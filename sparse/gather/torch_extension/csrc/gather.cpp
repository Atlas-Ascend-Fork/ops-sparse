// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// One-dimensional Gather and aten::index_select wrappers backed by aclsparseGather.

#include <algorithm>
#include <optional>
#include <tuple>

#include <torch/extension.h>

#include "aclsparse_common.h"

namespace {

using cann_ops_sparse::AclSparseContext;
using cann_ops_sparse::RunAclSparse;

aclDataType ToAclDataType(at::ScalarType scalarType)
{
    switch (scalarType) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kComplexFloat:
            return ACL_COMPLEX64;
        default:
            TORCH_CHECK(false,
                "aclsparseGather supports float16, bfloat16, float32 and complex64, but got ", scalarType);
    }
}

void CheckTensorContract(const at::Tensor &input, const at::Tensor &index)
{
    TORCH_CHECK(input.defined(), "input must be a defined tensor");
    TORCH_CHECK(index.defined(), "index must be a defined tensor");
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
        "input must be on an Ascend NPU, but got ", input.device());
    TORCH_CHECK(index.device().type() == c10::DeviceType::PrivateUse1,
        "index must be on an Ascend NPU, but got ", index.device());
    TORCH_CHECK(input.device() == index.device(), "input and index must be on the same NPU device");
    TORCH_CHECK(input.dim() == 1, "input must be one-dimensional, but got ", input.dim(), " dimensions");
    TORCH_CHECK(index.dim() == 1, "index must be one-dimensional, but got ", index.dim(), " dimensions");
    TORCH_CHECK(input.layout() == at::kStrided && index.layout() == at::kStrided,
        "input and index must use strided layout");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must use int32, but got ", index.scalar_type());
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
            input.scalar_type() == at::kFloat || input.scalar_type() == at::kComplexFloat,
        "input has unsupported dtype ", input.scalar_type());
    TORCH_CHECK(input.numel() > 0 || index.numel() == 0,
        "non-empty index requires a non-empty input, but got input length ", input.numel(),
        " and index length ", index.numel());
    TORCH_CHECK(!input.requires_grad(), "aclsparseGather is forward-only and does not support autograd");
}

class ConstDnVecGuard {
public:
    ConstDnVecGuard(int64_t size, const void *values, aclDataType valueType)
    {
        ACLSPARSE_CHECK(aclsparseCreateConstDnVec(&value_, size, values, valueType));
    }

    ~ConstDnVecGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroyDnVec(value_);
        }
    }

    ConstDnVecGuard(const ConstDnVecGuard &) = delete;
    ConstDnVecGuard &operator=(const ConstDnVecGuard &) = delete;

    aclsparseConstDnVecDescr_t get() const
    {
        return value_;
    }

private:
    aclsparseConstDnVecDescr_t value_ = nullptr;
};

class SpVecGuard {
public:
    SpVecGuard(int64_t size, int64_t nnz, void *indices, void *values, aclDataType valueType,
        aclsparseIndexBase_t indexBase)
    {
        ACLSPARSE_CHECK(aclsparseCreateSpVec(&value_, size, nnz, indices, values, ACL_SPARSE_INDEX_32I,
            indexBase, valueType));
    }

    ~SpVecGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroySpVec(value_);
        }
    }

    SpVecGuard(const SpVecGuard &) = delete;
    SpVecGuard &operator=(const SpVecGuard &) = delete;

    aclsparseSpVecDescr_t get() const
    {
        return value_;
    }

private:
    aclsparseSpVecDescr_t value_ = nullptr;
};

void GatherSlice(AclSparseContext &context, aclsparseConstDnVecDescr_t dense, int64_t inputSize,
    const at::Tensor &index, at::Tensor &output, aclDataType valueType, aclsparseIndexBase_t indexBase)
{
    SpVecGuard sparse(inputSize, index.numel(), const_cast<void *>(index.const_data_ptr()),
        output.mutable_data_ptr(), valueType, indexBase);
    ACLSPARSE_CHECK(aclsparseGather(context.handle(), dense, sparse.get()));
}

using IndexImplWeakPtr = c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;

class ValidatedIndexBoundsCache {
public:
    bool Matches(const at::Tensor &index, int64_t inputSize, int64_t base) const
    {
        const c10::TensorImpl *impl = index.unsafeGetTensorImpl();
        const auto &versionCounter = impl->version_counter();
        if (inputSize_ != inputSize || base_ != base) {
            return false;
        }
        if (!indexImpl_.has_value() || !versionCounter.enabled()) {
            return false;
        }
        const auto cachedImpl = indexImpl_->lock();
        return cachedImpl && cachedImpl.get() == impl && version_ == versionCounter.current_version();
    }

    void Store(const at::Tensor &index, int64_t inputSize, int64_t base)
    {
        const c10::TensorImpl *impl = index.unsafeGetTensorImpl();
        const auto &versionCounter = impl->version_counter();
        if (!versionCounter.enabled()) {
            indexImpl_.reset();
            return;
        }
        indexImpl_.emplace(index.getIntrusivePtr());
        version_ = versionCounter.current_version();
        inputSize_ = inputSize;
        base_ = base;
    }

private:
    std::optional<IndexImplWeakPtr> indexImpl_;
    uint32_t version_ = 0;
    int64_t inputSize_ = 0;
    int64_t base_ = 0;
};

void CheckIndexBounds(const at::Tensor &index, int64_t inputSize, int64_t base)
{
    // A weak, versioned single-entry cache avoids retaining device storage while
    // eliding repeated reductions for an unchanged index tensor on this thread.
    thread_local ValidatedIndexBoundsCache cache;
    if (cache.Matches(index, inputSize, base)) {
        return;
    }
    // The index tensor stays on NPU. Reading the scalar reduction results is
    // required to reject an invalid index before it can reach the Gather kernel.
    const auto indexRange = at::aminmax(index);
    const int64_t minIndex = std::get<0>(indexRange).item<int64_t>();
    const int64_t maxIndex = std::get<1>(indexRange).item<int64_t>();
    TORCH_CHECK_INDEX(minIndex >= base && maxIndex - base < inputSize,
        "index out of range in self; expected every index to be in [", base, ", ", inputSize + base,
        "), but found range [", minIndex, ", ", maxIndex, "]");
    cache.Store(index, inputSize, base);
}

at::Tensor Gather(const at::Tensor &input, const at::Tensor &index, int64_t base)
{
    TORCH_CHECK(base == 0 || base == 1, "aclsparseGather base must be 0 or 1, but got ", base);
    CheckTensorContract(input, index);
    const auto indexBase = base == 0 ? ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;
    return RunAclSparse(input, [&](AclSparseContext &context) {
        at::Tensor output = at::empty({index.numel()}, input.options());
        if (index.numel() == 0) {
            return output;
        }

        CheckIndexBounds(index, input.numel(), base);
        const aclDataType valueType = ToAclDataType(input.scalar_type());
        ConstDnVecGuard dense(input.numel(), input.const_data_ptr(), valueType);
        // Record the shared storage before any launch, including exceptional multi-launch paths.
        context.RecordTensors(input, index, output);
        // Submit queued torch_npu work before directly launching on the same current stream.
        (void)c10_npu::getCurrentNPUStream(input.device().index()).stream(true);
        if (index.numel() <= input.numel()) {
            GatherSlice(context, dense.get(), input.numel(), index, output, valueType, indexBase);
        } else {
            // index_select allows repeated indices and an output longer than the input.
            // Keep the shared SpVec contract (nnz <= size) using zero-copy tensor views.
            for (int64_t offset = 0; offset < index.numel();) {
                const int64_t count = std::min(input.numel(), index.numel() - offset);
                const at::Tensor indexSlice = index.narrow(0, offset, count);
                at::Tensor outputSlice = output.narrow(0, offset, count);
                GatherSlice(context, dense.get(), input.numel(), indexSlice, outputSlice, valueType, indexBase);
                offset += count;
            }
        }
        return output;
    });
}

at::Tensor IndexSelect(const at::Tensor &input, const at::Tensor &index)
{
    return Gather(input, index, 0);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("gather", &Gather, "NPU Gather with zero-based or one-based indices");
    module.def("index_select", &IndexSelect, "One-dimensional NPU index_select backed by aclsparseGather");
}
