// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Sparse → dense C++ wrapper for aten::_to_dense (CSR / CSC / COO).
//
// arch22（Atlas A2/A3，910B/910_93）与 arch35（Ascend 950）经 aclsparseSparseToDense
// 公开 API 按 SoC 分发，本 wrapper 两架构共用。python 适配层已在所有 SoC 上对未
// coalesce COO 预归并（Tensor.to_dense 对重复坐标求和），经由此 wrapper 的 COO 为
// coalesced；COO 路径仍取 _values()/_indices() 私有访问器以兼容直接调用（公开
// .values()/.indices() 在未 coalesce 张量上直接 RuntimeError，torch 2.7 行为）——
// 原生算子对重复坐标的契约为任一写入值（aclsparse API 层约定）。

#include <limits>

#include <torch/extension.h>

#include "aclsparse_common.h"

namespace {

using cann_ops_sparse::AclSparseContext;
using cann_ops_sparse::AclSparseWorkspace;
using cann_ops_sparse::RunAclSparse;

constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();

class ConstSpMatGuard {
public:
    ConstSpMatGuard() = default;

    ~ConstSpMatGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroySpMat(value_);
        }
    }

    ConstSpMatGuard(const ConstSpMatGuard &) = delete;
    ConstSpMatGuard &operator=(const ConstSpMatGuard &) = delete;

    aclsparseConstSpMatDescr_t *address()
    {
        return &value_;
    }

    aclsparseConstSpMatDescr_t get() const
    {
        return value_;
    }

private:
    aclsparseConstSpMatDescr_t value_ = nullptr;
};

class DnMatGuard {
public:
    DnMatGuard() = default;

    ~DnMatGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroyDnMat(value_);
        }
    }

    DnMatGuard(const DnMatGuard &) = delete;
    DnMatGuard &operator=(const DnMatGuard &) = delete;

    aclsparseDnMatDescr_t *address()
    {
        return &value_;
    }

    aclsparseDnMatDescr_t get() const
    {
        return value_;
    }

private:
    aclsparseDnMatDescr_t value_ = nullptr;
};

aclDataType ToAclDataType(at::ScalarType type)
{
    switch (type) {
        case at::kChar:
            return ACL_INT8;
        case at::kInt:
            return ACL_INT32;
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
                "SparseToDense supports int8, int32, float16, bfloat16, float32 and complex64, but got ",
                type);
    }
}

void CheckSparseInput(const at::Tensor &tensor)
{
    TORCH_CHECK(tensor.defined(), "sparse input must be a defined tensor");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
        "SparseToDense requires an Ascend NPU tensor, but got device ", tensor.device());
    TORCH_CHECK(tensor.dim() == 2, "SparseToDense supports only 2-D matrices, but got ", tensor.dim(),
        " dimensions");
    TORCH_CHECK(tensor.is_sparse() || tensor.layout() == at::kSparseCsr || tensor.layout() == at::kSparseCsc,
        "SparseToDense expects a sparse layout, but got ", tensor.layout());
    TORCH_CHECK(tensor.size(0) <= kInt32Max && tensor.size(1) <= kInt32Max && tensor._nnz() <= kInt32Max,
        "SparseToDense exceeds the int32 limit: shape ", tensor.sizes(), " with nnz ", tensor._nnz(),
        " must stay within ", kInt32Max);
}

at::Tensor ContiguousInt32(const at::Tensor &index)
{
    return index.to(at::kInt).contiguous();
}

void CreateSpMatFromSparse(ConstSpMatGuard &descriptor, const at::Tensor &sparse, aclDataType valueType,
    at::Tensor &primary, at::Tensor &secondary, at::Tensor &values)
{
    const int64_t rows = sparse.size(0);
    const int64_t cols = sparse.size(1);
    const int64_t nnz = sparse._nnz();

    if (sparse.layout() == at::kSparseCsr) {
        values = sparse.values().contiguous();
        primary = ContiguousInt32(sparse.crow_indices());
        secondary = ContiguousInt32(sparse.col_indices());
        ACLSPARSE_CHECK(aclsparseCreateConstCsr(descriptor.address(), rows, cols, nnz,
            primary.const_data_ptr(), secondary.const_data_ptr(), values.const_data_ptr(), ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, valueType));
        return;
    }

    if (sparse.layout() == at::kSparseCsc) {
        values = sparse.values().contiguous();
        primary = ContiguousInt32(sparse.ccol_indices());
        secondary = ContiguousInt32(sparse.row_indices());
        ACLSPARSE_CHECK(aclsparseCreateConstCsc(descriptor.address(), rows, cols, nnz,
            primary.const_data_ptr(), secondary.const_data_ptr(), values.const_data_ptr(),
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, valueType));
        return;
    }

    TORCH_CHECK(sparse.layout() == at::kSparse, "unsupported sparse layout ", sparse.layout());
    // hybrid COO（dense_dim>0 → values 非 1-D）不支持，显式报错；未 coalesce COO
    // 可接受（见文件头说明），公开访问器在未 coalesce 张量上不可用。
    TORCH_CHECK(sparse.sparse_dim() == 2 && sparse.dense_dim() == 0,
        "SparseToDense supports only plain 2-D COO tensors, but got sparse_dim=",
        sparse.sparse_dim(), " dense_dim=", sparse.dense_dim());
    values = sparse._values().contiguous();
    const at::Tensor indices = sparse._indices().to(at::kInt).contiguous();
    primary = indices.select(0, 0).contiguous();
    secondary = indices.select(0, 1).contiguous();
    ACLSPARSE_CHECK(aclsparseCreateConstCoo(descriptor.address(), rows, cols, nnz, primary.const_data_ptr(),
        secondary.const_data_ptr(), values.const_data_ptr(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
        valueType));
}

at::Tensor SparseToDense(const at::Tensor &self)
{
    CheckSparseInput(self);
    const int64_t rows = self.size(0);
    const int64_t cols = self.size(1);
    if (rows == 0 || cols == 0 || self._nnz() == 0) {
        return at::zeros({rows, cols}, self.options().layout(at::kStrided));
    }

    return RunAclSparse(self, [&](AclSparseContext &context) {
        const aclDataType valueType = ToAclDataType(self.scalar_type());
        ConstSpMatGuard spMat;
        at::Tensor primary;
        at::Tensor secondary;
        at::Tensor values;
        CreateSpMatFromSparse(spMat, self, valueType, primary, secondary, values);

        // torch_npu 的 task-queue 模式（TASK_QUEUE_ENABLE=2 默认档）会延迟下发本线程
        // 的 aclnn 算子——上方 ContiguousInt32 的 int64→int32 索引 cast 即经此路径。
        // aclsparseSparseToDense（arch22）的校验链以原生 aclrt 时序同步读输入数组，
        // 必须先确保这些算子已实际入队，否则 D2H 可能读到复用内存中的陈旧数据
        //（CSR int32 直传正常、int64 输入触发 cast 后可能读到上一调用残留）。
        // stream(true) 在返回 rtStream 前清空当前 stream 的 task queue（torch_npu
        // NPUStream 语义），保证后续原生 aclrt 操作排在 cast 之后。
        (void)c10_npu::getCurrentNPUStream().stream(true);

        // Kernel path clears dense then scatters nnz; prefer empty to avoid a
        // pre-fill zeros() racing / poisoning subsequent ACLSparse launches.
        at::Tensor dense = at::empty({rows, cols}, values.options().layout(at::kStrided));
        context.RecordTensors(primary, secondary, values, dense);

        DnMatGuard dnMat;
        const int64_t ld = dense.stride(0);
        ACLSPARSE_CHECK(aclsparseCreateDnMat(dnMat.address(), rows, cols, ld, dense.data_ptr(), valueType,
            ACL_SPARSE_ORDER_ROW));

        size_t bufferSize = 0;
        ACLSPARSE_CHECK(aclsparseSparseToDense_bufferSize(context.handle(), spMat.get(), dnMat.get(),
            ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize));
        AclSparseWorkspace workspace(context, values, bufferSize);
        ACLSPARSE_CHECK(aclsparseSparseToDense(context.handle(), spMat.get(), dnMat.get(),
            ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, workspace.data()));
        return dense;
    });
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_to_dense", &SparseToDense, "Convert CSR/CSC/COO sparse tensor to dense on NPU");
}
