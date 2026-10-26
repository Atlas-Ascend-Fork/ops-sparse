/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// ACLSparse 三阶段 SDDMM 的 ATen 桥接层。
//
// 本文件把 PyTorch 侧的稀疏张量/稠密张量转成 ACLSparse 描述符，按
// BufferSize -> Preprocess -> Execute 的顺序调用公开 C API，并原地更新 C 的
// values 数组。计算全部在 NPU 上完成，不存在 CPU fallback 分支。
//
// sddmm_csr / sddmm_bsr_pattern 接收索引与 values 张量。
// sparse_sampled_addmm_csr 从 PyTorch CSR 张量提取分量后调用同一套逻辑。

#include "aclsparse_common.h"

#include <c10/core/Scalar.h>
#include <pybind11/complex.h>
#include <torch/extension.h>
#include "torch_npu/csrc/framework/OpCommand.h"

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

extern "C" aclsparseStatus_t sddmm_index32_launch(uint8_t* source, uint8_t* dest, uint32_t count, uint32_t cores,
                                                  void* stream);

extern "C" aclsparseStatus_t sddmm_call_local(aclsparseHandle_t handle, aclsparseOperation_t ox,
                                              aclsparseOperation_t oy, const void* alpha, aclsparseConstDnMatDescr_t x,
                                              aclsparseConstDnMatDescr_t y, const void* beta, aclsparseSpMatDescr_t c,
                                              aclDataType compute, aclsparseSDDMMAlg_t alg, void* buffer);

namespace {

constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();

aclDataType ToAclDataType(at::ScalarType type)
{
    switch (type) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kComplexFloat:
            return ACL_COMPLEX64;
        default:
            TORCH_CHECK(false, "SDDMM supports float16, bfloat16, float32 and complex64, got ", type);
    }
}

// 任务书 §2.3.5 支持矩阵：complex64 按复数累加，其余组合一律以 FP32 累加
// （FP16/FP16、BF16/BF16 也是 FP32 computeType）。
aclDataType ComputeTypeFor(at::ScalarType valueType)
{
    return valueType == at::kComplexFloat ? ACL_COMPLEX64 : ACL_FLOAT;
}

// alpha/beta 按 computeType 打包成 Host 标量。ptrMode 固定为 HOST，因此这里
// 直接持有字节，生命周期覆盖整个三阶段调用。
struct HostScalar {
    alignas(8) std::array<uint8_t, 8> bytes{};

    const void* Data() const { return bytes.data(); }
};

template <typename T>
void StoreScalar(HostScalar& destination, const T& value)
{
    static_assert(sizeof(T) <= sizeof(destination.bytes), "scalar does not fit the host buffer");
    std::copy_n(reinterpret_cast<const uint8_t*>(&value), sizeof(value), destination.bytes.data());
}

HostScalar MakeScalar(const c10::Scalar& scalar, aclDataType computeType)
{
    HostScalar result{};
    if (computeType == ACL_COMPLEX64) {
        const c10::complex<float> value = scalar.toComplexFloat();
        StoreScalar(result, aclsparseComplex{value.real(), value.imag()});
    } else {
        const c10::complex<double> value = scalar.toComplexDouble();
        TORCH_CHECK(value.imag() == 0.0, "a real SDDMM dtype cannot use a complex alpha/beta");
        StoreScalar(result, scalar.toFloat());
    }
    return result;
}

class DnMatGuard {
public:
    ~DnMatGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroyDnMat(value_);
        }
    }

    aclsparseDnMatDescr_t* Out() { return &value_; }

    aclsparseDnMatDescr_t Get() const { return value_; }

private:
    aclsparseDnMatDescr_t value_ = nullptr;
};

class SpMatGuard {
public:
    ~SpMatGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroySpMat(value_);
        }
    }

    aclsparseSpMatDescr_t* Out() { return &value_; }

    aclsparseSpMatDescr_t Get() const { return value_; }

private:
    aclsparseSpMatDescr_t value_ = nullptr;
};

at::Tensor Index32(const at::Tensor& input)
{
    TORCH_CHECK(input.is_privateuseone() && input.layout() == at::kStrided && input.dim() == 1,
                "SDDMM indices must be one-dimensional NPU arrays");
    TORCH_CHECK(input.scalar_type() == at::kInt || input.scalar_type() == at::kLong,
                "SDDMM indices must have int32 or int64 dtype");
    TORCH_CHECK(input.numel() <= kInt32Max, "SDDMM index count exceeds int32 range");
    if (input.scalar_type() == at::kInt) {
        return input.contiguous();
    }
    c10::OptionalDeviceGuard guard(input.device());
    auto source = input.contiguous();
    auto result = at::empty(input.sizes(), input.options().dtype(at::kInt));
    const auto stream = c10_npu::getCurrentNPUStream(input.device().index());
    const uint32_t count = input.numel();
    const uint32_t cores = std::min(uint32_t(32), (count + 1023) / 1024);
    ACLSPARSE_CHECK(sddmm_index32_launch(static_cast<uint8_t*>(source.data_ptr()),
                                         static_cast<uint8_t*>(result.data_ptr()), count, cores, stream.stream()));
    c10_npu::NPUCachingAllocator::recordStream(source.storage().data_ptr(), stream);
    c10_npu::NPUCachingAllocator::recordStream(result.storage().data_ptr(), stream);
    return result;
}

int64_t ReadIndexBase(const at::Tensor& rows)
{
    TORCH_CHECK(rows.is_privateuseone() && rows.dim() == 1 && rows.numel() > 0 &&
                    (rows.scalar_type() == at::kInt || rows.scalar_type() == at::kLong),
                "CSR row offsets must be a nonempty int32 or int64 NPU array");
    c10::OptionalDeviceGuard guard(rows.device());
    struct HostIndex {
        void* data = nullptr;
        ~HostIndex()
        {
            if (data) {
                (void)aclrtFreeHost(data);
            }
        }
    };
    static thread_local HostIndex host;
    if (!host.data) {
        TORCH_CHECK(aclrtMallocHost(&host.data, 64) == ACL_SUCCESS, "CSR base allocation failed");
    }
    const auto stream = c10_npu::getCurrentNPUStream(rows.device().index()).stream();
    TORCH_CHECK(aclrtMemcpyAsync(host.data, 64, rows.data_ptr(), rows.element_size(), ACL_MEMCPY_DEVICE_TO_HOST,
                                 stream) == ACL_SUCCESS,
                "CSR base copy failed");
    TORCH_CHECK(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "CSR base synchronization failed");
    return rows.scalar_type() == at::kInt ? *static_cast<int32_t*>(host.data) : *static_cast<int64_t*>(host.data);
}

// Keep caller storage alive, but reject metadata changes that invalidate descriptors.
struct TensorBinding {
    at::Tensor source;
    void* pointer = nullptr;
    std::vector<int64_t> sizes, strides;
    void Bind(const at::Tensor& tensor)
    {
        source = tensor;
        pointer = tensor.data_ptr();
        sizes = tensor.sizes().vec();
        strides = tensor.strides().vec();
    }
    bool Refresh(at::Tensor& target, bool index) const
    {
        TORCH_CHECK(source.data_ptr() == pointer && source.sizes().equals(sizes) && source.strides().equals(strides),
                    "prepared SDDMM binding changed; prepare again");
        if (target.data_ptr() != pointer) {
            target.copy_(index ? Index32(source) : source);
            return true;
        }
        return false;
    }
};

void CheckCsrShapes(const at::Tensor& rowPtr, const at::Tensor& colInd, const at::Tensor& values, int64_t m,
                    int64_t nnz)
{
    TORCH_CHECK(rowPtr.scalar_type() == at::kInt && rowPtr.numel() == m + 1,
                "csr row offsets must be an int32 array of length rows + 1");
    TORCH_CHECK(colInd.scalar_type() == at::kInt && colInd.numel() == nnz,
                "csr column indices must be an int32 array of length nnz");
    TORCH_CHECK(values.scalar_type() == at::kFloat || values.scalar_type() == at::kHalf ||
                    values.scalar_type() == at::kBFloat16 || values.scalar_type() == at::kComplexFloat,
                "unsupported values dtype ", values.scalar_type());
    // values 既是输入又是输出，核函数按 nnz 个标量原地更新，长度必须严格相等。
    TORCH_CHECK(values.dim() == 1 && values.is_contiguous() && values.numel() == nnz,
                "csr values must hold exactly nnz elements, got ", values.numel(), " for nnz = ", nnz);
    TORCH_CHECK(m >= 0 && m <= kInt32Max, "rows exceed the int32 limit");
}

void CheckDenseOperands(const at::Tensor& x, const at::Tensor& y, int64_t m, int64_t n, int64_t k);

// 描述符只引用 device 指针，因此这里保证每个数组都是连续且整型宽度符合契约。
void ExecuteCsr(cann_ops_sparse::AclSparseContext& context, const at::Tensor& rowPtr, const at::Tensor& colInd,
                at::Tensor& values, const at::Tensor& x, const at::Tensor& y, int64_t m, int64_t n, int64_t k,
                int64_t nnz, aclsparseIndexBase_t idxBase, aclDataType xValueType, aclDataType cValueType,
                const HostScalar& alpha, const HostScalar& beta, aclDataType computeType)
{
    DnMatGuard matX;
    DnMatGuard matY;
    SpMatGuard matC;
    ACLSPARSE_CHECK(aclsparseCreateDnMat(matX.Out(), m, k, k, x.mutable_data_ptr(), xValueType, ACL_SPARSE_ORDER_ROW));
    ACLSPARSE_CHECK(aclsparseCreateDnMat(matY.Out(), k, n, n, y.mutable_data_ptr(), xValueType, ACL_SPARSE_ORDER_ROW));
    ACLSPARSE_CHECK(aclsparseCreateCsr(matC.Out(), m, n, nnz, rowPtr.mutable_data_ptr(), colInd.mutable_data_ptr(),
                                       values.mutable_data_ptr(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, idxBase,
                                       cValueType));

    const aclsparseHandle_t handle = context.handle();
    constexpr aclsparseOperation_t op = ACL_SPARSE_OP_NON_TRANSPOSE;
    constexpr aclsparseSDDMMAlg_t alg = ACL_SPARSE_SDDMM_ALG_DEFAULT;

    size_t bufferSize = 0;
    ACLSPARSE_CHECK(aclsparseSDDMMBufferSize(handle, op, op, alpha.Data(), matX.Get(), matY.Get(), beta.Data(),
                                             matC.Get(), computeType, alg, &bufferSize));
    cann_ops_sparse::AclSparseWorkspace buffer(context, values, bufferSize);
    ACLSPARSE_CHECK(sddmm_call_local(handle, op, op, alpha.Data(), matX.Get(), matY.Get(), beta.Data(), matC.Get(),
                                     computeType, alg, buffer.data()));
}

// Caller-owned three-stage state for repeated execution over one immutable CSR
// pattern and one X/Y binding. The public ACLSparse API prepares once and
// executes repeatedly; the ordinary ATen path keeps call-local resources.
class PreparedCsr {
public:
    PreparedCsr(const at::Tensor& rowPtrIn, const at::Tensor& colIndIn, const at::Tensor& valuesIn,
                const at::Tensor& xIn, const at::Tensor& yIn, const c10::Scalar& alphaScalar,
                const c10::Scalar& betaScalar, int64_t idxBaseValue)
        : values_(valuesIn)
    {
        ValidateOperands(rowPtrIn, colIndIn, xIn, yIn, idxBaseValue);
        rowPtr_ = Index32(rowPtrIn);
        colInd_ = Index32(colIndIn);
        // Raw descriptors need physical conjugated data, not ATen's lazy bit.
        // Keep original bindings below so Execute refreshes materialized copies.
        x_ = xIn.resolve_conj().resolve_neg().contiguous();
        y_ = yIn.resolve_conj().resolve_neg();
        y_ = y_.stride(0) == 1 && y_.stride(1) == y_.size(0) ? y_ : y_.contiguous();
        const std::array<at::Tensor, 5> sources{rowPtrIn, colIndIn, valuesIn, xIn, yIn};
        for (size_t i = 0; i < sources.size(); ++i) {
            bindings_[i].Bind(sources[i]);
        }

        m_ = x_.size(0);
        k_ = x_.size(1);
        n_ = y_.size(1);
        nnz_ = colInd_.numel();
        CheckCsrShapes(rowPtr_, colInd_, values_, m_, nnz_);
        TORCH_CHECK(x_.dim() == 2 && y_.dim() == 2 && y_.size(0) == k_ && x_.scalar_type() == y_.scalar_type(),
                    "prepared SDDMM dense shape/dtype mismatch");
        const bool yColumnMajor = y_.stride(0) == 1 && y_.stride(1) == k_;
        TORCH_CHECK(n_ <= kInt32Max && k_ <= kInt32Max && nnz_ <= kInt32Max,
                    "SDDMM dimensions and nnz must fit in int32");

        xValueType_ = ToAclDataType(x_.scalar_type());
        cValueType_ = ToAclDataType(values_.scalar_type());
        computeType_ = ComputeTypeFor(values_.scalar_type());
        alpha_ = MakeScalar(alphaScalar, computeType_);
        beta_ = MakeScalar(betaScalar, computeType_);
        idxBase_ = idxBaseValue == 0 ? ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;

        PrepareDescriptors(yColumnMajor);
    }

    PreparedCsr(const PreparedCsr&) = delete;
    PreparedCsr& operator=(const PreparedCsr&) = delete;

    at::Tensor Execute()
    {
        return cann_ops_sparse::RunAclSparse(values_, [&](cann_ops_sparse::AclSparseContext& context) {
            RefreshBindings(context);
            ACLSPARSE_CHECK(aclsparseSDDMM(context.handle(), kOp, kOp, alpha_.Data(), matX_.Get(), matY_.Get(),
                                           beta_.Data(), matC_.Get(), computeType_, kAlg, WorkspaceData()));
            context.RecordTensors(rowPtr_, colInd_, values_, x_, y_, workspace_);
            return values_;
        });
    }

    // For the version-checked binding owned by SddmmNpuEntry only. All converted
    // operands are already prepared and immutable for this invocation. Queue
    // the *entire* public Execute on torch_npu's launch queue, in stream order
    // with the caller's copies/events, instead of draining that queue on the
    // interpreter thread before every call. Public PreparedSddmmCsr.execute()
    // keeps its refresh/validation semantics above.
    static void EnqueueBound(const std::shared_ptr<PreparedCsr>& self)
    {
        c10::OptionalDeviceGuard guard(self->values_.device());
        const auto stream = c10_npu::getCurrentNPUStream(self->values_.device().index());
        auto handle = cann_ops_sparse::AclSparseHandleCache::Acquire(self->values_.device().index(),
                                                                     stream.stream(false));
        for (const auto& tensor : {self->rowPtr_, self->colInd_, self->values_, self->x_, self->y_, self->workspace_}) {
            if (tensor.defined()) {
                c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
            }
        }
        // Strong captures keep descriptors/workspace and the handle alive even
        // if the next call replaces the thread-local binding before launch.
        at_npu::native::OpCommand::RunOpApiV2("aclsparseSDDMM", [self, handle]() -> int {
            std::lock_guard<std::mutex> lock(handle->mutex);
            ACLSPARSE_CHECK(aclsparseSDDMM(handle->handle(), kOp, kOp, self->alpha_.Data(), self->matX_.Get(),
                                           self->matY_.Get(), self->beta_.Data(), self->matC_.Get(), self->computeType_,
                                           kAlg, self->WorkspaceData()));
            return ACL_SUCCESS;
        });
    }

    int64_t WorkspaceBytes() const { return static_cast<int64_t>(workspaceBytes_); }

private:
    void ValidateOperands(const at::Tensor& rowPtrIn, const at::Tensor& colIndIn, const at::Tensor& xIn,
                          const at::Tensor& yIn, int64_t idxBaseValue) const
    {
        TORCH_CHECK(idxBaseValue == 0 || idxBaseValue == 1, "index base must be 0 or 1");
        TORCH_CHECK(rowPtrIn.is_privateuseone() && colIndIn.is_privateuseone() && values_.is_privateuseone() &&
                        xIn.is_privateuseone() && yIn.is_privateuseone(),
                    "SDDMM requires every operand to live on an Ascend NPU");
        TORCH_CHECK(values_.is_contiguous(), "prepared SDDMM requires contiguous values");
        TORCH_CHECK(xIn.layout() == at::kStrided && yIn.layout() == at::kStrided && xIn.dim() == 2 && yIn.dim() == 2,
                    "prepared SDDMM expects two-dimensional dense inputs");
        for (const auto& tensor : {rowPtrIn, colIndIn, xIn, yIn}) {
            TORCH_CHECK(tensor.device() == values_.device(), "SDDMM operands must share one NPU device");
        }
    }
    void PrepareDescriptors(bool yColumnMajor)
    {
        cann_ops_sparse::RunAclSparse(values_, [&](cann_ops_sparse::AclSparseContext& context) {
            ACLSPARSE_CHECK(aclsparseCreateDnMat(matX_.Out(), m_, k_, k_, x_.mutable_data_ptr(), xValueType_,
                                                 ACL_SPARSE_ORDER_ROW));
            ACLSPARSE_CHECK(aclsparseCreateDnMat(matY_.Out(), k_, n_, yColumnMajor ? k_ : n_, y_.mutable_data_ptr(),
                                                 xValueType_,
                                                 yColumnMajor ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW));
            ACLSPARSE_CHECK(aclsparseCreateCsr(matC_.Out(), m_, n_, nnz_, rowPtr_.mutable_data_ptr(),
                                               colInd_.mutable_data_ptr(), values_.mutable_data_ptr(),
                                               ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, idxBase_, cValueType_));

            const aclsparseHandle_t handle = context.handle();
            ACLSPARSE_CHECK(aclsparseSDDMMBufferSize(handle, kOp, kOp, alpha_.Data(), matX_.Get(), matY_.Get(),
                                                     beta_.Data(), matC_.Get(), computeType_, kAlg, &workspaceBytes_));
            TORCH_CHECK(workspaceBytes_ <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                        "SDDMM workspace exceeds the ATen allocation limit");
            if (workspaceBytes_ != 0) {
                workspace_ = at::empty({static_cast<int64_t>(workspaceBytes_)}, values_.options().dtype(at::kByte));
            }
            ACLSPARSE_CHECK(aclsparseSDDMMPreprocess(handle, kOp, kOp, alpha_.Data(), matX_.Get(), matY_.Get(),
                                                     beta_.Data(), matC_.Get(), computeType_, kAlg, WorkspaceData()));
            context.RecordTensors(rowPtr_, colInd_, values_, x_, y_, workspace_);
        });
    }

    void RefreshBindings(cann_ops_sparse::AclSparseContext& context)
    {
        std::array<at::Tensor*, 5> targets{&rowPtr_, &colInd_, &values_, &x_, &y_};
        bool copied = false;
        for (size_t i = 0; i < targets.size(); ++i) {
            copied |= bindings_[i].Refresh(*targets[i], i < 2);
            context.RecordTensors(bindings_[i].source);
        }
        if (copied) {
            // Submit queued ATen copies before launching raw ACL validation.
            (void)c10_npu::getCurrentNPUStream(values_.device().index()).stream(true);
        }
    }

    void* WorkspaceData() const { return workspace_.defined() ? workspace_.data_ptr() : nullptr; }

    static constexpr aclsparseOperation_t kOp = ACL_SPARSE_OP_NON_TRANSPOSE;
    static constexpr aclsparseSDDMMAlg_t kAlg = ACL_SPARSE_SDDMM_ALG_DEFAULT;

    at::Tensor rowPtr_;
    std::array<TensorBinding, 5> bindings_;
    at::Tensor colInd_;
    at::Tensor values_;
    at::Tensor x_;
    at::Tensor y_;
    at::Tensor workspace_;
    DnMatGuard matX_;
    DnMatGuard matY_;
    SpMatGuard matC_;
    HostScalar alpha_;
    HostScalar beta_;
    int64_t m_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t nnz_ = 0;
    size_t workspaceBytes_ = 0;
    aclsparseIndexBase_t idxBase_ = ACL_SPARSE_INDEX_BASE_ZERO;
    aclDataType xValueType_ = ACL_DT_UNDEFINED;
    aclDataType cValueType_ = ACL_DT_UNDEFINED;
    aclDataType computeType_ = ACL_DT_UNDEFINED;
};

// 校验 X/Y 的形状与 C 的 [m, n] 相容，并返回规整后的二维布局。
void CheckDenseOperands(const at::Tensor& x, const at::Tensor& y, int64_t m, int64_t n, int64_t k)
{
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "SDDMM expects two-dimensional X and Y");
    TORCH_CHECK(x.size(0) == m && x.size(1) == k, "X must be [m, k] = [", m, ", ", k, "], got ", x.sizes());
    TORCH_CHECK(y.size(0) == k && y.size(1) == n, "Y must be [k, n] = [", k, ", ", n, "], got ", y.sizes());
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "X and Y must share one dtype");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "X and Y must be contiguous for a strided-batch-1 launch");
}

void InitializeCsrOutput(cann_ops_sparse::AclSparseContext& context, const at::Tensor& inputValues, at::Tensor& values)
{
    // Complex clone dispatches through AICPU Identity on this CANN version.
    // A byte copy preserves the out-of-place contract without that operator.
    if (inputValues.defined() && inputValues.numel() != 0) {
        const auto stream = c10_npu::getCurrentNPUStream(values.device().index()).stream();
        const size_t bytes = inputValues.numel() * inputValues.element_size();
        const aclError status = aclrtMemcpyAsync(values.mutable_data_ptr(), bytes, inputValues.data_ptr(), bytes,
                                                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        TORCH_CHECK(status == ACL_SUCCESS, "SDDMM output initialization failed: ", status);
        context.RecordTensors(inputValues);
    }
}

at::Tensor SddmmCsrImpl(const at::Tensor& rowPtrIn, const at::Tensor& colIndIn, at::Tensor values,
                        const at::Tensor& xIn, const at::Tensor& yIn, const c10::Scalar& alphaScalar,
                        const c10::Scalar& betaScalar, int64_t idxBaseValue, bool outOfPlace = false)
{
    TORCH_CHECK(idxBaseValue == 0 || idxBaseValue == 1, "index base must be 0 or 1");
    TORCH_CHECK(rowPtrIn.is_privateuseone() && colIndIn.is_privateuseone() && values.is_privateuseone() &&
                    xIn.is_privateuseone() && yIn.is_privateuseone(),
                "SDDMM requires every operand to live on an Ascend NPU");
    for (const auto& tensor : {rowPtrIn, colIndIn, xIn, yIn}) {
        TORCH_CHECK(tensor.device() == values.device(), "SDDMM operands must share one NPU device");
    }

    at::Tensor inputValues;
    if (outOfPlace) {
        if (betaScalar.toComplexDouble() != c10::complex<double>(0.0, 0.0)) {
            inputValues = values.contiguous();
        }
        values = at::empty(values.sizes(), values.options());
    }

    TORCH_CHECK(xIn.layout() == at::kStrided && yIn.layout() == at::kStrided && xIn.dim() == 2 && yIn.dim() == 2,
                "SDDMM expects two-dimensional dense inputs");
    at::Tensor rowPtr = Index32(rowPtrIn);
    at::Tensor colInd = Index32(colIndIn);
    at::Tensor x = xIn.resolve_conj().resolve_neg().contiguous();
    at::Tensor y = yIn.resolve_conj().resolve_neg().contiguous();

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t n = y.size(1);
    const int64_t nnz = colInd.numel();
    CheckCsrShapes(rowPtr, colInd, values, m, nnz);
    CheckDenseOperands(x, y, m, n, k);
    TORCH_CHECK(n <= kInt32Max && k <= kInt32Max && nnz <= kInt32Max, "SDDMM dimensions and nnz must fit in int32");

    const aclDataType xValueType = ToAclDataType(x.scalar_type());
    const aclDataType cValueType = ToAclDataType(values.scalar_type());
    const aclDataType computeType = ComputeTypeFor(values.scalar_type());
    const HostScalar alpha = MakeScalar(alphaScalar, computeType);
    const HostScalar beta = MakeScalar(betaScalar, computeType);
    const aclsparseIndexBase_t idxBase = idxBaseValue == 0 ? ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;

    return cann_ops_sparse::RunAclSparse(values, [&](cann_ops_sparse::AclSparseContext& context) {
        InitializeCsrOutput(context, inputValues, values);
        ExecuteCsr(context, rowPtr, colInd, values, x, y, m, n, k, nnz, idxBase, xValueType, cValueType, alpha, beta,
                   computeType);
        context.RecordTensors(rowPtr, colInd, values, x, y);
        return values;
    });
}

} // namespace

/// 原始张量入口：索引/values 已是 CSR 分量，X/Y 为 [m, k] 与 [k, n] 稠密矩阵。
at::Tensor SddmmCsr(const at::Tensor& rowPtr, const at::Tensor& colInd, const at::Tensor& values, const at::Tensor& x,
                    const at::Tensor& y, const c10::Scalar& alpha, const c10::Scalar& beta, int64_t idxBase)
{
    return SddmmCsrImpl(rowPtr, colInd, values, x, y, alpha, beta, idxBase);
}

/// Public ATen CSR backend: self is read-only and result owns its values.
at::Tensor SparseSampledAddmmCsr(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
                                 const c10::Scalar& beta, const c10::Scalar& alpha)
{
    TORCH_CHECK(self.layout() == at::kSparseCsr, "self must use sparse CSR layout");
    TORCH_CHECK(self.is_privateuseone(), "self must be on an Ascend NPU");
    TORCH_CHECK(self.dim() == 2, "self must be a two-dimensional sparse matrix");
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "mat1 and mat2 must be two-dimensional dense matrices");
    TORCH_CHECK(self.size(0) == mat1.size(0) && mat1.size(1) == mat2.size(0) && mat2.size(1) == self.size(1),
                "sampled_addmm shape mismatch: [", self.sizes(), "] x [", mat1.sizes(), "] x [", mat2.sizes(), "]");
    // mat1/mat2 (X/Y) and self (C) are deliberately allowed to differ: the
    // support matrix pairs FP16 or BF16 X/Y with an FP32 C. Only X and Y must
    // agree, since they are the two operands of the dot product.
    TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "mat1 and mat2 must share one dtype, got ",
                mat1.scalar_type(), " and ", mat2.scalar_type());

    at::Tensor crowInd = self.crow_indices();
    // `sparse_sampled_addmm` 的 ATen 签名没有索引基值参数，但 CSR 张量自己就
    // 声明了它：crow_indices[0] 为 0 是 C 风格，为 1 是 Fortran 风格。按张量
    // 实际携带的基值下发，而不是假定 0，才能让 1 基算例真正走到 C API 的
    // ACL_SPARSE_INDEX_BASE_ONE 分支。
    const int64_t idxBaseValue = ReadIndexBase(crowInd);
    TORCH_CHECK(idxBaseValue == 0 || idxBaseValue == 1,
                "sparse_sampled_addmm expects a 0- or 1-based CSR matrix, got crow_indices[0]=", idxBaseValue);

    // The public ATen operation is out-of-place; the ACLSparse API updates C.
    at::Tensor result = SddmmCsrImpl(crowInd, self.col_indices(), self.values(), mat1, mat2, alpha, beta, idxBaseValue,
                                     true);
    return at::sparse_csr_tensor(self.crow_indices(), self.col_indices(), result, self.sizes(),
                                 result.options().layout(at::kSparseCsr));
}

// pybind11 没有 c10::Scalar 的 type caster，因此导出层用 std::complex<double>
// 收 alpha/beta（实数会落到虚部为 0 的复数上），内部再还原成 c10::Scalar。
at::Tensor SddmmCsrPy(const at::Tensor& rowPtr, const at::Tensor& colInd, const at::Tensor& values, const at::Tensor& x,
                      const at::Tensor& y, const std::complex<double>& alpha, const std::complex<double>& beta,
                      int64_t idxBase)
{
    return SddmmCsr(rowPtr, colInd, values, x, y, c10::Scalar(c10::complex<double>(alpha.real(), alpha.imag())),
                    c10::Scalar(c10::complex<double>(beta.real(), beta.imag())), idxBase);
}

at::Tensor SparseSampledAddmmCsrPy(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
                                   const std::complex<double>& beta, const std::complex<double>& alpha)
{
    return SparseSampledAddmmCsr(self, mat1, mat2, c10::Scalar(c10::complex<double>(beta.real(), beta.imag())),
                                 c10::Scalar(c10::complex<double>(alpha.real(), alpha.imag())));
}

// Native BSR descriptors own preprocessing; dense batch count one broadcasts.
void CheckBsrOperands(const at::Tensor& blockRowPtrIn, const at::Tensor& blockColIndIn, const at::Tensor& values,
                      const at::Tensor& xIn, const at::Tensor& yIn, int64_t base, int64_t bd)
{
    TORCH_CHECK(base == 0 || base == 1, "BSR base must be 0 or 1");
    TORCH_CHECK(bd >= 2 && bd <= 128 && (bd & (bd - 1)) == 0, "invalid BSR block size");
    const int64_t safeBd = bd == 0 ? 1 : bd;
    TORCH_CHECK(values.is_privateuseone() && values.is_contiguous() && values.dim() == 3 && values.size(1) == safeBd &&
                    values.size(2) == safeBd,
                "invalid BSR values layout");
    TORCH_CHECK(xIn.dim() == 3 && yIn.dim() == 3 && xIn.size(0) > 0 && yIn.size(0) > 0 && xIn.size(2) == yIn.size(1),
                "invalid BSR dense batch shapes");
    for (const auto& tensor : {blockRowPtrIn, blockColIndIn, xIn, yIn}) {
        TORCH_CHECK(tensor.device() == values.device(), "BSR operands must share one NPU device");
    }
}

void CheckBsrDimensions(const at::Tensor& blockRowPtrIn, const at::Tensor& blockColIndIn, const at::Tensor& values,
                        const at::Tensor& xIn, const at::Tensor& yIn, int64_t safeBd, int64_t batch)
{
    const int64_t m = xIn.size(1), k = xIn.size(2), n = yIn.size(2);
    const int64_t blockDim = safeBd == 0 ? 1 : safeBd;
    TORCH_CHECK(batch >= 1 && batch <= 65535 && (xIn.size(0) == 1 || xIn.size(0) == batch) &&
                    (yIn.size(0) == 1 || yIn.size(0) == batch),
                "incompatible BSR batch counts");
    TORCH_CHECK(m % blockDim == 0 && n % blockDim == 0 && blockRowPtrIn.numel() == m / blockDim + 1,
                "BSR dimensions must match block rows");
    TORCH_CHECK(m <= kInt32Max && n <= kInt32Max && k <= kInt32Max && values.numel() <= kInt32Max,
                "BSR dimensions exceed int32 capacity");
    TORCH_CHECK(blockRowPtrIn.dim() == 1 && blockColIndIn.dim() == 1 && blockColIndIn.numel() == values.size(0),
                "invalid BSR index shapes");
    TORCH_CHECK(blockRowPtrIn.scalar_type() == at::kInt && blockColIndIn.scalar_type() == at::kInt,
                "native BSR requires int32 indices");
}

void CheckBsrTypes(const at::Tensor& x, const at::Tensor& y, aclDataType xType, aclDataType cType)
{
    TORCH_CHECK(x.scalar_type() == y.scalar_type() &&
                    ((xType == ACL_FLOAT && cType == ACL_FLOAT) || (xType == ACL_COMPLEX64 && cType == ACL_COMPLEX64) ||
                     (xType == ACL_FLOAT16 && (cType == ACL_FLOAT16 || cType == ACL_FLOAT)) ||
                     (xType == ACL_BF16 && (cType == ACL_BF16 || cType == ACL_FLOAT))),
                "unsupported BSR input/output dtype pair");
}

at::Tensor PrepareBsrOutput(at::Tensor& blockRowPtr, at::Tensor& colInd, const at::Tensor& values, int64_t batch,
                            int64_t safeBd)
{
    auto result = batch == 1 ? values.unsqueeze(0) :
                               at::empty({batch, values.size(0), safeBd, safeBd}, values.options());
    if (batch > 1) {
        result.copy_(values.unsqueeze(0).expand_as(result));
        blockRowPtr = blockRowPtr.unsqueeze(0).repeat({batch, 1});
        colInd = colInd.unsqueeze(0).repeat({batch, 1});
    }
    return result;
}

at::Tensor SddmmBsrPatternPy(const at::Tensor& blockRowPtrIn, const at::Tensor& blockColIndIn, const at::Tensor& values,
                             const at::Tensor& xIn, const at::Tensor& yIn, const std::complex<double>& alphaValue,
                             const std::complex<double>& betaValue, int64_t base, int64_t bd, bool columnMajor,
                             int64_t batch)
{
    CheckBsrOperands(blockRowPtrIn, blockColIndIn, values, xIn, yIn, base, bd);
    const int64_t safeBd = bd == 0 ? 1 : bd;
    CheckBsrDimensions(blockRowPtrIn, blockColIndIn, values, xIn, yIn, safeBd, batch);
    const int64_t m = xIn.size(1), k = xIn.size(2), n = yIn.size(2);
    auto colInd = blockColIndIn.contiguous();
    auto blockRowPtr = blockRowPtrIn.to(at::kInt).contiguous();
    auto x = xIn.resolve_conj().resolve_neg().contiguous();
    auto y = yIn.resolve_conj().resolve_neg().contiguous();
    CheckDenseOperands(x[0], y[0], m, n, k);
    auto result = PrepareBsrOutput(blockRowPtr, colInd, values, batch, safeBd);
    const aclDataType xType = ToAclDataType(x.scalar_type());
    const aclDataType cType = ToAclDataType(values.scalar_type());
    const aclDataType compute = ComputeTypeFor(values.scalar_type());
    const HostScalar alpha = MakeScalar(c10::Scalar(c10::complex<double>(alphaValue.real(), alphaValue.imag())),
                                        compute);
    const HostScalar beta = MakeScalar(c10::Scalar(c10::complex<double>(betaValue.real(), betaValue.imag())), compute);
    CheckBsrTypes(x, y, xType, cType);
    return cann_ops_sparse::RunAclSparse(values, [&](cann_ops_sparse::AclSparseContext& context) {
        DnMatGuard dx, dy;
        SpMatGuard dc;
        ACLSPARSE_CHECK(aclsparseCreateDnMat(dx.Out(), m, k, k, x.mutable_data_ptr(), xType, ACL_SPARSE_ORDER_ROW));
        ACLSPARSE_CHECK(aclsparseCreateDnMat(dy.Out(), k, n, n, y.mutable_data_ptr(), xType, ACL_SPARSE_ORDER_ROW));
        ACLSPARSE_CHECK(aclsparseDnMatSetStridedBatch(dx.Get(), x.size(0), m * k));
        ACLSPARSE_CHECK(aclsparseDnMatSetStridedBatch(dy.Get(), y.size(0), k * n));
        ACLSPARSE_CHECK(aclsparseCreateBsr(dc.Out(), m / safeBd, n / safeBd, values.size(0), safeBd, safeBd,
                                           blockRowPtr.mutable_data_ptr(), colInd.mutable_data_ptr(),
                                           result.mutable_data_ptr(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                           base ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO, cType,
                                           columnMajor ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW));
        ACLSPARSE_CHECK(aclsparseBsrSetStridedBatch(dc.Get(), batch, m / safeBd + 1, values.size(0), values.numel()));
        constexpr auto op = ACL_SPARSE_OP_NON_TRANSPOSE;
        constexpr auto alg = ACL_SPARSE_SDDMM_ALG_DEFAULT;
        size_t bytes = 0;
        ACLSPARSE_CHECK(aclsparseSDDMMBufferSize(context.handle(), op, op, alpha.Data(), dx.Get(), dy.Get(),
                                                 beta.Data(), dc.Get(), compute, alg, &bytes));
        cann_ops_sparse::AclSparseWorkspace buffer(context, values, bytes);
        ACLSPARSE_CHECK(sddmm_call_local(context.handle(), op, op, alpha.Data(), dx.Get(), dy.Get(), beta.Data(),
                                         dc.Get(), compute, alg, buffer.data()));
        context.RecordTensors(blockRowPtr, colInd, values, x, y, result);
        return result;
    });
}

std::shared_ptr<PreparedCsr> PrepareSddmmCsrPy(const at::Tensor& rowPtr, const at::Tensor& colInd,
                                               const at::Tensor& values, const at::Tensor& x, const at::Tensor& y,
                                               const std::complex<double>& alpha, const std::complex<double>& beta,
                                               int64_t idxBase)
{
    return std::make_shared<PreparedCsr>(rowPtr, colInd, values, x, y,
                                         c10::Scalar(c10::complex<double>(alpha.real(), alpha.imag())),
                                         c10::Scalar(c10::complex<double>(beta.real(), beta.imag())), idxBase);
}

// ---------------------------------------------------------------------------
// torch.ops.ops_sparse_test.sddmm_npu
// ---------------------------------------------------------------------------
//
// Native dispatcher adapter. CSR single-batch calls reuse versioned bindings;
// CSR batches and BSR calls execute through native three-stage bindings.

namespace {

// Retain the exact operands and their metadata. An allocator reusing an address
// must not make a different tensor look like the previously prepared pattern.
struct CachedCsrBinding {
    at::Tensor source;
    void* pointer = nullptr;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    at::ScalarType dtype = at::kFloat;
    c10::Device device{c10::DeviceType::CPU};
    uint32_t version = 0;
    bool hasVersion = false;

    void Bind(const at::Tensor& tensor)
    {
        source = tensor;
        pointer = tensor.data_ptr();
        sizes = tensor.sizes().vec();
        strides = tensor.strides().vec();
        dtype = tensor.scalar_type();
        device = tensor.device();
        const auto& counter = tensor.unsafeGetTensorImpl()->version_counter();
        hasVersion = counter.enabled();
        version = hasVersion ? counter.current_version() : 0;
    }

    bool Matches(const at::Tensor& tensor, bool trackVersion) const
    {
        if (!source.defined() || source.unsafeGetTensorImpl() != tensor.unsafeGetTensorImpl() ||
            pointer != tensor.data_ptr() || !tensor.sizes().equals(sizes) || !tensor.strides().equals(strides) ||
            dtype != tensor.scalar_type() || device != tensor.device()) {
            return false;
        }
        if (!trackVersion) {
            return true; // C values are the mutable in/out buffer, not a pattern dependency.
        }
        // Snapshot the version at preparation, rather than comparing two live
        // aliases of the same counter. Inference tensors have no counter, so
        // conservatively prepare again rather than reuse stale numeric data.
        const auto& counter = tensor.unsafeGetTensorImpl()->version_counter();
        return hasVersion && counter.enabled() && version == counter.current_version();
    }
};

struct PreparedCsrCache {
    CachedCsrBinding rowPtr;
    CachedCsrBinding colInd;
    CachedCsrBinding values;
    CachedCsrBinding x;
    CachedCsrBinding y;
    double alpha = 0.0;
    double beta = 0.0;
    int64_t base = 0;
    int64_t streamId = 0;
    std::shared_ptr<PreparedCsr> prepared;
};

thread_local PreparedCsrCache g_preparedCsr;

PreparedCsrCache BuildCsrCache(const at::Tensor& rowPtr, const at::Tensor& colInd, const at::Tensor& values,
                               const at::Tensor& x, const at::Tensor& y, double alpha, double beta, int64_t base,
                               int64_t streamId)
{
    // x[0]/y[0] create new TensorImpls on every call. Cache the original
    // three-dimensional operands and make those views only on a miss.
    // Otherwise identical inputs would repeat all three API stages.
    PreparedCsrCache next;
    const auto xMatrix = x[0].contiguous();
    auto yMatrix = y[0];
    // The versioned binding owns dense preprocessing too. For real CSR
    // reductions, keep Y's K dimension contiguous across executions instead
    // of transposing the same immutable matrix inside every C API Execute.
    // No result is cached: changed operands rebuild this state, and every
    // invocation still computes against the current mutable C values.
    const bool realType = yMatrix.scalar_type() == at::kHalf || yMatrix.scalar_type() == at::kBFloat16 ||
                          yMatrix.scalar_type() == at::kFloat;
    if (realType && yMatrix.size(0) >= 32 && yMatrix.size(1) >= 32 &&
        colInd.numel() >= (yMatrix.size(1) - 1) / yMatrix.size(0) + 1) {
        yMatrix = yMatrix.transpose(0, 1).contiguous().transpose(0, 1);
    } else {
        yMatrix = yMatrix.contiguous();
    }
    next.prepared = std::make_shared<PreparedCsr>(rowPtr, colInd, values, xMatrix, yMatrix, c10::Scalar(alpha),
                                                  c10::Scalar(beta), base);
    next.rowPtr.Bind(rowPtr);
    next.colInd.Bind(colInd);
    next.values.Bind(values);
    next.x.Bind(x);
    next.y.Bind(y);
    next.alpha = alpha;
    next.beta = beta;
    next.base = base;
    next.streamId = streamId;
    return next;
}

at::Tensor ExecuteCsrBatch(const at::Tensor& rowPtr, const at::Tensor& colInd, const at::Tensor& values,
                           const at::Tensor& x, const at::Tensor& y, double alpha, double beta, int64_t base,
                           int64_t batch)
{
    std::vector<at::Tensor> outputs;
    outputs.reserve(static_cast<size_t>(batch));
    for (int64_t index = 0; index < batch; ++index) {
        auto output = values.clone();
        auto prepared = std::make_shared<PreparedCsr>(rowPtr, colInd, output, x[x.size(0) == 1 ? 0 : index],
                                                      y[y.size(0) == 1 ? 0 : index], c10::Scalar(alpha),
                                                      c10::Scalar(beta), base);
        outputs.push_back(prepared->Execute());
    }
    return at::stack(outputs);
}

at::Tensor SddmmNpuEntry(std::string fmt, const at::Tensor& rowPtr, const at::Tensor& colInd, const at::Tensor& values,
                         const at::Tensor& x, const at::Tensor& y, double alpha, double beta, int64_t base,
                         int64_t blockDim, std::string direction, int64_t batch)
{
    if (fmt == "bsr") {
        TORCH_CHECK(direction == "row" || direction == "col", "BSR direction must be row or col");
        return SddmmBsrPatternPy(rowPtr, colInd, values, x, y, {alpha, 0.0}, {beta, 0.0}, base, blockDim,
                                 direction == "col", batch);
    }
    TORCH_CHECK(fmt == "csr", "unsupported sparse format: ", fmt);
    TORCH_CHECK(x.dim() == 3 && y.dim() == 3, "SDDMM dense inputs must be three-dimensional");
    TORCH_CHECK(batch >= 1 && batch <= 65535, "invalid SDDMM batch count");
    TORCH_CHECK((x.size(0) == 1 || x.size(0) == batch) && (y.size(0) == 1 || y.size(0) == batch),
                "incompatible SDDMM batch counts");
    if (batch > 1) {
        return ExecuteCsrBatch(rowPtr, colInd, values, x, y, alpha, beta, base, batch);
    }

    TORCH_CHECK(values.is_privateuseone(), "SDDMM values must live on an Ascend NPU");
    const auto streamId = c10_npu::getCurrentNPUStream(values.device().index()).id();
    const bool hit = g_preparedCsr.prepared && g_preparedCsr.streamId == streamId && g_preparedCsr.alpha == alpha &&
                     g_preparedCsr.beta == beta && g_preparedCsr.base == base &&
                     g_preparedCsr.rowPtr.Matches(rowPtr, true) && g_preparedCsr.colInd.Matches(colInd, true) &&
                     g_preparedCsr.values.Matches(values, false) && g_preparedCsr.x.Matches(x, true) &&
                     g_preparedCsr.y.Matches(y, true);

    if (!hit) {
        g_preparedCsr = BuildCsrCache(rowPtr, colInd, values, x, y, alpha, beta, base, streamId);
    }

    PreparedCsr::EnqueueBound(g_preparedCsr.prepared);
    return values.unsqueeze(0);
}

} // namespace

TORCH_LIBRARY(ops_sparse_test, m)
{
    m.def("sddmm_npu(str fmt, Tensor row_ptr, Tensor col_ind, Tensor values, Tensor x, "
          "Tensor y, float alpha, float beta, int base, int block_dim, str direction, "
          "int batch) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_sparse_test, PrivateUse1, m) { m.impl("sddmm_npu", &SddmmNpuEntry); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "tensor_version",
        [](const at::Tensor& tensor) { return tensor.unsafeGetTensorImpl()->version_counter().current_version(); },
        "Read the version used by the native prepared-binding cache");
    pybind11::class_<PreparedCsr, std::shared_ptr<PreparedCsr>>(m, "PreparedSddmmCsr")
        .def("execute", &PreparedCsr::Execute)
        .def_property_readonly("workspace_bytes", &PreparedCsr::WorkspaceBytes);
    m.def("prepare_sddmm_csr", &PrepareSddmmCsrPy,
          "Prepare reusable ACLSparse descriptors, workspace and preprocessing for CSR SDDMM");
    m.def("sddmm_csr", &SddmmCsrPy, "ACLSparse three-stage SDDMM over CSR");
    m.def("sparse_sampled_addmm_csr", &SparseSampledAddmmCsrPy, "aten::sparse_sampled_addmm backend for NPU CSR");
    m.def("sddmm_bsr_pattern", &SddmmBsrPatternPy, "Native BSR three-stage API with explicit output batch count");
}
