/*
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 *
 * Optional PyTorch/torch_npu bridge for the A5 aclSPARSE SpMV path.  The
 * bridge deliberately contains no CPU implementation: every registered NPU
 * entry validates PrivateUse1 tensors and submits the existing aclsparse
 * three-stage API on the current NPU stream.
 */

#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

constexpr int64_t kIndexBaseZero = 0;
constexpr int64_t kIndexBaseOne = 1;

[[noreturn]] void ThrowStatus(const char *api, aclsparseStatus_t status)
{
    TORCH_CHECK(false, api, " failed with aclSPARSE status ", static_cast<int>(status));
}

void CheckStatus(const char *api, aclsparseStatus_t status)
{
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        ThrowStatus(api, status);
    }
}

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be on an NPU device");
}

void CheckSameDevice(const at::Tensor &first, const at::Tensor &second,
                     const char *firstName, const char *secondName)
{
    TORCH_CHECK(first.device() == second.device(), firstName, " and ", secondName,
                " must be on the same NPU device");
}

void CheckVector(const at::Tensor &tensor, const char *name)
{
    CheckNpuTensor(tensor, name);
    TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckIndex(const at::Tensor &tensor, const char *name)
{
    CheckVector(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kInt,
                name, " must use int32 indices for A5 SpMV");
}

aclsparseOperation_t ParseOperation(const std::string &trans)
{
    TORCH_CHECK(trans == "N" || trans == "T" || trans == "H",
                "SpMV trans must be 'N', 'T', or 'H', got '", trans, "'");
    if (trans == "H") {
        return ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    }
    return trans == "T" ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
}

aclsparseIndexBase_t ParseBase(int64_t base)
{
    TORCH_CHECK(base == kIndexBaseZero || base == kIndexBaseOne,
                "SpMV index base must be 0 or 1, got ", base);
    return base == kIndexBaseOne ? ACL_SPARSE_INDEX_BASE_ONE
                                 : ACL_SPARSE_INDEX_BASE_ZERO;
}

struct ScalarStorage {
    int32_t integer = 0;
    float real = 0.0f;
    aclsparseComplex complex{0.0f, 0.0f};
};

struct ScalarPair {
    ScalarStorage alpha;
    ScalarStorage beta;
    const void *alphaPtr = nullptr;
    const void *betaPtr = nullptr;
    aclDataType computeType = ACL_FLOAT;
};

ScalarPair MakeScalars(const at::Scalar &alpha, const at::Scalar &beta,
                       aclDataType computeType)
{
    ScalarPair result;
    result.computeType = computeType;
    if (computeType == ACL_INT32) {
        TORCH_CHECK(alpha.isIntegral() && beta.isIntegral(),
                    "INT32 SpMV alpha/beta must be integral scalars");
        const int64_t alphaValue = alpha.toLong();
        const int64_t betaValue = beta.toLong();
        TORCH_CHECK(alphaValue >= std::numeric_limits<int32_t>::min() &&
                        alphaValue <= std::numeric_limits<int32_t>::max() &&
                        betaValue >= std::numeric_limits<int32_t>::min() &&
                        betaValue <= std::numeric_limits<int32_t>::max(),
                    "INT32 SpMV alpha/beta must fit int32");
        result.alpha.integer = static_cast<int32_t>(alphaValue);
        result.beta.integer = static_cast<int32_t>(betaValue);
    } else if (computeType == ACL_COMPLEX64) {
        const auto alphaValue = alpha.toComplexDouble();
        const auto betaValue = beta.toComplexDouble();
        result.alpha.complex = {static_cast<float>(alphaValue.real()),
                                static_cast<float>(alphaValue.imag())};
        result.beta.complex = {static_cast<float>(betaValue.real()),
                               static_cast<float>(betaValue.imag())};
    } else {
        result.alpha.real = static_cast<float>(alpha.toDouble());
        result.beta.real = static_cast<float>(beta.toDouble());
    }
    return result;
}

class SpmvHandle {
public:
    explicit SpmvHandle(aclrtStream stream)
    {
        CheckStatus("aclsparseSetStream", aclsparseSetStream(Get(), stream));
        CheckStatus("aclsparseSetPointerMode",
                    aclsparseSetPointerMode(Get(), ACL_SPARSE_POINTER_MODE_HOST));
    }

    aclsparseHandle_t Get()
    {
        return reinterpret_cast<aclsparseHandle_t>(&context_);
    }

    SpmvHandle(const SpmvHandle &) = delete;
    SpmvHandle &operator=(const SpmvHandle &) = delete;

private:
    // This in-tree bridge uses SpMV's explicit Tensor workspace exclusively.
    // The host context owns no device memory and is not retained by kernels;
    // its stack lifetime needs neither aclsparseCreate nor aclsparseDestroy.
    aclsparseContext context_;
};

struct DescriptorGuard {
    aclsparseConstSpMatDescr_t mat = nullptr;
    aclsparseConstDnVecDescr_t x = nullptr;
    aclsparseDnVecDescr_t y = nullptr;

    ~DescriptorGuard()
    {
        if (y != nullptr) {
            (void)aclsparseDestroyDnVec(y);
        }
        if (x != nullptr) {
            (void)aclsparseDestroyDnVec(x);
        }
        if (mat != nullptr) {
            (void)aclsparseDestroySpMat(mat);
        }
    }
};

at::Tensor AllocateWorkspace(const at::Tensor &values, size_t bytes,
                             const c10_npu::NPUStream &stream)
{
    TORCH_CHECK(bytes > 0, "aclsparseSpMV returned an empty workspace");
    TORCH_CHECK(bytes <= std::numeric_limits<size_t>::max() - 63U,
                "SpMV workspace size overflows size_t");
    TORCH_CHECK(bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) - 63U,
                "SpMV workspace is too large for a Tensor dimension");
    auto options = values.options().layout(at::kStrided).dtype(at::kByte);
    // The extra bytes let us repair an allocator alignment that is weaker than
    // the public 64-byte aclSPARSE workspace contract.
    at::Tensor raw = at::empty({static_cast<int64_t>(bytes + 63U)}, options);
    const auto address = reinterpret_cast<uintptr_t>(raw.data_ptr());
    const size_t offset = (64U - (address & 63U)) & 63U;
    at::Tensor aligned = raw.narrow(0, static_cast<int64_t>(offset),
                                    static_cast<int64_t>(bytes));
    TORCH_CHECK((reinterpret_cast<uintptr_t>(aligned.data_ptr()) & 63U) == 0U,
                "failed to align SpMV workspace");
    // Tensor destruction is allowed after this function returns; record the
    // stream so torch_npu's caching allocator inserts the required event.
    c10_npu::NPUCachingAllocator::recordStream(raw.storage().data_ptr(), stream);
    return aligned;
}

void ValidateCsr(const at::Tensor &rowPtr, const at::Tensor &colInd,
                 const at::Tensor &values, const at::Tensor &x,
                 const at::Tensor &y, aclsparseOperation_t op,
                 aclsparseIndexBase_t base)
{
    CheckIndex(rowPtr, "row_ptr");
    CheckIndex(colInd, "col_ind");
    CheckVector(values, "values");
    CheckVector(x, "x");
    CheckVector(y, "y");
    CheckSameDevice(rowPtr, colInd, "row_ptr", "col_ind");
    CheckSameDevice(rowPtr, values, "row_ptr", "values");
    CheckSameDevice(rowPtr, x, "row_ptr", "x");
    CheckSameDevice(rowPtr, y, "row_ptr", "y");
    // The kernel updates Y in place and treats all other tensors as read-only.
    // Reject shared storage up front, including sliced/offset aliases that
    // would not necessarily have identical data_ptr() values.
    TORCH_CHECK(!rowPtr.is_alias_of(colInd) && !rowPtr.is_alias_of(values) &&
                    !rowPtr.is_alias_of(x) && !rowPtr.is_alias_of(y) &&
                    !colInd.is_alias_of(values) && !colInd.is_alias_of(x) &&
                    !colInd.is_alias_of(y) && !values.is_alias_of(x) &&
                    !values.is_alias_of(y) && !x.is_alias_of(y),
                "SpMV inputs and output must not share storage");
    TORCH_CHECK(rowPtr.numel() > 0, "row_ptr must contain at least one entry");
    const bool transpose = op == ACL_SPARSE_OP_TRANSPOSE ||
                           op == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    const int64_t rows = transpose ? x.numel() : y.numel();
    const int64_t cols = transpose ? y.numel() : x.numel();
    TORCH_CHECK(rows < std::numeric_limits<int64_t>::max() &&
                    rowPtr.numel() == rows + 1,
                "row_ptr length must equal matrix rows + 1");
    TORCH_CHECK(colInd.numel() == values.numel(),
                "col_ind and values must have identical nnz");
    TORCH_CHECK(rows >= 0 && cols >= 0 &&
                    (rows == 0 || cols <= std::numeric_limits<int64_t>::max() / rows),
                "SpMV matrix shape overflows int64");
    // Repeated CSR coordinates are valid and may make nnz larger than rows*cols.
    // The aclsparse layer enforces the I32/shape/workspace limits without
    // imposing a uniqueness assumption on the sparse pattern.
    (void)base;
}

struct SpmvTypes {
    aclDataType value = ACL_FLOAT;
    aclDataType compute = ACL_FLOAT;
};

SpmvTypes ResolveSpmvTypes(const at::Tensor &values, const at::Tensor &y)
{
    const auto valueType = values.scalar_type();
    const auto outputType = y.scalar_type();
    if (valueType == at::kChar && outputType == at::kInt) {
        return {ACL_INT8, ACL_INT32};
    }
    if (valueType == at::kChar && outputType == at::kFloat) {
        return {ACL_INT8, ACL_FLOAT};
    }
    if (valueType == at::kHalf) {
        TORCH_CHECK(outputType == at::kHalf || outputType == at::kFloat,
                    "FP16 SpMV output must be float16 or float32");
        return {ACL_FLOAT16, ACL_FLOAT};
    }
    if (valueType == at::kBFloat16) {
        TORCH_CHECK(outputType == at::kBFloat16 || outputType == at::kFloat,
                    "BF16 SpMV output must be bfloat16 or float32");
        return {ACL_BF16, ACL_FLOAT};
    }
    if (valueType == at::kFloat && outputType == at::kFloat) {
        return {ACL_FLOAT, ACL_FLOAT};
    }
    if (valueType == at::kComplexFloat && outputType == at::kComplexFloat) {
        return {ACL_COMPLEX64, ACL_COMPLEX64};
    }
    if (valueType == at::kFloat && outputType == at::kComplexFloat) {
        return {ACL_FLOAT, ACL_COMPLEX64};
    }
    TORCH_CHECK(false, "unsupported A5 SpMV dtype combination");
    return {};
}

aclDataType ToAclDataType(at::ScalarType type)
{
    switch (type) {
        case at::kChar: return ACL_INT8;
        case at::kInt: return ACL_INT32;
        case at::kHalf: return ACL_FLOAT16;
        case at::kBFloat16: return ACL_BF16;
        case at::kFloat: return ACL_FLOAT;
        case at::kComplexFloat: return ACL_COMPLEX64;
        default: TORCH_CHECK(false, "unsupported A5 SpMV tensor dtype");
    }
    return ACL_FLOAT;
}

void BindScalarPointers(ScalarPair *scalars)
{
    if (scalars->computeType == ACL_INT32) {
        scalars->alphaPtr = &scalars->alpha.integer;
        scalars->betaPtr = &scalars->beta.integer;
    } else if (scalars->computeType == ACL_COMPLEX64) {
        scalars->alphaPtr = &scalars->alpha.complex;
        scalars->betaPtr = &scalars->beta.complex;
    } else {
        scalars->alphaPtr = &scalars->alpha.real;
        scalars->betaPtr = &scalars->beta.real;
    }
}

void CreateSpmvDescriptors(
    DescriptorGuard *descriptors, const at::Tensor &rowPtr,
    const at::Tensor &colInd, const at::Tensor &values, const at::Tensor &x,
    const at::Tensor &y, aclsparseOperation_t op,
    aclsparseIndexBase_t indexBase, const SpmvTypes &types)
{
    const bool transpose = op == ACL_SPARSE_OP_TRANSPOSE ||
                           op == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    const int64_t rows = transpose ? x.numel() : y.numel();
    const int64_t cols = transpose ? y.numel() : x.numel();
    CheckStatus("aclsparseCreateConstCsr", aclsparseCreateConstCsr(
        &descriptors->mat, rows, cols, colInd.numel(), rowPtr.data_ptr(),
        colInd.data_ptr(), values.data_ptr(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_32I, indexBase, types.value));
    CheckStatus("aclsparseCreateConstDnVec", aclsparseCreateConstDnVec(
        &descriptors->x, x.numel(), x.data_ptr(), ToAclDataType(x.scalar_type())));
    CheckStatus("aclsparseCreateDnVec", aclsparseCreateDnVec(
        &descriptors->y, y.numel(), y.mutable_data_ptr(),
        ToAclDataType(y.scalar_type())));
}

void ExecuteSpmv(aclsparseHandle_t handle, const at::Tensor &values, aclsparseOperation_t op,
                 const ScalarPair &scalars, const DescriptorGuard &descriptors,
                 const c10_npu::NPUStream &stream)
{
    size_t workspaceBytes = 0U;
    CheckStatus("aclsparseSpMVGetBufferSize", aclsparseSpMVGetBufferSize(
        handle, op, scalars.alphaPtr, descriptors.mat, descriptors.x,
        scalars.betaPtr, descriptors.y, scalars.computeType,
        ACL_SPARSE_SPMV_ALG_DEFAULT, &workspaceBytes));
    at::Tensor workspace = AllocateWorkspace(values, workspaceBytes, stream);
    CheckStatus("aclsparseSpMVPreprocess", aclsparseSpMVPreprocess(
        handle, op, scalars.alphaPtr, descriptors.mat, descriptors.x,
        scalars.betaPtr, descriptors.y, scalars.computeType,
        ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.data_ptr()));
    CheckStatus("aclsparseSpMV", aclsparseSpMV(
        handle, op, scalars.alphaPtr, descriptors.mat, descriptors.x,
        scalars.betaPtr, descriptors.y, scalars.computeType,
        ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.data_ptr()));
}

at::Tensor RunSpmv(const at::Tensor &rowPtr, const at::Tensor &colInd,
                   const at::Tensor &values, const at::Tensor &x,
                   const at::Tensor &y, const at::Scalar &alpha,
                   const at::Scalar &beta, const std::string &trans,
                   int64_t base)
{
    const auto op = ParseOperation(trans);
    const auto indexBase = ParseBase(base);
    ValidateCsr(rowPtr, colInd, values, x, y, op, indexBase);
    TORCH_CHECK(values.scalar_type() == x.scalar_type(),
                "values and x must have the same dtype");
    const SpmvTypes types = ResolveSpmvTypes(values, y);
    ScalarPair scalars = MakeScalars(alpha, beta, types.compute);
    BindScalarPointers(&scalars);
    auto stream = c10_npu::getCurrentNPUStream();
    // Direct ACL kernel launches bypass torch_npu's queued dispatcher. Drain
    // pending tensor copies on the current stream first so input/reset writes
    // cannot execute after SpMV and overwrite its output.
    SpmvHandle handle(stream.stream(true));

    DescriptorGuard descriptors;
    CreateSpmvDescriptors(&descriptors, rowPtr, colInd, values, x, y,
                          op, indexBase, types);
    ExecuteSpmv(handle.Get(), values, op, scalars, descriptors, stream);
    return y;
}

at::Tensor SpmvNpu(const at::Tensor &rowPtr, const at::Tensor &colInd,
                   const at::Tensor &values, const at::Tensor &x,
                   const at::Tensor &y, const at::Scalar &alpha,
                   const at::Scalar &beta, const std::string &trans,
                   int64_t base)
{
    return RunSpmv(rowPtr, colInd, values, x, y, alpha, beta, trans, base);
}

at::Tensor SpmvMv(const at::Tensor &matrix, const at::Tensor &vector)
{
    CheckNpuTensor(matrix, "mat");
    CheckVector(vector, "vec");
    TORCH_CHECK(matrix.layout() == at::kSparseCsr && matrix.dim() == 2,
                "aten::mv SpMV bridge requires a two-dimensional CSR matrix");
    CheckSameDevice(matrix, vector, "mat", "vec");
    TORCH_CHECK(matrix.size(1) == vector.numel(),
                "SpMV matrix columns must equal vector length, got ",
                matrix.size(1), " columns and ", vector.numel(), " elements");
    const auto outputType = matrix.scalar_type() == at::kChar ? at::kInt : matrix.scalar_type();
    const int64_t outputSize = matrix.size(0);
    at::Tensor output = at::empty({outputSize}, matrix.options().layout(at::kStrided).dtype(outputType));
    return RunSpmv(matrix.crow_indices(), matrix.col_indices(), matrix.values(), vector,
                   output, at::Scalar(1), at::Scalar(0), "N", 0);
}

}  // namespace

TORCH_LIBRARY(ops_sparse, m)
{
    m.def("spmv(Tensor mat, Tensor vec) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_sparse, PrivateUse1, m)
{
    m.impl("spmv", TORCH_FN(SpmvMv));
}

// A CSR tensor carries the layout-specific dispatch key in recent PyTorch
// releases. Register the explicit custom entry on that key as well, so the
// test hook remains usable when its first argument is a CSR tensor rather than
// a strided NPU tensor.
TORCH_LIBRARY_IMPL(ops_sparse, SparseCsrPrivateUse1, m)
{
    m.impl("spmv", TORCH_FN(SpmvMv));
}

TORCH_LIBRARY(ops_sparse_test, m)
{
    m.def("spmv_npu(Tensor row_ptr, Tensor col_ind, Tensor values, Tensor x, Tensor y, Scalar alpha, Scalar beta, str trans, int base) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_sparse_test, PrivateUse1, m)
{
    m.impl("spmv_npu", TORCH_FN(SpmvNpu));
}

// Sparse CSR tensors carry the layout-specific dispatch key on PyTorch 2.7+
// and torch_npu 26+.  Register the same implementation there so torch.mv does
// not fall through to a generic (or CPU) kernel for an NPU CSR tensor.
TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m)
{
    m.impl("mv", TORCH_FN(SpmvMv));
}
