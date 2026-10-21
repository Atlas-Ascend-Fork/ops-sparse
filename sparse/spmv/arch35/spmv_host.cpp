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
 */

#include <cstdint>
#include <limits>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "spmv.h"
#include "spmv_kernel.h"

namespace {

uint32_t SpmvBlockDim()
{
    const uint32_t cores = GetAivCoreCount();
    return cores;
}

bool IsScalarType(aclDataType computeType)
{
    return computeType == ACL_INT32 || computeType == ACL_FLOAT ||
           computeType == ACL_COMPLEX64;
}

bool IsTransposeOperation(aclsparseOperation_t opA)
{
    return opA == ACL_SPARSE_OP_TRANSPOSE ||
           opA == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
}

bool IsSupportedCombo(aclDataType valueType, aclDataType xType,
                      aclDataType yType, aclDataType computeType)
{
    if (valueType != xType) {
        return false;
    }
    if (computeType == ACL_INT32) {
        return valueType == ACL_INT8 && yType == ACL_INT32;
    }
    if (computeType == ACL_COMPLEX64) {
        return (valueType == ACL_COMPLEX64 && yType == ACL_COMPLEX64) ||
               (valueType == ACL_FLOAT && yType == ACL_COMPLEX64);
    }
    if (computeType != ACL_FLOAT) {
        return false;
    }
    return (valueType == ACL_INT8 && yType == ACL_FLOAT) ||
           (valueType == ACL_FLOAT16 &&
               (yType == ACL_FLOAT16 || yType == ACL_FLOAT)) ||
           (valueType == ACL_BF16 &&
               (yType == ACL_BF16 || yType == ACL_FLOAT)) ||
           (valueType == ACL_FLOAT && yType == ACL_FLOAT);
}

bool IsSupportedAlg(aclsparseSpMVAlg_t alg)
{
    return alg == ACL_SPARSE_SPMV_ALG_DEFAULT ||
           alg == ACL_SPARSE_SPMV_CSR_ALG1 ||
           alg == ACL_SPARSE_SPMV_CSR_ALG2;
}

int32_t ValueTypeId(aclDataType type)
{
    switch (type) {
        case ACL_INT8: return SPMV35_VALUE_I8;
        case ACL_FLOAT16: return SPMV35_VALUE_F16;
        case ACL_BF16: return SPMV35_VALUE_BF16;
        case ACL_FLOAT: return SPMV35_VALUE_F32;
        case ACL_COMPLEX64: return SPMV35_VALUE_C64;
        default: return -1;
    }
}

int32_t OutputTypeId(aclDataType type)
{
    switch (type) {
        case ACL_INT32: return SPMV35_OUTPUT_I32;
        case ACL_FLOAT16: return SPMV35_OUTPUT_F16;
        case ACL_BF16: return SPMV35_OUTPUT_BF16;
        case ACL_FLOAT: return SPMV35_OUTPUT_F32;
        case ACL_COMPLEX64: return SPMV35_OUTPUT_C64;
        default: return -1;
    }
}

int32_t ComputeTypeId(aclDataType type)
{
    switch (type) {
        case ACL_INT32: return SPMV35_COMPUTE_I32;
        case ACL_FLOAT: return SPMV35_COMPUTE_F32;
        case ACL_COMPLEX64: return SPMV35_COMPUTE_C64;
        default: return -1;
    }
}

aclsparseStatus_t ValidateHandle(aclsparseHandle_t handle)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpMV", "handle must be non-null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    // The public handle is opaque.  Check its private signature before any
    // field access so stale or foreign handles fail deterministically.
    if (SpmvArch35ToHandle(handle)->signature != kAclSparseHandleSignature) {
        OP_LOGE("aclsparseSpMV", "invalid handle signature; expected an aclSPARSE handle");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (SpmvArch35ToHandle(handle)->stream == nullptr) {
        OP_LOGE("aclsparseSpMV", "handle stream must be set before use");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

size_t ScalarAlignment(aclDataType computeType)
{
    switch (computeType) {
        case ACL_INT32:
            return alignof(int32_t);
        case ACL_COMPLEX64:
            return alignof(aclsparseComplex);
        case ACL_FLOAT:
            return alignof(float);
        default:
            return 0U;
    }
}

aclsparseStatus_t ValidatePointerMode(aclsparseHandle_t handle,
                                      const void *alpha, const void *beta,
                                      aclDataType computeType)
{
    aclsparsePointerMode_t mode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseStatus_t st = aclsparseGetPointerMode(handle, &mode);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (mode != ACL_SPARSE_POINTER_MODE_HOST && mode != ACL_SPARSE_POINTER_MODE_DEVICE) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const size_t alignment = ScalarAlignment(computeType);
    if (alignment == 0U || reinterpret_cast<uintptr_t>(alpha) % alignment != 0U ||
        reinterpret_cast<uintptr_t>(beta) % alignment != 0U) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmvDescriptorSetup(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg)
{
    aclsparseStatus_t st = ValidateHandle(handle);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (matA == nullptr || vecX == nullptr || vecY == nullptr) {
        OP_LOGE("aclsparseSpMV", "matrix and vector descriptors must be non-null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!SpmvArch35DescriptorsValid(matA, vecX, vecY)) {
        OP_LOGE("aclsparseSpMV", "invalid descriptor signature; expected matrix and dense vector descriptors");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (SpmvBlockDim() == 0U) {
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE && !IsTransposeOperation(opA)) {
        OP_LOGE("aclsparseSpMV", "op=%d; expected N/T/H", static_cast<int>(opA));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!IsScalarType(computeType) || !IsSupportedAlg(alg)) {
        OP_LOGE("aclsparseSpMV", "computeType=%d alg=%d; expected INT32/FLOAT/COMPLEX64 and DEFAULT/CSR_ALG1/CSR_ALG2",
                static_cast<int>(computeType), static_cast<int>(alg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    st = ValidatePointerMode(handle, alpha, beta, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool SpmvStrideFits(uint64_t length, int64_t stride, size_t elementBytes)
{
    if (stride <= 0 || elementBytes == 0U) {
        return false;
    }
    const uint64_t maxOffset = static_cast<uint64_t>(
        std::numeric_limits<int64_t>::max()) / elementBytes;
    return length <= 1U ||
           static_cast<uint64_t>(stride) <= maxOffset / (length - 1U);
}

size_t SpmvElementBytes(aclDataType type)
{
    return type == ACL_INT8 ? 1U
         : (type == ACL_FLOAT16 || type == ACL_BF16) ? 2U
         : type == ACL_COMPLEX64 ? 8U : 4U;
}

aclsparseStatus_t ValidateSpmvFormat(const aclsparseSpMatDescr *mat)
{
    if (mat->format != ACL_SPARSE_FORMAT_CSR ||
        mat->ptrType != ACL_SPARSE_INDEX_32I || mat->IdxType != ACL_SPARSE_INDEX_32I ||
        (mat->baseType != ACL_SPARSE_INDEX_BASE_ZERO &&
         mat->baseType != ACL_SPARSE_INDEX_BASE_ONE)) {
        OP_LOGE("aclsparseSpMV", "format=%d rowIndexType=%d colIndexType=%d base=%d; expected CSR=%d, I32=%d, base 0/1",
                static_cast<int>(mat->format), static_cast<int>(mat->ptrType),
                static_cast<int>(mat->IdxType), static_cast<int>(mat->baseType),
                static_cast<int>(ACL_SPARSE_FORMAT_CSR), static_cast<int>(ACL_SPARSE_INDEX_32I));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmvShape(const aclsparseSpMatDescr *mat)
{
    if (mat->nnz > 0U && (mat->rows == 0U || mat->cols == 0U)) {
        OP_LOGE("aclsparseSpMV", "rows=%llu cols=%llu nnz=%llu; zero dimensions require nnz=0",
                static_cast<unsigned long long>(mat->rows), static_cast<unsigned long long>(mat->cols),
                static_cast<unsigned long long>(mat->nnz));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const uint64_t maxI32 = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    const uint64_t baseOffset = mat->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1U : 0U;
    // Row offsets and column indices are consumed as signed int32_t on A5.
    // A base-one CSR needs one extra representable value for both the
    // terminal row offset and the largest stored column index.
    if (mat->rows > maxI32 || mat->cols > maxI32 ||
        mat->nnz > maxI32 - baseOffset) {
        OP_LOGE("aclsparseSpMV", "rows=%llu cols=%llu nnz=%llu; dimensions must fit INT32_MAX, nnz <= %llu",
                static_cast<unsigned long long>(mat->rows), static_cast<unsigned long long>(mat->cols),
                static_cast<unsigned long long>(mat->nnz), static_cast<unsigned long long>(maxI32 - baseOffset));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (mat->rows != 0U && mat->cols > UINT64_MAX / mat->rows) {
        OP_LOGE("aclsparseSpMV", "rows=%llu cols=%llu; product must fit uint64",
                static_cast<unsigned long long>(mat->rows), static_cast<unsigned long long>(mat->cols));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmvMatrixMetadata(
    const aclsparseSpMatDescr *mat, const aclsparseDnVecDescr *x,
    const aclsparseDnVecDescr *y, aclDataType computeType)
{
    aclsparseStatus_t status = ValidateSpmvFormat(mat);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateSpmvShape(mat);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    // Repeated coordinates are valid, so nnz is not bounded by rows*cols.
    if (!IsSupportedCombo(mat->valueType, x->valueType, y->valueType, computeType)) {
        OP_LOGE("aclsparseSpMV", "unsupported dtype combination A=%d x=%d y=%d compute=%d; see A5 dtype matrix",
                static_cast<int>(mat->valueType), static_cast<int>(x->valueType),
                static_cast<int>(y->valueType), static_cast<int>(computeType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!SpmvStrideFits(mat->rows + 1U, mat->ptrStride, sizeof(int32_t)) ||
        !SpmvStrideFits(mat->nnz, mat->idxStride, sizeof(int32_t)) ||
        !SpmvStrideFits(mat->nnz, mat->valueStride, SpmvElementBytes(mat->valueType)) ||
        !SpmvStrideFits(x->nums, x->stride, SpmvElementBytes(x->valueType)) ||
        !SpmvStrideFits(y->nums, y->stride, SpmvElementBytes(y->valueType))) {
        OP_LOGE("aclsparseSpMV", "strides row=%lld col=%lld values=%lld x=%lld y=%lld; expected positive, addressable strides",
                static_cast<long long>(mat->ptrStride), static_cast<long long>(mat->idxStride),
                static_cast<long long>(mat->valueStride), static_cast<long long>(x->stride),
                static_cast<long long>(y->stride));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool HasInvalidSpmvAliases(const aclsparseSpMatDescr *mat,
                           const aclsparseDnVecDescr *x,
                           const aclsparseDnVecDescr *y)
{
    if (mat->ptrs != nullptr &&
        (mat->ptrs == mat->idxs || mat->ptrs == mat->values ||
         mat->ptrs == x->values || mat->ptrs == y->values)) {
        return true;
    }
    if (mat->idxs != nullptr &&
        (mat->idxs == mat->values || mat->idxs == x->values ||
         mat->idxs == y->values)) {
        return true;
    }
    return (mat->values != nullptr &&
            (mat->values == x->values || mat->values == y->values)) ||
           (x->values != nullptr && x->values == y->values);
}

aclsparseStatus_t ValidateSpmvVectorSizes(
    aclsparseOperation_t opA, const aclsparseSpMatDescr *mat,
    const aclsparseDnVecDescr *x, const aclsparseDnVecDescr *y)
{
    const uint64_t xRequired = IsTransposeOperation(opA) ? mat->rows : mat->cols;
    const uint64_t yRequired = IsTransposeOperation(opA) ? mat->cols : mat->rows;
    if (x->nums != xRequired || y->nums != yRequired) {
        OP_LOGE("aclsparseSpMV", "vector dimensions x=%llu y=%llu; expected x=%llu y=%llu for op=%d",
                static_cast<unsigned long long>(x->nums), static_cast<unsigned long long>(y->nums),
                static_cast<unsigned long long>(xRequired), static_cast<unsigned long long>(yRequired),
                static_cast<int>(opA));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmvStoragePointers(
    aclsparseOperation_t opA, const aclsparseSpMatDescr *mat,
    const aclsparseDnVecDescr *x, const aclsparseDnVecDescr *y)
{
    const uint64_t xRequired = IsTransposeOperation(opA) ? mat->rows : mat->cols;
    const uint64_t yRequired = IsTransposeOperation(opA) ? mat->cols : mat->rows;
    if ((mat->rows > 0U && mat->ptrs == nullptr) ||
        (mat->nnz > 0U && (mat->idxs == nullptr || mat->values == nullptr)) ||
        (xRequired > 0U && x->values == nullptr) ||
        (yRequired > 0U && y->values == nullptr)) {
        OP_LOGE("aclsparseSpMV", "required storage is unbound: row=%p col=%p values=%p x=%p y=%p",
                mat->ptrs, mat->idxs, mat->values, x->values, y->values);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Y is updated in place while A and X are read-only. Exact pointer
    // aliases are rejected here. For device buffers, pointer equality is the
    // strongest check available without a size-bearing ABI; the caller must
    // also ensure non-overlapping ranges when pointers are offset into one
    // allocation.
    if (HasInvalidSpmvAliases(mat, x, y)) {
        OP_LOGE("aclsparseSpMV", "CSR, x and y storage must not alias");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmvWorkspace(bool requireBuffer, void *buffer)
{
    if (requireBuffer && buffer == nullptr) {
        OP_LOGE("aclsparseSpMV", "workspace must be allocated using GetBufferSize");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (buffer != nullptr && reinterpret_cast<uintptr_t>(buffer) % kSpmvArch35WorkspaceAlignment != 0U) {
        OP_LOGE("aclsparseSpMV", "workspace=%p; expected 64-byte alignment", buffer);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmvInputs(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg)
{
    aclsparseStatus_t st = ValidateSpmvDescriptorSetup(
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    auto *mat = SpmvArch35ToMat(matA);
    auto *x = SpmvArch35ToVec(vecX);
    auto *y = SpmvArch35ToVec(vecY);
    st = ValidateSpmvMatrixMetadata(mat, x, y, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return ValidateSpmvVectorSizes(opA, mat, x, y);
}

aclsparseStatus_t ValidateSpmvExecutionInputs(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, void *buffer)
{
    aclsparseStatus_t st = ValidateSpmvInputs(
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateSpmvStoragePointers(
        opA, SpmvArch35ToMat(matA), SpmvArch35ToVec(vecX), SpmvArch35ToVec(vecY));
    return st == ACL_SPARSE_STATUS_SUCCESS
        ? ValidateSpmvWorkspace(true, buffer) : st;
}

aclsparseStatus_t ReadHostScalars(aclsparseHandle_t handle, const void *alpha,
                                  const void *beta, aclDataType computeType,
                                  SpmvArch35TilingData *tiling)
{
    aclsparsePointerMode_t mode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseStatus_t st = aclsparseGetPointerMode(handle, &mode);
    if (st != ACL_SPARSE_STATUS_SUCCESS || tiling == nullptr) {
        return st == ACL_SPARSE_STATUS_SUCCESS ? ACL_SPARSE_STATUS_INVALID_VALUE : st;
    }
    if (mode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        tiling->alphaPtr = reinterpret_cast<uint64_t>(alpha);
        tiling->betaPtr = reinterpret_cast<uint64_t>(beta);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (computeType == ACL_INT32) {
        tiling->alphaInt = *reinterpret_cast<const int32_t *>(alpha);
        tiling->betaInt = *reinterpret_cast<const int32_t *>(beta);
    } else if (computeType == ACL_COMPLEX64) {
        const auto *a = reinterpret_cast<const aclsparseComplex *>(alpha);
        const auto *b = reinterpret_cast<const aclsparseComplex *>(beta);
        tiling->alphaReal = a->x;
        tiling->alphaImag = a->y;
        tiling->betaReal = b->x;
        tiling->betaImag = b->y;
    } else {
        tiling->alphaReal = *reinterpret_cast<const float *>(alpha);
        tiling->betaReal = *reinterpret_cast<const float *>(beta);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

void InitializeTiling(const aclsparseSpMatDescr *mat, const aclsparseDnVecDescr *x,
                      const aclsparseDnVecDescr *y,
                      aclsparseOperation_t opA, aclDataType computeType,
                      aclsparseSpMVAlg_t alg, SpmvArch35TilingData *tiling)
{
    *tiling = {};
    tiling->rows = static_cast<uint32_t>(mat->rows);
    tiling->cols = static_cast<uint32_t>(mat->cols);
    tiling->nnz = static_cast<uint32_t>(mat->nnz);
    tiling->indexBase = static_cast<uint32_t>(mat->baseType);
    tiling->rowPtrBase = tiling->indexBase;
    const uint64_t outputLength = IsTransposeOperation(opA) ? mat->cols : mat->rows;
    tiling->rowsPerBlock = outputLength == 0U
        ? 1U : static_cast<uint32_t>((outputLength + SpmvBlockDim() - 1U) / SpmvBlockDim());
    tiling->valueType = static_cast<uint32_t>(ValueTypeId(mat->valueType));
    tiling->outputType = static_cast<uint32_t>(OutputTypeId(y->valueType));
    tiling->computeType = static_cast<uint32_t>(ComputeTypeId(computeType));
    tiling->opA = static_cast<uint32_t>(opA);
    tiling->alg = static_cast<uint32_t>(alg);
    tiling->rowOffsetsStride = mat->ptrStride;
    tiling->colIndStride = mat->idxStride;
    tiling->valuesStride = mat->valueStride;
    tiling->xStride = x->stride;
    tiling->yStride = y->stride;
}

aclsparseStatus_t ConfigureTransposeWorkspace(const aclsparseSpMatDescr *mat,
                                              aclDataType computeType,
                                              SpmvArch35TilingData *tiling)
{
    size_t offset = 0U;
    if (!SpmvArch35AccumulatorOffset(mat->rows, &offset)) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    size_t workspaceBytes = 0U;
    if (!SpmvArch35WorkspaceBytes(mat->rows, mat->cols,
                                  static_cast<uint32_t>(ComputeTypeId(computeType)),
                                  true, &workspaceBytes)) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    (void)workspaceBytes;
    tiling->accumulatorOffset = static_cast<uint64_t>(offset);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildTiling(aclsparseHandle_t handle, aclsparseOperation_t opA,
                              const void *alpha, aclsparseConstSpMatDescr_t matA,
                              const void *beta, aclsparseConstDnVecDescr_t vecX,
                              aclsparseDnVecDescr_t vecY, aclDataType computeType,
                              aclsparseSpMVAlg_t alg, void *buffer,
                              SpmvArch35TilingData *tiling)
{
    if (tiling == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *mat = SpmvArch35ToMat(matA);
    auto *x = SpmvArch35ToVec(vecX);
    auto *y = SpmvArch35ToVec(vecY);
    InitializeTiling(mat, x, y, opA, computeType, alg, tiling);
    // Preprocess has no row-metadata launch for a zero-row CSR.  Keep the
    // original pointer in that case: transpose finalization can still apply
    // beta to a non-empty Y without ever dereferencing rowPtr.
    tiling->useNormalizedRowPtr =
        (mat->rows > 0U &&
         SpmvArch35ActivePlanMatches(mat, x, y, opA, computeType, alg, buffer,
                                     SpmvArch35ToHandle(handle)->stream,
                                     SpmvArch35ToHandle(handle)->stream_epoch))
            ? 1U : 0U;
    if (tiling->useNormalizedRowPtr != 0U) {
        // Preprocess normalizes only row offsets. Column indices retain the
        // descriptor's original base and must still be adjusted in kernel.
        tiling->rowPtrBase = 0U;
        tiling->rowOffsetsStride = 1;
    }
    if (IsTransposeOperation(opA)) {
        const aclsparseStatus_t st = ConfigureTransposeWorkspace(mat, computeType, tiling);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }
    OP_LOGD("aclsparseSpMV", "A5 tiling rows=%u cols=%u nnz=%u op=%u alg=%u value=%u output=%u compute=%u normalized=%u",
            tiling->rows, tiling->cols, tiling->nnz, tiling->opA, tiling->alg,
            tiling->valueType, tiling->outputType, tiling->computeType,
            tiling->useNormalizedRowPtr);
    return ReadHostScalars(handle, alpha, beta, computeType, tiling);
}

aclsparseStatus_t PreprocessSpmv(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, void *externalBuffer)
{
    aclsparseStatus_t st = ValidateSpmvInputs(
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // Preprocess only reads row offsets; values and vectors may be bound later.
    if (SpmvArch35ToMat(matA)->rows > 0U && SpmvArch35ToMat(matA)->ptrs == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    st = ValidateSpmvWorkspace(true, externalBuffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    size_t required = 0U;
    st = aclsparseSpMVGetBufferSize(handle, opA, alpha, matA, vecX, beta, vecY,
                                    computeType, alg, &required);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // The public ABI has no buffer-size argument. The active-buffer association
    // records identity only; callers must allocate the exact GetBufferSize value.
    (void)required;
    auto *mat = SpmvArch35ToMat(matA);
    // This workspace will hold SpMV metadata, invalidating a cached SpMM/SDDMM
    // layout if the caller reuses the same allocation.
    if (mat->activeBuffer == externalBuffer) {
        mat->activeBuffer = nullptr;
    }
    if (mat->rows > 0U) {
        // Normalize on one AIV to avoid cross-core writes to the same cache line.
        spmv_arch35_preprocess_kernel_do(
            reinterpret_cast<GM_ADDR>(mat->ptrs), reinterpret_cast<GM_ADDR>(externalBuffer),
            static_cast<uint32_t>(mat->rows), static_cast<int32_t>(mat->baseType),
            mat->ptrStride, 1U, SpmvArch35ToHandle(handle)->stream);
    }
    SpmvArch35RecordActivePlan(mat, SpmvArch35ToVec(vecX),
                               SpmvArch35ToVec(vecY), opA, computeType, alg,
                               externalBuffer, SpmvArch35ToHandle(handle)->stream,
                               SpmvArch35ToHandle(handle)->stream_epoch);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ExecuteSpmv(
    void *externalBuffer, aclsparseHandle_t handle,
    aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg)
{
    aclsparseStatus_t st = ValidateSpmvExecutionInputs(
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    auto *mat = SpmvArch35ToMat(matA);
    const uint64_t outputLength = IsTransposeOperation(opA) ? mat->cols : mat->rows;
    if (outputLength == 0U) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    SpmvArch35TilingData tiling{};
    st = BuildTiling(handle, opA, alpha, matA, beta, vecX, vecY,
                     computeType, alg, externalBuffer, &tiling);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    auto *x = SpmvArch35ToVec(vecX);
    auto *y = SpmvArch35ToVec(vecY);
    // BuildTiling must inspect the old cache ownership before invalidating it.
    if (mat->activeBuffer == externalBuffer) {
        mat->activeBuffer = nullptr;
        mat->activeSpmvBuffer = nullptr;
    }
    spmv_arch35_kernel_do(
        reinterpret_cast<GM_ADDR>(mat->ptrs), reinterpret_cast<GM_ADDR>(mat->idxs),
        reinterpret_cast<GM_ADDR>(mat->values), reinterpret_cast<GM_ADDR>(x->values),
        reinterpret_cast<GM_ADDR>(y->values), reinterpret_cast<GM_ADDR>(externalBuffer),
        tiling, SpmvBlockDim(), SpmvArch35ToHandle(handle)->stream);
    OP_LOGI("aclsparseSpMV", "A5 kernel launched blockDim=%u op=%u alg=%u",
            SpmvBlockDim(), tiling.opA, tiling.alg);
    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

extern "C" aclsparseStatus_t aclsparseSpMVGetBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, size_t *bufferSize)
{
    if (bufferSize == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *bufferSize = 0U;
    aclsparseStatus_t st = ValidateSpmvInputs(
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    auto *mat = SpmvArch35ToMat(matA);
    if (!SpmvArch35WorkspaceBytes(mat->rows, mat->cols,
                                  static_cast<uint32_t>(ComputeTypeId(computeType)),
                                  IsTransposeOperation(opA), bufferSize)) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseSpMVPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, void *externalBuffer)
{
    return PreprocessSpmv(handle, opA, alpha, matA, vecX, beta, vecY,
                          computeType, alg, externalBuffer);
}

extern "C" aclsparseStatus_t aclsparseSpMV(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, void *externalBuffer)
{
    return ExecuteSpmv(externalBuffer, handle, opA, alpha, matA, vecX,
                       beta, vecY, computeType, alg);
}
