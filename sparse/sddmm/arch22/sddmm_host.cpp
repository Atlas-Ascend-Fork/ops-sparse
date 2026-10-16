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

#include <cstdint>
#include <climits>
#include <vector>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#define HOST_SDDMM
#include "sddmm.h"
#include "sddmm_kernel.h"

namespace {

constexpr uint32_t kSmallRowsThreshold = 64;
constexpr uint64_t kNnzKMulticoreThreshold = 12000;
constexpr uint32_t kKBlockLen = 4088;

// GetAivCoreCount returns 0 when the platform query fails; clamp to 1 so the
// caller never divides by zero when partitioning work across cores.
inline uint32_t SddmmGetBlockDim() {
    const uint32_t aivNum = GetAivCoreCount();
    return (aivNum > 0) ? aivNum : 1;
}

// Scalar / enum arguments and nullptr checks, before any descriptor is touched.
aclsparseStatus_t SddmmValidateArgs(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matXDesc, aclsparseConstDnMatDescr_t matYDesc,
    const void *beta, aclsparseSpMatDescr_t matCDesc, aclDataType computeType,
    aclsparseSDDMMAlg_t alg) {

    CHECK_RET(handle != nullptr,
              OP_LOGE("aclsparseSDDMM", "handle is nullptr");
              return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(alpha != nullptr && beta != nullptr,
              OP_LOGE("aclsparseSDDMM", "alpha/beta is nullptr");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(matXDesc != nullptr && matYDesc != nullptr && matCDesc != nullptr,
              OP_LOGE("aclsparseSDDMM", "descriptor is nullptr");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(alg == ACL_SPARSE_SDDMM_ALG_DEFAULT,
              OP_LOGE("aclsparseSDDMM", "unsupported algorithm");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    CHECK_RET(computeType == ACL_FLOAT || computeType == ACL_FLOAT16,
              OP_LOGE("aclsparseSDDMM", "only ACL_FLOAT/ACL_FLOAT16 computeType supported");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    CHECK_RET(opX == ACL_SPARSE_OP_NON_TRANSPOSE || opX == ACL_SPARSE_OP_TRANSPOSE,
              OP_LOGE("aclsparseSDDMM", "opX conjugate not supported");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    CHECK_RET(opY == ACL_SPARSE_OP_NON_TRANSPOSE || opY == ACL_SPARSE_OP_TRANSPOSE,
              OP_LOGE("aclsparseSDDMM", "opY conjugate not supported");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// CSR format / index type / dtype consistency / layout of the three descriptors.
aclsparseStatus_t SddmmValidateDescriptors(
    struct aclsparseDnMatDescr *matX, struct aclsparseDnMatDescr *matY,
    struct aclsparseSpMatDescr *matC, aclDataType computeType) {

    CHECK_RET(matC->format == ACL_SPARSE_FORMAT_CSR,
              OP_LOGE("aclsparseSDDMM", "only CSR format supported");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    CHECK_RET(matC->baseType == ACL_SPARSE_INDEX_BASE_ZERO,
              OP_LOGE("aclsparseSDDMM", "only 0-based index supported");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    aclsparseStatus_t idxSt = AclsparseValidateSupportedCsrIndexTypes(matC->ptrType, matC->IdxType);
    CHECK_RET(idxSt == ACL_SPARSE_STATUS_SUCCESS,
              OP_LOGE("aclsparseSDDMM", "only ACL_SPARSE_INDEX_32I supported");
              return idxSt);
    CHECK_RET((matX->valueType == ACL_FLOAT || matX->valueType == ACL_FLOAT16) &&
              matX->valueType == matY->valueType && matX->valueType == matC->valueType &&
              matX->valueType == computeType,
              OP_LOGE("aclsparseSDDMM", "dtype must be consistent ACL_FLOAT or ACL_FLOAT16");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    CHECK_RET(matX->order == ACL_SPARSE_ORDER_ROW || matX->order == ACL_SPARSE_ORDER_COL,
              OP_LOGE("aclsparseSDDMM", "matX order must be ROW or COL");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(matY->order == ACL_SPARSE_ORDER_ROW || matY->order == ACL_SPARSE_ORDER_COL,
              OP_LOGE("aclsparseSDDMM", "matY order must be ROW or COL");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    const int64_t xMinLd = (matX->order == ACL_SPARSE_ORDER_ROW) ? matX->cols : matX->rows;
    CHECK_RET(matX->ld >= xMinLd,
              OP_LOGE("aclsparseSDDMM", "matX ld (%ld) < minimum (%ld)", matX->ld, xMinLd);
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(matX->ld <= static_cast<int64_t>(UINT32_MAX),
              OP_LOGE("aclsparseSDDMM", "matX ld (%ld) exceeds UINT32_MAX", matX->ld);
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    const int64_t yMinLd = (matY->order == ACL_SPARSE_ORDER_ROW) ? matY->cols : matY->rows;
    CHECK_RET(matY->ld >= yMinLd,
              OP_LOGE("aclsparseSDDMM", "matY ld (%ld) < minimum (%ld)", matY->ld, yMinLd);
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(matY->ld <= static_cast<int64_t>(UINT32_MAX),
              OP_LOGE("aclsparseSDDMM", "matY ld (%ld) exceeds UINT32_MAX", matY->ld);
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Dimension bounds and the X / Y / C shape agreement implied by opX / opY.
aclsparseStatus_t SddmmValidateShapes(
    aclsparseOperation_t opX, aclsparseOperation_t opY,
    struct aclsparseDnMatDescr *matX, struct aclsparseDnMatDescr *matY,
    struct aclsparseSpMatDescr *matC) {

    CHECK_RET(matC->rows <= INT32_MAX && matC->cols <= INT32_MAX && matC->nnz <= INT32_MAX,
              OP_LOGE("aclsparseSDDMM", "dimensions exceed INT32_MAX");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(matX->rows <= INT32_MAX && matX->cols <= INT32_MAX,
              OP_LOGE("aclsparseSDDMM", "matX dimensions exceed INT32_MAX");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(matY->rows <= INT32_MAX && matY->cols <= INT32_MAX,
              OP_LOGE("aclsparseSDDMM", "matY dimensions exceed INT32_MAX");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    // 零维守卫：公共层 CreateDnMat 放行 rows/cols 为 0 的稠密矩阵，维度一致性
    // 检查对 0==0 也会放行，但 arch22 kernel 不支持零维输入，此处 fail-fast。
    // 与 arch35 SDDMM / arch22 SpMM 的零维行为保持一致（同 API 跨架构一致）。
    CHECK_RET(matX->rows > 0 && matX->cols > 0 && matY->rows > 0 &&
              matY->cols > 0 && matC->rows > 0 && matC->cols > 0,
              OP_LOGE("aclsparseSDDMM", "zero dimension not supported "
                      "(X=%ldx%ld, Y=%ldx%ld, C=%lux%lu)",
                      static_cast<long>(matX->rows), static_cast<long>(matX->cols),
                      static_cast<long>(matY->rows), static_cast<long>(matY->cols),
                      static_cast<unsigned long>(matC->rows),
                      static_cast<unsigned long>(matC->cols));
              return ACL_SPARSE_STATUS_INVALID_VALUE);

    const int64_t xEffRows = (opX == ACL_SPARSE_OP_NON_TRANSPOSE) ? matX->rows : matX->cols;
    const int64_t xEffCols = (opX == ACL_SPARSE_OP_NON_TRANSPOSE) ? matX->cols : matX->rows;
    const int64_t yDimN = (opY == ACL_SPARSE_OP_NON_TRANSPOSE) ? matY->cols : matY->rows;
    const int64_t yDimK = (opY == ACL_SPARSE_OP_NON_TRANSPOSE) ? matY->rows : matY->cols;

    CHECK_RET(xEffRows == static_cast<int64_t>(matC->rows),
              OP_LOGE("aclsparseSDDMM", "X effective rows (%ld) != C rows (%lu)", xEffRows, matC->rows);
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(yDimN == static_cast<int64_t>(matC->cols),
              OP_LOGE("aclsparseSDDMM", "Y effective rows (%ld) != C cols (%lu)", yDimN, matC->cols);
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(xEffCols == yDimK,
              OP_LOGE("aclsparseSDDMM", "X cols (%ld) != Y cols (%ld)", xEffCols, yDimK);
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// CSR data pointer validity: ptrs must never be null (kernel reads ptrs[m]
// unconditionally); when nnz > 0, idxs and values must also be non-null.
aclsparseStatus_t SddmmValidateCsrData(struct aclsparseSpMatDescr *matC) {
    CHECK_RET(matC->ptrs != nullptr,
              OP_LOGE("aclsparseSDDMM", "matC->ptrs is null (kernel reads ptrs[m] for nnz)");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    if (matC->nnz > 0) {
        CHECK_RET(matC->idxs != nullptr,
                  OP_LOGE("aclsparseSDDMM", "matC has nnz=%lu but idxs is null",
                          static_cast<unsigned long>(matC->nnz));
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(matC->values != nullptr,
                  OP_LOGE("aclsparseSDDMM", "matC has nnz=%lu but values is null",
                          static_cast<unsigned long>(matC->nnz));
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Validate CSR rowOffsets monotonicity: rowOffsets[0] == 0, rowOffsets[i] <=
// rowOffsets[i+1], and rowOffsets[m] == nnz.  Requires a D2H copy of ptrs.
aclsparseStatus_t SddmmValidateCsrStructure(struct aclsparseSpMatDescr *matC) {
    const uint64_t m = matC->rows;
    const uint64_t expectedNnz = matC->nnz;
    const size_t ptrBytes = (m + 1) * sizeof(int32_t);

    std::vector<int32_t> rowOff(m + 1);
    aclError aclRet = aclrtMemcpy(rowOff.data(), ptrBytes, matC->ptrs, ptrBytes,
                                  ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS,
              OP_LOGE("aclsparseSDDMM", "D2H copy of rowOffsets failed: %d", static_cast<int>(aclRet));
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    CHECK_RET(rowOff[0] == 0,
              OP_LOGE("aclsparseSDDMM", "rowOffsets[0] = %d, expected 0", rowOff[0]);
              return ACL_SPARSE_STATUS_INVALID_VALUE);

    for (uint64_t i = 0; i < m; ++i) {
        CHECK_RET(rowOff[i] <= rowOff[i + 1],
                  OP_LOGE("aclsparseSDDMM", "rowOffsets not monotonic at row %lu: %d > %d",
                          static_cast<unsigned long>(i), rowOff[i], rowOff[i + 1]);
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
    }

    CHECK_RET(static_cast<uint64_t>(rowOff[m]) == expectedNnz,
              OP_LOGE("aclsparseSDDMM", "rowOffsets[m]=%d != nnz=%lu",
                      rowOff[m], static_cast<unsigned long>(expectedNnz));
              return ACL_SPARSE_STATUS_INVALID_VALUE);

    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t SddmmValidateCommon(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matXDesc, aclsparseConstDnMatDescr_t matYDesc,
    const void *beta, aclsparseSpMatDescr_t matCDesc, aclDataType computeType,
    aclsparseSDDMMAlg_t alg,
    struct aclsparseContext **hOut,
    struct aclsparseDnMatDescr **xOut,
    struct aclsparseDnMatDescr **yOut,
    struct aclsparseSpMatDescr **cOut) {

    aclsparseStatus_t st = SddmmValidateArgs(handle, opX, opY, alpha, matXDesc, matYDesc,
                                            beta, matCDesc, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    auto *h = SddmmToInternalHandle(handle);
    auto *matX = SddmmToDnMatInner(matXDesc);
    auto *matY = SddmmToDnMatInner(matYDesc);
    auto *matC = SddmmToSpMatInner(matCDesc);

    st = SddmmValidateDescriptors(matX, matY, matC, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = SddmmValidateShapes(opX, opY, matX, matY, matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = SddmmValidateCsrData(matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    *hOut = h;
    *xOut = matX;
    *yOut = matY;
    *cOut = matC;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Pick the core count. Tiny problems stay single-core to avoid launch overhead
// dominating; huge-K shapes partition by nonzero instead of by row. The result is
// always >= 1 so the caller's division by blockDim is safe.
uint32_t SddmmSelectBlockDim(uint32_t totalRows, uint32_t nnz, uint32_t maxK,
                              bool nnzPartition, bool isFp16, uint32_t aivNum) {
    const bool hugeK = (maxK > kKBlockLen);
    const uint64_t nnzKProduct = static_cast<uint64_t>(nnz) * maxK;

    uint32_t blockDim;
    if (nnzPartition) {
        blockDim = (aivNum < nnz) ? aivNum : nnz;
    } else if (!isFp16 && totalRows <= kSmallRowsThreshold && !hugeK && nnzKProduct < kNnzKMulticoreThreshold) {
        blockDim = 1;
    } else {
        blockDim = (aivNum < totalRows) ? aivNum : totalRows;
    }
    return (blockDim == 0) ? 1u : blockDim;
}

SddmmTilingData SddmmBuildTiling(
    aclsparseOperation_t opX, aclsparseOperation_t opY,
    float alphaF, float betaF,
    struct aclsparseDnMatDescr *matX,
    struct aclsparseDnMatDescr *matY,
    struct aclsparseSpMatDescr *matC) {

    SddmmTilingData td{};

    const uint32_t nnz = static_cast<uint32_t>(matC->nnz);
    const uint32_t matARows = static_cast<uint32_t>(matX->rows);
    const uint32_t matACols = static_cast<uint32_t>(matX->cols);
    const uint32_t matBRows = static_cast<uint32_t>(matY->rows);
    const uint32_t matBCols = static_cast<uint32_t>(matY->cols);

    const uint32_t rowCount = static_cast<uint32_t>(matC->rows);
    const uint32_t totalRows = rowCount;

    const uint32_t maxK = (matACols > matARows) ? matACols : matARows;
    const bool isFp16 = (matX->valueType == ACL_FLOAT16);
    const uint32_t innerK = (opX == ACL_SPARSE_OP_NON_TRANSPOSE) ? matACols : matARows;
    const uint32_t aivNum = SddmmGetBlockDim();
    // A single-chunk FP16 row can expose one independent dot product per core
    // when all nonzeros fit on the device and row partitioning leaves cores idle.
    const bool fp16SingleWave = isFp16 && innerK == kKBlockLen &&
                                totalRows < aivNum && nnz <= aivNum;
    const bool nnzPartition = innerK >= kKBlockLen && (nnz > totalRows) &&
                              (!isFp16 || fp16SingleWave);
    uint32_t blockDim = SddmmSelectBlockDim(totalRows, nnz, maxK, nnzPartition, isFp16, aivNum);
    if (blockDim == 0) {
        blockDim = 1u;
    }

    td.nnz = nnz;
    td.rowCount = rowCount;
    td.matARows = matARows;
    td.matACols = matACols;
    td.matBRows = matBRows;
    td.matBCols = matBCols;
    td.blockDim = blockDim;
    td.rowsPerCore = totalRows / blockDim;
    td.remainderRows = totalRows % blockDim;
    td.kBlockLen = kKBlockLen;
    td.nnzPartition = nnzPartition ? 1u : 0u;
    td.nnzPerCore = nnzPartition ? (nnz / blockDim) : 0u;
    td.remainderNnz = nnzPartition ? (nnz % blockDim) : 0u;
    td.alphaHost = alphaF;
    td.betaHost = betaF;
    td.opX = (opX == ACL_SPARSE_OP_TRANSPOSE) ? 1u : 0u;
    td.opY = (opY == ACL_SPARSE_OP_TRANSPOSE) ? 1u : 0u;
    td.ldx = static_cast<uint32_t>(matX->ld);
    td.ldy = static_cast<uint32_t>(matY->ld);
    td.orderX = (matX->order == ACL_SPARSE_ORDER_COL) ? 1u : 0u;
    td.orderY = (matY->order == ACL_SPARSE_ORDER_COL) ? 1u : 0u;
    td.dataType = (matX->valueType == ACL_FLOAT16) ? 1u : 0u;

    return td;
}



} // anonymous namespace

aclsparseStatus_t aclsparseSDDMMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, size_t *size) {

    CHECK_RET(size != nullptr,
              OP_LOGE("aclsparseSDDMM", "size is nullptr");
              return ACL_SPARSE_STATUS_INVALID_VALUE);

    struct aclsparseContext *h = nullptr;
    struct aclsparseDnMatDescr *xInner = nullptr;
    struct aclsparseDnMatDescr *yInner = nullptr;
    struct aclsparseSpMatDescr *cInner = nullptr;
    aclsparseStatus_t st = SddmmValidateCommon(
        handle, opX, opY, alpha, matX, matY, beta, matC, computeType, alg,
        &h, &xInner, &yInner, &cInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    *size = 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSDDMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer) {

    struct aclsparseContext *h = nullptr;
    struct aclsparseDnMatDescr *xInner = nullptr;
    struct aclsparseDnMatDescr *yInner = nullptr;
    struct aclsparseSpMatDescr *cInner = nullptr;
    aclsparseStatus_t st = SddmmValidateCommon(
        handle, opX, opY, alpha, matX, matY, beta, matC, computeType, alg,
        &h, &xInner, &yInner, &cInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    (void)h;
    (void)xInner;
    (void)yInner;
    (void)cInner;
    (void)buffer;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSDDMM(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer) {

    struct aclsparseContext *h = nullptr;
    struct aclsparseDnMatDescr *xInner = nullptr;
    struct aclsparseDnMatDescr *yInner = nullptr;
    struct aclsparseSpMatDescr *cInner = nullptr;
    aclsparseStatus_t st = SddmmValidateCommon(
        handle, opX, opY, alpha, matX, matY, beta, matC, computeType, alg,
        &h, &xInner, &yInner, &cInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    CHECK_RET(h->stream != nullptr,
              OP_LOGE("aclsparseSDDMM", "stream is nullptr");
              return ACL_SPARSE_STATUS_INVALID_VALUE);

    st = SddmmValidateCsrStructure(cInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    float alphaF = SddmmReadScalarToF32(alpha, computeType);
    float betaF = SddmmReadScalarToF32(beta, computeType);

    SddmmTilingData td = SddmmBuildTiling(opX, opY, alphaF, betaF, xInner, yInner, cInner);

    GM_ADDR matBArg = static_cast<GM_ADDR>(yInner->values);

    sddmm_kernel_launch(
        static_cast<GM_ADDR>(xInner->values),
        matBArg,
        static_cast<GM_ADDR>(cInner->ptrs),
        static_cast<GM_ADDR>(cInner->idxs),
        static_cast<GM_ADDR>(cInner->values),
        td,
        td.blockDim,
        h->stream);

    return ACL_SPARSE_STATUS_SUCCESS;
}
