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
#include "sddmm.h"
#include "sddmm_launch.h"

namespace {

constexpr uint32_t kSmallRowsThreshold = 64;
constexpr uint64_t kNnzKMulticoreThreshold = 12000;
constexpr uint32_t kKBlockLen = 4088;

// Scalar / enum arguments and nullptr checks, before any descriptor is touched.
aclsparseStatus_t SddmmValidateArgs(aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
                                    const void* alpha, aclsparseConstDnMatDescr_t matXDesc,
                                    aclsparseConstDnMatDescr_t matYDesc, const void* beta,
                                    aclsparseSpMatDescr_t matCDesc, aclDataType computeType, aclsparseSDDMMAlg_t alg)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclsparseSDDMM", "handle is nullptr");
              return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(alpha != nullptr && beta != nullptr, OP_LOGE("aclsparseSDDMM", "alpha/beta is nullptr");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(matXDesc != nullptr && matYDesc != nullptr && matCDesc != nullptr,
              OP_LOGE("aclsparseSDDMM", "descriptor is nullptr");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(alg == ACL_SPARSE_SDDMM_ALG_DEFAULT, OP_LOGE("aclsparseSDDMM", "unsupported algorithm");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    // Which of these is legal is decided against matC's dtype in
    // SddmmValidateDescriptors; here we only reject values that no row of the
    // support matrix can ever claim.
    CHECK_RET(computeType == ACL_FLOAT || computeType == ACL_FLOAT16 || computeType == ACL_COMPLEX64,
              OP_LOGE("aclsparseSDDMM", "unsupported computeType %d", static_cast<int>(computeType));
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    CHECK_RET(opX == ACL_SPARSE_OP_NON_TRANSPOSE || opX == ACL_SPARSE_OP_TRANSPOSE,
              OP_LOGE("aclsparseSDDMM", "opX conjugate not supported");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    CHECK_RET(opY == ACL_SPARSE_OP_NON_TRANSPOSE || opY == ACL_SPARSE_OP_TRANSPOSE,
              OP_LOGE("aclsparseSDDMM", "opY conjugate not supported");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Map an ACL dtype onto the tiling code the kernel dispatches on.
uint32_t SddmmDtypeCodeOf(aclDataType valueType)
{
    switch (valueType) {
        case ACL_FLOAT16:
            return SDDMM_DTYPE_FP16;
        case ACL_BF16:
            return SDDMM_DTYPE_BF16;
        case ACL_COMPLEX64:
            return SDDMM_DTYPE_COMPLEX64;
        default:
            return SDDMM_DTYPE_FP32;
    }
}

aclsparseStatus_t SddmmValidateLayouts(aclsparseDnMatDescr* matX, aclsparseDnMatDescr* matY, aclsparseSpMatDescr* matC)
{
    CHECK_RET(matX->order == ACL_SPARSE_ORDER_ROW || matX->order == ACL_SPARSE_ORDER_COL,
              OP_LOGE("aclsparseSDDMM", "matX order must be ROW or COL");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(matY->order == ACL_SPARSE_ORDER_ROW || matY->order == ACL_SPARSE_ORDER_COL,
              OP_LOGE("aclsparseSDDMM", "matY order must be ROW or COL");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    const int64_t xMinLd = (matX->order == ACL_SPARSE_ORDER_ROW) ? matX->cols : matX->rows;
    CHECK_RET(matX->ld >= xMinLd, OP_LOGE("aclsparseSDDMM", "matX ld (%ld) < minimum (%ld)", matX->ld, xMinLd);
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(matX->ld <= static_cast<int64_t>(UINT32_MAX),
              OP_LOGE("aclsparseSDDMM", "matX ld (%ld) exceeds UINT32_MAX", matX->ld);
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    const int64_t yMinLd = (matY->order == ACL_SPARSE_ORDER_ROW) ? matY->cols : matY->rows;
    CHECK_RET(matY->ld >= yMinLd, OP_LOGE("aclsparseSDDMM", "matY ld (%ld) < minimum (%ld)", matY->ld, yMinLd);
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(matY->ld <= static_cast<int64_t>(UINT32_MAX),
              OP_LOGE("aclsparseSDDMM", "matY ld (%ld) exceeds UINT32_MAX", matY->ld);
              return ACL_SPARSE_STATUS_INVALID_VALUE);

    // The complex64 row splits X into two fp32 planes and reads Y through an fp32
    // view, so both are indexed as flat row-major arrays: a leading dimension
    // wider than the row would put padding between rows and be read as data.
    if (matC->valueType == ACL_COMPLEX64) {
        CHECK_RET(matX->order == ACL_SPARSE_ORDER_ROW && matX->ld == matX->cols,
                  OP_LOGE("aclsparseSDDMM",
                          "complex64 needs matX row-major and contiguous "
                          "(ld %ld, cols %ld)",
                          matX->ld, matX->cols);
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);
        CHECK_RET(matY->order == ACL_SPARSE_ORDER_ROW && matY->ld == matY->cols,
                  OP_LOGE("aclsparseSDDMM",
                          "complex64 needs matY row-major and contiguous "
                          "(ld %ld, cols %ld)",
                          matY->ld, matY->cols);
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// CSR format / index type / dtype consistency / layout of the three descriptors.
aclsparseStatus_t SddmmValidateDescriptors(struct aclsparseDnMatDescr* matX, struct aclsparseDnMatDescr* matY,
                                           struct aclsparseSpMatDescr* matC, aclDataType computeType)
{
    CHECK_RET(matC->format == ACL_SPARSE_FORMAT_CSR, OP_LOGE("aclsparseSDDMM", "only CSR format supported");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    CHECK_RET(matC->baseType == ACL_SPARSE_INDEX_BASE_ZERO || matC->baseType == ACL_SPARSE_INDEX_BASE_ONE,
              OP_LOGE("aclsparseSDDMM", "index base must be ZERO or ONE, got %d", static_cast<int>(matC->baseType));
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    aclsparseStatus_t idxSt = AclsparseValidateSupportedCsrIndexTypes(matC->ptrType, matC->IdxType);
    CHECK_RET(idxSt == ACL_SPARSE_STATUS_SUCCESS, OP_LOGE("aclsparseSDDMM", "only ACL_SPARSE_INDEX_32I supported");
              return idxSt);
    CHECK_RET(matX->valueType == matY->valueType, OP_LOGE("aclsparseSDDMM", "matX and matY must share one dtype");
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    CHECK_RET(SddmmDtypePairSupported(matX->valueType, matC->valueType),
              OP_LOGE("aclsparseSDDMM", "unsupported matX/matY (%d) + matC values (%d) pair",
                      static_cast<int>(matX->valueType), static_cast<int>(matC->valueType));
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    CHECK_RET(SddmmComputeTypeSupported(matX->valueType, matC->valueType, computeType),
              OP_LOGE("aclsparseSDDMM", "computeType (%d) must match the matC values dtype (%d)",
                      static_cast<int>(computeType), static_cast<int>(matC->valueType));
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    return SddmmValidateLayouts(matX, matY, matC);
}

// Dimension bounds and the X / Y / C shape agreement implied by opX / opY.
aclsparseStatus_t SddmmValidateShapes(aclsparseOperation_t opX, aclsparseOperation_t opY,
                                      struct aclsparseDnMatDescr* matX, struct aclsparseDnMatDescr* matY,
                                      struct aclsparseSpMatDescr* matC)
{
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
    CHECK_RET(matX->rows > 0 && matX->cols > 0 && matY->rows > 0 && matY->cols > 0 && matC->rows > 0 && matC->cols > 0,
              OP_LOGE("aclsparseSDDMM",
                      "zero dimension not supported "
                      "(X=%ldx%ld, Y=%ldx%ld, C=%lux%lu)",
                      static_cast<long>(matX->rows), static_cast<long>(matX->cols), static_cast<long>(matY->rows),
                      static_cast<long>(matY->cols), static_cast<unsigned long>(matC->rows),
                      static_cast<unsigned long>(matC->cols));
              return ACL_SPARSE_STATUS_INVALID_VALUE);

    // The doubled-pattern rewriting is expressed in the (X @ Y) orientation
    // only; a transposed operand would need its own split / view derivation.
    if (matC->valueType == ACL_COMPLEX64) {
        CHECK_RET(opX == ACL_SPARSE_OP_NON_TRANSPOSE && opY == ACL_SPARSE_OP_NON_TRANSPOSE,
                  OP_LOGE("aclsparseSDDMM",
                          "complex64 requires opX = opY = NON_TRANSPOSE "
                          "(got opX %d, opY %d)",
                          static_cast<int>(opX), static_cast<int>(opY));
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);
    }

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
    CHECK_RET(xEffCols == yDimK, OP_LOGE("aclsparseSDDMM", "X cols (%ld) != Y cols (%ld)", xEffCols, yDimK);
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// CSR data pointer validity: ptrs must never be null (kernel reads ptrs[m]
// unconditionally); when nnz > 0, idxs and values must also be non-null.
aclsparseStatus_t SddmmValidateCsrData(struct aclsparseSpMatDescr* matC)
{
    CHECK_RET(matC->ptrs != nullptr, OP_LOGE("aclsparseSDDMM", "matC->ptrs is null (kernel reads ptrs[m] for nnz)");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    if (matC->nnz > 0) {
        CHECK_RET(matC->idxs != nullptr,
                  OP_LOGE("aclsparseSDDMM", "matC has nnz=%lu but idxs is null", static_cast<unsigned long>(matC->nnz));
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(matC->values != nullptr, OP_LOGE("aclsparseSDDMM", "matC has nnz=%lu but values is null",
                                                   static_cast<unsigned long>(matC->nnz));
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

struct SddmmDescriptors {
    aclsparseContext* h = nullptr;
    aclsparseDnMatDescr *xInner = nullptr, *yInner = nullptr;
    aclsparseSpMatDescr* cInner = nullptr;
};

aclsparseStatus_t SddmmValidateCommon(aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
                                      const void* alpha, aclsparseConstDnMatDescr_t matXDesc,
                                      aclsparseConstDnMatDescr_t matYDesc, const void* beta,
                                      aclsparseSpMatDescr_t matCDesc, aclDataType computeType, aclsparseSDDMMAlg_t alg,
                                      SddmmDescriptors& descriptors)
{
    aclsparseStatus_t st = SddmmValidateArgs(handle, opX, opY, alpha, matXDesc, matYDesc, beta, matCDesc, computeType,
                                             alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    auto* h = SddmmToInternalHandle(handle);
    auto* matX = SddmmToDnMatInner(matXDesc);
    auto* matY = SddmmToDnMatInner(matYDesc);
    auto* matC = SddmmToSpMatInner(matCDesc);

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

    descriptors = {h, matX, matY, matC};
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Pick the core count. Tiny problems stay single-core to avoid launch overhead
// dominating; huge-K shapes partition by nonzero instead of by row. The result is
// always >= 1 so the caller's division by blockDim is safe.
uint32_t SddmmSelectBlockDim(uint32_t totalRows, uint32_t nnz, uint32_t maxK, bool nnzPartition, bool isFp16 = false)
{
    const uint32_t aivNum = SddmmGetBlockDim();
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

SddmmTilingData SddmmBuildTiling(aclsparseOperation_t opX, aclsparseOperation_t opY, float alphaF, float betaF,
                                 float alphaImagF, float betaImagF, struct aclsparseDnMatDescr* matX,
                                 struct aclsparseDnMatDescr* matY, struct aclsparseSpMatDescr* matC)
{
    SddmmTilingData td{};

    const uint32_t nnz = static_cast<uint32_t>(matC->nnz);
    const uint32_t matARows = static_cast<uint32_t>(matX->rows);
    const uint32_t matACols = static_cast<uint32_t>(matX->cols);
    const uint32_t matBRows = static_cast<uint32_t>(matY->rows);
    const uint32_t matBCols = static_cast<uint32_t>(matY->cols);

    const uint32_t rowCount = static_cast<uint32_t>(matC->rows);
    const uint32_t totalRows = rowCount;

    // Partitioning must follow the reduction dimension after op(X), not the
    // larger physical dimension of X. Using max(rows, cols) classified every
    // tall performance shape (M=4096..8192, K=128) as huge-K and sent FP32 to
    // the per-nonzero path, where each slot repeated the row lookup and A load.
    const uint32_t innerK = (opX == ACL_SPARSE_OP_NON_TRANSPOSE) ? matACols : matARows;
    const bool isFp16 = matX->valueType == ACL_FLOAT16;
    const uint32_t cores = SddmmGetBlockDim();
    const bool fp16SingleWave = isFp16 && innerK == kKBlockLen && totalRows < cores && nnz <= cores;
    const bool nnzPartition = innerK >= kKBlockLen && nnz > totalRows &&
                              (matX->valueType == ACL_FLOAT || fp16SingleWave);
    uint32_t blockDim = SddmmSelectBlockDim(totalRows, nnz, innerK, nnzPartition, isFp16);
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
    td.alphaImagHost = alphaImagF;
    td.betaImagHost = betaImagF;
    td.opX = (opX == ACL_SPARSE_OP_TRANSPOSE) ? 1u : 0u;
    td.opY = (opY == ACL_SPARSE_OP_TRANSPOSE) ? 1u : 0u;
    td.ldx = static_cast<uint32_t>(matX->ld);
    td.ldy = static_cast<uint32_t>(matY->ld);
    td.orderX = (matX->order == ACL_SPARSE_ORDER_COL) ? 1u : 0u;
    td.orderY = (matY->order == ACL_SPARSE_ORDER_COL) ? 1u : 0u;
    td.xType = SddmmDtypeCodeOf(matX->valueType);
    td.cType = SddmmDtypeCodeOf(matC->valueType);
    td.idxBase = (matC->baseType == ACL_SPARSE_INDEX_BASE_ONE) ? 1u : 0u;

    return td;
}

// One of the two fp32 passes that stand in for a complex64 SDDMM.
//
// The shapes match the caller's problem except that B is the [k, 2n] fp32 view
// of the complex Y (which needs no repacking -- a complex64 array already *is*
// that float array), the reduction length K is unchanged, and the pattern has
// been doubled so every logical entry owns one slot per real column. alpha/beta
// stay at 1/0: the real multipliers are applied later by the combine pass, and
// the real kernel's beta==0 path is exactly what keeps it from touching the
// product buffers.
SddmmTilingData SddmmBuildComplexTiling(uint32_t rows, uint32_t k, uint32_t cols, uint32_t doubledNnz, uint32_t idxBase,
                                        uint32_t yPitch)
{
    SddmmTilingData td{};
    const uint32_t maxK = k;
    // Same rule as the real fp32 path: split by nonzero only when K is too long
    // for one block and the pattern is denser than the row count.
    const bool nnzPartition = (maxK > kKBlockLen) && (doubledNnz > rows);
    uint32_t blockDim = SddmmSelectBlockDim(rows, doubledNnz, maxK, nnzPartition);
    if (blockDim == 0) {
        blockDim = 1u;
    }

    td.nnz = doubledNnz;
    td.rowCount = rows;
    td.matARows = rows;
    td.matACols = k;
    td.matBRows = k;
    td.matBCols = 2u * cols;
    td.blockDim = blockDim;
    td.rowsPerCore = rows / blockDim;
    td.remainderRows = rows % blockDim;
    td.kBlockLen = kKBlockLen;
    td.nnzPartition = nnzPartition ? 1u : 0u;
    td.nnzPerCore = nnzPartition ? (doubledNnz / blockDim) : 0u;
    td.remainderNnz = nnzPartition ? (doubledNnz % blockDim) : 0u;
    td.alphaHost = 1.0f;
    td.betaHost = 0.0f;
    td.alphaImagHost = 0.0f;
    td.betaImagHost = 0.0f;
    td.opX = 0;
    td.opY = 0;
    td.ldx = k;
    td.ldy = (yPitch != 0) ? yPitch : 2u * cols;
    td.orderX = 0;
    td.orderY = (yPitch != 0) ? 1u : 0u;
    td.xType = SDDMM_DTYPE_FP32;
    td.cType = SDDMM_DTYPE_FP32;
    td.idxBase = idxBase;

    return td;
}

// Drive the complex64 row: two fp32 SDDMM passes over the doubled pattern, then
// one pass that folds the four real products back into complex64 and applies
// alpha / beta. `buffer` holds the split X planes, the expanded pattern and the
// two product runs, so it must be the allocation the caller sized with
// aclsparseSDDMMBufferSize.
aclsparseStatus_t SddmmRunComplex(struct aclsparseContext* h, struct aclsparseDnMatDescr* matX,
                                  struct aclsparseDnMatDescr* matY, struct aclsparseSpMatDescr* matC, float alphaReal,
                                  float alphaImag, float betaReal, float betaImag, void* buffer)
{
    CHECK_RET(buffer != nullptr, OP_LOGE("aclsparseSDDMM", "complex64 needs the buffer reported by "
                                                           "aclsparseSDDMMBufferSize (got nullptr)");
              return ACL_SPARSE_STATUS_INVALID_VALUE);

    const uint32_t rows = static_cast<uint32_t>(matC->rows);
    const uint32_t k = static_cast<uint32_t>(matX->cols);
    const uint32_t cols = static_cast<uint32_t>(matY->cols);
    const uint32_t nnz = static_cast<uint32_t>(matC->nnz);
    // Both the doubled pattern and the doubled column count are addressed with
    // 32-bit indices downstream.
    CHECK_RET(cols <= (static_cast<uint32_t>(INT32_MAX) / 2u) && nnz <= (static_cast<uint32_t>(INT32_MAX) / 2u),
              OP_LOGE("aclsparseSDDMM", "complex64 dims too large: cols %u, nnz %u", cols, nnz);
              return ACL_SPARSE_STATUS_NOT_SUPPORTED);

    const uint32_t idxBase = (matC->baseType == ACL_SPARSE_INDEX_BASE_ONE) ? 1u : 0u;
    const SddmmComplexPlan plan = SddmmComplexPlanOf(rows, k, cols, nnz);
    const SddmmTilingData inner = SddmmBuildComplexTiling(rows, k, cols, 2u * nnz, idxBase,
                                                          plan.useYt ? plan.yPitch : 0u);

    return sddmm_complex_launch(static_cast<GM_ADDR>(matX->values), static_cast<GM_ADDR>(matY->values),
                                static_cast<GM_ADDR>(matC->ptrs), static_cast<GM_ADDR>(matC->idxs),
                                static_cast<GM_ADDR>(matC->values), rows, k, nnz, idxBase, alphaReal, alphaImag,
                                betaReal, betaImag, inner, buffer, h->stream);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// B-operand transpose (bulk path)
// ---------------------------------------------------------------------------

// Largest ldy the batched kernel can stage: the repeat stride that walks the UB
// tile is an 8-bit block count.
constexpr uint32_t kSddmmMaxYtPitch = 255u * 8u;

// Element size of the transposed Y, or 0 when this storage dtype has no
// transpose entry point. complex64 maps to 0 on purpose: its fp32 view is what
// gets transposed, and only the complex row knows about that view.
uint32_t SddmmYtElemSize(aclDataType valueType)
{
    switch (valueType) {
        case ACL_FLOAT:
            return static_cast<uint32_t>(sizeof(float));
        case ACL_FLOAT16:
        case ACL_BF16:
            return static_cast<uint32_t>(sizeof(uint16_t));
        default:
            return 0;
    }
}

// What the bulk path needs to describe its own copy of Y. BufferSize, Preprocess
// and the kernel launch all derive it from the same inputs, so they cannot
// disagree about the layout.
struct SddmmYtPlan {
    bool use = false;
    uint64_t bytes = 0;
    uint32_t pitch = 0;
    uint32_t rows = 0; // the reduction length K
    uint32_t cols = 0; // the number of B columns n
    uint32_t elemSize = 0;
};

// The kernel's bContig test, mirrored here so the host can ask SddmmPlanYt the
// question the kernel would answer: a column of B is contiguous in GM when the
// memory order and the requested transpose agree. Both false-bContig layouts
// (row-major Y with opY == N, column-major Y with opY == T) address B[k][j] as
// memY[j + k*ldy], so one transpose serves them both.
inline bool SddmmBColumnContiguous(uint32_t orderYCol, uint32_t opYValue)
{
    return (orderYCol == 0 && opYValue == 1) || (orderYCol == 1 && opYValue == 0);
}

// Is a transposed copy of Y worth building (and usable) for this problem?
//
// The copy moves n * K elements once, while the kernel saves a factor of about
// 32 / sizeof(T) on every element of B it stages -- and it stages nnz * K of
// them. So the swap wins exactly when Y is re-read many times over, i.e. when
// the pattern covers the columns more than once. `nnz >= cols` is the crude
// version of that test; below it the original (gather) path stays in charge.
//
// Everything the transpose needs -- K, n, ldy -- comes out of the descriptors
// here, and each of the three API entry points calls this one function, so the
// size it reports and the layout the kernel assumes cannot drift apart.
SddmmYtPlan SddmmPlanYt(aclsparseOperation_t opX, aclsparseOperation_t opY, struct aclsparseDnMatDescr* xInner,
                        struct aclsparseDnMatDescr* yInner, struct aclsparseSpMatDescr* cInner)
{
    SddmmYtPlan plan{};
    plan.elemSize = SddmmYtElemSize(yInner->valueType);
    const bool bContig = SddmmBColumnContiguous((yInner->order == ACL_SPARSE_ORDER_COL) ? 1u : 0u,
                                                (opY == ACL_SPARSE_OP_TRANSPOSE) ? 1u : 0u);
    if (bContig || plan.elemSize == 0) {
        return plan;
    }
    // The reduction length and the number of B columns, resolved the way the
    // kernel's SetupLayout resolves them.
    const uint32_t k = (opX == ACL_SPARSE_OP_NON_TRANSPOSE) ? static_cast<uint32_t>(xInner->cols) :
                                                              static_cast<uint32_t>(xInner->rows);
    const uint32_t cols = (opY == ACL_SPARSE_OP_NON_TRANSPOSE) ? static_cast<uint32_t>(yInner->cols) :
                                                                 static_cast<uint32_t>(yInner->rows);
    const uint32_t reqRows = static_cast<uint32_t>(cInner->rows);
    // Below the tile size the transpose kernel has nothing to do, and a pattern
    // sparser than the column count does not re-read Y enough to pay for it.
    if (k < 32u || cols < kYTransTile || reqRows == 0) {
        return plan;
    }
    // memY[j + k*ldy] has to stay inside the n * ldy element allocation, which
    // is exactly the statement ldy >= n.
    const uint32_t ldy = static_cast<uint32_t>(yInner->ld);
    if (ldy < cols) {
        return plan;
    }
    plan.pitch = SddmmYtPitch(k);
    if (plan.pitch > kSddmmMaxYtPitch) {
        return plan;
    }
    if (static_cast<uint64_t>(cInner->nnz) * k < static_cast<uint64_t>(cols)) {
        return plan;
    }
    plan.use = true;
    plan.rows = k;
    plan.cols = cols;
    plan.bytes = SddmmYtBytes(k, cols, plan.elemSize);
    return plan;
}

aclsparseStatus_t SddmmCsrBufferSize(aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
                                     const void* alpha, aclsparseConstDnMatDescr_t matX,
                                     aclsparseConstDnMatDescr_t matY, const void* beta, aclsparseSpMatDescr_t matC,
                                     aclDataType computeType, aclsparseSDDMMAlg_t alg, size_t* size)
{
    CHECK_RET(size != nullptr, OP_LOGE("aclsparseSDDMM", "size is nullptr"); return ACL_SPARSE_STATUS_INVALID_VALUE);

    SddmmDescriptors descriptors;
    const auto st = SddmmValidateCommon(handle, opX, opY, alpha, matX, matY, beta, matC, computeType, alg, descriptors);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    const auto [h, xInner, yInner, cInner] = descriptors;

    // The real rows need scratch only when B gets transposed; the complex64 row
    // additionally needs room for the split X planes, the doubled sparsity
    // pattern and the two product runs. Either way aclsparseSDDMMPreprocess is
    // what fills the buffer in.
    if (cInner->valueType == ACL_COMPLEX64) {
        *size = static_cast<size_t>(SddmmComplexPlanOf(cInner->rows, xInner->cols, yInner->cols, cInner->nnz).total);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    *size = static_cast<size_t>(SddmmPlanYt(opX, opY, xInner, yInner, cInner).bytes);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t SddmmCsrPrepare(aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
                                  const void* alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
                                  const void* beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
                                  aclsparseSDDMMAlg_t alg, void* buffer)
{
    SddmmDescriptors descriptors;
    const auto st = SddmmValidateCommon(handle, opX, opY, alpha, matX, matY, beta, matC, computeType, alg, descriptors);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    const auto [h, xInner, yInner, cInner] = descriptors;

    // Only the complex64 row has work to do up front: splitting X and expanding
    // the pattern are both pure functions of the inputs. Everything reads and
    // writes within the bounds the caller declared (rows, k, nnz), so this stage
    // is memory-safe even before aclsparseSDDMM runs its CSR structure checks.
    if (cInner->valueType == ACL_COMPLEX64) {
        CHECK_RET(buffer != nullptr, OP_LOGE("aclsparseSDDMM", "complex64 preprocess needs the buffer reported by "
                                                               "aclsparseSDDMMBufferSize (got nullptr)");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(h->stream != nullptr, OP_LOGE("aclsparseSDDMM", "stream is nullptr");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        return sddmm_complex_prepare(static_cast<GM_ADDR>(xInner->values), static_cast<GM_ADDR>(yInner->values),
                                     static_cast<GM_ADDR>(cInner->ptrs), static_cast<GM_ADDR>(cInner->idxs),
                                     static_cast<uint32_t>(cInner->rows), static_cast<uint32_t>(xInner->cols),
                                     static_cast<uint32_t>(yInner->cols), static_cast<uint32_t>(cInner->nnz),
                                     (cInner->baseType == ACL_SPARSE_INDEX_BASE_ONE) ? 1u : 0u, SddmmGetBlockDim(),
                                     buffer, h->stream);
    }

    // Real rows: build the transposed copy of Y when BufferSize asked for one.
    const SddmmYtPlan yt = SddmmPlanYt(opX, opY, xInner, yInner, cInner);
    if (!yt.use) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    CHECK_RET(buffer != nullptr, OP_LOGE("aclsparseSDDMM", "transposed-B preprocess needs the buffer reported by "
                                                           "aclsparseSDDMMBufferSize (got nullptr)");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(h->stream != nullptr, OP_LOGE("aclsparseSDDMM", "stream is nullptr");
              return ACL_SPARSE_STATUS_INVALID_VALUE);
    return sddmm_y_transpose_launch(static_cast<GM_ADDR>(yInner->values), static_cast<GM_ADDR>(buffer), yt.rows,
                                    yt.cols, static_cast<uint32_t>(yInner->ld), yt.pitch,
                                    SddmmDtypeCodeOf(yInner->valueType), SddmmGetBlockDim(), h->stream);
}

bool SddmmEffectiveY(aclsparseOperation_t opX, aclsparseOperation_t opY, aclsparseDnMatDescr* xInner,
                     aclsparseDnMatDescr* yInner, aclsparseSpMatDescr* cInner, void* buffer,
                     aclsparseDnMatDescr& ytDesc)
{
    const SddmmYtPlan yt = SddmmPlanYt(opX, opY, xInner, yInner, cInner);
    if (yt.use && buffer != nullptr) {
        ytDesc = *yInner;
        ytDesc.rows = yt.rows;
        ytDesc.cols = yt.cols;
        ytDesc.ld = yt.pitch;
        ytDesc.order = ACL_SPARSE_ORDER_COL;
        ytDesc.values = buffer;
        return true;
    }

    return false;
}

aclsparseStatus_t SddmmCsrExecute(aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
                                  const void* alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
                                  const void* beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
                                  aclsparseSDDMMAlg_t alg, void* buffer, void* rowHints, void* patternFlags,
                                  uint32_t patternFlagWords)
{
    SddmmDescriptors descriptors;
    const auto st = SddmmValidateCommon(handle, opX, opY, alpha, matX, matY, beta, matC, computeType, alg, descriptors);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    const auto [h, xInner, yInner, cInner] = descriptors;

    CHECK_RET(h->stream != nullptr, OP_LOGE("aclsparseSDDMM", "stream is nullptr");
              return ACL_SPARSE_STATUS_INVALID_VALUE);

    float alphaF = SddmmReadScalarToF32(alpha, computeType);
    float betaF = SddmmReadScalarToF32(beta, computeType);
    float alphaImagF = SddmmReadScalarImagToF32(alpha, computeType);
    float betaImagF = SddmmReadScalarImagToF32(beta, computeType);

    if (cInner->valueType == ACL_COMPLEX64) {
        return SddmmRunComplex(h, xInner, yInner, cInner, alphaF, alphaImagF, betaF, betaImagF, buffer);
    }

    // When Preprocess laid Y out as Yt, describe *that* to the kernel: K rows of
    // n columns, column-major with ld == pitch. Both then fall out for free --
    // the kernel's bContig test passes (column-major + no transpose), which is
    // what turns on the batched tile path, and B[k][j] lands at j * ldy, exactly
    // where the transpose wrote it.
    //
    // The plan is a pure function of the descriptors, so the three entry points
    // agree on whether Yt exists without any of them storing state. A caller who
    // sized the buffer with a real aclsparseSDDMMBufferSize (i.e. non-zero) gets
    // a non-null pointer back from AclSparseWorkspace and takes this path; one
    // who passed nullptr or a zero-sized buffer keeps the gather path, which is
    // still correct, just slower.
    struct aclsparseDnMatDescr ytDesc;
    struct aclsparseDnMatDescr* effY = yInner;
    aclsparseOperation_t effOpY = opY;
    if (SddmmEffectiveY(opX, opY, xInner, yInner, cInner, buffer, ytDesc)) {
        effY = &ytDesc;
        effOpY = ACL_SPARSE_OP_NON_TRANSPOSE;
    }

    SddmmTilingData td = SddmmBuildTiling(opX, effOpY, alphaF, betaF, alphaImagF, betaImagF, xInner, effY, cInner);
    td.rowHints = reinterpret_cast<uintptr_t>(rowHints);
    td.patternFlags = reinterpret_cast<uintptr_t>(patternFlags);
    td.patternFlagWords = patternFlagWords;

    GM_ADDR matBArg = static_cast<GM_ADDR>(effY->values);

    aclsparseStatus_t launchStatus = sddmm_kernel_launch(
        static_cast<GM_ADDR>(xInner->values), matBArg, static_cast<GM_ADDR>(cInner->ptrs),
        static_cast<GM_ADDR>(cInner->idxs), static_cast<GM_ADDR>(cInner->values), td, td.blockDim, h->stream);
    // Launch returns NOT_SUPPORTED when no entry point matches the tiling; do
    // not report success for a kernel that never ran.
    CHECK_RET(launchStatus == ACL_SPARSE_STATUS_SUCCESS,
              OP_LOGE("aclsparseSDDMM", "kernel launch rejected the tiling: status %d", static_cast<int>(launchStatus));
              return launchStatus);

    return ACL_SPARSE_STATUS_SUCCESS;
}
