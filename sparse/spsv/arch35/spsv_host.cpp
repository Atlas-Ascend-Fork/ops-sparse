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

#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <new>
#include "log/log.h"
#include "cann_ops_sparse.h"
#include "spsv.h"
#include "spsv_kernel.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_spsv_descr.h"
#include "aclsparse_host_utils.h"

namespace {

static aclsparseStatus_t ValidateSpSVIndexTypes(const char *tag,
    const aclsparseSpMatDescr *mat)
{
    if (mat->baseType != ACL_SPARSE_INDEX_BASE_ZERO &&
        mat->baseType != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(tag, "unsupported baseType=%d, only ZERO or ONE supported", mat->baseType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (mat->format != ACL_SPARSE_FORMAT_CSR &&
        mat->format != ACL_SPARSE_FORMAT_CSC &&
        mat->format != ACL_SPARSE_FORMAT_COO &&
        mat->format != ACL_SPARSE_FORMAT_SLICED_ELL) {
        OP_LOGE(tag, "unsupported format=%d", mat->format);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // Validate individual index types (ptrType and IdxType independently).
    if (mat->ptrType != ACL_SPARSE_INDEX_32I && mat->ptrType != ACL_SPARSE_INDEX_64I) {
        OP_LOGE(tag, "unsupported ptrType=%d", mat->ptrType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (mat->IdxType != ACL_SPARSE_INDEX_32I && mat->IdxType != ACL_SPARSE_INDEX_64I) {
        OP_LOGE(tag, "unsupported idxType=%d", mat->IdxType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // Reject (I32, I64): ptrType is I32 cannot address nnz > INT32_MAX;
    // colInd being I64 in this combo is unsupported (no practical use case).
    // Allowed combinations: (I32, I32), (I64, I32), (I64, I64).
    if (mat->ptrType == ACL_SPARSE_INDEX_32I && mat->IdxType == ACL_SPARSE_INDEX_64I) {
        OP_LOGE(tag, "unsupported index combination: ptrType=I32, idxType=I64");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpSVCommonParams(
    const char *tag,
    aclsparseConstSpMatDescr_t matA,
    aclsparseSpSVDescr_t spsvDescr,
    aclDataType computeType,
    aclsparseSpSVAlg_t alg)
{
    if (matA == nullptr) {
        OP_LOGE(tag, "matA is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (spsvDescr == nullptr) {
        OP_LOGE(tag, "spsvDescr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != ACL_FLOAT) {
        OP_LOGE(tag, "unsupported computeType=%d, only ACL_FLOAT supported", computeType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (alg != ACL_SPARSE_SPSV_ALG_DEFAULT) {
        OP_LOGE(tag, "unsupported alg=%d", alg);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    auto *mat = spsv::ToMatInner(matA);
    if (mat->valueType != ACL_FLOAT) {
        OP_LOGE(tag, "unsupported matrix valueType=%d, only ACL_FLOAT supported", mat->valueType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (mat->rows > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(tag, "matrix rows=%" PRIu64 " exceeds INT32_MAX (kernel uses int32_t internally)", mat->rows);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // nnz > INT32_MAX requires I64 index type (ptrType=ACL_SPARSE_INDEX_64I).
    // I32 + nnz > INT32_MAX is explicitly rejected below.
    if (mat->ptrType == ACL_SPARSE_INDEX_32I && mat->nnz > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(tag, "I32 index type cannot represent nnz=%" PRIu64 " > INT32_MAX", mat->nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (mat->rows != mat->cols) {
        OP_LOGE(tag, "matrix must be square, rows=%" PRIu64 " cols=%" PRIu64, mat->rows, mat->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ValidateSpSVIndexTypes(tag, mat);
}

static aclsparseStatus_t ValidateOpA(const char *tag, aclsparseOperation_t opA)
{
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE &&
        opA != ACL_SPARSE_OP_TRANSPOSE &&
        opA != ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) {
        OP_LOGE(tag, "invalid opA=%d", opA);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static bool NeedsTranspose(aclsparseOperation_t opA)
{
    return opA != ACL_SPARSE_OP_NON_TRANSPOSE;
}

// Single source of truth for workspace layout. Returned struct carries all
// sub-allocation offsets plus the total size, so callers no longer need to
// pass 12 out-parameters.
struct WorkspaceOffsets {
    int64_t diagPtrOffset{-1};
    int64_t levelPtrOffset{-1};
    int64_t levelRowOffset{-1};
    int64_t validCountOffset{-1};
    int64_t csrRowPtrOffset{-1};
    int64_t csrColIndOffset{-1};
    int64_t csrValuesOffset{-1};
    int64_t permOffset{-1};
    int64_t transRowPtrOffset{-1};
    int64_t transColIndOffset{-1};
    int64_t transValuesOffset{-1};
    int64_t transPermOffset{-1};
    size_t totalSize{0};
};

// Compute all workspace sub-allocation offsets.
// - The first offset is 512B-aligned (kAlign) to satisfy GM base alignment.
// - Subsequent sub-allocations are 64B-aligned (kInternalAlign) to reduce
//   workspace bloat for small matrices.
// - When nnz==0 all offsets are -1 (uniform sentinel), totalSize is 0.
// - permType controls the element size for nnz-scaled workspace arrays
//   (perm, transPerm, diagPtr): 0=int32_t, 1=int64_t (nnz > INT32_MAX).
static WorkspaceOffsets ComputeWorkspaceOffsets(
    int64_t m, int64_t nnz, int32_t format, size_t rowPtrSize, size_t colIndSize,
    bool needsTrans, int32_t permType, int32_t idxBase)
{
    using spsv::AlignUp;
    using spsv::AlignUpInternal;

    WorkspaceOffsets out{};

    if (nnz == 0) {
        return out;
    }

    // permSize: element size for nnz-scaled arrays (perm, transPerm, diagPtr).
    // When permType==1 (nnz > INT32_MAX with I64 index), use 8-byte elements.
    size_t permSize = (permType == 1) ? sizeof(int64_t) : sizeof(int32_t);

    // First sub-allocation uses 512B alignment (GM base).
    size_t offset = AlignUp(sizeof(SpsvTilingData));

    // Build workspace CSR copy for: COO(2), SELL(3), or 1-based CSR(0)/CSC(1).
    // For 1-based inputs, workspace CSR holds 0-based indices for downstream ops.
    // rowPtrSize: element width for row-offset arrays (csrRowPtr, transRowPtr)
    // colIndSize: element width for column-index arrays (csrColInd, transColInd)
    if (format == 2 || format == 3 || idxBase == 1) {
        out.csrRowPtrOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(m + 1) * rowPtrSize);
        out.csrColIndOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * colIndSize);
        out.csrValuesOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * sizeof(float));
        out.permOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * permSize);
    }

    out.diagPtrOffset = static_cast<int64_t>(offset);
    offset += AlignUpInternal(static_cast<size_t>(m) * permSize);
    out.levelPtrOffset = static_cast<int64_t>(offset);
    offset += AlignUpInternal(static_cast<size_t>(m + 1) * sizeof(int32_t));
    out.levelRowOffset = static_cast<int64_t>(offset);
    offset += AlignUpInternal(static_cast<size_t>(m) * sizeof(int32_t));
    out.validCountOffset = static_cast<int64_t>(offset);
    offset += AlignUpInternal(static_cast<size_t>(m) * sizeof(int32_t));

    if (needsTrans) {
        out.transRowPtrOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(m + 1) * rowPtrSize);
        out.transColIndOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * colIndSize);
        out.transValuesOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * sizeof(float));
        out.transPermOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * permSize);
    }

    out.totalSize = offset;
    return out;
}

static size_t ComputeWorkspaceSize(int64_t m, int64_t nnz, int32_t format,
    size_t rowPtrSize, size_t colIndSize,
    bool needsTrans, int32_t permType, int32_t idxBase)
{
    return ComputeWorkspaceOffsets(m, nnz, format, rowPtrSize, colIndSize,
        needsTrans, permType, idxBase).totalSize;
}

static float ReadHostScalar(const void *p)
{
    if (p == nullptr) { return 0.0f; }
    return *static_cast<const float *>(p);
}

// Debug override: when SPSV_FORCE_PERMT_64=1 is set, force int64 workspace
// arrays (perm/transPerm/diagPtr) regardless of nnz. Used for validating
// the <IdxT, int64_t> kernel template instantiation on small matrices.
static bool ShouldForcePermType64()
{
    const char *env = std::getenv("SPSV_FORCE_PERMT_64");
    return env && env[0] == '1' && env[1] == '\0';
}

// Select SIMT thread count based on matrix dimension. The if-else chain
// picks the smallest power-of-two >= m (capped at kSimtMaxThreads) to
// avoid launching excess idle threads on small problems.
static uint32_t ComputeNthreads(int64_t m)
{
    if (m <= 64) {
        return 64u;
    }
    if (m <= 128) {
        return 128u;
    }
    if (m <= 256) {
        return 256u;
    }
    return spsv::kSimtMaxThreads;
}

static void ComputeFormatWorkspace(
    SpsvTilingData &tiling, aclsparseSpSVDescr_t spsvDescr,
    int64_t diagPtrOff, int64_t levelPtrOff, int64_t levelRowOff, int64_t validCountOff,
    int64_t csrRowPtrOff, int64_t csrColIndOff, int64_t csrValuesOff, int64_t permOff,
    int64_t transRowPtrOff, int64_t transColIndOff, int64_t transValuesOff, int64_t transPermOff)
{
    tiling.levelPtrOffset = levelPtrOff;
    tiling.levelRowOffset = levelRowOff;
    tiling.diagPtrOffset = diagPtrOff;
    tiling.validCountOffset = validCountOff;
    tiling.numLevels = 0;
    tiling.csrRowPtrOffset = csrRowPtrOff;
    tiling.csrColIndOffset = csrColIndOff;
    tiling.csrValuesOffset = csrValuesOff;
    tiling.permOffset = permOff;
    tiling.transRowPtrOffset = transRowPtrOff;
    tiling.transColIndOffset = transColIndOff;
    tiling.transValuesOffset = transValuesOff;
    tiling.transPermOffset = transPermOff;

    spsvDescr->diagPtrOffset = diagPtrOff;
    spsvDescr->csrValuesOffset = csrValuesOff;
    spsvDescr->permOffset = permOff;
    spsvDescr->transValuesOffset = transValuesOff;
    spsvDescr->transPermOffset = transPermOff;
}

static SpsvTilingData BuildTilingData(
    aclsparseConstSpMatDescr_t matA,
    float alpha,
    aclsparseSpSVDescr_t spsvDescr)
{
    (void)matA; // All attributes read from spsvDescr cache (cuSPARSE convention).
    // Use cached attributes from analysis phase (cuSPARSE convention: solve
    // and updateMatrix read from descriptor, not from matA).
    int32_t format = spsvDescr->cachedFormat;
    int32_t fillMode = spsvDescr->cachedFillMode;
    int32_t diagType = spsvDescr->cachedDiagType;
    int32_t op = spsvDescr->cachedOpA;

    if (format == 1) {  // CSC: cachedFormat=1 means CSC
        format = 0;
        fillMode = (fillMode == 0) ? 1 : 0;
        op = (op == 0) ? 1 : 0;
    }

    int64_t m = spsvDescr->cachedM;
    int64_t nnz = spsvDescr->cachedNnz;
    size_t rowPtrSize = spsvDescr->cachedIdxSize;
    size_t colIndSize = spsvDescr->cachedColIndSize;
    bool needsTrans = (op != 0);

    int32_t permType = spsvDescr->cachedPermType;

    WorkspaceOffsets offsets = ComputeWorkspaceOffsets(
        m, nnz, format, rowPtrSize, colIndSize, needsTrans, permType, spsvDescr->cachedIdxBase);

    uint32_t nthreads = ComputeNthreads(m);

    SpsvTilingData tiling{};
    tiling.m = m;
    tiling.nnz = nnz;
    tiling.alpha = alpha;
    tiling.fillMode = fillMode;
    tiling.diagType = diagType;
    tiling.opA = op;
    tiling.format = format;
    tiling.indexType = spsvDescr->cachedIndexType;
    tiling.colIndType = spsvDescr->cachedColIndType;
    tiling.idxBase = spsvDescr->cachedIdxBase;
    tiling.numSlices = spsvDescr->cachedNumSlices;
    tiling.sliceWidth = spsvDescr->cachedSliceWidth;
    tiling.nthreads = nthreads;
    tiling.permType = permType;
    // numBlocks is computed and overwritten by ComputeNumBlocks in
    // LaunchSpSVAnalysisKernel / LaunchSpSVSolveKernel before launch.
    tiling.numBlocks = 0;

    ComputeFormatWorkspace(tiling, spsvDescr,
        offsets.diagPtrOffset, offsets.levelPtrOffset,
        offsets.levelRowOffset, offsets.validCountOffset,
        offsets.csrRowPtrOffset, offsets.csrColIndOffset,
        offsets.csrValuesOffset, offsets.permOffset,
        offsets.transRowPtrOffset, offsets.transColIndOffset,
        offsets.transValuesOffset, offsets.transPermOffset);

    return tiling;
}

static void GetCsrPointers(
    aclsparseConstSpMatDescr_t matA,
    uint8_t *&rowPtr, uint8_t *&colInd, uint8_t *&values)
{
    auto *mat = spsv::ToMatInner(matA);
    rowPtr = reinterpret_cast<uint8_t *>(mat->ptrs);
    colInd = reinterpret_cast<uint8_t *>(mat->idxs);
    values = reinterpret_cast<uint8_t *>(mat->values);
}

// Compute the number of AI cores (blocks) to launch.
// Deep level-set heuristic: when estimated average level width < 256 rows,
// the per-level SyncAll overhead in multi-core solve dominates. Fall back to
// single core to avoid paying that overhead for negligible parallelism.
// estimatedNumLevels ≈ nnz/m is a rough lower bound; numLevels is only known
// after analysis, so this estimate may miss some deep cases.
static uint32_t ComputeNumBlocks(int64_t m, int64_t nnz, uint32_t nthreads)
{
    if (nthreads == 0) {
        return 1;
    }
    if (m <= static_cast<int64_t>(nthreads)) {
        return 1;
    }
    // Deep level-set: average level width < 256 → SyncAll cost dominates.
    if (nnz > 0 && m > 0) {
        int64_t estimatedNumLevels = nnz / m;
        if (estimatedNumLevels <= 0) {
            estimatedNumLevels = 1;
        }
        int64_t avgLevelWidth = m / estimatedNumLevels;
        if (avgLevelWidth < 256) {
            return 1;
        }
    }
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        return 1;
    }
    uint64_t usefulCores = (static_cast<uint64_t>(m) + nthreads - 1u) / nthreads;
    uint32_t numBlocks = (usefulCores < aivCoreNum) ? static_cast<uint32_t>(usefulCores) : aivCoreNum;
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    return numBlocks;
}

struct SpSVLaunchContext {
    struct aclsparseContext *h;
    aclrtStream useStream;
    SpsvTilingData tiling;
    uint8_t *rowPtr;
    uint8_t *colInd;
    uint8_t *values;
};

static aclsparseStatus_t PrepareSpSVContext(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseSpSVDescr_t spsvDescr,
    SpSVLaunchContext &ctx)
{
    if (alpha == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    ctx.h = spsv::ToInternalHandle(handle);
    ctx.useStream = ctx.h->stream;
    ctx.tiling = BuildTilingData(matA, 0.0f /*alpha*/, spsvDescr);
    if (ctx.h->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        ctx.tiling.alphaDevicePtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alpha));
    } else {
        ctx.tiling.alpha = ReadHostScalar(alpha);
        ctx.tiling.alphaDevicePtr = 0;
    }
    ctx.rowPtr = nullptr;
    ctx.colInd = nullptr;
    ctx.values = nullptr;
    GetCsrPointers(matA, ctx.rowPtr, ctx.colInd, ctx.values);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Cache all matA attributes into descriptor BEFORE BuildTilingData so that
// solve/updateMatrix can rely on cached values (cuSPARSE convention).
static void CacheMatrixAttributes(
    aclsparseConstSpMatDescr_t matA, aclsparseOperation_t opA,
    aclsparseSpSVDescr_t spsvDescr)
{
    auto *mat = spsv::ToMatInner(matA);
    spsvDescr->cachedM = static_cast<int64_t>(mat->rows);
    spsvDescr->cachedNnz = static_cast<int64_t>(mat->nnz);
    spsvDescr->cachedFormat = spsv::FormatToInt(mat->format);
    spsvDescr->cachedDiagType = static_cast<int32_t>(mat->diagType);
    spsvDescr->cachedIdxBase = (mat->baseType == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    spsvDescr->cachedOpA = static_cast<int32_t>(opA);
    spsvDescr->cachedFillMode = static_cast<int32_t>(mat->fillMode);
    spsvDescr->cachedNumSlices = static_cast<int32_t>(mat->numSlices);
    spsvDescr->cachedSliceWidth = static_cast<int32_t>(mat->sliceNnz);
    // ptrType -> rowPtr (indexType/indexSize); IdxType -> colInd (colIndType/colIndSize)
    spsvDescr->cachedIdxSize = (mat->ptrType == ACL_SPARSE_INDEX_64I) ? 8u : 4u;
    spsvDescr->cachedIndexType = (mat->ptrType == ACL_SPARSE_INDEX_64I) ? 1 : 0;
    spsvDescr->cachedColIndSize = (mat->IdxType == ACL_SPARSE_INDEX_64I) ? 8u : 4u;
    spsvDescr->cachedColIndType = (mat->IdxType == ACL_SPARSE_INDEX_64I) ? 1 : 0;
    spsvDescr->cachedPermType = (mat->nnz > static_cast<int64_t>(INT32_MAX)) ? 1 : 0;
    if (spsvDescr->cachedPermType == 0 && ShouldForcePermType64()) {
        spsvDescr->cachedPermType = 1;
    }
    spsvDescr->currentValues = mat->values;
}

static aclsparseStatus_t LaunchSpSVAnalysisKernel(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseSpSVDescr_t spsvDescr,
    void *externalBuffer)
{
    SpSVLaunchContext ctx{};
    aclsparseStatus_t st = PrepareSpSVContext(handle, opA, alpha, matA, spsvDescr, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    uint32_t numBlocks = ComputeNumBlocks(ctx.tiling.m, ctx.tiling.nnz, ctx.tiling.nthreads);
    ctx.tiling.numBlocks = numBlocks;

    OP_LOGI("aclsparseSpSV_analysis",
            "launching analysis kernel: m=%" PRId64 " nnz=%" PRId64 " format=%d numBlocks=%u",
            ctx.tiling.m, ctx.tiling.nnz, ctx.tiling.format, numBlocks);

    GM_ADDR gmRowPtr = reinterpret_cast<GM_ADDR>(ctx.rowPtr);
    GM_ADDR gmColInd = reinterpret_cast<GM_ADDR>(ctx.colInd);
    GM_ADDR gmValues = reinterpret_cast<GM_ADDR>(ctx.values);
    GM_ADDR gmWorkspace = reinterpret_cast<GM_ADDR>(externalBuffer);

    if (numBlocks <= 1) {
        spsv_analysis_kernel_do(
            gmRowPtr, gmColInd, gmValues, gmWorkspace,
            numBlocks, ctx.tiling, ctx.useStream);
    } else {
        // Multi-core analysis runs as three independent launches on the same
        // stream. Stream ordering gives the cross-kernel GM visibility the old
        // in-kernel SyncAll barriers provided, without launching idle blocks
        // for the serial / final phases.
        spsv_analysis_serial_kernel_do(
            gmRowPtr, gmColInd, gmValues, gmWorkspace,
            ctx.tiling, ctx.useStream);
        spsv_analysis_parallel_kernel_do(
            gmRowPtr, gmColInd, gmWorkspace,
            numBlocks, ctx.tiling, ctx.useStream);
        spsv_analysis_final_kernel_do(
            gmWorkspace, ctx.tiling, ctx.useStream);
    }

    spsvDescr->analysisLaunched = true;
    spsvDescr->workspaceBuffer = externalBuffer;

    return ACL_SPARSE_STATUS_SUCCESS;
}

// Shared null-check for SpSV dense vector values (Launch + ZeroNnz paths).
static aclsparseStatus_t ValidateSpSVDenseVecValues(
    aclsparseConstDnVecDescr_t vecX, aclsparseDnVecDescr_t vecY,
    struct aclsparseDnVecDescr **vecXOut, struct aclsparseDnVecDescr **vecYOut)
{
    auto *vecXInner = spsv::ToVecInner(vecX);
    auto *vecYInner = spsv::ToVecInnerMut(vecY);
    if (vecXInner->values == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "vecX.values is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecYInner->values == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "vecY.values is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *vecXOut = vecXInner;
    *vecYOut = vecYInner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchSpSVSolveKernel(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY,
    aclsparseSpSVDescr_t spsvDescr)
{
    SpSVLaunchContext ctx{};
    aclsparseStatus_t st = PrepareSpSVContext(handle, opA, alpha, matA, spsvDescr, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    uint8_t *rowPtr = ctx.rowPtr;
    uint8_t *colInd = ctx.colInd;
    uint8_t *values = ctx.values;

    if (ctx.tiling.transRowPtrOffset >= 0) {
        uint8_t *ws = reinterpret_cast<uint8_t *>(spsvDescr->workspaceBuffer);
        rowPtr = ws + ctx.tiling.transRowPtrOffset;
        colInd = ws + ctx.tiling.transColIndOffset;
        values = ws + ctx.tiling.transValuesOffset;
    } else if (ctx.tiling.csrRowPtrOffset >= 0) {
        uint8_t *ws = reinterpret_cast<uint8_t *>(spsvDescr->workspaceBuffer);
        rowPtr = ws + ctx.tiling.csrRowPtrOffset;
        colInd = ws + ctx.tiling.csrColIndOffset;
        values = ws + ctx.tiling.csrValuesOffset;
    }

    if (spsvDescr->currentValues != nullptr && ctx.tiling.transRowPtrOffset < 0 &&
        ctx.tiling.csrRowPtrOffset < 0) {
        values = reinterpret_cast<uint8_t *>(spsvDescr->currentValues);
    }

    struct aclsparseDnVecDescr *vecXInner = nullptr;
    struct aclsparseDnVecDescr *vecYInner = nullptr;
    aclsparseStatus_t vecSt = ValidateSpSVDenseVecValues(vecX, vecY, &vecXInner, &vecYInner);
    if (vecSt != ACL_SPARSE_STATUS_SUCCESS) {
        return vecSt;
    }

    uint8_t *vecXPtr = reinterpret_cast<uint8_t *>(vecXInner->values);
    uint8_t *vecYPtr = reinterpret_cast<uint8_t *>(vecYInner->values);

    uint32_t numBlocks = ComputeNumBlocks(ctx.tiling.m, ctx.tiling.nnz, ctx.tiling.nthreads);
    ctx.tiling.numBlocks = numBlocks;

    OP_LOGI("aclsparseSpSV_solve", "launching solve kernel: m=%" PRId64 " numBlocks=%u",
            ctx.tiling.m, numBlocks);

    spsv_solve_kernel_do(
        reinterpret_cast<GM_ADDR>(rowPtr),
        reinterpret_cast<GM_ADDR>(colInd),
        reinterpret_cast<GM_ADDR>(values),
        reinterpret_cast<GM_ADDR>(vecXPtr),
        reinterpret_cast<GM_ADDR>(vecYPtr),
        reinterpret_cast<GM_ADDR>(spsvDescr->workspaceBuffer),
        numBlocks, ctx.tiling, ctx.useStream);

    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t HandleSolveZeroNnz(
    aclsparseHandle_t handle, const void *alpha,
    aclsparseConstDnVecDescr_t vecX, aclsparseDnVecDescr_t vecY,
    aclsparseSpSVDescr_t spsvDescr)
{
    if (alpha == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *h = spsv::ToInternalHandle(handle);
    struct aclsparseDnVecDescr *vecXInner = nullptr;
    struct aclsparseDnVecDescr *vecYInner = nullptr;
    aclsparseStatus_t vecSt = ValidateSpSVDenseVecValues(vecX, vecY, &vecXInner, &vecYInner);
    if (vecSt != ACL_SPARSE_STATUS_SUCCESS) {
        return vecSt;
    }
    SpsvTilingData tiling{};
    tiling.m = spsvDescr->cachedM;
    tiling.nthreads = spsv::kSimtMaxThreads;

    tiling.alpha = 0.0f;
    if (h->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        tiling.alphaDevicePtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alpha));
    } else {
        tiling.alpha = ReadHostScalar(alpha);
        tiling.alphaDevicePtr = 0;
    }

    if (spsvDescr->cachedDiagType == 1) {
        spsv_scale_copy_kernel_do(
            reinterpret_cast<GM_ADDR>(vecXInner->values),
            reinterpret_cast<GM_ADDR>(vecYInner->values),
            1, tiling, h->stream);
    } else {  // NON_UNIT: singular matrix, produce Inf/NaN per IEEE-754
        spsv_scale_inf_kernel_do(
            reinterpret_cast<GM_ADDR>(vecXInner->values),
            reinterpret_cast<GM_ADDR>(vecYInner->values),
            1, tiling, h->stream);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Sentinel value for csrValuesOffset: indicates the update-diag kernel should
// write directly to the user's CSR values buffer (passed as a separate GM_ADDR)
// rather than into a workspace-internal CSR copy.
constexpr int64_t kCsrValuesDirectWrite = -2;

static void UpdateMatrixGeneral(
    aclsparseSpSVDescr_t spsvDescr, void *newValues,
    uint8_t *ws, uint8_t *newVals, SpsvTilingData &tiling, aclrtStream useStream)
{
    if (spsvDescr->transValuesOffset >= 0 && spsvDescr->transPermOffset >= 0) {
        // Use perm mapping for ALL transposed formats (CSC, COO+TRANSPOSE,
        // SELL+TRANSPOSE): update_values_kernel performs
        //   dst[pos] = newVals[perm[pos]]   // perm: CSR-of-A index -> source-format index
        tiling.csrValuesOffset = spsvDescr->transValuesOffset;
        tiling.permOffset = spsvDescr->transPermOffset;
        spsv_update_values_kernel_do(
            reinterpret_cast<GM_ADDR>(newVals),
            reinterpret_cast<GM_ADDR>(ws),
            1, tiling, useStream);
    } else if (spsvDescr->csrValuesOffset >= 0) {
        spsv_update_values_kernel_do(
            reinterpret_cast<GM_ADDR>(newVals),
            reinterpret_cast<GM_ADDR>(ws),
            1, tiling, useStream);
    } else {
        spsvDescr->currentValues = newValues;
    }
}

static void UpdateMatrixDiagonal(
    aclsparseSpSVDescr_t spsvDescr,
    uint8_t *ws, uint8_t *newVals, SpsvTilingData &tiling, aclrtStream useStream)
{
    if (spsvDescr->transValuesOffset >= 0) {
        tiling.csrValuesOffset = spsvDescr->transValuesOffset;
        spsv_update_diag_kernel_do(
            reinterpret_cast<GM_ADDR>(newVals),
            reinterpret_cast<GM_ADDR>(ws),
            1, tiling, useStream);
    } else if (spsvDescr->csrValuesOffset >= 0) {
        spsv_update_diag_kernel_do(
            reinterpret_cast<GM_ADDR>(newVals),
            reinterpret_cast<GM_ADDR>(ws),
            1, tiling, useStream);
    } else {
        tiling.csrValuesOffset = kCsrValuesDirectWrite;
        spsv_update_diag_csr_kernel_do(
            reinterpret_cast<GM_ADDR>(newVals),
            reinterpret_cast<GM_ADDR>(spsvDescr->currentValues),
            reinterpret_cast<GM_ADDR>(ws),
            1, tiling, useStream);
    }
}

static aclsparseStatus_t ValidateSpSVSolveConsistency(
    const char *tag, aclsparseOperation_t opA,
    aclsparseConstSpMatDescr_t matA, aclsparseSpSVDescr_t spsvDescr)
{
    if (static_cast<int32_t>(opA) != spsvDescr->cachedOpA) {
        OP_LOGE(tag,
                "opA mismatch: solve opA=%d but analysis opA=%d",
                static_cast<int32_t>(opA), spsvDescr->cachedOpA);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *mat = spsv::ToMatInner(matA);
    if (static_cast<int64_t>(mat->rows) != spsvDescr->cachedM) {
        OP_LOGE(tag,
                "matrix rows changed after analysis: current=%" PRId64 ", cached=%" PRId64,
                static_cast<int64_t>(mat->rows), spsvDescr->cachedM);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (static_cast<int64_t>(mat->nnz) != spsvDescr->cachedNnz) {
        OP_LOGE(tag,
                "matrix nnz changed after analysis: current=%" PRId64 ", cached=%" PRId64,
                static_cast<int64_t>(mat->nnz), spsvDescr->cachedNnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Values pointer: reject swap unless updateMatrix has been called or
    // solve will use a workspace copy (CSR/trans offsets active).
    if (!spsvDescr->updateMatrixCalled &&
        spsvDescr->csrValuesOffset < 0 && spsvDescr->transValuesOffset < 0 &&
        mat->values != spsvDescr->currentValues) {
        OP_LOGE(tag,
                "matrix values pointer changed after analysis "
                "(use aclsparseSpSV_updateMatrix to update values)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static void InitUpdateTilingData(aclsparseSpSVDescr_t spsvDescr, SpsvTilingData &tiling)
{
    tiling = SpsvTilingData{};
    tiling.m = spsvDescr->cachedM;
    tiling.nnz = spsvDescr->cachedNnz;
    tiling.diagPtrOffset = spsvDescr->diagPtrOffset;
    tiling.csrValuesOffset = spsvDescr->csrValuesOffset;
    tiling.permOffset = spsvDescr->permOffset;
    tiling.permType = spsvDescr->cachedPermType;
    tiling.nthreads = spsv::kSimtMaxThreads;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *spsvDescr)
{
    if (spsvDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *inner = new (std::nothrow) aclsparseSpSVDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    *spsvDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr)
{
    if (spsvDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    delete spsvDescr;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpSV_bufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr, size_t *bufferSize)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpSV_bufferSize", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (bufferSize == nullptr) {
        OP_LOGE("aclsparseSpSV_bufferSize", "bufferSize is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclsparseSpSV_bufferSize", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st = ValidateOpA("aclsparseSpSV_bufferSize", opA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    st = ValidateSpSVCommonParams(
        "aclsparseSpSV_bufferSize", matA, spsvDescr, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    auto *mat = spsv::ToMatInner(matA);
    int64_t m = static_cast<int64_t>(mat->rows);
    int64_t nnz = static_cast<int64_t>(mat->nnz);
    int32_t format = spsv::FormatToInt(mat->format);
    bool needsTrans = NeedsTranspose(opA);
    if (mat->format == ACL_SPARSE_FORMAT_CSC) {
        format = 0;
        needsTrans = !needsTrans;
    }
    // rowPtrSize corresponds to ptrType; colIndSize corresponds to IdxType.
    size_t rowPtrSize = (mat->ptrType == ACL_SPARSE_INDEX_64I) ? 8u : 4u;
    size_t colIndSize = (mat->IdxType == ACL_SPARSE_INDEX_64I) ? 8u : 4u;
    int32_t permType = (nnz > static_cast<int64_t>(INT32_MAX)) ? 1 : 0;
    if (permType == 0 && ShouldForcePermType64()) {
        permType = 1;
    }
    int32_t idxBase = (mat->baseType == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    *bufferSize = ComputeWorkspaceSize(m, nnz, format, rowPtrSize, colIndSize,
        needsTrans, permType, idxBase);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpSV_analysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr, void *externalBuffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpSV_analysis", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclsparseSpSV_analysis", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st = ValidateOpA("aclsparseSpSV_analysis", opA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    st = ValidateSpSVCommonParams(
        "aclsparseSpSV_analysis", matA, spsvDescr, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    auto *mat = spsv::ToMatInner(matA);
    CacheMatrixAttributes(matA, opA, spsvDescr);
    if (mat->nnz == 0) {
        spsvDescr->analysisLaunched = true;
        spsvDescr->workspaceBuffer = nullptr;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    if (externalBuffer == nullptr) {
        OP_LOGE("aclsparseSpSV_analysis", "externalBuffer is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return LaunchSpSVAnalysisKernel(handle, opA, alpha, matA, spsvDescr, externalBuffer);
}

aclsparseStatus_t aclsparseSpSV_solve(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st = ValidateOpA("aclsparseSpSV_solve", opA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    st = ValidateSpSVCommonParams(
        "aclsparseSpSV_solve", matA, spsvDescr, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    if (vecX == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "vecX is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecY == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "vecY is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!spsvDescr->analysisLaunched) {
        OP_LOGE("aclsparseSpSV_solve", "analysis not launched, call aclsparseSpSV_analysis first");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    st = ValidateSpSVSolveConsistency("aclsparseSpSV_solve", opA, matA, spsvDescr);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (spsvDescr->cachedNnz == 0) {
        return HandleSolveZeroNnz(handle, alpha, vecX, vecY, spsvDescr);
    }

    return LaunchSpSVSolveKernel(handle, opA, alpha, matA, vecX, vecY, spsvDescr);
}

aclsparseStatus_t aclsparseSpSV_updateMatrix(
    aclsparseHandle_t handle, aclsparseSpSVDescr_t spsvDescr,
    void *newValues, aclsparseSpSVUpdate_t updatePart)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpSV_updateMatrix", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (spsvDescr == nullptr) {
        OP_LOGE("aclsparseSpSV_updateMatrix", "spsvDescr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (newValues == nullptr) {
        OP_LOGE("aclsparseSpSV_updateMatrix", "newValues is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!spsvDescr->analysisLaunched) {
        OP_LOGE("aclsparseSpSV_updateMatrix", "analysis not launched");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (spsvDescr->cachedNnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (spsvDescr->workspaceBuffer == nullptr) {
        OP_LOGE("aclsparseSpSV_updateMatrix", "workspaceBuffer is nullptr after analysis");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto *h = spsv::ToInternalHandle(handle);
    aclrtStream useStream = h->stream;

    SpsvTilingData tiling;
    InitUpdateTilingData(spsvDescr, tiling);

    uint8_t *ws = reinterpret_cast<uint8_t *>(spsvDescr->workspaceBuffer);
    uint8_t *newVals = reinterpret_cast<uint8_t *>(newValues);

    if (updatePart == ACL_SPARSE_SPSV_UPDATE_GENERAL) {
        UpdateMatrixGeneral(spsvDescr, newValues, ws, newVals, tiling, useStream);
        spsvDescr->updateMatrixCalled = true;
    } else if (updatePart == ACL_SPARSE_SPSV_UPDATE_DIAGONAL) {
        UpdateMatrixDiagonal(spsvDescr, ws, newVals, tiling, useStream);
        spsvDescr->updateMatrixCalled = true;
    } else {
        OP_LOGE("aclsparseSpSV_updateMatrix", "unsupported updatePart=%d", updatePart);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
