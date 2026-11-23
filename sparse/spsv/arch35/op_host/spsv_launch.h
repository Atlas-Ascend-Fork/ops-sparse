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

#ifndef OP_HOST_SPSV_LAUNCH_H_
#define OP_HOST_SPSV_LAUNCH_H_

namespace
{
struct SpSVLaunchContext {
    struct aclsparseContext* h;
    aclrtStream useStream;
    SpsvTilingData tiling;
    uint8_t* rowPtr;
    uint8_t* colInd;
    uint8_t* values;
};

static aclsparseStatus_t PrepareSpSVContext(aclsparseHandle_t handle, aclsparseOperation_t opA,
                                            const void* alpha, aclsparseConstSpMatDescr_t matA,
                                            aclsparseSpSVDescr_t spsvDescr, SpSVLaunchContext& ctx)
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
        if (ctx.tiling.valueType == 1) {
            aclsparseComplex c = ReadHostComplexScalar(alpha);
            ctx.tiling.alpha = c.x;
            ctx.tiling.alphaImag = c.y;
        } else {
            ctx.tiling.alpha = ReadHostScalar(alpha);
        }
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
static void CacheMatrixAttributes(aclsparseConstSpMatDescr_t matA, aclsparseOperation_t opA,
                                  aclsparseSpSVDescr_t spsvDescr)
{
    spsvDescr->cachedNumLevels = 0;
    spsvDescr->analysisSnapshotCurrent = false;
    auto* mat = spsv::ToMatInner(matA);
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
    spsvDescr->cachedValueType = (mat->valueType == ACL_COMPLEX64) ? 1 : 0;
    spsvDescr->cachedPermType = (mat->nnz > static_cast<int64_t>(INT32_MAX)) ? 1 : 0;
    if (spsvDescr->cachedPermType == 0 && ShouldForcePermType64()) {
        spsvDescr->cachedPermType = 1;
    }
    spsvDescr->currentValues = mat->values;
}

static void EnqueueSpSVAnalysis(const SpSVLaunchContext& ctx, void* externalBuffer)
{
    uint32_t numBlocks = ctx.tiling.numBlocks;
    GM_ADDR gmRowPtr = reinterpret_cast<GM_ADDR>(ctx.rowPtr);
    GM_ADDR gmColInd = reinterpret_cast<GM_ADDR>(ctx.colInd);
    GM_ADDR gmValues = reinterpret_cast<GM_ADDR>(ctx.values);
    GM_ADDR gmWorkspace = reinterpret_cast<GM_ADDR>(externalBuffer);

    if (ctx.tiling.valueType == 1) {
        // Same three-launch split as the float path above: the complex kernel
        // is the serial phase, and the parallel and final phases take no
        // values, so they are the float instantiations unchanged.  Stream
        // ordering gives the cross-kernel GM visibility.
        spsv_analysis_complex_kernel_do(gmRowPtr, gmColInd, gmValues, gmWorkspace, ctx.tiling, ctx.useStream);
        if (numBlocks > 1) {
            spsv_analysis_parallel_kernel_do(gmRowPtr, gmColInd, gmWorkspace, numBlocks, ctx.tiling,
                                             ctx.useStream);
            spsv_analysis_final_kernel_do(gmWorkspace, ctx.tiling, ctx.useStream);
        }
    } else if (numBlocks <= 1) {
        spsv_analysis_kernel_do(gmRowPtr, gmColInd, gmValues, gmWorkspace, numBlocks, ctx.tiling,
                                ctx.useStream);
    } else {
        // Multi-core analysis runs as three independent launches on the same
        // stream. Stream ordering gives the cross-kernel GM visibility the old
        // in-kernel SyncAll barriers provided, without launching idle blocks
        // for the serial / final phases.
        spsv_analysis_serial_kernel_do(gmRowPtr, gmColInd, gmValues, gmWorkspace, ctx.tiling, ctx.useStream);
        spsv_analysis_parallel_kernel_do(gmRowPtr, gmColInd, gmWorkspace, numBlocks, ctx.tiling,
                                         ctx.useStream);
        spsv_analysis_final_kernel_do(gmWorkspace, ctx.tiling, ctx.useStream);
    }
}

static aclsparseStatus_t LaunchSpSVAnalysisKernel(aclsparseHandle_t handle, aclsparseOperation_t opA,
                                                  const void* alpha, aclsparseConstSpMatDescr_t matA,
                                                  aclsparseSpSVDescr_t spsvDescr, void* externalBuffer)
{
    SpSVLaunchContext ctx{};
    aclsparseStatus_t st = PrepareSpSVContext(handle, opA, alpha, matA, spsvDescr, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = SpsvPrepareScheduleCache(spsvDescr);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    ctx.tiling.numBlocks = ComputeNumBlocks(ctx.tiling.m, ctx.tiling.nnz, ctx.tiling.nthreads);
    OP_LOGI("aclsparseSpSV_analysis",
            "launching analysis kernel: m=%" PRId64 " nnz=%" PRId64 " format=%d numBlocks=%u", ctx.tiling.m,
            ctx.tiling.nnz, ctx.tiling.format, ctx.tiling.numBlocks);
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    EnqueueSpSVAnalysis(ctx, externalBuffer);

    // clear-then-launch-then-peek（评审响应）：Peek 只读不清除，陈旧线程级错误
    // 会被归因本次下发且一次失败后永久粘滞；下发前 aclrtGetLastError 清除。
    // 下发失败不得静默按成功返回；异步执行错误仍由调用方同步观察（issue #357）。
    const aclError analysisEnqueueStatus = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (analysisEnqueueStatus != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpSV_analysis", "kernel enqueue failed: %d",
                static_cast<int>(analysisEnqueueStatus));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    st = SpsvRecordScheduleCache(spsvDescr, externalBuffer, ctx.useStream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    spsvDescr->analysisLaunched = true;
    spsvDescr->workspaceBuffer = externalBuffer;

    return ACL_SPARSE_STATUS_SUCCESS;
}

// Shared null-check for SpSV dense vector values (Launch + ZeroNnz paths).
static aclsparseStatus_t ValidateSpSVDenseVecValues(aclsparseConstDnVecDescr_t vecX,
                                                    aclsparseDnVecDescr_t vecY,
                                                    struct aclsparseDnVecDescr** vecXOut,
                                                    struct aclsparseDnVecDescr** vecYOut)
{
    auto* vecXInner = spsv::ToVecInner(vecX);
    auto* vecYInner = spsv::ToVecInnerMut(vecY);
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

static aclsparseStatus_t ValidateSpSVDenseVecTypes(const char* tag, aclsparseConstSpMatDescr_t matA,
                                                   aclsparseConstDnVecDescr_t vecX,
                                                   aclsparseDnVecDescr_t vecY, aclDataType computeType)
{
    if (vecX == nullptr || vecY == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    auto* mat = spsv::ToMatInner(matA);
    auto* x = spsv::ToVecInner(vecX);
    auto* y = spsv::ToVecInnerMut(vecY);
    if (x->valueType != computeType || y->valueType != computeType || x->valueType != y->valueType) {
        OP_LOGE(tag, "vecX/vecY valueType must match computeType");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (x->nums != mat->rows || y->nums != mat->rows) {
        OP_LOGE(tag, "vecX/vecY length must equal matrix rows");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Solve 内核实参的存储区路由（评审响应：自 LaunchSpSVSolveKernel 提取，压函数行数至 50 内）：
// 转置/CSR workspace 副本优先，无副本时回落用户 currentValues。
static void RouteSpSVSolveStorage(aclsparseSpSVDescr_t spsvDescr, const SpSVLaunchContext& ctx,
                                  uint8_t*& rowPtr, uint8_t*& colInd, uint8_t*& values)
{
    if (ctx.tiling.transRowPtrOffset >= 0) {
        uint8_t* ws = reinterpret_cast<uint8_t*>(spsvDescr->workspaceBuffer);
        rowPtr = ws + ctx.tiling.transRowPtrOffset;
        colInd = ws + ctx.tiling.transColIndOffset;
        values = ws + ctx.tiling.transValuesOffset;
    } else if (ctx.tiling.csrRowPtrOffset >= 0) {
        uint8_t* ws = reinterpret_cast<uint8_t*>(spsvDescr->workspaceBuffer);
        rowPtr = ws + ctx.tiling.csrRowPtrOffset;
        colInd = ws + ctx.tiling.csrColIndOffset;
        values = ws + ctx.tiling.csrValuesOffset;
    }
    if (spsvDescr->currentValues != nullptr && ctx.tiling.transRowPtrOffset < 0 &&
        ctx.tiling.csrRowPtrOffset < 0) {
        values = reinterpret_cast<uint8_t*>(spsvDescr->currentValues);
    }
}

static aclsparseStatus_t PreserveOriginalRhsIfInPlace(const SpSVLaunchContext& ctx,
                                                      aclsparseSpSVDescr_t spsvDescr, uint8_t* vecYPtr,
                                                      uint8_t*& vecXPtr)
{
    if (vecXPtr != vecYPtr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    size_t valueBytes = ctx.tiling.valueType == 1 ? 2 * sizeof(float) : sizeof(float);
    size_t bytes = static_cast<size_t>(ctx.tiling.m) * valueBytes;
    size_t offset = static_cast<size_t>(ctx.tiling.yWideOffset);
    if (ctx.tiling.valueType == 1) {
        offset += static_cast<size_t>(ctx.tiling.m) * 4 * sizeof(float);
    }
    uint8_t* savedX = reinterpret_cast<uint8_t*>(spsvDescr->workspaceBuffer) + offset;
    aclError copyStatus =
        aclrtMemcpyAsync(savedX, bytes, vecXPtr, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.useStream);
    if (copyStatus != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpSV_solve", "RHS snapshot enqueue failed: %d", copyStatus);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    vecXPtr = savedX;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchSpSVSolveKernel(aclsparseHandle_t handle, aclsparseOperation_t opA,
                                               const void* alpha, aclsparseConstSpMatDescr_t matA,
                                               aclsparseConstDnVecDescr_t vecX, aclsparseDnVecDescr_t vecY,
                                               aclsparseSpSVDescr_t spsvDescr)
{
    SpSVLaunchContext ctx{};
    aclsparseStatus_t st = PrepareSpSVContext(handle, opA, alpha, matA, spsvDescr, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    uint8_t* rowPtr = ctx.rowPtr;
    uint8_t* colInd = ctx.colInd;
    uint8_t* values = ctx.values;
    RouteSpSVSolveStorage(spsvDescr, ctx, rowPtr, colInd, values);

    struct aclsparseDnVecDescr* vecXInner = nullptr;
    struct aclsparseDnVecDescr* vecYInner = nullptr;
    aclsparseStatus_t vecSt = ValidateSpSVDenseVecValues(vecX, vecY, &vecXInner, &vecYInner);
    if (vecSt != ACL_SPARSE_STATUS_SUCCESS) {
        return vecSt;
    }

    uint8_t* vecXPtr = reinterpret_cast<uint8_t*>(vecXInner->values);
    uint8_t* vecYPtr = reinterpret_cast<uint8_t*>(vecYInner->values);

    // A fallback can reread this row after the fast path wrote Y. Preserve
    // the ORIGINAL RHS on the same stream, before any solve write, without
    // synchronizing the host. Out-of-place solves pay no copy.
    aclsparseStatus_t snapshotStatus = PreserveOriginalRhsIfInPlace(ctx, spsvDescr, vecYPtr, vecXPtr);
    if (snapshotStatus != ACL_SPARSE_STATUS_SUCCESS) {
        return snapshotStatus;
    }

    uint32_t numBlocks = ComputeNumBlocks(ctx.tiling.m, ctx.tiling.nnz, ctx.tiling.nthreads);

    st = SpsvReadScheduleCache(spsvDescr, ctx.tiling.numLevels);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    numBlocks = SelectComplexScaledBlocks(ctx.tiling, numBlocks);
    // The complex path decides between one block and many on the device, from
    // the level structure analysis recorded; see SpsvSolveComplexDispatch.
    ctx.tiling.numBlocks = numBlocks;

    OP_LOGI("aclsparseSpSV_solve", "launching solve kernel: m=%" PRId64 " numBlocks=%u", ctx.tiling.m,
            numBlocks);

    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    spsv_solve_kernel_do(reinterpret_cast<GM_ADDR>(rowPtr), reinterpret_cast<GM_ADDR>(colInd),
                         reinterpret_cast<GM_ADDR>(values), reinterpret_cast<GM_ADDR>(vecXPtr),
                         reinterpret_cast<GM_ADDR>(vecYPtr),
                         reinterpret_cast<GM_ADDR>(spsvDescr->workspaceBuffer), numBlocks, ctx.tiling,
                         ctx.useStream);

    // clear-then-launch-then-peek（评审响应）：Peek 只读不清除，下发前
    // aclrtGetLastError 清除陈旧错误（issue #357）。
    const aclError solveEnqueueStatus = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (solveEnqueueStatus != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpSV_solve", "kernel enqueue failed: %d", static_cast<int>(solveEnqueueStatus));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t HandleSolveZeroNnz(aclsparseHandle_t handle, const void* alpha,
                                            aclsparseConstDnVecDescr_t vecX, aclsparseDnVecDescr_t vecY,
                                            aclsparseSpSVDescr_t spsvDescr)
{
    if (alpha == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* h = spsv::ToInternalHandle(handle);
    struct aclsparseDnVecDescr* vecXInner = nullptr;
    struct aclsparseDnVecDescr* vecYInner = nullptr;
    aclsparseStatus_t vecSt = ValidateSpSVDenseVecValues(vecX, vecY, &vecXInner, &vecYInner);
    if (vecSt != ACL_SPARSE_STATUS_SUCCESS) {
        return vecSt;
    }
    SpsvTilingData tiling{};
    tiling.m = spsvDescr->cachedM;
    tiling.nthreads = spsv::kSimtMaxThreads;
    tiling.valueType = spsvDescr->cachedValueType;

    tiling.alpha = 0.0f;
    tiling.alphaImag = 0.0f;
    if (h->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        tiling.alphaDevicePtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alpha));
    } else {
        if (tiling.valueType == 1) {
            aclsparseComplex c = ReadHostComplexScalar(alpha);
            tiling.alpha = c.x;
            tiling.alphaImag = c.y;
        } else {
            tiling.alpha = ReadHostScalar(alpha);
        }
        tiling.alphaDevicePtr = 0;
    }

    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    if (spsvDescr->cachedDiagType == 1) {
        spsv_scale_copy_kernel_do(reinterpret_cast<GM_ADDR>(vecXInner->values),
                                  reinterpret_cast<GM_ADDR>(vecYInner->values), 1, tiling, h->stream);
    } else {  // NON_UNIT: singular matrix, produce Inf/NaN per IEEE-754
        spsv_scale_inf_kernel_do(reinterpret_cast<GM_ADDR>(vecXInner->values),
                                 reinterpret_cast<GM_ADDR>(vecYInner->values), 1, tiling, h->stream);
    }
    // ZeroNnz 的 scale 内核同样不得静默假成功；clear-then-launch-then-peek
    // 只归因本次下发（issue #357）。
    const aclError zeroNnzEnqueueStatus = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (zeroNnzEnqueueStatus != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpSV_solve", "zero-nnz kernel enqueue failed: %d",
                static_cast<int>(zeroNnzEnqueueStatus));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Sentinel value for csrValuesOffset: indicates the update-diag kernel should
// write directly to the user's CSR values buffer (passed as a separate GM_ADDR)
// rather than into a workspace-internal CSR copy.
constexpr int64_t kCsrValuesDirectWrite = -2;

static void UpdateMatrixGeneral(aclsparseSpSVDescr_t spsvDescr, void* newValues, uint8_t* ws,
                                uint8_t* newVals, SpsvTilingData& tiling, aclrtStream useStream)
{
    if (spsvDescr->transValuesOffset >= 0 && spsvDescr->transPermOffset >= 0) {
        // Use perm mapping for ALL transposed formats (CSC, COO+TRANSPOSE,
        // SELL+TRANSPOSE): update_values_kernel performs
        //   dst[pos] = newVals[perm[pos]]   // perm: CSR-of-A index -> source-format index
        tiling.csrValuesOffset = spsvDescr->transValuesOffset;
        tiling.permOffset = spsvDescr->transPermOffset;
        spsv_update_values_kernel_do(reinterpret_cast<GM_ADDR>(newVals), reinterpret_cast<GM_ADDR>(ws), 1,
                                     tiling, useStream);
    } else if (spsvDescr->csrValuesOffset >= 0) {
        spsv_update_values_kernel_do(reinterpret_cast<GM_ADDR>(newVals), reinterpret_cast<GM_ADDR>(ws), 1,
                                     tiling, useStream);
    } else {
        spsvDescr->currentValues = newValues;
    }
}

static void UpdateMatrixDiagonal(aclsparseSpSVDescr_t spsvDescr, uint8_t* ws, uint8_t* newVals,
                                 SpsvTilingData& tiling, aclrtStream useStream)
{
    if (spsvDescr->transValuesOffset >= 0) {
        tiling.csrValuesOffset = spsvDescr->transValuesOffset;
        spsv_update_diag_kernel_do(reinterpret_cast<GM_ADDR>(newVals), reinterpret_cast<GM_ADDR>(ws), 1,
                                   tiling, useStream);
    } else if (spsvDescr->csrValuesOffset >= 0) {
        spsv_update_diag_kernel_do(reinterpret_cast<GM_ADDR>(newVals), reinterpret_cast<GM_ADDR>(ws), 1,
                                   tiling, useStream);
    } else {
        tiling.csrValuesOffset = kCsrValuesDirectWrite;
        spsv_update_diag_csr_kernel_do(reinterpret_cast<GM_ADDR>(newVals),
                                       reinterpret_cast<GM_ADDR>(spsvDescr->currentValues),
                                       reinterpret_cast<GM_ADDR>(ws), 1, tiling, useStream);
    }
}

static aclsparseStatus_t ValidateSpSVSolveConsistency(const char* tag, aclsparseOperation_t opA,
                                                      aclsparseConstSpMatDescr_t matA,
                                                      aclsparseSpSVDescr_t spsvDescr)
{
    if (static_cast<int32_t>(opA) != spsvDescr->cachedOpA) {
        OP_LOGE(tag, "opA mismatch: solve opA=%d but analysis opA=%d", static_cast<int32_t>(opA),
                spsvDescr->cachedOpA);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* mat = spsv::ToMatInner(matA);
    if (static_cast<int64_t>(mat->rows) != spsvDescr->cachedM) {
        OP_LOGE(tag, "matrix rows changed after analysis: current=%" PRId64 ", cached=%" PRId64,
                static_cast<int64_t>(mat->rows), spsvDescr->cachedM);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (static_cast<int64_t>(mat->nnz) != spsvDescr->cachedNnz) {
        OP_LOGE(tag, "matrix nnz changed after analysis: current=%" PRId64 ", cached=%" PRId64,
                static_cast<int64_t>(mat->nnz), spsvDescr->cachedNnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Values pointer: reject swap unless updateMatrix has been called or
    // solve will use a workspace copy (CSR/trans offsets active).
    if (!spsvDescr->updateMatrixCalled && spsvDescr->csrValuesOffset < 0 &&
        spsvDescr->transValuesOffset < 0 && mat->values != spsvDescr->currentValues) {
        OP_LOGE(tag,
                "matrix values pointer changed after analysis "
                "(use aclsparseSpSV_updateMatrix to update values)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static void InitUpdateTilingData(aclsparseSpSVDescr_t spsvDescr, SpsvTilingData& tiling)
{
    tiling = SpsvTilingData{};
    tiling.m = spsvDescr->cachedM;
    tiling.nnz = spsvDescr->cachedNnz;
    tiling.diagPtrOffset = spsvDescr->diagPtrOffset;
    tiling.csrValuesOffset = spsvDescr->csrValuesOffset;
    tiling.permOffset = spsvDescr->permOffset;
    tiling.permType = spsvDescr->cachedPermType;
    tiling.nthreads = spsv::kSimtMaxThreads;
    tiling.valueType = spsvDescr->cachedValueType;
    tiling.opA = spsvDescr->cachedOpA;
}

} // namespace

#endif // OP_HOST_SPSV_LAUNCH_H_
