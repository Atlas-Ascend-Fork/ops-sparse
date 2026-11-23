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

/* analysis internals: format normalisation to workspace CSR, transpose and
   permutation, level scheduling, and the phases the analysis kernels call. */

#ifndef SPSV_ANALYSIS_IMPL_H_
#define SPSV_ANALYSIS_IMPL_H_

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "../spsv_kernel.h"
#include "../math/spsv_arith.h"
#include "spsv_format_impl.h"

// Matches the .cpp this was split out of, and the repo's other
// arch35 implementation headers.
using namespace AscendC;

// kSimtMaxThreads is defined in spsv_tiling_data.h (included via spsv_kernel.h).

// Parameter-list macros: eliminate codecheck duplicate-lines warnings by
// declaring shared function parameter lists in a single place. Each macro is
// expanded into the function signature of both the serial and parallel
// variant, so the parameter list never appears twice in the source.

// Dual-template parameter macros for mixed index widths (RowPtrT, ColIndT).
// RowPtrT: element type for row-offset arrays (ptrType from matrix descriptor).
// ColIndT: element type for column-index arrays (idxType from matrix descriptor).
#define SPSV_WORKSPACE_CSR_PARAMS(RowPtrT, ColIndT)                                                                    \
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd, __gm__ const float *values, __gm__ uint8_t *workspace, \
        int64_t m, int64_t nnz, int64_t csrRowPtrOffset, int64_t csrColIndOffset, int64_t csrValuesOffset,             \
        int64_t permOffset, int64_t transRowPtrOffset, int64_t transColIndOffset, int64_t transValuesOffset,           \
        int64_t transPermOffset, int64_t diagPtrOffset, int32_t format, int32_t opA, int32_t numSlices,                \
        int32_t sliceWidth, int32_t permType, int32_t idxBase
// Declare and initialize workspace CSR pointers.
// Declares 10 local variables (wsRowPtr, wsColInd, wsValues, tRowPtr, tColInd,
// tValues, srcRowPtr, srcColInd, srcValues, srcIdxBase) from macro parameters.
#define SPSV_WORKSPACE_CSR_SETUP(RowPtrT, ColIndT)                  \
    __gm__ RowPtrT* wsRowPtr = nullptr;                             \
    __gm__ ColIndT* wsColInd = nullptr;                             \
    __gm__ float* wsValues = nullptr;                               \
    __gm__ RowPtrT* tRowPtr = nullptr;                              \
    __gm__ ColIndT* tColInd = nullptr;                              \
    __gm__ float* tValues = nullptr;                                \
    __gm__ const RowPtrT* srcRowPtr = rowPtr;                       \
    __gm__ const ColIndT* srcColInd = colInd;                       \
    __gm__ const float* srcValues = values;                         \
    int32_t srcIdxBase = idxBase;                                   \
    if (format == 2 || format == 3 || idxBase == 1) {               \
        wsRowPtr = (__gm__ RowPtrT*)(workspace + csrRowPtrOffset);  \
        wsColInd = (__gm__ ColIndT*)(workspace + csrColIndOffset);  \
        wsValues = (__gm__ float*)(workspace + csrValuesOffset);    \
    }                                                               \
    if (opA != 0 && transRowPtrOffset >= 0) {                       \
        tRowPtr = (__gm__ RowPtrT*)(workspace + transRowPtrOffset); \
        tColInd = (__gm__ ColIndT*)(workspace + transColIndOffset); \
        tValues = (__gm__ float*)(workspace + transValuesOffset);   \
    }
// ---------------------------------------------------------------------------
// Shared workspace-setup helper: factors out the common pointer-alias setup
// used by SpsvBuildWorkspaceCsrImpl regardless of the PARALLEL template flag.
// ---------------------------------------------------------------------------

// SpsvSetupFormatWsPtrs has been removed — workspace pointer setup is now
// done inline in SpsvBuildWorkspaceCsr / SpsvBuildWorkspaceCsrParallel so
// that permType-based branching determines whether perm/transPerm/diagPtr
// are int32_t* or int64_t* (__gm__ void* casts at each use site).

template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline int32_t SpsvComputeLevelForRow(__gm__ const RowPtrT* useRowPtr, __gm__ const ColIndT* useColInd,
                                                      __gm__ const PermT* diagPtr, int64_t row, bool forward, int64_t m)
{
    int32_t maxLvl = 0;
    RowPtrT rs = useRowPtr[row];
    RowPtrT re = useRowPtr[row + 1];
    for (RowPtrT p = rs; p < re; p++) {
        int32_t col = static_cast<int32_t>(useColInd[p]);
        if (col < 0 || col >= static_cast<int32_t>(m)) {
            continue;
        }
        bool isDep = forward ? (col < static_cast<int32_t>(row)) : (col > static_cast<int32_t>(row));
        if (isDep) {
            int32_t dep = static_cast<int32_t>(diagPtr[col]) + 1;
            if (dep > maxLvl) {
                maxLvl = dep;
            }
        }
    }
    return maxLvl;
}
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvComputeLevels(__gm__ const RowPtrT* useRowPtr, __gm__ const ColIndT* useColInd,
                                              __gm__ PermT* diagPtr, int64_t m, bool forward)
{
    for (int64_t i = 0; i < m; i++) {
        diagPtr[i] = 0;
    }
    if (forward) {
        for (int64_t i = 0; i < m; i++) {
            diagPtr[i] = SpsvComputeLevelForRow<RowPtrT, ColIndT, PermT>(useRowPtr, useColInd, diagPtr, i, true, m);
        }
    } else {
        for (int64_t i = m - 1; i >= 0; i--) {
            diagPtr[i] = SpsvComputeLevelForRow<RowPtrT, ColIndT, PermT>(useRowPtr, useColInd, diagPtr, i, false, m);
        }
    }
}
template <typename PermT>
__simt_callee__ inline int32_t SpsvBuildLevelHistogram(__gm__ PermT* diagPtr, __gm__ int32_t* levelPtr, int64_t m)
{
    int32_t numLevels = 0;
    for (int64_t i = 0; i < m; i++) {
        if (static_cast<int32_t>(diagPtr[i]) > numLevels) {
            numLevels = static_cast<int32_t>(diagPtr[i]);
        }
    }
    numLevels += 1;
    for (int32_t k = 0; k <= numLevels; k++) {
        levelPtr[k] = 0;
    }
    for (int64_t i = 0; i < m; i++) {
        levelPtr[static_cast<int32_t>(diagPtr[i]) + 1]++;
    }
    for (int32_t k = 0; k < numLevels; k++) {
        levelPtr[k + 1] += levelPtr[k];
    }
    return numLevels;
}
template <typename RowPtrT, typename ColIndT>
__simt_callee__ inline void SpsvResolveEffectivePtrs(__gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd,
                                                     __gm__ uint8_t* workspace, int32_t fillMode, int32_t opA,
                                                     int64_t csrRowPtrOffset, int64_t csrColIndOffset,
                                                     int64_t transRowPtrOffset, int64_t transColIndOffset,
                                                     int32_t format, int32_t idxBase, __gm__ const RowPtrT** effRowPtr,
                                                     __gm__ const ColIndT** effColInd, bool* effForward)
{
    *effRowPtr = rowPtr;
    *effColInd = colInd;
    int32_t effFillMode = fillMode;
    int32_t effOpA = opA;

    // Use workspace CSR if format requires conversion (COO/SELL) OR
    // if input is 1-based CSR (analysis phase built 0-based copy there).
    if (format == 2 || format == 3 || idxBase == 1) {
        *effRowPtr = (__gm__ const RowPtrT*)(workspace + csrRowPtrOffset);
        *effColInd = (__gm__ const ColIndT*)(workspace + csrColIndOffset);
    }
    if (effOpA != 0 && transRowPtrOffset >= 0) {
        *effRowPtr = (__gm__ const RowPtrT*)(workspace + transRowPtrOffset);
        *effColInd = (__gm__ const ColIndT*)(workspace + transColIndOffset);
        effFillMode = (effFillMode == 0) ? 1 : 0;
        effOpA = 0;
    }
    *effForward = false;
    if (effFillMode == 0 && effOpA == 0) {
        *effForward = true;
    }
    if (effFillMode == 1 && effOpA != 0) {
        *effForward = true;
    }
}
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvComputeDiagAndValidCount(__gm__ const RowPtrT* effRowPtr,
                                                         __gm__ const ColIndT* effColInd, __gm__ PermT* diagPtr,
                                                         __gm__ int32_t* validCount, int64_t row, bool effForward)
{
    RowPtrT rs = effRowPtr[row];
    RowPtrT re = effRowPtr[row + 1];

    diagPtr[row] = static_cast<PermT>(-1);
    for (RowPtrT p = rs; p < re; p++) {
        if (effColInd[p] == static_cast<ColIndT>(row)) {
            diagPtr[row] = static_cast<PermT>(p);
            break;
        }
    }

    int32_t vc = 0;
    if (effForward) {
        for (RowPtrT p = rs; p < re; p++) {
            if (effColInd[p] < static_cast<ColIndT>(row)) {
                vc++;
            } else {
                break;
            }
        }
    } else {
        // Guard against empty row (re == rs): without this, p = re - 1 would
        // underflow if RowPtrT were ever instantiated as unsigned.
        if (re > rs) {
            for (RowPtrT p = re - 1; p >= rs; p--) {
                if (effColInd[p] > static_cast<ColIndT>(row)) {
                    vc++;
                } else {
                    break;
                }
            }
        }
    }
    validCount[row] = vc;
}
// Serial workspace CSR build. Delegates to SPSV_CONVERT_FORMAT_TO_CSR_SERIAL /
// SPSV_TRANSPOSE_CSR_IF_NEEDED_SERIAL macros for the single-core analysis path.
// permType selects the element width of perm/transPerm/diagPtr workspace
// arrays: 0=int32_t, 1=int64_t (nnz > INT32_MAX).
// idxBase: 0 for 0-based input, 1 for 1-based (Fortran indexing).
// Dual-template: RowPtrT for row-offset arrays, ColIndT for column-index arrays.
template <typename RowPtrT, typename ColIndT>
__simt_callee__ inline void SpsvBuildWorkspaceCsr(SPSV_WORKSPACE_CSR_PARAMS(RowPtrT, ColIndT))
{
    SPSV_WORKSPACE_CSR_SETUP(RowPtrT, ColIndT)

    SPSV_CONVERT_FORMAT_TO_CSR_SERIAL(RowPtrT, ColIndT);
    SPSV_TRANSPOSE_CSR_IF_NEEDED_SERIAL(RowPtrT, ColIndT);
}
// Parallel workspace CSR build. Delegates to SPSV_CONVERT_FORMAT_TO_CSR_PARALLEL /
// SPSV_TRANSPOSE_CSR_IF_NEEDED_PARALLEL macros for the multi-core serial phase path.
// When permType==1 (nnz > INT32_MAX), falls back to serial conversion
// in thread 0 to avoid int64 atomic_add dependency.
// TODO(ops-perf): when nnz > INT32_MAX, format conversion falls back to single-thread
// serial execution which may be slow for very large matrices. Consider multi-core
// parallel with per-thread range partitioning and final reduction.
// idxBase: 0 for 0-based input, 1 for 1-based (Fortran indexing).
// Dual-template: RowPtrT for row-offset arrays, ColIndT for column-index arrays.
template <typename RowPtrT, typename ColIndT>
__simt_callee__ inline void SpsvBuildWorkspaceCsrParallel(SPSV_WORKSPACE_CSR_PARAMS(RowPtrT, ColIndT))
{
    SPSV_WORKSPACE_CSR_SETUP(RowPtrT, ColIndT)

    SPSV_CONVERT_FORMAT_TO_CSR_PARALLEL(RowPtrT, ColIndT);
    SPSV_TRANSPOSE_CSR_IF_NEEDED_PARALLEL(RowPtrT, ColIndT);
}
// Parallel level histogram build using all threads in the block.
// Thread 0 sequentially finds maxLevel and zeros levelPtr (O(m) + O(numLevels),
// fast for moderate m). All threads then scatter diagPtr[i] into levelPtr via
// asc_atomic_add (O(m) parallelized). Thread 0 does the final prefix-sum
// (O(numLevels), very small).
// On return, tilingOut->numLevels holds the level count, and levelPtr contains
// the correct prefix-sum of per-level row counts. SpsvFinalizeLevelPtr (called
// by the separate SpsvAnalysisFinalPhase kernel after the parallel scatter)
// right-shifts levelPtr and writes numLevels.
template <typename PermT>
__simt_callee__ inline void SpsvBuildLevelHistogramParallel(__gm__ PermT* diagPtr, __gm__ int32_t* levelPtr,
                                                            __gm__ SpsvTilingData* tilingOut, int64_t m)
{
    if (threadIdx.x == 0) {
        int32_t numLevels = 0;
        for (int64_t i = 0; i < m; i++) {
            int32_t lvl = static_cast<int32_t>(diagPtr[i]);
            if (lvl > numLevels) {
                numLevels = lvl;
            }
        }
        numLevels += 1;
        tilingOut->numLevels = numLevels;
        tilingOut->primaryStatus = 0;
        for (int32_t k = 0; k <= numLevels; k++) {
            levelPtr[k] = 0;
        }
    }
    asc_syncthreads();

    int32_t numLevels = tilingOut->numLevels;

    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        asc_atomic_add(&levelPtr[static_cast<int32_t>(diagPtr[i]) + 1], 1);
    }
    asc_syncthreads();

    if (threadIdx.x == 0) {
        for (int32_t k = 0; k < numLevels; k++) {
            levelPtr[k + 1] += levelPtr[k];
        }
    }
    asc_syncthreads();
}
// Finalize levelPtr prefix-sum and write runtime-computed numLevels into
// the SpsvTilingData at workspace offset 0.
// Convention: the host reserves AlignUp(sizeof(SpsvTilingData)) bytes at the
// start of workspace for this purpose; the analysis kernel writes here and
// the solve kernel reads numLevels from here (NOT from the tiling parameter).
__simt_callee__ inline void SpsvFinalizeLevelPtr(__gm__ int32_t* levelPtr, __gm__ SpsvTilingData* tilingOut,
                                                 int32_t numLevels)
{
    for (int32_t k = numLevels; k >= 1; k--) {
        levelPtr[k] = levelPtr[k - 1];
    }
    levelPtr[0] = 0;
    tilingOut->numLevels = numLevels;
    tilingOut->primaryStatus = 0;
}
// Resolves effective row/col pointers from workspace offsets and computes the
// level structure on the resolved matrix. Encapsulates the declare-pointer,
// call SpsvResolveEffectivePtrs, call SpsvComputeLevels pattern that
// appeared identically in SpsvAnalysisCommon and SpsvAnalysisSerialPhase.
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvResolveAndComputeLevels(__gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd,
                                                        __gm__ uint8_t* workspace, __gm__ PermT* diagPtr, int64_t m,
                                                        int32_t fillMode, int32_t opA, int64_t csrRowPtrOffset,
                                                        int64_t csrColIndOffset, int64_t transRowPtrOffset,
                                                        int64_t transColIndOffset, int32_t format, int32_t idxBase)
{
    __gm__ const RowPtrT* useRowPtr = nullptr;
    __gm__ const ColIndT* useColInd = nullptr;
    bool forward = false;
    SpsvResolveEffectivePtrs<RowPtrT, ColIndT>(rowPtr, colInd, workspace, fillMode, opA, csrRowPtrOffset,
                                               csrColIndOffset, transRowPtrOffset, transColIndOffset, format, idxBase,
                                               &useRowPtr, &useColInd, &forward);
    SpsvComputeLevels<RowPtrT, ColIndT, PermT>(useRowPtr, useColInd, diagPtr, m, forward);
}
// SPSV_ANALYSIS_PARAMS: shared parameter list for analysis VF functions.
// Kept as a macro because SIMT __simt_vf__ function signatures must be
// explicitly spelled out for the VF compiler; a struct-based approach is
// not supported by the asc_vf_call template instantiation mechanism.
#define SPSV_ANALYSIS_PARAMS                                                                                           \
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd, __gm__ const float *values, __gm__ uint8_t *workspace, \
        int64_t m, int64_t nnz, int32_t fillMode, int32_t opA, int64_t levelPtrOffset, int64_t levelRowOffset,         \
        int64_t diagPtrOffset, int64_t validCountOffset, int64_t csrRowPtrOffset, int64_t csrColIndOffset,             \
        int64_t csrValuesOffset, int64_t permOffset, int64_t transRowPtrOffset, int64_t transColIndOffset,             \
        int64_t transValuesOffset, int64_t transPermOffset, int32_t format, int32_t numSlices, int32_t sliceWidth,     \
        int32_t permType, int32_t idxBase
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline int32_t SpsvAnalysisCommon(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ const float* values, __gm__ uint8_t* workspace,
    __gm__ PermT* diagPtr, __gm__ int32_t* levelPtr, int64_t m, int64_t nnz, int32_t fillMode, int32_t opA,
    int64_t csrRowPtrOffset, int64_t csrColIndOffset, int64_t csrValuesOffset, int64_t permOffset,
    int64_t transRowPtrOffset, int64_t transColIndOffset, int64_t transValuesOffset, int64_t transPermOffset,
    int64_t diagPtrOffset, int32_t format, int32_t numSlices, int32_t sliceWidth, int32_t permType, int32_t idxBase)
{
    SpsvBuildWorkspaceCsr<RowPtrT, ColIndT>(rowPtr, colInd, values, workspace, m, nnz, csrRowPtrOffset, csrColIndOffset,
                                            csrValuesOffset, permOffset, transRowPtrOffset, transColIndOffset,
                                            transValuesOffset, transPermOffset, diagPtrOffset, format, opA, numSlices,
                                            sliceWidth, permType, idxBase);

    // Resolve effective pointers and compute level structure in a single call.
    // Pass idxBase so it can redirect to workspace CSR for 1-based input.
    SpsvResolveAndComputeLevels<RowPtrT, ColIndT, PermT>(rowPtr, colInd, workspace, diagPtr, m, fillMode, opA,
                                                         csrRowPtrOffset, csrColIndOffset, transRowPtrOffset,
                                                         transColIndOffset, format, idxBase);

    return SpsvBuildLevelHistogram<PermT>(diagPtr, levelPtr, m);
}
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvAnalysisSimtCompute(SPSV_ANALYSIS_PARAMS)
{
    __gm__ int32_t* levelPtr = (__gm__ int32_t*)(workspace + levelPtrOffset);
    __gm__ int32_t* levelRow = (__gm__ int32_t*)(workspace + levelRowOffset);
    __gm__ PermT* diagPtr = (__gm__ PermT*)(workspace + diagPtrOffset);
    __gm__ int32_t* validCount = (__gm__ int32_t*)(workspace + validCountOffset);
    __gm__ SpsvTilingData* tilingOut = (__gm__ SpsvTilingData*)workspace;

    int32_t numLevels = 0;

    if (threadIdx.x == 0) {
        numLevels = SpsvAnalysisCommon<RowPtrT, ColIndT, PermT>(
            rowPtr, colInd, values, workspace, diagPtr, levelPtr, m, nnz, fillMode, opA, csrRowPtrOffset,
            csrColIndOffset, csrValuesOffset, permOffset, transRowPtrOffset, transColIndOffset, transValuesOffset,
            transPermOffset, diagPtrOffset, format, numSlices, sliceWidth, permType, idxBase);
    }

    // thread 0 wrote diagPtr/levelPtr to GM above. asc_threadfence_block
    // ensures those GM writes are visible to all other threads in this block
    // before asc_syncthreads releases them to read diagPtr[i].
    asc_threadfence_block();
    asc_syncthreads();

    __gm__ const RowPtrT* effRowPtr = nullptr;
    __gm__ const ColIndT* effColInd = nullptr;
    bool effForward = false;
    SpsvResolveEffectivePtrs<RowPtrT, ColIndT>(rowPtr, colInd, workspace, fillMode, opA, csrRowPtrOffset,
                                               csrColIndOffset, transRowPtrOffset, transColIndOffset, format, idxBase,
                                               &effRowPtr, &effColInd, &effForward);

    // DEFENSIVE: host ValidateSpSVCommonParams guarantees m <= INT32_MAX.
    // All levelRow/validCount arrays are int32_t; diagPtr values are level
    // numbers ≤ numLevels ≤ m, cast to int32_t is safe.
    // The same guarantee covers SpsvAnalysisParallelPhase.
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        int32_t lvl = static_cast<int32_t>(diagPtr[i]);
        // Scatter rows into level buckets via atomic_add on the prefix-sum
        // array. levelPtr[lvl] starts at the bucket's base offset; each
        // atomic_add returns a unique slot and increments the counter.
        // After all rows are scattered, levelPtr values are "corrupted"
        // (shifted by the bucket width). SpsvFinalizeLevelPtr right-shifts
        // the array to restore correct prefix-sum boundaries.
        int32_t pos = asc_atomic_add(&levelPtr[lvl], 1);
        levelRow[pos] = static_cast<int32_t>(i);
    }

    asc_syncthreads();

    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        SpsvComputeDiagAndValidCount<RowPtrT, ColIndT, PermT>(effRowPtr, effColInd, diagPtr, validCount, i, effForward);
    }

    if (threadIdx.x == 0) {
        SpsvFinalizeLevelPtr(levelPtr, tilingOut, numLevels);
    }

    asc_syncthreads();
}
// Serial phase: launched as a standalone single-block kernel with nthreads
// threads. FORMAT CONVERSION and HISTOGRAM are parallelized across all threads
// in the block; LEVEL COMPUTATION runs on thread 0 only (sequential chain).
// Splitting this into its own single-block launch avoids scheduling the
// (numBlocks-1) idle blocks the old in-kernel SyncAll design required.
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvAnalysisSerialPhase(SPSV_ANALYSIS_PARAMS)
{
    __gm__ int32_t* levelPtr = (__gm__ int32_t*)(workspace + levelPtrOffset);
    __gm__ PermT* diagPtr = (__gm__ PermT*)(workspace + diagPtrOffset);
    __gm__ SpsvTilingData* tilingOut = (__gm__ SpsvTilingData*)workspace;

    // Phase 1: Format conversion (all threads in block).
    // SpsvBuildWorkspaceCsrParallel ends with asc_syncthreads() internally
    // when it does any work; for CSR (format == 0) without transpose it is
    // a no-op, which is fine because no shared state needs synchronizing.
    // When permType==1 (nnz > INT32_MAX), the parallel function internally
    // falls back to serial conversion (thread 0 only).
    SpsvBuildWorkspaceCsrParallel<RowPtrT, ColIndT>(
        rowPtr, colInd, values, workspace, m, nnz, csrRowPtrOffset, csrColIndOffset, csrValuesOffset, permOffset,
        transRowPtrOffset, transColIndOffset, transValuesOffset, transPermOffset, diagPtrOffset, format, opA, numSlices,
        sliceWidth, permType, idxBase);

    // Phase 2: Resolve effective pointers and compute level structure
    // (thread 0 only — true sequential dependency: level[i] depends on
    // diagPtr[col] for col in row i's dependencies).
    if (threadIdx.x == 0) {
        SpsvResolveAndComputeLevels<RowPtrT, ColIndT, PermT>(rowPtr, colInd, workspace, diagPtr, m, fillMode, opA,
                                                             csrRowPtrOffset, csrColIndOffset, transRowPtrOffset,
                                                             transColIndOffset, format, idxBase);
    }
    // Publish thread 0's GM levels before the block builds the histogram.
    asc_threadfence_block();
    asc_syncthreads();

    // Phase 3: Parallel histogram build (all threads in block).
    // Writes tilingOut->numLevels and the prefix-summed levelPtr array.
    SpsvBuildLevelHistogramParallel<PermT>(diagPtr, levelPtr, tilingOut, m);
}
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvAnalysisParallelPhase(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ uint8_t* workspace, int64_t m, int32_t fillMode,
    int32_t opA, int64_t levelPtrOffset, int64_t levelRowOffset, int64_t diagPtrOffset, int64_t validCountOffset,
    int64_t csrRowPtrOffset, int64_t csrColIndOffset, int64_t transRowPtrOffset, int64_t transColIndOffset,
    int32_t format, int32_t numBlocks, int32_t idxBase)
{
    __gm__ int32_t* levelPtr = (__gm__ int32_t*)(workspace + levelPtrOffset);
    __gm__ int32_t* levelRow = (__gm__ int32_t*)(workspace + levelRowOffset);
    __gm__ PermT* diagPtr = (__gm__ PermT*)(workspace + diagPtrOffset);
    __gm__ int32_t* validCount = (__gm__ int32_t*)(workspace + validCountOffset);

    __gm__ const RowPtrT* parRowPtr = nullptr;
    __gm__ const ColIndT* parColInd = nullptr;
    bool parForward = false;
    SpsvResolveEffectivePtrs<RowPtrT, ColIndT>(rowPtr, colInd, workspace, fillMode, opA, csrRowPtrOffset,
                                               csrColIndOffset, transRowPtrOffset, transColIndOffset, format, idxBase,
                                               &parRowPtr, &parColInd, &parForward);

    int64_t globalTid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                        static_cast<int64_t>(threadIdx.x);
    int64_t globalStride = static_cast<int64_t>(numBlocks) * static_cast<int64_t>(blockDim.x);

    // DEFENSIVE: host guarantees m <= INT32_MAX; all row-index → int32_t
    // casts in this parallel phase (levelRow scatter, diagPtr, validCount)
    // are safe from silent truncation.
    for (int64_t i = globalTid; i < m; i += globalStride) {
        int32_t lvl = static_cast<int32_t>(diagPtr[i]);
        int32_t pos = asc_atomic_add(&levelPtr[lvl], 1);
        levelRow[pos] = static_cast<int32_t>(i);
    }

    for (int64_t i = globalTid; i < m; i += globalStride) {
        SpsvComputeDiagAndValidCount<RowPtrT, ColIndT, PermT>(parRowPtr, parColInd, diagPtr, validCount, i, parForward);
    }
}
// Final phase: launched as a standalone single-block kernel. Only thread 0
// finalizes the levelPtr prefix-sum. Same rationale as SpsvAnalysisSerialPhase.
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvAnalysisFinalPhase(__gm__ uint8_t* workspace,
                                                                                             int64_t levelPtrOffset)
{
    __gm__ int32_t* levelPtr = (__gm__ int32_t*)(workspace + levelPtrOffset);
    __gm__ SpsvTilingData* tilingOut = (__gm__ SpsvTilingData*)workspace;

    if (threadIdx.x == 0) {
        int32_t numLevels = tilingOut->numLevels;
        SpsvFinalizeLevelPtr(levelPtr, tilingOut, numLevels);
    }
}

#endif // SPSV_ANALYSIS_IMPL_H_
