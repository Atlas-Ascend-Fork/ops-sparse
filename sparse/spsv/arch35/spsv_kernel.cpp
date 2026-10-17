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
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "spsv_kernel.h"

// using-namespace is acceptable in this .cpp translation unit (not a header).
// AscendC symbols (TPipe, SetFlag, SyncAll, etc.) are used extensively throughout.
using namespace AscendC;

// kSimtMaxThreads is defined in spsv_tiling_data.h (included via spsv_kernel.h).

// Parameter-list macros: eliminate codecheck duplicate-lines warnings by
// declaring shared function parameter lists in a single place. Each macro is
// expanded into the function signature of both the serial and parallel
// variant, so the parameter list never appears twice in the source.

// Dual-template parameter macros for mixed index widths (RowPtrT, ColIndT).
// RowPtrT: element type for row-offset arrays (ptrType from matrix descriptor).
// ColIndT: element type for column-index arrays (idxType from matrix descriptor).

#define SPSV_COO_BUILD_PARAMS(RowPtrT, ColIndT, PermT) \
    __gm__ const RowPtrT *cooRowInd,   \
    __gm__ const ColIndT *cooColInd,   \
    __gm__ const float *cooValues, \
    __gm__ RowPtrT *wsRowPtr,          \
    __gm__ ColIndT *wsColInd,          \
    __gm__ float *wsValues,      \
    __gm__ PermT *perm,          \
    __gm__ PermT *scratch,       \
    int64_t m, int64_t nnz, int32_t idxBase

#define SPSV_SLICED_ELL_PARAMS(RowPtrT, ColIndT, PermT) \
    __gm__ const RowPtrT *slicePtr,    \
    __gm__ const ColIndT *sellColInd,  \
    __gm__ const float *sellValues, \
    __gm__ RowPtrT *wsRowPtr,          \
    __gm__ ColIndT *wsColInd,          \
    __gm__ float *wsValues,      \
    __gm__ PermT *perm,          \
    int64_t m, int32_t numSlices, int32_t sliceWidth, int32_t idxBase

#define SPSV_TRANS_PARAMS(RowPtrT, ColIndT, PermT) \
    __gm__ const RowPtrT *rowPtr, \
    __gm__ const ColIndT *colInd, \
    __gm__ const float *values, \
    __gm__ RowPtrT *transRowPtr,  \
    __gm__ ColIndT *transColInd,  \
    __gm__ float *transValues, \
    __gm__ PermT *transPerm, \
    __gm__ PermT *scratch, \
    int64_t m, int64_t nnz, int32_t idxBase

#define SPSV_WORKSPACE_CSR_PARAMS(RowPtrT, ColIndT) \
    __gm__ const RowPtrT *rowPtr, \
    __gm__ const ColIndT *colInd, \
    __gm__ const float *values, \
    __gm__ uint8_t *workspace, \
    int64_t m, int64_t nnz, \
    int64_t csrRowPtrOffset, int64_t csrColIndOffset, \
    int64_t csrValuesOffset, int64_t permOffset, \
    int64_t transRowPtrOffset, int64_t transColIndOffset, \
    int64_t transValuesOffset, int64_t transPermOffset, \
    int64_t diagPtrOffset, \
    int32_t format, int32_t opA, int32_t numSlices, int32_t sliceWidth, \
    int32_t permType, int32_t idxBase

// Declare and initialize workspace CSR pointers.
// Declares 10 local variables (wsRowPtr, wsColInd, wsValues, tRowPtr, tColInd,
// tValues, srcRowPtr, srcColInd, srcValues, srcIdxBase) from macro parameters.
#define SPSV_WORKSPACE_CSR_SETUP(RowPtrT, ColIndT)                              \
    __gm__ RowPtrT *wsRowPtr = nullptr;                                         \
    __gm__ ColIndT *wsColInd = nullptr;                                         \
    __gm__ float *wsValues = nullptr;                                           \
    __gm__ RowPtrT *tRowPtr = nullptr;                                          \
    __gm__ ColIndT *tColInd = nullptr;                                          \
    __gm__ float *tValues = nullptr;                                            \
    __gm__ const RowPtrT *srcRowPtr = rowPtr;                                   \
    __gm__ const ColIndT *srcColInd = colInd;                                   \
    __gm__ const float *srcValues = values;                                     \
    int32_t srcIdxBase = idxBase;                                               \
    if (format == 2 || format == 3 || idxBase == 1) {                           \
        wsRowPtr = (__gm__ RowPtrT *)(workspace + csrRowPtrOffset);             \
        wsColInd = (__gm__ ColIndT *)(workspace + csrColIndOffset);             \
        wsValues = (__gm__ float *)(workspace + csrValuesOffset);               \
    }                                                                           \
    if (opA != 0 && transRowPtrOffset >= 0) {                                   \
        tRowPtr = (__gm__ RowPtrT *)(workspace + transRowPtrOffset);            \
        tColInd = (__gm__ ColIndT *)(workspace + transColIndOffset);            \
        tValues = (__gm__ float *)(workspace + transValuesOffset);              \
    }

// Setup workspace pointer helpers below are called from SpsvBuildWorkspaceCsr
// and SpsvBuildWorkspaceCsrParallel with runtime permType branching.

// Serial COO -> CSR conversion (single-thread, thread 0 only).
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvBuildCsrFromCoo(SPSV_COO_BUILD_PARAMS(RowPtrT, ColIndT, PermT))
{
    // wsRowPtr has m+1 elements (allocated by host via ComputeWorkspaceSize).
    // idxBase: 0 for 0-based COO, 1 for 1-based (Fortran). Subtract idxBase
    // when reading row/col indices to normalize to 0-based workspace CSR.
    for (int64_t i = 0; i <= m; i++) {
        wsRowPtr[i] = 0;
    }
    for (int64_t p = 0; p < nnz; p++) {
        // Bounds check in original type first to avoid int64-to-int32 truncation
        // silently wrapping large unsigned values into [0, m). Host guarantees
        // valid input; this guard prevents silent GM corruption on illegal data.
        RowPtrT rawR = cooRowInd[p] - static_cast<RowPtrT>(idxBase);
        ColIndT rawC = cooColInd[p] - static_cast<ColIndT>(idxBase);
        if (rawR < 0 || rawR >= static_cast<RowPtrT>(m) || rawC < 0 || rawC >= static_cast<ColIndT>(m)) {
            continue;
        }
        int32_t r = static_cast<int32_t>(rawR);
        wsRowPtr[r + 1]++;
    }
    for (int64_t i = 0; i < m; i++) {
        wsRowPtr[i + 1] += wsRowPtr[i];
    }
    for (int64_t i = 0; i < m; i++) {
        scratch[i] = 0;
    }
    for (int64_t p = 0; p < nnz; p++) {
        // Bounds check in original type first to avoid int64-to-int32 truncation.
        RowPtrT rawR = cooRowInd[p] - static_cast<RowPtrT>(idxBase);
        ColIndT rawC = cooColInd[p] - static_cast<ColIndT>(idxBase);
        if (rawR < 0 || rawR >= static_cast<RowPtrT>(m) || rawC < 0 || rawC >= static_cast<ColIndT>(m)) {
            continue;
        }
        int32_t row = static_cast<int32_t>(rawR);
        RowPtrT pos = wsRowPtr[row] + static_cast<RowPtrT>(scratch[row]);
        scratch[row]++;
        // Write 0-based column index to workspace CSR directly in ColIndT
        // to remain consistent with the parallel path (Issue 12 fix).
        wsColInd[pos] = cooColInd[p] - static_cast<ColIndT>(idxBase);
        wsValues[pos] = cooValues[p];
        perm[pos] = static_cast<PermT>(p);
    }
}

// Parallel COO -> CSR conversion. All threads in the block participate using
// atomic histogram and scatter (multi-core serial phase path).
// idxBase: 0 for 0-based COO, 1 for 1-based (Fortran).
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvBuildCsrFromCooParallel(SPSV_COO_BUILD_PARAMS(RowPtrT, ColIndT, PermT))
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i <= m;
         i += static_cast<int64_t>(blockDim.x)) {
        wsRowPtr[i] = 0;
    }
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < m;
         i += static_cast<int64_t>(blockDim.x)) {
        scratch[i] = 0;
    }
    asc_syncthreads();

    for (int64_t p = static_cast<int64_t>(threadIdx.x);
         p < nnz;
         p += static_cast<int64_t>(blockDim.x)) {
        // Bounds check in original type first to avoid int64-to-int32 truncation
        // bypassing the range check (issues #149/#150).
        RowPtrT rawR = cooRowInd[p] - static_cast<RowPtrT>(idxBase);
        if (rawR < 0 || rawR >= static_cast<RowPtrT>(m)) {
            continue;
        }
        int32_t r = static_cast<int32_t>(rawR);
        asc_atomic_add(&wsRowPtr[r + 1], 1);
    }
    asc_syncthreads();

    if (threadIdx.x == 0) {
        for (int64_t i = 0; i < m; i++) {
            wsRowPtr[i + 1] += wsRowPtr[i];
        }
    }
    asc_syncthreads();

    for (int64_t p = static_cast<int64_t>(threadIdx.x);
         p < nnz;
         p += static_cast<int64_t>(blockDim.x)) {
        RowPtrT rawR = cooRowInd[p] - static_cast<RowPtrT>(idxBase);
        if (rawR < 0 || rawR >= static_cast<RowPtrT>(m)) {
            continue;
        }
        int32_t row = static_cast<int32_t>(rawR);
        int32_t pos = static_cast<int32_t>(wsRowPtr[row]) +
                      asc_atomic_add(&scratch[row], 1);
        // Write 0-based column index to workspace CSR.
        wsColInd[pos] = cooColInd[p] - static_cast<ColIndT>(idxBase);
        wsValues[pos] = cooValues[p];
        perm[pos] = static_cast<PermT>(p);
    }
    asc_syncthreads();
}

template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvBuildCsrFromSlicedEll(SPSV_SLICED_ELL_PARAMS(RowPtrT, ColIndT, PermT))
{
    int64_t nnzCounter = 0;
    for (int64_t i = 0; i < m; i++) {
        wsRowPtr[i] = static_cast<RowPtrT>(nnzCounter);
        for (int32_t s = 0; s < numSlices; s++) {
            int64_t sliceStart = static_cast<int64_t>(slicePtr[s]);
            for (int32_t k = 0; k < sliceWidth; k++) {
                int64_t p = sliceStart + i * sliceWidth + k;
                if (sellColInd[p] >= 0) {
                    wsColInd[nnzCounter] = sellColInd[p] - static_cast<ColIndT>(idxBase);
                    wsValues[nnzCounter] = sellValues[p];
                    perm[nnzCounter] = static_cast<PermT>(p);
                    nnzCounter++;
                }
            }
        }
    }
    wsRowPtr[m] = static_cast<RowPtrT>(nnzCounter);
}

// Serial CSR transpose (single-thread, thread 0 only).
// idxBase: 0 for 0-based source CSR, 1 for 1-based (Fortran). Subtract idxBase
// when reading source rowPtr and colInd to normalize to 0-based transposed CSR.
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvTransposeCsr(SPSV_TRANS_PARAMS(RowPtrT, ColIndT, PermT))
{
    for (int64_t i = 0; i <= m; i++) {
        transRowPtr[i] = 0;
    }
    for (int64_t i = 0; i < m; i++) {
        RowPtrT rowStart = rowPtr[i] - static_cast<RowPtrT>(idxBase);
        RowPtrT rowEnd = rowPtr[i + 1] - static_cast<RowPtrT>(idxBase);
        for (RowPtrT p = rowStart; p < rowEnd; p++) {
            int32_t col = static_cast<int32_t>(colInd[p]) - idxBase;
            // Bounds check: skip entries with column index out of range
            // [0, m) to prevent out-of-bounds write to transRowPtr
            // (which has m+1 entries).
            if (col < 0 || col >= static_cast<int32_t>(m)) {
                continue;
            }
            transRowPtr[col + 1]++;
        }
    }
    for (int64_t i = 0; i < m; i++) {
        transRowPtr[i + 1] += transRowPtr[i];
    }
    for (int64_t i = 0; i < m; i++) {
        scratch[i] = 0;
    }
    for (int64_t i = 0; i < m; i++) {
        RowPtrT rowStart = rowPtr[i] - static_cast<RowPtrT>(idxBase);
        RowPtrT rowEnd = rowPtr[i + 1] - static_cast<RowPtrT>(idxBase);
        for (RowPtrT p = rowStart; p < rowEnd; p++) {
            int32_t col = static_cast<int32_t>(colInd[p]) - idxBase;
            // Bounds check: skip entries with column index out of range
            // [0, m) to prevent out-of-bounds read from
            // transRowPtr/scratch arrays.
            if (col < 0 || col >= static_cast<int32_t>(m)) {
                continue;
            }
            RowPtrT pos = transRowPtr[col] + static_cast<RowPtrT>(scratch[col]);
            scratch[col]++;
            transColInd[pos] = static_cast<ColIndT>(i);
            transValues[pos] = values[p];
            transPerm[pos] = static_cast<PermT>(p);
        }
    }
}

// Helper macros for SpsvTransposeCsrParallel to factor out the common row-iteration
// and column-bounds-check structure shared between the histogram and scatter passes.
// idxBase is applied to rowPtr and colInd reads in SpsvTransposeCsrParallel below.
// RowPtrT is for iterating rowPtr offsets; ColIndT is for reading colInd values.
#define SPSV_TRANSPOSE_FOREACH_ROW(RowPtrT, idxBase)                           \
    for (int64_t i = static_cast<int64_t>(threadIdx.x);                        \
         i < m;                                                                \
         i += static_cast<int64_t>(blockDim.x))                                \
        for (RowPtrT p = rowPtr[i] - static_cast<RowPtrT>(idxBase);            \
             p < rowPtr[i + 1] - static_cast<RowPtrT>(idxBase); p++)

#define SPSV_TRANSPOSE_BOUNDSCHECK_COL(ColIndT, idxBase)                       \
    int32_t col = static_cast<int32_t>(colInd[p]) - idxBase;                   \
    if (col < 0 || col >= static_cast<int32_t>(m)) {                           \
        continue;                                                              \
    }

// Parallel CSR transpose. All threads in the block participate using atomic
// histogram and scatter.
// idxBase: 0 for 0-based source CSR, 1 for 1-based (Fortran).
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvTransposeCsrParallel(SPSV_TRANS_PARAMS(RowPtrT, ColIndT, PermT))
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i <= m;
         i += static_cast<int64_t>(blockDim.x)) {
        transRowPtr[i] = 0;
    }
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < m;
         i += static_cast<int64_t>(blockDim.x)) {
        scratch[i] = 0;
    }
    asc_syncthreads();

    SPSV_TRANSPOSE_FOREACH_ROW(RowPtrT, idxBase) {
        SPSV_TRANSPOSE_BOUNDSCHECK_COL(ColIndT, idxBase)
        asc_atomic_add(&transRowPtr[col + 1], 1);
    }
    asc_syncthreads();

    if (threadIdx.x == 0) {
        for (int64_t i = 0; i < m; i++) {
            transRowPtr[i + 1] += transRowPtr[i];
        }
    }
    asc_syncthreads();

    SPSV_TRANSPOSE_FOREACH_ROW(RowPtrT, idxBase) {
        SPSV_TRANSPOSE_BOUNDSCHECK_COL(ColIndT, idxBase)
        RowPtrT pos = transRowPtr[col] + static_cast<RowPtrT>(
                       asc_atomic_add(&scratch[col], 1));
        transColInd[pos] = static_cast<ColIndT>(i);
        transValues[pos] = values[p];
        transPerm[pos] = static_cast<PermT>(p);
    }
    asc_syncthreads();
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
__simt_callee__ inline int32_t SpsvComputeLevelForRow(
    __gm__ const RowPtrT *useRowPtr,
    __gm__ const ColIndT *useColInd,
    __gm__ const PermT *diagPtr,
    int64_t row, bool forward, int64_t m)
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
__simt_callee__ inline void SpsvComputeLevels(
    __gm__ const RowPtrT *useRowPtr,
    __gm__ const ColIndT *useColInd,
    __gm__ PermT *diagPtr,
    int64_t m, bool forward)
{
    for (int64_t i = 0; i < m; i++) {
        diagPtr[i] = 0;
    }
    if (forward) {
        for (int64_t i = 0; i < m; i++) {
            diagPtr[i] = SpsvComputeLevelForRow<RowPtrT, ColIndT, PermT>(
                useRowPtr, useColInd, diagPtr, i, true, m);
        }
    } else {
        for (int64_t i = m - 1; i >= 0; i--) {
            diagPtr[i] = SpsvComputeLevelForRow<RowPtrT, ColIndT, PermT>(
                useRowPtr, useColInd, diagPtr, i, false, m);
        }
    }
}

template <typename PermT>
__simt_callee__ inline int32_t SpsvBuildLevelHistogram(
    __gm__ PermT *diagPtr, __gm__ int32_t *levelPtr, int64_t m)
{
    int32_t numLevels = 0;
    for (int64_t i = 0; i < m; i++) {
        if (static_cast<int32_t>(diagPtr[i]) > numLevels) { numLevels = static_cast<int32_t>(diagPtr[i]); }
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
__simt_callee__ inline void SpsvResolveEffectivePtrs(
    __gm__ const RowPtrT *rowPtr,
    __gm__ const ColIndT *colInd,
    __gm__ uint8_t *workspace,
    int32_t fillMode, int32_t opA,
    int64_t csrRowPtrOffset, int64_t csrColIndOffset,
    int64_t transRowPtrOffset, int64_t transColIndOffset,
    int32_t format, int32_t idxBase,
    __gm__ const RowPtrT **effRowPtr,
    __gm__ const ColIndT **effColInd,
    bool *effForward)
{
    *effRowPtr = rowPtr;
    *effColInd = colInd;
    int32_t effFillMode = fillMode;
    int32_t effOpA = opA;

    // Use workspace CSR if format requires conversion (COO/SELL) OR
    // if input is 1-based CSR (analysis phase built 0-based copy there).
    if (format == 2 || format == 3 || idxBase == 1) {
        *effRowPtr = (__gm__ const RowPtrT *)(workspace + csrRowPtrOffset);
        *effColInd = (__gm__ const ColIndT *)(workspace + csrColIndOffset);
    }
    if (effOpA != 0 && transRowPtrOffset >= 0) {
        *effRowPtr = (__gm__ const RowPtrT *)(workspace + transRowPtrOffset);
        *effColInd = (__gm__ const ColIndT *)(workspace + transColIndOffset);
        effFillMode = (effFillMode == 0) ? 1 : 0;
        effOpA = 0;
    }
    *effForward = false;
    if (effFillMode == 0 && effOpA == 0) { *effForward = true; }
    if (effFillMode == 1 && effOpA != 0) { *effForward = true; }
}

template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvComputeDiagAndValidCount(
    __gm__ const RowPtrT *effRowPtr,
    __gm__ const ColIndT *effColInd,
    __gm__ PermT *diagPtr,
    __gm__ int32_t *validCount,
    int64_t row, bool effForward)
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

// ---------------------------------------------------------------------------
// Parallel variants for the multi-core serial phase.
// All threads in the single block participate (blockDim.x == nthreads).
// asc_syncthreads() provides block-level synchronization between phases.
// These functions are used ONLY in SpsvAnalysisSerialPhase; the single-core
// path (SpsvAnalysisSimtCompute) continues to use the sequential helpers
// above via SpsvAnalysisCommon.
// ---------------------------------------------------------------------------

// Parallel SLICED_ELL -> CSR conversion using all threads in the block.
// Two-pass approach (no atomics needed on row-level output buffers):
// Pass 1: Count valid entries per row (parallel over rows, no shared counter).
// Pass 2: Exclusive prefix-sum wsRowPtr to compute per-row start offsets
//         (thread 0 only, O(m)).
// Pass 3: Scatter (parallel over rows; each thread writes its row's entries
//         at wsRowPtr[i] + k without atomics since rows are disjoint).
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvBuildCsrFromSlicedEllParallel(SPSV_SLICED_ELL_PARAMS(RowPtrT, ColIndT, PermT))
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < m;
         i += static_cast<int64_t>(blockDim.x)) {
        int64_t count = 0;
        for (int32_t s = 0; s < numSlices; s++) {
            int64_t sliceStart = static_cast<int64_t>(slicePtr[s]);
            for (int32_t k = 0; k < sliceWidth; k++) {
                int64_t p = sliceStart + i * sliceWidth + k;
                if (sellColInd[p] >= 0) {
                    count++;
                }
            }
        }
        wsRowPtr[i] = static_cast<RowPtrT>(count);
    }
    asc_syncthreads();

    if (threadIdx.x == 0) {
        int64_t running = 0;
        for (int64_t i = 0; i < m; i++) {
            int64_t cnt = static_cast<int64_t>(wsRowPtr[i]);
            wsRowPtr[i] = static_cast<RowPtrT>(running);
            running += cnt;
        }
        wsRowPtr[m] = static_cast<RowPtrT>(running);
    }
    asc_syncthreads();

    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < m;
         i += static_cast<int64_t>(blockDim.x)) {
        int64_t pos = static_cast<int64_t>(wsRowPtr[i]);
        for (int32_t s = 0; s < numSlices; s++) {
            int64_t sliceStart = static_cast<int64_t>(slicePtr[s]);
            for (int32_t k = 0; k < sliceWidth; k++) {
                int64_t p = sliceStart + i * sliceWidth + k;
                if (sellColInd[p] >= 0) {
                    wsColInd[pos] = sellColInd[p] - static_cast<ColIndT>(idxBase);
                    wsValues[pos] = sellValues[p];
                    perm[pos] = static_cast<PermT>(p);
                    pos++;
                }
            }
        }
    }
    asc_syncthreads();
}

// ConvertFormatToCsr: dispatch format conversion to serial helpers (COO/SELL/1-based CSR).
// Requires local variables: wsRowPtr, wsColInd, wsValues, workspace, rowPtr, colInd, values,
// permOffset, diagPtrOffset, m, nnz, numSlices, sliceWidth, format, permType, idxBase.
// Outputs: updates srcRowPtr, srcColInd, srcValues, srcIdxBase.
#define SPSV_CONVERT_FORMAT_TO_CSR_SERIAL(RowPtrT, ColIndT) \
    do { \
        if (format == 2) { \
            if (permType == 0) { \
                __gm__ int32_t *permWs = (__gm__ int32_t *)(workspace + permOffset); \
                __gm__ int32_t *dgPtr = (__gm__ int32_t *)(workspace + diagPtrOffset); \
                SpsvBuildCsrFromCoo<RowPtrT, ColIndT, int32_t>(rowPtr, colInd, values, \
                    wsRowPtr, wsColInd, wsValues, permWs, dgPtr, m, nnz, idxBase); \
            } else { \
                __gm__ int64_t *permWs = (__gm__ int64_t *)(workspace + permOffset); \
                __gm__ int64_t *dgPtr = (__gm__ int64_t *)(workspace + diagPtrOffset); \
                SpsvBuildCsrFromCoo<RowPtrT, ColIndT, int64_t>(rowPtr, colInd, values, \
                    wsRowPtr, wsColInd, wsValues, permWs, dgPtr, m, nnz, idxBase); \
            } \
            srcRowPtr = wsRowPtr; \
            srcColInd = wsColInd; \
            srcValues = wsValues; \
            srcIdxBase = 0; \
        } else if (format == 3) { \
            if (permType == 0) { \
                __gm__ int32_t *permWs = (__gm__ int32_t *)(workspace + permOffset); \
                SpsvBuildCsrFromSlicedEll<RowPtrT, ColIndT, int32_t>(rowPtr, colInd, values, \
                    wsRowPtr, wsColInd, wsValues, permWs, m, numSlices, sliceWidth, idxBase); \
            } else { \
                __gm__ int64_t *permWs = (__gm__ int64_t *)(workspace + permOffset); \
                SpsvBuildCsrFromSlicedEll<RowPtrT, ColIndT, int64_t>(rowPtr, colInd, values, \
                    wsRowPtr, wsColInd, wsValues, permWs, m, numSlices, sliceWidth, idxBase); \
            } \
            srcRowPtr = wsRowPtr; \
            srcColInd = wsColInd; \
            srcValues = wsValues; \
            srcIdxBase = 0; \
        } else if (format == 0 && idxBase == 1) { \
            RowPtrT rowBase = static_cast<RowPtrT>(idxBase); \
            ColIndT colBase = static_cast<ColIndT>(idxBase); \
            for (int64_t i = 0; i <= m; i++) { \
                wsRowPtr[i] = rowPtr[i] - rowBase; \
            } \
            for (int64_t i = 0; i < nnz; i++) { \
                wsColInd[i] = colInd[i] - colBase; \
                wsValues[i] = values[i]; \
            } \
            if (permType == 0) { \
                __gm__ int32_t *permWs = (__gm__ int32_t *)(workspace + permOffset); \
                for (int64_t i = 0; i < nnz; i++) { \
                    permWs[i] = static_cast<int32_t>(i); \
                } \
            } else { \
                __gm__ int64_t *permWs = (__gm__ int64_t *)(workspace + permOffset); \
                for (int64_t i = 0; i < nnz; i++) { \
                    permWs[i] = static_cast<int64_t>(i); \
                } \
            } \
            srcRowPtr = wsRowPtr; \
            srcColInd = wsColInd; \
            srcValues = wsValues; \
            srcIdxBase = 0; \
        } \
    } while (0)

// ConvertFormatToCsrParallel: dispatch format conversion to parallel helpers with int64 serial fallback.
// Same local variable requirements as SPSV_CONVERT_FORMAT_TO_CSR_SERIAL.
#define SPSV_CONVERT_FORMAT_TO_CSR_PARALLEL(RowPtrT, ColIndT) \
    do { \
        if (format == 2) { \
            if (permType == 0) { \
                __gm__ int32_t *permWs = (__gm__ int32_t *)(workspace + permOffset); \
                __gm__ int32_t *dgPtr = (__gm__ int32_t *)(workspace + diagPtrOffset); \
                SpsvBuildCsrFromCooParallel<RowPtrT, ColIndT, int32_t>(rowPtr, colInd, values, \
                    wsRowPtr, wsColInd, wsValues, permWs, dgPtr, m, nnz, idxBase); \
            } else { \
                if (threadIdx.x == 0) { \
                    __gm__ int64_t *permWs = (__gm__ int64_t *)(workspace + permOffset); \
                    __gm__ int64_t *dgPtr = (__gm__ int64_t *)(workspace + diagPtrOffset); \
                    SpsvBuildCsrFromCoo<RowPtrT, ColIndT, int64_t>(rowPtr, colInd, values, \
                        wsRowPtr, wsColInd, wsValues, permWs, dgPtr, m, nnz, idxBase); \
                } \
                asc_syncthreads(); \
            } \
            srcRowPtr = wsRowPtr; \
            srcColInd = wsColInd; \
            srcValues = wsValues; \
            srcIdxBase = 0; \
        } else if (format == 3) { \
            if (permType == 0) { \
                __gm__ int32_t *permWs = (__gm__ int32_t *)(workspace + permOffset); \
                SpsvBuildCsrFromSlicedEllParallel<RowPtrT, ColIndT, int32_t>(rowPtr, colInd, values, \
                    wsRowPtr, wsColInd, wsValues, permWs, m, numSlices, sliceWidth, idxBase); \
            } else { \
                if (threadIdx.x == 0) { \
                    __gm__ int64_t *permWs = (__gm__ int64_t *)(workspace + permOffset); \
                    SpsvBuildCsrFromSlicedEll<RowPtrT, ColIndT, int64_t>(rowPtr, colInd, values, \
                        wsRowPtr, wsColInd, wsValues, permWs, m, numSlices, sliceWidth, idxBase); \
                } \
                asc_syncthreads(); \
            } \
            srcRowPtr = wsRowPtr; \
            srcColInd = wsColInd; \
            srcValues = wsValues; \
            srcIdxBase = 0; \
        } else if (format == 0 && idxBase == 1) { \
            RowPtrT rowBase = static_cast<RowPtrT>(idxBase); \
            ColIndT colBase = static_cast<ColIndT>(idxBase); \
            for (int64_t i = static_cast<int64_t>(threadIdx.x); i <= m; \
                 i += static_cast<int64_t>(blockDim.x)) { \
                wsRowPtr[i] = rowPtr[i] - rowBase; \
            } \
            if (permType == 0) { \
                __gm__ int32_t *permWs = (__gm__ int32_t *)(workspace + permOffset); \
                for (int64_t i = static_cast<int64_t>(threadIdx.x); i < nnz; \
                     i += static_cast<int64_t>(blockDim.x)) { \
                    wsColInd[i] = colInd[i] - colBase; \
                    wsValues[i] = values[i]; \
                    permWs[i] = static_cast<int32_t>(i); \
                } \
            } else { \
                __gm__ int64_t *permWs = (__gm__ int64_t *)(workspace + permOffset); \
                for (int64_t i = static_cast<int64_t>(threadIdx.x); i < nnz; \
                     i += static_cast<int64_t>(blockDim.x)) { \
                    wsColInd[i] = colInd[i] - colBase; \
                    wsValues[i] = values[i]; \
                    permWs[i] = static_cast<int64_t>(i); \
                } \
            } \
            asc_syncthreads(); \
            srcRowPtr = wsRowPtr; \
            srcColInd = wsColInd; \
            srcValues = wsValues; \
            srcIdxBase = 0; \
        } \
    } while (0)

// TransposeCsrIfNeeded: transpose src CSR into workspace if opA != 0 (serial version).
// Requires local variables: srcRowPtr, srcColInd, srcValues, srcIdxBase,
// tRowPtr, tColInd, tValues, workspace, transPermOffset, diagPtrOffset,
// transRowPtrOffset, opA, permType, m, nnz.
#define SPSV_TRANSPOSE_CSR_IF_NEEDED_SERIAL(RowPtrT, ColIndT) \
    do { \
        if (opA != 0 && transRowPtrOffset >= 0) { \
            if (permType == 0) { \
                __gm__ int32_t *tPermWs = (__gm__ int32_t *)(workspace + transPermOffset); \
                __gm__ int32_t *dgPtr = (__gm__ int32_t *)(workspace + diagPtrOffset); \
                SpsvTransposeCsr<RowPtrT, ColIndT, int32_t>(srcRowPtr, srcColInd, srcValues, \
                    tRowPtr, tColInd, tValues, tPermWs, dgPtr, m, nnz, srcIdxBase); \
            } else { \
                __gm__ int64_t *tPermWs = (__gm__ int64_t *)(workspace + transPermOffset); \
                __gm__ int64_t *dgPtr = (__gm__ int64_t *)(workspace + diagPtrOffset); \
                SpsvTransposeCsr<RowPtrT, ColIndT, int64_t>(srcRowPtr, srcColInd, srcValues, \
                    tRowPtr, tColInd, tValues, tPermWs, dgPtr, m, nnz, srcIdxBase); \
            } \
        } \
    } while (0)

// TransposeCsrIfNeededParallel: transpose src CSR into workspace if opA != 0 (parallel, int64 serial fallback).
#define SPSV_TRANSPOSE_CSR_IF_NEEDED_PARALLEL(RowPtrT, ColIndT) \
    do { \
        if (opA != 0 && transRowPtrOffset >= 0) { \
            if (permType == 0) { \
                __gm__ int32_t *tPermWs = (__gm__ int32_t *)(workspace + transPermOffset); \
                __gm__ int32_t *dgPtr = (__gm__ int32_t *)(workspace + diagPtrOffset); \
                SpsvTransposeCsrParallel<RowPtrT, ColIndT, int32_t>(srcRowPtr, srcColInd, srcValues, \
                    tRowPtr, tColInd, tValues, tPermWs, dgPtr, m, nnz, srcIdxBase); \
            } else { \
                if (threadIdx.x == 0) { \
                    __gm__ int64_t *tPermWs = (__gm__ int64_t *)(workspace + transPermOffset); \
                    __gm__ int64_t *dgPtr = (__gm__ int64_t *)(workspace + diagPtrOffset); \
                    SpsvTransposeCsr<RowPtrT, ColIndT, int64_t>(srcRowPtr, srcColInd, srcValues, \
                        tRowPtr, tColInd, tValues, tPermWs, dgPtr, m, nnz, srcIdxBase); \
                } \
                asc_syncthreads(); \
            } \
        } \
    } while (0)

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
__simt_callee__ inline void SpsvBuildLevelHistogramParallel(
    __gm__ PermT *diagPtr,
    __gm__ int32_t *levelPtr,
    __gm__ SpsvTilingData *tilingOut,
    int64_t m)
{
    if (threadIdx.x == 0) {
        int32_t numLevels = 0;
        for (int64_t i = 0; i < m; i++) {
            int32_t lvl = static_cast<int32_t>(diagPtr[i]);
            if (lvl > numLevels) { numLevels = lvl; }
        }
        numLevels += 1;
        tilingOut->numLevels = numLevels;
        for (int32_t k = 0; k <= numLevels; k++) {
            levelPtr[k] = 0;
        }
    }
    asc_syncthreads();

    int32_t numLevels = tilingOut->numLevels;

    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < m;
         i += static_cast<int64_t>(blockDim.x)) {
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
__simt_callee__ inline void SpsvFinalizeLevelPtr(
    __gm__ int32_t *levelPtr,
    __gm__ SpsvTilingData *tilingOut,
    int32_t numLevels)
{
    for (int32_t k = numLevels; k >= 1; k--) {
        levelPtr[k] = levelPtr[k - 1];
    }
    levelPtr[0] = 0;
    tilingOut->numLevels = numLevels;
}

// Resolves effective row/col pointers from workspace offsets and computes the
// level structure on the resolved matrix. Encapsulates the declare-pointer,
// call SpsvResolveEffectivePtrs, call SpsvComputeLevels pattern that
// appeared identically in SpsvAnalysisCommon and SpsvAnalysisSerialPhase.
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvResolveAndComputeLevels(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ uint8_t *workspace,
    __gm__ PermT *diagPtr,
    int64_t m,
    int32_t fillMode, int32_t opA,
    int64_t csrRowPtrOffset, int64_t csrColIndOffset,
    int64_t transRowPtrOffset, int64_t transColIndOffset,
    int32_t format, int32_t idxBase)
{
    __gm__ const RowPtrT *useRowPtr = nullptr;
    __gm__ const ColIndT *useColInd = nullptr;
    bool forward = false;
    SpsvResolveEffectivePtrs<RowPtrT, ColIndT>(rowPtr, colInd, workspace,
        fillMode, opA, csrRowPtrOffset, csrColIndOffset,
        transRowPtrOffset, transColIndOffset, format, idxBase,
        &useRowPtr, &useColInd, &forward);
    SpsvComputeLevels<RowPtrT, ColIndT, PermT>(useRowPtr, useColInd, diagPtr, m, forward);
}

// SPSV_ANALYSIS_PARAMS: shared parameter list for analysis VF functions.
// Kept as a macro because SIMT __simt_vf__ function signatures must be
// explicitly spelled out for the VF compiler; a struct-based approach is
// not supported by the asc_vf_call template instantiation mechanism.
#define SPSV_ANALYSIS_PARAMS \
    __gm__ const RowPtrT *rowPtr, \
    __gm__ const ColIndT *colInd, \
    __gm__ const float *values, \
    __gm__ uint8_t *workspace, \
    int64_t m, int64_t nnz, \
    int32_t fillMode, int32_t opA, \
    int64_t levelPtrOffset, int64_t levelRowOffset, int64_t diagPtrOffset, \
    int64_t validCountOffset, \
    int64_t csrRowPtrOffset, int64_t csrColIndOffset, int64_t csrValuesOffset, int64_t permOffset, \
    int64_t transRowPtrOffset, int64_t transColIndOffset, int64_t transValuesOffset, \
    int64_t transPermOffset, \
    int32_t format, int32_t numSlices, int32_t sliceWidth, int32_t permType, int32_t idxBase

template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline int32_t SpsvAnalysisCommon(
    __gm__ const RowPtrT *rowPtr,
    __gm__ const ColIndT *colInd,
    __gm__ const float *values,
    __gm__ uint8_t *workspace,
    __gm__ PermT *diagPtr,
    __gm__ int32_t *levelPtr,
    int64_t m, int64_t nnz,
    int32_t fillMode, int32_t opA,
    int64_t csrRowPtrOffset, int64_t csrColIndOffset, int64_t csrValuesOffset, int64_t permOffset,
    int64_t transRowPtrOffset, int64_t transColIndOffset, int64_t transValuesOffset,
    int64_t transPermOffset,
    int64_t diagPtrOffset,
    int32_t format, int32_t numSlices, int32_t sliceWidth,
    int32_t permType, int32_t idxBase)
{
    SpsvBuildWorkspaceCsr<RowPtrT, ColIndT>(rowPtr, colInd, values, workspace,
        m, nnz, csrRowPtrOffset, csrColIndOffset, csrValuesOffset, permOffset,
        transRowPtrOffset, transColIndOffset, transValuesOffset, transPermOffset,
        diagPtrOffset, format, opA, numSlices, sliceWidth, permType, idxBase);

    // Resolve effective pointers and compute level structure in a single call.
    // Pass idxBase so it can redirect to workspace CSR for 1-based input.
    SpsvResolveAndComputeLevels<RowPtrT, ColIndT, PermT>(rowPtr, colInd, workspace, diagPtr, m,
        fillMode, opA,
        csrRowPtrOffset, csrColIndOffset,
        transRowPtrOffset, transColIndOffset, format, idxBase);

    return SpsvBuildLevelHistogram<PermT>(diagPtr, levelPtr, m);
}

template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvAnalysisSimtCompute(
    SPSV_ANALYSIS_PARAMS)
{
    __gm__ int32_t *levelPtr = (__gm__ int32_t *)(workspace + levelPtrOffset);
    __gm__ int32_t *levelRow = (__gm__ int32_t *)(workspace + levelRowOffset);
    __gm__ PermT *diagPtr = (__gm__ PermT *)(workspace + diagPtrOffset);
    __gm__ int32_t *validCount = (__gm__ int32_t *)(workspace + validCountOffset);
    __gm__ SpsvTilingData *tilingOut = (__gm__ SpsvTilingData *)workspace;

    int32_t numLevels = 0;

    if (threadIdx.x == 0) {
        numLevels = SpsvAnalysisCommon<RowPtrT, ColIndT, PermT>(rowPtr, colInd, values, workspace,
            diagPtr, levelPtr, m, nnz, fillMode, opA,
            csrRowPtrOffset, csrColIndOffset, csrValuesOffset, permOffset,
            transRowPtrOffset, transColIndOffset, transValuesOffset, transPermOffset,
            diagPtrOffset, format, numSlices, sliceWidth, permType, idxBase);
    }

    // thread 0 wrote diagPtr/levelPtr to GM above. asc_threadfence_block
    // ensures those GM writes are visible to all other threads in this block
    // before asc_syncthreads releases them to read diagPtr[i].
    asc_threadfence_block();
    asc_syncthreads();

    __gm__ const RowPtrT *effRowPtr = nullptr;
    __gm__ const ColIndT *effColInd = nullptr;
    bool effForward = false;
    SpsvResolveEffectivePtrs<RowPtrT, ColIndT>(rowPtr, colInd, workspace,
        fillMode, opA, csrRowPtrOffset, csrColIndOffset,
        transRowPtrOffset, transColIndOffset, format, idxBase,
        &effRowPtr, &effColInd, &effForward);

    // DEFENSIVE: host ValidateSpSVCommonParams guarantees m <= INT32_MAX.
    // All levelRow/validCount arrays are int32_t; diagPtr values are level
    // numbers ≤ numLevels ≤ m, cast to int32_t is safe.
    // The same guarantee covers SpsvAnalysisParallelPhase.
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < m;
         i += static_cast<int64_t>(blockDim.x)) {
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

    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < m;
         i += static_cast<int64_t>(blockDim.x)) {
        SpsvComputeDiagAndValidCount<RowPtrT, ColIndT, PermT>(effRowPtr, effColInd,
            diagPtr, validCount, i, effForward);
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
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvAnalysisSerialPhase(
    SPSV_ANALYSIS_PARAMS)
{
    __gm__ int32_t *levelPtr = (__gm__ int32_t *)(workspace + levelPtrOffset);
    __gm__ PermT *diagPtr = (__gm__ PermT *)(workspace + diagPtrOffset);
    __gm__ SpsvTilingData *tilingOut = (__gm__ SpsvTilingData *)workspace;

    // Phase 1: Format conversion (all threads in block).
    // SpsvBuildWorkspaceCsrParallel ends with asc_syncthreads() internally
    // when it does any work; for CSR (format == 0) without transpose it is
    // a no-op, which is fine because no shared state needs synchronizing.
    // When permType==1 (nnz > INT32_MAX), the parallel function internally
    // falls back to serial conversion (thread 0 only).
    SpsvBuildWorkspaceCsrParallel<RowPtrT, ColIndT>(rowPtr, colInd, values, workspace,
        m, nnz, csrRowPtrOffset, csrColIndOffset, csrValuesOffset, permOffset,
        transRowPtrOffset, transColIndOffset, transValuesOffset, transPermOffset,
        diagPtrOffset, format, opA, numSlices, sliceWidth, permType, idxBase);

    // Phase 2: Resolve effective pointers and compute level structure
    // (thread 0 only — true sequential dependency: level[i] depends on
    // diagPtr[col] for col in row i's dependencies).
    if (threadIdx.x == 0) {
        SpsvResolveAndComputeLevels<RowPtrT, ColIndT, PermT>(rowPtr, colInd, workspace, diagPtr, m,
            fillMode, opA,
            csrRowPtrOffset, csrColIndOffset,
            transRowPtrOffset, transColIndOffset, format, idxBase);
    }
    asc_syncthreads();

    // Phase 3: Parallel histogram build (all threads in block).
    // Writes tilingOut->numLevels and the prefix-summed levelPtr array.
    SpsvBuildLevelHistogramParallel<PermT>(diagPtr, levelPtr, tilingOut, m);
}

template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvAnalysisParallelPhase(
    __gm__ const RowPtrT *rowPtr,
    __gm__ const ColIndT *colInd,
    __gm__ uint8_t *workspace,
    int64_t m,
    int32_t fillMode, int32_t opA,
    int64_t levelPtrOffset, int64_t levelRowOffset, int64_t diagPtrOffset,
    int64_t validCountOffset,
    int64_t csrRowPtrOffset, int64_t csrColIndOffset,
    int64_t transRowPtrOffset, int64_t transColIndOffset,
    int32_t format, int32_t numBlocks, int32_t idxBase)
{
    __gm__ int32_t *levelPtr = (__gm__ int32_t *)(workspace + levelPtrOffset);
    __gm__ int32_t *levelRow = (__gm__ int32_t *)(workspace + levelRowOffset);
    __gm__ PermT *diagPtr = (__gm__ PermT *)(workspace + diagPtrOffset);
    __gm__ int32_t *validCount = (__gm__ int32_t *)(workspace + validCountOffset);

    __gm__ const RowPtrT *parRowPtr = nullptr;
    __gm__ const ColIndT *parColInd = nullptr;
    bool parForward = false;
    SpsvResolveEffectivePtrs<RowPtrT, ColIndT>(rowPtr, colInd, workspace,
        fillMode, opA, csrRowPtrOffset, csrColIndOffset,
        transRowPtrOffset, transColIndOffset, format, idxBase,
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
        SpsvComputeDiagAndValidCount<RowPtrT, ColIndT, PermT>(parRowPtr, parColInd,
            diagPtr, validCount, i, parForward);
    }
}

// Final phase: launched as a standalone single-block kernel. Only thread 0
// finalizes the levelPtr prefix-sum. Same rationale as SpsvAnalysisSerialPhase.
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvAnalysisFinalPhase(
    __gm__ uint8_t *workspace,
    int64_t levelPtrOffset)
{
    __gm__ int32_t *levelPtr = (__gm__ int32_t *)(workspace + levelPtrOffset);
    __gm__ SpsvTilingData *tilingOut = (__gm__ SpsvTilingData *)workspace;

    if (threadIdx.x == 0) {
        int32_t numLevels = tilingOut->numLevels;
        SpsvFinalizeLevelPtr(levelPtr, tilingOut, numLevels);
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_callee__ inline void SpsvSolveRow(
    __gm__ const RowPtrT *rowPtr,
    __gm__ const ColIndT *colInd,
    __gm__ const float *values,
    __gm__ const float *vecX,
    __gm__ float *vecY,
    __gm__ const PermT *diagPtr,
    __gm__ const int32_t *validCount,
    float alpha, int32_t row)
{
    float sum = alpha * vecX[row];
    RowPtrT rs = rowPtr[row];
    RowPtrT re = rowPtr[row + 1];
    int32_t vc = validCount[row];

    if constexpr (FORWARD) {
        RowPtrT validEnd = rs + static_cast<RowPtrT>(vc);
        // #pragma unroll: the VF compiler may ignore unrecognised pragmas;
        // when recognised it hints loop unrolling for the hot inner loop.
        #pragma unroll 4
        for (RowPtrT p = rs; p < validEnd; p++) {
            sum -= values[p] * vecY[static_cast<int32_t>(colInd[p])];
        }
        // Cleanup loop for unsorted column indices beyond validCount.
        // validCount already covers the sorted-index case; this branch is
        // a safety net for unsorted CSR where col < row entries may appear
        // after the sorted prefix. The if-branch is acceptable here because
        // this path is rarely taken for well-formed sorted CSR input.
        for (RowPtrT p = validEnd; p < re; p++) {
            int32_t col = static_cast<int32_t>(colInd[p]);
            if (col < row) {
                sum -= values[p] * vecY[col];
            }
        }
    } else {
        RowPtrT validStart = re - static_cast<RowPtrT>(vc);
        #pragma unroll 4
        for (RowPtrT p = validStart; p < re; p++) {
            sum -= values[p] * vecY[static_cast<int32_t>(colInd[p])];
        }
        // Cleanup loop for unsorted column indices (backward variant).
        for (RowPtrT p = rs; p < validStart; p++) {
            int32_t col = static_cast<int32_t>(colInd[p]);
            if (col > row) {
                sum -= values[p] * vecY[col];
            }
        }
    }

    if constexpr (NON_UNIT) {
        // diagPtr[row] == -1 means no diagonal element exists in this row
        // (singular matrix). Divide by 0.0f so that IEEE-754 semantics
        // produce Inf/NaN, matching cuSPARSE behavior.
        sum /= (diagPtr[row] >= 0) ? values[diagPtr[row]] : 0.0f;
    }
    vecY[row] = sum;
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvSolveSimtCompute(
    __gm__ const RowPtrT *rowPtr,
    __gm__ const ColIndT *colInd,
    __gm__ const float *values,
    __gm__ const float *vecX,
    __gm__ float *vecY,
    __gm__ const int32_t *levelPtr,
    __gm__ const int32_t *levelRow,
    __gm__ const PermT *diagPtr,
    __gm__ const int32_t *validCount,
    float alpha, int64_t m, int32_t numLevels)
{
    for (int32_t level = 0; level < numLevels; level++) {
        int32_t levelStart = levelPtr[level];
        int32_t levelEnd = levelPtr[level + 1];
        int32_t levelWidth = levelEnd - levelStart;

        for (int32_t idx = static_cast<int32_t>(threadIdx.x);
             idx < levelWidth;
             idx += static_cast<int32_t>(blockDim.x)) {
            int32_t row = levelRow[levelStart + idx];
            SpsvSolveRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT>(rowPtr, colInd, values,
                vecX, vecY, diagPtr, validCount, alpha, row);
        }

        asc_syncthreads();
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvSolveLevelSimtCompute(
    __gm__ const RowPtrT *rowPtr,
    __gm__ const ColIndT *colInd,
    __gm__ const float *values,
    __gm__ const float *vecX,
    __gm__ float *vecY,
    __gm__ const int32_t *levelPtr,
    __gm__ const int32_t *levelRow,
    __gm__ const PermT *diagPtr,
    __gm__ const int32_t *validCount,
    float alpha, int32_t level, int32_t numBlocks)
{
    int32_t globalTid = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) +
                        static_cast<int32_t>(threadIdx.x);
    int32_t globalStride = numBlocks * static_cast<int32_t>(blockDim.x);

    int32_t levelStart = levelPtr[level];
    int32_t levelEnd = levelPtr[level + 1];
    int32_t levelWidth = levelEnd - levelStart;

    for (int32_t idx = globalTid; idx < levelWidth; idx += globalStride) {
        int32_t row = levelRow[levelStart + idx];
        SpsvSolveRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT>(rowPtr, colInd, values,
            vecX, vecY, diagPtr, validCount, alpha, row);
    }
}

template <typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvUpdateValuesSimtCompute(
    __gm__ const float *newValues,
    __gm__ float *csrValues,
    __gm__ const PermT *perm,
    int64_t nnz)
{
    for (int64_t p = static_cast<int64_t>(threadIdx.x);
         p < nnz;
         p += static_cast<int64_t>(blockDim.x)) {
        csrValues[p] = newValues[perm[p]];
    }
}

template <typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvUpdateDiagSimtCompute(
    __gm__ const float *newDiagValues,
    __gm__ float *values,
    __gm__ const PermT *diagPtr,
    int64_t m)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < m;
         i += static_cast<int64_t>(blockDim.x)) {
        PermT dp = diagPtr[i];
        if (dp >= 0) {
            values[dp] = newDiagValues[i];
        }
    }
}

// SpsvFillZero / SpsvScaleCopy / SpsvCopyValues: small-vector helper kernels.
// These operate on vectors of length m (matrix dimension), which is typically
// small for triangular solve workloads. SIMT per-element writes are adequate
// here; a DataCopy/Duplicate path would add TPipe/TQue setup overhead that
// outweighs the benefit for these small transfers.
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvFillZeroSimtCompute(
    __gm__ float *vecY,
    int64_t m)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < m;
         i += static_cast<int64_t>(blockDim.x)) {
        vecY[i] = 0.0f;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvScaleCopySimtCompute(
    __gm__ const float *vecX,
    __gm__ float *vecY,
    float alpha, int64_t m)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < m;
         i += static_cast<int64_t>(blockDim.x)) {
        vecY[i] = alpha * vecX[i];
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvScaleInfSimtCompute(
    __gm__ const float *vecX,
    __gm__ float *vecY,
    float alpha, int64_t m)
{
    const float inf = 1.0f / 0.0f;
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < m;
         i += static_cast<int64_t>(blockDim.x)) {
        vecY[i] = (alpha * vecX[i]) * inf;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvCopyValuesSimtCompute(
    __gm__ const float *src,
    __gm__ float *dst,
    int64_t nnz)
{
    for (int64_t p = static_cast<int64_t>(threadIdx.x);
         p < nnz;
         p += static_cast<int64_t>(blockDim.x)) {
        dst[p] = src[p];
    }
}

// Dispatch macros for asc_vf_call. Kept as macros (not inline template
// functions) because asc_vf_call requires the VF function template to be
// spelled out as a non-type template argument, and the SIMT compiler does
// not support forwarding this through an intermediate template wrapper.
#define SPSV_CALL_ANALYSIS(RowPtrT, ColIndT, PermT, Func, rp, ci, val, ws, t) \
    asc_vf_call<Func<RowPtrT, ColIndT, PermT>>(dim3{t.nthreads, 1, 1}, \
        (__gm__ const RowPtrT *)rp, (__gm__ const ColIndT *)ci, \
        (__gm__ const float *)val, (__gm__ uint8_t *)ws, \
        t.m, t.nnz, t.fillMode, t.opA, \
        t.levelPtrOffset, t.levelRowOffset, t.diagPtrOffset, t.validCountOffset, \
        t.csrRowPtrOffset, t.csrColIndOffset, t.csrValuesOffset, t.permOffset, \
        t.transRowPtrOffset, t.transColIndOffset, t.transValuesOffset, t.transPermOffset, \
        t.format, t.numSlices, t.sliceWidth, t.permType, t.idxBase)

#define SPSV_CALL_ANALYSIS_PAR(RowPtrT, ColIndT, PermT, rp, ci, ws, t) \
    asc_vf_call<SpsvAnalysisParallelPhase<RowPtrT, ColIndT, PermT>>(dim3{t.nthreads, 1, 1}, \
        (__gm__ const RowPtrT *)rp, (__gm__ const ColIndT *)ci, \
        (__gm__ uint8_t *)ws, t.m, t.fillMode, t.opA, \
        t.levelPtrOffset, t.levelRowOffset, t.diagPtrOffset, t.validCountOffset, \
        t.csrRowPtrOffset, t.csrColIndOffset, \
        t.transRowPtrOffset, t.transColIndOffset, \
        t.format, static_cast<int32_t>(t.numBlocks), t.idxBase)

#define SPSV_CALL_SOLVE(RowPtrT, ColIndT, PermT, FWD, NU, Func, rp, ci, val, vx, vy, \
    lp, lr, dp, vc, a, e1, e2, nt) \
    asc_vf_call<Func<RowPtrT, ColIndT, FWD, NU, PermT>>(dim3{nt, 1, 1}, \
        (__gm__ const RowPtrT *)rp, (__gm__ const ColIndT *)ci, \
        (__gm__ const float *)val, (__gm__ const float *)vx, (__gm__ float *)vy, \
        lp, lr, dp, vc, a, e1, e2)

#define SPSV_DISPATCH_SOLVE(RowPtrT, ColIndT, PermT, Func, rp, ci, val, vx, vy, \
    lp, lr, dp, vc, a, e1, e2, nt, fwd, nonUnit) \
    do { \
        if (fwd && nonUnit) { \
            SPSV_CALL_SOLVE(RowPtrT, ColIndT, PermT, true, true, Func, rp, ci, val, vx, vy, \
                lp, lr, dp, vc, a, e1, e2, nt); \
        } else if (fwd) { \
            SPSV_CALL_SOLVE(RowPtrT, ColIndT, PermT, true, false, Func, rp, ci, val, vx, vy, \
                lp, lr, dp, vc, a, e1, e2, nt); \
        } else if (nonUnit) { \
            SPSV_CALL_SOLVE(RowPtrT, ColIndT, PermT, false, true, Func, rp, ci, val, vx, vy, \
                lp, lr, dp, vc, a, e1, e2, nt); \
        } else { \
            SPSV_CALL_SOLVE(RowPtrT, ColIndT, PermT, false, false, Func, rp, ci, val, vx, vy, \
                lp, lr, dp, vc, a, e1, e2, nt); \
        } \
    } while (0)

// Single-core analysis path: one block runs the whole analysis (format
// conversion, level computation, scatter, diag/validCount, finalize) using
// block-level asc_syncthreads. The multi-core path is split into three
// separate host-side kernel launches (serial / parallel / final) below.
extern "C" __global__ __aicore__ void spsv_analysis_kernel(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // Dispatch on (indexType, colIndType, permType):
    //   I32 ptr + I32 col + permType=0: PermT int32.
    //   I32 ptr + I32 col + permType=1: PermT int64 (SPSV_FORCE_PERMT_64=1).
    //   I64 ptr + I32 col + permType=0: PermT int32.
    //   I64 ptr + I32 col + permType=1: PermT int64 (nnz > INT32_MAX).
    //   I64 ptr + I64 col + permType=0: PermT int32.
    //   I64 ptr + I64 col + permType=1: PermT int64.
    if (tiling.indexType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int32_t, int32_t, int32_t, SpsvAnalysisSimtCompute,
                rowPtr, colInd, values, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS(int32_t, int32_t, int64_t, SpsvAnalysisSimtCompute,
                rowPtr, colInd, values, workspace, tiling);
        }
    } else if (tiling.colIndType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int64_t, int32_t, int32_t, SpsvAnalysisSimtCompute,
                rowPtr, colInd, values, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS(int64_t, int32_t, int64_t, SpsvAnalysisSimtCompute,
                rowPtr, colInd, values, workspace, tiling);
        }
    } else {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int64_t, int64_t, int32_t, SpsvAnalysisSimtCompute,
                rowPtr, colInd, values, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS(int64_t, int64_t, int64_t, SpsvAnalysisSimtCompute,
                rowPtr, colInd, values, workspace, tiling);
        }
    }
}

// Multi-core analysis, phase 1 (serial): single-block launch. Performs format
// conversion, sequential level computation, histogram and prefix-sum. Stream
// ordering relative to the next launch provides cross-kernel GM visibility,
// replacing the in-kernel SyncAll the old single-launch design relied on.
extern "C" __global__ __aicore__ void spsv_analysis_serial_kernel(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.indexType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int32_t, int32_t, int32_t, SpsvAnalysisSerialPhase,
                rowPtr, colInd, values, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS(int32_t, int32_t, int64_t, SpsvAnalysisSerialPhase,
                rowPtr, colInd, values, workspace, tiling);
        }
    } else if (tiling.colIndType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int64_t, int32_t, int32_t, SpsvAnalysisSerialPhase,
                rowPtr, colInd, values, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS(int64_t, int32_t, int64_t, SpsvAnalysisSerialPhase,
                rowPtr, colInd, values, workspace, tiling);
        }
    } else {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int64_t, int64_t, int32_t, SpsvAnalysisSerialPhase,
                rowPtr, colInd, values, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS(int64_t, int64_t, int64_t, SpsvAnalysisSerialPhase,
                rowPtr, colInd, values, workspace, tiling);
        }
    }
}

// Multi-core analysis, phase 2 (parallel): launched with numBlocks blocks.
// Scatter (levelRow fill), diagonal-index and validCount computation. Each
// block works on a disjoint row partition with no cross-block data dependency,
// so no barrier is needed within this kernel.
extern "C" __global__ __aicore__ void spsv_analysis_parallel_kernel(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.indexType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS_PAR(int32_t, int32_t, int32_t, rowPtr, colInd, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS_PAR(int32_t, int32_t, int64_t, rowPtr, colInd, workspace, tiling);
        }
    } else if (tiling.colIndType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS_PAR(int64_t, int32_t, int32_t, rowPtr, colInd, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS_PAR(int64_t, int32_t, int64_t, rowPtr, colInd, workspace, tiling);
        }
    } else {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS_PAR(int64_t, int64_t, int32_t, rowPtr, colInd, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS_PAR(int64_t, int64_t, int64_t, rowPtr, colInd, workspace, tiling);
        }
    }
}

// Multi-core analysis, phase 3 (final): single-block launch. Restores the
// levelPtr prefix-sum (right-shift) and writes numLevels.
extern "C" __global__ __aicore__ void spsv_analysis_final_kernel(
    GM_ADDR workspace, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    asc_vf_call<SpsvAnalysisFinalPhase>(dim3{tiling.nthreads, 1, 1},
        (__gm__ uint8_t *)workspace,
        tiling.levelPtrOffset);
}

template <typename RowPtrT, typename ColIndT, typename PermT>
__aicore__ inline void SpsvSolveSingleCore(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
    GM_ADDR vecX, GM_ADDR vecY,
    __gm__ const int32_t *levelPtr, __gm__ const int32_t *levelRow,
    __gm__ const PermT *diagPtr, __gm__ const int32_t *validCount,
    float alpha, int64_t m, int32_t numLevels, uint32_t nthreads,
    bool fwd, bool nonUnit)
{
    SPSV_DISPATCH_SOLVE(RowPtrT, ColIndT, PermT, SpsvSolveSimtCompute,
        rowPtr, colInd, values, vecX, vecY,
        levelPtr, levelRow, diagPtr, validCount,
        alpha, m, numLevels, nthreads, fwd, nonUnit);
}

template <typename RowPtrT, typename ColIndT, typename PermT>
__aicore__ inline void SpsvSolveMultiCore(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
    GM_ADDR vecX, GM_ADDR vecY,
    __gm__ const int32_t *levelPtr, __gm__ const int32_t *levelRow,
    __gm__ const PermT *diagPtr, __gm__ const int32_t *validCount,
    float alpha, int32_t numLevels, uint32_t nthreads,
    int32_t numBlocks, bool fwd, bool nonUnit)
{
    for (int32_t level = 0; level < numLevels; level++) {
        SPSV_DISPATCH_SOLVE(RowPtrT, ColIndT, PermT, SpsvSolveLevelSimtCompute,
            rowPtr, colInd, values, vecX, vecY,
            levelPtr, levelRow, diagPtr, validCount,
            alpha, level, numBlocks, nthreads, fwd, nonUnit);

        // SIMT VF writes go directly to GM. SyncAll on arch35 implements
        // cross-core synchronization via GM-based flag operations (SetFlag /
        // WaitFlag). The GM writes in the flag protocol provide the memory
        // ordering guarantee (cross-core GM visibility) in addition to the
        // execution synchronization barrier; a separate memory fence is not
        // required here.
        SyncAll();
    }
}

// Dispatch solve based on RowPtrT/ColIndT/PermT and numBlocks (single-core vs multi-core).
#define SPSV_DISPATCH_SOLVE_BLOCKS(RowPtrT, ColIndT, PermT, rowPtr, colInd, values, \
        vecX, vecY, levelPtr, levelRow, diagPtr, validCount, alpha, m, \
        numLevels, nthreads, numBlocks, fwd, nonUnit) \
    do { \
        if ((numBlocks) <= 1) { \
            SpsvSolveSingleCore<RowPtrT, ColIndT, PermT>( \
                (rowPtr), (colInd), (values), (vecX), (vecY), \
                (levelPtr), (levelRow), (diagPtr), (validCount), \
                (alpha), (m), (numLevels), (nthreads), (fwd), (nonUnit)); \
        } else { \
            SpsvSolveMultiCore<RowPtrT, ColIndT, PermT>( \
                (rowPtr), (colInd), (values), (vecX), (vecY), \
                (levelPtr), (levelRow), (diagPtr), (validCount), \
                (alpha), (numLevels), (nthreads), \
                static_cast<int32_t>(numBlocks), (fwd), (nonUnit)); \
        } \
    } while (0)

// Prepare solve kernel local variables from tiling data.
// Reads levelPtr/levelRow/validCount pointers, numLevels from workspace,
// computes fwd/nonUnit flags, and resolves alpha (host or device pointer).
__aicore__ inline void PrepareSolveKernelArgs(
    const SpsvTilingData &tiling,
    __gm__ uint8_t *workspace,
    __gm__ const int32_t *&levelPtr,
    __gm__ const int32_t *&levelRow,
    __gm__ const int32_t *&validCount,
    int32_t &numLevels,
    bool &fwd,
    bool &nonUnit,
    float &alpha)
{
    levelPtr = (__gm__ const int32_t *)(workspace + tiling.levelPtrOffset);
    levelRow = (__gm__ const int32_t *)(workspace + tiling.levelRowOffset);
    validCount = (__gm__ const int32_t *)(workspace + tiling.validCountOffset);

    // numLevels is read from workspace (written by the analysis kernel),
    // NOT from tiling.numLevels (which is 0 in the host-supplied params).
    __gm__ SpsvTilingData *tilingWs = (__gm__ SpsvTilingData *)workspace;
    numLevels = tilingWs->numLevels;

    int32_t fm = tiling.fillMode;
    if (tiling.opA != 0) {
        fm = (fm == 0) ? 1 : 0;
    }
    fwd = (fm == 0);
    // diagType == 0 corresponds to ACL_SPARSE_DIAG_TYPE_NON_UNIT (non-unit
    // diagonal), meaning the solve must divide by the diagonal element.
    nonUnit = (tiling.diagType == 0);

    alpha = tiling.alpha;
    if (tiling.alphaDevicePtr != 0) {
        // alphaDevicePtr is a device pointer to a float. The caller must
        // ensure at least 4-byte alignment (float requirement).
        alpha = *(__gm__ const float *)(tiling.alphaDevicePtr);
    }
}

extern "C" __global__ __aicore__ void spsv_solve_kernel(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
    GM_ADDR vecX, GM_ADDR vecY, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ const int32_t *levelPtr;
    __gm__ const int32_t *levelRow;
    __gm__ const int32_t *validCount;
    int32_t numLevels;
    bool fwd;
    bool nonUnit;
    float alpha;
    PrepareSolveKernelArgs(tiling, (__gm__ uint8_t *)workspace,
        levelPtr, levelRow, validCount, numLevels, fwd, nonUnit, alpha);

    // Dispatch based on (indexType, colIndType, permType)
    if (tiling.indexType == 0 && tiling.colIndType == 0) {
        // (I32, I32)
        if (tiling.permType == 0) {
            SPSV_DISPATCH_SOLVE_BLOCKS(int32_t, int32_t, int32_t,
                rowPtr, colInd, values, vecX, vecY, levelPtr, levelRow,
                (__gm__ const int32_t *)(workspace + tiling.diagPtrOffset), validCount,
                alpha, tiling.m, numLevels, tiling.nthreads, tiling.numBlocks, fwd, nonUnit);
        } else {
            SPSV_DISPATCH_SOLVE_BLOCKS(int32_t, int32_t, int64_t,
                rowPtr, colInd, values, vecX, vecY, levelPtr, levelRow,
                (__gm__ const int64_t *)(workspace + tiling.diagPtrOffset), validCount,
                alpha, tiling.m, numLevels, tiling.nthreads, tiling.numBlocks, fwd, nonUnit);
        }
    } else if (tiling.indexType == 1 && tiling.colIndType == 0) {
        // (I64, I32)
        if (tiling.permType == 0) {
            SPSV_DISPATCH_SOLVE_BLOCKS(int64_t, int32_t, int32_t,
                rowPtr, colInd, values, vecX, vecY, levelPtr, levelRow,
                (__gm__ const int32_t *)(workspace + tiling.diagPtrOffset), validCount,
                alpha, tiling.m, numLevels, tiling.nthreads, tiling.numBlocks, fwd, nonUnit);
        } else {
            SPSV_DISPATCH_SOLVE_BLOCKS(int64_t, int32_t, int64_t,
                rowPtr, colInd, values, vecX, vecY, levelPtr, levelRow,
                (__gm__ const int64_t *)(workspace + tiling.diagPtrOffset), validCount,
                alpha, tiling.m, numLevels, tiling.nthreads, tiling.numBlocks, fwd, nonUnit);
        }
    } else {
        // (I64, I64)
        if (tiling.permType == 0) {
            SPSV_DISPATCH_SOLVE_BLOCKS(int64_t, int64_t, int32_t,
                rowPtr, colInd, values, vecX, vecY, levelPtr, levelRow,
                (__gm__ const int32_t *)(workspace + tiling.diagPtrOffset), validCount,
                alpha, tiling.m, numLevels, tiling.nthreads, tiling.numBlocks, fwd, nonUnit);
        } else {
            SPSV_DISPATCH_SOLVE_BLOCKS(int64_t, int64_t, int64_t,
                rowPtr, colInd, values, vecX, vecY, levelPtr, levelRow,
                (__gm__ const int64_t *)(workspace + tiling.diagPtrOffset), validCount,
                alpha, tiling.m, numLevels, tiling.nthreads, tiling.numBlocks, fwd, nonUnit);
        }
    }
}

extern "C" __global__ __aicore__ void spsv_update_values_kernel(
    GM_ADDR newValues, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ float *csrValues = (__gm__ float *)(workspace + tiling.csrValuesOffset);

    if (tiling.permType == 0) {
        __gm__ const int32_t *perm = (__gm__ const int32_t *)(workspace + tiling.permOffset);
        asc_vf_call<SpsvUpdateValuesSimtCompute<int32_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, csrValues, perm, tiling.nnz);
    } else {
        __gm__ const int64_t *perm = (__gm__ const int64_t *)(workspace + tiling.permOffset);
        asc_vf_call<SpsvUpdateValuesSimtCompute<int64_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, csrValues, perm, tiling.nnz);
    }
}

extern "C" __global__ __aicore__ void spsv_update_diag_kernel(
    GM_ADDR newValues, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ float *csrValues = (__gm__ float *)(workspace + tiling.csrValuesOffset);

    if (tiling.permType == 0) {
        __gm__ const int32_t *diagPtr = (__gm__ const int32_t *)(workspace + tiling.diagPtrOffset);
        asc_vf_call<SpsvUpdateDiagSimtCompute<int32_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, csrValues, diagPtr, tiling.m);
    } else {
        __gm__ const int64_t *diagPtr = (__gm__ const int64_t *)(workspace + tiling.diagPtrOffset);
        asc_vf_call<SpsvUpdateDiagSimtCompute<int64_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, csrValues, diagPtr, tiling.m);
    }
}

extern "C" __global__ __aicore__ void spsv_update_diag_csr_kernel(
    GM_ADDR newValues, GM_ADDR values, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.permType == 0) {
        __gm__ const int32_t *diagPtr = (__gm__ const int32_t *)(workspace + tiling.diagPtrOffset);
        asc_vf_call<SpsvUpdateDiagSimtCompute<int32_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, (__gm__ float *)values, diagPtr, tiling.m);
    } else {
        __gm__ const int64_t *diagPtr = (__gm__ const int64_t *)(workspace + tiling.diagPtrOffset);
        asc_vf_call<SpsvUpdateDiagSimtCompute<int64_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, (__gm__ float *)values, diagPtr, tiling.m);
    }
}

extern "C" __global__ __aicore__ void spsv_fill_zero_kernel(
    GM_ADDR vecY, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    asc_vf_call<SpsvFillZeroSimtCompute>(dim3{tiling.nthreads, 1, 1},
        (__gm__ float *)vecY, tiling.m);
}

extern "C" __global__ __aicore__ void spsv_scale_copy_kernel(
    GM_ADDR vecX, GM_ADDR vecY, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    float alpha = tiling.alpha;
    if (tiling.alphaDevicePtr != 0) {
        alpha = *(__gm__ const float *)(tiling.alphaDevicePtr);
    }

    asc_vf_call<SpsvScaleCopySimtCompute>(dim3{tiling.nthreads, 1, 1},
        (__gm__ const float *)vecX, (__gm__ float *)vecY, alpha, tiling.m);
}

extern "C" __global__ __aicore__ void spsv_scale_inf_kernel(
    GM_ADDR vecX, GM_ADDR vecY, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    float alpha = tiling.alpha;
    if (tiling.alphaDevicePtr != 0) {
        alpha = *(__gm__ const float *)(tiling.alphaDevicePtr);
    }

    asc_vf_call<SpsvScaleInfSimtCompute>(dim3{tiling.nthreads, 1, 1},
        (__gm__ const float *)vecX, (__gm__ float *)vecY, alpha, tiling.m);
}

extern "C" __global__ __aicore__ void spsv_copy_values_kernel(
    GM_ADDR src, GM_ADDR dst, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    asc_vf_call<SpsvCopyValuesSimtCompute>(dim3{tiling.nthreads, 1, 1},
        (__gm__ const float *)src, (__gm__ float *)dst, tiling.nnz);
}

// ---------------------------------------------------------------------------
// Host-side kernel launchers
// These functions are compiled with host toolchain and invoke the device
// kernels via the <<<>>> launch syntax. Kept in this file per repo convention
// (spmm/gpsv follow the same pattern).
// ---------------------------------------------------------------------------

void spsv_analysis_kernel_do(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_analysis_kernel<<<numBlocks, nullptr, stream>>>(
        rowPtr, colInd, values, workspace, tiling);
}

void spsv_analysis_serial_kernel_do(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR workspace,
    const SpsvTilingData &tiling, void *stream)
{
    // SIMT kernel requires fixed block size (__launch_bounds__ = 2048 threads).
    // Only thread 0 executes the serial logic; remaining threads idle at
    // asc_syncthreads. This is a SIMT hardware constraint and cannot be
    // reduced to a smaller block size.
    spsv_analysis_serial_kernel<<<1, nullptr, stream>>>(
        rowPtr, colInd, values, workspace, tiling);
}

void spsv_analysis_parallel_kernel_do(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_analysis_parallel_kernel<<<numBlocks, nullptr, stream>>>(
        rowPtr, colInd, workspace, tiling);
}

void spsv_analysis_final_kernel_do(
    GM_ADDR workspace, const SpsvTilingData &tiling, void *stream)
{
    // Same SIMT fixed-block-size constraint as serial phase: 2048 threads
    // are launched but only thread 0 performs the finalization; others idle.
    spsv_analysis_final_kernel<<<1, nullptr, stream>>>(workspace, tiling);
}

void spsv_solve_kernel_do(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
    GM_ADDR vecX, GM_ADDR vecY, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_solve_kernel<<<numBlocks, nullptr, stream>>>(
        rowPtr, colInd, values, vecX, vecY, workspace, tiling);
}

void spsv_update_values_kernel_do(
    GM_ADDR newValues, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_update_values_kernel<<<numBlocks, nullptr, stream>>>(
        newValues, workspace, tiling);
}

void spsv_update_diag_kernel_do(
    GM_ADDR newValues, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_update_diag_kernel<<<numBlocks, nullptr, stream>>>(
        newValues, workspace, tiling);
}

void spsv_update_diag_csr_kernel_do(
    GM_ADDR newValues, GM_ADDR values, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_update_diag_csr_kernel<<<numBlocks, nullptr, stream>>>(
        newValues, values, workspace, tiling);
}

void spsv_fill_zero_kernel_do(
    GM_ADDR vecY,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_fill_zero_kernel<<<numBlocks, nullptr, stream>>>(vecY, tiling);
}

void spsv_scale_copy_kernel_do(
    GM_ADDR vecX, GM_ADDR vecY,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_scale_copy_kernel<<<numBlocks, nullptr, stream>>>(vecX, vecY, tiling);
}

void spsv_scale_inf_kernel_do(
    GM_ADDR vecX, GM_ADDR vecY,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_scale_inf_kernel<<<numBlocks, nullptr, stream>>>(vecX, vecY, tiling);
}

void spsv_copy_values_kernel_do(
    GM_ADDR src, GM_ADDR dst,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_copy_values_kernel<<<numBlocks, nullptr, stream>>>(src, dst, tiling);
}
