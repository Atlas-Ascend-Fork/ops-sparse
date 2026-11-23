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

#ifndef SPSV_FORMAT_IMPL_H_
#define SPSV_FORMAT_IMPL_H_

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "../spsv_kernel.h"
#include "../math/spsv_arith.h"

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

// kSimtMaxThreads is defined in spsv_tiling_data.h (included via spsv_kernel.h).

// Parameter-list macros: eliminate codecheck duplicate-lines warnings by
// declaring shared function parameter lists in a single place. Each macro is
// expanded into the function signature of both the serial and parallel
// variant, so the parameter list never appears twice in the source.

// Dual-template parameter macros for mixed index widths (RowPtrT, ColIndT).
// RowPtrT: element type for row-offset arrays (ptrType from matrix descriptor).
// ColIndT: element type for column-index arrays (idxType from matrix descriptor).

// ValueT is float for the real path and SpsvComplex64 for the complex one;
// the conversion is the same either way, and writing it twice is what the
// duplicate-code finding was about.
#define SPSV_COO_BUILD_PARAMS(RowPtrT, ColIndT, PermT, ValueT)                                           \
    __gm__ const RowPtrT *cooRowInd, __gm__ const ColIndT *cooColInd, __gm__ const ValueT *cooValues,    \
        __gm__ RowPtrT *wsRowPtr, __gm__ ColIndT *wsColInd, __gm__ ValueT *wsValues, __gm__ PermT *perm, \
        __gm__ PermT *scratch, int64_t m, int64_t nnz, int32_t idxBase
#define SPSV_SLICED_ELL_PARAMS(RowPtrT, ColIndT, PermT)                                                            \
    __gm__ const RowPtrT *slicePtr, __gm__ const ColIndT *sellColInd, __gm__ const float *sellValues,              \
        __gm__ RowPtrT *wsRowPtr, __gm__ ColIndT *wsColInd, __gm__ float *wsValues, __gm__ PermT *perm, int64_t m, \
        int32_t numSlices, int32_t sliceWidth, int32_t idxBase
#define SPSV_TRANS_PARAMS(RowPtrT, ColIndT, PermT)                                                                    \
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd, __gm__ const float *values,                           \
        __gm__ RowPtrT *transRowPtr, __gm__ ColIndT *transColInd, __gm__ float *transValues, __gm__ PermT *transPerm, \
        __gm__ PermT *scratch, int64_t m, int64_t nnz, int32_t idxBase
// Setup workspace pointer helpers below are called from SpsvBuildWorkspaceCsr
// and SpsvBuildWorkspaceCsrParallel with runtime permType branching.

// A COO entry decoded to 0-based indices, or false when either is outside the
// matrix.
//
// Checked in the original index type, before any narrowing: an int64 index
// cast to int32 can wrap into [0, m) and bypass the range check (issues
// #149/#150).  All four passes -- serial and parallel, histogram and scatter
// -- decode through here, because the two builders have to agree on which
// entries count.  Counting an entry the scatter will skip, and storing a
// column the solve will index vecY with, are the same defect from opposite
// ends; when only the row was checked on the parallel path, the same illegal
// input was dropped by one builder and written into the workspace CSR by the
// other.
template <typename RowPtrT, typename ColIndT>
__simt_callee__ inline bool SpsvDecodeCooEntry(__gm__ const RowPtrT* cooRowInd, __gm__ const ColIndT* cooColInd,
                                               int64_t p, int32_t idxBase, int64_t m, int32_t& row, ColIndT& col)
{
    RowPtrT rawR = cooRowInd[p] - static_cast<RowPtrT>(idxBase);
    ColIndT rawC = cooColInd[p] - static_cast<ColIndT>(idxBase);
    if (rawR < 0 || rawR >= static_cast<RowPtrT>(m) || rawC < 0 || rawC >= static_cast<ColIndT>(m)) {
        return false;
    }
    row = static_cast<int32_t>(rawR);
    col = rawC;
    return true;
}

// Serial COO -> CSR conversion (single-thread, thread 0 only).
template <typename RowPtrT, typename ColIndT, typename PermT, typename ValueT>
__simt_callee__ inline void SpsvBuildCsrFromCoo(SPSV_COO_BUILD_PARAMS(RowPtrT, ColIndT, PermT, ValueT))
{
    // wsRowPtr has m+1 elements (allocated by host via ComputeWorkspaceSize).
    // idxBase: 0 for 0-based COO, 1 for 1-based (Fortran). Subtract idxBase
    // when reading row/col indices to normalize to 0-based workspace CSR.
    for (int64_t i = 0; i <= m; i++) {
        wsRowPtr[i] = 0;
    }
    for (int64_t p = 0; p < nnz; p++) {
        int32_t r = 0;
        ColIndT c = 0;
        if (!SpsvDecodeCooEntry(cooRowInd, cooColInd, p, idxBase, m, r, c)) {
            continue;
        }
        wsRowPtr[r + 1]++;
    }
    for (int64_t i = 0; i < m; i++) {
        wsRowPtr[i + 1] += wsRowPtr[i];
    }
    for (int64_t i = 0; i < m; i++) {
        scratch[i] = 0;
    }
    for (int64_t p = 0; p < nnz; p++) {
        int32_t row = 0;
        ColIndT col = 0;
        if (!SpsvDecodeCooEntry(cooRowInd, cooColInd, p, idxBase, m, row, col)) {
            continue;
        }
        RowPtrT pos = wsRowPtr[row] + static_cast<RowPtrT>(scratch[row]);
        scratch[row]++;
        wsColInd[pos] = col;
        SpsvAssignValue(&wsValues[pos], &cooValues[p]);
        perm[pos] = static_cast<PermT>(p);
    }
}
// Parallel COO -> CSR conversion. All threads in the block participate using
// atomic histogram and scatter (multi-core serial phase path).
// idxBase: 0 for 0-based COO, 1 for 1-based (Fortran).
template <typename RowPtrT, typename ColIndT, typename PermT, typename ValueT>
__simt_callee__ inline void SpsvBuildCsrFromCooParallel(SPSV_COO_BUILD_PARAMS(RowPtrT, ColIndT, PermT, ValueT))
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i <= m; i += static_cast<int64_t>(blockDim.x)) {
        wsRowPtr[i] = 0;
    }
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        scratch[i] = 0;
    }
    asc_syncthreads();

    for (int64_t p = static_cast<int64_t>(threadIdx.x); p < nnz; p += static_cast<int64_t>(blockDim.x)) {
        int32_t r = 0;
        ColIndT c = 0;
        if (!SpsvDecodeCooEntry(cooRowInd, cooColInd, p, idxBase, m, r, c)) {
            continue;
        }
        asc_atomic_add(&wsRowPtr[r + 1], 1);
    }
    asc_syncthreads();

    if (threadIdx.x == 0) {
        for (int64_t i = 0; i < m; i++) {
            wsRowPtr[i + 1] += wsRowPtr[i];
        }
    }
    asc_syncthreads();

    for (int64_t p = static_cast<int64_t>(threadIdx.x); p < nnz; p += static_cast<int64_t>(blockDim.x)) {
        int32_t row = 0;
        ColIndT col = 0;
        if (!SpsvDecodeCooEntry(cooRowInd, cooColInd, p, idxBase, m, row, col)) {
            continue;
        }
        int32_t pos = static_cast<int32_t>(wsRowPtr[row]) + asc_atomic_add(&scratch[row], 1);
        wsColInd[pos] = col;
        SpsvAssignValue(&wsValues[pos], &cooValues[p]);
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
#define SPSV_TRANSPOSE_FOREACH_ROW(RowPtrT, idxBase)                                                                   \
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x))                  \
        for (RowPtrT p = rowPtr[i] - static_cast<RowPtrT>(idxBase); p < rowPtr[i + 1] - static_cast<RowPtrT>(idxBase); \
             p++)
#define SPSV_TRANSPOSE_BOUNDSCHECK_COL(ColIndT, idxBase)     \
    int32_t col = static_cast<int32_t>(colInd[p]) - idxBase; \
    if (col < 0 || col >= static_cast<int32_t>(m)) {         \
        continue;                                            \
    }
// Parallel CSR transpose. All threads in the block participate using atomic
// histogram and scatter.
// idxBase: 0 for 0-based source CSR, 1 for 1-based (Fortran).
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvTransposeCsrParallel(SPSV_TRANS_PARAMS(RowPtrT, ColIndT, PermT))
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i <= m; i += static_cast<int64_t>(blockDim.x)) {
        transRowPtr[i] = 0;
    }
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        scratch[i] = 0;
    }
    asc_syncthreads();

    SPSV_TRANSPOSE_FOREACH_ROW(RowPtrT, idxBase)
    {
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

    SPSV_TRANSPOSE_FOREACH_ROW(RowPtrT, idxBase)
    {
        SPSV_TRANSPOSE_BOUNDSCHECK_COL(ColIndT, idxBase)
        RowPtrT pos = transRowPtr[col] + static_cast<RowPtrT>(asc_atomic_add(&scratch[col], 1));
        transColInd[pos] = static_cast<ColIndT>(i);
        transValues[pos] = values[p];
        transPerm[pos] = static_cast<PermT>(p);
    }
    asc_syncthreads();
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
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
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

    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
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
#define SPSV_CONVERT_FORMAT_TO_CSR_SERIAL(RowPtrT, ColIndT)                                                           \
    do {                                                                                                              \
        if (format == 2) {                                                                                            \
            if (permType == 0) {                                                                                      \
                __gm__ int32_t* permWs = (__gm__ int32_t*)(workspace + permOffset);                                   \
                __gm__ int32_t* dgPtr = (__gm__ int32_t*)(workspace + diagPtrOffset);                                 \
                SpsvBuildCsrFromCoo<RowPtrT, ColIndT, int32_t>(rowPtr, colInd, values, wsRowPtr, wsColInd, wsValues,  \
                                                               permWs, dgPtr, m, nnz, idxBase);                       \
            } else {                                                                                                  \
                __gm__ int64_t* permWs = (__gm__ int64_t*)(workspace + permOffset);                                   \
                __gm__ int64_t* dgPtr = (__gm__ int64_t*)(workspace + diagPtrOffset);                                 \
                SpsvBuildCsrFromCoo<RowPtrT, ColIndT, int64_t>(rowPtr, colInd, values, wsRowPtr, wsColInd, wsValues,  \
                                                               permWs, dgPtr, m, nnz, idxBase);                       \
            }                                                                                                         \
            srcRowPtr = wsRowPtr;                                                                                     \
            srcColInd = wsColInd;                                                                                     \
            srcValues = wsValues;                                                                                     \
            srcIdxBase = 0;                                                                                           \
        } else if (format == 3) {                                                                                     \
            if (permType == 0) {                                                                                      \
                __gm__ int32_t* permWs = (__gm__ int32_t*)(workspace + permOffset);                                   \
                SpsvBuildCsrFromSlicedEll<RowPtrT, ColIndT, int32_t>(                                                 \
                    rowPtr, colInd, values, wsRowPtr, wsColInd, wsValues, permWs, m, numSlices, sliceWidth, idxBase); \
            } else {                                                                                                  \
                __gm__ int64_t* permWs = (__gm__ int64_t*)(workspace + permOffset);                                   \
                SpsvBuildCsrFromSlicedEll<RowPtrT, ColIndT, int64_t>(                                                 \
                    rowPtr, colInd, values, wsRowPtr, wsColInd, wsValues, permWs, m, numSlices, sliceWidth, idxBase); \
            }                                                                                                         \
            srcRowPtr = wsRowPtr;                                                                                     \
            srcColInd = wsColInd;                                                                                     \
            srcValues = wsValues;                                                                                     \
            srcIdxBase = 0;                                                                                           \
        } else if (format == 0 && idxBase == 1) {                                                                     \
            RowPtrT rowBase = static_cast<RowPtrT>(idxBase);                                                          \
            ColIndT colBase = static_cast<ColIndT>(idxBase);                                                          \
            for (int64_t i = 0; i <= m; i++) {                                                                        \
                wsRowPtr[i] = rowPtr[i] - rowBase;                                                                    \
            }                                                                                                         \
            for (int64_t i = 0; i < nnz; i++) {                                                                       \
                wsColInd[i] = colInd[i] - colBase;                                                                    \
                wsValues[i] = values[i];                                                                              \
            }                                                                                                         \
            if (permType == 0) {                                                                                      \
                __gm__ int32_t* permWs = (__gm__ int32_t*)(workspace + permOffset);                                   \
                for (int64_t i = 0; i < nnz; i++) {                                                                   \
                    permWs[i] = static_cast<int32_t>(i);                                                              \
                }                                                                                                     \
            } else {                                                                                                  \
                __gm__ int64_t* permWs = (__gm__ int64_t*)(workspace + permOffset);                                   \
                for (int64_t i = 0; i < nnz; i++) {                                                                   \
                    permWs[i] = static_cast<int64_t>(i);                                                              \
                }                                                                                                     \
            }                                                                                                         \
            srcRowPtr = wsRowPtr;                                                                                     \
            srcColInd = wsColInd;                                                                                     \
            srcValues = wsValues;                                                                                     \
            srcIdxBase = 0;                                                                                           \
        }                                                                                                             \
    } while (0)
// ConvertFormatToCsrParallel: dispatch format conversion to parallel helpers with int64 serial fallback.
// Same local variable requirements as SPSV_CONVERT_FORMAT_TO_CSR_SERIAL.
#define SPSV_CONVERT_FORMAT_TO_CSR_PARALLEL(RowPtrT, ColIndT)                                                         \
    do {                                                                                                              \
        if (format == 2) {                                                                                            \
            if (permType == 0) {                                                                                      \
                __gm__ int32_t* permWs = (__gm__ int32_t*)(workspace + permOffset);                                   \
                __gm__ int32_t* dgPtr = (__gm__ int32_t*)(workspace + diagPtrOffset);                                 \
                SpsvBuildCsrFromCooParallel<RowPtrT, ColIndT, int32_t>(rowPtr, colInd, values, wsRowPtr, wsColInd,    \
                                                                       wsValues, permWs, dgPtr, m, nnz, idxBase);     \
            } else {                                                                                                  \
                if (threadIdx.x == 0) {                                                                               \
                    __gm__ int64_t* permWs = (__gm__ int64_t*)(workspace + permOffset);                               \
                    __gm__ int64_t* dgPtr = (__gm__ int64_t*)(workspace + diagPtrOffset);                             \
                    SpsvBuildCsrFromCoo<RowPtrT, ColIndT, int64_t>(rowPtr, colInd, values, wsRowPtr, wsColInd,        \
                                                                   wsValues, permWs, dgPtr, m, nnz, idxBase);         \
                }                                                                                                     \
                asc_syncthreads();                                                                                    \
            }                                                                                                         \
            srcRowPtr = wsRowPtr;                                                                                     \
            srcColInd = wsColInd;                                                                                     \
            srcValues = wsValues;                                                                                     \
            srcIdxBase = 0;                                                                                           \
        } else if (format == 3) {                                                                                     \
            if (permType == 0) {                                                                                      \
                __gm__ int32_t* permWs = (__gm__ int32_t*)(workspace + permOffset);                                   \
                SpsvBuildCsrFromSlicedEllParallel<RowPtrT, ColIndT, int32_t>(                                         \
                    rowPtr, colInd, values, wsRowPtr, wsColInd, wsValues, permWs, m, numSlices, sliceWidth, idxBase); \
            } else {                                                                                                  \
                if (threadIdx.x == 0) {                                                                               \
                    __gm__ int64_t* permWs = (__gm__ int64_t*)(workspace + permOffset);                               \
                    SpsvBuildCsrFromSlicedEll<RowPtrT, ColIndT, int64_t>(rowPtr, colInd, values, wsRowPtr, wsColInd,  \
                                                                         wsValues, permWs, m, numSlices, sliceWidth,  \
                                                                         idxBase);                                    \
                }                                                                                                     \
                asc_syncthreads();                                                                                    \
            }                                                                                                         \
            srcRowPtr = wsRowPtr;                                                                                     \
            srcColInd = wsColInd;                                                                                     \
            srcValues = wsValues;                                                                                     \
            srcIdxBase = 0;                                                                                           \
        } else if (format == 0 && idxBase == 1) {                                                                     \
            RowPtrT rowBase = static_cast<RowPtrT>(idxBase);                                                          \
            ColIndT colBase = static_cast<ColIndT>(idxBase);                                                          \
            for (int64_t i = static_cast<int64_t>(threadIdx.x); i <= m; i += static_cast<int64_t>(blockDim.x)) {      \
                wsRowPtr[i] = rowPtr[i] - rowBase;                                                                    \
            }                                                                                                         \
            if (permType == 0) {                                                                                      \
                __gm__ int32_t* permWs = (__gm__ int32_t*)(workspace + permOffset);                                   \
                for (int64_t i = static_cast<int64_t>(threadIdx.x); i < nnz; i += static_cast<int64_t>(blockDim.x)) { \
                    wsColInd[i] = colInd[i] - colBase;                                                                \
                    wsValues[i] = values[i];                                                                          \
                    permWs[i] = static_cast<int32_t>(i);                                                              \
                }                                                                                                     \
            } else {                                                                                                  \
                __gm__ int64_t* permWs = (__gm__ int64_t*)(workspace + permOffset);                                   \
                for (int64_t i = static_cast<int64_t>(threadIdx.x); i < nnz; i += static_cast<int64_t>(blockDim.x)) { \
                    wsColInd[i] = colInd[i] - colBase;                                                                \
                    wsValues[i] = values[i];                                                                          \
                    permWs[i] = static_cast<int64_t>(i);                                                              \
                }                                                                                                     \
            }                                                                                                         \
            asc_syncthreads();                                                                                        \
            srcRowPtr = wsRowPtr;                                                                                     \
            srcColInd = wsColInd;                                                                                     \
            srcValues = wsValues;                                                                                     \
            srcIdxBase = 0;                                                                                           \
        }                                                                                                             \
    } while (0)
// Fold the format-normalisation permutation into the transpose permutation.
//
// The transpose records, for each entry of op(A), where that entry sat in the
// CSR it was given.  When that CSR is the workspace copy built from COO,
// SLICED_ELL, a 1-based index array or a conjugated CSC, those positions are
// meaningless to updateMatrix, which is handed the caller's values in the
// original layout.  perm already maps workspace-CSR position -> source-format
// position, so composing the two leaves transPerm indexing the caller's buffer.
//
// count is the number of entries the transpose actually wrote; entries whose
// column index was out of range were skipped and left untouched, and the bounds
// check keeps those from indexing perm.
template <typename PermT>
__simt_callee__ inline void SpsvComposeTransPerm(__gm__ PermT* transPerm, __gm__ const PermT* perm, int64_t count,
                                                 int64_t nnz)
{
    for (int64_t p = 0; p < count; p++) {
        int64_t q = static_cast<int64_t>(transPerm[p]);
        if (q >= 0 && q < nnz) {
            transPerm[p] = perm[q];
        }
    }
}
// Same composition spread over the block; the caller syncs around it.
template <typename PermT>
__simt_callee__ inline void SpsvComposeTransPermParallel(__gm__ PermT* transPerm, __gm__ const PermT* perm,
                                                         int64_t count, int64_t nnz)
{
    for (int64_t p = static_cast<int64_t>(threadIdx.x); p < count; p += static_cast<int64_t>(blockDim.x)) {
        int64_t q = static_cast<int64_t>(transPerm[p]);
        if (q >= 0 && q < nnz) {
            transPerm[p] = perm[q];
        }
    }
}
// The parallel transpose has no int64 form: composing the permutation is
// sequential in the index type.  Both the serial macro and the parallel one
// therefore run this same body -- the parallel one on a single thread -- so
// it is named once here instead of written out twice.
#define SPSV_TRANSPOSE_CSR_INT64_BODY(RowPtrT, ColIndT)                                                              \
    __gm__ int64_t* tPermWs = (__gm__ int64_t*)(workspace + transPermOffset);                                        \
    __gm__ int64_t* dgPtr = (__gm__ int64_t*)(workspace + diagPtrOffset);                                            \
    SpsvTransposeCsr<RowPtrT, ColIndT, int64_t>(srcRowPtr, srcColInd, srcValues, tRowPtr, tColInd, tValues, tPermWs, \
                                                dgPtr, m, nnz, srcIdxBase);                                          \
    if (normalised) {                                                                                                \
        SpsvComposeTransPerm<int64_t>(tPermWs, (__gm__ const int64_t*)(workspace + permOffset),                      \
                                      static_cast<int64_t>(tRowPtr[m]), nnz);                                        \
    }
// TransposeCsrIfNeeded: transpose src CSR into workspace if opA != 0 (serial version).
// Requires local variables: srcRowPtr, srcColInd, srcValues, srcIdxBase,
// tRowPtr, tColInd, tValues, workspace, transPermOffset, diagPtrOffset,
// transRowPtrOffset, opA, permType, m, nnz.
#define SPSV_TRANSPOSE_CSR_IF_NEEDED_SERIAL(RowPtrT, ColIndT)                                                  \
    do {                                                                                                       \
        if (opA != 0 && transRowPtrOffset >= 0) {                                                              \
            bool normalised = (srcRowPtr == wsRowPtr);                                                         \
            if (permType == 0) {                                                                               \
                __gm__ int32_t* tPermWs = (__gm__ int32_t*)(workspace + transPermOffset);                      \
                __gm__ int32_t* dgPtr = (__gm__ int32_t*)(workspace + diagPtrOffset);                          \
                SpsvTransposeCsr<RowPtrT, ColIndT, int32_t>(srcRowPtr, srcColInd, srcValues, tRowPtr, tColInd, \
                                                            tValues, tPermWs, dgPtr, m, nnz, srcIdxBase);      \
                if (normalised) {                                                                              \
                    SpsvComposeTransPerm<int32_t>(tPermWs, (__gm__ const int32_t*)(workspace + permOffset),    \
                                                  static_cast<int64_t>(tRowPtr[m]), nnz);                      \
                }                                                                                              \
            } else {                                                                                           \
                SPSV_TRANSPOSE_CSR_INT64_BODY(RowPtrT, ColIndT)                                                \
            }                                                                                                  \
        }                                                                                                      \
    } while (0)
// TransposeCsrIfNeededParallel: transpose src CSR into workspace if opA != 0 (parallel, int64 serial fallback).
#define SPSV_TRANSPOSE_CSR_IF_NEEDED_PARALLEL(RowPtrT, ColIndT)                                                        \
    do {                                                                                                               \
        if (opA != 0 && transRowPtrOffset >= 0) {                                                                      \
            bool normalised = (srcRowPtr == wsRowPtr);                                                                 \
            if (permType == 0) {                                                                                       \
                __gm__ int32_t* tPermWs = (__gm__ int32_t*)(workspace + transPermOffset);                              \
                __gm__ int32_t* dgPtr = (__gm__ int32_t*)(workspace + diagPtrOffset);                                  \
                SpsvTransposeCsrParallel<RowPtrT, ColIndT, int32_t>(srcRowPtr, srcColInd, srcValues, tRowPtr, tColInd, \
                                                                    tValues, tPermWs, dgPtr, m, nnz, srcIdxBase);      \
                if (normalised) {                                                                                      \
                    SpsvComposeTransPermParallel<int32_t>(tPermWs, (__gm__ const int32_t*)(workspace + permOffset),    \
                                                          static_cast<int64_t>(tRowPtr[m]), nnz);                      \
                    asc_syncthreads();                                                                                 \
                }                                                                                                      \
            } else {                                                                                                   \
                if (threadIdx.x == 0) {                                                                                \
                    SPSV_TRANSPOSE_CSR_INT64_BODY(RowPtrT, ColIndT)                                                    \
                }                                                                                                      \
                asc_syncthreads();                                                                                     \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#endif // SPSV_FORMAT_IMPL_H_
