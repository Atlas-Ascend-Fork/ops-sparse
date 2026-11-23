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

#ifndef OP_KERNEL_ANALYSIS_SPSV_COMPLEX_ANALYSIS_H_
#define OP_KERNEL_ANALYSIS_SPSV_COMPLEX_ANALYSIS_H_

// Dispatch macros for asc_vf_call. Kept as macros (not inline template
// functions) because asc_vf_call requires the VF function template to be
// spelled out as a non-type template argument, and the SIMT compiler does
// not support forwarding this through an intermediate template wrapper.
#define SPSV_CALL_ANALYSIS(RowPtrT, ColIndT, PermT, Func, rp, ci, val, ws, t)                                    \
    asc_vf_call<Func<RowPtrT, ColIndT, PermT>>(                                                                  \
        dim3{t.nthreads, 1, 1}, (__gm__ const RowPtrT*)rp, (__gm__ const ColIndT*)ci, (__gm__ const float*)val,  \
        (__gm__ uint8_t*)ws, t.m, t.nnz, t.fillMode, t.opA, t.levelPtrOffset, t.levelRowOffset, t.diagPtrOffset, \
        t.validCountOffset, t.csrRowPtrOffset, t.csrColIndOffset, t.csrValuesOffset, t.permOffset,               \
        t.transRowPtrOffset, t.transColIndOffset, t.transValuesOffset, t.transPermOffset, t.format, t.numSlices, \
        t.sliceWidth, t.permType, t.idxBase)

#define SPSV_CALL_ANALYSIS_PAR(RowPtrT, ColIndT, PermT, rp, ci, ws, t)                                                 \
    asc_vf_call<SpsvAnalysisParallelPhase<RowPtrT, ColIndT, PermT>>(                                                   \
        dim3{t.nthreads, 1, 1}, (__gm__ const RowPtrT*)rp, (__gm__ const ColIndT*)ci, (__gm__ uint8_t*)ws, t.m,        \
        t.fillMode, t.opA, t.levelPtrOffset, t.levelRowOffset, t.diagPtrOffset, t.validCountOffset, t.csrRowPtrOffset, \
        t.csrColIndOffset, t.transRowPtrOffset, t.transColIndOffset, t.format, static_cast<int32_t>(t.numBlocks),      \
        t.idxBase)

// Dedicated complex64 analysis path.  The legacy analysis helpers use float
// value pointers for FP32 and therefore cannot safely materialize an 8-byte
// complex value into CSR/transpose workspace.  This path performs the same
// structural analysis while copying values as interleaved {real, imag} pairs.

// SLICED_ELL -> workspace CSR for complex values.  Mirrors
// SpsvBuildCsrFromSlicedEll.
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvBuildCsrFromSlicedEllComplex(
    __gm__ const RowPtrT* slicePtr, __gm__ const ColIndT* sellColInd, __gm__ const SpsvComplex64* sellValues,
    __gm__ RowPtrT* wsRowPtr, __gm__ ColIndT* wsColInd, __gm__ SpsvComplex64* wsValues, __gm__ PermT* perm, int64_t m,
    int32_t numSlices, int32_t sliceWidth, int32_t idxBase)
{
    int64_t nnzCounter = 0;
    for (int64_t i = 0; i < m; i++) {
        wsRowPtr[i] = static_cast<RowPtrT>(nnzCounter);
        for (int32_t sl = 0; sl < numSlices; sl++) {
            int64_t sliceStart = static_cast<int64_t>(slicePtr[sl]);
            for (int32_t k = 0; k < sliceWidth; k++) {
                int64_t p = sliceStart + i * sliceWidth + k;
                if (sellColInd[p] >= 0) {
                    wsColInd[nnzCounter] = sellColInd[p] - static_cast<ColIndT>(idxBase);
                    wsValues[nnzCounter].real = sellValues[p].real;
                    wsValues[nnzCounter].imag = sellValues[p].imag;
                    perm[nnzCounter] = static_cast<PermT>(p);
                    nnzCounter++;
                }
            }
        }
    }
    wsRowPtr[m] = static_cast<RowPtrT>(nnzCounter);
}

// COO and SLICED_ELL carry their own layout and are normalised into the
// workspace CSR; 1-based CSR/CSC only needs the index shift.  conjValues
// covers CSC with CONJUGATE_TRANSPOSE: the host rewrites that to a plain
// solve on the conjugate of the stored matrix, and the conjugation has
// nowhere else to happen because that path builds no transpose.  Returns
// whether the workspace copy is now the one to read.
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline bool SpsvComplexNormaliseToCsr(__gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd,
                                                      __gm__ const SpsvComplex64* values, __gm__ RowPtrT* csrRp,
                                                      __gm__ ColIndT* csrCi, __gm__ SpsvComplex64* csrVal,
                                                      __gm__ PermT* perm, __gm__ PermT* diagPtr, int64_t m, int64_t nnz,
                                                      int32_t format, int32_t numSlices, int32_t sliceWidth,
                                                      int32_t conjValues, int32_t idxBase)
{
    bool converted = false;
    if (format == 2) {
        SpsvBuildCsrFromCoo<RowPtrT, ColIndT, PermT>(rowPtr, colInd, values, csrRp, csrCi, csrVal, perm, diagPtr, m,
                                                     nnz, idxBase);
        converted = true;
    } else if (format == 3) {
        SpsvBuildCsrFromSlicedEllComplex<RowPtrT, ColIndT, PermT>(rowPtr, colInd, values, csrRp, csrCi, csrVal, perm, m,
                                                                  numSlices, sliceWidth, idxBase);
        converted = true;
    } else if (idxBase == 1 || conjValues != 0) {
        for (int64_t i = 0; i <= m; ++i) {
            csrRp[i] = rowPtr[i] - static_cast<RowPtrT>(idxBase);
        }
        for (int64_t i = 0; i < nnz; ++i) {
            csrCi[i] = colInd[i] - static_cast<ColIndT>(idxBase);
            csrVal[i].real = values[i].real;
            csrVal[i].imag = values[i].imag;
            perm[i] = static_cast<PermT>(i);
        }
        converted = true;
    }
    if (converted) {
        if (conjValues != 0) {
            int64_t effNnz = static_cast<int64_t>(csrRp[m]);
            for (int64_t i = 0; i < effNnz; ++i) {
                csrVal[i].imag = -csrVal[i].imag;
            }
        }
    }
    return converted;
}

// op(A) materialised into the transpose workspace, conjugated for H.  The
// per-column cursors are accumulated in diagPtr, which the level pass
// overwrites immediately afterwards, so they need no workspace of their own.
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvComplexBuildTranspose(__gm__ const RowPtrT* srcRp, __gm__ const ColIndT* srcCi,
                                                      __gm__ const SpsvComplex64* srcVal, __gm__ RowPtrT* trRp,
                                                      __gm__ ColIndT* trCi, __gm__ SpsvComplex64* trVal,
                                                      __gm__ PermT* trPerm, __gm__ PermT* perm, __gm__ PermT* diagPtr,
                                                      int64_t m, int64_t nnz, int32_t opA, bool converted)
{
    for (int64_t i = 0; i <= m; ++i)
        trRp[i] = 0;
    for (int64_t i = 0; i < m; ++i) {
        RowPtrT rs = srcRp[i];
        RowPtrT re = srcRp[i + 1];
        for (int64_t p = static_cast<int64_t>(rs); p < re; ++p) {
            ColIndT c = srcCi[p];
            if (c >= 0 && c < static_cast<ColIndT>(m))
                trRp[c + 1]++;
        }
    }
    for (int64_t i = 0; i < m; ++i)
        trRp[i + 1] += trRp[i];
    for (int64_t i = 0; i < m; ++i)
        diagPtr[i] = 0;
    for (int64_t i = 0; i < m; ++i) {
        RowPtrT rs = srcRp[i];
        RowPtrT re = srcRp[i + 1];
        for (int64_t p = static_cast<int64_t>(rs); p < re; ++p) {
            ColIndT c = srcCi[p];
            if (c < 0 || c >= static_cast<ColIndT>(m))
                continue;
            RowPtrT pos = trRp[c] + static_cast<RowPtrT>(diagPtr[c]++);
            trCi[pos] = static_cast<ColIndT>(i);
            trVal[pos].real = srcVal[p].real;
            trVal[pos].imag = srcVal[p].imag;
            if (opA == 2)
                trVal[pos].imag = -trVal[pos].imag;
            trPerm[pos] = static_cast<PermT>(p);
        }
    }
    if (converted) {
        SpsvComposeTransPerm<PermT>(trPerm, perm, static_cast<int64_t>(trRp[m]), nnz);
    }
}

// One row's level: one past the deepest level among the dependencies that
// are already ordered, which for a lower triangle are the columns before the
// diagonal and for an upper one the columns after it.  [lo, hi] says which.
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline int32_t SpsvComplexRowLevel(__gm__ const RowPtrT* effRp, __gm__ const ColIndT* effCi,
                                                   __gm__ const PermT* diagPtr, int64_t i, int32_t lo, int32_t hi)
{
    int32_t lvl = 0;
    for (int64_t p = static_cast<int64_t>(effRp[i]); p < effRp[i + 1]; ++p) {
        int32_t c = static_cast<int32_t>(effCi[p]);
        if (c < lo || c > hi)
            continue;
        int32_t d = static_cast<int32_t>(diagPtr[c]) + 1;
        if (d > lvl)
            lvl = d;
    }
    return lvl;
}

// Each row's level: one past the deepest level among the dependencies it
// has already-ordered.  Walking forward for a lower triangle and backward for
// an upper one means every dependency is settled before it is read.
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline void SpsvComplexAssignLevels(__gm__ const RowPtrT* effRp, __gm__ const ColIndT* effCi,
                                                    __gm__ PermT* diagPtr, int64_t m, bool forward)
{
    for (int64_t i = 0; i < m; ++i)
        diagPtr[i] = 0;
    if (forward) {
        for (int64_t i = 0; i < m; ++i) {
            diagPtr[i] = static_cast<PermT>(
                SpsvComplexRowLevel<RowPtrT, ColIndT, PermT>(effRp, effCi, diagPtr, i, 0, static_cast<int32_t>(i) - 1));
        }
    } else {
        for (int64_t ii = m; ii > 0; --ii) {
            int64_t i = ii - 1;
            diagPtr[i] = static_cast<PermT>(SpsvComplexRowLevel<RowPtrT, ColIndT, PermT>(
                effRp, effCi, diagPtr, i, static_cast<int32_t>(i) + 1, static_cast<int32_t>(m) - 1));
        }
    }
}

// The three things solve reads: which level each row belongs to, where the
// diagonal sits in each row, and how many of the row's dependencies are
// already ordered before it.  Returns the number of levels.
template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ inline int32_t SpsvComplexBuildLevels(__gm__ const RowPtrT* effRp, __gm__ const ColIndT* effCi,
                                                      __gm__ PermT* diagPtr, __gm__ int32_t* levelPtr,
                                                      __gm__ int32_t* levelRow, __gm__ int32_t* validCount, int64_t m,
                                                      bool forward)
{
    SpsvComplexAssignLevels<RowPtrT, ColIndT, PermT>(effRp, effCi, diagPtr, m, forward);

    int32_t numLevels = 0;
    for (int64_t i = 0; i < m; ++i) {
        int32_t lvl = static_cast<int32_t>(diagPtr[i]);
        if (lvl > numLevels)
            numLevels = lvl;
    }
    ++numLevels;
    for (int32_t k = 0; k <= numLevels; ++k)
        levelPtr[k] = 0;
    for (int64_t i = 0; i < m; ++i)
        levelPtr[static_cast<int32_t>(diagPtr[i]) + 1]++;
    for (int32_t k = 0; k < numLevels; ++k)
        levelPtr[k + 1] += levelPtr[k];
    // Use validCount as temporary bucket cursors so levelPtr remains the
    // immutable prefix-sum consumed by solve.
    for (int32_t k = 0; k < numLevels; ++k)
        validCount[k] = levelPtr[k];
    for (int64_t i = 0; i < m; ++i) {
        int32_t lvl = static_cast<int32_t>(diagPtr[i]);
        levelRow[validCount[lvl]++] = static_cast<int32_t>(i);
    }
    for (int64_t i = 0; i < m; ++i) {
        diagPtr[i] = static_cast<PermT>(-1);
        for (int64_t p = static_cast<int64_t>(effRp[i]); p < effRp[i + 1]; ++p) {
            if (effCi[p] == static_cast<ColIndT>(i)) {
                diagPtr[i] = static_cast<PermT>(p);
                break;
            }
        }
        int32_t vc = 0;
        if (forward) {
            for (int64_t p = static_cast<int64_t>(effRp[i]); p < effRp[i + 1] && effCi[p] < static_cast<ColIndT>(i);
                 ++p)
                ++vc;
        } else if (effRp[i + 1] > effRp[i]) {
            for (int64_t p = static_cast<int64_t>(effRp[i + 1]) - 1;
                 p >= effRp[i] && effCi[p] > static_cast<ColIndT>(i); --p)
                ++vc;
        }
        validCount[i] = vc;
    }
    return numLevels;
}

template <typename RowPtrT, typename ColIndT, typename PermT>
__simt_callee__ __attribute__((always_inline)) inline void SpsvBuildComplexAnalysisLevels(
    __gm__ const RowPtrT* effRp, __gm__ const ColIndT* effCi, __gm__ PermT* diagPtr, __gm__ int32_t* levelPtr,
    __gm__ int32_t* levelRow, __gm__ int32_t* validCount, __gm__ uint8_t* workspace, int64_t m, int32_t fillMode,
    int32_t opA, int32_t numBlocks)
{
    int32_t effFill = fillMode;
    if (opA != 0)
        effFill = (effFill == 0) ? 1 : 0;
    bool forward = (effFill == 0);
    if (numBlocks > 1) {
        // the levelRow scatter, diagPtr and validCount are the shared
        // parallel phase's work; this leaves it the levels in diagPtr
        SpsvComplexAssignLevels<RowPtrT, ColIndT, PermT>(effRp, effCi, diagPtr, m, forward);
    } else {
        __gm__ SpsvTilingData* tw = (__gm__ SpsvTilingData*)workspace;
        tw->numLevels = SpsvComplexBuildLevels<RowPtrT, ColIndT, PermT>(effRp, effCi, diagPtr, levelPtr, levelRow,
                                                                        validCount, m, forward);
    }
}

template <typename PermT>
__simt_callee__ __attribute__((always_inline)) inline void SpsvFinishComplexAnalysisHistogram(
    __gm__ uint8_t* workspace, int64_t diagPtrOffset, int64_t levelPtrOffset, int64_t m, int32_t numBlocks)
{
    if (numBlocks > 1) {
        // Publish thread 0's GM levels before the histogram's block barrier.
        asc_threadfence_block();
        // collective, so the handles are taken again out here rather than
        // reaching into thread 0's scope
        SpsvBuildLevelHistogramParallel<PermT>((__gm__ PermT*)(workspace + diagPtrOffset),
                                               (__gm__ int32_t*)(workspace + levelPtrOffset),
                                               (__gm__ SpsvTilingData*)workspace, m);
    }
}

template <typename RowPtrT, typename ColIndT, typename PermT>

#endif // OP_KERNEL_ANALYSIS_SPSV_COMPLEX_ANALYSIS_H_
