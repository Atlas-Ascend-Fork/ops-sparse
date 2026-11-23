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

#ifndef OP_KERNEL_DISPATCH_SPSV_SOLVE_ENTRY_H_
#define OP_KERNEL_DISPATCH_SPSV_SOLVE_ENTRY_H_

#include "../validation/spsv_complex_candidate.h"
#include "../validation/spsv_complex_scaled.h"

__aicore__ inline void SpsvComplexLaunchDiagRecip(GM_ADDR values, GM_ADDR workspace, GM_ADDR effX,
                                                  __gm__ float *rowPrep, SpsvComplex64 alphaComplex,
                                                  bool nonUnit, const SpsvTilingData &tiling)
{
    int32_t levels = ((__gm__ const SpsvTilingData *)workspace)->numLevels;
    // Deep schedules amortize the independent signed residual certificate.
    // Shallow and small schedules retain their ordinary forward-error bounds.
    bool deferredCertificate = tiling.m >= 1024 && levels >= (tiling.m + 3) / 4;
    int32_t blockId = static_cast<int32_t>(AscendC::GetBlockIdx());
    int32_t blocks = static_cast<int32_t>(tiling.numBlocks);
    if (tiling.permType == 0) {
        asc_vf_call<SpsvComplexDiagRecipCompute<int32_t>>(
            dim3{tiling.nthreads, 1, 1}, (__gm__ const SpsvComplex64 *)values,
            nonUnit ? (__gm__ const int32_t *)((__gm__ uint8_t *)workspace + tiling.diagPtrOffset)
                    : (__gm__ const int32_t *)nullptr,
            (__gm__ const SpsvComplex64 *)effX, alphaComplex, rowPrep, tiling.m, blockId, blocks,
            deferredCertificate);
    } else {
        asc_vf_call<SpsvComplexDiagRecipCompute<int64_t>>(
            dim3{tiling.nthreads, 1, 1}, (__gm__ const SpsvComplex64 *)values,
            nonUnit ? (__gm__ const int64_t *)((__gm__ uint8_t *)workspace + tiling.diagPtrOffset)
                    : (__gm__ const int64_t *)nullptr,
            (__gm__ const SpsvComplex64 *)effX, alphaComplex, rowPrep, tiling.m, blockId, blocks,
            deferredCertificate);
    }
    SpsvSyncSolveBlocks();
}

// With no triangular dependencies, the rounded answer is never consumed by
// another row. A bounded FP32 product/divide can therefore use the task's
// component tolerance directly, retaining the exact path for cancellation.
__simt_callee__ inline bool SpsvTryIndependentComplexRow(SpsvComplex64 alpha, SpsvComplex64 x,
                                                         SpsvComplex64 diagonal, SpsvDFComplex &answer)
{
    if (!(SpsvFastComponent(alpha.real) && SpsvFastComponent(alpha.imag) && SpsvFastComponent(x.real) &&
          SpsvFastComponent(x.imag) && SpsvFastComponent(diagonal.real) &&
          SpsvFastComponent(diagonal.imag))) {
        return false;
    }
    float scale = SpsvMaxAbs2(diagonal.real, diagonal.imag);
    if (scale == 0.0f) {
        return false;
    }
    float dr = diagonal.real / scale, di = diagonal.imag / scale;
    float inv = 1.0f / (dr * dr + di * di);
    SpsvComplex64 rhs = SpsvComplexMul(alpha, x);
    SpsvComplex64 q = SpsvComplexDivByDiag(rhs, {dr * inv, -di * inv}, scale);
    // At most 32 rounded operations, normalized diagonal norm in [1,2].
    // 128*u times the absolute product norm encloses input-product,
    // normalization, reciprocal and quotient roundoff, with spare factors
    // for the positive majorant's rounding. The admitted operand range keeps
    // input products and scale factors away from overflow and underflow.
    float norm = (SpsvAbsF(alpha.real) + SpsvAbsF(alpha.imag)) * (SpsvAbsF(x.real) + SpsvAbsF(x.imag));
    float error = norm / scale * 0x1p-17f;
    float reBudget = SpsvAbsF(q.real) * 0x1p-13f + 0x1p-18f;
    float imBudget = SpsvAbsF(q.imag) * 0x1p-13f + 0x1p-18f;
    float reCap = SpsvAbsF(q.real) * 0x1p-21f;
    float imCap = SpsvAbsF(q.imag) * 0x1p-21f;
    reCap = reCap > 0.0025f ? reCap : 0.0025f;
    imCap = imCap > 0.0025f ? imCap : 0.0025f;
    reBudget = reBudget < reCap ? reBudget : reCap;
    imBudget = imBudget < imCap ? imBudget : imCap;
    bool realOnly = alpha.imag == 0.0f && x.imag == 0.0f && diagonal.imag == 0.0f;
    if (!(SpsvIsFiniteF(error) && error <= reBudget && (realOnly || error <= imBudget))) {
        return false;
    }
    answer = {{q.real, 0.0f}, {realOnly ? 0.0f : q.imag, 0.0f}};
    return true;
}

// All rows in one analysis level have no triangular dependencies. Fuse
// RHS and diagonal division, avoiding a separate reciprocal launch and grid
// synchronization for diagonal matrices.
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvComplexDiagonalCompute(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd, __gm__ const SpsvComplex64 *values,
    __gm__ const volatile SpsvComplex64 *vecX, __gm__ SpsvComplex64 *vecY, __gm__ float *yWide,
    __gm__ const int32_t *diagPtr, SpsvComplex64 alpha, int64_t m, int32_t blockId, int32_t numBlocks,
    bool nonUnit, bool forward)
{
    int64_t start = static_cast<int64_t>(blockId) * blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(numBlocks) * blockDim.x;
    for (int64_t r = start; r < m; r += step) {
        int32_t row = static_cast<int32_t>(r);
        SpsvComplex64 diagonal{1.0f, 0.0f};
        if (nonUnit) {
            int32_t p = diagPtr[row];
            diagonal = p >= 0 ? SpsvComplex64{values[p].real, values[p].imag} : SpsvComplex64{0.0f, 0.0f};
        }
        SpsvDFComplex answer{};
        // Ill-conditioned inputs retain the certified path; exponent-domain
        // integer arithmetic covers extreme operands and cancellation.
        SpsvComplex64 x{vecX[row].real, vecX[row].imag};
        bool accepted = SpsvTryIndependentComplexRow(alpha, x, diagonal, answer);
        if (!accepted) {
            accepted = SpsvTryCertifiedComplexRedo<int32_t, int32_t, false>(
                rowPtr, colInd, values, vecX, yWide, alpha, row, forward, nonUnit, diagonal, answer);
        }
        if (!accepted) {
            bool finite = SpsvIsFiniteF(alpha.real) && SpsvIsFiniteF(alpha.imag) &&
                          SpsvIsFiniteF(vecX[row].real) && SpsvIsFiniteF(vecX[row].imag) &&
                          SpsvIsFiniteF(diagonal.real) && SpsvIsFiniteF(diagonal.imag);
            if (finite && (diagonal.real != 0.0f || diagonal.imag != 0.0f)) {
                answer = SpsvComplexExactRow<int32_t, int32_t>(rowPtr, colInd, values, vecX, yWide, alpha,
                                                               row, forward, nonUnit, diagonal);
            } else {
                SpsvComplex64 x{vecX[row].real, vecX[row].imag};
                SpsvComplex64 rhs = SpsvComplexMul(alpha, x);
                float scale = SpsvMaxAbs2(diagonal.real, diagonal.imag);
                float dr = diagonal.real / scale, di = diagonal.imag / scale;
                float inv = 1.0f / (dr * dr + di * di);
                SpsvComplex64 q = SpsvComplexDivByDiag(rhs, {dr * inv, -di * inv}, scale);
                answer = {{q.real, 0.0f}, {q.imag, 0.0f}};
            }
        }
        vecY[row].real = answer.re.hi;
        vecY[row].imag = answer.im.hi;
        int64_t at = 4 * r;
        yWide[at] = answer.re.hi;
        yWide[at + 1] = answer.re.lo;
        yWide[at + 2] = answer.im.hi;
        yWide[at + 3] = answer.im.lo;
    }
}

__aicore__ inline bool SpsvTryComplexProposalSchedule(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
                                                      GM_ADDR vecY, __gm__ float *yWide,
                                                      __gm__ float *rowPrep, __gm__ const int32_t *levelPtr,
                                                      __gm__ const int32_t *levelRow, int32_t numLevels,
                                                      bool fwd, bool nonUnit, const SpsvTilingData &tiling)
{
    // An uncertified proposal has a separate, small VF. The mandatory
    // completed-row scan and original-equation recovery certify every row.
    // Sparse medium-depth graphs spend most time propagating broad
    // forward bounds. The signed candidate is accepted only by a later
    // original-equation certificate, exactly as for the deeper schedules.
    bool sparseDeep = tiling.nnz / tiling.m <= 6 && numLevels >= (tiling.m + 15) / 16;
    if (tiling.m >= 1024 && (numLevels >= (tiling.m + 3) / 4 || sparseDeep)) {
        int32_t firstLevel =
            (tiling.nnz / tiling.m < 8 && tiling.numBlocks > 1 && levelPtr[1] - levelPtr[0] >= 32) ? 1 : 0;
        if (firstLevel != 0) {
            int32_t block = static_cast<int32_t>(AscendC::GetBlockIdx());
            int32_t blocks = static_cast<int32_t>(tiling.numBlocks);
#define SPSV_INITIAL(F, N)                                                                                \
    asc_vf_call<SpsvComplexProposalInitialLevel<N>>(dim3{256, 1, 1}, (__gm__ SpsvComplex64 *)vecY, yWide, \
                                                    rowPrep, levelPtr, levelRow, block, blocks)
            if (fwd && nonUnit) {
                SPSV_INITIAL(true, true);
            } else if (fwd) {
                SPSV_INITIAL(true, false);
            } else if (nonUnit) {
                SPSV_INITIAL(false, true);
            } else {
                SPSV_INITIAL(false, false);
            }
#undef SPSV_INITIAL
            SpsvSyncSolveBlocks();
        }
        if (AscendC::GetBlockIdx() == 0) {
#define SPSV_PROPOSAL(F, N)                                                                           \
    asc_vf_call<SpsvComplexProposalSchedule<F, N>>(                                                   \
        dim3{256, 1, 1}, (__gm__ const int32_t *)rowPtr, (__gm__ const int32_t *)colInd,              \
        (__gm__ const SpsvComplex64 *)values, (__gm__ SpsvComplex64 *)vecY, yWide, rowPrep, levelPtr, \
        levelRow, numLevels, firstLevel)
            if (fwd && nonUnit) {
                SPSV_PROPOSAL(true, true);
            } else if (fwd) {
                SPSV_PROPOSAL(true, false);
            } else if (nonUnit) {
                SPSV_PROPOSAL(false, true);
            } else {
                SPSV_PROPOSAL(false, false);
            }
#undef SPSV_PROPOSAL
        }
        if (tiling.numBlocks > 1) {
            SpsvSyncSolveBlocks();
        }
        return true;
    }

    return false;
}

__aicore__ inline bool SpsvTryScaledComplexEntry(const SpsvComplexCertificateData& data, GM_ADDR workspace,
                                                 const SpsvTilingData& tiling, int32_t levels)
{
    if (!SpsvShouldTryScaledComplex(tiling, levels) || (tiling.diagType == 0 && tiling.m <= 2048)) {
        return false;
    }
    auto status = SpsvTryScaledComplex(data, tiling.m);
    // Pending centers carry infinite error bounds. Keep the proposal and
    // let the original certificate and exact recovery validate every row.
    return status != SpsvScaledStatus::Unsupported;
}

// The complex64 half of the solve entry.  Split out of spsv_solve_kernel,
// which carried both halves in 233 lines at a cyclomatic complexity of 45.
__aicore__ inline void SpsvSolveComplexEntry(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX,
                                             GM_ADDR vecY, GM_ADDR workspace, const SpsvTilingData &tiling)
{
    __gm__ const int32_t *levelPtr =
        (__gm__ const int32_t *)((__gm__ uint8_t *)workspace + tiling.levelPtrOffset);
    __gm__ const int32_t *levelRow =
        (__gm__ const int32_t *)((__gm__ uint8_t *)workspace + tiling.levelRowOffset);
    __gm__ const int32_t *validCount =
        (__gm__ const int32_t *)((__gm__ uint8_t *)workspace + tiling.validCountOffset);
    __gm__ SpsvTilingData *tilingWs = (__gm__ SpsvTilingData *)workspace;
    int32_t numLevels = tilingWs->numLevels;
    int32_t fm = tiling.fillMode;
    // Analysis materializes op(A) into the transpose workspace for T/H.
    // The effective triangular orientation is therefore flipped, just as
    // in the FP32 solve preparation path below.
    if (tiling.opA != 0) {
        fm = (fm == 0) ? 1 : 0;
    }
    bool fwd = (fm == 0);
    bool nonUnit = (tiling.diagType == 0);
    SpsvComplex64 alphaComplex = ReadComplexAlpha(tiling);
    // Wide (hi, lo) shadow of Y: the substitution carries already-solved
    // entries at ~48 significand bits so that components formed by
    // cancellation stay inside the per-element accuracy ceiling.
    __gm__ float *yWide = (__gm__ float *)((__gm__ uint8_t *)workspace + tiling.yWideOffset);
    GM_ADDR effX = vecX;  // Host snapshots an aliased RHS before enqueueing.
    SpsvResolveComplexSolveCsr(rowPtr, colInd, values, workspace, tiling);

    if (numLevels == 1) {
        asc_vf_call<SpsvComplexDiagonalCompute>(
            dim3{tiling.nthreads, 1, 1}, (__gm__ const int32_t *)rowPtr, (__gm__ const int32_t *)colInd,
            (__gm__ const SpsvComplex64 *)values, (__gm__ const volatile SpsvComplex64 *)effX,
            (__gm__ SpsvComplex64 *)vecY, yWide, (__gm__ const int32_t *)(workspace + tiling.diagPtrOffset),
            alphaComplex, tiling.m, static_cast<int32_t>(AscendC::GetBlockIdx()),
            static_cast<int32_t>(tiling.numBlocks), nonUnit, fwd);
        return;
    }
    __gm__ float *rowPrep =
        yWide +
        6 * tiling.m;  // eighteen floats per row, including certified inverse, diagonal and RHS bounds
    SpsvComplexLaunchDiagRecip(values, workspace, effX, rowPrep, alphaComplex, nonUnit, tiling);

    auto scaled = SpsvMakeComplexCertificateData(rowPtr, colInd, values, effX, vecY, (GM_ADDR)yWide,
                                                 workspace + tiling.diagPtrOffset, alphaComplex, fwd, nonUnit);
    if (SpsvTryScaledComplexEntry(scaled, workspace, tiling, numLevels)) {
        return;
    }

    if (SpsvTryComplexProposalSchedule(rowPtr, colInd, values, vecY, yWide, rowPrep, levelPtr, levelRow,
                                       numLevels, fwd, nonUnit, tiling)) {
        return;
    }

    // A matrix whose rows are this short is solved by an instantiation that
    // carries the short-row solver; everything else by the one that does
    // not, so neither carries a branch for the other.  The average only
    // selects: that instantiation still checks each row, because a
    // power-law matrix can average two nonzeros and hold one row of
    // thousands.
    bool shortRows = (tiling.m > 0) && (tiling.nnz / tiling.m < static_cast<int64_t>(kSpsvWarpRowMinNnz));

    // ValidateSpSVCommonParams rejects wider indices/permutations for Complex64.
    // Instantiate only that public API combination.
    SPSV_CALL_COMPLEX_PASS(int32_t, int32_t, int32_t);
}
// One float pass per (row-pointer, column-index) width, over both
// permutation widths.
#define SPSV_CALL_FLOAT_PASS(RowPtrT, ColIndT)                                                           \
    do {                                                                                                 \
        if (tiling.permType == 0) {                                                                      \
            SPSV_DISPATCH_SOLVE_REFINE(                                                                  \
                RowPtrT, ColIndT, int32_t, rowPtr, colInd, values, vecX, vecY, levelPtr, levelRow,       \
                (__gm__ const int32_t *)(workspace + tiling.diagPtrOffset), validCount, alpha, tiling.m, \
                numLevels, tiling.nthreads, tiling.numBlocks, fwd, nonUnit);                             \
        } else {                                                                                         \
            SPSV_DISPATCH_SOLVE_REFINE(                                                                  \
                RowPtrT, ColIndT, int64_t, rowPtr, colInd, values, vecX, vecY, levelPtr, levelRow,       \
                (__gm__ const int64_t *)(workspace + tiling.diagPtrOffset), validCount, alpha, tiling.m, \
                numLevels, tiling.nthreads, tiling.numBlocks, fwd, nonUnit);                             \
        }                                                                                                \
    } while (0)

// The float32 half of the solve entry.
__aicore__ inline void SpsvSolveFloatEntry(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX,
                                           GM_ADDR vecY, GM_ADDR workspace, const SpsvTilingData &tiling)
{
    __gm__ const int32_t *levelPtr;
    __gm__ const int32_t *levelRow;
    __gm__ const int32_t *validCount;
    int32_t numLevels;
    bool fwd;
    bool nonUnit;
    float alpha;
    PrepareSolveKernelArgs(tiling, (__gm__ uint8_t *)workspace, levelPtr, levelRow, validCount, numLevels,
                           fwd, nonUnit, alpha);

    __gm__ float *yLow = (__gm__ float *)(workspace + tiling.yWideOffset) + tiling.m;
    // Dispatch based on (indexType, colIndType, permType)
    if (tiling.indexType == 0 && tiling.colIndType == 0) {
        SPSV_CALL_FLOAT_PASS(int32_t, int32_t);
    } else if (tiling.indexType == 1 && tiling.colIndType == 0) {
        SPSV_CALL_FLOAT_PASS(int64_t, int32_t);
    } else {
        SPSV_CALL_FLOAT_PASS(int64_t, int64_t);
    }
}

#endif // OP_KERNEL_DISPATCH_SPSV_SOLVE_ENTRY_H_
