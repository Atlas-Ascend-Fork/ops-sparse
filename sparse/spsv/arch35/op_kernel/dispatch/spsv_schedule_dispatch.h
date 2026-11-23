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

#ifndef OP_KERNEL_DISPATCH_SPSV_SCHEDULE_DISPATCH_H_
#define OP_KERNEL_DISPATCH_SPSV_SCHEDULE_DISPATCH_H_

#include "spsv_direction_dispatch.h"

// Check product and reduction error during each row's normal traversal.
// Uncertified rows retry with wider arithmetic and, when necessary, the
// integer accumulator. Every scheduled path carries solution low words to
// subsequent rows, avoiding a separate whole-matrix refinement pass.
template <typename RowPtrT, typename ColIndT, typename PermT>
__aicore__ inline void SpsvSolveFloatDispatch(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX,
                                              GM_ADDR vecY, __gm__ float* yLow, __gm__ const int32_t* levelPtr,
                                              __gm__ const int32_t* levelRow, __gm__ const PermT* diagPtr,
                                              __gm__ const int32_t* validCount, float alpha, int64_t m,
                                              int32_t numLevels, uint32_t nthreads, uint32_t numBlocks, bool fwd,
                                              bool nonUnit)
{
    // Row-local checks and retries are part of this single scheduled pass.
    SpsvSolveFloatDispatchPass<RowPtrT, ColIndT, PermT>(rowPtr, colInd, values, vecX, vecY, yLow, levelPtr, levelRow,
                                                        diagPtr, validCount, alpha, m, numLevels, nthreads, numBlocks,
                                                        fwd, nonUnit);
}

// Match SPSV_DISPATCH_SOLVE_BLOCKS at the call site. The FP32 dispatch owns
// the scheduled solve, its forward-error scan, and any dependency retry.
#define SPSV_DISPATCH_SOLVE_REFINE(RowPtrT, ColIndT, PermT, rowPtr, colInd, values, vecX, vecY, levelPtr, levelRow, \
                                   diagPtr, validCount, alpha, m, numLevels, nthreads, numBlocks, fwd, nonUnit)     \
    SpsvSolveFloatDispatch<RowPtrT, ColIndT, PermT>((rowPtr), (colInd), (values), (vecX), (vecY), yLow, (levelPtr), \
                                                    (levelRow), (diagPtr), (validCount), (alpha), (m), (numLevels), \
                                                    (nthreads), (numBlocks), (fwd), (nonUnit))

// Prepare solve kernel local variables from tiling data.
// Reads levelPtr/levelRow/validCount pointers, numLevels from workspace,
// computes fwd/nonUnit flags, and resolves alpha (host or device pointer).
__aicore__ inline void PrepareSolveKernelArgs(const SpsvTilingData& tiling, __gm__ uint8_t* workspace,
                                              __gm__ const int32_t*& levelPtr, __gm__ const int32_t*& levelRow,
                                              __gm__ const int32_t*& validCount, int32_t& numLevels, bool& fwd,
                                              bool& nonUnit, float& alpha)
{
    levelPtr = (__gm__ const int32_t*)(workspace + tiling.levelPtrOffset);
    levelRow = (__gm__ const int32_t*)(workspace + tiling.levelRowOffset);
    validCount = (__gm__ const int32_t*)(workspace + tiling.validCountOffset);

    // numLevels is read from workspace (written by the analysis kernel),
    // NOT from tiling.numLevels (which is 0 in the host-supplied params).
    __gm__ SpsvTilingData* tilingWs = (__gm__ SpsvTilingData*)workspace;
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
        alpha = *(__gm__ const float*)(tiling.alphaDevicePtr);
    }
}

__aicore__ inline SpsvComplex64 ReadComplexAlpha(const SpsvTilingData& tiling)
{
    if (tiling.alphaDevicePtr != 0) {
        __gm__ const float* p = (__gm__ const float*)(tiling.alphaDevicePtr);
        return {p[0], p[1]};
    }
    return {tiling.alpha, tiling.alphaImag};
}

// One level across every block, and the whole schedule on one block: the two
// shapes the complex dispatch chooses between, each spelled once instead of
// once per (orientation, diagonal) combination.
#define SPSV_CALL_COMPLEX_LEVEL(RowPtrT, ColIndT, PermT, SR, FWD, NU)                                           \
    asc_vf_call<SpsvSolveComplexLevelSimtCompute<RowPtrT, ColIndT, FWD, NU, PermT, SR>>(                        \
        dim3{nthreads, 1, 1}, rp, ci, vv, xx, yy, yWide, levelPtr, levelRow, rowPrep, validCount, alpha, level, \
        static_cast<int32_t>(numBlocks))

#define SPSV_CALL_COMPLEX_WHOLE(RowPtrT, ColIndT, PermT, SR, FWD, NU)                                                  \
    SPSV_CALL_SOLVE_COMPLEX(RowPtrT, ColIndT, PermT, FWD, NU, SR, rowPtr, colInd, values, vecX, vecY, yWide, levelPtr, \
                            levelRow, rowPrep, validCount, alpha, m, numLevels, nthreads)

template <typename RowPtrT, typename ColIndT, typename PermT, bool SHORT_ROWS>
__aicore__ inline void SpsvLaunchComplexChain(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX,
                                              GM_ADDR vecY, __gm__ float* yWide, __gm__ const int32_t* levelPtr,
                                              __gm__ const int32_t* levelRow, __gm__ const float* rowPrep,
                                              __gm__ const int32_t* validCount, SpsvComplex64 alpha, int64_t m,
                                              int32_t numLevels, uint32_t nthreads, uint32_t numBlocks, bool fwd,
                                              bool nonUnit)
{
    if (AscendC::GetBlockIdx() == 0) {
        __gm__ const RowPtrT* rp = (__gm__ const RowPtrT*)rowPtr;
        __gm__ const ColIndT* ci = (__gm__ const ColIndT*)colInd;
        __gm__ const SpsvComplex64* vv = (__gm__ const SpsvComplex64*)values;
        __gm__ const SpsvComplex64* xx = (__gm__ const SpsvComplex64*)vecX;
        __gm__ SpsvComplex64* yy = (__gm__ SpsvComplex64*)vecY;
#define SPSV_CHAIN(F, N)                                                                  \
    asc_vf_call<SpsvSolveComplexChainCompute<RowPtrT, ColIndT, F, N, PermT, SHORT_ROWS>>( \
        dim3{32, 1, 1}, rp, ci, vv, xx, yy, yWide, levelPtr, levelRow, rowPrep, validCount, alpha, 0, numLevels)
        SPSV_DISPATCH_DIRECTION_AND_DIAGONAL(fwd, nonUnit, SPSV_CHAIN);
#undef SPSV_CHAIN
    }
    if (numBlocks > 1) {
        SpsvSyncSolveBlocks();
    }
}

#include "../validation/spsv_complex_distributed_row.h"

template <typename RowPtrT, typename ColIndT, typename PermT, bool SHORT_ROWS>
__aicore__ inline void SpsvLaunchComplexLevels(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX,
                                               GM_ADDR vecY, __gm__ float* yWide, __gm__ const int32_t* levelPtr,
                                               __gm__ const int32_t* levelRow, __gm__ const float* rowPrep,
                                               __gm__ const int32_t* validCount, SpsvComplex64 alpha, int64_t m,
                                               int32_t numLevels, uint32_t nthreads, uint32_t numBlocks, bool fwd,
                                               bool nonUnit)
{
    __gm__ const RowPtrT* rp = (__gm__ const RowPtrT*)rowPtr;
    __gm__ const ColIndT* ci = (__gm__ const ColIndT*)colInd;
    __gm__ const SpsvComplex64* vv = (__gm__ const SpsvComplex64*)values;
    __gm__ const SpsvComplex64* xx = (__gm__ const SpsvComplex64*)vecX;
    __gm__ SpsvComplex64* yy = (__gm__ SpsvComplex64*)vecY;
    for (int32_t level = 0; level < numLevels; ++level) {
        if (numLevels <= kSpsvDistributedMaxLevels && levelPtr[level+1]-levelPtr[level] == 1 &&
            SpsvTryLaunchDistributedRow(rp, ci, vv, xx, yy, yWide, rowPrep, alpha, m,
                levelRow[levelPtr[level]], numBlocks, fwd, nonUnit)) {
            SpsvSyncSolveBlocks();
            continue;
        }
#define SPSV_LEVEL_DIRECTION(F, N) SPSV_CALL_COMPLEX_LEVEL(RowPtrT, ColIndT, PermT, SHORT_ROWS, F, N)
        SPSV_DISPATCH_DIRECTION_AND_DIAGONAL(fwd, nonUnit, SPSV_LEVEL_DIRECTION);
#undef SPSV_LEVEL_DIRECTION
        SpsvSyncSolveBlocks();
    }
}

template <typename RowPtrT, typename ColIndT, typename PermT, bool SHORT_ROWS, bool FORWARD, bool NON_UNIT>
__aicore__ inline void SpsvLaunchComplexSplit(__gm__ const RowPtrT* rp, __gm__ const ColIndT* ci,
                                              __gm__ const SpsvComplex64* vv, __gm__ const SpsvComplex64* xx,
                                              __gm__ SpsvComplex64* yy, __gm__ float* yWide,
                                              __gm__ const int32_t* levelPtr, __gm__ const int32_t* levelRow,
                                              __gm__ const float* rowPrep, __gm__ const int32_t* validCount,
                                              SpsvComplex64 alpha, int32_t prefix, int32_t suffix, int32_t firstWidth,
                                              int32_t numLevels, uint32_t nthreads)
{
    asc_vf_call<SpsvSolveComplexSimtCompute<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, SHORT_ROWS>>(
        dim3{nthreads, 1, 1}, rp, ci, vv, xx, yy, yWide, levelPtr, levelRow, rowPrep, validCount, alpha, 0, prefix);
    if (suffix > prefix) {
        asc_vf_call<SpsvSolveComplexChainCompute<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, SHORT_ROWS>>(
            dim3{32, 1, 1}, rp, ci, vv, xx, yy, yWide, levelPtr, levelRow, rowPrep, validCount, alpha, firstWidth,
            SpsvComplexSingletonSuffixBegin(firstWidth, suffix, prefix));
    }
    if (suffix < numLevels) {
        asc_vf_call<SpsvSolveComplexSimtCompute<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, SHORT_ROWS>>(
            dim3{nthreads, 1, 1}, rp, ci, vv, xx, yy, yWide, levelPtr, levelRow, rowPrep, validCount, alpha, suffix,
            numLevels);
    }
}

template <typename RowPtrT, typename ColIndT, typename PermT, bool SHORT_ROWS>
__aicore__ inline void SpsvLaunchComplexSingletonSuffix(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX,
                                                        GM_ADDR vecY, __gm__ float* yWide,
                                                        __gm__ const int32_t* levelPtr, __gm__ const int32_t* levelRow,
                                                        __gm__ const float* rowPrep, __gm__ const int32_t* validCount,
                                                        SpsvComplex64 alpha, int64_t m, int32_t numLevels,
                                                        uint32_t nthreads, uint32_t numBlocks, bool fwd, bool nonUnit,
                                                        int32_t prefix)
{
    int32_t firstWidth = levelPtr[prefix];
    if (AscendC::GetBlockIdx() == 0) {
        __gm__ const RowPtrT* rp = (__gm__ const RowPtrT*)rowPtr;
        __gm__ const ColIndT* ci = (__gm__ const ColIndT*)colInd;
        __gm__ const SpsvComplex64* vv = (__gm__ const SpsvComplex64*)values;
        __gm__ const SpsvComplex64* xx = (__gm__ const SpsvComplex64*)vecX;
        __gm__ SpsvComplex64* yy = (__gm__ SpsvComplex64*)vecY;
        int32_t suffix = numLevels;
        while (suffix > prefix) {
            int32_t row = levelRow[SpsvComplexSingletonSuffixBegin(firstWidth, suffix, prefix) - 1];
            if (rp[row + 1] - rp[row] < static_cast<RowPtrT>(256)) {
                break;
            }
            --suffix;
        }
#define SPSV_SPLIT_DIRECTION(F, N)                                                                                   \
    SpsvLaunchComplexSplit<RowPtrT, ColIndT, PermT, SHORT_ROWS, F, N>(rp, ci, vv, xx, yy, yWide, levelPtr, levelRow, \
                                                                      rowPrep, validCount, alpha, prefix, suffix,    \
                                                                      firstWidth, numLevels, nthreads)
        SPSV_DISPATCH_DIRECTION_AND_DIAGONAL(fwd, nonUnit, SPSV_SPLIT_DIRECTION);
#undef SPSV_SPLIT_DIRECTION
    }
}

template <typename RowPtrT, typename ColIndT, typename PermT, bool SHORT_ROWS>
__aicore__ inline void SpsvSolveComplexDispatchPass(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX,
                                                    GM_ADDR vecY, __gm__ float* yWide, __gm__ const int32_t* levelPtr,
                                                    __gm__ const int32_t* levelRow, __gm__ const float* rowPrep,
                                                    __gm__ const int32_t* validCount, SpsvComplex64 alpha, int64_t m,
                                                    int32_t numLevels, uint32_t nthreads, uint32_t numBlocks, bool fwd,
                                                    bool nonUnit)
{
    if (numLevels == m) {
        SpsvLaunchComplexChain<RowPtrT, ColIndT, PermT, SHORT_ROWS>(rowPtr, colInd, values, vecX, vecY, yWide, levelPtr,
                                                                    levelRow, rowPrep, validCount, alpha, m, numLevels,
                                                                    nthreads, numBlocks, fwd, nonUnit);
        return;
    }
    // Shallow schedules remain single-block to avoid synchronization overhead;
    // deeper schedules use all blocks with a barrier between levels.  A level
    // holding fewer than kSpsvComplexMultiBlockMinLevelWidth rows cannot fill them, and
    // the per-level launch and grid barrier then dominate.
    int32_t meanLevelWidth = (numLevels > 0) ? static_cast<int32_t>(m / static_cast<int64_t>(numLevels)) : 1;
    if (numBlocks > 1 && meanLevelWidth >= kSpsvComplexMultiBlockMinLevelWidth) {
        SpsvLaunchComplexLevels<RowPtrT, ColIndT, PermT, SHORT_ROWS>(rowPtr, colInd, values, vecX, vecY, yWide,
                                                                     levelPtr, levelRow, rowPrep, validCount, alpha, m,
                                                                     numLevels, nthreads, numBlocks, fwd, nonUnit);
        return;
    }
    if (numBlocks > 1 && AscendC::GetBlockIdx() != 0) {
        // The single-block schedule below runs on block 0; the rest only need
        // to be here for the barrier that follows.
        SpsvSyncSolveBlocks();
        return;
    }
    // A wide first level followed by singleton levels has a compact row
    // order. Keep its long serial part in the small, warp-only VF; retain
    // block cooperation for the contiguous suffix of long rows. An earlier
    // long row is still handled correctly by the warp path.
    // Each analysis level is nonempty. The row-count equality therefore
    // proves that every remaining level is a singleton, without a full scan.
    // Limit prefix discovery to 64 levels to bound its scalar setup cost.
    int32_t prefix = SpsvComplexSingletonPrefix(levelPtr, numLevels, m);
    if (prefix < numLevels && static_cast<int64_t>(levelPtr[prefix]) + numLevels - prefix == m) {
        SpsvLaunchComplexSingletonSuffix<RowPtrT, ColIndT, PermT, SHORT_ROWS>(
            rowPtr, colInd, values, vecX, vecY, yWide, levelPtr, levelRow, rowPrep, validCount, alpha, m, numLevels,
            nthreads, numBlocks, fwd, nonUnit, prefix);
    } else {
#define SPSV_WHOLE_DIRECTION(F, N) SPSV_CALL_COMPLEX_WHOLE(RowPtrT, ColIndT, PermT, SHORT_ROWS, F, N)
        SPSV_DISPATCH_DIRECTION_AND_DIAGONAL(fwd, nonUnit, SPSV_WHOLE_DIRECTION);
#undef SPSV_WHOLE_DIRECTION
    }

    if (numBlocks > 1) {
        SpsvSyncSolveBlocks();
    }
}

// One complex pass per (row-pointer, column-index, permutation) width.  The
// short-row instantiation is a separate template so that neither carries the
// other's branch; spelling the twelve combinations out cost a hundred lines of
// identical calls.
#define SPSV_CALL_COMPLEX_PASS(RowPtrT, ColIndT, PermT)                                                           \
    do {                                                                                                          \
        if (shortRows) {                                                                                          \
            SpsvSolveComplexDispatchPass<RowPtrT, ColIndT, PermT, true>(                                          \
                rowPtr, colInd, values, effX, vecY, yWide, levelPtr, levelRow, rowPrep, validCount, alphaComplex, \
                tiling.m, numLevels, tiling.nthreads, tiling.numBlocks, fwd, nonUnit);                            \
        } else {                                                                                                  \
            SpsvSolveComplexDispatchPass<RowPtrT, ColIndT, PermT, false>(                                         \
                rowPtr, colInd, values, effX, vecY, yWide, levelPtr, levelRow, rowPrep, validCount, alphaComplex, \
                tiling.m, numLevels, tiling.nthreads, tiling.numBlocks, fwd, nonUnit);                            \
        }                                                                                                         \
    } while (0)

// 1/diag and alpha*x for every row, computed once off the dependency chain:
// see SpsvComplexDiagRecipCompute.  Unit diagonals pass a null diagPtr and so
// skip the divide entirely.

#endif // OP_KERNEL_DISPATCH_SPSV_SCHEDULE_DISPATCH_H_
