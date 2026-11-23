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

#ifndef OP_KERNEL_DISPATCH_SPSV_REAL_DISPATCH_H_
#define OP_KERNEL_DISPATCH_SPSV_REAL_DISPATCH_H_

#include "spsv_direction_dispatch.h"

template <typename RowPtrT, typename ColIndT, typename PermT>
__aicore__ inline void SpsvSolveSingleCore(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX, GM_ADDR vecY,
                                           __gm__ float* yLow, __gm__ const int32_t* levelPtr,
                                           __gm__ const int32_t* levelRow, __gm__ const PermT* diagPtr,
                                           __gm__ const int32_t* validCount, float alpha, int64_t m, int32_t numLevels,
                                           uint32_t nthreads, bool fwd, bool nonUnit)
{
    SPSV_DISPATCH_SOLVE(RowPtrT, ColIndT, PermT, SpsvSolveSimtCompute, rowPtr, colInd, values, vecX, vecY, levelPtr,
                        levelRow, diagPtr, validCount, alpha, m, numLevels, nthreads, fwd, nonUnit);
}

template <typename RowPtrT, typename ColIndT, typename PermT>
__aicore__ inline void SpsvSolveMultiCore(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX, GM_ADDR vecY,
                                          __gm__ float* yLow, __gm__ const int32_t* levelPtr,
                                          __gm__ const int32_t* levelRow, __gm__ const PermT* diagPtr,
                                          __gm__ const int32_t* validCount, float alpha, int32_t numLevels,
                                          uint32_t nthreads, int32_t numBlocks, bool fwd, bool nonUnit)
{
    if (numLevels == 1) {
#define SPSV_INDEPENDENT_LEVEL(F, N)                                                                            \
    asc_vf_call<SpsvSolveLevelSimtCompute<RowPtrT, ColIndT, F, N, PermT>>(                                       \
        dim3{nthreads, 1, 1}, (__gm__ const RowPtrT*)rowPtr, (__gm__ const ColIndT*)colInd,                       \
        (__gm__ const float*)values, (__gm__ const float*)vecX, (__gm__ float*)vecY, (__gm__ float*)nullptr,       \
        levelPtr, levelRow, diagPtr, validCount, alpha, 0, numBlocks, 1)
        SPSV_DISPATCH_DIRECTION_AND_DIAGONAL(fwd, nonUnit, SPSV_INDEPENDENT_LEVEL);
#undef SPSV_INDEPENDENT_LEVEL
        return;
    }
    // Very shallow level schedules have little parallel work to amortize the
    // cross-core barrier.  Let block 0 execute the complete schedule with
    // block-local barriers; this also avoids one device-side VF launch per
    // level.  Other blocks exit before touching vecY, so no duplicate writes
    // occur.
    if (numLevels <= 3) {
        if (AscendC::GetBlockIdx() == 0) {
            int64_t m = static_cast<int64_t>(levelPtr[numLevels]);
            SpsvSolveSingleCore<RowPtrT, ColIndT, PermT>(rowPtr, colInd, values, vecX, vecY, yLow, levelPtr, levelRow,
                                                         diagPtr, validCount, alpha, m, numLevels, nthreads, fwd,
                                                         nonUnit);
        }
        return;
    }
    __gm__ const RowPtrT* rp = (__gm__ const RowPtrT*)rowPtr;
    int64_t m = levelPtr[numLevels];
    int64_t nnz = static_cast<int64_t>(rp[m]) - rp[0];
    int32_t lanes = nnz <= 4 * m ? 1 : nnz <= 32 * m ? 4 : 32;
    for (int32_t level = 0; level < numLevels; level++) {
#define SPSV_LEVEL(F, N)                                                                                             \
    asc_vf_call<SpsvSolveLevelSimtCompute<RowPtrT, ColIndT, F, N, PermT>>(                                           \
        dim3{nthreads, 1, 1}, rp, (__gm__ const ColIndT*)colInd, (__gm__ const float*)values,                        \
        (__gm__ const float*)vecX, (__gm__ float*)vecY, yLow, levelPtr, levelRow, diagPtr, validCount, alpha, level, \
        numBlocks, lanes)
        SPSV_DISPATCH_DIRECTION_AND_DIAGONAL(fwd, nonUnit, SPSV_LEVEL);
#undef SPSV_LEVEL

        // All blocks, including those with no rows in this level, join.
        SpsvSyncSolveBlocks();
    }
}

// Dispatch solve based on RowPtrT/ColIndT/PermT and numBlocks (single-core vs multi-core).
#define SPSV_DISPATCH_SOLVE_BLOCKS(RowPtrT, ColIndT, PermT, rowPtr, colInd, values, vecX, vecY, levelPtr, levelRow,  \
                                   diagPtr, validCount, alpha, m, numLevels, nthreads, numBlocks, fwd, nonUnit)      \
    do {                                                                                                             \
        if ((numBlocks) <= 1) {                                                                                      \
            SpsvSolveSingleCore<RowPtrT, ColIndT, PermT>((rowPtr), (colInd), (values), (vecX), (vecY), yLow,         \
                                                         (levelPtr), (levelRow), (diagPtr), (validCount), (alpha),   \
                                                         (m), (numLevels), (nthreads), (fwd), (nonUnit));            \
        } else {                                                                                                     \
            SpsvSolveMultiCore<RowPtrT, ColIndT, PermT>(                                                             \
                (rowPtr), (colInd), (values), (vecX), (vecY), yLow, (levelPtr), (levelRow), (diagPtr), (validCount), \
                (alpha), (numLevels), (nthreads), static_cast<int32_t>(numBlocks), (fwd), (nonUnit));                \
        }                                                                                                            \
    } while (0)

// One complete single-precision solve, single- or multi-block.  Both callees
// leave every block at the same barrier count, so the caller may rendezvous afterward.

template <typename RowPtrT, typename ColIndT, typename PermT>
__aicore__ inline void SpsvSolveFloatChainDispatch(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
                                                   GM_ADDR vecX, GM_ADDR vecY, __gm__ float* yLow,
                                                   __gm__ const int32_t* levelRow,
                                                   __gm__ const PermT* diagPtr, float alpha,
                                                   int32_t numLevels, bool fwd, bool nonUnit)
{
    if (AscendC::GetBlockIdx() == 0) {
#define SPSV_SHADOW_CHAIN(F, N)                                                                      \
    asc_vf_call<SpsvSolveFloatChainCompute<RowPtrT, ColIndT, F, N, PermT>>(                          \
        dim3{32, 1, 1}, (__gm__ const RowPtrT*)rowPtr, (__gm__ const ColIndT*)colInd,                \
        (__gm__ const float*)values, (__gm__ const float*)vecX, (__gm__ float*)vecY, yLow, levelRow, \
        diagPtr, alpha, numLevels)
        SPSV_DISPATCH_DIRECTION_AND_DIAGONAL(fwd, nonUnit, SPSV_SHADOW_CHAIN);
#undef SPSV_SHADOW_CHAIN
    }
}

template <typename RowPtrT, typename ColIndT, typename PermT>
__aicore__ inline void SpsvSolveFloatDispatchPass(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
                                                  GM_ADDR vecX, GM_ADDR vecY, __gm__ float* yLow,
                                                  __gm__ const int32_t* levelPtr,
                                                  __gm__ const int32_t* levelRow, __gm__ const PermT* diagPtr,
                                                  __gm__ const int32_t* validCount, float alpha, int64_t m,
                                                  int32_t numLevels, uint32_t nthreads, uint32_t numBlocks,
                                                  bool fwd, bool nonUnit)
{
    if (numLevels == m) {
        SpsvSolveFloatChainDispatch<RowPtrT, ColIndT, PermT>(
            rowPtr, colInd, values, vecX, vecY, yLow, levelRow, diagPtr, alpha, numLevels, fwd, nonUnit);
        return;
    }
    if (numLevels == 1) {
        yLow = nullptr;
    }
    // How wide the levels actually are, rather than what the host guessed
    // before analysis had run; see the note on ComputeNumBlocks.  A level that
    // cannot fill the blocks turns every extra block into a grid barrier and a
    // launch with nothing to do.
    int32_t meanLevelWidth = (numLevels > 0) ? static_cast<int32_t>(m / static_cast<int64_t>(numLevels)) : 1;
    if (meanLevelWidth >= 2 && meanLevelWidth <= 64 && numLevels > 3) {
        if (AscendC::GetBlockIdx() == 0) {
            __gm__ const RowPtrT* rp = (__gm__ const RowPtrT*)rowPtr;
            if (static_cast<int64_t>(rp[m]) - rp[0] <= 4 * m) {
                SpsvSolveSingleCore<RowPtrT, ColIndT, PermT>(rowPtr, colInd, values, vecX, vecY, yLow,
                                                             levelPtr, levelRow, diagPtr, validCount, alpha,
                                                             m, numLevels, nthreads, fwd, nonUnit);
                return;
            }
            SPSV_DISPATCH_SOLVE(RowPtrT, ColIndT, PermT, SpsvSolveGroupedCompute, rowPtr, colInd, values,
                                vecX, vecY, levelPtr, levelRow, diagPtr, validCount, alpha, m, numLevels, 256,
                                fwd, nonUnit);
        }
        return;
    }
    if (numBlocks > 1 && meanLevelWidth >= kSpsvFloatMultiBlockMinLevelWidth) {
        SpsvSolveMultiCore<RowPtrT, ColIndT, PermT>(rowPtr, colInd, values, vecX, vecY, yLow, levelPtr,
                                                    levelRow, diagPtr, validCount, alpha, numLevels, nthreads,
                                                    static_cast<int32_t>(numBlocks), fwd, nonUnit);
        return;
    }
    if (numBlocks > 1 && AscendC::GetBlockIdx() != 0) {
        // Block 0 owns the whole schedule.  The others take no part and touch
        // no output; the caller's rendezvous after this pass is where they rejoin,
        // and the single-block path takes no cross-core rendezvous of its own, so the counts
        // still match.
        return;
    }
    SpsvSolveSingleCore<RowPtrT, ColIndT, PermT>(rowPtr, colInd, values, vecX, vecY, yLow, levelPtr, levelRow,
                                                 diagPtr, validCount, alpha, m, numLevels, nthreads, fwd,
                                                 nonUnit);
}

#endif // OP_KERNEL_DISPATCH_SPSV_REAL_DISPATCH_H_
