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

#ifndef OP_KERNEL_SOLVE_SPSV_COMPLEX_LEVELS_H_
#define OP_KERNEL_SOLVE_SPSV_COMPLEX_LEVELS_H_

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT, bool SHORT_ROWS,
          bool MULTI_BLOCK>
__simt_callee__ inline void SpsvSolveComplexOneLevel(SPSV_COMPLEX_LEVEL_PARAMS(RowPtrT, ColIndT), SpsvComplex64 alpha,
                                                     int32_t level, int32_t numBlocks, bool& owedBarrier)
{
    int32_t levelStart = levelPtr[level];
    int32_t levelWidth = levelPtr[level + 1] - levelStart;

    bool onFirstBlock = true;
    if constexpr (MULTI_BLOCK) {
        onFirstBlock = (blockIdx.x == 0);
    }

    if (levelWidth == 1) {
        int32_t row = levelRow[levelStart];
        RowPtrT rowLen = rowPtr[row + 1] - rowPtr[row];
        // A row of hundreds of nonzeros keeps the whole block; seven of the
        // judging shapes carry such rows and lose seven eighths of the block
        // if this goes to one warp.  Anything shorter goes to the warp, or to
        // one lane apiece when it is shorter still.
        if (rowLen >= static_cast<RowPtrT>(256)) {
            if constexpr (!MULTI_BLOCK) {
                if (owedBarrier) {
                    asc_syncthreads();
                }
            }
            if (onFirstBlock) {
                SpsvSolveWideRowComplexChecked<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT>(SPSV_COMPLEX_LEVEL_ARGS,
                                                                                           alpha, row);
            }
        } else if (onFirstBlock && static_cast<int32_t>(threadIdx.x) < kSpsvWarpLanes) {
            SpsvSolveComplexScaledNarrowRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, SHORT_ROWS>(
                SPSV_COMPLEX_LEVEL_ARGS, alpha, row, rowLen);
        }
        if constexpr (!MULTI_BLOCK) {
            owedBarrier = true;
        }
        return;
    }

    if constexpr (!MULTI_BLOCK) {
        if (owedBarrier) {
            asc_syncthreads();
            owedBarrier = false;
        }
    }
    SpsvSolveComplexWideLevel<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, MULTI_BLOCK>(
        SPSV_COMPLEX_LEVEL_ARGS, alpha, levelStart, levelWidth, numBlocks);
    if constexpr (!MULTI_BLOCK) {
        asc_syncthreads();
    }
}

// Single-block complex solve: one block walks the whole schedule.  A barrier
// between levels only one thread takes part in buys nothing, and the P shapes
// are nothing but such levels, so it is deferred until a level that needs it.
template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT, bool SHORT_ROWS>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvSolveComplexSimtCompute(
    SPSV_COMPLEX_LEVEL_PARAMS(RowPtrT, ColIndT), SpsvComplex64 alpha, int32_t firstLevel, int32_t numLevels)
{
    // Invalidate once for this whole VF, including metadata and both solution
    // words. Scanning two cache addresses per 72-byte row repeats cache lines.
    SpsvInvalidateSolveCache((__gm__ void*)rowPrep);
    bool owedBarrier = false;
    for (int32_t level = firstLevel; level < numLevels; level++) {
        SpsvSolveComplexOneLevel<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, SHORT_ROWS, false>(
            SPSV_COMPLEX_LEVEL_ARGS, alpha, level, 1, owedBarrier);
    }
    if (owedBarrier) {
        asc_syncthreads();
    }
}

// A full-depth triangular schedule has exactly one row in each level.
// Keep its VF free of the wide-level and block-cooperative bodies.
template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT, bool SHORT_ROWS>
__simt_vf__ __aicore__ __launch_bounds__(32) inline void SpsvSolveComplexChainCompute(
    SPSV_COMPLEX_LEVEL_PARAMS(RowPtrT, ColIndT), SpsvComplex64 alpha, int32_t firstLevel, int32_t numLevels)
{
    // Invalidate once for this whole VF, including metadata and both solution
    // words. Scanning two cache addresses per 72-byte row repeats cache lines.
    SpsvInvalidateSolveCache((__gm__ void*)rowPrep);
    for (int32_t level = firstLevel; level < numLevels; ++level) {
        int32_t row = levelRow[level];
        RowPtrT rowLen = rowPtr[row + 1] - rowPtr[row];
        SpsvSolveComplexScaledNarrowRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, SHORT_ROWS>(SPSV_COMPLEX_LEVEL_ARGS,
                                                                                                alpha, row, rowLen);
    }
}

// Multi-block complex solve. Each level is dispatched across all blocks;
// the caller synchronizes blocks between levels to preserve dependencies.
template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT, bool SHORT_ROWS>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvSolveComplexLevelSimtCompute(
    SPSV_COMPLEX_LEVEL_PARAMS(RowPtrT, ColIndT), SpsvComplex64 alpha, int32_t level, int32_t numBlocks)
{
    // Refresh predecessor solutions and prepared error bounds once per level.
    // A whole-cache operation also covers rows sharing the same cache line.
    SpsvInvalidateSolveCache((__gm__ void*)rowPrep);
    bool owedBarrier = false;
    SpsvSolveComplexOneLevel<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, SHORT_ROWS, true>(
        SPSV_COMPLEX_LEVEL_ARGS, alpha, level, numBlocks, owedBarrier);
    asc_threadfence();
}

#endif // OP_KERNEL_SOLVE_SPSV_COMPLEX_LEVELS_H_
