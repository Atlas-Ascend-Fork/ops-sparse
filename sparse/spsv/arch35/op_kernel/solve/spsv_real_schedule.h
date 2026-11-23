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

#ifndef OP_KERNEL_SOLVE_SPSV_REAL_SCHEDULE_H_
#define OP_KERNEL_SOLVE_SPSV_REAL_SCHEDULE_H_

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT,
          typename PermT, int LANES>
__simt_callee__ inline void SpsvSolveFloatScheduledRow(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX, __gm__ float *vecY,
    __gm__ float *yLow, __gm__ const PermT *diagPtr, float alpha, int32_t row)
{
    if (rowPtr[row + 1] - rowPtr[row] <= 4) {
        if (static_cast<int32_t>(threadIdx.x) % LANES == 0) {
            // Short rows need only the conservative low-word magnitude
            // enclosure. Keep the same dependency bound and exact retry.
            SpsvSolveFloatShadowRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, 1, false>(
                rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, alpha, row);
        }
    } else {
        SpsvSolveFloatShadowRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, LANES>(
            rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, alpha, row);
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_callee__ inline void SpsvSolveWideRowChecked(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX, __gm__ float *vecY,
    __gm__ float *yLow, __gm__ const PermT *diagPtr, float alpha, int32_t row)
{
    // Hundreds of threads already shorten the dot-product chain. The cheaper,
    // conservative certificate is sufficient here; preserve the same retry
    // thresholds and exact fallback when that certificate cannot certify a row.
    __ubuf__ float partials[64 * 5];
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    int32_t threads = static_cast<int32_t>(blockDim.x);
    int32_t lane = tid & 31, warp = tid / 32, warps = threads / 32;
    SpsvDF sum{0.0f, 0.0f};
    float error = 0.0f, predecessorError = 0.0f;
    uint32_t flags = 0;
    if (tid == 0) {
        SpsvAccEnclosed(sum.hi, sum.lo, error, alpha, vecX[row], 0.0f);
        flags |= SpsvFloatOperandFlags(alpha, vecX[row], 0.0f);
    }
    for (SpsvCsrCursor<RowPtrT> q = static_cast<SpsvCsrCursor<RowPtrT>>(rowPtr[row]) + tid;
         q < rowPtr[row + 1]; q += threads) {
        int32_t col = static_cast<int32_t>(colInd[q]);
        if (FORWARD ? col < row : col > row) {
            float a = values[q], y = vecY[col], low = yLow[2 * static_cast<int64_t>(col)];
            flags |= SpsvFloatOperandFlags(a, y, low);
            predecessorError = SpsvFloatDependencyAcc(predecessorError,
                SpsvAbsF(a), yLow[2 * static_cast<int64_t>(col) + 1]);
            SpsvAccEnclosed(sum.hi, sum.lo, error, -a, y, low);
        }
    }
    sum = SpsvReduceFloatShadow<32, false>(sum, error, predecessorError, flags);
    if (lane == 0) {
        partials[warp * 5] = sum.hi; partials[warp * 5 + 1] = sum.lo;
        partials[warp * 5 + 4] = predecessorError;
        partials[warp * 5 + 2] = error; partials[warp * 5 + 3] = static_cast<float>(flags);
    }
    asc_syncthreads();
    if (warp == 0) {
        sum = {0.0f, 0.0f}; error = 0.0f; predecessorError = 0.0f; flags = 0;
        for (int32_t w = lane; w < warps; w += 32) {
            sum = SpsvMergeEnclosed(sum, {partials[w * 5], partials[w * 5 + 1]}, error);
            error += partials[w * 5 + 2];
            predecessorError += partials[w * 5 + 4];
            flags |= static_cast<uint32_t>(partials[w * 5 + 3]);
        }
        sum = SpsvReduceFloatShadow<32, false>(sum, error, predecessorError, flags);
        float diagonal = NON_UNIT ? (diagPtr[row] >= 0 ? values[diagPtr[row]] : 0.0f) : 1.0f;
        flags |= SpsvFloatOperandFlags(diagonal, 0.0f, 0.0f);
        SpsvFinishFloatShadow<RowPtrT, ColIndT, FORWARD, 32>(
            rowPtr, colInd, values, vecX, vecY, yLow, alpha, row, diagonal, sum, error, predecessorError, flags);
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_callee__ inline void SpsvSolveRowWarpChecked(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX, __gm__ float *vecY,
    __gm__ float *yLow, __gm__ const PermT *diagPtr, __gm__ const int32_t *validCount,
    float alpha, int32_t row, RowPtrT rowLen)
{
    SpsvSolveFloatShadowRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, 32>(
        rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, alpha, row);
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_callee__ inline void SpsvSolveRowChecked(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX, __gm__ float *vecY,
    __gm__ float *yLow, __gm__ const PermT *diagPtr, __gm__ const int32_t *validCount,
    float alpha, int32_t row)
{
    SpsvSolveFloatScheduledRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, 1>(
        rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, alpha, row);
}

template <typename RowPtrT, typename ColIndT, bool FORWARD>
__simt_callee__ __attribute__((always_inline)) inline void SpsvAccumulateFloatChain(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX, __gm__ const float *vecY,
    __gm__ const float *yLow, int32_t row, RowPtrT start, RowPtrT end,
    int32_t lane, int32_t stride, bool serial, float alpha, SpsvDF &sum,
    float &error, float &predecessorError, uint32_t &flags)
{
    if (lane == 0) {
        if (serial) { SpsvAccEnclosed(sum.hi, sum.lo, error, alpha, vecX[row], 0.0f); }
        else { SpsvAccFloatEnclosed(sum.hi, sum.lo, error, alpha, vecX[row], 0.0f); }
        flags |= SpsvFloatOperandFlags(alpha, vecX[row], 0.0f);
    }
    for (SpsvCsrCursor<RowPtrT> p = static_cast<SpsvCsrCursor<RowPtrT>>(start) + lane; p < end; p += stride) {
        int32_t col = static_cast<int32_t>(colInd[p]);
        if (FORWARD ? col < row : col > row) {
            float a = values[p], y = vecY[col], low = yLow[2 * static_cast<int64_t>(col)];
            flags |= SpsvFloatOperandFlags(a, y, low);
            predecessorError = SpsvFloatDependencyAcc(predecessorError,
            SpsvAbsF(a), yLow[2 * static_cast<int64_t>(col) + 1]);
            if (serial) { SpsvAccEnclosed(sum.hi, sum.lo, error, -a, y, low); }
            else { SpsvAccFloatEnclosed(sum.hi, sum.lo, error, -a, y, low); }
        }
    }
    if (!serial) {
        flags = asc_reduce_max(flags);
        int32_t width = end - start <= 4 ? 4 : end - start <= 8 ? 8 : end - start <= 16 ? 16 : 32;
        sum = SpsvWarpEnclosed(sum, error, width);
        for (int32_t mask = 1; mask < 32; mask <<= 1) {
            predecessorError += asc_shfl_xor(predecessorError, mask);
        }
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD>
__simt_callee__ __attribute__((always_inline)) inline void SpsvFinishFloatChainRow(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX, __gm__ float *vecY,
    __gm__ float *yLow, int32_t row, RowPtrT start, RowPtrT end,
    float diagonal, float alpha, float predecessorError,
    SpsvDF sum, float error, uint32_t flags)
{
    uint32_t db;
    __builtin_memcpy(&db, &diagonal, sizeof(db));
    db &= 0x7fffffffU;
    flags |= (db - 1U < 0x007fffffU) ? 2U : 0U;
    SpsvDF answer = SpsvDFDivF(SpsvTwoSum(sum.hi, sum.lo), diagonal);
    float bound = (error * 0x1p-21f + static_cast<float>(static_cast<int64_t>(end) - start + 1) * 0x1p-120f) /
        SpsvAbsF(diagonal) + SpsvAbsF(answer.hi) * 0x1p-45f;
    SpsvDF originalAnswer = answer;
    float budget = SpsvAbsF(answer.hi) * 0x1p-32f;
    budget = budget > 0x1p-45f ? budget : 0x1p-45f;
    bool risk = flags || end - start > (1 << 20) || !SpsvIsFiniteF(bound) ||
        !SpsvIsFiniteF(answer.hi) || !SpsvIsFiniteF(answer.lo) || bound > budget;
    if ((flags & 1U) || !SpsvIsFiniteF(diagonal) || diagonal == 0.0f) {
        // Preserve IEEE propagation for non-finite operands/singular rows.
        answer = {SpsvPropagateFloatRow<RowPtrT, ColIndT, FORWARD>(
            rowPtr, colInd, values, vecX, vecY, row, alpha, diagonal), 0.0f};
    } else if (risk) {
        if (static_cast<int32_t>(threadIdx.x) == 0) {
            answer = SpsvFloatShadowExact<RowPtrT, ColIndT, FORWARD>(
                rowPtr, colInd, values, vecX, vecY, yLow, row, alpha, diagonal);
        }
        answer.hi = asc_shfl(answer.hi, 0);
        answer.lo = asc_shfl(answer.lo, 0);
    }
    vecY[row] = answer.hi;
    yLow[2 * static_cast<int64_t>(row)] = answer.lo;
    float displacement = SpsvFloatBoundAdd(SpsvAbsF(originalAnswer.hi - answer.hi),
        SpsvAbsF(originalAnswer.lo - answer.lo));
    float enclosure = SpsvFloatBoundAdd(SpsvFloatBoundAdd(bound, displacement),
        SpsvFloatBoundDiv(SpsvFloatDependencyBound(predecessorError,
                static_cast<int64_t>(rowPtr[row + 1]) - rowPtr[row]), SpsvAbsF(diagonal)));
    yLow[2 * static_cast<int64_t>(row) + 1] = (flags & 2U) == 0 && SpsvIsFiniteF(enclosure) &&
        static_cast<int64_t>(end) - start <= (1 << 20) ? enclosure : __builtin_inff();
}


// A two-word candidate is never accepted by itself: every radius is infinite.
// A later original-equation certificate or unchanged exact retry proves it.
template<typename RowPtrT,typename ColIndT,bool FORWARD,bool NON_UNIT,typename PermT>
__simt_callee__ inline void SpsvFloatCandidateChain(
    __gm__ const RowPtrT*rp,__gm__ const ColIndT*ci,__gm__ const float*a,
    __gm__ const float*x,__gm__ float*y,__gm__ float*state,
    __gm__ const int32_t*lr,__gm__ const PermT*dp,float alpha,int32_t levels)
{
    int32_t lane=static_cast<int32_t>(threadIdx.x);
    for(int32_t level=0;level<levels;++level){
        int32_t row=lr[level];RowPtrT first=rp[row],end=rp[row+1];
        SpsvDF sum=lane==0?SpsvTwoProd(alpha,x[row]):SpsvDF{0.f,0.f};
        for(SpsvCsrCursor<RowPtrT>p=static_cast<SpsvCsrCursor<RowPtrT>>(first)+lane;p<end;p+=32){
            int64_t col=ci[p];
            if(FORWARD?col<row:col>row){
                auto product=SpsvFastMul(SpsvDF{y[col],state[2*col]},-a[p]);
                sum=SpsvFastAdd(sum,product);
            }
        }
        int32_t width=end-first<=4?4:end-first<=8?8:end-first<=16?16:32;
        for(int32_t distance=1;distance<width;distance*=2){
            auto high=SpsvTwoSum(sum.hi,asc_shfl_xor(sum.hi,distance));
            sum.lo=(sum.lo+asc_shfl_xor(sum.lo,distance))+high.lo;sum.hi=high.hi;
        }
        if(lane==0){
            auto answer=SpsvTwoSum(sum.hi,sum.lo);
            if constexpr(NON_UNIT)answer=SpsvDFDivF(answer,dp[row]>=0?a[dp[row]]:0.f);
            y[row]=answer.hi;state[2*static_cast<int64_t>(row)]=answer.lo;
            state[2*static_cast<int64_t>(row)+1]=__builtin_inff();
        }
    }
    asc_threadfence();
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(32) inline void SpsvSolveFloatChainCompute(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX, __gm__ float *vecY,
    __gm__ float *yLow, __gm__ const int32_t *levelRow,
    __gm__ const PermT *diagPtr, float alpha, int32_t numLevels)
{
    if(numLevels>=128&&static_cast<int64_t>(rowPtr[numLevels])-rowPtr[0]<=12LL*numLevels){
        SpsvFloatCandidateChain<RowPtrT,ColIndT,FORWARD,NON_UNIT,PermT>(
            rowPtr,colInd,values,vecX,vecY,yLow,levelRow,diagPtr,alpha,numLevels);
        return;
    }

    for (int32_t level = 0; level < numLevels; ++level) {
        int32_t row = levelRow[level];
        RowPtrT start = rowPtr[row], end = rowPtr[row + 1];
        bool serial = end - start < 4;
        int32_t lane = serial ? 0 : static_cast<int32_t>(threadIdx.x);
        int32_t stride = serial ? 1 : 32;
        SpsvDF sum{0.0f, 0.0f};
        float error = 0.0f, predecessorError = 0.0f;
        uint32_t flags = 0;
        SpsvAccumulateFloatChain<RowPtrT, ColIndT, FORWARD>(rowPtr, colInd, values,
            vecX, vecY, yLow, row, start, end, lane, stride, serial, alpha,
            sum, error, predecessorError, flags);
        float diagonal = NON_UNIT ? (diagPtr[row] >= 0 ? values[diagPtr[row]] : 0.0f) : 1.0f;
        SpsvFinishFloatChainRow<RowPtrT, ColIndT, FORWARD>(rowPtr, colInd, values,
            vecX, vecY, yLow, row, start, end, diagonal,
            alpha, predecessorError, sum, error, flags);
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_callee__ __attribute__((always_inline)) inline void SpsvSolveFloatSingleton(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX, __gm__ float *vecY,
    __gm__ float *yLow, __gm__ const int32_t *levelRow,
    __gm__ const PermT *diagPtr, __gm__ const int32_t *validCount,
    float alpha, int32_t levelStart, bool &owedBarrier)
{
    int32_t wideRow = levelRow[levelStart];
    RowPtrT wideLen = rowPtr[wideRow + 1] - rowPtr[wideRow];
    if (wideLen >= static_cast<RowPtrT>(256)) {
        // The cooperative path reads Y across the whole block, so it
        // has to see the deferred writes first.
        if (owedBarrier) {
            asc_syncthreads();
        }
        SpsvSolveWideRowChecked<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT>(
            rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, alpha, wideRow);
        owedBarrier = true;
        return;
    }
    // One row, one warp: its lanes read back their own writes, and
    // the other warp has no stake until a wider level arrives.
    if (static_cast<int32_t>(threadIdx.x) < kSpsvWarpLanes) {
        SpsvSolveRowWarpChecked<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT>(
            rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, validCount, alpha,
            wideRow, wideLen);
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT, bool GROUPED>
__simt_callee__ __attribute__((always_inline)) inline void SpsvSolveFloatSchedule(
    __gm__ const RowPtrT *rowPtr,
    __gm__ const ColIndT *colInd,
    __gm__ const float *values,
    __gm__ const float *vecX,
    __gm__ float *vecY,
    __gm__ float *yLow,
    __gm__ const int32_t *levelPtr,
    __gm__ const int32_t *levelRow,
    __gm__ const PermT *diagPtr,
    __gm__ const int32_t *validCount,
    float alpha, int64_t m, int32_t numLevels)
{
    // Set while thread 0 has written rows the rest of the block has not yet
    // synchronised on.  Every thread reaches the same value from the same level
    // widths, so the block never disagrees about where a barrier falls.
    bool owedBarrier = false;
    for (int32_t level = 0; level < numLevels; level++) {
        int32_t levelStart = levelPtr[level];
        int32_t levelEnd = levelPtr[level + 1];
        int32_t levelWidth = levelEnd - levelStart;

        if (levelWidth == 1) {
            SpsvSolveFloatSingleton<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT>(
                rowPtr, colInd, values, vecX, vecY, yLow, levelRow,
                diagPtr, validCount, alpha, levelStart, owedBarrier);
            owedBarrier = true;
            continue;
        }

        if (owedBarrier) {
            asc_syncthreads();
            owedBarrier = false;
        }
        for (int64_t idx = static_cast<int64_t>(threadIdx.x) / (GROUPED ? 8 : 1);
             idx < levelWidth;
             idx += static_cast<int64_t>(blockDim.x) / (GROUPED ? 8 : 1)) {
            int32_t row = levelRow[levelStart + idx];
            if constexpr (GROUPED) {
                SpsvSolveFloatShadowRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, 8>(
                    rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, alpha, row);
            } else {
                SpsvSolveRowChecked<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT>(
                    rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, validCount, alpha, row);
            }
        }

        asc_syncthreads();
    }
    if (owedBarrier) {
        asc_syncthreads();
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvSolveSimtCompute(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX, __gm__ float *vecY,
    __gm__ float *yLow, __gm__ const int32_t *levelPtr, __gm__ const int32_t *levelRow,
    __gm__ const PermT *diagPtr, __gm__ const int32_t *validCount,
    float alpha, int64_t m, int32_t numLevels)
{
    SpsvSolveFloatSchedule<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, false>(
        rowPtr, colInd, values, vecX, vecY, yLow, levelPtr, levelRow,
        diagPtr, validCount, alpha, m, numLevels);
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvSolveGroupedCompute(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX, __gm__ float *vecY,
    __gm__ float *yLow, __gm__ const int32_t *levelPtr, __gm__ const int32_t *levelRow,
    __gm__ const PermT *diagPtr, __gm__ const int32_t *validCount,
    float alpha, int64_t m, int32_t numLevels)
{
    SpsvSolveFloatSchedule<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, true>(
        rowPtr, colInd, values, vecX, vecY, yLow, levelPtr, levelRow,
        diagPtr, validCount, alpha, m, numLevels);
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_callee__ __attribute__((always_inline)) inline void SpsvSolveFloatSingletonLevel(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX, __gm__ float *vecY,
    __gm__ float *yLow, __gm__ const int32_t *levelRow,
    __gm__ const PermT *diagPtr, __gm__ const int32_t *validCount,
    float alpha, int32_t levelStart)
{
    int32_t wideRow = levelRow[levelStart];
    RowPtrT wideLen = rowPtr[wideRow + 1] - rowPtr[wideRow];
    if (wideLen >= static_cast<RowPtrT>(256)) {
        if (blockIdx.x == 0) {
            SpsvSolveWideRowChecked<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT>(
                rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, alpha, wideRow);
        }
        asc_threadfence();
        return;
    }
    // One row: block 0's first warp takes it, rather than the one thread
    // the strided loop below would have left it to.  The caller's cross-core rendezvous
    // publishes the result to every other block.
    if (blockIdx.x == 0 && static_cast<int32_t>(threadIdx.x) < kSpsvWarpLanes) {
        SpsvSolveRowWarpChecked<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT>(
            rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, validCount, alpha,
            wideRow, wideLen);
    }
    asc_threadfence();
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvSolveLevelSimtCompute(
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd,
    __gm__ const float *values, __gm__ const float *vecX,
    __gm__ float *vecY, __gm__ float *yLow,
    __gm__ const int32_t *levelPtr, __gm__ const int32_t *levelRow,
    __gm__ const PermT *diagPtr, __gm__ const int32_t *validCount,
    float alpha, int32_t level, int32_t numBlocks, int32_t lanes)
{
    // The preceding grid rendezvous publishes both solution words. Invalidate
    // once per block instead of issuing two cache instructions per producer row.
    SpsvInvalidateSolveCache((__gm__ void *)vecY);
    int32_t globalTid = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) +
                        static_cast<int32_t>(threadIdx.x);
    int32_t globalStride = numBlocks * static_cast<int32_t>(blockDim.x);

    int32_t levelStart = levelPtr[level];
    int32_t levelEnd = levelPtr[level + 1];
    int32_t levelWidth = levelEnd - levelStart;

    if (levelWidth == 1) {
        SpsvSolveFloatSingletonLevel<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT>(
            rowPtr, colInd, values, vecX, vecY, yLow, levelRow,
            diagPtr, validCount, alpha, levelStart);
        return;
    }

    for (int64_t idx = globalTid / lanes; idx < levelWidth; idx += globalStride / lanes) {
        int32_t row = levelRow[levelStart + idx];
        if (lanes == 1) {
            SpsvSolveFloatScheduledRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, 1>(
                rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, alpha, row);
        } else if (lanes == 4) {
            SpsvSolveFloatShadowRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, 4>(
                rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, alpha, row);
        } else {
            SpsvSolveFloatScheduledRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, PermT, 32>(
                rowPtr, colInd, values, vecX, vecY, yLow, diagPtr, alpha, row);
        }
    }
    // Publish BOTH solution words before returning to the cross-core
    // rendezvous. A control barrier alone does not drain SIMT GM stores.
    asc_threadfence();
}

#endif // OP_KERNEL_SOLVE_SPSV_REAL_SCHEDULE_H_
