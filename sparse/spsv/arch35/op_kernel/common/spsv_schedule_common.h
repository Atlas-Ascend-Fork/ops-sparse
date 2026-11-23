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

#ifndef OP_KERNEL_COMMON_SPSV_SCHEDULE_COMMON_H_
#define OP_KERNEL_COMMON_SPSV_SCHEDULE_COMMON_H_

#include "../spsv_kernel.h"

// Average nonzeros per row below which a matrix is solved by the short-row
// instantiation.  Measured on this part, running a row whole on every lane
// costs about 360 ns per nonzero while the reduction it avoids is worth about
// 500 ns, so only the shortest rows come out ahead -- and the banded shapes,
// with two nonzeros a row, are exactly those.
constexpr int32_t kSpsvWarpRowMinNnz = 4;

// The magnitude above which a warp sum needs room, and the exact powers of
// two it makes room with.  Thirty-two values of at most 2^122 cannot pass
// 2^127 however the tree pairs them up, and 2^-6 brings any finite float
// under 2^122.  Both are powers of two, so scaling by them is exact -- in
// both words of a (hi, lo) pair as well -- and costs a multiply rather than
// an ldexpf call.
constexpr float kSpsvReduceSafeMag = 5.3169119e36f; // 2^122
constexpr float kSpsvReduceDown = 0.015625f;        // 2^-6
constexpr float kSpsvReduceUp = 64.0f;              // 2^6

__aicore__ inline void SpsvSyncSolveBlocks()
{
    // VF runs on PIPE_V. Publish completion only after every SIMT producer
    // has finished; PIPE_S must wait before dispatching the next level's VF.
    AscendC::CrossCoreSetFlag<0, ::PIPE_V>(0);
    AscendC::CrossCoreWaitFlag<0, ::PIPE_S>(0);
}

__simt_callee__ inline void SpsvInvalidateSolveCache(__gm__ void* address)
{
    // A grid barrier orders producers but does not invalidate another AIV's
    // SIMT cache, including cache lines containing not-yet-written neighbours.
    if (threadIdx.x == 0) {
        asc_dcci_entire(address);
    }
    asc_syncthreads();
}

// Width and positive stride are at most INT32_MAX; therefore the terminal
// sum is at most UINT32_MAX - 3. The unsigned cursor stops at the I32 width
// before this out-of-range value is used as a row index.
__simt_callee__ inline uint32_t SpsvNextComplexLevelRow(uint32_t index, int32_t stride)
{
    return index + static_cast<uint32_t>(stride);
}

__aicore__ inline void SpsvResolveComplexSolveCsr(GM_ADDR& rowPtr, GM_ADDR& colInd, GM_ADDR& values, GM_ADDR workspace,
                                                  const SpsvTilingData& tiling)
{
    // Transpose and conjugate transpose consume the effective CSR prepared
    // by analysis and updated on the same stream.
    if (tiling.opA != 0 && tiling.transValuesOffset >= 0) {
        rowPtr = workspace + tiling.transRowPtrOffset;
        colInd = workspace + tiling.transColIndOffset;
        values = workspace + tiling.transValuesOffset;
    }
}

// Return the bounded prefix candidate; callers retain the row-count proof.
__aicore__ inline int32_t SpsvComplexSingletonPrefix(__gm__ const int32_t* levelPtr, int32_t numLevels, int64_t m)
{
    int32_t prefix = 1;
    while (prefix < numLevels && prefix < 64 && static_cast<int64_t>(levelPtr[prefix]) + numLevels - prefix != m) {
        ++prefix;
    }
    return prefix;
}

#endif // OP_KERNEL_COMMON_SPSV_SCHEDULE_COMMON_H_
