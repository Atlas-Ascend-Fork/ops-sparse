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
#ifndef SPSV_SIMT_COMMON_H_
#define SPSV_SIMT_COMMON_H_

template <typename T>
__simt_callee__ __attribute__((always_inline)) inline void SpsvSimtRefresh(__gm__ const T* input)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)input);
    }
    asc_syncthreads();
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvPublishBlockRisk128(
    uint32_t risk, __gm__ uint32_t* risks, int32_t block)
{
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    risk = asc_reduce_max(risk);
    __ubuf__ uint32_t warpRisk[4];
    if ((tid & 31) == 0) {
        warpRisk[tid / 32] = risk;
    }
    asc_syncthreads();
    if (tid == 0) {
        for (int32_t warp = 1; warp < 4; ++warp) {
            risk |= warpRisk[warp];
        }
        risks[block] = risk;
    }
    asc_threadfence();
}

__simt_callee__ inline int64_t SpsvDyadicActiveRow(int64_t step, int64_t m, bool forward,
                                                 __gm__ const int32_t* grid, int32_t invalid)
{
    int64_t row = forward ? m - 1 - step : step;
    return grid[row] == invalid ? -1 : row;
}

struct SpsvAffineUpEntry {
    int64_t node;
    int32_t row;
    int32_t column;
};

template <int32_t WIDTH>
__simt_callee__ inline SpsvAffineUpEntry SpsvAffineUpPosition(int64_t entry, int64_t first)
{
    return {first + entry / (WIDTH * (WIDTH + 1)),
            static_cast<int32_t>((entry % (WIDTH * (WIDTH + 1))) / (WIDTH + 1)),
            static_cast<int32_t>(entry % (WIDTH + 1))};
}

#endif // SPSV_SIMT_COMMON_H_
