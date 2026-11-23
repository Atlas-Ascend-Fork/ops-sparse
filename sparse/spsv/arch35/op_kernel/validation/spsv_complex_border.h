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

#ifndef SPSV_COMPLEX_BORDER_H_
#define SPSV_COMPLEX_BORDER_H_

#include "../common/spsv_simt_common.h"

#include "spsv_complex_border_defect.h"

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvTailPrepare(const SpsvComplexCertificateData data,
                                                                          int64_t m, int64_t limit, int32_t block,
                                                                          int32_t blocks, __gm__ float* buffer)
{
    SpsvSimtRefresh(buffer);
    int32_t tid = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    int64_t count = limit / WIDTH;
    SpsvPrepareTripleParts<WIDTH>(data, m, count, tid, warp, block, blocks, buffer);
    asc_threadfence();
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_vf__ __aicore__ __launch_bounds__(128) inline void SpsvTailHeadResults(__gm__ const float* tree,
                                                                              __gm__ const float* states,
                                                                              int64_t leaves, int64_t limit,
                                                                              int32_t block, int32_t blocks,
                                                                              __gm__ float* corrections)
{
    SpsvSimtRefresh(tree);
    for (int64_t at = static_cast<int64_t>(block) * 128 + threadIdx.x; at < limit;
         at += static_cast<int64_t>(blocks) * 128) {
        SpsvTriplePrefixStore(corrections + at * 7, SpsvTripleTreeResult<WIDTH>(tree, states, leaves, at));
    }
    asc_threadfence();
}

__simt_callee__ inline int64_t SpsvBorderColumnBegin(const SpsvComplexCertificateData& data, int64_t row,
                                                     int64_t firstColumn)
{
    int64_t lower = data.rowPtr[row], upper = data.rowPtr[row + 1];
    while (lower < upper) {
        int64_t middle = lower + (upper - lower) / 2;
        if (data.colInd[middle] < firstColumn) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return lower;
}

template <int32_t GROUP_ROWS>
__simt_callee__ __attribute__((always_inline)) inline SpsvTriplePrefixValue SpsvBorderInverseDot(
    const SpsvComplexCertificateData& data, int64_t m, int64_t first, int64_t at,
    int64_t group, int32_t tid, __gm__ const float* local)
{
    int64_t row = data.forward ? at : m - 1 - at;
    int64_t firstColumn = data.forward ? first + tid : row + 1;
    int64_t lastColumn = data.forward ? row - 1 : m - 1 - (first + tid);
    int64_t lower = SpsvBorderColumnBegin(data, row, firstColumn);
    auto sum = SpsvTriplePrefixZero();
    for (int64_t p = lower; p < data.rowPtr[row + 1]; ++p) {
        if (data.colInd[p] > lastColumn) {
            break;
        }
        int64_t column = data.colInd[p], source = data.forward ? column : m - 1 - column;
        if (source < first || source >= at) {
            continue;
        }
        auto value = SpsvTriplePrefixLoad(local + ((group * GROUP_ROWS + source - first) * 32 + tid) * 7);
        sum = SpsvTriplePrefixAdd(
            sum, SpsvTriplePrefixMulComplexScalar(-data.values[p].real, -data.values[p].imag, value));
    }
    return sum;
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvTailInverse(const SpsvComplexCertificateData data,
                                                                          int64_t m, int64_t limit, int32_t block,
                                                                          int32_t blocks, __gm__ float* inverses,
                                                                          __gm__ float* defects, __gm__ float* local)
{
    SpsvSimtRefresh(data.wide);
    int32_t tid = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    int64_t groups = (m - limit + 31) / 32;
    for (int64_t first = static_cast<int64_t>(block); first < groups; first += static_cast<int64_t>(blocks) * 8) {
        int64_t group = first + static_cast<int64_t>(warp) * blocks;
        asc_syncthreads();
        if (group >= groups) {
            continue;
        }
        for (int32_t row = 0; row < 32; ++row) {
            int64_t at = limit + 32 * group + row, r = data.forward ? at : m - 1 - at;
            auto sum = SpsvTriplePrefixZero();
            if (at < m && row > tid) {
                sum = SpsvBorderInverseDot<32>(data, m, limit + 32 * group, at, group, tid, local);
                sum = SpsvTriplePrefixMul(sum, SpsvTriplePrefixLoad(inverses + (at - limit) * 7));
            }
            if (row == tid) {
                sum = SpsvTriplePrefixAdd(sum, {{1.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, 0.f});
            }
            SpsvTriplePrefixStore(local + ((group * 32 + row) * 32 + tid) * 7, sum);
        }
    }
    asc_threadfence();
}

__simt_callee__ inline SpsvTriplePrefixValue SpsvTailWarpSum(SpsvTriplePrefixValue z, int32_t lane)
{
    for (int32_t mask = 16; mask > 0; mask /= 2) {
        SpsvTriplePrefixValue other{
            {asc_shfl_xor(z.re.hi, mask), asc_shfl_xor(z.re.lo, mask), asc_shfl_xor(z.re.tail, mask)},
            {asc_shfl_xor(z.im.hi, mask), asc_shfl_xor(z.im.lo, mask), asc_shfl_xor(z.im.tail, mask)},
            asc_shfl_xor(z.radius, mask)};
        if (lane < mask) {
            z = SpsvTriplePrefixAdd(z, other);
        }
    }
    return z;
}
// Each inverse column is an independent triangular recurrence. Assign a
// whole warp to its dot products, then distribute columns over blocks. Only
// this column's already completed rows live in block-local storage.
__simt_callee__ inline void SpsvTailStoreColumnState(__ubuf__ float* state, int32_t row, SpsvTriplePrefixValue sum)
{
    int64_t offset = row * 7;
    state[offset] = sum.re.hi;
    state[offset + 1] = sum.re.lo;
    state[offset + 2] = sum.re.tail;
    state[offset + 3] = sum.im.hi;
    state[offset + 4] = sum.im.lo;
    state[offset + 5] = sum.im.tail;
    state[offset + 6] = sum.radius;
}

__simt_callee__ inline SpsvTriplePrefixValue SpsvTailColumnDot(const SpsvComplexCertificateData& data, int64_t m,
                                                               int64_t first, int32_t column, int64_t at, int32_t lane,
                                                               __ubuf__ const float* state)
{
    int64_t r = data.forward ? at : m - 1 - at;
    auto sum = SpsvTriplePrefixZero();
    int64_t firstColumn = data.forward ? first + column : r + 1;
    int64_t lastColumn = data.forward ? r - 1 : m - 1 - (first + column);
    int64_t lower = SpsvBorderColumnBegin(data, r, firstColumn);
    for (int64_t p = lower + lane; p < data.rowPtr[r + 1]; p += 32) {
        if (data.colInd[p] > lastColumn) {
            break;
        }
        int64_t source = data.forward ? data.colInd[p] : m - 1 - data.colInd[p];
        if (source < first + column || source >= at) {
            continue;
        }
        int64_t offset = (source - first) * 7;
        SpsvTriplePrefixValue v{{state[offset], state[offset + 1], state[offset + 2]},
                                {state[offset + 3], state[offset + 4], state[offset + 5]},
                                state[offset + 6]};
        sum = SpsvTriplePrefixAdd(sum, SpsvTriplePrefixMulComplexScalar(-data.values[p].real, -data.values[p].imag, v));
    }
    return sum;
}

__simt_vf__ __aicore__ __launch_bounds__(32) inline void SpsvTailInverseParallelColumns(
    const SpsvComplexCertificateData data, int64_t m, int64_t limit, int32_t block, int32_t blocks,
    __gm__ const float* inverses, __gm__ float* local)
{
    SpsvSimtRefresh(inverses);
    int32_t lane = static_cast<int32_t>(threadIdx.x);
    int64_t groups = (m - limit + 31) / 32;
    __ubuf__ float state[32 * kSpsvTriplePrefixWords];
    for (int64_t task = block; task < groups * 32; task += blocks) {
        int64_t group = task / 32;
        int32_t column = static_cast<int32_t>(task % 32);
        int64_t first = limit + group * 32;
        for (int32_t row = 0; row < 32; ++row) {
            int64_t at = first + row;
            int64_t r = data.forward ? at : m - 1 - at;
            auto sum = SpsvTriplePrefixZero();
            if (at < m && row > column) {
                sum = SpsvTailColumnDot(data, m, first, column, at, lane, state);
            }
            sum = SpsvTailWarpSum(sum, lane);
            if (lane == 0) {
                if (at < m && row > column) {
                    sum = SpsvTriplePrefixMul(sum, SpsvTriplePrefixLoad(inverses + (at - limit) * 7));
                }
                if (row == column) {
                    sum = SpsvTriplePrefixAdd(sum, {{1.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, 0.f});
                }
                SpsvTailStoreColumnState(state, row, sum);
                SpsvTriplePrefixStore(local + ((group * 32 + row) * 32 + column) * 7, sum);
            }
            asc_syncthreads();
        }
    }
    asc_threadfence();
}

__simt_callee__ inline void SpsvTailApplyConstant(const SpsvComplexCertificateData& data, int64_t m, int64_t limit,
                                                  int64_t group, int32_t row, int32_t lane,
                                                  __gm__ const float* inverses, __gm__ const float* defects,
                                                  __gm__ const float* corrections, __gm__ float* constants)
{
    int64_t at = limit + 32 * group + row, r = data.forward ? at : m - 1 - at;
    auto z = SpsvTriplePrefixZero();
    if (at < m) {
        for (int64_t p = data.rowPtr[r] + lane; p < data.rowPtr[r + 1]; p += 32) {
            int64_t c = data.colInd[p], source = data.forward ? c : m - 1 - c;
            if (source >= limit + 32 * group) {
                continue;
            }
            z = SpsvTriplePrefixAdd(z,
                                    SpsvTriplePrefixMulComplexScalar(-data.values[p].real, -data.values[p].imag,
                                                                     SpsvTriplePrefixLoad(corrections + source * 7)));
        }
    }
    z = SpsvTailWarpSum(z, lane);
    if (lane == 0) {
        if (at < m) {
            z = SpsvTriplePrefixAdd(SpsvTriplePrefixLoad(defects + (at - limit) * 7),
                                    SpsvTriplePrefixMul(z, SpsvTriplePrefixLoad(inverses + (at - limit) * 7)));
        }
        SpsvTriplePrefixStore(constants + (32 * group + row) * 7, z);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvTailApply(
    const SpsvComplexCertificateData data, int64_t m, int64_t limit, int32_t block, __gm__ const float* inverses,
    __gm__ const float* defects, __gm__ const float* local, __gm__ float* constants, __gm__ float* corrections)
{
    if (block != 0) {
        return;
    }
    SpsvSimtRefresh(corrections);
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    int64_t groups = (m - limit + 31) / 32;
    for (int64_t group = 0; group < groups; ++group) {
        for (int32_t row = warp; row < 32; row += 8) {
            SpsvTailApplyConstant(data, m, limit, group, row, lane, inverses, defects, corrections, constants);
        }
        asc_syncthreads();
        for (int32_t row = warp; row < 32; row += 8) {
            int64_t at = limit + 32 * group + row;
            auto z = SpsvTriplePrefixZero();
            if (at < m) {
                z = SpsvTriplePrefixMul(SpsvTriplePrefixLoad(local + ((group * 32 + row) * 32 + lane) * 7),
                                        SpsvTriplePrefixLoad(constants + (32 * group + lane) * 7));
            }
            z = SpsvTailWarpSum(z, lane);
            if (at < m && lane == 0) {
                SpsvTriplePrefixStore(corrections + at * 7, z);
            }
        }
        asc_syncthreads();
    }
    asc_threadfence();
}

__simt_vf__ __aicore__ __launch_bounds__(128) inline void SpsvTailPublish(__gm__ const float* input,
                                                                          __gm__ SpsvComplex64* y, __gm__ float* wide,
                                                                          int64_t m, bool forward, int32_t block,
                                                                          int32_t blocks, __gm__ uint32_t* risks)
{
    SpsvSimtRefresh(input);
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    __gm__ float* prep = wide + 6 * m;
    uint32_t risk = 0U;
    for (int64_t r = static_cast<int64_t>(block) * 128 + tid; r < m; r += static_cast<int64_t>(blocks) * 128) {
        int64_t at = forward ? r : m - 1 - r;
        SpsvTriplePrefixValue z = SpsvTriplePrefixLoad(input + at * 7);
        risk |= SpsvTriplePublishRow(z, y, wide, prep, r);
    }
    SpsvPublishBlockRisk128(risk, risks, block);
}

template <int32_t GROUP_ROWS>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvHeadInverse(const SpsvComplexCertificateData data,
                                                                          int64_t m, int64_t head, int32_t block,
                                                                          int32_t blocks, __gm__ float* inverses,
                                                                          __gm__ float* defects, __gm__ float* local)
{
    SpsvSimtRefresh(data.wide);
    int32_t tid = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    int64_t groups = (head + GROUP_ROWS - 1) / GROUP_ROWS;
    for (int64_t first = static_cast<int64_t>(block); first < groups; first += static_cast<int64_t>(blocks) * 8) {
        int64_t group = first + static_cast<int64_t>(warp) * blocks;
        asc_syncthreads();
        if (group >= groups) {
            continue;
        }
        for (int32_t row = 0; row < GROUP_ROWS; ++row) {
            int64_t at = GROUP_ROWS * group + row, r = data.forward ? at : m - 1 - at;
            auto sum = SpsvTriplePrefixZero();
            if (at < head && row > tid) {
                sum = SpsvBorderInverseDot<GROUP_ROWS>(data, m, GROUP_ROWS * group, at, group, tid, local);
                sum = SpsvTriplePrefixMul(sum, SpsvTriplePrefixLoad(inverses + at * 7));
            }
            if (row == tid) {
                sum = SpsvTriplePrefixAdd(sum, {{1.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, 0.f});
            }
            SpsvTriplePrefixStore(local + ((group * GROUP_ROWS + row) * 32 + tid) * 7, sum);
        }
    }
    asc_threadfence();
}

#endif // SPSV_COMPLEX_BORDER_H_
