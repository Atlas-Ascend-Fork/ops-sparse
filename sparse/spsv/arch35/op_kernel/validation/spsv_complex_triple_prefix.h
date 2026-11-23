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

// A three-word, componentwise enclosure of the signed shadow correction.
// Only finite, bounded-bandwidth graphs can use this path; any unsupported
// graph or unsuccessful certificate retains dyadic/CRT reconstruction.
// Tree composition evaluates every affine matrix only once, followed by a
// boundary-state downsweep. Original rounding-cell/budget tests are unchanged.
#ifndef SPSV_COMPLEX_TRIPLE_PREFIX_H_
#define SPSV_COMPLEX_TRIPLE_PREFIX_H_

#include "../common/spsv_simt_common.h"
#include "spsv_complex_triple_arith.h"
template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline int64_t SpsvTriplePrefixAddress(int64_t block, int32_t row, int32_t column)
{
    return (block * (WIDTH * (WIDTH + 1)) + row * (WIDTH + 1) + column) * kSpsvTriplePrefixWords;
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline uint32_t SpsvTriplePrefixValidateRows(const SpsvComplexCertificateData& data, int64_t m,
                                                             int32_t block, int32_t blocks, int32_t tid)
{
    uint32_t bad = 0U;
    // Validate the complete input graph independently of any numerical
    // cancellation in the proposed affine transforms.
    for (int64_t r = static_cast<int64_t>(block) * 32 + tid; r < m; r += static_cast<int64_t>(blocks) * 32) {
        bad |= !SpsvCertificateRowFinite(data, r, 64);
        if (data.nonUnit) {
            SpsvTriplePrefixValue inverse = SpsvTriplePrefixInverse(data, m, r);
            bad |= data.diagPtr[r] < 0 || !(inverse.radius >= 0.f) || !SpsvBlockFiniteNormal(inverse.re.hi) ||
                   !SpsvBlockFiniteNormal(inverse.re.lo) || !SpsvBlockFiniteNormal(inverse.im.hi) ||
                   !SpsvBlockFiniteNormal(inverse.im.lo) || !SpsvIsFiniteF(inverse.radius);
        }
        for (int64_t p = data.rowPtr[r]; p < data.rowPtr[r + 1]; ++p) {
            int64_t distance = data.forward ? r - data.colInd[p] : static_cast<int64_t>(data.colInd[p]) - r;
            bad |= distance > WIDTH || !SpsvBlockFiniteNormal(data.values[p].real) ||
                   !SpsvBlockFiniteNormal(data.values[p].imag);
        }
    }
    return bad;
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline void SpsvTriplePrefixPrepareDefects(const SpsvComplexCertificateData& data, int64_t m,
                                                           int64_t part, int32_t tid, __gm__ float* buffer,
                                                           __gm__ float* inverses)
{
    // Each residual depends only on the immutable input and solved shadow,
    // so compute the block's row defects in parallel before the affine
    // recurrence. The block barrier publishes them to the constant lane.
    if (tid < WIDTH) {
        int64_t logical = part * WIDTH + tid;
        int64_t r = data.forward ? logical : m - 1 - logical;
        SpsvTriplePrefixValue defectValue = SpsvTriplePrefixZero();
        if (logical < m) {
            SpsvTriplePrefixValue verified = SpsvTriplePrefixVerifiedInverse(data, m, r);
            SpsvTriplePrefixStore(inverses + logical * kSpsvTriplePrefixWords, verified);
            float defect[7];
            if (SpsvTriplePrefixTightDefect(data, r, defect)) {
                defectValue = {{defect[0], defect[1], defect[2]}, {defect[3], defect[4], defect[5]}, defect[6]};
                if (data.nonUnit) {
                    defectValue = SpsvTriplePrefixMul(defectValue, verified);
                }
            } else {
                defectValue.radius = __builtin_inff();
            }
        }
        SpsvTriplePrefixStore(buffer + SpsvTriplePrefixAddress<WIDTH>(part, tid, WIDTH), defectValue);
    }
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline SpsvTriplePrefixValue SpsvTriplePrefixPrepareRow(const SpsvComplexCertificateData& data,
                                                                        int64_t m, int64_t part, int32_t row,
                                                                        int32_t tid, __gm__ float* buffer,
                                                                        __gm__ float* inverses)
{
    int64_t logical = part * WIDTH + row;
    int64_t r = data.forward ? logical : m - 1 - logical;
    SpsvTriplePrefixValue z = SpsvTriplePrefixZero();
    if (logical < m) {
        if (tid == WIDTH) {
            z = SpsvTriplePrefixLoad(buffer + SpsvTriplePrefixAddress<WIDTH>(part, row, WIDTH));
        }
        SpsvTriplePrefixValue dependencies = data.nonUnit ? SpsvTriplePrefixZero() : z;
        for (int64_t p = data.rowPtr[r]; p < data.rowPtr[r + 1]; ++p) {
            int64_t c = data.colInd[p], at = data.forward ? c : m - 1 - c;
            if (at >= logical) {
                continue;
            }
            SpsvTriplePrefixValue source = SpsvTriplePrefixZero();
            if (at >= part * WIDTH) {
                source = SpsvTriplePrefixLoad(
                    buffer + SpsvTriplePrefixAddress<WIDTH>(part, static_cast<int32_t>(at - part * WIDTH), tid));
            } else if (at - (part - 1) * WIDTH == tid) {
                source.re.hi = 1.f;
            } else if (at < (part - 1) * WIDTH) {
                z.radius = __builtin_inff();
            }
            dependencies = SpsvTriplePrefixAdd(
                dependencies,
                SpsvTriplePrefixMul({{-data.values[p].real, 0.f}, {-data.values[p].imag, 0.f}, 0.f}, source));
        }
        if (data.nonUnit) {
            dependencies = SpsvTriplePrefixMul(dependencies,
                                               SpsvTriplePrefixLoad(inverses + logical * kSpsvTriplePrefixWords));
            z = SpsvTriplePrefixAdd(z, dependencies);
        } else {
            z = dependencies;
        }
    }
    return z;
}

template <int32_t WIDTH>
__simt_callee__ __attribute__((always_inline)) inline void SpsvPrepareTripleParts(
    const SpsvComplexCertificateData& data, int64_t m, int64_t count, int32_t tid, int32_t warp,
    int32_t block, int32_t blocks, __gm__ float* buffer)
{
    for (int64_t first = static_cast<int64_t>(block) * 8; first < count; first += static_cast<int64_t>(blocks) * 8) {
        int64_t part = first + warp;
        if (part < count) {
            SpsvTriplePrefixPrepareDefects<WIDTH>(data, m, part, tid, buffer,
                                                  buffer + count * (WIDTH * (WIDTH + 1)) * kSpsvTriplePrefixWords);
        }
        asc_syncthreads();
        if (part >= count || tid >= (WIDTH + 1)) {
            continue;
        }
        for (int32_t row = 0; row < WIDTH; ++row) {
            SpsvTriplePrefixValue z = SpsvTriplePrefixPrepareRow<WIDTH>(
                data, m, part, row, tid, buffer, buffer + count * (WIDTH * (WIDTH + 1)) * kSpsvTriplePrefixWords);
            SpsvTriplePrefixStore(buffer + SpsvTriplePrefixAddress<WIDTH>(part, row, tid), z);
        }
    }
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvTriplePrefixPrepare(
    __gm__ const int32_t* rp, __gm__ const int32_t* ci, __gm__ const SpsvComplex64* a, __gm__ const SpsvComplex64* rhs,
    __gm__ const SpsvComplex64* y, __gm__ const int32_t* dp, __gm__ float* wide, SpsvComplex64 alpha, int64_t m,
    bool forward, bool nonUnit, int32_t block, int32_t blocks, __gm__ float* buffer, __gm__ uint32_t* invalid)
{
    SpsvSimtRefresh(wide);
    int32_t tid = static_cast<int32_t>(threadIdx.x) % 32;
    // Each warp owns one part. All warps participate in the uniform block
    // barriers, including a partial final group of parts.
    int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
    uint32_t bad = 0U;
    SpsvComplexCertificateData data{rp, ci, a, rhs, dp, y, wide, alpha, forward, nonUnit};
    bad = SpsvTriplePrefixValidateRows<WIDTH>(data, m, block, blocks, tid);
    bad = asc_reduce_max(bad);
    if (threadIdx.x == 0) {
        invalid[block] = bad;
    }
    int64_t count = (m + WIDTH - 1) / WIDTH;
    SpsvPrepareTripleParts<WIDTH>(data, m, count, tid, warp, block, blocks, buffer);
    asc_threadfence();
}

// Work-efficient affine scan. A tree node represents right(left(boundary)).
// Padding leaves are identities and are never read from uninitialized memory.
template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline SpsvTriplePrefixValue SpsvTripleTreeLoad(__gm__ const float* tree, int64_t node, int32_t row,
                                                                int32_t column, int64_t leaves, int64_t count)
{
    if (node >= leaves + count) {
        SpsvTriplePrefixValue z = SpsvTriplePrefixZero();
        if (row == column) {
            z.re.hi = 1.f;
        }
        return z;
    }
    return SpsvTriplePrefixLoad(tree + SpsvTriplePrefixAddress<WIDTH>(node, row, column));
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth, bool SHORT_PRODUCT = false>
__simt_vf__ __aicore__ __launch_bounds__(512) inline void SpsvTripleTreeUp(__gm__ float* tree, int64_t leaves,
                                                                           int64_t count, int64_t first, int32_t block,
                                                                           int32_t blocks)
{
    SpsvSimtRefresh(tree);
    for (int64_t entry = static_cast<int64_t>(block) * 512 + threadIdx.x; entry < first * (WIDTH * (WIDTH + 1));
         entry += static_cast<int64_t>(blocks) * 512) {
        auto position = SpsvAffineUpPosition<WIDTH>(entry, first);
        SpsvTriplePrefixValue z = SpsvTriplePrefixZero();
        if (position.column == WIDTH) {
            z = SpsvTripleTreeLoad<WIDTH>(tree, 2 * position.node + 1, position.row, position.column, leaves, count);
        }
        for (int32_t k = 0; k < WIDTH; ++k) {
            z = SpsvTriplePrefixAdd(z, SpsvTriplePrefixTreeMul<SHORT_PRODUCT>(
                                           SpsvTripleTreeLoad<WIDTH>(tree, 2 * position.node + 1, position.row, k, leaves, count),
                                           SpsvTripleTreeLoad<WIDTH>(tree, 2 * position.node, k, position.column, leaves, count)));
        }
        SpsvTriplePrefixStore(tree + SpsvTriplePrefixAddress<WIDTH>(position.node, position.row, position.column), z);
    }
    asc_threadfence();
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline SpsvTriplePrefixValue SpsvTripleTreeState(__gm__ const float* states, int64_t node, int32_t row)
{
    if (node == 1) {
        return SpsvTriplePrefixZero();
    }
    return SpsvTriplePrefixLoad(states + (node * WIDTH + row) * kSpsvTriplePrefixWords);
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth, bool SHORT_PRODUCT = false>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvTripleTreeDown(__gm__ const float* tree,
                                                                             __gm__ float* states, int64_t leaves,
                                                                             int64_t count, int64_t first,
                                                                             int32_t block, int32_t blocks)
{
    SpsvSimtRefresh(tree);
    for (int64_t entry = static_cast<int64_t>(block) * 256 + threadIdx.x; entry < 2 * first * WIDTH;
         entry += static_cast<int64_t>(blocks) * 256) {
        int64_t child = 2 * first + entry / WIDTH;
        int64_t parent = child / 2;
        int32_t row = static_cast<int32_t>(entry % WIDTH);
        SpsvTriplePrefixValue z = SpsvTripleTreeState<WIDTH>(states, parent, row);
        if ((child & 1) != 0) {
            z = SpsvTripleTreeLoad<WIDTH>(tree, child - 1, row, WIDTH, leaves, count);
            for (int32_t k = 0; k < WIDTH; ++k) {
                z = SpsvTriplePrefixAdd(z, SpsvTriplePrefixTreeMul<SHORT_PRODUCT>(
                                               SpsvTripleTreeLoad<WIDTH>(tree, child - 1, row, k, leaves, count),
                                               SpsvTripleTreeState<WIDTH>(states, parent, k)));
            }
        }
        SpsvTriplePrefixStore(states + (child * WIDTH + row) * kSpsvTriplePrefixWords, z);
    }
    asc_threadfence();
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline SpsvTriplePrefixValue SpsvTripleTreeResult(__gm__ const float* tree, __gm__ const float* states,
                                                                  int64_t leaves, int64_t at)
{
    int64_t node = leaves + at / WIDTH;
    int32_t row = static_cast<int32_t>(at % WIDTH);
    SpsvTriplePrefixValue z = SpsvTriplePrefixLoad(tree + SpsvTriplePrefixAddress<WIDTH>(node, row, WIDTH));
    for (int32_t k = 0; k < WIDTH; ++k) {
        z = SpsvTriplePrefixAdd(
            z, SpsvTriplePrefixMul(SpsvTriplePrefixLoad(tree + SpsvTriplePrefixAddress<WIDTH>(node, row, k)),
                                   SpsvTripleTreeState<WIDTH>(states, node, k)));
    }
    return z;
}

__simt_callee__ inline uint32_t SpsvTriplePublishRow(SpsvTriplePrefixValue z, __gm__ SpsvComplex64* y,
                                                     __gm__ float* wide, __gm__ float* prep, int64_t r)
{
    uint32_t risk = 0U;
    float er = SpsvFloatBoundAdd(
        z.radius, SpsvFloatBoundAdd(SpsvFloatBoundAdd(SpsvAbsF(z.re.hi), SpsvAbsF(z.re.lo)), SpsvAbsF(z.re.tail)));
    float ei = SpsvFloatBoundAdd(
        z.radius, SpsvFloatBoundAdd(SpsvFloatBoundAdd(SpsvAbsF(z.im.hi), SpsvAbsF(z.im.lo)), SpsvAbsF(z.im.tail)));
    if (prep[18 * r + 9] >= 0.f && er < prep[18 * r + 9]) {
        prep[18 * r + 9] = er;
    }
    if (prep[18 * r + 10] >= 0.f && ei < prep[18 * r + 10]) {
        prep[18 * r + 10] = ei;
    }
    float high = y[r].real, low = wide[4 * r + 1], bound = prep[18 * r + 9];
    if (SpsvFloatNeedsDependencyRetry(high, low, bound) &&
        SpsvTriplePrefixCorrectComponent(high, low, bound, z.re, z.radius)) {
        y[r].real = high;
        wide[4 * r] = high;
        wide[4 * r + 1] = low;
        prep[18 * r + 9] = bound;
    }
    high = y[r].imag;
    low = wide[4 * r + 3];
    bound = prep[18 * r + 10];
    if (SpsvFloatNeedsDependencyRetry(high, low, bound) &&
        SpsvTriplePrefixCorrectComponent(high, low, bound, z.im, z.radius)) {
        y[r].imag = high;
        wide[4 * r + 2] = high;
        wide[4 * r + 3] = low;
        prep[18 * r + 10] = bound;
    }
    risk |= SpsvFloatNeedsDependencyRetry(y[r].real, wide[4 * r + 1], prep[18 * r + 9]);
    risk |= SpsvFloatNeedsDependencyRetry(y[r].imag, wide[4 * r + 3], prep[18 * r + 10]);
    return risk;
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_vf__ __aicore__ __launch_bounds__(128) inline void SpsvTriplePrefixPublish(
    __gm__ const float* input, __gm__ const float* states, int64_t leaves, __gm__ SpsvComplex64* y, __gm__ float* wide,
    int64_t m, bool forward, int32_t block, int32_t blocks, __gm__ uint32_t* risks)
{
    SpsvSimtRefresh(input);
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    __gm__ float* prep = wide + 6 * m;
    uint32_t risk = 0U;
    for (int64_t r = static_cast<int64_t>(block) * 128 + tid; r < m; r += static_cast<int64_t>(blocks) * 128) {
        int64_t at = forward ? r : m - 1 - r;
        SpsvTriplePrefixValue z = SpsvTripleTreeResult<WIDTH>(input, states, leaves, at);
        risk |= SpsvTriplePublishRow(z, y, wide, prep, r);
    }
    SpsvPublishBlockRisk128(risk, risks, block);
}

__aicore__ inline void SpsvTriplePrefixBarrier()
{
    AscendC::CrossCoreSetFlag<0, PIPE_V>(0);
    AscendC::CrossCoreWaitFlag<0, PIPE_S>(0);
}

#endif
