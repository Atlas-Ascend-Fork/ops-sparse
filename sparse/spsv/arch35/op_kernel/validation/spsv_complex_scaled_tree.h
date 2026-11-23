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
#ifndef SPSV_COMPLEX_SCALED_TREE_H_
#define SPSV_COMPLEX_SCALED_TREE_H_

#include "spsv_complex_scaled_math.h"
#include "spsv_complex_triple_prefix.h"

constexpr int32_t kSpsvScaledMaxBandwidth = 16;

template <typename Data>
__simt_callee__ inline bool SpsvScaledInputsFinite(const Data& data, int64_t row)
{
    if (!SpsvBlockFiniteNormal(data.alpha.real) || !SpsvBlockFiniteNormal(data.alpha.imag) ||
        !SpsvBlockFiniteNormal(SpsvScaledRhs(data, row).real) ||
        !SpsvBlockFiniteNormal(SpsvScaledRhs(data, row).imag)) {
        return false;
    }
    if (!data.nonUnit) {
        return true;
    }
    int32_t d = data.diagPtr[row];
    return d >= 0 && (SpsvScaledCoefficient(data, d).real != 0.f || SpsvScaledCoefficient(data, d).imag != 0.f);
}

template <typename Data>
__simt_callee__ inline uint32_t SpsvScaledRowBandwidth(const Data& data, int64_t row, int64_t m, bool ignoreTail)
{
    constexpr uint32_t invalid = kSpsvScaledMaxBandwidth + 2U;
    if (!SpsvScaledInputsFinite(data, row)) {
        return invalid;
    }
    if (ignoreTail && row == (data.forward ? m - 1 : 0)) {
        return 0;
    }
    uint32_t maximum = 0;
    for (int64_t p = data.rowPtr[row]; p < data.rowPtr[row + 1]; ++p) {
        int64_t distance = data.forward ? row - data.colInd[p] : static_cast<int64_t>(data.colInd[p]) - row;
        if (!SpsvBlockFiniteNormal(SpsvScaledCoefficient(data, p).real) ||
            !SpsvBlockFiniteNormal(SpsvScaledCoefficient(data, p).imag) || !SpsvScaledColumnOrder(data, row, p)) {
            return invalid;
        }
        if (distance > kSpsvScaledMaxBandwidth) {
            int64_t logical = data.forward ? row : m - 1 - row;
            if (logical != m - 1) {
                return invalid;
            }
            maximum = kSpsvScaledMaxBandwidth + 1U;
            continue;
        }
        if (distance > static_cast<int64_t>(maximum)) {
            maximum = static_cast<uint32_t>(distance);
        }
    }
    return maximum;
}

template <typename Data>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvScaledValidate(const Data data, int64_t m, int32_t block,
                                                                             int32_t blocks, __gm__ uint32_t* flags,
                                                                             bool ignoreTail)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)data.wide);
    }
    asc_syncthreads();
    uint32_t maximum = 0;
    for (int64_t r = static_cast<int64_t>(block) * 256 + threadIdx.x; r < m; r += static_cast<int64_t>(blocks) * 256) {
        uint32_t width = SpsvScaledRowBandwidth(data, r, m, ignoreTail);
        maximum = maximum > width ? maximum : width;
    }
    maximum = asc_reduce_max(maximum);
    __ubuf__ uint32_t warp[8];
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    if ((tid & 31) == 0) {
        warp[tid / 32] = maximum;
    }
    asc_syncthreads();
    if (tid == 0) {
        for (int32_t w = 1; w < 8; ++w) {
            maximum = maximum > warp[w] ? maximum : warp[w];
        }
        flags[block] = maximum;
    }
    asc_threadfence();
}

template <int32_t WORDS, typename Data>
__simt_vf__ __aicore__ __launch_bounds__(128) inline void SpsvScaledSeed(const Data data, int64_t m, int64_t activeM,
                                                                         int32_t block, int32_t blocks, int64_t leaves,
                                                                         int32_t width, __gm__ float* tree,
                                                                         __gm__ float* inverses)
{
    for (int64_t at = static_cast<int64_t>(block) * 128 + threadIdx.x; at < activeM;
         at += static_cast<int64_t>(blocks) * 128) {
        int64_t r = data.forward ? at : m - 1 - at;
        auto inv = SpsvScaledInverse<WORDS>(data, m, r);
        SpsvScaledStore(inverses + at * kSpsvScaledWords, inv);
        SpsvComplex64 rhs{SpsvScaledRhs(data, r).real, SpsvScaledRhs(data, r).imag};
        auto value = SpsvScaledMul(SpsvScaledFromComplex<WORDS>(data.alpha), SpsvScaledFromComplex<WORDS>(rhs));
        SpsvScaledStore(tree + SpsvScaledAt(leaves + at / width, static_cast<int32_t>(at % width), width, width),
                        value);
    }
    asc_threadfence();
}

template <int32_t WORDS, typename Data>
__simt_callee__ __attribute__((always_inline)) inline SpsvScaledQuad<WORDS> SpsvScaledPrepareValue(
    const Data& data, int64_t m, int64_t activeM, int64_t part, int64_t logical, int32_t row, int32_t lane,
    int32_t width, int64_t leaves, __gm__ const float* tree, __gm__ const float* inverses)
{
    auto value = SpsvScaledZero<WORDS>();
    if (logical < activeM) {
        int64_t r = data.forward ? logical : m - 1 - logical;
        if (lane == width) {
            value = SpsvScaledLoad<WORDS>(tree + SpsvScaledAt(leaves + part, row, width, width));
        }
        for (int64_t p = data.rowPtr[r]; p < data.rowPtr[r + 1]; ++p) {
            int64_t source = data.forward ? data.colInd[p] : m - 1 - data.colInd[p];
            if (source >= logical) {
                continue;
            }
            auto prior = SpsvScaledZero<WORDS>();
            if (source >= part * width) {
                prior = SpsvScaledLoad<WORDS>(tree + SpsvScaledAt(leaves + part,
                                                                  static_cast<int32_t>(source - part * width),
                                                                  lane, width));
            } else if (source - (part - 1) * width == lane) {
                prior = SpsvScaledFromComplex<WORDS>({1.f, 0.f});
            }
            auto coefficient = SpsvScaledFromComplex<WORDS>(
                {-SpsvScaledCoefficient(data, p).real, -SpsvScaledCoefficient(data, p).imag});
            value = SpsvScaledAdd(value, SpsvScaledMul(coefficient, prior));
        }
        value = SpsvScaledMul(value, SpsvScaledLoad<WORDS>(inverses + logical * kSpsvScaledWords));
    }
    return value;
}

template <int32_t WORDS, typename Data>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvScaledPrepare(const Data data, int64_t m, int64_t activeM,
                                                                            int32_t block, int32_t blocks,
                                                                            int64_t leaves, int64_t count,
                                                                            int32_t width, __gm__ float* tree,
                                                                            __gm__ const float* inverses)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)tree);
    }
    asc_syncthreads();
    int32_t lane = static_cast<int32_t>(threadIdx.x) & 31, warp = static_cast<int32_t>(threadIdx.x) / 32;
    for (int64_t part = static_cast<int64_t>(block) * 8 + warp; part < count;
         part += static_cast<int64_t>(blocks) * 8) {
        if (lane > width) {
            continue;
        }
        for (int32_t row = 0; row < width; ++row) {
            int64_t logical = part * width + row;
            auto value = SpsvScaledPrepareValue<WORDS>(
                data, m, activeM, part, logical, row, lane, width, leaves, tree, inverses);
            SpsvScaledStore(tree + SpsvScaledAt(leaves + part, row, lane, width), value);
        }
    }
    asc_threadfence();
}

template <int32_t WORDS>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledTreeLoad(__gm__ const float* tree, int64_t node, int32_t row,
                                                                int32_t col, int64_t leaves, int64_t count,
                                                                int32_t width)
{
    if (node >= leaves + count) {
        return row == col ? SpsvScaledFromComplex<WORDS>({1.f, 0.f}) : SpsvScaledZero<WORDS>();
    }
    return SpsvScaledLoad<WORDS>(tree + SpsvScaledAt(node, row, col, width));
}

template <int32_t WORDS>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvScaledTreeUp(__gm__ float* tree, int64_t leaves,
                                                                           int64_t count, int64_t first, int32_t width,
                                                                           int32_t block, int32_t blocks)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)tree);
    }
    asc_syncthreads();
    int64_t entries = width * (width + 1);
    for (int64_t at = static_cast<int64_t>(block) * 256 + threadIdx.x; at < first * entries;
         at += static_cast<int64_t>(blocks) * 256) {
        int64_t node = first + at / entries;
        int32_t row = static_cast<int32_t>((at % entries) / (width + 1)), col = static_cast<int32_t>(at % (width + 1));
        auto value = col == width ? SpsvScaledTreeLoad<WORDS>(tree, 2 * node + 1, row, col, leaves, count, width) :
                                    SpsvScaledZero<WORDS>();
        for (int32_t k = 0; k < width; ++k) {
            value = SpsvScaledAdd(
                value, SpsvScaledMul(SpsvScaledTreeLoad<WORDS>(tree, 2 * node + 1, row, k, leaves, count, width),
                                     SpsvScaledTreeLoad<WORDS>(tree, 2 * node, k, col, leaves, count, width)));
        }
        SpsvScaledStore(tree + SpsvScaledAt(node, row, col, width), value);
    }
    asc_threadfence();
}

template <int32_t WORDS>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledState(__gm__ const float* state, int64_t node, int32_t row,
                                                             int32_t width)
{
    return node == 1 ? SpsvScaledZero<WORDS>() : SpsvScaledLoad<WORDS>(state + (node * width + row) * kSpsvScaledWords);
}

template <int32_t WORDS>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvScaledTreeDown(__gm__ const float* tree,
                                                                             __gm__ float* state, int64_t leaves,
                                                                             int64_t count, int64_t first,
                                                                             int32_t width, int32_t block,
                                                                             int32_t blocks)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)state);
    }
    asc_syncthreads();
    for (int64_t at = static_cast<int64_t>(block) * 256 + threadIdx.x; at < 2 * first * width;
         at += static_cast<int64_t>(blocks) * 256) {
        int64_t node = 2 * first + at / width;
        int32_t row = static_cast<int32_t>(at % width);
        auto value = SpsvScaledState<WORDS>(state, node / 2, row, width);
        if ((node & 1) != 0) {
            value = SpsvScaledTreeLoad<WORDS>(tree, node - 1, row, width, leaves, count, width);
            for (int32_t k = 0; k < width; ++k) {
                value = SpsvScaledAdd(
                    value, SpsvScaledMul(SpsvScaledTreeLoad<WORDS>(tree, node - 1, row, k, leaves, count, width),
                                         SpsvScaledState<WORDS>(state, node / 2, k, width)));
            }
        }
        SpsvScaledStore(state + (node * width + row) * kSpsvScaledWords, value);
    }
    asc_threadfence();
}

template <int32_t WORDS>
__simt_callee__ inline bool SpsvScaledComponent(const SpsvDefectExpansion& center, float radius, int32_t exponent,
                                                float& high, float& low, float& bound)
{
    if (!SpsvIsFiniteF(radius)) {
        return false;
    }
    // Beyond the binary64 exponent range, later cancellations and IEEE
    // nonfinite propagation depend on the exact original recurrence.
    if (exponent >= 1023) {
        return false;
    }
    float rest = radius;
    for (int32_t j = 1; j < WORDS; ++j) {
        rest = SpsvFloatBoundAdd(rest, SpsvAbsF(center.word[j]));
    }
    float lower = SpsvBoundDown(SpsvAbsF(center.word[0]) - rest);
    int64_t thresholdExponent = 128LL - exponent;
    if (lower > 0.f && thresholdExponent <= 127) {
        float threshold = thresholdExponent < -126 ? 0x1p-126f : SpsvPow2(static_cast<int32_t>(thresholdExponent));
        if (lower > threshold) {
            high = center.word[0] < 0.f ? -__builtin_inff() : __builtin_inff();
            low = 0.f;
            bound = kSpsvCertifiedOverflowRadius;
            return true;
        }
    }
    float words[4] = {0.f, 0.f, 0.f, 0.f};
    for (int32_t j = 0; j < WORDS; ++j) {
        words[j] = ldexpf(center.word[j], exponent);
        if (!SpsvIsFiniteF(words[j])) {
            return false;
        }
    }
    auto a = SpsvTwoSum(words[0], words[1]);
    auto b = SpsvTwoSum(a.lo, words[2]);
    auto c = SpsvTwoSum(a.hi, b.hi);
    auto d = SpsvTwoSum(c.lo, words[3]);
    auto answer = SpsvTwoSum(c.hi, d.hi);
    float error = SpsvFloatBoundAdd(SpsvFloatBoundUp(ldexpf(radius, exponent)),
                                    SpsvFloatBoundUp(SpsvAbsF(b.lo) + SpsvAbsF(d.lo) + 0x1p-122f));
    high = answer.hi;
    low = answer.lo;
    bound = error;
    return !SpsvFloatNeedsDependencyRetry(answer.hi, answer.lo, error);
}

template <int32_t WORDS, typename Data>
__simt_callee__ inline uint32_t SpsvPublishScaledRow(const Data& data, int64_t m, int64_t row,
                                                     const SpsvScaledQuad<WORDS>& value)
{
    uint32_t pending = 0;
    for (int32_t component = 0; component < SpsvScaledComponentCount(data); ++component) {
        float high = 0.f, low = 0.f, bound = __builtin_inff();
        if (!SpsvScaledComponent<WORDS>(component == 0 ? value.re : value.im, value.radius, value.exponent, high, low,
                                        bound)) {
            pending = 1U;
            bound = __builtin_inff();
        }
        SpsvScaledWriteComponent(data, m, row, component, high, low, bound);
    }
    return pending;
}

template <int32_t WORDS, typename Data>
__simt_vf__ __aicore__ __launch_bounds__(128) inline void SpsvScaledPublish(
    const Data data, int64_t m, int64_t activeM, __gm__ const float* tree, __gm__ const float* state, int64_t leaves,
    int32_t width, int32_t block, int32_t blocks, __gm__ uint32_t* risk, __gm__ float* solved)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)state);
    }
    asc_syncthreads();
    uint32_t pending = 0;
    for (int64_t at = static_cast<int64_t>(block) * 128 + threadIdx.x; at < activeM;
         at += static_cast<int64_t>(blocks) * 128) {
        int64_t node = leaves + at / width, r = data.forward ? at : m - 1 - at;
        int32_t row = static_cast<int32_t>(at % width);
        auto value = SpsvScaledLoad<WORDS>(tree + SpsvScaledAt(node, row, width, width));
        for (int32_t k = 0; k < width; ++k) {
            value = SpsvScaledAdd(value, SpsvScaledMul(SpsvScaledLoad<WORDS>(tree + SpsvScaledAt(node, row, k, width)),
                                                       SpsvScaledState<WORDS>(state, node, k, width)));
        }
        SpsvScaledStore(solved + at * kSpsvScaledWords, value);
        pending |= SpsvPublishScaledRow(data, m, r, value);
    }
    pending = asc_reduce_max(pending);
    __ubuf__ uint32_t warp[4];
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    if ((tid & 31) == 0) {
        warp[tid / 32] = pending;
    }
    asc_syncthreads();
    if (tid == 0) {
        for (int32_t w = 1; w < 4; ++w) {
            pending |= warp[w];
        }
        risk[block] = pending;
    }
    asc_threadfence();
}

template <int32_t WORDS>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledTailShuffle(const SpsvScaledQuad<WORDS>& a, int32_t offset)
{
    auto b = SpsvScaledZero<WORDS>();
    for (int32_t j = 0; j < 4; ++j) {
        b.re.word[j] = asc_shfl_down(a.re.word[j], offset, 32);
        b.im.word[j] = asc_shfl_down(a.im.word[j], offset, 32);
    }
    b.radius = asc_shfl_down(a.radius, offset, 32);
    b.exponent = asc_shfl_down(a.exponent, offset, 32);
    return b;
}
template <int32_t WORDS, typename Data>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledMergeTerminalPartials(const Data& data, int64_t m, int64_t row,
                                                                             __ubuf__ const float* partial,
                                                                             __ubuf__ const int32_t* exponent)
{
    SpsvComplex64 rhs{SpsvScaledRhs(data, row).real, SpsvScaledRhs(data, row).imag};
    auto result = SpsvScaledMul(SpsvScaledFromComplex<WORDS>(data.alpha), SpsvScaledFromComplex<WORDS>(rhs));
    for (int32_t w = 0; w < 8; ++w) {
        auto a = SpsvScaledZero<WORDS>();
        for (int32_t j = 0; j < 4; ++j) {
            a.re.word[j] = partial[w * 10 + j];
            a.im.word[j] = partial[w * 10 + 4 + j];
        }
        a.radius = partial[w * 10 + 8];
        a.exponent = exponent[w];
        result = SpsvScaledAdd(result, a);
    }
    return SpsvScaledMul(result, SpsvScaledInverse<WORDS>(data, m, row));
}

template <int32_t WORDS, typename Data>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvScaledTerminalRow(const Data data, int64_t m,
                                                                                __gm__ const float* solved,
                                                                                __gm__ uint32_t* risk)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)solved);
    }
    asc_syncthreads();
    int64_t row = data.forward ? m - 1 : 0;
    auto sum = SpsvScaledZero<WORDS>();
    for (int64_t p = static_cast<int64_t>(data.rowPtr[row]) + threadIdx.x; p < data.rowPtr[row + 1]; p += 256) {
        int64_t at = data.forward ? data.colInd[p] : m - 1 - data.colInd[p];
        if (at >= m - 1) {
            continue;
        }
        auto coefficient = SpsvScaledFromComplex<WORDS>(
            {-SpsvScaledCoefficient(data, p).real, -SpsvScaledCoefficient(data, p).imag});
        sum = SpsvScaledAdd(sum, SpsvScaledMul(coefficient, SpsvScaledLoad<WORDS>(solved + at * kSpsvScaledWords)));
    }
    int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
    for (int32_t offset = 16; offset > 0; offset /= 2) {
        auto b = SpsvScaledTailShuffle(sum, offset);
        if (lane + offset < 32) {
            sum = SpsvScaledAdd(sum, b);
        }
    }
    __ubuf__ float partial[80];
    __ubuf__ int32_t exponent[8];
    int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
    if (lane == 0) {
        for (int32_t j = 0; j < 4; ++j) {
            partial[warp * 10 + j] = sum.re.word[j];
            partial[warp * 10 + 4 + j] = sum.im.word[j];
        }
        partial[warp * 10 + 8] = sum.radius;
        exponent[warp] = sum.exponent;
    }
    asc_syncthreads();
    if (threadIdx.x == 0) {
        auto result = SpsvScaledMergeTerminalPartials<WORDS>(data, m, row, partial, exponent);
        risk[0] |= SpsvPublishScaledRow(data, m, row, result);
    }
    asc_threadfence();
}

#endif // SPSV_COMPLEX_SCALED_TREE_H_
