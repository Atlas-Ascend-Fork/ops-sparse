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

// The candidate carries infinite radii. Only a later certificate against
// the original equations or exact reconstruction can accept its result.
#ifndef SPSV_COMPLEX_CANDIDATE_H_
#define SPSV_COMPLEX_CANDIDATE_H_

__simt_callee__ inline SpsvDF SpsvProposalPairMul(SpsvDF a, SpsvDF b)
{
    return SpsvFastAdd(SpsvFastMul(a, b.hi), SpsvFastMul(a, b.lo));
}

__simt_callee__ inline SpsvDF SpsvProposalWarpSum(SpsvDF value, int32_t width)
{
    for (int32_t mask = 1; mask < width; mask <<= 1) {
        float high = asc_shfl_xor(value.hi, mask);
        float low = asc_shfl_xor(value.lo, mask);
        auto sum = SpsvTwoSum(value.hi, high);
        value.hi = sum.hi;
        value.lo = (value.lo + low) + sum.lo;
    }
    value = SpsvTwoSum(value.hi, value.lo);
    // Only lane zero consumes this reduction: it publishes a row or
    // writes the warp partial. Broadcasting to the other lanes is redundant.
    return value;
}

template <bool NON_UNIT>
__simt_callee__ inline void SpsvProposalPublish(__gm__ SpsvComplex64* y, __gm__ float* wide, __gm__ float* prep,
                                                int32_t r, SpsvDF re, SpsvDF im)
{
    int64_t meta = 18 * static_cast<int64_t>(r);
    if constexpr (NON_UNIT) {
        SpsvDF ir{prep[meta + 11], prep[meta + 12]};
        SpsvDF ii{prep[meta + 13], prep[meta + 14]};
        SpsvDF qr = SpsvFastAdd(SpsvProposalPairMul(re, ir), SpsvDFNeg(SpsvProposalPairMul(im, ii)));
        SpsvDF qi = SpsvFastAdd(SpsvProposalPairMul(re, ii), SpsvProposalPairMul(im, ir));
        re = qr;
        im = qi;
        if (prep[meta + 6] < 0.f) {
            re = SpsvFastAdd(re, {prep[meta + 2], prep[meta + 3]});
            im = SpsvFastAdd(im, {prep[meta + 4], prep[meta + 5]});
        }
    }
    int64_t at = 4 * static_cast<int64_t>(r);
    y[r].real = re.hi;
    y[r].imag = im.hi;
    wide[at] = re.hi;
    wide[at + 1] = re.lo;
    wide[at + 2] = im.hi;
    wide[at + 3] = im.lo;
    prep[meta + 9] = __builtin_inff();
    prep[meta + 10] = __builtin_inff();
}

__simt_callee__ inline bool SpsvProposalMergeBlock(SpsvDF& re, SpsvDF& im, __ubuf__ float* partials)
{
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    int32_t warp = tid / 32;
    if ((tid & 31) == 0) {
        partials[4 * warp] = re.hi;
        partials[4 * warp + 1] = re.lo;
        partials[4 * warp + 2] = im.hi;
        partials[4 * warp + 3] = im.lo;
    }
    asc_syncthreads();
    if (tid >= 32) {
        return false;
    }
    re = tid < 8 ? SpsvDF{partials[4 * tid], partials[4 * tid + 1]} : SpsvDF{0.f, 0.f};
    im = tid < 8 ? SpsvDF{partials[4 * tid + 2], partials[4 * tid + 3]} : SpsvDF{0.f, 0.f};
    re = SpsvProposalWarpSum(re, 8);
    im = SpsvProposalWarpSum(im, 8);
    return true;
}

template <bool FORWARD, bool NON_UNIT, int32_t GROUP>
__simt_callee__ inline void SpsvProposalRow(__gm__ const int32_t* rp, __gm__ const int32_t* ci,
                                            __gm__ const SpsvComplex64* a, __gm__ SpsvComplex64* y, __gm__ float* wide,
                                            __gm__ float* prep, int32_t r, __ubuf__ float* partials)
{
    int32_t lane = static_cast<int32_t>(threadIdx.x) & (GROUP - 1);
    int64_t meta = 18 * static_cast<int64_t>(r);
    float rh = 0.f, rl = 0.f, ih = 0.f, il = 0.f;
    if (lane == 0 && (!NON_UNIT || prep[meta + 6] >= 0.f)) {
        rh = prep[meta + 2];
        rl = prep[meta + 3];
        ih = prep[meta + 4];
        il = prep[meta + 5];
    }
    for (int64_t p = static_cast<int64_t>(rp[r]) + lane; p < rp[r + 1]; p += GROUP) {
        int32_t c = ci[p];
        if (FORWARD ? c < r : c > r) {
            SpsvAccumComplexDF(rh, rl, ih, il, a[p].real, a[p].imag, wide, c);
        }
    }
    SpsvDF re = SpsvTwoSum(rh, rl), im = SpsvTwoSum(ih, il);
    if constexpr (GROUP > 1) {
        int32_t width = 32;
        if constexpr (GROUP == 32) {
            int32_t length = rp[r + 1] - rp[r];
            width = length <= 4 ? 4 : length <= 8 ? 8 : length <= 16 ? 16 : 32;
        }
        re = SpsvProposalWarpSum(re, width);
        im = SpsvProposalWarpSum(im, width);
    }
    if constexpr (GROUP == 256) {
        if (!SpsvProposalMergeBlock(re, im, partials)) {
            return;
        }
    }
    if (lane == 0) {
        SpsvProposalPublish<NON_UNIT>(y, wide, prep, r, re, im);
    }
}

template <bool NON_UNIT>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvComplexProposalInitialLevel(
    __gm__ SpsvComplex64* y, __gm__ float* wide, __gm__ float* prep, __gm__ const int32_t* lp, __gm__ const int32_t* lr,
    int32_t block, int32_t blocks)
{
    SpsvInvalidateSolveCache((__gm__ void*)prep);
    // Analysis guarantees that this level has no triangular predecessors.
    // Keep the original DF publication and infinite-radius acceptance gate;
    // no CSR scan or warp reduction is needed for these independent rows.
    int64_t start = static_cast<int64_t>(lp[0]) + static_cast<int64_t>(block) * 256 + threadIdx.x;
    for (int64_t at = start; at < lp[1]; at += static_cast<int64_t>(blocks) * 256) {
        int32_t row = lr[at];
        int64_t meta = 18 * static_cast<int64_t>(row);
        SpsvDF re{0.f, 0.f}, im{0.f, 0.f};
        if (!NON_UNIT || prep[meta + 6] >= 0.f) {
            re = SpsvTwoSum(prep[meta + 2], prep[meta + 3]);
            im = SpsvTwoSum(prep[meta + 4], prep[meta + 5]);
        }
        SpsvProposalPublish<NON_UNIT>(y, wide, prep, row, re, im);
    }
    asc_threadfence();
}

template <bool FORWARD, bool NON_UNIT>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvComplexProposalSchedule(
    __gm__ const int32_t* rp, __gm__ const int32_t* ci, __gm__ const SpsvComplex64* a, __gm__ SpsvComplex64* y,
    __gm__ float* wide, __gm__ float* prep, __gm__ const int32_t* lp, __gm__ const int32_t* lr, int32_t levels,
    int32_t firstLevel)
{
    SpsvInvalidateSolveCache((__gm__ void*)prep);
    int32_t tid = static_cast<int32_t>(threadIdx.x), warp = tid / 32;
    __ubuf__ float partials[32];
    bool owed = false;
    bool blockWrites = false;
    for (int32_t level = firstLevel; level < levels; ++level) {
        int32_t begin = lp[level], end = lp[level + 1];
        if (blockWrites) {
            asc_syncthreads();
            blockWrites = false;
            owed = false;
        }
        if (end - begin == 1) {
            int32_t r = lr[begin], length = rp[r + 1] - rp[r];
            if (length >= 256) {
                if (owed) {
                    asc_syncthreads();
                }
                SpsvProposalRow<FORWARD, NON_UNIT, 256>(rp, ci, a, y, wide, prep, r, partials);
                blockWrites = true;
            } else if (length <= 4) {
                if (tid == 0) {
                    SpsvProposalRow<FORWARD, NON_UNIT, 1>(rp, ci, a, y, wide, prep, r, partials);
                }
            } else if (warp == 0) {
                SpsvProposalRow<FORWARD, NON_UNIT, 32>(rp, ci, a, y, wide, prep, r, partials);
            }
            // Other warps may advance through singleton levels, but they
            // cannot consume warp zero's result until this debt is paid.
            owed = true;
        } else {
            if (owed) {
                asc_syncthreads();
                owed = false;
            }
            for (int32_t at = begin + warp; at < end; at += 8) {
                int32_t r = lr[at];
                SpsvProposalRow<FORWARD, NON_UNIT, 32>(rp, ci, a, y, wide, prep, r, partials);
            }
            blockWrites = true;
        }
    }
    if (owed || blockWrites) {
        asc_syncthreads();
    }
    asc_threadfence();
}
#endif
