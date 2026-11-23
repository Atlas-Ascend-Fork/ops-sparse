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
#ifndef SPSV_BLOCK_BOUND_H_
#define SPSV_BLOCK_BOUND_H_

#include "spsv_float_bound.h"
#include "../recovery/spsv_rational.h"

constexpr int32_t kSpsvBoundBlockRows = 32;
constexpr int32_t kSpsvBoundBandwidth = 16;

__simt_callee__ inline bool SpsvBlockFiniteNormal(float value)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    bits &= 0x7fffffffU;
    return bits == 0 || (bits >= 0x00800000U && bits < 0x7f800000U);
}

// Tight error-free-transform accumulator. The radius uses the same units
// and FTZ floor as the normal FP32 shadow pass.
__simt_callee__ inline void SpsvBlockAcc(SpsvDF& sum, float& error, float a, SpsvDF y)
{
    float ph = a * y.hi, pl = a * y.lo;
    SpsvDF product = SpsvTwoSum(fmaf(a, y.hi, -ph), pl);
    SpsvDF high = SpsvTwoSum(sum.hi, ph);
    SpsvDF middle = SpsvTwoSum(high.lo, product.hi);
    SpsvDF low = SpsvTwoSum(sum.lo, middle.hi);
    error += (SpsvAbsF(fmaf(a, y.lo, -pl)) + SpsvAbsF(product.lo) + SpsvAbsF(middle.lo) + SpsvAbsF(low.lo)) * 0x1p23f;
    sum = {high.hi, low.hi};
}

__simt_callee__ inline float SpsvBoundDown(float value)
{
    if (!(value > 0.0f)) {
        return 0.0f;
    }
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    bits = bits > 0x00800002U ? bits - 2U : 0U;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}

// Radius around (Y_hi+Y_lo) that certifies either the binary32 rounding
// cell or the stricter task budget used by the dependency retry.
__simt_callee__ inline float SpsvFloatAllowedRadius(float y, float low)
{
    if (!SpsvIsFiniteF(y) || !SpsvIsFiniteF(low)) {
        return 0.0f;
    }
    float budget = SpsvAbsF(y) * 0x1p-12f;
    budget = budget > 0x1p-18f ? budget : 0x1p-18f;
    budget = budget < 0.0005f ? budget : 0.0005f;
    float allowed = SpsvBoundDown(budget - SpsvAbsF(low));
    uint32_t bits;
    __builtin_memcpy(&bits, &y, sizeof(bits));
    uint32_t magnitude = bits & 0x7fffffffU;
    if ((magnitude >> 23) > 24U) {
        uint32_t gapBits = ((magnitude >> 23) - 24U) << 23;
        float upper;
        __builtin_memcpy(&upper, &gapBits, sizeof(upper));
        float lower = (magnitude & 0x7fffffU) == 0 ? upper * 0.5f : upper;
        float delta = (bits >> 31) != 0 ? -low : low;
        upper -= delta;
        lower += delta;
        float slack = SpsvBoundDown(upper < lower ? upper : lower);
        allowed = allowed > slack ? allowed : slack;
    }
    return allowed;
}

__simt_callee__ inline float SpsvBlockMagnitude(SpsvDF value, float error, float floor)
{
    value = SpsvTwoSum(value.hi, value.lo);
    return SpsvFloatBoundAdd(SpsvFloatBoundAdd(SpsvAbsF(value.hi), SpsvAbsF(value.lo)),
                             SpsvFloatBoundUp(error * 0x1p-20f + floor));
}

#endif // SPSV_BLOCK_BOUND_H_
