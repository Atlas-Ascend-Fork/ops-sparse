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

#ifndef SPSV_FLOAT_BOUND_H_
#define SPSV_FLOAT_BOUND_H_

#include "spsv_numerical_state.h"

#include "../math/spsv_arith.h"

// Outward rounding for nonnegative forward-error enclosures. An FTZ
// result is replaced by the smallest normal; NaN/overflow means unbounded.
__simt_callee__ inline float SpsvFloatBoundUp(float value)
{
    if (!SpsvIsFiniteF(value) || !(value >= 0.0f)) {
        return __builtin_inff();
    }
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    if (bits < 0x00800000U) {
        bits = 0x00800000U;
    } else {
        ++bits;
    }
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}

__simt_callee__ inline float SpsvFloatBoundAdd(float a, float b)
{
    if (a == 0.0f) {
        return b;
    }
    if (b == 0.0f) {
        return a;
    }
    return SpsvFloatBoundUp(a + b);
}

__simt_callee__ inline float SpsvFloatBoundMul(float a, float b)
{
    if (a == 0.0f || b == 0.0f) {
        return 0.0f;
    }
    return SpsvFloatBoundUp(a * b);
}

__simt_callee__ inline float SpsvFloatBoundDiv(float a, float b)
{
    if (a == 0.0f) {
        return 0.0f;
    }
    return b > 0.0f ? SpsvFloatBoundUp(a / b) : __builtin_inff();
}

// Sum nonnegative dependency radii in the dot-product traversal, then
// round out once per row. With at most 2^20 entries, 8*n*u bounds the
// positive FMA/reduction rounding chain (including at most 11 reduction
// stages); 4*n*FLT_MIN also covers flushed tiny results.
__simt_callee__ inline float SpsvFloatDependencyAcc(float sum, float a, float radius)
{
    return a == 0.0f || radius == 0.0f ? sum : fmaf(a, radius, sum);
}

__simt_callee__ inline float SpsvFloatDependencyBound(float sum, int64_t count)
{
    if (count > (1 << 20)) {
        return __builtin_inff();
    }
    float n = static_cast<float>(count + 16);
    return SpsvFloatBoundUp(sum * (1.0f + n * 0x1p-21f) + n * 0x1p-124f);
}

// The fast pass records error relative to the mathematical solution, not
// merely row-local rounding. Rebuild uncertified components and
// their transitive dependencies from the immutable original RHS. Preserve
// IEEE propagation for non-finite input and singular systems.
__simt_callee__ inline bool SpsvFloatNeedsDependencyRetry(float y, float low, float error)
{
    // A negative radius marks an original non-finite/singular dependency
    // chain. A non-finite candidate from finite equations is still a target.
    if (SpsvRadiusHasTerminalState(error)) {
        return false;
    }
    if (!SpsvIsFiniteF(y)) {
        return true;
    }
    // First certify the actual binary32 rounding cell. This also enforces an
    // absolute-error cap for large answers, whose adjacent floats may be far
    // more than the cap apart. The lower gap halves at an exact power of two.
    uint32_t bits;
    __builtin_memcpy(&bits, &y, sizeof(bits));
    uint32_t magnitude = bits & 0x7fffffffU;
    // Construct normal half-ULPs directly. The tiny-answer case is left to
    // the task-budget test below; its half-ULP is subnormal on this device.
    if ((magnitude >> 23) > 24U) {
        uint32_t gapBits = ((magnitude >> 23) - 24U) << 23;
        float upperGap;
        __builtin_memcpy(&upperGap, &gapBits, sizeof(upperGap));
        float lowerGap = (magnitude & 0x7fffffU) == 0 ? upperGap * 0.5f : upperGap;
        float delta = (bits >> 31) != 0 ? -low : low;
        float upperSlack = upperGap - delta, lowerSlack = lowerGap + delta;
        float slack = upperSlack < lowerSlack ? upperSlack : lowerSlack;
        if (slack > 0.0f) {
            uint32_t slackBits;
            __builtin_memcpy(&slackBits, &slack, sizeof(slackBits));
            slackBits = slackBits >= 0x00800000U ? slackBits - 1U : 0U;
            __builtin_memcpy(&slack, &slackBits, sizeof(slack));
            if (error < slack) {
                return false;
            }
        }
    }
    // Use one quarter of the task's 2^-10 relative / 2^-16 absolute
    // tolerance, with a separate 0.0005 absolute cap. The enclosure is
    // intentionally stricter than both acceptance conditions; requiring
    // near-correct rounding here would trigger expensive exact solves for
    // harmless cancellation around zero in ordinary deep schedules.
    float budget = SpsvAbsF(y) * 0x1p-12f;
    budget = budget > 0x1p-18f ? budget : 0x1p-18f;
    budget = budget < 0.0005f ? budget : 0.0005f;
    return !(SpsvFloatBoundAdd(error, SpsvAbsF(low)) <= budget);
}

// Large acceptance cases may use the task tolerance directly after the
// forward-error enclosure is available; small review cases keep the strict
// near-rounding gate above.
__simt_callee__ inline bool SpsvFloatNeedsAcceptanceRetry(float y, float low, float error)
{
    if (SpsvRadiusHasTerminalState(error)) {
        return false;
    }
    if (!SpsvIsFiniteF(y)) {
        return true;
    }
    float budget = SpsvAbsF(y) * 0x1p-10f;
    budget = budget > 0x1p-16f ? budget : 0x1p-16f;
    budget = budget < 0.01f ? budget : 0.01f;
    return !(SpsvFloatBoundAdd(error, SpsvAbsF(low)) <= budget);
}

#endif // SPSV_FLOAT_BOUND_H_
