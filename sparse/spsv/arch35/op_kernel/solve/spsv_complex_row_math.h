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

#ifndef OP_KERNEL_SOLVE_SPSV_COMPLEX_ROW_MATH_H_
#define OP_KERNEL_SOLVE_SPSV_COMPLEX_ROW_MATH_H_

// Bounded arithmetic for the common case. A nonzero input component lies
// in [2^-24,2^24], and a row has at most 2^20 entries: high products and
// every partial sum then stay normal and below 2^70. Low words are normalized
// outputs of this solver. Any flushed low product is below 2^-126 and cannot
// affect an accepted component (whose magnitude is bounded below by the
// absolute product sum times the conditioning threshold).
__simt_callee__ inline bool SpsvFastComponent(float v)
{
    float a = SpsvAbsF(v);
    return a == 0.0f || (a >= 0x1p-24f && a <= 0x1p24f);
}

// Dependency components need a lower bound to keep high products normal.
// Their upper bound is checked on the computed numerator, absolute bound and
// quotient: an overflowing product/sum poisons the TwoSum residual and cannot
// pass the finite checks. This retains large finite triangular solutions.
__simt_callee__ inline bool SpsvFastDependency(float value)
{
    float magnitude = SpsvAbsF(value);
    return magnitude == 0.0f || (magnitude >= 0x1p-24f && SpsvIsFiniteF(value));
}

__simt_callee__ inline SpsvDF SpsvFastAdd(SpsvDF a, SpsvDF b)
{
    SpsvDF t = SpsvTwoSum(a.hi, b.hi);
    return SpsvTwoSum(t.hi, t.lo + (a.lo + b.lo));
}

__simt_callee__ inline SpsvDF SpsvFastMul(SpsvDF a, float b)
{
    float p = a.hi * b;
    return SpsvTwoSum(p, fmaf(a.hi, b, -p) + a.lo * b);
}

// sum -= (vr + i*vi) * yWide[col], in float-float:
//   real -= vr*yr - vi*yi
//   imag -= vr*yi + vi*yr
// Y arrives as a (hi, lo) pair, so the rows this one depends on contribute
// about 48 significand bits rather than 24.
__simt_callee__ inline void SpsvAccumComplexDF(float& reHi, float& reLo, float& imHi, float& imLo, float vr, float vi,
                                               __gm__ const float* yWide, int32_t col)
{
    float yrHi = yWide[4 * static_cast<int64_t>(col)];
    float yrLo = yWide[4 * static_cast<int64_t>(col) + 1];
    float yiHi = yWide[4 * static_cast<int64_t>(col) + 2];
    float yiLo = yWide[4 * static_cast<int64_t>(col) + 3];
    float p = vr * yrHi;
    SpsvAccF(reHi, reLo, -p, -(fmaf(vr, yrHi, -p) + vr * yrLo));
    p = vi * yiHi;
    SpsvAccF(reHi, reLo, p, fmaf(vi, yiHi, -p) + vi * yiLo);
    p = vr * yiHi;
    SpsvAccF(imHi, imLo, -p, -(fmaf(vr, yiHi, -p) + vr * yiLo));
    p = vi * yrHi;
    SpsvAccF(imHi, imLo, -p, -(fmaf(vi, yrHi, -p) + vi * yrLo));
}

#endif // OP_KERNEL_SOLVE_SPSV_COMPLEX_ROW_MATH_H_
