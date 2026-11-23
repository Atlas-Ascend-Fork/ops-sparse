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
#ifndef SPSV_COMPLEX_SCALED_MATH_H_
#define SPSV_COMPLEX_SCALED_MATH_H_

#include "../common/spsv_scaled_data.h"
#include "spsv_complex_defect.h"

template <int32_t WORDS>
struct SpsvScaledQuad {
    SpsvDefectExpansion re;
    SpsvDefectExpansion im;
    float radius;
    int32_t exponent;
};
constexpr int32_t kSpsvScaledWords = 10;

template <int32_t WORDS>
__simt_callee__ inline void SpsvScaledExpansionAdd(SpsvDefectExpansion& sum, float term)
{
    for (int32_t i = 0; i < WORDS; ++i) {
        auto next = SpsvTwoSum(sum.word[i], term);
        sum.word[i] = next.hi;
        term = next.lo;
    }
    // A complex product collects fewer than 128 positive loss terms.
    // The final 1+2^-10 inflation exceeds gamma_128, and its 2^-108
    // additive term covers all possible flushed intermediate losses.
    sum.discarded += SpsvAbsF(term);
}

template <int32_t WORDS>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledZero()
{
    return {{{0.f, 0.f, 0.f, 0.f}, 0.f}, {{0.f, 0.f, 0.f, 0.f}, 0.f}, 0.f, 0};
}

template <int32_t WORDS>
__simt_callee__ inline float SpsvScaledNorm(const SpsvScaledQuad<WORDS>& a)
{
    float norm = 0.f;
    for (int32_t j = 0; j < WORDS; ++j) {
        norm += SpsvAbsF(a.re.word[j]);
        norm += SpsvAbsF(a.im.word[j]);
    }
    return norm == 0.f ? 0.f : SpsvFloatBoundUp(norm * (1.f + 0x1p-20f) + 0x1p-122f);
}

template <int32_t WORDS>
__simt_callee__ inline bool SpsvScaledIsZero(const SpsvScaledQuad<WORDS>& a)
{
    bool zero = a.radius == 0.f;
    for (int32_t j = 0; j < WORDS; ++j) {
        zero &= a.re.word[j] == 0.f && a.im.word[j] == 0.f;
    }
    return zero;
}

template <int32_t WORDS>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledNormalize(SpsvScaledQuad<WORDS> a)
{
    float maximum = a.radius;
    bool finite = SpsvIsFiniteF(a.radius);
    for (int32_t j = 0; j < WORDS; ++j) {
        finite &= SpsvIsFiniteF(a.re.word[j]) && SpsvIsFiniteF(a.im.word[j]);
        maximum = SpsvMaxAbs2(maximum, a.re.word[j]);
        maximum = SpsvMaxAbs2(maximum, a.im.word[j]);
    }
    if (!finite || !SpsvIsFiniteF(maximum) || a.radius < 0.f) {
        a = SpsvScaledZero<WORDS>();
        a.radius = __builtin_inff();
        return a;
    }
    if (maximum == 0.f) {
        return SpsvScaledZero<WORDS>();
    }
    int32_t shift = SpsvExpOf(maximum);
    int64_t exponent = static_cast<int64_t>(a.exponent) + shift;
    if (exponent < -1000000000LL || exponent > 1000000000LL) {
        a = SpsvScaledZero<WORDS>();
        a.radius = __builtin_inff();
        return a;
    }
    auto scale = SpsvScaleFactors(-shift);
    for (int32_t j = 0; j < WORDS; ++j) {
        a.re.word[j] = (a.re.word[j] * scale.f1) * scale.f2;
        a.im.word[j] = (a.im.word[j] * scale.f1) * scale.f2;
    }
    // Eight center words and one radius can flush during normalization.
    // 2^-108 exceeds their combined possible FTZ loss by over 2^14.
    a.radius = SpsvFloatBoundAdd(SpsvFloatBoundUp((a.radius * scale.f1) * scale.f2), 0x1p-108f);
    a.exponent = static_cast<int32_t>(exponent);
    a.re.discarded = a.im.discarded = 0.f;
    return a;
}

template <int32_t WORDS>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledFromComplex(SpsvComplex64 value)
{
    auto a = SpsvScaledZero<WORDS>();
    a.re.word[0] = value.real;
    a.im.word[0] = value.imag;
    return SpsvScaledNormalize(a);
}

template <int32_t WORDS>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledLoad(__gm__ const float* p)
{
    auto a = SpsvScaledZero<WORDS>();
    for (int32_t j = 0; j < WORDS; ++j) {
        a.re.word[j] = p[j];
        a.im.word[j] = p[j + 4];
    }
    a.radius = p[8];
    __builtin_memcpy(&a.exponent, p + 9, sizeof(a.exponent));
    return a;
}

template <int32_t WORDS>
__simt_callee__ inline void SpsvScaledStore(__gm__ float* p, const SpsvScaledQuad<WORDS>& a)
{
    for (int32_t j = 0; j < WORDS; ++j) {
        p[j] = a.re.word[j];
        p[j + 4] = a.im.word[j];
    }
    p[8] = a.radius;
    __builtin_memcpy(p + 9, &a.exponent, sizeof(a.exponent));
}

template <int32_t WORDS>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledAlign(SpsvScaledQuad<WORDS> a, int32_t exponent)
{
    int64_t delta = static_cast<int64_t>(a.exponent) - exponent;
    if (delta < -120) {
        // A normalized center has each of its eight words below two.
        a = SpsvScaledZero<WORDS>();
        a.radius = 0x1p-115f;
    } else {
        float scale = SpsvPow2(static_cast<int32_t>(delta));
        for (int32_t j = 0; j < WORDS; ++j) {
            a.re.word[j] = a.re.word[j] * scale;
            a.im.word[j] = a.im.word[j] * scale;
        }
        a.radius = SpsvFloatBoundAdd(SpsvFloatBoundUp(a.radius * scale), 0x1p-108f);
    }
    a.exponent = exponent;
    return a;
}

template <int32_t WORDS>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledAdd(SpsvScaledQuad<WORDS> a, SpsvScaledQuad<WORDS> b)
{
    if (!SpsvIsFiniteF(a.radius) || !SpsvIsFiniteF(b.radius)) {
        auto z = SpsvScaledZero<WORDS>();
        z.radius = __builtin_inff();
        return z;
    }
    if (SpsvScaledIsZero(a)) {
        return b;
    }
    if (SpsvScaledIsZero(b)) {
        return a;
    }
    int32_t exponent = a.exponent > b.exponent ? a.exponent : b.exponent;
    a = SpsvScaledAlign(a, exponent);
    b = SpsvScaledAlign(b, exponent);
    for (int32_t j = 0; j < WORDS; ++j) {
        SpsvScaledExpansionAdd<WORDS>(a.re, b.re.word[j]);
        SpsvScaledExpansionAdd<WORDS>(a.im, b.im.word[j]);
    }
    float local = SpsvFloatBoundUp((a.re.discarded + a.im.discarded) * (1.f + 0x1p-10f) + 0x1p-108f);
    a.radius = SpsvFloatBoundAdd(SpsvFloatBoundAdd(a.radius, b.radius), local);
    return SpsvScaledNormalize(a);
}

template <int32_t WORDS>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledNeg(SpsvScaledQuad<WORDS> a)
{
    for (int32_t j = 0; j < WORDS; ++j) {
        a.re.word[j] = -a.re.word[j];
        a.im.word[j] = -a.im.word[j];
    }
    return a;
}

template <int32_t WORDS>
static __simt_callee__ __attribute__((noinline)) void SpsvScaledProductAcc(SpsvDefectExpansion& out,
                                                                           const SpsvDefectExpansion& a,
                                                                           const SpsvDefectExpansion& b, float sign)
{
    for (int32_t i = 0; i < WORDS; ++i) {
        if (a.word[i] == 0.f) {
            continue;
        }
        for (int32_t j = 0; j < WORDS; ++j) {
            if (b.word[j] == 0.f) {
                continue;
            }
            float product = a.word[i] * b.word[j];
            if (i + j >= WORDS) {
                // Omit only from the center. Charge the complete absolute
                // product, its rounding and any FTZ loss to the enclosure.
                out.discarded += SpsvAbsF(product) * (1.f + 0x1p-22f) + 0x1p-124f;
                continue;
            }
            SpsvScaledExpansionAdd<WORDS>(out, sign * product);
            float residual = fmaf(a.word[i], b.word[j], -product);
            if (i + j == WORDS - 1) {
                out.discarded += SpsvAbsF(residual) + 0x1p-124f;
            } else {
                SpsvScaledExpansionAdd<WORDS>(out, sign * residual);
            }
        }
    }
}

template <int32_t WORDS>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledMul(SpsvScaledQuad<WORDS> a, SpsvScaledQuad<WORDS> b)
{
    if (!SpsvIsFiniteF(a.radius) || !SpsvIsFiniteF(b.radius)) {
        auto z = SpsvScaledZero<WORDS>();
        z.radius = __builtin_inff();
        return z;
    }
    if (SpsvScaledIsZero(a) || SpsvScaledIsZero(b)) {
        return SpsvScaledZero<WORDS>();
    }
    auto z = SpsvScaledZero<WORDS>();
    int64_t exponent = static_cast<int64_t>(a.exponent) + b.exponent;
    if (exponent < -1000000000LL || exponent > 1000000000LL) {
        z.radius = __builtin_inff();
        return z;
    }
    z.exponent = static_cast<int32_t>(exponent);
    SpsvScaledProductAcc<WORDS>(z.re, a.re, b.re, 1.f);
    SpsvScaledProductAcc<WORDS>(z.re, a.im, b.im, -1.f);
    SpsvScaledProductAcc<WORDS>(z.im, a.re, b.im, 1.f);
    SpsvScaledProductAcc<WORDS>(z.im, a.im, b.re, 1.f);
    float propagated = SpsvFloatBoundAdd(SpsvFloatBoundMul(SpsvScaledNorm(a), b.radius),
                                         SpsvFloatBoundMul(SpsvScaledNorm(b), a.radius));
    propagated = SpsvFloatBoundAdd(propagated, SpsvFloatBoundMul(2.f, SpsvFloatBoundMul(a.radius, b.radius)));
    float local = SpsvFloatBoundUp((z.re.discarded + z.im.discarded) * (1.f + 0x1p-10f) + 0x1p-108f);
    z.radius = SpsvFloatBoundAdd(propagated, local);
    return SpsvScaledNormalize(z);
}

template <int32_t WORDS, typename Data>
__simt_callee__ inline SpsvScaledQuad<WORDS> SpsvScaledInverse(const Data& data, int64_t m, int64_t row)
{
    auto one = SpsvScaledFromComplex<WORDS>({1.f, 0.f});
    if (!data.nonUnit) {
        return one;
    }
    SpsvComplex64 diagonalValue{SpsvScaledCoefficient(data, data.diagPtr[row]).real,
                                SpsvScaledCoefficient(data, data.diagPtr[row]).imag};
    float scale = SpsvMaxAbs2(diagonalValue.real, diagonalValue.imag);
    float re = diagonalValue.real / scale, im = diagonalValue.imag / scale;
    float norm = re * re + im * im;
    float seed[4] = {(re / norm) / scale, 0.f, (-im / norm) / scale, 0.f};
    const float* p = seed;
    auto q = SpsvScaledZero<WORDS>();
    q.re.word[0] = p[0];
    q.re.word[1] = p[1];
    q.im.word[0] = p[2];
    q.im.word[1] = p[3];
    q = SpsvScaledNormalize(q);
    SpsvComplex64 diagonal{SpsvScaledCoefficient(data, data.diagPtr[row]).real,
                           SpsvScaledCoefficient(data, data.diagPtr[row]).imag};
    auto d = SpsvScaledFromComplex<WORDS>(diagonal);
    for (int32_t iteration = 0; iteration < 3; ++iteration) {
        auto residual = SpsvScaledAdd(one, SpsvScaledNeg(SpsvScaledMul(d, q)));
        // Newton constructs a candidate. Only its final independent
        // original-diagonal residual is used as an inverse certificate.
        residual.radius = 0.f;
        q = SpsvScaledAdd(q, SpsvScaledMul(q, residual));
        q.radius = 0.f;
    }
    auto residual = SpsvScaledAdd(one, SpsvScaledNeg(SpsvScaledMul(d, q)));
    float defect = SpsvFloatBoundUp(
        ldexpf(SpsvFloatBoundAdd(SpsvScaledNorm(residual), residual.radius), residual.exponent));
    if (!(defect < 0.5f)) {
        q.radius = __builtin_inff();
        return q;
    }
    q.radius = SpsvFloatBoundMul(SpsvScaledNorm(q), SpsvFloatBoundDiv(defect, SpsvBoundDown(1.f - defect)));
    return SpsvScaledNormalize(q);
}

__simt_callee__ inline int64_t SpsvScaledAt(int64_t node, int32_t row, int32_t column, int32_t width)
{
    return (node * width * (width + 1) + row * (width + 1) + column) * kSpsvScaledWords;
}

#endif // SPSV_COMPLEX_SCALED_MATH_H_
