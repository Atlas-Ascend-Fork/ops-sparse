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

#ifndef SPSV_COMPLEX_TRIPLE_ARITH_H_
#define SPSV_COMPLEX_TRIPLE_ARITH_H_

#include "spsv_complex_prefix.h"

struct SpsvTriplePrefixScalar {
    float hi;
    float lo;
    float tail;
};
constexpr int32_t kSpsvTriplePrefixWidth = 10;
constexpr int32_t kSpsvTriplePrefixColumns = kSpsvTriplePrefixWidth + 1;
constexpr int32_t kSpsvTriplePrefixEntries = kSpsvTriplePrefixWidth * kSpsvTriplePrefixColumns;
constexpr int32_t kSpsvTriplePrefixWords = 7;
constexpr int32_t kSpsvTriplePrefixMatrixWords = kSpsvTriplePrefixEntries * kSpsvTriplePrefixWords;
constexpr int32_t kSpsvTriplePrefixStateWords = kSpsvTriplePrefixWidth * kSpsvTriplePrefixWords;
constexpr int32_t kSpsvTriplePrefixNodeWords = kSpsvTriplePrefixMatrixWords + kSpsvTriplePrefixStateWords;

struct SpsvTriplePrefixValue {
    SpsvTriplePrefixScalar re;
    SpsvTriplePrefixScalar im;
    float radius;
};

__simt_callee__ inline bool SpsvCertificateRowFinite(const SpsvComplexCertificateData& data, int64_t row,
                                                     int32_t maxEntries)
{
    return SpsvIsFiniteF(data.alpha.real) && SpsvIsFiniteF(data.alpha.imag) &&
           SpsvBlockFiniteNormal(data.rhs[row].real) && SpsvBlockFiniteNormal(data.rhs[row].imag) &&
           SpsvBlockFiniteNormal(data.vecY[row].real) && SpsvBlockFiniteNormal(data.vecY[row].imag) &&
           SpsvBlockFiniteNormal(data.wide[4 * row + 1]) && SpsvBlockFiniteNormal(data.wide[4 * row + 3]) &&
           data.rowPtr[row + 1] - data.rowPtr[row] <= maxEntries;
}

__simt_callee__ inline SpsvTriplePrefixValue SpsvTriplePrefixInverse(const SpsvComplexCertificateData& data, int64_t m,
                                                                     int64_t row)
{
    if (!data.nonUnit) {
        return {{1.f, 0.f}, {0.f, 0.f}, 0.f};
    }
    __gm__ const float* prep = data.wide + 6 * m + 18 * row;
    return {{prep[11], prep[12]}, {prep[13], prep[14]}, prep[15]};
}

__simt_callee__ inline bool SpsvCertificateInverseFinite(const SpsvComplexCertificateData& data, int64_t m, int64_t row)
{
    auto inverse = SpsvTriplePrefixInverse(data, m, row);
    return !data.nonUnit || (data.diagPtr[row] >= 0 && inverse.radius >= 0.f && SpsvIsFiniteF(inverse.radius) &&
                             SpsvBlockFiniteNormal(inverse.re.hi) && SpsvBlockFiniteNormal(inverse.re.lo) &&
                             SpsvBlockFiniteNormal(inverse.im.hi) && SpsvBlockFiniteNormal(inverse.im.lo));
}

__simt_callee__ inline SpsvTriplePrefixValue SpsvTriplePrefixZero() { return {{0.f, 0.f}, {0.f, 0.f}, 0.f}; }

__simt_callee__ inline bool SpsvTriplePrefixIsZero(SpsvTriplePrefixValue a)
{
    return a.re.hi == 0.f && a.re.lo == 0.f && a.re.tail == 0.f && a.im.hi == 0.f && a.im.lo == 0.f &&
           a.im.tail == 0.f && a.radius == 0.f;
}
__simt_callee__ inline SpsvTriplePrefixValue SpsvTriplePrefixLoad(__gm__ const float* p)
{
    return {{p[0], p[1], p[2]}, {p[3], p[4], p[5]}, p[6]};
}
__simt_callee__ inline void SpsvTriplePrefixStore(__gm__ float* p, SpsvTriplePrefixValue a)
{
    p[0] = a.re.hi;
    p[1] = a.re.lo;
    p[2] = a.re.tail;
    p[3] = a.im.hi;
    p[4] = a.im.lo;
    p[5] = a.im.tail;
    p[6] = a.radius;
}
__simt_callee__ inline float SpsvTriplePrefixNorm(SpsvTriplePrefixValue a)
{
    float v = SpsvFloatBoundAdd(SpsvAbsF(a.re.hi), SpsvAbsF(a.re.lo));
    v = SpsvFloatBoundAdd(v, SpsvAbsF(a.re.tail));
    v = SpsvFloatBoundAdd(v, SpsvAbsF(a.im.hi));
    v = SpsvFloatBoundAdd(v, SpsvAbsF(a.im.lo));
    return SpsvFloatBoundAdd(v, SpsvAbsF(a.im.tail));
}
__simt_callee__ inline void SpsvTriplePrefixAcc(SpsvTriplePrefixScalar& s, float x, float& lost)
{
    SpsvDF a = SpsvTwoSum(s.hi, x);
    s.hi = a.hi;
    SpsvDF b = SpsvTwoSum(s.lo, a.lo);
    s.lo = b.hi;
    SpsvDF c = SpsvTwoSum(s.tail, b.lo);
    s.tail = c.hi;
    lost += SpsvAbsF(c.lo);
}
__simt_callee__ inline bool SpsvTriplePrefixTightDefect(const SpsvComplexCertificateData& data, int64_t row, float* out)
{
    if (data.rowPtr[row + 1] - data.rowPtr[row] > 64 || (data.nonUnit && data.diagPtr[row] < 0)) {
        return false;
    }
    SpsvDefectExpansion re{{0.f, 0.f, 0.f, 0.f}, 0.f};
    SpsvDefectExpansion im{{0.f, 0.f, 0.f, 0.f}, 0.f};
    SpsvComplex64 diagonal = data.nonUnit ? SpsvComplex64{data.values[data.diagPtr[row]].real,
                                                          data.values[data.diagPtr[row]].imag} :
                                            SpsvComplex64{1.f, 0.f};
    bool valid = SpsvComplexTightDefectTerms(data, row, diagonal, re, im);
    if (!valid) {
        return false;
    }

    SpsvTriplePrefixScalar r{0.f, 0.f, 0.f}, i{0.f, 0.f, 0.f};
    float lost = 0.f;
    for (int32_t j = 3; j >= 0; --j) {
        SpsvTriplePrefixAcc(r, re.word[j], lost);
        SpsvTriplePrefixAcc(i, im.word[j], lost);
    }
    float n = static_cast<float>(data.rowPtr[row + 1] - data.rowPtr[row] + 4);
    float radius = SpsvFloatBoundUp((re.discarded + im.discarded + lost) * (1.f + n * 0x1p-16f) + n * 0x1p-110f);
    if (!SpsvIsFiniteF(r.hi) || !SpsvIsFiniteF(r.lo) || !SpsvIsFiniteF(r.tail) || !SpsvIsFiniteF(i.hi) ||
        !SpsvIsFiniteF(i.lo) || !SpsvIsFiniteF(i.tail) || !SpsvIsFiniteF(radius)) {
        return false;
    }
    out[0] = r.hi;
    out[1] = r.lo;
    out[2] = r.tail;
    out[3] = i.hi;
    out[4] = i.lo;
    out[5] = i.tail;
    out[6] = radius;
    return true;
}

__simt_callee__ inline SpsvTriplePrefixValue SpsvTriplePrefixAdd(SpsvTriplePrefixValue a, SpsvTriplePrefixValue b)
{
    if (SpsvTriplePrefixIsZero(a)) {
        return b;
    }
    if (SpsvTriplePrefixIsZero(b)) {
        return a;
    }
    float lost = 0.f;
    SpsvTriplePrefixAcc(a.re, b.re.hi, lost);
    SpsvTriplePrefixAcc(a.re, b.re.lo, lost);
    SpsvTriplePrefixAcc(a.re, b.re.tail, lost);
    SpsvTriplePrefixAcc(a.im, b.im.hi, lost);
    SpsvTriplePrefixAcc(a.im, b.im.lo, lost);
    SpsvTriplePrefixAcc(a.im, b.im.tail, lost);
    a.radius = SpsvFloatBoundAdd(SpsvFloatBoundAdd(a.radius, b.radius),
                                 SpsvFloatBoundUp(lost * (1.f + 0x1p-10f) + 0x1p-108f));
    return a;
}
__simt_callee__ inline void SpsvTriplePrefixProductWord(SpsvTriplePrefixScalar& s, float a, SpsvTriplePrefixScalar b,
                                                        float sign, float& lost)
{
    if (a != 0.f) {
        if (b.hi != 0.f) {
            float product = a * b.hi;
            SpsvTriplePrefixAcc(s, sign * product, lost);
            SpsvTriplePrefixAcc(s, sign * fmaf(a, b.hi, -product), lost);
        }
        if (b.lo != 0.f) {
            float product = a * b.lo;
            SpsvTriplePrefixAcc(s, sign * product, lost);
            SpsvTriplePrefixAcc(s, sign * fmaf(a, b.lo, -product), lost);
        }
        if (b.tail != 0.f) {
            float product = a * b.tail;
            SpsvTriplePrefixAcc(s, sign * product, lost);
            SpsvTriplePrefixAcc(s, sign * fmaf(a, b.tail, -product), lost);
        }
    }
}

__simt_callee__ inline void SpsvTriplePrefixProductAcc(SpsvTriplePrefixScalar& s, SpsvTriplePrefixScalar a,
                                                       SpsvTriplePrefixScalar b, float sign, float& lost)
{
    // Keep high, low and tail products in their original order.
    SpsvTriplePrefixProductWord(s, a.hi, b, sign, lost);
    SpsvTriplePrefixProductWord(s, a.lo, b, sign, lost);
    SpsvTriplePrefixProductWord(s, a.tail, b, sign, lost);
}
__simt_callee__ inline SpsvTriplePrefixValue SpsvTriplePrefixMul(SpsvTriplePrefixValue a, SpsvTriplePrefixValue b)
{
    if (SpsvTriplePrefixIsZero(a) || SpsvTriplePrefixIsZero(b)) {
        return SpsvTriplePrefixZero();
    }
    // Multiplication by the exact complex identity introduces no error.
    if (a.re.hi == 1.f && a.re.lo == 0.f && a.re.tail == 0.f && a.im.hi == 0.f && a.im.lo == 0.f && a.im.tail == 0.f &&
        a.radius == 0.f) {
        return b;
    }
    if (b.re.hi == 1.f && b.re.lo == 0.f && b.re.tail == 0.f && b.im.hi == 0.f && b.im.lo == 0.f && b.im.tail == 0.f &&
        b.radius == 0.f) {
        return a;
    }
    SpsvTriplePrefixValue z = SpsvTriplePrefixZero();
    float lost = 0.f;
    SpsvTriplePrefixProductAcc(z.re, a.re, b.re, 1.f, lost);
    SpsvTriplePrefixProductAcc(z.re, a.im, b.im, -1.f, lost);
    SpsvTriplePrefixProductAcc(z.im, a.re, b.im, 1.f, lost);
    SpsvTriplePrefixProductAcc(z.im, a.im, b.re, 1.f, lost);
    z.radius = SpsvFloatBoundAdd(SpsvFloatBoundAdd(SpsvFloatBoundMul(SpsvTriplePrefixNorm(a), b.radius),
                                                   SpsvFloatBoundMul(SpsvTriplePrefixNorm(b), a.radius)),
                                 SpsvFloatBoundAdd(SpsvFloatBoundMul(2.f, SpsvFloatBoundMul(a.radius, b.radius)),
                                                   SpsvFloatBoundUp(lost * (1.f + 0x1p-10f) + 0x1p-104f)));
    return z;
}
// Compose two-word operands directly into a three-word center. TwoSum
// keeps the leading cross products exact; only the final tail sum and the
// low*low FMA residual enter the outward radius. No normalization of the
// input operands is assumed. The original full-product attempt remains.
__simt_callee__ inline SpsvTriplePrefixScalar SpsvTreePairProduct(SpsvTriplePrefixScalar a, SpsvTriplePrefixScalar b,
                                                                  float& discarded)
{
    float high = a.hi * b.hi;
    float residual = fmaf(a.hi, b.hi, -high);
    float cross0 = a.hi * b.lo, cross1 = a.lo * b.hi, lowProduct = a.lo * b.lo;
    float error0 = fmaf(a.hi, b.lo, -cross0), error1 = fmaf(a.lo, b.hi, -cross1);
    float lowResidual = fmaf(a.lo, b.lo, -lowProduct);
    SpsvDF cross = SpsvTwoSum(cross0, cross1);
    SpsvDF middle = SpsvTwoSum(residual, cross.hi);
    SpsvDF head = SpsvTwoSum(high, middle.hi);
    SpsvDF low0 = SpsvTwoSum(head.lo, middle.lo);
    SpsvDF low1 = SpsvTwoSum(low0.hi, cross.lo);
    float tail0 = low0.lo + low1.lo;
    float tail1 = tail0 + error0;
    float tail2 = tail1 + error1;
    float tail = tail2 + lowProduct;
    float cost = SpsvAbsF(tail0) + SpsvAbsF(tail1) + SpsvAbsF(tail2) + SpsvAbsF(tail);
    float error = SpsvFloatBoundAdd(SpsvAbsF(lowResidual), SpsvFloatBoundUp(cost * 0x1p-20f + 0x1p-104f));
    discarded = SpsvFloatBoundAdd(discarded, error);
    return {head.hi, low1.hi, tail};
}

__simt_callee__ inline void SpsvTriplePrefixScalarProductAcc(SpsvTriplePrefixScalar& s, SpsvTriplePrefixScalar b,
                                                             float scalar, float sign, float& lost)
{
    if (b.hi != 0.f) {
        float product = scalar * b.hi;
        SpsvTriplePrefixAcc(s, sign * product, lost);
        SpsvTriplePrefixAcc(s, sign * fmaf(scalar, b.hi, -product), lost);
    }
    if (b.lo != 0.f) {
        float product = scalar * b.lo;
        SpsvTriplePrefixAcc(s, sign * product, lost);
        SpsvTriplePrefixAcc(s, sign * fmaf(scalar, b.lo, -product), lost);
    }
    if (b.tail != 0.f) {
        float product = scalar * b.tail;
        SpsvTriplePrefixAcc(s, sign * product, lost);
        SpsvTriplePrefixAcc(s, sign * fmaf(scalar, b.tail, -product), lost);
    }
}

// Specialized exact enclosure for a one-word complex coefficient times a triple value.
__simt_callee__ inline SpsvTriplePrefixValue SpsvTriplePrefixMulComplexScalar(float ar, float ai,
                                                                              SpsvTriplePrefixValue b)
{
    if ((ar == 0.f && ai == 0.f) || SpsvTriplePrefixIsZero(b)) {
        return SpsvTriplePrefixZero();
    }
    SpsvTriplePrefixValue z = SpsvTriplePrefixZero();
    float lost = 0.f;
    SpsvTriplePrefixScalarProductAcc(z.re, b.re, ar, 1.f, lost);
    SpsvTriplePrefixScalarProductAcc(z.re, b.im, ai, -1.f, lost);
    SpsvTriplePrefixScalarProductAcc(z.im, b.im, ar, 1.f, lost);
    SpsvTriplePrefixScalarProductAcc(z.im, b.re, ai, 1.f, lost);
    float norm = SpsvFloatBoundAdd(SpsvAbsF(ar), SpsvAbsF(ai));
    z.radius = SpsvFloatBoundAdd(SpsvFloatBoundMul(norm, b.radius),
                                 SpsvFloatBoundUp(lost * (1.f + 0x1p-10f) + 0x1p-104f));
    return z;
}

template <bool SHORT_PRODUCT>
__simt_callee__ inline SpsvTriplePrefixValue SpsvTriplePrefixTreeMul(SpsvTriplePrefixValue a, SpsvTriplePrefixValue b)
{
    if constexpr (SHORT_PRODUCT) {
        if (SpsvTriplePrefixIsZero(a) || SpsvTriplePrefixIsZero(b)) {
            return SpsvTriplePrefixZero();
        }
        float tailA = SpsvFloatBoundAdd(SpsvAbsF(a.re.tail), SpsvAbsF(a.im.tail));
        float tailB = SpsvFloatBoundAdd(SpsvAbsF(b.re.tail), SpsvAbsF(b.im.tail));
        a.radius = SpsvFloatBoundAdd(a.radius, tailA);
        b.radius = SpsvFloatBoundAdd(b.radius, tailB);
        a.re.tail = a.im.tail = b.re.tail = b.im.tail = 0.f;
        float local = 0.f;
        auto rr = SpsvTreePairProduct(a.re, b.re, local);
        auto ii = SpsvTreePairProduct(a.im, b.im, local);
        auto ri = SpsvTreePairProduct(a.re, b.im, local);
        auto ir = SpsvTreePairProduct(a.im, b.re, local);
        auto center = SpsvTriplePrefixAdd({rr, ri, local}, {{-ii.hi, -ii.lo, -ii.tail}, ir, 0.f});
        float propagated = SpsvFloatBoundAdd(SpsvFloatBoundMul(SpsvTriplePrefixNorm(a), b.radius),
                                             SpsvFloatBoundMul(SpsvTriplePrefixNorm(b), a.radius));
        center.radius = SpsvFloatBoundAdd(
            propagated,
            SpsvFloatBoundAdd(SpsvFloatBoundMul(2.f, SpsvFloatBoundMul(a.radius, b.radius)), center.radius));
        return center;
    }
    return SpsvTriplePrefixMul(a, b);
}

__simt_callee__ inline SpsvTriplePrefixValue SpsvTriplePrefixNeg(SpsvTriplePrefixValue a)
{
    a.re.hi = -a.re.hi;
    a.re.lo = -a.re.lo;
    a.re.tail = -a.re.tail;
    a.im.hi = -a.im.hi;
    a.im.lo = -a.im.lo;
    a.im.tail = -a.im.tail;
    return a;
}
__simt_callee__ inline SpsvTriplePrefixValue SpsvTriplePrefixVerifiedInverse(const SpsvComplexCertificateData& data,
                                                                             int64_t m, int64_t row)
{
    if (!data.nonUnit) {
        return {{1.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, 0.f};
    }
    __gm__ const float* prep = data.wide + 6 * m + 18 * row;
    if (data.diagPtr[row] < 0) {
        return {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, __builtin_inff()};
    }
    SpsvTriplePrefixValue q{{prep[11], prep[12], 0.f}, {prep[13], prep[14], 0.f}, 0.f};
    SpsvTriplePrefixValue d{
        {data.values[data.diagPtr[row]].real, 0.f, 0.f}, {data.values[data.diagPtr[row]].imag, 0.f, 0.f}, 0.f};
    SpsvTriplePrefixValue one{{1.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, 0.f};
    SpsvTriplePrefixValue residual = SpsvTriplePrefixAdd(one, SpsvTriplePrefixNeg(SpsvTriplePrefixMul(d, q)));
    // Newton only proposes a center; the independent residual below
    // certifies that proposal, including all rounding in its construction.
    residual.radius = 0.f;
    q = SpsvTriplePrefixAdd(q, SpsvTriplePrefixMul(q, residual));
    q.radius = 0.f;
    residual = SpsvTriplePrefixAdd(one, SpsvTriplePrefixNeg(SpsvTriplePrefixMul(d, q)));
    q.radius = SpsvFloatBoundMul(SpsvFloatBoundAdd(SpsvTriplePrefixNorm(residual), residual.radius), prep[8]);
    return q;
}

__simt_callee__ inline float SpsvTriplePrefixTightScalarNorm(SpsvTriplePrefixScalar a)
{
    SpsvDF x = SpsvTwoSum(a.hi, a.lo);
    SpsvDF y = SpsvTwoSum(x.lo, a.tail);
    SpsvDF z = SpsvTwoSum(x.hi, y.hi);
    return SpsvFloatBoundAdd(SpsvFloatBoundAdd(SpsvAbsF(z.hi), SpsvAbsF(z.lo)),
                             SpsvFloatBoundAdd(SpsvAbsF(y.lo), 0x1p-108f));
}
__simt_callee__ inline float SpsvTriplePrefixTightCenterNorm(SpsvTriplePrefixValue a)
{
    return SpsvFloatBoundAdd(SpsvTriplePrefixTightScalarNorm(a.re), SpsvTriplePrefixTightScalarNorm(a.im));
}

__simt_callee__ inline bool SpsvTriplePrefixCorrectComponent(float& high, float& low, float& bound,
                                                             SpsvTriplePrefixScalar correction, float radius)
{
    SpsvTriplePrefixValue value{{high, low, 0.f}, {0.f, 0.f, 0.f}, 0.f};
    SpsvTriplePrefixValue delta{correction, {0.f, 0.f, 0.f}, radius};
    value = SpsvTriplePrefixAdd(value, delta);
    SpsvDF proposed = SpsvTwoSum(value.re.hi, value.re.lo);
    proposed = SpsvTwoSum(proposed.hi, proposed.lo + value.re.tail);
    SpsvTriplePrefixValue represented{{proposed.hi, proposed.lo, 0.f}, {0.f, 0.f, 0.f}, 0.f};
    SpsvTriplePrefixValue difference = SpsvTriplePrefixAdd(value, SpsvTriplePrefixNeg(represented));
    float enclosure = SpsvFloatBoundAdd(SpsvTriplePrefixTightCenterNorm(difference), difference.radius);
    if (SpsvFloatNeedsDependencyRetry(proposed.hi, proposed.lo, enclosure)) {
        return false;
    }
    high = proposed.hi;
    low = proposed.lo;
    bound = enclosure;
    return true;
}

#endif // SPSV_COMPLEX_TRIPLE_ARITH_H_
