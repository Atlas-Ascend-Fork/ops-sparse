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

#ifndef OP_KERNEL_VALIDATION_SPSV_ENCLOSED_ARITHMETIC_H_
#define OP_KERNEL_VALIDATION_SPSV_ENCLOSED_ARITHMETIC_H_

// BEGIN SPSV_CERTIFIED_COMPLEX_RETRY
// The bounded retry is an optional filter before the full exponent-domain
// integer accumulator. Every multiplication must have an exact float/FMA
// decomposition on flush-to-zero hardware. The smallest product bit is
// 2^(ea+eb-46), hence ea+eb >= -78 keeps it normal. The upper gate
// keeps each product finite. A nonfinite intermediate sum poisons
// its TwoSum residual and is rejected by the final error certificate.
__simt_callee__ __attribute__((always_inline)) inline bool SpsvCertifiedProduct(float a, float b, SpsvDF& product,
                                                                                float& discarded)
{
    uint32_t bitsA, bitsB;
    __builtin_memcpy(&bitsA, &a, sizeof(bitsA));
    __builtin_memcpy(&bitsB, &b, sizeof(bitsB));
    bitsA &= 0x7FFFFFFFu;
    bitsB &= 0x7FFFFFFFu;
    if (bitsA >= 0x7F800000u || bitsB >= 0x7F800000u || (bitsA != 0 && bitsA < 0x00800000u) ||
        (bitsB != 0 && bitsB < 0x00800000u)) {
        return false;
    }
    if (bitsA == 0 || bitsB == 0) {
        product = {0.0f, 0.0f};
        return true;
    }
    int32_t exponent = static_cast<int32_t>(bitsA >> 23) + static_cast<int32_t>(bitsB >> 23) - 254;
    if (exponent > 125) {
        return false;
    }
    float high = a * b;
    product = {high, fmaf(a, b, -high)};
    // A normal operand can produce a subnormal product or FMA residual.
    // Each flushed result has magnitude below FLT_MIN; enclose both rather
    // than rejecting a whole long row for a negligible low-word product.
    discarded += 0x1p-124f;
    return true;
}

__simt_callee__ __attribute__((always_inline)) inline SpsvDF SpsvCertifiedComplexAdd(SpsvDF a, SpsvDF b,
                                                                                     float& discarded)
{
    SpsvDF high = SpsvTwoSum(a.hi, b.hi);
    SpsvDF low = SpsvTwoSum(a.lo, b.lo);
    SpsvDF middle = SpsvTwoSum(high.lo, low.hi);
    SpsvDF head = SpsvTwoSum(high.hi, middle.hi);
    // Feed the third words back into the low word before enclosing only
    // their unrepresented tails. Do not discard recoverable cancellation.
    SpsvDF tail = SpsvTwoSum(head.lo, middle.lo);
    SpsvDF last = SpsvTwoSum(tail.hi, low.lo);
    SpsvDF result = SpsvTwoSum(head.hi, last.hi);
    discarded += SpsvAbsF(tail.lo) + SpsvAbsF(last.lo);
    return result;
}

__simt_callee__ __attribute__((always_inline)) inline bool SpsvCertifiedComplexTerm(SpsvDF& sum, float& discarded,
                                                                                    float a, float b)
{
    SpsvDF product;
    if (!SpsvCertifiedProduct(a, b, product, discarded)) {
        return false;
    }
    sum = SpsvCertifiedComplexAdd(sum, product, discarded);
    return true;
}

__simt_callee__ __attribute__((always_inline)) inline bool SpsvCertifiedComplexResidual(
    SpsvDFComplex numerator, SpsvDFComplex quotient, SpsvComplex64 diagonal, float& discardedRe, float& discardedIm,
    SpsvDFComplex& residual)
{
    residual = numerator;
    // Numerator - diagonal * (quotient.hi + quotient.lo), with every
    // product and every omitted third word enclosed independently.
    return SpsvCertifiedComplexTerm(residual.re, discardedRe, -diagonal.real, quotient.re.hi) &&
           SpsvCertifiedComplexTerm(residual.re, discardedRe, -diagonal.real, quotient.re.lo) &&
           SpsvCertifiedComplexTerm(residual.re, discardedRe, diagonal.imag, quotient.im.hi) &&
           SpsvCertifiedComplexTerm(residual.re, discardedRe, diagonal.imag, quotient.im.lo) &&
           SpsvCertifiedComplexTerm(residual.im, discardedIm, -diagonal.real, quotient.im.hi) &&
           SpsvCertifiedComplexTerm(residual.im, discardedIm, -diagonal.real, quotient.im.lo) &&
           SpsvCertifiedComplexTerm(residual.im, discardedIm, -diagonal.imag, quotient.re.hi) &&
           SpsvCertifiedComplexTerm(residual.im, discardedIm, -diagonal.imag, quotient.re.lo);
}

__simt_callee__ __attribute__((always_inline)) inline bool SpsvCertifiedComplexQuotient(
    SpsvDFComplex numerator, float discardedRe, float discardedIm, SpsvComplex64 diagonal, SpsvDFComplex& quotient,
    bool taskTolerance = false)
{
    numerator.re = SpsvTwoSum(numerator.re.hi, numerator.re.lo);
    numerator.im = SpsvTwoSum(numerator.im.hi, numerator.im.lo);
    float scale = SpsvMaxAbs2(diagonal.real, diagonal.imag);
    if (scale == 0.0f || !(scale >= 0x1p-24f && scale <= 0x1p24f)) {
        return false;
    }
    float dr = diagonal.real / scale, di = diagonal.imag / scale;
    float inverse = 1.0f / (dr * dr + di * di);
    SpsvComplex64 reciprocal{dr * inverse, -di * inverse};
    SpsvComplex64 q = SpsvComplexDivByDiag({numerator.re.hi, numerator.im.hi}, reciprocal, scale);
    quotient = {{q.real, 0.0f}, {q.imag, 0.0f}};
    SpsvDFComplex residual;
    float ignoredRe = discardedRe, ignoredIm = discardedIm;
    if (!SpsvCertifiedComplexResidual(numerator, quotient, diagonal, ignoredRe, ignoredIm, residual)) {
        return false;
    }
    SpsvComplex64 correction = SpsvComplexDivByDiag({residual.re.hi + residual.re.lo, residual.im.hi + residual.im.lo},
                                                    reciprocal, scale);
    quotient.re = SpsvTwoSum(q.real, correction.real);
    quotient.im = SpsvTwoSum(q.imag, correction.imag);
    if (!SpsvCertifiedComplexResidual(numerator, quotient, diagonal, discardedRe, discardedIm, residual)) {
        return false;
    }
    // |1/diag|_1 <= 2 / max(|dr|, |di|). Four covers rounding of the
    // positive error sums (fewer than 2^24 terms); a second factor of four
    // encloses this norm bound and the final division. The tiny floor also
    // encloses a flushed bound without affecting the acceptance threshold.
    float errorRe = 4.0f * (SpsvAbsF(residual.re.hi) + SpsvAbsF(residual.re.lo) + discardedRe);
    float errorIm = 4.0f * (SpsvAbsF(residual.im.hi) + SpsvAbsF(residual.im.lo) + discardedIm);
    float error = 4.0f * (errorRe + errorIm) / scale + 0x1p-60f;
    float relativeBudget = taskTolerance ? 0x1p-19f : 0x1p-32f;
    float floorBudget = taskTolerance ? 1.0e-2f : 0x1p-40f;
    float budgetRe = SpsvAbsF(quotient.re.hi) * relativeBudget;
    float budgetIm = SpsvAbsF(quotient.im.hi) * relativeBudget;
    budgetRe = budgetRe > floorBudget ? budgetRe : floorBudget;
    budgetIm = budgetIm > floorBudget ? budgetIm : floorBudget;
    return SpsvIsFiniteF(error) && error <= budgetRe && error <= budgetIm;
}

// Carry a third float through the retry. TwoSum propagates every omitted
// word to the next tier; only a fourth word enters the positive error bound.
// The product gate keeps the least product bit normal on FTZ hardware.
struct SpsvTriple {
    float hi = 0.0f;
    float mid = 0.0f;
    float lo = 0.0f;
};

__simt_callee__ inline void SpsvTripleAdd(SpsvTriple& sum, float value, float& discarded)
{
    SpsvDF head = SpsvTwoSum(sum.hi, value);
    SpsvDF middle = SpsvTwoSum(sum.mid, head.lo);
    SpsvDF tail = SpsvTwoSum(sum.lo, middle.lo);
    sum.hi = head.hi;
    sum.mid = middle.hi;
    sum.lo = tail.hi;
    discarded += SpsvAbsF(tail.lo);
}

__simt_callee__ inline bool SpsvTripleTerm(SpsvTriple& sum, float& discarded, float a, float b)
{
    SpsvDF product;
    if (!SpsvCertifiedProduct(a, b, product, discarded)) {
        return false;
    }
    SpsvTripleAdd(sum, product.hi, discarded);
    SpsvTripleAdd(sum, product.lo, discarded);
    return true;
}

__simt_callee__ inline SpsvDF SpsvTripleFinish(SpsvTriple sum, float& discarded)
{
    return SpsvCertifiedComplexAdd({sum.hi, 0.0f}, {sum.mid, sum.lo}, discarded);
}

// The error accumulator is in units of single-precision roundoff.
// Each rounded low-word operation contributes its computed magnitude.
__simt_callee__ inline void SpsvAccEnclosed(float& hi, float& lo, float& error, float a, float yh, float yl)
{
    float ph = a * yh;
    float pl = a * yl;
    float productLow = fmaf(a, yh, -ph) + pl;
    SpsvDF head = SpsvTwoSum(hi, ph);
    float correction = head.lo + productLow;
    float low = lo + correction;
    // Track the magnitudes of the actual rounded low-word operations.
    // This also encloses lazy low sums without assuming an O(n) norm bound.
    error += SpsvAbsF(pl) + SpsvAbsF(productLow) + SpsvAbsF(correction) + SpsvAbsF(low);
    hi = head.hi;
    lo = low;
}

__simt_callee__ inline SpsvDF SpsvMergeEnclosed(SpsvDF a, SpsvDF b, float& error)
{
    SpsvDF head = SpsvTwoSum(a.hi, b.hi);
    float low = a.lo + b.lo;
    float middle = head.lo + low;
    error += SpsvAbsF(low) + SpsvAbsF(middle);
    return SpsvTwoSum(head.hi, middle);
}

__simt_callee__ inline SpsvDF SpsvWarpEnclosed(SpsvDF sum, float& error, int32_t width = 32)
{
    // Delay normalization and collect every lane's nonnegative merge cost.
    // Summing even the duplicate butterfly nodes is conservative and avoids
    // carrying a third value through every shuffle stage.
    for (int32_t mask = 1; mask < width; mask <<= 1) {
        SpsvDF other{asc_shfl_xor(sum.hi, mask), asc_shfl_xor(sum.lo, mask)};
        SpsvDF head = SpsvTwoSum(sum.hi, other.hi);
        float low = sum.lo + other.lo;
        float middle = head.lo + low;
        error += SpsvAbsF(low) + SpsvAbsF(middle);
        sum = {head.hi, middle};
    }
    error = asc_reduce_add(error);
    sum = SpsvTwoSum(sum.hi, sum.lo);
    return {asc_shfl(sum.hi, 0), asc_shfl(sum.lo, 0)};
}

// Each small group owns a different row. Keep both the solution and its
// nonnegative error bound inside that group; a warp-wide reduction would mix
// unrelated rows. Butterfly merge costs are deliberately counted repeatedly.
template <int GROUP>
__simt_callee__ inline SpsvDF SpsvGroupEnclosed(SpsvDF sum, float& error)
{
    for (int32_t mask = 1; mask < GROUP; mask <<= 1) {
        SpsvDF other{asc_shfl_xor(sum.hi, mask), asc_shfl_xor(sum.lo, mask)};
        SpsvDF head = SpsvTwoSum(sum.hi, other.hi);
        float low = sum.lo + other.lo;
        float middle = head.lo + low;
        error += SpsvAbsF(low) + SpsvAbsF(middle);
        sum = {head.hi, middle};
    }
    for (int32_t mask = 1; mask < GROUP; mask <<= 1) {
        error += asc_shfl_xor(error, mask);
    }
    sum = SpsvTwoSum(sum.hi, sum.lo);
    int32_t leader = static_cast<int32_t>(threadIdx.x) & (31 & ~(GROUP - 1));
    return {asc_shfl(sum.hi, leader), asc_shfl(sum.lo, leader)};
}

// BEGIN SPSV_REAL_WIDE_CERTIFICATE
// A normal float product has at most 48 significant bits. Rejecting a
// combined exponent below -78 keeps its last bit above the FTZ boundary.
// The caller reserves exponent headroom for every product in the whole row.
__simt_callee__ __attribute__((always_inline)) inline bool SpsvRealWideProduct(float a, float b, int32_t upper,
                                                                               SpsvDF& product)
{
    uint32_t ba, bb;
    __builtin_memcpy(&ba, &a, sizeof(ba));
    __builtin_memcpy(&bb, &b, sizeof(bb));
    ba &= 0x7FFFFFFFu;
    bb &= 0x7FFFFFFFu;
    if (ba >= 0x7F800000u || bb >= 0x7F800000u || (ba != 0 && ba < 0x00800000u) || (bb != 0 && bb < 0x00800000u)) {
        return false;
    }
    if (ba == 0 || bb == 0) {
        product = {0.0f, 0.0f};
        return true;
    }
    int32_t exponent = static_cast<int32_t>(ba >> 23) + static_cast<int32_t>(bb >> 23) - 254;
    if (exponent < -78 || exponent > upper) {
        return false;
    }
    float high = a * b;
    product = {high, fmaf(a, b, -high)};
    return true;
}

__simt_callee__ __attribute__((always_inline)) inline bool SpsvRealWideTerm(SpsvDF& sum, float& discarded, float a,
                                                                            float b, int32_t upper)
{
    SpsvDF product;
    if (!SpsvRealWideProduct(a, b, upper, product)) {
        return false;
    }
    sum = SpsvCertifiedComplexAdd(sum, product, discarded);
    return true;
}

__simt_callee__ __attribute__((always_inline)) inline bool SpsvRealWideQuotient(SpsvDF numerator, float discarded,
                                                                                float diagonal, SpsvDF& answer)
{
    if (!SpsvIsFiniteF(diagonal) || SpsvAbsF(diagonal) < 0x1p-24f || SpsvAbsF(diagonal) > 0x1p24f) {
        return false;
    }
    numerator = SpsvTwoSum(numerator.hi, numerator.lo);
    answer = SpsvDFDivF(numerator, diagonal);
    SpsvDF residual = numerator;
    if (!SpsvRealWideTerm(residual, discarded, -diagonal, answer.hi, 123) ||
        !SpsvRealWideTerm(residual, discarded, -diagonal, answer.lo, 123)) {
        return false;
    }
    // Positive sums contain fewer than 2^24 additions. Eight encloses their
    // rounding, the residual norm, and the last division. The absolute floor
    // covers an error-bound term flushed by the target arithmetic.
    float error = 8.0f * (SpsvAbsF(residual.hi) + SpsvAbsF(residual.lo) + discarded) / SpsvAbsF(diagonal) + 0x1p-60f;
    float budget = SpsvAbsF(answer.hi) * 0x1p-32f;
    budget = budget > 0x1p-40f ? budget : 0x1p-40f;
    return SpsvIsFiniteF(error) && error <= budget;
}

template <bool WARP>
__simt_callee__ inline void SpsvTripleWarpMerge(SpsvTriple& sum, float& discarded, uint32_t& rejected)
{
    if constexpr (WARP) {
        int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
        for (int32_t offset = 16; offset > 0; offset >>= 1) {
            SpsvTriple other{asc_shfl_xor(sum.hi, offset), asc_shfl_xor(sum.mid, offset), asc_shfl_xor(sum.lo, offset)};
            float otherDiscarded = asc_shfl_xor(discarded, offset);
            uint32_t otherRejected = asc_shfl_xor(rejected, offset);
            if (lane < offset) {
                SpsvTripleAdd(sum, other.hi, discarded);
                SpsvTripleAdd(sum, other.mid, discarded);
                SpsvTripleAdd(sum, other.lo, discarded);
                discarded += otherDiscarded;
                rejected |= otherRejected;
            }
        }
    }
}

#endif // OP_KERNEL_VALIDATION_SPSV_ENCLOSED_ARITHMETIC_H_
