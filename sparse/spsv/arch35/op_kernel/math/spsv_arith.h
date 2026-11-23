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

/* complex64 and float-float (hi, lo) arithmetic used by the complex solve. */

#ifndef SPSV_ARITH_H_
#define SPSV_ARITH_H_

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"

// Matches the .cpp this was split out of, and the repo's other
// arch35 implementation headers.
using namespace AscendC;

// I32 offsets are nonnegative and at most INT32_MAX. An unsigned cursor
// also represents the terminal start+lane / p+stride (at most INT32_MAX
// plus the launch width), without signed overflow or 64-bit loop overhead.
// I64 descriptors retain their full offset range.
template <typename RowPtrT>
using SpsvCsrCursor = typename std::conditional<sizeof(RowPtrT) == sizeof(int32_t), uint32_t, int64_t>::type;

// Complex64 device value used by the dedicated CSR/zero-based path.  Keep the
// representation binary-compatible with aclsparseComplex without including
// the host API header in device compilation.
struct SpsvComplex64 {
    float real;
    float imag;
};

// Copy one stored value.  A __gm__-qualified struct gets no implicit
// operator=, so the conversions that are shared between the real and complex
// paths go through this rather than a plain assignment.
__simt_callee__ inline void SpsvAssignValue(__gm__ float* dst, __gm__ const float* src) { *dst = *src; }

__simt_callee__ inline void SpsvAssignValue(__gm__ SpsvComplex64* dst, __gm__ const SpsvComplex64* src)
{
    dst->real = src->real;
    dst->imag = src->imag;
}

__simt_callee__ inline SpsvComplex64 SpsvComplexMul(SpsvComplex64 a, SpsvComplex64 b)
{
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

__simt_callee__ inline float SpsvAbsF(float x) { return fabsf(x); }

// False for an infinity and for a NaN, both of which fail the comparison.
__simt_callee__ inline bool SpsvIsFiniteF(float x) { return SpsvAbsF(x) <= 3.4028235e38f; }

// The larger of two magnitudes.  Used to normalise a complex value by its
// dominant component.
__simt_callee__ inline float SpsvMaxAbs2(float a, float b)
{
    float x = SpsvAbsF(a);
    float y = SpsvAbsF(b);
    return (x >= y) ? x : y;
}

// 2^e as a float, for -126 <= e <= 127: the exponent field, written.  Used
// to build a row's scale factor once and apply it with a multiply, which is
// exact and is not a call.
__simt_callee__ inline float SpsvPow2(int32_t e)
{
    uint32_t bits = static_cast<uint32_t>(e + 127) << 23;
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

// The binary exponent to normalise by, or zero when there is nothing to
// normalise.  This part flushes subnormals to zero, so a non-zero finite
// float is normal and its exponent is the exponent field; ilogbf would read
// the same number with a call, and the scan below asks for two of these per
// term.  The guard is here in one place so that a zero, an infinity or a NaN
// is left alone.
__simt_callee__ inline int32_t SpsvExpOf(float x)
{
    float m = SpsvAbsF(x);
    if (!(m > 0.0f && SpsvIsFiniteF(m))) {
        return 0;
    }
    uint32_t bits;
    __builtin_memcpy(&bits, &m, sizeof(bits));
    return static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
}

// 2^e as a pair of factors, because the shift a row accumulates in can go
// below the smallest representable power of two.  Two of them span
// [-252, 254] and e is saturated into that span first: outside it the second
// exponent leaves SpsvPow2's range, where the field is written through and
// wraps -- SpsvPow2(-128) is 0xff800000, negative infinity, and a right-hand
// side that should have flushed to zero poisons the row instead.
//
// Saturating is what every caller wants, because each hands over an operand
// normalised to a mantissa: below the span, 2^-252 times one is under the
// smallest normal and flushes to zero, which is what the arithmetic would
// have done; above it, 2^254 times one overflows, likewise.  Inside the span
// both factors are exact, so applying them in turn is exact.
struct SpsvScalePair {
    float f1;
    float f2;
};

__simt_callee__ inline SpsvScalePair SpsvScaleFactors(int32_t e)
{
    int32_t c = (e < -252) ? -252 : ((e > 254) ? 254 : e);
    int32_t e1 = (c < -126) ? -126 : ((c > 127) ? 127 : c);
    return SpsvScalePair{SpsvPow2(e1), SpsvPow2(c - e1)};
}

// The exponent to normalise a complex value by: that of its larger component.
__simt_callee__ inline int32_t SpsvExpOfMax2(float a, float b) { return SpsvExpOf(SpsvMaxAbs2(a, b)); }

// v / diag, with the diagonal held as a normalised reciprocal and a separate
// scale.  Which side of the multiply the scale goes on follows the scale, and
// that is exact in range whenever the quotient itself is representable:
//
//   scale >= 1  divide first.  The division cannot overflow, and each of the
//               two products that make up a component is bounded by the
//               divided operand, so neither leaves range before their
//               difference -- the answer -- is formed.
//   scale <  1  multiply first.  The reciprocal's modulus is at most one so
//               neither product can grow, and the division that follows
//               produces the answer directly.
//
// Both branches of the finish go through here, which is what the pre-divided
// branch did not do before: it multiplied first whatever the scale was, and a
// finite dot product of -3e38 in both components over a diagonal of 2+i then
// formed -3.6e38.
//
// What ordering cannot reach is a quotient that is itself out of range while
// something later brings it back.  That is the precompute's recovery, and it
// goes through SpsvComplexScaledProductDivByDiag instead: every product
// normalised by its own two operands' exponents, aligned only inside the
// sums.
//
// Operand and scale are brought into [1, 2) by exact powers of two before the
// multiply, and a single power-of-two rescale closes it.  That bounds every
// intermediate: the normalised operand's components are below 2 and the
// reciprocal's modulus is at most 1, so a component of the product -- a sum
// of two terms -- cannot exceed 4, and dividing by a scale in [1, 2) cannot
// grow it.  The closing rescale leaves range exactly when the answer does.
//
// Choosing an order of multiply and divide instead does not work.  Neither
// order is safe when the scale is near one, because a component of a complex
// product can reach sqrt(2) times the operand's modulus and that modulus may
// already be past FLT_MAX: alpha = 3e38(1+i) over a diagonal of 1+0.5i needs
// 3.6e38 whichever way round it is taken, while the answer, 3.12e38, is
// representable.
//
// Rows inside kSpsvDivFastMaxV / kSpsvDivFastMinScale skip the normalising
// entirely; see there for why the plain expression is safe for them and what
// normalising every row cost.
//
// extraExp is for a caller that has already scaled its operands by a power of
// two and needs the way back.  It belongs here rather than in a second step
// because an intermediate can be subnormal and this part flushes subnormals
// to zero: taken separately, 3e38 * 3e38 / 3e38 comes back as zero.
//
// A zero diagonal is a singular row and stays non-finite: the reciprocal is
// formed as diag / scale, so a zero scale has already made it NaN and that
// NaN carries through the multiply below.
__simt_callee__ inline SpsvComplex64 SpsvComplexDivByDiag(SpsvComplex64 v, SpsvComplex64 recip, float scale)
{
    // A zero scale means a zero diagonal, which is a singular row and has to
    // stay non-finite.  It does without dividing by zero here: the reciprocal
    // is formed as diag / scale, so a zero scale has already made it NaN, and
    // that NaN carries through the multiply below.
    float by = (scale != 0.0f) ? scale : 1.0f;
    if (scale >= 1.0f) {
        v.real /= by;
        v.imag /= by;
        return SpsvComplexMul(v, recip);
    }
    SpsvComplex64 q = SpsvComplexMul(v, recip);
    return SpsvComplex64{q.real / by, q.imag / by};
}

// Float-float helpers.  The AI core rejects double precision operations, so the
// complex substitution carries its running sum and the already-solved part of Y
// as an unevaluated (hi, lo) pair, which yields about 48 significand bits out of
// single-precision arithmetic alone.  Triangular solves with a unit diagonal
// amplify the solution by orders of magnitude, so its small components are
// formed by cancellation between large terms; plain fp32 leaves those with an
// absolute error well above the per-element ceiling the task book requires.
// Ordinary third-word rounding and range loss need different retry policies.
// Slot seven otherwise records nonnegative local loss flags. This negative
// token selects an untrusted proposal whose global bounds stay infinite until
// an independent original-equation certificate or exact recovery succeeds.
constexpr float kSpsvDeferredComplexCertificate = -256.0f;

constexpr uint32_t kSpsvRoundLoss = 1;
constexpr uint32_t kSpsvRangeLoss = 2;

struct SpsvDF {
    float hi;
    float lo;
    uint32_t lost = 0; // Sticky bitmask; neither class is cleared by a later add.
};

__simt_callee__ inline SpsvDF SpsvTwoSum(float a, float b)
{
    float s = a + b;
    float bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

__simt_callee__ inline SpsvDF SpsvTwoProd(float a, float b)
{
    float p = a * b;
    return {p, fmaf(a, b, -p),
            kSpsvRangeLoss * static_cast<uint32_t>((a != 0.0f && b != 0.0f && p == 0.0f) || !SpsvIsFiniteF(p) ||
                                                   (a != 0.0f && b != 0.0f && SpsvExpOf(a) + SpsvExpOf(b) < -80))};
}

__simt_callee__ inline SpsvDF SpsvDFNeg(SpsvDF a) { return {-a.hi, -a.lo, a.lost}; }

__simt_callee__ inline SpsvDF SpsvDFAdd(SpsvDF a, SpsvDF b)
{
    SpsvDF s = SpsvTwoSum(a.hi, b.hi);
    SpsvDF low = SpsvTwoSum(a.lo, b.lo);
    SpsvDF merge = SpsvTwoSum(s.lo, low.hi);
    SpsvDF out = SpsvTwoSum(s.hi, merge.hi);
    // Both low-word merges can discard a third tier, even when the high
    // word is finite. Never clear this bit after subsequent cancellation.
    out.lost = a.lost | b.lost | (kSpsvRoundLoss * static_cast<uint32_t>(low.lo != 0.0f || merge.lo != 0.0f));
    return out;
}

// (a.hi + a.lo) * b, where b is exact by construction -- a matrix entry,
// alpha, or a reciprocal computed once.
__simt_callee__ inline SpsvDF SpsvDFMulF(SpsvDF a, float b)
{
    SpsvDF p = SpsvTwoProd(a.hi, b);
    SpsvDF out = SpsvDFAdd(p, SpsvTwoProd(a.lo, b));
    out.lost |= a.lost;
    return out;
}

__simt_callee__ inline void SpsvAccF(float& hi, float& lo, float pHi, float pLo)
{
    SpsvDF t = SpsvTwoSum(hi, pHi);
    hi = t.hi;
    lo += t.lo + pLo;
}

// A float-float pair scaled by an exact power of two: exact in both words
// until the result leaves range, which is what lets a whole row be carried in
// a scaled domain and brought back at the end.
__simt_callee__ inline SpsvDF SpsvDFScale2(SpsvDF v, int32_t e)
{
    float hi = ldexpf(v.hi, e);
    float lo = ldexpf(v.lo, e);
    uint32_t lost = v.lost | (kSpsvRangeLoss * static_cast<uint32_t>(ldexpf(hi, -e) != v.hi || ldexpf(lo, -e) != v.lo));
    return SpsvDF{hi, lo, lost};
}

// SpsvDFScale2 for an exponent a row can actually produce: at most the bit
// width of its nonzero count.  Inside SpsvPow2's range the factor is a normal
// power of two, so the scaling is one multiply in each word and, unlike
// ldexpf, not a call.  Exact, for the reason SpsvDFScale2 gives.
__simt_callee__ inline SpsvDF SpsvDFScaleRow(SpsvDF v, int32_t e)
{
    float f = SpsvPow2(e);
    if (f == 0.0f || !SpsvIsFiniteF(f)) {
        return SpsvDFScale2(v, e);
    }
    float hi = v.hi * f;
    float lo = v.lo * f;
    return SpsvDF{hi, lo, v.lost | (kSpsvRangeLoss * static_cast<uint32_t>(hi / f != v.hi || lo / f != v.lo))};
}

// The float-float twin of SpsvComplexDivByDiag, and the way a row divides by
// its diagonal.
// A complex value carried as two float-float pairs.  Passed and returned by
// value: this is on the path every row takes, where the pairs want to stay in
// registers, and a reference parameter asks for an address instead.
struct SpsvDFComplex {
    SpsvDF re;
    SpsvDF im;
};

// A float-float pair with a binary exponent of its own.
//
// The precompute's recovery needs one because a product of finite operands can
// want 254 bits of exponent, and because two output components can sit 2^127
// apart -- alpha = F+i against x = 2+iF puts F in the real part and F^2 in the
// imaginary one.  No single shared exponent holds both.
struct SpsvDFExp {
    SpsvDF m;
    int32_t e;
};

// u * v, with each operand normalised by its own exponent, so the mantissa is
// in [1, 4) and nothing is subnormal however large or small u and v are.
__simt_callee__ inline SpsvDFExp SpsvDFExpProd(float u, float v)
{
    int32_t eu = SpsvExpOf(u);
    int32_t ev = SpsvExpOf(v);
    SpsvDFExp r;
    r.m = SpsvTwoProd(ldexpf(u, -eu), ldexpf(v, -ev));
    r.e = eu + ev;
    return r;
}

__simt_callee__ inline SpsvDFExp SpsvDFExpMulF(SpsvDFExp a, float v)
{
    int32_t ev = SpsvExpOf(v);
    a.m = SpsvDFMulF(a.m, ldexpf(v, -ev));
    a.e += ev;
    return a;
}

__simt_callee__ inline SpsvDFExp SpsvDFExpNeg(SpsvDFExp a)
{
    a.m = SpsvDFNeg(a.m);
    return a;
}

// a + b at the larger exponent.  This is the only place anything is brought
// down, and it is the only place where losing a term is right: what flushes
// here is small against the term it is added to.
//
// A zero is returned as the other side untouched rather than aligned to, so
// that a product with a zero operand -- whose exponent is whatever the other
// operand's was -- cannot drag the sum's exponent up and flush the term that
// matters.
__simt_callee__ inline SpsvDFExp SpsvDFExpAdd(SpsvDFExp a, SpsvDFExp b)
{
    if (a.m.hi == 0.0f && a.m.lo == 0.0f) {
        b.m.lost |= a.m.lost;
        return b;
    }
    if (b.m.hi == 0.0f && b.m.lo == 0.0f) {
        a.m.lost |= b.m.lost;
        return a;
    }
    SpsvDFExp r;
    r.e = (a.e > b.e) ? a.e : b.e;
    r.m = SpsvDFAdd(SpsvDFScale2(a.m, a.e - r.e), SpsvDFScale2(b.m, b.e - r.e));
    return r;
}

// (a * b) / diag when a * b on its own leaves range.
//
// Dividing one operand by the diagonal first does not work: alpha / diag can
// itself be unrepresentable while alpha * x / diag is not (alpha = 3e38(1+i),
// diag = 1+0.5i, x = 0.65-0.65i wants 3.6e38 for the first and 3.12e38 for
// the second).  Normalising only the larger operand does not work either,
// because the other one can be just as large.
//
// Normalising each operand by its larger component does not work, which is
// what this used to do and what the review found: alpha = F+i comes down by
// 2^-127, alpha.imag flushes, and alpha.imag * x.imag is where the real
// part's F comes from -- wrong by one, and finite, so nothing downstream
// notices.  A component that is small against its own pair's dominant one is
// not small against the term it meets in the other output component.
//
// So every product normalises its own two operands and keeps its own
// exponent, the two output components keep separate exponents, and the only
// place anything comes down is inside a sum.  The division by the diagonal
// mixes the components, so its four products are formed the same way.
__simt_callee__ inline void SpsvComplexScaledProductDivByDiag(SpsvComplex64 a, SpsvComplex64 b, SpsvComplex64 recip,
                                                              float scale, SpsvDF& outRe, SpsvDF& outIm)
{
    SpsvDFExp pRe = SpsvDFExpAdd(SpsvDFExpProd(a.real, b.real), SpsvDFExpNeg(SpsvDFExpProd(a.imag, b.imag)));
    SpsvDFExp pIm = SpsvDFExpAdd(SpsvDFExpProd(a.real, b.imag), SpsvDFExpProd(a.imag, b.real));
    SpsvDFExp qRe = SpsvDFExpAdd(SpsvDFExpMulF(pRe, recip.real), SpsvDFExpNeg(SpsvDFExpMulF(pIm, recip.imag)));
    SpsvDFExp qIm = SpsvDFExpAdd(SpsvDFExpMulF(pRe, recip.imag), SpsvDFExpMulF(pIm, recip.real));
    // the stored scale, divided out by its mantissa with its exponent carried
    // into the one rescale apiece; see SpsvComplexDivByDiag on a zero scale
    int32_t ks = SpsvExpOf(scale);
    float ss = ldexpf(scale, -ks);
    float by = (ss != 0.0f) ? ss : 1.0f;
    outRe = SpsvDFScale2(SpsvDF{qRe.m.hi / by, qRe.m.lo / by, qRe.m.lost}, qRe.e - ks);
    outIm = SpsvDFScale2(SpsvDF{qIm.m.hi / by, qIm.m.lo / by, qIm.m.lost}, qIm.e - ks);
}

#endif // SPSV_ARITH_H_
