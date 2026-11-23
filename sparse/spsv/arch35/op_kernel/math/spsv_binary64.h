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

#ifndef SPSV_BINARY64_H_
#define SPSV_BINARY64_H_
#include <cstdint>
#ifndef SPSV_SOFT
#define SPSV_SOFT __simt_callee__ inline
#endif
#ifndef SPSV_SOFT_CLZ
#define SPSV_SOFT_CLZ(value) AscendC::Simt::Clz(value)
#endif
#ifndef SPSV_SOFT_NAMESPACE
#define SPSV_SOFT_NAMESPACE spsv_soft
#endif
namespace SPSV_SOFT_NAMESPACE {
constexpr uint64_t Sign = 0x8000000000000000ULL;
constexpr uint64_t Fraction = 0x000fffffffffffffULL;
constexpr uint64_t Hidden = 0x0010000000000000ULL;
constexpr uint64_t Infinity = 0x7ff0000000000000ULL;
constexpr uint64_t NaN = 0x7ff8000000000000ULL;
constexpr uint64_t One = 0x3ff0000000000000ULL;
SPSV_SOFT bool IsNaN(uint64_t x) { return (x & ~Sign) > Infinity; }
SPSV_SOFT bool IsInf(uint64_t x) { return (x & ~Sign) == Infinity; }
SPSV_SOFT bool IsZero(uint64_t x) { return (x & ~Sign) == 0; }
SPSV_SOFT uint64_t Neg(uint64_t x) { return x ^ Sign; }
SPSV_SOFT uint64_t Jam(uint64_t x, int amount)
{
    if (amount <= 0) {
        return x;
    }
    if (amount >= 64) {
        return x != 0;
    }
    return (x >> amount) | ((x << (64 - amount)) != 0);
}
SPSV_SOFT uint64_t Pack(uint64_t sign, int exponent, uint64_t extended)
{
    if (exponent <= 0) {
        extended = Jam(extended, 1 - exponent);
        exponent = 0;
    }
    uint64_t significand = extended >> 3;
    uint64_t rounding = extended & 7;
    significand += rounding > 4 || (rounding == 4 && (significand & 1));
    if (significand >= (Hidden << 1)) {
        significand >>= 1;
        ++exponent;
    } else if (exponent == 0 && significand >= Hidden) {
        exponent = 1;
    }
    if (exponent >= 2047) {
        return sign | Infinity;
    }
    return sign | (static_cast<uint64_t>(exponent) << 52) | (significand & Fraction);
}
SPSV_SOFT uint64_t AddGeneral(uint64_t a, uint64_t b)
{
    if (IsNaN(a) || IsNaN(b)) {
        return NaN;
    }
    if (IsInf(a) || IsInf(b)) {
        if (IsInf(a) && IsInf(b) && ((a ^ b) & Sign)) {
            return NaN;
        }
        return IsInf(a) ? a : b;
    }
    if ((a & ~Sign) < (b & ~Sign)) {
        uint64_t swap = a;
        a = b;
        b = swap;
    }
    int ea = static_cast<int>((a >> 52) & 2047), eb = static_cast<int>((b >> 52) & 2047);
    uint64_t ma = ((a & Fraction) | (ea ? Hidden : 0)) << 3;
    uint64_t mb = ((b & Fraction) | (eb ? Hidden : 0)) << 3;
    ea = ea ? ea : 1;
    eb = eb ? eb : 1;
    mb = Jam(mb, ea - eb);
    uint64_t result = 0;
    if (((a ^ b) & Sign) == 0) {
        result = ma + mb;
        if (result >= (Hidden << 4)) {
            result = Jam(result, 1);
            ++ea;
        }
    } else {
        result = ma - mb;
        if (result == 0) {
            return 0;
        }
    }
    if (result == 0) {
        return a & Sign;
    }
    int shift = SPSV_SOFT_CLZ(result) - 8;
    result <<= shift;
    ea -= shift;
    return Pack(a & Sign, ea, result);
}
SPSV_SOFT __attribute__((always_inline)) void OrderNormalOperands(
    uint64_t& a, uint64_t& b, uint32_t& ea, uint32_t& eb)
{
    if ((a & ~Sign) < (b & ~Sign)) {
        uint64_t value = a;
        a = b;
        b = value;
        uint32_t exponent = ea;
        ea = eb;
        eb = exponent;
    }
}
// The caller has already proved both operands are finite normal numbers.
// Cancellation, underflow and overflow still retain the general fallback.
SPSV_SOFT uint64_t AddNormal(uint64_t a, uint64_t b)
{
    uint32_t ea = static_cast<uint32_t>((a >> 52) & 2047U);
    uint32_t eb = static_cast<uint32_t>((b >> 52) & 2047U);
    OrderNormalOperands(a, b, ea, eb);
    uint32_t gap = ea - eb;
    // Even below a power of two the smaller operand is strictly below half
    // the preceding binary64 spacing. No rounding boundary can be crossed.
    if (gap >= 55U) {
        return a;
    }
    uint64_t ma = ((a & Fraction) | Hidden) << 3;
    uint64_t mb = ((b & Fraction) | Hidden) << 3;
    uint64_t sticky = ((mb << ((0U - gap) & 63U)) != 0U) & (gap != 0U);
    mb = (mb >> gap) | sticky;
    uint64_t result = ((a ^ b) & Sign) != 0U ? ma - mb : ma + mb;
    if (result == 0U) {
        return 0U;
    }
    int32_t shift = SPSV_SOFT_CLZ(result) - 8;
    int32_t exponent = static_cast<int32_t>(ea) - shift;
    if (exponent <= 0 || exponent >= 2047) {
        return AddGeneral(a, b);
    }
    uint32_t left = shift > 0 ? static_cast<uint32_t>(shift) : 0U;
    result = shift < 0 ? ((result >> 1) | (result & 1U)) : result << left;
    uint64_t significand = result >> 3;
    uint32_t rounding = static_cast<uint32_t>(result & 7U);
    significand += rounding > 4U || (rounding == 4U && (significand & 1U));
    // Including the hidden bit in the addition carries a rounding overflow
    // into the exponent without a second normalization branch.
    return (a & Sign) | ((static_cast<uint64_t>(exponent - 1) << 52) + significand);
}
// Replay uses this specialization only while all preceding solution
// components have binary64 exponents [256, 2000], and all off-diagonal
// coefficients are ordinary nonzero FP32 values of magnitude <= 8.
// Products then have exponents >= 130; a nonzero complex term is >= 78.
// CSR order and <=12288 unique columns bound accumulated exponents to
// [26, 2039]. Thus neither normalization underflow nor overflow is possible.
// Zero operands retain the exact nearest-even signed-zero rule.
SPSV_SOFT bool ReplayOperandBounded(uint64_t x)
{
    uint32_t exponent = static_cast<uint32_t>((x >> 52) & 2047U);
    return exponent - 256U <= 2000U - 256U;
}
SPSV_SOFT uint64_t AddReplayBounded(uint64_t a, uint64_t b)
{
    uint32_t ea = static_cast<uint32_t>((a >> 52) & 2047U);
    uint32_t eb = static_cast<uint32_t>((b >> 52) & 2047U);
    if (ea == 0U || eb == 0U) {
        return ea == 0U ? (eb == 0U ? a & b & Sign : b) : a;
    }
    OrderNormalOperands(a, b, ea, eb);
    uint32_t gap = ea - eb;
    // Even below a power of two the smaller operand is strictly below half
    // the preceding binary64 spacing. No rounding boundary can be crossed.
    if (gap >= 55U) {
        return a;
    }
    uint64_t ma = ((a & Fraction) | Hidden) << 3;
    uint64_t mb = ((b & Fraction) | Hidden) << 3;
    uint64_t discarded = mb & ((uint64_t{1} << gap) - 1U);
    mb = (mb >> gap) | (discarded != 0U);
    uint64_t result = ((a ^ b) & Sign) != 0U ? ma - mb : ma + mb;
    if (result == 0U) {
        return 0U;
    }
    int32_t shift = SPSV_SOFT_CLZ(result) - 8;
    int32_t exponent = static_cast<int32_t>(ea) - shift;
    uint32_t left = shift > 0 ? static_cast<uint32_t>(shift) : 0U;
    result = shift < 0 ? ((result >> 1) | (result & 1U)) : result << left;
    uint64_t significand = (result + 3U + ((result >> 3) & 1U)) >> 3;
    // Including the hidden bit in the addition carries a rounding overflow
    // into the exponent without a second normalization branch.
    return (a & Sign) | ((static_cast<uint64_t>(exponent - 1) << 52) + significand);
}
SPSV_SOFT uint64_t Add(uint64_t a, uint64_t b)
{
    uint32_t ea = static_cast<uint32_t>((a >> 52) & 2047U);
    uint32_t eb = static_cast<uint32_t>((b >> 52) & 2047U);
    return ea - 1U < 2046U && eb - 1U < 2046U ? AddNormal(a, b) : AddGeneral(a, b);
}
SPSV_SOFT uint64_t Sub(uint64_t a, uint64_t b) { return Add(a, Neg(b)); }
SPSV_SOFT void Unpack(uint64_t a, int& exponent, uint64_t& significand)
{
    exponent = static_cast<int>((a >> 52) & 2047);
    significand = a & Fraction;
    if (exponent) {
        significand |= Hidden;
        return;
    }
    exponent = 1;
    while (significand && significand < Hidden) {
        significand <<= 1;
        --exponent;
    }
}
SPSV_SOFT uint64_t Mul(uint64_t a, uint64_t b)
{
    uint64_t sign = (a ^ b) & Sign;
    if (IsNaN(a) || IsNaN(b)) {
        return NaN;
    }
    if (IsInf(a) || IsInf(b)) {
        return (IsZero(a) || IsZero(b)) ? NaN : sign | Infinity;
    }
    if (IsZero(a) || IsZero(b)) {
        return sign;
    }
    int ea, eb;
    uint64_t ma, mb;
    Unpack(a, ea, ma);
    Unpack(b, eb, mb);
    uint64_t a0 = static_cast<uint32_t>(ma), a1 = ma >> 32;
    uint64_t b0 = static_cast<uint32_t>(mb), b1 = mb >> 32;
    uint64_t low = a0 * b0;
    uint64_t middle = a1 * b0 + (low >> 32);
    uint64_t high = a1 * b1 + (middle >> 32);
    middle = (middle & 0xffffffffULL) + a0 * b1;
    high += middle >> 32;
    low = (middle << 32) | static_cast<uint32_t>(low);
    int extra = (high & (1ULL << 41)) != 0;
    int shift = 49 + extra;
    uint64_t result = (high << (64 - shift)) | (low >> shift);
    result |= (low << (64 - shift)) != 0;
    return Pack(sign, ea + eb - 1023 + extra, result);
}

// These helpers instantiate in the existing scalar and SIMT namespaces.
// Preserve their distinct packing policies while sharing exact significand math.
#include "spsv_binary64_products.inc"
struct WideProduct {
    uint64_t high, low;
};
SPSV_SOFT WideProduct MultiplyWide(uint64_t a, uint64_t b)
{
    uint64_t a0 = static_cast<uint32_t>(a), a1 = a >> 32, b0 = static_cast<uint32_t>(b), b1 = b >> 32;
    uint64_t low = a0 * b0, middle = a1 * b0 + (low >> 32), high = a1 * b1 + (middle >> 32);
    middle = (middle & 0xffffffffULL) + a0 * b1;
    high += middle >> 32;
    return {high, (middle << 32) | static_cast<uint32_t>(low)};
}
SPSV_SOFT uint64_t WideShift(WideProduct p, int shift) { return (p.high << (64 - shift)) | (p.low >> shift); }
SPSV_SOFT uint64_t ReciprocalMantissa(uint64_t mb)
{
    uint64_t reciprocal = static_cast<uint64_t>((1.f / static_cast<float>(mb)) * 0x1p114f);
    for (int iteration = 0; iteration < 2; ++iteration) {
        uint64_t estimate = WideShift(MultiplyWide(mb, reciprocal), 52);
        uint64_t target = 1ULL << 62;
        if (estimate <= target) {
            reciprocal += WideShift(MultiplyWide(reciprocal, target - estimate), 62);
        } else {
            reciprocal -= WideShift(MultiplyWide(reciprocal, estimate - target), 62);
        }
    }
    return reciprocal;
}
template <bool PREPARED>
SPSV_SOFT uint64_t DivisionCore(uint64_t a, uint64_t b, uint64_t reciprocal)
{
    uint64_t sign = (a ^ b) & Sign;
    if (IsNaN(a) || IsNaN(b) || (IsInf(a) && IsInf(b)) || (IsZero(a) && IsZero(b))) {
        return NaN;
    }
    if (IsInf(a) || IsZero(b)) {
        return sign | Infinity;
    }
    if (IsZero(a) || IsInf(b)) {
        return sign;
    }
    int ea, eb;
    uint64_t ma, mb;
    Unpack(a, ea, ma);
    Unpack(b, eb, mb);
    int exponent = ea - eb + 1023;
    if (ma < mb) {
        ma <<= 1;
        --exponent;
    }
    if constexpr (!PREPARED) {
        reciprocal = ReciprocalMantissa(mb);
    }
    uint64_t quotient = WideShift(MultiplyWide(ma, reciprocal), 59);
    WideProduct product = MultiplyWide(quotient, mb);
    WideProduct dividend{ma >> 9, ma << 55};
    WideProduct remainder{dividend.high - product.high - (dividend.low < product.low), dividend.low - product.low};
    while (remainder.high & Sign) {
        --quotient;
        uint64_t old = remainder.low;
        remainder.low += mb;
        remainder.high += remainder.low < old;
    }
    while (remainder.high || remainder.low >= mb) {
        ++quotient;
        remainder.high -= remainder.low < mb;
        remainder.low -= mb;
    }
    quotient |= remainder.low != 0;
    return Pack(sign, exponent, quotient);
}
SPSV_SOFT uint64_t Div(uint64_t a, uint64_t b)
{
    return DivisionCore<false>(a, b, 0);
}
SPSV_SOFT uint64_t From32(uint32_t x)
{
    uint64_t sign = static_cast<uint64_t>(x & 0x80000000U) << 32;
    int exponent = static_cast<int>((x >> 23) & 255);
    uint64_t fraction = x & 0x7fffffU;
    if (exponent == 255) {
        return sign | Infinity | (fraction << 29);
    }
    if (exponent == 0) {
        if (fraction == 0) {
            return sign;
        }
        exponent = 1;
        while (fraction < (1U << 23)) {
            fraction <<= 1;
            --exponent;
        }
        fraction &= 0x7fffffU;
    }
    return sign | (static_cast<uint64_t>(exponent - 127 + 1023) << 52) | (fraction << 29);
}
SPSV_SOFT uint32_t To32(uint64_t x)
{
    uint32_t sign = static_cast<uint32_t>(x >> 32) & 0x80000000U;
    if (IsNaN(x)) {
        return sign | 0x7fc00000U;
    }
    if (IsInf(x)) {
        return sign | 0x7f800000U;
    }
    if (IsZero(x)) {
        return sign;
    }
    int exponent;
    uint64_t significand;
    Unpack(x, exponent, significand);
    exponent = exponent - 1023 + 127;
    uint64_t extended = Jam(significand, 26);
    if (exponent <= 0) {
        extended = Jam(extended, 1 - exponent);
        exponent = 0;
    }
    uint32_t mantissa = static_cast<uint32_t>(extended >> 3);
    int remainder = static_cast<int>(extended & 7);
    mantissa += remainder > 4 || (remainder == 4 && (mantissa & 1));
    if (mantissa >= (1U << 24)) {
        mantissa >>= 1;
        ++exponent;
    } else if (exponent == 0 && mantissa >= (1U << 23)) {
        exponent = 1;
    }
    if (exponent >= 255) {
        return sign | 0x7f800000U;
    }
    return sign | (static_cast<uint32_t>(exponent) << 23) | (mantissa & 0x7fffffU);
}
struct Complex {
    uint64_t re, im;
};
SPSV_SOFT uint64_t Box(uint64_t x) { return (x & Sign) | (IsInf(x) ? One : 0); }
SPSV_SOFT Complex RecoverProduct(Complex a, Complex b, uint64_t rr, uint64_t ii, uint64_t ri, uint64_t ir, Complex z)
{
    bool fix = false;
    if (IsInf(a.re) || IsInf(a.im)) {
        a = {Box(a.re), Box(a.im)};
        if (IsNaN(b.re)) {
            b.re &= Sign;
        }
        if (IsNaN(b.im)) {
            b.im &= Sign;
        }
        fix = true;
    }
    if (IsInf(b.re) || IsInf(b.im)) {
        b = {Box(b.re), Box(b.im)};
        if (IsNaN(a.re)) {
            a.re &= Sign;
        }
        if (IsNaN(a.im)) {
            a.im &= Sign;
        }
        fix = true;
    }
    if (!fix && (IsInf(rr) || IsInf(ii) || IsInf(ri) || IsInf(ir))) {
        if (IsNaN(a.re)) {
            a.re &= Sign;
        }
        if (IsNaN(a.im)) {
            a.im &= Sign;
        }
        if (IsNaN(b.re)) {
            b.re &= Sign;
        }
        if (IsNaN(b.im)) {
            b.im &= Sign;
        }
        fix = true;
    }
    if (fix) {
        z = {Mul(Infinity, Sub(Mul(a.re, b.re), Mul(a.im, b.im))),
             Mul(Infinity, Add(Mul(a.re, b.im), Mul(a.im, b.re)))};
    }
    return z;
}

SPSV_SOFT Complex Product(Complex a, Complex b)
{
    uint64_t rr = Mul(a.re, b.re), ii = Mul(a.im, b.im), ri = Mul(a.re, b.im), ir = Mul(a.im, b.re);
    Complex z{Sub(rr, ii), Add(ri, ir)};
    return IsNaN(z.re) && IsNaN(z.im) ? RecoverProduct(a, b, rr, ii, ri, ir, z) : z;
}
// The replay admits finite FP32 diagonals with both components in the
// ordinary binary64 range. The caller checks these restrictions.
SPSV_SOFT Complex Quotient(Complex a, Complex b)
{
    // A tiny numerator component needs a common exact exponent shift.
    // The admitted float diagonals have both components >= 2^-24 and <= 1,
    // so denominator-only and zero-ratio exceptional branches cannot occur.
    uint64_t ar = a.re & ~Sign, ai = a.im & ~Sign;
    if ((ar < Hidden && ai < 0x7c9fffffffffffffULL) || (ai < Hidden && ar < 0x7c9fffffffffffffULL)) {
        constexpr uint64_t scale = 0x4330000000000000ULL;
        a = {Mul(a.re, scale), Mul(a.im, scale)};
        b = {Mul(b.re, scale), Mul(b.im, scale)};
    }

    uint64_t ratio, denominator;
    Complex z;
    if ((b.re & ~Sign) < (b.im & ~Sign)) {
        ratio = Div(b.re, b.im);
        denominator = Add(Mul(b.re, ratio), b.im);
        z = {Div(Add(Mul(a.re, ratio), a.im), denominator), Div(Sub(Mul(a.im, ratio), a.re), denominator)};
    } else {
        ratio = Div(b.im, b.re);
        denominator = Add(Mul(b.im, ratio), b.re);
        z = {Div(Add(Mul(a.im, ratio), a.re), denominator), Div(Sub(a.im, Mul(a.re, ratio)), denominator)};
    }
    if (IsNaN(z.re) && IsNaN(z.im) && (IsInf(a.re) || IsInf(a.im))) {
        a = {Box(a.re), Box(a.im)};
        z = {Mul(Infinity, Add(Mul(a.re, b.re), Mul(a.im, b.im))),
             Mul(Infinity, Sub(Mul(a.im, b.re), Mul(a.re, b.im)))};
    }
    return z;
}
} // namespace spsv_soft
#endif
