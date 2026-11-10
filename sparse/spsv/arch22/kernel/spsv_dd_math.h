/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_A2_DD_MATH_H_
#define SPSV_A2_DD_MATH_H_
#include <cstdint>

#ifndef SPSV_DD_AICORE
#define SPSV_DD_AICORE __aicore__
#define SPSV_DD_LOCAL_AICORE
#endif

namespace spsv_complex {
// Component operations resolve to the real-number backend by argument type.
template <typename T>
SPSV_DD_AICORE inline T Add(T a, T b)
{
    return {Add(a.real, b.real), Add(a.imag, b.imag)};
}
template <typename T>
SPSV_DD_AICORE inline T Sub(T a, T b)
{
    return {Sub(a.real, b.real), Sub(a.imag, b.imag)};
}
template <typename T>
SPSV_DD_AICORE inline T Mul(T a, T b)
{
    return {Sub(Mul(a.real, b.real), Mul(a.imag, b.imag)),
        Add(Mul(a.real, b.imag), Mul(a.imag, b.real))};
}
}

// FP32 high/low arithmetic, without FP64 or SIMT-only fmaf intrinsics. Splitting
// each finite float into two 12-significand-bit parts recovers product rounding.
namespace spsv_dd {
struct Real { float hi; float lo; };
struct Complex { Real real; Real imag; };
SPSV_DD_AICORE inline bool Finite(float x)
{
    union { float f; uint32_t u; } v;
    v.f = x;
    return (v.u & 0x7f800000u) != 0x7f800000u;
}
SPSV_DD_AICORE inline float Split(float x)
{
    union { float f; uint32_t u; } v;
    v.f = x;
    v.u &= 0xfffff000u;
    return v.f;
}
SPSV_DD_AICORE inline Real Sum(float a, float b)
{
    float hi = a + b;
    if (!Finite(a) || !Finite(b) || !Finite(hi)) return {hi, 0};
    float virtualB = hi - a;
    return {hi, (a - (hi - virtualB)) + (b - virtualB)};
}
SPSV_DD_AICORE inline Real Add(Real a, Real b)
{
    Real s = Sum(a.hi, b.hi);
    if (!Finite(s.hi)) return s;
    return Sum(s.hi, s.lo + a.lo + b.lo);
}
SPSV_DD_AICORE inline Real Negate(Real x) { return {-x.hi, -x.lo}; }
SPSV_DD_AICORE inline Real Sub(Real a, Real b) { return Add(a, Negate(b)); }
SPSV_DD_AICORE inline Real Product(float a, float b)
{
    float hi = a * b;
    if (!Finite(a) || !Finite(b) || !Finite(hi)) return {hi, 0};
    float ah = Split(a);
    float bh = Split(b);
    float al = a - ah;
    float bl = b - bh;
    float error = ((ah * bh - hi) + ah * bl + al * bh) + al * bl;
    return {hi, error};
}
SPSV_DD_AICORE inline Real Mul(Real a, Real b)
{
    Real p = Product(a.hi, b.hi);
    if (!Finite(p.hi)) return p;
    return Sum(p.hi, p.lo + a.hi * b.lo + a.lo * b.hi + a.lo * b.lo);
}
SPSV_DD_AICORE inline Real Div(Real x, Real d)
{
    float hi = x.hi / d.hi;
    if (!Finite(hi) || !Finite(x.hi) || !Finite(d.hi) || d.hi == 0) return {hi, 0};
    Real remainder = Sub(x, Mul(d, {hi, 0}));
    return Sum(hi, (remainder.hi + remainder.lo) / d.hi);
}
using spsv_complex::Add;
using spsv_complex::Sub;
using spsv_complex::Mul;
SPSV_DD_AICORE inline float Abs(float x) { return x < 0 ? -x : x; }
SPSV_DD_AICORE inline float Nan()
{
    union { uint32_t u; float f; } v;
    v.u = 0x7fc00000u;
    return v.f;
}
SPSV_DD_AICORE inline Complex DivUnscaled(Complex x, Complex d)
{
    if (d.real.hi == 0 && d.imag.hi == 0) return {{Nan(), 0}, {Nan(), 0}};
    if (Abs(d.real.hi) >= Abs(d.imag.hi)) {
        Real ratio = Div(d.imag, d.real);
        Real denominator = Add(d.real, Mul(d.imag, ratio));
        return {Div(Add(x.real, Mul(x.imag, ratio)), denominator),
            Div(Sub(x.imag, Mul(x.real, ratio)), denominator)};
    }
    Real ratio = Div(d.real, d.imag);
    Real denominator = Add(d.imag, Mul(d.real, ratio));
    return {Div(Add(Mul(x.real, ratio), x.imag), denominator),
        Div(Sub(Mul(x.imag, ratio), x.real), denominator)};
}
}

// Finite extreme-range intermediates use a separate binary exponent, not an
// FP32 multiplication followed by an attempted recovery from Inf or zero.
namespace spsv_scaled {
struct Real { spsv_dd::Real fraction; int exponent; };
struct Complex { Real real; Real imag; };
// The workspace exponent stays int32. Guard bits bound scalar/span temporaries.
constexpr int kExponentLimit = 0x7fffffff - 4096;

SPSV_DD_AICORE inline uint32_t Bits(float x)
{
    union { float value; uint32_t bits; } v;
    v.value = x;
    return v.bits;
}
SPSV_DD_AICORE inline float FromBits(uint32_t bits)
{
    union { uint32_t bits; float value; } v;
    v.bits = bits;
    return v.value;
}
SPSV_DD_AICORE inline bool NeedsScaling(float x)
{
    uint32_t magnitude = Bits(x) & 0x7fffffffu;
    // Unsigned interval tests exclude zero/Inf/NaN without branch chains.
    // The bounded normal interval is [2^-50, 2^50], including both endpoints.
    return ((magnitude - 1u) < 0x7f7fffffu) &
        ((magnitude - 0x26800000u) > 0x32000000u);
}
SPSV_DD_AICORE inline Real FromFloat(float x)
{
    uint32_t bits = Bits(x);
    uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude == 0 || magnitude >= 0x7f800000u) return {{x, 0}, 0};
    int exponent = static_cast<int>(magnitude >> 23) - 127;
    uint32_t significand = magnitude & 0x007fffffu;
    if (magnitude < 0x00800000u) {
        exponent = -126;
        while (significand < 0x00800000u) {
            significand <<= 1;
            --exponent;
        }
    }
    return {{FromBits((bits & 0x80000000u) | 0x3f800000u |
        (significand & 0x007fffffu)), 0}, exponent};
}
SPSV_DD_AICORE inline float ScaleFloat(float x, int64_t shift)
{
    Real normalized = FromFloat(x);
    uint32_t bits = Bits(normalized.fraction.hi);
    uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude == 0 || magnitude >= 0x7f800000u) return x;
    uint32_t sign = bits & 0x80000000u;
    if (shift > int64_t{127} - normalized.exponent)
        return FromBits(sign | 0x7f800000u);
    if (shift < int64_t{-150} - normalized.exponent)
        return FromBits(sign);
    int64_t exponent = static_cast<int64_t>(normalized.exponent) + shift;
    if (exponent >= -126) return FromBits(sign |
        (static_cast<uint32_t>(exponent + 127) << 23) | (bits & 0x007fffffu));
    // Form subnormal results with integer rounding, independent of the
    // device arithmetic mode's treatment of subnormal intermediates.
    int64_t right = -126 - exponent;
    if (right > 24) return FromBits(sign);
    uint32_t significand = (bits & 0x007fffffu) | 0x00800000u;
    uint32_t rounded = significand >> right;
    uint32_t remainder = significand & ((1u << right) - 1u);
    uint32_t halfway = 1u << (right - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
    return FromBits(sign | rounded);
}
SPSV_DD_AICORE inline Real Normalize(spsv_dd::Real fraction, int64_t exponent)
{
    fraction = spsv_dd::Sum(fraction.hi, fraction.lo);
    Real normalized = FromFloat(fraction.hi);
    if (fraction.hi == 0 || !spsv_dd::Finite(fraction.hi)) return normalized;
    const int64_t upper = static_cast<int64_t>(kExponentLimit) - normalized.exponent;
    const int64_t lower = -static_cast<int64_t>(kExponentLimit) - normalized.exponent;
    uint32_t sign = Bits(fraction.hi) & 0x80000000u;
    if (exponent > upper)
        return FromFloat(FromBits(sign | 0x7f800000u));
    if (exponent < lower)
        return FromFloat(FromBits(sign));
    normalized.fraction.lo = ScaleFloat(fraction.lo, -static_cast<int64_t>(normalized.exponent));
    normalized.exponent = static_cast<int>(static_cast<int64_t>(normalized.exponent) + exponent);
    return normalized;
}
SPSV_DD_AICORE inline Real FromPair(spsv_dd::Real value)
{
    if (value.lo == 0) return FromFloat(value.hi);
    return Normalize(value, 0);
}
SPSV_DD_AICORE inline float Round(Real value)
{
    value = Normalize(value.fraction, value.exponent);
    uint32_t bits = Bits(value.fraction.hi);
    uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude == 0 || magnitude >= 0x7f800000u)
        return value.fraction.hi;
    if (value.exponent >= -126)
        return ScaleFloat(value.fraction.hi + value.fraction.lo, value.exponent);
    int64_t right = -126 - static_cast<int64_t>(value.exponent);
    if (right > 24)
        return FromBits(bits & 0x80000000u);
    uint32_t significand = (bits & 0x007fffffu) | 0x00800000u;
    uint32_t rounded = significand >> right;
    uint32_t remainder = significand & ((1u << right) - 1u);
    uint32_t halfway = 1u << (right - 1);
    uint32_t low = Bits(value.fraction.lo);
    bool lowNonzero = (low & 0x7fffffffu) != 0;
    bool lowPositive = ((low ^ bits) & 0x80000000u) == 0;
    bool tieUp = lowNonzero ? lowPositive : (rounded & 1u) != 0;
    if (remainder > halfway || (remainder == halfway && tieUp))
        ++rounded;
    return FromBits((bits & 0x80000000u) | rounded);
}
SPSV_DD_AICORE inline spsv_dd::Real ToPair(Real value)
{
    if (value.exponent < -126)
        return {Round(value), 0};
    float hi = ScaleFloat(value.fraction.hi, value.exponent);
    return {hi, spsv_dd::Finite(hi) ? ScaleFloat(value.fraction.lo, value.exponent) : 0};
}
SPSV_DD_AICORE inline Real Negate(Real value) { return {spsv_dd::Negate(value.fraction), value.exponent}; }
SPSV_DD_AICORE inline Real Add(Real a, Real b)
{
    if (a.fraction.hi == 0) return b;
    if (b.fraction.hi == 0) return a;
    int exponent = a.exponent > b.exponent ? a.exponent : b.exponent;
    spsv_dd::Real left{ScaleFloat(a.fraction.hi, static_cast<int64_t>(a.exponent) - exponent),
                       ScaleFloat(a.fraction.lo, static_cast<int64_t>(a.exponent) - exponent)};
    spsv_dd::Real right{ScaleFloat(b.fraction.hi, static_cast<int64_t>(b.exponent) - exponent),
                        ScaleFloat(b.fraction.lo, static_cast<int64_t>(b.exponent) - exponent)};
    return Normalize(spsv_dd::Add(left, right), exponent);
}
SPSV_DD_AICORE inline Real Sub(Real a, Real b) { return Add(a, Negate(b)); }
SPSV_DD_AICORE inline Real Mul(Real a, Real b)
{
    return Normalize(spsv_dd::Mul(a.fraction, b.fraction), static_cast<int64_t>(a.exponent) + b.exponent);
}
SPSV_DD_AICORE inline Real Div(Real x, Real d)
{
    return Normalize(spsv_dd::Div(x.fraction, d.fraction), static_cast<int64_t>(x.exponent) - d.exponent);
}
SPSV_DD_AICORE inline Complex FromPair(spsv_dd::Complex value) { return {FromPair(value.real), FromPair(value.imag)}; }
SPSV_DD_AICORE inline spsv_dd::Complex ToPair(Complex value) { return {ToPair(value.real), ToPair(value.imag)}; }
using spsv_complex::Add;
using spsv_complex::Sub;
using spsv_complex::Mul;
SPSV_DD_AICORE inline Complex Div(Complex x, Complex d)
{
    if (d.real.fraction.hi == 0 && d.imag.fraction.hi == 0)
        return {FromFloat(spsv_dd::Nan()), FromFloat(spsv_dd::Nan())};
    // Products are bounded significands with independent exponents. This
    // covers both Smith branches without an overflowing denominator or an
    // underflowing ratio, including very different component magnitudes.
    Real denominator = Add(Mul(d.real, d.real), Mul(d.imag, d.imag));
    return {Div(Add(Mul(x.real, d.real), Mul(x.imag, d.imag)), denominator),
        Div(Sub(Mul(x.imag, d.real), Mul(x.real, d.imag)), denominator)};
}
}

namespace spsv_dd {
SPSV_DD_AICORE static __attribute__((noinline)) Complex DivScaled(Complex x, Complex d)
{
    return spsv_scaled::ToPair(spsv_scaled::Div(spsv_scaled::FromPair(x), spsv_scaled::FromPair(d)));
}
SPSV_DD_AICORE inline Complex Div(Complex x, Complex d)
{
    if (spsv_scaled::NeedsScaling(x.real.hi) || spsv_scaled::NeedsScaling(x.imag.hi) ||
        spsv_scaled::NeedsScaling(d.real.hi) || spsv_scaled::NeedsScaling(d.imag.hi))
        return DivScaled(x, d);
    return DivUnscaled(x, d);
}
}
#ifdef SPSV_DD_LOCAL_AICORE
#undef SPSV_DD_LOCAL_AICORE
#undef SPSV_DD_AICORE
#endif
#endif
