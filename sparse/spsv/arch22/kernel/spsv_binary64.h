/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_SPSV_BINARY64_H_
#define SPSV_ARCH22_SPSV_BINARY64_H_
#include <cstdint>

#ifndef SPSV_SOFT64_FN
#define SPSV_SOFT64_FN inline
#endif
#ifndef SPSV_EXTENDED64_FN
#define SPSV_EXTENDED64_FN inline
#endif

namespace spsv_soft64 {
struct Real {
    uint64_t bits;
};
struct Wide {
    uint64_t hi;
    uint64_t lo;
};
struct Parts {
    uint64_t significand;
    int exponent;
    bool negative;
};
constexpr uint64_t kSign = uint64_t{1} << 63;
constexpr uint64_t kFraction = (uint64_t{1} << 52) - 1;
constexpr uint64_t kInfinity = uint64_t{0x7ff} << 52;
constexpr uint64_t kQuietNan = kInfinity | (uint64_t{1} << 51);

SPSV_SOFT64_FN bool IsFinite(Real x) { return (x.bits & kInfinity) != kInfinity; }
SPSV_SOFT64_FN bool IsZero(Real x) { return (x.bits & ~kSign) == 0; }
SPSV_SOFT64_FN bool IsNan(Real x) { return (x.bits & ~kSign) > kInfinity; }
SPSV_SOFT64_FN bool IsInf(Real x) { return (x.bits & ~kSign) == kInfinity; }
SPSV_SOFT64_FN Real Negate(Real x) { return {x.bits ^ kSign}; }
SPSV_SOFT64_FN Real Zero(bool negative) { return {negative ? kSign : 0}; }
SPSV_SOFT64_FN Real Infinity(bool negative) { return {kInfinity | (negative ? kSign : 0)}; }
SPSV_SOFT64_FN Real Nan() { return {kQuietNan}; }
// 2 denotes unordered (either operand is NaN).
SPSV_SOFT64_FN int CompareMagnitude(Real a, Real b)
{
    if (IsNan(a) || IsNan(b))
        return 2;
    uint64_t aa = a.bits & ~kSign;
    uint64_t bb = b.bits & ~kSign;
    return aa < bb ? -1 : (aa > bb ? 1 : 0);
}

SPSV_SOFT64_FN uint64_t RightJam(uint64_t x, unsigned distance)
{
    if (distance == 0)
        return x;
    if (distance >= 64)
        return x != 0;
    return (x >> distance) | ((x << (64 - distance)) != 0);
}
SPSV_SOFT64_FN bool WideZero(Wide x) { return (x.hi | x.lo) == 0; }
SPSV_SOFT64_FN Wide LeftWide(Wide x, unsigned distance)
{
    if (distance == 0)
        return x;
    if (distance >= 128)
        return {0, 0};
    if (distance >= 64)
        return {x.lo << (distance - 64), 0};
    return {(x.hi << distance) | (x.lo >> (64 - distance)), x.lo << distance};
}
SPSV_SOFT64_FN Wide RightWideJam(Wide x, unsigned distance)
{
    if (distance == 0)
        return x;
    if (distance >= 128)
        return {0, !WideZero(x)};
    if (distance == 64)
        return {0, x.hi | (x.lo != 0)};
    if (distance > 64) {
        unsigned highDistance = distance - 64;
        return {0, (x.hi >> highDistance) | ((x.lo != 0) || ((x.hi << (64 - highDistance)) != 0))};
    }
    return {x.hi >> distance, (x.hi << (64 - distance)) | (x.lo >> distance) | ((x.lo << (64 - distance)) != 0)};
}

// Precondition: finite and nonzero. exponent is the power of its leading bit.
SPSV_SOFT64_FN Parts Unpack(Real x)
{
    uint64_t sig = x.bits & kFraction;
    int field = static_cast<int>((x.bits >> 52) & 0x7ff);
    int exponent = field == 0 ? -1022 : field - 1023;
    if (field != 0)
        sig |= uint64_t{1} << 52;
    else {
        const unsigned shift = static_cast<unsigned>(__builtin_clzll(sig)) - 11;
        sig <<= shift;
        exponent -= static_cast<int>(shift);
    }
    return {sig, exponent, (x.bits & kSign) != 0};
}

// sig has a leading bit at 55 plus guard/round/sticky at bits 2..0.
SPSV_SOFT64_FN Real RoundPack(bool negative, int exponent, uint64_t sig)
{
    if (exponent < -1022) {
        sig = RightJam(sig, static_cast<unsigned>(-1022 - exponent));
        exponent = -1022;
    }
    uint64_t tail = sig & 7;
    uint64_t rounded = sig >> 3;
    if (tail > 4 || (tail == 4 && (rounded & 1)))
        ++rounded;
    if (rounded >= (uint64_t{1} << 53)) {
        rounded >>= 1;
        ++exponent;
    }
    if (exponent > 1023)
        return Infinity(negative);
    if (rounded == 0)
        return Zero(negative);
    uint64_t field = exponent == -1022 && rounded < (uint64_t{1} << 52) ? 0 : static_cast<uint64_t>(exponent + 1023);
    return {(negative ? kSign : 0) | (field << 52) | (rounded & kFraction)};
}

SPSV_SOFT64_FN Real FromFloatBits(uint32_t bits)
{
    bool negative = (bits >> 31) != 0;
    uint32_t fraction = bits & 0x7fffff;
    uint32_t field = (bits >> 23) & 0xff;
    if (field == 255)
        return fraction != 0 ? Nan() : Infinity(negative);
    if (field == 0 && fraction == 0)
        return Zero(negative);
    int exponent = field == 0 ? -126 : static_cast<int>(field) - 127;
    if (field == 0) {
        const unsigned shift = static_cast<unsigned>(__builtin_clz(fraction)) - 8;
        fraction <<= shift;
        exponent -= static_cast<int>(shift);
    }
    return {(negative ? kSign : 0) | (static_cast<uint64_t>(exponent + 1023) << 52) |
            (static_cast<uint64_t>(fraction & 0x7fffff) << 29)};
}
SPSV_SOFT64_FN uint32_t ToFloatBits(Real x)
{
    uint32_t sign = (x.bits & kSign) != 0 ? uint32_t{1} << 31 : 0;
    if (IsNan(x))
        return sign | 0x7fc00000;
    if (IsInf(x))
        return sign | 0x7f800000;
    if (IsZero(x))
        return sign;
    Parts p = Unpack(x);
    uint64_t sig = RightJam(p.significand, 26);
    if (p.exponent < -126) {
        sig = RightJam(sig, static_cast<unsigned>(-126 - p.exponent));
        p.exponent = -126;
    }
    uint64_t rounded = sig >> 3;
    uint64_t tail = sig & 7;
    if (tail > 4 || (tail == 4 && (rounded & 1)))
        ++rounded;
    if (rounded >= (uint64_t{1} << 24)) {
        rounded >>= 1;
        ++p.exponent;
    }
    if (p.exponent > 127)
        return sign | 0x7f800000;
    uint32_t field = p.exponent == -126 && rounded < 0x800000 ? 0 : static_cast<uint32_t>(p.exponent + 127);
    return sign | (field << 23) | (static_cast<uint32_t>(rounded) & 0x7fffff);
}

SPSV_SOFT64_FN Real AddFinite(Real a, Real b)
{
    Parts pa = Unpack(a);
    Parts pb = Unpack(b);
    if (pa.exponent < pb.exponent || (pa.exponent == pb.exponent && pa.significand < pb.significand)) {
        Parts temporary = pa;
        pa = pb;
        pb = temporary;
    }
    uint64_t left = pa.significand << 3;
    uint64_t right = RightJam(pb.significand << 3, static_cast<unsigned>(pa.exponent - pb.exponent));
    uint64_t result = pa.negative == pb.negative ? left + right : left - right;
    if (result == 0)
        return Zero(false);
    if (result >= (uint64_t{1} << 56)) {
        result = RightJam(result, 1);
        ++pa.exponent;
    }
    const unsigned shift = static_cast<unsigned>(__builtin_clzll(result)) - 8;
    result <<= shift;
    pa.exponent -= static_cast<int>(shift);
    return RoundPack(pa.negative, pa.exponent, result);
}
SPSV_SOFT64_FN Real Add(Real a, Real b)
{
    if (IsNan(a) || IsNan(b))
        return Nan();
    if (IsInf(a) && IsInf(b) && ((a.bits ^ b.bits) & kSign))
        return Nan();
    if (IsInf(a))
        return a;
    if (IsInf(b))
        return b;
    if (IsZero(a) && IsZero(b))
        return {(a.bits & b.bits) & kSign};
    if (IsZero(a))
        return b;
    if (IsZero(b))
        return a;
    return AddFinite(a, b);
}
SPSV_SOFT64_FN Real Sub(Real a, Real b) { return Add(a, Negate(b)); }

// Exact 53x53 product using four 32-bit limbs, no native wider integer type.
SPSV_SOFT64_FN Wide Product53(uint64_t a, uint64_t b)
{
    constexpr uint64_t lowMask = 0xffffffff;
    uint64_t ll = (a & lowMask) * (b & lowMask);
    uint64_t lh = (a & lowMask) * (b >> 32);
    uint64_t hl = (a >> 32) * (b & lowMask);
    uint64_t middle = (ll >> 32) + (lh & lowMask) + (hl & lowMask);
    return {(a >> 32) * (b >> 32) + (lh >> 32) + (hl >> 32) + (middle >> 32), (ll & lowMask) | (middle << 32)};
}
SPSV_SOFT64_FN Real Mul(Real a, Real b)
{
    bool negative = ((a.bits ^ b.bits) & kSign) != 0;
    if (IsNan(a) || IsNan(b))
        return Nan();
    if ((IsInf(a) && IsZero(b)) || (IsInf(b) && IsZero(a)))
        return Nan();
    if (IsInf(a) || IsInf(b))
        return Infinity(negative);
    if (IsZero(a) || IsZero(b))
        return Zero(negative);
    Parts pa = Unpack(a);
    Parts pb = Unpack(b);
    Wide product = Product53(pa.significand, pb.significand);
    bool high = (product.hi & (uint64_t{1} << 41)) != 0;
    unsigned shift = high ? 50 : 49;
    Wide reduced = RightWideJam(product, shift);
    return RoundPack(negative, pa.exponent + pb.exponent + static_cast<int>(high), reduced.lo);
}
SPSV_SOFT64_FN Real Div(Real a, Real b)
{
    bool negative = ((a.bits ^ b.bits) & kSign) != 0;
    if (IsNan(a) || IsNan(b))
        return Nan();
    if ((IsInf(a) && IsInf(b)) || (IsZero(a) && IsZero(b)))
        return Nan();
    if (IsInf(a) || IsZero(b))
        return Infinity(negative);
    if (IsZero(a) || IsInf(b))
        return Zero(negative);
    Parts pa = Unpack(a);
    Parts pb = Unpack(b);
    int exponent = pa.exponent - pb.exponent;
    uint64_t numerator = pa.significand;
    if (numerator < pb.significand) {
        numerator <<= 1;
        --exponent;
    }
    uint64_t remainder = numerator - pb.significand;
    uint64_t quotient = 1;
    for (unsigned bit = 0; bit < 55; ++bit) {
        remainder <<= 1;
        quotient <<= 1;
        if (remainder >= pb.significand) {
            remainder -= pb.significand;
            quotient |= 1;
        }
    }
    return RoundPack(negative, exponent, quotient | (remainder != 0));
}

} // namespace spsv_soft64

namespace spsv_extended64 {
using Binary = spsv_soft64::Real;
struct Real {
    Binary fraction;
    int exponent;
};
struct Complex {
    Real real;
    Real imag;
};
// Reserve room for row-product and accumulator-span offsets.
constexpr int kExponentLimit = 0x7fffffff - 4096;
constexpr uint64_t kUnitExponent = uint64_t{1023} << 52;

SPSV_EXTENDED64_FN uint32_t Bits(float value)
{
    union {
        float value;
        uint32_t bits;
    } data;
    data.value = value;
    return data.bits;
}
SPSV_EXTENDED64_FN float Float(uint32_t bits)
{
    union {
        float value;
        uint32_t bits;
    } data;
    data.bits = bits;
    return data.value;
}
SPSV_EXTENDED64_FN Real Normalize(Binary value, int64_t exponent)
{
    if (!spsv_soft64::IsFinite(value) || spsv_soft64::IsZero(value))
        return {value, 0};
    const auto parts = spsv_soft64::Unpack(value);
    if (exponent > static_cast<int64_t>(kExponentLimit) + 1074)
        return {spsv_soft64::Infinity(parts.negative), 0};
    if (exponent < -static_cast<int64_t>(kExponentLimit) - 1023)
        return {spsv_soft64::Zero(parts.negative), 0};
    exponent += parts.exponent;
    if (exponent > kExponentLimit)
        return {spsv_soft64::Infinity(parts.negative), 0};
    if (exponent < -static_cast<int64_t>(kExponentLimit))
        return {spsv_soft64::Zero(parts.negative), 0};
    return {{(value.bits & spsv_soft64::kSign) | kUnitExponent | (parts.significand & spsv_soft64::kFraction)},
            static_cast<int>(exponent)};
}
SPSV_EXTENDED64_FN Real FromFloat(float value) { return Normalize(spsv_soft64::FromFloatBits(Bits(value)), 0); }
SPSV_EXTENDED64_FN Real Negate(Real value) { return {spsv_soft64::Negate(value.fraction), value.exponent}; }
SPSV_EXTENDED64_FN Real Add(Real left, Real right)
{
    if (!spsv_soft64::IsFinite(left.fraction) || !spsv_soft64::IsFinite(right.fraction))
        return Normalize(spsv_soft64::Add(left.fraction, right.fraction), 0);
    if (spsv_soft64::IsZero(left.fraction)) {
        return spsv_soft64::IsZero(right.fraction) ? Normalize(spsv_soft64::Add(left.fraction, right.fraction), 0) :
                                                     right;
    }
    if (spsv_soft64::IsZero(right.fraction))
        return left;
    if (left.exponent < right.exponent) {
        const auto temporary = left;
        left = right;
        right = temporary;
    }
    const int64_t gap = static_cast<int64_t>(left.exponent) - right.exponent;
    // Beyond 64 exponent bits the smaller normalized operand cannot affect RNE.
    if (gap > 64)
        return left;
    const Binary aligned{(right.fraction.bits & (spsv_soft64::kSign | spsv_soft64::kFraction)) |
                         (static_cast<uint64_t>(1023 - gap) << 52)};
    return Normalize(spsv_soft64::Add(left.fraction, aligned), left.exponent);
}
SPSV_EXTENDED64_FN Real Sub(Real left, Real right) { return Add(left, Negate(right)); }
SPSV_EXTENDED64_FN Real Mul(Real left, Real right)
{
    return Normalize(spsv_soft64::Mul(left.fraction, right.fraction),
                     static_cast<int64_t>(left.exponent) + right.exponent);
}
SPSV_EXTENDED64_FN Real Div(Real left, Real right)
{
    return Normalize(spsv_soft64::Div(left.fraction, right.fraction),
                     static_cast<int64_t>(left.exponent) - right.exponent);
}
SPSV_EXTENDED64_FN int CompareMagnitude(Real left, Real right)
{
    if (!spsv_soft64::IsFinite(left.fraction) || !spsv_soft64::IsFinite(right.fraction) ||
        spsv_soft64::IsZero(left.fraction) || spsv_soft64::IsZero(right.fraction))
        return spsv_soft64::CompareMagnitude(left.fraction, right.fraction);
    if (left.exponent != right.exponent)
        return left.exponent > right.exponent ? 1 : -1;
    return spsv_soft64::CompareMagnitude(left.fraction, right.fraction);
}
SPSV_EXTENDED64_FN Complex Product(Complex left, Complex right)
{
    return {Sub(Mul(left.real, right.real), Mul(left.imag, right.imag)),
            Add(Mul(left.real, right.imag), Mul(left.imag, right.real))};
}
SPSV_EXTENDED64_FN Complex Divide(Complex value, Complex divisor)
{
    if (CompareMagnitude(divisor.real, divisor.imag) >= 0) {
        const Real ratio = Div(divisor.imag, divisor.real);
        const Real denominator = Add(divisor.real, Mul(divisor.imag, ratio));
        return {Div(Add(value.real, Mul(value.imag, ratio)), denominator),
                Div(Sub(value.imag, Mul(value.real, ratio)), denominator)};
    }
    const Real ratio = Div(divisor.real, divisor.imag);
    const Real denominator = Add(divisor.imag, Mul(divisor.real, ratio));
    return {Div(Add(Mul(value.real, ratio), value.imag), denominator),
            Div(Sub(Mul(value.imag, ratio), value.real), denominator)};
}
SPSV_EXTENDED64_FN uint32_t ToFloatBits(Real value)
{
    if (!spsv_soft64::IsFinite(value.fraction) || spsv_soft64::IsZero(value.fraction))
        return spsv_soft64::ToFloatBits(value.fraction);
    const uint32_t sign = static_cast<uint32_t>(value.fraction.bits >> 32) & 0x80000000u;
    if (value.exponent > 127)
        return sign | 0x7f800000u;
    if (value.exponent < -150)
        return sign;
    const Binary expanded{(value.fraction.bits & (spsv_soft64::kSign | spsv_soft64::kFraction)) |
                          (static_cast<uint64_t>(value.exponent + 1023) << 52)};
    return spsv_soft64::ToFloatBits(expanded);
}

struct Magnitude {
    uint64_t significand = 0;
    int64_t highest = 0;
    float special = 0;
};
SPSV_EXTENDED64_FN void AppendWindow(Magnitude& value, uint64_t significand, int highest, int64_t first, float special)
{
    if ((Bits(special) & 0x7fffffffu) >= 0x7f800000u) {
        value.special = special;
        return;
    }
    if (significand == 0)
        return;
    const int64_t exponent = first + highest;
    if (value.significand != 0) {
        const int64_t distance = exponent - value.highest;
        const unsigned shift = distance > 63 ? 64 : static_cast<unsigned>(distance);
        significand |= spsv_soft64::RightJam(value.significand, shift);
    }
    value.significand = significand;
    value.highest = exponent;
}
SPSV_EXTENDED64_FN Real Finish(const Magnitude& value, bool negative)
{
    if ((Bits(value.special) & 0x7fffffffu) >= 0x7f800000u)
        return negative ? Negate(FromFloat(value.special)) : FromFloat(value.special);
    if (value.significand == 0)
        return FromFloat(0);
    const auto result = Normalize(spsv_soft64::RoundPack(false, 0, value.significand), value.highest);
    return negative ? Negate(result) : result;
}

template <class Accumulator>
SPSV_EXTENDED64_FN void Accumulate(Accumulator& sum, float coefficient, Real value, int64_t first, bool subtract)
{
    const uint32_t bits = Bits(coefficient), magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u || !spsv_soft64::IsFinite(value.fraction)) {
        sum.AddProduct(coefficient, Float(spsv_soft64::ToFloatBits(value.fraction)), subtract);
        return;
    }
    if (magnitude == 0 || spsv_soft64::IsZero(value.fraction))
        return;
    const int field = static_cast<int>(magnitude >> 23);
    const uint64_t significand = (magnitude & 0x7fffffu) | (field == 0 ? 0 : 0x800000u);
    const auto previous = spsv_soft64::Unpack(value.fraction);
    const uint64_t lowProduct = significand * (previous.significand & 0xffffffffu);
    const uint64_t highProduct = significand * (previous.significand >> 32);
    const uint64_t low = lowProduct + (highProduct << 32);
    const spsv_soft64::Wide product{(highProduct >> 32) + (low < lowProduct), low};
    const int64_t shift = (field == 0 ? -149 : field - 150) + previous.exponent - 52 +
                          static_cast<int64_t>(value.exponent) - first;
    const int word = static_cast<int>(shift >= 0 ? shift / 32 : (shift - 31) / 32);
    const auto shifted = spsv_soft64::LeftWide(product, static_cast<unsigned>(shift - static_cast<int64_t>(word) * 32));
    const bool negative = (((bits >> 31) != 0) != previous.negative) != subtract;
    sum.AddRawWord(word, static_cast<uint32_t>(shifted.lo), negative);
    sum.AddRawWord(word + 1, static_cast<uint32_t>(shifted.lo >> 32), negative);
    sum.AddRawWord(word + 2, static_cast<uint32_t>(shifted.hi), negative);
    sum.AddRawWord(word + 3, static_cast<uint32_t>(shifted.hi >> 32), negative);
}
} // namespace spsv_extended64

#endif // SPSV_ARCH22_SPSV_BINARY64_H_
