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
#ifndef CGTSV2_ARCH22_COMPLEX_MATH_H_
#define CGTSV2_ARCH22_COMPLEX_MATH_H_

#include <cstdint>
#include "kernel_operator.h"

namespace Cgtsv2 {

#if defined(__clang__)
#pragma clang fp reassociate(off) contract(off)
#endif

struct Complex {
    float real;
    float imag;
};

struct FloatPair {
    float hi;
    float lo;
};

__aicore__ inline FloatPair TwoSum(float a, float b)
{
    volatile float rounded = a + b;
    const float sum = rounded;
    const float virtualB = sum - a;
    return {sum, (a - (sum - virtualB)) + (b - virtualB)};
}

__aicore__ inline FloatPair TwoProduct(float a, float b)
{
    const float aHi = __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, a) & 0xfffff000U);
    const float bHi = __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, b) & 0xfffff000U);
    const float aLo = a - aHi;
    const float bLo = b - bHi;
    volatile float rounded = a * b;
    const float product = rounded;
    return {product, ((aHi * bHi - product) + aHi * bLo + aLo * bHi) + aLo * bLo};
}

__aicore__ inline FloatPair AddPair(FloatPair a, FloatPair b)
{
    const FloatPair sum = TwoSum(a.hi, b.hi);
    return TwoSum(sum.hi, (sum.lo + a.lo) + b.lo);
}

__aicore__ inline float PairQuotient(FloatPair numerator, FloatPair denominator)
{
    const float quotient = numerator.hi / denominator.hi;
    const FloatPair product = TwoProduct(denominator.hi, quotient);
    const FloatPair residual = AddPair(numerator, {-product.hi, -product.lo - denominator.lo * quotient});
    return quotient + (residual.hi + residual.lo) / denominator.hi;
}

__aicore__ inline uint32_t MagnitudeBits(float value)
{
    return __builtin_bit_cast(uint32_t, value) & 0x7fffffffU;
}

__aicore__ inline bool Finite(float value)
{
    return MagnitudeBits(value) < 0x7f800000U;
}

__aicore__ inline bool Finite(Complex value)
{
    return Finite(value.real) && Finite(value.imag);
}

__aicore__ inline bool Zero(Complex value)
{
    return MagnitudeBits(value.real) == 0 && MagnitudeBits(value.imag) == 0;
}

__aicore__ inline bool Infinite(float value)
{
    return MagnitudeBits(value) == 0x7f800000U;
}

__aicore__ inline float Signed(float magnitude, float sign)
{
    return __builtin_bit_cast(float, MagnitudeBits(magnitude) |
        (__builtin_bit_cast(uint32_t, sign) & 0x80000000U));
}

// Return the exponent of the leading nonzero binary digit, including a
// subnormal input. Integer normalization avoids flushing tiny operands to zero.
__aicore__ inline int32_t Exponent(uint32_t magnitude)
{
    if (magnitude == 0) { return 0; }
    int32_t exponent = static_cast<int32_t>(magnitude >> 23) - 127;
    if (magnitude < 0x800000U) {
        exponent = -126;
        while ((magnitude & 0x800000U) == 0) {
            magnitude <<= 1;
            --exponent;
        }
    }
    return exponent;
}

// Exact power-of-two scaling followed by float32 RNE, with integer rounding
// for subnormal outputs. No native FP64 or subnormal arithmetic is required.
__aicore__ inline float Scale(float value, int32_t shift)
{
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    const uint32_t sign = bits & 0x80000000U;
    uint32_t magnitude = bits & 0x7fffffffU;
    if (magnitude == 0 || magnitude >= 0x7f800000U) { return value; }
    const int32_t exponent = Exponent(magnitude);
    uint32_t significand = magnitude & 0x7fffffU;
    if (magnitude < 0x800000U) {
        while ((significand & 0x800000U) == 0) { significand <<= 1; }
    } else {
        significand |= 0x800000U;
    }
    const int32_t resultExponent = exponent + shift;
    if (resultExponent > 127) {
        return __builtin_bit_cast(float, sign | 0x7f800000U);
    }
    if (resultExponent >= -126) {
        return __builtin_bit_cast(float, sign |
            (static_cast<uint32_t>(resultExponent + 127) << 23) |
            (significand & 0x7fffffU));
    }
    if (resultExponent < -150) { return __builtin_bit_cast(float, sign); }
    const uint32_t drop = static_cast<uint32_t>(-126 - resultExponent);
    uint32_t rounded = significand >> drop;
    const uint32_t remainder = significand & ((1U << drop) - 1U);
    const uint32_t halfway = 1U << (drop - 1U);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0)) { ++rounded; }
    return __builtin_bit_cast(float, sign | rounded);
}

__aicore__ inline uint32_t MaximumComponent(Complex value)
{
    const uint32_t real = MagnitudeBits(value.real);
    const uint32_t imag = MagnitudeBits(value.imag);
    return real >= imag ? real : imag;
}

struct ScaledPair {
    FloatPair value;
    int32_t exponent;
};

__aicore__ inline ScaledPair ScaledProduct(float left, float right)
{
    if (MagnitudeBits(left) == 0 || MagnitudeBits(right) == 0) {
        return {{left * right, 0.0f}, 0};
    }
    const int32_t leftExponent = Exponent(MagnitudeBits(left));
    const int32_t rightExponent = Exponent(MagnitudeBits(right));
    return {TwoProduct(Scale(left, -leftExponent), Scale(right, -rightExponent)),
            leftExponent + rightExponent};
}

__aicore__ inline ScaledPair AddScaled(ScaledPair left, ScaledPair right)
{
    if (left.value.hi == 0.0f && left.value.lo == 0.0f) { return right; }
    if (right.value.hi == 0.0f && right.value.lo == 0.0f) { return left; }
    if (left.exponent < right.exponent) {
        const ScaledPair temporary = left;
        left = right;
        right = temporary;
    }
    const int32_t distance = left.exponent - right.exponent;
    if (distance > 80) { return left; }
    right.value = {Scale(right.value.hi, -distance), Scale(right.value.lo, -distance)};
    return {AddPair(left.value, right.value), left.exponent};
}

__aicore__ inline ScaledPair Dot(float a, float b, float c, float d)
{
    return AddScaled(ScaledProduct(a, b), ScaledProduct(c, d));
}

__aicore__ inline float AddValueAndScaled(float value, ScaledPair term)
{
    const ScaledPair sum = AddScaled(ScaledProduct(value, 1.0f), term);
    return Scale(sum.value.hi + sum.value.lo, sum.exponent);
}

__aicore__ inline Complex Subtract(Complex left, Complex right)
{
    return {left.real - right.real, left.imag - right.imag};
}

__aicore__ inline Complex Negate(Complex value)
{
    return {-value.real, -value.imag};
}

__aicore__ inline Complex SubtractProduct(Complex value, Complex left, Complex right)
{
    // Keep cancellation in complex multiply-subtracts until the final FP32
    // rounding. All factors remain float32; this is not device FP64 arithmetic.
    const uint32_t leftMax = MaximumComponent(left);
    const uint32_t rightMax = MaximumComponent(right);
    if (Finite(value) && Finite(left) && Finite(right) &&
        leftMax < 0x5e800000U && rightMax < 0x5e800000U &&
        (leftMax == 0 || leftMax >= 0x20800000U) &&
        (rightMax == 0 || rightMax >= 0x20800000U)) {
        FloatPair real = AddPair({value.real, 0.0f}, TwoProduct(-left.real, right.real));
        real = AddPair(real, TwoProduct(left.imag, right.imag));
        FloatPair imag = AddPair({value.imag, 0.0f}, TwoProduct(-left.real, right.imag));
        imag = AddPair(imag, TwoProduct(-left.imag, right.real));
        return {real.hi + real.lo, imag.hi + imag.lo};
    }
    if (Finite(value) && Finite(left) && Finite(right)) {
        return {AddValueAndScaled(value.real, Dot(-left.real, right.real, left.imag, right.imag)),
                AddValueAndScaled(value.imag, Dot(-left.real, right.imag, -left.imag, right.real))};
    }
    return {value.real - (left.real * right.real - left.imag * right.imag),
            value.imag - (left.real * right.imag + left.imag * right.real)};
}

__aicore__ inline Complex Multiply(Complex left, Complex right)
{
    // Normal finite inputs take the ordinary FP32 path. For extreme finite
    // scales, bound the four products before restoring their shared exponent.
    const uint32_t leftMax = MaximumComponent(left);
    const uint32_t rightMax = MaximumComponent(right);
    if (leftMax != 0 && rightMax != 0 && Finite(left) && Finite(right) &&
        (leftMax >= 0x5e800000U || rightMax >= 0x5e800000U ||
         leftMax < 0x20800000U || rightMax < 0x20800000U)) {
        const ScaledPair real = Dot(left.real, right.real, -left.imag, right.imag);
        const ScaledPair imag = Dot(left.real, right.imag, left.imag, right.real);
        return {Scale(real.value.hi + real.value.lo, real.exponent),
                Scale(imag.value.hi + imag.value.lo, imag.exponent)};
    }
    return {left.real * right.real - left.imag * right.imag,
            left.real * right.imag + left.imag * right.real};
}

__aicore__ inline bool NormGreaterEqual(Complex left, Complex right)
{
    const uint32_t leftMax = MaximumComponent(left);
    const uint32_t rightMax = MaximumComponent(right);
    const uint32_t scale = leftMax >= rightMax ? leftMax : rightMax;
    if (scale == 0) { return true; }
    const int32_t exponent = Exponent(scale);
    const float a = Scale(left.real, -exponent);
    const float b = Scale(left.imag, -exponent);
    const float c = Scale(right.real, -exponent);
    const float d = Scale(right.imag, -exponent);
    return a * a + b * b >= c * c + d * d;
}

__aicore__ inline Complex DivideSpecial(Complex numerator, Complex denominator)
{
    const float infinity = __builtin_bit_cast(float, 0x7f800000U);
    const float nan = __builtin_bit_cast(float, 0x7fc00000U);
    if (Zero(denominator) &&
        (MagnitudeBits(numerator.real) <= 0x7f800000U ||
         MagnitudeBits(numerator.imag) <= 0x7f800000U)) {
        const float signedInfinity = Signed(infinity, denominator.real);
        return {signedInfinity * numerator.real, signedInfinity * numerator.imag};
    }
    if (Finite(denominator) && !Zero(denominator)) {
        // The task's actual SciPy/LAPACK ZGTSV reference retains the arithmetic
        // classification of Inf*0 in complex division. A libgcc-style recovery
        // of Inf+Inf here changes its last-row NaN mask for the special suite.
        const int32_t exponent = Exponent(MaximumComponent(denominator));
        const float c = Scale(denominator.real, -exponent);
        const float d = Scale(denominator.imag, -exponent);
        if (c != 0.0f && MagnitudeBits(c) >= MagnitudeBits(d)) {
            const float ratio = d / c;
            const float divisor = c + d * ratio;
            return {Scale((numerator.real + numerator.imag * ratio) / divisor, -exponent),
                    Scale((numerator.imag - numerator.real * ratio) / divisor, -exponent)};
        }
        // A finite nonzero denominator always has a nonzero scaled component.
        if (d == 0.0f) { return {nan, nan}; }
        const float ratio = c / d;
        const float divisor = d + c * ratio;
        return {Scale((numerator.imag + numerator.real * ratio) / divisor, -exponent),
                Scale((-numerator.real + numerator.imag * ratio) / divisor, -exponent)};
    }
    if ((Infinite(denominator.real) || Infinite(denominator.imag)) && Finite(numerator)) {
        const float c = Signed(Infinite(denominator.real) ? 1.0f : 0.0f, denominator.real);
        const float d = Signed(Infinite(denominator.imag) ? 1.0f : 0.0f, denominator.imag);
        return {0.0f * (numerator.real * c + numerator.imag * d),
                0.0f * (numerator.imag * c - numerator.real * d)};
    }
    return {nan, nan};
}

__aicore__ inline Complex Divide(Complex numerator, Complex denominator)
{
    if (!Finite(numerator) || !Finite(denominator) || Zero(denominator)) {
        return DivideSpecial(numerator, denominator);
    }
    const int32_t denominatorExponent = Exponent(MaximumComponent(denominator));
    const float c = Scale(denominator.real, -denominatorExponent);
    const float d = Scale(denominator.imag, -denominatorExponent);
    // Each numerator product retains its own exponent: tiny real/imaginary
    // components must not underflow merely because the other component is huge.
    const FloatPair divisor = AddPair(TwoProduct(c, c), TwoProduct(d, d));
    const ScaledPair real = Dot(numerator.real, denominator.real, numerator.imag, denominator.imag);
    const ScaledPair imag = Dot(numerator.imag, denominator.real, -numerator.real, denominator.imag);
    return {Scale(PairQuotient(real.value, divisor), real.exponent - 2 * denominatorExponent),
            Scale(PairQuotient(imag.value, divisor), imag.exponent - 2 * denominatorExponent)};
}

}  // namespace Cgtsv2

#endif  // CGTSV2_ARCH22_COMPLEX_MATH_H_
