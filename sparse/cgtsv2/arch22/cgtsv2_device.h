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
#ifndef CGTSV2_ARCH22_DEVICE_H_
#define CGTSV2_ARCH22_DEVICE_H_

#include <cstdint>
#include "kernel_operator.h"
#include "cgtsv2_kernel.h"
#include "cgtsv2_complex_math.h"

namespace Cgtsv2 {

#if defined(__clang__)
#pragma clang fp reassociate(off) contract(off)
#endif

__aicore__ inline bool DirectRowEligible(Complex lower, Complex diagonal, Complex upper)
{
    // A conservative value-based guard, independent of case IDs and labels.
    if (!Finite(lower) || !Finite(diagonal) || !Finite(upper) ||
        MaximumComponent(lower) > 0x41800000U || MaximumComponent(upper) > 0x41800000U ||
        MaximumComponent(diagonal) > 0x41000000U || MaximumComponent(diagonal) < 0x3f800000U) {
        return false;
    }
    const float lowerMax = __builtin_bit_cast(float, MaximumComponent(lower));
    const float lowerMin = __builtin_bit_cast(float,
        MagnitudeBits(lower.real) < MagnitudeBits(lower.imag) ?
        MagnitudeBits(lower.real) : MagnitudeBits(lower.imag));
    const float upperMax = __builtin_bit_cast(float, MaximumComponent(upper));
    const float upperMin = __builtin_bit_cast(float,
        MagnitudeBits(upper.real) < MagnitudeBits(upper.imag) ?
        MagnitudeBits(upper.real) : MagnitudeBits(upper.imag));
    // max+0.5*min bounds the Euclidean off-diagonal norm from above;
    // max(abs(d.real),abs(d.imag)) bounds the diagonal norm from below.
    const float offBound = (lowerMax + 0.5f * lowerMin) + (upperMax + 0.5f * upperMin);
    return __builtin_bit_cast(float, MaximumComponent(diagonal)) > 1.250031f * offBound;
}

__aicore__ inline bool FastScale(Complex value)
{
    const uint32_t maximum = MaximumComponent(value);
    return maximum <= 0x5d800000U && (maximum == 0 || maximum >= 0x21800000U);
}

__aicore__ inline Complex DivideDispatch(Complex numerator, Complex denominator, bool fast)
{
    if (!fast || !FastScale(numerator) || !FastScale(denominator) || Zero(denominator)) {
        return Divide(numerator, denominator);
    }
    if (MagnitudeBits(denominator.real) >= MagnitudeBits(denominator.imag)) {
        const float ratio = denominator.imag / denominator.real;
        const float divisor = denominator.real + denominator.imag * ratio;
        return {(numerator.real + numerator.imag * ratio) / divisor,
                (numerator.imag - numerator.real * ratio) / divisor};
    }
    const float ratio = denominator.real / denominator.imag;
    const float divisor = denominator.imag + denominator.real * ratio;
    return {(numerator.imag + numerator.real * ratio) / divisor,
            (-numerator.real + numerator.imag * ratio) / divisor};
}

__aicore__ inline Complex SubtractProductDispatch(Complex value, Complex left, Complex right, bool fast)
{
    if (fast && FastScale(value) && FastScale(left) && FastScale(right)) {
        return {value.real - (left.real * right.real - left.imag * right.imag),
                value.imag - (left.real * right.imag + left.imag * right.real)};
    }
    return SubtractProduct(value, left, right);
}

__aicore__ inline bool NormGreaterEqualDispatch(Complex left, Complex right, bool fast)
{
    if (fast && FastScale(left) && FastScale(right)) {
        return left.real * left.real + left.imag * left.imag >=
            right.real * right.real + right.imag * right.imag;
    }
    return NormGreaterEqual(left, right);
}

__aicore__ inline void SubtractScaledProduct(ScaledPair &real, ScaledPair &imag,
    Complex coefficient, Complex value)
{
    real = AddScaled(real, Dot(-coefficient.real, value.real, coefficient.imag, value.imag));
    imag = AddScaled(imag, Dot(-coefficient.real, value.imag, -coefficient.imag, value.real));
}

__aicore__ inline Complex Residual(Complex original, Complex lower, Complex diagonal,
    Complex upper, Complex previous, Complex current, Complex next)
{
    ScaledPair real = ScaledProduct(original.real, 1.0f);
    ScaledPair imag = ScaledProduct(original.imag, 1.0f);
    SubtractScaledProduct(real, imag, lower, previous);
    SubtractScaledProduct(real, imag, diagonal, current);
    SubtractScaledProduct(real, imag, upper, next);
    return {Scale(real.value.hi + real.value.lo, real.exponent),
            Scale(imag.value.hi + imag.value.lo, imag.exponent)};
}

template <typename Tensor>
__aicore__ inline Complex Read(Tensor &tensor, int64_t index)
{
    return {tensor.GetValue(2 * index), tensor.GetValue(2 * index + 1)};
}

template <typename Tensor>
__aicore__ inline void Write(Tensor &tensor, int64_t index, Complex value)
{
    tensor.SetValue(2 * index, value.real);
    tensor.SetValue(2 * index + 1, value.imag);
}

// The byte offsets are private diagnostics, not a synchronous public info ABI:
// [0] flags: bit0 singular, bit1 nonfinite factor/input, bit2 all original
// coefficient rows meet the conservative direct-solve guard; bit3 extended
// significands/exponents are active and shared RHS scratch has one writer;
// bit4 certifies the finite, bounded diagonally dominant Jacobi path;
// [1] first singular row, [2] first nonfinite row; UINT32_MAX means absent;
// [3] reserved, always zero. Only the one factor core writes these words.
constexpr uint32_t kSingular = 1U;
constexpr uint32_t kNonfinite = 2U;
constexpr uint32_t kDirectEligible = 4U;
constexpr uint32_t kExpanded = 8U;
constexpr uint32_t kJacobi = 16U;
constexpr uint32_t kNoRow = 0xffffffffU;

// Same exponent/significand strategy as gtsv2/arch22, with an independent
// representation for each complex component. No native FP64 instructions.
struct WideReal {
    float hi;
    float lo;
    int64_t exponent;
};

struct WideComplex {
    WideReal real;
    WideReal imag;
};

__aicore__ inline WideReal Widen(float value)
{
    const uint32_t magnitude = MagnitudeBits(value);
    if (magnitude == 0 || magnitude >= 0x7f800000U) { return {value, 0.0f, 0}; }
    const int32_t exponent = Exponent(magnitude);
    return {Scale(value, -exponent), 0.0f, exponent};
}

__aicore__ inline WideComplex Widen(Complex value)
{
    return {Widen(value.real), Widen(value.imag)};
}

__aicore__ inline WideReal NormalizeWide(FloatPair value, int64_t exponent)
{
    value = TwoSum(value.hi, value.lo);
    WideReal result = Widen(value.hi);
    if (value.hi == 0.0f || !Finite(value.hi)) { return result; }
    result.lo = Scale(value.lo, -static_cast<int32_t>(result.exponent));
    result.exponent += exponent;
    return result;
}

__aicore__ inline WideReal Negate(WideReal value)
{
    return {-value.hi, -value.lo, value.exponent};
}

__aicore__ inline WideReal AddWide(WideReal left, WideReal right)
{
    if (!Finite(left.hi) || !Finite(right.hi)) { return Widen(left.hi + right.hi); }
    if (left.hi == 0.0f && left.lo == 0.0f) { return right; }
    if (right.hi == 0.0f && right.lo == 0.0f) { return left; }
    if (left.exponent < right.exponent) {
        const WideReal temporary = left;
        left = right;
        right = temporary;
    }
    const int64_t distance = left.exponent - right.exponent;
    if (distance > 80) { return left; }
    const FloatPair small{Scale(right.hi, -static_cast<int32_t>(distance)),
                          Scale(right.lo, -static_cast<int32_t>(distance))};
    return NormalizeWide(AddPair({left.hi, left.lo}, small), left.exponent);
}

__aicore__ inline WideReal MultiplyWide(WideReal left, WideReal right)
{
    if (left.hi == 0.0f || right.hi == 0.0f || !Finite(left.hi) || !Finite(right.hi)) {
        return Widen(left.hi * right.hi);
    }
    FloatPair product = TwoProduct(left.hi, right.hi);
    product.lo += (left.hi * right.lo + left.lo * right.hi) + left.lo * right.lo;
    return NormalizeWide(product, left.exponent + right.exponent);
}

__aicore__ inline WideReal DivideWide(WideReal left, WideReal right)
{
    if (left.hi == 0.0f || right.hi == 0.0f || !Finite(left.hi) || !Finite(right.hi)) {
        return Widen(left.hi / right.hi);
    }
    const float quotient = left.hi / right.hi;
    const FloatPair product = TwoProduct(right.hi, quotient);
    const FloatPair residual = AddPair({left.hi, left.lo},
        {-product.hi, -product.lo - right.lo * quotient});
    return NormalizeWide(TwoSum(quotient, (residual.hi + residual.lo) / right.hi),
        left.exponent - right.exponent);
}

__aicore__ inline WideComplex AddWide(WideComplex left, WideComplex right)
{
    return {AddWide(left.real, right.real), AddWide(left.imag, right.imag)};
}

__aicore__ inline WideComplex Negate(WideComplex value)
{
    return {Negate(value.real), Negate(value.imag)};
}

__aicore__ inline WideComplex MultiplyWide(WideComplex left, WideComplex right)
{
    return {AddWide(MultiplyWide(left.real, right.real), Negate(MultiplyWide(left.imag, right.imag))),
            AddWide(MultiplyWide(left.real, right.imag), MultiplyWide(left.imag, right.real))};
}

__aicore__ inline WideComplex DivideWide(WideComplex left, WideComplex right)
{
    const WideReal denominator = AddWide(MultiplyWide(right.real, right.real),
        MultiplyWide(right.imag, right.imag));
    return {DivideWide(AddWide(MultiplyWide(left.real, right.real), MultiplyWide(left.imag, right.imag)),
                       denominator),
            DivideWide(AddWide(MultiplyWide(left.imag, right.real), Negate(MultiplyWide(left.real, right.imag))),
                       denominator)};
}

__aicore__ inline bool Zero(WideComplex value)
{
    return value.real.hi == 0.0f && value.imag.hi == 0.0f;
}

__aicore__ inline bool Finite(WideComplex value)
{
    return Finite(value.real.hi) && Finite(value.imag.hi);
}

__aicore__ inline bool AbsGreaterEqualWide(WideReal left, WideReal right)
{
    if (right.hi == 0.0f) { return true; }
    if (left.hi == 0.0f) { return false; }
    if (left.exponent != right.exponent) { return left.exponent > right.exponent; }
    const float leftHi = left.hi < 0.0f ? -left.hi : left.hi;
    const float rightHi = right.hi < 0.0f ? -right.hi : right.hi;
    if (leftHi != rightHi) { return leftHi > rightHi; }
    return (left.hi < 0.0f ? -left.lo : left.lo) >= (right.hi < 0.0f ? -right.lo : right.lo);
}

__aicore__ inline WideReal NormSquareWide(WideComplex value)
{
    return AddWide(MultiplyWide(value.real, value.real), MultiplyWide(value.imag, value.imag));
}

__aicore__ inline bool NormGreaterEqualWide(WideComplex left, WideComplex right)
{
    return AbsGreaterEqualWide(NormSquareWide(left), NormSquareWide(right));
}

__aicore__ inline float NarrowWide(WideReal value)
{
    if (value.hi == 0.0f || !Finite(value.hi)) { return value.hi; }
    const uint32_t bits = __builtin_bit_cast(uint32_t, value.hi);
    const uint32_t sign = bits & 0x80000000U;
    if (value.exponent < -126) {
        if (value.exponent < -150) { return __builtin_bit_cast(float, sign); }
        const uint32_t shift = static_cast<uint32_t>(-126 - value.exponent);
        const uint32_t significand = (bits & 0x7fffffU) | 0x800000U;
        uint32_t rounded = significand >> shift;
        const uint32_t remainder = significand & ((1U << shift) - 1U);
        const uint32_t halfway = 1U << (shift - 1U);
        const float low = sign != 0 ? -value.lo : value.lo;
        if (remainder > halfway || (remainder == halfway &&
            (low > 0.0f || (low == 0.0f && (rounded & 1U) != 0)))) { ++rounded; }
        return __builtin_bit_cast(float, sign | rounded);
    }
    const WideReal rounded = Widen(value.hi + value.lo);
    const int64_t exponent = value.exponent + rounded.exponent;
    if (exponent > 127) { return __builtin_bit_cast(float, sign | 0x7f800000U); }
    return __builtin_bit_cast(float, sign | (static_cast<uint32_t>(exponent + 127) << 23) |
        (__builtin_bit_cast(uint32_t, rounded.hi) & 0x7fffffU));
}

__aicore__ inline Complex NarrowWide(WideComplex value)
{
    return {NarrowWide(value.real), NarrowWide(value.imag)};
}

__aicore__ inline WideReal SumWideTerms(WideReal *terms, int32_t count)
{
    // Include the RHS before rounding any complex product sum. Cancel the
    // largest signed terms first so tiny cross terms survive that cancellation.
    for (int32_t i = 1; i < count; ++i) {
        for (int32_t j = i; j > 0 && AbsGreaterEqualWide(terms[j], terms[j - 1]); --j) {
            const WideReal temporary = terms[j];
            terms[j] = terms[j - 1];
            terms[j - 1] = temporary;
        }
    }
    WideReal result = Widen(0.0f);
    for (int32_t i = 0; i < count; ++i) { result = AddWide(result, terms[i]); }
    return result;
}

__aicore__ inline WideComplex SubtractProductWide(WideComplex value, WideComplex left, WideComplex right)
{
    WideReal real[3] = {value.real, Negate(MultiplyWide(left.real, right.real)),
        MultiplyWide(left.imag, right.imag)};
    WideReal imag[3] = {value.imag, Negate(MultiplyWide(left.real, right.imag)),
        Negate(MultiplyWide(left.imag, right.real))};
    return {SumWideTerms(real, 3), SumWideTerms(imag, 3)};
}

__aicore__ inline WideComplex SubtractTwoProductsWide(WideComplex value,
    WideComplex firstLeft, WideComplex firstRight, WideComplex secondLeft, WideComplex secondRight)
{
    WideReal real[5] = {value.real,
        Negate(MultiplyWide(firstLeft.real, firstRight.real)), MultiplyWide(firstLeft.imag, firstRight.imag),
        Negate(MultiplyWide(secondLeft.real, secondRight.real)), MultiplyWide(secondLeft.imag, secondRight.imag)};
    WideReal imag[5] = {value.imag,
        Negate(MultiplyWide(firstLeft.real, firstRight.imag)), Negate(MultiplyWide(firstLeft.imag, firstRight.real)),
        Negate(MultiplyWide(secondLeft.real, secondRight.imag)), Negate(MultiplyWide(secondLeft.imag, secondRight.real))};
    return {SumWideTerms(real, 5), SumWideTerms(imag, 5)};
}

__aicore__ inline WideComplex ResidualWide(Complex original, Complex lower, Complex diagonal,
    Complex upper, WideComplex previous, WideComplex current, WideComplex next)
{
    const WideComplex l = Widen(lower);
    const WideComplex d = Widen(diagonal);
    const WideComplex u = Widen(upper);
    WideReal real[7] = {Widen(original.real),
        Negate(MultiplyWide(l.real, previous.real)), MultiplyWide(l.imag, previous.imag),
        Negate(MultiplyWide(d.real, current.real)), MultiplyWide(d.imag, current.imag),
        Negate(MultiplyWide(u.real, next.real)), MultiplyWide(u.imag, next.imag)};
    WideReal imag[7] = {Widen(original.imag),
        Negate(MultiplyWide(l.real, previous.imag)), Negate(MultiplyWide(l.imag, previous.real)),
        Negate(MultiplyWide(d.real, current.imag)), Negate(MultiplyWide(d.imag, current.real)),
        Negate(MultiplyWide(u.real, next.imag)), Negate(MultiplyWide(u.imag, next.real))};
    return {SumWideTerms(real, 7), SumWideTerms(imag, 7)};
}


// Keep the correction separate until cancellation with the original equation.
// Forming current + correction first can discard the low part being recovered.
__aicore__ inline WideComplex CorrectedResidualWide(Complex original,
    const Complex *coefficients, const WideComplex *current, const WideComplex *correction)
{
    WideReal real[13];
    WideReal imag[13];
    real[0] = Widen(original.real);
    imag[0] = Widen(original.imag);
    for (int32_t i = 0; i < 3; ++i) {
        const WideComplex coefficient = Widen(coefficients[i]);
        for (int32_t part = 0; part < 2; ++part) {
            const WideComplex value = part == 0 ? current[i] : correction[i];
            const int32_t index = 1 + 4 * i + 2 * part;
            real[index] = Negate(MultiplyWide(coefficient.real, value.real));
            real[index + 1] = MultiplyWide(coefficient.imag, value.imag);
            imag[index] = Negate(MultiplyWide(coefficient.real, value.imag));
            imag[index + 1] = Negate(MultiplyWide(coefficient.imag, value.real));
        }
    }
    return {SumWideTerms(real, 13), SumWideTerms(imag, 13)};
}

}  // namespace Cgtsv2

#endif  // CGTSV2_ARCH22_DEVICE_H_
