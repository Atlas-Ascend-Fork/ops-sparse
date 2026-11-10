/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_NUMERIC_H_
#define SPSV_NUMERIC_H_
#include "kernel_operator.h"
#include "spsv_dd_math.h"
#include "spsv_exact_sum.h"

namespace spsv_a2 {
using namespace AscendC;

struct Complex { float real; float imag; };
__aicore__ inline float Abs(float x) { return __builtin_fabsf(x); }
__aicore__ inline float Nan()
{
    union { uint32_t bits; float value; } x;
    x.bits = 0x7fc00000u;
    return x.value;
}
__aicore__ inline bool Finite(float x)
{
    union { float value; uint32_t bits; } v;
    v.value = x;
    return (v.bits & 0x7f800000u) != 0x7f800000u;
}
__aicore__ inline Complex Multiply(Complex a, Complex b)
{
    return {a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real};
}
__aicore__ inline bool NeedsScaling(Complex value)
{
    return spsv_scaled::NeedsScaling(value.real) | spsv_scaled::NeedsScaling(value.imag);
}
__aicore__ inline bool NeedsCheckedValue(float value)
{
    uint32_t magnitude = spsv_scaled::Bits(value) & 0x7fffffffu;
    return (magnitude != 0) & ((magnitude - 0x26800000u) > 0x32000000u);
}
__aicore__ inline bool NeedsCheckedValue(Complex value)
{
    return NeedsCheckedValue(value.real) | NeedsCheckedValue(value.imag);
}
template <int32_t Components, bool Sum = false>
__aicore__ inline void TrackProductMagnitude(Complex a, Complex b, Complex& largest)
{
    float real = Abs(a.real * b.real);
    if constexpr (Components == 2)
        real += Abs(a.imag * b.imag);
    if constexpr (Sum)
        largest.real += real;
    else if (real > largest.real)
        largest.real = real;
    if constexpr (Components == 2) {
        float imag = Abs(a.real * b.imag) + Abs(a.imag * b.real);
        if constexpr (Sum)
            largest.imag += imag;
        else if (imag > largest.imag)
            largest.imag = imag;
    }
}
__aicore__ inline bool NeedsExactResidual(Complex rhs, Complex largest)
{
    constexpr float cancellationRatio = 0x1p-12f;
    return (largest.real > 0 && Abs(rhs.real) < largest.real * cancellationRatio) ||
           (largest.imag > 0 && Abs(rhs.imag) < largest.imag * cancellationRatio);
}
constexpr int64_t kMaxBoundedTerms = 1LL << 20;
__aicore__ inline bool NeedsExactProducts(Complex rhs, Complex magnitudes, int64_t terms)
{
    // For n <= 2^20 and FP32 unit roundoff 2^-24, gamma(2*n+2) < 1/6.
    // A 5/4 headroom bounds downward rounding of the positive magnitude sum.
    constexpr float errorRatio = 0x1.4p-12f;
    return terms > kMaxBoundedTerms || (magnitudes.real > 0 && Abs(rhs.real) < magnitudes.real * errorRatio) ||
           (magnitudes.imag > 0 && Abs(rhs.imag) < magnitudes.imag * errorRatio);
}
__aicore__ inline bool NeedsCompactWidePrecision(Complex magnitudes, spsv_dd::Complex diagonal, bool unit)
{
    float divisor = unit ?
                        1 :
                        (Abs(diagonal.real.hi) > Abs(diagonal.imag.hi) ? Abs(diagonal.real.hi) : Abs(diagonal.imag.hi));
    float magnitude = magnitudes.real > magnitudes.imag ? magnitudes.real : magnitudes.imag;
    return divisor > 0 && Finite(divisor) && Finite(magnitude) && magnitude > divisor * 0x1p36f;
}
__aicore__ inline bool DependencyErrorExceedsBudget(float error, float result)
{
    // An internal dependency can be amplified later. A public absolute-error
    // allowance must not hide a lost nonzero residual here.
    return error > Abs(result) * 0x1p-11f;
}
__aicore__ inline bool NeedsDependencyReplay(Complex magnitudes, spsv_dd::Complex diagonal, bool unit, Complex result)
{
    const float real = unit ? 1 : Abs(diagonal.real.hi);
    const float imag = unit ? 0 : Abs(diagonal.imag.hi);
    const float divisor = real > imag ? real : imag;
    if (!(divisor > 0) || !Finite(divisor))
        return false;
    // FP32 unit roundoff is 2^-24. The positive sums include all products,
    // with the same 5/4 downward-rounding headroom as NeedsExactProducts.
    constexpr float roundingRatio = 0x1.4p-24f;
    const float realError = magnitudes.real * roundingRatio;
    const float imagError = magnitudes.imag * roundingRatio;
    const float dr = real / divisor;
    const float di = imag / divisor;
    // |d|^2/divisor^2 >= 1; omitting this factor overestimates both errors
    // after complex division and retains coupling between the components.
    const float realBound = (dr * realError + di * imagError) / divisor;
    const float imagBound = (di * realError + dr * imagError) / divisor;
    return DependencyErrorExceedsBudget(realBound, result.real) || DependencyErrorExceedsBudget(imagBound, result.imag);
}
__aicore__ inline spsv_dd::Complex ToPair(Complex value)
{
    return {{value.real, 0}, {value.imag, 0}};
}
__aicore__ inline float RoundReal(spsv_dd::Real value)
{
    if ((spsv_scaled::Bits(value.hi) & 0x7fffffffu) < 0x00800000u)
        return (spsv_scaled::Bits(value.hi) & 0x7fffffffu) == 0 ? value.lo : value.hi;
    return value.hi + value.lo;
}
__aicore__ inline Complex RoundPair(spsv_dd::Complex value)
{
    return {RoundReal(value.real), RoundReal(value.imag)};
}
__aicore__ inline Complex Divide(Complex x, Complex d, int32_t components)
{
    if (components == 1) return {x.real / d.real, 0};
    if (NeedsScaling(x) || NeedsScaling(d))
        return RoundPair(spsv_dd::DivScaled(ToPair(x), ToPair(d)));
    if (d.real == 0 && d.imag == 0) return {Nan(), Nan()};
    if (Abs(d.real) >= Abs(d.imag)) {
        float ratio = d.imag / d.real;
        float denominator = d.real + d.imag * ratio;
        return {(x.real + x.imag * ratio) / denominator,
            (x.imag - x.real * ratio) / denominator};
    }
    float ratio = d.real / d.imag;
    float denominator = d.imag + d.real * ratio;
    return {(x.real * ratio + x.imag) / denominator,
        (x.imag * ratio - x.real) / denominator};
}
__aicore__ inline float SubtractCompensated(float rhs, float sum, float correction)
{
    spsv_dd::Real difference = spsv_dd::Sum(rhs, -sum);
    return difference.hi + (difference.lo - correction);
}
__aicore__ inline void AccumulateDiagonal(Complex value, spsv_dd::Complex &diagonal,
    int32_t components, bool &seen, bool &scaled)
{
    // Two-word sums can lose small terms before repeated cancellation even
    // in the normal range. Replay every duplicate diagonal exactly, not only
    // those whose approximate sum becomes zero or needs exponent scaling.
    scaled |= seen;
    seen = true;
    diagonal.real = spsv_dd::Add(diagonal.real, {value.real, 0});
    if (components == 2) diagonal.imag = spsv_dd::Add(diagonal.imag, {value.imag, 0});
}
__aicore__ inline Complex Divide(Complex x, spsv_dd::Complex diagonal, int32_t components)
{
    if (diagonal.real.lo == 0 && diagonal.imag.lo == 0)
        return Divide(x, Complex{diagonal.real.hi, diagonal.imag.hi}, components);
    if (components == 1) return {RoundReal(spsv_dd::Div(ToPair(x).real, diagonal.real)), 0};
    return RoundPair(spsv_dd::Div(ToPair(x), diagonal));
}
__aicore__ static __attribute__((noinline)) Complex ScaledProductQuotient(Complex alpha, Complex x,
    spsv_scaled::Complex divisor, bool unit, int32_t components)
{
    auto left = spsv_scaled::FromPair(ToPair(alpha));
    auto right = spsv_scaled::FromPair(ToPair(x));
    if (components == 1) {
        auto rhs = spsv_scaled::Mul(left.real, right.real);
        auto result = spsv_scaled::ToPair(unit ? rhs : spsv_scaled::Div(rhs, divisor.real));
        return {RoundReal(result), 0};
    }
    auto rhs = spsv_scaled::Mul(left, right);
    return RoundPair(spsv_scaled::ToPair(unit ? rhs : spsv_scaled::Div(rhs, divisor)));
}
template <bool Checked = true>
__aicore__ inline void Accumulate(float value, float& sum, float& correction, bool& scaled)
{
    float next = sum + value;
    bool finiteInputs = !Checked || (Finite(sum) && Finite(value));
    if (!Checked || (finiteInputs && Finite(next))) {
        correction += Abs(sum) >= Abs(value) ? (sum - next) + value :
            (value - next) + sum;
    } else {
        // Recompute finite arithmetic overflow, not intentional IEEE
        // propagation from a previously nonfinite dependency.
        scaled = scaled || finiteInputs;
        correction = 0;
    }
    sum = next;
}

} // namespace spsv_a2
#endif
