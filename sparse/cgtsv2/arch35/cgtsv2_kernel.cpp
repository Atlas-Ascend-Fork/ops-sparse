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

#include <limits>
#include "acl/acl_rt.h"

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "cgtsv2_kernel.h"
#include "log/log.h"

namespace {
struct Complex {
    float real;
    float imag;
};
static_assert(sizeof(Complex) == 8, "complex64 layout");

__simt_callee__ inline Complex ReadComplex(__gm__ const Complex* values, int64_t i)
{
    return {values[i].real, values[i].imag};
}

__simt_callee__ inline void WriteComplex(__gm__ Complex* values, int64_t i, Complex value)
{
    values[i].real = value.real;
    values[i].imag = value.imag;
}

__simt_callee__ inline bool IsZero(Complex x)
{
    return x.real == 0.0f && x.imag == 0.0f;
}

__simt_callee__ inline Complex SubProduct(Complex value, Complex factor, Complex other)
{
    // A structurally zero multiplier must not mix unrelated Inf/NaN components.
    if (IsZero(factor)) {
        return value;
    }
    return {fmaf(-factor.real, other.real, fmaf(factor.imag, other.imag, value.real)),
        fmaf(-factor.real, other.imag, fmaf(-factor.imag, other.real, value.imag))};
}

__simt_callee__ inline float DivideComponent(float numerator, float denominator)
{
    if (denominator != 0.0f) {
        return numerator / denominator;
    }
    // Preserve IEEE classification and the sign of a zero denominator.
    if (numerator == 0.0f || isnan(numerator)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const bool negative = static_cast<bool>(__builtin_signbit(numerator)) !=
        static_cast<bool>(__builtin_signbit(denominator));
    return negative ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
}

__simt_callee__ inline Complex Divide(Complex a, Complex b)
{
    // Preserve IEEE complex arithmetic for nonfinite numerators. In that
    // domain the cross terms (including Inf*0 and NaN*0) affect classification.
    if (b.imag == 0.0f && isfinite(a.real) && isfinite(a.imag)) {
        return {a.real / b.real, a.imag / b.real};
    }
    if (b.real == 0.0f && isfinite(a.real) && isfinite(a.imag)) {
        return {a.imag / b.imag, -a.real / b.imag};
    }
    // Smith division avoids squaring the denominator and its overflow/underflow.
    if (fabsf(b.real) >= fabsf(b.imag)) {
        const float ratio = b.imag / b.real;
        const float denominator = fmaf(b.imag, ratio, b.real);
        return {DivideComponent(fmaf(a.imag, ratio, a.real), denominator),
            DivideComponent(fmaf(-a.real, ratio, a.imag), denominator)};
    }
    const float ratio = b.real / b.imag;
    const float denominator = fmaf(b.real, ratio, b.imag);
    return {DivideComponent(fmaf(a.real, ratio, a.imag), denominator),
        DivideComponent(fmaf(a.imag, ratio, -a.real), denominator)};
}

struct Mantissa {
    float hi;
    float lo;
};
struct Wide {
    float hi;
    float lo;
    int32_t exponent = 0;
};
struct WideComplex {
    Wide real;
    Wide imag;
};
static_assert(sizeof(WideComplex) == 24, "independent component exponents workspace layout");

__simt_callee__ inline float ScaleBinary(float value, int32_t shift)
{
    // Exact power-of-two multiplication also preserves cancellation of large
    // identical inputs; no approximate exponential is used for scalbn.
    while (shift > 32 && value != 0.0f && isfinite(value)) {
        value *= 0x1p32f;
        shift -= 32;
    }
    while (shift < -32 && value != 0.0f && isfinite(value)) {
        value *= 0x1p-32f;
        shift += 32;
    }
    if (value == 0.0f || !isfinite(value)) {
        return value;
    }
    union {
        uint32_t bits;
        float value;
    } power{static_cast<uint32_t>(shift + 127) << 23};
    return value * power.value;
}

__simt_callee__ inline Mantissa ScaleMantissa(Wide a, int32_t exponent)
{
    const int32_t shift = a.exponent - exponent;
    return {ScaleBinary(a.hi, shift), ScaleBinary(a.lo, shift)};
}

__simt_callee__ inline Wide Normalize(Wide a)
{
    if (a.hi == 0.0f) {
        a.hi = a.lo;
        a.lo = 0.0f;
    }
    float magnitude = fabsf(a.hi);
    if (!isfinite(magnitude) || magnitude == 0.0f) {
        a.exponent = 0;
        return a;
    }
    // Normalize each component separately: a large real part must never scale
    // a small imaginary part out of range before later real cancellation.
    while (magnitude > 65536.0f) {
        a.hi *= 0x1p-32f;
        a.lo *= 0x1p-32f;
        a.exponent += 32;
        magnitude *= 0x1p-32f;
    }
    while (magnitude < 0x1p-16f) {
        a.hi *= 0x1p32f;
        a.lo *= 0x1p32f;
        a.exponent -= 32;
        magnitude *= 0x1p32f;
    }
    return a;
}

__simt_callee__ inline Mantissa AddMantissa(Mantissa a, Mantissa b)
{
    const float sum = a.hi + b.hi;
    if (!isfinite(sum)) {
        return {sum, 0.0f};
    }
    const float v = sum - a.hi;
    const float error = ((a.hi - (sum - v)) + (b.hi - v)) + (a.lo + b.lo);
    const float hi = sum + error;
    return {hi, error - (hi - sum)};
}

__simt_callee__ inline Mantissa NegateMantissa(Mantissa a)
{
    return {-a.hi, -a.lo};
}

__simt_callee__ inline Mantissa MultiplyMantissa(Mantissa a, Mantissa b)
{
    const float product = a.hi * b.hi;
    if (!isfinite(product)) {
        return {product, 0.0f};
    }
    const float error = fmaf(a.hi, b.hi, -product) +
        ((a.hi * b.lo + a.lo * b.hi) + a.lo * b.lo);
    const float hi = product + error;
    return {hi, error - (hi - product)};
}

__simt_callee__ inline Mantissa DivideMantissa(Mantissa a, Mantissa b)
{
    const float quotient = DivideComponent(a.hi, b.hi);
    if (!isfinite(quotient)) {
        return {quotient, 0.0f};
    }
    const Mantissa remainder = AddMantissa(a, NegateMantissa(MultiplyMantissa({quotient, 0.0f}, b)));
    return AddMantissa({quotient, 0.0f}, {DivideComponent(remainder.hi + remainder.lo, b.hi), 0.0f});
}

__simt_callee__ inline bool IsWideZero(Wide a)
{
    return a.hi == 0.0f && a.lo == 0.0f;
}

__simt_callee__ inline Wide NegateWide(Wide a)
{
    return {-a.hi, -a.lo, a.exponent};
}

__simt_callee__ inline Wide AddWide(Wide a, Wide b)
{
    if (IsWideZero(a)) {
        return b;
    }
    if (IsWideZero(b)) {
        return a;
    }
    const int32_t exponent = a.exponent > b.exponent ? a.exponent : b.exponent;
    const Mantissa sum = AddMantissa(ScaleMantissa(a, exponent), ScaleMantissa(b, exponent));
    return Normalize({sum.hi, sum.lo, exponent});
}

__simt_callee__ inline Wide MultiplyWide(Wide a, Wide b)
{
    const Mantissa product = MultiplyMantissa({a.hi, a.lo}, {b.hi, b.lo});
    return Normalize({product.hi, product.lo, a.exponent + b.exponent});
}

__simt_callee__ inline Wide DivideWide(Wide a, Wide b)
{
    const Mantissa quotient = DivideMantissa({a.hi, a.lo}, {b.hi, b.lo});
    return Normalize({quotient.hi, quotient.lo, a.exponent - b.exponent});
}

__simt_callee__ inline WideComplex Promote(Complex a)
{
    return {Normalize({a.real, 0.0f}), Normalize({a.imag, 0.0f})};
}

__simt_callee__ inline Complex Narrow(WideComplex a)
{
    return {ScaleBinary(a.real.hi + a.real.lo, a.real.exponent),
        ScaleBinary(a.imag.hi + a.imag.lo, a.imag.exponent)};
}

__simt_callee__ inline bool IsWideZero(WideComplex a)
{
    return IsWideZero(a.real) && IsWideZero(a.imag);
}

__simt_callee__ inline bool GreaterEqualWide(Wide a, Wide b)
{
    const int32_t exponent = a.exponent > b.exponent ? a.exponent : b.exponent;
    const Mantissa left = ScaleMantissa(a, exponent);
    const Mantissa right = ScaleMantissa(b, exponent);
    return left.hi > right.hi || (left.hi == right.hi && left.lo >= right.lo);
}

__simt_callee__ inline bool KeepWidePivot(WideComplex a, WideComplex b)
{
    if (IsWideZero(b)) {
        return true;
    }
    if (IsWideZero(a)) {
        return false;
    }
    const Wide normA = AddWide(MultiplyWide(a.real, a.real), MultiplyWide(a.imag, a.imag));
    const Wide normB = AddWide(MultiplyWide(b.real, b.real), MultiplyWide(b.imag, b.imag));
    return GreaterEqualWide(normA, normB);
}

__simt_callee__ inline WideComplex SubWideProduct(WideComplex a, WideComplex b, WideComplex c)
{
    if (IsWideZero(b) || IsWideZero(c)) {
        return a;
    }
    WideComplex product{AddWide(MultiplyWide(b.real, c.real), NegateWide(MultiplyWide(b.imag, c.imag))),
        AddWide(MultiplyWide(b.real, c.imag), MultiplyWide(b.imag, c.real))};
    return {AddWide(a.real, NegateWide(product.real)), AddWide(a.imag, NegateWide(product.imag))};
}

__simt_callee__ inline WideComplex DivideWideComplex(WideComplex a, WideComplex b)
{
    // IEEE cross terms are significant for nonfinite numerators (Inf*0 and
    // NaN*0). Use the full complex formula in that domain, as in Divide.
    const bool finiteNumerator = isfinite(a.real.hi) && isfinite(a.imag.hi);
    if (IsWideZero(b.imag) && finiteNumerator) {
        return {DivideWide(a.real, b.real), DivideWide(a.imag, b.real)};
    }
    if (IsWideZero(b.real) && finiteNumerator) {
        return {DivideWide(a.imag, b.imag), NegateWide(DivideWide(a.real, b.imag))};
    }
    Wide absReal{fabsf(b.real.hi), b.real.hi < 0.0f ? -b.real.lo : b.real.lo, b.real.exponent};
    Wide absImag{fabsf(b.imag.hi), b.imag.hi < 0.0f ? -b.imag.lo : b.imag.lo, b.imag.exponent};
    if (GreaterEqualWide(absReal, absImag)) {
        const Wide ratio = DivideWide(b.imag, b.real);
        const Wide denominator = AddWide(b.real, MultiplyWide(b.imag, ratio));
        return {DivideWide(AddWide(a.real, MultiplyWide(a.imag, ratio)), denominator),
            DivideWide(AddWide(a.imag, NegateWide(MultiplyWide(a.real, ratio))), denominator)};
    }
    const Wide ratio = DivideWide(b.real, b.imag);
    const Wide denominator = AddWide(b.imag, MultiplyWide(b.real, ratio));
    return {DivideWide(AddWide(a.imag, MultiplyWide(a.real, ratio)), denominator),
        DivideWide(AddWide(NegateWide(a.real), MultiplyWide(a.imag, ratio)), denominator)};
}

__simt_callee__ inline WideComplex ReadWide(__gm__ const WideComplex* p, int64_t i)
{
    return {{p[i].real.hi, p[i].real.lo, p[i].real.exponent},
        {p[i].imag.hi, p[i].imag.lo, p[i].imag.exponent}};
}

__simt_callee__ inline void WriteWide(__gm__ WideComplex* p, int64_t i, WideComplex a)
{
    p[i].real.hi = a.real.hi;
    p[i].real.lo = a.real.lo;
    p[i].real.exponent = a.real.exponent;
    p[i].imag.hi = a.imag.hi;
    p[i].imag.lo = a.imag.lo;
    p[i].imag.exponent = a.imag.exponent;
}

__simt_callee__ inline void SolveWideColumn(__gm__ const Complex* dl, __gm__ const Complex* d,
    __gm__ const Complex* du, __gm__ Complex* b, __gm__ WideComplex* ws,
    int32_t m, int32_t ldb, uint32_t column, uint32_t slot)
{
    const int64_t offset = static_cast<int64_t>(column) * ldb;
    __gm__ WideComplex* diag = ws + static_cast<int64_t>(slot) * 4 * m;
    __gm__ WideComplex* upper = diag + m;
    __gm__ WideComplex* upper2 = upper + m;
    __gm__ WideComplex* transformed = upper2 + m;
    const WideComplex zero{{0.0f, 0.0f}, {0.0f, 0.0f}};
    WideComplex pivot = Promote(ReadComplex(d, 0));
    WideComplex pivotUpper = Promote(ReadComplex(du, 0));
    WideComplex rhs = Promote(ReadComplex(b, offset));
    for (int32_t i = 1; i < m; ++i) {
        const WideComplex lower = Promote(ReadComplex(dl, i));
        const WideComplex nextDiag = Promote(ReadComplex(d, i));
        const WideComplex nextUpper = i + 1 < m ? Promote(ReadComplex(du, i)) : zero;
        const WideComplex nextRhs = Promote(ReadComplex(b, offset + i));
        if (KeepWidePivot(pivot, lower)) {
            const WideComplex factor = IsWideZero(lower) ? zero : DivideWideComplex(lower, pivot);
            WriteWide(diag, i - 1, pivot);
            WriteWide(upper, i - 1, pivotUpper);
            WriteWide(upper2, i - 1, zero);
            WriteWide(transformed, i - 1, rhs);
            pivot = SubWideProduct(nextDiag, factor, pivotUpper);
            rhs = SubWideProduct(nextRhs, factor, rhs);
            pivotUpper = nextUpper;
        } else {
            const WideComplex factor = DivideWideComplex(pivot, lower);
            WriteWide(diag, i - 1, lower);
            WriteWide(upper, i - 1, nextDiag);
            WriteWide(upper2, i - 1, nextUpper);
            WriteWide(transformed, i - 1, nextRhs);
            pivot = SubWideProduct(pivotUpper, factor, nextDiag);
            rhs = SubWideProduct(rhs, factor, nextRhs);
            pivotUpper = SubWideProduct(zero, factor, nextUpper);
        }
    }
    WideComplex x1 = DivideWideComplex(rhs, pivot);
    WideComplex x2 = zero;
    WriteComplex(b, offset + m - 1, Narrow(x1));
    for (int32_t i = m - 2; i >= 0; --i) {
        WideComplex numerator = SubWideProduct(ReadWide(transformed, i), ReadWide(upper, i), x1);
        if (i + 2 < m) {
            numerator = SubWideProduct(numerator, ReadWide(upper2, i), x2);
        }
        const WideComplex x = DivideWideComplex(numerator, ReadWide(diag, i));
        WriteComplex(b, offset + i, Narrow(x));
        x2 = x1;
        x1 = x;
    }
}

__simt_callee__ inline float Magnitude(Complex a)
{
    const float scale = fmaxf(fabsf(a.real), fabsf(a.imag));
    if (scale == 0.0f) {
        return 0.0f;
    }
    const float r = a.real / scale;
    const float i = a.imag / scale;
    return scale * sqrtf(fmaf(r, r, i * i));
}

__simt_callee__ inline bool InPcrRange(float value)
{
    const float magnitude = fabsf(value);
    return magnitude == 0.0f || (magnitude >= 0x1p-32f && magnitude <= 0x1p32f);
}

__simt_callee__ inline bool InPcrRange(Complex value)
{
    return InPcrRange(value.real) && InPcrRange(value.imag);
}

// A data-dependent stability gate, independent of case IDs and problem sizes.
// Bound coefficient and RHS ranges as well as row dominance before any write
// to B. Extreme inputs take the extended-range path, even if A is dominant.
__simt_vf__ __aicore__ __launch_bounds__(cgtsv2::kThreads) inline void Probe(
    __gm__ const Complex* dl, __gm__ const Complex* d, __gm__ const Complex* du,
    __gm__ const Complex* b, __gm__ uint32_t* flags, int32_t m, int32_t n, int32_t ldb)
{
    uint32_t stable = 1;
    const uint32_t lane = blockIdx.x * cgtsv2::kThreads + threadIdx.x;
    constexpr uint32_t stride = cgtsv2::kThreads * cgtsv2::kThreads;
    for (int64_t i = lane; i < m; i += stride) {
        const float diagonal = Magnitude(ReadComplex(d, i));
        const float lower = i > 0 ? Magnitude(ReadComplex(dl, i)) : 0.0f;
        const float upper = i + 1 < m ? Magnitude(ReadComplex(du, i)) : 0.0f;
        if (!InPcrRange(ReadComplex(d, i)) ||
            (i > 0 && !InPcrRange(ReadComplex(dl, i))) ||
            (i + 1 < m && !InPcrRange(ReadComplex(du, i))) ||
            !(diagonal > 0.0f && diagonal >= 1.25f * (lower + upper))) {
            stable = 0;
        }
    }
    for (int64_t i = lane; i < static_cast<int64_t>(m) * n; i += stride) {
        if (!InPcrRange(ReadComplex(b, (i / m) * ldb + i % m))) {
            stable = 0;
        }
    }
    const uint32_t blockStable = static_cast<uint32_t>(asc_all(static_cast<int32_t>(stable)));
    if (threadIdx.x == 0) {
        flags[blockIdx.x] = blockStable;
    }
}

__simt_callee__ inline bool Stable(__gm__ const uint32_t* flags)
{
    for (uint32_t i = 0; i < cgtsv2::kThreads; ++i) {
        if (flags[i] == 0) {
            return false;
        }
    }
    return true;
}

// Parallel cyclic reduction shares matrix coefficients across all RHS columns.
// It is used only after the device has verified strong diagonal dominance.
__simt_vf__ __aicore__ __launch_bounds__(cgtsv2::kParallelThreads) inline void PcrInit(
    __gm__ const Complex* dl, __gm__ const Complex* d, __gm__ const Complex* du,
    __gm__ const Complex* b, __gm__ Complex* ws, __gm__ const uint32_t* flags,
    int32_t m, int32_t ldb, uint32_t base, uint32_t columns, uint32_t blocks)
{
    if (!Stable(flags)) {
        return;
    }
    const int64_t rhsCount = static_cast<int64_t>(m) * columns;
    const int64_t coefficientCount = static_cast<int64_t>(m) * 3;
    const int64_t count = rhsCount > coefficientCount ? rhsCount : coefficientCount;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * cgtsv2::kParallelThreads + threadIdx.x;
         i < count; i += static_cast<int64_t>(blocks) * cgtsv2::kParallelThreads) {
        if (i < coefficientCount) {
            const int32_t row = static_cast<int32_t>(i % m);
            Complex value;
            if (i < m) {
                value = row == 0 ? Complex{0.0f, 0.0f} : ReadComplex(dl, row);
            } else if (i < 2LL * m) {
                value = ReadComplex(d, row);
            } else {
                value = row + 1 < m ? ReadComplex(du, row) : Complex{0.0f, 0.0f};
            }
            WriteComplex(ws, i, value);
        }
        if (i < rhsCount) {
            const int64_t source = (static_cast<int64_t>(base) + i / m) * ldb + i % m;
            WriteComplex(ws + 8LL * m, i, ReadComplex(b, source));
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(cgtsv2::kParallelThreads) inline void PcrCoefficients(
    __gm__ Complex* ws, __gm__ const uint32_t* flags, int32_t m, int32_t distance,
    uint32_t parity, uint32_t blocks)
{
    if (!Stable(flags)) {
        return;
    }
    __gm__ Complex* src = ws + static_cast<int64_t>(parity) * 3 * m;
    __gm__ Complex* dst = ws + static_cast<int64_t>(1 - parity) * 3 * m;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * cgtsv2::kParallelThreads + threadIdx.x;
         row < m; row += static_cast<int64_t>(blocks) * cgtsv2::kParallelThreads) {
        Complex diagonal = ReadComplex(src, m + row);
        Complex lower{0.0f, 0.0f};
        Complex upper{0.0f, 0.0f};
        Complex left{0.0f, 0.0f};
        Complex right{0.0f, 0.0f};
        if (row >= distance) {
            left = Divide(ReadComplex(src, row), ReadComplex(src, m + row - distance));
            diagonal = SubProduct(diagonal, left, ReadComplex(src, 2LL * m + row - distance));
            lower = SubProduct(lower, left, ReadComplex(src, row - distance));
        }
        if (row + distance < m) {
            right = Divide(ReadComplex(src, 2LL * m + row), ReadComplex(src, m + row + distance));
            diagonal = SubProduct(diagonal, right, ReadComplex(src, row + distance));
            upper = SubProduct(upper, right, ReadComplex(src, 2LL * m + row + distance));
        }
        WriteComplex(dst, row, lower);
        WriteComplex(dst, m + row, diagonal);
        WriteComplex(dst, 2LL * m + row, upper);
        WriteComplex(ws + 6LL * m, row, left);
        WriteComplex(ws + 7LL * m, row, right);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(cgtsv2::kParallelThreads) inline void PcrRhs(
    __gm__ Complex* ws, __gm__ const uint32_t* flags, int32_t m, uint32_t columns,
    uint32_t slots, int32_t distance, uint32_t parity, uint32_t blocks)
{
    if (!Stable(flags)) {
        return;
    }
    const int64_t pitch = static_cast<int64_t>(m) * slots;
    __gm__ Complex* src = ws + 8LL * m + parity * pitch;
    __gm__ Complex* dst = ws + 8LL * m + (1 - parity) * pitch;
    const int64_t count = static_cast<int64_t>(m) * columns;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * cgtsv2::kParallelThreads + threadIdx.x;
         i < count; i += static_cast<int64_t>(blocks) * cgtsv2::kParallelThreads) {
        const int32_t row = static_cast<int32_t>(i % m);
        Complex value = ReadComplex(src, i);
        if (row >= distance) {
            value = SubProduct(value, ReadComplex(ws + 6LL * m, row), ReadComplex(src, i - distance));
        }
        if (row + distance < m) {
            value = SubProduct(value, ReadComplex(ws + 7LL * m, row), ReadComplex(src, i + distance));
        }
        WriteComplex(dst, i, value);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(cgtsv2::kParallelThreads) inline void PcrFinish(
    __gm__ Complex* b, __gm__ const Complex* ws, __gm__ const uint32_t* flags,
    int32_t m, int32_t ldb, uint32_t base, uint32_t columns, uint32_t slots,
    uint32_t parity, uint32_t blocks)
{
    if (!Stable(flags)) {
        return;
    }
    const __gm__ Complex* rhs = ws + 8LL * m + static_cast<int64_t>(parity) * m * slots;
    const __gm__ Complex* diagonal = ws + static_cast<int64_t>(parity) * 3 * m + m;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * cgtsv2::kParallelThreads + threadIdx.x;
         i < static_cast<int64_t>(m) * columns; i += static_cast<int64_t>(blocks) * cgtsv2::kParallelThreads) {
        const int32_t row = static_cast<int32_t>(i % m);
        const int64_t target = (static_cast<int64_t>(base) + i / m) * ldb + row;
        WriteComplex(b, target, Divide(ReadComplex(rhs, i), ReadComplex(diagonal, row)));
    }
}

__simt_vf__ __aicore__ __launch_bounds__(cgtsv2::kThreads) inline void Compute(
    __gm__ const Complex* dl, __gm__ const Complex* d, __gm__ const Complex* du,
    __gm__ Complex* b, __gm__ Complex* ws, __gm__ const uint32_t* flags,
    int32_t m, int32_t n, int32_t ldb, uint32_t slots)
{
    if (Stable(flags)) {
        return;
    }
    const uint32_t slot = blockIdx.x * cgtsv2::kThreads + threadIdx.x;
    if (slot < slots) {
        for (uint64_t column = slot; column < static_cast<uint32_t>(n); column += slots) {
            SolveWideColumn(dl, d, du, b, reinterpret_cast<__gm__ WideComplex*>(ws),
                m, ldb, static_cast<uint32_t>(column), slot);
        }
    }
}

__simt_callee__ inline void PcrBlockStep(__gm__ Complex* ws, int32_t m, int32_t distance, uint32_t parity)
{
    __gm__ const Complex* src = ws + static_cast<int64_t>(parity) * 4 * m;
    __gm__ Complex* dst = ws + static_cast<int64_t>(1 - parity) * 4 * m;
    for (int64_t row = threadIdx.x; row < m; row += cgtsv2::kParallelThreads) {
        Complex diagonal = ReadComplex(src, m + row);
        Complex rhs = ReadComplex(src, 3LL * m + row);
        Complex lower{0.0f, 0.0f};
        Complex upper{0.0f, 0.0f};
        if (row >= distance) {
            const Complex left = Divide(ReadComplex(src, row), ReadComplex(src, m + row - distance));
            diagonal = SubProduct(diagonal, left, ReadComplex(src, 2LL * m + row - distance));
            lower = SubProduct(lower, left, ReadComplex(src, row - distance));
            rhs = SubProduct(rhs, left, ReadComplex(src, 3LL * m + row - distance));
        }
        if (row + distance < m) {
            const Complex right = Divide(ReadComplex(src, 2LL * m + row), ReadComplex(src, m + row + distance));
            diagonal = SubProduct(diagonal, right, ReadComplex(src, row + distance));
            upper = SubProduct(upper, right, ReadComplex(src, 2LL * m + row + distance));
            rhs = SubProduct(rhs, right, ReadComplex(src, 3LL * m + row + distance));
        }
        WriteComplex(dst, row, lower);
        WriteComplex(dst, m + row, diagonal);
        WriteComplex(dst, 2LL * m + row, upper);
        WriteComplex(dst, 3LL * m + row, rhs);
    }
}

// A block owns one RHS and both ping-pong buffers. Reuse the same bounded slot
// for its subsequent columns; no cross-block barrier or host batch loop exists.
__simt_vf__ __aicore__ __launch_bounds__(cgtsv2::kParallelThreads) inline void PcrColumns(
    __gm__ const Complex* dl, __gm__ const Complex* d, __gm__ const Complex* du,
    __gm__ Complex* b, __gm__ Complex* ws, __gm__ const uint32_t* flags,
    int32_t m, int32_t n, int32_t ldb, uint32_t blocks)
{
    if (!Stable(flags)) {
        return;
    }
    ws += static_cast<int64_t>(blockIdx.x) * 8 * m;
    for (uint64_t column = blockIdx.x; column < static_cast<uint32_t>(n); column += blocks) {
        const int64_t offset = static_cast<int64_t>(column) * ldb;
        for (int64_t row = threadIdx.x; row < m; row += cgtsv2::kParallelThreads) {
            WriteComplex(ws, row, row == 0 ? Complex{0.0f, 0.0f} : ReadComplex(dl, row));
            WriteComplex(ws, m + row, ReadComplex(d, row));
            WriteComplex(ws, 2LL * m + row, row + 1 < m ? ReadComplex(du, row) : Complex{0.0f, 0.0f});
            WriteComplex(ws, 3LL * m + row, ReadComplex(b, offset + row));
        }
        asc_threadfence_block();
        asc_syncthreads();
        uint32_t parity = 0;
        for (int64_t distance = 1; distance < m; distance *= 2) {
            PcrBlockStep(ws, m, static_cast<int32_t>(distance), parity);
            asc_threadfence_block();
            asc_syncthreads();
            parity = 1 - parity;
        }
        __gm__ const Complex* src = ws + static_cast<int64_t>(parity) * 4 * m;
        for (int64_t row = threadIdx.x; row < m; row += cgtsv2::kParallelThreads) {
            WriteComplex(b, offset + row, Divide(ReadComplex(src, 3LL * m + row), ReadComplex(src, m + row)));
        }
        // All readers finish before the next column reinitializes these slots.
        asc_threadfence_block();
        asc_syncthreads();
    }
}
} // namespace

extern "C" __global__ __aicore__ void cgtsv2_probe_kernel(GM_ADDR dl, GM_ADDR d, GM_ADDR du,
    GM_ADDR b, GM_ADDR workspace, Cgtsv2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<Probe>(dim3{cgtsv2::kThreads}, reinterpret_cast<__gm__ const Complex*>(dl),
        reinterpret_cast<__gm__ const Complex*>(d), reinterpret_cast<__gm__ const Complex*>(du),
        reinterpret_cast<__gm__ const Complex*>(b), reinterpret_cast<__gm__ uint32_t*>(workspace),
        tiling.m, tiling.n, tiling.ldb);
}

extern "C" __global__ __aicore__ void cgtsv2_kernel(GM_ADDR dl, GM_ADDR d, GM_ADDR du,
    GM_ADDR b, GM_ADDR workspace, Cgtsv2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<Compute>(dim3{cgtsv2::kThreads}, reinterpret_cast<__gm__ const Complex*>(dl),
        reinterpret_cast<__gm__ const Complex*>(d), reinterpret_cast<__gm__ const Complex*>(du),
        reinterpret_cast<__gm__ Complex*>(b),
        reinterpret_cast<__gm__ Complex*>(workspace + cgtsv2::kAlignment),
        reinterpret_cast<__gm__ const uint32_t*>(workspace),
        tiling.m, tiling.n, tiling.ldb, tiling.slots);
}

extern "C" __global__ __aicore__ void cgtsv2_pcr_kernel(GM_ADDR dl, GM_ADDR d, GM_ADDR du,
    GM_ADDR b, GM_ADDR workspace, Cgtsv2TilingData tiling, uint32_t stage,
    uint32_t base, uint32_t columns, int32_t distance, uint32_t parity, uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* payload = reinterpret_cast<__gm__ Complex*>(workspace + cgtsv2::kAlignment);
    const auto* flags = reinterpret_cast<__gm__ const uint32_t*>(workspace);
    if (stage == 0) {
        asc_vf_call<PcrInit>(dim3{cgtsv2::kParallelThreads},
            reinterpret_cast<__gm__ const Complex*>(dl), reinterpret_cast<__gm__ const Complex*>(d),
            reinterpret_cast<__gm__ const Complex*>(du), reinterpret_cast<__gm__ const Complex*>(b),
            payload, flags, tiling.m, tiling.ldb, base, columns, blocks);
    } else if (stage == 1) {
        asc_vf_call<PcrCoefficients>(dim3{cgtsv2::kParallelThreads}, payload, flags,
            tiling.m, distance, parity, blocks);
    } else if (stage == 2) {
        asc_vf_call<PcrRhs>(dim3{cgtsv2::kParallelThreads}, payload, flags,
            tiling.m, columns, tiling.slots, distance, parity, blocks);
    } else {
        asc_vf_call<PcrFinish>(dim3{cgtsv2::kParallelThreads}, reinterpret_cast<__gm__ Complex*>(b),
            payload, flags, tiling.m, tiling.ldb, base, columns, tiling.slots, parity, blocks);
    }
}

extern "C" __global__ __aicore__ void cgtsv2_pcr_columns_kernel(GM_ADDR dl, GM_ADDR d, GM_ADDR du,
    GM_ADDR b, GM_ADDR workspace, Cgtsv2TilingData tiling, uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<PcrColumns>(dim3{cgtsv2::kParallelThreads}, reinterpret_cast<__gm__ const Complex*>(dl),
        reinterpret_cast<__gm__ const Complex*>(d), reinterpret_cast<__gm__ const Complex*>(du),
        reinterpret_cast<__gm__ Complex*>(b),
        reinterpret_cast<__gm__ Complex*>(workspace + cgtsv2::kAlignment),
        reinterpret_cast<__gm__ const uint32_t*>(workspace), tiling.m, tiling.n, tiling.ldb, blocks);
}

namespace {
bool LaunchSucceeded(const char* stage, const Cgtsv2TilingData& tiling, void* stream)
{
    const aclError error = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (error == ACL_SUCCESS) {
        return true;
    }
    OP_LOGE("aclsparseCgtsv2", "%s enqueue failed: error=%d m=%d n=%d ldb=%d stream=%p",
        stage, error, tiling.m, tiling.n, tiling.ldb, stream);
    return false;
}

aclsparseStatus_t LaunchBatchedPcr(uint8_t* dl, uint8_t* d, uint8_t* du, uint8_t* b,
    uint8_t* workspace, const Cgtsv2TilingData& tiling, void* stream)
{
    const uint32_t coefficientBlocks = static_cast<uint32_t>(
        (static_cast<int64_t>(tiling.m) + cgtsv2::kParallelThreads - 1) / cgtsv2::kParallelThreads);
    const uint32_t matrixBlocks = coefficientBlocks < tiling.cores ? coefficientBlocks : tiling.cores;
    for (int64_t base = 0; base < tiling.n; base += tiling.slots) {
        const uint32_t remaining = static_cast<uint32_t>(tiling.n - base);
        const uint32_t columns = remaining < tiling.slots ? remaining : tiling.slots;
        const int64_t count = static_cast<int64_t>(tiling.m) * columns;
        const int64_t initCount = count > 3LL * tiling.m ? count : 3LL * tiling.m;
        const int64_t initGrid = (initCount + cgtsv2::kParallelThreads - 1) / cgtsv2::kParallelThreads;
        const int64_t rhsGrid = (count + cgtsv2::kParallelThreads - 1) / cgtsv2::kParallelThreads;
        const uint32_t initBlocks = static_cast<uint32_t>(initGrid < tiling.cores ? initGrid : tiling.cores);
        const uint32_t rhsBlocks = static_cast<uint32_t>(rhsGrid < tiling.cores ? rhsGrid : tiling.cores);
        cgtsv2_pcr_kernel<<<initBlocks, nullptr, stream>>>(dl, d, du, b, workspace, tiling,
            0, static_cast<uint32_t>(base), columns, 0, 0, initBlocks);
        if (!LaunchSucceeded("pcr_init", tiling, stream)) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        uint32_t parity = 0;
        for (int64_t distance = 1; distance < tiling.m; distance *= 2) {
            cgtsv2_pcr_kernel<<<matrixBlocks, nullptr, stream>>>(dl, d, du, b, workspace, tiling,
                1, static_cast<uint32_t>(base), columns, static_cast<int32_t>(distance), parity, matrixBlocks);
            if (!LaunchSucceeded("pcr_coefficients", tiling, stream)) {
                return ACL_SPARSE_STATUS_EXECUTION_FAILED;
            }
            cgtsv2_pcr_kernel<<<rhsBlocks, nullptr, stream>>>(dl, d, du, b, workspace, tiling,
                2, static_cast<uint32_t>(base), columns, static_cast<int32_t>(distance), parity, rhsBlocks);
            if (!LaunchSucceeded("pcr_rhs", tiling, stream)) {
                return ACL_SPARSE_STATUS_EXECUTION_FAILED;
            }
            parity = 1 - parity;
        }
        cgtsv2_pcr_kernel<<<rhsBlocks, nullptr, stream>>>(dl, d, du, b, workspace, tiling,
            3, static_cast<uint32_t>(base), columns, 0, parity, rhsBlocks);
        if (!LaunchSucceeded("pcr_finish", tiling, stream)) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
} // namespace

aclsparseStatus_t cgtsv2_kernel_do(uint8_t* dl, uint8_t* d, uint8_t* du, uint8_t* b,
    uint8_t* workspace, const Cgtsv2TilingData& tiling, void* stream)
{
    // Attribute thread-local errors to this chain, preserving asynchronous execution.
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    cgtsv2_probe_kernel<<<cgtsv2::kThreads, nullptr, stream>>>(dl, d, du, b, workspace, tiling);
    if (!LaunchSucceeded("probe", tiling, stream)) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    cgtsv2_kernel<<<tiling.blocks, nullptr, stream>>>(dl, d, du, b, workspace, tiling);
    if (!LaunchSucceeded("twofold", tiling, stream)) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    if (static_cast<uint32_t>(tiling.n) > tiling.slots) {
        const uint32_t blocks = tiling.slots < tiling.cores ? tiling.slots : tiling.cores;
        cgtsv2_pcr_columns_kernel<<<blocks, nullptr, stream>>>(dl, d, du, b, workspace, tiling, blocks);
        return LaunchSucceeded("pcr_columns", tiling, stream) ?
            ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return LaunchBatchedPcr(dl, d, du, b, workspace, tiling, stream);
}
