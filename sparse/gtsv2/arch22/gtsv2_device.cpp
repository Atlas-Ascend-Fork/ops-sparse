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

/*!
 * \file gtsv2_device.cpp
 * \brief DAV_2201 aclsparseSgtsv2 kernels.
 *
 * The coefficient matrix is common to every RHS.  A one-core factor kernel
 * performs pivoted tridiagonal LU once, then a multi-core solve kernel reuses
 * those factors for all RHS columns.  For m <= kGtsv2UbMaxM, each solve core
 * keeps the factors and one complete RHS in UB.  Larger systems use a compact
 * GM streaming fallback. A single UB-resident RHS fuses factorization and
 * substitution into one kernel without spilling factors to workspace. These
 * direct paths preserve the FP32 operation order of the pivoted Thomas algorithm.
 * Strongly row-dominant, finite, bounded-normalized-RHS systems instead use
 * a guarded 40-iteration vector Jacobi path to avoid long scalar dependency
 * chains on A2/A3. This never relaxes the public precision contract.
 * Other finite systems use pivoted LU with two-float significands and separate
 * exponents, avoiding overflow in factors, RHS elimination and back substitution.
 */

#include "gtsv2_device.h"

using namespace AscendC;

// Error-free transforms require their written FP32 evaluation order. They do
// not use FP64 instructions, CPU fallback, or a changed public float ABI.
// The device compiler can contract across inlined transforms even with the
// pragma below. Volatile rounding barriers materialize the leading sum/product
// before computing its error, preventing that error from being counted twice.
#if defined(__clang__)
#pragma clang fp reassociate(off) contract(off)
#endif


__aicore__ inline bool Gtsv2Finite(float value)
{
    constexpr float maxFinite = 0x1.fffffep127f;
    return value <= maxFinite && value >= -maxFinite;
}

__aicore__ inline Gtsv2FloatPair Gtsv2TwoSum(float a, float b)
{
    volatile float roundedSum = a + b;
    float sum = roundedSum;
    float bVirtual = sum - a;
    float error = (a - (sum - bVirtual)) + (b - bVirtual);
    return {sum, error};
}

__aicore__ inline Gtsv2FloatPair Gtsv2TwoProduct(float a, float b)
{
    // Splitting the significand by bits avoids the overflow of splitter*a.
    float aHi = __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, a) & 0xfffff000U);
    float bHi = __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, b) & 0xfffff000U);
    float aLo = a - aHi;
    float bLo = b - bHi;
    volatile float roundedProduct = a * b;
    float product = roundedProduct;
    float error = ((aHi * bHi - product) + aHi * bLo + aLo * bHi) + aLo * bLo;
    return {product, error};
}

__aicore__ inline Gtsv2FloatPair Gtsv2PairAdd(Gtsv2FloatPair a, Gtsv2FloatPair b)
{
    Gtsv2FloatPair sum = Gtsv2TwoSum(a.hi, b.hi);
    float error = (sum.lo + a.lo) + b.lo;
    return Gtsv2TwoSum(sum.hi, error);
}

__aicore__ inline float Gtsv2AccurateResidual(
    float b, float dl, float d, float du, float previous, float x, float next)
{
    Gtsv2FloatPair residual{b, 0.0f};
    residual = Gtsv2PairAdd(residual, Gtsv2TwoProduct(-dl, previous));
    residual = Gtsv2PairAdd(residual, Gtsv2TwoProduct(-d, x));
    residual = Gtsv2PairAdd(residual, Gtsv2TwoProduct(-du, next));
    return residual.hi + residual.lo;
}

__aicore__ inline Gtsv2Scaled Gtsv2ScaledFromFloat(float value)
{
    uint32_t bits = __builtin_bit_cast(uint32_t, value);
    uint32_t magnitude = bits & 0x7fffffffU;
    if (magnitude == 0 || magnitude >= 0x7f800000U) {
        return {value, 0.0f, 0};
    }
    int64_t exponent = static_cast<int32_t>(magnitude >> 23) - 127;
    uint32_t fraction = magnitude & 0x7fffffU;
    if (magnitude < 0x800000U) {
        exponent = -126;
        while ((fraction & 0x800000U) == 0) {
            fraction <<= 1;
            --exponent;
        }
    }
    float hi = __builtin_bit_cast(float, (bits & 0x80000000U) | 0x3f800000U | (fraction & 0x7fffffU));
    return {hi, 0.0f, exponent};
}

__aicore__ inline float Gtsv2ScaleSignificand(float value, int32_t exponent)
{
    return value * __builtin_bit_cast(float, static_cast<uint32_t>(exponent + 127) << 23);
}

__aicore__ inline Gtsv2Scaled Gtsv2ScaledNormalize(Gtsv2FloatPair value, int64_t exponent)
{
    value = Gtsv2TwoSum(value.hi, value.lo);
    Gtsv2Scaled normalized = Gtsv2ScaledFromFloat(value.hi);
    if (value.hi == 0.0f || !Gtsv2Finite(value.hi)) {
        return normalized;
    }
    normalized.lo = Gtsv2ScaleSignificand(value.lo, -static_cast<int32_t>(normalized.exponent));
    normalized.exponent += exponent;
    return normalized;
}

__aicore__ inline Gtsv2Scaled Gtsv2ScaledNegate(Gtsv2Scaled value)
{
    return {-value.hi, -value.lo, value.exponent};
}

__aicore__ inline Gtsv2Scaled Gtsv2ScaledAdd(Gtsv2Scaled a, Gtsv2Scaled b)
{
    if (!Gtsv2Finite(a.hi) || !Gtsv2Finite(b.hi)) {
        return Gtsv2ScaledFromFloat(a.hi + b.hi);
    }
    if (a.hi == 0.0f && b.hi == 0.0f) { return Gtsv2ScaledFromFloat(a.hi + b.hi); }
    if (a.hi == 0.0f) { return b; }
    if (b.hi == 0.0f) { return a; }
    if (a.exponent < b.exponent) {
        Gtsv2Scaled temporary = a; a = b; b = temporary;
    }
    const int64_t difference = a.exponent - b.exponent;
    // More than the two-float significand plus a guard word. This is ordinary
    // precision rounding, not a truncation of the stored exponent range.
    if (difference > 80) { return a; }
    Gtsv2FloatPair small{Gtsv2ScaleSignificand(b.hi, -static_cast<int32_t>(difference)),
                         Gtsv2ScaleSignificand(b.lo, -static_cast<int32_t>(difference))};
    return Gtsv2ScaledNormalize(Gtsv2PairAdd({a.hi, a.lo}, small), a.exponent);
}

__aicore__ inline Gtsv2Scaled Gtsv2ScaledMultiply(Gtsv2Scaled a, Gtsv2Scaled b)
{
    if (a.hi == 0.0f || b.hi == 0.0f || !Gtsv2Finite(a.hi) || !Gtsv2Finite(b.hi)) {
        return Gtsv2ScaledFromFloat(a.hi * b.hi);
    }
    Gtsv2FloatPair product = Gtsv2TwoProduct(a.hi, b.hi);
    product.lo += (a.hi * b.lo + a.lo * b.hi) + a.lo * b.lo;
    return Gtsv2ScaledNormalize(product, a.exponent + b.exponent);
}

__aicore__ inline Gtsv2Scaled Gtsv2ScaledDivide(Gtsv2Scaled a, Gtsv2Scaled b)
{
    if (a.hi == 0.0f || b.hi == 0.0f || !Gtsv2Finite(a.hi) || !Gtsv2Finite(b.hi)) {
        return Gtsv2ScaledFromFloat(a.hi / b.hi);
    }
    float quotient = a.hi / b.hi;
    Gtsv2FloatPair product = Gtsv2TwoProduct(b.hi, quotient);
    product.lo += b.lo * quotient;
    Gtsv2FloatPair residual = Gtsv2PairAdd({a.hi, a.lo}, {-product.hi, -product.lo});
    float correction = (residual.hi + residual.lo) / b.hi;
    return Gtsv2ScaledNormalize(Gtsv2TwoSum(quotient, correction), a.exponent - b.exponent);
}

__aicore__ inline Gtsv2Scaled Gtsv2ScaledSubtractProduct(
    Gtsv2Scaled value, Gtsv2Scaled factor, Gtsv2Scaled other)
{
    return Gtsv2ScaledAdd(value, Gtsv2ScaledNegate(Gtsv2ScaledMultiply(factor, other)));
}

__aicore__ inline bool Gtsv2ScaledAbsGreaterEqual(Gtsv2Scaled a, Gtsv2Scaled b)
{
    if (b.hi == 0.0f) { return true; }
    if (a.hi == 0.0f) { return false; }
    if (a.exponent != b.exponent) { return a.exponent > b.exponent; }
    float aHi = a.hi < 0.0f ? -a.hi : a.hi;
    float bHi = b.hi < 0.0f ? -b.hi : b.hi;
    if (aHi != bHi) { return aHi > bHi; }
    return (a.hi < 0.0f ? -a.lo : a.lo) >= (b.hi < 0.0f ? -b.lo : b.lo);
}

__aicore__ inline float Gtsv2ScaledToFloat(Gtsv2Scaled value)
{
    if (value.hi == 0.0f || !Gtsv2Finite(value.hi)) { return value.hi; }
    uint32_t bits = __builtin_bit_cast(uint32_t, value.hi);
    uint32_t sign = bits & 0x80000000U;
    if (value.exponent < -126) {
        if (value.exponent < -150) { return __builtin_bit_cast(float, sign); }
        // Integer round-to-nearest-even also preserves subnormal outputs on
        // vector cores whose arithmetic flushes subnormal operands/results.
        uint32_t shift = static_cast<uint32_t>(-126 - value.exponent);
        uint32_t significand = (bits & 0x7fffffU) | 0x800000U;
        uint32_t result = significand >> shift;
        uint32_t remainder = significand & ((1U << shift) - 1U);
        uint32_t halfway = 1U << (shift - 1U);
        float low = sign != 0 ? -value.lo : value.lo;
        if (remainder > halfway || (remainder == halfway &&
            (low > 0.0f || (low == 0.0f && (result & 1U) != 0)))) {
            ++result;
        }
        return __builtin_bit_cast(float, sign | result);
    }
    Gtsv2Scaled rounded = Gtsv2ScaledFromFloat(value.hi + value.lo);
    int64_t exponent = value.exponent + rounded.exponent;
    if (exponent > 127) { return __builtin_bit_cast(float, sign | 0x7f800000U); }
    return __builtin_bit_cast(float, sign | (static_cast<uint32_t>(exponent + 127) << 23) |
        (__builtin_bit_cast(uint32_t, rounded.hi) & 0x7fffffU));
}

// Independent exponents prevent overflow, but unbalanced rows can still cause
// long pivot chains with cancellation during back substitution. Scale each
// original row near unity before choosing pivots. Adjust only the exponent:
// converting the scaled coefficients to FP32 would erase tiny terms in a row.
__aicore__ inline int64_t Gtsv2ScaledRowExponent(Gtsv2Scaled lower, Gtsv2Scaled diagonal, Gtsv2Scaled upper)
{
    Gtsv2Scaled largest = diagonal;
    if (lower.hi != 0.0f && (largest.hi == 0.0f || lower.exponent > largest.exponent)) { largest = lower; }
    if (upper.hi != 0.0f && (largest.hi == 0.0f || upper.exponent > largest.exponent)) { largest = upper; }
    return largest.exponent;
}

template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
__aicore__ inline void Gtsv2ScaledLu<FloatTensor, ExponentTensor, PivotTensor>::Init(
    FloatTensor significands, ExponentTensor exponents, PivotTensor pivots, FloatTensor original, int32_t m)
{
    significands_ = significands; exponents_ = exponents; pivots_ = pivots; original_ = original; m_ = m;
}

template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
__aicore__ inline Gtsv2Scaled Gtsv2ScaledLu<FloatTensor, ExponentTensor, PivotTensor>::Get(int64_t index)
{
    return {significands_.GetValue(2 * index), significands_.GetValue(2 * index + 1),
            exponents_.GetValue(index)};
}

template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
__aicore__ inline void Gtsv2ScaledLu<FloatTensor, ExponentTensor, PivotTensor>::Set(int64_t index, Gtsv2Scaled value)
{
    significands_.SetValue(2 * index, value.hi);
    significands_.SetValue(2 * index + 1, value.lo);
    exponents_.SetValue(index, value.exponent);
}

template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
template <typename CoefficientTensor>
__aicore__ inline bool Gtsv2ScaledLu<FloatTensor, ExponentTensor, PivotTensor>::Factor(
    CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du, bool equilibrateRows)
{
    equilibrateRows_ = equilibrateRows;
    Gtsv2Scaled diagonal = Gtsv2ScaledFromFloat(d.GetValue(0));
    Gtsv2Scaled upper = Gtsv2ScaledFromFloat(du.GetValue(0));
    const int64_t firstScale = equilibrateRows_ ?
        Gtsv2ScaledRowExponent(Gtsv2ScaledFromFloat(0.0f), diagonal, upper) : 0;
    diagonal.exponent -= firstScale;
    upper.exponent -= firstScale;
    for (int32_t row = 1; row < m_; ++row) {
        Gtsv2Scaled lower = Gtsv2ScaledFromFloat(dl.GetValue(row));
        Gtsv2Scaled current = Gtsv2ScaledFromFloat(d.GetValue(row));
        Gtsv2Scaled currentUpper = Gtsv2ScaledFromFloat(row + 1 == m_ ? 0.0f : du.GetValue(row));
        const int64_t scale = equilibrateRows_ ? Gtsv2ScaledRowExponent(lower, current, currentUpper) : 0;
        lower.exponent -= scale;
        current.exponent -= scale;
        currentUpper.exponent -= scale;
        bool swapped = !Gtsv2ScaledAbsGreaterEqual(diagonal, lower);
        Gtsv2Scaled pivot = swapped ? lower : diagonal;
        if (pivot.hi == 0.0f) { return false; }
        Gtsv2Scaled factor = Gtsv2ScaledDivide(swapped ? diagonal : lower, pivot);
        Set(row - 1, pivot);
        Set(1LL * m_ + row - 1, swapped ? current : upper);
        Set(2LL * m_ + row - 1, swapped ? currentUpper : Gtsv2ScaledFromFloat(0.0f));
        Set(3LL * m_ + row - 1, factor);
        pivots_.SetValue(row - 1, static_cast<uint8_t>(swapped));
        diagonal = swapped ? Gtsv2ScaledSubtractProduct(upper, factor, current) :
                             Gtsv2ScaledSubtractProduct(current, factor, upper);
        upper = swapped ? Gtsv2ScaledNegate(Gtsv2ScaledMultiply(factor, currentUpper)) : currentUpper;
    }
    Set(m_ - 1, diagonal);
    return diagonal.hi != 0.0f;
}

template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
template <typename RhsTensor, typename CoefficientTensor>
__aicore__ inline void Gtsv2ScaledLu<FloatTensor, ExponentTensor, PivotTensor>::SolveColumn(
    RhsTensor rhs, int64_t offset, CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du)
{
    const int64_t base = 4LL * m_;
    for (int32_t row = 0; row < m_; ++row) {
        const int64_t scale = equilibrateRows_ ? Gtsv2ScaledRowExponent(
            Gtsv2ScaledFromFloat(row == 0 ? 0.0f : dl.GetValue(row)),
            Gtsv2ScaledFromFloat(d.GetValue(row)),
            Gtsv2ScaledFromFloat(row + 1 == m_ ? 0.0f : du.GetValue(row))) : 0;
        Gtsv2Scaled current = Gtsv2ScaledFromFloat(rhs.GetValue(offset + row));
        current.exponent -= scale;  // Apply the original row scale before pivoting the RHS.
        Set(base + row, current);
    }
    SolvePreparedRhs();
}

template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
__aicore__ inline void Gtsv2ScaledLu<FloatTensor, ExponentTensor, PivotTensor>::SolvePreparedRhs()
{
    const int64_t base = 4LL * m_;
    for (int32_t row = 1; row < m_; ++row) {
        Gtsv2Scaled previous = Get(base + row - 1);
        Gtsv2Scaled current = Get(base + row);
        Gtsv2Scaled factor = Get(3LL * m_ + row - 1);
        bool swapped = pivots_.GetValue(row - 1) != 0;
        if (swapped) { Set(base + row - 1, current); }
        Set(base + row, swapped ? Gtsv2ScaledSubtractProduct(previous, factor, current) :
                                 Gtsv2ScaledSubtractProduct(current, factor, previous));
    }
    Gtsv2Scaled next = Gtsv2ScaledDivide(Get(base + m_ - 1), Get(m_ - 1));
    Gtsv2Scaled next2 = Gtsv2ScaledFromFloat(0.0f);
    Set(base + m_ - 1, next);
    for (int32_t row = m_ - 2; row >= 0; --row) {
        Gtsv2Scaled numerator = Gtsv2ScaledSubtractProduct(Get(base + row), Get(1LL * m_ + row), next);
        if (row + 2 < m_) { numerator = Gtsv2ScaledSubtractProduct(numerator, Get(2LL * m_ + row), next2); }
        Gtsv2Scaled value = Gtsv2ScaledDivide(numerator, Get(row));
        next2 = next; next = value;
        Set(base + row, value);
    }
}

__aicore__ inline Gtsv2Scaled Gtsv2ScaledResidual(float b, float lower, float diagonal, float upper,
    float previous, float x, float next, Gtsv2Scaled &denominator)
{
    Gtsv2Scaled terms[4] = {Gtsv2ScaledFromFloat(b),
        Gtsv2ScaledMultiply(Gtsv2ScaledFromFloat(-lower), Gtsv2ScaledFromFloat(previous)),
        Gtsv2ScaledMultiply(Gtsv2ScaledFromFloat(-diagonal), Gtsv2ScaledFromFloat(x)),
        Gtsv2ScaledMultiply(Gtsv2ScaledFromFloat(-upper), Gtsv2ScaledFromFloat(next))};
    denominator = Gtsv2ScaledFromFloat(0.0f);
    for (int32_t i = 0; i < 4; ++i) {
        denominator = Gtsv2ScaledAdd(denominator,
            terms[i].hi < 0.0f ? Gtsv2ScaledNegate(terms[i]) : terms[i]);
    }
    // Cancel the largest products before adding small terms. Keep the residual
    // in expanded range, including values smaller than the FP32 subnormal range.
    for (int32_t i = 1; i < 4; ++i) {
        for (int32_t j = i; j > 0 && Gtsv2ScaledAbsGreaterEqual(terms[j], terms[j - 1]); --j) {
            Gtsv2Scaled temporary = terms[j]; terms[j] = terms[j - 1]; terms[j - 1] = temporary;
        }
    }
    Gtsv2Scaled residual = Gtsv2ScaledFromFloat(0.0f);
    for (int32_t i = 0; i < 4; ++i) { residual = Gtsv2ScaledAdd(residual, terms[i]); }
    return residual;
}

template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
template <typename RhsTensor, typename CoefficientTensor>
__aicore__ inline float Gtsv2ScaledLu<FloatTensor, ExponentTensor, PivotTensor>::MaxBackwardError(
    RhsTensor rhs, int64_t offset, CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du)
{
    const int64_t base = 4LL * m_;
    float maximum = 0.0f;
    for (int32_t row = 0; row < m_; ++row) {
        const float previous = row == 0 ? 0.0f : Gtsv2ScaledToFloat(Get(base + row - 1));
        const float x = Gtsv2ScaledToFloat(Get(base + row));
        const float next = row + 1 == m_ ? 0.0f : Gtsv2ScaledToFloat(Get(base + row + 1));
        if (!Gtsv2Finite(previous) || !Gtsv2Finite(x) || !Gtsv2Finite(next)) { return 2.0f; }
        Gtsv2Scaled denominator = Gtsv2ScaledFromFloat(0.0f);
        Gtsv2Scaled residual = Gtsv2ScaledResidual(rhs.GetValue(offset + row),
            row == 0 ? 0.0f : dl.GetValue(row), d.GetValue(row), row + 1 == m_ ? 0.0f : du.GetValue(row),
            previous, x, next, denominator);
        if (residual.hi < 0.0f) { residual = Gtsv2ScaledNegate(residual); }
        const float error = denominator.hi == 0.0f ? 0.0f :
            Gtsv2ScaledToFloat(Gtsv2ScaledDivide(residual, denominator));
        if (!Gtsv2Finite(error)) { return 2.0f; }
        if (error > maximum) { maximum = error; }
    }
    return maximum;
}

template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
template <typename RhsTensor, typename CoefficientTensor>
__aicore__ inline void Gtsv2ScaledLu<FloatTensor, ExponentTensor, PivotTensor>::Solve(
    RhsTensor rhs, int64_t offset, CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du)
{
    SolveColumn(rhs, offset, dl, d, du);
    const float firstError = MaxBackwardError(rhs, offset, dl, d, du);
    // Row equilibration can change pivots adversely when solution components
    // have very different magnitudes. Preserve original B until both choices
    // can be compared using their final FP32 output, not just internal precision.
    constexpr float retryThreshold = 1e-6f;
    if (firstError > retryThreshold) {
        const bool firstMode = equilibrateRows_;
        const bool alternateValid = Factor(dl, d, du, !firstMode);
        if (alternateValid) { SolveColumn(rhs, offset, dl, d, du); }
        if (!alternateValid || MaxBackwardError(rhs, offset, dl, d, du) > firstError) {
            Factor(dl, d, du, firstMode);
            SolveColumn(rhs, offset, dl, d, du);
        }
    }
    // Retain original B separately while the fifth Scaled array is reused for
    // residuals and corrections. The selected factorization remains unchanged
    // during refinement; a small backward error is not a stopping criterion.
    for (int32_t row = 0; row < m_; ++row) {
        original_.SetValue(row, rhs.GetValue(offset + row));
        rhs.SetValue(offset + row, Gtsv2ScaledToFloat(Get(4LL * m_ + row)));
    }
    RefineColumn(rhs, offset, dl, d, du);
}

template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
template <typename RhsTensor, typename CoefficientTensor>
__aicore__ inline bool Gtsv2ScaledLu<FloatTensor, ExponentTensor, PivotTensor>::PrepareResidual(
    RhsTensor rhs, int64_t offset, CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du)
{
    for (int32_t row = 0; row < m_; ++row) {
        const float previous = row == 0 ? 0.0f : rhs.GetValue(offset + row - 1);
        const float x = rhs.GetValue(offset + row);
        const float next = row + 1 == m_ ? 0.0f : rhs.GetValue(offset + row + 1);
        if (!Gtsv2Finite(previous) || !Gtsv2Finite(x) || !Gtsv2Finite(next)) { return false; }
        const float lower = row == 0 ? 0.0f : dl.GetValue(row);
        const float upper = row + 1 == m_ ? 0.0f : du.GetValue(row);
        Gtsv2Scaled denominator = Gtsv2ScaledFromFloat(0.0f);
        Gtsv2Scaled residual = Gtsv2ScaledResidual(original_.GetValue(row), lower, d.GetValue(row), upper,
            previous, x, next, denominator);
        residual.exponent -= equilibrateRows_ ? Gtsv2ScaledRowExponent(Gtsv2ScaledFromFloat(lower),
            Gtsv2ScaledFromFloat(d.GetValue(row)), Gtsv2ScaledFromFloat(upper)) : 0;
        Set(4LL * m_ + row, residual);
    }
    return true;
}

template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
template <typename RhsTensor>
__aicore__ inline bool Gtsv2ScaledLu<FloatTensor, ExponentTensor, PivotTensor>::ApplyCorrection(
    RhsTensor rhs, int64_t offset)
{
    for (int32_t row = 0; row < m_; ++row) {
        Gtsv2Scaled current = Gtsv2ScaledFromFloat(rhs.GetValue(offset + row));
        Gtsv2Scaled correction = Get(4LL * m_ + row);
        // A zero correction must not change an existing signed zero.
        Gtsv2Scaled sum = correction.hi == 0.0f ? current : Gtsv2ScaledAdd(current, correction);
        if (!Gtsv2Finite(Gtsv2ScaledToFloat(sum))) { return false; }
        Set(4LL * m_ + row, sum);
    }
    for (int32_t row = 0; row < m_; ++row) {
        rhs.SetValue(offset + row, Gtsv2ScaledToFloat(Get(4LL * m_ + row)));
    }
    return true;
}

template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
template <typename RhsTensor, typename CoefficientTensor>
__aicore__ inline void Gtsv2ScaledLu<FloatTensor, ExponentTensor, PivotTensor>::RefineColumn(
    RhsTensor rhs, int64_t offset, CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du)
{
    // Small backward error alone does not bound forward error for an
    // ill-conditioned system. Always try both corrections for finite output.
    constexpr int32_t maxCorrections = 2;
    for (int32_t iteration = 0; iteration < maxCorrections; ++iteration) {
        if (!PrepareResidual(rhs, offset, dl, d, du)) { return; }
        SolvePreparedRhs();
        if (!ApplyCorrection(rhs, offset)) { return; }
    }
}

template <typename Tensor>
__aicore__ inline bool Gtsv2NeedsRefinement(Tensor dl, Tensor d, Tensor du, int32_t m)
{
    bool strongDominance = true;
    for (int32_t row = 0; row < m; ++row) {
        float lower = row == 0 ? 0.0f : dl.GetValue(row);
        float diagonal = d.GetValue(row);
        float upper = row + 1 == m ? 0.0f : du.GetValue(row);
        if (!Gtsv2Finite(lower) || !Gtsv2Finite(diagonal) || !Gtsv2Finite(upper)) {
            return false;  // Preserve existing nonfinite propagation.
        }
        float absLower = lower < 0 ? -lower : lower;
        float absD = diagonal < 0 ? -diagonal : diagonal;
        float absUpper = upper < 0 ? -upper : upper;
        strongDominance = strongDominance && absD > 2.0f * (absLower + absUpper);
    }
    return !strongDominance;
}

__aicore__ inline void Gtsv2Refinement::Init(
        AscendC::GlobalTensor<float> dl, AscendC::GlobalTensor<float> d,
        AscendC::GlobalTensor<float> du, AscendC::GlobalTensor<float> factors,
        AscendC::GlobalTensor<uint8_t> pivots, AscendC::GlobalTensor<float> b,
        GM_ADDR workspace, int32_t m, int32_t n, int32_t ldb)
{
    dl_ = dl; d_ = d; du_ = du; factors_ = factors; pivots_ = pivots; b_ = b;
    m_ = m; n_ = n; ldb_ = ldb;
    const uint64_t factorBytes = (kGtsv2PackedFactorBytesPerRow * m + 127ULL) & ~127ULL;
    auto *scratch = reinterpret_cast<__gm__ float *>(workspace + factorBytes);
    original_.SetGlobalBuffer(scratch, m);
    correction_.SetGlobalBuffer(scratch + m, m);
}

__aicore__ inline void Gtsv2Refinement::Process()
{
    for (int64_t column = 0; column < n_; ++column) {
        const int64_t offset = column * ldb_;
        bool finite = true;
        for (int32_t row = 0; row < m_; ++row) {
            float value = b_.GetValue(offset + row);
            original_.SetValue(row, value);
            finite = finite && Gtsv2Finite(value);
        }
        Solve(b_, offset);
        if (finite && AllFinite(b_, offset)) {
            RefineColumn(offset);
        }
    }
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
        AscendC::DcciDst::CACHELINE_OUT>(b_);
}

__aicore__ inline void Gtsv2Refinement::SolveNonfiniteColumn(int64_t offset)
{
    Solve(b_, offset);
}

__aicore__ inline bool Gtsv2Refinement::AllFinite(AscendC::GlobalTensor<float> values, int64_t offset)
{
    for (int32_t row = 0; row < m_; ++row) {
        if (!Gtsv2Finite(values.GetValue(offset + row))) {
            return false;
        }
    }
    return true;
}

__aicore__ inline void Gtsv2Refinement::Solve(AscendC::GlobalTensor<float> rhs, int64_t offset)
{
    for (int32_t row = 1; row < m_; ++row) {
        float previous = rhs.GetValue(offset + row - 1);
        float current = rhs.GetValue(offset + row);
        float factor = factors_.GetValue(3LL * m_ + row - 1);
        if (pivots_.GetValue(row - 1) == 0) {
            rhs.SetValue(offset + row, current - factor * previous);
        } else {
            rhs.SetValue(offset + row - 1, current);
            rhs.SetValue(offset + row, previous - factor * current);
        }
    }
    // Zero pivots intentionally retain the Legacy API's Inf/NaN behavior;
    // refinement is skipped when this initial solve is nonfinite.
    float next = rhs.GetValue(offset + m_ - 1) / factors_.GetValue(m_ - 1);
    float next2 = 0.0f;
    rhs.SetValue(offset + m_ - 1, next);
    for (int32_t row = m_ - 2; row >= 0; --row) {
        float numerator = rhs.GetValue(offset + row) - factors_.GetValue(m_ + row) * next;
        if (row + 2 < m_) {
            numerator -= factors_.GetValue(2LL * m_ + row) * next2;
        }
        float x = numerator / factors_.GetValue(row);
        next2 = next;
        next = x;
        rhs.SetValue(offset + row, x);
    }
}

__aicore__ inline bool Gtsv2Refinement::FormResidual(int64_t offset)
{
    bool finite = true;
    for (int32_t row = 0; row < m_; ++row) {
        float lower = row == 0 ? 0.0f : dl_.GetValue(row);
        float upper = row + 1 == m_ ? 0.0f : du_.GetValue(row);
        float previous = row == 0 ? 0.0f : b_.GetValue(offset + row - 1);
        float next = row + 1 == m_ ? 0.0f : b_.GetValue(offset + row + 1);
        float residual = Gtsv2AccurateResidual(original_.GetValue(row), lower,
            d_.GetValue(row), upper, previous, b_.GetValue(offset + row), next);
        correction_.SetValue(row, residual);
        finite = finite && Gtsv2Finite(residual);
    }
    return finite;
}

__aicore__ inline void Gtsv2Refinement::RefineColumn(int64_t offset)
{
    constexpr int32_t maxCorrections = 2;
    for (int32_t iteration = 0; iteration < maxCorrections; ++iteration) {
        if (!FormResidual(offset)) {
            return;
        }
        Solve(correction_, 0);
        if (!AllFinite(correction_, 0)) {
            return;
        }
        for (int32_t row = 0; row < m_; ++row) {
            b_.SetValue(offset + row, b_.GetValue(offset + row) + correction_.GetValue(row));
        }
    }
}

__aicore__ inline void Gtsv2RangeSolver::Init(
        AscendC::GlobalTensor<float> dl, AscendC::GlobalTensor<float> d,
        AscendC::GlobalTensor<float> du, AscendC::GlobalTensor<float> factors,
        AscendC::GlobalTensor<uint8_t> pivots, AscendC::GlobalTensor<float> b,
        GM_ADDR workspace, int32_t m, int32_t n, int32_t ldb)
{
    dl_ = dl; d_ = d; du_ = du; b_ = b; m_ = m; n_ = n; ldb_ = ldb;
    legacy_.Init(dl, d, du, factors, pivots, b, workspace, m, n, ldb);
    const uint64_t factorBytes = (kGtsv2PackedFactorBytesPerRow * m + 127ULL) & ~127ULL;
    auto *scratch = workspace + factorBytes;
    significands_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch), 10ULL * m);
    exponents_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(scratch + 40ULL * m), 5ULL * m);
    original_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + 80ULL * m), m);
    pivots_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(scratch + 84ULL * m), m);
}

__aicore__ inline void Gtsv2RangeSolver::Process()
{
    if (!FiniteCoefficients()) {
        legacy_.Process();
        return;
    }
    Gtsv2ScaledLu<AscendC::GlobalTensor<float>, AscendC::GlobalTensor<int64_t>,
        AscendC::GlobalTensor<uint8_t>> solver;
    solver.Init(significands_, exponents_, pivots_, original_, m_);
    if (!solver.Factor(dl_, d_, du_) && !solver.Factor(dl_, d_, du_, false)) {
        // An exact zero pivot retains the Legacy API's Inf/NaN behavior.
        legacy_.Process();
        return;
    }
    for (int64_t column = 0; column < n_; ++column) {
        const int64_t offset = column * ldb_;
        bool finite = true;
        for (int32_t row = 0; row < m_; ++row) {
            finite = finite && Gtsv2Finite(b_.GetValue(offset + row));
        }
        if (finite) {
            solver.Solve(b_, offset, dl_, d_, du_);
        } else {
            legacy_.SolveNonfiniteColumn(offset);
        }
    }
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
        AscendC::DcciDst::CACHELINE_OUT>(b_);
}

__aicore__ inline bool Gtsv2RangeSolver::FiniteCoefficients()
{
    for (int32_t row = 0; row < m_; ++row) {
        if (!Gtsv2Finite(d_.GetValue(row)) ||
            (row != 0 && !Gtsv2Finite(dl_.GetValue(row))) ||
            (row + 1 != m_ && !Gtsv2Finite(du_.GetValue(row)))) {
            return false;
        }
    }
    return true;
}

__aicore__ inline bool Gtsv2JacobiEligible(
    int32_t m, AscendC::LocalTensor<float> lower, AscendC::LocalTensor<float> diagonal,
    AscendC::LocalTensor<float> upper, AscendC::LocalTensor<float> rhs,
    AscendC::LocalTensor<float> temporary, AscendC::LocalTensor<float> sumStorage,
    AscendC::LocalTensor<float> boundStorage, event_t vectorToScalar)
{
    using namespace AscendC;
    // Test before changing any coefficient or RHS. Summation intentionally
    // rejects finite inputs whose absolute sum overflows as well as Inf/NaN.
    Abs(temporary, rhs, m);
    Abs(sumStorage, diagonal, m);
    PipeBarrier<PIPE_V>();
    Add(temporary, temporary, sumStorage, m);
    Abs(sumStorage, lower, m);
    PipeBarrier<PIPE_V>();
    Add(temporary, temporary, sumStorage, m);
    Abs(sumStorage, upper, m);
    PipeBarrier<PIPE_V>();
    Add(temporary, temporary, sumStorage, m);
    PipeBarrier<PIPE_V>();
    ReduceSum(sumStorage, temporary, sumStorage[8], m);
    SetFlag<HardEvent::V_S>(vectorToScalar);
    WaitFlag<HardEvent::V_S>(vectorToScalar);
    const float absoluteSum = sumStorage.GetValue(0);
    if (!Gtsv2Finite(absoluteSum)) {
        return false;
    }
    Abs(temporary, diagonal, m);
    PipeBarrier<PIPE_V>();
    ReduceMin(boundStorage, temporary, boundStorage[8], m, false);
    SetFlag<HardEvent::V_S>(vectorToScalar);
    WaitFlag<HardEvent::V_S>(vectorToScalar);
    const float minimumDiagonal = boundStorage.GetValue(0);
    // The caller establishes strict diagonal dominance. Keep the divisor
    // precondition local too, before evaluating the normalized RHS bound.
    if (!(minimumDiagonal > 0.0f)) {
        return false;
    }
    Abs(temporary, rhs, m);
    PipeBarrier<PIPE_V>();
    ReduceMax(boundStorage, temporary, boundStorage[8], m, false);
    SetFlag<HardEvent::V_S>(vectorToScalar);
    WaitFlag<HardEvent::V_S>(vectorToScalar);
    const float bound = boundStorage.GetValue(0) / minimumDiagonal;
    return Gtsv2Finite(bound) && bound <= 4.0f;
}

__aicore__ inline bool Gtsv2TryJacobi(
    int32_t m,
    AscendC::LocalTensor<float> lower, AscendC::LocalTensor<float> diagonal,
    AscendC::LocalTensor<float> upper, AscendC::LocalTensor<float> rhs,
    AscendC::LocalTensor<float> temporary,
    AscendC::LocalTensor<float> xStorage, AscendC::LocalTensor<float> nextStorage,
    event_t scalarToVector, event_t vectorToScalar)
{
    using namespace AscendC;
    SetFlag<HardEvent::S_V>(scalarToVector);
    WaitFlag<HardEvent::S_V>(scalarToVector);
    if (!Gtsv2JacobiEligible(m, lower, diagonal, upper, rhs, temporary,
                              xStorage, nextStorage, vectorToScalar)) {
        return false;
    }

    Div(lower, lower, diagonal, m);
    Div(upper, upper, diagonal, m);
    Div(rhs, rhs, diagonal, m);
    PipeBarrier<PIPE_V>();
    // The original diagonal is dead after normalization. Reuse it as gather
    // offsets; each ping-pong tensor has 8-float guards on both ends.
    auto indices = diagonal.ReinterpretCast<uint32_t>();
    CreateVecIndex(diagonal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0), m);
    PipeBarrier<PIPE_V>();
    ShiftLeft(indices, indices, static_cast<uint32_t>(2), m);
    const uint32_t guardedRows = ((m + 7) & ~7) + 16;
    Duplicate(xStorage, 0.0f, guardedRows);
    Duplicate(nextStorage, 0.0f, guardedRows);
    PipeBarrier<PIPE_V>();
    for (int32_t iteration = 0; iteration < kGtsv2JacobiIters; ++iteration) {
        auto next = nextStorage[8];
        Gather(next, xStorage, indices, 7 * sizeof(float), m);
        Gather(temporary, xStorage, indices, 9 * sizeof(float), m);
        PipeBarrier<PIPE_V>();
        Mul(next, next, lower, m);
        Mul(temporary, temporary, upper, m);
        PipeBarrier<PIPE_V>();
        Add(next, next, temporary, m);
        PipeBarrier<PIPE_V>();
        Sub(next, rhs, next, m);
        PipeBarrier<PIPE_V>();
        auto previous = xStorage;
        xStorage = nextStorage;
        nextStorage = previous;
    }
    Adds(rhs, xStorage[8], 0.0f, m);
    SetFlag<HardEvent::V_S>(vectorToScalar);
    WaitFlag<HardEvent::V_S>(vectorToScalar);
    return true;
}

__aicore__ inline int32_t Gtsv2IterateWideJacobi(
    int32_t m, int32_t stride, AscendC::LocalTensor<float> storage,
    AscendC::GlobalTensor<float> rhsGm, int64_t offset,
    event_t vToMte2, event_t mte2ToV)
{
    using namespace AscendC;
    auto lower = storage;
    auto upper = storage[stride];
    auto indices = storage[4 * stride].ReinterpretCast<uint32_t>();
    auto temporary = storage[5 * stride];
    const DataCopyExtParams copy{1, static_cast<uint32_t>(m * sizeof(float)), 0, 0, 0};
    const DataCopyPadExtParams<float> padding{false, 0, 0, 0.0f};
    int32_t currentBase = 2 * stride;
    int32_t nextBase = 3 * stride;
    for (int32_t iteration = 0; iteration < kGtsv2JacobiIters; ++iteration) {
        auto next = storage[nextBase];
        Gather(next, storage, indices, (currentBase - 1) * sizeof(float), m);
        Gather(temporary, storage, indices, (currentBase + 1) * sizeof(float), m);
        PipeBarrier<PIPE_V>();
        Mul(next, next, lower, m);
        Mul(temporary, temporary, upper, m);
        PipeBarrier<PIPE_V>();
        Add(next, next, temporary, m);
        SetFlag<HardEvent::V_MTE2>(vToMte2);
        WaitFlag<HardEvent::V_MTE2>(vToMte2);
        DataCopyPad(temporary, rhsGm[offset], copy, padding);
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        Sub(next, temporary, next, m);
        PipeBarrier<PIPE_V>();
        int32_t oldBase = currentBase;
        currentBase = nextBase;
        nextBase = oldBase;
    }
    return currentBase;
}

__aicore__ inline void Gtsv2InitializeWideJacobi(
    int32_t m, int32_t stride, AscendC::LocalTensor<float> storage)
{
    using namespace AscendC;
    auto diagonal = storage[4 * stride];
    auto indices = diagonal.ReinterpretCast<uint32_t>();
    CreateVecIndex(diagonal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0), m);
    PipeBarrier<PIPE_V>();
    ShiftLeft(indices, indices, static_cast<uint32_t>(2), m);
    Duplicate(storage[2 * stride], 0.0f, stride);
    Duplicate(storage[3 * stride], 0.0f, stride);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline bool Gtsv2TryWideJacobi(
    int32_t m, AscendC::LocalTensor<float> storage,
    AscendC::GlobalTensor<float> lowerGm, AscendC::GlobalTensor<float> diagonalGm,
    AscendC::GlobalTensor<float> upperGm, AscendC::GlobalTensor<float> rhsGm,
    int64_t offset, event_t sToV, event_t vToS, event_t mte2ToV,
    event_t vToMte2, event_t vToMte3, event_t mte3ToV)
{
    using namespace AscendC;
    const int32_t stride = (m + 7) & ~7;
    auto lower = storage;
    auto upper = storage[stride];
    auto initialRhs = storage[2 * stride];
    auto reduction = storage[3 * stride];
    auto diagonal = storage[4 * stride];
    auto temporary = storage[5 * stride];
    const DataCopyExtParams copy{1, static_cast<uint32_t>(m * sizeof(float)), 0, 0, 0};
    const DataCopyPadExtParams<float> padding{false, 0, 0, 0.0f};
    DataCopyPad(lower, lowerGm, copy, padding);
    DataCopyPad(upper, upperGm, copy, padding);
    DataCopyPad(diagonal, diagonalGm, copy, padding);
    DataCopyPad(initialRhs, rhsGm[offset], copy, padding);
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);
    if (!Gtsv2JacobiEligible(m, lower, diagonal, upper, initialRhs, temporary,
                              reduction, reduction, vToS)) {
        return false;
    }
    lower.SetValue(0, 0.0f);
    upper.SetValue(m - 1, 0.0f);
    SetFlag<HardEvent::S_V>(sToV);
    WaitFlag<HardEvent::S_V>(sToV);
    Div(lower, lower, diagonal, m);
    Div(upper, upper, diagonal, m);
    Div(initialRhs, initialRhs, diagonal, m);
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    DataCopyPad(rhsGm[offset], initialRhs, copy);
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
    Gtsv2InitializeWideJacobi(m, stride, storage);
    const int32_t currentBase = Gtsv2IterateWideJacobi(
        m, stride, storage, rhsGm, offset, vToMte2, mte2ToV);
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    DataCopyPad(rhsGm[offset], storage[currentBase], copy);
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
    return true;
}

namespace {

__aicore__ inline float Gtsv2Abs(float value)
{
    return value < 0.0f ? -value : value;
}

__aicore__ inline float Gtsv2PivotRatio(float numerator, float denominator)
{
    if (denominator == 0.0f) {
        return __builtin_bit_cast(float, uint32_t{0x7fc00000U});
    }
    return numerator / denominator;
}

__aicore__ inline DataCopyExtParams OneBlockCopy(uint32_t bytes)
{
    return DataCopyExtParams{1, bytes, 0, 0, 0};
}

template <typename T>
__aicore__ inline DataCopyPadExtParams<T> NoPadding()
{
    return DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0)};
}

__aicore__ inline uint64_t Gtsv2FactorBytes(int32_t m)
{
    return static_cast<uint64_t>(kGtsv2FactorArrays) * static_cast<uint64_t>(m) * kFloatBytes;
}

__aicore__ inline int32_t Gtsv2AlignedRows(int32_t m)
{
    constexpr int32_t rowsPerDataBlock = 32 / sizeof(float);
    return (m + rowsPerDataBlock - 1) & ~(rowsPerDataBlock - 1);
}

__aicore__ inline uint32_t Gtsv2FactorStorageBytes(int32_t m)
{
    return static_cast<uint32_t>(kGtsv2FactorArrays) *
           static_cast<uint32_t>(Gtsv2AlignedRows(m)) * kFloatBytes;
}

template <bool SingleRhs>
__aicore__ inline void Gtsv2FactorKernel<SingleRhs>::Init(
        GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR workspace, GM_ADDR B,
        const Gtsv2TilingData &tiling)
{
    m_ = tiling.m;
    n_ = tiling.n;
    ldb_ = tiling.ldb;
    workspace_ = workspace;
    useUb_ = tiling.useUb != 0;
    dlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dl), m_);
    dGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d), m_);
    duGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(du), m_);

    BGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(B), static_cast<uint64_t>(ldb_) * n_);
    auto *workspaceBytes = reinterpret_cast<__gm__ uint8_t *>(workspace);
    factorGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(workspaceBytes),
        static_cast<uint64_t>(kGtsv2FactorArrays) * m_);
    pivotGm_.SetGlobalBuffer(
        workspaceBytes + Gtsv2FactorBytes(m_), static_cast<uint64_t>(m_));

    if (useUb_) {
        pipe_.InitBuffer(factorBuf_, Gtsv2FactorStorageBytes(m_));
        if constexpr (SingleRhs) {
            pipe_.InitBuffer(rhsBuf_, static_cast<uint32_t>(m_) * kFloatBytes);
            if (m_ >= kGtsv2JacobiMinM && m_ <= kGtsv2JacobiMaxM) {
                pipe_.InitBuffer(jacobiBuf_, 2U * (Gtsv2AlignedRows(m_) + 16) * kFloatBytes);
            }
        } else {
            pipe_.InitBuffer(pivotBuf_, static_cast<uint32_t>(m_));
            pipe_.InitBuffer(checkBuf_, static_cast<uint32_t>(Gtsv2AlignedRows(m_)) * kFloatBytes);
        }
    }
}

template <bool SingleRhs>
__aicore__ inline void Gtsv2FactorKernel<SingleRhs>::Process()
{
    if (useUb_) {
        FactorInUb();
    } else {
        needsRefinement_ = Gtsv2NeedsRefinement(dlGm_, dGm_, duGm_, m_) || NeedsRangeInGm();
        FactorInGm();
    }
}

template <bool SingleRhs>
__aicore__ inline bool Gtsv2FactorKernel<SingleRhs>::NeedsRefinementInUb(
        LocalTensor<float> lower, LocalTensor<float> diagonal, LocalTensor<float> upper,
        LocalTensor<float> margin, LocalTensor<float> temporary)
{
    if (m_ < kGtsv2JacobiMinM) {
        return Gtsv2NeedsRefinement(lower, diagonal, upper, m_);
    }
    // The unused endpoints are ignored by the public tridiagonal ABI.
    lower.SetValue(0, 0.0f);
    upper.SetValue(m_ - 1, 0.0f);
    event_t scalarToVector = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(scalarToVector);
    WaitFlag<HardEvent::S_V>(scalarToVector);
    Abs(margin, lower, m_);
    Abs(temporary, upper, m_);
    PipeBarrier<PIPE_V>();
    Add(margin, margin, temporary, m_);
    PipeBarrier<PIPE_V>();
    Muls(margin, margin, 2.0f, m_);
    Abs(temporary, diagonal, m_);
    PipeBarrier<PIPE_V>();
    Sub(margin, temporary, margin, m_);
    PipeBarrier<PIPE_V>();
    // After Sub, temporary is dead and can hold both the reduced value
    // and scratch (two entries per 64 FP32 inputs, 32-byte aligned).
    ReduceMin(temporary, margin, temporary[8], m_, false);
    event_t vectorToScalar = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(vectorToScalar);
    WaitFlag<HardEvent::V_S>(vectorToScalar);
    return !(temporary.GetValue(0) > 0.0f);
}

template <bool SingleRhs>
__aicore__ inline bool Gtsv2FactorKernel<SingleRhs>::NeedsRangeInGm()
{
    for (int32_t row = 0; row < m_; ++row) {
        float diagonal = Gtsv2Abs(dGm_.GetValue(row));
        if (!(diagonal >= 0x1p-24f && diagonal <= 0x1p24f)) { return true; }
    }
    for (int64_t column = 0; column < n_; ++column) {
        for (int32_t row = 0; row < m_; ++row) {
            if (!(Gtsv2Abs(BGm_.GetValue(column * ldb_ + row)) <= 0x1p24f)) { return true; }
        }
    }
    return false;
}

template <bool SingleRhs>
__aicore__ inline float Gtsv2FactorKernel<SingleRhs>::ReduceMagnitude(
        LocalTensor<float> input, LocalTensor<float> temporary, LocalTensor<float> reduction, bool minimum)
{
    auto sToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
    auto vToS = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::S_V>(sToV);
    WaitFlag<HardEvent::S_V>(sToV);
    Abs(temporary, input, m_);
    PipeBarrier<PIPE_V>();
    if (minimum) {
        ReduceMin(reduction, temporary, reduction[8], m_, false);
    } else {
        ReduceMax(reduction, temporary, reduction[8], m_, false);
    }
    SetFlag<HardEvent::V_S>(vToS);
    WaitFlag<HardEvent::V_S>(vToS);
    return reduction.GetValue(0);
}

template <bool SingleRhs>
__aicore__ inline bool Gtsv2FactorKernel<SingleRhs>::NeedsRangeInUb(
        LocalTensor<float> diagonal, LocalTensor<float> temporary,
        LocalTensor<float> reduction, event_t mte2ToScalar)
{
    if (m_ < kGtsv2JacobiMinM) { return NeedsRangeInGm(); }
    if (!(ReduceMagnitude(diagonal, temporary, reduction, true) >= 0x1p-24f) ||
        !(ReduceMagnitude(diagonal, temporary, reduction, false) <= 0x1p24f)) {
        return true;
    }
    for (int64_t column = 0; column < n_; ++column) {
        // Both buffers are dead after each reduction. Wait for all vector
        // reads before reusing the reduction storage as a DMA destination.
        auto vToMte2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(vToMte2);
        WaitFlag<HardEvent::V_MTE2>(vToMte2);
        DataCopyPad(reduction, BGm_[column * ldb_],
            OneBlockCopy(static_cast<uint32_t>(m_) * kFloatBytes), NoPadding<float>());
        SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
        WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
        if (!(ReduceMagnitude(reduction, temporary, reduction, false) <= 0x1p24f)) { return true; }
    }
    return false;
}

template <bool SingleRhs>
__aicore__ inline bool Gtsv2FactorKernel<SingleRhs>::TrySingleRhsInUb(
        LocalTensor<float> multiplier, LocalTensor<float> dWork, LocalTensor<float> duWork,
        LocalTensor<float> rhs, LocalTensor<float> du2Work, event_t mte2ToScalar)
{
    const uint32_t diagonalBytes = static_cast<uint32_t>(m_) * kFloatBytes;
    if (needsRefinement_) {
        FactorInGm();
        Gtsv2RangeSolver refinement;
        refinement.Init(dlGm_, dGm_, duGm_, factorGm_, pivotGm_, BGm_,
            workspace_, m_, 1, ldb_);
        refinement.Process();
        return true;
    }
    DataCopyPad(rhs, BGm_, OneBlockCopy(diagonalBytes), NoPadding<float>());
    SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
    WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
    if (m_ >= kGtsv2JacobiMinM && m_ <= kGtsv2JacobiMaxM) {
        auto scratch = jacobiBuf_.Get<float>();
        auto sToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        auto vToS = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        if (Gtsv2TryJacobi(m_, multiplier, dWork, duWork, rhs, du2Work,
                           scratch, scratch[Gtsv2AlignedRows(m_) + 16], sToV, vToS)) {
            StoreSingleRhsInUb(rhs);
            return true;
        }
    }
    return false;
}

template <bool SingleRhs>
__aicore__ inline void Gtsv2FactorKernel<SingleRhs>::UpdateUbRhsOrPivot(
        LocalTensor<float> rhs, LocalTensor<uint8_t> pivots, int32_t row, float factor, bool swapped)
{
    if constexpr (SingleRhs) {
        if (swapped) {
            float previous = rhs.GetValue(row - 1);
            float current = rhs.GetValue(row);
            rhs.SetValue(row - 1, current);
            rhs.SetValue(row, previous - factor * current);
        } else {
            rhs.SetValue(row, rhs.GetValue(row) - factor * rhs.GetValue(row - 1));
        }
    } else {
        pivots.SetValue(row - 1, static_cast<uint8_t>(swapped));
    }
}

template <bool SingleRhs>
__aicore__ inline void Gtsv2FactorKernel<SingleRhs>::FactorRowsInUb(
        LocalTensor<float> dWork, LocalTensor<float> duWork, LocalTensor<float> du2Work,
        LocalTensor<float> multiplier, LocalTensor<uint8_t> pivots, LocalTensor<float> rhs,
        float &pivotD, float &pivotDu)
{
    float currentD = dWork.GetValue(1);
    float currentDu = duWork.GetValue(1);

    for (int32_t row = 1; row < m_; ++row) {
        float lower = multiplier.GetValue(row);
        float nextD;
        float nextDu;
        float factor;

        if (Gtsv2Abs(pivotD) >= Gtsv2Abs(lower)) {
            factor = Gtsv2PivotRatio(lower, pivotD);
            dWork.SetValue(row - 1, pivotD);
            duWork.SetValue(row - 1, pivotDu);
            du2Work.SetValue(row - 1, 0.0f);
            UpdateUbRhsOrPivot(rhs, pivots, row, factor, false);
            nextD = currentD - factor * pivotDu;
            nextDu = currentDu;
        } else {
            factor = Gtsv2PivotRatio(pivotD, lower);
            dWork.SetValue(row - 1, lower);
            duWork.SetValue(row - 1, currentD);
            du2Work.SetValue(row - 1, currentDu);
            UpdateUbRhsOrPivot(rhs, pivots, row, factor, true);
            nextD = pivotDu - factor * currentD;
            nextDu = -factor * currentDu;
        }
        if constexpr (!SingleRhs) {
            multiplier.SetValue(row - 1, factor);
        }
        pivotD = nextD;
        pivotDu = nextDu;

        if (row + 1 < m_) {
            currentD = dWork.GetValue(row + 1);
            currentDu = duWork.GetValue(row + 1);
        }
    }
}

template <bool SingleRhs>
__aicore__ inline void Gtsv2FactorKernel<SingleRhs>::StoreFactorsFromUb(
        LocalTensor<float> dWork, LocalTensor<float> duWork, LocalTensor<float> du2Work,
        LocalTensor<float> multiplier, LocalTensor<uint8_t> pivots)
{
    const uint32_t diagonalBytes = static_cast<uint32_t>(m_) * kFloatBytes;
    event_t scalarToMte3 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(scalarToMte3);
    WaitFlag<HardEvent::S_MTE3>(scalarToMte3);
    DataCopyPad(factorGm_, dWork, OneBlockCopy(diagonalBytes));
    DataCopyPad(factorGm_[m_], duWork, OneBlockCopy(diagonalBytes));
    DataCopyPad(factorGm_[2 * m_], du2Work,
                OneBlockCopy(diagonalBytes));
    DataCopyPad(factorGm_[3 * m_], multiplier,
                OneBlockCopy(diagonalBytes));
    DataCopyPad(pivotGm_, pivots,
                OneBlockCopy(static_cast<uint32_t>(m_)));
    PipeBarrier<PIPE_MTE3>();
}

template <bool SingleRhs>
__aicore__ inline bool Gtsv2FactorKernel<SingleRhs>::NeedsProtectedSolveInUb(
        LocalTensor<float> multiplier, LocalTensor<float> diagonal, LocalTensor<float> upper,
        LocalTensor<float> temporary, LocalTensor<float> rhs, event_t mte2ToScalar)
{
    LocalTensor<float> check;
    if constexpr (SingleRhs) {
        check = rhs;  // RHS has not been loaded yet; reuse its UB allocation.
    } else {
        check = checkBuf_.Get<float>();
    }
    return NeedsRefinementInUb(multiplier, diagonal, upper, temporary, check) ||
        NeedsRangeInUb(diagonal, temporary, check, mte2ToScalar);
}

template <bool SingleRhs>
__aicore__ inline void Gtsv2FactorKernel<SingleRhs>::FactorInUb()
{
    LocalTensor<float> factors = factorBuf_.Get<float>();
    LocalTensor<float> dWork = factors;
    const int32_t localStride = Gtsv2AlignedRows(m_);
    LocalTensor<float> duWork = factors[localStride];
    LocalTensor<float> du2Work = factors[2 * localStride];
    // dl is consumed from left to right and overwritten behind the cursor
    // with the elimination multiplier.
    LocalTensor<float> multiplier = factors[3 * localStride];
    LocalTensor<uint8_t> pivots;
    LocalTensor<float> rhs;
    if constexpr (SingleRhs) {
        rhs = rhsBuf_.Get<float>();
    } else {
        pivots = pivotBuf_.Get<uint8_t>();
    }

    const uint32_t diagonalBytes = static_cast<uint32_t>(m_) * kFloatBytes;
    DataCopyPad(dWork, dGm_, OneBlockCopy(diagonalBytes), NoPadding<float>());
    DataCopyPad(duWork, duGm_, OneBlockCopy(diagonalBytes), NoPadding<float>());
    DataCopyPad(multiplier, dlGm_, OneBlockCopy(diagonalBytes), NoPadding<float>());
    event_t mte2ToScalar = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
    WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);

    needsRefinement_ = NeedsProtectedSolveInUb(multiplier, dWork, duWork, du2Work, rhs, mte2ToScalar);
    if constexpr (SingleRhs) {
        if (TrySingleRhsInUb(multiplier, dWork, duWork, rhs, du2Work, mte2ToScalar)) {
            return;
        }
    }

    float pivotD = dWork.GetValue(0);
    float pivotDu = duWork.GetValue(0);
    FactorRowsInUb(dWork, duWork, du2Work, multiplier, pivots, rhs, pivotD, pivotDu);

    dWork.SetValue(m_ - 1, pivotD);
    duWork.SetValue(m_ - 1, pivotDu);
    du2Work.SetValue(m_ - 1, 0.0f);
    if constexpr (SingleRhs) {
        SolveSingleRhsInUb(dWork, duWork, du2Work, rhs);
        return;
    } else {
        multiplier.SetValue(m_ - 1, 0.0f);
        pivots.SetValue(m_ - 1, static_cast<uint8_t>(needsRefinement_));
    }

    StoreFactorsFromUb(dWork, duWork, du2Work, multiplier, pivots);
}

template <bool SingleRhs>
__aicore__ inline void Gtsv2FactorKernel<SingleRhs>::SolveSingleRhsInUb(
        LocalTensor<float> diagonal, LocalTensor<float> upper,
        LocalTensor<float> upper2, LocalTensor<float> rhs)
{
    float xNext = rhs.GetValue(m_ - 1) / diagonal.GetValue(m_ - 1);
    float xNext2 = 0.0f;
    rhs.SetValue(m_ - 1, xNext);
    for (int32_t row = m_ - 2; row >= 0; --row) {
        float numerator = rhs.GetValue(row) - upper.GetValue(row) * xNext;
        if (row + 2 < m_) {
            numerator -= upper2.GetValue(row) * xNext2;
        }
        float x = numerator / diagonal.GetValue(row);
        xNext2 = xNext;
        xNext = x;
        rhs.SetValue(row, x);
    }
    event_t scalarToMte3 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(scalarToMte3);
    WaitFlag<HardEvent::S_MTE3>(scalarToMte3);
    DataCopyPad(BGm_, rhs, OneBlockCopy(static_cast<uint32_t>(m_) * kFloatBytes));
    PipeBarrier<PIPE_MTE3>();
}

template <bool SingleRhs>
__aicore__ inline void Gtsv2FactorKernel<SingleRhs>::StoreSingleRhsInUb(LocalTensor<float> rhs)
{
    event_t ready = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(ready);
    WaitFlag<HardEvent::S_MTE3>(ready);
    DataCopyPad(BGm_, rhs, OneBlockCopy(static_cast<uint32_t>(m_) * kFloatBytes));
    PipeBarrier<PIPE_MTE3>();
}

template <bool SingleRhs>
__aicore__ inline void Gtsv2FactorKernel<SingleRhs>::FactorInGm()
{
    const int64_t dOffset = 0;
    const int64_t duOffset = m_;
    const int64_t du2Offset = 2LL * m_;
    const int64_t multiplierOffset = 3LL * m_;

    float pivotD = dGm_.GetValue(0);
    float pivotDu = duGm_.GetValue(0);
    float currentD = dGm_.GetValue(1);
    float currentDu = duGm_.GetValue(1);

    for (int32_t row = 1; row < m_; ++row) {
        float lower = dlGm_.GetValue(row);
        float nextD;
        float nextDu;
        float factor;
        uint8_t swapped;

        if (Gtsv2Abs(pivotD) >= Gtsv2Abs(lower)) {
            factor = Gtsv2PivotRatio(lower, pivotD);
            factorGm_.SetValue(dOffset + row - 1, pivotD);
            factorGm_.SetValue(duOffset + row - 1, pivotDu);
            factorGm_.SetValue(du2Offset + row - 1, 0.0f);
            nextD = currentD - factor * pivotDu;
            nextDu = currentDu;
            swapped = 0;
        } else {
            factor = Gtsv2PivotRatio(pivotD, lower);
            factorGm_.SetValue(dOffset + row - 1, lower);
            factorGm_.SetValue(duOffset + row - 1, currentD);
            factorGm_.SetValue(du2Offset + row - 1, currentDu);
            nextD = pivotDu - factor * currentD;
            nextDu = -factor * currentDu;
            swapped = 1;
        }
        factorGm_.SetValue(multiplierOffset + row - 1, factor);
        pivotGm_.SetValue(row - 1, swapped);
        pivotD = nextD;
        pivotDu = nextDu;

        if (row + 1 < m_) {
            currentD = dGm_.GetValue(row + 1);
            currentDu = duGm_.GetValue(row + 1);
        }
    }

    factorGm_.SetValue(dOffset + m_ - 1, pivotD);
    factorGm_.SetValue(duOffset + m_ - 1, pivotDu);
    factorGm_.SetValue(du2Offset + m_ - 1, 0.0f);
    factorGm_.SetValue(multiplierOffset + m_ - 1, 0.0f);
    // The unused final pivot byte carries the matrix refinement policy.
    pivotGm_.SetValue(m_ - 1, static_cast<uint8_t>(needsRefinement_));

    // The following solve kernel is ordered on the same stream and must
    // observe the scalar stores above.
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
        DcciDst::CACHELINE_OUT>(factorGm_);
}

__aicore__ inline void Gtsv2SolveKernel::Init(
        GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B, GM_ADDR workspace, const Gtsv2TilingData &tiling)
{
    m_ = tiling.m;
    n_ = tiling.n;
    ldb_ = tiling.ldb;
    blockNum_ = tiling.blockNum;
    blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    useUb_ = tiling.useUb != 0;
    workspace_ = workspace;
    dlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dl), m_);
    dGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d), m_);
    duGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(du), m_);

    BGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(B),
        static_cast<uint64_t>(ldb_) * static_cast<uint64_t>(n_));
    auto *workspaceBytes = reinterpret_cast<__gm__ uint8_t *>(workspace);
    factorGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(workspaceBytes),
        static_cast<uint64_t>(kGtsv2FactorArrays) * m_);
    pivotGm_.SetGlobalBuffer(
        workspaceBytes + Gtsv2FactorBytes(m_), static_cast<uint64_t>(m_));

    if (useUb_) {
        if (m_ > kGtsv2JacobiMaxM) {
            pipe_.InitBuffer(wideJacobiBuf_, 6U * Gtsv2AlignedRows(m_) * kFloatBytes);
            return;
        }
        pipe_.InitBuffer(factorBuf_, Gtsv2FactorStorageBytes(m_));
        pipe_.InitBuffer(pivotBuf_, static_cast<uint32_t>(m_));
        pipe_.InitBuffer(rhsBuf_, static_cast<uint32_t>(m_) * kFloatBytes);
        if (m_ >= kGtsv2JacobiMinM && m_ <= kGtsv2JacobiMaxM) {
            pipe_.InitBuffer(jacobiBuf_, 2U * (Gtsv2AlignedRows(m_) + 16) * kFloatBytes);
        }
    }
}

__aicore__ inline void Gtsv2SolveKernel::Process()
{
    if (pivotGm_.GetValue(m_ - 1) != 0) {
        if (blockIdx_ == 0) {
            Gtsv2RangeSolver refinement;
            refinement.Init(dlGm_, dGm_, duGm_, factorGm_, pivotGm_, BGm_,
                workspace_, m_, n_, ldb_);
            refinement.Process();
        }
        return;
    }
    if (useUb_) {
        if (m_ > kGtsv2JacobiMaxM) {
            SolveWideJacobiInUb();
            return;
        }
        SolveInUb();
    } else {
        SolveInGm();
    }
}

__aicore__ inline void Gtsv2SolveKernel::SolveOneRhsUb(
        LocalTensor<float> factors, LocalTensor<uint8_t> pivots,
        LocalTensor<float> rhs)
{
    LocalTensor<float> dFactor = factors;
    const int32_t localStride = Gtsv2AlignedRows(m_);
    LocalTensor<float> duFactor = factors[localStride];
    LocalTensor<float> du2Factor = factors[2 * localStride];
    LocalTensor<float> multiplier = factors[3 * localStride];

    for (int32_t row = 1; row < m_; ++row) {
        float previous = rhs.GetValue(row - 1);
        float current = rhs.GetValue(row);
        float factor = multiplier.GetValue(row - 1);
        if (pivots.GetValue(row - 1) == 0) {
            rhs.SetValue(row, current - factor * previous);
        } else {
            rhs.SetValue(row - 1, current);
            rhs.SetValue(row, previous - factor * current);
        }
    }

    float xNext = rhs.GetValue(m_ - 1) / dFactor.GetValue(m_ - 1);
    float xNext2 = 0.0f;
    rhs.SetValue(m_ - 1, xNext);
    for (int32_t row = m_ - 2; row >= 0; --row) {
        float numerator = rhs.GetValue(row) - duFactor.GetValue(row) * xNext;
        if (row + 2 < m_) {
            numerator -= du2Factor.GetValue(row) * xNext2;
        }
        float x = numerator / dFactor.GetValue(row);
        xNext2 = xNext;
        xNext = x;
        rhs.SetValue(row, x);
    }
}

__aicore__ inline bool Gtsv2SolveKernel::TryJacobiForColumn(
        LocalTensor<float> factors, LocalTensor<float> rhs, event_t rhsReady,
        event_t jacobiSToV, event_t jacobiVToS)
{
    const int32_t localStride = Gtsv2AlignedRows(m_);
    const uint32_t rhsBytes = static_cast<uint32_t>(m_) * kFloatBytes;
    bool solved = false;
    if (m_ >= kGtsv2JacobiMinM && m_ <= kGtsv2JacobiMaxM) {
        DataCopyPad(factors, dGm_, OneBlockCopy(rhsBytes), NoPadding<float>());
        DataCopyPad(factors[localStride], duGm_, OneBlockCopy(rhsBytes), NoPadding<float>());
        DataCopyPad(factors[3 * localStride], dlGm_, OneBlockCopy(rhsBytes), NoPadding<float>());
        SetFlag<HardEvent::MTE2_S>(rhsReady);
        WaitFlag<HardEvent::MTE2_S>(rhsReady);
        factors[3 * localStride].SetValue(0, 0.0f);
        factors[localStride].SetValue(m_ - 1, 0.0f);
        auto scratch = jacobiBuf_.Get<float>();
        solved = Gtsv2TryJacobi(m_, factors[3 * localStride], factors,
            factors[localStride], rhs, factors[2 * localStride],
            scratch, scratch[localStride + 16], jacobiSToV, jacobiVToS);
        if (!solved) {
            for (int32_t array = 0; array < kGtsv2FactorArrays; ++array) {
                DataCopyPad(factors[array * localStride], factorGm_[array * m_],
                    OneBlockCopy(rhsBytes), NoPadding<float>());
            }
            SetFlag<HardEvent::MTE2_S>(rhsReady);
            WaitFlag<HardEvent::MTE2_S>(rhsReady);
        }
    }
    return solved;
}

__aicore__ inline void Gtsv2SolveKernel::SolveInUb()
{
    LocalTensor<float> factors = factorBuf_.Get<float>();
    LocalTensor<uint8_t> pivots = pivotBuf_.Get<uint8_t>();
    LocalTensor<float> rhs = rhsBuf_.Get<float>();

    const int32_t localStride = Gtsv2AlignedRows(m_);
    const uint32_t factorArrayBytes =
        static_cast<uint32_t>(m_) * kFloatBytes;
    DataCopyPad(factors, factorGm_, OneBlockCopy(factorArrayBytes),
                NoPadding<float>());
    DataCopyPad(factors[localStride], factorGm_[m_],
                OneBlockCopy(factorArrayBytes), NoPadding<float>());
    DataCopyPad(factors[2 * localStride], factorGm_[2 * m_],
                OneBlockCopy(factorArrayBytes), NoPadding<float>());
    DataCopyPad(factors[3 * localStride], factorGm_[3 * m_],
                OneBlockCopy(factorArrayBytes), NoPadding<float>());
    DataCopyPad(pivots, pivotGm_,
                OneBlockCopy(static_cast<uint32_t>(m_)),
                NoPadding<uint8_t>());
    event_t mte2ToScalar = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
    WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);

    const uint32_t rhsBytes = static_cast<uint32_t>(m_) * kFloatBytes;
    event_t rhsReady = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    event_t rhsComputed = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    event_t rhsStored = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    // Allocate once: unaligned ldb can route many RHS through one core.
    auto jacobiSToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
    auto jacobiVToS = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
    for (int64_t rhsIndex = blockIdx_; rhsIndex < n_; rhsIndex += blockNum_) {
        int64_t columnOffset = static_cast<int64_t>(rhsIndex) * ldb_;
        DataCopyPad(rhs, BGm_[columnOffset], OneBlockCopy(rhsBytes),
                    NoPadding<float>());
        SetFlag<HardEvent::MTE2_S>(rhsReady);
        WaitFlag<HardEvent::MTE2_S>(rhsReady);

        bool solved = TryJacobiForColumn(factors, rhs, rhsReady, jacobiSToV, jacobiVToS);
        if (!solved) {
            SolveOneRhsUb(factors, pivots, rhs);
        }

        SetFlag<HardEvent::S_MTE3>(rhsComputed);
        WaitFlag<HardEvent::S_MTE3>(rhsComputed);
        DataCopyPad(BGm_[columnOffset], rhs, OneBlockCopy(rhsBytes));
        // The next iteration reuses rhs for GM->UB.  Synchronize the
        // MTE3 reader with the following MTE2 writer explicitly; a
        // PIPE_MTE3 barrier only orders instructions within MTE3.
        SetFlag<HardEvent::MTE3_MTE2>(rhsStored);
        WaitFlag<HardEvent::MTE3_MTE2>(rhsStored);
    }
}

__aicore__ inline void Gtsv2SolveKernel::SolveWideJacobiInUb()
{
    auto data = wideJacobiBuf_.Get<float>();
    const int32_t stride = Gtsv2AlignedRows(m_);
    auto sToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
    auto vToS = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
    auto mte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
    auto vToMte2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
    auto vToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
    auto mte3ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_V));
    auto mte2ToS = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
    auto sToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
    auto mte3ToMte2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
    const uint32_t bytes = static_cast<uint32_t>(m_) * kFloatBytes;
    for (int64_t column = blockIdx_; column < n_; column += blockNum_) {
        const int64_t offset = column * ldb_;
        if (Gtsv2TryWideJacobi(m_, data, dlGm_, dGm_, duGm_, BGm_, offset,
                              sToV, vToS, mte2ToV, vToMte2, vToMte3, mte3ToV)) {
            // The helper drains MTE3 for V, but the next column begins
            // with MTE2 writes into the same X/RHS storage. Order that
            // producer explicitly before reusing any UB array.
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            continue;
        }
        // Preflight rejected the input before writing B. Restore LU
        // factors and use the existing direct solve for this RHS only.
        SetFlag<HardEvent::V_MTE2>(vToMte2);
        WaitFlag<HardEvent::V_MTE2>(vToMte2);
        auto pivots = wideJacobiBuf_.Get<uint8_t>()[5 * stride * kFloatBytes];
        auto rhs = data[4 * stride];
        for (int32_t array = 0; array < kGtsv2FactorArrays; ++array) {
            DataCopyPad(data[array * stride], factorGm_[array * m_],
                OneBlockCopy(bytes), NoPadding<float>());
        }
        DataCopyPad(pivots, pivotGm_, OneBlockCopy(m_), NoPadding<uint8_t>());
        DataCopyPad(rhs, BGm_[offset], OneBlockCopy(bytes), NoPadding<float>());
        SetFlag<HardEvent::MTE2_S>(mte2ToS);
        WaitFlag<HardEvent::MTE2_S>(mte2ToS);
        SolveOneRhsUb(data, pivots, rhs);
        SetFlag<HardEvent::S_MTE3>(sToMte3);
        WaitFlag<HardEvent::S_MTE3>(sToMte3);
        DataCopyPad(BGm_[offset], rhs, OneBlockCopy(bytes));
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
    }
}

__aicore__ inline void Gtsv2SolveKernel::SolveInGm()
{
    const int64_t dOffset = 0;
    const int64_t duOffset = m_;
    const int64_t du2Offset = 2LL * m_;
    const int64_t multiplierOffset = 3LL * m_;

    for (int64_t rhsIndex = blockIdx_; rhsIndex < n_; rhsIndex += blockNum_) {
        int64_t columnOffset = static_cast<int64_t>(rhsIndex) * ldb_;
        for (int32_t row = 1; row < m_; ++row) {
            float previous = BGm_.GetValue(columnOffset + row - 1);
            float current = BGm_.GetValue(columnOffset + row);
            float factor = factorGm_.GetValue(multiplierOffset + row - 1);
            if (pivotGm_.GetValue(row - 1) == 0) {
                BGm_.SetValue(columnOffset + row,
                              current - factor * previous);
            } else {
                BGm_.SetValue(columnOffset + row - 1, current);
                BGm_.SetValue(columnOffset + row,
                              previous - factor * current);
            }
        }

        float xNext = BGm_.GetValue(columnOffset + m_ - 1) /
                      factorGm_.GetValue(dOffset + m_ - 1);
        float xNext2 = 0.0f;
        BGm_.SetValue(columnOffset + m_ - 1, xNext);
        for (int32_t row = m_ - 2; row >= 0; --row) {
            float numerator = BGm_.GetValue(columnOffset + row) -
                factorGm_.GetValue(duOffset + row) * xNext;
            if (row + 2 < m_) {
                numerator -= factorGm_.GetValue(du2Offset + row) * xNext2;
            }
            float x = numerator / factorGm_.GetValue(dOffset + row);
            xNext2 = xNext;
            xNext = x;
            BGm_.SetValue(columnOffset + row, x);
        }
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
        DcciDst::CACHELINE_OUT>(BGm_);
}

}  // namespace

extern "C" __global__ __aicore__ void gtsv2_factor_kernel(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B, GM_ADDR workspace,
    const Gtsv2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Gtsv2FactorKernel<> kernel;
    kernel.Init(dl, d, du, workspace, B, tiling);
    kernel.Process();
    ResetMask();
}

extern "C" __global__ __aicore__ void gtsv2_solve_kernel(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B, GM_ADDR workspace, const Gtsv2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Gtsv2SolveKernel kernel;
    kernel.Init(dl, d, du, B, workspace, tiling);
    kernel.Process();
    ResetMask();
}

extern "C" __global__ __aicore__ void gtsv2_single_rhs_kernel(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B, GM_ADDR workspace,
    const Gtsv2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Gtsv2FactorKernel<true> kernel;
    kernel.Init(dl, d, du, workspace, B, tiling);
    kernel.Process();
    ResetMask();
}

void gtsv2_factor_kernel_do(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B, GM_ADDR workspace,
    const Gtsv2TilingData &tiling, void *stream)
{
    gtsv2_factor_kernel<<<1, nullptr, stream>>>(dl, d, du, B, workspace, tiling);
}

void gtsv2_solve_kernel_do(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B, GM_ADDR workspace, const Gtsv2TilingData &tiling,
    uint32_t blockNum, void *stream)
{
    gtsv2_solve_kernel<<<blockNum, nullptr, stream>>>(dl, d, du, B, workspace, tiling);
}

void gtsv2_single_rhs_kernel_do(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B, GM_ADDR workspace,
    const Gtsv2TilingData &tiling, void *stream)
{
    gtsv2_single_rhs_kernel<<<1, nullptr, stream>>>(dl, d, du, B, workspace, tiling);
}
