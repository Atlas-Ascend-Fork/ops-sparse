/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_SPSV_EXPANSION64_H_
#define SPSV_ARCH22_SPSV_EXPANSION64_H_
#include "spsv_binary64.h"
#include "spsv_tiling_data.h"

namespace spsv_expansion64 {
namespace base = spsv_extended64;
struct Real {
    base::Real high;
    base::Real low;
};
struct Complex {
    Real real;
    Real imag;
};
SPSV_EXTENDED64_FN Real FromFloat(float value) { return {base::FromFloat(value), base::FromFloat(0)}; }
SPSV_EXTENDED64_FN bool HasTail(Real value) { return !spsv_soft64::IsZero(value.low.fraction); }
SPSV_EXTENDED64_FN __attribute__((noinline)) Real Normalize(base::Real high, base::Real low)
{
    if (!spsv_soft64::IsFinite(high.fraction))
        return {high, base::FromFloat(0)};
    const auto sum = base::Add(high, low);
    if (!spsv_soft64::IsFinite(sum.fraction))
        return {sum, base::FromFloat(0)};
    const auto virtualLow = base::Sub(sum, high);
    const auto error = base::Add(base::Sub(high, base::Sub(sum, virtualLow)), base::Sub(low, virtualLow));
    return {sum, error};
}
SPSV_EXTENDED64_FN void Insert(base::Real* terms, int& count, base::Real value)
{
    int retained = 0;
    for (int i = 0; i < count; ++i) {
        const auto pair = Normalize(value, terms[i]);
        if (!spsv_soft64::IsZero(pair.low.fraction))
            terms[retained++] = pair.low;
        value = pair.high;
    }
    terms[retained++] = value;
    count = retained;
}
SPSV_EXTENDED64_FN __attribute__((noinline)) Real Add(Real a, Real b)
{
    if (!spsv_soft64::IsFinite(a.high.fraction) || (spsv_soft64::IsZero(b.high.fraction) && !HasTail(b)))
        return a;
    if (!spsv_soft64::IsFinite(b.high.fraction))
        return b;
    base::Real terms[4]{};
    int count = 0;
    Insert(terms, count, a.low);
    Insert(terms, count, a.high);
    Insert(terms, count, b.low);
    Insert(terms, count, b.high);
    if (count == 1)
        return {terms[0], base::FromFloat(0)};
    auto low = terms[count - 2];
    for (int i = count - 3; i >= 0; --i)
        low = base::Add(low, terms[i]);
    return Normalize(terms[count - 1], low);
}
SPSV_EXTENDED64_FN __attribute__((noinline)) Real Div(Real value, Real divisor)
{
    // Keep binary64 division rounding for the leading component. Propagate
    // the independent tail instead of rounding it into that component.
    const auto high = base::Div(value.high, divisor.high);
    if (!spsv_soft64::IsFinite(high.fraction) || (!HasTail(value) && !HasTail(divisor)))
        return {high, base::FromFloat(0)};
    const auto remainder = base::Sub(value.low, base::Mul(high, divisor.low));
    return Normalize(high, base::Div(remainder, divisor.high));
}
SPSV_EXTENDED64_FN base::Complex High(Complex value) { return {value.real.high, value.imag.high}; }
SPSV_EXTENDED64_FN base::Complex Low(Complex value) { return {value.real.low, value.imag.low}; }
SPSV_EXTENDED64_FN __attribute__((noinline)) Complex Product(base::Complex coefficient, Complex value)
{
    // Preserve the established binary64 complex-product rounding, while
    // carrying previously retained tails through the same linear operation.
    const auto high = base::Product(coefficient, High(value));
    if (!HasTail(value.real) && !HasTail(value.imag))
        return {{high.real, base::FromFloat(0)}, {high.imag, base::FromFloat(0)}};
    const auto low = base::Product(coefficient, Low(value));
    return {Normalize(high.real, low.real), Normalize(high.imag, low.imag)};
}
SPSV_EXTENDED64_FN __attribute__((noinline)) Complex Divide(Complex value, Complex divisor)
{
    const auto high = base::Divide(High(value), High(divisor));
    if (!spsv_soft64::IsFinite(high.real.fraction) || !spsv_soft64::IsFinite(high.imag.fraction))
        return {{high.real, base::FromFloat(0)}, {high.imag, base::FromFloat(0)}};
    if (!HasTail(value.real) && !HasTail(value.imag) && !HasTail(divisor.real) && !HasTail(divisor.imag))
        return {{high.real, base::FromFloat(0)}, {high.imag, base::FromFloat(0)}};
    const auto product = base::Product(high, Low(divisor));
    const base::Complex remainder{base::Sub(value.real.low, product.real), base::Sub(value.imag.low, product.imag)};
    const auto low = base::Divide(remainder, High(divisor));
    return {Normalize(high.real, low.real), Normalize(high.imag, low.imag)};
}
SPSV_EXTENDED64_FN uint32_t ToFloatBits(Real value)
{
    auto high = value.high;
    if (HasTail(value) && spsv_soft64::IsFinite(high.fraction) && !spsv_soft64::IsZero(high.fraction)) {
        // A retained tail can decide an FP32 tie even when it cannot change
        // the binary64 leading component. Nudge only exact FP32 midpoints.
        const int shift = high.exponent >= -126 ? 29 : -97 - high.exponent;
        const uint64_t significand = spsv_soft64::Unpack(high.fraction).significand;
        if (shift > 0 && shift <= 53 && (significand & ((uint64_t{1} << shift) - 1)) == (uint64_t{1} << (shift - 1))) {
            const bool sameSign = ((high.fraction.bits ^ value.low.fraction.bits) & spsv_soft64::kSign) == 0;
            high = base::Normalize({sameSign ? high.fraction.bits + 1 : high.fraction.bits - 1}, high.exponent);
        }
    }
    return base::ToFloatBits(high);
}
} // namespace spsv_expansion64

namespace spsv_series64 {
namespace base = spsv_extended64;
struct Real {
    // Only the initialized prefix [0, count) is ever read or copied.
    base::Real parts[spsv_a2_ub::kWideCompactTerms];
    int count = 0;
    SPSV_EXTENDED64_FN Real() {}
    SPSV_EXTENDED64_FN Real(const Real& source) : count(source.count)
    {
        for (int i = 0; i < count; ++i)
            parts[i] = source.parts[i];
    }
    SPSV_EXTENDED64_FN Real& operator=(const Real& source)
    {
        count = source.count;
        for (int i = 0; i < count; ++i)
            parts[i] = source.parts[i];
        return *this;
    }
};
struct Complex {
    Real real;
    Real imag;
};
SPSV_EXTENDED64_FN base::Real At(const Real& value, int i)
{
    return i < value.count ? value.parts[i] : base::FromFloat(0);
}
SPSV_EXTENDED64_FN Real FromBase(base::Real value)
{
    Real result;
    result.parts[0] = value;
    result.count = spsv_soft64::IsZero(value.fraction) ? 0 : 1;
    return result;
}
SPSV_EXTENDED64_FN Complex FromBase(base::Complex value) { return {FromBase(value.real), FromBase(value.imag)}; }
SPSV_EXTENDED64_FN base::Complex Head(const Complex& value) { return {At(value.real, 0), At(value.imag, 0)}; }
SPSV_EXTENDED64_FN void Append(Real& value, base::Real part)
{
    if (!spsv_soft64::IsZero(part.fraction) && value.count < static_cast<int>(spsv_a2_ub::kWideCompactTerms))
        value.parts[value.count++] = part;
}
SPSV_EXTENDED64_FN Real WithoutHead(const Real& value)
{
    Real result;
    for (int i = 1; i < value.count; ++i)
        Append(result, value.parts[i]);
    return result;
}
SPSV_EXTENDED64_FN Complex WithoutHead(const Complex& value)
{
    return {WithoutHead(value.real), WithoutHead(value.imag)};
}
SPSV_EXTENDED64_FN bool StablePair(base::Real high, base::Real low)
{
    // Certify RNE leaves high unchanged, including the shorter spacing on
    // the side of a power of two facing zero and even-significand ties.
    if (spsv_soft64::IsZero(high.fraction) || spsv_soft64::IsZero(low.fraction) ||
        !spsv_soft64::IsFinite(high.fraction) || !spsv_soft64::IsFinite(low.fraction))
        return false;
    const bool opposite = ((high.fraction.bits ^ low.fraction.bits) & spsv_soft64::kSign) != 0;
    const bool power = (high.fraction.bits & spsv_soft64::kFraction) == 0;
    const int64_t halfSpacing = static_cast<int64_t>(high.exponent) - 53 - (opposite && power ? 1 : 0);
    return low.exponent < halfSpacing ||
           (low.exponent == halfSpacing && (low.fraction.bits & spsv_soft64::kFraction) == 0 &&
            (high.fraction.bits & 1) == 0);
}
SPSV_EXTENDED64_FN int StablePrefix(const Real& value)
{
    int count = value.count == 0 ? 0 : 1;
    while (count < value.count && StablePair(value.parts[count - 1], value.parts[count]))
        ++count;
    return count;
}
SPSV_EXTENDED64_FN __attribute__((noinline)) void AddTerm(Real& value, base::Real term, int capacity)
{
    if (!spsv_soft64::IsFinite(At(value, 0).fraction) || !spsv_soft64::IsFinite(term.fraction)) {
        value = FromBase(base::Add(At(value, 0), term));
        return;
    }
    const int stablePrefix = StablePrefix(value);
    if (stablePrefix == value.count && (value.count == 0 || StablePair(value.parts[value.count - 1], term))) {
        if (value.count < capacity)
            Append(value, term);
        return;
    }
    base::Real ascending[spsv_a2_ub::kWideCompactTerms + 1];
    int count = 0;
    for (int i = value.count - 1; i >= 0; --i) {
        const auto pair = spsv_expansion64::Normalize(term, value.parts[i]);
        const bool unchanged = pair.high.fraction.bits == value.parts[i].fraction.bits &&
                               pair.high.exponent == value.parts[i].exponent &&
                               pair.low.fraction.bits == term.fraction.bits && pair.low.exponent == term.exponent;
        if (!spsv_soft64::IsZero(pair.low.fraction))
            ascending[count++] = pair.low;
        term = pair.high;
        if (unchanged && i < stablePrefix) {
            // Every remaining adjacent pair has the same certificate, so
            // the ordinary insertion would reproduce this prefix bitwise.
            for (int j = i; j >= 0; --j)
                ascending[count++] = value.parts[j];
            term = base::FromFloat(0);
            break;
        }
    }
    if (!spsv_soft64::IsZero(term.fraction))
        ascending[count++] = term;
    if (count > capacity) {
        ascending[1] = base::Add(ascending[1], ascending[0]);
        for (int i = 1; i < count; ++i)
            ascending[i - 1] = ascending[i];
        --count;
    }
    value.count = 0;
    for (int i = count - 1; i >= 0; --i)
        Append(value, ascending[i]);
}
SPSV_EXTENDED64_FN void SubtractProduct(Complex& value, base::Complex coefficient, const Complex& source, int capacity)
{
    const int count = source.real.count > source.imag.count ? source.real.count : source.imag.count;
    for (int i = 0; i < count; ++i) {
        const auto product = base::Product(coefficient, {At(source.real, i), At(source.imag, i)});
        AddTerm(value.real, base::Negate(product.real), capacity);
        AddTerm(value.imag, base::Negate(product.imag), capacity);
    }
}
SPSV_EXTENDED64_FN __attribute__((noinline)) Complex Divide(const Complex& source, const Complex& divisor, int capacity,
                                                            bool complex)
{
    const auto leading = Head(divisor);
    const auto high = complex ? base::Divide(Head(source), leading) :
                                base::Complex{base::Div(At(source.real, 0), leading.real), base::FromFloat(0)};
    auto result = FromBase(high);
    if (!spsv_soft64::IsFinite(high.real.fraction) || !spsv_soft64::IsFinite(high.imag.fraction))
        return result;
    if (divisor.real.count <= 1 && divisor.imag.count <= 1) {
        const int count = source.real.count > source.imag.count ? source.real.count : source.imag.count;
        for (int i = 1; i < capacity && i < count; ++i) {
            const base::Complex part{At(source.real, i), At(source.imag, i)};
            const auto next = complex ? base::Divide(part, leading) :
                                        base::Complex{base::Div(part.real, leading.real), base::FromFloat(0)};
            AddTerm(result.real, next.real, capacity);
            AddTerm(result.imag, next.imag, capacity);
        }
        return result;
    }
    const auto tail = WithoutHead(divisor);
    auto value = WithoutHead(source);
    SubtractProduct(value, high, tail, capacity);
    for (int i = 1; i < capacity && (value.real.count != 0 || value.imag.count != 0); ++i) {
        const auto next = complex ? base::Divide(Head(value), leading) :
                                    base::Complex{base::Div(At(value.real, 0), leading.real), base::FromFloat(0)};
        AddTerm(result.real, next.real, capacity);
        AddTerm(result.imag, next.imag, capacity);
        value = WithoutHead(value);
        SubtractProduct(value, next, tail, capacity);
    }
    return result;
}
SPSV_EXTENDED64_FN uint32_t ToFloatBits(const Real& value)
{
    return spsv_expansion64::ToFloatBits({At(value, 0), At(value, 1)});
}
} // namespace spsv_series64

#endif // SPSV_ARCH22_SPSV_EXPANSION64_H_
