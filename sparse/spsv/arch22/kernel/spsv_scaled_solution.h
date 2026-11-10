/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_SCALED_SOLUTION_H_
#define SPSV_SCALED_SOLUTION_H_
#include "spsv_kernel_memory.h"
#include "spsv_numeric.h"
#include "spsv_wide_solution.h"
#include "spsv_unit_vector_math.h"

namespace spsv_a2 {
using namespace AscendC;

template <int32_t Components, uint32_t YCapacity>
__aicore__ __attribute__((always_inline)) inline spsv_scaled::Complex ExactRow(
    int64_t row, Complex alpha, Complex x, bool unit, UbCache<int32_t, spsv_a2_ub::kRowCapacity>& offsets,
    UbCache<int32_t, spsv_a2_ub::kColumnCapacity>& columns, UbCache<float, spsv_a2_ub::kValueCapacity>& values,
    UbCache<float, YCapacity>& y, LocalTensor<float>* low = nullptr)
{
    spsv_scaled::ExactResidualSum<Components> residual;
    residual.AddProduct(alpha.real, alpha.imag, ToPair(x), false);
    spsv_scaled::ExactDiagonalSum<Components> diagonal;
    for (int64_t p = offsets.Read(row); p < offsets.Read(row + 1); ++p) {
        int32_t col = columns.Read(p);
        Complex coefficient{values.Read(p * Components),
            Components == 2 ? values.Read(p * Components + 1) : 0};
        if (col == row) {
            if (!unit) diagonal.Add(coefficient.real, coefficient.imag);
            continue;
        }
        int64_t position = static_cast<int64_t>(col) * Components;
        spsv_dd::Complex previous{{y.Read(position), low == nullptr ? 0 : low->GetValue(position)},
            {Components == 2 ? y.Read(position + 1) : 0,
                Components == 2 && low != nullptr ? low->GetValue(position + 1) : 0}};
        residual.AddProduct(coefficient.real, coefficient.imag, previous, true);
    }
    auto rhs = residual.Value();
    if (!unit) {
        auto divisor = diagonal.Value();
        if constexpr (Components == 1) rhs.real = spsv_scaled::Div(rhs.real, divisor.real);
        else rhs = spsv_scaled::Div(rhs, divisor);
    }
    return rhs;
}

__aicore__ inline bool NeedsScaledDependency(spsv_scaled::Real value)
{
    if (value.fraction.hi == 0 || !spsv_dd::Finite(value.fraction.hi))
        return false;
    // Keep the top binade too: final FP32 rounding can overflow it even
    // though the internal significand/exponent still represents a finite value.
    if (value.exponent < -126 || value.exponent >= 127)
        return true;
    return value.fraction.lo != 0 && value.exponent + spsv_scaled::FromFloat(value.fraction.lo).exponent < -126;
}
__aicore__ inline bool NeedsScaledDependency(spsv_scaled::Complex value)
{
    return NeedsScaledDependency(value.real) || NeedsScaledDependency(value.imag);
}

template <int32_t Components, uint32_t YCapacity>
using WideSolution = spsv_wide::Solution<
    Components, UbCache<int32_t, spsv_a2_ub::kRowCapacity>, UbCache<int32_t, spsv_a2_ub::kColumnCapacity>,
    UbCache<float, spsv_a2_ub::kValueCapacity>, UbCache<float, spsv_a2_ub::kInputCapacity>, UbCache<float, YCapacity>>;

// Once a row requires scaled replay, retain both significand words and the
// exponent for all remaining dependencies. Public rounding happens only after
// the triangular solve, including for aliased X/Y. Reuse the existing caches.
template <int32_t Components, uint32_t YCapacity>
class ScaledSolution {
public:
    __aicore__ inline ScaledSolution(const SpsvA2Tiling& tiling, UbCache<int32_t, spsv_a2_ub::kRowCapacity>& row,
                                     UbCache<int32_t, spsv_a2_ub::kColumnCapacity>& col,
                                     UbCache<float, spsv_a2_ub::kValueCapacity>& value,
                                     UbCache<float, spsv_a2_ub::kInputCapacity>& x, UbCache<float, YCapacity>& y)
        : t_(tiling), row_(row), col_(col), value_(value), x_(x), y_(y)
    {}

    __aicore__ __attribute__((noinline)) void Continue(int64_t step, Complex alpha, GM_ADDR input, GM_ADDR output,
                                                       GM_ADDR workspace, LocalTensor<float>* low = nullptr)
    {
        Preserve(step, input, output, workspace, low);
        for (; step < t_.m; ++step) {
            int64_t row = Row(step);
            int64_t offset = row * Components;
            Complex source{x_.Read(offset), Components == 2 ? x_.Read(offset + 1) : 0};
            bool replay = false;
            auto result = Compute(row, alpha, source, replay);
            if (replay) {
                WideSolution<Components, YCapacity> solution(t_, row_, col_, value_, x_, y_);
                solution.Run(alpha, input, output, workspace);
                return;
            }
            Write(offset, result.real);
            if constexpr (Components == 2)
                Write(offset + 1, result.imag);
        }
        x_.Rebind(output, t_.m * Components);
        for (int64_t offset = 0; offset < t_.m * Components; ++offset)
            x_.Write(offset, spsv_scaled::Round(Read(offset)));
        x_.Flush();
        y_.Flush();
    }

private:
    __aicore__ inline int64_t Row(int64_t step) const
    {
        bool forward = t_.transpose ? !t_.lower : t_.lower;
        return forward ? step : t_.m - 1 - step;
    }
    __aicore__ inline spsv_scaled::Real Read(int64_t offset)
    {
        return {{y_.Read(offset * 3), y_.Read(offset * 3 + 1)},
                static_cast<int32_t>(spsv_scaled::Bits(y_.Read(offset * 3 + 2)))};
    }
    __aicore__ inline spsv_scaled::Complex Previous(int64_t col)
    {
        if constexpr (Components == 1)
            return {Read(col), spsv_scaled::FromFloat(0)};
        else
            return {Read(col * 2), Read(col * 2 + 1)};
    }
    __aicore__ inline void Write(int64_t offset, spsv_scaled::Real value)
    {
        if (value.exponent > spsv_scaled::kExponentLimit || value.exponent < -spsv_scaled::kExponentLimit)
            value = spsv_scaled::Normalize(value.fraction, value.exponent);
        y_.Write(offset * 3, value.fraction.hi);
        y_.Write(offset * 3 + 1, value.fraction.lo);
        y_.Write(offset * 3 + 2, spsv_scaled::FromBits(static_cast<uint32_t>(value.exponent)));
    }
    __aicore__ inline void Preserve(int64_t steps, GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                    LocalTensor<float>* low)
    {
        y_.Rebind(workspace + t_.solutionOffset, t_.m * Components * 3);
        x_.Rebind(output, t_.m * Components);
        for (int64_t step = 0; step < steps; ++step) {
            for (int part = 0; part < Components; ++part) {
                int64_t offset = Row(step) * Components + part;
                const spsv_dd::Real value{x_.Read(offset), low == nullptr ? 0 : low->GetValue(offset)};
                Write(offset, spsv_scaled::FromPair(value));
            }
        }
        x_.Rebind(input, t_.m * Components);
    }
    struct ExponentSpan {
        int first = 0;
        int last = 0;
        bool seen = false;
    };
    __aicore__ inline void TrackPart(float a, float b, int scale, ExponentSpan& span)
    {
        uint32_t left = spsv_scaled::Bits(a) & 0x7fffffffu;
        uint32_t right = spsv_scaled::Bits(b) & 0x7fffffffu;
        if (left == 0 || right == 0 || left >= 0x7f800000u || right >= 0x7f800000u)
            return;
        int exponent = spsv_scaled::FromFloat(a).exponent + spsv_scaled::FromFloat(b).exponent + scale;
        if (!span.seen || exponent - 46 < span.first)
            span.first = exponent - 46;
        if (!span.seen || exponent + 1 > span.last)
            span.last = exponent + 1;
        span.seen = true;
    }
    __aicore__ inline void Track(float a, spsv_scaled::Real b, ExponentSpan& span)
    {
        TrackPart(a, b.fraction.hi, b.exponent, span);
        TrackPart(a, b.fraction.lo, b.exponent, span);
    }
    __aicore__ inline void Track(Complex a, spsv_scaled::Complex b, ExponentSpan& span)
    {
        Track(a.real, b.real, span);
        if constexpr (Components == 2) {
            Track(a.imag, b.imag, span);
            Track(a.real, b.imag, span);
            Track(a.imag, b.real, span);
        }
    }
    __aicore__ inline Complex Coefficient(int64_t position)
    {
        return {value_.Read(position * Components), Components == 2 ? value_.Read(position * Components + 1) : 0};
    }
    struct FastResidual {
        spsv_scaled::Complex value{};
        spsv_scaled::Complex magnitude{};
    };
    struct ResidualInfo {
        spsv_scaled::Complex magnitude{};
        int termBits = 0;
        bool valid = false;
    };
    __aicore__ inline spsv_dd::Real AlignFast(spsv_dd::Real value, int64_t shift)
    {
        if ((value.hi == 0 && value.lo == 0) || shift == 0)
            return value;
        // The omitted tail is below the pair-roundoff allowance in FastErrorFits.
        // Keep the scaling factor normal; cancellation still takes the exact path.
        if (shift < -100)
            return {0, 0};
        const float factor = spsv_scaled::FromBits(static_cast<uint32_t>(shift + 127) << 23);
        return {value.hi * factor, value.lo * factor};
    }
    __aicore__ inline bool AddFastTerm(spsv_scaled::Real a, spsv_scaled::Real b, bool negative, spsv_scaled::Real& sum,
                                       spsv_scaled::Real& magnitude)
    {
        if (a.fraction.hi == 0 || b.fraction.hi == 0)
            return true;
        const int64_t exponent = static_cast<int64_t>(a.exponent) + b.exponent;
        if (exponent > spsv_scaled::kExponentLimit || exponent < -spsv_scaled::kExponentLimit)
            return false;
        auto term = spsv_dd::Mul(a.fraction, b.fraction);
        if (negative)
            term = spsv_dd::Negate(term);
        const int common = magnitude.fraction.hi == 0 ?
                               static_cast<int>(exponent) :
                               (magnitude.exponent > exponent ? magnitude.exponent : static_cast<int>(exponent));
        const auto previous = AlignFast(sum.fraction, static_cast<int64_t>(sum.exponent) - common);
        const auto positive = AlignFast(magnitude.fraction, static_cast<int64_t>(magnitude.exponent) - common);
        term = AlignFast(term, exponent - common);
        sum = {spsv_dd::Add(previous, term), common};
        magnitude = {{positive.hi + Abs(term.hi) + Abs(term.lo), 0}, common};
        return true;
    }
    __aicore__ inline bool FiniteScaled(spsv_scaled::Complex value)
    {
        return Finite(value.real.fraction.hi) && Finite(value.imag.fraction.hi);
    }
    __aicore__ inline bool AddFastProduct(Complex a, spsv_scaled::Complex b, bool negative, FastResidual& sum)
    {
        if (!Finite(a.real) || !Finite(a.imag) || !FiniteScaled(b))
            return false;
        const auto real = spsv_scaled::FromFloat(a.real);
        if (!AddFastTerm(real, b.real, negative, sum.value.real, sum.magnitude.real))
            return false;
        if constexpr (Components == 2) {
            const auto imag = spsv_scaled::FromFloat(a.imag);
            if (!AddFastTerm(imag, b.imag, !negative, sum.value.real, sum.magnitude.real) ||
                !AddFastTerm(real, b.imag, negative, sum.value.imag, sum.magnitude.imag) ||
                !AddFastTerm(imag, b.real, negative, sum.value.imag, sum.magnitude.imag))
                return false;
        }
        return FiniteScaled(sum.value) && FiniteScaled(sum.magnitude);
    }
    __aicore__ inline int64_t FastBudgetExponent(spsv_scaled::Real value)
    {
        // Four public FP32 ULPs, including subnormals. Cancellation that
        // cannot meet this local error budget retains exact replay.
        const int64_t exponent = value.fraction.hi == 0 ? -147 : static_cast<int64_t>(value.exponent) - 21;
        return exponent < -147 ? -147 : exponent;
    }
    __aicore__ inline int MaxScaledExponent(spsv_scaled::Complex value)
    {
        if (value.real.fraction.hi == 0)
            return value.imag.exponent;
        if (value.imag.fraction.hi == 0)
            return value.real.exponent;
        return value.real.exponent > value.imag.exponent ? value.real.exponent : value.imag.exponent;
    }
    __aicore__ inline bool FastErrorFits(const FastResidual& sum, spsv_scaled::Complex divisor,
                                         spsv_scaled::Complex result, int termBits)
    {
        if (!FiniteScaled(divisor) || !FiniteScaled(result))
            return false;
        const bool dr = divisor.real.fraction.hi != 0;
        const bool di = Components == 2 && divisor.imag.fraction.hi != 0;
        if (!dr && !di)
            return false;
        const bool mr = sum.magnitude.real.fraction.hi != 0;
        const bool mi = Components == 2 && sum.magnitude.imag.fraction.hi != 0;
        if (!mr && !mi)
            return true;
        const int magnitude = MaxScaledExponent(sum.magnitude);
        const int diagonal = MaxScaledExponent(divisor);
        // Sixteen pair roundoffs (2^-47) per term, plus five exponent bits
        // for alignment tails, positive magnitude sums and complex division.
        const int64_t error = static_cast<int64_t>(magnitude) - diagonal + termBits - 38;
        const bool realError = (mr && dr) || (mi && di);
        const bool imagError = (mr && di) || (mi && dr);
        return (!realError || error < FastBudgetExponent(result.real)) &&
               (!imagError || error < FastBudgetExponent(result.imag));
    }
    __aicore__ inline spsv_scaled::Complex DivideScaled(spsv_scaled::Complex value, spsv_scaled::Complex divisor)
    {
        if (t_.unit)
            return value;
        if constexpr (Components == 1)
            value.real = spsv_scaled::Div(value.real, divisor.real);
        else
            value = spsv_scaled::Div(value, divisor);
        return value;
    }
    __aicore__ inline bool TryFastResidual(int64_t row, Complex alpha, Complex source, spsv_scaled::Complex divisor,
                                           spsv_scaled::Complex& result, ResidualInfo& info)
    {
        const int64_t begin = row_.Read(row);
        const int64_t end = row_.Read(row + 1);
        int64_t terms = end - begin + 1;
        if (terms > kMaxBoundedTerms)
            return false;
        FastResidual sum;
        if (!AddFastProduct(alpha, spsv_scaled::FromPair(ToPair(source)), false, sum))
            return false;
        for (int64_t p = begin; p < end; ++p) {
            const int32_t col = col_.Read(p);
            if (col != row && !AddFastProduct(Coefficient(p), Previous(col), true, sum))
                return false;
        }
        int termBits = 0;
        while (terms > 1) {
            terms = (terms + 1) / 2;
            ++termBits;
        }
        sum.value.real = spsv_scaled::Normalize(sum.value.real.fraction, sum.value.real.exponent);
        sum.value.imag = spsv_scaled::Normalize(sum.value.imag.fraction, sum.value.imag.exponent);
        sum.magnitude.real = spsv_scaled::Normalize(sum.magnitude.real.fraction, sum.magnitude.real.exponent);
        sum.magnitude.imag = spsv_scaled::Normalize(sum.magnitude.imag.fraction, sum.magnitude.imag.exponent);
        if (sum.value.real.fraction.hi == 0)
            sum.value.real = spsv_scaled::FromFloat(0);
        if (sum.value.imag.fraction.hi == 0)
            sum.value.imag = spsv_scaled::FromFloat(0);
        info.magnitude = sum.magnitude;
        info.termBits = termBits;
        info.valid = true;
        result = DivideScaled(sum.value, divisor);
        return FastErrorFits(sum, divisor, result, termBits);
    }
    __aicore__ inline bool PrefixBudgetFits(int64_t error, spsv_scaled::Real value)
    {
        // Internal dependencies have no public absolute-error floor: a tiny
        // residual can be amplified by a later coefficient.
        return value.fraction.hi != 0 && error < static_cast<int64_t>(value.exponent) - 11;
    }
    __aicore__ inline bool RequiresPrefixReplay(int64_t row, const ResidualInfo& info, spsv_scaled::Complex divisor,
                                                spsv_scaled::Complex result)
    {
        if (!FiniteScaled(divisor) || !FiniteScaled(result))
            return false;
        const bool dr = divisor.real.fraction.hi != 0;
        const bool di = Components == 2 && divisor.imag.fraction.hi != 0;
        if (!dr && !di)
            return false;
        if (!info.valid)
            return static_cast<int64_t>(row_.Read(row + 1)) - row_.Read(row) + 1 > kMaxBoundedTerms;
        const bool mr = info.magnitude.real.fraction.hi != 0;
        const bool mi = Components == 2 && info.magnitude.imag.fraction.hi != 0;
        if (!mr && !mi)
            return false;
        // Two-word dependency rounding needs its own budget, independent of
        // whether this row's residual was recomputed exactly.
        const int64_t error = static_cast<int64_t>(MaxScaledExponent(info.magnitude)) - MaxScaledExponent(divisor) +
                              info.termBits - 41;
        const bool realError = (mr && dr) || (mi && di);
        const bool imagError = (mr && di) || (mi && dr);
        return (realError && !PrefixBudgetFits(error, result.real)) ||
               (imagError && !PrefixBudgetFits(error, result.imag));
    }
    __aicore__ inline spsv_scaled::Complex ScaledDiagonal(int64_t row)
    {
        if (t_.unit)
            return {spsv_scaled::FromFloat(1), spsv_scaled::FromFloat(0)};
        spsv_scaled::ExactDiagonalSum<Components> diagonal;
        for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
            if (col_.Read(p) == row) {
                auto a = Coefficient(p);
                diagonal.Add(a.real, a.imag);
            }
        }
        return diagonal.Value();
    }
    __aicore__ inline ExponentSpan ResidualSpan(int64_t row, Complex alpha, Complex source)
    {
        ExponentSpan span;
        Track(alpha, spsv_scaled::FromPair(ToPair(source)), span);
        for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
            int32_t col = col_.Read(p);
            if (col != row)
                Track(Coefficient(p), Previous(col), span);
        }
        // More than 32 guard bits cover carries from every legal stored term.
        span.last += 33;
        return span;
    }
    __aicore__ inline void FillWindow(spsv_scaled::ExactResidualSum<Components>& sum, int64_t row, Complex alpha,
                                      Complex source, int exponent)
    {
        sum.AddScaledProduct(alpha.real, alpha.imag, spsv_scaled::FromPair(ToPair(source)), exponent, false);
        for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
            int32_t col = col_.Read(p);
            if (col == row)
                continue;
            auto a = Coefficient(p);
            sum.AddScaledProduct(a.real, a.imag, Previous(col), exponent, true);
        }
    }
    __aicore__ inline spsv_scaled::Complex Residual(int64_t row, Complex alpha, Complex source, bool& negativeReal,
                                                    bool& negativeImag)
    {
        auto span = ResidualSpan(row, alpha, source);
        spsv_scaled::Complex result{spsv_scaled::FromFloat(0), spsv_scaled::FromFloat(0)};
        int64_t realCarry = 0;
        int64_t imagCarry = 0;
        for (int first = span.first; first <= span.last; first += 640) {
            spsv_scaled::ExactResidualSum<Components> sum;
            sum.BeginWindow(realCarry, imagCarry, negativeReal, negativeImag);
            int exponent = first + 298;
            FillWindow(sum, row, alpha, source, exponent);
            realCarry = sum.Carry(0);
            if constexpr (Components == 2)
                imagCarry = sum.Carry(1);
            auto digit = sum.UnsignedValue();
            digit.real.exponent += exponent;
            digit.imag.exponent += exponent;
            result = spsv_scaled::Add(result, digit);
        }
        negativeReal = realCarry < 0;
        negativeImag = imagCarry < 0;
        return result;
    }
    __aicore__ inline spsv_scaled::Complex Compute(int64_t row, Complex alpha, Complex source, bool& replay)
    {
        const auto diagonal = ScaledDiagonal(row);
        spsv_scaled::Complex fast;
        ResidualInfo info;
        if (TryFastResidual(row, alpha, source, diagonal, fast, info)) {
            replay = RequiresPrefixReplay(row, info, diagonal, fast);
            return fast;
        }
        bool negativeReal = false;
        bool negativeImag = false;
        auto result = Residual(row, alpha, source, negativeReal, negativeImag);
        if (negativeReal || negativeImag) {
            bool realSign = negativeReal;
            bool imagSign = negativeImag;
            result = Residual(row, alpha, source, negativeReal, negativeImag);
            if (realSign)
                result.real = spsv_scaled::Negate(result.real);
            if (imagSign)
                result.imag = spsv_scaled::Negate(result.imag);
        }
        result = DivideScaled(result, diagonal);
        replay = RequiresPrefixReplay(row, info, diagonal, result);
        return result;
    }
    const SpsvA2Tiling& t_;
    UbCache<int32_t, spsv_a2_ub::kRowCapacity>& row_;
    UbCache<int32_t, spsv_a2_ub::kColumnCapacity>& col_;
    UbCache<float, spsv_a2_ub::kValueCapacity>& value_;
    UbCache<float, spsv_a2_ub::kInputCapacity>& x_;
    UbCache<float, YCapacity>& y_;
};

} // namespace spsv_a2
#endif
