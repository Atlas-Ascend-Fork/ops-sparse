/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_WIDE_SOLUTION_H_
#define SPSV_WIDE_SOLUTION_H_
#include "spsv_tiling_data.h"
#include "spsv_exact_sum.h"
#define SPSV_EXTENDED64_FN __aicore__ inline
#include "spsv_expansion64.h"
#undef SPSV_EXTENDED64_FN

namespace spsv_wide {
template <class Solver, class Address>
class UnitPrecision;
template <int Components, class RowCache, class ColumnCache, class ValueCache, class InputCache, class OutputCache>
class Solution {
    template <class, class>
    friend class UnitPrecision;
    using Base = spsv_extended64::Real;
    using Real = spsv_series64::Real;
    using Pair = spsv_series64::Complex;
    using Sum = spsv_scaled::ExactProductSum;
    static constexpr int kWindowBits = Sum::kBitCount;
    // Complex products combine leading and tail components; include all
    // intermediate rounding carries when selecting occupied windows.
    static constexpr int kProductHeadroom = Components == 1 ? 1 : 4;
    struct Input {
        float real;
        float imag;
    };
    struct Span {
        int64_t first = 0;
        int64_t last = 0;
        bool seen = false;
    };

    struct NextWindow {
        int64_t base;
        int64_t lower;
        int64_t first;
    };

public:
    __aicore__ inline Solution(const SpsvA2Tiling& tiling, RowCache& row, ColumnCache& column, ValueCache& values,
                               InputCache& input, OutputCache& output)
        : t_(tiling), row_(row), column_(column), values_(values), input_(input), output_(output)
    {}

    template <class Alpha, class Address>
    __aicore__ inline bool UnitMain(const Alpha& sourceAlpha, Address input, Address output, Address workspace)
    {
        output_.Rebind(workspace + t_.solutionOffset, t_.m * Components * StorageWords());
        input_.Rebind(input, t_.m * Components);
        return UnitPrecision<Solution, Address>(*this, output, workspace).Run({sourceAlpha.real, sourceAlpha.imag});
    }
    template <class Alpha, class Address, class Math>
    __aicore__ inline bool UnitMain(const Alpha& sourceAlpha, Address input, Address output, Address workspace,
                                    Math& math)
    {
        output_.Rebind(workspace + t_.solutionOffset, t_.m * Components * StorageWords());
        input_.Rebind(input, t_.m * Components);
        return UnitPrecision<Solution, Address>(*this, output, workspace)
            .Run({sourceAlpha.real, sourceAlpha.imag}, math);
    }
    template <class Alpha, class Address>
    __aicore__ __attribute__((noinline)) void Run(const Alpha& sourceAlpha, Address input, Address output,
                                                  Address workspace)
    {
        if (UnitMain(sourceAlpha, input, output, workspace))
            return;
        const Input alpha{sourceAlpha.real, sourceAlpha.imag};
        const bool forward = t_.transpose ? !t_.lower : t_.lower;
        for (int64_t step = 0; step < t_.m; ++step) {
            const int64_t row = forward ? step : t_.m - 1 - step;
            const auto value = Compute(row, alpha);
            Write(row * Components, value.real);
            if constexpr (Components == 2)
                Write(row * Components + 1, value.imag);
        }
        input_.Rebind(output, t_.m * Components);
        for (int64_t offset = 0; offset < t_.m * Components; ++offset)
            input_.Write(offset, spsv_extended64::Float(Rounded(offset)));
        input_.Flush();
        output_.Flush();
    }

private:
    __aicore__ inline Base ReadPart(int64_t offset)
    {
        const uint64_t low = spsv_extended64::Bits(output_.Read(offset));
        const uint64_t high = spsv_extended64::Bits(output_.Read(offset + 1));
        return {{low | (high << 32)}, static_cast<int32_t>(spsv_extended64::Bits(output_.Read(offset + 2)))};
    }
    __aicore__ inline void WritePart(int64_t offset, Base value)
    {
        output_.Write(offset, spsv_extended64::Float(static_cast<uint32_t>(value.fraction.bits)));
        output_.Write(offset + 1, spsv_extended64::Float(static_cast<uint32_t>(value.fraction.bits >> 32)));
        output_.Write(offset + 2, spsv_extended64::Float(static_cast<uint32_t>(value.exponent)));
    }
    __aicore__ inline uint32_t StorageWords() { return 3u * static_cast<uint32_t>(t_.wideTerms) + 1u; }
    __aicore__ inline uint32_t Rounded(int64_t offset)
    {
        const int64_t first = offset * StorageWords();
        const uint32_t count = spsv_extended64::Bits(output_.Read(first + StorageWords() - 1));
        const auto high = count == 0 ? spsv_extended64::FromFloat(0) : ReadPart(first);
        const auto low = count < 2 ? spsv_extended64::FromFloat(0) : ReadPart(first + 3);
        return spsv_expansion64::ToFloatBits({high, low});
    }
    __aicore__ inline Real Read(int64_t offset)
    {
        const int64_t first = offset * StorageWords();
        const uint32_t count = spsv_extended64::Bits(output_.Read(first + StorageWords() - 1));
        Real value;
        for (uint32_t i = 0; i < count && i < static_cast<uint32_t>(t_.wideTerms); ++i)
            spsv_series64::Append(value, ReadPart(first + i * 3));
        return value;
    }
    __aicore__ inline void Write(int64_t offset, const Real& value)
    {
        const int64_t first = offset * StorageWords();
        for (int i = 0; i < value.count; ++i)
            WritePart(first + i * 3, value.parts[i]);
        output_.Write(first + StorageWords() - 1, spsv_extended64::Float(static_cast<uint32_t>(value.count)));
    }
    __aicore__ inline Pair Previous(int64_t column)
    {
        return {Read(column * Components), Components == 2 ? Read(column * Components + 1) : Real{}};
    }
    __aicore__ inline Input Coefficient(int64_t position)
    {
        return {values_.Read(position * Components), Components == 2 ? values_.Read(position * Components + 1) : 0};
    }
    __aicore__ inline Pair FloatPair(Input value)
    {
        return spsv_series64::FromBase(
            spsv_extended64::Complex{spsv_extended64::FromFloat(value.real), spsv_extended64::FromFloat(value.imag)});
    }
    __aicore__ inline Pair Divide(const Pair& value, const Pair& divisor)
    {
        return t_.unit ? value : spsv_series64::Divide(value, divisor, t_.wideTerms, Components == 2);
    }
    template <class Accumulator>
    __aicore__ inline Real Extract(Accumulator& sum, int64_t first)
    {
        Real value;
        for (int i = 0; i < t_.wideTerms; ++i) {
            int highest;
            const auto bits = sum.RoundedNormalized64(highest);
            const auto part = spsv_extended64::Normalize({bits}, first + highest);
            if (spsv_soft64::IsZero(part.fraction))
                break;
            spsv_series64::Append(value, part);
            if (!spsv_soft64::IsFinite(part.fraction))
                break;
            spsv_extended64::Accumulate(sum, 1, part, first, true);
        }
        return value;
    }
    __aicore__ inline Pair Diagonal(int64_t row)
    {
        if (t_.unit)
            return FloatPair({1, 0});
        int64_t position = -1;
        for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
            if (column_.Read(p) != row)
                continue;
            if (position >= 0)
                return SumDiagonal(row);
            position = p;
        }
        return FloatPair(position < 0 ? Input{0, 0} : Coefficient(position));
    }
    __aicore__ inline Pair SumDiagonal(int64_t row)
    {
        if (t_.unit)
            return FloatPair({1, 0});
        Sum sum[Components];
        for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
            if (column_.Read(p) != row)
                continue;
            const auto coefficient = Coefficient(p);
            sum[0].AddProduct(coefficient.real, 1);
            if constexpr (Components == 2)
                sum[1].AddProduct(coefficient.imag, 1);
        }
        const auto real = Extract(sum[0], -298);
        if constexpr (Components == 1)
            return {real, Real{}};
        else
            return {real, Extract(sum[1], -298)};
    }
    __aicore__ inline void Track(Span& span, Base value, int lowBits)
    {
        if (!spsv_soft64::IsFinite(value.fraction) || spsv_soft64::IsZero(value.fraction))
            return;
        const int64_t first = static_cast<int64_t>(value.exponent) - lowBits;
        const int64_t last = static_cast<int64_t>(value.exponent) + kProductHeadroom;
        if (!span.seen || first < span.first)
            span.first = first;
        if (!span.seen || last > span.last)
            span.last = last;
        span.seen = true;
    }
    __aicore__ inline void Track(NextWindow& next, Base value, int lowBits)
    {
        if (!spsv_soft64::IsFinite(value.fraction) || spsv_soft64::IsZero(value.fraction))
            return;
        const int64_t first = static_cast<int64_t>(value.exponent) - lowBits;
        if (static_cast<int64_t>(value.exponent) + kProductHeadroom < next.lower)
            return;
        const int64_t window = first <= next.lower ? next.lower :
                                                     next.base + (first - next.base) / kWindowBits * kWindowBits;
        if (window < next.first)
            next.first = window;
    }
    template <class Tracker>
    __aicore__ inline void TrackScalarProduct(Tracker& span, float coefficient, Base value)
    {
        uint32_t bits = spsv_extended64::Bits(coefficient) & 0x7fffffffu;
        if (bits == 0 || bits >= 0x7f800000u || spsv_soft64::IsZero(value.fraction) ||
            !spsv_soft64::IsFinite(value.fraction))
            return;
        int exponent = static_cast<int>(bits >> 23) - 127;
        if ((bits >> 23) == 0) {
            exponent = -126;
            while (bits < 0x00800000u) {
                bits <<= 1;
                --exponent;
            }
        }
        const Base product{value.fraction, exponent + value.exponent};
        Track(span, product, 75);
    }
    template <class Tracker>
    __aicore__ inline void TrackPair(Tracker& span, float coefficient, const Real& value)
    {
        for (int i = 0; i < value.count; ++i)
            TrackScalarProduct(span, coefficient, value.parts[i]);
    }
    template <class Tracker>
    __aicore__ inline void TrackProduct(Tracker& span, Input coefficient, const Pair& value)
    {
        TrackPair(span, coefficient.real, value.real);
        if constexpr (Components == 2) {
            TrackPair(span, coefficient.imag, value.imag);
            TrackPair(span, coefficient.real, value.imag);
            TrackPair(span, coefficient.imag, value.real);
        }
    }
    __aicore__ inline int64_t FindNextWindow(int64_t row, Input alpha, Input source, Span span, int64_t lower)
    {
        NextWindow next{span.first, lower, span.first + (span.last - span.first) / kWindowBits * kWindowBits};
        TrackProduct(next, alpha, FloatPair(source));
        TrackProduct(next, {1, 0}, correction_);
        for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
            const int32_t column = column_.Read(p);
            if (column != row)
                TrackProduct(next, Coefficient(p), Previous(column));
        }
        return next.first;
    }
    __aicore__ inline void EmptyWindows(spsv_extended64::Magnitude* magnitude, int64_t* carry, int64_t first,
                                        int64_t end)
    {
        for (int component = 0; component < Components; ++component) {
            if (carry[component] == 0)
                continue;
            Sum sum;
            sum.AddCarry(carry[component]);
            uint64_t significand;
            int highest;
            float special;
            sum.UnsignedWindow(significand, highest, special);
            carry[component] = sum.Carry();
            spsv_extended64::AppendWindow(magnitude[component], significand, highest, first, special);
            // Further empty windows contain only zero or sign extension.
            if (carry[component] < 0 && end - first > kWindowBits)
                spsv_extended64::AppendWindow(magnitude[component], (uint64_t{1} << 56) - 1, (kWindowBits - 1),
                                              end - kWindowBits, 0);
        }
    }
    __aicore__ inline void TrackStored(Span& span, Input coefficient, int64_t row)
    {
        for (int component = 0; component < Components; ++component) {
            const int64_t offset = (row * Components + component) * StorageWords();
            uint32_t count = spsv_extended64::Bits(output_.Read(offset + StorageWords() - 1));
            if (count > static_cast<uint32_t>(t_.wideTerms))
                count = t_.wideTerms;
            if (count == 0)
                continue;
            const Base first = ReadPart(offset);
            const Base last = count == 1 ? first : ReadPart(offset + (count - 1) * 3);
            TrackScalarProduct(span, coefficient.real, first);
            TrackScalarProduct(span, coefficient.real, last);
            if constexpr (Components == 2) {
                TrackScalarProduct(span, coefficient.imag, first);
                TrackScalarProduct(span, coefficient.imag, last);
            }
        }
    }
    __aicore__ inline Span ResidualSpan(int64_t row, Input alpha, Input source)
    {
        Span span;
        TrackProduct(span, alpha, FloatPair(source));
        for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
            const int32_t column = column_.Read(p);
            if (column != row)
                TrackStored(span, Coefficient(p), column);
        }
        span.last += 33;
        return span;
    }
    template <class Accumulator>
    __aicore__ inline void AccumulateScalar(Accumulator& sum, float coefficient, const Real& value, int64_t first,
                                            bool subtract)
    {
        for (int i = 0; i < value.count; ++i)
            spsv_extended64::Accumulate(sum, coefficient, value.parts[i], first, subtract);
    }
    __aicore__ inline bool IntersectsWindow(Input coefficient, const Pair& value, int64_t first, int bits)
    {
        if (!spsv_dd::Finite(coefficient.real) || !spsv_dd::Finite(coefficient.imag) ||
            !spsv_soft64::IsFinite(spsv_series64::At(value.real, 0).fraction) ||
            !spsv_soft64::IsFinite(spsv_series64::At(value.imag, 0).fraction))
            return true;
        Span span;
        TrackProduct(span, coefficient, value);
        return span.seen && span.first <= first + (bits - 1) && span.last >= first;
    }
    template <class Accumulator>
    __aicore__ inline void Accumulate(Accumulator* sums, Input coefficient, const Pair& value, int64_t first,
                                      bool subtract, bool negativeReal, bool negativeImag, bool complete = false)
    {
        if (!complete && !IntersectsWindow(coefficient, value, first, Accumulator::kBitCount))
            return;
        if constexpr (Components == 1) {
            AccumulateScalar(sums[0], coefficient.real, value.real, first, subtract != negativeReal);
        } else {
            const int count = value.real.count > value.imag.count ? value.real.count : value.imag.count;
            for (int i = 0; i < count; ++i) {
                const auto product = spsv_extended64::Product(
                    {spsv_extended64::FromFloat(coefficient.real), spsv_extended64::FromFloat(coefficient.imag)},
                    {spsv_series64::At(value.real, i), spsv_series64::At(value.imag, i)});
                spsv_extended64::Accumulate(sums[0], 1, product.real, first, subtract != negativeReal);
                spsv_extended64::Accumulate(sums[1], 1, product.imag, first, subtract != negativeImag);
            }
        }
    }
    template <class Accumulator>
    __aicore__ inline void FillWindow(Accumulator* sums, int64_t row, Input alpha, Input source, int64_t first,
                                      bool negativeReal, bool negativeImag, bool complete = false)
    {
        Accumulate(sums, alpha, FloatPair(source), first, false, negativeReal, negativeImag, complete);
        Accumulate(sums, {1, 0}, correction_, first, true, negativeReal, negativeImag, complete);
        for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
            const int32_t column = column_.Read(p);
            if (column != row)
                Accumulate(sums, Coefficient(p), Previous(column), first, true, negativeReal, negativeImag, complete);
        }
    }
    __aicore__ inline spsv_extended64::Complex ResidualPass(int64_t row, Input alpha, Input source, Span span,
                                                            bool negativeReal, bool negativeImag, bool& realSign,
                                                            bool& imagSign)
    {
        spsv_extended64::Magnitude magnitude[Components];
        int64_t carry[Components]{};
        for (int64_t first = span.first; first <= span.last;) {
            Sum sums[Components];
            for (int component = 0; component < Components; ++component)
                sums[component].AddCarry(carry[component]);
            FillWindow(sums, row, alpha, source, first, negativeReal, negativeImag);
            for (int component = 0; component < Components; ++component) {
                uint64_t significand;
                int highest;
                float special;
                carry[component] = sums[component].Carry();
                sums[component].UnsignedWindow(significand, highest, special);
                spsv_extended64::AppendWindow(magnitude[component], significand, highest, first, special);
            }
            const int64_t lower = first + kWindowBits;
            if (lower > span.last)
                break;
            const int64_t next = FindNextWindow(row, alpha, source, span, lower);
            if (next > lower)
                EmptyWindows(magnitude, carry, lower, next);
            first = next;
        }
        realSign = carry[0] < 0;
        imagSign = Components == 2 && carry[Components - 1] < 0;
        const auto real = spsv_extended64::Finish(magnitude[0], negativeReal);
        if constexpr (Components == 1)
            return {real, spsv_extended64::FromFloat(0)};
        else
            return {real, spsv_extended64::Finish(magnitude[1], negativeImag)};
    }
    __aicore__ inline spsv_extended64::Complex SignedResidual(int64_t row, Input alpha, Input source, Span span)
    {
        bool realSign = false, imagSign = false;
        auto result = ResidualPass(row, alpha, source, span, false, false, realSign, imagSign);
        if (realSign || imagSign) {
            bool ignoredReal, ignoredImag;
            result = ResidualPass(row, alpha, source, span, realSign, imagSign, ignoredReal, ignoredImag);
        }
        return result;
    }
    template <class Accumulator>
    __aicore__ inline Pair FixedResidual(int64_t row, Input alpha, Input source, int64_t first)
    {
        Accumulator sums[Components];
        FillWindow(sums, row, alpha, source, first, false, false, true);
        const auto real = Extract(sums[0], first);
        if constexpr (Components == 1)
            return {real, Real{}};
        else
            return {real, Extract(sums[1], first)};
    }
    __aicore__ inline Pair Residual(int64_t row, Input alpha, Input source)
    {
        correction_ = {};
        const auto span = ResidualSpan(row, alpha, source);
        const int64_t width = span.last - span.first;
        if (width < (kWindowBits - 1))
            return FixedResidual<Sum>(row, alpha, source, span.first);
        // Select a complete row accumulator before summation. Each product
        // is visited once; extracting tails never rescans the dependencies.
        if (width < 2047)
            return FixedResidual<spsv_scaled::ExactAccumulator<64, -298>>(row, alpha, source, span.first);
        if (width < 8191)
            return FixedResidual<spsv_scaled::ExactAccumulator<256, -298>>(row, alpha, source, span.first);
        Pair value;
        for (int i = 0; i < t_.wideTerms; ++i) {
            const auto part = SignedResidual(row, alpha, source, span);
            if (spsv_soft64::IsZero(part.real.fraction) && spsv_soft64::IsZero(part.imag.fraction))
                break;
            spsv_series64::Append(value.real, part.real);
            spsv_series64::Append(value.imag, part.imag);
            if (!spsv_soft64::IsFinite(part.real.fraction) || !spsv_soft64::IsFinite(part.imag.fraction))
                break;
            correction_ = value;
        }
        return value;
    }
    __aicore__ inline Pair SingleDiagonal(int64_t row, int64_t position, Input alpha, Input source)
    {
        const auto a = spsv_series64::Head(FloatPair(alpha));
        const auto x = spsv_series64::Head(FloatPair(source));
        const auto result = Components == 1 ? spsv_extended64::Complex{spsv_extended64::Mul(a.real, x.real),
                                                                       spsv_extended64::FromFloat(0)} :
                                              spsv_extended64::Product(a, x);
        return Divide(spsv_series64::FromBase(result), FloatPair(t_.unit ? Input{1, 0} : Coefficient(position)));
    }
    __aicore__ inline Pair Compute(int64_t row, Input alpha)
    {
        const Input source{input_.Read(row * Components), Components == 2 ? input_.Read(row * 2 + 1) : 0};
        const int64_t begin = row_.Read(row), end = row_.Read(row + 1);
        if (end - begin == 1 && column_.Read(begin) == row)
            return SingleDiagonal(row, begin, alpha, source);
        const auto residual = Residual(row, alpha, source);
        return Divide(residual, Diagonal(row));
    }
    Pair correction_{};
    const SpsvA2Tiling& t_;
    RowCache& row_;
    ColumnCache& column_;
    ValueCache& values_;
    InputCache& input_;
    OutputCache& output_;
};
} // namespace spsv_wide
#include "spsv_unit_precision.h"
#endif // SPSV_WIDE_SOLUTION_H_
