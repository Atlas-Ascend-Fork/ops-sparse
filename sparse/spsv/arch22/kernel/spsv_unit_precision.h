/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_SPSV_UNIT_PRECISION_H_
#define SPSV_ARCH22_SPSV_UNIT_PRECISION_H_
#include "spsv_dd_math.h"
#include "spsv_binary64.h"
#include "spsv_tiling_data.h"
#include "spsv_unit_integer.h"

#ifndef SPSV_UNIT_BOUND_FN
#define SPSV_UNIT_BOUND_FN __aicore__ inline
#endif
#ifndef SPSV_UNIT_PRECISION_AICORE
#define SPSV_UNIT_PRECISION_AICORE __aicore__
#endif

namespace spsv_unit_bound {
// For finite inputs, positive upper bounds keep a normalized FP32 fraction.
// Aligned operands stay normal; sums and fraction products lie in [1, 4].
// Rounding upward by one ulp encloses RN error before renormalization.
struct Real {
    float fraction = 0;
    int exponent = 0;
};
SPSV_UNIT_BOUND_FN Real Positive(float value)
{
    uint32_t bits = spsv_extended64::Bits(value) & 0x7fffffffu;
    if (bits == 0)
        return {};
    int exponent = static_cast<int>(bits >> 23) - 127;
    if ((bits >> 23) == 0) {
        exponent = -126;
        while (bits < 0x00800000u) {
            bits <<= 1;
            --exponent;
        }
    }
    return {spsv_extended64::Float((bits & 0x7fffffu) | 0x3f800000u), exponent};
}
SPSV_UNIT_BOUND_FN Real Up(float value, int exponent)
{
    const uint32_t bits = spsv_extended64::Bits(value) + 1;
    exponent += static_cast<int>(bits >> 23) - 127;
    return {spsv_extended64::Float((bits & 0x7fffffu) | 0x3f800000u), exponent};
}
SPSV_UNIT_BOUND_FN bool Greater(Real a, Real b)
{
    if (b.fraction == 0)
        return a.fraction != 0;
    if (a.fraction == 0)
        return false;
    return a.exponent > b.exponent || (a.exponent == b.exponent && a.fraction > b.fraction);
}
SPSV_UNIT_BOUND_FN Real Add(Real a, Real b)
{
    if (a.fraction == 0)
        return b;
    if (b.fraction == 0)
        return a;
    if (a.exponent < b.exponent) {
        const auto saved = a;
        a = b;
        b = saved;
    }
    const int64_t shift = static_cast<int64_t>(a.exponent) - b.exponent;
    if (shift > 24)
        return Up(a.fraction, a.exponent);
    const float scale = spsv_extended64::Float(static_cast<uint32_t>(127 - shift) << 23);
    return Up(a.fraction + b.fraction * scale, a.exponent);
}
SPSV_UNIT_BOUND_FN Real Mul(Real a, Real b)
{
    if (a.fraction == 0 || b.fraction == 0)
        return {};
    return Up(a.fraction * b.fraction, a.exponent + b.exponent);
}
} // namespace spsv_unit_bound

namespace spsv_wide {
struct ScalarUnitMath {
    static constexpr bool kAvailable = false;
};
template <class Solver, class Address>
class UnitPrecision {
    using Base = spsv_unit_bound::Real;
    using Input = typename Solver::Input;
    static constexpr int kWords = spsv_a2_ub::kUnitPrecisionWords;
    static constexpr unsigned kSpanBits = spsv_a2_ub::kUnitSpanBits;
    static_assert(2 * kSpanBits < 31, "Packed span metadata must fit positive I32 values");
    static constexpr uint32_t kSpanMask = (1u << kSpanBits) - 1;
    static constexpr uint32_t kNegativeBit = 1u << (2 * kSpanBits);
    static constexpr int kSmallWords = 4;
    static constexpr int kVectorMinimumWords = 32;
    // CSR begin/end, one scalar span, and three growth-bound words.
    static_assert(spsv_a2_ub::SolutionWords(spsv_a2_ub::kWideRegularTerms) >= 2 + 2 + 3,
                  "Unit metadata must fit the solution region");
    static_assert(static_cast<uint64_t>(spsv_a2_ub::kResidentCapacity) * spsv_a2_ub::kUnitPrecisionStride <=
                      0x7fffffffu,
                  "Packed state offsets must fit I32 metadata");

public:
    SPSV_UNIT_PRECISION_AICORE inline UnitPrecision(Solver& solver, Address output, Address workspace)
        : s_(solver), output_(output), workspace_(workspace)
    {}
    SPSV_UNIT_PRECISION_AICORE inline bool Run(Input alpha)
    {
        ScalarUnitMath math;
        return Run(alpha, math);
    }
    template <class Math>
    SPSV_UNIT_PRECISION_AICORE __attribute__((noinline)) bool Run(Input alpha, Math& math)
    {
        if (!s_.t_.unit || s_.t_.precisionOffset == 0)
            return false;
        cursor_ = 0;
        sourceFirst_ = 256;
        vector_ = false;
        const int64_t metadataWords = s_.t_.m * MetadataStride();
        s_.output_.Rebind(workspace_ + s_.t_.solutionOffset + metadataWords * sizeof(int32_t), s_.t_.m * 3);
        if (!Prepare(alpha)) {
            Restore();
            return false;
        }
        if constexpr (Math::kAvailable) {
            vector_ = count_ > kVectorMinimumWords;
            if (vector_)
                math.Configure(workspace_ + s_.t_.precisionOffset,
                               s_.t_.m * s_.t_.components * spsv_a2_ub::kUnitPrecisionStride);
        }
        wordBits_ = vector_ ? spsv_a2_ub::kUnitDigitBits : 32;
        InitializeRows();
        s_.row_.Rebind(workspace_ + s_.t_.solutionOffset, metadataWords);
        const int stride = vector_ ? spsv_a2_ub::kUnitPrecisionStride : count_ + 2;
        s_.output_.Rebind(workspace_ + s_.t_.precisionOffset, s_.t_.m * s_.t_.components * stride);
        Commit(Solve(alpha, math));
        return true;
    }

private:
    SPSV_UNIT_PRECISION_AICORE inline int64_t Row(int64_t step)
    {
        return (s_.t_.transpose ? !s_.t_.lower : s_.t_.lower) ? step : s_.t_.m - 1 - step;
    }
    struct WordSpan {
        int64_t offset;
        int begin;
        int end;
        bool negative;
    };
    SPSV_UNIT_PRECISION_AICORE inline int MetadataStride() { return 2 + 2 * s_.t_.components; }
    SPSV_UNIT_PRECISION_AICORE inline int64_t MetadataIndex(int64_t row, int component)
    {
        return row * MetadataStride() + 2 + component * 2;
    }
    SPSV_UNIT_PRECISION_AICORE inline WordSpan Previous(int64_t row, int component)
    {
        const int64_t index = MetadataIndex(row, component);
        const uint32_t code = static_cast<uint32_t>(s_.row_.Read(index + 1));
        return {s_.row_.Read(index), static_cast<int>(code & kSpanMask),
                static_cast<int>((code >> kSpanBits) & kSpanMask), (code & (kNegativeBit)) != 0};
    }
    SPSV_UNIT_PRECISION_AICORE inline void InitializeRows()
    {
        s_.output_.Rebind(workspace_ + s_.t_.solutionOffset, s_.t_.m * MetadataStride());
        for (int64_t row = 0; row < s_.t_.m; ++row) {
            const int64_t index = row * MetadataStride();
            s_.output_.Write(index, spsv_extended64::Float(static_cast<uint32_t>(s_.row_.Read(row))));
            s_.output_.Write(index + 1, spsv_extended64::Float(static_cast<uint32_t>(s_.row_.Read(row + 1))));
            for (int component = 0; component < s_.t_.components; ++component) {
                s_.output_.Write(MetadataIndex(row, component), 0);
                s_.output_.Write(MetadataIndex(row, component) + 1, 0);
            }
        }
        s_.output_.Flush();
    }
    SPSV_UNIT_PRECISION_AICORE inline Base Positive(float value) { return spsv_unit_bound::Positive(value); }
    SPSV_UNIT_PRECISION_AICORE inline Base Add(Base a, Base b) { return spsv_unit_bound::Add(a, b); }
    SPSV_UNIT_PRECISION_AICORE inline Base Mul(Base a, Base b) { return spsv_unit_bound::Mul(a, b); }
    SPSV_UNIT_PRECISION_AICORE inline Base Norm(Input value) { return Add(Positive(value.real), Positive(value.imag)); }
    SPSV_UNIT_PRECISION_AICORE inline bool Finite(Input value) { return spsv_dd::Finite(value.real) && spsv_dd::Finite(value.imag); }
    SPSV_UNIT_PRECISION_AICORE inline int LowestBit(float value)
    {
        const uint32_t bits = spsv_extended64::Bits(value) & 0x7fffffffu;
        const uint32_t field = bits >> 23;
        uint32_t fraction = (bits & 0x7fffffu) | (field == 0 ? 0 : 0x800000u);
        if (fraction == 0)
            return 256;
        int exponent = field == 0 ? -149 : static_cast<int>(field) - 150;
        while ((fraction & 1u) == 0) {
            fraction >>= 1;
            ++exponent;
        }
        return exponent;
    }
    SPSV_UNIT_PRECISION_AICORE inline int LowestBit(Input value)
    {
        const int real = LowestBit(value.real), imag = LowestBit(value.imag);
        return real < imag ? real : imag;
    }
    SPSV_UNIT_PRECISION_AICORE inline Base ReadBound(int64_t row)
    {
        const int64_t offset = row * 3;
        return {s_.output_.Read(offset), static_cast<int32_t>(spsv_extended64::Bits(s_.output_.Read(offset + 1)))};
    }
    SPSV_UNIT_PRECISION_AICORE inline void StoreBound(int64_t row, Base value)
    {
        const int64_t offset = row * 3;
        s_.output_.Write(offset, value.fraction);
        s_.output_.Write(offset + 1, spsv_extended64::Float(static_cast<uint32_t>(value.exponent)));
        s_.output_.Write(offset + 2, 0);
    }
    SPSV_UNIT_PRECISION_AICORE inline bool PrepareRow(int64_t row, Input alpha, Base& largest, Base& source, int64_t& terms)
    {
        const Input x{s_.input_.Read(row * s_.t_.components), s_.t_.components == 2 ? s_.input_.Read(row * 2 + 1) : 0};
        if (!Finite(x))
            return false;
        const int inputFirst = LowestBit(x);
        if (inputFirst < sourceFirst_)
            sourceFirst_ = inputFirst;
        const auto localSource = Mul(Norm(alpha), Norm(x));
        if (spsv_unit_bound::Greater(localSource, source))
            source = localSource;
        auto bound = Positive(1);
        const int64_t begin = s_.row_.Read(row), end = s_.row_.Read(row + 1);
        if (end - begin + 1 > terms)
            terms = end - begin + 1;
        for (int64_t p = begin; p < end; ++p) {
            const int32_t col = s_.column_.Read(p);
            if (col == row)
                continue;
            const auto coefficient = s_.Coefficient(p);
            if (!Finite(coefficient))
                return false;
            bound = Add(bound, Mul(Norm(coefficient), ReadBound(col)));
        }
        if (!spsv_dd::Finite(bound.fraction))
            return false;
        StoreBound(row, bound);
        if (spsv_unit_bound::Greater(bound, largest))
            largest = bound;
        return true;
    }
    SPSV_UNIT_PRECISION_AICORE inline bool Prepare(Input alpha)
    {
        if (!Finite(alpha))
            return false;
        Base largest = Positive(1), source = Positive(0);
        int64_t terms = 1;
        for (int64_t step = 0; step < s_.t_.m; ++step)
            if (!PrepareRow(Row(step), alpha, largest, source, terms))
                return false;
        if (source.fraction == 0)
            return false;
        termBits_ = 0;
        for (int64_t count = terms; count > 1; count = (count + 1) / 2)
            ++termBits_;
        // G bounds the infinity norm of the absolute triangular inverse.
        // Each coefficient product loses less than 2^first per source word.
        // G * maxTerms * maxWords * components * 2^first < 2^-24.
        first_ = -(static_cast<int64_t>(largest.exponent) + termBits_ + spsv_a2_ub::kUnitPrecisionLogWords +
                   (s_.t_.components == 2 ? 1 : 0) + 1 + 24);
        // Retain every bit of the actual alpha*x products, including subnormals.
        // Normal inputs need not reserve all 298 fractional FP32-product bits.
        const int sourceFirst = sourceFirst_ + LowestBit(alpha);
        if (sourceFirst < first_)
            first_ = sourceFirst;
        const int64_t bits = static_cast<int64_t>(largest.exponent) + source.exponent + 3 - first_;
        if (bits > kWords * 32)
            return false;
        count_ = static_cast<int>(bits > 32 ? (bits + 31) / 32 : 1);
        return true;
    }
    template <class Sum>
    SPSV_UNIT_PRECISION_AICORE inline void Accumulate(Sum* sums, Input coefficient, int64_t col)
    {
        for (int component = 0; component < s_.t_.components; ++component) {
            const auto span = Previous(col, component);
            if (span.end == span.begin)
                continue;
            const bool negative = !span.negative;
            sums[component].AddProduct(coefficient.real, s_.output_, span.offset, negative, span.end - span.begin,
                                       span.begin);
            if (s_.t_.components == 2)
                sums[1 - component].AddProduct(coefficient.imag, s_.output_, span.offset, negative != (component != 0),
                                               span.end - span.begin, span.begin);
        }
    }
    SPSV_UNIT_PRECISION_AICORE inline int LocalWords(int64_t row, Input alpha, Input x)
    {
        if (count_ <= kSmallWords)
            return count_;
        const auto source = Mul(Norm(alpha), Norm(x));
        int64_t bits = source.fraction == 0 ? 1 : static_cast<int64_t>(source.exponent) + 1 - first_;
        const int64_t begin = s_.row_.Read(row * MetadataStride()), end = s_.row_.Read(row * MetadataStride() + 1);
        for (int64_t p = begin; p < end; ++p) {
            const int32_t col = s_.column_.Read(p);
            if (col == row)
                continue;
            int highest = 0;
            for (int component = 0; component < s_.t_.components; ++component) {
                const auto span = Previous(col, component);
                if (span.end > span.begin && span.end > highest)
                    highest = span.end;
            }
            const auto value = s_.Coefficient(p);
            const uint32_t real = spsv_extended64::Bits(value.real) & 0x7fffffffu;
            const uint32_t imag = spsv_extended64::Bits(value.imag) & 0x7fffffffu;
            const uint32_t magnitude = real > imag ? real : imag;
            if (highest == 0 || magnitude == 0)
                continue;
            const int field = static_cast<int>(magnitude >> 23);
            const int exponent = field == 0 ? -126 : field - 126;
            const int64_t bound = highest * wordBits_ + exponent + (s_.t_.components == 2 ? 2 : 0);
            if (bound > bits)
                bits = bound;
        }
        bits += termBits_ + 2;
        const int needed = bits > 32 ? static_cast<int>((bits + 31) / 32) : 1;
        return needed < count_ ? needed : count_;
    }
    template <class Sum>
    SPSV_UNIT_PRECISION_AICORE inline bool Store(int64_t row, int component, Sum& sum)
    {
        if (!sum.Valid())
            return false;
        const bool negative = sum.MakeAbsolute();
        if constexpr (Sum::kVector) {
            const auto packed = sum.Pack(cursor_, negative);
            if (!packed.valid)
                return false;
            StoreSpan(row, component, packed.offset, packed.begin, packed.end, negative);
            cursor_ += packed.words;
            return true;
        }
        int begin = 0, end = sum.Count();
        while (begin < end && sum.Word(begin) == 0)
            ++begin;
        while (end > begin && sum.Word(end - 1) == 0)
            --end;
        if (begin == end)
            begin = end = 0;
        for (int word = begin; word < end; ++word)
            s_.output_.Write(cursor_ + word - begin, spsv_extended64::Float(sum.Word(word)));
        s_.output_.Write(cursor_ + end - begin, spsv_extended64::Float(sum.FloatBits(first_, negative)));
        StoreSpan(row, component, cursor_, begin, end, negative);
        cursor_ += end - begin + 1;
        return true;
    }
    SPSV_UNIT_PRECISION_AICORE inline void StoreSpan(int64_t row, int component, int64_t offset, int begin, int end, bool negative)
    {
        const uint32_t code = static_cast<uint32_t>(begin) | (static_cast<uint32_t>(end) << kSpanBits) |
                              (negative ? kNegativeBit : 0);
        const int64_t index = MetadataIndex(row, component);
        s_.row_.Write(index, static_cast<int32_t>(offset));
        s_.row_.Write(index + 1, static_cast<int32_t>(code));
    }
    template <class Math>
    SPSV_UNIT_PRECISION_AICORE inline bool SolveRow(int64_t row, Input alpha, Math& math)
    {
        const Input x{s_.input_.Read(row * s_.t_.components), s_.t_.components == 2 ? s_.input_.Read(row * 2 + 1) : 0};
        const int words = LocalWords(row, alpha, x);
        if constexpr (Math::kAvailable) {
            if (vector_) {
                math.BeginRow(words, first_, s_.t_.components);
                typename Math::Integer sums[2]{{math, 0}, {math, 1}};
                return RowTerms(row, alpha, x, sums);
            }
        }
        UnitInteger sums[2]{UnitInteger(words), UnitInteger(words)};
        return RowTerms(row, alpha, x, sums);
    }
    template <class Sum>
    SPSV_UNIT_PRECISION_AICORE inline bool RowTerms(int64_t row, Input alpha, Input x, Sum* sums)
    {
        sums[0].AddSource(alpha.real, x.real, first_, false);
        if (s_.t_.components == 2) {
            sums[0].AddSource(alpha.imag, x.imag, first_, true);
            sums[1].AddSource(alpha.real, x.imag, first_, false);
            sums[1].AddSource(alpha.imag, x.real, first_, false);
        }
        const int64_t begin = s_.row_.Read(row * MetadataStride()), end = s_.row_.Read(row * MetadataStride() + 1);
        for (int64_t p = begin; p < end; ++p) {
            const int32_t col = s_.column_.Read(p);
            if (col != row)
                Accumulate(sums, s_.Coefficient(p), col);
        }
        for (int component = 0; component < s_.t_.components; ++component)
            if (!Store(row, component, sums[component]))
                return false;
        return true;
    }
    template <class Math>
    SPSV_UNIT_PRECISION_AICORE inline bool Solve(Input alpha, Math& math)
    {
        for (int64_t step = 0; step < s_.t_.m; ++step)
            if (!SolveRow(Row(step), alpha, math))
                return false;
        return true;
    }
    SPSV_UNIT_PRECISION_AICORE inline void Restore()
    {
        s_.output_.Rebind(workspace_ + s_.t_.solutionOffset, s_.t_.m * s_.t_.components * s_.StorageWords());
    }
    SPSV_UNIT_PRECISION_AICORE inline void Commit(bool valid)
    {
        // Preserve all source values until the forward solve has finished.
        // A violated integer range invariant is exposed, never recomputed.
        s_.input_.Rebind(workspace_ + s_.t_.inputOffset, s_.t_.m * s_.t_.components);
        for (int64_t i = 0; i < s_.t_.m * s_.t_.components; ++i) {
            const auto span = valid ? Previous(i / s_.t_.components, static_cast<int>(i % s_.t_.components)) :
                                      WordSpan{0, 0, 0, false};
            s_.input_.Write(i, valid ? s_.output_.Read(span.offset + (vector_ ? (span.end - span.begin + 1) / 2 :
                                                                                span.end - span.begin)) :
                                       spsv_extended64::Float(0x7fc00000u));
        }
        s_.output_.Rebind(output_, s_.t_.m * s_.t_.components);
        for (int64_t i = 0; i < s_.t_.m * s_.t_.components; ++i)
            s_.output_.Write(i, s_.input_.Read(i));
        s_.input_.Flush();
        s_.output_.Flush();
        s_.row_.Rebind(workspace_ + s_.t_.rowOffset, s_.t_.m + 1);
    }
    Solver& s_;
    Address output_, workspace_;
    int count_ = 0;
    int wordBits_ = 32;
    bool vector_ = false;
    int termBits_ = 0;
    int64_t cursor_ = 0;
    int sourceFirst_ = 256;
    int64_t first_ = 0;
};
} // namespace spsv_wide

#endif // SPSV_ARCH22_SPSV_UNIT_PRECISION_H_
