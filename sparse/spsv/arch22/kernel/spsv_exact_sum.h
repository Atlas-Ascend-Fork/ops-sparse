/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_A2_EXACT_SUM_H_
#define SPSV_A2_EXACT_SUM_H_
#ifndef SPSV_EXACT_SUM_FN
#define SPSV_EXACT_SUM_FN __aicore__ inline
#endif

#include "spsv_dd_math.h"
#ifndef SPSV_SOFT64_FN
#define SPSV_SOFT64_FN SPSV_EXACT_SUM_FN
#define SPSV_EXACT_LOCAL_SOFT64_FN
#endif
#ifndef SPSV_EXTENDED64_FN
#define SPSV_EXTENDED64_FN SPSV_EXACT_SUM_FN
#define SPSV_EXACT_LOCAL_EXTENDED64_FN
#endif
#include "spsv_binary64.h"
#ifdef SPSV_EXACT_LOCAL_EXTENDED64_FN
#undef SPSV_EXACT_LOCAL_EXTENDED64_FN
#undef SPSV_EXTENDED64_FN
#endif
#ifdef SPSV_EXACT_LOCAL_SOFT64_FN
#undef SPSV_EXACT_LOCAL_SOFT64_FN
#undef SPSV_SOFT64_FN
#endif

namespace spsv_scaled {
// Every finite FP32 value is an integer multiple of 2^-149. A signed
// 320-bit accumulator covers all 277 significand positions plus the carry
// from at most INT32_MAX stored entries, without rounding before cancellation.
template <int WordCount, int MinimumExponent>
class ExactAccumulator {
public:
    static constexpr int kBitCount = WordCount * 32;
    SPSV_EXACT_SUM_FN void Add(float value)
    {
        uint32_t bits = Bits(value);
        uint32_t magnitude = bits & 0x7fffffffu;
        if (magnitude >= 0x7f800000u) {
            special_ += value;
            return;
        }
        uint32_t exponent = magnitude >> 23;
        uint32_t significand = magnitude & 0x007fffffu;
        if (exponent != 0) significand |= 0x00800000u;
        uint32_t shift = (exponent == 0 ? 0 : exponent - 1) + (-149 - MinimumExponent);
        uint64_t shifted = static_cast<uint64_t>(significand) << (shift % 32);
        bool subtract = (bits >> 31) != 0;
        AddWord(shift / 32, static_cast<uint32_t>(shifted), subtract);
        AddWord(shift / 32 + 1, static_cast<uint32_t>(shifted >> 32), subtract);
    }
    SPSV_EXACT_SUM_FN void AddProduct(float a, float b, bool negate = false, int64_t scale = 0)
    {
        static_assert(MinimumExponent <= -298, "Exact products require the FP32 product scale");
        uint32_t left = Bits(a);
        uint32_t right = Bits(b);
        uint32_t lm = left & 0x7fffffffu;
        uint32_t rm = right & 0x7fffffffu;
        if (lm >= 0x7f800000u || rm >= 0x7f800000u) {
            special_ += negate ? -(a * b) : a * b;
            return;
        }
        uint32_t le = lm >> 23;
        uint32_t re = rm >> 23;
        uint64_t product = static_cast<uint64_t>((lm & 0x7fffffu) | (le == 0 ? 0 : 0x800000u)) *
            ((rm & 0x7fffffu) | (re == 0 ? 0 : 0x800000u));
        int64_t shift = static_cast<int64_t>(le == 0 ? 0 : le - 1) + (re == 0 ? 0 : re - 1) + (-298 - MinimumExponent) +
                        scale;
        int word = static_cast<int>(shift >= 0 ? shift / 32 : (shift - 31) / 32);
        uint32_t bits = static_cast<uint32_t>(shift - static_cast<int64_t>(word) * 32);
        bool subtract = (((left ^ right) >> 31) != 0) != negate;
        AddWord(word, static_cast<uint32_t>(product << bits), subtract);
        AddWord(word + 1, static_cast<uint32_t>(product >> (32 - bits)), subtract);
        if (bits != 0)
            AddWord(word + 2, static_cast<uint32_t>(product >> (64 - bits)), subtract);
    }
    // Round a signed window without consuming it; the caller can subtract
    // the leading component and extract an independently scaled remainder.
    SPSV_EXACT_SUM_FN uint64_t RoundedNormalized64(int& highest)
    {
        highest = 0;
        if (!spsv_dd::Finite(special_))
            return spsv_soft64::FromFloatBits(Bits(special_)).bits;
        int first = 0;
        while (first < kWords && words_[first] == 0)
            ++first;
        if (first == kWords)
            return 0;
        const bool negative = (words_[kWords - 1] >> 31) != 0;
        int word = kWords - 1;
        while (word > first && MagnitudeWord(word, first, negative) == 0)
            --word;
        highest = word * 32 + 31 - static_cast<int>(__builtin_clz(MagnitudeWord(word, first, negative)));
        const uint64_t significand = SignedSignificand(highest, first, negative);
        return spsv_soft64::RoundPack(negative, 0, significand).bits;
    }
    SPSV_EXACT_SUM_FN Real Value()
    {
        if (!spsv_dd::Finite(special_)) return FromFloat(special_);
        bool negative = (words_[kWords - 1] >> 31) != 0;
        if (negative) NegateWords();
        Real sum = UnsignedValue();
        return negative ? Negate(sum) : sum;
    }
    SPSV_EXACT_SUM_FN Real UnsignedValue()
    {
        if (!spsv_dd::Finite(special_))
            return FromFloat(special_);
        Real sum = FromFloat(0);
        for (int word = kWords - 1; word >= 0; --word) {
            Real high = FromFloat(static_cast<float>(static_cast<int32_t>(words_[word] >> 16)));
            high.exponent += word * 32 + 16 + MinimumExponent;
            sum = spsv_scaled::Add(sum, high);
            Real low = FromFloat(static_cast<float>(static_cast<int32_t>(words_[word] & 0xffffu)));
            low.exponent += word * 32 + MinimumExponent;
            sum = spsv_scaled::Add(sum, low);
        }
        return sum;
    }
    SPSV_EXACT_SUM_FN void AddCarry(int64_t value)
    {
        bool subtract = value < 0;
        uint64_t magnitude = subtract ? 0u - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
        AddWord(0, static_cast<uint32_t>(magnitude), subtract);
        AddWord(1, static_cast<uint32_t>(magnitude >> 32), subtract);
    }
    SPSV_EXACT_SUM_FN int64_t Carry() const { return carry_; }
    SPSV_EXACT_SUM_FN void AddRawWord(int word, uint32_t value, bool subtract) { AddWord(word, value, subtract); }
    SPSV_EXACT_SUM_FN bool UnsignedWindow(uint64_t& significand, int& highest, float& special) const
    {
        significand = 0;
        highest = 0;
        special = special_;
        if (!spsv_dd::Finite(special))
            return false;
        int word = kWords - 1;
        while (word > 0 && words_[word] == 0)
            --word;
        if (words_[word] == 0)
            return false;
        int bit = 31;
        while ((words_[word] >> bit) == 0)
            --bit;
        highest = word * 32 + bit;
        significand = Binary64Significand(highest);
        return true;
    }
    SPSV_EXACT_SUM_FN float RoundedValue()
    {
        static_assert(MinimumExponent == -149, "FP32 rounding requires the FP32 integer scale");
        if (!spsv_dd::Finite(special_)) return special_;
        const uint32_t sign = words_[kWords - 1] & 0x80000000u;
        if (sign != 0) NegateWords();
        int word = kWords - 1;
        while (word > 0 && words_[word] == 0) --word;
        if (words_[word] == 0) return 0;
        int bit = 31;
        while ((words_[word] >> bit) == 0) --bit;
        int highest = word * 32 + bit;
        // Subnormals are exact integer multiples of 2^-149. Do not form
        // them through floating arithmetic, which may flush them to zero.
        if (highest < 23) return FromBits(sign | words_[0]);
        const int shift = highest - 23;
        const int first = shift / 32;
        uint64_t window = words_[first];
        if (first + 1 < kWords) window |= static_cast<uint64_t>(words_[first + 1]) << 32;
        uint32_t significand = static_cast<uint32_t>(window >> (shift % 32));
        if (RoundUp(shift, significand)) ++significand;
        if (significand == 0x01000000u) {
            significand >>= 1;
            ++highest;
        }
        const uint32_t exponent = static_cast<uint32_t>(highest - 22);
        if (exponent >= 255) return FromBits(sign | 0x7f800000u);
        return FromBits(sign | (exponent << 23) | (significand & 0x007fffffu));
    }
private:
    SPSV_EXACT_SUM_FN uint32_t MagnitudeWord(int word, int first, bool negative) const
    {
        if (word < first || word >= kWords)
            return 0;
        const uint32_t value = words_[word];
        return !negative ? value : (word == first ? 0u - value : ~value);
    }
    SPSV_EXACT_SUM_FN uint64_t SignedSignificand(int highest, int first, bool negative) const
    {
        const int low = highest - 55;
        if (low < 0) {
            const uint64_t value = MagnitudeWord(0, first, negative) |
                                   (static_cast<uint64_t>(MagnitudeWord(1, first, negative)) << 32);
            return value << -low;
        }
        const int word = low / 32;
        const unsigned shift = static_cast<unsigned>(low % 32);
        uint64_t value = (static_cast<uint64_t>(MagnitudeWord(word, first, negative)) >> shift) |
                         (static_cast<uint64_t>(MagnitudeWord(word + 1, first, negative)) << (32 - shift));
        if (shift > 8)
            value |= static_cast<uint64_t>(MagnitudeWord(word + 2, first, negative)) << (64 - shift);
        if (first < word || (shift != 0 && (MagnitudeWord(word, first, negative) & ((1u << shift) - 1u)) != 0))
            value |= 1;
        return value;
    }
    SPSV_EXACT_SUM_FN uint64_t Binary64Significand(int highest) const
    {
        int shift = highest - 55;
        if (shift < 0) {
            uint64_t low = words_[0];
            if (kWords > 1)
                low |= static_cast<uint64_t>(words_[1]) << 32;
            return low << (-shift);
        }
        int first = shift / 32;
        spsv_soft64::Wide window{0, words_[first]};
        if (first + 1 < kWords)
            window.lo |= static_cast<uint64_t>(words_[first + 1]) << 32;
        if (first + 2 < kWords)
            window.hi = words_[first + 2];
        uint64_t significand = spsv_soft64::RightWideJam(window, static_cast<unsigned>(shift % 32)).lo;
        for (int word = 0; word < first; ++word) {
            if (words_[word] != 0) {
                significand |= 1;
                break;
            }
        }
        return significand;
    }
    SPSV_EXACT_SUM_FN bool RoundUp(int shift, uint32_t significand) const
    {
        if (shift == 0) return false;
        const int guard = shift - 1;
        const uint32_t mask = 1u << (guard % 32);
        if ((words_[guard / 32] & mask) == 0) return false;
        if ((significand & 1u) != 0 || (words_[guard / 32] & (mask - 1u)) != 0) return true;
        for (int word = 0; word < guard / 32; ++word) {
            if (words_[word] != 0) return true;
        }
        return false;
    }
    SPSV_EXACT_SUM_FN void AddWord(int word, uint32_t value, bool subtract)
    {
        if (word < 0 || word >= kWords)
            return;
        uint32_t carry = value;
        for (int index = word; index < kWords && carry != 0; ++index) {
            uint32_t previous = words_[index];
            uint32_t next = subtract ? previous - carry : previous + carry;
            words_[index] = next;
            carry = subtract ? static_cast<uint32_t>(previous < carry) : static_cast<uint32_t>(next < previous);
        }
        carry_ += subtract ? -static_cast<int64_t>(carry) : static_cast<int64_t>(carry);
    }
    SPSV_EXACT_SUM_FN void NegateWords()
    {
        uint64_t carry = 1;
        for (int word = 0; word < kWords; ++word) {
            uint64_t next = static_cast<uint64_t>(~words_[word]) + carry;
            words_[word] = static_cast<uint32_t>(next);
            carry = next >> 32;
        }
    }
    static constexpr int kWords = WordCount;
    volatile uint32_t words_[kWords]{};
    float special_ = 0;
    int64_t carry_ = 0;
};

using ExactSum = ExactAccumulator<10, -149>;
// Products of two FP32 values are integer multiples of 2^-298. Twenty
// words retain every product bit and the carries from INT32_MAX entries.
using ExactProductSum = ExactAccumulator<20, -298>;
template <int32_t Components>
class ExactResidualSum {
public:
    SPSV_EXACT_SUM_FN void AddProduct(float real, float imag, spsv_dd::Complex value, bool subtract)
    {
        sums_[0].AddProduct(real, value.real.hi, subtract);
        if (value.real.lo != 0) sums_[0].AddProduct(real, value.real.lo, subtract);
        if constexpr (Components == 2) {
            sums_[0].AddProduct(imag, value.imag.hi, !subtract);
            if (value.imag.lo != 0) sums_[0].AddProduct(imag, value.imag.lo, !subtract);
            sums_[1].AddProduct(real, value.imag.hi, subtract);
            if (value.imag.lo != 0) sums_[1].AddProduct(real, value.imag.lo, subtract);
            sums_[1].AddProduct(imag, value.real.hi, subtract);
            if (value.real.lo != 0) sums_[1].AddProduct(imag, value.real.lo, subtract);
        }
    }
    SPSV_EXACT_SUM_FN Complex Value()
    {
        if constexpr (Components == 1) return {sums_[0].Value(), FromFloat(0)};
        else return {sums_[0].Value(), sums_[1].Value()};
    }
    SPSV_EXACT_SUM_FN void BeginWindow(int64_t real, int64_t imag, bool negateReal, bool negateImag)
    {
        sums_[0].AddCarry(real);
        negate_[0] = negateReal;
        if constexpr (Components == 2) {
            sums_[1].AddCarry(imag);
            negate_[1] = negateImag;
        }
    }
    SPSV_EXACT_SUM_FN int64_t Carry(int component) const { return sums_[component].Carry(); }
    SPSV_EXACT_SUM_FN Complex UnsignedValue()
    {
        if constexpr (Components == 1)
            return {sums_[0].UnsignedValue(), FromFloat(0)};
        else
            return {sums_[0].UnsignedValue(), sums_[1].UnsignedValue()};
    }
    SPSV_EXACT_SUM_FN void AddScaledProduct(float real, float imag, Complex value, int exponent, bool subtract)
    {
        AddRealProduct(0, real, value.real, exponent, subtract);
        if constexpr (Components == 2) {
            AddRealProduct(0, imag, value.imag, exponent, !subtract);
            AddRealProduct(1, real, value.imag, exponent, subtract);
            AddRealProduct(1, imag, value.real, exponent, subtract);
        }
    }

private:
    SPSV_EXACT_SUM_FN void AddRealProduct(int component, float coefficient, Real value, int exponent, bool subtract)
    {
        int64_t scale = static_cast<int64_t>(value.exponent) - exponent;
        sums_[component].AddProduct(coefficient, value.fraction.hi, subtract != negate_[component], scale);
        if (value.fraction.lo != 0)
            sums_[component].AddProduct(coefficient, value.fraction.lo, subtract != negate_[component], scale);
    }
    ExactProductSum sums_[Components];
    bool negate_[Components]{};
};

template <int32_t Components>
class ExactDiagonalSum {
public:
    SPSV_EXACT_SUM_FN void Add(float real, float imag)
    {
        sums_[0].Add(real);
        if constexpr (Components == 2) sums_[1].Add(imag);
    }
    SPSV_EXACT_SUM_FN Complex Value()
    {
        if constexpr (Components == 1) return {sums_[0].Value(), FromFloat(0)};
        else return {sums_[0].Value(), sums_[1].Value()};
    }
private:
    ExactSum sums_[Components];
};
}
#endif
