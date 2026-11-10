/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_UNIT_INTEGER_H_
#define SPSV_UNIT_INTEGER_H_
#ifndef SPSV_UNIT_INTEGER_FN
#define SPSV_UNIT_INTEGER_FN __aicore__ inline
#endif

namespace spsv_wide {
// A bounded fixed-point integer. All discarded product bits lie below its
// common binary point; UnitPrecision bounds their amplification before use.
class UnitInteger {
public:
    static constexpr bool kVector = false;
    SPSV_UNIT_INTEGER_FN explicit UnitInteger(int count) : count_(count)
    {
        for (int i = 0; i < count_; ++i)
            words_[i] = 0;
    }
    SPSV_UNIT_INTEGER_FN uint32_t Word(int i) const { return words_[i]; }
    SPSV_UNIT_INTEGER_FN int Count() const { return count_; }
    SPSV_UNIT_INTEGER_FN bool Valid() const
    {
        const bool negative = (words_[count_ - 1] >> 31) != 0;
        return !overflow_ && carry_ == (negative ? -1 : 0);
    }
    SPSV_UNIT_INTEGER_FN bool MakeAbsolute()
    {
        const bool negative = carry_ < 0;
        if (negative) {
            uint64_t carry = 1;
            for (int i = 0; i < count_; ++i) {
                const uint64_t next = static_cast<uint64_t>(~words_[i]) + carry;
                words_[i] = static_cast<uint32_t>(next);
                carry = next >> 32;
            }
        }
        return negative;
    }
    SPSV_UNIT_INTEGER_FN void AddCoefficient(float coefficient, uint32_t word, int index, bool negative)
    {
        const uint32_t bits = spsv_extended64::Bits(coefficient);
        const uint32_t magnitude = bits & 0x7fffffffu;
        const uint32_t field = magnitude >> 23;
        const uint64_t fraction = (magnitude & 0x7fffffu) | (field == 0 ? 0 : 0x800000u);
        Add(fraction * word, static_cast<int64_t>(index) * 32 + (field == 0 ? -149 : static_cast<int>(field) - 150),
            negative != ((bits >> 31) != 0));
    }
    template <class Cache>
    SPSV_UNIT_INTEGER_FN void AddProduct(float coefficient, Cache& source, int64_t offset, bool negative,
                                         int sourceWords = -1, int firstWord = 0)
    {
        const uint32_t bits = spsv_extended64::Bits(coefficient);
        const uint32_t magnitude = bits & 0x7fffffffu, field = magnitude >> 23;
        const uint32_t fraction = (magnitude & 0x7fffffu) | (field == 0 ? 0 : 0x800000u);
        if (fraction == 0)
            return;
        const int shift = (field == 0 ? -149 : static_cast<int>(field) - 150) + firstWord * 32;
        const int sourceCount = sourceWords < 0 ? count_ : sourceWords;
        ProductState state;
        state.word = shift >= 0 ? shift / 32 : (shift - 31) / 32;
        state.shift = static_cast<unsigned>(shift - state.word * 32);
        state.negative = negative != ((bits >> 31) != 0);
        uint64_t productCarry = 0;
        for (int i = 0; i < sourceCount;) {
            uint32_t available = 0;
            const auto span = source.ReadSpan(offset + i, available);
            const uint32_t take = available < static_cast<uint32_t>(sourceCount - i) ? available : sourceCount - i;
            for (uint32_t j = 0; j < take; ++j) {
                const uint64_t product = static_cast<uint64_t>(spsv_extended64::Bits(span.GetValue(j))) * fraction +
                                         productCarry;
                PushProduct(state, static_cast<uint32_t>(product));
                productCarry = product >> 32;
            }
            i += take;
        }
        PushProduct(state, static_cast<uint32_t>(productCarry));
        PushProduct(state, 0);
        while (state.word < count_ && state.carry != 0)
            AddDigit(state, 0);
        carry_ += state.negative ? -static_cast<int64_t>(state.carry) : static_cast<int64_t>(state.carry);
    }
    SPSV_UNIT_INTEGER_FN void AddSource(float a, float b, int64_t first, bool negative)
    {
        const uint32_t ab = spsv_extended64::Bits(a), bb = spsv_extended64::Bits(b);
        const uint32_t ae = (ab & 0x7fffffffu) >> 23, be = (bb & 0x7fffffffu) >> 23;
        const uint64_t af = (ab & 0x7fffffu) | (ae == 0 ? 0 : 0x800000u);
        const uint64_t bf = (bb & 0x7fffffu) | (be == 0 ? 0 : 0x800000u);
        const int64_t shift = (ae == 0 ? -149 : static_cast<int>(ae) - 150) +
                              (be == 0 ? -149 : static_cast<int>(be) - 150) - first;
        Add(af * bf, shift, negative != (((ab ^ bb) >> 31) != 0));
    }
    SPSV_UNIT_INTEGER_FN uint32_t FloatBits(int64_t first, bool negative) const
    {
        int word = count_ - 1;
        while (word > 0 && words_[word] == 0)
            --word;
        if (words_[word] == 0)
            return 0;
        const int bit = HighestBit(words_[word]);
        const int highest = word * 32 + bit;
        const uint32_t sign = negative ? 0x80000000u : 0;
        if (first + highest > 127)
            return sign | 0x7f800000u;
        if (first + highest < -150)
            return sign;
        const uint64_t significand = Significand(highest);
        const uint64_t jammed = (significand >> 3) | ((significand & 7) != 0);
        const uint64_t bits = (negative ? spsv_soft64::kSign : 0) | spsv_extended64::kUnitExponent |
                              (jammed & spsv_soft64::kFraction);
        return spsv_extended64::ToFloatBits({{bits}, static_cast<int>(first + highest)});
    }

private:
    struct ProductState {
        int64_t word = 0;
        unsigned shift = 0;
        uint32_t shiftedCarry = 0;
        uint32_t carry = 0;
        bool negative = false;
    };
    SPSV_UNIT_INTEGER_FN void AddDigit(ProductState& state, uint32_t digit)
    {
        const int64_t word = state.word++;
        if (word < 0)
            return;
        if (word >= count_) {
            overflow_ = overflow_ || digit != 0;
            return;
        }
        const uint32_t old = words_[word];
        if (state.negative) {
            const uint64_t subtrahend = static_cast<uint64_t>(digit) + state.carry;
            words_[word] = old - static_cast<uint32_t>(subtrahend);
            state.carry = static_cast<uint64_t>(old) < subtrahend;
        } else {
            const uint64_t sum = static_cast<uint64_t>(old) + digit + state.carry;
            words_[word] = static_cast<uint32_t>(sum);
            state.carry = static_cast<uint32_t>(sum >> 32);
        }
    }
    SPSV_UNIT_INTEGER_FN void PushProduct(ProductState& state, uint32_t digit)
    {
        const uint32_t shifted = (digit << state.shift) | state.shiftedCarry;
        state.shiftedCarry = state.shift == 0 ? 0 : digit >> (32 - state.shift);
        AddDigit(state, shifted);
    }
    SPSV_UNIT_INTEGER_FN int HighestBit(uint32_t value) const
    {
        int bit = 0;
        if (value >= 0x10000u) {
            value >>= 16;
            bit += 16;
        }
        if (value >= 0x100u) {
            value >>= 8;
            bit += 8;
        }
        if (value >= 0x10u) {
            value >>= 4;
            bit += 4;
        }
        if (value >= 4u) {
            value >>= 2;
            bit += 2;
        }
        return bit + (value >= 2u ? 1 : 0);
    }
    SPSV_UNIT_INTEGER_FN uint64_t Significand(int highest) const
    {
        const int low = highest - 55;
        if (low < 0) {
            const uint64_t head = words_[0] | (count_ > 1 ? static_cast<uint64_t>(words_[1]) << 32 : 0);
            return head << -low;
        }
        const int word = low / 32;
        const unsigned shift = static_cast<unsigned>(low % 32);
        uint64_t value = (static_cast<uint64_t>(words_[word]) >> shift) |
                         (static_cast<uint64_t>(words_[word + 1]) << (32 - shift));
        if (shift > 8)
            value |= static_cast<uint64_t>(words_[word + 2]) << (64 - shift);
        // highest bounds all head reads. Once a sticky bit is found, no
        // remaining lower word can change the rounded significand.
        for (int i = 0; i < word; ++i)
            if (words_[i] != 0)
                return value | 1;
        if (shift != 0 && (words_[word] & ((1u << shift) - 1u)) != 0)
            value |= 1;
        return value;
    }
    SPSV_UNIT_INTEGER_FN void Add(uint64_t product, int64_t shift, bool negative)
    {
        if (product == 0)
            return;
        const int64_t word = shift >= 0 ? shift / 32 : (shift - 31) / 32;
        const unsigned bit = static_cast<unsigned>(shift - word * 32);
        const uint64_t low = product << bit;
        const uint64_t high = bit == 0 ? 0 : product >> (64 - bit);
        AddWord(word, static_cast<uint32_t>(low), negative);
        AddWord(word + 1, static_cast<uint32_t>(low >> 32), negative);
        AddWord(word + 2, static_cast<uint32_t>(high), negative);
    }
    SPSV_UNIT_INTEGER_FN void AddWord(int64_t word, uint32_t value, bool negative)
    {
        if (word < 0 || value == 0)
            return;
        if (word >= count_) {
            overflow_ = true;
            return;
        }
        uint32_t carry = value;
        for (int i = static_cast<int>(word); i < count_ && carry != 0; ++i) {
            const uint32_t old = words_[i];
            const uint32_t next = negative ? old - carry : old + carry;
            words_[i] = next;
            carry = negative ? old < carry : next < old;
        }
        carry_ += negative ? -static_cast<int64_t>(carry) : static_cast<int64_t>(carry);
    }
    volatile uint32_t words_[spsv_a2_ub::kUnitPrecisionWords];
    int count_;
    int64_t carry_ = 0;
    bool overflow_ = false;
};
} // namespace spsv_wide
#endif // SPSV_UNIT_INTEGER_H_
