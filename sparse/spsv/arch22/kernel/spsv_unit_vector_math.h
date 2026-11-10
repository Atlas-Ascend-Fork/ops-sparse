/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_SPSV_UNIT_VECTOR_MATH_H_
#define SPSV_ARCH22_SPSV_UNIT_VECTOR_MATH_H_
#include "kernel_operator.h"
#include "spsv_tiling_data.h"
#include "spsv_binary64.h"

namespace spsv_unit_carry {
constexpr int kTableWords = 72;
constexpr int kTableBytes = spsv_a2_ub::kUnitCarryBytes;
__aicore__ inline void Initialize(AscendC::LocalTensor<int32_t> table)
{
    // Monotone maps on incoming carries {-1,0,1}; the last map is identity.
    const int maps[8][3]{{-1, -1, -1}, {-1, -1, 0}, {-1, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {-1, 0, 1}};
    for (int a = 0; a < 8; ++a) {
        for (int b = 0; b < 8; ++b) {
            int code = 0;
            for (; code < 8; ++code)
                if (maps[code][0] == maps[a][maps[b][0] + 1] && maps[code][1] == maps[a][maps[b][1] + 1] &&
                    maps[code][2] == maps[a][maps[b][2] + 1])
                    break;
            table.SetValue(a * 8 + b, code);
        }
        table.SetValue(64 + a, maps[a][1]);
    }
}
class Normalizer {
public:
    __aicore__ inline Normalizer(AscendC::LocalTensor<int32_t> value, AscendC::LocalTensor<int32_t> a,
                                 AscendC::LocalTensor<int32_t> b, AscendC::LocalTensor<int32_t> c,
                                 AscendC::LocalTensor<int32_t> table, int count)
        : value_(value), a_(a), b_(b), c_(c), table_(table), count_(count)
    {}
    __aicore__ inline int64_t Run()
    {
        int64_t carry = 0;
        for (int step = 0; step < 3; ++step)
            carry += CarryStep();
        States();
        for (int distance = 1; distance < count_; distance *= 2)
            Prefix(distance);
        AscendC::ShiftLeft(a_.ReinterpretCast<uint32_t>(), c_.ReinterpretCast<uint32_t>(), uint32_t{2}, count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(b_, table_[64], a_.ReinterpretCast<uint32_t>(), 0, count_);
        AscendC::PipeBarrier<PIPE_V>();
        Indices(c_, 1);
        AscendC::Gather(a_, b_, c_.ReinterpretCast<uint32_t>(), 0, count_);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        carry += b_.GetValue(count_ - 1);
        a_.SetValue(0, 0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::ShiftLeft(b_.ReinterpretCast<uint32_t>(), b_.ReinterpretCast<uint32_t>(), uint32_t{12}, count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(value_, value_, b_, count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(value_, value_, a_, count_);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        return carry;
    }

private:
    __aicore__ inline void Indices(AscendC::LocalTensor<int32_t> dst, int distance)
    {
        // On A2/A3 CreateVecIndex initializes the first UB block through
        // Scalar stores before expanding it with Vector instructions.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::CreateVecIndex(dst, static_cast<int32_t>(-distance), count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(dst, dst, int32_t{0}, count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(dst.ReinterpretCast<uint32_t>(), dst.ReinterpretCast<uint32_t>(), uint32_t{2}, count_);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline int32_t CarryStep()
    {
        AscendC::ShiftRight(a_, value_, int32_t{12}, count_);
        AscendC::Duplicate(c_, int32_t{4095}, count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(value_.ReinterpretCast<uint16_t>(), value_.ReinterpretCast<uint16_t>(),
                     c_.ReinterpretCast<uint16_t>(), count_ * 2);
        AscendC::PipeBarrier<PIPE_V>();
        Indices(c_, 1);
        AscendC::Gather(b_, a_, c_.ReinterpretCast<uint32_t>(), 0, count_);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const int32_t carry = a_.GetValue(count_ - 1);
        b_.SetValue(0, 0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::Add(value_, value_, b_, count_);
        AscendC::PipeBarrier<PIPE_V>();
        return carry;
    }
    __aicore__ inline void States()
    {
        AscendC::Adds(a_, value_, int32_t{2}, count_);
        AscendC::Adds(b_, value_, int32_t{-4094}, count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(a_, a_, int32_t{3}, count_);
        AscendC::Maxs(b_, b_, int32_t{0}, count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(c_, a_, b_, count_);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Prefix(int distance)
    {
        Indices(a_, distance);
        AscendC::Gather(b_, c_, a_.ReinterpretCast<uint32_t>(), 0, count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(b_, int32_t{7}, distance);
        AscendC::ShiftLeft(a_.ReinterpretCast<uint32_t>(), c_.ReinterpretCast<uint32_t>(), uint32_t{3}, count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(a_, a_, b_, count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(a_.ReinterpretCast<uint32_t>(), a_.ReinterpretCast<uint32_t>(), uint32_t{2}, count_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(c_, table_, a_.ReinterpretCast<uint32_t>(), 0, count_);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::LocalTensor<int32_t> value_, a_, b_, c_, table_;
    int count_;
};
} // namespace spsv_unit_carry

namespace spsv_unit_vector {
constexpr int kDigitBits = spsv_a2_ub::kUnitDigitBits;
constexpr int kMask = (1 << kDigitBits) - 1;
constexpr int kMaximumDigits = spsv_a2_ub::kUnitMaximumDigits;
constexpr int kCapacity = spsv_a2_ub::kUnitVectorCapacity;
constexpr int kPrefix = spsv_a2_ub::kUnitPrefixDigits;
constexpr int kShortCapacity = spsv_a2_ub::kUnitShortCapacity;
constexpr int kBytes = spsv_a2_ub::kUnitMathBytes;
constexpr int kScalarCarryMaximumDigits = 256;
constexpr int kProductPrefixDigits = 8;
constexpr int kSelectionWords = 48;
static_assert(kMaximumDigits + kProductPrefixDigits <= kCapacity, "Product source must fit the vector buffer");
struct Packed {
    int64_t offset;
    int begin;
    int end;
    int words;
    bool valid;
};
// A coefficient uses at most three radix-4096 chunks. Truncation at the
// common binary point loses less than two units per shifted chunk, below
// the 512-unit budget reserved by the precision planner. Normalizing after
// 16 products keeps all signed digit sums below 2^30.
class Math {
public:
    static constexpr bool kAvailable = true;
    class Integer;
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(storage_, kBytes);
        sum_[0] = storage_.Get<int32_t>();
        sum_[1] = sum_[0][kCapacity];
        source_ = sum_[1][kCapacity];
        temporary_ = source_[kCapacity];
        coefficient_ = temporary_[kCapacity];
        short_ = storage_.Get<int16_t>()[5 * kCapacity * 2];
        table_ = storage_.Get<int32_t>()[(kBytes - spsv_a2_ub::kUnitCarryBytes) / 4];
        spsv_unit_carry::Initialize(table_);
        scalarDirty_ = true;
    }
    __aicore__ inline void Configure(GM_ADDR base, int64_t words)
    {
        global_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(base), words * 2);
        extent_ = words;
    }
    __aicore__ inline void BeginRow(int words, int64_t first, int components)
    {
        words_ = words;
        digits_ = (words * 32 + 11) / 12;
        first_ = first;
        for (int c = 0; c < components; ++c) {
            AscendC::Duplicate(sum_[c], int32_t{0}, digits_);
            carry_[c] = correction_[c] = 0;
            pending_[c] = 0;
            valid_[c] = true;
        }
        VectorToScalar();
        scalarDirty_ = true;
    }
    __aicore__ inline void AddSource(int c, float a, float b, bool negative)
    {
        const uint32_t ab = spsv_extended64::Bits(a), bb = spsv_extended64::Bits(b);
        const int ae = static_cast<int>((ab & 0x7fffffffu) >> 23), be = static_cast<int>((bb & 0x7fffffffu) >> 23);
        const uint64_t af = (ab & 0x7fffffu) | (ae == 0 ? 0 : 0x800000u);
        const uint64_t bf = (bb & 0x7fffffu) | (be == 0 ? 0 : 0x800000u);
        const int64_t shift = (ae == 0 ? -149 : ae - 150) + (be == 0 ? -149 : be - 150) - first_;
        const int word = static_cast<int>(shift >= 0 ? shift / 12 : (shift - 11) / 12);
        const unsigned bit = static_cast<unsigned>(shift - static_cast<int64_t>(word) * 12);
        const uint64_t value = (af * bf) << bit;
        negative = negative != (((ab ^ bb) >> 31) != 0);
        for (int i = 0; i < 5; ++i) {
            const int digit = static_cast<int>((value >> (i * 12)) & kMask);
            if (digit == 0)
                continue;
            if (word + i < 0 || word + i >= digits_) {
                valid_[c] = false;
                continue;
            }
            const int32_t old = sum_[c].GetValue(word + i);
            sum_[c].SetValue(word + i, negative ? old - digit : old + digit);
        }
        scalarDirty_ = true;
    }
    __aicore__ inline void AddProduct(int c, float coefficient, int64_t offset, bool negative, int count, int begin)
    {
        const uint32_t bits = spsv_extended64::Bits(coefficient), magnitude = bits & 0x7fffffffu;
        const int field = static_cast<int>(magnitude >> 23);
        const uint64_t fraction = (magnitude & 0x7fffffu) | (field == 0 ? 0 : 0x800000u);
        if (fraction == 0 || count == 0)
            return;
        if (pending_[c] == 16)
            Normalize(c);
        if (!PrepareProduct(offset, count)) {
            valid_[c] = false;
            return;
        }
        const int shift = (field == 0 ? -149 : field - 150) + begin * 12;
        const int word = shift >= 0 ? shift / 12 : (shift - 11) / 12;
        const unsigned bit = static_cast<unsigned>(shift - word * 12);
        const uint64_t value = fraction << bit;
        negative = negative != ((bits >> 31) != 0);
        for (int part = 0; part < 3; ++part) {
            const int digit = static_cast<int>((value >> (part * 12)) & kMask);
            if (digit != 0)
                ProductChunk(c, count, word + part, digit, negative);
        }
        ++pending_[c];
    }
    __aicore__ inline bool Valid(int c)
    {
        Normalize(c);
        if (carry_[c] != 0 && carry_[c] != -1)
            return false;
        const int bit = (words_ * 32 - 1) % 12;
        const uint32_t top = static_cast<uint32_t>(sum_[c].GetValue(digits_ - 1));
        const uint32_t extensionMask = kMask & ~((1u << bit) - 1u);
        return valid_[c] && (top & extensionMask) == (carry_[c] < 0 ? extensionMask : 0);
    }
    __aicore__ inline bool MakeAbsolute(int c)
    {
        int first = 0;
        while (first < digits_ && sum_[c].GetValue(first) == 0)
            ++first;
        const bool negative = carry_[c] < 0;
        if (negative && first < digits_) {
            const int32_t initial = sum_[c].GetValue(first);
            ScalarToVector();
            AscendC::Duplicate(coefficient_, int32_t{kMask}, digits_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(sum_[c], coefficient_, sum_[c], digits_);
            AscendC::PipeBarrier<PIPE_V>();
            if (first != 0)
                AscendC::Duplicate(sum_[c], int32_t{0}, first);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            sum_[c].SetValue(first, (1 << 12) - initial);
            scalarDirty_ = true;
        }
        firstDigit_[c] = first;
        int last = digits_ - 1;
        while (last >= first && sum_[c].GetValue(last) == 0)
            --last;
        lastDigit_[c] = last;
        return negative;
    }
    __aicore__ inline uint32_t FloatBits(int c, bool negative) const
    {
        if (lastDigit_[c] < firstDigit_[c])
            return 0;
        uint32_t top = static_cast<uint32_t>(sum_[c].GetValue(lastDigit_[c]));
        int bit = 0;
        while ((top >>= 1) != 0)
            ++bit;
        const int highest = lastDigit_[c] * 12 + bit;
        const uint32_t sign = negative ? 0x80000000u : 0;
        if (first_ + highest > 127)
            return sign | 0x7f800000u;
        if (first_ + highest < -150)
            return sign;
        const uint64_t significand = Significand(c, highest);
        const uint64_t jammed = (significand >> 3) | ((significand & 7) != 0);
        const uint64_t bits = (negative ? spsv_soft64::kSign : 0) | spsv_extended64::kUnitExponent |
                              (jammed & spsv_soft64::kFraction);
        return spsv_extended64::ToFloatBits({{bits}, static_cast<int>(first_ + highest)});
    }
    __aicore__ inline Packed Pack(int c, int64_t cursor, bool negative)
    {
        const bool empty = lastDigit_[c] < firstDigit_[c];
        const int begin = empty ? 0 : firstDigit_[c] / 8 * 8;
        const int end = empty ? 0 : lastDigit_[c] + 1;
        const int count = end - begin;
        const int even = (count + 1) / 2 * 2;
        const int words = (kPrefix + even + 2) / 2;
        if (cursor < 0 || cursor + words > extent_)
            return {0, 0, 0, 0, false};
        const Packed result{cursor + kPrefix / 2, begin, end, words, true};
        PackData(c, cursor, negative, result);
        return result;
    }

private:
    __aicore__ inline void PackData(int c, int64_t cursor, bool negative, const Packed& result)
    {
        const int count = result.end - result.begin;
        const int even = (count + 1) / 2 * 2;
        ScalarToVector();
        AscendC::Duplicate(short_, int16_t{0}, kPrefix);
        if (count != 0)
            AscendC::Cast(short_[kPrefix], sum_[c][result.begin], AscendC::RoundMode::CAST_NONE, count);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        if (even != count)
            short_.SetValue(kPrefix + count, 0);
        const uint32_t bits = FloatBits(c, negative);
        short_.SetValue(kPrefix + even, static_cast<int16_t>(bits));
        short_.SetValue(kPrefix + even + 1, static_cast<int16_t>(bits >> 16));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(global_[cursor * 2], short_,
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(result.words * 4), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
    __aicore__ inline void VectorToScalar() const
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    __aicore__ inline void ScalarToVector()
    {
        if (scalarDirty_) {
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            scalarDirty_ = false;
        }
    }
    __aicore__ inline bool Load(int64_t start, int count)
    {
        if (start < 0 || count < 0 || count > kShortCapacity || start + count > extent_ * 2)
            return false;
        LoadBlock(start, count);
        return true;
    }
    __aicore__ inline void LoadBlock(int64_t start, int count)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopyPad(short_, global_[start],
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(count * 2), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<int16_t>{false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }
    __aicore__ inline bool PrepareProduct(int64_t offset, int count)
    {
        const int total = count + kProductPrefixDigits;
        if (count < 0 || total > kCapacity || !Load(offset * 2 - kProductPrefixDigits, total))
            return false;
        ConvertProduct(total);
        return true;
    }
    __aicore__ inline void ConvertProduct(int total)
    {
        ScalarToVector();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(coefficient_.ReinterpretCast<float>(), short_, AscendC::RoundMode::CAST_NONE, total);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(source_, coefficient_.ReinterpretCast<float>(), AscendC::RoundMode::CAST_RINT, total);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(coefficient_, int32_t{-1}, kSelectionWords);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ProductChunk(int c, int count, int start, int coefficient, bool negative)
    {
        const int skip = start < 0 ? -start : 0;
        if (skip > count)
            return;
        const int low = start < 0 ? 0 : start;
        const int pad = low % 8;
        const int take = count - skip + pad;
        if (low - pad + take > digits_) {
            valid_[c] = false;
            return;
        }
        if (skip != 0) {
            const int product = short_.GetValue(kProductPrefixDigits + skip - 1) * coefficient;
            correction_[c] += negative ? -(product >> 12) : product >> 12;
        }
        if (skip == count)
            return;
        const int begin = kProductPrefixDigits + skip - pad;
        if (begin < 0 || begin >= 32) {
            valid_[c] = false;
            return;
        }
        SelectProduct(c, count, begin, low - pad, take, coefficient, negative);
    }
    __aicore__ inline void SelectProduct(int c, int count, int begin, int destination, int take, int coefficient,
                                         bool negative)
    {
        // FP32 coefficients have least exponent >=-149, so skip<=13.
        // Only the first mask word changes; Counter N limits the upper end.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        coefficient_.SetValue(0, static_cast<int32_t>(0xffffffffu << begin));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        uint64_t kept = 0;
        AscendC::GatherMask<int32_t, uint32_t>(temporary_, source_, coefficient_.ReinterpretCast<uint32_t>(), true,
                                               static_cast<uint32_t>(count + kProductPrefixDigits),
                                               AscendC::GatherMaskParams{1, 1, 0, 0}, kept);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(temporary_, temporary_, static_cast<int32_t>(coefficient), take);
        AscendC::PipeBarrier<PIPE_V>();
        if (negative)
            AscendC::Sub(sum_[c][destination], sum_[c][destination], temporary_, take);
        else
            AscendC::Add(sum_[c][destination], sum_[c][destination], temporary_, take);
    }
    __aicore__ inline void Normalize(int c)
    {
        VectorToScalar();
        if (digits_ > kScalarCarryMaximumDigits) {
            sum_[c].SetValue(0, sum_[c].GetValue(0) + static_cast<int32_t>(correction_[c]));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            carry_[c] += spsv_unit_carry::Normalizer(sum_[c], source_, temporary_, coefficient_, table_, digits_).Run();
            correction_[c] = 0;
            pending_[c] = 0;
            scalarDirty_ = true;
            return;
        }
        int64_t carry = correction_[c];
        for (int i = 0; i < digits_; ++i) {
            const int64_t value = static_cast<int64_t>(sum_[c].GetValue(i)) + carry;
            const uint32_t digit = static_cast<uint32_t>(value) & kMask;
            sum_[c].SetValue(i, static_cast<int32_t>(digit));
            carry = (value - digit) / (1 << 12);
        }
        carry_[c] += carry;
        correction_[c] = 0;
        pending_[c] = 0;
        scalarDirty_ = true;
    }
    __aicore__ inline uint64_t Digit(int c, int i) const
    {
        return i < digits_ ? static_cast<uint32_t>(sum_[c].GetValue(i)) : 0;
    }
    __aicore__ inline uint64_t Significand(int c, int highest) const
    {
        const int low = highest - 55;
        const int first = low < 0 ? 0 : low / 12;
        const unsigned shift = low < 0 ? 0 : static_cast<unsigned>(low % 12);
        uint64_t value = 0;
        for (int i = 0; i < 5; ++i)
            value |= Digit(c, first + i) << (i * 12);
        if (low < 0)
            return value << -low;
        value >>= shift;
        if (shift > 4)
            value |= Digit(c, first + 5) << (60 - shift);
        if (firstDigit_[c] < first || (shift != 0 && (Digit(c, first) & ((1u << shift) - 1u)) != 0))
            value |= 1;
        return value;
    }
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage_;
    AscendC::LocalTensor<int32_t> sum_[2], source_, temporary_, coefficient_, table_;
    AscendC::LocalTensor<int16_t> short_;
    AscendC::GlobalTensor<int16_t> global_;
    int64_t extent_ = 0, first_ = 0, carry_[2]{}, correction_[2]{};
    int words_ = 0, digits_ = 0, pending_[2]{}, firstDigit_[2]{}, lastDigit_[2]{};
    bool scalarDirty_ = false, valid_[2]{};
};
class Math::Integer {
public:
    static constexpr bool kVector = true;
    __aicore__ inline Integer(Math& owner, int component) : owner_(owner), component_(component) {}
    __aicore__ inline void AddSource(float a, float b, int64_t, bool negative)
    {
        owner_.AddSource(component_, a, b, negative);
    }
    template <class Cache>
    __aicore__ inline void AddProduct(float coefficient, Cache&, int64_t offset, bool negative, int count, int first)
    {
        owner_.AddProduct(component_, coefficient, offset, negative, count, first);
    }
    __aicore__ inline bool Valid() { return owner_.Valid(component_); }
    __aicore__ inline bool MakeAbsolute() { return owner_.MakeAbsolute(component_); }
    __aicore__ inline uint32_t FloatBits(int64_t, bool negative) const
    {
        return owner_.FloatBits(component_, negative);
    }
    __aicore__ inline Packed Pack(int64_t cursor, bool negative) { return owner_.Pack(component_, cursor, negative); }
    __aicore__ inline int Count() const { return owner_.digits_; }
    __aicore__ inline uint32_t Word(int i) const { return static_cast<uint32_t>(owner_.sum_[component_].GetValue(i)); }

private:
    Math& owner_;
    int component_;
};
} // namespace spsv_unit_vector

#endif // SPSV_ARCH22_SPSV_UNIT_VECTOR_MATH_H_
