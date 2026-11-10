/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "device_buffer.h"

// These checks exercise portable scalar arithmetic, independently of launch.
#define SPSV_DD_AICORE
#include "spsv_dd_math.h"
#undef SPSV_DD_AICORE
#include "spsv_binary64.h"
#define SPSV_EXACT_SUM_FN inline
#include "spsv_exact_sum.h"
#undef SPSV_EXACT_SUM_FN
#include "spsv_expansion64.h"
#include "spsv_tiling_data.h"
#define SPSV_UNIT_BOUND_FN inline
#define SPSV_UNIT_INTEGER_FN inline
#define SPSV_UNIT_PRECISION_AICORE
#include "spsv_unit_precision.h"
#undef SPSV_UNIT_PRECISION_AICORE
#undef SPSV_UNIT_INTEGER_FN
#undef SPSV_UNIT_BOUND_FN

namespace {
TEST(SpSVHostExponentOnly, ScaleFloatHandlesFullShiftDomain)
{
    for (float value : {1.0f, -1.0f, std::numeric_limits<float>::denorm_min(),
                        -std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max()}) {
        auto large = spsv_scaled::ScaleFloat(value, std::numeric_limits<int64_t>::max());
        auto small = spsv_scaled::ScaleFloat(value, std::numeric_limits<int64_t>::min());
        EXPECT_TRUE(std::isinf(large));
        EXPECT_EQ(std::signbit(large), std::signbit(value));
        EXPECT_EQ(small, 0.0f);
        EXPECT_EQ(std::signbit(small), std::signbit(value));
        EXPECT_EQ(spsv_scaled::ScaleFloat(value, 0), value);
    }
    EXPECT_TRUE(std::isnan(
        spsv_scaled::ScaleFloat(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<int64_t>::max())));
    EXPECT_EQ(spsv_scaled::ScaleFloat(-std::numeric_limits<float>::infinity(), std::numeric_limits<int64_t>::min()),
              -std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::signbit(spsv_scaled::ScaleFloat(-0.0f, std::numeric_limits<int64_t>::max())));
}

void VerifyExponentLimit(float sign)
{
    const auto limit = spsv_scaled::kExponentLimit;
    const auto inside = spsv_scaled::Normalize({sign, 0}, limit);
    const auto below = spsv_scaled::Normalize({sign, 0}, -static_cast<int64_t>(limit));
    EXPECT_EQ(inside.exponent, limit);
    EXPECT_EQ(inside.fraction.hi, sign);
    EXPECT_EQ(below.exponent, -limit);
    EXPECT_EQ(below.fraction.hi, sign);
    for (int64_t exponent : {static_cast<int64_t>(limit) + 1, std::numeric_limits<int64_t>::max()}) {
        const auto result = spsv_scaled::Normalize({sign, 0}, exponent);
        EXPECT_TRUE(std::isinf(result.fraction.hi));
        EXPECT_EQ(result.exponent, 0);
        EXPECT_EQ(std::signbit(result.fraction.hi), std::signbit(sign));
    }
    for (int64_t exponent : {-static_cast<int64_t>(limit) - 1, std::numeric_limits<int64_t>::min()}) {
        const auto result = spsv_scaled::Normalize({sign, 0}, exponent);
        EXPECT_EQ(result.fraction.hi, 0.0f);
        EXPECT_EQ(result.exponent, 0);
        EXPECT_EQ(std::signbit(result.fraction.hi), std::signbit(sign));
    }
}

TEST(SpSVHostExponentOnly, ArithmeticPromotesBeforeCombiningExponents)
{
    static_assert(sizeof(spsv_scaled::Real) == 12, "The legacy scaled scalar contains two words and an exponent");
    VerifyExponentLimit(1.0f);
    VerifyExponentLimit(-1.0f);
    const spsv_scaled::Real large{{1, 0}, std::numeric_limits<int>::max()};
    const spsv_scaled::Real small{{1, 0}, std::numeric_limits<int>::min()};
    EXPECT_TRUE(std::isinf(spsv_scaled::Mul(large, spsv_scaled::FromFloat(2)).fraction.hi));
    EXPECT_EQ(spsv_scaled::Div(small, spsv_scaled::FromFloat(2)).fraction.hi, 0.0f);
    EXPECT_TRUE(std::isinf(spsv_scaled::Add(large, small).fraction.hi));
    EXPECT_EQ(spsv_scaled::Round(small), 0.0f);
}
uint64_t WideBits(double value) { return spsv_test::BitCopy<uint64_t>(value); }

void ExpectExtendedValue(spsv_extended64::Real actual, double expected)
{
    const auto wanted = spsv_extended64::Normalize({WideBits(expected)}, 0);
    EXPECT_EQ(actual.fraction.bits, wanted.fraction.bits);
    EXPECT_EQ(actual.exponent, wanted.exponent);
}

TEST(SpSVHostExponentOnly, ExtendedBinary64MatchesNativeInNormalRange)
{
    std::mt19937_64 generator(0x535056455854ULL);
    for (int sample = 0; sample < 4096; ++sample) {
        const uint64_t aBits = (generator() & 0x800fffffffffffffULL) |
                               (static_cast<uint64_t>(923 + generator() % 201) << 52);
        const uint64_t bBits = (generator() & 0x800fffffffffffffULL) |
                               (static_cast<uint64_t>(923 + generator() % 201) << 52);
        const double a = spsv_test::BitCopy<double>(aBits);
        const double b = spsv_test::BitCopy<double>(bBits);
        const auto left = spsv_extended64::Normalize({aBits}, 0);
        const auto right = spsv_extended64::Normalize({bBits}, 0);
        SCOPED_TRACE("extended_sample=" + std::to_string(sample));
        ExpectExtendedValue(spsv_extended64::Add(left, right), a + b);
        ExpectExtendedValue(spsv_extended64::Sub(left, right), a - b);
        ExpectExtendedValue(spsv_extended64::Mul(left, right), a * b);
        ExpectExtendedValue(spsv_extended64::Div(left, right), a / b);
    }
}

TEST(SpSVHostExponentOnly, ExtendedBinary64RetainsVirtualExponent)
{
    auto value = spsv_extended64::FromFloat(1);
    for (int step = 0; step < 11; ++step)
        value = spsv_extended64::Div(value, spsv_extended64::FromFloat(0x1p100f));
    EXPECT_EQ(value.exponent, -1100);
    EXPECT_EQ(value.fraction.bits, WideBits(1));
    EXPECT_EQ(spsv_extended64::ToFloatBits(value), 0u);
    for (int step = 0; step < 11; ++step)
        value = spsv_extended64::Div(value, spsv_extended64::FromFloat(0x1p-100f));
    ExpectExtendedValue(value, 1);
    EXPECT_EQ(spsv_extended64::ToFloatBits(value), 0x3f800000u);
    const auto large = spsv_extended64::Normalize({WideBits(-2)}, std::numeric_limits<int64_t>::max());
    EXPECT_EQ(large.fraction.bits, WideBits(-std::numeric_limits<double>::infinity()));
    const auto small = spsv_extended64::Normalize({WideBits(-0x1p-1022)}, std::numeric_limits<int64_t>::min());
    EXPECT_EQ(small.fraction.bits, WideBits(-0.0));
}
TEST(SpSVHostExponentOnly, ExpansionPreservesSeparatedTail)
{
    namespace base = spsv_extended64;
    namespace expansion = spsv_expansion64;
    for (const int exponent : {-5000, -1100, -125, -110, -101, -100, -99, -54, -25}) {
        for (const float sign : {-1.0f, 1.0f}) {
            const auto tail = base::Normalize({WideBits(sign)}, exponent);
            const auto value = expansion::Normalize(base::FromFloat(1), tail);
            const auto recovered = expansion::Normalize(base::Sub(value.high, base::FromFloat(1)), value.low);
            EXPECT_EQ(recovered.high.fraction.bits, tail.fraction.bits);
            EXPECT_EQ(recovered.high.exponent, tail.exponent);
            EXPECT_TRUE(spsv_soft64::IsZero(recovered.low.fraction));
        }
    }
}

TEST(SpSVHostExponentOnly, ExpansionTailDecidesFloatMidpoint)
{
    namespace base = spsv_extended64;
    namespace expansion = spsv_expansion64;
    for (const double midpoint : {0x1p-150, 1.0 + 0x1p-24, 0x1.fffffep127 + 0x1p103}) {
        const auto leading = base::Normalize({WideBits(midpoint)}, 0);
        for (const float sign : {-1.0f, 1.0f}) {
            const auto tail = base::Normalize({WideBits(1)}, leading.exponent - 200);
            const auto positive = expansion::Normalize(leading, sign > 0 ? tail : base::Negate(tail));
            const auto negative = expansion::Normalize(base::Negate(leading), sign > 0 ? base::Negate(tail) : tail);
            const float wanted = static_cast<float>(std::nextafter(midpoint, sign > 0 ? INFINITY : 0));
            EXPECT_EQ(expansion::ToFloatBits(positive), spsv_test::BitCopy<uint32_t>(wanted));
            EXPECT_EQ(expansion::ToFloatBits(negative), spsv_test::BitCopy<uint32_t>(-wanted));
        }
    }
}
struct UnitWordSpan {
    const uint32_t* words;
    float GetValue(uint32_t index) const { return spsv_test::BitCopy<float>(words[index]); }
};
struct UnitWordSource {
    std::vector<uint32_t> words;
    uint32_t window;
    UnitWordSpan ReadSpan(int64_t index, uint32_t& available)
    {
        available = std::min(window - static_cast<uint32_t>(index % window),
                             static_cast<uint32_t>(words.size() - index));
        return {words.data() + index};
    }
};
void TrimUnitSource(const UnitWordSource& source, bool packed, int& begin, int& end)
{
    if (!packed)
        return;
    while (begin < end && source.words[begin] == 0)
        ++begin;
    while (end > begin && source.words[end - 1] == 0)
        --end;
}
void VerifyStreamProduct(float coefficient, std::initializer_list<uint32_t> input,
                         std::initializer_list<uint32_t> wanted, bool negative, int initial = 0)
{
    std::vector<uint32_t> expected(wanted);
    expected.resize(8);
    for (const uint32_t window : {1u, 3u, 8u}) {
        SCOPED_TRACE("window=" + std::to_string(window));
        UnitWordSource source{std::vector<uint32_t>(input), window};
        source.words.resize(8);
        for (const bool packed : {false, true}) {
            int begin = 0, end = 8;
            TrimUnitSource(source, packed, begin, end);
            spsv_wide::UnitInteger sum(8);
            sum.AddSource(initial, 1, 0, negative);
            sum.AddProduct(coefficient, source, begin, negative, end - begin, begin);
            ASSERT_TRUE(sum.Valid());
            EXPECT_EQ(sum.MakeAbsolute(), negative != std::signbit(coefficient));
            for (int i = 0; i < 8; ++i)
                EXPECT_EQ(sum.Word(i), expected[i]);
        }
    }
}
TEST(SpSVHostExponentOnly, UnitIntegerStreamsCarriesAndShiftedProducts)
{
    for (const bool negative : {false, true}) {
        VerifyStreamProduct(1, {0xffffffffu, 0xffffffffu}, {0, 0, 1}, negative, 1);
        VerifyStreamProduct(1.5f, {0xfffffffeu, 0xffffffffu}, {0, 0x80000000u, 1}, negative, 3);
        VerifyStreamProduct(0x1p-40f, {0xffffffffu, 0xffffffffu}, {0xffffffu}, negative);
        VerifyStreamProduct(0x1p-149f, {0, 0, 0, 0, 0x00200000u}, {1}, negative);
        VerifyStreamProduct(0x1p100f, {0xffffffffu, 0xffffffffu}, {0, 0, 0, 0xfffffff0u, 0xffffffffu, 0xf}, negative);
        VerifyStreamProduct(-1, {7}, {7}, negative);
    }
}
TEST(SpSVHostExponentOnly, EmptyIntegerSpanDoesNotReadOrChangeTheAccumulator)
{
    UnitWordSource source{std::vector<uint32_t>(8), 3};
    spsv_wide::UnitInteger sum(4);
    sum.AddSource(1, 1, 0, false);
    sum.AddProduct(1, source, 8, true, 0, 8);
    ASSERT_TRUE(sum.Valid());
    EXPECT_FALSE(sum.MakeAbsolute());
    EXPECT_EQ(sum.Word(0), 1u);
    for (int i = 1; i < 4; ++i)
        EXPECT_EQ(sum.Word(i), 0u);
}

TEST(SpSVHostExponentOnly, PositiveBoundsEncloseFiniteProductsAndSums)
{
    namespace bound = spsv_unit_bound;
    const float values[] = {0,
                            std::numeric_limits<float>::denorm_min(),
                            0x1p-126f,
                            0x1p-25f,
                            0x1p-24f,
                            0.1f,
                            1,
                            std::nextafter(2.0f, 0.0f),
                            std::numeric_limits<float>::max()};
    for (float a : values) {
        const auto first = bound::Positive(a);
        EXPECT_EQ(std::ldexp(static_cast<long double>(first.fraction), first.exponent), a);
        for (float b : values) {
            const auto sum = bound::Add(first, bound::Positive(b));
            const auto product = bound::Mul(first, bound::Positive(b));
            EXPECT_GE(std::ldexp(static_cast<long double>(sum.fraction), sum.exponent),
                      static_cast<long double>(a) + b);
            EXPECT_GE(std::ldexp(static_cast<long double>(product.fraction), product.exponent),
                      static_cast<long double>(a) * b);
        }
    }
}

TEST(SpSVHostExponentOnly, SeriesCopiesOnlyItsInitializedPrefix)
{
    namespace series = spsv_series64;
    for (int count : {0, 1, 2, 19, 20}) {
        series::Real source;
        for (int i = 0; i < count; ++i)
            series::Append(source, spsv_extended64::Normalize({WideBits(1)}, -110 * i));
        series::Real copy(source);
        auto assigned = series::FromBase(spsv_extended64::FromFloat(-1));
        assigned = copy;
        EXPECT_EQ(assigned.count, count);
        for (int i = 0; i < count; ++i) {
            EXPECT_EQ(assigned.parts[i].fraction.bits, source.parts[i].fraction.bits);
            EXPECT_EQ(assigned.parts[i].exponent, source.parts[i].exponent);
        }
        EXPECT_TRUE(spsv_soft64::IsZero(series::At(assigned, count).fraction));
        assigned = series::Real{};
        EXPECT_EQ(assigned.count, 0);
        EXPECT_TRUE(spsv_soft64::IsZero(series::At(assigned, 0).fraction));
    }
}

TEST(SpSVHostExponentOnly, UnitIntegerConversionCoversWordAndExponentBoundaries)
{
    for (const int highest : {0, 1, 23, 31, 32, 54, 55, 56, 63}) {
        const uint64_t value = (uint64_t{1} << highest) | 1;
        for (const bool negative : {false, true}) {
            for (const int first : {-300, -151, -150, -149, -127, -126, -103, 0, 64, 128}) {
                spsv_wide::UnitInteger sum(8);
                sum.AddCoefficient(1, static_cast<uint32_t>(value), 0, negative);
                sum.AddCoefficient(1, static_cast<uint32_t>(value >> 32), 1, negative);
                ASSERT_TRUE(sum.Valid());
                const bool sign = sum.MakeAbsolute();
                const long double magnitude = std::ldexp(static_cast<long double>(value), first);
                const float expected = static_cast<float>(negative ? -magnitude : magnitude);
                EXPECT_EQ(sum.FloatBits(first, sign), spsv_test::BitCopy<uint32_t>(expected));
            }
        }
    }
}

void VerifyUnitMidpoint(int kind, bool negative, bool positiveTail)
{
    constexpr int first = -298;
    spsv_wide::UnitInteger sum(32);
    if (kind == 0) {
        sum.AddSource(1, 1, first, negative);
        sum.AddSource(0x1p-24f, 1, first, negative);
    } else if (kind == 1) {
        sum.AddSource(0x1p-75f, 0x1p-75f, first, negative);
    } else {
        sum.AddSource(std::numeric_limits<float>::max(), 1, first, negative);
        sum.AddSource(0x1p50f, 0x1p53f, first, negative);
    }
    const float tiny = std::numeric_limits<float>::denorm_min();
    sum.AddSource(tiny, tiny, first, !positiveTail);
    const bool away = positiveTail != negative;
    float expected = kind == 0 ? (away ? 1 + 0x1p-23f : 1) :
                     kind == 1 ? (away ? tiny : 0) :
                                 (away ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::max());
    if (negative)
        expected = -expected;
    ASSERT_TRUE(sum.Valid());
    const bool sign = sum.MakeAbsolute();
    EXPECT_EQ(sum.FloatBits(first, sign), spsv_test::BitCopy<uint32_t>(expected));
}
TEST(SpSVHostExponentOnly, UnitIntegerStickyTailDecidesFloatMidpoint)
{
    for (int kind = 0; kind < 3; ++kind)
        for (bool negative : {false, true})
            for (bool positiveTail : {false, true})
                VerifyUnitMidpoint(kind, negative, positiveTail);
}
TEST(SpSVHostExponentOnly, StableExpansionPrefixHonorsDirectionalSpacingAndTies)
{
    namespace base = spsv_extended64;
    const auto one = base::FromFloat(1);
    const base::Real odd{{base::kUnitExponent | 1}, 0};
    const base::Real half{{base::kUnitExponent}, -53};
    const base::Real quarter{{base::kUnitExponent}, -54};
    const auto aboveQuarter = base::Normalize(spsv_soft64::FromFloatBits(0x3fc00000u), -54);
    EXPECT_TRUE(spsv_series64::StablePair(one, half));
    EXPECT_TRUE(spsv_series64::StablePair(one, base::Negate(quarter)));
    EXPECT_FALSE(spsv_series64::StablePair(one, base::Negate(aboveQuarter)));
    EXPECT_TRUE(spsv_series64::StablePair(one, aboveQuarter));
    EXPECT_FALSE(spsv_series64::StablePair(odd, half));
    EXPECT_FALSE(spsv_series64::StablePair(odd, base::Negate(half)));
    EXPECT_TRUE(spsv_series64::StablePair(base::Negate(one), quarter));
    EXPECT_FALSE(spsv_series64::StablePair(base::Negate(one), aboveQuarter));
    EXPECT_TRUE(spsv_series64::StablePair(base::Negate(one), base::Negate(aboveQuarter)));
}

template <int Words>
void VerifySignedWindowRounding(int highest)
{
    for (int mask = 0; mask < 8; ++mask) {
        const bool negative = (mask & 1) != 0;
        const bool odd = (mask & 2) != 0;
        const bool sticky = (mask & 4) != 0;
        spsv_scaled::ExactAccumulator<Words, -298> sum;
        sum.AddRawWord(highest / 32, 1u << (highest % 32), negative);
        if (odd)
            sum.AddRawWord((highest - 52) / 32, 1u << ((highest - 52) % 32), negative);
        sum.AddRawWord((highest - 53) / 32, 1u << ((highest - 53) % 32), negative);
        if (sticky)
            sum.AddRawWord(0, 1, negative);
        const uint64_t expected = spsv_extended64::kUnitExponent | (negative ? spsv_soft64::kSign : 0) |
                                  (odd ? 2 : (sticky ? 1 : 0));
        for (int repeat = 0; repeat < 2; ++repeat) {
            int actualHighest;
            EXPECT_EQ(sum.RoundedNormalized64(actualHighest), expected);
            EXPECT_EQ(actualHighest, highest);
        }
    }
}
TEST(SpSVHostExponentOnly, SignedAccumulatorWindowRetainsStickyBitsWithoutConsumption)
{
    for (const int highest : {55, 56, 63, 64, 95, 127, 318, 638})
        VerifySignedWindowRounding<20>(highest);
    VerifySignedWindowRounding<64>(2046);
    VerifySignedWindowRounding<256>(8190);
}
} // namespace
