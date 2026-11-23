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
#ifndef SPSV_EXACT_H_
#define SPSV_EXACT_H_

#include "spsv_arith.h"

// Retry-only signed two's-complement accumulator. Cancellation preserves
// every exponent tier. A float product of three operands occupies
// bits -447..383; 1024 bits starting at -512 also cover an int64-sized row.
// One signed accumulator is reused for the two complex components (128 bytes).
constexpr int32_t kSpsvExactLimbs = 64;
constexpr int32_t kSpsvExactOrigin = -512;
struct SpsvExactSum {
    uint16_t digits[kSpsvExactLimbs];
};

__simt_callee__ inline uint32_t SpsvFloatMantissa(float v, int32_t& e, bool& negative)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    negative = (bits >> 31) != 0;
    uint32_t exponent = (bits >> 23) & 255;
    e = exponent == 0 ? -149 : static_cast<int32_t>(exponent) - 150;
    return (bits & 0x7fffffU) | (exponent == 0 ? 0 : 0x800000U);
}

__simt_callee__ inline void SpsvExactClear(SpsvExactSum& s)
{
    for (int32_t i = 0; i < kSpsvExactLimbs; ++i) {
        s.digits[i] = 0;
    }
}

__simt_callee__ inline void SpsvExactDigit(uint16_t* digits, uint32_t value, int32_t bit, bool subtract)
{
    int32_t index = bit / 16;
    uint32_t carry = value << (bit % 16);
    while (carry != 0 && index < kSpsvExactLimbs) {
        uint32_t part = carry & 65535U;
        if (subtract) {
            int32_t difference = static_cast<int32_t>(digits[index]) - static_cast<int32_t>(part);
            digits[index++] = static_cast<uint16_t>(difference);
            carry = (carry >> 16) + static_cast<uint32_t>(difference < 0);
        } else {
            uint32_t sum = digits[index] + part;
            digits[index++] = static_cast<uint16_t>(sum);
            carry = (carry >> 16) + (sum >> 16);
        }
    }
}

// The 72-bit integer product is formed without any floating point rounding,
// overflow or underflow. c=1 also gives an exact two-operand product.
__simt_callee__ inline void SpsvExactProduct(SpsvExactSum& s, float a, float b, float c, bool subtract)
{
    int32_t ea, eb, ec;
    bool na, nb, nc;
    uint32_t ma = SpsvFloatMantissa(a, ea, na);
    uint32_t mb = SpsvFloatMantissa(b, eb, nb);
    uint32_t mc = SpsvFloatMantissa(c, ec, nc);
    uint64_t ab = static_cast<uint64_t>(ma) * mb;
    uint64_t low = (ab & 0xffffffffULL) * mc;
    uint64_t high = (ab >> 32) * mc + (low >> 32);
    bool negative = na ^ nb ^ nc ^ subtract;
    int32_t bit = ea + eb + ec - kSpsvExactOrigin;
    SpsvExactDigit(s.digits, static_cast<uint32_t>(low) & 65535U, bit, negative);
    SpsvExactDigit(s.digits, static_cast<uint32_t>(low >> 16) & 65535U, bit + 16, negative);
    SpsvExactDigit(s.digits, static_cast<uint32_t>(high) & 65535U, bit + 32, negative);
    SpsvExactDigit(s.digits, static_cast<uint32_t>(high >> 16) & 65535U, bit + 48, negative);
    SpsvExactDigit(s.digits, static_cast<uint32_t>(high >> 32), bit + 64, negative);
}

// Cancel in integer space BEFORE rounding to a float-float significand.
// The independent exponent is retained through division and final rescaling.
__simt_callee__ inline SpsvDFExp SpsvExactFinish(SpsvExactSum& s)
{
    int32_t top = kSpsvExactLimbs - 1;
    while (top >= 0 && s.digits[top] == 0) {
        --top;
    }
    if (top < 0) {
        return {{0.0f, 0.0f}, 0};
    }
    bool negative = (s.digits[kSpsvExactLimbs - 1] & 0x8000U) != 0;
    if (negative) {
        uint32_t carry = 1;
        for (int32_t i = 0; i < kSpsvExactLimbs; ++i) {
            uint32_t digit = static_cast<uint32_t>(~s.digits[i] & 65535U) + carry;
            s.digits[i] = static_cast<uint16_t>(digit);
            carry = digit >> 16;
        }
        top = kSpsvExactLimbs - 1;
        while (top >= 0 && s.digits[top] == 0) {
            --top;
        }
    }
    SpsvDF mantissa{static_cast<float>(s.digits[top]), 0.0f};
    for (int32_t i = 1; i <= 4 && top >= i; ++i) {
        mantissa = SpsvDFAdd(mantissa, {ldexpf(static_cast<float>(s.digits[top - i]), -16 * i), 0.0f});
    }
    if (negative) {
        mantissa = SpsvDFNeg(mantissa);
    }
    return {mantissa, kSpsvExactOrigin + 16 * top};
}

__simt_callee__ inline SpsvDF SpsvExactQuotient(SpsvDFExp numerator, SpsvDFExp denominator)
{
    float q = numerator.m.hi / denominator.m.hi;
    float residual = fmaf(-q, denominator.m.hi, numerator.m.hi) + numerator.m.lo - q * denominator.m.lo;
    SpsvDF result = SpsvTwoSum(q, residual / denominator.m.hi);
    return SpsvDFScale2(result, numerator.e - denominator.e);
}

#endif // SPSV_EXACT_H_
