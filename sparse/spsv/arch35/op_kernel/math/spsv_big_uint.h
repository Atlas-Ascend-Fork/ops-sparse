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

#ifndef SPSV_BIG_UINT_H_
#define SPSV_BIG_UINT_H_

#include "spsv_exact.h"

struct SpsvBigUInt {
    __gm__ uint16_t* digit;
    int64_t size;
};

__simt_callee__ inline int32_t SpsvUIntBits(uint32_t v)
{
    int32_t bits = 0;
    while (v != 0) {
        ++bits;
        v >>= 1;
    }
    return bits;
}

__simt_callee__ inline int64_t SpsvBigBits(const SpsvBigUInt& a)
{
    return a.size == 0 ? 0 : 16 * (a.size - 1) + SpsvUIntBits(a.digit[a.size - 1]);
}

__simt_callee__ inline void SpsvBigSet(SpsvBigUInt& a, uint16_t value)
{
    a.size = value == 0 ? 0 : 1;
    a.digit[0] = value;
}

__simt_callee__ inline void SpsvBigTrim(SpsvBigUInt& a)
{
    while (a.size > 0 && a.digit[a.size - 1] == 0) {
        --a.size;
    }
}

__simt_callee__ inline void SpsvBigMul(SpsvBigUInt& a, uint32_t v)
{
    uint64_t carry = 0;
    for (int64_t i = 0; i < a.size; ++i) {
        uint64_t product = static_cast<uint64_t>(a.digit[i]) * v + carry;
        a.digit[i] = static_cast<uint16_t>(product);
        carry = product >> 16;
    }
    while (carry != 0) {
        a.digit[a.size++] = static_cast<uint16_t>(carry);
        carry >>= 16;
    }
    SpsvBigTrim(a);
}

__simt_callee__ inline void SpsvBigShift(SpsvBigUInt& a, int64_t bits)
{
    if (a.size == 0 || bits == 0) {
        return;
    }
    int64_t words = bits / 16;
    int32_t shift = static_cast<int32_t>(bits % 16);
    int64_t oldSize = a.size;
    for (int64_t i = oldSize; i-- > 0;) {
        a.digit[i + words] = a.digit[i];
    }
    for (int64_t i = 0; i < words; ++i) {
        a.digit[i] = 0;
    }
    a.size += words;
    if (shift != 0) {
        uint32_t carry = 0;
        for (int64_t i = words; i < a.size; ++i) {
            uint32_t value = (static_cast<uint32_t>(a.digit[i]) << shift) | carry;
            a.digit[i] = static_cast<uint16_t>(value);
            carry = value >> 16;
        }
        if (carry != 0) {
            a.digit[a.size++] = static_cast<uint16_t>(carry);
        }
    }
}

__simt_callee__ inline uint32_t SpsvBigMod(const SpsvBigUInt& a, uint32_t p)
{
    if (p < 2) {
        return 0;
    }
    uint32_t remainder = 0;
    for (int64_t i = a.size; i-- > 0;) {
        if (p <= 65535U) {
            remainder = ((remainder << 16) + a.digit[i]) % p;
        } else {
            // Keep division in the native 32-bit domain. For p < 2^31,
            // doubling a residue cannot overflow uint32_t.
            for (int32_t bit = 15; bit >= 0; --bit) {
                uint32_t doubled = remainder + remainder;
                remainder = doubled >= p ? doubled - p : doubled;
                remainder += (a.digit[i] >> bit) & 1U;
                if (remainder == p) {
                    remainder = 0;
                }
            }
        }
    }
    return remainder;
}

// X += M*t, with 0 <= X < M and t < p, before replacing M by M*p.
__simt_callee__ inline void SpsvBigCrtAdd(SpsvBigUInt& x, const SpsvBigUInt& m, uint32_t t)
{
    int64_t oldSize = x.size;
    uint64_t carry = 0;
    for (int64_t i = 0; i < m.size || carry != 0; ++i) {
        uint64_t value = carry + (i < oldSize ? x.digit[i] : 0);
        if (i < m.size) {
            value += static_cast<uint64_t>(m.digit[i]) * t;
        }
        x.digit[i] = static_cast<uint16_t>(value);
        carry = value >> 16;
        x.size = i + 1;
    }
    SpsvBigTrim(x);
}

__simt_callee__ inline uint16_t SpsvBigShiftDigit(const SpsvBigUInt& a, int64_t at, int64_t shift)
{
    int64_t i = at - shift / 16;
    int32_t bit = static_cast<int32_t>(shift % 16);
    uint32_t value = i >= 0 && i < a.size ? static_cast<uint32_t>(a.digit[i]) << bit : 0;
    if (bit != 0 && i > 0 && i - 1 < a.size) {
        value |= a.digit[i - 1] >> (16 - bit);
    }
    return static_cast<uint16_t>(value);
}

__simt_callee__ inline int32_t SpsvBigCompareShift(const SpsvBigUInt& a, const SpsvBigUInt& b, int64_t shift)
{
    int64_t ab = SpsvBigBits(a), bb = SpsvBigBits(b) + shift;
    if (ab != bb) {
        return ab < bb ? -1 : 1;
    }
    for (int64_t i = a.size; i-- > 0;) {
        uint16_t other = SpsvBigShiftDigit(b, i, shift);
        if (a.digit[i] != other) {
            return a.digit[i] < other ? -1 : 1;
        }
    }
    return 0;
}

__simt_callee__ inline void SpsvBigSubtractShift(SpsvBigUInt& a, const SpsvBigUInt& b, int64_t shift)
{
    int32_t borrow = 0;
    for (int64_t i = 0; i < a.size; ++i) {
        int32_t value = static_cast<int32_t>(a.digit[i]) - SpsvBigShiftDigit(b, i, shift) - borrow;
        a.digit[i] = static_cast<uint16_t>(value);
        borrow = value < 0;
    }
    SpsvBigTrim(a);
}

// Convert the negative CRT representative X-M into magnitude M-X in place.
__simt_callee__ inline void SpsvBigComplement(SpsvBigUInt& x, const SpsvBigUInt& m)
{
    int64_t oldSize = x.size;
    int32_t borrow = 0;
    for (int64_t i = 0; i < m.size; ++i) {
        int32_t value = static_cast<int32_t>(m.digit[i]) - (i < oldSize ? x.digit[i] : 0) - borrow;
        x.digit[i] = static_cast<uint16_t>(value);
        borrow = value < 0;
    }
    x.size = m.size;
    SpsvBigTrim(x);
}

// Correctly rounded binary32 N/D. Only 24 quotient bits are needed. Form
// subnormals by their bit patterns so SIMT FTZ cannot discard the result.
__simt_callee__ inline float SpsvRationalRound(SpsvBigUInt& n, const SpsvBigUInt& d, bool negative)
{
    uint32_t sign = negative ? 0x80000000U : 0U;
    uint32_t bits = sign;
    if (n.size != 0) {
        int64_t exponent = SpsvBigBits(n) - SpsvBigBits(d);
        if (exponent >= 0) {
            if (SpsvBigCompareShift(n, d, exponent) < 0) {
                --exponent;
            }
        } else if (SpsvBigCompareShift(d, n, -exponent) > 0) {
            --exponent;
        }
        if (exponent > 127) {
            bits |= 0x7f800000U;
        } else if (exponent >= -150) {
            int64_t shift = exponent < -126 ? -149 : exponent - 23;
            if (shift < 0) {
                SpsvBigShift(n, -shift);
                shift = 0;
            }
            uint32_t q = 0;
            for (int32_t k = 24; k >= 0; --k) {
                if (SpsvBigCompareShift(n, d, shift + k) >= 0) {
                    SpsvBigSubtractShift(n, d, shift + k);
                    q |= 1U << k;
                }
            }
            SpsvBigShift(n, 1);
            int32_t halfway = SpsvBigCompareShift(n, d, shift);
            q += halfway > 0 || (halfway == 0 && (q & 1U) != 0);
            if (exponent < -126) {
                bits |= q;
            } else {
                if (q >= 0x1000000U) {
                    q >>= 1;
                    ++exponent;
                }
                bits |= exponent > 127 ? 0x7f800000U
                                       : (static_cast<uint32_t>(exponent + 127) << 23) | (q & 0x7fffffU);
            }
        }
    }
    float answer;
    __builtin_memcpy(&answer, &bits, sizeof(answer));
    return answer;
}

#endif  // SPSV_BIG_UINT_H_
