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

/*!
 * \file aclsparse_fp16_utils.h
 * \brief Host-side fp16 <-> fp32 conversion utilities for ops-sparse.
 *
 * These helpers are intentionally kept header-only and self-contained so that
 * both library code and unit tests can share the same bit-accurate conversion
 * routines without duplicating logic.
 */

#ifndef ACLSPARSE_FP16_UTILS_H
#define ACLSPARSE_FP16_UTILS_H

#include <cmath>
#include <cstdint>
#include <limits>

namespace aclsparse {

namespace {

// Type-punning helper that avoids std::memcpy, which some static analyzers
// flag as an unsafe function. The union approach is supported by the compilers
// used for CANN (Clang/GCC) and produces identical code for this same-size
// float <-> uint32_t conversion.
union FloatUint32 {
    float f;
    uint32_t u;
};

} // namespace

inline float Float16BitsToFloat32(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1u;
    int32_t exp = ((h >> 10) & 0x1Fu) - 15;
    uint32_t mant = h & 0x3FFu;

    if (exp == 16) {
        return mant ? std::numeric_limits<float>::quiet_NaN()
                    : (sign ? -std::numeric_limits<float>::infinity()
                            : std::numeric_limits<float>::infinity());
    }

    float val = 0.0f;
    if (exp == -15) {
        if (mant != 0) {
            val = std::ldexp(static_cast<float>(mant) / 1024.0f, -14);
        }
    } else {
        val = std::ldexp(1.0f + static_cast<float>(mant) / 1024.0f, exp);
    }
    return sign ? -val : val;
}

inline uint16_t Float32ToFloat16Bits(float f)
{
    FloatUint32 converter;
    converter.f = f;
    uint32_t x = converter.u;

    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp == 128) {
        // Keep NaNs distinct from infinity, including payloads lost by narrowing.
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }
    if (exp >= 16) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (exp < -25) {
        return static_cast<uint16_t>(sign);
    }

    uint32_t shift = 13;
    uint32_t half = 0;
    if (exp >= -14) {
        half = static_cast<uint32_t>(exp + 15) << 10;
    } else {
        // Restore the implicit bit before shifting into the half subnormal range.
        mant |= 0x800000u;
        shift = static_cast<uint32_t>(-exp - 1);
    }
    half |= mant >> shift;
    uint32_t remainder = mant & ((1u << shift) - 1u);
    uint32_t midpoint = 1u << (shift - 1);
    if (remainder > midpoint || (remainder == midpoint && (half & 1u))) {
        // Increment the full magnitude so a mantissa carry advances the exponent.
        ++half;
    }
    return static_cast<uint16_t>(sign | half);
}

} // namespace aclsparse

#endif // ACLSPARSE_FP16_UTILS_H
