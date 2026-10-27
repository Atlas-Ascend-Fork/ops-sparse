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

#ifndef SDDMM_ARCH35_SIMT_CONVERT_H_
#define SDDMM_ARCH35_SIMT_CONVERT_H_

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace sddmm_arch35 {

static __simt_callee__ __aicore__ inline uint16_t SddmmFloatToHalf(float value)
{
    uint32_t bits = 0U;
    __builtin_memcpy(&bits, &value, sizeof(float));
    const uint32_t sign = (bits >> 31) & 1U;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU);
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent == 0xFF) {
        const uint16_t payload = mantissa == 0U ? 0U : static_cast<uint16_t>(0x0200U | (mantissa >> 13));
        return static_cast<uint16_t>((sign << 15) | 0x7C00U | payload);
    }

    int32_t halfExponent = exponent - 127 + 15;
    if (halfExponent >= 0x1F) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00U);
    }
    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return static_cast<uint16_t>(sign << 15);
        }
        const uint32_t significand = mantissa | 0x800000U;
        const int32_t shift = 14 - halfExponent;
        uint32_t rounded = significand >> shift;
        const uint32_t mask = (1U << shift) - 1U;
        const uint32_t remainder = significand & mask;
        const uint32_t halfway = 1U << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) {
            ++rounded;
        }
        return static_cast<uint16_t>((sign << 15) | rounded);
    }

    const uint32_t roundingBias = 0x0FFFU + ((mantissa >> 13) & 1U);
    uint32_t roundedMantissa = (mantissa + roundingBias) >> 13;
    if (roundedMantissa > 0x3FFU) {
        roundedMantissa = 0U;
        ++halfExponent;
    }
    if (halfExponent >= 0x1F) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00U);
    }
    return static_cast<uint16_t>((sign << 15) |
        (static_cast<uint32_t>(halfExponent) << 10) | roundedMantissa);
}

static __simt_callee__ __aicore__ inline float SddmmBf16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

static __simt_callee__ __aicore__ inline uint16_t SddmmFloatToBf16(float value)
{
    uint32_t bits = 0U;
    __builtin_memcpy(&bits, &value, sizeof(float));
    if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0U) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040U);
    }
    const uint32_t lsb = (bits >> 16) & 1U;
    return static_cast<uint16_t>((bits + 0x7FFFU + lsb) >> 16);
}

} // namespace sddmm_arch35

#endif // SDDMM_ARCH35_SIMT_CONVERT_H_
