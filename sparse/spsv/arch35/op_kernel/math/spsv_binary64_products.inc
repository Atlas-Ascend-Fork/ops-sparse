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

// Included inside SPSV_SOFT_NAMESPACE by spsv_binary64.h. Deliberately has no
// include guard: the same translation unit instantiates scalar and SIMT helpers.
SPSV_SOFT __attribute__((always_inline)) void Unpack32(uint32_t a, int& exponent, uint64_t& significand)
{
    exponent = static_cast<int>((a >> 23) & 255U);
    significand = a & 0x7fffffU;
    if (exponent != 0) {
        significand |= 1U << 23;
    } else {
        exponent = 1;
        while (significand < (1U << 23)) {
            significand <<= 1;
            --exponent;
        }
    }
}
struct ExtendedProduct32 {
    uint64_t bits;
    int extra;
};
SPSV_SOFT __attribute__((always_inline)) ExtendedProduct32 Multiply32Significands(uint64_t ma, uint64_t mb)
{
    uint64_t first = static_cast<uint32_t>(mb) * ma;
    uint64_t middle = (mb >> 32) * ma + (first >> 32);
    uint64_t high = middle >> 32, low = (middle << 32) | static_cast<uint32_t>(first);
    int extra = (high & (1ULL << 12)) != 0, shift = 20 + extra;
    uint64_t extended = (high << (64 - shift)) | (low >> shift);
    extended |= (low << (64 - shift)) != 0;
    return {extended, extra};
}
SPSV_SOFT __attribute__((always_inline)) bool TryMultiply32Special(uint32_t a, uint64_t b, uint64_t& sign,
                                                                    uint64_t& result)
{
    sign = (static_cast<uint64_t>(a & 0x80000000U) << 32) ^ (b & Sign);
    uint32_t magnitude = a & 0x7fffffffU;
    if (magnitude > 0x7f800000U || IsNaN(b)) {
        result = NaN;
        return true;
    }
    if (magnitude == 0x7f800000U || IsInf(b)) {
        result = (magnitude == 0 || IsZero(b)) ? NaN : sign | Infinity;
        return true;
    }
    if (magnitude == 0 || IsZero(b)) {
        result = sign;
        return true;
    }
    return false;
}
template <bool NORMAL_PACK>
SPSV_SOFT __attribute__((always_inline)) uint64_t Multiply32(uint32_t a, uint64_t b)
{
    uint64_t sign, special;
    if (TryMultiply32Special(a, b, sign, special)) {
        return special;
    }
    int ea, eb;
    uint64_t ma, mb;
    Unpack32(a, ea, ma);
    if constexpr (NORMAL_PACK) {
        eb = static_cast<int>((b >> 52) & 2047U);
        mb = (b & Fraction) | Hidden;
        if (eb == 0) {
            Unpack(b, eb, mb);
        }
    } else {
        Unpack(b, eb, mb);
    }
    auto product = Multiply32Significands(ma, mb);
    int exponent = ea + eb - 127 + product.extra;
    if constexpr (NORMAL_PACK) {
        if (exponent > 0 && exponent < 2047) {
            uint64_t significand = product.bits >> 3;
            uint32_t rounding = static_cast<uint32_t>(product.bits & 7U);
            significand += rounding > 4U || (rounding == 4U && (significand & 1U));
            return sign | ((static_cast<uint64_t>(exponent - 1) << 52) + significand);
        }
    }
    return Pack(sign, exponent, product.bits);
}
SPSV_SOFT __attribute__((always_inline)) uint64_t MultiplyNormal32(uint32_t a, uint64_t b)
{
    uint64_t sign = (static_cast<uint64_t>(a & 0x80000000U) << 32) ^ (b & Sign);
    auto product = Multiply32Significands((a & 0x7fffffU) | (1U << 23), (b & Fraction) | Hidden);
    int exponent = static_cast<int>((a >> 23) & 255U) + static_cast<int>((b >> 52) & 2047U) - 127 + product.extra;
    uint64_t significand = (product.bits + 3U + ((product.bits >> 3) & 1U)) >> 3;
    return sign | ((static_cast<uint64_t>(exponent - 1) << 52) + significand);
}

