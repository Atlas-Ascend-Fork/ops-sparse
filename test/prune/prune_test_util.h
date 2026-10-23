/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_PRUNE_TEST_UTIL_H_
#define TEST_PRUNE_TEST_UTIL_H_

// =============================================================================
// Shared test utilities for prune test (decoupled from alg_set_attribute /
// matmul test headers). Contains:
//   - FP16/BF16 bit-pattern <-> FP32 conversion helpers
//   - AlgSetAttrDtypeTrait (all dtype specializations: float / uint16_t /
//     bf16_bits_t / int8_t)
//   - Data generation: GenFp32Matrix / ToStorage* / GenTestMatrix /
//     MatmulGenTestMatrix
//   - Structured pruning primitives: PickLargestAbs / PruneRowFp32 /
//     PruneColFp32
//
// Extracted from the source repo's alg_set_attribute_golden.h and
// matmul_golden_types.h so that prune tests are self-contained (no cross-
// operator includes).
// =============================================================================

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace sparse_test {

// -----------------------------------------------------------------------------
// FP32 -> FP16 bit pattern (IEEE754 round-to-nearest-even).
// -----------------------------------------------------------------------------
inline uint16_t Fp32ToFp16Bits(float v)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t rawExp = (bits >> 23) & 0xFFu;
    int32_t exp = static_cast<int32_t>(rawExp) - 127;
    uint32_t mant = bits & 0x007FFFFFu;

    // NaN / Inf: preserve NaN (mantissa nonzero) and Inf semantics.
    if (rawExp == 0xFFu) {
        uint32_t nanBit = (mant != 0u) ? 0x0200u : 0u;
        return static_cast<uint16_t>(sign | 0x7C00u | nanBit);
    }

    // Overflow to Inf.
    if (exp >= 16) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    // Normalized fp16: exp in [-14, 15].
    if (exp >= -14) {
        uint32_t mant10 = mant >> 13;          // high 10 mantissa bits
        uint32_t remainder = mant & 0x1FFFu;   // low 13 bits rounding residue
        uint32_t result = (static_cast<uint32_t>(exp + 15) << 10) | mant10;
        if (remainder > 0x1000u || (remainder == 0x1000u && (mant10 & 1u))) {
            result += 1u;  // carry may naturally overflow into exponent (incl. norm->Inf)
        }
        return static_cast<uint16_t>(sign | result);
    }

    // Subnormal fp16: exp in [-25, -15], otherwise underflow to 0.
    if (exp >= -25) {
        uint32_t significand = 0x00800000u | mant;       // 24-bit significand with hidden bit
        uint32_t shift = static_cast<uint32_t>(-exp - 1); // 14..24
        uint32_t frac = significand >> shift;
        uint32_t remainder = significand & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1u);
        if (remainder > half || (remainder == half && (frac & 1u))) {
            frac += 1u;  // carry may naturally promote to smallest normal
        }
        return static_cast<uint16_t>(sign | frac);
    }
    return static_cast<uint16_t>(sign);
}

// -----------------------------------------------------------------------------
// FP16 bit pattern -> FP32 (lossless).
// -----------------------------------------------------------------------------
inline float Fp16BitsToFp32(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t shift = __builtin_clz(mant) - 21;
            mant <<= shift;
            exp = 1;
            f = sign | ((exp + 127 - 15 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float v;
    __builtin_memcpy(&v, &f, sizeof(float));
    return v;
}

// -----------------------------------------------------------------------------
// bf16_bits_t: distinct storage type for BF16 bit patterns.
// Same size as uint16_t but a distinct type for template specialization,
// so AlgSetAttrDtypeTrait<bf16_bits_t> can use BF16-specific conversion
// (different from FP16's AlgSetAttrDtypeTrait<uint16_t>).
// -----------------------------------------------------------------------------
struct bf16_bits_t {
    uint16_t value = 0;
    bf16_bits_t() = default;
    explicit bf16_bits_t(uint16_t v) : value(v) {}
};

// -----------------------------------------------------------------------------
// BF16 <-> FP32 conversion (aligned with NPU SpltFp32ToBf16 / SpltBf16ToFp32).
// BF16 is the upper 16 bits of FP32 (sign 1 + exponent 8 + mantissa 7).
// -----------------------------------------------------------------------------
inline float Bf16BitsToFp32(uint16_t bf16Bits)
{
    uint32_t fp32Bits = static_cast<uint32_t>(bf16Bits) << 16;
    float f;
    __builtin_memcpy(&f, &fp32Bits, sizeof(float));
    return f;
}

inline uint16_t Fp32ToBf16Bits(float fVal)
{
    uint32_t u32;
    __builtin_memcpy(&u32, &fVal, sizeof(float));
    // RNE (round-to-nearest-even), matching NPU SpltFp32ToBf16.
    const uint32_t lsb = (u32 >> 16) & 1u;
    const uint32_t roundingBias = 0x7FFFu + lsb;
    u32 += roundingBias;
    return static_cast<uint16_t>(u32 >> 16);
}

// -----------------------------------------------------------------------------
// Trait: map storage type T to (a) whether it is FP16 bits, and (b) convert
// one element of T to FP32.
// T = float        -> FP32 path
// T = uint16_t     -> FP16 bit pattern path
// T = bf16_bits_t  -> BF16 bit pattern path
// T = int8_t       -> INT8 path
// -----------------------------------------------------------------------------
template <typename T>
struct AlgSetAttrDtypeTrait;

template <>
struct AlgSetAttrDtypeTrait<float> {
    static constexpr bool kIsFp16 = false;
    static float toFp32(float v) { return v; }
    static float fromFp32(float v) { return v; }
};

template <>
struct AlgSetAttrDtypeTrait<uint16_t> {
    static constexpr bool kIsFp16 = true;
    static float toFp32(uint16_t v) { return Fp16BitsToFp32(v); }
    static uint16_t fromFp32(float v) { return Fp32ToFp16Bits(v); }
};

template <>
struct AlgSetAttrDtypeTrait<bf16_bits_t> {
    static constexpr bool kIsFp16 = false;  // not FP16 bit pattern
    static float toFp32(bf16_bits_t v) { return Bf16BitsToFp32(v.value); }
    static bf16_bits_t fromFp32(float v) { return bf16_bits_t(Fp32ToBf16Bits(v)); }
};

template <>
struct AlgSetAttrDtypeTrait<int8_t> {
    static constexpr bool kIsFp16 = false;
    static float toFp32(int8_t v) { return static_cast<float>(static_cast<int32_t>(v)); }
    static int8_t fromFp32(float v) {
        int32_t iv = static_cast<int32_t>(std::round(v));
        if (iv > 127) { iv = 127; }
        if (iv < -128) { iv = -128; }
        return static_cast<int8_t>(iv);
    }
};

template <>
struct AlgSetAttrDtypeTrait<int32_t> {
    static constexpr bool kIsFp16 = false;
    static float toFp32(int32_t v) { return static_cast<float>(v); }
    static int32_t fromFp32(float v) {
        // round-half-away-from-zero（对齐 kernel INT32 输出舍入模式）
        return (v >= 0.0f) ? static_cast<int32_t>(v + 0.5f)
                           : static_cast<int32_t>(v - 0.5f);
    }
};

// -----------------------------------------------------------------------------
// Shared test data generation utilities.
// -----------------------------------------------------------------------------

// Generate a row-major FP32 matrix filled with uniform random values in
// [lo, hi). Returns FP32; callers quantise to T when needed.
inline std::vector<float> GenFp32Matrix(int64_t rows, int64_t cols, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = dist(rng);
    }
    return v;
}

// Quantise an FP32 vector to the storage type T (FP32: identity; FP16: bits).
inline std::vector<float> ToStorageFp32(const std::vector<float>& v) {
    return v;  // FP32 path: no conversion
}

inline std::vector<uint16_t> ToStorageFp16(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = Fp32ToFp16Bits(v[i]);
    }
    return out;
}

inline std::vector<bf16_bits_t> ToStorageBf16(const std::vector<float>& v) {
    std::vector<bf16_bits_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = bf16_bits_t(Fp32ToBf16Bits(v[i]));
    }
    return out;
}

inline std::vector<int8_t> ToStorageInt8(const std::vector<float>& v) {
    std::vector<int8_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        int32_t iv = static_cast<int32_t>(std::round(v[i]));
        if (iv > 127) { iv = 127; }
        if (iv < -128) { iv = -128; }
        out[i] = static_cast<int8_t>(iv);
    }
    return out;
}

// Generate a row-major matrix of type T from uniform random FP32
// values in [lo, hi). Combines GenFp32Matrix + ToStorage* into one call.
template <typename T>
inline std::vector<T> GenTestMatrix(int64_t rows, int64_t cols, float lo, float hi, uint32_t seed) {
    auto fp32 = GenFp32Matrix(rows, cols, lo, hi, seed);
    if constexpr (AlgSetAttrDtypeTrait<T>::kIsFp16) {
        return ToStorageFp16(fp32);
    } else {
        return ToStorageFp32(fp32);
    }
}

// Unified test matrix generation for all prune dtypes (FP32/FP16/BF16/INT8).
template <typename T>
inline std::vector<T> MatmulGenTestMatrix(int64_t rows, int64_t cols,
                                           float lo, float hi, uint32_t seed)
{
    auto fp32 = GenFp32Matrix(rows, cols, lo, hi, seed);
    if constexpr (std::is_same_v<T, float>) {
        return fp32;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return ToStorageFp16(fp32);
    } else if constexpr (std::is_same_v<T, bf16_bits_t>) {
        return ToStorageBf16(fp32);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return ToStorageInt8(fp32);
    } else {
        return std::vector<T>{};
    }
}

// -----------------------------------------------------------------------------
// Structured pruning primitives (shared by golden references).
//
// Generic row-wise / column-wise N:M structured pruning on FP32 data:
//   - groupSize = 4, keepCount = 2  -> 2:4 sparsity (FP16/BF16/INT8 path)
//   - groupSize = 2, keepCount = 1  -> 1:2 sparsity (FP32 path)
//
// For each group, keep the `keepCount` elements with the largest absolute
// value, zero the rest. Ties are broken by the smaller index. Tail groups
// smaller than `groupSize` are zeroed.
// -----------------------------------------------------------------------------
inline void PickLargestAbs(const float* g, int groupSize, int keepCount, bool* keep)
{
    for (int c = 0; c < keepCount; ++c) {
        int32_t best = -1;
        float bestVal = -1.0f;
        for (int t = 0; t < groupSize; ++t) {
            if (keep[t]) { continue; }
            float av = std::fabs(g[t]);
            if (best < 0 || av > bestVal) { bestVal = av; best = t; }
        }
        if (best >= 0) { keep[best] = true; }
    }
}

inline void PruneRowFp32(const std::vector<float>& A, std::vector<float>& A_pruned,
                         int32_t m, int32_t k, int32_t groupSize, int32_t keepCount)
{
    A_pruned.assign(static_cast<size_t>(m) * static_cast<size_t>(k), 0.0f);
    for (int32_t i = 0; i < m; ++i) {
        const float* row = A.data() + static_cast<int64_t>(i) * k;
        float* outRow = A_pruned.data() + static_cast<int64_t>(i) * k;
        for (int32_t j = 0; j + groupSize <= k; j += groupSize) {
            float g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int t = 0; t < groupSize; ++t) { g[t] = row[j + t]; }
            bool keep[4] = {false, false, false, false};
            PickLargestAbs(g, groupSize, keepCount, keep);
            for (int t = 0; t < groupSize; ++t) {
                if (keep[t]) { outRow[j + t] = g[t]; }
            }
        }
        // tail group (k % groupSize != 0): leave zeroed.
    }
}

inline void PruneColFp32(const std::vector<float>& A, std::vector<float>& A_pruned,
                         int32_t m, int32_t k, int32_t groupSize, int32_t keepCount)
{
    A_pruned.assign(static_cast<size_t>(m) * static_cast<size_t>(k), 0.0f);
    for (int32_t j = 0; j < k; ++j) {
        for (int32_t i = 0; i + groupSize <= m; i += groupSize) {
            float g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int t = 0; t < groupSize; ++t) {
                g[t] = A[static_cast<int64_t>(i + t) * k + j];
            }
            bool keep[4] = {false, false, false, false};
            PickLargestAbs(g, groupSize, keepCount, keep);
            for (int t = 0; t < groupSize; ++t) {
                if (keep[t]) {
                    A_pruned[static_cast<int64_t>(i + t) * k + j] = g[t];
                }
            }
        }
        // tail group (m % groupSize != 0): leave zeroed.
    }
}

}  // namespace sparse_test

#endif  // TEST_PRUNE_TEST_UTIL_H_
