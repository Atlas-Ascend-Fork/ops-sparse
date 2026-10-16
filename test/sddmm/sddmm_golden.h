/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SDDMM_SDDMM_GOLDEN_H_
#define TEST_SDDMM_SDDMM_GOLDEN_H_

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "verify.h"  // applyMixedTolerance / Verifier (MIXED_TOLERANCE mode)

namespace sparse_test {

// ============================================================================
// SDDMM sparse CSR structure (FP64 values).
// Used by both golden and NPU wrapper (via MakeSddmmSparsity) so that the
// sparsity pattern and C initial values are bit-for-bit identical on both
// sides — this is the precondition for element-wise precision comparison.
// ============================================================================
struct SddmmCsr {
    std::vector<int32_t> rowOffsets;  // size = m + 1, 0-based
    std::vector<int32_t> colIndices;  // size = nnz, ascending within each row
    std::vector<double> values;       // size = nnz, C initial values (FP64)
    int64_t m = 0;
    int64_t n = 0;
    int64_t nnz = 0;
};

// ============================================================================
// Deterministic sparsity pattern generator (per requirement §1).
//
// Position (i, j) is nonzero  <=>  (i * 7 + j * 13) % 100 < int(ratio * 100)
//
// `int(ratio * 100)` uses C++ truncation toward zero (e.g. ratio=0.3 -> 29 or
// 30 depending on FP representation; golden and NPU share the SAME function so
// the resulting pattern is always identical regardless of the exact threshold).
//
// `values` are generated with std::mt19937 + uniform_real_distribution<double>
// over [value_lo, value_hi], seeded by `seed` for reproducibility.
// ============================================================================

inline SddmmCsr MakeSddmmSparsity(int64_t m, int64_t n, double sparsity_ratio,
                                  double value_lo, double value_hi, uint32_t seed) {
    SddmmCsr csr;
    csr.m = m;
    csr.n = n;
    if (m <= 0 || n <= 0) {
        return csr;
    }
    csr.rowOffsets.assign(static_cast<size_t>(m) + 1, 0);

    int32_t threshold = static_cast<int32_t>(sparsity_ratio * 100);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(value_lo, value_hi);

    int64_t nnz = 0;
    for (int64_t i = 0; i < m; i++) {
        csr.rowOffsets[static_cast<size_t>(i)] = static_cast<int32_t>(nnz);
        for (int64_t j = 0; j < n; j++) {
            // (i*7 + j*13) is always non-negative for i,j >= 0; % 100 is safe.
            int32_t key = static_cast<int32_t>(((i * 7) + (j * 13)) % 100);
            if (key < threshold) {
                csr.colIndices.push_back(static_cast<int32_t>(j));
                csr.values.push_back(dist(rng));
                nnz++;
            }
        }
    }
    csr.rowOffsets[static_cast<size_t>(m)] = static_cast<int32_t>(nnz);
    csr.nnz = nnz;
    return csr;
}

// ============================================================================
// SDDMM golden reference (FP64).
//
// For each nonzero position (i, j) (the p-th CSR entry):
//   dot    = sum_{t=0}^{k-1} X[?, ?] * Y[?, ?]        (FP64 accumulation)
//   C_out[p] = alpha * dot + beta * C_init[p]          (FP64 arithmetic)
//
// X layout: row-major. The descriptor shape depends on opX:
//   opX=NON_TRANSPOSE: X is m×k row-major.  X[i, t] = X[i*k + t]
//   opX=TRANSPOSE:     X is k×m row-major.  X[t, i] = X[t*m + i]
//
// opY=NON_TRANSPOSE: Y descriptor is k×n row-major. Computes X·Y.
//   dot = sum_t X[i,t] * Y[t,j],  Y[t*n + j]
// opY=TRANSPOSE:     Y descriptor is n×k row-major. Computes X·Y^T.
//   dot = sum_t X[i,t] * Y[j,t],  Y[j*k + t]
//
// C_init[p] == csrC.values[p] (FP64).
//
// Returns std::vector<double> of length nnz.
// ============================================================================

inline std::vector<double> SddmmGolden(int64_t m, int64_t n, int64_t k,
                                       const std::vector<double>& X,
                                       const std::vector<double>& Y,
                                       const SddmmCsr& csrC,
                                       double alpha, double beta,
                                       aclsparseOperation_t opX = ACL_SPARSE_OP_NON_TRANSPOSE,
                                       aclsparseOperation_t opY = ACL_SPARSE_OP_NON_TRANSPOSE) {
    std::vector<double> out(static_cast<size_t>(csrC.nnz), 0.0);
    if (csrC.nnz == 0) {
        return out;
    }
    const bool xTransposed = (opX == ACL_SPARSE_OP_TRANSPOSE);
    const bool yTransposed = (opY == ACL_SPARSE_OP_TRANSPOSE);
    for (int64_t i = 0; i < m; i++) {
        int32_t s = csrC.rowOffsets[static_cast<size_t>(i)];
        int32_t e = csrC.rowOffsets[static_cast<size_t>(i + 1)];
        for (int32_t p = s; p < e; p++) {
            int32_t j = csrC.colIndices[static_cast<size_t>(p)];
            double dot = 0.0;
            for (int64_t t = 0; t < k; t++) {
                double xVal;
                if (!xTransposed) {
                    // X is m×k row-major: X[i, t] = X[i*k + t]
                    xVal = X[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(t)];
                } else {
                    // X is k×m row-major: X[t, i] = X[t*m + i]
                    xVal = X[static_cast<size_t>(t) * static_cast<size_t>(m) + static_cast<size_t>(i)];
                }
                double yVal;
                if (!yTransposed) {
                    // Y is k×n row-major: Y[t, j] = Y[t*n + j]
                    yVal = Y[static_cast<size_t>(t) * static_cast<size_t>(n) + static_cast<size_t>(j)];
                } else {
                    // Y is n×k row-major: Y[j, t] = Y[j*k + t]
                    yVal = Y[static_cast<size_t>(j) * static_cast<size_t>(k) + static_cast<size_t>(t)];
                }
                dot += xVal * yVal;
            }
            out[static_cast<size_t>(p)] = alpha * dot + beta * csrC.values[static_cast<size_t>(p)];
        }
    }
    return out;
}

// ============================================================================
// FP16 bit conversion helpers (reference: spmm arch35 test).
// Host stores FP16 as uint16_t IEEE-754 bit patterns.
//
// 注意：sparse/sddmm/arch35/sddmm_host.cpp 中有同功能实现 SddmmFp16BitsToFp32，
// 两者算法一致；测试侧独立保留以避免测试代码依赖算子库内部头文件。
// ============================================================================

inline uint16_t Fp32ToFp16Bits(float v) {
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t rawExp = (bits >> 23) & 0xFFu;
    int32_t exp = static_cast<int32_t>(rawExp) - 127;
    uint32_t mant = bits & 0x007FFFFFu;

    // NaN / Inf
    if (rawExp == 0xFFu) {
        uint32_t nanBit = (mant != 0u) ? 0x0200u : 0u;
        return static_cast<uint16_t>(sign | 0x7C00u | nanBit);
    }
    // Overflow to Inf
    if (exp >= 16) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    // Normalized fp16: exp ∈ [-14, 15], round-to-nearest-even
    if (exp >= -14) {
        uint32_t mant10 = mant >> 13;
        uint32_t remainder = mant & 0x1FFFu;
        uint32_t result = (static_cast<uint32_t>(exp + 15) << 10) | mant10;
        if (remainder > 0x1000u || (remainder == 0x1000u && (mant10 & 1u))) {
            result += 1u;
        }
        return static_cast<uint16_t>(sign | result);
    }
    // Subnormal fp16: exp ∈ [-25, -15], otherwise underflow to 0
    if (exp >= -25) {
        uint32_t significand = 0x00800000u | mant;
        uint32_t shift = static_cast<uint32_t>(-exp - 1);
        uint32_t frac = significand >> shift;
        uint32_t remainder = significand & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1u);
        if (remainder > half || (remainder == half && (frac & 1u))) {
            frac += 1u;
        }
        return static_cast<uint16_t>(sign | frac);
    }
    return static_cast<uint16_t>(sign);
}

inline float Fp16BitsToFp32(uint16_t h) {
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
            f = sign | ((exp + 127 - 15 - shift) << 23) | ((mant << 13) & 0x7FFFFFu);
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

// ============================================================================
// Host data conversion helpers: FP64 -> dtype-sized host vectors for NPU input.
// Shared by sddmm_test.cpp and sddmm_perf.cpp to avoid duplicate definitions.
// ============================================================================

inline std::vector<float> DoublesToFp32(const std::vector<double>& v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = static_cast<float>(v[i]);
    }
    return out;
}

inline std::vector<uint16_t> DoublesToFp16(const std::vector<double>& v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = Fp32ToFp16Bits(static_cast<float>(v[i]));
    }
    return out;
}

}  // namespace sparse_test

#endif  // TEST_SDDMM_SDDMM_GOLDEN_H_
