/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GTSV2_STRIDED_BATCH_GTSV2_STRIDED_BATCH_GOLDEN_H_
#define TEST_GTSV2_STRIDED_BATCH_GTSV2_STRIDED_BATCH_GOLDEN_H_

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace sparse_test {

// ============================================================================
// Tridiagonal strided-batch data generator for gtsv2StridedBatch
//
// Generates four strided vectors (dl, d, du, x) describing a batch of
// tridiagonal linear systems A^{(i)} y^{(i)} = x^{(i)}. Each batch starts at
// offset b * batchStride. Constraints guaranteed by the generator:
//   - dl[b*batchStride + 0]      == 0  (lower sub-diagonal padding)
//   - du[b*batchStride + m - 1]  == 0  (upper sub-diagonal padding)
//   - |d[i]| > |dl[i]| + |du[i]| for well_cond / diag_dom / extreme_val
//     (diagonal dominance, avoids Inf/NaN for non-singular patterns)
//   - singular pattern: at least one d[k] == 0 (triggers divide-by-zero,
//     cuSPARSE-aligned Inf/NaN behaviour)
//
// Memory layout (batchCount = 3, batchStride = m + pad):
//   dl: [dl0[0..m-1], pad] [dl1[0..m-1], pad] [dl2[0..m-1], pad]
//   d :  same layout
//   du:  same layout
//   x :  same layout (IN: RHS, OUT: solution, in-place)
// ============================================================================

struct TridiagStridedMatrix {
    std::vector<float> dl;  // sub-diagonal (batchStride apart, dl[b*stride + 0] = 0)
    std::vector<float> d;   // main diagonal (batchStride apart)
    std::vector<float> du;  // super-diagonal (batchStride apart, du[b*stride + m-1] = 0)
    std::vector<float> x;   // RHS / solution (batchStride apart)
    int m = 0;
    int batchCount = 0;
    int batchStride = 0;
};

enum class TridiagPattern { kWellCond, kDiagDom, kConstDiag, kMixedSign, kExtreme, kSingular };

inline float fillUniform(std::mt19937& g, double lo, double hi)
{
    std::uniform_real_distribution<float> dist(static_cast<float>(lo), static_cast<float>(hi));
    return dist(g);
}

inline float fillNonZero(std::mt19937& g, double lo, double hi)
{
    std::uniform_real_distribution<float> mag(static_cast<float>(lo), static_cast<float>(hi));
    std::uniform_int_distribution<int> sign(0, 1);
    float v = mag(g);
    return sign(g) ? -v : v;
}

using TridiagStrategy = void (*)(std::mt19937&, int, int, float&, float&, float&);

inline void stratWellCond(std::mt19937& g, int, int, float& dv, float& dlv, float& duv)
{
    dv = fillNonZero(g, 1.0, 5.0);
    dlv = fillUniform(g, -0.8, 0.8);
    duv = fillUniform(g, -0.8, 0.8);
    if (std::abs(dv) <= std::abs(dlv) + std::abs(duv)) {
        float sgn = (dv >= 0.0f) ? 1.0f : -1.0f;
        dv = sgn * (std::abs(dlv) + std::abs(duv) + 1.0f);
    }
}

inline void stratDiagDom(std::mt19937& g, int, int, float& dv, float& dlv, float& duv)
{
    dv = fillNonZero(g, 1.0, 5.0);
    dlv = fillUniform(g, -0.4, 0.4);
    duv = fillUniform(g, -0.4, 0.4);
}

inline void stratConstDiag(std::mt19937&, int, int, float& dv, float& dlv, float& duv)
{
    dv = 2.0f;
    dlv = -1.0f;
    duv = -1.0f;
}

inline void stratMixedSign(std::mt19937& g, int, int, float& dv, float& dlv, float& duv)
{
    dv = fillNonZero(g, 0.5, 3.0);
    dlv = fillUniform(g, -2.0, 2.0);
    duv = fillUniform(g, -2.0, 2.0);
    if (std::abs(dv) <= std::abs(dlv) + std::abs(duv)) {
        float sgn = (dv >= 0.0f) ? 1.0f : -1.0f;
        dv = sgn * (std::abs(dlv) + std::abs(duv) + 1.0f);
    }
}

inline void stratExtreme(std::mt19937& g, int i, int, float& dv, float& dlv, float& duv)
{
    std::uniform_real_distribution<float> big(1.0e5f, 1.0e6f);
    std::uniform_real_distribution<float> tiny(1.0e-6f, 1.0e-5f);
    float mag = (i % 2 == 0) ? big(g) : tiny(g);
    bool neg = (g() & 1u) != 0u;
    dv = neg ? -mag : mag;
    dlv = fillUniform(g, -0.8, 0.8);
    duv = fillUniform(g, -0.8, 0.8);
    if (std::abs(dv) <= std::abs(dlv) + std::abs(duv)) {
        float sgn = (dv >= 0.0f) ? 1.0f : -1.0f;
        dv = sgn * (std::abs(dlv) + std::abs(duv) + 1.0f);
    }
}

inline void stratSingular(std::mt19937& g, int i, int m, float& dv, float& dlv, float& duv)
{
    bool isMiddle = (i == m / 2);
    dv = isMiddle ? 0.0f : fillNonZero(g, 1.0, 5.0);
    dlv = fillUniform(g, -0.8, 0.8);
    duv = fillUniform(g, -0.8, 0.8);
}

inline TridiagPattern resolvePattern(const std::string& pattern)
{
    if (pattern == "well_cond") return TridiagPattern::kWellCond;
    if (pattern == "diag_dom") return TridiagPattern::kDiagDom;
    if (pattern == "const_diag") return TridiagPattern::kConstDiag;
    if (pattern == "mixed_sign") return TridiagPattern::kMixedSign;
    if (pattern == "extreme_val") return TridiagPattern::kExtreme;
    if (pattern == "singular") return TridiagPattern::kSingular;
    return TridiagPattern::kWellCond;
}

inline TridiagStrategy strategyFor(TridiagPattern p)
{
    switch (p) {
        case TridiagPattern::kDiagDom:    return stratDiagDom;
        case TridiagPattern::kConstDiag:  return stratConstDiag;
        case TridiagPattern::kMixedSign:  return stratMixedSign;
        case TridiagPattern::kExtreme:    return stratExtreme;
        case TridiagPattern::kSingular:   return stratSingular;
        case TridiagPattern::kWellCond:
        default:                          return stratWellCond;
    }
}

inline TridiagStridedMatrix makeSparseTridiagStrided(
    int m, int batchCount, int batchStride,
    const std::string& pattern = "well_cond",
    uint32_t seed = 42)
{
    TridiagStridedMatrix out;
    out.m = m;
    out.batchCount = batchCount;
    out.batchStride = batchStride;

    if (m <= 0 || batchCount <= 0 || batchStride < m) {
        return out;
    }

    const int64_t total = static_cast<int64_t>(batchCount) * batchStride;
    out.dl.assign(total, 0.0f);
    out.d.assign(total, 0.0f);
    out.du.assign(total, 0.0f);
    out.x.assign(total, 0.0f);

    const TridiagStrategy strat = strategyFor(resolvePattern(pattern));

    for (int b = 0; b < batchCount; b++) {
        std::mt19937 brng(seed + static_cast<uint32_t>(b) * 1000003u);
        int64_t base = static_cast<int64_t>(b) * batchStride;

        for (int i = 0; i < m; i++) {
            float dv = 0.0f;
            float dlv = 0.0f;
            float duv = 0.0f;
            strat(brng, i, m, dv, dlv, duv);

            out.d[base + i] = dv;
            out.dl[base + i] = dlv;
            out.du[base + i] = duv;
            out.x[base + i] = fillUniform(brng, -5.0, 5.0);
        }

        out.dl[base + 0] = 0.0f;          // dl[0] = 0
        out.du[base + m - 1] = 0.0f;      // du[m-1] = 0
    }

    return out;
}

// ============================================================================
// Golden reference for aclsparseSgtsv2StridedBatch
//
// Implements the Thomas algorithm (TDMA — Tri-Diagonal Matrix Algorithm) in
// FP64 as the CPU reference baseline. Each batch is solved independently in
// place (the input RHS x is overwritten by the solution y).
//
// Algorithm (per batch, FP64 arithmetic):
//   Forward elimination:
//     for j = 1 .. m-1:
//         w  = dl[j] / d[j-1]
//         d[j]  -= w * du[j-1]
//         x[j]  -= w * x[j-1]
//   Backward substitution:
//     x[m-1] /= d[m-1]
//     for j = m-2 .. 0:
//         x[j] = (x[j] - du[j] * x[j+1]) / d[j]
//
// Why Thomas (FP64) instead of Eigen::SparseMatrix<double>:
//   - Eigen has no native tridiagonal solver; building a general SparseMatrix
//     and calling a generic solver is both slower and semantically misaligned.
//   - Thomas is O(m) and directly mirrors gtsv2's mathematical meaning.
//   - FP64 keeps the reference free of accumulation error and serves as a
//     reliable baseline for the NPU's FP32 CR (Cyclic Reduction) kernel.
//   - Algorithm heterogeneity (Golden Thomas scalar serial vs NPU CR vector
//     parallel) means floating-point rounding orders differ; the FP32
//     rtol/atol thresholds absorb these discrepancies.
//
// Input layout:
//   dl / d / du / x are strided vectors of length batchCount * batchStride;
//   batch b starts at offset b * batchStride.
//   dl[b*stride + 0]     == 0 (caller guarantees)
//   du[b*stride + m - 1] == 0 (caller guarantees)
//
// Output: x is overwritten in place with the solution.
// ============================================================================

// Thomas algorithm forward elimination (FP64, single batch)
inline void ThomasForwardElim(
    std::vector<double>& dL, std::vector<double>& dD,
    std::vector<double>& dU, std::vector<double>& dX, int m)
{
    for (int j = 1; j < m; j++) {
        double w = 0.0;
        if (dD[j - 1] != 0.0) {
            w = dL[j] / dD[j - 1];
        } else {
            // singular matrix: propagate Inf to mirror cuSPARSE/no-pivot behaviour
            w = (dL[j] == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
        }
        dD[j] -= w * dU[j - 1];
        dX[j] -= w * dX[j - 1];
    }
}

// Thomas algorithm backward substitution (FP64, single batch)
inline void ThomasBackwardSub(
    const std::vector<double>& dU,
    std::vector<double>& dD, std::vector<double>& dX, int m)
{
    if (dD[m - 1] != 0.0) {
        dX[m - 1] /= dD[m - 1];
    } else {
        dX[m - 1] = (dX[m - 1] == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                                        : std::numeric_limits<double>::infinity();
    }
    for (int j = m - 2; j >= 0; j--) {
        if (dD[j] != 0.0) {
            dX[j] = (dX[j] - dU[j] * dX[j + 1]) / dD[j];
        } else {
            dX[j] = (dX[j] - dU[j] * dX[j + 1] == 0.0)
                        ? std::numeric_limits<double>::quiet_NaN()
                        : std::numeric_limits<double>::infinity();
        }
    }
}

inline void Sgtsv2StridedBatchGolden(
    int m,
    const std::vector<double>& dl,
    const std::vector<double>& d,
    const std::vector<double>& du,
    std::vector<double>& x,
    int batchCount,
    int batchStride)
{
    if (m <= 0 || batchCount <= 0) {
        return;
    }

    for (int b = 0; b < batchCount; b++) {
        int64_t base = static_cast<int64_t>(b) * batchStride;

        std::vector<double> dL(dl.begin() + base, dl.begin() + base + m);
        std::vector<double> dD(d.begin()  + base, d.begin()  + base + m);
        std::vector<double> dU(du.begin() + base, du.begin() + base + m);
        std::vector<double> dX(x.begin()  + base, x.begin()  + base + m);

        dL[0] = 0.0;
        dU[m - 1] = 0.0;

        ThomasForwardElim(dL, dD, dU, dX, m);
        ThomasBackwardSub(dU, dD, dX, m);

        for (int i = 0; i < m; i++) {
            x[base + i] = dX[i];
        }
    }
}

// ============================================================================
// Convenience wrapper: accept a TridiagStridedMatrix and return a fresh
// FP64 solution vector. The input TridiagStridedMatrix is left unchanged.
// ============================================================================

inline std::vector<double> Sgtsv2StridedBatchGoldenFromTridiag(
    const TridiagStridedMatrix& tri)
{
    // Promote FP32 input to FP64 for the reference computation
    std::vector<double> dl(tri.dl.size());
    std::vector<double> d(tri.d.size());
    std::vector<double> du(tri.du.size());
    std::vector<double> x(tri.x.size());
    for (size_t i = 0; i < tri.dl.size(); i++) {
        dl[i] = static_cast<double>(tri.dl[i]);
        d[i]  = static_cast<double>(tri.d[i]);
        du[i] = static_cast<double>(tri.du[i]);
        x[i]  = static_cast<double>(tri.x[i]);
    }

    Sgtsv2StridedBatchGolden(tri.m, dl, d, du, x, tri.batchCount, tri.batchStride);
    return x;
}

}  // namespace sparse_test

#endif  // TEST_GTSV2_STRIDED_BATCH_GTSV2_STRIDED_BATCH_GOLDEN_H_
