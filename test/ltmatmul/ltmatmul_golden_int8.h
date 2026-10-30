/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_MATMUL_GOLDEN_INT8_H_
#define TEST_MATMUL_GOLDEN_INT8_H_

// =============================================================================
// INT8-specific golden functions for aclsparseLtMatmul.
//
// Split from matmul_golden.h to reduce file size.
// Included by matmul_golden.h; users who include matmul_golden.h automatically
// get these functions.
//
// Dependencies (defined in matmul_golden_types.h, available via include):
//   - MatmulPruneInputA (prune helper)
// =============================================================================

#include "ltmatmul_golden_types.h"  // shared types, conversions, prune helpers

namespace sparse_test {

// -----------------------------------------------------------------------------
// MatmulInt32Accumulate extracted from the common (i,p,j)
// INT32 accumulation loop shared by MatmulAlphaBetaInt32 (scalar alpha/beta)
// and MatmulAlphaBetaInt32Vec (per-row vector alpha/beta).
// D_int32[i][j] = Σ A_pruned[i][p] * B[p][j]  (int32 accumulation)
// -----------------------------------------------------------------------------
inline void MatmulInt32Accumulate(
    const std::vector<int8_t>& A_pruned,
    const std::vector<int8_t>& Bf,
    std::vector<int32_t>& Df,
    int32_t m, int32_t k, int32_t n)
{
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    Df.assign(mn, 0);
    for (int32_t i = 0; i < m; ++i) {
        const int8_t* aRow = A_pruned.data() + static_cast<int64_t>(i) * k;
        int32_t* dRow = Df.data() + static_cast<int64_t>(i) * n;
        for (int32_t p = 0; p < k; ++p) {
            int8_t a = aRow[p];
            if (a == 0) { continue; }  // skip pruned zeros
            const int8_t* bRow = Bf.data() + static_cast<int64_t>(p) * n;
            for (int32_t j = 0; j < n; ++j) {
                dRow[j] += static_cast<int32_t>(a) * static_cast<int32_t>(bRow[j]);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// INT32 accumulation matmul for INT8 input.
// D_int32[i][j] = Σ A_pruned[i][p] * B[p][j]  (int32 accumulation)
// Then applies alpha/beta: D = round(α·float(acc) + β·float(C))
// When α=1.0, β=0.0: D = acc (exact integer, no rounding needed).
// -----------------------------------------------------------------------------
inline void MatmulAlphaBetaInt32(
    const std::vector<int8_t>& A_pruned,
    const std::vector<int8_t>& Bf,
    const std::vector<int8_t>& Cf,
    std::vector<int32_t>& Df,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta)
{
    MatmulInt32Accumulate(A_pruned, Bf, Df, m, k, n);
    // Apply alpha/beta scaling (only when non-trivial).
    // B5: with beta==0 and alpha an int32-domain integral value, mirror the
    // kernel's int32-domain scaling (Muls<int32_t>, exact even for
    // |sum| > 2^24); otherwise (including non-integral alpha / beta!=0)
    // keep the float domain (mirroring the kernel's Cast->Muls rounding
    // path — for |sum| <= 2^24 double and float are bit-identical, zero
    // regression).
    if (alpha != 1.0f || beta != 0.0f) {
        const bool int32Domain = (beta == 0.0f) && SpltAlphaIsIntegralInt32Host(alpha);
        for (int32_t i = 0; i < m; ++i) {
            const int8_t* cRow = Cf.data() + static_cast<int64_t>(i) * n;
            int32_t* dRow = Df.data() + static_cast<int64_t>(i) * n;
            for (int32_t j = 0; j < n; ++j) {
                if (int32Domain) {
                    dRow[j] = static_cast<int32_t>(
                        static_cast<int64_t>(alpha) * static_cast<int64_t>(dRow[j]));
                } else {
                    float scaled = alpha * static_cast<float>(dRow[j])
                                 + beta * static_cast<float>(cRow[j]);
                    dRow[j] = static_cast<int32_t>(std::round(scaled));
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Per-row vector scaling variant: alpha/beta are float[M] arrays.
// D[i,j] = alphaVec[i] * acc[i,j] + betaVec[i] * C[i,j]
// Reuses MatmulInt32Accumulate for the common accumulation loop.
// -----------------------------------------------------------------------------
inline void MatmulAlphaBetaInt32Vec(
    const std::vector<int8_t>& A_pruned,
    const std::vector<int8_t>& Bf,
    const std::vector<int8_t>& Cf,
    std::vector<int32_t>& Df,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec,
    const std::vector<float>& betaVec)
{
    MatmulInt32Accumulate(A_pruned, Bf, Df, m, k, n);
    for (int32_t i = 0; i < m; ++i) {
        const int8_t* cRow = Cf.data() + static_cast<int64_t>(i) * n;
        int32_t* dRow = Df.data() + static_cast<int64_t>(i) * n;
        float aV = alphaVec[static_cast<size_t>(i)];
        float bV = betaVec.empty() ? 0.0f : betaVec[static_cast<size_t>(i)];
        if (aV != 1.0f || bV != 0.0f) {
            for (int32_t j = 0; j < n; ++j) {
                float scaled = aV * static_cast<float>(dRow[j])
                             + bV * static_cast<float>(cRow[j]);
                dRow[j] = static_cast<int32_t>(std::round(scaled));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Saturate INT32 → INT8 cast.
// -----------------------------------------------------------------------------
inline void SaturateCastToInt8(const std::vector<int32_t>& src, std::vector<int8_t>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        float v = static_cast<float>(src[i]);
        if (v > 127.0f) { v = 127.0f; }
        if (v < -128.0f) { v = -128.0f; }
        int32_t rounded = static_cast<int32_t>(std::round(v));
        if (rounded > 127) { rounded = 127; }
        if (rounded < -128) { rounded = -128; }
        dst[i] = static_cast<int8_t>(rounded);
    }
}

// -----------------------------------------------------------------------------
// INT8 prune: prune int8_t matrix (group=4, keep=2, compare absolute values).
// Promotes to FP32, prunes, converts back (zeros stay zeros, kept values exact).
//
// pruneAlg dispatch (mirrors FP32/FP16 MatmulPruneInputAWithAlg in
// ltmatmul_golden_types.h):
//   "TILE"  -> SpMMAPruneGolden<int8_t> TILE path (4x4 tile 2:2, row/col joint
//              constraint; edge tiles degenerate to STRIP).
//   "STRIP" -> existing 1D top-N per group (default, backward compatible).
// -----------------------------------------------------------------------------
inline void PruneInt8Matrix(const std::vector<int8_t>& A, std::vector<int8_t>& A_pruned,
                             int32_t m, int32_t k, bool isColOrder,
                             const std::string& pruneAlg = "STRIP")
{
    if (pruneAlg == "TILE") {
        // TILE prune: delegate to SpMMAPruneGolden<int8_t>'s TILE path.
        // SpMMAPruneGolden promotes int8->FP32, prunes in FP32 (4x4 tile 2:2),
        // and converts back via AlgSetAttrDtypeTrait<int8_t>::fromFp32
        // (round + saturate). Kept int8 values are exact (int8->FP32 is lossless
        // and round of an integer float is itself); zeros stay zero.
        // Direction semantics, per branch (post batch-2 view derivation —
        // do NOT read the 8th argument as alongRow):
        //   STRIP (below): grouping always runs along the logical (m,k) rows
        //     (the K dim) regardless of view — MatmulPruneInputA ->
        //     PruneRowFp32 is direction-blind (under (b) semantics the ColView
        //     STRIP and the Row variant share identical group sets);
        //   TILE (here): SpMMAPruneGolden's 8th param is isRowOrder, with
        //     alongRow = (transA != isRowOrder) derived inside. Passing
        //     (transA=false, isRowOrder=!isColOrder) selects the ColView
        //     same-view branch (transpose + Col variant + transpose back)
        //     exactly when isColOrder is true. isColOrder here carries the
        //     caller-derived sparse-side VIEW ((trans == !order), i.e.
        //     ColView=true — see GoldenIsColOrder in ltmatmul_test.cpp), not
        //     the raw physical order, so N+COL / T+ROW take the same-view
        //     TILE branch, matching the kernel's view dispatch. Same
        //     convention as the FP32/FP16 MatmulPruneInputAWithAlg.
        SpMMAPruneGolden<int8_t>(A, A_pruned, m, k, 0, 1, false, !isColOrder, "TILE");
        return;
    }
    // STRIP path: promote to FP32, prune (1D top-N), convert back.
    std::vector<float> Af(static_cast<size_t>(m) * k);
    for (size_t i = 0; i < Af.size(); ++i) {
        Af[i] = static_cast<float>(A[i]);
    }
    std::vector<float> Af_pruned;
    MatmulPruneInputA<int8_t>(Af, Af_pruned, m, k);
    A_pruned.resize(Af.size());
    for (size_t i = 0; i < Af.size(); ++i) {
        A_pruned[i] = (Af_pruned[i] == 0.0f)
            ? static_cast<int8_t>(0)
            : static_cast<int8_t>(std::round(Af_pruned[i]));
    }
}

// -----------------------------------------------------------------------------
// PruneInt8AndMatmul / PruneInt8AndMatmulVec: shared prune+matmul
// sequences extracted from LtMatmulGoldenRunInt8 / LtMatmulGoldenRunInt8WithEpilogue
// (scalar) and their Vec counterparts to eliminate duplicate code blocks.
// -----------------------------------------------------------------------------
inline void PruneInt8AndMatmul(const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C, std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n, float alpha, float beta,
    bool isColOrder, bool pruneB, const std::string& pruneAlg)
{
    std::vector<int8_t> A_pruned;
    std::vector<int8_t> B_pruned;
    if (pruneB) {
        PruneInt8Matrix(B, B_pruned, k, n, isColOrder, pruneAlg);
        MatmulAlphaBetaInt32(A, B_pruned, C, D_int32, m, k, n, alpha, beta);
    } else {
        PruneInt8Matrix(A, A_pruned, m, k, isColOrder, pruneAlg);
        MatmulAlphaBetaInt32(A_pruned, B, C, D_int32, m, k, n, alpha, beta);
    }
}

inline void PruneInt8AndMatmulVec(const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C, std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    bool isColOrder, bool pruneB, const std::string& pruneAlg)
{
    std::vector<int8_t> A_pruned;
    std::vector<int8_t> B_pruned;
    if (pruneB) {
        PruneInt8Matrix(B, B_pruned, k, n, isColOrder, pruneAlg);
        MatmulAlphaBetaInt32Vec(A, B_pruned, C, D_int32, m, k, n, alphaVec, betaVec);
    } else {
        PruneInt8Matrix(A, A_pruned, m, k, isColOrder, pruneAlg);
        MatmulAlphaBetaInt32Vec(A_pruned, B, C, D_int32, m, k, n, alphaVec, betaVec);
    }
}

// -----------------------------------------------------------------------------
// Golden entry: sparse×dense INT8 (INT32 accumulation).
// output_dtype: "INT32" → D is int32_t; "INT8" → D is int8_t (saturated).
// pruneAlg: "STRIP" (default) or "TILE".
// -----------------------------------------------------------------------------
inline void LtMatmulGoldenRunInt8(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg = "STRIP")
{
    PruneInt8AndMatmul(A, B, C, D_int32, m, k, n, alpha, beta, isColOrder, pruneB, pruneAlg);
}

inline void LtMatmulGoldenRunInt8WithNpuPrune(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    const std::vector<int8_t>& npuPrunedA,
    bool isSparseA)
{
    std::vector<int32_t> D_int32;
    if (isSparseA && !npuPrunedA.empty()) {
        MatmulAlphaBetaInt32(npuPrunedA, B, C, D_int32, m, k, n, alpha, beta);
    } else if (!isSparseA) {
        std::vector<int8_t> B_pruned;
        PruneInt8Matrix(B, B_pruned, k, n, false);
        MatmulAlphaBetaInt32(A, B_pruned, C, D_int32, m, k, n, alpha, beta);
    } else {
        std::vector<int8_t> A_pruned;
        PruneInt8Matrix(A, A_pruned, m, k, false);
        MatmulAlphaBetaInt32(A_pruned, B, C, D_int32, m, k, n, alpha, beta);
    }
    SaturateCastToInt8(D_int32, D_int8);
}

inline void LtMatmulGoldenRunInt8VecWithNpuPrune(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec,
    const std::vector<float>& betaVec,
    const std::vector<int8_t>& npuPrunedA,
    bool isSparseA)
{
    std::vector<int32_t> D_int32;
    if (isSparseA && !npuPrunedA.empty()) {
        MatmulAlphaBetaInt32Vec(npuPrunedA, B, C, D_int32, m, k, n, alphaVec, betaVec);
    } else if (!isSparseA) {
        std::vector<int8_t> B_pruned;
        PruneInt8Matrix(B, B_pruned, k, n, false);
        MatmulAlphaBetaInt32Vec(A, B_pruned, C, D_int32, m, k, n, alphaVec, betaVec);
    } else {
        std::vector<int8_t> A_pruned;
        PruneInt8Matrix(A, A_pruned, m, k, false);
        MatmulAlphaBetaInt32Vec(A_pruned, B, C, D_int32, m, k, n, alphaVec, betaVec);
    }
    SaturateCastToInt8(D_int32, D_int8);
}

// -----------------------------------------------------------------------------
// Golden entry: dense×dense INT8 (INT32 accumulation, no prune).
// -----------------------------------------------------------------------------
inline void LtMatmulDenseGoldenRunInt8(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta)
{
    // No prune — direct INT32 matmul on dense A.
    MatmulAlphaBetaInt32(A, B, C, D_int32, m, k, n, alpha, beta);
}

inline void LtMatmulDenseGoldenRunInt8ToInt8(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta)
{
    std::vector<int32_t> D_int32;
    LtMatmulDenseGoldenRunInt8(A, B, C, D_int32, m, k, n, alpha, beta);
    SaturateCastToInt8(D_int32, D_int8);
}

// -----------------------------------------------------------------------------
// Vector scaling golden entries: alpha/beta are per-row float[M] vectors.
// -----------------------------------------------------------------------------

// sparse×dense INT8 with vector scaling (INT32 output)
inline void LtMatmulGoldenRunInt8Vec(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg = "STRIP")
{
    PruneInt8AndMatmulVec(A, B, C, D_int32, m, k, n, alphaVec, betaVec,
                           isColOrder, pruneB, pruneAlg);
}

// dense×dense INT8 with vector scaling (INT32 output)
inline void LtMatmulDenseGoldenRunInt8Vec(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec)
{
    MatmulAlphaBetaInt32Vec(A, B, C, D_int32, m, k, n, alphaVec, betaVec);
}

// dense×dense INT8 with vector scaling (INT8 output)
inline void LtMatmulDenseGoldenRunInt8ToInt8Vec(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec)
{
    std::vector<int32_t> D_int32;
    LtMatmulDenseGoldenRunInt8Vec(A, B, C, D_int32, m, k, n, alphaVec, betaVec);
    SaturateCastToInt8(D_int32, D_int8);
}

// =============================================================================
// Epilogue golden helpers for INT8→INT32 path.
//
// Per requirement doc §2.1.1 INT8→INT32 path:
//   Step 5: ApplyBiasInt32 — bias is FP32, rounds to INT32 (±0.5 rounding error,
//           tolerated by atol=1 per requirement doc §2.6)
//   Step 6: ApplyActivationInt32 — ReLU is integer comparison (no rounding);
//           GeLU computes in FP32 then rounds back to INT32
//   Step 7: INT32 output (no truncation) or SaturateCastToInt8 for INT8 output
// =============================================================================

// Step 5 (INT32 domain): per-row bias broadcast.
// D_int32[i][j] += round(biasVec[i]).
// bias is FP32; round-to-nearest introduces ±0.5 error, covered by atol=1.
inline void ApplyBiasInt32(std::vector<int32_t>& D_int32,
                            const std::vector<float>& biasVec,
                            int32_t m, int32_t n)
{
    if (biasVec.empty()) { return; }
    for (int32_t i = 0; i < m; ++i) {
        int32_t b = static_cast<int32_t>(std::round(biasVec[static_cast<size_t>(i)]));
        int32_t* dRow = D_int32.data() + static_cast<int64_t>(i) * n;
        for (int32_t j = 0; j < n; ++j) { dRow[j] += b; }
    }
}

// Step 6 (INT32 domain): activation post-processing.
//   ReLU: D = min(ub, max(thr, D)) — integer comparison, thresholds rounded to int.
//   GeLU: convert to FP32, compute sigmoid polynomial, round back to INT32.
// actType==0 is a no-op.
//
// When reluUb exceeds INT32 range (e.g. FLT_MAX meaning "no upper bound"),
// the float→int32 cast is undefined behavior. We detect this and skip the
// upper clamp, matching the NPU's FP32-domain behavior where FLT_MAX is a
// no-op clamp. Same safeguard applies to reluThr below INT32 range.
inline void ApplyActivationInt32(std::vector<int32_t>& D_int32,
                                   int32_t actType,
                                   float reluUb, float reluThr,
                                   float geluScale,
                                   int32_t m, int32_t n)
{
    if (actType == 0) { return; }
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    if (actType == 1) {
        // ReLU: integer thresholds (round ub/thr to int for exact comparison).
        // Guard against UB when reluUb/reluThr exceed INT32 range.
        // 2147483520.0f is the largest float ≤ INT32_MAX (2³¹−128).
        constexpr float kInt32MaxF = 2147483520.0f;
        constexpr float kInt32MinF = -2147483648.0f;
        const bool hasUpperBound = (reluUb <= kInt32MaxF);
        const bool hasLowerBound = (reluThr >= kInt32MinF);
        int32_t thrInt = hasLowerBound ? static_cast<int32_t>(std::round(reluThr))
                                      : INT32_MIN;
        int32_t ubInt  = hasUpperBound ? static_cast<int32_t>(std::round(reluUb))
                                      : INT32_MAX;
        for (size_t i = 0; i < mn; ++i) {
            int32_t v = D_int32[i];
            if (hasLowerBound && v < thrInt) { v = thrInt; }
            if (hasUpperBound && v > ubInt) { v = ubInt; }
            D_int32[i] = v;
        }
    } else if (actType == 2) {
        // GeLU: FP32 computation then round back to INT32
        constexpr float kSqrt8OverPi = 1.5957691216057308f;
        for (size_t i = 0; i < mn; ++i) {
            float x = static_cast<float>(D_int32[i]);
            float x3 = x * x * x;
            float t = kSqrt8OverPi * (x + 0.044715f * x3);
            float sigmoid = 1.0f / (1.0f + std::exp(-t));
            float result = geluScale * x * sigmoid;
            D_int32[i] = static_cast<int32_t>(std::round(result));
        }
    }
}

// Convenience: apply bias + activation in sequence (INT32 domain).
inline void ApplyEpilogueInt32(std::vector<int32_t>& D_int32,
                                const std::vector<float>& biasVec,
                                int32_t actType,
                                float reluUb, float reluThr, float geluScale,
                                int32_t m, int32_t n)
{
    ApplyBiasInt32(D_int32, biasVec, m, n);
    ApplyActivationInt32(D_int32, actType, reluUb, reluThr, geluScale, m, n);
}

// =============================================================================
// INT8 epilogue golden entries (sparse×dense + dense×dense, INT32 + INT8 output).
//
// These extend the existing INT8 entries with bias + activation (Steps 5-6).
// When biasVec is empty and actType==0, they reduce to the existing entries.
// =============================================================================

// sparse×dense INT8→INT32 with epilogue (scalar alpha/beta)
inline void LtMatmulGoldenRunInt8WithEpilogue(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg,
    const std::vector<float>& biasVec,
    int32_t actType, float reluUb, float reluThr, float geluScale)
{
    PruneInt8AndMatmul(A, B, C, D_int32, m, k, n, alpha, beta, isColOrder, pruneB, pruneAlg);
    ApplyEpilogueInt32(D_int32, biasVec, actType, reluUb, reluThr, geluScale, m, n);
}

// sparse×dense INT8→INT32 with epilogue (vector alpha/beta)
inline void LtMatmulGoldenRunInt8WithEpilogueVec(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg,
    const std::vector<float>& biasVec,
    int32_t actType, float reluUb, float reluThr, float geluScale)
{
    PruneInt8AndMatmulVec(A, B, C, D_int32, m, k, n, alphaVec, betaVec,
                           isColOrder, pruneB, pruneAlg);
    ApplyEpilogueInt32(D_int32, biasVec, actType, reluUb, reluThr, geluScale, m, n);
}

// sparse×dense INT8→INT8 with epilogue (scalar alpha/beta)
inline void LtMatmulGoldenRunInt8ToInt8WithEpilogue(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg,
    const std::vector<float>& biasVec,
    int32_t actType, float reluUb, float reluThr, float geluScale)
{
    std::vector<int32_t> D_int32;
    LtMatmulGoldenRunInt8WithEpilogue(A, B, C, D_int32, m, k, n,
                                        alpha, beta, isColOrder, pruneB, pruneAlg,
                                        biasVec, actType, reluUb, reluThr, geluScale);
    SaturateCastToInt8(D_int32, D_int8);
}

// sparse×dense INT8→INT8 with epilogue (vector alpha/beta)
inline void LtMatmulGoldenRunInt8ToInt8WithEpilogueVec(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg,
    const std::vector<float>& biasVec,
    int32_t actType, float reluUb, float reluThr, float geluScale)
{
    std::vector<int32_t> D_int32;
    LtMatmulGoldenRunInt8WithEpilogueVec(A, B, C, D_int32, m, k, n,
                                           alphaVec, betaVec, isColOrder, pruneB, pruneAlg,
                                           biasVec, actType, reluUb, reluThr, geluScale);
    SaturateCastToInt8(D_int32, D_int8);
}

// INT8→INT8 with NPU-pruned A + epilogue (scalar/vector alpha/beta)
inline void LtMatmulGoldenRunInt8WithNpuPruneEpilogue(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    const std::vector<int8_t>& npuPrunedA,
    bool isSparseA,
    const std::vector<float>& biasVec,
    int32_t actType, float reluUb, float reluThr, float geluScale)
{
    std::vector<int32_t> D_int32;
    if (isSparseA && !npuPrunedA.empty()) {
        MatmulAlphaBetaInt32(npuPrunedA, B, C, D_int32, m, k, n, alpha, beta);
    } else if (!isSparseA && !npuPrunedA.empty()) {
        // B-sparse: use NPU's pruned B (already in k×n layout)
        MatmulAlphaBetaInt32(A, npuPrunedA, C, D_int32, m, k, n, alpha, beta);
    } else if (!isSparseA) {
        std::vector<int8_t> B_pruned;
        PruneInt8Matrix(B, B_pruned, k, n, false);
        MatmulAlphaBetaInt32(A, B_pruned, C, D_int32, m, k, n, alpha, beta);
    } else {
        std::vector<int8_t> A_pruned;
        PruneInt8Matrix(A, A_pruned, m, k, false);
        MatmulAlphaBetaInt32(A_pruned, B, C, D_int32, m, k, n, alpha, beta);
    }
    ApplyEpilogueInt32(D_int32, biasVec, actType, reluUb, reluThr, geluScale, m, n);
    SaturateCastToInt8(D_int32, D_int8);
}

inline void LtMatmulGoldenRunInt8VecWithNpuPruneEpilogue(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    const std::vector<int8_t>& npuPrunedA,
    bool isSparseA,
    const std::vector<float>& biasVec,
    int32_t actType, float reluUb, float reluThr, float geluScale)
{
    std::vector<int32_t> D_int32;
    if (isSparseA && !npuPrunedA.empty()) {
        MatmulAlphaBetaInt32Vec(npuPrunedA, B, C, D_int32, m, k, n, alphaVec, betaVec);
    } else if (!isSparseA && !npuPrunedA.empty()) {
        // B-sparse: use NPU's pruned B (already in k×n layout)
        MatmulAlphaBetaInt32Vec(A, npuPrunedA, C, D_int32, m, k, n, alphaVec, betaVec);
    } else if (!isSparseA) {
        std::vector<int8_t> B_pruned;
        PruneInt8Matrix(B, B_pruned, k, n, false);
        MatmulAlphaBetaInt32Vec(A, B_pruned, C, D_int32, m, k, n, alphaVec, betaVec);
    } else {
        std::vector<int8_t> A_pruned;
        PruneInt8Matrix(A, A_pruned, m, k, false);
        MatmulAlphaBetaInt32Vec(A_pruned, B, C, D_int32, m, k, n, alphaVec, betaVec);
    }
    ApplyEpilogueInt32(D_int32, biasVec, actType, reluUb, reluThr, geluScale, m, n);
    SaturateCastToInt8(D_int32, D_int8);
}

}  // namespace sparse_test

#endif  // TEST_MATMUL_GOLDEN_INT8_H_
