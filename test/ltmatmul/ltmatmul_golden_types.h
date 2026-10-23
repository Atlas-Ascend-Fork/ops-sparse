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

#ifndef TEST_MATMUL_GOLDEN_TYPES_H_
#define TEST_MATMUL_GOLDEN_TYPES_H_

// =============================================================================
// Shared types, conversions, and helpers for the ltmatmul test golden reference.
//
// Provides:
//   - CPU golden step helpers: PromoteInputsToFp32, PruneInputA,
//     MatmulAccumulateFp32, MatmulAlphaBetaFp32, TruncateToOutput
//   - LtMatmulGoldenRun, LtMatmulAlgSetAttributeGolden, AlgSetAttrToFloat
//   - MatmulDtypeTrait, MatmulPruneInputA, vector-scaling golden entries
//
// Includes:
//   - prune_test_util.h: shared utilities (AlgSetAttrDtypeTrait, bf16_bits_t,
//     PruneRowFp32, PruneColFp32, Fp32ToFp16Bits, Fp16BitsToFp32, etc.)
//   - prune_golden.h: SpMMAPruneGolden (TILE prune path)
// =============================================================================

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// Shared CPU golden utilities (AlgSetAttrDtypeTrait, bf16_bits_t, PruneRowFp32,
// PruneColFp32, PickLargestAbs, Fp32ToFp16Bits, Fp16BitsToFp32,
// GenFp32Matrix, ToStorage*, GenTestMatrix, etc.) from prune_test_util.h.
#include "../prune/prune_test_util.h"
// Reuse TILE prune golden (SpMMAPruneGolden with TILE mode)
#include "../prune/prune_golden.h"

namespace sparse_test {

// -----------------------------------------------------------------------------
// CPU golden step helpers (merged from alg_set_attribute_golden.h).
//
// Formula: D = alpha * StructuredPrune(A) * B + beta * C
// Compute path: input T(FP16/FP32) -> promote to FP32 -> structured prune
// -> alpha/beta post-processing -> truncate back to T
//
// Each step is a self-contained function; the main entry point is a 4-step
// sequence.
// -----------------------------------------------------------------------------

// Step 1: Promote inputs A/B/C from storage type T to FP32 (row-major).
template <typename T>
inline void PromoteInputsToFp32(const std::vector<T>& A, const std::vector<T>& B,
                                const std::vector<T>& C,
                                std::vector<float>& Af, std::vector<float>& Bf,
                                std::vector<float>& Cf,
                                int32_t m, int32_t k, int32_t n)
{
    using Trait = AlgSetAttrDtypeTrait<T>;
    const size_t mk = static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t kn = static_cast<size_t>(k) * static_cast<size_t>(n);
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    Af.resize(mk);
    Bf.resize(kn);
    Cf.resize(mn);
    for (size_t i = 0; i < mk; ++i) { Af[i] = Trait::toFp32(A[i]); }
    for (size_t i = 0; i < kn; ++i) { Bf[i] = Trait::toFp32(B[i]); }
    for (size_t i = 0; i < mn; ++i) { Cf[i] = Trait::toFp32(C[i]); }
}

// Step 2: Structured prune on A (FP32), dtype-aware + direction-aware.
template <typename T>
inline void PruneInputA(const std::vector<float>& Af, std::vector<float>& A_pruned,
                        int32_t m, int32_t k, bool isColOrder)
{
    using Trait = AlgSetAttrDtypeTrait<T>;
    const int32_t groupSize = Trait::kIsFp16 ? 4 : 2;
    const int32_t keepCount = Trait::kIsFp16 ? 2 : 1;
    if (isColOrder) {
        PruneColFp32(Af, A_pruned, m, k, groupSize, keepCount);
    } else {
        PruneRowFp32(Af, A_pruned, m, k, groupSize, keepCount);
    }
}

// Step 3: Dense matmul + alpha/beta post-processing (FP32 accumulation).
// Loop ordering (i, p, j) is cache-friendly. Pruned-zero elements of A are
// skipped (mathematically a no-op, doubles throughput for 50%-sparse matrix).
//
// MatmulAccumulateFp32 extracted from the common (i,p,j)
// accumulation loop shared by MatmulAlphaBetaFp32 (scalar alpha/beta) and
// MatmulAlphaBetaFp32Vec (per-row vector alpha/beta below).
inline void MatmulAccumulateFp32(const std::vector<float>& A_pruned,
                                 const std::vector<float>& Bf,
                                 std::vector<float>& Df,
                                 int32_t m, int32_t k, int32_t n)
{
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    Df.assign(mn, 0.0f);
    for (int32_t i = 0; i < m; ++i) {
        const float* aRow = A_pruned.data() + static_cast<int64_t>(i) * k;
        float* dRow = Df.data() + static_cast<int64_t>(i) * n;
        for (int32_t p = 0; p < k; ++p) {
            float a = aRow[p];
            if (a == 0.0f) { continue; }
            const float* bRow = Bf.data() + static_cast<int64_t>(p) * n;
            for (int32_t j = 0; j < n; ++j) {
                dRow[j] += a * bRow[j];
            }
        }
    }
}

inline void MatmulAlphaBetaFp32(const std::vector<float>& A_pruned,
                                 const std::vector<float>& Bf,
                                 const std::vector<float>& Cf,
                                 std::vector<float>& Df,
                                 int32_t m, int32_t k, int32_t n,
                                 float alpha, float beta)
{
    MatmulAccumulateFp32(A_pruned, Bf, Df, m, k, n);
    for (int32_t i = 0; i < m; ++i) {
        const float* cRow = Cf.data() + static_cast<int64_t>(i) * n;
        float* dRow = Df.data() + static_cast<int64_t>(i) * n;
        if (beta == 0.0f) {
            for (int32_t j = 0; j < n; ++j) { dRow[j] = alpha * dRow[j]; }
        } else {
            for (int32_t j = 0; j < n; ++j) { dRow[j] = alpha * dRow[j] + beta * cRow[j]; }
        }
    }
}

// Step 4: Truncate FP32 result back to storage type T.
template <typename T>
inline void TruncateToOutput(const std::vector<float>& Df, std::vector<T>& D, size_t mn)
{
    using Trait = AlgSetAttrDtypeTrait<T>;
    D.resize(mn);
    if (Trait::kIsFp16) {
        constexpr float kFp16Max = 65504.0f;
        for (size_t i = 0; i < mn; ++i) {
            float v = Df[i];
            if (v > kFp16Max) { v = kFp16Max; }
            if (v < -kFp16Max) { v = -kFp16Max; }
            D[i] = Trait::fromFp32(v);
        }
    } else {
        for (size_t i = 0; i < mn; ++i) { D[i] = Trait::fromFp32(Df[i]); }
    }
}

// -----------------------------------------------------------------------------
// Golden entry: D = alpha * StructuredPrune(A) * B + beta * C
//
// Template parameter T:
//   float    -> FP32 input/output
//   uint16_t -> FP16 bit-pattern input/output
//
// alg_config_id / split_k do NOT affect the golden numerical result (they only
// influence NPU internal algorithm selection); they are accepted to align with
// the operator API signature.
//
// isColOrder selects the prune direction:
//   false (ROW) -> PruneRowFp32 (groups along each row's K dimension)
//   true  (COL) -> PruneColFp32 (groups along each column's M dimension)
// This matches the NPU prune kernel's pruneAlongRow logic:
//   pruneAlongRow = (transA != isRowOrder) ? 1 : 0
// For NON_TRANSPOSE + COL order, pruneAlongRow=0 (column-wise pruning).
// The matmul kernel always reads A_pruned as row-major (NDExtLayoutPtn), so
// the golden matmul loop is identical regardless of order.
// -----------------------------------------------------------------------------
template <typename T>
inline void LtMatmulGoldenRun(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    int32_t alg_config_id, int32_t split_k,
    bool isColOrder, bool pruneB)
{
    (void)alg_config_id;  // golden is independent of algorithm selection
    (void)split_k;        // golden is independent of split-K

    // Step 1: promote inputs to FP32 (row-major).
    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);

    // Step 2: structured prune on the sparse matrix (FP32), dtype-aware + direction-aware.
    std::vector<float> Df;
    if (pruneB) {
        std::vector<float> B_pruned;
        PruneInputA<T>(Bf, B_pruned, k, n, isColOrder);
        // Step 3: dense matmul + alpha/beta post-processing (FP32 accumulation).
        MatmulAlphaBetaFp32(Af, B_pruned, Cf, Df, m, k, n, alpha, beta);
    } else {
        std::vector<float> A_pruned;
        PruneInputA<T>(Af, A_pruned, m, k, isColOrder);
        MatmulAlphaBetaFp32(A_pruned, Bf, Cf, Df, m, k, n, alpha, beta);
    }

    // Step 4: truncate back to T.
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

template <typename T>
void LtMatmulAlgSetAttributeGolden(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    int32_t alg_config_id, int32_t split_k,
    bool isColOrder = false)  // default ROW for backward compat
{
    LtMatmulGoldenRun<T>(A, B, C, D, m, k, n, alpha, beta,
                          alg_config_id, split_k, isColOrder, false);
}

// -----------------------------------------------------------------------------
// Helper: convert a T storage vector to FP32 for verification comparison.
// -----------------------------------------------------------------------------
template <typename T>
inline std::vector<float> AlgSetAttrToFloat(const std::vector<T>& v)
{
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = AlgSetAttrDtypeTrait<T>::toFp32(v[i]);
    }
    return out;
}

// =============================================================================
// Matmul-specific golden types and helpers (original ltmatmul_golden_types.h).
// =============================================================================

// -----------------------------------------------------------------------------
// MatmulDtypeTrait: prune group parameters per dtype.
// FP16/BF16/INT8: group=4, keep=2 (2:4 sparsity).
// FP32: group=2, keep=1 (1:2 sparsity).
// -----------------------------------------------------------------------------
template <typename T>
struct MatmulDtypeTrait {
    static constexpr int32_t kGroupSize = AlgSetAttrDtypeTrait<T>::kIsFp16 ? 4 : 2;
    static constexpr int32_t kKeepCount = AlgSetAttrDtypeTrait<T>::kIsFp16 ? 2 : 1;
};

template <>
struct MatmulDtypeTrait<bf16_bits_t> {
    static constexpr int32_t kGroupSize = 4;
    static constexpr int32_t kKeepCount = 2;
};

template <>
struct MatmulDtypeTrait<int8_t> {
    static constexpr int32_t kGroupSize = 4;
    static constexpr int32_t kKeepCount = 2;
};

// -----------------------------------------------------------------------------
// MatmulPruneInputA: dtype-aware + direction-aware prune for all matmul dtypes.
// Extends PruneInputA<T> to support BF16 (group=4) and INT8 (group=4).
// -----------------------------------------------------------------------------
template <typename T>
inline void MatmulPruneInputA(const std::vector<float>& Af, std::vector<float>& A_pruned,
                               int32_t m, int32_t k, bool isColOrder)
{
    const int32_t groupSize = MatmulDtypeTrait<T>::kGroupSize;
    const int32_t keepCount = MatmulDtypeTrait<T>::kKeepCount;
    if (isColOrder) {
        PruneColFp32(Af, A_pruned, m, k, groupSize, keepCount);
    } else {
        PruneRowFp32(Af, A_pruned, m, k, groupSize, keepCount);
    }
}

// -----------------------------------------------------------------------------
// MatmulPruneInputAWithAlg: TILE prune support via prune_golden.h.
// When pruneAlg is "TILE", uses SpMMAPruneGolden's TILE path (2D tile-level
// pruning with row/col joint constraint). Falls back to STRIP otherwise.
// -----------------------------------------------------------------------------
template <typename T>
inline void MatmulPruneInputAWithAlg(const std::vector<float>& Af, std::vector<float>& A_pruned,
                                      int32_t m, int32_t k, bool isColOrder,
                                      const std::string& pruneAlg)
{
    if (pruneAlg == "TILE") {
        // TILE prune: delegate to SpMMAPruneGolden's TILE path.
        // SpMMAPruneGolden works on the storage type T; we convert Af->T, prune, convert back.
        std::vector<T> A_storage(Af.size());
        for (size_t i = 0; i < Af.size(); ++i) {
            A_storage[i] = AlgSetAttrDtypeTrait<T>::fromFp32(Af[i]);
        }
        std::vector<T> A_pruned_storage;
        const bool alongRow = !isColOrder;  // alongRow = !isColOrder for NON_TRANSPOSE
        SpMMAPruneGolden<T>(A_storage, A_pruned_storage, m, k, 0, 1, false, !isColOrder, "TILE");
        A_pruned.resize(A_pruned_storage.size());
        for (size_t i = 0; i < A_pruned_storage.size(); ++i) {
            A_pruned[i] = AlgSetAttrDtypeTrait<T>::toFp32(A_pruned_storage[i]);
        }
    } else {
        MatmulPruneInputA<T>(Af, A_pruned, m, k, isColOrder);
    }
}

// -----------------------------------------------------------------------------
// Per-row vector scaling FP32 matmul: alpha/beta are float[M] arrays.
// D[i,j] = alphaVec[i] * acc[i,j] + betaVec[i] * C[i,j]
//
// Reuses MatmulAccumulateFp32 for the common (i,p,j)
// accumulation loop, eliminating duplication with MatmulAlphaBetaFp32.
// -----------------------------------------------------------------------------
inline void MatmulAlphaBetaFp32Vec(const std::vector<float>& A_pruned,
                                    const std::vector<float>& Bf,
                                    const std::vector<float>& Cf,
                                    std::vector<float>& Df,
                                    int32_t m, int32_t k, int32_t n,
                                    const std::vector<float>& alphaVec,
                                    const std::vector<float>& betaVec)
{
    MatmulAccumulateFp32(A_pruned, Bf, Df, m, k, n);
    for (int32_t i = 0; i < m; ++i) {
        float* dRow = Df.data() + static_cast<int64_t>(i) * n;
        const float* cRow = Cf.data() + static_cast<int64_t>(i) * n;
        float aV = alphaVec[static_cast<size_t>(i)];
        float bV = betaVec.empty() ? 0.0f : betaVec[static_cast<size_t>(i)];
        if (bV == 0.0f) {
            for (int32_t j = 0; j < n; ++j) { dRow[j] = aV * dRow[j]; }
        } else {
            for (int32_t j = 0; j < n; ++j) { dRow[j] = aV * dRow[j] + bV * cRow[j]; }
        }
    }
}

// -----------------------------------------------------------------------------
// Helper: convert storage vector to FP32 for verification.
// -----------------------------------------------------------------------------
template <typename T>
inline std::vector<float> MatmulToFloat(const std::vector<T>& v)
{
    return AlgSetAttrToFloat<T>(v);  // reuse existing helper
}

inline std::vector<float> MatmulToFloat(const std::vector<int32_t>& v)
{
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<float>(v[i]);
    }
    return out;
}

// -----------------------------------------------------------------------------
// Helper: generate a per-row alpha/beta vector (random floats in [0.5, 1.5]).
// -----------------------------------------------------------------------------
inline std::vector<float> GenScalingVector(int32_t m, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    std::vector<float> v(static_cast<size_t>(m));
    for (size_t i = 0; i < v.size(); ++i) { v[i] = dist(rng); }
    return v;
}

// =============================================================================
// Epilogue golden helpers (bias + activation), aligned with cuSPARSELt semantics.
//
// Formula: D = Activation(alpha · op(A_pruned) · op(B) + beta · C + bias)
//   - bias: per-row broadcast (bias[i] added to every element of row i)
//   - ReLU: D[i][j] = min(upperBound, max(threshold, D[i][j]))
//   - GeLU: D[i][j] = gelu_scaling · x · sigmoid(√(8/π) · (x + 0.044715·x³))
// All epilogue math runs in FP32 (even for FP16/BF16/INT8 outputs), per the
// cuSPARSELt math definition. INT8 path rounds bias to INT32 first.
// =============================================================================

// Activation type enum (mirrors the aclsparseLt epilogue encoding).
enum EpilogueActivationType {
    EPILOGUE_ACT_NONE = 0,
    EPILOGUE_ACT_RELU = 1,
    EPILOGUE_ACT_GELU = 2,
};

// Step 5 (FP32 domain): per-row bias broadcast.
// Df[i][j] += biasVec[i]  for all j in [0, n).
// If biasVec is empty, this is a no-op (backward compatible).
inline void ApplyBiasFp32(std::vector<float>& Df,
                          const std::vector<float>& biasVec,
                          int32_t m, int32_t n)
{
    if (biasVec.empty()) { return; }
    for (int32_t i = 0; i < m; ++i) {
        float b = biasVec[static_cast<size_t>(i)];
        float* dRow = Df.data() + static_cast<int64_t>(i) * n;
        for (int32_t j = 0; j < n; ++j) { dRow[j] += b; }
    }
}

// Step 6 (FP32 domain): activation post-processing.
//   ReLU: Df[i][j] = min(ub, max(thr, Df[i][j]))
//   GeLU: Df[i][j] = gelu_scaling · x · sigmoid(√(8/π) · (x + 0.044715·x³))
// actType==0 is a no-op (backward compatible).
inline void ApplyActivationFp32(std::vector<float>& Df,
                                 int32_t actType,
                                 float reluUb, float reluThr,
                                 float geluScale,
                                 int32_t m, int32_t n)
{
    if (actType == EPILOGUE_ACT_NONE) { return; }
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    if (actType == EPILOGUE_ACT_RELU) {
        // G2 归零语义（与 kernel Select(x<=thr, 0, min(x, ub)) 一致）：
        // v <= thr → 0（clamp-to-zero，非 clamp-to-threshold）；thr < v <= ub → v；v > ub → ub。
        // 注意 <= 边界（v == thr 时归零）。
        for (size_t i = 0; i < mn; ++i) {
            float v = Df[i];
            Df[i] = (v <= reluThr) ? 0.0f : std::min(v, reluUb);
        }
    } else if (actType == EPILOGUE_ACT_GELU) {
        // GeLU (tanh approximation): 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))
        // sigmoid form: gelu_scaling · x · sigmoid(√(8/π)·(x+0.044715·x³))
        // sigmoid(t) = 1/(1+e^{-t}); √(8/π) ≈ 1.5957691216057308f
        constexpr float kSqrt8OverPi = 1.5957691216057308f;
        const size_t total = mn;
        for (size_t i = 0; i < total; ++i) {
            float x = Df[i];
            float x3 = x * x * x;
            float t = kSqrt8OverPi * (x + 0.044715f * x3);
            float sigmoid = 1.0f / (1.0f + std::exp(-t));
            Df[i] = geluScale * x * sigmoid;
        }
    }
}

// Convenience: apply bias + activation in sequence (FP32 domain).
inline void ApplyEpilogueFp32(std::vector<float>& Df,
                              const std::vector<float>& biasVec,
                              int32_t actType,
                              float reluUb, float reluThr, float geluScale,
                              int32_t m, int32_t n)
{
    ApplyBiasFp32(Df, biasVec, m, n);
    ApplyActivationFp32(Df, actType, reluUb, reluThr, geluScale, m, n);
}

// -----------------------------------------------------------------------------
// Helper: generate a per-row bias vector.
// Special values controlled by biasSeed pattern:
//   seed%10==0 → all-zero bias (tests bias no-op)
//   seed%10==1 → all-positive bias (2.0)
//   seed%10==2 → all-negative bias (-2.0)
//   seed%10==3 → INT8 max bias (127.0, for INT8 path saturation test)
//   seed%10==4 → INT8 min bias (-128.0)
//   otherwise  → uniform random in [biasLow, biasHigh]
// -----------------------------------------------------------------------------
inline std::vector<float> GenBiasVector(int32_t m, uint32_t biasSeed,
                                         float biasLow = -2.0f, float biasHigh = 2.0f)
{
    std::vector<float> v(static_cast<size_t>(m));
    int pattern = static_cast<int>(biasSeed % 10u);
    if (pattern == 0) {
        // all-zero (SP1)
        for (size_t i = 0; i < v.size(); ++i) { v[i] = 0.0f; }
    } else if (pattern == 1) {
        // all-positive (SP2)
        for (size_t i = 0; i < v.size(); ++i) { v[i] = 2.0f; }
    } else if (pattern == 2) {
        // all-negative (SP3)
        for (size_t i = 0; i < v.size(); ++i) { v[i] = -2.0f; }
    } else if (pattern == 3) {
        // INT8 max (SP9)
        for (size_t i = 0; i < v.size(); ++i) { v[i] = 127.0f; }
    } else if (pattern == 4) {
        // INT8 min (SP10)
        for (size_t i = 0; i < v.size(); ++i) { v[i] = -128.0f; }
    } else {
        std::mt19937 rng(biasSeed);
        std::uniform_real_distribution<float> dist(biasLow, biasHigh);
        for (size_t i = 0; i < v.size(); ++i) { v[i] = dist(rng); }
    }
    return v;
}

}  // namespace sparse_test

#endif  // TEST_MATMUL_GOLDEN_TYPES_H_
