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

#ifndef TEST_MATMUL_GOLDEN_H_
#define TEST_MATMUL_GOLDEN_H_

// =============================================================================
// CPU Golden reference for aclsparseLtMatmul (4 dtype × 2 path).
//
// Paths:
//   sparse×dense: D = α·op(A_pruned)·op(B) + β·C  (with 2:4 prune front-end)
//   dense×dense:  D = α·op(A)·op(B) + β·C          (no prune)
//
// dtype dispatch:
//   FP32/FP16: reuse LtMatmulGoldenRun<T> from ltmatmul_golden_types.h
//   BF16:      new bf16_bits_t trait + LtMatmulGoldenRun<bf16_bits_t>
//   INT8:      LtMatmulGoldenRunInt8 (INT32 accumulation) [in matmul_golden_int8.h]
//
// This header includes:
//   - ltmatmul_golden_types.h: shared types, conversions, prune helpers, CPU golden steps
//   - matmul_golden_int8.h:  INT8-specific golden functions
// And provides FP32/FP16/BF16 golden entry points (with/without vector scaling).
// =============================================================================

#include <algorithm>

#include "ltmatmul_golden_types.h"  // bf16_bits_t, conversions, prune helpers, CPU golden steps
#include "ltmatmul_golden_int8.h"   // INT8 golden entries

namespace sparse_test {

// -----------------------------------------------------------------------------
// PruneAndMatmulFp32 / PruneAndMatmulFp32Vec: shared prune+matmul
// sequences extracted from LtMatmulGoldenRunWithPruneAlg / LtMatmulGoldenRunWithEpilogue
// (scalar) and their Vec counterparts to eliminate duplicate code blocks.
// -----------------------------------------------------------------------------
template <typename T>
inline void PruneAndMatmulFp32(const std::vector<float>& Af, const std::vector<float>& Bf,
    const std::vector<float>& Cf, std::vector<float>& Df,
    int32_t m, int32_t k, int32_t n, float alpha, float beta,
    bool isColOrder, bool pruneB, const std::string& pruneAlg)
{
    if (pruneB) {
        std::vector<float> B_pruned;
        MatmulPruneInputAWithAlg<T>(Bf, B_pruned, k, n, isColOrder, pruneAlg);
        MatmulAlphaBetaFp32(Af, B_pruned, Cf, Df, m, k, n, alpha, beta);
    } else {
        std::vector<float> A_pruned;
        MatmulPruneInputAWithAlg<T>(Af, A_pruned, m, k, isColOrder, pruneAlg);
        MatmulAlphaBetaFp32(A_pruned, Bf, Cf, Df, m, k, n, alpha, beta);
    }
}

template <typename T>
inline void PruneAndMatmulFp32Vec(const std::vector<float>& Af, const std::vector<float>& Bf,
    const std::vector<float>& Cf, std::vector<float>& Df,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    bool isColOrder, bool pruneB, const std::string& pruneAlg)
{
    if (pruneB) {
        std::vector<float> B_pruned;
        MatmulPruneInputAWithAlg<T>(Bf, B_pruned, k, n, isColOrder, pruneAlg);
        MatmulAlphaBetaFp32Vec(Af, B_pruned, Cf, Df, m, k, n, alphaVec, betaVec);
    } else {
        std::vector<float> A_pruned;
        MatmulPruneInputAWithAlg<T>(Af, A_pruned, m, k, isColOrder, pruneAlg);
        MatmulAlphaBetaFp32Vec(A_pruned, Bf, Cf, Df, m, k, n, alphaVec, betaVec);
    }
}

// -----------------------------------------------------------------------------
// Golden entry: sparse×dense for FP32/FP16/BF16 (FP32 accumulation).
// Reuses LtMatmulGoldenRun<T> from ltmatmul_golden_types.h, but with
// MatmulDtypeTrait for prune group parameters.
// For BF16, T=bf16_bits_t uses the BF16 trait.
//
// pruneAlg: "STRIP" (default) or "TILE".
// -----------------------------------------------------------------------------
template <typename T>
inline void LtMatmulGoldenRunWithPruneAlg(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    int32_t alg_config_id, int32_t split_k,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg = "STRIP")
{
    (void)alg_config_id;
    (void)split_k;

    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);
    std::vector<float> Df;
    PruneAndMatmulFp32<T>(Af, Bf, Cf, Df, m, k, n, alpha, beta, isColOrder, pruneB, pruneAlg);
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

// -----------------------------------------------------------------------------
// Golden entry: dense×dense for FP32/FP16/BF16 (skip prune).
// D = α·A·B + β·C, FP32 accumulation, no prune front-end.
// -----------------------------------------------------------------------------
template <typename T>
inline void LtMatmulDenseGoldenRun(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    int32_t alg_config_id, int32_t split_k)
{
    (void)alg_config_id;
    (void)split_k;

    // Step 1: promote inputs to FP32 (no prune).
    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);

    // Step 2: dense matmul + alpha/beta.
    std::vector<float> Df;
    MatmulAlphaBetaFp32(Af, Bf, Cf, Df, m, k, n, alpha, beta);

    // Step 3: truncate back to T.
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

// -----------------------------------------------------------------------------
// Vector scaling golden entries: alpha/beta are per-row float[M] vectors.
// D[i,j] = alphaVec[i] * acc[i,j] + betaVec[i] * C[i,j]
// -----------------------------------------------------------------------------

// sparse×dense FP32/FP16/BF16 with vector scaling
template <typename T>
inline void LtMatmulGoldenRunWithPruneAlgVec(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg = "STRIP")
{
    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);
    std::vector<float> Df;
    PruneAndMatmulFp32Vec<T>(Af, Bf, Cf, Df, m, k, n, alphaVec, betaVec,
                              isColOrder, pruneB, pruneAlg);
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

// dense×dense FP32/FP16/BF16 with vector scaling
template <typename T>
inline void LtMatmulDenseGoldenRunVec(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec)
{
    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);
    std::vector<float> Df;
    MatmulAlphaBetaFp32Vec(Af, Bf, Cf, Df, m, k, n, alphaVec, betaVec);
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

// =============================================================================
// Epilogue golden entries: D = Activation(alpha·op(A_pruned)·op(B) + beta·C + bias)
//
// New functions extend the existing scalar/vector entries with bias + activation
// post-processing (Step 5 + Step 6). All epilogue
// math runs in FP32; bias is per-row broadcast; activation is ReLU/GeLU.
//
// When biasVec is empty and actType==0, these reduce to the existing entries
// (backward compatible).
// =============================================================================

// sparse×dense FP32/FP16/BF16 with epilogue (scalar alpha/beta)
template <typename T>
inline void LtMatmulGoldenRunWithEpilogue(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    int32_t alg_config_id, int32_t split_k,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg,
    const std::vector<float>& biasVec,
    int32_t actType, float reluUb, float reluThr, float geluScale)
{
    (void)alg_config_id;
    (void)split_k;

    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);
    std::vector<float> Df;
    PruneAndMatmulFp32<T>(Af, Bf, Cf, Df, m, k, n, alpha, beta, isColOrder, pruneB, pruneAlg);
    // Step 5 + 6: bias + activation (FP32 domain)
    ApplyEpilogueFp32(Df, biasVec, actType, reluUb, reluThr, geluScale, m, n);
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

// sparse×dense FP32/FP16/BF16 with epilogue (vector alpha/beta)
template <typename T>
inline void LtMatmulGoldenRunWithEpilogueVec(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg,
    const std::vector<float>& biasVec,
    int32_t actType, float reluUb, float reluThr, float geluScale)
{
    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);
    std::vector<float> Df;
    PruneAndMatmulFp32Vec<T>(Af, Bf, Cf, Df, m, k, n, alphaVec, betaVec,
                              isColOrder, pruneB, pruneAlg);
    ApplyEpilogueFp32(Df, biasVec, actType, reluUb, reluThr, geluScale, m, n);
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

// dense×dense FP32/FP16/BF16 with epilogue (scalar alpha/beta)
template <typename T>
inline void LtMatmulDenseGoldenRunWithEpilogue(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    int32_t alg_config_id, int32_t split_k,
    const std::vector<float>& biasVec,
    int32_t actType, float reluUb, float reluThr, float geluScale)
{
    (void)alg_config_id;
    (void)split_k;
    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);
    std::vector<float> Df;
    MatmulAlphaBetaFp32(Af, Bf, Cf, Df, m, k, n, alpha, beta);
    ApplyEpilogueFp32(Df, biasVec, actType, reluUb, reluThr, geluScale, m, n);
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

// dense×dense FP32/FP16/BF16 with epilogue (vector alpha/beta)
template <typename T>
inline void LtMatmulDenseGoldenRunWithEpilogueVec(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    const std::vector<float>& biasVec,
    int32_t actType, float reluUb, float reluThr, float geluScale)
{
    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);
    std::vector<float> Df;
    MatmulAlphaBetaFp32Vec(Af, Bf, Cf, Df, m, k, n, alphaVec, betaVec);
    ApplyEpilogueFp32(Df, biasVec, actType, reluUb, reluThr, geluScale, m, n);
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

// -----------------------------------------------------------------------------
// Batch epilogue wrapper (stage 2): iterate over batches, each batch runs the
// full epilogue chain independently. biasStride==0 → shared bias across batches;
// biasStride>0 → bias pointer advances by biasStride per batch.
//
// batchStride is the element stride for A/B/C/D matrices (same stride for all
// four. BATCH_STRIDE set per-matrix in MatDesc;
// golden assumes caller passes contiguous data with batchStride spacing).
// -----------------------------------------------------------------------------
template <typename T>
inline void LtMatmulGoldenRunWithEpilogueBatch(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    int32_t alg_config_id, int32_t split_k,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg,
    const std::vector<float>& biasVecFlat,  // length = numBatches * (biasStrideShared? m : m), see below
    int64_t biasStride,                     // 0=shared, >0=per-batch offset in biasVecFlat
    int32_t actType, float reluUb, float reluThr, float geluScale,
    int32_t numBatches, int64_t batchStride)
{
    (void)alg_config_id;
    (void)split_k;
    const size_t mk = static_cast<size_t>(m) * k;
    const size_t kn = static_cast<size_t>(k) * n;
    const size_t mn = static_cast<size_t>(m) * n;
    const int64_t bs = (batchStride > 0) ? batchStride
                       : static_cast<int64_t>(std::max({mk, kn, mn}));

    D.resize(static_cast<size_t>(numBatches) * mn);
    for (int32_t b = 0; b < numBatches; ++b) {
        const size_t offA = static_cast<size_t>(b) * bs;
        // A/B/C/D are flat; slice per-batch views (non-owning).
        std::vector<T> Ab(A.begin() + offA, A.begin() + offA + mk);
        std::vector<T> Bb(B.begin() + offA, B.begin() + offA + kn);
        std::vector<T> Cb(C.begin() + offA, C.begin() + offA + mn);
        std::vector<T> Db;
        // bias: shared (biasStride==0) or per-batch slice
        std::vector<float> biasB;
        if (!biasVecFlat.empty()) {
            if (biasStride == 0) {
                biasB.assign(biasVecFlat.begin(), biasVecFlat.begin() + m);
            } else {
                size_t biasOff = static_cast<size_t>(b) * static_cast<size_t>(biasStride);
                biasB.assign(biasVecFlat.begin() + biasOff,
                             biasVecFlat.begin() + biasOff + m);
            }
        }
        LtMatmulGoldenRunWithEpilogue<T>(Ab, Bb, Cb, Db, m, k, n,
                                          alpha, beta, alg_config_id, split_k,
                                          isColOrder, pruneB, pruneAlg,
                                          biasB, actType, reluUb, reluThr, geluScale);
        std::copy(Db.begin(), Db.end(), D.begin() + static_cast<size_t>(b) * mn);
    }
}

}  // namespace sparse_test

#endif  // TEST_MATMUL_GOLDEN_H_
