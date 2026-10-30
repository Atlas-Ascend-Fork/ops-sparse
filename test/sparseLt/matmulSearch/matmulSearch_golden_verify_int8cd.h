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

#ifndef TEST_SPARSELT_MATMULSEARCH_GOLDEN_VERIFY_INT8CD_H_
#define TEST_SPARSELT_MATMULSEARCH_GOLDEN_VERIFY_INT8CD_H_

// =============================================================================
// matmulSearch 测试共享层（二）之 Int8CD 专项段：H 组超集组合
// （INT8 A/B × C=D=CDT）的专用 golden 计算函数族。自
// matmulSearch_golden_verify.h 拆出（codecheck 超大头文件项：原文件
// NBNC 501 > 500，仅超 1），函数体语句逐位不变。
// 依赖说明（同 matmulSearch_npu_wrapper_int8cd.h 惯例）：本头由
// matmulSearch_golden_verify.h 末尾 include（其前置定义
// ExtractPrunedBatch / BiasSliceOfBatch 及经 test_helpers 链可见的
// ExtractInt8CDPruned / ComputeInt8CDGoldenCore / ApplyEpilogueFp32 /
// TruncateToOutput* 等）；不单独 include。
// =============================================================================

namespace matmulsearch_test {

// ----------------------------------------------------------------------------
// H 组超集组合（INT8 A/B × C=D=CDT ∈ {INT32, FP16, BF16}）：独立链 + 专用 golden。
// golden：acc = Σ A_pruned[i,k]*B[k,j]（int32 精确，共享层 ComputeInt8CDGoldenCore）；
// ref = α*acc + β*Cf（FP 域，Cf 由 CDT 提升）；D = cast<CDT>(ref)。C 数据为
// 离散小整数域（[-5,5]）经 CDT 存储（FP16/BF16 精确可表）。
// batch 分支：per-batch 切片（hA/hB/hC/npuPrunedA 按 batchStride 切段，
// bias 按 biasStride>0 逐 batch 独立偏移切片），输出 packed 拼接
// （numBatches * m*n）——与 ComputeSearchGoldenFloat 的 batch 分支同范式。
// 验证口径：C=D=INT32 → MERE atol=1/rtol=0/lim=65（β*C 半整数经 kernel
// round-half-away-from-zero 舍入，±0.5 差由 atol 吸收——C4 先例口径）；
// C=D=FP16/BF16 → 按 D dtype 混合容差（FP 域无舍入问题）。
// ----------------------------------------------------------------------------
// B5 (golden numeric path mirrors the kernel): the int8cd golden's
// epilogue + quantization tail.
// int32 output: with bias/act, applied in the float domain (the original
// numeric path — convert to float then ApplyEpilogue; for |sum| <= 2^24
// double and float are bit-identical, zero regression); otherwise direct
// conversion from double (TruncateToOutputInt32FromDouble, exact in the B5
// domain); FP16/BF16 output: the original float pipeline is unchanged.
template <typename CDT, typename AccT>
inline void ApplyInt8CDGoldenEpilogueAndTruncate(std::vector<AccT>& Df,
    std::vector<CDT>& D, size_t mn, const std::vector<float>& biasVec,
    int32_t actType, const MatmulSearchParam& p)
{
    if constexpr (std::is_same_v<CDT, int32_t>) {
        if (actType != 0 || !biasVec.empty()) {
            std::vector<float> DfF(Df.begin(), Df.end());
            ApplyEpilogueFp32(DfF, biasVec, actType,
                              p.reluUpperBound, p.reluThreshold, p.geluScaling, p.m, p.n);
            Df.assign(DfF.begin(), DfF.end());
        }
        TruncateToOutputInt32FromDouble(Df, D, mn);
    } else {
        ApplyEpilogueFp32(Df, biasVec, actType,
                          p.reluUpperBound, p.reluThreshold, p.geluScaling, p.m, p.n);
        TruncateToOutput<CDT>(Df, D, mn);
    }
}

template <typename CDT>
inline void ComputeSearchGoldenInt8CD(const MatmulSearchParam& p,
    const MatmulNpuResult& npu, const std::vector<int8_t>& hA,
    const std::vector<int8_t>& hB, const std::vector<CDT>& hC,
    const SearchVecs& vecs, std::vector<CDT>& hGoldenD)
{
    const size_t mk = static_cast<size_t>(p.m) * p.k;
    const size_t kn = static_cast<size_t>(p.k) * p.n;
    const size_t mn = static_cast<size_t>(p.m) * p.n;
    const bool useVec = p.alpha_vector_scaling || p.beta_vector_scaling;
    const int32_t actType = GoldenActType(p.activationType);
    if (!p.hasBatch()) {
        const std::vector<int8_t> pruned = ExtractInt8CDPruned(npu, p.isSparseA(), mk);
        auto Df = ComputeInt8CDGoldenCore<CDT>(p.m, p.k, p.n, p.isSparseA(),
            useVec, GoldenAlphaOf(p), GoldenBetaOf(p), vecs.alphaVec, vecs.betaVec,
            hA, hB, pruned, hC);
        ApplyInt8CDGoldenEpilogueAndTruncate<CDT>(Df, hGoldenD, mn,
            vecs.biasVec, actType, p);
        return;
    }
    // batch：per-batch 切片（stride padding 丢弃），输出 packed 拼接
    const int64_t bs = p.batchStride;
    const size_t prunedSize = p.isSparseA() ? mk : kn;
    hGoldenD.resize(static_cast<size_t>(p.numBatches) * mn);
    for (int32_t b = 0; b < p.numBatches; ++b) {
        const size_t off = static_cast<size_t>(b) * static_cast<size_t>(bs);
        const std::vector<int8_t> Ab(hA.begin() + off, hA.begin() + off + mk);
        const std::vector<int8_t> Bb(hB.begin() + off, hB.begin() + off + kn);
        const std::vector<CDT> Cb(hC.begin() + off, hC.begin() + off + mn);
        const std::vector<int8_t> pruned = ExtractPrunedBatch<int8_t>(
            npu, prunedSize, static_cast<size_t>(b), bs);
        const std::vector<float> biasB = BiasSliceOfBatch(p, vecs, b);
        auto Df = ComputeInt8CDGoldenCore<CDT>(p.m, p.k, p.n, p.isSparseA(),
            useVec, GoldenAlphaOf(p), GoldenBetaOf(p), vecs.alphaVec, vecs.betaVec,
            Ab, Bb, pruned, Cb);
        std::vector<CDT> Db;
        ApplyInt8CDGoldenEpilogueAndTruncate<CDT>(Df, Db, mn, biasB, actType, p);
        std::copy(Db.begin(), Db.end(), hGoldenD.begin() + static_cast<size_t>(b) * mn);
    }
}

}  // namespace matmulsearch_test

#endif  // TEST_SPARSELT_MATMULSEARCH_GOLDEN_VERIFY_INT8CD_H_
