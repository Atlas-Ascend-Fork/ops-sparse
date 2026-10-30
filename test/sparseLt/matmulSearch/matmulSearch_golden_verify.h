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

#ifndef TEST_SPARSELT_MATMULSEARCH_GOLDEN_VERIFY_H_
#define TEST_SPARSELT_MATMULSEARCH_GOLDEN_VERIFY_H_

// =============================================================================
// matmulSearch 测试共享层（二）：数值 golden 计算 + 精度验证（口径表硬断言 +
// 记录落盘）+ 搜索链统一后置断言。
//
// 权威精度断言（精度判定口径表）：逐输出张量实现可执行硬断言——
//   matD 数值：FP32/FP16/BF16 混合容差（frame getMixedToleranceDefaults 默认值
//   与口径表逐位一致：FP32 rtol=2^-10/atol=2^-16/lim=1e-2；FP16 2^-9/1e-1；
//   BF16 2^-6/1.0；matched_ratio>=0.99）；INT8/INT32 输出两级分派
//   （ltmatmul 权威口径同构）：纯整数路径（α=1∧β=0∧无 bias/act∧标量）
//   EXACT（误差=0）；FP 域介入（α≠1/β≠0/bias/GELU/向量缩放）MERE
//   atol=1/rtol=0/lim=65（INT32 出；C3/C4 承载）。
//   algSelection 三元组：Tier-1 + Tier-2 合法候选集谓词（matmulSearch_golden.h）。
//   plan.workspaceSize：白盒重算一致 + 方案 A 单调性。
//   搜索有效性：T 组 4 条（case120-123）为【计时观测用例】——ratio>1.10 输出
//   WARN 并记 INFO(observed)，无数值门禁（防环境噪声假阳性）；另设
//   ratio<=2.5 的防呆级 sanity 硬门禁（拦 Search 功能性退化，与噪声
//   观测口径正交）。阻塞语义 / in-place：声明性验证。
// =============================================================================

#include "matmulSearch_test_helpers.h"

namespace matmulsearch_test {

// ============================================================================
// 数值 golden 计算（NPU 实际剪枝矩阵回读范式；核心数值函数复用
// ltmatmul_golden.h / ltmatmul_golden_types.h / ltmatmul_golden_int8.h）
// ============================================================================

// per-batch 剪枝矩阵提取（A-sparse: (m,k)；B-sparse: (k,n)）。
template <typename T>
inline std::vector<T> ExtractPrunedBatch(const MatmulNpuResult& npu,
    size_t elems, size_t batchIdx, int64_t bs)
{
    std::vector<T> npuPrunedT(elems);
    const size_t byteOff = static_cast<size_t>(batchIdx) * static_cast<size_t>(bs) * sizeof(T);
    std::copy(npu.npuPrunedA.begin() + byteOff,
              npu.npuPrunedA.begin() + byteOff + elems * sizeof(T),
              reinterpret_cast<int8_t*>(npuPrunedT.data()));
    return npuPrunedT;
}

// per-batch bias 切片（biasStride>0：offset=b*stride；否则整份共享）。
inline std::vector<float> BiasSliceOfBatch(const MatmulSearchParam& p,
    const SearchVecs& vecs, int32_t b)
{
    if (p.biasStride > 0) {
        const size_t off = static_cast<size_t>(b) * static_cast<size_t>(p.biasStride);
        return std::vector<float>(vecs.biasVec.begin() + off,
                                  vecs.biasVec.begin() + off + p.m);
    }
    return vecs.biasVec;
}

// 单 batch 数学域计算（math-domain：A=(m,k)、B=(k,n)、C=(m,n)）；bias 为该
// batch 的切片（biasStride>0 逐 batch 独立偏移，否则共享）。
template <typename T>
inline void ComputeGoldenOneBatch(const MatmulSearchParam& p, const SearchVecs& vecs,
    bool useVec, const std::vector<T>& Ab, const std::vector<T>& Bb,
    const std::vector<T>& Cb, const std::vector<T>* prunedT,
    const std::vector<float>& biasB, std::vector<T>& Db)
{
    const size_t mn = static_cast<size_t>(p.m) * p.n;
    const int32_t actType = GoldenActType(p.activationType);
    std::vector<float> Af, Bf, Cf;
    if (prunedT != nullptr && p.isSparseA()) {
        PromoteInputsToFp32<T>(*prunedT, Bb, Cb, Af, Bf, Cf, p.m, p.k, p.n);
    } else if (prunedT != nullptr) {  // B-sparse：pruned 为 B_pruned（(k,n) 布局）
        PromoteInputsToFp32<T>(Ab, *prunedT, Cb, Af, Bf, Cf, p.m, p.k, p.n);
    } else {
        PromoteInputsToFp32<T>(Ab, Bb, Cb, Af, Bf, Cf, p.m, p.k, p.n);
    }
    std::vector<float> Df;
    if (useVec) {
        MatmulAlphaBetaFp32Vec(Af, Bf, Cf, Df, p.m, p.k, p.n, vecs.alphaVec, vecs.betaVec);
    } else {
        MatmulAlphaBetaFp32(Af, Bf, Cf, Df, p.m, p.k, p.n, GoldenAlphaOf(p), GoldenBetaOf(p));
    }
    ApplyEpilogueFp32(Df, biasB, actType,
                      p.reluUpperBound, p.reluThreshold, p.geluScaling, p.m, p.n);
    TruncateToOutput<T>(Df, Db, mn);
}

// 浮点（FP32/FP16/BF16）：A-sparse / B-sparse / dense，含 vec scaling + epilogue；
// batch 分支（浮点 B-sparse×batch）——per-batch 切片（npuPrunedA 同为
// batch-strided），标量 α/β 与 shared bias 跨 batch 共享，输出 packed 拼接。
template <typename T>
inline void ComputeSearchGoldenFloat(const MatmulSearchParam& p,
    const MatmulNpuResult& npu, const std::vector<T>& hA, const std::vector<T>& hB,
    const std::vector<T>& hC, const SearchVecs& vecs, std::vector<T>& hGoldenD)
{
    const size_t mn = static_cast<size_t>(p.m) * p.n;
    const bool useVec = p.alpha_vector_scaling || p.beta_vector_scaling;
    if (!p.hasBatch()) {
        const bool hasNpuPrune = !npu.npuPrunedA.empty();
        const std::vector<float> bias0 = BiasSliceOfBatch(p, vecs, 0);
        if (hasNpuPrune && p.isSparseA()) {
            const auto pruned = ExtractPrunedBatch<T>(npu, static_cast<size_t>(p.m) * p.k, 0, 0);
            ComputeGoldenOneBatch<T>(p, vecs, useVec, hA, hB, hC, &pruned, bias0, hGoldenD);
        } else if (hasNpuPrune) {
            const auto pruned = ExtractPrunedBatch<T>(npu, static_cast<size_t>(p.k) * p.n, 0, 0);
            ComputeGoldenOneBatch<T>(p, vecs, useVec, hA, hB, hC, &pruned, bias0, hGoldenD);
        } else {
            ComputeGoldenOneBatch<T>(p, vecs, useVec, hA, hB, hC, nullptr, bias0, hGoldenD);
        }
        return;
    }
    // batch：per-batch 切片（stride padding 丢弃），输出 packed 拼接；
    // bias per-batch 切片（biasStride>0 逐 batch 独立偏移）
    const size_t mk = static_cast<size_t>(p.m) * p.k;
    const size_t kn = static_cast<size_t>(p.k) * p.n;
    const int64_t bs = p.batchStride;
    const bool hasNpuPrune = !npu.npuPrunedA.empty();
    hGoldenD.resize(static_cast<size_t>(p.numBatches) * mn);
    for (int32_t b = 0; b < p.numBatches; ++b) {
        const size_t off = static_cast<size_t>(b) * static_cast<size_t>(bs);
        const std::vector<T> Ab(hA.begin() + off, hA.begin() + off + mk);
        const std::vector<T> Bb(hB.begin() + off, hB.begin() + off + kn);
        const std::vector<T> Cb(hC.begin() + off, hC.begin() + off + mn);
        const std::vector<float> biasB = BiasSliceOfBatch(p, vecs, b);
        std::vector<T> Db;
        if (hasNpuPrune) {
            const auto pruned = ExtractPrunedBatch<T>(npu, p.isSparseA() ? mk : kn,
                                                      static_cast<size_t>(b), bs);
            ComputeGoldenOneBatch<T>(p, vecs, useVec, Ab, Bb, Cb, &pruned, biasB, Db);
        } else {
            ComputeGoldenOneBatch<T>(p, vecs, useVec, Ab, Bb, Cb, nullptr, biasB, Db);
        }
        std::copy(Db.begin(), Db.end(), hGoldenD.begin() + static_cast<size_t>(b) * mn);
    }
}

// INT8→INT32（A-sparse / B-sparse，含 batch per-slice 与 epilogue：α/β 缩放、
// bias+GeLU 在 INT32 累加域后经 ApplyEpilogueInt32 处理，判定按两级分派口径见
// VerifyInt8Int32SearchOutput）。B-sparse 分支 npuPrunedA 为 B_pruned
// （(k,n) 布局，batch-strided）。
inline void ComputeSearchGoldenInt8Int32(const MatmulSearchParam& p,
    const MatmulNpuResult& npu, const std::vector<int8_t>& hA, const std::vector<int8_t>& hB,
    const std::vector<int8_t>& hC, const SearchVecs& vecs, std::vector<int32_t>& hGoldenD)
{
    const size_t mk = static_cast<size_t>(p.m) * p.k;
    const size_t kn = static_cast<size_t>(p.k) * p.n;
    const size_t mn = static_cast<size_t>(p.m) * p.n;
    const int64_t bs = (p.numBatches > 1 && p.batchStride > 0) ? p.batchStride
                       : static_cast<int64_t>(std::max({mk, kn, mn}));
    hGoldenD.resize(static_cast<size_t>(p.numBatches) * mn);
    const int32_t actType = GoldenActType(p.activationType);
    for (int32_t b = 0; b < p.numBatches; ++b) {
        const size_t off = static_cast<size_t>(b) * static_cast<size_t>(bs);
        std::vector<int32_t> Db;
        if (p.isSparseA()) {
            // A-sparse：npuPrunedA 为 A_pruned（(m,k) 布局）
            std::vector<int8_t> prunedA(npu.npuPrunedA.begin() + off,
                                         npu.npuPrunedA.begin() + off + mk);
            std::vector<int8_t> Bb(hB.begin() + off, hB.begin() + off + kn);
            std::vector<int8_t> Cb(hC.begin() + off, hC.begin() + off + mn);
            MatmulAlphaBetaInt32(prunedA, Bb, Cb, Db, p.m, p.k, p.n,
                                 GoldenAlphaOf(p), GoldenBetaOf(p));
        } else {
            // B-sparse：npuPrunedA 为 B_pruned（(k,n) 布局，剪枝已转置完成）
            std::vector<int8_t> Ab(hA.begin() + off, hA.begin() + off + mk);
            std::vector<int8_t> prunedB(npu.npuPrunedA.begin() + off,
                                         npu.npuPrunedA.begin() + off + kn);
            std::vector<int8_t> Cb(hC.begin() + off, hC.begin() + off + mn);
            MatmulAlphaBetaInt32(Ab, prunedB, Cb, Db, p.m, p.k, p.n,
                                 GoldenAlphaOf(p), GoldenBetaOf(p));
        }
        // epilogue（INT32 域：bias FP32 加法 + GeLU/ReLU；biasStride>0 时
        // 逐 batch 独立偏移切片，否则共享）
        const std::vector<float> biasB = (p.biasStride > 0)
            ? std::vector<float>(vecs.biasVec.begin() + static_cast<size_t>(b) * p.biasStride,
                                 vecs.biasVec.begin() + static_cast<size_t>(b) * p.biasStride + p.m)
            : vecs.biasVec;
        ApplyEpilogueInt32(Db, biasB, actType,
                           p.reluUpperBound, p.reluThreshold, p.geluScaling, p.m, p.n);
        std::copy(Db.begin(), Db.end(), hGoldenD.begin() + static_cast<size_t>(b) * mn);
    }
}

// INT8→INT8（饱和转换确定性，EXACT；A-sparse / B-sparse；含 batch per-slice——
// npuPrunedA 为 batch-strided 全量，per-batch 切片后拼接 packed 输出；
// B-sparse 分支 npuPrunedA 为 B_pruned（(k,n) 布局，剪枝已转置完成））
inline void ComputeSearchGoldenInt8Int8(const MatmulSearchParam& p,
    const MatmulNpuResult& npu, const std::vector<int8_t>& hA, const std::vector<int8_t>& hB,
    const std::vector<int8_t>& hC, std::vector<int8_t>& hGoldenD)
{
    const size_t mk = static_cast<size_t>(p.m) * p.k;
    const size_t kn = static_cast<size_t>(p.k) * p.n;
    const size_t mn = static_cast<size_t>(p.m) * p.n;
    const int64_t bs = (p.numBatches > 1 && p.batchStride > 0) ? p.batchStride
                       : static_cast<int64_t>(std::max({mk, kn, mn}));
    hGoldenD.resize(static_cast<size_t>(p.numBatches) * mn);
    for (int32_t b = 0; b < p.numBatches; ++b) {
        const size_t off = static_cast<size_t>(b) * static_cast<size_t>(bs);
        std::vector<int32_t> d32b;
        if (p.isSparseA()) {
            std::vector<int8_t> prunedA(npu.npuPrunedA.begin() + off,
                                         npu.npuPrunedA.begin() + off + mk);
            std::vector<int8_t> Bb(hB.begin() + off, hB.begin() + off + kn);
            std::vector<int8_t> Cb(hC.begin() + off, hC.begin() + off + mn);
            MatmulAlphaBetaInt32(prunedA, Bb, Cb, d32b, p.m, p.k, p.n,
                                 GoldenAlphaOf(p), GoldenBetaOf(p));
        } else {
            std::vector<int8_t> Ab(hA.begin() + off, hA.begin() + off + mk);
            std::vector<int8_t> prunedB(npu.npuPrunedA.begin() + off,
                                         npu.npuPrunedA.begin() + off + kn);
            std::vector<int8_t> Cb(hC.begin() + off, hC.begin() + off + mn);
            MatmulAlphaBetaInt32(Ab, prunedB, Cb, d32b, p.m, p.k, p.n,
                                 GoldenAlphaOf(p), GoldenBetaOf(p));
        }
        std::vector<int8_t> d8b;
        SaturateCastToInt8(d32b, d8b);
        std::copy(d8b.begin(), d8b.end(), hGoldenD.begin() + static_cast<size_t>(b) * mn);
    }
}

// ============================================================================
// 数值验证（口径表硬断言 + 记录落盘；交叉核对统计仅记录）
// ============================================================================

// 浮点输出：混合容差（frame 默认 = 口径表逐位一致，不覆盖）
template <typename T>
inline bool VerifyFloatSearchOutput(const std::vector<T>& hGoldenD, const std::vector<T>& hNpuD,
    const std::string& caseId, const std::string& tensorTag)
{
    const std::vector<float> goldenF = MatmulToFloat(hGoldenD);
    const std::vector<float> npuF = MatmulToFloat(hNpuD);
    const aclDataType aclDtype = NpuDtypeTrait<T>::kAclDtype;
    VerifyConfig cfg;
    applyMixedTolerance(cfg, aclDtype, goldenF.data(), goldenF.size());  // 口径表取值
    const bool ok = Verifier::verifyVector(npuF, goldenF, cfg, caseId);
    // 交叉核对统计（辅助信息，不作判定依据）——共享层 sparse_test::
    // ComputeErrorMetrics（ltmatmul_test_utils.h，跨文件共享）
    double maxAbsErr = 0.0, maxRelErr = 0.0;
    ComputeErrorMetrics(goldenF, npuF, maxAbsErr, maxRelErr);
    const auto& d = getMixedToleranceDefaults(aclDtype);
    SearchAssertionRecord rec;
    rec.caseId = caseId;
    rec.tensor = tensorTag;
    rec.metric = "MIXED_TOLERANCE(|e|<=atol+rtol|g| & ratio>=0.99 & maxAbsErr<=max(fixed,32ULP))";
    rec.measured = "maxAbsErr=" + std::to_string(maxAbsErr) +
        ",maxRelErr=" + std::to_string(maxRelErr);
    rec.threshold = "atol=" + std::to_string(d.atol) + ",rtol=" + std::to_string(d.rtol) +
        ",maxAbsErrLimit=" + std::to_string(d.maxAbsErrorLimitFixed) + ",matchedRatio>=0.99";
    rec.verdict = VerdictOf(ok);
    AppendSearchRecord(rec);
    return ok;
}

// INT32 输出 MERE 判定 + 落盘共享段（VerifyInt8Int32SearchOutput 与
// VerifyInt8CDInt32Output 的共用 MERE 段）：
// atol=1 / rtol=0 / maxAbsErrLim=65（bias FP32→INT32 round / GeLU round /
// β*C 半整数 round-half-away 的既有权威口径）。withRelErr 控制落盘
// measured 字段格式（Int8Int32 侧原含 maxRelErr、Int8CD 侧原仅 maxAbsErr）；
// expectOk 控制是否在本函数内附带 EXPECT_TRUE（Int8Int32 侧原有、Int8CD
// 侧由调用方断言）——两者均保持拆分前各自的原始行为。
inline bool VerifyInt32MereAndRecord(const std::vector<float>& goldenF,
    const std::vector<float>& npuF, const std::string& caseId,
    const std::string& tensorTag, const std::string& metricTag,
    bool withRelErr, bool expectOk)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenF.data(), goldenF.size());
    cfg.SetMixedTol(1.0, 0.0);
    cfg.SetMixedMaxAbsErrLim(65.0);
    const bool ok = Verifier::verifyVector(npuF, goldenF, cfg, caseId);
    double maxAbsErr = 0.0, maxRelErr = 0.0;
    ComputeErrorMetrics(goldenF, npuF, maxAbsErr, maxRelErr);  // 交叉核对统计（不作判定）
    SearchAssertionRecord rec;
    rec.caseId = caseId;
    rec.tensor = tensorTag;
    rec.metric = metricTag;
    rec.measured = withRelErr
        ? ("maxAbsErr=" + std::to_string(maxAbsErr) + ",maxRelErr=" + std::to_string(maxRelErr))
        : ("maxAbsErr=" + std::to_string(maxAbsErr));
    rec.threshold = "atol=1.000000,rtol=0.000000,maxAbsErrLimit=65.000000,matchedRatio>=0.99";
    rec.verdict = VerdictOf(ok);
    AppendSearchRecord(rec);
    if (expectOk) {
        EXPECT_TRUE(ok) << caseId << " INT32 MERE failed: maxAbsErr=" << maxAbsErr
                        << " (atol=1, maxAbsErrLim=65)";
    }
    return ok;
}

// INT32 输出：两级分派（ltmatmul VerifyInt8Int32Output 权威分派规则同构）：
//   纯整数路径（α=1 ∧ β=0 ∧ 无 bias/activation ∧ 标量缩放）→ EXACT（误差=0）；
//   FP 域介入（α≠1 / β≠0 / bias / GeLU / 向量缩放）→ MERE atol=1 / rtol=0 /
//   maxAbsErrLim=65（bias FP32→INT32 round / GeLU round 的既有权威口径）。
inline bool VerifyInt8Int32SearchOutput(const MatmulSearchParam& p,
    const std::vector<int32_t>& golden, const std::vector<int32_t>& npu,
    const std::string& caseId, const std::string& tensorTag)
{
    EXPECT_EQ(golden.size(), npu.size()) << caseId << ": size mismatch";
    const bool hasBiasOrActivation = p.hasBias() || p.hasActivation();
    const bool hasVecScaling = (p.alpha_vector_scaling != 0) || (p.beta_vector_scaling != 0);
    const bool pureIntegerPath = (GoldenAlphaOf(p) == 1.0f) && (GoldenBetaOf(p) == 0.0f) &&
                                 !hasBiasOrActivation && !hasVecScaling;
    if (pureIntegerPath) {
        int64_t mismatches = 0;
        for (size_t i = 0; i < golden.size() && i < npu.size(); ++i) {
            if (npu[i] != golden[i]) {
                if (mismatches < 5) {
                    std::cout << "[" << caseId << "] mismatch at " << i
                              << ": golden=" << golden[i] << " npu=" << npu[i] << "\n";
                }
                ++mismatches;
            }
        }
        const bool ok = (mismatches == 0) && (golden.size() == npu.size());
        SearchAssertionRecord rec;
        rec.caseId = caseId;
        rec.tensor = tensorTag;
        rec.metric = "EXACT (pure-integer path: alpha==1 && beta==0 && no epilogue)";
        rec.measured = "mismatches=" + std::to_string(mismatches) + "/" + std::to_string(golden.size());
        rec.threshold = "error=0";
        rec.verdict = VerdictOf(ok);
        AppendSearchRecord(rec);
        return ok;
    }
    // FP 域介入 → MERE（atol=1, rtol=0, maxAbsErrLim=65）
    const std::vector<float> goldenF = MatmulToFloat(golden);
    const std::vector<float> npuF = MatmulToFloat(npu);
    return VerifyInt32MereAndRecord(goldenF, npuF, caseId, tensorTag,
        "MERE (fp-domain involved: atol=1/rtol=0/lim=65, ltmatmul authority)",
        /*withRelErr=*/true, /*expectOk=*/true);
}

// INT8 输出：EXACT（逐元素 int8 相等；含饱和转换确定性）
inline bool VerifyExactInt8SearchOutput(const std::vector<int8_t>& golden,
    const std::vector<int8_t>& npu, const std::string& caseId, const std::string& tensorTag)
{
    EXPECT_EQ(golden.size(), npu.size()) << caseId << ": size mismatch";
    int64_t mismatches = 0;
    for (size_t i = 0; i < golden.size() && i < npu.size(); ++i) {
        if (npu[i] != golden[i]) {
            if (mismatches < 5) {
                std::cout << "[" << caseId << "] mismatch at " << i
                          << ": golden=" << static_cast<int32_t>(golden[i])
                          << " npu=" << static_cast<int32_t>(npu[i]) << "\n";
            }
            ++mismatches;
        }
    }
    const bool ok = (mismatches == 0) && (golden.size() == npu.size());
    SearchAssertionRecord rec;
    rec.caseId = caseId;
    rec.tensor = tensorTag;
    rec.metric = "EXACT";
    rec.measured = "mismatches=" + std::to_string(mismatches) + "/" + std::to_string(golden.size());
    rec.threshold = "error=0";
    rec.verdict = VerdictOf(ok);
    AppendSearchRecord(rec);
    return ok;
}

// INT32 出（C=D=INT32）：MERE atol=1（β*C 半整数舍入吸收）
inline bool VerifyInt8CDInt32Output(const MatmulSearchParam& p,
    const std::vector<int32_t>& golden, const std::vector<int32_t>& npu,
    const std::string& caseId, const std::string& tensorTag)
{
    (void)p;
    const std::vector<float> goldenF = MatmulToFloat(golden);
    const std::vector<float> npuF = MatmulToFloat(npu);
    return VerifyInt32MereAndRecord(goldenF, npuF, caseId, tensorTag,
        "MERE (int8-x-cd-i32: atol=1/rtol=0/lim=65, round-half-away absorbed)",
        /*withRelErr=*/false, /*expectOk=*/false);
}

// ============================================================================
// 搜索链统一后置断言（正常用例后置编排）
// ============================================================================
// 步骤 2：阻塞语义声明性检查（返回后无同步直接回读成功；值比对见步骤 6）。
template <typename OutT>
inline void AssertBlockingSemantics(const MatmulSearchNpuResult<OutT>& result,
    const std::string& caseId)
{
    EXPECT_EQ(result.blockedReadRet, ACL_SUCCESS) << caseId << ": blocked read matD failed";
    SearchAssertionRecord rec;
    rec.caseId = caseId; rec.tensor = "blocking";
    rec.metric = "declarative(no-sync memcpy after Search returns)";
    rec.measured = "aclrtMemcpy ret=" + std::to_string(result.blockedReadRet);
    rec.threshold = "ACL_SUCCESS (Search is blocking wrt streams[0])";
    rec.verdict = VerdictOf(result.blockedReadRet == ACL_SUCCESS);
    AppendSearchRecord(rec);
}

// 步骤 4 记录：三元组 Tier-1 + Tier-2 合法候选集判定落盘。
inline void RecordCandidateVerdict(const CandidateVerdict& cand, int32_t maxCfgId,
    const std::string& caseId)
{
    SearchAssertionRecord rec;
    rec.caseId = caseId; rec.tensor = "algSelection(cfg,splitK,splitKMode)";
    rec.metric = "candidate-legal(Tier1+Tier2+wsFit"
                 "+fusedForcedIfDefaultPreset)";
    rec.measured = cand.Brief();
    rec.threshold = "cfg in [0," + std::to_string(maxCfgId) + "), s in [1,K]"
                    ", mode in {0,1}, power-of-2 two-kernel, wsReq<=wsLimit";
    rec.verdict = VerdictOf(cand.AllOk());
    AppendSearchRecord(rec);
}

// 步骤 5 记录：plan 一致性判定落盘。
inline void RecordPlanConsistency(const PlanConsistencyVerdict& pc, const std::string& caseId)
{
    SearchAssertionRecord rec;
    rec.caseId = caseId; rec.tensor = "plan.workspaceSize";
    rec.metric = "ws-recompute-equal + monotonic(<=wsPre)";
    rec.measured = pc.Brief();
    rec.threshold = "wsPost==compute_ws_layout(effSplitK).totalBytes && wsPost<=wsPre";
    rec.verdict = VerdictOf(pc.AllOk());
    AppendSearchRecord(rec);
}

// 步骤 7：计时观测（check_timing=1；ratio 为信息性观测）：event 计时含不可消除
// 的环境噪声（host-device 提交/调度 gap 瞬态，形态为孤立单点/成对/短时段），
// 断言化判 FAIL 会产生环境性假阳性——故 ratio>1.10 仅输出 WARN 并记
// INFO(observed)，无数值门禁。测量协议为逐次交错+顺序平衡 + min + 每侧 20
// 样本，ratio 数值照常写入断言日志。另设 ratio<=2.5 防呆级 sanity 硬门禁
// （远宽于观测阈，不与噪声冲突；仅拦 Search 功能性退化——候选空间缩窄 /
// 候选系统性失败后误选等，见函数体内注释）。其余断言（1-6 步）不受影响。
template <typename OutT>
inline void ObserveTimingRatio(const MatmulSearchParam& p,
    const MatmulSearchNpuResult<OutT>& result, const std::string& caseId)
{
    (void)p;
    ASSERT_TRUE(result.hasTiming) << caseId << ": timing baseline missing";
    ASSERT_TRUE(result.tDefault.valid && result.tSel.valid);
    // 防御：默认 fused plan ws 需求 <= 预置 wsPre（方案 A 单调性）——保留硬断言
    EXPECT_LE(result.wsDefaultNeed, result.wsPre) << caseId << ": default plan ws > preset ws";
    const double selStat = result.tSel.minMs;             // 统计量 min
    const double defStat = result.tDefault.minMs;
    const double ratio = (defStat > 0) ? selStat / defStat : 0.0;
    // 防呆级 sanity 硬门禁（ratio <= 2.5 补偿通道）：远宽于
    // 1.10 观测阈，不与共享设备计时噪声冲突（观测口径不变——ratio>1.10
    // 仍仅 WARN）；仅拦截 Search 功能性退化（如候选空间被缩窄致恒返回
    // 预置配置、候选执行系统性失败后误选等），此类退化下选中配置耗时
    // 相对默认配置的劣化远超噪声量级。
    ASSERT_LE(ratio, 2.5)
        << caseId << ": timing sanity gate tripped (ratio=" << ratio
        << " > 2.5) — search likely degenerated (check candidate space / "
           "candidate execution failures)";
    const bool overThreshold = ratio > 1.10;
    if (overThreshold) {
        // 信息性 WARN（不判 FAIL）：计时含环境噪声（kernel 纯时长稳定而
        // event 区间主体为 host-device 提交/调度 gap，gap 存在瞬态波动），
        // 超阈不构成功能/性能缺陷判据
        std::cout << "[WARN][timing-observed] " << caseId
                  << ": ratio=min(c*)/min(def)=" << ratio
                  << " > 1.10（计时含环境噪声，超阈不判 FAIL，信息性观测）"
                  << " [T(default) min=" << defStat
                  << "ms (mean=" << result.tDefault.meanMs
                  << ", CV=" << result.tDefault.cv << ")"
                  << " | T(c*) min=" << selStat
                  << "ms (mean=" << result.tSel.meanMs
                  << ", CV=" << result.tSel.cv << ")"
                  << " [interleaved 块明细: def前半 min=" << result.tDefaultWin1.minMs
                  << " def后半 min=" << result.tDefaultWin2.minMs
                  << " c*前半 min=" << result.tSelWin1.minMs
                  << " c*后半 min=" << result.tSelWin2.minMs
                  << " | def热身(warmup-only) min=" << result.tDefaultPre.minMs << "]"
                  << std::endl;
    }
    SearchAssertionRecord rec;
    rec.caseId = caseId; rec.tensor = "T(c*)";
    rec.metric = "informational-observed: ratio = min(c*,20)/min(default,20) "
                 "(order-balanced interleaved, 20 pairs)";
    rec.measured = "ratio=" + std::to_string(ratio) + " min(c*)=" +
        std::to_string(selStat) + "ms(mean=" + std::to_string(result.tSel.meanMs) +
        ",CV=" + std::to_string(result.tSel.cv) + ") min(def)=" +
        std::to_string(defStat) + "ms(mean=" +
        std::to_string(result.tDefault.meanMs) + ",CV=" +
        std::to_string(result.tDefault.cv) + ",warmup-only-min=" +
        std::to_string(result.tDefaultPre.minMs) + ")";
    rec.threshold = "1.10 (reference only, not enforced)";
    rec.verdict = overThreshold ? "INFO(observed, ratio>1.10 WARNed)"
                                : "INFO(observed)";
    AppendSearchRecord(rec);
}


// (b) A_pruned 回退区容量（元素）——与库内 splt_apruned_span_elems 同口径：
// A-sparse 取稀疏侧描述符物理跨度（rows/cols 为 transA ? (k,m) : (m,k)，
// ld 取 ld_sparse 覆盖或自然紧凑值），非 A-sparse 维持 m*k。
inline int64_t SearchAprunedSpanElems(const MatmulSearchParam& p)
{
    if (!p.isSparseA()) { return static_cast<int64_t>(p.m) * p.k; }
    const int64_t rows = p.isTransA() ? p.k : p.m;
    const int64_t cols = p.isTransA() ? p.m : p.k;
    const int64_t ld = (p.ldSparse > 0) ? p.ldSparse
        : ((p.isColOrder()) ? rows : cols);
    return p.isColOrder() ? cols * ld : rows * ld;
}

template <typename OutT>
inline void AssertSearchPostconditions(const MatmulSearchParam& p,
    const MatmulSearchNpuResult<OutT>& result, int32_t inDtypeCode,
    const std::string& caseId, bool verifySearchD)
{
    // 1. Search 返回码（expect_status 全 SUCCESS）
    ASSERT_EQ(result.searchRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << ": Search failed";
    // 2. 阻塞语义声明性检查：返回后无同步直接回读成功（值比对见步骤 6）
    AssertBlockingSemantics(result, caseId);
    // 3. 三元组回读 + maxCfgId 查询
    ASSERT_EQ(result.algGetCfgRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(result.algGetSplitRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(result.algGetSplitKModeRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(result.algGetMaxCfgIdRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_GT(result.maxCfgId, 0);
    // 4. Tier-1 + Tier-2 合法候选集谓词（含 ws 方案 A 裁剪 + 默认预置推论）
    const CandidateVerdict cand = AssertCandidateLegal(
        result.selCfg, result.selSplitK, result.selSplitKMode,
        p.k, result.maxCfgId, result.wsPre,
        p.m, p.n, p.k, inDtypeCode, p.numBatches, p.isDefaultPreset(),
        SearchAprunedSpanElems(p));
    EXPECT_TRUE(cand.AllOk()) << caseId << ": candidate illegal: " << cand.Brief();
    RecordCandidateVerdict(cand, result.maxCfgId, caseId);
    // 5. plan 一致性：wsPost == 白盒重算 且 <= wsPre（方案 A 单调性）
    ASSERT_EQ(result.wsPostRet, ACL_SPARSE_STATUS_SUCCESS);
    const PlanConsistencyVerdict pc = AssertPlanConsistency(
        result.wsPre, result.wsPost, p.m, p.n, p.k, cand.effSplitK,
        inDtypeCode, p.numBatches, SearchAprunedSpanElems(p));
    EXPECT_TRUE(pc.AllOk()) << caseId << ": plan ws mismatch: " << pc.Brief();
    RecordPlanConsistency(pc, caseId);
    // 6. 搜索后再执行（plan 生效性；计时类末次测量输出兼作）
    ASSERT_TRUE(result.rerunExecuted) << caseId << ": rerun/last-measure not executed";
    ASSERT_EQ(result.rerunRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << ": rerun matmul failed";
    // 7. 计时观测（check_timing=1；ratio 为信息性观测，无数值门禁）
    if (p.check_timing) {
        ObserveTimingRatio(p, result, caseId);
    }
    (void)verifySearchD;  // 数值比对由调用方按 dtype 编排（B2 in-place 跳过 search 输出）
}

// 从 batch-strided NPU 输出提取 packed 输出（ltmatmul ExtractPackedFromBatchStrided
// 同构：每 batch 取前 mn 个有效元素，丢弃 stride padding；batch=1 原样返回）。
template <typename T>
inline std::vector<T> ExtractPackedFromBatchStrided(const std::vector<T>& batchStrided,
    int32_t m, int32_t n, int32_t numBatches, int64_t batchStride)
{
    if (numBatches <= 1) { return batchStrided; }
    const size_t mn = static_cast<size_t>(m) * n;
    std::vector<T> packed(static_cast<size_t>(numBatches) * mn);
    for (int32_t b = 0; b < numBatches; ++b) {
        const size_t srcOff = static_cast<size_t>(b) * static_cast<size_t>(batchStride);
        const size_t dstOff = static_cast<size_t>(b) * mn;
        std::copy(batchStrided.begin() + srcOff,
                  batchStrided.begin() + srcOff + mn,
                  packed.begin() + dstOff);
    }
    return packed;
}

// C/D 布局感知解包：NPU 输出按 (cdOrder, ldCd) 物理布局 + batch-strided
// 摆放 → packed 逻辑输出（每 batch 解包 m*n 后拼接）。ROW 紧凑时与
// ExtractPackedFromBatchStrided 等价（既有用例数值零变化锚点）。
template <typename T>
inline std::vector<T> ExtractSearchOutputPacked(const std::vector<T>& npuOut,
    const MatmulSearchParam& p)
{
    return sparse_test::ExtractCdLogicalBatch<T>(npuOut, p.m, p.n, ToNpuLayoutOpts(p),
                                                 p.numBatches, p.batchStride);
}

}  // namespace matmulsearch_test

// Int8CD 专项 golden 函数族（H 组超集组合链专用）拆出为独立头
//（codecheck 超大头文件项：本文件 NBNC 501 > 500）——函数体语句逐位不变，
// 依赖本文件前置定义（ExtractPrunedBatch / BiasSliceOfBatch），故在文件
// 末尾 include（同 matmulSearch_npu_wrapper_int8cd.h 惯例，不单独 include）。
#include "matmulSearch_golden_verify_int8cd.h"

#endif  // TEST_SPARSELT_MATMULSEARCH_GOLDEN_VERIFY_H_
