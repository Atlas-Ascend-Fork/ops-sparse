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

/**
 * @file matmulSearch_test.cpp
 * @brief GTest 入口：TEST_P CSV 驱动（L0/L1，88 用例：D 组 dtype×路径、S 组
 *        workspace 语义、T 组计时观测、B 组行为语义、NA 组 shape 边界、SP 组
 *        特殊值、C 组多场景耦合、E 组正交补测、F 组 A-sparse×batch 与规模档、
 *        H 组 dtype 矩阵超集、J 组 ReLU 归零区分性、P 组全维叠加）
 *        + TEST_F 异常路径（本文件 X01~X19：单因子变异 + batch 回退域）。
 *        合计 131 用例（88 CSV 驱动 + 43 TEST_F：X01~X19 20 条 /
 *        X20~X23 4 条（xcases）/ X31~X50 19 条（live），X24~X29/X36
 *        空号不重排）。
 *
 *        同 target 多源组织：
 *          matmulSearch_test_helpers.h      共享基础设施（env/断言记录/glue/
 *                                          输入生成/异常基座公共段/BuildSearchArgs）
 *          matmulSearch_golden_verify.h     golden 计算 + 精度验证 + 后置断言
 *          matmulSearch_test.cpp            本文件（TEST_P + X01~X19）
 *          matmulSearch_test_xcases.cpp     X20~X23（dtype 超集组合描述符层
 *                                          接受性）
 *          matmulSearch_test_live.cpp       X31~X50（补齐 + 探针收编组）
 *

 * 权威精度断言（精度判定口径表）：逐输出张量实现可执行硬断言——
 *   matD 数值：FP32/FP16/BF16 混合容差（frame getMixedToleranceDefaults 默认值
 *   与口径表逐位一致：FP32 rtol=2^-10/atol=2^-16/lim=1e-2；FP16 2^-9/1e-1；
 *   BF16 2^-6/1.0；matched_ratio>=0.99）；INT8/INT32 输出两级分派
 *   （ltmatmul 权威口径同构）：纯整数路径（α=1∧β=0∧无 bias/act∧标量）
 *   EXACT（误差=0）；FP 域介入（α≠1/β≠0/bias/GELU/向量缩放）MERE
 *   atol=1/rtol=0/lim=65（INT32 出；C3/C4 承载）。
 *   algSelection 三元组：Tier-1 + Tier-2 合法候选集谓词（matmulSearch_golden.h）。
 *   plan.workspaceSize：白盒重算一致 + 方案 A 单调性。
 *   搜索有效性：T 组 4 条（case120-123）为【计时观测用例】——测量协议为
 *   逐次交错+顺序平衡（min，每侧 20 样本），ratio 数值照常落盘断言日志；
 *   ratio>1.10 输出 WARN（含两侧明细与环境噪声说明）并记 INFO(observed)——
 *   无 FAIL 路径（event 计时含 host-device 提交/调度 gap 瞬态噪声，形态为
 *   孤立单点/成对/短时段，任何单值统计量均无法全免疫，断言化即环境性假阳性）。
 *   阻塞语义 / in-place：声明性验证。
 * 每条用例产出「输出张量 / 判定指标 / 实测值 / 阈值 / 结论」记录，随结果落盘
 * .cannbot/aclsparseLtMatmulSearch/tmp/precision_assertions.log（JSONL）。
 * 自建宽松容差不设——口径表为唯一通过依据；max_abs_error / max_rel_error /
 * matched_ratio 统计仅作交叉核对信息记录（不作判定）。
 */

#include "test_common.h"

#include "matmulSearch_golden_verify.h"  // 共享层（含 test_helpers / wrapper / golden 链）
#include "ltmatmul_batch_neg_helper.h"   // LtBatchMatDescKit（X18/X19 与 ltmatmul WB_L2 共享）

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace sparse_test;
using namespace matmulsearch_test;

// ============================================================================
// SearchAclEnvManager 静态成员定义（单 TU 定义点；头文件仅声明）
// ============================================================================
std::once_flag matmulsearch_test::SearchAclEnvManager::once_;
std::unique_ptr<sparse_test::AclEnvScope> matmulsearch_test::SearchAclEnvManager::instance_;

namespace {

// ============================================================================
// 链路前置返回码断言（四 runner 共用前缀）
// ============================================================================

// algSel 预置四返回码（全部链共有前缀；搜索链结果类型为
// MatmulSearchNpuResult<OutT>——algSetItersRet 为其扩展字段）。
template <typename OutT>
inline void AssertAlgPresetRets(const MatmulSearchNpuResult<OutT>& r, const std::string& caseId)
{
    ASSERT_EQ(r.algSetCfgRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " preset cfg set failed";
    ASSERT_EQ(r.algSetSplitRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " preset splitK set failed";
    ASSERT_EQ(r.algSetSplitKModeRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " preset mode set failed";
    ASSERT_EQ(r.algSetItersRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " search_iterations set failed";
}

// Search 失败前置闸（设备级错误级联防护）：记录 FAIL 归因并终止本用例。
inline void RecordSearchFailAndAbort(const std::string& caseId, aclsparseStatus_t ret)
{
    SearchAssertionRecord rec;
    rec.caseId = caseId; rec.tensor = "searchRet";
    rec.metric = "search-return";
    rec.measured = "ret=" + std::to_string(static_cast<int32_t>(ret));
    rec.threshold = "SUCCESS";
    rec.verdict = "FAIL(search-failed, device-error-cascade-guard)";
    AppendSearchRecord(rec);
    FAIL() << caseId << ": Search failed ret=" << static_cast<int32_t>(ret)
           << "（设备级错误防护闸：本用例终止，详见断言日志归因记录）";
}

// ============================================================================
// 用例主体（浮点 / INT8→INT32 / INT8→INT8；输入生成与断言段拆分见上）
// ============================================================================

template <typename T>
inline void RunSearchCaseFloat(const MatmulSearchParam& p)
{
    const uint32_t seed = SeedForSearchCase(p.case_id);
    const std::string caseId = p.caseId();
    std::vector<T> hA, hB, hC;
    GenSearchInputs<T>(p, seed, hA, hB, hC);
    const SearchVecs vecs = GenSearchVecs(p, seed);

    std::vector<T> dummyD;
    auto args = BuildSearchArgs<T, T>(p, hA, hB, hC, vecs, dummyD);
    auto result = RunMatmulSearchNpu<T, T>(args);

    // 链路前置返回码
    AssertAlgPresetRets(result, caseId);
    ASSERT_EQ(result.descSetRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " descSetAttribute failed";
    ASSERT_EQ(result.batchSetRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " batch MatDescSetAttribute failed";
    if (!p.isDensePath()) {
        ASSERT_EQ(result.pruneRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " prune failed";
    }
    ASSERT_EQ(result.wsPreRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " GetWorkspace(pre) failed";
    ASSERT_EQ(result.wsDefaultRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " default baseline failed";

    // 数值 golden（NPU 实际剪枝矩阵回读范式）
    // Search 失败前置闸：设备级错误（AI Core Error 级联）时后续设备操作不可靠，
    // 立即终止本用例（FAIL 已由下方断言记录）——探针归因见 .cannbot 报告
    if (result.searchRet != ACL_SPARSE_STATUS_SUCCESS) {
        RecordSearchFailAndAbort(caseId, result.searchRet);
        return;
    }

    std::vector<T> hGoldenD;
    ComputeSearchGoldenFloat<T>(p, result, hA, hB, hC, vecs, hGoldenD);

    const int32_t dtypeCode = SpltDtypeCodeOf(NpuDtypeTrait<T>::kAclDtype);
    AssertSearchPostconditions<T>(p, result, dtypeCode, caseId, !p.inplace);

    // 数值比对：Search 返回 matD（非 in-place）+ 搜索后再执行 matmul
    // batch 用例 NPU 输出为 batch-strided，提取 packed 后比对
    if (!p.inplace) {
        auto hDSearchPacked = ExtractSearchOutputPacked<T>(result.hDSearch, p);
        EXPECT_TRUE(VerifyFloatSearchOutput<T>(hGoldenD, hDSearchPacked, caseId, "matD.search"));
    } else {
        RecordInplaceDeclarative(p, result);
    }
    auto hDFinalPacked = ExtractSearchOutputPacked<T>(result.hDFinal, p);
    EXPECT_TRUE(VerifyFloatSearchOutput<T>(hGoldenD, hDFinalPacked, caseId, "matD.rerun"));
    // SP3（ReLU ub=0）显式交叉核对：确定性全零输出（辅助信息）
    if (p.activationType == 1 && p.reluUpperBound == 0.0f) {
        const auto goldenF = MatmulToFloat(hGoldenD);
        bool allZero = !goldenF.empty();
        for (float v : goldenF) { if (v != 0.0f) { allZero = false; break; } }
        SearchAssertionRecord rec;
        rec.caseId = caseId; rec.tensor = "matD";
        rec.metric = "cross-check(all-zero, SP3)";
        rec.measured = allZero ? "all zeros" : "non-zero present";
        rec.threshold = "all zeros (golden side)";
        rec.verdict = "INFO";
        AppendSearchRecord(rec);
    }
}

inline void RunSearchCaseInt8Int32(const MatmulSearchParam& p)
{
    const uint32_t seed = SeedForSearchCase(p.case_id);
    const std::string caseId = p.caseId();
    std::vector<int8_t> hA, hB, hC;
    GenSearchInputs<int8_t>(p, seed, hA, hB, hC);  // 含 batch 布局（D4/C4）
    const SearchVecs vecs = GenSearchVecs(p, seed);

    std::vector<int32_t> dummyD;
    auto args = BuildSearchArgs<int8_t, int32_t>(p, hA, hB, hC, vecs, dummyD);
    auto result = RunMatmulSearchNpu<int8_t, int32_t>(args);

    AssertAlgPresetRets(result, caseId);
    ASSERT_EQ(result.batchSetRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " batch set failed";
    ASSERT_EQ(result.pruneRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " prune failed";
    ASSERT_EQ(result.wsPreRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " GetWorkspace(pre) failed";
    ASSERT_EQ(result.wsDefaultRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " default baseline failed";

    std::vector<int32_t> hGoldenD;
    ComputeSearchGoldenInt8Int32(p, result, hA, hB, hC, vecs, hGoldenD);

    const int32_t dtypeCode = SpltDtypeCodeOf(ACL_INT8);
    AssertSearchPostconditions<int32_t>(p, result, dtypeCode, caseId, !p.inplace);

    if (!p.inplace) {
        auto hDSearchPacked = ExtractSearchOutputPacked<int32_t>(result.hDSearch, p);
        EXPECT_TRUE(VerifyInt8Int32SearchOutput(p, hGoldenD, hDSearchPacked, caseId, "matD.search"));
    } else {
        RecordInplaceDeclarative(p, result);
    }
    auto hDFinalPacked = ExtractSearchOutputPacked<int32_t>(result.hDFinal, p);
    EXPECT_TRUE(VerifyInt8Int32SearchOutput(p, hGoldenD, hDFinalPacked, caseId, "matD.rerun"));
}

inline void RunSearchCaseInt8Int8(const MatmulSearchParam& p)
{
    const uint32_t seed = SeedForSearchCase(p.case_id);
    const std::string caseId = p.caseId();
    std::vector<int8_t> hA, hB, hC;
    GenSearchInputs<int8_t>(p, seed, hA, hB, hC);
    const SearchVecs vecs = GenSearchVecs(p, seed);

    std::vector<int8_t> dummyD;
    auto args = BuildSearchArgs<int8_t, int8_t>(p, hA, hB, hC, vecs, dummyD);
    auto result = RunMatmulSearchNpu<int8_t, int8_t>(args);

    AssertAlgPresetRets(result, caseId);
    ASSERT_EQ(result.pruneRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " prune failed";
    ASSERT_EQ(result.wsPreRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " GetWorkspace(pre) failed";
    ASSERT_EQ(result.wsDefaultRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " default baseline failed";

    std::vector<int8_t> hGoldenD;
    ComputeSearchGoldenInt8Int8(p, result, hA, hB, hC, hGoldenD);

    const int32_t dtypeCode = SpltDtypeCodeOf(ACL_INT8);
    AssertSearchPostconditions<int8_t>(p, result, dtypeCode, caseId, !p.inplace);

    if (!p.inplace) {
        auto hDSearchPacked = ExtractSearchOutputPacked<int8_t>(result.hDSearch, p);
        EXPECT_TRUE(VerifyExactInt8SearchOutput(hGoldenD, hDSearchPacked, caseId, "matD.search"));
    } else {
        RecordInplaceDeclarative(p, result);
    }
    auto hDFinalPacked = ExtractSearchOutputPacked<int8_t>(result.hDFinal, p);
    EXPECT_TRUE(VerifyExactInt8SearchOutput(hGoldenD, hDFinalPacked, caseId, "matD.rerun"));
}

// ----------------------------------------------------------------------------
// H 组超集组合（INT8 A/B × C=D=CDT）用例主体（输入生成 / args 组装段 +
// RunSearchChainInt8CD 链执行 + 两级验证）。batch（num_batches>1）为真
// 接入：四矩阵 batch 属性经链铺设段设置，输入按 batch-strided 铺设，
// golden 逐 batch 计算拼接，输出解包 packed 后比对。
// ----------------------------------------------------------------------------

// Int8CD 输入生成：A/B 离散小整数域（[-5,5]）+ C 离散小整数域经 CDT 存储
// （FP16/BF16 精确可表；共享层 GenInt8CDInputC）+ vecs（按值返回——
// SearchVecs 指针成员指向返回值自身，必须 auto 接收，禁止输出参数赋值传递）。
// batch 时 A/B 按 GenBatchStridedMatrix 范式逐 batch 生成独立数据并按
// batchStride 铺设（段间 padding 置零），C 走共享层 GenInt8CDInputCBatch。
template <typename CDT>
SearchVecs PrepareInt8CDSearchInputs(const MatmulSearchParam& p, uint32_t seed,
    std::vector<int8_t>& hA, std::vector<int8_t>& hB, std::vector<CDT>& hC)
{
    if (p.hasBatch()) {
        auto genBatch = [&p](int32_t rows, int32_t cols, uint32_t s) {
            const size_t total = static_cast<size_t>(p.numBatches) *
                                 static_cast<size_t>(p.batchStride);
            std::vector<int8_t> data(total, int8_t{0});
            for (int32_t b = 0; b < p.numBatches; ++b) {
                auto bd = MatmulGenTestMatrix<int8_t>(rows, cols, -5.0f, 5.0f,
                    s + static_cast<uint32_t>(b) * 100);
                std::copy(bd.begin(), bd.end(),
                          data.begin() + static_cast<size_t>(b) * p.batchStride);
            }
            return data;
        };
        hA = genBatch(p.m, p.k, seed);
        hB = genBatch(p.k, p.n, seed + 1);
        GenInt8CDInputCBatch<CDT>(p.m, p.n, p.numBatches, p.batchStride, hC);
    } else {
        hA = MatmulGenTestMatrix<int8_t>(p.m, p.k, -5.0f, 5.0f, seed);
        hB = MatmulGenTestMatrix<int8_t>(p.k, p.n, -5.0f, 5.0f, seed + 1);
        GenInt8CDInputC<CDT>(p.m, p.n, hC);
    }
    return GenSearchVecs(p, seed);
}

// Int8CD 搜索链参数组装（H 组超集链专用：C 走独立参数、C/D dtype=CDT；
// dummyD 由调用方持有——args 基类 hD 为引用成员）。
template <typename CDT>
MatmulSearchRunArgs<int8_t, CDT, CDT> BuildInt8CDSearchArgs(const MatmulSearchParam& p,
    const std::vector<int8_t>& hA, const std::vector<int8_t>& hB,
    std::vector<CDT>& dummyD, const SearchVecs& vecs)
{
    MatmulSearchRunArgs<int8_t, CDT, CDT> args{
        p.m, p.k, p.n, hA, hB, hA, dummyD,   // 基类 hC 绑 hA 占位（C 走独立参数）
        p.alpha, p.beta, p.algConfigId, p.splitK, p.splitKMode,
        ToNpuOrder(p), ToNpuOp(p.isTransA()), ToNpuOp(p.isTransB()),
        p.isTilePrune() ? ACLSPARSELT_PRUNE_SPMMA_TILE : ACLSPARSELT_PRUNE_SPMMA_STRIP,
        p.alpha_vector_scaling, p.beta_vector_scaling, vecs.alphaPtr, vecs.betaPtr};
    args.biasEnabled = p.biasEnabled;
    args.biasStride = p.biasStride;
    args.activationType = p.activationType;
    args.reluUpperBound = p.reluUpperBound;
    args.reluThreshold = p.reluThreshold;
    args.geluScaling = p.geluScaling;
    args.numBatches = p.numBatches;
    args.batchStride = p.batchStride;
    args.isSparseA = p.isSparseA();
    args.isDensePath = false;
    args.searchIterations = p.search_iterations;
    args.checkTiming = (p.check_timing == 1);
    args.matANull = (p.mat_a_null == 1);
    args.alphaNull = (p.alpha_null == 1);
    args.betaNull = (p.beta_null == 1);
    args.numStreams = p.num_streams;
    args.biasVec = vecs.biasPtr;
    args.biasEltSize = sizeof(float);   // INT8 超集链 C 为 FP 域 → bias FP32
    args.biasDtype = ACL_FLOAT;
    args.layout = ToNpuLayoutOpts(p);
    return args;
}

template <typename CDT>
inline void RunSearchCaseInt8CD(const MatmulSearchParam& p)
{
    const uint32_t seed = SeedForSearchCase(p.case_id);
    const std::string caseId = p.caseId();
    std::vector<int8_t> hA, hB;
    std::vector<CDT> hC;
    const SearchVecs vecs = PrepareInt8CDSearchInputs<CDT>(p, seed, hA, hB, hC);

    std::vector<CDT> dummyD;
    auto args = BuildInt8CDSearchArgs<CDT>(p, hA, hB, dummyD, vecs);

    auto result = RunSearchChainInt8CD<CDT>(args, hA, hB, hC);

    AssertAlgPresetRets(result, caseId);
    ASSERT_EQ(result.batchSetRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " batch MatDescSetAttribute failed";
    ASSERT_EQ(result.pruneRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " prune failed";
    ASSERT_EQ(result.wsPreRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << " GetWorkspace(pre) failed";

    std::vector<CDT> hGoldenD;
    ComputeSearchGoldenInt8CD<CDT>(p, result, hA, hB, hC, vecs, hGoldenD);

    const int32_t dtypeCode = SpltDtypeCodeOf(ACL_INT8);
    AssertSearchPostconditions<CDT>(p, result, dtypeCode, caseId, !p.inplace);

    // batch 真接入断言：NPU 输出有效元素数 == numBatches*m*n（回读为
    // batch-strided 全量，解包 packed 后为 numBatches*mn）
    const size_t mn = static_cast<size_t>(p.m) * p.n;
    if (!p.inplace) {
        ASSERT_EQ(result.hDSearch.size(),
                  BatchTotalElems(mn, p.numBatches, p.batchStride))
            << caseId << ": search output elems != batch-strided total";
    }
    ASSERT_EQ(result.hDFinal.size(), BatchTotalElems(mn, p.numBatches, p.batchStride))
        << caseId << ": rerun output elems != batch-strided total";
    ASSERT_EQ(hGoldenD.size(), static_cast<size_t>(p.numBatches) * mn)
        << caseId << ": golden elems != numBatches*m*n";

    if (!p.inplace) {
        auto hDSearchPacked = ExtractSearchOutputPacked<CDT>(result.hDSearch, p);
        ASSERT_EQ(hDSearchPacked.size(), static_cast<size_t>(p.numBatches) * mn)
            << caseId << ": packed search output != numBatches*m*n";
        if constexpr (std::is_same_v<CDT, int32_t>) {
            EXPECT_TRUE(VerifyInt8CDInt32Output(p, hGoldenD, hDSearchPacked, caseId, "matD.search"));
        } else {
            EXPECT_TRUE(VerifyFloatSearchOutput<CDT>(hGoldenD, hDSearchPacked, caseId, "matD.search"));
        }
    } else {
        RecordInplaceDeclarative(p, result);
    }
    auto hDFinalPacked = ExtractSearchOutputPacked<CDT>(result.hDFinal, p);
    ASSERT_EQ(hDFinalPacked.size(), static_cast<size_t>(p.numBatches) * mn)
        << caseId << ": packed rerun output != numBatches*m*n";
    if constexpr (std::is_same_v<CDT, int32_t>) {
        EXPECT_TRUE(VerifyInt8CDInt32Output(p, hGoldenD, hDFinalPacked, caseId, "matD.rerun"));
    } else {
        EXPECT_TRUE(VerifyFloatSearchOutput<CDT>(hGoldenD, hDFinalPacked, caseId, "matD.rerun"));
    }
}

}  // namespace

// ============================================================================
// TEST_P：CSV 驱动参数化用例（L0 + L1，88 行）
// ============================================================================
class MatmulSearchTest : public testing::TestWithParam<sparse_test::MatmulSearchParam>
{
  public:
    static void SetUpTestSuite() { (void)matmulsearch_test::SearchAclEnvManager::Get(); }
    static void TearDownTestSuite() {}

  protected:
    sparse_test::MatmulSearchParam param_;
    void SetUp() override { param_ = GetParam(); }
};

TEST_P(MatmulSearchTest, CsvDriven)
{
    const auto& p = param_;
    ASSERT_TRUE(p.expectSuccess()) << p.caseId() << ": v1 全部用例 expect_status=SUCCESS";
    if (!ValidateParamCombo(p)) { return; }

    // 链构建失败（描述符/plan guard throw）不作为裸异常冒泡：给出明确归因。
    // 已知支持域冲突（参数调整须回退方案裁决，不在测试侧自行改参）：
    //   - structured 对齐倍数：FP32=8 / FP16·BF16=16 / INT8=32（rows/cols/ld 均须倍数）
    //   - matA/matB 有且仅一个 structured（Matmul 支持域无 dense×dense 路径）
    //   命中示例：D2(FP16,m=520%16!=0)、D4(INT8,m=48%32!=0)、D5(dense×dense)。
    try {
        if (p.isFp32()) {
            RunSearchCaseFloat<float>(p);
        } else if (p.isFp16()) {
            RunSearchCaseFloat<uint16_t>(p);
        } else if (p.isBf16()) {
            RunSearchCaseFloat<bf16_bits_t>(p);
        } else if (p.isInt8()) {
            if (p.isInt32Output()) {
                RunSearchCaseInt8Int32(p);
            } else if (p.isInt8Output()) {
                RunSearchCaseInt8Int8(p);
            } else if (p.output_dtype == "FP16") {
                RunSearchCaseInt8CD<uint16_t>(p);      // H 组超集：INT8 × C=D=FP16
            } else if (p.output_dtype == "BF16") {
                RunSearchCaseInt8CD<bf16_bits_t>(p);   // H 组超集：INT8 × C=D=BF16
            } else if (p.output_dtype == "I32CD") {
                RunSearchCaseInt8CD<int32_t>(p);       // H 组超集：INT8 × C=D=INT32（独立 C dtype）
            } else {
                FAIL() << p.caseId() << ": INT8 requires output_dtype INT8/INT32/FP16/BF16/I32CD";
            }
        } else {
            FAIL() << p.caseId() << ": unsupported dtype " << p.dtype;
        }
    } catch (const std::runtime_error& e) {
        ADD_FAILURE()
            << p.caseId() << ": 搜索链构建失败: " << e.what()
            << " —— 若为描述符约束拒绝（structured 对齐倍数 FP32=8/FP16,BF16=16/INT8=32，"
               "或 matA/matB 须恰一个 structured），则为用例参数与 aclsparseLtMatmul 支持域"
               "冲突（支持域继承 aclsparseLtMatmul），回退方案裁决参数调整；"
               "不在测试侧自行改参。";
    }
}

INSTANTIATE_TEST_SUITE_P(MatmulSearchCases, MatmulSearchTest,
                         testing::ValuesIn(sparse_test::GetCasesFromCsv<sparse_test::MatmulSearchParam>(
                             "matmulSearch_test.csv")),
                         [](const testing::TestParamInfo<sparse_test::MatmulSearchParam>& info)
                         {
                             std::string name = info.param.caseId();
                             std::replace(name.begin(), name.end(), '-', '_');
                             return name;
                         });

// ============================================================================
// TEST_F：L2 异常路径 X01~X19（有效 FP32 128³ 默认预置全链为基座，单因子变异；
// X17/X18/X19 独立基座；matmulPlanInit 先例范式）
// （MatmulSearchExceptionTest fixture 定义在 matmulSearch_test_helpers.h——
// 拆分后三个 TU 共享）
// ============================================================================

// ---- X01: handle = nullptr → HANDLE_IS_NULLPTR ----
TEST_F(MatmulSearchExceptionTest, X01HandleNullptr)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    p.handle = nullptr;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// ---- X02: *handle = nullptr（handle 指向空句柄变量）→ HANDLE_IS_NULLPTR ----
TEST_F(MatmulSearchExceptionTest, X02HandleStarNull)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    aclsparseLtHandle_t hNull = nullptr;
    p.handle = &hNull;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// ---- X03: plan = nullptr → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X03PlanNullptr)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    p.searchPlan = nullptr;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X04: *plan = nullptr（未初始化/Destroy 后）→ INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X04PlanStarNull)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    aclsparseLtMatmulPlan_t planNull = nullptr;
    p.searchPlan = &planNull;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X05: streams = nullptr → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X05StreamsNullptr)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    p.streams = nullptr;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X06: numStreams = 0 → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X06NumStreamsZero)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    p.numStreams = 0;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X07: numStreams = -1 → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X07NumStreamsNegative)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    p.numStreams = -1;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X08: matB = nullptr → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X08MatBNullptr)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    p.matB = nullptr;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X09: matC = nullptr → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X09MatCNullptr)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    p.matC = nullptr;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X10: matD = nullptr → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X10MatDNullptr)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    p.matD = nullptr;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X11: workspace = nullptr → INSUFFICIENT_RESOURCES ----
TEST_F(MatmulSearchExceptionTest, X11WorkspaceNullptr)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    p.workspace = nullptr;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
}

// ---- X12: workspace 未 16 字节对齐 → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X12WorkspaceMisaligned)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    sparse_test::DeviceBuffer misaligned = sparse_test::DeviceBuffer::alloc(ctx.workspaceSize + 16);
    p.workspace = static_cast<uint8_t*>(misaligned.get()) + 1;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X13: matB 未 16 字节对齐 → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X13MatBMisaligned)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    const size_t kn = 128 * 128;
    sparse_test::DeviceBuffer misaligned = sparse_test::DeviceBuffer::alloc(kn * sizeof(float) + 16);
    p.matB = static_cast<uint8_t*>(misaligned.get()) + 1;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X14: matC 未 16 字节对齐 → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X14MatCMisaligned)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    const size_t mn = 128 * 128;
    sparse_test::DeviceBuffer misaligned = sparse_test::DeviceBuffer::alloc(mn * sizeof(float) + 16);
    p.matC = static_cast<uint8_t*>(misaligned.get()) + 1;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X15: matD 未 16 字节对齐 → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X15MatDMisaligned)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    const size_t mn = 128 * 128;
    sparse_test::DeviceBuffer misaligned = sparse_test::DeviceBuffer::alloc(mn * sizeof(float) + 16);
    p.matD = static_cast<uint8_t*>(misaligned.get()) + 1;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X16: matA（非空）未 16 字节对齐 → INVALID_VALUE ----
TEST_F(MatmulSearchExceptionTest, X16MatAMisaligned)
{
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    const size_t mk = 128 * 128;
    sparse_test::DeviceBuffer misaligned = sparse_test::DeviceBuffer::alloc(mk * sizeof(float) + 16);
    p.matA = static_cast<uint8_t*>(misaligned.get()) + 1;
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X17: AlgSetAttribute(SEARCH_ITERATIONS, 0) 于合法 algSelection → INVALID_VALUE ----
// （搜索输入域守卫；独立基座：16³ FP32 描述符 + 合法 algSelection，无需完整链）
TEST_F(MatmulSearchExceptionTest, X17SearchIterationsZeroRejected)
{
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::SparseLtMatDescGuard matA(handle.get(), 16, 16, 16, 16, ACL_FLOAT,
                                           ACL_SPARSE_ORDER_ROW,
                                           ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    sparse_test::SparseLtDnMatDescGuard matB(handle.get(), 16, 16, 16, 16, ACL_FLOAT,
                                             ACL_SPARSE_ORDER_ROW);
    sparse_test::SparseLtDnMatDescGuard matC(handle.get(), 16, 16, 16, 16, ACL_FLOAT,
                                             ACL_SPARSE_ORDER_ROW);
    sparse_test::SparseLtDnMatDescGuard matD(handle.get(), 16, 16, 16, 16, ACL_FLOAT,
                                             ACL_SPARSE_ORDER_ROW);
    sparse_test::SparseLtMatmulDescGuard mm(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
                                            ACL_SPARSE_OP_NON_TRANSPOSE,
                                            matA.get(), matB.get(), matC.get(), matD.get(),
                                            ACL_SPARSE_COMPUTE_32F);
    sparse_test::SparseLtAlgSelectionGuard algSel(handle.get(), mm.get(),
                                                  ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    int32_t zeroIters = 0;
    auto ret = aclsparseLtMatmulAlgSetAttribute(handle.get(), algSel.ptr(),
        ACLSPARSELT_MATMUL_SEARCH_ITERATIONS, &zeroIters, sizeof(int32_t));
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
    // 对照：合法值（>=1）应被接受（搜索域正向确认）
    int32_t validIters = 5;
    ret = aclsparseLtMatmulAlgSetAttribute(handle.get(), algSel.ptr(),
        ACLSPARSELT_MATMUL_SEARCH_ITERATIONS, &validIters, sizeof(int32_t));
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// ---- X18/X19 基座共用段：kit 四描述符 + matmulDesc 收养进基座成员。----
static void AdoptBatchDescKit(matmulsearch_test::SearchChainCtxBase& ctx,
                               sparse_test::LtBatchMatDescKit& kit)
{
    ctx.matA = std::move(kit.matA);
    ctx.matB = std::move(kit.matB);
    ctx.matC = std::move(kit.matC);
    ctx.matD = std::move(kit.matD);
    ctx.matmulDesc = std::move(kit.matmulDesc);
}

// ---- X18/X19 基座共用段：algSel + SEARCH_ITERATIONS 预置（PlanInit 前
// Set）+ plan + 流。tag 区分基座 throw 消息（两基座错误语义原样保留）。----
static void InitBatchCtxAlgSelItersPlanStream(
    matmulsearch_test::SearchChainCtxBase& ctx, const char* tag)
{
    ctx.algSel = std::make_unique<sparse_test::SparseLtAlgSelectionGuard>(
        ctx.handle.get(), ctx.matmulDesc->get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    int32_t iters = 5;
    if (aclsparseLtMatmulAlgSetAttribute(ctx.handle.get(), ctx.algSel->ptr(),
            ACLSPARSELT_MATMUL_SEARCH_ITERATIONS, &iters, sizeof(int32_t))
            != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(tag) + ": search iterations set failed");
    }
    ctx.plan = std::make_unique<sparse_test::SparseLtPlanGuard>(
        ctx.handle.get(), ctx.matmulDesc->get(), ctx.algSel->get());
    if (aclrtCreateStream(&ctx.stream) != ACL_SUCCESS) {
        throw std::runtime_error(std::string(tag) + ": aclrtCreateStream failed");
    }
}

// ---- X18 基座：A-sparse batch=3 全链 ----
// handle → mat 描述符（batch 属性须在 MatmulDescriptorInit 前设置于 4 个
// mat 描述符，一致性校验要求 A/B/C/D 相同）→ matmulDescr → algSel +
// SEARCH_ITERATIONS 预置 → plan → ws → 流与 B/C/D 缓冲。matA 不作为 Search
// 入参（X18 注入 matA=nullptr 触发 batch>1 回退域拒绝）。构造失败 throw
// （SearchExceptionCtx 同范式）。继承 SearchChainCtxBase 共享成员段
// （mm/ws/wsSize 即基类 matmulDesc/dWorkspace/workspaceSize，dA/dPruned
// 不分配——X18 不需要剪枝输入）。
// 四描述符 + batch 属性经 LtBatchMatDescKit（与 ltmatmul WB_L2 系共享）。
struct SearchBatchExceptionCtx : matmulsearch_test::SearchChainCtxBase {
    SearchBatchExceptionCtx()
        : SearchChainCtxBase()
    {
        InitMatDescriptors();
        InitAlgSelPlanAndBuffers();
    }

private:
    // mat 描述符 + batch 属性（numBatches=3 / batchStride=16384，须在
    // MatmulDescriptorInit 前设置于 4 个 mat 描述符）+ matmulDescr——
    // 共享 kit 构造后收养（kit 语义：四描述符一致性 batch 属性设置）。
    void InitMatDescriptors()
    {
        sparse_test::LtBatchMatDescKit kit(handle.get(), 3, 16384, 0,
            "SearchBatchExceptionCtx");
        AdoptBatchDescKit(*this, kit);
    }

    // algSel + SEARCH_ITERATIONS 预置（PlanInit 前 Set）+ plan + ws + 流与
    // B/C/D 缓冲（matA 不作为 Search 入参——X18 注入 nullptr）。algSel/
    // plan/流经共用段 helper（InitBatchCtxAlgSelItersPlanStream）。
    void InitAlgSelPlanAndBuffers()
    {
        InitBatchCtxAlgSelItersPlanStream(*this, "SearchBatchExceptionCtx");
        constexpr int32_t k = 128, n = 128;
        constexpr int32_t numBatches = 3;
        constexpr int64_t batchStride = 16384;
        const size_t total = static_cast<size_t>(numBatches) * batchStride;
        dB = sparse_test::DeviceBuffer::copyFrom(
            std::vector<float>(static_cast<size_t>(k) * n, 1.0f).data(),
            static_cast<size_t>(k) * n * sizeof(float));
        dC = sparse_test::DeviceBuffer::copyFrom(
            std::vector<float>(total, 0.5f).data(), total * sizeof(float));
        dD = sparse_test::DeviceBuffer::alloc(total * sizeof(float));
        // ws 预备（含 aPruned 区；回退前置语义由用例侧声明）
        if (aclsparseLtMatmulGetWorkspace(handle.get(), plan->cptr(), &workspaceSize)
                != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("SearchBatchExceptionCtx: GetWorkspace failed");
        }
        dWorkspace = sparse_test::DeviceBuffer::alloc(workspaceSize);
    }
};

// ---- X18: matA=nullptr × A-sparse × batch>1 → INVALID_VALUE（回退路径域收紧；
//      B3 单 batch 回退用例为正向对照，本用例为负向边界）----
TEST_F(MatmulSearchExceptionTest, X18MatANullWithBatchRejected)
{
    SearchBatchExceptionCtx ctx;
    // matA=nullptr 触发拒绝：numBatches>1 时 workspace aPruned 区为单 batch
    // 容量，kernel 按 batchStrideA 推进读取将越界（校验层拦截）。
    aclsparseLtMatmulPlan_t planRaw = ctx.plan->get();
    aclrtStream stream = ctx.stream;
    auto ret = aclsparseLtMatmulSearch(ctx.handle.get(), &planRaw,
        &ctx.alpha, nullptr, ctx.dB.get(), &ctx.beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- X19 基座：batch 布局溢出描述符（batchStride / ldB 可注入）----
// batchStride=2^61 × elemSize=4 的乘积（X19a，V3 字节溢出）/ 上界内巨 ld
// 的 (rows-1)*ld physSpan 跨度（X19b，V1 stride < span）在受检查计算下
// 干净拒绝（INVALID_VALUE）；内部 Matmul 的 batch 布局校验拦截后，
// Search 侧表现为全部候选执行失败 → EXECUTION_FAILED（快照恢复 +
// 无 UB/无设备错误）。ld > INT32_MAX 的更大值已由描述符层直接拒绝，
// 不经本链。
struct SearchBatchOverflowCtx : matmulsearch_test::SearchChainCtxBase {
    explicit SearchBatchOverflowCtx(int64_t batchStride, int64_t ldB)
        : SearchChainCtxBase()
    {
        InitOverflowDescriptors(batchStride, ldB);
        InitAlgSelPlanAndBuffers();
    }

private:
    // mat 描述符（batch 属性先于 MatmulDescriptorInit 设置于四矩阵）+
    // matmulDescr：batchStride / ldB（B dense）为注入的溢出布局参数——
    // 共享 kit（numBatches=2）构造后收养。
    void InitOverflowDescriptors(int64_t batchStride, int64_t ldB)
    {
        sparse_test::LtBatchMatDescKit kit(handle.get(), 2, batchStride, ldB,
            "SearchBatchOverflowCtx");
        AdoptBatchDescKit(*this, kit);
    }

    // algSel + SEARCH_ITERATIONS 预置（PlanInit 前 Set）+ plan + ws + 流与
    // A/B/C/D 缓冲（matA 为 A_pruned 占位；候选执行在校验层被拒绝，缓冲
    // 不会被实际读取——按单 batch 尺寸分配即可）。algSel/plan/流经共用段
    // helper（InitBatchCtxAlgSelItersPlanStream）。
    void InitAlgSelPlanAndBuffers()
    {
        InitBatchCtxAlgSelItersPlanStream(*this, "SearchBatchOverflowCtx");
        constexpr int32_t m = 128, k = 128, n = 128;
        const size_t mk = static_cast<size_t>(m) * k;
        const size_t kn = static_cast<size_t>(k) * n;
        const size_t mn = static_cast<size_t>(m) * n;
        dA = sparse_test::DeviceBuffer::copyFrom(
            std::vector<float>(mk, 1.0f).data(), mk * sizeof(float));
        dB = sparse_test::DeviceBuffer::copyFrom(
            std::vector<float>(kn, 1.0f).data(), kn * sizeof(float));
        dC = sparse_test::DeviceBuffer::copyFrom(
            std::vector<float>(mn, 0.5f).data(), mn * sizeof(float));
        dD = sparse_test::DeviceBuffer::alloc(mn * sizeof(float));
        if (aclsparseLtMatmulGetWorkspace(handle.get(), plan->cptr(), &workspaceSize)
                != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("SearchBatchOverflowCtx: GetWorkspace failed");
        }
        dWorkspace = sparse_test::DeviceBuffer::alloc(workspaceSize);
    }
};

// ---- X19a: batchStride=2^61（对齐通过，内部 Matmul 的 V3 字节溢出拒绝）→
//      全部候选失败 → EXECUTION_FAILED ----
TEST_F(MatmulSearchExceptionTest, X19aBatchStrideOverflowAllCandidatesRejected)
{
    SearchBatchOverflowCtx ctx(1LL << 61, 0);
    aclsparseLtMatmulPlan_t planRaw = ctx.plan->get();
    aclrtStream stream = ctx.stream;
    auto ret = aclsparseLtMatmulSearch(ctx.handle.get(), &planRaw,
        &ctx.alpha, ctx.dA.get(), ctx.dB.get(), &ctx.beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_EXECUTION_FAILED);
}

// ---- X19b: ld=INT32_MAX 上界值（2147483644，4 对齐）+ batchStride=2^37 <
//      V1 物理跨度（(128-1)*ld+128 ≈ 2.7e11）→ 内部 Matmul 的 V1 span
// 拒绝 → 全部候选失败 → EXECUTION_FAILED ----
// 演变说明：原 ld=2^58 形态测 V1 span 乘加链溢出；描述符层补 ld >
// INT32_MAX 拒绝后巨 ld 域被封死，该溢出面不再可达，
// V1 的受检查 span 计算保留为纵深防御。本用例改用上界内巨 ld 验证
// V1 的 span 语义拦截（batchStride < 物理跨度），与 X19a（V3 字节溢出）
// 拦截路径互补。
TEST_F(MatmulSearchExceptionTest, X19bBatchLdSpanOverflowAllCandidatesRejected)
{
    // ld=2147483644：INT32_MAX 内的最大 4 对齐值（描述符可创建，
    // FP32 dense 对齐/ROW ld>=cols 均通过）
    SearchBatchOverflowCtx ctx(1LL << 37, 2147483644);
    aclsparseLtMatmulPlan_t planRaw = ctx.plan->get();
    aclrtStream stream = ctx.stream;
    auto ret = aclsparseLtMatmulSearch(ctx.handle.get(), &planRaw,
        &ctx.alpha, ctx.dA.get(), ctx.dB.get(), &ctx.beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_EXECUTION_FAILED);
}
