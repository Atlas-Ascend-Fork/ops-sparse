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
 * @file matmulSearch_test_live.cpp
 * @brief TEST_F X31~X50——补齐组（X31~X37）+ 探针收编组（X38~X44）+
 *        探针收编组（X45~X50）。基座构造复用 matmulSearch_test_helpers.h
 *        共享层（SearchChainCtxBase / MakeSearchDescQuartet /
 *        InitExcChainDataAndWs / PruneCtxInput / SetVecAttrIf / SetSearchIters）。
 */

#include "test_common.h"

#include "matmulSearch_golden_verify.h"  // 共享层（含 test_helpers / wrapper / golden 链）

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace sparse_test;
using namespace matmulsearch_test;

// ============================================================================
// 补齐组（X31~X37）：N1 vec×nullptr / N2 vec×host 指针 / N4 库级 API /
// N9 SEARCH_ITERATIONS 默认值；N7 SPARSE_MAT_POINTER 组随属性
// 还原删除，编号 X36 空缺不重排）
// ============================================================================

// ---- X31-X34 共用基座：vec 属性按正规时序（PlanInit 前 SetAttribute——
// plan 的 tilingData 在 PlanInit 时固化 vec 编码，后置 Set 不进 td）。
// 描述符 → SetVecAttr → algSel + iters → plan → ws → prune → buffers。
// 构造段复用共享层 helper（与 SearchExceptionCtx 共用）。
struct VecSearchExceptionCtx : SearchChainCtxBase {
    // enableAlphaVec / enableBetaVec：ALPHA/BETA_VECTOR_SCALING 属性开关
    //（在 PlanInit 前设置——编码进 matD->alignment 高位并固化于 plan td）
    explicit VecSearchExceptionCtx(bool enableAlphaVec, bool enableBetaVec)
    {
        MakeSearchDescQuartet<float>(handle, 128, 128, 128, matA, matB, matC, matD);
        matmulDesc = std::make_unique<sparse_test::SparseLtMatmulDescGuard>(
            handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            matA->get(), matB->get(), matC->get(), matD->get(), ACL_SPARSE_COMPUTE_32F);
        // ★ vec 属性在 PlanInit 前设置（正规调用时序）
        SetVecAttrIf(handle, *matmulDesc, enableAlphaVec,
                     ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,
                     "VecSearchExceptionCtx: alpha vec set failed");
        SetVecAttrIf(handle, *matmulDesc, enableBetaVec,
                     ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING,
                     "VecSearchExceptionCtx: beta vec set failed");
        algSel = std::make_unique<sparse_test::SparseLtAlgSelectionGuard>(
            handle.get(), matmulDesc->get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
        SetSearchIters(handle, *algSel, 5, "VecSearchExceptionCtx: iters set failed");
        plan = std::make_unique<sparse_test::SparseLtPlanGuard>(
            handle.get(), matmulDesc->get(), algSel->get());
        if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
            throw std::runtime_error("VecSearchExceptionCtx: aclrtCreateStream failed");
        }
        InitExcChainDataAndWs(handle, *plan, 128, 128, 128, dA, dB, dC, dD, dPruned,
                              dWorkspace, workspaceSize);
        PruneCtxInput(handle, *matmulDesc, dA, dPruned, stream);
    }
};

// X31-X34 共用：Search 入口 + Matmul 直调对照（单次执行校验直接透出
// INVALID_VALUE）。expectedSearchRet 区分两类拦截位置（Search 入口
// 对 vec×nullptr 已作前置校验）：
//   nullptr（X31/X32）→ 入口校验 INVALID_VALUE（与文档返回码声明一致）；
//   host 指针（X33/X34）→ 入口仅拦 nullptr，32B 对齐/device 判定仍由
//   候选执行链内的 Matmul 执行期校验承载 → 全候选失败聚合
//   EXECUTION_FAILED。
inline void ExpectVecPointerRejected(VecSearchExceptionCtx& ctx,
    const void* alpha, const void* beta, aclsparseStatus_t expectedSearchRet,
    const char* tag)
{
    // Search 入口
    aclsparseLtMatmulPlan_t planRaw = ctx.plan->get();
    aclrtStream stream = ctx.stream;
    auto searchRet = aclsparseLtMatmulSearch(ctx.handle.get(), &planRaw,
        alpha, ctx.dPruned.get(), ctx.dB.get(), beta,
        ctx.dC.get(), ctx.dD.get(), ctx.dWorkspace.get(), &stream, 1);
    EXPECT_EQ(searchRet, expectedSearchRet) << tag << ": Search";
    // Matmul 直调对照：拦截码直接透出
    aclsparseLtConstMatmulPlan_t planC = ctx.plan->get();
    auto matmulRet = aclsparseLtMatmul(ctx.handle.get(), &planC,
        alpha, ctx.dPruned.get(), ctx.dB.get(), beta,
        ctx.dC.get(), ctx.dD.get(), ctx.dWorkspace.get(), &stream, 1);
    EXPECT_EQ(matmulRet, ACL_SPARSE_STATUS_INVALID_VALUE) << tag << ": Matmul";
}

// ---- X31 (N1)：α-vec=1 ∧ alpha=nullptr → 拒绝（U2 X5 ①）。NVIDIA 契约：
//      向量缩放启用时对应指针必须为有效 device float[M] 向量；vec 模式 null
//      属参数错误（标量模式 α/β null 容忍为已声明超集，B6/B7 正向对照）。
//      对照合法域：α-vec=1 ∧ β-vec=0 ∧ β=null 合法（CSV case227）。
//      Search 入口前置校验（与 Matmul 侧
//      validate_vector_scaling_pointers nullptr 段同构）→ INVALID_VALUE
//      （此前经候选执行失败聚合为 EXECUTION_FAILED，与文档声明不符）----
TEST_F(MatmulSearchExceptionTest, X31AlphaVecNullptrRejected)
{
    VecSearchExceptionCtx ctx(/*enableAlphaVec=*/true, /*enableBetaVec=*/false);
    ExpectVecPointerRejected(ctx, /*alpha=*/nullptr, /*beta=*/&ctx.beta,
                             ACL_SPARSE_STATUS_INVALID_VALUE, "X31");
}

// ---- X32 (N1)：β-vec=1（隐含提升 α-vec）∧ beta=nullptr → 拒绝（U2 X5 ②）。
//      alpha 传有效 device 指针（隔离 β 分支的拦截语义）。入口前置校验
//      → INVALID_VALUE（同 X31）----
TEST_F(MatmulSearchExceptionTest, X32BetaVecNullptrRejected)
{
    VecSearchExceptionCtx ctx(/*enableAlphaVec=*/false, /*enableBetaVec=*/true);
    ExpectVecPointerRejected(ctx, /*alpha=*/ctx.dB.get(), /*beta=*/nullptr,
                             ACL_SPARSE_STATUS_INVALID_VALUE, "X32");
}

// ---- X33 (N2)：α-vec=1 ∧ alpha=host 栈指针 → 拒绝（既有防御零覆盖
//      收口：aclrtPointerGetAttributes 非 Device 内存提前拒绝，替代 GM 侧
//      MTE 非法访问的 AI Core Error 设备级崩溃）。入口仅拦 nullptr，
//      host 指针由候选执行链拦截 → 全候选失败聚合 EXECUTION_FAILED----
TEST_F(MatmulSearchExceptionTest, X33AlphaVecHostPointerRejected)
{
    VecSearchExceptionCtx ctx(/*enableAlphaVec=*/true, /*enableBetaVec=*/false);
    float stackAlpha[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    ExpectVecPointerRejected(ctx, /*alpha=*/stackAlpha, /*beta=*/&ctx.beta,
                             ACL_SPARSE_STATUS_EXECUTION_FAILED, "X33");
}

// ---- X34 (N2)：β-vec=1 ∧ beta=host 堆指针 → 拒绝（heap 变异；alpha 有效）。
//      同 X33：候选执行链聚合 EXECUTION_FAILED----
TEST_F(MatmulSearchExceptionTest, X34BetaVecHostPointerRejected)
{
    VecSearchExceptionCtx ctx(/*enableAlphaVec=*/false, /*enableBetaVec=*/true);
    auto* heapBeta = new float[128];
    for (int i = 0; i < 128; ++i) { heapBeta[i] = 0.5f; }
    ExpectVecPointerRejected(ctx, /*alpha=*/ctx.dB.get(), /*beta=*/heapBeta,
                             ACL_SPARSE_STATUS_EXECUTION_FAILED, "X34");
    delete[] heapBeta;
}

// ---- X35 (N4)：库级 API 四函数 + Init *handle 非空防御（G0 功能面收口）。
//      GetVersion == MAJOR*10000+MINOR*100+PATCH（三分量经 GetProperty 自洽
//      断言，不硬编码版本值）；GetProperty 非法枚举拒绝；ErrorName/String
//      含未识别码 "unrecognized error code" ----
TEST_F(MatmulSearchExceptionTest, X35LibraryLevelApiSmoke)
{
    // Init *handle 非空防御：已有有效句柄上重复 Init → INVALID_VALUE（防覆盖泄漏）
    sparse_test::SparseLtHandleGuard handle;
    aclsparseLtHandle_t handleAlias = *handle.get();
    EXPECT_EQ(aclsparseLtInit(&handleAlias), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(handleAlias, *handle.get());  // 原句柄未被覆盖

    // GetVersion：MAJOR*10000+MINOR*100+PATCH 编码（分量经 GetProperty 自洽）
    int major = -1, minor = -1, patch = -1;
    ASSERT_EQ(aclsparseLtGetProperty(ACLSPARSELT_MAJOR_VERSION, &major),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtGetProperty(ACLSPARSELT_MINOR_VERSION, &minor),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtGetProperty(ACLSPARSELT_PATCH_LEVEL, &patch),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_GE(major, 0);
    EXPECT_GE(minor, 0);
    EXPECT_GE(patch, 0);
    int version = 0;
    ASSERT_EQ(aclsparseLtGetVersion(handle.get(), &version), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(version, major * 10000 + minor * 100 + patch);

    // GetProperty 非法枚举 → INVALID_VALUE
    int dummy = 0;
    auto badProp = static_cast<aclsparseLtLibraryPropertyType_t>(99);
    EXPECT_EQ(aclsparseLtGetProperty(badProp, &dummy), ACL_SPARSE_STATUS_INVALID_VALUE);
    // GetProperty null value → INVALID_VALUE
    EXPECT_EQ(aclsparseLtGetProperty(ACLSPARSELT_MAJOR_VERSION, nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    // GetVersion null handle / null version → 拒绝
    EXPECT_NE(aclsparseLtGetVersion(nullptr, &version), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseLtGetVersion(handle.get(), nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);

    // GetErrorName / GetErrorString：已识别码 + 未识别码
    EXPECT_STREQ(aclsparseLtGetErrorName(ACL_SPARSE_STATUS_SUCCESS),
                 "ACL_SPARSE_STATUS_SUCCESS");
    EXPECT_STREQ(aclsparseLtGetErrorName(ACL_SPARSE_STATUS_INVALID_VALUE),
                 "ACL_SPARSE_STATUS_INVALID_VALUE");
    EXPECT_STREQ(aclsparseLtGetErrorString(ACL_SPARSE_STATUS_SUCCESS),
                 "The operation completed successfully");
    const auto unknownCode = static_cast<aclsparseStatus_t>(9999);
    EXPECT_STREQ(aclsparseLtGetErrorName(unknownCode), "unrecognized error code");
    EXPECT_STREQ(aclsparseLtGetErrorString(unknownCode), "unrecognized error code");
}

// ---- X37 (N9)：不 Set SEARCH_ITERATIONS → Search SUCCESS →
//      AlgGetAttribute(SEARCH_ITERATIONS) 回读 == 5（默认值端到端 + Get 回读
//      零覆盖收口；全部既有 CSV 用例显式 Set，默认路径此前未端到端走过）----
TEST_F(MatmulSearchExceptionTest, X37SearchIterationsDefaultRoundtrip)
{
    // SearchExceptionCtx 基座不 Set iters（默认 5）——直接以默认值走 Search
    SearchExceptionCtx ctx;
    SearchRawParams p = DefaultRawParams(ctx);
    EXPECT_EQ(SearchRaw(p), ACL_SPARSE_STATUS_SUCCESS);
    // Get 回读默认值
    int32_t gotIters = -1;
    EXPECT_EQ(aclsparseLtMatmulAlgGetAttribute(ctx.handle.get(), ctx.algSel->cptr(),
        ACLSPARSELT_MATMUL_SEARCH_ITERATIONS, &gotIters, sizeof(int32_t)),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(gotIters, 5);
    // 对照：显式 Set 后回读新值（get 分支写路径）
    int32_t newIters = 3;
    EXPECT_EQ(aclsparseLtMatmulAlgSetAttribute(ctx.handle.get(), ctx.algSel->ptr(),
        ACLSPARSELT_MATMUL_SEARCH_ITERATIONS, &newIters, sizeof(int32_t)),
        ACL_SPARSE_STATUS_SUCCESS);
    gotIters = -1;
    EXPECT_EQ(aclsparseLtMatmulAlgGetAttribute(ctx.handle.get(), ctx.algSel->cptr(),
        ACLSPARSELT_MATMUL_SEARCH_ITERATIONS, &gotIters, sizeof(int32_t)),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(gotIters, 3);
}

// ============================================================================
// 探针收编组（X38~X44）：Y1 vec 活读时序 / Y2 dense 非紧凑 ld 直读 /
// Y3 bias 防御校验——临时探针（probe_y1_live_read /
// probe_y2_pack_bigshape / probe_y3_bias_negative，51 断言全绿）的正式
// 用例化。分支覆盖（算子侧
// 执行期路径）：
//   X38/X39/X44 → matmul_host fill_matmul_tiling_fields / validate_vector_scaling_
//      pointers 的活读分支（PlanInit 后开 / 关 vec；X44 另覆盖 biasChunkMode
//      活读重算降级路径——fill_matmul_tiling_fields 执行期重调
//      compute_bias_chunk_mode）；
//   X40/X41 → cube GM 布局直通行距（td.ldB 直通 NDExt/DNExt 行距参数，
//      MTE2 按行距跳过行尾 pad；matmul_host / matmul_search_host 两条
//      执行链）+ 紧凑零 padding 锚点（ldB == n 时布局参数与紧凑形态
//      逐位相同）；
//   X42/X43 → validate_bias_params 的 B1（32B 对齐）/ B2（stride >= 0）分支
//      + 合法域直通（广播 stride=0）。
// ============================================================================

// 状态类断言记录（返回码/防御判定 → JSONL 落盘；数值类经
// VerifyFloatSearchOutput 落盘，与 CSV 用例同一断言日志）
inline void Record22DStatusCheck(const std::string& caseId, const std::string& item,
    const std::string& measured, const std::string& expected, bool ok)
{
    SearchAssertionRecord rec;
    rec.caseId = caseId;
    rec.tensor = item;
    rec.metric = "declarative(expected-status, probe-converted)";
    rec.measured = measured;
    rec.threshold = expected;
    rec.verdict = VerdictOf(ok);
    AppendSearchRecord(rec);
}

// X38/X39/X42/X43 共用基座：FP32 128³ A-sparse 随机数据全链 + 剪枝回读
//（NPU 剪枝回读范式：golden 与执行共用同一剪枝产物）。prePlanAlphaVec=true
// 时 ALPHA_VECTOR_SCALING 于 PlanInit 前置 1（plan 快照含 vec-on，供 X39
//「后关」活读验证）；false = PlanInit 时无 vec（X38/X42/X43 基座）。
// SEARCH_ITERATIONS 预置 1（X42/X43 的 Search 入口子断言用，控制耗时）。
// X45~X50 扩展：模板参数 T 参数化 dtype（默认 float；X47/X48 以
// uint16_t 实例化 FP16 链），尾参 initBatches/splitK/dataBatches 参数化
// batch 属性（MatmulDescriptorInit 前置）与 two-stage 计划——既有 FP32
// 128³ 调用点显式 <float> 实例化。构造段见
// InitLiveReadInputs / InitLiveReadPlan / AllocLiveReadBuffers；
// 描述符/数据/ws 段复用共享层 helper（与 SearchExceptionCtx 共用）。
template <typename T = float>
struct LiveReadChainCtx : SearchChainCtxBase {
    // 默认维度 128³（X38/X39/X42/X43 基座）；大 m / 大 k 场景经 ctor 尾参覆盖
    static constexpr int32_t kM = 128;
    static constexpr int32_t kK = 128;
    static constexpr int32_t kN = 128;

    int32_t m = kM;             // 实例维度（golden/回读尺寸按此计算）
    int32_t k = kK;
    int32_t n = kN;
    int32_t batches = 1;        // 描述符活读 batch 数（SetLiveBatches 同步更新）
    int32_t allocBatches = 1;   // 数据/设备缓冲分配容量（batch 排布，>= batches）
    std::vector<T> hA;          // 随机输入（allocBatches*m*k，剪枝前）
    std::vector<T> hB;          // 随机输入（allocBatches*k*n）
    std::vector<T> hC;          // 随机输入（allocBatches*m*n）
    std::vector<T> hApruned;    // 剪枝回读（batches*m*k，golden 基准）

    // dimM/dimK/dimN/batch/splitK 尾置默认参数：既有 128³ 单 batch fused
    // 调用点零变化（X44 传大 m；X45/X46 传 batch/splitK）。
    //   initBatches>1：四矩阵 NUM_BATCHES + per-matrix 紧凑 BATCH_STRIDE 于
    //     aclsparseLtMatmulDescriptorInit 前设置（跨矩阵一致性校验要求）；
    //   splitK>1：AlgSelection 预置 SPLIT_K + TWO_KERNELS（two-stage 计划，
    //     PlanInit 冻结 temp 区需求——Z1 workspace 快照语义的触发前提）；
    //   dataBatches：数据/设备缓冲容量（活读 batch 增大场景预留，防止快照
    //     失配拒绝路径意外放行时 kernel 越出用例自有缓冲；0 = 取 initBatches）。
    explicit LiveReadChainCtx(uint32_t seed, bool prePlanAlphaVec = false,
                              int32_t dimM = kM, int32_t dimK = kK, int32_t dimN = kN,
                              int32_t initBatches = 1, int32_t splitK = 1,
                              int32_t dataBatches = 0)
        : m(dimM), k(dimK), n(dimN),
          batches(initBatches > 1 ? initBatches : 1),
          allocBatches(std::max(batches, dataBatches))
    {
        InitLiveReadInputs(seed);
        MakeSearchDescQuartet<T>(handle, m, k, n, matA, matB, matC, matD);
        if (batches > 1 &&
            SetLiveBatches(batches) != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("LiveReadChainCtx: batch attrs set failed");
        }
        matmulDesc = std::make_unique<sparse_test::SparseLtMatmulDescGuard>(
            handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            matA->get(), matB->get(), matC->get(), matD->get(),
            sparse_test::MatmulComputeTrait<T>::kComputeType);
        SetVecAttrIf(handle, *matmulDesc, prePlanAlphaVec,
                     ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,
                     "LiveReadChainCtx: pre-plan alpha vec set failed");
        algSel = std::make_unique<sparse_test::SparseLtAlgSelectionGuard>(
            handle.get(), matmulDesc->get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
        SetSearchIters(handle, *algSel, 1, "LiveReadChainCtx: search iterations set failed");
        if (splitK > 1) {
            const int32_t twoKernels = ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS;
            if (aclsparseLtMatmulAlgSetAttribute(handle.get(), algSel->ptr(),
                    ACLSPARSELT_MATMUL_SPLIT_K, &splitK, sizeof(int32_t))
                    != ACL_SPARSE_STATUS_SUCCESS
                || aclsparseLtMatmulAlgSetAttribute(handle.get(), algSel->ptr(),
                    ACLSPARSELT_MATMUL_SPLIT_K_MODE, &twoKernels, sizeof(int32_t))
                    != ACL_SPARSE_STATUS_SUCCESS) {
                throw std::runtime_error("LiveReadChainCtx: splitK attrs set failed");
            }
        }
        plan = std::make_unique<sparse_test::SparseLtPlanGuard>(
            handle.get(), matmulDesc->get(), algSel->get());
        if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
            throw std::runtime_error("LiveReadChainCtx: aclrtCreateStream failed");
        }
        AllocLiveReadBuffers();
        PruneCtxInput(handle, *matmulDesc, dA, dPruned, stream);
        // 剪枝按 ctor 时活读 batch 属性产出（prune 单调用化：一次调用处理
        // 全 batch），回读 batches*mk；batch 增大场景（X46）恢复后沿用
        // 构造时的 1-batch 剪枝产物
        hApruned.resize(static_cast<size_t>(batches) * static_cast<size_t>(m) * k);
        dPruned.copyToHost(hApruned.data(), hApruned.size() * NpuDtypeTrait<T>::kEltSize);
    }

    LiveReadChainCtx(const LiveReadChainCtx&) = delete;
    LiveReadChainCtx& operator=(const LiveReadChainCtx&) = delete;

    // 多 batch 数据按 (allocBatches*rows, cols) 整体生成——紧凑 batch 排布
    // 下每 batch 即一段连续 rows*cols
    void InitLiveReadInputs(uint32_t seed)
    {
        hA = MatmulGenTestMatrix<T>(static_cast<int64_t>(allocBatches) * m, k,
                                    -1.0f, 1.0f, seed);
        hB = MatmulGenTestMatrix<T>(static_cast<int64_t>(allocBatches) * k, n,
                                    -1.0f, 1.0f, seed + 1);
        hC = MatmulGenTestMatrix<T>(static_cast<int64_t>(allocBatches) * m, n,
                                    -1.0f, 1.0f, seed + 2);
    }

    // 设备缓冲（allocBatches 容量）+ GetWorkspace + ws 分配。
    void AllocLiveReadBuffers()
    {
        constexpr size_t kElt = NpuDtypeTrait<T>::kEltSize;
        const size_t mk = static_cast<size_t>(m) * k;
        const size_t kn = static_cast<size_t>(k) * n;
        const size_t mn = static_cast<size_t>(m) * n;
        dA = sparse_test::DeviceBuffer::copyFrom(hA.data(), allocBatches * mk * kElt);
        dB = sparse_test::DeviceBuffer::copyFrom(hB.data(), allocBatches * kn * kElt);
        dC = sparse_test::DeviceBuffer::copyFrom(hC.data(), allocBatches * mn * kElt);
        dD = sparse_test::DeviceBuffer::alloc(allocBatches * mn * kElt);
        dPruned = sparse_test::DeviceBuffer::alloc(allocBatches * mk * kElt);
        aclsparseStatus_t wsRet =
            aclsparseLtMatmulGetWorkspace(handle.get(), plan->cptr(), &workspaceSize);
        if (wsRet != ACL_SPARSE_STATUS_SUCCESS || workspaceSize == 0) {
            throw std::runtime_error("LiveReadChainCtx: GetWorkspace failed");
        }
        dWorkspace = sparse_test::DeviceBuffer::alloc(workspaceSize);
    }

    // 活读 batch 属性设置（X46）：四矩阵 NUM_BATCHES + per-matrix 紧凑
    // BATCH_STRIDE（A=m*k、B=k*n、C/D=m*n，与 aclsparseLtMatDescriptorInit
    // 前置设置同式），并同步 batches 活读态
    aclsparseStatus_t SetLiveBatches(int32_t numBatches)
    {
        const int64_t strides[4] = {static_cast<int64_t>(m) * k,
                                    static_cast<int64_t>(k) * n,
                                    static_cast<int64_t>(m) * n,
                                    static_cast<int64_t>(m) * n};
        aclsparseLtMatDescriptor_t mats[4] = {matA->get(), matB->get(),
                                              matC->get(), matD->get()};
        for (int32_t i = 0; i < 4; ++i) {
            aclsparseStatus_t st = aclsparseLtMatDescSetAttribute(handle.get(), &mats[i],
                ACLSPARSELT_MAT_NUM_BATCHES, &numBatches, sizeof(numBatches));
            if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
            st = aclsparseLtMatDescSetAttribute(handle.get(), &mats[i],
                ACLSPARSELT_MAT_BATCH_STRIDE, &strides[i], sizeof(strides[i]));
            if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
        }
        batches = numBatches;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    // Matmul 直调（α/β 指针由用例注入——活读时序场景的变量）
    aclsparseStatus_t RunMatmul(const void* alpha, const void* beta)
    {
        aclsparseLtConstMatmulPlan_t cplan = plan->get();
        return aclsparseLtMatmul(handle.get(), &cplan, alpha, dPruned.get(), dB.get(),
                                 beta, dC.get(), dD.get(), dWorkspace.get(), &stream, 1);
    }

    // Search 直调（候选执行链共用同一执行期校验路径）
    aclsparseStatus_t RunSearch(const void* alpha, const void* beta)
    {
        aclsparseLtMatmulPlan_t planRaw = plan->get();
        return aclsparseLtMatmulSearch(handle.get(), &planRaw, alpha, dPruned.get(),
                                       dB.get(), beta, dC.get(), dD.get(),
                                       dWorkspace.get(), &stream, 1);
    }

    // 读回 matD 的 batch 0 段（m*n；既有用例语义）
    std::vector<T> ReadD() const { return ReadDBatch(0); }

    // 读回 matD 指定 batch 段（batch 排布下第 batchIdx 个 m*n 段）
    std::vector<T> ReadDBatch(int32_t batchIdx) const
    {
        const size_t mn = static_cast<size_t>(m) * n;
        const size_t bytes = mn * sparse_test::NpuDtypeTrait<T>::kEltSize;
        std::vector<T> out(mn);
        const void* src = static_cast<const uint8_t*>(dD.raw())
            + static_cast<size_t>(batchIdx) * bytes;
        if (aclrtMemcpy(out.data(), bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST)
                != ACL_SUCCESS) {
            throw std::runtime_error("LiveReadChainCtx: ReadDBatch memcpy failed");
        }
        return out;
    }
};

// golden（FP32 域，NPU 剪枝回读范式）：D = α（标量或逐行向量）·(A_pruned×B)
// + bias[i]；β 恒 nullptr（语义 0.0）——X38/X39/X42/X46 用例统一（X44 双 vec
// 走 MatmulAlphaBetaFp32Vec 组合，见用例内）。维度取 ctx 实例值（X44 大 m）；
// 限 batches==1（hApruned/hB 为单 batch 剪枝产物；X45 双 batch 走
// ComputeBatchGoldenFp32 按段组装）。
inline std::vector<float> ComputeLiveReadGolden(const LiveReadChainCtx<float>& c,
    float alphaScalar, const std::vector<float>* alphaVec, const std::vector<float>* biasVec)
{
    std::vector<float> Df;
    MatmulAlphaBetaFp32(c.hApruned, c.hB, c.hC, Df, c.m, c.k, c.n, 1.0f, 0.0f);
    for (int32_t i = 0; i < c.m; ++i) {
        const float aV = (alphaVec != nullptr)
            ? (*alphaVec)[static_cast<size_t>(i)] : alphaScalar;
        for (int32_t j = 0; j < c.n; ++j) {
            Df[static_cast<size_t>(i) * c.n + j] *= aV;
        }
    }
    if (biasVec != nullptr && !biasVec->empty()) {
        ApplyBiasFp32(Df, *biasVec, c.m, c.n);
    }
    return Df;
}

// ---- X38 (Y1/P1+P2+P3)：PlanInit 后开 ALPHA_VECTOR_SCALING + device 向量 →
//      向量分支执行正确（逐行 α 数值正确）。快照语义下该组合为「标量分支
//      host 解引用 device 指针」的崩溃域——活读语义消除；附带活读 validate
//      联动断言：vec 开启态 host 栈指针 / nullptr → INVALID_VALUE（校验按
//      活读状态生效，而非 PlanInit 快照）----
TEST_F(MatmulSearchExceptionTest, X38LiveReadVecOnAfterPlanInit)
{
    LiveReadChainCtx<float> ctx(20260381u);
    int enable = 1;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &enable, sizeof(int)),
        ACL_SPARSE_STATUS_SUCCESS) << "post-PlanInit SetAttribute(ALPHA_VEC=1)";
    const auto alphaVec = GenScalingVector(ctx.m, 20260382u);
    sparse_test::DeviceBuffer dAlphaVec = sparse_test::DeviceBuffer::copyFrom(
        alphaVec.data(), alphaVec.size() * sizeof(float));
    aclsparseStatus_t st = ctx.RunMatmul(dAlphaVec.get(), nullptr);
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS) << "vec-on + device vec Matmul";
    ASSERT_EQ(aclrtSynchronizeStream(ctx.stream), ACL_SUCCESS);
    const auto hD = ctx.ReadD();
    const auto golden = ComputeLiveReadGolden(ctx, 1.0f, &alphaVec, nullptr);
    EXPECT_TRUE(VerifyFloatSearchOutput<float>(golden, hD, "X38-vec-on",
                "matD(live-read-vector-branch)"));
    // 活读 validate 联动（探针 P2）：vec 开启态 alpha=host 栈指针 → 拒绝
    float hostAlpha = 1.5f;
    st = ctx.RunMatmul(&hostAlpha, nullptr);
    const bool p2Ok = (st == ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_TRUE(p2Ok) << "vec-on + host-ptr alpha, ret="
                      << static_cast<int32_t>(st);
    Record22DStatusCheck("X38", "alpha(vec-on,host-ptr)",
        "ret=" + std::to_string(static_cast<int32_t>(st)), "INVALID_VALUE", p2Ok);
    // 活读 validate 联动（探针 P3）：vec 开启态 alpha=nullptr → 拒绝
    st = ctx.RunMatmul(nullptr, nullptr);
    const bool p3Ok = (st == ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_TRUE(p3Ok) << "vec-on + null alpha, ret=" << static_cast<int32_t>(st);
    Record22DStatusCheck("X38", "alpha(vec-on,nullptr)",
        "ret=" + std::to_string(static_cast<int32_t>(st)), "INVALID_VALUE", p3Ok);
}

// ---- X39 (Y1/P4)：PlanInit（快照 vec-on）后关 ALPHA_VECTOR_SCALING → 标量
//      分支执行正确（host 标量 α 数值正确；α=2.5 区分缩放真实生效）----
TEST_F(MatmulSearchExceptionTest, X39LiveReadVecOffAfterPlanInit)
{
    LiveReadChainCtx<float> ctx(20260391u, /*prePlanAlphaVec=*/true);
    int disable = 0;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &disable, sizeof(int)),
        ACL_SPARSE_STATUS_SUCCESS) << "post-PlanInit SetAttribute(ALPHA_VEC=0)";
    const float alphaScalar = 2.5f;
    aclsparseStatus_t st = ctx.RunMatmul(&alphaScalar, nullptr);
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS) << "vec-off + host scalar Matmul";
    ASSERT_EQ(aclrtSynchronizeStream(ctx.stream), ACL_SUCCESS);
    const auto hD = ctx.ReadD();
    const auto golden = ComputeLiveReadGolden(ctx, alphaScalar, nullptr, nullptr);
    EXPECT_TRUE(VerifyFloatSearchOutput<float>(golden, hD, "X39-vec-off",
                "matD(live-read-scalar-branch)"));
}

// pad 列污染值（探针方法论：cube GM 布局直读 ld 时若误读 pad 列，输出
// 立即远超容差而被数值断言捕获——Y2 实现报告「过程发现」节实证有效）。
// FP16 用 +Inf 位模式：被剪枝零元素相乘亦得 NaN（inf×0），任意 pad 读取
// 均可检。
template <typename T>
inline T DensePackPadPoison()
{
    if constexpr (std::is_same_v<T, float>) {
        return 9.99e9f;
    } else {
        return static_cast<T>(0x7C00U);  // FP16 +Inf 位模式
    }
}

// Y2 场景 runner（探针 S1/S2 正式化）：大 viewCols 非紧凑布局端到端。
//   - B 物理 (k, ldB) ROW：有效列 [0,n) 随机，pad 列 [n,ldB) 污染值；
//   - withSearch=true：aclsparseLtMatmulSearch 端到端（iters=1——覆盖
//     matmul_search_host 的候选执行链：COL 翻转决策 + 全候选直接复用
//     调用方原始指针）+ aclsparseLtMatmul 直调（matmul_host 执行链）
//     双入口数值验证；
//   - golden：A_pruned 回读 × B_logical（仅有效列），FP32 累加域。
// 链/缓冲段见 DensePackScenarioCtx、pad 布局段见
// MakeDensePackPaddedB、双入口验证段见 VerifyDensePackEntry。
namespace {

// pad 污染物理 B 生成：有效列 [0,n) 取 hBlogical，pad 列 [n,ldB) 污染值。
template <typename T>
std::vector<T> MakeDensePackPaddedB(const std::vector<T>& hBlogical,
    int32_t k, int32_t n, int64_t ldB)
{
    std::vector<T> hBphys(static_cast<size_t>(k) * static_cast<size_t>(ldB), T{0});
    for (int32_t p = 0; p < k; ++p) {
        for (int64_t j = 0; j < ldB; ++j) {
            hBphys[static_cast<size_t>(p) * static_cast<size_t>(ldB)
                   + static_cast<size_t>(j)] =
                (j < n) ? hBlogical[static_cast<size_t>(p) * n + j]
                        : DensePackPadPoison<T>();
        }
    }
    return hBphys;
}

// Y2 链基座：描述符（B dense 带 ldB）+ algSel(iters=1) + plan + ws + buffers
// + prune + 剪枝回读。
template <typename T>
struct DensePackScenarioCtx {
    static constexpr int32_t kIters = 1;

    sparse_test::SparseLtHandleGuard handle;
    sparse_test::SparseLtMatDescGuard matA;
    sparse_test::SparseLtDnMatDescGuard matB;
    sparse_test::SparseLtDnMatDescGuard matC;
    sparse_test::SparseLtDnMatDescGuard matD;
    std::unique_ptr<sparse_test::SparseLtMatmulDescGuard> mm;
    std::unique_ptr<sparse_test::SparseLtAlgSelectionGuard> algSel;
    std::unique_ptr<sparse_test::SparseLtPlanGuard> plan;
    sparse_test::DeviceBuffer ws;
    size_t wsBytes = 0;
    sparse_test::DeviceBuffer dA, dB, dC, dD, dPruned;
    sparse_test::SearchStreamGuard streamGuard;
    std::vector<T> hApruned;

    DensePackScenarioCtx(int32_t m, int32_t k, int32_t n, int64_t ldB,
        const std::vector<T>& hA, const std::vector<T>& hBphys, const std::vector<T>& hC,
        const std::string& tag)
        : matA(handle.get(), m, k, k, 16, sparse_test::NpuDtypeTrait<T>::kAclDtype,
               ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT),
          matB(handle.get(), k, n, ldB, 16, sparse_test::NpuDtypeTrait<T>::kAclDtype,
               ACL_SPARSE_ORDER_ROW),
          matC(handle.get(), m, n, n, 16, sparse_test::NpuDtypeTrait<T>::kAclDtype,
               ACL_SPARSE_ORDER_ROW),
          matD(handle.get(), m, n, n, 16, sparse_test::NpuDtypeTrait<T>::kAclDtype,
               ACL_SPARSE_ORDER_ROW),
          streamGuard()
    {
        constexpr size_t kElt = sparse_test::NpuDtypeTrait<T>::kEltSize;
        const size_t mk = static_cast<size_t>(m) * k;
        const size_t mn = static_cast<size_t>(m) * n;
        mm = std::make_unique<sparse_test::SparseLtMatmulDescGuard>(
            handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            matA.get(), matB.get(), matC.get(), matD.get(),
            sparse_test::MatmulComputeTrait<T>::kComputeType);
        algSel = std::make_unique<sparse_test::SparseLtAlgSelectionGuard>(
            handle.get(), mm->get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
        SetSearchIters(handle, *algSel, kIters, "DensePackScenarioCtx: iters set failed");
        plan = std::make_unique<sparse_test::SparseLtPlanGuard>(
            handle.get(), mm->get(), algSel->get());
        if (aclsparseLtMatmulGetWorkspace(handle.get(), plan->cptr(), &wsBytes)
                != ACL_SPARSE_STATUS_SUCCESS || wsBytes == 0) {
            throw std::runtime_error(tag + ": GetWorkspace failed");
        }
        ws = sparse_test::DeviceBuffer::alloc(wsBytes);
        dA = sparse_test::DeviceBuffer::copyFrom(hA.data(), mk * kElt);
        dB = sparse_test::DeviceBuffer::copyFrom(hBphys.data(), hBphys.size() * kElt);
        dC = sparse_test::DeviceBuffer::copyFrom(hC.data(), mn * kElt);
        dD = sparse_test::DeviceBuffer::alloc(mn * kElt);
        dPruned = sparse_test::DeviceBuffer::alloc(mk * kElt);

        aclsparseLtConstMatmulDescriptor_t cmd = mm->get();
        if (aclsparseLtSpMMAPrune(handle.get(), &cmd, dA.get(), dPruned.get(),
                                  ACLSPARSELT_PRUNE_SPMMA_STRIP, streamGuard.get())
                != ACL_SPARSE_STATUS_SUCCESS ||
            aclrtSynchronizeStream(streamGuard.get()) != ACL_SUCCESS) {
            throw std::runtime_error(tag + ": prune failed");
        }
        hApruned.resize(mk);
        dPruned.copyToHost(hApruned.data(), mk * kElt);
    }
};

// Y2 单入口执行 + 数值验证（Search 或 Matmul；entryTag = "Search"/"Matmul"）。
template <typename T>
void VerifyDensePackEntry(DensePackScenarioCtx<T>& c, const std::string& tag,
    const std::string& entryTag, bool viaSearch, const std::vector<T>& hGolden)
{
    constexpr size_t kElt = sparse_test::NpuDtypeTrait<T>::kEltSize;
    const size_t mn = hGolden.size();
    sparse_test::DeviceBuffer dOut = sparse_test::DeviceBuffer::alloc(mn * kElt);
    const float alpha = 1.0f;
    aclrtStream stream = c.streamGuard.get();
    aclsparseStatus_t st;
    if (viaSearch) {
        aclsparseLtMatmulPlan_t planRaw = c.plan->get();
        st = aclsparseLtMatmulSearch(c.handle.get(), &planRaw, &alpha,
            c.dPruned.get(), c.dB.get(), nullptr, c.dC.get(), dOut.get(),
            c.ws.get(), &stream, 1);
    } else {
        aclsparseLtConstMatmulPlan_t cplan = c.plan->get();
        st = aclsparseLtMatmul(c.handle.get(), &cplan, &alpha,
            c.dPruned.get(), c.dB.get(), nullptr, c.dC.get(), dOut.get(),
            c.ws.get(), &stream, 1);
    }
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS) << tag << ": " << entryTag << " failed";
    ASSERT_EQ(aclrtSynchronizeStream(c.streamGuard.get()), ACL_SUCCESS);
    std::vector<T> hOut(mn);
    dOut.copyToHost(hOut.data(), mn * kElt);
    EXPECT_TRUE(VerifyFloatSearchOutput<T>(hGolden, hOut, tag,
                viaSearch ? "matD.search(ld-direct-read)" : "matD.matmul(ld-direct-read)"));
}

}  // namespace

template <typename T>
inline void RunDensePackBigShapeScenario(const std::string& tag, int32_t m, int32_t k,
    int32_t n, int64_t ldB, uint32_t seed, bool withSearch)
{
    const size_t mn = static_cast<size_t>(m) * n;

    const auto hA = MatmulGenTestMatrix<T>(m, k, -1.0f, 1.0f, seed);
    const auto hBlogical = MatmulGenTestMatrix<T>(k, n, -1.0f, 1.0f, seed + 1);
    const auto hC = MatmulGenTestMatrix<T>(m, n, -1.0f, 1.0f, seed + 2);
    const auto hBphys = MakeDensePackPaddedB<T>(hBlogical, k, n, ldB);

    DensePackScenarioCtx<T> ctx(m, k, n, ldB, hA, hBphys, hC, tag);

    // golden：A_pruned × B_logical（FP32 累加域；α=1.0 标量、β=nullptr→0.0）
    std::vector<float> Af, Bf, Cf, Df;
    PromoteInputsToFp32<T>(ctx.hApruned, hBlogical, hC, Af, Bf, Cf, m, k, n);
    MatmulAlphaBetaFp32(Af, Bf, Cf, Df, m, k, n, 1.0f, 0.0f);
    std::vector<T> hGolden;
    TruncateToOutput<T>(Df, hGolden, mn);

    if (withSearch) {
        VerifyDensePackEntry<T>(ctx, tag, "Search", /*viaSearch=*/true, hGolden);
    }
    VerifyDensePackEntry<T>(ctx, tag, "Matmul", /*viaSearch=*/false, hGolden);
}

// ---- X40 (Y2/S1)：大 viewCols 非紧凑布局端到端（cube GM 布局直通行距
//      ldB：MTE2 搬运按行距跳过行尾 pad，无 UB 中转缓冲上限约束）。
//      FP32 n=8192/ldB=8200（pad 8 列）、FP16 n=7952/ldB=7968（pad 16 列，
//      尾行非对齐边界附近）；pad 列污染值佐证直读路径不读 pad；
//      Search（候选执行链）+ Matmul（直调执行链）双入口数值验证 ----
TEST_F(MatmulSearchExceptionTest, X40DensePackColumnChunkedBigShape)
{
    RunDensePackBigShapeScenario<float>("X40-FP32-8192-ldB8200", 128, 128, 8192, 8200,
                                        20260401u, /*withSearch=*/true);
    RunDensePackBigShapeScenario<uint16_t>("X40-FP16-7952-ldB7968", 128, 128, 7952, 7968,
                                           20260402u, /*withSearch=*/true);
}

// ---- X41 (Y2/S2)：紧凑路径对照（零 padding 锚点）——ldB==n=8192 时
//      行距与逻辑宽度一致（布局参数与紧凑形态逐位相同），同 shape 大 n
//      数值正确性独立验证 ----
TEST_F(MatmulSearchExceptionTest, X41DensePackCompactAnchorBigShape)
{
    RunDensePackBigShapeScenario<float>("X41-FP32-8192-compact", 128, 128, 8192, 8192,
                                        20260411u, /*withSearch=*/false);
}

// ---- X42 (Y3/N1+P1+P2)：biasPointer 16B 对齐非 32B（malloc 基址+16）→
//      INVALID_VALUE（DataCopyPad GM 源硬件要求）；Search 入口候选执行链
//      拦截 → 全候选失败聚合 EXECUTION_FAILED。恢复合法（32B 对齐 +
//      stride=0 广播）→ SUCCESS + 数值正确（合法域不误拒 + host 侧拒绝无
//      设备级联）----
TEST_F(MatmulSearchExceptionTest, X42BiasPointerMisalignRejected)
{
    LiveReadChainCtx<float> ctx(20260421u);
    sparse_test::DeviceBuffer dBias = sparse_test::DeviceBuffer::alloc(
        static_cast<size_t>(ctx.m) * sizeof(float) + 32);
    void* misaligned = static_cast<uint8_t*>(dBias.get()) + 16;
    const uintptr_t baseAddr = reinterpret_cast<uintptr_t>(dBias.get());
    const uintptr_t probeAddr = reinterpret_cast<uintptr_t>(misaligned);
    ASSERT_TRUE((baseAddr % 32) == 0 && (probeAddr % 16) == 0 && (probeAddr % 32) != 0)
        << "precondition: bias base 32B-aligned, probe ptr 16B-aligned-not-32B";
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BIAS_POINTER, &misaligned, sizeof(void*)),
        ACL_SPARSE_STATUS_SUCCESS);
    const float alpha = 2.0f;
    aclsparseStatus_t st = ctx.RunMatmul(&alpha, nullptr);
    const bool n1Ok = (st == ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_TRUE(n1Ok) << "misaligned biasPointer, ret=" << static_cast<int32_t>(st);
    Record22DStatusCheck("X42", "biasPointer(16B-not-32B)",
        "ret=" + std::to_string(static_cast<int32_t>(st)), "INVALID_VALUE", n1Ok);
    st = ctx.RunSearch(&alpha, nullptr);
    const bool n1SearchOk = (st == ACL_SPARSE_STATUS_EXECUTION_FAILED);
    EXPECT_TRUE(n1SearchOk) << "misaligned biasPointer Search, ret="
                            << static_cast<int32_t>(st);
    Record22DStatusCheck("X42", "biasPointer(16B-not-32B)/Search",
        "ret=" + std::to_string(static_cast<int32_t>(st)),
        "EXECUTION_FAILED(all-candidates-rejected)", n1SearchOk);
    // 恢复 + 正向对照：32B 对齐 + stride=0 广播 → SUCCESS + 数值正确
    const auto biasVec = GenBiasVector(ctx.m, 20260422u);
    ASSERT_EQ(aclrtMemcpy(dBias.get(), static_cast<size_t>(ctx.m) * sizeof(float),
        biasVec.data(), static_cast<size_t>(ctx.m) * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);
    void* biasAligned = dBias.get();
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BIAS_POINTER, &biasAligned, sizeof(void*)),
        ACL_SPARSE_STATUS_SUCCESS);
    const int64_t stride0 = 0;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BIAS_STRIDE, &stride0, sizeof(int64_t)),
        ACL_SPARSE_STATUS_SUCCESS);
    st = ctx.RunMatmul(&alpha, nullptr);
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS) << "recovered bias Matmul";
    ASSERT_EQ(aclrtSynchronizeStream(ctx.stream), ACL_SUCCESS);
    const auto hD = ctx.ReadD();
    const auto golden = ComputeLiveReadGolden(ctx, alpha, nullptr, &biasVec);
    EXPECT_TRUE(VerifyFloatSearchOutput<float>(golden, hD, "X42-bias-recovered",
                "matD(bias-broadcast-stride0)"));
}

// ---- X43 (Y3/N2)：biasStride 负值（指针对齐合法）→ INVALID_VALUE（负值在
//      kernel batch 循环回走越界，host 侧拦截）；Search 入口同聚合
//      EXECUTION_FAILED；恢复 stride=0（广播）→ SUCCESS（库仍可用）----
TEST_F(MatmulSearchExceptionTest, X43BiasStrideNegativeRejected)
{
    LiveReadChainCtx<float> ctx(20260431u);
    sparse_test::DeviceBuffer dBias = sparse_test::DeviceBuffer::alloc(
        static_cast<size_t>(ctx.m) * sizeof(float));
    void* biasPtr = dBias.get();
    ASSERT_TRUE((reinterpret_cast<uintptr_t>(biasPtr) % 32) == 0)
        << "precondition: bias ptr 32B-aligned";
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BIAS_POINTER, &biasPtr, sizeof(void*)),
        ACL_SPARSE_STATUS_SUCCESS);
    const int64_t strideNeg = -8;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BIAS_STRIDE, &strideNeg, sizeof(int64_t)),
        ACL_SPARSE_STATUS_SUCCESS);
    const float alpha = 2.0f;
    aclsparseStatus_t st = ctx.RunMatmul(&alpha, nullptr);
    const bool n2Ok = (st == ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_TRUE(n2Ok) << "negative biasStride, ret=" << static_cast<int32_t>(st);
    Record22DStatusCheck("X43", "biasStride(-8)",
        "ret=" + std::to_string(static_cast<int32_t>(st)), "INVALID_VALUE", n2Ok);
    st = ctx.RunSearch(&alpha, nullptr);
    const bool n2SearchOk = (st == ACL_SPARSE_STATUS_EXECUTION_FAILED);
    EXPECT_TRUE(n2SearchOk) << "negative biasStride Search, ret="
                            << static_cast<int32_t>(st);
    Record22DStatusCheck("X43", "biasStride(-8)/Search",
        "ret=" + std::to_string(static_cast<int32_t>(st)),
        "EXECUTION_FAILED(all-candidates-rejected)", n2SearchOk);
    // 恢复：stride=0（广播，合法域）→ SUCCESS（数值正确性由 X42 正向对照覆盖）
    const int64_t stride0 = 0;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BIAS_STRIDE, &stride0, sizeof(int64_t)),
        ACL_SPARSE_STATUS_SUCCESS);
    st = ctx.RunMatmul(&alpha, nullptr);
    const bool recOk = (st == ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_TRUE(recOk) << "recovered stride=0 Matmul, ret=" << static_cast<int32_t>(st);
    Record22DStatusCheck("X43", "biasStride(0,broadcast)/recovered",
        "ret=" + std::to_string(static_cast<int32_t>(st)), "SUCCESS", recOk);
}

// ---- X44 (Y1/P5)：PlanInit（无 vec/bias）后设 bias + 双 vec + 大 m →
//      biasChunkMode 执行期活读重算。m=20992 时 fused 全量加载 UB 需求
//      (α+β+bias)·m·4 = 246KB，叠加 tile 缓冲 > 248KB（get_ub_size()=253952）
//      ——PlanInit 快照 biasChunkMode=0（当时无 vec/bias）过期，快照语义下
//      fused kernel InitBuffer 超 UB；活读语义下 fill_matmul_tiling_fields
//      执行期重调 compute_bias_chunk_mode 降级 per-chunk 加载。断言 SUCCESS +
//      全元素数值正确（QA 探针 P5 形态：BETA_VECTOR_SCALING=1 隐含提升
//      ALPHA，bias 广播 stride=0）----
TEST_F(MatmulSearchExceptionTest, X44LiveReadBiasChunkModeRecomputed)
{
    // 大 m：全量加载超 UB 域（LiveReadChainCtx 尾参覆盖维度，k/n 维持 128）
    LiveReadChainCtx<float> ctx(20260441u, /*prePlanAlphaVec=*/false,
                                /*dimM=*/20992, /*dimK=*/128, /*dimN=*/128);
    // PlanInit 后设置：BETA_VECTOR_SCALING=1（隐含提升 ALPHA——alpha 须 device
    // 向量）+ bias（stride=0 广播），均为执行期活读属性
    int enable = 1;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING, &enable, sizeof(int)),
        ACL_SPARSE_STATUS_SUCCESS) << "post-PlanInit SetAttribute(BETA_VEC=1)";
    const auto alphaVec = GenScalingVector(ctx.m, 20260442u);
    const auto betaVec = GenScalingVector(ctx.m, 20260443u);
    const auto biasVec = GenBiasVector(ctx.m, 20260447u);  // seed%10=7 → [-2,2] 随机
    sparse_test::DeviceBuffer dAlphaVec = sparse_test::DeviceBuffer::copyFrom(
        alphaVec.data(), alphaVec.size() * sizeof(float));
    sparse_test::DeviceBuffer dBetaVec = sparse_test::DeviceBuffer::copyFrom(
        betaVec.data(), betaVec.size() * sizeof(float));
    sparse_test::DeviceBuffer dBias = sparse_test::DeviceBuffer::copyFrom(
        biasVec.data(), biasVec.size() * sizeof(float));
    void* biasPtr = dBias.get();
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BIAS_POINTER, &biasPtr, sizeof(void*)),
        ACL_SPARSE_STATUS_SUCCESS);
    const int64_t stride0 = 0;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BIAS_STRIDE, &stride0, sizeof(int64_t)),
        ACL_SPARSE_STATUS_SUCCESS);
    aclsparseStatus_t st = ctx.RunMatmul(dAlphaVec.get(), dBetaVec.get());
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS)
        << "post-PlanInit bias+double-vec(large m) Matmul";
    ASSERT_EQ(aclrtSynchronizeStream(ctx.stream), ACL_SUCCESS);
    const auto hD = ctx.ReadD();
    // golden：D[i,j] = α[i]·(A_pruned×B)[i,j] + β[i]·C[i,j] + bias[i]（双 vec
    // + bias 广播；NPU 剪枝回读范式，FP32 域）
    std::vector<float> Af, Bf, Cf, golden;
    PromoteInputsToFp32<float>(ctx.hApruned, ctx.hB, ctx.hC, Af, Bf, Cf, ctx.m, ctx.k, ctx.n);
    MatmulAlphaBetaFp32Vec(Af, Bf, Cf, golden, ctx.m, ctx.k, ctx.n, alphaVec, betaVec);
    ApplyBiasFp32(golden, biasVec, ctx.m, ctx.n);
    EXPECT_TRUE(VerifyFloatSearchOutput<float>(golden, hD, "X44-biaschunk-recompute",
                "matD(live-read-biasChunkMode-perchunk)"));
}

// ============================================================================
// 探针收编组（X45~X50）：Z1 workspace 活读复检 / Z2 vec UB 预算 / vec
// 指针 32B 对齐——临时探针 probe_z1z2（Z1/Z2/vec32B 三组 44 断言
// 全绿）的正式用例化（修复：matmul_host
// validate_workspace_sufficiency / validate_scaling_ub_budget /
// validate_vector_scaling_pointers + kernel SpltLoadScalingUB chunk 模式与
// fused epilogue tile 段惰性加载）。分支覆盖（算子侧执行期路径）：
//   X45/X46 → matmul_host validate_workspace_sufficiency 活读重算分支；
//   X47 → kernel SpltLoadScalingUB scalingChunkMode tile 级槽位；
//   X48 → matmul_host validate_scaling_ub_budget two-stage 拒绝分支；
//   X49/X50 → validate_vector_scaling_pointers 的 alpha/beta 32B 对齐分支。
// ============================================================================

// X45 golden：batch 段提取（紧凑 batch 排布第 batchIdx 段）+ 标量 α（β 恒
// nullptr 语义 0.0，C 不参与）
inline std::vector<float> ComputeBatchGoldenFp32(const LiveReadChainCtx<float>& c,
    int32_t batchIdx, float alphaScalar)
{
    const size_t mk = static_cast<size_t>(c.m) * c.k;
    const size_t kn = static_cast<size_t>(c.k) * c.n;
    const size_t mn = static_cast<size_t>(c.m) * c.n;
    const size_t offA = static_cast<size_t>(batchIdx) * mk;
    const size_t offB = static_cast<size_t>(batchIdx) * kn;
    std::vector<float> Af(c.hApruned.begin() + offA, c.hApruned.begin() + offA + mk);
    std::vector<float> Bf(c.hB.begin() + offB, c.hB.begin() + offB + kn);
    std::vector<float> Cf(mn, 0.0f);
    std::vector<float> Df;
    MatmulAlphaBetaFp32(Af, Bf, Cf, Df, c.m, c.k, c.n, alphaScalar, 0.0f);
    return Df;
}

// X47/X50 golden：双 vec（batch=1 语义）——FP32 累加域 D[i,j] =
// α[i]·(A_pruned×B)[i,j] + β[i]·C[i,j]，输出量化回 T（FP16 与 device 输出
// 同 dtype 逐位口径）
template <typename T>
inline std::vector<T> ComputeDualVecGolden(const LiveReadChainCtx<T>& c,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec)
{
    std::vector<float> Af, Bf, Cf, Df;
    PromoteInputsToFp32<T>(c.hApruned, c.hB, c.hC, Af, Bf, Cf, c.m, c.k, c.n);
    MatmulAlphaBetaFp32Vec(Af, Bf, Cf, Df, c.m, c.k, c.n, alphaVec, betaVec);
    std::vector<T> out;
    TruncateToOutput<T>(Df, out, static_cast<size_t>(c.m) * c.n);
    return out;
}

// ---- X45 (Z1/A1)：正常时序 batch=2 × splitK=2（two-stage）→ Matmul 成功 +
//      逐 batch 数值正确——活读 workspace 重算 == PlanInit 快照（同式同参
//      恒等），零影响锚点（修复不误拒合法域）----
TEST_F(MatmulSearchExceptionTest, X45NormalOrderBatch2SplitK2Anchor)
{
    // initBatches=2：batch 属性于 MatmulDescriptorInit 前置（正常时序一致性）
    LiveReadChainCtx<float> ctx(20260451u, /*prePlanAlphaVec=*/false,
                                /*dimM=*/128, /*dimK=*/128, /*dimN=*/128,
                                /*initBatches=*/2, /*splitK=*/2, /*dataBatches=*/2);
    const float alpha = 1.5f;
    aclsparseStatus_t st = ctx.RunMatmul(&alpha, nullptr);
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS) << "normal-order batch=2 splitK=2 Matmul";
    ASSERT_EQ(aclrtSynchronizeStream(ctx.stream), ACL_SUCCESS);
    for (int32_t b = 0; b < 2; ++b) {
        const auto hD = ctx.ReadDBatch(b);
        const auto golden = ComputeBatchGoldenFp32(ctx, b, alpha);
        EXPECT_TRUE(VerifyFloatSearchOutput<float>(golden, hD,
                    "X45-normal-b" + std::to_string(b),
                    "matD(batch=" + std::to_string(b) + ",splitK=2,normal-order)"));
    }
}

// ---- X46 (Z1/A2)：PlanInit(batch=1, splitK=2) 后活读 NUM_BATCHES 1→2 →
//      workspace 快照失配（temp 区按活读 batch 重算超出 PlanInit 冻结值）→
//      INSUFFICIENT_RESOURCES 干净拒绝；恢复 batch=1 → Matmul 成功且数值
//      正确——拒绝发生在 host 校验层（validate_workspace_sufficiency，无
//      kernel launch / memset），设备状态未污染 ----
TEST_F(MatmulSearchExceptionTest, X46PostPlanBatchGrowInsufficientResources)
{
    // dataBatches=2：活读增大场景预留缓冲容量（对齐探针保守做法，防快照
    // 失配拒绝路径意外放行时 kernel 越出用例自有缓冲）
    LiveReadChainCtx<float> ctx(20260461u, /*prePlanAlphaVec=*/false,
                                /*dimM=*/128, /*dimK=*/128, /*dimN=*/128,
                                /*initBatches=*/1, /*splitK=*/2, /*dataBatches=*/2);
    const float alpha = 1.5f;
    // batch=1 基线（活读重算 == 快照，恒通过）
    ASSERT_EQ(ctx.RunMatmul(&alpha, nullptr), ACL_SPARSE_STATUS_SUCCESS)
        << "batch=1 baseline Matmul";
    ASSERT_EQ(aclrtSynchronizeStream(ctx.stream), ACL_SUCCESS);
    // PlanInit 后增大 NUM_BATCHES（活读契约允许）× splitK=2 → temp 快照失配
    ASSERT_EQ(ctx.SetLiveBatches(2), ACL_SPARSE_STATUS_SUCCESS)
        << "post-PlanInit SetNumBatches(2)";
    aclsparseStatus_t st = ctx.RunMatmul(&alpha, nullptr);
    const bool rejectOk = (st == ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
    EXPECT_TRUE(rejectOk) << "live batch 1->2 x splitK=2, ret="
                          << static_cast<int32_t>(st);
    Record22DStatusCheck("X46", "liveBatch(1->2)xsplitK(2)",
        "ret=" + std::to_string(static_cast<int32_t>(st)),
        "INSUFFICIENT_RESOURCES", rejectOk);
    // 恢复 batch=1 → 成功 + 数值正确（拒绝路径无设备写，状态干净）
    ASSERT_EQ(ctx.SetLiveBatches(1), ACL_SPARSE_STATUS_SUCCESS)
        << "restore SetNumBatches(1)";
    st = ctx.RunMatmul(&alpha, nullptr);
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS) << "restored batch=1 Matmul";
    ASSERT_EQ(aclrtSynchronizeStream(ctx.stream), ACL_SUCCESS);
    const auto hD = ctx.ReadD();
    const auto golden = ComputeLiveReadGolden(ctx, alpha, nullptr, nullptr);
    EXPECT_TRUE(VerifyFloatSearchOutput<float>(golden, hD, "X46-restored-batch1",
                "matD(state-unpolluted-after-reject)"));
}

// ---- X47 (Z2/B1)：FP16 m=16384 双 vec（vecBytes=128KB 叠加 tile ~133KB
//      > 248KB UB，超 fused 全量加载 ~14.6K 阈值域）→ per-tile chunk 惰性
//      降级正确执行——PlanInit 后 BETA_VECTOR_SCALING=1（隐含提升 ALPHA），
//      全元素数值正确（chunk 化仅 tile 段加载 + 消费索引平移，数值不变）----
TEST_F(MatmulSearchExceptionTest, X47VecScalingPerTileChunkedLargeM)
{
    LiveReadChainCtx<uint16_t> ctx(20260471u, /*prePlanAlphaVec=*/false,
                                   /*dimM=*/16384, /*dimK=*/128, /*dimN=*/128);
    int enable = 1;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING, &enable, sizeof(int)),
        ACL_SPARSE_STATUS_SUCCESS) << "post-PlanInit SetAttribute(BETA_VEC=1)";
    const auto alphaVec = GenScalingVector(ctx.m, 20260472u);
    const auto betaVec = GenScalingVector(ctx.m, 20260473u);
    sparse_test::DeviceBuffer dAlphaVec = sparse_test::DeviceBuffer::copyFrom(
        alphaVec.data(), alphaVec.size() * sizeof(float));
    sparse_test::DeviceBuffer dBetaVec = sparse_test::DeviceBuffer::copyFrom(
        betaVec.data(), betaVec.size() * sizeof(float));
    aclsparseStatus_t st = ctx.RunMatmul(dAlphaVec.get(), dBetaVec.get());
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS)
        << "FP16 m=16384 dual-vec (over-UB-threshold) Matmul (chunk degrade)";
    ASSERT_EQ(aclrtSynchronizeStream(ctx.stream), ACL_SUCCESS);
    const auto hD = ctx.ReadD();
    const auto golden = ComputeDualVecGolden(ctx, alphaVec, betaVec);
    EXPECT_TRUE(VerifyFloatSearchOutput<uint16_t>(golden, hD, "X47-vec-chunk-16384",
                "matD(FP16,dual-vec,per-tile-chunk)"));
}

// ---- X48 (Z2/B4)：FP16 m=32768 × splitK=2 two-stage 极窄域双 vec
//      （vecBytes=256KB + 16KB 固定缓冲 > 248KB UB 预算，阈值 m>~29.5K）→
//      NOT_SUPPORTED 可诊断拒绝（不崩；拒绝在 prepare 前置校验链，无
//      kernel 访问）----
TEST_F(MatmulSearchExceptionTest, X48TwoStageVecOverBudgetRejected)
{
    LiveReadChainCtx<uint16_t> ctx(20260481u, /*prePlanAlphaVec=*/false,
                                   /*dimM=*/32768, /*dimK=*/256, /*dimN=*/128,
                                   /*initBatches=*/1, /*splitK=*/2);
    int enable = 1;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING, &enable, sizeof(int)),
        ACL_SPARSE_STATUS_SUCCESS) << "post-PlanInit SetAttribute(BETA_VEC=1)";
    const auto alphaVec = GenScalingVector(ctx.m, 20260482u);
    const auto betaVec = GenScalingVector(ctx.m, 20260483u);
    sparse_test::DeviceBuffer dAlphaVec = sparse_test::DeviceBuffer::copyFrom(
        alphaVec.data(), alphaVec.size() * sizeof(float));
    sparse_test::DeviceBuffer dBetaVec = sparse_test::DeviceBuffer::copyFrom(
        betaVec.data(), betaVec.size() * sizeof(float));
    aclsparseStatus_t st = ctx.RunMatmul(dAlphaVec.get(), dBetaVec.get());
    const bool rejectOk = (st == ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_TRUE(rejectOk) << "two-stage vec UB over budget, ret="
                          << static_cast<int32_t>(st);
    Record22DStatusCheck("X48", "twoStage(m=32768,splitK=2)dual-vec",
        "ret=" + std::to_string(static_cast<int32_t>(st)),
        "NOT_SUPPORTED(diagnosable-reject)", rejectOk);
}

// ---- X49 (vec32B/C1+C2)：alpha / beta 向量指针 16B 对齐非 32B（32B 对齐
//      基址 + 16B 偏移）→ INVALID_VALUE（DataCopyPad GM 源硬件要求，
//      validate_vector_scaling_pointers 活读生效；beta 负向时 alpha 用合法
//      32B 指针隔离单变量）----
TEST_F(MatmulSearchExceptionTest, X49VecPointerMisalign32BRejected)
{
    LiveReadChainCtx<float> ctx(20260491u);
    const auto alphaVec = GenScalingVector(ctx.m, 20260492u);
    const auto betaVec = GenScalingVector(ctx.m, 20260493u);
    sparse_test::DeviceBuffer dAlphaVec = sparse_test::DeviceBuffer::alloc(
        static_cast<size_t>(ctx.m) * sizeof(float) + 64);
    sparse_test::DeviceBuffer dBetaVec = sparse_test::DeviceBuffer::alloc(
        static_cast<size_t>(ctx.m) * sizeof(float) + 64);
    ASSERT_EQ(aclrtMemcpy(dAlphaVec.get(), static_cast<size_t>(ctx.m) * sizeof(float),
        alphaVec.data(), static_cast<size_t>(ctx.m) * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dBetaVec.get(), static_cast<size_t>(ctx.m) * sizeof(float),
        betaVec.data(), static_cast<size_t>(ctx.m) * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    // 构造 16B 对齐但非 32B 对齐的指针（32B 对齐基址 + 16B 偏移）
    const uintptr_t alphaBase = (reinterpret_cast<uintptr_t>(dAlphaVec.get()) + 31u)
                                & ~static_cast<uintptr_t>(31u);
    const uintptr_t betaBase = (reinterpret_cast<uintptr_t>(dBetaVec.get()) + 31u)
                               & ~static_cast<uintptr_t>(31u);
    const uintptr_t alphaProbe = alphaBase + 16u;
    const uintptr_t betaProbe = betaBase + 16u;
    ASSERT_TRUE((alphaProbe % 16) == 0 && (alphaProbe % 32) != 0
                && (betaProbe % 16) == 0 && (betaProbe % 32) != 0)
        << "precondition: probe ptrs 16B-aligned-not-32B";
    // C1：alpha 向量非 32B → INVALID_VALUE
    int enable = 1;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &enable, sizeof(int)),
        ACL_SPARSE_STATUS_SUCCESS) << "SetAttribute(ALPHA_VEC=1)";
    aclsparseStatus_t st = ctx.RunMatmul(reinterpret_cast<void*>(alphaProbe), nullptr);
    const bool c1Ok = (st == ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_TRUE(c1Ok) << "alpha vec 16B-aligned-not-32B, ret="
                      << static_cast<int32_t>(st);
    Record22DStatusCheck("X49", "alphaVec(16B-not-32B)",
        "ret=" + std::to_string(static_cast<int32_t>(st)), "INVALID_VALUE", c1Ok);
    // C2：beta 向量非 32B（alpha 用合法 32B 指针）→ INVALID_VALUE
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING, &enable, sizeof(int)),
        ACL_SPARSE_STATUS_SUCCESS) << "SetAttribute(BETA_VEC=1)";
    st = ctx.RunMatmul(reinterpret_cast<void*>(alphaBase),
                       reinterpret_cast<void*>(betaProbe));
    const bool c2Ok = (st == ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_TRUE(c2Ok) << "beta vec 16B-aligned-not-32B, ret="
                      << static_cast<int32_t>(st);
    Record22DStatusCheck("X49", "betaVec(16B-not-32B)",
        "ret=" + std::to_string(static_cast<int32_t>(st)), "INVALID_VALUE", c2Ok);
}

// ---- X50 (vec32B 合法对照 = B2 探针)：FP32 m=128 双 vec 正常域（vecBytes
//      1KB << 248KB）→ 全量加载路径 SUCCESS + 数值正确——合法 32B 指针不
//      误拒 + Z2 fused 修复零变化锚点（阈值以下仍走全量加载）----
TEST_F(MatmulSearchExceptionTest, X50VecPointerAligned32BGolden)
{
    LiveReadChainCtx<float> ctx(20260501u);
    int enable = 1;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(ctx.handle.get(), &ctx.matmulDesc->get(),
        ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING, &enable, sizeof(int)),
        ACL_SPARSE_STATUS_SUCCESS) << "SetAttribute(BETA_VEC=1, implies alpha)";
    const auto alphaVec = GenScalingVector(ctx.m, 20260502u);
    const auto betaVec = GenScalingVector(ctx.m, 20260503u);
    sparse_test::DeviceBuffer dAlphaVec = sparse_test::DeviceBuffer::copyFrom(
        alphaVec.data(), alphaVec.size() * sizeof(float));
    sparse_test::DeviceBuffer dBetaVec = sparse_test::DeviceBuffer::copyFrom(
        betaVec.data(), betaVec.size() * sizeof(float));
    ASSERT_TRUE((reinterpret_cast<uintptr_t>(dAlphaVec.get()) % 32) == 0
                && (reinterpret_cast<uintptr_t>(dBetaVec.get()) % 32) == 0)
        << "precondition: vec ptrs 32B-aligned";
    aclsparseStatus_t st = ctx.RunMatmul(dAlphaVec.get(), dBetaVec.get());
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS)
        << "dual-vec 32B-aligned Matmul (full-load normal domain)";
    ASSERT_EQ(aclrtSynchronizeStream(ctx.stream), ACL_SUCCESS);
    const auto hD = ctx.ReadD();
    const auto golden = ComputeDualVecGolden(ctx, alphaVec, betaVec);
    EXPECT_TRUE(VerifyFloatSearchOutput<float>(golden, hD, "X50-vec32b-aligned",
                "matD(dual-vec,full-load)"));
}
