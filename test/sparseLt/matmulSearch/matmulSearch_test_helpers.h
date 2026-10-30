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

#ifndef TEST_SPARSELT_MATMULSEARCH_TEST_HELPERS_H_
#define TEST_SPARSELT_MATMULSEARCH_TEST_HELPERS_H_

// =============================================================================
// matmulSearch 测试共享基础设施：helpers / golden_verify 两个头 +
// matmulSearch_test.cpp（TEST_P + X01~X19）/ matmulSearch_test_xcases.cpp
// （X20~X23 dtype 超集组合）/ matmulSearch_test_live.cpp（X31~X50
// 补齐 + 探针收编组），同 target 多源链接。
//
// 本头：ACL 环境单例、权威断言记录（JSONL 落盘）、参数 glue、输入生成、
// 异常基座公共段（SearchChainCtxBase / MakeSearchDescQuartet 等——三个
// ctx 的重复成员/构造段提取）、SearchExceptionCtx 基座、BuildSearchArgs。
//
// golden 计算 / 数值验证 / 搜索链后置断言在 matmulSearch_golden_verify.h。
// =============================================================================

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_common.h"

#include "ltmatmul_exception_helper.h"  // MatmulRawParams：SearchRawParams 公共字段基类（codecheck8 消重）
#include "ltmatmul_golden.h"
#include "ltmatmul_golden_int8.h"
#include "matmulSearch_golden.h"
#include "matmulSearch_npu_wrapper.h"
#include "matmulSearch_param.h"

namespace matmulsearch_test {

using namespace sparse_test;

// ============================================================================
// 全局 ACL 环境单例（AclEnvScope，call_once；matmulPlanInit 范式）
// ============================================================================
class SearchAclEnvManager
{
  public:
    static sparse_test::AclEnvScope& Get()
    {
        std::call_once(once_, []() { instance_ = std::make_unique<sparse_test::AclEnvScope>(); });
        return *instance_;
    }

  private:
    static std::once_flag once_;
    static std::unique_ptr<sparse_test::AclEnvScope> instance_;
};
// 注：静态成员定义（once_/instance_）位于 matmulSearch_test.cpp（本头被多个
// TU include，单一定义点避免重复符号）。

// ============================================================================
// 权威断言记录（输出张量 / 判定指标 / 实测值 / 阈值 / 结论 → JSONL 落盘）
// ============================================================================
struct SearchAssertionRecord {
    std::string caseId;
    std::string tensor;     // matD / matD.search / matD.rerun / algSelection / plan.workspaceSize / T(c*) / blocking / inplace
    std::string metric;     // MIXED_TOLERANCE / EXACT / candidate-legal(Tier1+Tier2) / ws-recompute+monotonic / timing<=1.10x / declarative
    std::string measured;   // 实测值
    std::string threshold;  // 阈值 / 期望
    std::string verdict;    // PASS / FAIL
};

inline std::string SearchAssertLogPath()
{
    // 环境变量优先（绝对路径），否则向上锚定含 .cannbot 的目录（build/test/
    // matmulSearch → 仓根），保证随结果落盘到任务约定的 .cannbot 中间产物区。
    if (const char* env = std::getenv("MATMULSEARCH_ASSERT_LOG")) { return env; }
    namespace fs = std::filesystem;
    static const std::string anchored = []() -> std::string {
        for (const char* c : {".", "..", "../..", "../../..", "../../../.."}) {
            fs::path probe = fs::path(c) / ".cannbot";
            std::error_code ec;
            if (fs::exists(probe, ec) && fs::is_directory(probe, ec)) {
                return (fs::path(c) / ".cannbot" / "aclsparseLtMatmulSearch" / "tmp" /
                        "precision_assertions.log").string();
            }
        }
        return ".cannbot/aclsparseLtMatmulSearch/tmp/precision_assertions.log";
    }();
    return anchored;
}

inline void AppendSearchRecord(const SearchAssertionRecord& r)
{
    namespace fs = std::filesystem;
    fs::path p(SearchAssertLogPath());
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);  // 目录不存在时创建；失败则下方静默跳过
    std::ofstream ofs(p, std::ios::app);
    if (!ofs.is_open()) { return; }
    ofs << "{\"caseId\":\"" << r.caseId
        << "\",\"tensor\":\"" << r.tensor
        << "\",\"metric\":\"" << r.metric
        << "\",\"measured\":\"" << r.measured
        << "\",\"threshold\":\"" << r.threshold
        << "\",\"verdict\":\"" << r.verdict << "\"}\n";
}

inline std::string VerdictOf(bool ok) { return ok ? "PASS" : "FAIL"; }

// ============================================================================
// 数据生成 glue（核心生成函数复用 ltmatmul_golden_types.h / prune_test_util.h；
// MatmulGenTestMatrix / GenScalingVector / GenBiasVector / Fp32ToFp16Bits 等
// 均来自头文件；此处仅组装）。
// 注意：SearchVecs 与 ltmatmul AlphaBetaVecs 同构——alphaPtr/betaPtr/biasPtr
// 指向返回值自身成员，必须「按值返回 + auto 接收」（NRVO）传递；经输出参数
// 拷贝/移动赋值传递会悬空（vec scaling / bias 用例读垃圾）。
// ============================================================================
inline uint32_t SeedForSearchCase(int32_t caseId) { return 4000u + static_cast<uint32_t>(caseId); }

inline aclsparseLtPruneAlg_t ToNpuPruneAlg(const MatmulSearchParam& p)
{
    return p.isTilePrune() ? ACLSPARSELT_PRUNE_SPMMA_TILE : ACLSPARSELT_PRUNE_SPMMA_STRIP;
}

inline aclsparseOrder_t ToNpuOrder(const MatmulSearchParam& p)
{
    return p.isColOrder() ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW;
}

inline aclsparseOperation_t ToNpuOp(bool trans)
{
    return trans ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
}

// CSV 布局列 → MatmulLayoutOpts（ld_sparse/ld_dense/dense_order/cd_order/ld_cd；
// 缺省全零/row = 既有紧凑行为）
inline sparse_test::MatmulLayoutOpts ToNpuLayoutOpts(const MatmulSearchParam& p)
{
    sparse_test::MatmulLayoutOpts o;
    o.ldSparse = p.ldSparse;
    o.ldDense = p.ldDense;
    o.denseOrder = p.isDenseColOrder() ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW;
    o.cdOrder = p.isCdColOrder() ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW;
    o.ldCd = p.ldCd;
    o.directFeed = (p.directFeed != 0);
    return o;
}

// F1: activation_type=3（GELU_SCALING only）蕴含 GeLU，golden 按 GeLU（ltmatmul 同范式）
inline int32_t GoldenActType(int32_t activationType)
{
    return (activationType == 3) ? EPILOGUE_ACT_GELU : activationType;
}

// alpha/beta nullptr 语义（B6/B7）：golden 按 α=1.0 / β=0.0
inline float GoldenAlphaOf(const MatmulSearchParam& p) { return p.alpha_null ? 1.0f : p.alpha; }
inline float GoldenBetaOf(const MatmulSearchParam& p) { return p.beta_null ? 0.0f : p.beta; }

struct SearchVecs {
    std::vector<float> alphaVec;
    std::vector<float> betaVec;
    std::vector<float> biasVec;  // per-row bias（空 = 无）
    const std::vector<float>* alphaPtr = nullptr;
    const std::vector<float>* betaPtr = nullptr;
    const std::vector<float>* biasPtr = nullptr;
};

inline SearchVecs GenSearchVecs(const MatmulSearchParam& p, uint32_t seed)
{
    SearchVecs v;
    // 向量缩放（D2）或标量填充（golden 的 vec/标量双路径共用）
    if (p.alpha_vector_scaling) {
        v.alphaVec = GenScalingVector(p.m, seed + 10);
    } else {
        v.alphaVec.assign(static_cast<size_t>(p.m), GoldenAlphaOf(p));
    }
    if (p.beta_vector_scaling) {
        v.betaVec = GenScalingVector(p.m, seed + 20);
    } else {
        v.betaVec.assign(static_cast<size_t>(p.m), GoldenBetaOf(p));
    }
    // bvec 隐含提升（BETA implies ALPHA）：beta 向量启用时库内隐含提升 ALPHA 编码位，
    // alpha 必须有 device 向量——alphaPtr 指向标量填充向量，SetVecScalingAttrs 据此
    // 分配 dAlphaVec（否则 host 标量指针被当 device 向量读取 → AI Core Error）
    v.alphaPtr = (p.alpha_vector_scaling || p.beta_vector_scaling) ? &v.alphaVec : nullptr;
    v.betaPtr = p.beta_vector_scaling ? &v.betaVec : nullptr;
    if (p.hasBias()) {
        // bias_stride>0（逐 batch 独立偏移）时生成全量 numBatches*biasStride
        // 布局（per batch 独立 GenBiasVector，ltmatmul GenBiasForParam 同范式）；
        // 否则单份共享 bias（m 个，跨 batch 广播）
        if (p.hasBatch() && p.biasStride > 0) {
            const size_t total = static_cast<size_t>(p.numBatches) *
                                 static_cast<size_t>(p.biasStride);
            v.biasVec.resize(total);
            for (int32_t b = 0; b < p.numBatches; ++b) {
                auto biasB = GenBiasVector(p.m, seed + 30 + static_cast<uint32_t>(b) * 7);
                const size_t off = static_cast<size_t>(b) * static_cast<size_t>(p.biasStride);
                std::copy(biasB.begin(), biasB.begin() + p.m, v.biasVec.begin() + off);
            }
        } else {
            v.biasVec = GenBiasVector(p.m, seed + 30);  // [-2,2] 域（ltmatmul 同范式）
        }
        // F2: bias dtype 跟随 C dtype —— FP16/BF16 经存储 dtype 舍入对齐 NPU
        if (p.isFp16()) {
            for (auto& b : v.biasVec) { b = Fp16BitsToFp32(Fp32ToFp16Bits(b)); }
        } else if (p.isBf16()) {
            for (auto& b : v.biasVec) { b = Bf16BitsToFp32(Fp32ToBf16Bits(b)); }
        }
        v.biasPtr = &v.biasVec;
    }
    return v;
}

template <typename T>
inline void GenSearchInputs(const MatmulSearchParam& p, uint32_t seed,
    std::vector<T>& hA, std::vector<T>& hB, std::vector<T>& hC)
{
    if (p.hasBatch()) {
        // 浮点 batch——batch-strided 布局（每 batch 前 matSize 个
        // 有效元素 + stride padding 零；ltmatmul GenBatchStridedMatrix 同构）
        auto genBatch = [&p](int32_t rows, int32_t cols, uint32_t s) {
            const size_t total = static_cast<size_t>(p.numBatches) * static_cast<size_t>(p.batchStride);
            std::vector<T> data(total, T{0});
            for (int32_t b = 0; b < p.numBatches; ++b) {
                auto bd = MatmulGenTestMatrix<T>(rows, cols, p.range_low, p.range_high,
                    s + static_cast<uint32_t>(b) * 100);
                std::copy(bd.begin(), bd.end(), data.begin() + static_cast<size_t>(b) * p.batchStride);
            }
            return data;
        };
        hA = genBatch(p.m, p.k, seed);
        hB = genBatch(p.k, p.n, seed + 1);
        hC = genBatch(p.m, p.n, seed + 2);
        return;
    }
    hA = MatmulGenTestMatrix<T>(p.m, p.k, p.range_low, p.range_high, seed);
    hB = MatmulGenTestMatrix<T>(p.k, p.n, p.range_low, p.range_high, seed + 1);
    hC = MatmulGenTestMatrix<T>(p.m, p.n, p.range_low, p.range_high, seed + 2);
}

// ============================================================================
// 异常基座公共段（X01~X17 / X31~X34 / X38~X50 三个 ctx 的共享成员与构造段）
// ============================================================================

// 三个异常基座的公共成员（handle → mat 描述符 → matmulDesc → algSel → plan
// + 显式流 + 设备缓冲 + 标量）；析构统一销毁流（三 ctx 原析构语义一致）。
struct SearchChainCtxBase {
    sparse_test::SparseLtHandleGuard handle;
    std::unique_ptr<sparse_test::SparseLtMatDescGuard> matA;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matB;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matC;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matD;
    std::unique_ptr<sparse_test::SparseLtMatmulDescGuard> matmulDesc;
    std::unique_ptr<sparse_test::SparseLtAlgSelectionGuard> algSel;
    std::unique_ptr<sparse_test::SparseLtPlanGuard> plan;

    aclrtStream stream = nullptr;
    sparse_test::DeviceBuffer dA;
    sparse_test::DeviceBuffer dB;
    sparse_test::DeviceBuffer dC;
    sparse_test::DeviceBuffer dD;
    sparse_test::DeviceBuffer dPruned;
    sparse_test::DeviceBuffer dWorkspace;
    size_t workspaceSize = 0;

    float alpha = 1.0f;
    float beta = 0.0f;

    ~SearchChainCtxBase()
    {
        if (stream) { aclrtDestroyStream(stream); }
    }
};

// A-structured + B/C/D dense 描述符四元组（紧凑 ROW，ld=物理列数；dtype 参数
// 化）——SearchExceptionCtx / VecSearchExceptionCtx / LiveReadChainCtx 构造
// 的共享段。
template <typename T>
inline void MakeSearchDescQuartet(sparse_test::SparseLtHandleGuard& handle,
    int32_t m, int32_t k, int32_t n,
    std::unique_ptr<sparse_test::SparseLtMatDescGuard>& matA,
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard>& matB,
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard>& matC,
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard>& matD)
{
    constexpr aclDataType kDtype = sparse_test::NpuDtypeTrait<T>::kAclDtype;
    matA = std::make_unique<sparse_test::SparseLtMatDescGuard>(
        handle.get(), m, k, k, 16, kDtype, ACL_SPARSE_ORDER_ROW,
        ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    matB = std::make_unique<sparse_test::SparseLtDnMatDescGuard>(
        handle.get(), k, n, n, 16, kDtype, ACL_SPARSE_ORDER_ROW);
    matC = std::make_unique<sparse_test::SparseLtDnMatDescGuard>(
        handle.get(), m, n, n, 16, kDtype, ACL_SPARSE_ORDER_ROW);
    matD = std::make_unique<sparse_test::SparseLtDnMatDescGuard>(
        handle.get(), m, n, n, 16, kDtype, ACL_SPARSE_ORDER_ROW);
}

// 异常基座的默认数据/缓冲/ws 段（hA=1.0 / hB=1.0 / hC=0.5 全量矩阵 +
// dA~dD/dPruned 拷贝分配 + GetWorkspace + ws 分配）——SearchExceptionCtx /
// VecSearchExceptionCtx 构造共享段。
inline void InitExcChainDataAndWs(sparse_test::SparseLtHandleGuard& handle,
    sparse_test::SparseLtPlanGuard& plan, int32_t m, int32_t k, int32_t n,
    sparse_test::DeviceBuffer& dA, sparse_test::DeviceBuffer& dB,
    sparse_test::DeviceBuffer& dC, sparse_test::DeviceBuffer& dD,
    sparse_test::DeviceBuffer& dPruned, sparse_test::DeviceBuffer& dWorkspace,
    size_t& workspaceSize)
{
    const size_t mk = static_cast<size_t>(m) * k;
    const size_t kn = static_cast<size_t>(k) * n;
    const size_t mn = static_cast<size_t>(m) * n;
    std::vector<float> hA(mk, 1.0f);
    std::vector<float> hB(kn, 1.0f);
    std::vector<float> hC(mn, 0.5f);
    dA = sparse_test::DeviceBuffer::copyFrom(hA.data(), mk * sizeof(float));
    dB = sparse_test::DeviceBuffer::copyFrom(hB.data(), kn * sizeof(float));
    dC = sparse_test::DeviceBuffer::copyFrom(hC.data(), mn * sizeof(float));
    dD = sparse_test::DeviceBuffer::alloc(mn * sizeof(float));
    dPruned = sparse_test::DeviceBuffer::alloc(mk * sizeof(float));
    aclsparseStatus_t wsRet =
        aclsparseLtMatmulGetWorkspace(handle.get(), plan.cptr(), &workspaceSize);
    if (wsRet != ACL_SPARSE_STATUS_SUCCESS || workspaceSize == 0) {
        throw std::runtime_error("search chain ctx: GetWorkspace failed");
    }
    dWorkspace = sparse_test::DeviceBuffer::alloc(workspaceSize);
}

// 基座剪枝 + 同步段（matA 有效输入填充 dPruned，STRIP）。
inline void PruneCtxInput(sparse_test::SparseLtHandleGuard& handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc,
    sparse_test::DeviceBuffer& dA, sparse_test::DeviceBuffer& dPruned,
    aclrtStream stream)
{
    aclsparseLtConstMatmulDescriptor_t mdForPrune = matmulDesc.get();
    aclsparseStatus_t pruneRet = aclsparseLtSpMMAPrune(
        handle.get(), &mdForPrune, dA.get(), dPruned.get(),
        ACLSPARSELT_PRUNE_SPMMA_STRIP, stream);
    if (pruneRet != ACL_SPARSE_STATUS_SUCCESS ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("search chain ctx: prune failed");
    }
}

// vec 属性条件设置（enable=false 直通）——VecSearch / LiveRead 基座共享段。
inline void SetVecAttrIf(sparse_test::SparseLtHandleGuard& handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc, bool enable,
    aclsparseLtMatmulDescAttribute_t attr, const char* errMsg)
{
    if (!enable) { return; }
    int one = 1;
    if (aclsparseLtMatmulDescSetAttribute(handle.get(), &matmulDesc.get(), attr,
            &one, sizeof(int)) != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(errMsg);
    }
}

// SEARCH_ITERATIONS 预置（PlanInit 前 Set）——VecSearch / LiveRead / X 用例
// 链共享段。
inline void SetSearchIters(sparse_test::SparseLtHandleGuard& handle,
    sparse_test::SparseLtAlgSelectionGuard& algSel, int32_t iters,
    const char* errMsg)
{
    if (aclsparseLtMatmulAlgSetAttribute(handle.get(), algSel.ptr(),
            ACLSPARSELT_MATMUL_SEARCH_ITERATIONS, &iters, sizeof(int32_t))
            != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(errMsg);
    }
}

// ============================================================================
// 异常基座：FP32 128³ A-sparse 默认预置全链（handle→descs→algSel(0,1,0)
// →plan→ws→prune→buffers）。SearchRaw 注入单因子变异（X01-X17 / X37）。
// ============================================================================
struct SearchExceptionCtx : SearchChainCtxBase {
    explicit SearchExceptionCtx(int32_t m = 128, int32_t k = 128, int32_t n = 128)
    {
        MakeSearchDescQuartet<float>(handle, m, k, n, matA, matB, matC, matD);
        matmulDesc = std::make_unique<sparse_test::SparseLtMatmulDescGuard>(
            handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            matA->get(), matB->get(), matC->get(), matD->get(), ACL_SPARSE_COMPUTE_32F);
        algSel = std::make_unique<sparse_test::SparseLtAlgSelectionGuard>(
            handle.get(), matmulDesc->get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
        plan = std::make_unique<sparse_test::SparseLtPlanGuard>(
            handle.get(), matmulDesc->get(), algSel->get());
        if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
            throw std::runtime_error("SearchExceptionCtx: aclrtCreateStream failed");
        }
        InitExcChainDataAndWs(handle, *plan, m, k, n, dA, dB, dC, dD, dPruned,
                              dWorkspace, workspaceSize);
        PruneCtxInput(handle, *matmulDesc, dA, dPruned, stream);
    }
};

// 单因子变异的原始 Search 调用。aclsparseLtMatmulSearch 与 aclsparseLtMatmul
// 的参数列表除 plan 外逐一同构——公共 10 字段继承 MatmulRawParams（codecheck8
// 跨文件重复段消重）。基类 plan（Const 单指针，Matmul 专用）在 Search 场景闲置
// （恒 nullptr，无副作用）；Search 的 plan 为可变双指针（API 内回写调优后的
// plan），独立命名为 searchPlan（不 shadow 基类字段）。
struct SearchRawParams : MatmulRawParams {
    aclsparseLtMatmulPlan_t* searchPlan = nullptr;
};

inline aclsparseStatus_t SearchRaw(const SearchRawParams& p)
{
    return aclsparseLtMatmulSearch(p.handle, p.searchPlan, p.alpha, p.matA, p.matB, p.beta,
                                   p.matC, p.matD, p.workspace, p.streams, p.numStreams);
}

// 基座默认参数（全部有效）
inline SearchRawParams DefaultRawParams(SearchExceptionCtx& ctx)
{
    SearchRawParams p;
    p.handle = ctx.handle.get();
    p.searchPlan = &ctx.plan->get();
    p.alpha = &ctx.alpha;
    p.matA = ctx.dPruned.get();
    p.matB = ctx.dB.get();
    p.beta = &ctx.beta;
    p.matC = ctx.dC.get();
    p.matD = ctx.dD.get();
    p.workspace = ctx.dWorkspace.get();
    p.streams = &ctx.stream;
    p.numStreams = 1;
    return p;
}

// ============================================================================
// 参数组合合法性（黑盒防御；方案用例矩阵内的互斥约束）
// 口径：INT8 + FP 域 epilogue（bias/GeLU/α≠1/β≠0）组合纳入——按 ltmatmul
// 权威口径两级分派判定（VerifyInt8Int32SearchOutput）：纯整数路径 EXACT；
// FP 域介入 MERE atol=1/rtol=0/lim=65。
// ============================================================================
inline bool ValidateParamCombo(const MatmulSearchParam& p)
{
    // matA=nullptr × batch>1 由 X18（TEST_F）覆盖负向（INVALID_VALUE）；CSV 正向无此组合
    if (p.mat_a_null && (p.isDensePath() || !p.isSparseA())) {
        ADD_FAILURE() << p.caseId() << ": mat_a_null 仅支持 A-sparse（回退路径域约束）";
        return false;
    }
    return true;
}

// ============================================================================
// BuildSearchArgs：MatmulSearchParam → 搜索链参数
// ============================================================================
template <typename InT, typename OutT>
inline MatmulSearchRunArgs<InT, OutT> BuildSearchArgs(const MatmulSearchParam& p,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    const SearchVecs& vecs, std::vector<OutT>& dummyD)
{
    MatmulSearchRunArgs<InT, OutT> args{
        p.m, p.k, p.n, hA, hB, hC, dummyD,
        p.alpha, p.beta, p.algConfigId, p.splitK, p.splitKMode,
        ToNpuOrder(p), ToNpuOp(p.isTransA()), ToNpuOp(p.isTransB()), ToNpuPruneAlg(p),
        p.alpha_vector_scaling, p.beta_vector_scaling, vecs.alphaPtr, vecs.betaPtr
    };
    args.biasEnabled = p.biasEnabled;
    args.biasStride = p.biasStride;
    args.activationType = p.activationType;
    args.reluUpperBound = p.reluUpperBound;
    args.reluThreshold = p.reluThreshold;
    args.geluScaling = p.geluScaling;
    args.numBatches = p.numBatches;
    args.batchStride = p.batchStride;
    args.biasVec = vecs.biasPtr;
    args.biasEltSize = BiasDtypeTrait<InT>::kBiasEltSize;
    args.biasDtype = BiasDtypeTrait<InT>::kBiasAclDtype;
    // 路径
    args.isSparseA = p.isSparseA();
    args.isDensePath = p.isDensePath();
    // 搜索扩展
    args.searchIterations = p.search_iterations;
    args.checkTiming = (p.check_timing == 1);
    args.inplace = (p.inplace == 1);
    args.matANull = (p.mat_a_null == 1);
    args.streamNull = (p.stream_null == 1);
    args.alphaNull = (p.alpha_null == 1);
    args.betaNull = (p.beta_null == 1);
    args.numStreams = p.num_streams;
    // ld/order 布局覆盖
    args.layout = ToNpuLayoutOpts(p);
    return args;
}

// in-place 用例（B2）记录：SUCCESS + 合法配置，matD 不做值断言（in-place 累积语义，值不可预测）
template <typename OutT>
inline void RecordInplaceDeclarative(const MatmulSearchParam& p,
    const MatmulSearchNpuResult<OutT>& result)
{
    SearchAssertionRecord rec;
    rec.caseId = p.caseId();
    rec.tensor = "matD(in-place)";
    rec.metric = "declarative(SUCCESS + legal candidate, no value assertion)";
    rec.measured = "searchRet=" + std::to_string(result.searchRet) +
        ", cfg=" + std::to_string(result.selCfg) + ", s=" + std::to_string(result.selSplitK) +
        ", mode=" + std::to_string(result.selSplitKMode);
    rec.threshold = "SUCCESS + Tier-1 legal";
    const bool ok = (result.searchRet == ACL_SPARSE_STATUS_SUCCESS);
    rec.verdict = VerdictOf(ok);
    AppendSearchRecord(rec);
}

}  // namespace matmulsearch_test

// ============================================================================
// X 用例共享 fixture（TEST_F 定义分布在 matmulSearch_test.cpp（X01~X19）/
// matmulSearch_test_xcases.cpp（X20~X23）/ matmulSearch_test_live.cpp
// （X31~X50）三个 TU——fixture 单一定义于本头，gtest 跨 TU 合并同一
// registry；原单文件内定义，拆分后上移共享层）。
// ============================================================================
class MatmulSearchExceptionTest : public testing::Test
{
  protected:
    void SetUp() override { (void)matmulsearch_test::SearchAclEnvManager::Get(); }
};

#endif  // TEST_SPARSELT_MATMULSEARCH_TEST_HELPERS_H_
