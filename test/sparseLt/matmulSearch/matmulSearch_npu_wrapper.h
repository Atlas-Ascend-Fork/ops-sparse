/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SPARSELT_MATMULSEARCH_NPU_WRAPPER_H_
#define TEST_SPARSELT_MATMULSEARCH_NPU_WRAPPER_H_

// =============================================================================
// aclsparseLtMatmulSearch NPU wrapper —— 搜索链。
//
// 与 ltmatmul 一次性 chain runner 不同：搜索链需在 plan/workspace 建立后
// 插入 Search 并保持 ctx 存活做后置断言，故拆为「建链 runner + 后置 helper」，
// 复用 ltmatmul_test_utils.h / ltmatmul_npu_wrapper.h / ltmatmul_npu_chain.h
// 的可分离助手（RAII Guard / Trait / Prepare* / Set*Attrs / 剪枝回读）。
//
// 计时统计与测量协议（SearchTimingStats / MatmulExecPointers /
// MeasurePlanTiming / MeasureInterleavedTiming / SearchDefaultBaseline 等）已拆
// 至 matmulSearch_timing.h；两 runner 的「属性 + plan + ws」与
// 「热身窗 + Search + 后置」公共段分别为 SetupSearchChainAttrsPlan /
// FinishSearchChainWithBaseline（A-sparse / B-sparse 链共用）。
//
// 链步骤：
//   1. handle → mat 描述符（A-sparse / B-sparse / dense×dense）→ matmulDescr
//      （+ vec/bias/act/batch 属性）→ Device buffers
//   2. AlgSelectionInit → AlgSetAttribute(预置 cfg/splitK/mode + SEARCH_ITERATIONS)
//      → GetWorkspace(wsPre) → 分配 ws → PlanInit
//   3. 稀疏路径 SpMMAPrune（dA→dPruned；mat_a_null=1 时 → ws+aPrunedOffset，
//      偏移经 compute_ws_layout 白盒求取）→ 回读剪枝结果供 golden
//   4. ★ aclsparseLtMatmulSearch(handle, plan, α, matA, matB, β, matC, matD,
//      ws, streams, numStreams)（计时类用例 T(default) 基线在 Search 前完成）
//   5. 后置：无同步回读 matD（阻塞语义声明性检查）→
//      AlgGetAttribute 三元组 → GetWorkspace(wsPost) → 搜索后再执行 matmul
//      （新鲜 C/D 缓冲 + 原 workspace 缓冲，S3 复用语义）→（计时）T(c*) 测量
//
// 数值比对与 Tier-1/Tier-2 谓词断言由 matmulSearch_test.cpp 编排（谓词函数
// 在 matmulSearch_golden.h）；本 wrapper 只负责执行与数据回收。
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"

#include "ltmatmul_npu_wrapper.h"     // MatmulNpuResult, guards, Set*Attrs（含 chain/utils 依赖）
#include "ltmatmul_npu_chain.h"       // PrepareMatmulChainContext / PrepareBSparseChainContext / chain ctx

#include "matmulSearch_golden.h"      // EffectiveSplitKOf / WhiteboxAPrunedOffset / SpltDtypeCodeOf
#include "matmulSearch_timing.h"      // SearchTimingStats / MatmulExecPointers / 测量协议 / 默认基线

namespace sparse_test {

// -----------------------------------------------------------------------------
// 搜索链运行参数：MatmulRunArgs（ltmatmul 基础参数）+ 搜索扩展。
// 注意：args.hD 在搜索链中不使用（Search 输出经 result.hDSearch / hDFinal 返回）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT, typename CType = InT>
struct MatmulSearchRunArgs : public MatmulRunArgs<InT, OutT> {
    // 路径分发（由 test.cpp 按 matrix_type / sparse_side 填充）
    bool isSparseA = true;
    bool isDensePath = false;
    // ---- 搜索扩展（CSV 新增列）----
    int32_t searchIterations = 5;
    bool checkTiming = false;
    bool inplace = false;      // matC 与 matD 传同一 Device 缓冲（B2）
    bool matANull = false;     // matA=nullptr，剪枝结果预写 ws+aPrunedOffset（B3，仅 A-sparse）
    bool streamNull = false;   // streams[0]=nullptr（默认流，B4）
    bool alphaNull = false;    // alpha 指针传 nullptr（语义 1.0，B7）
    bool betaNull = false;     // beta  指针传 nullptr（语义 0.0，B6）
    int32_t numStreams = 1;    // B5=2（双流数组，仅用 streams[0] 被使用）
    // ---- 计时协议（预热 3 + 测量 10 取 min）----
    int32_t timingWarmup = 3;
    int32_t timingMeasure = 10;
    // ld/order 布局覆盖已在基类 MatmulRunArgs::layout（默认紧凑）

};

// -----------------------------------------------------------------------------
// 搜索链结果：继承 ltmatmul MatmulNpuResult（复用其 Set/Get/剪枝字段），
// 扩展 Search 后置产出。输出向量类型随 OutT。
// -----------------------------------------------------------------------------
template <typename OutT>
struct MatmulSearchNpuResult : public MatmulNpuResult {
    // ---- 阶段执行标志 + 返回码 ----
    aclsparseStatus_t searchRet = ACL_SPARSE_STATUS_SUCCESS;
    bool rerunExecuted = false;      // 搜索后再执行 / T(c*) 末次测量是否完成
    aclsparseStatus_t rerunRet = ACL_SPARSE_STATUS_SUCCESS;
    // ---- 搜索后三元组（经 AlgGetAttribute 回读 plan 绑定的 algSelection）----
    aclsparseStatus_t algGetCfgRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algGetSplitRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algGetSplitKModeRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algGetMaxCfgIdRet = ACL_SPARSE_STATUS_SUCCESS;
    int32_t selCfg = -1;
    int32_t selSplitK = -1;
    int32_t selSplitKMode = -1;
    int32_t maxCfgId = -1;      // AlgGetAttribute(ALG_CONFIG_MAX_ID) 动态查询值
    // ---- workspace（方案 A）----
    aclsparseStatus_t algSetItersRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t wsPreRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t wsPostRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t wsDefaultRet = ACL_SPARSE_STATUS_SUCCESS;
    int64_t wsPre = -1;
    int64_t wsPost = -1;
    int64_t wsDefaultNeed = -1;  // 计时基线默认 plan 的 ws 需求（应 <= wsPre）
    // ---- 阻塞语义声明性检查：Search 返回后【无任何 aclrtSynchronize*】直接回读 ----
    aclError blockedReadRet = ACL_SUCCESS;
    std::vector<OutT> hDSearch;   // Search 返回的 matD（无同步直读；B2 in-place 不做值断言）
    // ---- 搜索后再执行（新鲜 C/D 缓冲 + 原 workspace 缓冲，plan 生效性 + S3 复用）----
    std::vector<OutT> hDFinal;
    // ---- 计时（check_timing=1；逐次交错 + 顺序平衡协议）----
    // 设计目标：T(default) 与 T(c*) 的样本在时间轴上完全混叠，环境噪声（并发
    // 负载的时段切换、同步后首执行位的位置效应）对两侧 min 的影响对称。
    // 序列：[def-热身窗(warmup-only，仅预热设备/流，不计统计)] → Search →
    // 逐次交错（对内顺序逐对反转：偶对 [def,c*] / 奇对 [c*,def]，order
    // balancing 抵消同步后首执行位效应）×20 对 → 补一次 c* 单独执行（生效性输出）。
    // 统计量 min、ε=10%、样本 20/侧（每窗预热 3 + 测量 10 对齐）。
    // tSelWin1/2 与 tDefaultWin1/2 = interleaved 块时间序前/后半（明细留痕字段）。
    bool hasTiming = false;
    SearchTimingStats tDefault;   // 合并：def窗1 ∪ def窗2（Search 后两窗，20 样本取 min）
    SearchTimingStats tSel;       // 合并：c*-窗1 ∪ c*-窗2（20 样本取 min）；末窗末次输出兼作生效性比对
    SearchTimingStats tDefaultPre;   // def-热身窗（warmup-only：仅预热设备/流，不计入 T(default) 统计，明细留痕）
    SearchTimingStats tSelWin1;      // c*-窗1（Search 后第 1 窗）
    SearchTimingStats tDefaultWin1;  // def-窗1（Search 后与 c* 窗紧邻交错）
    SearchTimingStats tDefaultWin2;  // def-窗2（Search 后第 4 窗）
    SearchTimingStats tSelWin2;      // c*-窗2（末窗末次输出兼作生效性比对）
    // ---- mat_a_null 回退路径 ----
    int64_t aPrunedOffsetUsed = -1;  // ws+aPrunedOffset 白盒偏移（B3）
};

// -----------------------------------------------------------------------------
// ExecMatmulOnceAndCopy：单次 matmul（同步）+ 回读输出（搜索后再执行 / S3）。
// -----------------------------------------------------------------------------
template <typename OutT>
inline aclsparseStatus_t ExecMatmulOnceAndCopy(aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* plan, const MatmulExecPointers& ptr,
    aclrtStream stream, size_t outElems, size_t outEltSize,
    DeviceBuffer& dOut, std::vector<OutT>& hOut)
{
    aclsparseStatus_t ret = aclsparseLtMatmul(handle, plan, ptr.alpha, ptr.matA, ptr.matB,
        ptr.beta, ptr.matC, ptr.matD, ptr.workspace, &stream, 1);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    hOut.resize(outElems);
    if (aclrtMemcpy(hOut.data(), outElems * outEltSize, dOut.get(),
                    outElems * outEltSize, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ret;
}

// -----------------------------------------------------------------------------
// SearchStreamGuard：显式流 RAII。
// -----------------------------------------------------------------------------
class SearchStreamGuard {
public:
    SearchStreamGuard() {
        if (aclrtCreateStream(&stream_) != ACL_SUCCESS) {
            throw std::runtime_error("SearchStreamGuard: aclrtCreateStream failed");
        }
    }
    ~SearchStreamGuard() { if (stream_) { aclrtDestroyStream(stream_); } }
    SearchStreamGuard(const SearchStreamGuard&) = delete;
    SearchStreamGuard& operator=(const SearchStreamGuard&) = delete;
    aclrtStream get() const { return stream_; }
private:
    aclrtStream stream_ = nullptr;
};

// -----------------------------------------------------------------------------
// RunPruneToTarget：SpMMAPrune 剪枝到指定目标指针（独立 dPruned 缓冲或
// ws+aPrunedOffset），batch 循环 + 同步 + 回读剪枝结果供 golden。
// 仿 ltmatmul RunPruneForASparse，目标指针参数化（方案 B3 回退路径）。
// -----------------------------------------------------------------------------
// Prune 单调用化（prune host batch 增强后）：描述符携带 numBatches/batchStride，
// 一次 aclsparseLtSpMMAPrune 调用处理全 batch——传 batch0 基址即可。host 侧
// 不得再按 batch 切片循环（切片会使末切片基址被算子按 numBatches 二次推进而
// 越界）。输出与输入共用 batchStride 推进；回读 totalElems（BatchTotalElems
// 语义）的布局与切片版一致，golden 消费方式不变。
inline bool RunPruneToTarget(MatmulNpuResult& result,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulDescriptor_t matmulDesc,
    DeviceBuffer& dIn, uint8_t* targetBase,
    size_t totalElems, size_t kInElt, aclsparseLtPruneAlg_t pruneAlg,
    aclrtStream stream)
{
    result.pruneRet = aclsparseLtSpMMAPrune(
        handle, &matmulDesc, dIn.get(), targetBase, pruneAlg, stream);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }
    result.npuPrunedA.resize(totalElems * kInElt);
    result.memcpyRet = aclrtMemcpy(result.npuPrunedA.data(), totalElems * kInElt,
        targetBase, totalElems * kInElt, ACL_MEMCPY_DEVICE_TO_HOST);
    if (result.memcpyRet != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// SetupSearchAlgPlanWs：algSel 预置（cfg/splitK/mode + SEARCH_ITERATIONS）→
// PlanInit → GetWorkspace(wsPre) → alloc ws。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT, typename CType = InT>
inline bool SetupSearchAlgPlanWs(const MatmulSearchRunArgs<InT, OutT, CType>& args,
    SparseLtHandleGuard& handle, SparseLtMatmulDescGuard& matmulDesc,
    MatmulSearchNpuResult<OutT>& result,
    std::unique_ptr<SparseLtAlgSelectionGuard>& algSel,
    std::unique_ptr<SparseLtPlanGuard>& plan, DeviceBuffer& dWorkspace, void*& wsPtr)
{
    algSel = std::make_unique<SparseLtAlgSelectionGuard>(handle.get(), matmulDesc.get(),
                                                         ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    // 预置三元组（本套件语义：alg_config_id/split_k/split_k_mode = 初始预置属性）
    if (!SetMatmulAlgAttributes(handle.get(), algSel->get(), args.algConfigId,
                                args.splitK, args.splitKMode, result)) {
        return false;
    }
    // SEARCH_ITERATIONS 预置（PlanInit 前 Set；X17：0 被拒绝）
    result.algSetItersRet = aclsparseLtMatmulAlgSetAttribute(
        handle.get(), algSel->ptr(), ACLSPARSELT_MATMUL_SEARCH_ITERATIONS,
        &args.searchIterations, sizeof(int32_t));
    if (result.algSetItersRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }

    plan = std::make_unique<SparseLtPlanGuard>(handle.get(), matmulDesc.get(), algSel->get());
    size_t wsPreBytes = 0;
    result.wsPreRet = aclsparseLtMatmulGetWorkspace(handle.get(), plan->cptr(), &wsPreBytes);
    result.wsPre = static_cast<int64_t>(wsPreBytes);
    if (result.wsPreRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    if (result.wsPre <= 0) {
        // compute_ws_layout 恒含 tiling+aPruned 区，wsPre 必 > 0；防御记录
        result.wsPreRet = ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
        return false;
    }
    dWorkspace = DeviceBuffer::alloc(static_cast<size_t>(result.wsPre));
    wsPtr = dWorkspace.get();
    return true;
}

// -----------------------------------------------------------------------------
// SearchChainSetupResult + SetupSearchChainAttrsPlan：两 runner 共用的
// 「batchSetRet 检查 → vec scaling → epilogue → algSel 预置 + plan + wsPre」
// 公共段（步骤 1 属性段 + 步骤 2），自 RunMatmulSearchChain /
// RunMatmulSearchBSparseChain 提取以消除重复代码（ltmatmul
// ChainSetupResult / SetupVecEpiPlanWorkspace 同范式）。vec/epi 的 Device
// 缓冲随 setup 存活至 runner 返回（与原局部变量生命周期一致）。
// -----------------------------------------------------------------------------
struct SearchChainSetupResult {
    VecScalingBuffers vecBufs;
    EpilogueBuffers epiBufs;
    std::unique_ptr<SparseLtAlgSelectionGuard> algSel;
    std::unique_ptr<SparseLtPlanGuard> plan;
    DeviceBuffer dWorkspace;
    void* wsPtr = nullptr;
};

template <typename InT, typename OutT>
inline bool SetupSearchChainAttrsPlan(SparseLtHandleGuard& handle,
    SparseLtMatmulDescGuard& matmulDesc, const MatmulSearchRunArgs<InT, OutT>& args,
    MatmulSearchNpuResult<OutT>& result, SearchChainSetupResult& setup)
{
    if (result.batchSetRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    setup.vecBufs = SetVecScalingAttrs(handle.get(), matmulDesc, args.m,
        args.alpha_vector_scaling, args.beta_vector_scaling, args.alphaVec, args.betaVec, result);
    if (result.descSetRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    setup.epiBufs = SetEpilogueAttrs(handle.get(), matmulDesc, args.m,
        args.biasEnabled, args.biasStride, args.activationType, args.reluUpperBound,
        args.reluThreshold, args.geluScaling, args.numBatches, args.batchStride,
        args.biasVec, result, args.biasEltSize, args.biasDtype);
    if (result.biasSetRet != ACL_SPARSE_STATUS_SUCCESS ||
        result.actSetRet != ACL_SPARSE_STATUS_SUCCESS) {
        return false;
    }
    return SetupSearchAlgPlanWs(args, handle, matmulDesc, result,
                                setup.algSel, setup.plan, setup.dWorkspace, setup.wsPtr);
}

// -----------------------------------------------------------------------------
// PruneASparseForSearch：A-sparse 剪枝（步骤 3）——独立 dPruned 缓冲或
// ws+aPrunedOffset 白盒偏移目标（mat_a_null 回退路径），batch 循环 + 回读。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT, typename CType = InT>
inline bool PruneASparseForSearch(const MatmulSearchRunArgs<InT, OutT, CType>& args,
    SparseLtHandleGuard& handle, SparseLtMatmulDescGuard& matmulDesc,
    DeviceBuffer& dA, DeviceBuffer& dPruned, void* wsPtr,
    size_t mk, size_t kInElt, aclrtStream stream,
    MatmulSearchNpuResult<OutT>& result, void*& matAPtr)
{
    const int32_t presetEffSplitK = EffectiveSplitKOf(args.splitKMode, args.splitK);
    const int32_t dtypeCode = SpltDtypeCodeOf(NpuDtypeTrait<InT>::kAclDtype);
    const size_t totalElems = BatchTotalElems(mk, args.numBatches, args.batchStride);
    uint8_t* targetBase = nullptr;
    if (args.matANull) {
        result.aPrunedOffsetUsed = WhiteboxAPrunedOffset(args.m, args.n, args.k,
            presetEffSplitK, dtypeCode, args.numBatches);
        targetBase = static_cast<uint8_t*>(wsPtr) + result.aPrunedOffsetUsed;
    } else {
        dPruned = DeviceBuffer::alloc(totalElems * kInElt);
        targetBase = static_cast<uint8_t*>(dPruned.get());
    }
    if (!RunPruneToTarget(result, handle.get(), matmulDesc.get(), dA,
            targetBase, totalElems, kInElt, args.pruneAlg, stream)) {
        return false;
    }
    matAPtr = args.matANull ? nullptr : dPruned.get();
    return true;
}

// -----------------------------------------------------------------------------
// BuildSearchExecPointers：按 args 组装执行指针
// （alpha/beta nullptr 语义、vec scaling、matA nullptr 回退由 matAPtr 传入）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT, typename CType = InT>
inline MatmulExecPointers BuildSearchExecPointers(const MatmulSearchRunArgs<InT, OutT, CType>& args,
    VecScalingBuffers& vecBufs, float& alphaF, float& betaF,
    void* matAPtr, void* matBPtr, void* matCPtr, void* matDPtr, void* wsPtr)
{
    MatmulExecPointers ptr;
    if (args.alphaNull) {
        ptr.alpha = nullptr;  // 语义 1.0（B7）
    } else if (vecBufs.dAlphaVec.get() != nullptr) {
        // 含 bvec 单独的隐含提升（BETA implies ALPHA）：dAlphaVec 已分配即用向量
        ptr.alpha = vecBufs.dAlphaVec.get();
    } else {
        ptr.alpha = &alphaF;
    }
    if (args.betaNull) {
        ptr.beta = nullptr;   // 语义 0.0（B6）
    } else if (vecBufs.dBetaVec.get() != nullptr) {
        ptr.beta = vecBufs.dBetaVec.get();
    } else {
        ptr.beta = &betaF;
    }
    ptr.matA = matAPtr;   // matANull 时为 nullptr（回退 ws+aPruned 区）
    ptr.matB = matBPtr;
    ptr.matC = matCPtr;   // inplace 时 == matDPtr（B2）
    ptr.matD = matDPtr;
    ptr.workspace = wsPtr;
    return ptr;
}

// -----------------------------------------------------------------------------
// QueryPostSearch：搜索后置查询（三元组 + maxCfgId + wsPost）。
// -----------------------------------------------------------------------------
template <typename OutT>
inline void QueryPostSearch(SparseLtHandleGuard& handle,
    SparseLtAlgSelectionGuard& algSel, SparseLtPlanGuard& plan,
    MatmulSearchNpuResult<OutT>& result)
{
    result.algGetCfgRet = aclsparseLtMatmulAlgGetAttribute(
        handle.get(), algSel.cptr(), ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
        &result.selCfg, sizeof(int32_t));
    result.algGetSplitRet = aclsparseLtMatmulAlgGetAttribute(
        handle.get(), algSel.cptr(), ACLSPARSELT_MATMUL_SPLIT_K,
        &result.selSplitK, sizeof(int32_t));
    result.algGetSplitKModeRet = aclsparseLtMatmulAlgGetAttribute(
        handle.get(), algSel.cptr(), ACLSPARSELT_MATMUL_SPLIT_K_MODE,
        &result.selSplitKMode, sizeof(int32_t));
    result.algGetMaxCfgIdRet = aclsparseLtMatmulAlgGetAttribute(
        handle.get(), algSel.cptr(), ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
        &result.maxCfgId, sizeof(int32_t));
    size_t wsPostBytes = 0;
    result.wsPostRet = aclsparseLtMatmulGetWorkspace(handle.get(), plan.cptr(), &wsPostBytes);
    result.wsPost = static_cast<int64_t>(wsPostBytes);
}

// -----------------------------------------------------------------------------
// BlockedReadMatD：Search 返回后【不调用任何 aclrtSynchronize*】直接 aclrtMemcpy
// 回读 matD —— 阻塞语义声明性检查（返回后无任何 aclrtSynchronize* 直接拷贝）。
// -----------------------------------------------------------------------------
template <typename OutT>
inline void BlockedReadMatD(DeviceBuffer& dD, size_t outElems, size_t outEltSize,
    MatmulSearchNpuResult<OutT>& result)
{
    result.hDSearch.resize(outElems);
    result.blockedReadRet = aclrtMemcpy(result.hDSearch.data(), outElems * outEltSize,
        dD.get(), outElems * outEltSize, ACL_MEMCPY_DEVICE_TO_HOST);
}

// -----------------------------------------------------------------------------
// RerunAfterSearch：搜索后再执行一次 matmul（新鲜 C/D 缓冲 + 原 workspace 缓冲），
// 验证「搜索后 plan 生效性」与 workspace 复用语义（原缓冲直接可用）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT, typename CType = InT>
inline void RerunAfterSearch(const MatmulSearchRunArgs<InT, OutT, CType>& args,
    SparseLtHandleGuard& handle, SparseLtPlanGuard& plan,
    const MatmulExecPointers& searchPtr, aclrtStream stream,
    const std::vector<CType>& hC, size_t cBytes, size_t outElems, size_t outEltSize,
    MatmulSearchNpuResult<OutT>& result)
{
    DeviceBuffer dC2 = DeviceBuffer::copyFrom(hC.data(), cBytes);   // 新鲜 C
    DeviceBuffer dD2 = DeviceBuffer::alloc(outElems * outEltSize);  // 新鲜 D
    MatmulExecPointers rerunPtr = searchPtr;
    rerunPtr.matC = dC2.get();
    rerunPtr.matD = dD2.get();
    result.rerunRet = ExecMatmulOnceAndCopy<OutT>(handle.get(), plan.cptr(), rerunPtr,
        stream, outElems, outEltSize, dD2, result.hDFinal);
    result.rerunExecuted = (result.rerunRet == ACL_SPARSE_STATUS_SUCCESS);
}

// -----------------------------------------------------------------------------
// SearchCallAndPostPass：★ Search 调用 + 后置（步骤 4b + 5），两 runner 共用。
// 计时编排：runner 侧已在 Search 前完成 def-热身窗（warmup-only）；
// Search 后执行【逐次交错】测量（对内顺序逐对反转，pairs = timingMeasure×2 = 20/侧）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT, typename CType = InT>
inline void SearchCallAndPostPass(const MatmulSearchRunArgs<InT, OutT, CType>& args,
    SparseLtHandleGuard& handle, SparseLtAlgSelectionGuard& algSel,
    SparseLtPlanGuard& plan, const MatmulExecPointers& execPtr,
    DeviceBuffer& dD, aclrtStream stream,
    SearchDefaultBaseline& baseline,
    const std::vector<CType>& hC, size_t cBytes,
    size_t totalOutElems, size_t kOutElt, size_t mn,
    MatmulSearchNpuResult<OutT>& result)
{
    // ---- 步骤 4b：★ aclsparseLtMatmulSearch ----
    aclrtStream secondStream = stream;
    std::unique_ptr<SearchStreamGuard> stream2;  // B5 双流（仅 streams[0] 被使用）
    if (args.numStreams > 1) {
        stream2 = std::make_unique<SearchStreamGuard>();
        secondStream = stream2->get();
    }
    aclrtStream streamsArr[2] = {
        args.streamNull ? static_cast<aclrtStream>(nullptr) : stream,
        secondStream
    };
    result.searchRet = aclsparseLtMatmulSearch(
        handle.get(), &plan.get(),
        execPtr.alpha, execPtr.matA, execPtr.matB, execPtr.beta,
        execPtr.matC, execPtr.matD, execPtr.workspace,
        streamsArr, args.numStreams);
    if (result.searchRet != ACL_SPARSE_STATUS_SUCCESS) { return; }

    // ---- 步骤 5：后置 ----
    // 5.1 阻塞语义声明性检查：无任何 aclrtSynchronize* 直接回读 matD
    BlockedReadMatD(dD, totalOutElems, kOutElt, result);
    // 5.2 三元组 + maxCfgId + wsPost 查询
    QueryPostSearch(handle, algSel, plan, result);

    if (args.checkTiming) {
        // 5.3 逐次交错 + 顺序平衡测量：对内顺序逐对反转（偶 [def,c*] /
        // 奇 [c*,def]），pairs=timingMeasure×2（每侧 20 样本，统计量 min）。
        const int32_t pairs = args.timingMeasure * 2;
        aclsparseStatus_t timingRet = MeasureInterleavedTiming(handle.get(), plan.cptr(),
            baseline.planDefault->cptr(), execPtr, stream, args.timingWarmup, pairs,
            result.tSel, result.tDefault);
        if (timingRet != ACL_SPARSE_STATUS_SUCCESS) {
            result.rerunRet = timingRet;
            return;
        }
        // 窗口明细留痕（interleaved 块前/后半，保持字段与日志格式兼容）
        SplitTimingStats(result.tSel, result.tSelWin1, result.tSelWin2);
        SplitTimingStats(result.tDefault, result.tDefaultWin1, result.tDefaultWin2);
        // 生效性比对输出：交错块后补一次 c* 单独执行（计时外）写入 dD 并回读
        // （顺序平衡后末位执行不再固定为 c*，故显式补执行）
        result.rerunRet = ExecMatmulOnceAndCopy<OutT>(handle.get(), plan.cptr(), execPtr,
            stream, totalOutElems, kOutElt, dD, result.hDFinal);
        result.rerunExecuted = (result.rerunRet == ACL_SPARSE_STATUS_SUCCESS);
    } else {
        // 5.3' 搜索后再执行 matmul（新鲜 C/D + 原 ws 缓冲）
        RerunAfterSearch<InT, OutT, CType>(args, handle, plan, execPtr, stream, hC, cBytes,
            totalOutElems, kOutElt, result);
    }
}

// -----------------------------------------------------------------------------
// RunDefaultBaselineWarmup：def-热身窗（check_timing；Search 前完成，
// warmup-only：仅预热设备/流，样本不计入 T(default) 统计——两侧统计窗口须
// 同处 Search 之后才能时段对称）。自 FinishSearchChainWithBaseline 与
// RunSearchChainInt8CD 的共用段；返回 false 时失败码已写入 result
// （BuildDefaultBaselinePlan → wsDefaultRet / MeasurePlanTiming → wsDefaultRet）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT, typename CType = InT>
inline bool RunDefaultBaselineWarmup(const MatmulSearchRunArgs<InT, OutT, CType>& args,
    SparseLtHandleGuard& handle, SparseLtMatmulDescGuard& matmulDesc,
    const MatmulExecPointers& execPtr, aclrtStream stream,
    SearchDefaultBaseline& baseline, MatmulSearchNpuResult<OutT>& result)
{
    if (!BuildDefaultBaselinePlan(handle.get(), matmulDesc, baseline,
                                  result.wsDefaultRet, result.wsDefaultNeed)) {
        return false;
    }
    aclsparseStatus_t timingRet = MeasurePlanTiming(handle.get(),
        baseline.planDefault->cptr(), execPtr, stream,
        args.timingWarmup, args.timingMeasure, result.tDefaultPre);
    if (timingRet != ACL_SPARSE_STATUS_SUCCESS) {
        result.wsDefaultRet = timingRet;
        return false;
    }
    result.hasTiming = true;
    return true;
}

// -----------------------------------------------------------------------------
// FinishSearchChainWithBaseline：两 runner 共用的收尾公共段（步骤 4a def-热身窗
// + 步骤 4b/5 Search + 后置），自 RunMatmulSearchChain / RunMatmulSearchBSparseChain
// 提取以消除重复代码。
// cdSpan 参数 = C/D 单 batch 物理 span（紧凑时 == m*n；调用方由 ctx.cdSpan
// 传入）；hCPhys = pack 后的 C host 数据（rerun 新鲜 C 上传用；紧凑时与 args.hC
// 等价，统一走本参数以简化调用方）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline MatmulSearchNpuResult<OutT> FinishSearchChainWithBaseline(
    const MatmulSearchRunArgs<InT, OutT>& args, SparseLtHandleGuard& handle,
    SparseLtMatmulDescGuard& matmulDesc,
    std::unique_ptr<SparseLtAlgSelectionGuard>& algSel,
    std::unique_ptr<SparseLtPlanGuard>& plan,
    const MatmulExecPointers& execPtr, DeviceBuffer& dD, aclrtStream stream,
    size_t cdSpan, size_t totalOutElems, size_t kInElt, size_t kOutElt,
    MatmulSearchNpuResult<OutT>& result,
    const std::vector<InT>& hCPhys)
{
    // ---- 步骤 4a：def-热身窗（共享段；warmup-only）----
    SearchDefaultBaseline baseline;
    if (args.checkTiming &&
        !RunDefaultBaselineWarmup(args, handle, matmulDesc, execPtr, stream,
                                  baseline, result)) {
        return result;
    }

    // ---- 步骤 4b + 5：Search + 后置（含交替窗口其余三窗）----
    const size_t cBytes = BatchTotalElems(cdSpan, args.numBatches, args.batchStride) * kInElt;
    SearchCallAndPostPass(args, handle, *algSel, *plan, execPtr, dD,
        stream, baseline, hCPhys, cBytes, totalOutElems, kOutElt, cdSpan, result);
    return result;
}

// -----------------------------------------------------------------------------
// 顶层 runner（A-sparse / dense×dense）：完整搜索链。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
MatmulSearchNpuResult<OutT> RunMatmulSearchChain(const MatmulSearchRunArgs<InT, OutT>& args,
    bool isSparseA, bool isDensePath)
{
    MatmulSearchNpuResult<OutT> result{};
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize;
    constexpr size_t kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    const size_t mk = static_cast<size_t>(args.m) * args.k;

    SearchStreamGuard streamGuard;
    aclrtStream stream = streamGuard.get();

    // ---- 步骤 1：描述符 + buffers（含 vec/bias/act/batch 属性 + 布局）----
    MatmulChainCtx ctx;
    std::vector<InT> hCPhys;
    PrepareMatmulChainContext<InT, OutT>(args.m, args.k, args.n,
        args.hA, args.hB, args.hC, args.order, args.opA, args.opB,
        isSparseA, isDensePath, args.numBatches, args.batchStride, result, ctx,
        args.layout, &hCPhys);
    // ---- 步骤 1 属性段 + 步骤 2：algSel 预置 + plan + wsPre（两 runner 公共段）----
    SearchChainSetupResult setup;
    if (!SetupSearchChainAttrsPlan<InT, OutT>(ctx.handle, *ctx.matmulDesc, args,
                                              result, setup)) { return result; }

    // ---- 步骤 3：剪枝（A-sparse；mat_a_null → ws+aPrunedOffset 白盒偏移）----
    DeviceBuffer dPruned;
    void* matAPtr = ctx.dA.get();
    if (!isDensePath && isSparseA) {
        if (!PruneASparseForSearch<InT, OutT>(args, ctx.handle, *ctx.matmulDesc, ctx.dA,
                dPruned, setup.wsPtr, mk, kInElt, stream, result, matAPtr)) {
            return result;
        }
    }
    void* matBPtr = ctx.dB.get();

    // ---- 指针组装（in-place / nullptr 语义 / vec scaling）----
    float alphaF = args.alpha;
    float betaF = args.beta;
    void* matCPtr = args.inplace ? ctx.dD.get() : ctx.dC.get();
    void* matDPtr = ctx.dD.get();
    MatmulExecPointers execPtr = BuildSearchExecPointers(args, setup.vecBufs, alphaF, betaF,
        matAPtr, matBPtr, matCPtr, matDPtr, setup.wsPtr);

    // ---- 步骤 4a + 4b + 5：def-热身窗 + Search + 后置（两 runner 公共段）----
    // 回读尺寸 = C/D 物理 span（batch 含 stride padding；紧凑时 == mn*nb）
    const size_t totalOutElems = BatchTotalElems(ctx.cdSpan, args.numBatches, args.batchStride);
    return FinishSearchChainWithBaseline<InT, OutT>(args, ctx.handle, *ctx.matmulDesc,
        setup.algSel, setup.plan, execPtr, ctx.dD, stream, ctx.cdSpan, totalOutElems,
        kInElt, kOutElt, result, hCPhys);
}

// -----------------------------------------------------------------------------
// 顶层 runner（B-sparse）：完整搜索链（matB 为剪枝产出；无 mat_a_null 路径）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
MatmulSearchNpuResult<OutT> RunMatmulSearchBSparseChain(const MatmulSearchRunArgs<InT, OutT>& args)
{
    MatmulSearchNpuResult<OutT> result{};
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize;
    constexpr size_t kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    const size_t kn = static_cast<size_t>(args.k) * args.n;

    SearchStreamGuard streamGuard;
    aclrtStream stream = streamGuard.get();

    // ---- 步骤 1：B-sparse 描述符 + buffers（布局：B ld/order + dense-A + C/D）----
    MatmulBSparseCtx ctx;
    std::vector<InT> hCPhys;
    PrepareBSparseChainContext<InT, OutT>(args.m, args.k, args.n,
        args.hA, args.hB, args.hC, args.order, args.opA, args.opB,
        args.numBatches, args.batchStride, result, ctx, args.layout, &hCPhys);
    // ---- 步骤 1 属性段 + 步骤 2：algSel 预置 + plan + wsPre（两 runner 公共段）----
    SearchChainSetupResult setup;
    if (!SetupSearchChainAttrsPlan<InT, OutT>(ctx.handle, *ctx.matmulDesc, args,
                                              result, setup)) { return result; }

    // ---- 步骤 3：B 剪枝 + 回读（pruned B 直接 (k,n) 布局，matmul/golden 共用）----
    // prune 缓冲上界 = B 物理 span（≥ kn 恒成立；prune 输出恒紧凑 (k,n)）
    DeviceBuffer dBPruned;
    void* matBPtr = nullptr;
    RunBSparsePruneAndTranspose<InT>(result, ctx.handle.get(), ctx.matmulDesc->get(),
        ctx.dB, dBPruned, ctx.bPhysSpan, kInElt, args.pruneAlg, matBPtr,
        args.numBatches, args.batchStride);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }

    float alphaF = args.alpha;
    float betaF = args.beta;
    void* matCPtr = args.inplace ? ctx.dD.get() : ctx.dC.get();
    void* matDPtr = ctx.dD.get();
    MatmulExecPointers execPtr = BuildSearchExecPointers(args, setup.vecBufs, alphaF, betaF,
        ctx.dA.get(), matBPtr, matCPtr, matDPtr, setup.wsPtr);

    // ---- 步骤 4a + 4b + 5：def-热身窗 + Search + 后置（两 runner 公共段）----
    const size_t totalOutElems = BatchTotalElems(ctx.cdSpan, args.numBatches, args.batchStride);
    return FinishSearchChainWithBaseline<InT, OutT>(args, ctx.handle, *ctx.matmulDesc,
        setup.algSel, setup.plan, execPtr, ctx.dD, stream, ctx.cdSpan, totalOutElems,
        kInElt, kOutElt, result, hCPhys);
}

// -----------------------------------------------------------------------------
// RunMatmulSearchNpu：路径分发（A-sparse / B-sparse；dense×dense 为域外路径
// ——matmul 描述符层约束 matA/matB 有且仅一个 structured）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline MatmulSearchNpuResult<OutT> RunMatmulSearchNpu(const MatmulSearchRunArgs<InT, OutT>& args)
{
    if (args.isDensePath) {
        return RunMatmulSearchChain<InT, OutT>(args, true, true);
    }
    if (args.isSparseA) {
        return RunMatmulSearchChain<InT, OutT>(args, true, false);
    }
    return RunMatmulSearchBSparseChain<InT, OutT>(args);
}

}  // namespace sparse_test

// RunSearchChainInt8CD（H 组超集链）位于本目录
// matmulSearch_npu_wrapper_int8cd.h，经本 include 传递，调用方无需感知。
#include "matmulSearch_npu_wrapper_int8cd.h"

#endif  // TEST_SPARSELT_MATMULSEARCH_NPU_WRAPPER_H_
