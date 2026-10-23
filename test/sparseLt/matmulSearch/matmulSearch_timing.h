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

#ifndef TEST_SPARSELT_MATMULSEARCH_TIMING_H_
#define TEST_SPARSELT_MATMULSEARCH_TIMING_H_

// =============================================================================
// aclsparseLtMatmulSearch 计时统计与测量协议。包含：
//   - SearchTimingStats / FinalizeTimingStats / SplitTimingStats：样本统计
//     （min/mean/median/CV；samplesMs 保持时间序留痕）
//   - MatmulExecPointers：一次 matmul / Search 执行的指针参数束
//   - MeasurePlanTiming：单 plan 预热 + event 计时
//   - ExecOneTimedMatmul / RunWarmupPairInOrder / RunTimedPairInOrder /
//     MeasureInterleavedTiming：逐次交错 + 顺序平衡测量
//   - SearchDefaultBaseline / BuildDefaultBaselinePlan：默认基线 plan
// 复用 ltmatmul 测试资产（RAII Guard / EventGuard / aclsparseLtMatmul），
// 不依赖 gtest；数值比对与断言编排见 matmulSearch_test.cpp。
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "acl/acl.h"

#include "ltmatmul_npu_wrapper.h"     // EventGuard, RAII guards, aclsparseLtMatmul（含 utils/chain 依赖）

namespace sparse_test {

// -----------------------------------------------------------------------------
// 计时统计（预热 + 多次测量；性能采集框架另做轮间 median/CV）。
// -----------------------------------------------------------------------------
struct SearchTimingStats {
    bool valid = false;
    std::vector<double> samplesMs;              // 每次测量的 event 耗时（全部保留，留痕可追溯）
    double minMs = 0.0;
    double meanMs = 0.0;
    double maxMs = 0.0;
    double medianMs = 0.0;
    double cv = 0.0;                            // stddev / mean（离散度）
};

inline void FinalizeTimingStats(SearchTimingStats& s)
{
    s.valid = !s.samplesMs.empty();
    if (!s.valid) { return; }
    // 统计字段基于排序副本计算，samplesMs 保持【时间序】（执行顺序）不被就地
    // 排序——SplitTimingStats 的前/后半拆分按时间序切分（时间序对快值样本的
    // 分布诊断有意义：孤立单点 vs 连续时段）。
    std::vector<double> sorted = s.samplesMs;
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    s.minMs = sorted.front();
    s.maxMs = sorted.back();
    double sum = 0.0;
    for (double v : sorted) { sum += v; }
    s.meanMs = sum / static_cast<double>(n);
    s.medianMs = (n % 2 == 0) ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                              : sorted[n / 2];
    if (n > 1 && std::fabs(s.meanMs) > 1e-12) {
        double sqSum = 0.0;
        for (double v : sorted) { sqSum += (v - s.meanMs) * (v - s.meanMs); }
        const double stddev = std::sqrt(sqSum / static_cast<double>(n - 1));
        s.cv = stddev / std::fabs(s.meanMs);
    }
}

// 把交替块样本按前/后半拆分（保持 result 窗口明细字段与日志格式兼容；
// 语义 = interleaved 块的前半/后半，非时间上分离的窗）。
// 按【时间序】前/后半拆分（samplesMs 保持执行顺序，不按数值排序——
// 时间序对快值样本的孤立/连续形态诊断有意义）。
inline void SplitTimingStats(const SearchTimingStats& whole,
    SearchTimingStats& firstHalf, SearchTimingStats& secondHalf)
{
    const size_t n = whole.samplesMs.size();
    const size_t half = n / 2;
    firstHalf.samplesMs.assign(whole.samplesMs.begin(), whole.samplesMs.begin() + half);
    secondHalf.samplesMs.assign(whole.samplesMs.begin() + half, whole.samplesMs.end());
    FinalizeTimingStats(firstHalf);
    FinalizeTimingStats(secondHalf);
}

// -----------------------------------------------------------------------------
// MatmulExecPointers：一次 matmul / Search 执行的全部指针参数。
// -----------------------------------------------------------------------------
struct MatmulExecPointers {
    const void* alpha = nullptr;
    const void* matA = nullptr;
    const void* matB = nullptr;
    const void* beta = nullptr;
    const void* matC = nullptr;
    void* matD = nullptr;
    void* workspace = nullptr;
};

// -----------------------------------------------------------------------------
// MeasurePlanTiming：对指定 plan 计时
// （warmup 次预热 + measure 次 event 计时，每次 sync 后取 elapsed）。
// 返回最后一次 matmul 的状态码；样本与统计写入 out。
// -----------------------------------------------------------------------------
inline aclsparseStatus_t MeasurePlanTiming(aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* plan, const MatmulExecPointers& ptr,
    aclrtStream stream, int32_t warmup, int32_t measure, SearchTimingStats& out)
{
    out.samplesMs.clear();
    aclsparseStatus_t ret = ACL_SPARSE_STATUS_SUCCESS;
    for (int32_t i = 0; i < warmup; ++i) {
        ret = aclsparseLtMatmul(handle, plan, ptr.alpha, ptr.matA, ptr.matB, ptr.beta,
                                ptr.matC, ptr.matD, ptr.workspace, &stream, 1);
        if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    for (int32_t i = 0; i < measure; ++i) {
        EventGuard evStart, evStop;
        evStart.record(stream);
        ret = aclsparseLtMatmul(handle, plan, ptr.alpha, ptr.matA, ptr.matB, ptr.beta,
                                ptr.matC, ptr.matD, ptr.workspace, &stream, 1);
        evStop.record(stream);
        if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        out.samplesMs.push_back(static_cast<double>(EventGuard::elapsedMs(evStart, evStop)));
    }
    FinalizeTimingStats(out);
    return ret;
}

// -----------------------------------------------------------------------------
// SearchDefaultBaseline：默认基线 plan（AlgSelectionInit 默认值 cfg=0/splitK=1/
// mode=0，绑定同一 matmulDesc，非所有权引用）。跨阶段存活（热身窗与
// Search 后的交错窗复用同一 plan）。
// -----------------------------------------------------------------------------
struct SearchDefaultBaseline {
    std::unique_ptr<SparseLtAlgSelectionGuard> algSelDefault;
    std::unique_ptr<SparseLtPlanGuard> planDefault;
    bool built = false;
};

// BuildDefaultBaselinePlan：另建默认 algSel+plan 并查询 ws 需求（不测量）。
// 默认 fused plan 的 ws 需求恒 <= 预置 wsPre（方案 A 单调性），复用同一 ws 缓冲；
// wsRet/wsNeed 出参供调用方写入 result（wsDefaultRet/wsDefaultNeed）做防御性断言
// （与 MatmulSearchNpuResult 解耦，保持本头不依赖 wrapper 结果类型）。
inline bool BuildDefaultBaselinePlan(aclsparseLtConstHandle_t handle,
    SparseLtMatmulDescGuard& matmulDesc, SearchDefaultBaseline& baseline,
    aclsparseStatus_t& wsRet, int64_t& wsNeed)
{
    baseline.algSelDefault = std::make_unique<SparseLtAlgSelectionGuard>(
        handle, matmulDesc.get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    baseline.planDefault = std::make_unique<SparseLtPlanGuard>(
        handle, matmulDesc.get(), baseline.algSelDefault->get());
    size_t defaultWs = 0;
    wsRet = aclsparseLtMatmulGetWorkspace(handle, baseline.planDefault->cptr(), &defaultWs);
    wsNeed = static_cast<int64_t>(defaultWs);
    if (wsRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    baseline.built = true;
    return true;
}

// -----------------------------------------------------------------------------
// ExecOneTimedMatmul：单次 matmul + 前后 event 计时（同步后取 elapsed）。
// -----------------------------------------------------------------------------
inline aclsparseStatus_t ExecOneTimedMatmul(aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* plan, const MatmulExecPointers& execPtr,
    aclrtStream stream, SearchTimingStats& out)
{
    EventGuard evStart, evStop;
    evStart.record(stream);
    aclsparseStatus_t ret = aclsparseLtMatmul(handle, plan, execPtr.alpha, execPtr.matA,
        execPtr.matB, execPtr.beta, execPtr.matC, execPtr.matD, execPtr.workspace,
        &stream, 1);
    evStop.record(stream);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    out.samplesMs.push_back(static_cast<double>(EventGuard::elapsedMs(evStart, evStop)));
    return ACL_SPARSE_STATUS_SUCCESS;
}

// -----------------------------------------------------------------------------
// RunWarmupPairInOrder：交替预热一对（顺序平衡：偶 [def,c*]、奇 [c*,def]）。
// -----------------------------------------------------------------------------
inline aclsparseStatus_t RunWarmupPairInOrder(aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* planDef, aclsparseLtConstMatmulPlan_t* planSel,
    const MatmulExecPointers& execPtr, aclrtStream stream, bool defFirst)
{
    aclsparseLtConstMatmulPlan_t* first = defFirst ? planDef : planSel;
    aclsparseLtConstMatmulPlan_t* second = defFirst ? planSel : planDef;
    aclsparseStatus_t ret = aclsparseLtMatmul(handle, first, execPtr.alpha, execPtr.matA,
        execPtr.matB, execPtr.beta, execPtr.matC, execPtr.matD, execPtr.workspace,
        &stream, 1);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    return aclsparseLtMatmul(handle, second, execPtr.alpha, execPtr.matA,
        execPtr.matB, execPtr.beta, execPtr.matC, execPtr.matD, execPtr.workspace,
        &stream, 1);
}

// -----------------------------------------------------------------------------
// RunTimedPairInOrder：计时一对（对内顺序逐对反转：偶 [def,c*]、奇 [c*,def]）。
// -----------------------------------------------------------------------------
inline aclsparseStatus_t RunTimedPairInOrder(aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* planDef, aclsparseLtConstMatmulPlan_t* planSel,
    const MatmulExecPointers& execPtr, aclrtStream stream, bool defFirst,
    SearchTimingStats& outDef, SearchTimingStats& outSel)
{
    if (defFirst) {
        aclsparseStatus_t ret = ExecOneTimedMatmul(handle, planDef, execPtr, stream, outDef);
        if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
        return ExecOneTimedMatmul(handle, planSel, execPtr, stream, outSel);
    }
    aclsparseStatus_t ret = ExecOneTimedMatmul(handle, planSel, execPtr, stream, outSel);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    return ExecOneTimedMatmul(handle, planDef, execPtr, stream, outDef);
}

// -----------------------------------------------------------------------------
// MeasureInterleavedTiming（逐次交错 + 顺序平衡）：交替执行并计时 def 与
// c* 两个 plan，两侧样本在时间轴上完全混叠。对内顺序逐对反转（偶数对
// [def,c*]、奇数对 [c*,def]，order balancing）——固定配对顺序会使先执行的
// 一侧恒占每次同步后的首执行位而产生系统性位置偏差，反转使两侧对称抵消。
// 统计量为每侧 min。pairs = 每侧样本数（timingMeasure×2）。
// -----------------------------------------------------------------------------
inline aclsparseStatus_t MeasureInterleavedTiming(aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* planSel, aclsparseLtConstMatmulPlan_t* planDef,
    const MatmulExecPointers& execPtr, aclrtStream stream,
    int32_t warmup, int32_t pairs,
    SearchTimingStats& outSel, SearchTimingStats& outDef)
{
    outSel.samplesMs.clear();
    outDef.samplesMs.clear();
    // 交替预热（顺序平衡：偶 [def,c*]、奇 [c*,def]）
    for (int32_t i = 0; i < warmup; ++i) {
        const aclsparseStatus_t ret = RunWarmupPairInOrder(handle, planDef, planSel,
                                                           execPtr, stream, i % 2 == 0);
        if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    // pairs 对交替计时，对内顺序逐对反转（order balancing）
    for (int32_t i = 0; i < pairs; ++i) {
        const aclsparseStatus_t ret = RunTimedPairInOrder(handle, planDef, planSel,
            execPtr, stream, i % 2 == 0, outDef, outSel);
        if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    }
    FinalizeTimingStats(outSel);
    FinalizeTimingStats(outDef);
    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace sparse_test

#endif  // TEST_SPARSELT_MATMULSEARCH_TIMING_H_
