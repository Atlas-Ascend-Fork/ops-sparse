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

#ifndef TEST_SPARSELT_MATMULSEARCH_GOLDEN_H_
#define TEST_SPARSELT_MATMULSEARCH_GOLDEN_H_

// =============================================================================
// 搜索行为 golden（aclsparseLtMatmulSearch 特有输出）—— 合法候选集 / plan 一致性 / 计时有效性。
//
// 本头文件只定义【纯函数谓词】（不依赖 gtest），由调用方（ST 断言编排 /
// 验收侧性能采集）将返回的结构转成硬断言并落盘「输出张量 / 判定指标 /
// 实测值 / 阈值 / 结论」记录。
//
// 数值 golden（matD）不在本文件：直接复用 test/ltmatmul/ltmatmul_golden.h
// （含 ltmatmul_golden_types.h / ltmatmul_golden_int8.h），经 wrapper /
// test.cpp 调用——搜索不改变数值计算路径。
//
// 白盒依赖：sparseLt/shared/aclsparselt_internal.h 的 compute_ws_layout /
// dtype_from_acl / SPLT_DTYPE_*（与 matmulPlanInit 测试 VerifyPlanFields 同范式）。
// =============================================================================

#include <cstdint>
#include <string>

#include "aclsparselt_internal.h"  // compute_ws_layout, dtype_from_acl, SPLT_DTYPE_*
#include "cann_ops_sparseLt.h"

namespace sparse_test {

// -----------------------------------------------------------------------------
// Split-K 语义（与库内 fill_splitk_tiling_dims 同构）：
//   ONE_KERNEL(0) 吸收 splitK（fused 单核 GEMM+reduction，effectiveSplitK=1）；
//   TWO_KERNELS(1) 保留 splitK（两段式 matmul+epilogue）。
// -----------------------------------------------------------------------------
inline constexpr int32_t kSplitKModeOneKernel = 0;
inline constexpr int32_t kSplitKModeTwoKernels = 1;

inline int32_t EffectiveSplitKOf(int32_t splitKMode, int32_t splitK)
{
    return (splitKMode == kSplitKModeOneKernel) ? 1 : splitK;
}

inline bool IsPowerOfTwo(int32_t v)
{
    return v > 0 && (v & (v - 1)) == 0;
}

// 输入 dtype → compute_ws_layout 的 dataType 编码（SPLT_DTYPE_*）。
// INT8 输入在 ws 布局中按输入 dtype 编码（elemSize=1）。
inline int32_t SpltDtypeCodeOf(aclDataType valueType)
{
    return dtype_from_acl(valueType);
}

// -----------------------------------------------------------------------------
// 白盒 workspace 需求重算（Tier-2 wsReq / plan 一致性）。
// wsReq(s) = compute_ws_layout(m, n, k, s, dataType, numBatches).totalBytes
// fused 类（s==1）totalBytes=tempOff，与库内跳过 temp 分配逻辑一致。
// -----------------------------------------------------------------------------
inline int64_t WhiteboxWsBytes(int32_t m, int32_t n, int32_t k, int32_t effSplitK,
                               int32_t dataTypeCode, int32_t numBatches,
                               int64_t aPrunedSpanElems)
{
    const WsLayout w = compute_ws_layout(m, n, k, effSplitK, dataTypeCode, numBatches,
                                         aPrunedSpanElems);
    return w.totalBytes;
}
// aPruned 区偏移（B3：matA=nullptr 回退 ws+aPrunedOffset，剪枝目标偏移白盒求取）。
inline int64_t WhiteboxAPrunedOffset(int32_t m, int32_t n, int32_t k, int32_t effSplitK,
                                     int32_t dataTypeCode, int32_t numBatches,
                                     int64_t aPrunedSpanElems)
{
    const WsLayout w = compute_ws_layout(m, n, k, effSplitK, dataTypeCode, numBatches,
                                         aPrunedSpanElems);
    return w.aPrunedOff;
}

// -----------------------------------------------------------------------------
// 合法候选集断言结果（Tier-1 + Tier-2 + 方案 A 裁剪推论）。
// ok = tier1Ok && tier2Ok（推论断言 defaultPresetFusedOnly 与调用方编排）。
// -----------------------------------------------------------------------------
struct CandidateVerdict {
    // 搜索后经 AlgGetAttribute 回读的选中三元组（输入）
    int32_t selCfg = -1;
    int32_t selSplitK = -1;
    int32_t selSplitKMode = -1;
    // 派生
    int32_t effSplitK = 0;
    // Tier-1（所有用例必断言）
    bool tier1Ok = false;
    std::string tier1Detail;
    // Tier-2（正常用例必断言）
    bool tier2Ok = false;
    std::string tier2Detail;
    // workspace 方案 A 裁剪（wsReq(effSplitK) <= wsLimit）
    bool wsFitOk = false;
    int64_t wsReq = -1;
    int64_t wsLimit = -1;
    // 推论断言（默认预置 fused-only 窗口 → 选中必为 fused 类）
    bool fusedForcedOk = true;   // 非默认预置时恒 true（不启用）
    std::string fusedForcedDetail;

    bool AllOk() const { return tier1Ok && tier2Ok && wsFitOk && fusedForcedOk; }
    std::string Brief() const;
};

// -----------------------------------------------------------------------------
// AssertCandidateLegal：两级谓词判定。
//
// 输入：
//   selCfg/selSplitK/selSplitKMode —— 搜索后 AlgGetAttribute 回读的选中三元组
//   K（= md_k(matmulDescr)）        —— splitK 上界（Tier-1: s ∈ [1, K]）
//   maxCfgId —— AlgGetAttribute(ALG_CONFIG_MAX_ID) 查询值（Tier-1: cfg ∈ [0, maxCfgId)；
//               取动态查询值而非硬编码）
//   wsLimit —— 搜索前经 aclsparseLtMatmulGetWorkspace 查询的 plan->workspaceSize
//   m/n/k、dataTypeCode、numBatches —— wsReq 白盒重算入参
//   defaultPresetFusedOnly —— 预置为默认配置（cfg=0,splitK=1,mode=0）时为 true：
//               wsLimit 不含 temp 区 → 全部两段式候选被裁剪 → 选中必为 fused 类
//               （断言 effSplitK==1，覆盖 workspace 不足候选裁剪的可观测行为）。
// -----------------------------------------------------------------------------
inline CandidateVerdict AssertCandidateLegal(int32_t selCfg, int32_t selSplitK,
    int32_t selSplitKMode, int32_t K, int32_t maxCfgId,
    int64_t wsLimit, int32_t m, int32_t n, int32_t k,
    int32_t dataTypeCode, int32_t numBatches, bool defaultPresetFusedOnly,
    int64_t aPrunedSpanElems)
{
    CandidateVerdict v;
    v.selCfg = selCfg;
    v.selSplitK = selSplitK;
    v.selSplitKMode = selSplitKMode;
    v.effSplitK = EffectiveSplitKOf(selSplitKMode, selSplitK);
    v.wsLimit = wsLimit;

    // ---- Tier-1（所有用例必断言）----
    bool cfgOk = (selCfg >= 0) && (selCfg < maxCfgId);
    bool splitOk = (selSplitK >= 1) && (selSplitK <= K);
    bool modeOk = (selSplitKMode == kSplitKModeOneKernel) || (selSplitKMode == kSplitKModeTwoKernels);
    v.tier1Ok = cfgOk && splitOk && modeOk;
    v.tier1Detail = "cfg=" + std::to_string(selCfg) + " in [0," + std::to_string(maxCfgId) +
        "), s=" + std::to_string(selSplitK) + " in [1," + std::to_string(K) +
        "), mode=" + std::to_string(selSplitKMode) + " in {0,1}";

    // ---- Tier-2（正常用例必断言；与开发方案中幂次集合 S 的具体取值解耦）----
    if (v.effSplitK == 1) {
        // fused 类（含 ONE_KERNEL 任意 splitK 与 TWO_KERNELS+splitK=1 的等价类去重）：
        // effectiveSplitK==1 即可；mode==ONE_KERNEL 时 s 仅受 Tier-1 约束（splitK 被 absorb）。
        v.tier2Ok = true;
        v.tier2Detail = "fused class (effSplitK==1)";
    } else {
        // 两段式类：s 为 2 的幂（有界幂次候选集约束）且 2 <= s <= K
        const bool pow2 = IsPowerOfTwo(selSplitK);
        const bool rangeOk = (selSplitK >= 2) && (selSplitK <= K);
        v.tier2Ok = pow2 && rangeOk;
        v.tier2Detail = std::string("two-kernel class: s=") + std::to_string(selSplitK) +
            (pow2 ? " is power of 2" : " NOT power of 2") +
            (rangeOk ? ", in [2,K]" : ", out of [2,K]");
    }

    // ---- workspace 方案 A 裁剪：wsReq(effSplitK) <= wsLimit ----
    v.wsReq = WhiteboxWsBytes(m, n, k, v.effSplitK, dataTypeCode, numBatches,
                              aPrunedSpanElems);
    v.wsFitOk = (v.wsReq <= wsLimit);

    // ---- 推论断言（默认预置 fused-only 窗口）----
    if (defaultPresetFusedOnly) {
        v.fusedForcedOk = (v.effSplitK == 1);
        v.fusedForcedDetail = "default preset (fused-only candidate window): effSplitK==" +
            std::to_string(v.effSplitK) + " (expect 1)";
    } else {
        v.fusedForcedOk = true;  // 非默认预置不启用该推论
        v.fusedForcedDetail = "n/a (two-kernel preset)";
    }
    return v;
}

inline std::string CandidateVerdict::Brief() const
{
    return "Tier1[" + tier1Detail + "]=" + (tier1Ok ? "OK" : "FAIL") +
        " Tier2[" + tier2Detail + "]=" + (tier2Ok ? "OK" : "FAIL") +
        " wsFit(wsReq=" + std::to_string(wsReq) + " <= wsLimit=" + std::to_string(wsLimit) + ")=" +
        (wsFitOk ? "OK" : "FAIL") +
        " fusedForced=" + (fusedForcedOk ? "OK" : "FAIL") +
        " => effSplitK=" + std::to_string(effSplitK);
}

// -----------------------------------------------------------------------------
// plan 一致性断言：
//   wsPost == compute_ws_layout(m,n,k,effSplitK,dtype,batches).totalBytes（白盒重算）
//   wsPost <= wsPre（方案 A 单调性）
// -----------------------------------------------------------------------------
struct PlanConsistencyVerdict {
    int64_t wsPre = -1;
    int64_t wsPost = -1;
    int64_t wsExpected = -1;
    bool equalOk = false;   // wsPost == 白盒重算
    bool monotonicOk = false;  // wsPost <= wsPre
    bool AllOk() const { return equalOk && monotonicOk; }
    std::string Brief() const
    {
        return "wsPost=" + std::to_string(wsPost) + " expected=" + std::to_string(wsExpected) +
            " (equal=" + (equalOk ? "OK" : "FAIL") + ") wsPre=" + std::to_string(wsPre) +
            " (monotonic=" + (monotonicOk ? "OK" : "FAIL") + ")";
    }
};

inline PlanConsistencyVerdict AssertPlanConsistency(int64_t wsPre, int64_t wsPost,
    int32_t m, int32_t n, int32_t k, int32_t effSplitK,
    int32_t dataTypeCode, int32_t numBatches, int64_t aPrunedSpanElems)
{
    PlanConsistencyVerdict v;
    v.wsPre = wsPre;
    v.wsPost = wsPost;
    v.wsExpected = WhiteboxWsBytes(m, n, k, effSplitK, dataTypeCode, numBatches,
                                   aPrunedSpanElems);
    v.equalOk = (wsPost == v.wsExpected);
    v.monotonicOk = (wsPost <= wsPre);
    return v;
}

}  // namespace sparse_test

#endif  // TEST_SPARSELT_MATMULSEARCH_GOLDEN_H_
