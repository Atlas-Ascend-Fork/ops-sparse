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

/*!
 * \file matmul_search_host.cpp
 * \brief aclsparseLtMatmulSearch host-side API implementation (auto-tuning).
 * Pure host-side control flow, no device kernel of its own: enumerate the
 * algorithm candidate space (algConfigId x splitKMode x splitK) -> rewrite the
 * plan-bound algSelection in place -> recompute tiling/workspace via the same
 * inline helpers aclsparseLtMatmulPlanInit uses -> time the existing public
 * aclsparseLtMatmul on streams[0] with an aclrt event pair (min over
 * searchIterations samples) -> write the winning configuration back into
 * plan/algSelection. The numerical path is 100% carried by the existing
 * aclsparseLtMatmul kernel; candidates only change tiling fields.
 * 候选失败与 matD 终态语义：
 *   - 个别候选执行失败 → 视为不可用跳过（OP_LOGW 留痕），搜索继续；
 *   - 全部候选失败 → EXECUTION_FAILED + 快照恢复；
 *   - 成功路径在回写 c* 后以 c* 无条件再执行一次（不计时），保证 matD
 *     终态为选中配置的完整输出（末尾失败候选可能留下部分写入脏区）。
 */

// Include order: C standard → C++ → CANN → local.
#include <cstdint>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "log/log.h"
#include "cann_ops_sparseLt.h"
#include "aclsparselt_handle_internal.h"
#include "shared/aclsparselt_internal.h"
#include "matmul/arch35/matmul_kernel.h"  // spltMatmulRunInternal

// ============================================================================
// ★ aclsparseLtMatmulSearch — the namesake API of this operator.
// ============================================================================
// Structured error propagation, no goto Validate / Snapshot /
// BuildCandidates / ApplyCandidate / RecomputePlan / TimeCandidate /
// RunSearchLoop / event create-cleanup are separate static helpers so every
// helper keeps each function small and single-purpose.

namespace {

// One candidate configuration: the (algConfigId, splitKMode, splitK) triple
// that gets written into the plan-bound algSelection.
struct SearchCandidate {
    int32_t algConfigId = 0;  ///< cube tiling config id, [0, kAlgConfigMaxId)
    int32_t splitKMode = 0;   ///< 0 = ONE_KERNEL (fused), 1 = TWO_KERNELS
    int32_t splitK = 1;       ///< split-k factor, [1, K]; fused rep normalized to 1
};

// Snapshot of the plan state on entry restored verbatim
// when any aclrt call or the internal Matmul fails, so a failed search leaves
// the plan usable with its original configuration (no side effects).
struct SearchSnapshot {
    int32_t algConfigId = 0;
    int32_t splitKMode = 0;
    int32_t splitK = 1;
    AclsparseltTilingData tilingData;
    size_t workspaceSize = 0;
};

// Candidate upper bound 2 fused reps + 2 x 5 power-of-two
// splitK values + 1 current-config fallback = 13. The AddCandidate cap keeps
// this locked bound invariant even if a future SoC reports a larger cube
// core count (on Ascend 950PR, coreNum = 32 makes the cap unreachable).
constexpr int32_t kSearchMaxCandidates = 13;

// Fixed-size stack candidate table (~156 B, no heap allocation).
struct CandidateList {
    SearchCandidate items[kSearchMaxCandidates];
    int32_t count = 0;
};

// Bundled pass-through arguments for the internal aclsparseLtMatmul timing
// calls (mirrors MatmulLaunchArgs in matmul_host.cpp). Every data pointer and
// scalar is forwarded verbatim to each timed execution the
// matA=nullptr A-sparse workspace fallback, alpha/beta vector scaling and
// in-place (matC == matD) semantics are all carried by the existing Matmul
// path — the search adds zero reinterpretation. dense 侧带 ld 的非紧凑
// 输入由 cube GM 布局直通行距（td.ldA/ldB）承载，候选循环直接复用
// 调用方原始指针。
struct MatmulTimingArgs {
    aclsparseLtConstHandle_t handle;
    const void* alpha;
    const void* matA;
    const void* matB;
    const void* beta;
    const void* matC;
    void* matD;
    void* workspace;
    aclrtStream* streams;
    int32_t numStreams;
    aclrtStream stream;         ///< streams[0], resolved once (nullptr = default stream)
    int32_t searchIterations;   ///< captured once at validate time (default 5)
};

// ValidateSearchPointerAlignment — 16-byte alignment check for all data
// pointers (predicate shared via splt_data_ptrs_aligned16,
// aclsparseLtMatmul 校验同构). matA is only checked when non-null; nullptr
// matA has its own A-sparse fallback condition checked separately.
static aclsparseStatus_t ValidateSearchPointerAlignment(const void* workspace,
                                                           const void* matA, const void* matB,
                                                           const void* matC, const void* matD)
{
    if (!splt_data_ptrs_aligned16(workspace, matA, matB, matC, matD)) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: data pointer not 16-byte aligned");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ValidateSearchMatAFallback — matA == nullptr 仅在 A-sparse 的
// workspace 回退路径合法（回退到 workspace[aPrunedOffset]）。该回退区为
// 单 batch 容量（m×k×elemSize），numBatches > 1 时 kernel 按 batchStrideA
// 推进读取将越界——多 batch 场景须先显式调用 aclsparseLtSpMMAPrune 产出
// A_pruned（batch 布局）再经 matA 传入（单 batch 回退行为不变）。
static aclsparseStatus_t ValidateSearchMatAFallback(
    const aclsparseLtMatmulDescriptor* md)
{
    if (!md_hasStructuredSparsity(md) || !md_isSparseA(md)) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: matA is null (dense x dense / B-sparse requires explicit A)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const int32_t numBatches = (md->matA->numBatches > 0) ? md->matA->numBatches : 1;
    if (numBatches > 1) {
        OP_LOGE(kSparseLtLogTag,
                "MatmulSearch: matA is null with numBatches=%d (batch > 1 requires explicit A_pruned; "
                "workspace aPruned region is single-batch)", numBatches);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ValidateSearchPointers — null / alignment / matA-fallback checks
//. outPlan returns the internal plan struct.
static aclsparseStatus_t ValidateSearchPointers(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulPlan_t* plan,
    const void* workspace,
    const void* matA, const void* matB, const void* matC, const void* matD,
    aclrtStream* streams, int32_t numStreams,
    aclsparseLtMatmulPlan*& outPlan)
{
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (plan == nullptr || *plan == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: plan is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* p = ToInternal(*plan);
    // algSelection is the search write-back target (defensive addition:
    // PlanInit guarantees it non-null).
    if (p->matmulDescr == nullptr || p->tilingData == nullptr || p->algSelection == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: plan internals null (matmulDescr/tilingData/algSelection)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // streams == nullptr means "default stream only" (lenient policy,
    // aligned with the cuSPARSELt official samples that pass
    // streams=nullptr / num_streams=0 directly); numStreams is not judged
    // in that case. When streams != nullptr, numStreams must be > 0.
    // streams[0] itself may be nullptr (ACL runtime default stream); only
    // the array pointer and numStreams are validated (same as
    // aclsparseLtMatmul). numStreams > 1 is not an error: only streams[0]
    // is used.
    if (streams == nullptr) {
        // ACL default stream — ta.stream is resolved at the call site.
    } else if (numStreams <= 0) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: numStreams<=0 with non-null streams");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (workspace == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: workspace is null");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    // B/C/D must be passed explicitly (same as aclsparseLtMatmul).
    if (matB == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: matB is null (must be passed explicitly)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matC == nullptr || matD == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: matC/matD is null (must be passed explicitly)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseStatus_t st = ValidateSearchPointerAlignment(workspace, matA, matB, matC, matD);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // matA == nullptr is only legal on the A-sparse workspace fallback path
    // (single batch; same condition source as resolve_matmul_pointers in
    // matmul_host.cpp — the workspace non-null part is already checked
    // above). Reject it up front so the search cannot fail mid-way.
    if (matA == nullptr) {
        st = ValidateSearchMatAFallback(p->matmulDescr);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }
    outPlan = p;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ValidateSearchParams — full up-front validation. Everything is
// checked once before the search starts so the timing loop has only
// environment failures left as a failure source.
// alpha/beta 的向量缩放 nullptr 前置校验（与 aclsparseLtMatmul 的
// validate_vector_scaling_pointers nullptr 段同构，返回码与文案口径
// 一致）：向量缩放启用（活读编码位）时对应指针必须为有效 device
// float[M] 向量，nullptr 属参数错误 → INVALID_VALUE（公开契约口径；
// 此前该条件仅在候选执行内部校验并经 TimeCandidate 聚合为
// EXECUTION_FAILED，与文档声明不符）。缩放向量的 32B 对齐与 device
// 内存判定仍由候选执行链内的 aclsparseLtMatmul 执行期校验承载。
static aclsparseStatus_t ValidateSearchParams(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulPlan_t* plan,
    const void* alpha, const void* beta,
    const void* workspace,
    const void* matA, const void* matB, const void* matC, const void* matD,
    aclrtStream* streams, int32_t numStreams,
    aclsparseLtMatmulPlan*& outPlan,
    int32_t& outSearchIterations)
{
    aclsparseStatus_t st = ValidateSearchPointers(handle, plan, workspace, matA, matB, matC,
                                                      matD, streams, numStreams, outPlan);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // Defensive dtype gate mirroring the aclsparseLtMatmul launcher check.
    const int32_t dt = dtype_from_acl(outPlan->matmulDescr->matA->valueType);
    if (dt != SPLT_DTYPE_FP32 && dt != SPLT_DTYPE_FP16 &&
        dt != SPLT_DTYPE_BF16 && dt != SPLT_DTYPE_INT8) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: invalid dataType=%d (only FP32/FP16/BF16/INT8)", (int)dt);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // 向量缩放 nullptr 前置校验（见函数头注释）。
    const bool alphaVec = decode_alpha_scaling(outPlan->matmulDescr);
    const bool betaVec = decode_beta_scaling(outPlan->matmulDescr);
    if (alphaVec && alpha == nullptr) {
        OP_LOGE(kSparseLtLogTag,
                "MatmulSearch: alpha vector scaling enabled but alpha is nullptr "
                "(requires a device float[M] vector)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (betaVec && beta == nullptr) {
        OP_LOGE(kSparseLtLogTag,
                "MatmulSearch: beta vector scaling enabled but beta is nullptr "
                "(requires a device float[M] vector)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Defensive: AlgSetAttribute already enforces > 0.
    const int32_t iters = outPlan->algSelection->searchIterations;
    if (iters <= 0) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: searchIterations=%d must be > 0", iters);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    outSearchIterations = iters;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// SavePlanState / RestorePlanState — entry-state snapshot the
// algSelection triple, the tilingData POD copy and workspaceSize. The
// workspace trim baseline is ALWAYS this saved value (no-overflow red line):
// the live plan->workspaceSize is rewritten per candidate during the search
// and must never be used as the comparison base.
static void SavePlanState(const aclsparseLtMatmulPlan* p, SearchSnapshot& snap)
{
    const auto* asel = p->algSelection;
    snap.algConfigId = asel->algConfigId;
    snap.splitKMode = asel->splitKMode;
    snap.splitK = asel->splitK;
    snap.tilingData = *p->tilingData;
    snap.workspaceSize = p->workspaceSize;
}

// RestorePlanState — restore the entry snapshot verbatim (failure path).
static void RestorePlanState(aclsparseLtMatmulPlan* p, const SearchSnapshot& snap)
{
    auto* asel = p->algSelection;
    asel->algConfigId = snap.algConfigId;
    asel->splitKMode = snap.splitKMode;
    asel->splitK = snap.splitK;
    *p->tilingData = snap.tilingData;
    p->workspaceSize = snap.workspaceSize;
}

// AddCandidate — append with the locked 13-entry cap. The cap
// is structurally unreachable on the current platform (coreNum = 32 caps the
// power set at 5 per algConfigId); it keeps the locked bound if a future SoC
// reports more cube cores.
static void AddCandidate(CandidateList& cands, const SearchCandidate& cand)
{
    if (cands.count >= kSearchMaxCandidates) {
        OP_LOGI(kSparseLtLogTag,
                "MatmulSearch: candidate cap %d reached, dropping (algConfigId=%d, splitKMode=%d, splitK=%d)",
                kSearchMaxCandidates, cand.algConfigId, cand.splitKMode, cand.splitK);
        return;
    }
    cands.items[cands.count] = cand;
    cands.count++;
}

// HasCandidate — exact-triple membership test (used by the fallback).
static bool HasCandidate(const CandidateList& cands, int32_t algConfigId,
                         int32_t splitKMode, int32_t splitK)
{
    for (int32_t i = 0; i < cands.count; i++) {
        const SearchCandidate& c = cands.items[i];
        if (c.algConfigId == algConfigId && c.splitKMode == splitKMode && c.splitK == splitK) {
            return true;
        }
    }
    return false;
}

// AppendPowerCandidates — two-stage splitK power-of-two set,
// emitted in execution order: s ascending (outer), algConfigId ascending
// (inner). s = 2 is always the two-stage representative point; larger powers
// extend only while the previous level has not yet saturated the cube cores
// (tiles_g * (s/2) < coreNum, coreNum from get_cube_core_num()). Every s is
// additionally trimmed by the workspace plan-A rule only
// candidates whose compute_ws_layout requirement fits the entry snapshot
// savedWorkspaceSize are evaluated, so no timed execution can overflow the
// user allocation. The requirement is independent of algConfigId, so the
// layout is computed once per s.
static void AppendPowerCandidates(const aclsparseLtMatmulDescriptor* md,
                                  int32_t dt, int32_t numBatches, int32_t coreNum,
                                  size_t savedWorkspaceSize, CandidateList& cands)
{
    const int32_t m = md_m(md);
    const int32_t n = md_n(md);
    const int32_t k = md_k(md);
    const int64_t k64 = static_cast<int64_t>(k);
    if (k64 < 2) {
        return;  // no two-stage candidate exists (normal degeneration)
    }
    // Largest allowed power per algConfigId (core-count cap; s = 2 always fits).
    // {2, 2} 初始化列表与 kAlgConfigMaxId == 2 脆弱耦合：常量扩容而列表
    // 未同步时尾部元素零初始化（maxPow[g]=0），下方 while 循环条件恒真
    // 且 maxPow[g] *= 2 不前进（0×2=0）→ 死循环。static_assert 在编译期
    // 锁定该耦合；扩容 kAlgConfigMaxId 时须同步扩此列表。
    static_assert(kAlgConfigMaxId == 2,
                  "maxPow init list must be extended when kAlgConfigMaxId grows");
    int64_t maxPow[kAlgConfigMaxId] = {2, 2};
    for (int32_t g = 0; g < kAlgConfigMaxId; g++) {
        const CubeTiling ct = resolve_cube_tiling(dt, g);
        const int64_t tiles = static_cast<int64_t>((m + ct.baseM - 1) / ct.baseM) *
                              static_cast<int64_t>((n + ct.baseN - 1) / ct.baseN);
        // Doubling maxPow -> 2*maxPow is allowed when 2*maxPow <= K and
        // tiles * (2*maxPow / 2) < coreNum.
        while (2 * maxPow[g] <= k64 && tiles * maxPow[g] < static_cast<int64_t>(coreNum)) {
            maxPow[g] *= 2;
        }
    }
    for (int64_t s = 2; s <= k64; s *= 2) {
        if (s > maxPow[0] && s > maxPow[1]) {
            break;  // both configs saturated the cores
        }
        const WsLayout wsl = compute_ws_layout(m, n, k, static_cast<int32_t>(s), dt, numBatches,
                                             splt_apruned_span_elems(md, m, k));
        if (static_cast<int64_t>(wsl.totalBytes) > static_cast<int64_t>(savedWorkspaceSize)) {
            continue;  // workspace plan-A trim
        }
        for (int32_t g = 0; g < kAlgConfigMaxId; g++) {
            if (s <= maxPow[g]) {
                AddCandidate(cands, {g, ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS, static_cast<int32_t>(s)});
            }
        }
    }
}

// AppendCurrentConfigFallback — make sure the entry
// configuration's equivalence class is represented so the per-candidate min
// structurally includes the current configuration ("search never regresses"). effectiveSplitK == 1 entry configs (ONE_KERNEL with any
// splitK, or TWO_KERNELS + splitK = 1) are already covered by the fused reps;
// a TWO_KERNELS entry with splitK = s0 outside the generated power set is
// appended explicitly. In the normal flow its workspace requirement equals
// savedWorkspaceSize (PlanInit computed the entry state with it), so the workspace
// trim passes; the trim is re-checked anyway to keep the plan-A no-overflow
// red line airtight even under the unsupported SetAttribute-after-PlanInit
// usage.
static void AppendCurrentConfigFallback(const aclsparseLtMatmulDescriptor* md,
                                        int32_t dt, int32_t numBatches,
                                        const SearchSnapshot& snap, CandidateList& cands)
{
    const bool effectiveOne = (snap.splitKMode == ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL) ||
                              (snap.splitK <= 1);
    const bool gValid = (snap.algConfigId >= 0) && (snap.algConfigId < kAlgConfigMaxId);
    const int32_t k = md_k(md);
    // splitK <= k keeps the closed candidate domain [1, K];
    // AlgSetAttribute guarantees it in the normal flow.
    if (effectiveOne || !gValid || snap.splitK > k) {
        return;
    }
    if (HasCandidate(cands, snap.algConfigId, ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS, snap.splitK)) {
        return;  // already generated by the power set
    }
    const WsLayout wsl = compute_ws_layout(md_m(md), md_n(md), k, snap.splitK, dt, numBatches,
                                            splt_apruned_span_elems(md, md_m(md), k));
    if (static_cast<int64_t>(wsl.totalBytes) > static_cast<int64_t>(snap.workspaceSize)) {
        return;  // workspace plan-A trim (defensive; see function comment)
    }
    AddCandidate(cands, {snap.algConfigId, ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS, snap.splitK});
}

// BuildCandidates — assemble the full candidate table
//   F (fused equivalence-class reps)  ∪  two-stage power set  ∪  current-config
//   fallback, all under the workspace plan-A trim. Execution order :
// fused first (g = 0 -> 1), then two-stage s ascending (outer) / g ascending
// (inner), fallback last. The order does not affect correctness (non
// in-place: every candidate rewrites matD with the same math).
static void BuildCandidates(const aclsparseLtMatmulDescriptor* md,
                            const SearchSnapshot& snap, CandidateList& cands)
{
    cands.count = 0;
    // F: fused equivalence class. ONE_KERNEL with any splitK is
    // kernel-equivalent to TWO_KERNELS + splitK = 1 (td.splitK == 1 decides
    // the dispatch), so each algConfigId is timed exactly once, normalized to
    // (g, ONE_KERNEL, 1). Workspace need is the layout minimum (tempOff,
    // never grows with g) — always <= savedWorkspaceSize, no trim required.
    AddCandidate(cands, {0, ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL, 1});
    AddCandidate(cands, {1, ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL, 1});
    const int32_t dt = dtype_from_acl(md->matA->valueType);
    const int32_t numBatches = (md->matA->numBatches > 0) ? md->matA->numBatches : 1;
    const int32_t coreNum = static_cast<int32_t>(get_cube_core_num());
    AppendPowerCandidates(md, dt, numBatches, coreNum, snap.workspaceSize, cands);
    AppendCurrentConfigFallback(md, dt, numBatches, snap, cands);
}

// ApplyCandidate — write the candidate triple straight into the plan-bound
// algSelection (direct field write + explicit
// recompute, no public AlgSetAttribute round-trip). The candidate value
// domain is structurally closed by the generation rules, so no
// per-candidate range re-check is needed.
static void ApplyCandidate(aclsparseLtMatmulAlgSelection* asel, const SearchCandidate& cand)
{
    asel->algConfigId = cand.algConfigId;
    asel->splitKMode = cand.splitKMode;
    asel->splitK = cand.splitK;
}

// RecomputePlan — recompute plan->tilingData / plan->workspaceSize for the
// candidate currently held in algSelection, exactly the aclsparseLtMatmulPlanInit
// path (same inline helpers, same parameter derivation): after
// the search, the plan state is indistinguishable from a fresh PlanInit run
// with the selected configuration. fill_tiling_dims re-reads the triple
// through the public aclsparseLtMatmulAlgGetAttribute, so it picks up the values
// ApplyCandidate just wrote (self-consistent chain, no bypass).
static void RecomputePlan(aclsparseLtConstHandle_t handle, aclsparseLtMatmulPlan* p)
{
    auto* md = p->matmulDescr;
    const int32_t dt = dtype_from_acl(md->matA->valueType);
    const int32_t m = md_m(md);
    const int32_t n = md_n(md);
    const int32_t k = md_k(md);
    const int32_t numBatches = (md->matA->numBatches > 0) ? md->matA->numBatches : 1;
    const aclsparseLtMatmulAlgSelection* asel = p->algSelection;
    const int32_t effSplitK =
        (asel->splitKMode == ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL) ? 1 : asel->splitK;
    const WsLayout wsl = compute_ws_layout(m, n, k, effSplitK, dt, numBatches,
                                            splt_apruned_span_elems(md, m, k));
    fill_tiling_dims(*p->tilingData, handle, md, asel, wsl, static_cast<int32_t>(get_cube_core_num()));
    p->workspaceSize = static_cast<size_t>(wsl.totalBytes);
}

// TimeCandidate — run the internal aclsparseLtMatmul searchIterations times on
// streams[0] and return the minimum elapsed time in ms. One
// iteration = RecordEvent(start) -> aclsparseLtMatmul (all pointers forwarded
// verbatim) -> RecordEvent(stop) -> SynchronizeEvent(stop) ->
// EventElapsedTime. The envelope includes the tiling H2D push (~230 B) — the
// same per-call cost the user pays on every subsequent Matmul call with the
// selected configuration. No warm-up: the min statistic is naturally robust
// to first-run cold start.
static aclsparseStatus_t TimeCandidate(const MatmulTimingArgs& ta,
                                       aclsparseLtMatmulPlan* p,
                                       aclrtEvent start, aclrtEvent stop,
                                       float& outMinMs)
{
    float tMin = 0.0f;
    for (int32_t i = 0; i < ta.searchIterations; i++) {
        if (aclrtRecordEvent(start, ta.stream) != ACL_SUCCESS) {
            OP_LOGE(kSparseLtLogTag, "MatmulSearch: aclrtRecordEvent(start) failed");
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        // Two-step const conversion C++ forbids the implicit
        // T** -> const T** conversion, so materialize the const plan handle
        // in a local (no reinterpret_cast).
        aclsparseLtConstMatmulPlan_t constPlan = p;
        aclsparseStatus_t st = spltMatmulRunInternal(ta.handle, &constPlan, ta.alpha,
                                                     ta.matA, ta.matB, ta.beta, ta.matC,
                                                     ta.matD, ta.workspace, ta.streams,
                                                     ta.numStreams);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            OP_LOGE(kSparseLtLogTag, "MatmulSearch: internal aclsparseLtMatmul failed, status=%d",
                    static_cast<int32_t>(st));
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        if (aclrtRecordEvent(stop, ta.stream) != ACL_SUCCESS) {
            OP_LOGE(kSparseLtLogTag, "MatmulSearch: aclrtRecordEvent(stop) failed");
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        if (aclrtSynchronizeEvent(stop) != ACL_SUCCESS) {
            OP_LOGE(kSparseLtLogTag, "MatmulSearch: aclrtSynchronizeEvent failed");
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        float ms = 0.0f;
        if (aclrtEventElapsedTime(&ms, start, stop) != ACL_SUCCESS) {
            OP_LOGE(kSparseLtLogTag, "MatmulSearch: aclrtEventElapsedTime failed");
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        if (i == 0 || ms < tMin) {
            tMin = ms;
        }
    }
    outMinMs = tMin;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// RunSearchLoop — evaluate the candidates apply -> recompute ->
// time, tracking the (candidate, min-time) best.
// 候选执行失败语义：
//   ① 个别候选执行失败 → 视为不可用候选，跳过并 OP_LOGW 留痕（搜索空间
//      降级，不中断搜索）；
//   ② 全部候选执行失败 → 返回 EXECUTION_FAILED（调用方恢复快照），
//      不得以空选中/默认配置静默成功。
static aclsparseStatus_t RunSearchLoop(const MatmulTimingArgs& ta,
                                       aclsparseLtMatmulPlan* p,
                                       const CandidateList& cands,
                                       aclrtEvent start, aclrtEvent stop,
                                       SearchCandidate& outBest, float& outBestTime)
{
    SearchCandidate best = cands.items[0];
    float bestTime = 0.0f;
    bool anySuccess = false;
    for (int32_t i = 0; i < cands.count; i++) {
        const SearchCandidate& cand = cands.items[i];
        ApplyCandidate(p->algSelection, cand);
        RecomputePlan(ta.handle, p);
        float tMin = 0.0f;
        aclsparseStatus_t st = TimeCandidate(ta, p, start, stop, tMin);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            // ① candidate unavailable: skip and leave a trace (not silent).
            OP_LOGW(kSparseLtLogTag,
                    "MatmulSearch: candidate (algConfigId=%d, splitKMode=%d, splitK=%d) failed, "
                    "treating as unavailable and skipping",
                    cand.algConfigId, cand.splitKMode, cand.splitK);
            continue;
        }
        OP_LOGD(kSparseLtLogTag,
                "MatmulSearch: candidate (algConfigId=%d, splitKMode=%d, splitK=%d) minTime=%.3f ms",
                cand.algConfigId, cand.splitKMode, cand.splitK, tMin);
        if (!anySuccess || tMin < bestTime) {
            bestTime = tMin;
            best = cand;
        }
        anySuccess = true;
    }
    if (!anySuccess) {
        // ② every candidate failed — the search has nothing to write back.
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: all %d candidate(s) failed", cands.count);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    outBest = best;
    outBestTime = bestTime;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// CreateSearchEvents — create the reusable start/stop event pair (created
// once for the whole search, not per
// candidate or per iteration; after SynchronizeEvent an event can be
// recorded again). Partial creation failure cleans up the already-created
// event (nothing has been modified yet, no snapshot restore).
static aclsparseStatus_t CreateSearchEvents(aclrtEvent& start, aclrtEvent& stop)
{
    start = nullptr;
    stop = nullptr;
    if (aclrtCreateEvent(&start) != ACL_SUCCESS) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: aclrtCreateEvent(start) failed");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    if (aclrtCreateEvent(&stop) != ACL_SUCCESS) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: aclrtCreateEvent(stop) failed");
        (void)aclrtDestroyEvent(start);
        start = nullptr;
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// DestroySearchEvents — destroy both events (null-guarded). Callers must have
// drained the stream first (DestroyEvent precondition).
static void DestroySearchEvents(aclrtEvent start, aclrtEvent stop)
{
    if (start != nullptr) {
        (void)aclrtDestroyEvent(start);
    }
    if (stop != nullptr) {
        (void)aclrtDestroyEvent(stop);
    }
}

// CleanupSearchEvents — failure-path cleanup: drain the stream (unchecked —
// the call is already failing; the sync also drains anything a failed
// internal Matmul may have enqueued so the DestroyEvent "tasks completed"
// precondition holds), then destroy both events.
static void CleanupSearchEvents(aclrtStream stream, aclrtEvent start, aclrtEvent stop)
{
    (void)aclrtSynchronizeStream(stream);
    DestroySearchEvents(start, stop);
}

// ExecuteSelectedOnce — one untimed execution of the internal
// aclsparseLtMatmul with the currently applied (selected) configuration.
// All pointers forwarded verbatim (same two-step const conversion as
// TimeCandidate).
static aclsparseStatus_t ExecuteSelectedOnce(const MatmulTimingArgs& ta, aclsparseLtMatmulPlan* p)
{
    aclsparseLtConstMatmulPlan_t constPlan = p;
    return spltMatmulRunInternal(ta.handle, &constPlan, ta.alpha, ta.matA, ta.matB, ta.beta,
                                 ta.matC, ta.matD, ta.workspace, ta.streams, ta.numStreams);
}

// FinishSearchSuccess — success-path tail (matD
// terminal-state contract):
//   ① idempotent write-back of the winning candidate (the plan is recomputed
//      even if the last timed candidate happened to be the best one);
//   ② one untimed final execution with c* so matD is left as the
//      selected configuration's COMPLETE output — a trailing failed candidate
//      may otherwise leave partially-written dirty regions in matD
//      (non in-place is a full overwrite, in-place accumulates one
//      more time which does not change the declared "value unpredictable"
//      semantics);
//   ③ blocking sync (checked — locks the "blocking w.r.t. streams[0]"
//      contract and surfaces async kernel errors of the final execution),
//      then destroy the events.
// On failure: restore the entry snapshot, clean up, return EXECUTION_FAILED.
static aclsparseStatus_t FinishSearchSuccess(const MatmulTimingArgs& ta,
                                             aclsparseLtMatmulPlan* p,
                                             const SearchSnapshot& snap,
                                             const SearchCandidate& best, float bestTime,
                                             aclrtEvent start, aclrtEvent stop)
{
    ApplyCandidate(p->algSelection, best);
    RecomputePlan(ta.handle, p);
    aclsparseStatus_t st = ExecuteSelectedOnce(ta, p);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kSparseLtLogTag,
                "MatmulSearch: final execution with the selected config failed, status=%d",
                static_cast<int32_t>(st));
        RestorePlanState(p, snap);
        CleanupSearchEvents(ta.stream, start, stop);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    OP_LOGI(kSparseLtLogTag,
            "MatmulSearch: done, best algConfigId=%d, splitKMode=%d, splitK=%d, minTime=%.3f ms",
            best.algConfigId, best.splitKMode, best.splitK, bestTime);
    if (aclrtSynchronizeStream(ta.stream) != ACL_SUCCESS) {
        OP_LOGE(kSparseLtLogTag, "MatmulSearch: final aclrtSynchronizeStream failed");
        RestorePlanState(p, snap);
        CleanupSearchEvents(ta.stream, start, stop);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    DestroySearchEvents(start, stop);
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace
// ============================================================================
// ★ aclsparseLtMatmulSearch — evaluate the algorithm candidate space and
// write the fastest configuration back into plan/algSelection. Blocking
// w.r.t. streams[0] .
// Validate / Snapshot / BuildCandidates / RunSearchLoop / FinishSearchSuccess
// 等既有编排链保持不变（日志时序逐条一致）。dense 侧带 ld 的非紧凑输入
// 由 cube GM 布局直通行距（td.ldA/td.ldB）承载，候选循环直接使用调用方
// 原始指针，无需入口预处理。
// ============================================================================
extern "C" aclsparseStatus_t aclsparseLtMatmulSearch(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulPlan_t* plan,
    const void* alpha,
    const void* matA, const void* matB,
    const void* beta,
    const void* matC, void* matD,
    void* workspace,
    aclrtStream* streams, int32_t numStreams)
{
    aclsparseLtMatmulPlan* p = nullptr;
    int32_t searchIterations = 0;
    aclsparseStatus_t st = ValidateSearchParams(handle, plan, alpha, beta, workspace, matA, matB,
                                                matC, matD, streams, numStreams, p, searchIterations);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // stream0 guard: streams may be nullptr (default-stream-only policy,
    // see ValidateSearchPointers) — dereferencing streams[0] unconditionally
    // would crash on that path. ta.streams is passed through to
    // spltMatmulRunInternal, which re-validates it with the same lenient rule.
    aclrtStream stream0 = (streams != nullptr) ? streams[0] : nullptr;
    MatmulTimingArgs ta = {handle, alpha, matA, matB, beta, matC, matD, workspace,
                           streams, numStreams, stream0, searchIterations};

    SearchSnapshot snap;
    SavePlanState(p, snap);

    CandidateList cands;
    BuildCandidates(p->matmulDescr, snap, cands);
    OP_LOGI(kSparseLtLogTag,
            "MatmulSearch: start, candidates=%d, searchIterations=%d, m=%d, n=%d, k=%d, dataType=%d",
            cands.count, searchIterations, md_m(p->matmulDescr), md_n(p->matmulDescr),
            md_k(p->matmulDescr), dtype_from_acl(p->matmulDescr->matA->valueType));

    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    // CreateSearchEvents 返回 aclsparseStatus_t——与 ACL runtime 的
    // ACL_SUCCESS（aclError）是两套枚举，不可混比（当前两成功码数值
    // 恰均为 0，混比无害但语义错误；若取值分歧将静默误判）。
    if (CreateSearchEvents(start, stop) != ACL_SPARSE_STATUS_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    SearchCandidate best;
    float bestTime = 0.0f;
    st = RunSearchLoop(ta, p, cands, start, stop, best, bestTime);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        // All candidates failed: restore the entry snapshot (failed
        // call leaves no plan/algSelection side effects), clean up, propagate.
        RestorePlanState(p, snap);
        CleanupSearchEvents(ta.stream, start, stop);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    // Success path: write-back of the selected candidate + final c* execution (matD
    // terminal-state contract) + checked blocking sync + event destroy.
    return FinishSearchSuccess(ta, p, snap, best, bestTime, start, stop);
}
