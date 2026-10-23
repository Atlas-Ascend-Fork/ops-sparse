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

#ifndef TEST_SPARSELT_MATMULSEARCH_NPU_WRAPPER_INT8CD_H_
#define TEST_SPARSELT_MATMULSEARCH_NPU_WRAPPER_INT8CD_H_

// =============================================================================
// RunSearchChainInt8CD：H 组超集组合链——INT8 A/B × C=D=CDT
// （CDT ∈ {int32_t, uint16_t, bf16_bits_t}，dtype 矩阵超集组合）。
//
// 链结构与主链（RunMatmulSearchChain / RunMatmulSearchBSparseChain）对齐：
//   1. 描述符 + buffers：ltmatmul 共享铺设段 PrepareInt8CDChainContext
//      （A/B 物理 (order, ld) 布局 + batch-strided 铺设；A/B/C/D 四矩阵
//      NUM_BATCHES/BATCH_STRIDE 属性先于 MatmulDescriptorInit——一致性
//      校验要求四矩阵相同；C/D 独立 CDT dtype 接入 (cdOrder, ldCd)；
//      A/B/C/D 设备缓冲按 batch 总量放大）。
//   2. vec/epi 属性 + algSel 预置 + plan + wsPre（ltmatmul 共享 helper
//      SetInt8CDVecEpiAttrs / SetupSearchAlgPlanWs）。
//   3. 剪枝（PruneInt8CDForSearch）：A-sparse 产出 A_pruned（batch 布局，
//      mat_a_null 时写入 ws+aPrunedOffset 白盒偏移目标）；B-sparse 剪枝
//      dB（物理 (n,k)）→ dPruned（(k,n) 行主序输出）。prune 输入/输出
//      缓冲按 batch 总量放大（描述符携带 batch 属性，单调用处理全 batch）。
//   4. 指针组装 + def-热身窗 + Search + 后置（复用主链
//      SearchCallAndPostPass；回读/搜索后再执行按 batch 总量）。
//
// 输入 hA/hB/hC 为 batch-strided 布局（每 batch 前 matSize 个有效元素 +
// stride padding），由调用方生成；输出经 ExtractSearchOutputPacked 解包
// packed 后比对。hC 为 CDT 域数据。
// 依赖说明：本头由 matmulSearch_npu_wrapper.h 末尾 include（其前置定义
// SetupSearchAlgPlanWs / RunPruneToTarget / BuildSearchExecPointers /
// SearchCallAndPostPass / RunDefaultBaselineWarmup 等）；不单独 include。
// =============================================================================

namespace sparse_test {

// -----------------------------------------------------------------------------
// PruneInt8CDForSearch：Int8CD 链剪枝段。
//   - A-sparse：A_pruned——独立 dPruned 缓冲（batch 总量）或
//     ws+aPrunedOffset 白盒偏移目标（mat_a_null 回退路径）。
//   - B-sparse：剪枝 dB（物理 (n,k)）→ dPruned（(k,n) 行主序输出；
//     prune 输出恒紧凑 (k,n)，缓冲上界 = max(bPhysSpan, kn)）。
// 描述符携带 numBatches/batchStride，单调用处理全 batch（传 batch0
// 基址；host 侧切片循环会使末切片基址被算子按 numBatches 二次推进而
// 越界，禁止切片）；输出与输入共用 batchStride 推进，回读
// totalElems（BatchTotalElems 语义）为 batch-strided 全量。
// -----------------------------------------------------------------------------
template <typename CDT>
inline bool PruneInt8CDForSearch(const MatmulSearchRunArgs<int8_t, CDT, CDT>& args,
    SparseLtHandleGuard& handle, SparseLtMatmulDescGuard& matmulDesc,
    DeviceBuffer& dA, DeviceBuffer& dB, DeviceBuffer& dPruned, void* wsPtr,
    size_t mk, size_t kn, size_t bPhysSpan, aclrtStream stream,
    MatmulSearchNpuResult<CDT>& result, void*& matAPtr, void*& matBPtr)
{
    constexpr size_t kInElt = 1;  // int8
    matBPtr = dB.get();
    if (args.isSparseA) {
        // A-sparse 剪枝复用主链 PruneASparseForSearch（dtypeCode 经
        // NpuDtypeTrait<int8_t>::kAclDtype = ACL_INT8 等价推导；白盒
        // aPrunedOffset / 独立 dPruned 缓冲与 batch 全量语义同源）。
        return PruneASparseForSearch(args, handle, matmulDesc, dA, dPruned,
            wsPtr, mk, kInElt, stream, result, matAPtr);
    }
    // B-sparse：剪枝 dB（物理 (n,k)）→ dPruned（(k,n) 行主序输出）
    const size_t pruneSpan = (bPhysSpan > kn) ? bPhysSpan : kn;
    const size_t totalElems = BatchTotalElems(pruneSpan, args.numBatches, args.batchStride);
    dPruned = DeviceBuffer::alloc(totalElems * kInElt);
    if (!RunPruneToTarget(result, handle.get(), matmulDesc.get(), dB,
            static_cast<uint8_t*>(dPruned.get()), totalElems, kInElt, args.pruneAlg, stream)) {
        return false;
    }
    matBPtr = dPruned.get();
    matAPtr = dA.get();
    return true;
}

// -----------------------------------------------------------------------------
// SetupInt8CDAttrsPlanWs：Int8CD 链「vec/epi 属性（ltmatmul 共享 helper）+
// algSel 预置 + plan + wsPre」段；输出经 SearchChainSetupResult 归拢
// （主链 SetupSearchChainAttrsPlan 同范式）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT, typename CType = InT>
inline bool SetupInt8CDAttrsPlanWs(SparseLtHandleGuard& handle,
    SparseLtMatmulDescGuard& matmulDesc, const MatmulSearchRunArgs<InT, OutT, CType>& args,
    MatmulSearchNpuResult<OutT>& result, SearchChainSetupResult& setup)
{
    if (!SetInt8CDVecEpiAttrs(handle, matmulDesc, args.m,
            args.alpha_vector_scaling, args.beta_vector_scaling, args.alphaVec, args.betaVec,
            args.biasEnabled, args.biasStride, args.activationType,
            args.reluUpperBound, args.reluThreshold, args.geluScaling,
            args.numBatches, args.batchStride, args.biasVec,
            setup.vecBufs, setup.epiBufs, result)) {
        return false;
    }
    return SetupSearchAlgPlanWs(args, handle, matmulDesc, result,
                                setup.algSel, setup.plan, setup.dWorkspace, setup.wsPtr);
}

// -----------------------------------------------------------------------------
// FinishInt8CDExecAndSearch：Int8CD 链「指针组装（in-place / nullptr 语义 /
// vec scaling）+ def-热身窗（共享段）+ Search + 后置（复用主链
// SearchCallAndPostPass）」收尾段。
// totalOutElems = C/D 单 batch 物理 span 的 batch 总量（回读与搜索后再
// 执行尺寸）；hCPhys = pack 后的 C host 数据（rerun 新鲜 C 上传用）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT, typename CType = InT>
inline void FinishInt8CDExecAndSearch(const MatmulSearchRunArgs<InT, OutT, CType>& args,
    SparseLtHandleGuard& handle, SparseLtMatmulDescGuard& matmulDesc,
    std::unique_ptr<SparseLtAlgSelectionGuard>& algSel,
    std::unique_ptr<SparseLtPlanGuard>& plan,
    SearchChainSetupResult& setup,
    void* matAPtr, void* matBPtr, DeviceBuffer& dC, DeviceBuffer& dD,
    aclrtStream stream, size_t totalOutElems, size_t kOutElt,
    const std::vector<CType>& hCPhys, MatmulSearchNpuResult<OutT>& result)
{
    float alphaF = args.alpha;
    float betaF = args.beta;
    void* matCPtr = args.inplace ? dD.get() : dC.get();
    void* matDPtr = dD.get();
    MatmulExecPointers execPtr = BuildSearchExecPointers(args, setup.vecBufs, alphaF, betaF,
        matAPtr, matBPtr, matCPtr, matDPtr, setup.wsPtr);
    SearchDefaultBaseline baseline;
    if (args.checkTiming &&
        !RunDefaultBaselineWarmup(args, handle, matmulDesc, execPtr, stream,
                                  baseline, result)) { return; }
    const size_t cBytes = totalOutElems * kOutElt;
    SearchCallAndPostPass<InT, OutT, CType>(args, handle, *algSel, *plan, execPtr,
        dD, stream, baseline, hCPhys, cBytes, totalOutElems, kOutElt, totalOutElems, result);
}

// -----------------------------------------------------------------------------
// RunSearchChainInt8CD：H 组超集组合链主体（步骤编排；铺设段为 ltmatmul
// 共享 PrepareInt8CDChainContext，属性/计划段与执行/后置段分别见
// SetupInt8CDAttrsPlanWs / FinishInt8CDExecAndSearch，剪枝段见
// PruneInt8CDForSearch）。
// -----------------------------------------------------------------------------
template <typename CDT>
MatmulSearchNpuResult<CDT> RunSearchChainInt8CD(
    const MatmulSearchRunArgs<int8_t, CDT, CDT>& args,
    const std::vector<int8_t>& hA, const std::vector<int8_t>& hB,
    const std::vector<CDT>& hC)
{
    using namespace sparse_test;
    constexpr size_t kOutElt = NpuDtypeTrait<CDT>::kEltSize;
    const int32_t m = args.m, k = args.k, n = args.n;
    const size_t mn = static_cast<size_t>(m) * n;
    const size_t mk = static_cast<size_t>(m) * k;
    MatmulSearchNpuResult<CDT> result{};

    SearchStreamGuard streamGuard;
    aclrtStream stream = streamGuard.get();

    // ---- 描述符 + buffers（A/B 物理布局 + batch 属性 + batch 总量缓冲；
    //      A/B/C/D 的 NUM_BATCHES/BATCH_STRIDE 先于 MatmulDescriptorInit）----
    Int8CDChainCtx<CDT> ctx;
    std::vector<CDT> hCPhys;
    PrepareInt8CDChainContext<CDT>(ctx, m, k, n, hA, hB, hC,
        args.opA, args.opB, args.isSparseA, args.order,
        args.numBatches, args.batchStride, args.layout, result, &hCPhys);

    // ---- vec/epi 属性 + algSel 预置 + plan + wsPre（epiBufs.dBias 由 attr
    //      持有 device 指针，经 setup 存活至链尾 exec 完成）----
    SearchChainSetupResult setup;
    if (!SetupInt8CDAttrsPlanWs(ctx.handle, *ctx.matmulDesc, args, result, setup)) { return result; }

    // ---- 剪枝（A-sparse：A_pruned；B-sparse：B_pruned (k,n) 行主序）----
    DeviceBuffer dPruned;
    void* matAPtr = nullptr;
    void* matBPtr = nullptr;
    if (!PruneInt8CDForSearch<CDT>(args, ctx.handle, *ctx.matmulDesc, ctx.dA, ctx.dB,
            dPruned, setup.wsPtr, mk, static_cast<size_t>(k) * n, ctx.lay.bPhysSize,
            stream, result, matAPtr, matBPtr)) {
        return result;
    }

    // ---- 指针组装 + def-热身窗 + Search + 后置（复用主链 SearchCallAndPostPass）----
    // 回读尺寸 = C/D 物理 span 的 batch 总量（batch 布局含 stride padding）
    FinishInt8CDExecAndSearch(args, ctx.handle, *ctx.matmulDesc, setup.algSel, setup.plan,
        setup, matAPtr, matBPtr, ctx.dC, ctx.dD, stream, ctx.totalCdElems, kOutElt,
        hCPhys, result);
    return result;
}

}  // namespace sparse_test

#endif  // TEST_SPARSELT_MATMULSEARCH_NPU_WRAPPER_INT8CD_H_
