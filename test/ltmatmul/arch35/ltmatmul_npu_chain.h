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

#ifndef TEST_LTMATMUL_NPU_CHAIN_H_
#define TEST_LTMATMUL_NPU_CHAIN_H_

// =============================================================================
// Matmul chain implementation（A-sparse / dense*dense 链 runner）。
//
// Provides the internal chain runners consumed by RunMatmulNpu:
//   - MatmulChainCtx: aggregated descriptor + buffer context (A-sparse / dense)
//   - PrepareMatmulChainBuffers / CreateMatmulABDescriptors /
//     PrepareMatmulChainContext: A-sparse / dense*dense context setup
//   - PreparePlanAndWorkspace: algSel + plan + workspace
//   - CopyPrunedToHost / RunPruneForASparse: prune readback / A-sparse prune
//   - MatmulRunArgs / ChainSetupResult / SetupVecEpiPlanWorkspace: shared
//     post-context sequence (vec scaling + epilogue + plan + workspace)
//   - RunMatmulChain: A-sparse / dense*dense chain runner
//
// B-sparse 链族 + 物理布局计算段（MatmulBSparseCtx /
// Compute{A,B}SparsePhysicalLayout / PrepareBSparseChainContext /
// RunBSparsePruneAndTranspose / RunMatmulBSparseChainWithTrans 等）位于
// ltmatmul_npu_chain_bsparse.h，由本文件尾部 include 转发。
//
// Include order: this header is included by ltmatmul_npu_wrapper.h AFTER
// MatmulNpuResult, the epilogue helpers (ltmatmul_npu_epilogue.h),
// SetMatmulAlgAttributes, VecScalingBuffers/SetVecScalingAttrs,
// SetBatchAttrsOnMatDesc/SetBatchAttrsOnAllMatDescs, BatchTotalElems, and
// ExecMatmulAndCopyResult have been defined. ltmatmul_test_utils.h (RAII
// guards, DeviceBuffer, PreparePhysicalA/B, traits) is included at the top
// of the wrapper, so all foundational types are also available.
// =============================================================================

// -----------------------------------------------------------------------------
// MatmulChainCtx: aggregated context for RunMatmulChain (descriptors + buffers).
// Uses unique_ptr for RAII guards since they have no default constructor.
// -----------------------------------------------------------------------------
struct MatmulChainCtx {
    sparse_test::SparseLtHandleGuard handle;
    std::unique_ptr<sparse_test::SparseLtMatDescGuard> matAStructured;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matADense;
    std::unique_ptr<sparse_test::SparseLtMatDescGuard> matBStructured;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matBDense;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matC;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matD;
    std::unique_ptr<sparse_test::SparseLtMatmulDescGuard> matmulDesc;
    sparse_test::DeviceBuffer dA;
    sparse_test::DeviceBuffer dB;
    sparse_test::DeviceBuffer dC;
    sparse_test::DeviceBuffer dD;
    size_t mk = 0;
    int32_t numBatches = 1;
    int64_t batchStride = 0;
    // C/D 布局（回读尺寸与解包由 runner / test.cpp 消费）
    sparse_test::MatmulLayoutOpts layout;
    size_t cdSpan = 0;  // C/D 单 batch 物理 span（元素数；紧凑时 == m*n）
};

// -----------------------------------------------------------------------------
// PrepareMatmulChainBuffers: compute physical layout + allocate device buffers.
//
// opts 尾参：ld/order 布局覆盖（默认 MatmulLayoutOpts{} = 既有
// 紧凑行为，全部既有调用点零变化）。覆盖语义：
//   - A structured（A-sparse）：ldSparse>0 → aLd=ldSparse + 物理重排 padded
//   - B dense（A-sparse）：ldDense>0 / denseOrder=COL → bLd + 物理重排
//   - C/D：cdOrder/ldCd → hCPhys（pack 后 host 数据）+ dC/dD 尺寸按物理 span
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline void PrepareMatmulChainBuffers(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    aclsparseOrder_t order, aclsparseOperation_t opA, aclsparseOperation_t opB,
    MatmulChainCtx& ctx,
    int64_t& physRowsA, int64_t& physColsA, int64_t& aLd,
    int64_t& bPhysRows, int64_t& bPhysCols, int64_t& bLd,
    const sparse_test::MatmulLayoutOpts& opts = sparse_test::MatmulLayoutOpts{},
    std::vector<InT>* hCPhysOut = nullptr)
{
    using namespace sparse_test;
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize;
    constexpr size_t kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    const bool transA = (opA == ACL_SPARSE_OP_TRANSPOSE);
    const bool transB = (opB == ACL_SPARSE_OP_TRANSPOSE);
    ctx.mk = static_cast<size_t>(m) * k;
    const size_t kn = static_cast<size_t>(k) * n;
    const size_t mn = static_cast<size_t>(m) * n;
    physRowsA = transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    physColsA = transA ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    bPhysRows = transB ? static_cast<int64_t>(n) : static_cast<int64_t>(k);
    bPhysCols = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n);
    std::vector<InT> hA_phys = PreparePhysicalA<InT>(hA, m, k, transA, order, aLd,
        ctx.numBatches, ctx.batchStride);
    if (opts.ldSparse > 0) {
        aLd = opts.ldSparse;
        hA_phys = PackBatchStridedPhysical<InT>(hA_phys,
            static_cast<int32_t>(physRowsA), static_cast<int32_t>(physColsA),
            order, aLd, ctx.numBatches, ctx.batchStride);
    }
    std::vector<InT> hB_phys = PreparePhysicalB<InT>(hB, k, n, transB,
        ctx.numBatches, ctx.batchStride);
    bLd = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n);
    if (opts.ldDense > 0 || opts.denseOrder == ACL_SPARSE_ORDER_COL) {
        bLd = (opts.ldDense > 0) ? opts.ldDense : bPhysRows;  // COL 紧凑 ld=physRows
        hB_phys = PackBatchStridedPhysical<InT>(hB_phys,
            static_cast<int32_t>(bPhysRows), static_cast<int32_t>(bPhysCols),
            opts.denseOrder, bLd, ctx.numBatches, ctx.batchStride);
    }
    const size_t aPhysSize = (order == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(physColsA) * static_cast<size_t>(aLd)
        : static_cast<size_t>(physRowsA) * static_cast<size_t>(aLd);
    // B dense 物理 span（denseOrder 感知：COL 布局下元素 (i,j) 位于
    // i + j*ld，物理缓冲 = physCols*ld；恒 ROW 口径会在 COL 且 k>n 时使
    // host vector 小于上传长度（越界读 host）、k<n 时截断设备输入）。
    const size_t bPhysSize = (opts.denseOrder == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(bPhysCols) * static_cast<size_t>(bLd)
        : static_cast<size_t>(bPhysRows) * static_cast<size_t>(bLd);
    // Batch: each matrix buffer is num_batches * batch_stride elements (with
    // padding between batches when batch_stride > matrix_size). The host data
    // must be pre-laid-out with this stride.
    const size_t totalAElems = BatchTotalElems(aPhysSize, ctx.numBatches, ctx.batchStride);
    const size_t totalBElems = BatchTotalElems(bPhysSize, ctx.numBatches, ctx.batchStride);
    const size_t cdSpan = CdSpanElems(m, n, opts);
    const size_t totalCElems = BatchTotalElems(cdSpan, ctx.numBatches, ctx.batchStride);
    const size_t totalDElems = BatchTotalElems(cdSpan, ctx.numBatches, ctx.batchStride);
    ctx.dA = DeviceBuffer::copyFrom(hA_phys.data(), totalAElems * kInElt);
    ctx.dB = DeviceBuffer::copyFrom(hB_phys.data(), totalBElems * kInElt);
    // C 按 (cdOrder, ldCd) 物理布局上传（ROW 紧凑快路径原样）；
    // hCPhysOut 供搜索链 rerun 重复上传同一物理布局。
    std::vector<InT> hCPhys = PackBatchStridedPhysical<InT>(hC, m, n,
        opts.cdOrder, CdLdOf(m, n, opts), ctx.numBatches, ctx.batchStride);
    ctx.dC = DeviceBuffer::copyFrom(hCPhys.data(), totalCElems * kInElt);
    ctx.dD = DeviceBuffer::alloc(totalDElems * kOutElt);
    if (hCPhysOut != nullptr) { *hCPhysOut = std::move(hCPhys); }
}

// -----------------------------------------------------------------------------
// CreateMatmulABDescriptors: create matA/matB descriptors based on path
// (dense*dense / A-sparse / B-sparse). Extracted from PrepareMatmulChainContext.
// dense B 侧 order/ld 由 opts 覆盖（denseOrder/ldDense）。
// -----------------------------------------------------------------------------
template <typename InT>
inline void CreateMatmulABDescriptors(
    MatmulChainCtx& ctx, bool isDensePath, bool isSparseA,
    aclsparseOrder_t order, int64_t physRowsA, int64_t physColsA, int64_t aLd,
    int64_t bPhysRows, int64_t bPhysCols, int64_t bLd,
    const sparse_test::MatmulLayoutOpts& opts,
    aclsparseLtMatDescriptor_t& matADesc, aclsparseLtMatDescriptor_t& matBDesc)
{
    using namespace sparse_test;
    constexpr aclDataType kInDtype = NpuDtypeTrait<InT>::kAclDtype;
    if (isDensePath) {
        ctx.matADense = std::make_unique<SparseLtDnMatDescGuard>(
            ctx.handle.get(), physRowsA, physColsA, aLd, 16, kInDtype, order);
        matADesc = ctx.matADense->get();
        ctx.matBDense = std::make_unique<SparseLtDnMatDescGuard>(
            ctx.handle.get(), bPhysRows, bPhysCols, bLd, 16, kInDtype,
            (opts.ldDense > 0 || opts.denseOrder == ACL_SPARSE_ORDER_COL)
                ? opts.denseOrder : ACL_SPARSE_ORDER_ROW);
        matBDesc = ctx.matBDense->get();
    } else if (isSparseA) {
        ctx.matAStructured = std::make_unique<SparseLtMatDescGuard>(
            ctx.handle.get(), physRowsA, physColsA, aLd, 16, kInDtype, order,
            ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        matADesc = ctx.matAStructured->get();
        ctx.matBDense = std::make_unique<SparseLtDnMatDescGuard>(
            ctx.handle.get(), bPhysRows, bPhysCols, bLd, 16, kInDtype,
            (opts.ldDense > 0 || opts.denseOrder == ACL_SPARSE_ORDER_COL)
                ? opts.denseOrder : ACL_SPARSE_ORDER_ROW);
        matBDesc = ctx.matBDense->get();
    } else {
        ctx.matADense = std::make_unique<SparseLtDnMatDescGuard>(
            ctx.handle.get(), physRowsA, physColsA, aLd, 16, kInDtype, order);
        matADesc = ctx.matADense->get();
        ctx.matBStructured = std::make_unique<SparseLtMatDescGuard>(
            ctx.handle.get(), bPhysRows, bPhysCols, bLd, 16, kInDtype,
            ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        matBDesc = ctx.matBStructured->get();
    }
}

// -----------------------------------------------------------------------------
// PrepareMatmulChainContext: compute physical layout + allocate device buffers
// + create descriptors + matmulDesc.
// opts 尾参（默认 = 既有紧凑行为）；hCPhysOut 输出 pack 后的 C host 数据
// （搜索链 rerun 用；nullptr = 调用方不需要）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline void PrepareMatmulChainContext(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    aclsparseOrder_t order, aclsparseOperation_t opA, aclsparseOperation_t opB,
    bool isSparseA, bool isDensePath, int32_t num_batches, int64_t batch_stride,
    MatmulNpuResult& result, MatmulChainCtx& ctx,
    const sparse_test::MatmulLayoutOpts& opts = sparse_test::MatmulLayoutOpts{},
    std::vector<InT>* hCPhysOut = nullptr)
{
    using namespace sparse_test;
    constexpr aclDataType kInDtype = NpuDtypeTrait<InT>::kAclDtype;
    constexpr aclDataType kOutDtype = NpuDtypeTrait<OutT>::kAclDtype;
    constexpr aclsparseComputeType_t kComputeType = MatmulComputeTrait<InT>::kComputeType;
    InitChainCtxBase(ctx, num_batches, batch_stride, opts, m, n);
    int64_t physRowsA = 0;
    int64_t physColsA = 0;
    int64_t aLd = 0;
    int64_t bPhysRows = 0;
    int64_t bPhysCols = 0;
    int64_t bLd = 0;
    PrepareMatmulChainBuffers<InT, OutT>(m, k, n, hA, hB, hC, order, opA, opB,
        ctx, physRowsA, physColsA, aLd, bPhysRows, bPhysCols, bLd, opts, hCPhysOut);
    aclsparseLtMatDescriptor_t matADesc = nullptr;
    aclsparseLtMatDescriptor_t matBDesc = nullptr;
    CreateMatmulABDescriptors<InT>(ctx, isDensePath, isSparseA, order,
        physRowsA, physColsA, aLd, bPhysRows, bPhysCols, bLd, opts, matADesc, matBDesc);
    // C/D 描述符接入 (cdOrder, ldCd)（C/D 同 order 同 ld，ValidateCDConsistency）
    const int64_t cdLd = CdLdOf(m, n, opts);
    ctx.matC = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, cdLd,
        16, kInDtype, opts.cdOrder);
    ctx.matD = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, cdLd,
        16, kOutDtype, opts.cdOrder);
    // Set batch attributes on all four matrix descriptors BEFORE MatmulDescriptorInit
    // (consistency check requires A/B/C/D numBatches to match).
    SetBatchAttrsOnAllMatDescs(ctx.handle.get(), &matADesc, &matBDesc,
        &ctx.matC->get(), &ctx.matD->get(),
        num_batches, batch_stride, result);
    ctx.matmulDesc = std::make_unique<SparseLtMatmulDescGuard>(ctx.handle.get(), opA, opB,
        matADesc, matBDesc, ctx.matC->get(), ctx.matD->get(), kComputeType);
}

// -----------------------------------------------------------------------------
// PreparePlanAndWorkspace: create algSel + plan + workspace.
// -----------------------------------------------------------------------------
inline bool PreparePlanAndWorkspace(MatmulNpuResult& result,
    sparse_test::SparseLtHandleGuard& handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc,
    int32_t algConfigId, int32_t splitK, int32_t splitKMode,
    std::unique_ptr<sparse_test::SparseLtAlgSelectionGuard>& algSel,
    std::unique_ptr<sparse_test::SparseLtPlanGuard>& plan,
    sparse_test::DeviceBuffer& dWorkspace, void*& wsPtr)
{
    using namespace sparse_test;
    algSel = std::make_unique<SparseLtAlgSelectionGuard>(handle.get(), matmulDesc.get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    if (!SetMatmulAlgAttributes(handle.get(), algSel->get(),
                                algConfigId, splitK, splitKMode, result)) {
        return false;
    }
    // getAttr roundtrip verification: read back the values set by SetMatmulAlgAttributes.
    result.algGetCfgRet = aclsparseLtMatmulAlgGetAttribute(
        handle.get(), algSel->cptr(),
        ACLSPARSELT_MATMUL_ALG_CONFIG_ID, &result.gotAlgConfigId, sizeof(int32_t));
    result.algGetSplitRet = aclsparseLtMatmulAlgGetAttribute(
        handle.get(), algSel->cptr(),
        ACLSPARSELT_MATMUL_SPLIT_K, &result.gotSplitK, sizeof(int32_t));
    result.algGetSplitKModeRet = aclsparseLtMatmulAlgGetAttribute(
        handle.get(), algSel->cptr(),
        ACLSPARSELT_MATMUL_SPLIT_K_MODE, &result.gotSplitKMode, sizeof(int32_t));
    plan = std::make_unique<SparseLtPlanGuard>(handle.get(), matmulDesc.get(), algSel->get());
    result.wsRet = aclsparseLtMatmulGetWorkspace(handle.get(), plan->cptr(), &result.workspaceSize);
    if (result.wsRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    dWorkspace = (result.workspaceSize > 0)
        ? DeviceBuffer::alloc(result.workspaceSize) : DeviceBuffer{};
    wsPtr = (result.workspaceSize > 0) ? dWorkspace.get() : nullptr;
    return true;
}

// -----------------------------------------------------------------------------
// [Prune 回读共享段] 剪枝结果回读 host：npuPrunedA 扩容 + D2H memcpy，
// 失败时置 pruneRet=EXECUTION_FAILED（ltmatmul RunPruneForASparse 与
// PruneInt8CDBSparse 共用）。同步检查段保留在各调用方（A-sparse 链
// 记录 result.syncRet，Int8CD 链内联判断）。
// -----------------------------------------------------------------------------
inline bool CopyPrunedToHost(MatmulNpuResult& result,
    sparse_test::DeviceBuffer& dPruned, size_t totalBytes)
{
    result.npuPrunedA.resize(totalBytes);
    result.memcpyRet = aclrtMemcpy(result.npuPrunedA.data(), totalBytes,
        dPruned.get(), totalBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (result.memcpyRet != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// RunPruneForASparse: run prune on A-sparse path, update matAPtr + npuPrunedA.
// When num_batches > 1, prune each batch separately (loop with batch stride
// offset). The pruned buffer is num_batches * batch_stride * kInElt bytes.
// -----------------------------------------------------------------------------
// prune 单调用化：描述符携带 numBatches/batchStride，一次调用处理全
// batch（传 batch0 基址）。host 侧切片循环会使末切片基址被算子按 numBatches
// 二次推进而越界——禁止切片。输出布局（batch-strided）与切片版一致。
// 回读段（CopyPrunedToHost）与 PruneInt8CDBSparse 共享。
inline bool RunPruneForASparse(MatmulNpuResult& result,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulDescriptor_t matmulDesc,
    sparse_test::DeviceBuffer& dA, sparse_test::DeviceBuffer& dPruned,
    size_t mk, size_t kInElt, aclsparseLtPruneAlg_t pruneAlg, void*& matAPtr,
    int32_t num_batches = 1, int64_t batch_stride = 0)
{
    aclrtStream stream = 0;
    const size_t totalElems = BatchTotalElems(mk, num_batches, batch_stride);
    dPruned = sparse_test::DeviceBuffer::alloc(totalElems * kInElt);
    result.pruneRet = aclsparseLtSpMMAPrune(
        handle, &matmulDesc, dA.get(), dPruned.get(), pruneAlg, stream);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    result.syncRet = aclrtSynchronizeStream(stream);
    if (result.syncRet != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }
    if (!CopyPrunedToHost(result, dPruned, totalElems * kInElt)) { return false; }
    matAPtr = dPruned.get();
    return true;
}

// -----------------------------------------------------------------------------
// MatmulRunArgs: common parameters for matmul chain functions.
// Extracted to eliminate duplicate parameter lists between RunMatmulChain,
// RunMatmulBSparseChainWithTrans, and RunMatmulNpu.
// layout: ld/order 布局覆盖（默认紧凑，既有用例零变化）。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
struct MatmulRunArgs {
    int32_t m, k, n;
    const std::vector<InT>& hA;
    const std::vector<InT>& hB;
    const std::vector<InT>& hC;
    std::vector<OutT>& hD;
    float alpha, beta;
    int32_t algConfigId, splitK, splitKMode;
    aclsparseOrder_t order;
    aclsparseOperation_t opA, opB;
    aclsparseLtPruneAlg_t pruneAlg;
    int32_t alpha_vector_scaling;
    int32_t beta_vector_scaling;
    const std::vector<float>* alphaVec;
    const std::vector<float>* betaVec;
    // Epilogue (backward-compatible defaults: all-off).
    int32_t biasEnabled = 0;
    int64_t biasStride = 0;
    int32_t activationType = 0;
    float reluUpperBound = FLT_MAX;
    float reluThreshold = 0.0f;
    float geluScaling = 1.0f;
    int32_t numBatches = 1;
    int64_t batchStride = 0;
    const std::vector<float>* biasVec = nullptr;
    // bias dtype follows C dtype (FP16 path → FP16 bias, BF16 → BF16,
    // INT8 → FP32 per cuSPARSELt). Set by RunMatmulNpu from BiasDtypeTrait<InT>.
    size_t biasEltSize = sizeof(float);
    aclDataType biasDtype = ACL_FLOAT;
    // ld/order 布局覆盖（ldSparse/ldDense/denseOrder/cdOrder/ldCd）
    sparse_test::MatmulLayoutOpts layout;
};

// -----------------------------------------------------------------------------
// ChainSetupResult + SetupVecEpiPlanWorkspace: common
// post-context sequence (batchSetRet check → vec scaling → epilogue → plan +
// workspace) extracted from RunMatmulChain / RunMatmulBSparseChainWithTrans
// to eliminate duplicate code.
// -----------------------------------------------------------------------------
struct ChainSetupResult {
    VecScalingBuffers vecBufs;
    EpilogueBuffers epiBufs;
    std::unique_ptr<sparse_test::SparseLtAlgSelectionGuard> algSel;
    std::unique_ptr<sparse_test::SparseLtPlanGuard> plan;
    sparse_test::DeviceBuffer dWorkspace;
    void* wsPtr = nullptr;
};

template <typename InT, typename OutT>
inline bool SetupVecEpiPlanWorkspace(
    sparse_test::SparseLtHandleGuard& handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc,
    const MatmulRunArgs<InT, OutT>& args, int32_t m,
    MatmulNpuResult& result, ChainSetupResult& out)
{
    if (result.batchSetRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    out.vecBufs = SetVecScalingAttrs(handle.get(), matmulDesc, m,
        args.alpha_vector_scaling, args.beta_vector_scaling, args.alphaVec, args.betaVec, result);
    if (result.descSetRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    out.epiBufs = SetEpilogueAttrs(handle.get(), matmulDesc, m,
        args.biasEnabled, args.biasStride,
        args.activationType, args.reluUpperBound, args.reluThreshold, args.geluScaling,
        args.numBatches, args.batchStride, args.biasVec, result,
        args.biasEltSize, args.biasDtype);
    if (result.biasSetRet != ACL_SPARSE_STATUS_SUCCESS ||
        result.actSetRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    return PreparePlanAndWorkspace(result, handle, matmulDesc, args.algConfigId,
                                    args.splitK, args.splitKMode, out.algSel, out.plan,
                                    out.dWorkspace, out.wsPtr);
}

// -----------------------------------------------------------------------------
// Run matmul chain for A-sparse or dense*dense path.
// (B-sparse path is handled separately by RunMatmulBSparseChainWithTrans
//  because it needs proper transB buffer lifetime management.)
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline MatmulNpuResult RunMatmulChain(
    const MatmulRunArgs<InT, OutT>& args, bool isSparseA, bool isDensePath)
{
    using namespace sparse_test;
    MatmulNpuResult result{};
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize;
    constexpr size_t kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    const size_t mn = static_cast<size_t>(args.m) * args.n;
    MatmulChainCtx ctx;
    PrepareMatmulChainContext<InT, OutT>(args.m, args.k, args.n,
                                         args.hA, args.hB, args.hC,
                                         args.order, args.opA, args.opB,
                                         isSparseA, isDensePath,
                                         args.numBatches, args.batchStride,
                                         result, ctx, args.layout);
    ChainSetupResult setup;
    if (!SetupVecEpiPlanWorkspace<InT, OutT>(ctx.handle, *ctx.matmulDesc, args, args.m,
                                                result, setup)) { return result; }
    DeviceBuffer dPruned;
    void* matAPtr = ctx.dA.get();
    if (!isDensePath && isSparseA) {
        if (!RunPruneForASparse(result, ctx.handle.get(), ctx.matmulDesc->get(),
                ctx.dA, dPruned, ctx.mk, kInElt, args.pruneAlg, matAPtr,
                args.numBatches, args.batchStride)) { return result; }
    }
    // 回读尺寸 = C/D 物理 span（batch 布局含 stride padding；紧凑时 == mn*nb）
    const size_t totalOutElems = BatchTotalElems(ctx.cdSpan, ctx.numBatches, ctx.batchStride);
    ExecMatmulAndCopyResult<OutT>(result, ctx.handle.get(), setup.plan->cptr(),
        matAPtr, ctx.dB.get(), ctx.dC.get(), ctx.dD.get(), setup.wsPtr, setup.vecBufs, args.alpha, args.beta,
        args.alpha_vector_scaling, args.beta_vector_scaling, ctx.dD, args.hD, totalOutElems, kOutElt);
    return result;
}

// -----------------------------------------------------------------------------
// B-sparse 链族 + 物理布局计算段（MatmulBSparseCtx / BSparsePhysicalLayout /
// Compute{A,B}SparsePhysicalLayout / AllocBSparseChainBuffers /
// PrepareBSparseChainContext / RunBSparsePruneAndTranspose /
// RunMatmulBSparseChainWithTrans）位于 ltmatmul_npu_chain_bsparse.h，
// 此处尾部 include 转发——本头的全部既有 include 点
// （ltmatmul_npu_wrapper.h / matmulSearch_npu_wrapper.h 等）对
// B-sparse 族符号零改动可见。
// -----------------------------------------------------------------------------
#include "ltmatmul_npu_chain_bsparse.h"


#endif  // TEST_LTMATMUL_NPU_CHAIN_H_
