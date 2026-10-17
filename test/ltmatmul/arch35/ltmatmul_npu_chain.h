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
// Matmul chain implementation extracted from
// ltmatmul_npu_wrapper.h to reduce header file size.
//
// Provides the internal chain runners consumed by RunMatmulNpu:
//   - MatmulChainCtx / MatmulBSparseCtx: aggregated descriptor + buffer context
//   - PrepareMatmulChainBuffers / CreateMatmulABDescriptors /
//     PrepareMatmulChainContext: A-sparse / dense*dense context setup
//   - PrepareBSparseChainContext: B-sparse context setup
//   - PreparePlanAndWorkspace: algSel + plan + workspace
//   - RunPruneForASparse / RunBSparsePruneAndTranspose: prune helpers
//   - MatmulRunArgs / ChainSetupResult / SetupVecEpiPlanWorkspace: shared
//     post-context sequence (vec scaling + epilogue + plan + workspace)
//   - RunMatmulChain: A-sparse / dense*dense chain runner
//   - RunMatmulBSparseChainWithTrans: B-sparse chain runner
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
// Extracted from RunMatmulChain to reduce NBNC.
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
};

// -----------------------------------------------------------------------------
// MatmulBSparseCtx: aggregated context for RunMatmulBSparseChainWithTrans.
// -----------------------------------------------------------------------------
struct MatmulBSparseCtx {
    sparse_test::SparseLtHandleGuard handle;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matA;
    std::unique_ptr<sparse_test::SparseLtMatDescGuard> matB;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matC;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matD;
    std::unique_ptr<sparse_test::SparseLtMatmulDescGuard> matmulDesc;
    sparse_test::DeviceBuffer dA;
    sparse_test::DeviceBuffer dB;
    sparse_test::DeviceBuffer dC;
    sparse_test::DeviceBuffer dD;
    size_t kn = 0;
    int32_t numBatches = 1;
    int64_t batchStride = 0;
};

// -----------------------------------------------------------------------------
// PrepareMatmulChainBuffers: compute physical layout + allocate device buffers.
// Extracted from PrepareMatmulChainContext to reduce NBNC.
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline void PrepareMatmulChainBuffers(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    aclsparseOrder_t order, aclsparseOperation_t opA, aclsparseOperation_t opB,
    MatmulChainCtx& ctx,
    int64_t& physRowsA, int64_t& physColsA, int64_t& aLd,
    int64_t& bPhysRows, int64_t& bPhysCols, int64_t& bLd)
{
    using namespace sparse_test;
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize;
    constexpr size_t kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    const bool transA = (opA == ACL_SPARSE_OP_TRANSPOSE);
    const bool transB = (opB == ACL_SPARSE_OP_TRANSPOSE);
    ctx.mk = static_cast<size_t>(m) * k;
    const size_t kn = static_cast<size_t>(k) * n;
    const size_t mn = static_cast<size_t>(m) * n;
    std::vector<InT> hA_phys = PreparePhysicalA<InT>(hA, m, k, transA, order, aLd,
        ctx.numBatches, ctx.batchStride);
    std::vector<InT> hB_phys = PreparePhysicalB<InT>(hB, k, n, transB,
        ctx.numBatches, ctx.batchStride);
    physRowsA = transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    physColsA = transA ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    const size_t aPhysSize = (order == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(physColsA) * static_cast<size_t>(aLd)
        : static_cast<size_t>(physRowsA) * static_cast<size_t>(aLd);
    const size_t bPhysSize = transB ? static_cast<size_t>(n) * k : kn;
    // Batch: each matrix buffer is num_batches * batch_stride elements (with
    // padding between batches when batch_stride > matrix_size). The host data
    // must be pre-laid-out with this stride.
    const size_t totalAElems = BatchTotalElems(aPhysSize, ctx.numBatches, ctx.batchStride);
    const size_t totalBElems = BatchTotalElems(bPhysSize, ctx.numBatches, ctx.batchStride);
    const size_t totalCElems = BatchTotalElems(mn, ctx.numBatches, ctx.batchStride);
    const size_t totalDElems = BatchTotalElems(mn, ctx.numBatches, ctx.batchStride);
    ctx.dA = DeviceBuffer::copyFrom(hA_phys.data(), totalAElems * kInElt);
    ctx.dB = DeviceBuffer::copyFrom(hB_phys.data(), totalBElems * kInElt);
    ctx.dC = DeviceBuffer::copyFrom(hC.data(), totalCElems * kInElt);
    ctx.dD = DeviceBuffer::alloc(totalDElems * kOutElt);
    bPhysRows = transB ? static_cast<int64_t>(n) : static_cast<int64_t>(k);
    bPhysCols = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n);
    bLd = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n);
}

// -----------------------------------------------------------------------------
// CreateMatmulABDescriptors: create matA/matB descriptors based on path
// (dense*dense / A-sparse / B-sparse). Extracted from PrepareMatmulChainContext.
// -----------------------------------------------------------------------------
template <typename InT>
inline void CreateMatmulABDescriptors(
    MatmulChainCtx& ctx, bool isDensePath, bool isSparseA,
    aclsparseOrder_t order, int64_t physRowsA, int64_t physColsA, int64_t aLd,
    int64_t bPhysRows, int64_t bPhysCols, int64_t bLd,
    aclsparseLtMatDescriptor_t& matADesc, aclsparseLtMatDescriptor_t& matBDesc)
{
    using namespace sparse_test;
    constexpr aclDataType kInDtype = NpuDtypeTrait<InT>::kAclDtype;
    if (isDensePath) {
        ctx.matADense = std::make_unique<SparseLtDnMatDescGuard>(
            ctx.handle.get(), physRowsA, physColsA, aLd, 16, kInDtype, order);
        matADesc = ctx.matADense->get();
        ctx.matBDense = std::make_unique<SparseLtDnMatDescGuard>(
            ctx.handle.get(), bPhysRows, bPhysCols, bLd, 16, kInDtype, ACL_SPARSE_ORDER_ROW);
        matBDesc = ctx.matBDense->get();
    } else if (isSparseA) {
        ctx.matAStructured = std::make_unique<SparseLtMatDescGuard>(
            ctx.handle.get(), physRowsA, physColsA, aLd, 16, kInDtype, order,
            ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        matADesc = ctx.matAStructured->get();
        ctx.matBDense = std::make_unique<SparseLtDnMatDescGuard>(
            ctx.handle.get(), bPhysRows, bPhysCols, bLd, 16, kInDtype, ACL_SPARSE_ORDER_ROW);
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
// + create descriptors + matmulDesc. Extracted from RunMatmulChain to reduce NBNC.
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline void PrepareMatmulChainContext(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    aclsparseOrder_t order, aclsparseOperation_t opA, aclsparseOperation_t opB,
    bool isSparseA, bool isDensePath, int32_t num_batches, int64_t batch_stride,
    MatmulNpuResult& result, MatmulChainCtx& ctx)
{
    using namespace sparse_test;
    constexpr aclDataType kInDtype = NpuDtypeTrait<InT>::kAclDtype;
    constexpr aclDataType kOutDtype = NpuDtypeTrait<OutT>::kAclDtype;
    constexpr aclsparseComputeType_t kComputeType = MatmulComputeTrait<InT>::kComputeType;
    ctx.numBatches = num_batches;
    ctx.batchStride = batch_stride;
    int64_t physRowsA = 0;
    int64_t physColsA = 0;
    int64_t aLd = 0;
    int64_t bPhysRows = 0;
    int64_t bPhysCols = 0;
    int64_t bLd = 0;
    PrepareMatmulChainBuffers<InT, OutT>(m, k, n, hA, hB, hC, order, opA, opB,
        ctx, physRowsA, physColsA, aLd, bPhysRows, bPhysCols, bLd);
    aclsparseLtMatDescriptor_t matADesc = nullptr;
    aclsparseLtMatDescriptor_t matBDesc = nullptr;
    CreateMatmulABDescriptors<InT>(ctx, isDensePath, isSparseA, order,
        physRowsA, physColsA, aLd, bPhysRows, bPhysCols, bLd, matADesc, matBDesc);
    ctx.matC = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, n, 16, kInDtype, ACL_SPARSE_ORDER_ROW);
    ctx.matD = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, n, 16, kOutDtype, ACL_SPARSE_ORDER_ROW);
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
// Extracted from RunMatmulChain/RunMatmulBSparseChainWithTrans to reduce NBNC.
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
// RunPruneForASparse: run prune on A-sparse path, update matAPtr + npuPrunedA.
// When num_batches > 1, prune each batch separately (loop with batch stride
// offset). The pruned buffer is num_batches * batch_stride * kInElt bytes.
// Extracted from RunMatmulChain to reduce NBNC.
// -----------------------------------------------------------------------------
inline bool RunPruneForASparse(MatmulNpuResult& result,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulDescriptor_t matmulDesc,
    sparse_test::DeviceBuffer& dA, sparse_test::DeviceBuffer& dPruned,
    size_t mk, size_t kInElt, aclsparseLtPruneAlg_t pruneAlg, void*& matAPtr,
    int32_t num_batches = 1, int64_t batch_stride = 0)
{
    aclrtStream stream = 0;
    const size_t totalElems = BatchTotalElems(mk, num_batches, batch_stride);
    dPruned = sparse_test::DeviceBuffer::alloc(totalElems * kInElt);
    const int32_t nb = (num_batches > 1) ? num_batches : 1;
    const int64_t bs = (num_batches > 1 && batch_stride > 0) ? batch_stride
                       : static_cast<int64_t>(mk);
    for (int32_t b = 0; b < nb; ++b) {
        const size_t byteOff = static_cast<size_t>(b) * static_cast<size_t>(bs) * kInElt;
        uint8_t* dInB = static_cast<uint8_t*>(dA.get()) + byteOff;
        uint8_t* dOutB = static_cast<uint8_t*>(dPruned.get()) + byteOff;
        result.pruneRet = aclsparseLtSpMMAPrune(
            handle, &matmulDesc, dInB, dOutB, pruneAlg, stream);
        if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    }
    result.syncRet = aclrtSynchronizeStream(stream);
    if (result.syncRet != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }
    result.npuPrunedA.resize(totalElems * kInElt);
    result.memcpyRet = aclrtMemcpy(result.npuPrunedA.data(), totalElems * kInElt,
        dPruned.get(), totalElems * kInElt, ACL_MEMCPY_DEVICE_TO_HOST);
    if (result.memcpyRet != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }
    matAPtr = dPruned.get();
    return true;
}

// -----------------------------------------------------------------------------
// MatmulRunArgs: common parameters for matmul chain functions.
// Extracted to eliminate duplicate parameter lists between RunMatmulChain,
// RunMatmulBSparseChainWithTrans, and RunMatmulNpu.
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
                                         result, ctx);
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
    const size_t totalOutElems = BatchTotalElems(mn, ctx.numBatches, ctx.batchStride);
    ExecMatmulAndCopyResult<OutT>(result, ctx.handle.get(), setup.plan->cptr(),
        matAPtr, ctx.dB.get(), ctx.dC.get(), ctx.dD.get(), setup.wsPtr, setup.vecBufs, args.alpha, args.beta,
        args.alpha_vector_scaling, args.beta_vector_scaling, ctx.dD, args.hD, totalOutElems, kOutElt);
    return result;
}

// -----------------------------------------------------------------------------
// PrepareBSparseChainContext: compute physical layout + allocate device buffers
// + create B-sparse descriptors. Extracted from RunMatmulBSparseChainWithTrans.
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline void PrepareBSparseChainContext(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    aclsparseOrder_t order, aclsparseOperation_t opA, aclsparseOperation_t opB,
    int32_t num_batches, int64_t batch_stride, MatmulNpuResult& result,
    MatmulBSparseCtx& ctx)
{
    using namespace sparse_test;
    constexpr aclDataType kInDtype = NpuDtypeTrait<InT>::kAclDtype;
    constexpr aclDataType kOutDtype = NpuDtypeTrait<OutT>::kAclDtype;
    constexpr aclsparseComputeType_t kComputeType = MatmulComputeTrait<InT>::kComputeType;
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize;
    constexpr size_t kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    ctx.numBatches = num_batches;
    ctx.batchStride = batch_stride;
    const bool transA = (opA == ACL_SPARSE_OP_TRANSPOSE);
    const bool transB = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const size_t mn = static_cast<size_t>(m) * n;
    ctx.kn = static_cast<size_t>(k) * n;
    int64_t aLd = 0;
    int64_t bLdOut = 0;
    std::vector<InT> hA_phys = PreparePhysicalA<InT>(hA, m, k, transA, ACL_SPARSE_ORDER_ROW, aLd,
        num_batches, batch_stride);
    std::vector<InT> hB_phys = PreparePhysicalA<InT>(hB, k, n, transB, order, bLdOut,
        num_batches, batch_stride);
    const int64_t physRowsA = transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    const int64_t physColsA = transA ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    const size_t aPhysSize = static_cast<size_t>(physRowsA) * static_cast<size_t>(aLd);
    const size_t bPhysSize = static_cast<size_t>(ctx.kn);
    // Batch: each matrix buffer is num_batches * batch_stride elements.
    const size_t totalAElems = BatchTotalElems(aPhysSize, num_batches, batch_stride);
    const size_t totalBElems = BatchTotalElems(bPhysSize, num_batches, batch_stride);
    const size_t totalCElems = BatchTotalElems(mn, num_batches, batch_stride);
    const size_t totalDElems = BatchTotalElems(mn, num_batches, batch_stride);
    ctx.dA = DeviceBuffer::copyFrom(hA_phys.data(), totalAElems * kInElt);
    ctx.dB = DeviceBuffer::copyFrom(hB_phys.data(), totalBElems * kInElt);
    ctx.dC = DeviceBuffer::copyFrom(hC.data(), totalCElems * kInElt);
    ctx.dD = DeviceBuffer::alloc(totalDElems * kOutElt);
    const int64_t bPhysRows = transB ? static_cast<int64_t>(n) : static_cast<int64_t>(k);
    const int64_t bPhysCols = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n);
    ctx.matA = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), physRowsA, physColsA, aLd, 16, kInDtype, ACL_SPARSE_ORDER_ROW);
    ctx.matB = std::make_unique<SparseLtMatDescGuard>(ctx.handle.get(), bPhysRows, bPhysCols, bLdOut, 16, kInDtype,
                                order, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    ctx.matC = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, n, 16, kInDtype, ACL_SPARSE_ORDER_ROW);
    ctx.matD = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, n, 16, kOutDtype, ACL_SPARSE_ORDER_ROW);
    // Set batch attributes on all four matrix descriptors BEFORE MatmulDescriptorInit.
    SetBatchAttrsOnAllMatDescs(ctx.handle.get(), &ctx.matA->get(), &ctx.matB->get(),
        &ctx.matC->get(), &ctx.matD->get(),
        num_batches, batch_stride, result);
    ctx.matmulDesc = std::make_unique<SparseLtMatmulDescGuard>(ctx.handle.get(), opA, opB,
        ctx.matA->get(), ctx.matB->get(), ctx.matC->get(), ctx.matD->get(), kComputeType);
}

// -----------------------------------------------------------------------------
// RunBSparsePruneAndTranspose: prune B for B-sparse path.
// Extracted from RunMatmulBSparseChainWithTrans to reduce NBNC.
//
// Key insight (verified via prune_kernel.cpp SpltPruneTransRowOrder path):
// The pruned B from aclsparseLtSpMMAPrune is ALWAYS in (k,n) row layout,
// regardless of transB. For transB=T, the prune kernel reads the physical
// (n,k) layout and transposes internally, outputting (k,n). For transB=F,
// the physical B is already (k,n) and prune keeps it. The matmul internally
// forces td.transB=0 for B-sparse and reads B as NDExt(k,n), so no host-side
// transpose is needed — the pruned B is used directly by both matmul and golden.
// -----------------------------------------------------------------------------
template <typename InT>
inline void RunBSparsePruneAndTranspose(MatmulNpuResult& result,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulDescriptor_t matmulDesc,
    sparse_test::DeviceBuffer& dB, sparse_test::DeviceBuffer& dBPruned,
    size_t kn, size_t kInElt,
    aclsparseLtPruneAlg_t pruneAlg, void*& matBPtr,
    int32_t numBatches = 1, int64_t batchStride = 0)
{
    using namespace sparse_test;
    aclrtStream stream = 0;
    const size_t totalElems = BatchTotalElems(kn, numBatches, batchStride);
    dBPruned = DeviceBuffer::alloc(totalElems * kInElt);
    const int32_t nb = (numBatches > 1) ? numBatches : 1;
    const int64_t bs = (numBatches > 1 && batchStride > 0) ? batchStride
                       : static_cast<int64_t>(kn);
    for (int32_t b = 0; b < nb; ++b) {
        const size_t byteOff = static_cast<size_t>(b) * static_cast<size_t>(bs) * kInElt;
        uint8_t* dInB = static_cast<uint8_t*>(dB.get()) + byteOff;
        uint8_t* dOutB = static_cast<uint8_t*>(dBPruned.get()) + byteOff;
        result.pruneRet = aclsparseLtSpMMAPrune(
            handle, &matmulDesc, dInB, dOutB, pruneAlg, stream);
        if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return; }
    }
    result.syncRet = aclrtSynchronizeStream(stream);
    if (result.syncRet != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return;
    }
    // Pruned B is already in (k,n) row layout — used directly by matmul (which
    // forces td.transB=0 for B-sparse) and by golden (which expects B[k][n]).
    matBPtr = dBPruned.get();
    // Copy pruned B back to host for golden use (already in k×n layout).
    result.npuPrunedA.resize(totalElems * kInElt);
    result.memcpyRet = aclrtMemcpy(result.npuPrunedA.data(), totalElems * kInElt,
        dBPruned.get(), totalElems * kInElt, ACL_MEMCPY_DEVICE_TO_HOST);
    if (result.memcpyRet != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return;
    }
}

// -----------------------------------------------------------------------------
// B-sparse chain with transB support (proper buffer lifetime management).
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline MatmulNpuResult RunMatmulBSparseChainWithTrans(
    const MatmulRunArgs<InT, OutT>& args)
{
    using namespace sparse_test;
    MatmulNpuResult result{};
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize;
    constexpr size_t kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    const size_t mn = static_cast<size_t>(args.m) * args.n;
    MatmulBSparseCtx ctx;
    PrepareBSparseChainContext<InT, OutT>(args.m, args.k, args.n,
                                          args.hA, args.hB, args.hC,
                                          args.order, args.opA, args.opB,
                                          args.numBatches, args.batchStride,
                                          result, ctx);
    ChainSetupResult setup;
    if (!SetupVecEpiPlanWorkspace<InT, OutT>(ctx.handle, *ctx.matmulDesc, args, args.m,
                                               result, setup)) { return result; }
    void* matBPtr = nullptr;
    DeviceBuffer dBPruned;
    RunBSparsePruneAndTranspose<InT>(result, ctx.handle.get(), ctx.matmulDesc->get(),
        ctx.dB, dBPruned, ctx.kn, kInElt, args.pruneAlg, matBPtr,
        args.numBatches, args.batchStride);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }
    const size_t totalOutElems = BatchTotalElems(mn, ctx.numBatches, ctx.batchStride);
    ExecMatmulAndCopyResult<OutT>(result, ctx.handle.get(), setup.plan->cptr(),
        ctx.dA.get(), matBPtr, ctx.dC.get(), ctx.dD.get(), setup.wsPtr, setup.vecBufs, args.alpha, args.beta,
        args.alpha_vector_scaling, args.beta_vector_scaling, ctx.dD, args.hD, totalOutElems, kOutElt);
    return result;
}

#endif  // TEST_LTMATMUL_NPU_CHAIN_H_
