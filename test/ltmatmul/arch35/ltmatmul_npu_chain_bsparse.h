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

#ifndef TEST_LTMATMUL_NPU_CHAIN_BSPARSE_H_
#define TEST_LTMATMUL_NPU_CHAIN_BSPARSE_H_

// =============================================================================
// B-sparse 链族 + 物理布局计算段。ltmatmul_npu_chain.h 尾部 include 转发，
// 既有引用点（ltmatmul_npu_wrapper.h / matmulSearch_npu_wrapper.h 等）
// 零改动。
//
// Provides:
//   - MatmulBSparseCtx: aggregated B-sparse descriptor + buffer context
//   - BSparsePhysicalLayout / ComputeBSparsePhysicalLayout /
//     ComputeASparsePhysicalLayout: (order, ld) 物理布局计算段（双侧物理
//     span；A-sparse 对偶段供 RunMatmulNpuInt8CD 的 batch/布局透传复用）
//   - AllocBSparseChainBuffers / PrepareBSparseChainContext: B-sparse
//     context setup
//   - RunBSparsePruneAndTranspose: B-sparse prune helper
//   - RunMatmulBSparseChainWithTrans: B-sparse chain runner
//
// Include order: 本头由 ltmatmul_npu_chain.h 尾部 include（在其
// MatmulRunArgs / ChainSetupResult / SetupVecEpiPlanWorkspace 共享段之后）；
// ltmatmul_npu_wrapper.h 的 MatmulNpuResult / VecScalingBuffers / epilogue
// helpers / SetBatchAttrsOnAllMatDescs / BatchTotalElems /
// ExecMatmulAndCopyResult 与 ltmatmul_test_utils.h 基建（RAII guards /
// DeviceBuffer / InitChainCtxBase / PreparePhysicalA/B /
// PackBatchStridedPhysical）经该 include 链先行可见。不单独 include。
// =============================================================================

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
    // C/D 布局（回读尺寸与解包由 runner / test.cpp 消费）
    sparse_test::MatmulLayoutOpts layout;
    size_t cdSpan = 0;
};

// -----------------------------------------------------------------------------
// BSparsePhysicalLayout + ComputeBSparsePhysicalLayout: B-sparse 链的
// 物理布局计算段（dense-A / structured-B 的 (order, ld) 物理数据 + 双侧
// 物理 span）。
// -----------------------------------------------------------------------------
template <typename InT>
struct BSparsePhysicalLayout {
    std::vector<InT> hA_phys;
    std::vector<InT> hB_phys;
    int64_t aLd = 0;         // dense-A 物理 ld（ROW 紧凑 = physColsA）
    int64_t bLdOut = 0;      // structured-B 物理 ld（prune 输入布局）
    int64_t physRowsA = 0;   // dense-A 物理行数（transA 决定 (m,k)/(k,m)）
    int64_t physColsA = 0;
    int64_t bPhysRows = 0;   // structured-B 物理行数（transB 决定 (k,n)/(n,k)）
    int64_t bPhysCols = 0;
    size_t aPhysSize = 0;    // dense-A 物理 span（prune 输入缓冲上界）
    size_t bPhysSize = 0;    // structured-B 物理 span（prune 输入缓冲上界）
};

template <typename InT>
inline BSparsePhysicalLayout<InT> ComputeBSparsePhysicalLayout(
    const std::vector<InT>& hA, const std::vector<InT>& hB,
    int32_t m, int32_t k, int32_t n, aclsparseOrder_t order,
    bool transA, bool transB, int32_t num_batches, int64_t batch_stride,
    const sparse_test::MatmulLayoutOpts& opts)
{
    using namespace sparse_test;
    BSparsePhysicalLayout<InT> lay;
    // dense-A 自然 compact（ROW 快路径）：transA 转置到 (k,m)/(m,k)，
    // 随后按 (denseOrder, ldDense) 装配。
    lay.hA_phys = PreparePhysicalA<InT>(hA, m, k, transA, ACL_SPARSE_ORDER_ROW, lay.aLd,
        num_batches, batch_stride);
    lay.physRowsA = transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    lay.physColsA = transA ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    if (opts.ldDense > 0 || opts.denseOrder == ACL_SPARSE_ORDER_COL) {
        // dense-A 布局覆盖（L8：COL + ld 的 dense-A 直读路径）
        lay.aLd = (opts.ldDense > 0) ? opts.ldDense : lay.physRowsA;
        lay.hA_phys = PackBatchStridedPhysical<InT>(lay.hA_phys,
            static_cast<int32_t>(lay.physRowsA), static_cast<int32_t>(lay.physColsA),
            opts.denseOrder, lay.aLd, num_batches, batch_stride);
    }
    // (b) B structured：逻辑 (k,n) → 描述符 (order, ldSparse) 布局
    //（PreparePhysicalA 内部完成自然 compact + (order, ld) 装配）。
    lay.hB_phys = PreparePhysicalA<InT>(hB, k, n, transB, order, lay.bLdOut,
        num_batches, batch_stride, opts.ldSparse);
    lay.bPhysRows = transB ? static_cast<int64_t>(n) : static_cast<int64_t>(k);
    lay.bPhysCols = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n);
    // A dense 物理 span（order 感知：COL 布局下 span = physCols*ld
    // ——元素 (i,j) 位于 i + j*ld，最大偏移 (physCols-1)*ld + physRows-1；恒 ROW
    // 语义会在 COL 时截断 dA（case224 根因：96*112=10752 < 实需 160*112=17920，
    // 后 64 列物理数据丢失 → cube 越界读）。对齐同函数 bPhysSize 与 A-sparse 链
    // PrepareMatmulChainBuffers 的 aPhysSize 同范式写法）
    lay.aPhysSize = (opts.denseOrder == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(lay.physColsA) * static_cast<size_t>(lay.aLd)
        : static_cast<size_t>(lay.physRowsA) * static_cast<size_t>(lay.aLd);
    // B 物理 span（prune 输入缓冲上界；(b) prune 输出保持 B 描述符
    // (order, ld) 布局——该 span 亦为输出缓冲上界）
    lay.bPhysSize = (order == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(lay.bPhysCols) * static_cast<size_t>(lay.bLdOut)
        : static_cast<size_t>(lay.bPhysRows) * static_cast<size_t>(lay.bLdOut);
    return lay;
}

// -----------------------------------------------------------------------------
// ComputeASparsePhysicalLayout: A-sparse 链的 A/B 物理布局段
// （ComputeBSparsePhysicalLayout 的对偶——A structured (m,k) 物理 +
// (order, ldSparse)；B dense (k,n)/(n,k) 物理 + (denseOrder, ldDense)）。
// 段内逻辑与 PrepareMatmulChainBuffers 的 A/B 布局段同范式（单 batch 紧凑时
// 逐位一致）；供 RunMatmulNpuInt8CD 的 batch/布局透传复用。
// -----------------------------------------------------------------------------
template <typename InT>
inline BSparsePhysicalLayout<InT> ComputeASparsePhysicalLayout(
    const std::vector<InT>& hA, const std::vector<InT>& hB,
    int32_t m, int32_t k, int32_t n, aclsparseOrder_t order,
    bool transA, bool transB, int32_t num_batches, int64_t batch_stride,
    const sparse_test::MatmulLayoutOpts& opts)
{
    using namespace sparse_test;
    BSparsePhysicalLayout<InT> lay;
    // (b) A structured: logical (m,k) -> descriptor (order, ldSparse) layout
    // (PreparePhysicalA internally does the natural compact + (order, ld)
    // assembly).
    lay.hA_phys = PreparePhysicalA<InT>(hA, m, k, transA, order, lay.aLd,
        num_batches, batch_stride, opts.ldSparse);
    lay.physRowsA = transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    lay.physColsA = transA ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    // B dense：物理 (k,n)/(n,k)（transB），(denseOrder, ldDense) 布局
    lay.hB_phys = PreparePhysicalB<InT>(hB, k, n, transB,
        num_batches, batch_stride);
    lay.bPhysRows = transB ? static_cast<int64_t>(n) : static_cast<int64_t>(k);
    lay.bPhysCols = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n);
    lay.bLdOut = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n);
    if (opts.ldDense > 0 || opts.denseOrder == ACL_SPARSE_ORDER_COL) {
        lay.bLdOut = (opts.ldDense > 0) ? opts.ldDense : lay.bPhysRows;  // COL 紧凑 ld=physRows
        lay.hB_phys = PackBatchStridedPhysical<InT>(lay.hB_phys,
            static_cast<int32_t>(lay.bPhysRows), static_cast<int32_t>(lay.bPhysCols),
            opts.denseOrder, lay.bLdOut, num_batches, batch_stride);
    }
    // A structured 物理 span（order 感知，同 ComputeBSparsePhysicalLayout 的
    // aPhysSize 范式）；B dense 物理 span（denseOrder 感知）
    lay.aPhysSize = (order == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(lay.physColsA) * static_cast<size_t>(lay.aLd)
        : static_cast<size_t>(lay.physRowsA) * static_cast<size_t>(lay.aLd);
    lay.bPhysSize = (opts.denseOrder == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(lay.bPhysCols) * static_cast<size_t>(lay.bLdOut)
        : static_cast<size_t>(lay.bPhysRows) * static_cast<size_t>(lay.bLdOut);
    return lay;
}

// AllocBSparseChainBuffers：B-sparse 链设备缓冲段（双侧 batch 总量 + dA/dB
// 拷贝 + C 的 (order,ld) 物理 pack + dC 拷贝 + dD 分配）。
template <typename InT, typename OutT>
inline void AllocBSparseChainBuffers(MatmulBSparseCtx& ctx,
    const std::vector<InT>& hC, const BSparsePhysicalLayout<InT>& lay,
    int32_t m, int32_t n, int32_t num_batches, int64_t batch_stride,
    const sparse_test::MatmulLayoutOpts& opts, std::vector<InT>* hCPhysOut)
{
    using namespace sparse_test;
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize;
    constexpr size_t kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    // Batch: each matrix buffer is num_batches * batch_stride elements.
    const size_t totalAElems = BatchTotalElems(lay.aPhysSize, num_batches, batch_stride);
    const size_t totalBElems = BatchTotalElems(lay.bPhysSize, num_batches, batch_stride);
    const size_t totalCElems = BatchTotalElems(ctx.cdSpan, num_batches, batch_stride);
    ctx.dA = DeviceBuffer::copyFrom(lay.hA_phys.data(), totalAElems * kInElt);
    ctx.dB = DeviceBuffer::copyFrom(lay.hB_phys.data(), totalBElems * kInElt);
    std::vector<InT> hCPhys = PackBatchStridedPhysical<InT>(hC, m, n,
        opts.cdOrder, CdLdOf(m, n, opts), num_batches, batch_stride);
    ctx.dC = DeviceBuffer::copyFrom(hCPhys.data(), totalCElems * kInElt);
    ctx.dD = DeviceBuffer::alloc(totalCElems * kOutElt);
    if (hCPhysOut != nullptr) { *hCPhysOut = std::move(hCPhys); }
}

// -----------------------------------------------------------------------------
// PrepareBSparseChainContext: compute physical layout + allocate device buffers
// + create B-sparse descriptors. Extracted from RunMatmulBSparseChainWithTrans.
// opts 尾参：ldSparse 覆盖 B structured 的 ld；ldDense/denseOrder 覆盖
// A dense 的布局；cdOrder/ldCd 覆盖 C/D（默认 = 既有紧凑行为）。
// 物理布局计算段（ComputeBSparsePhysicalLayout）与缓冲段
// （AllocBSparseChainBuffers）见上方。
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline void PrepareBSparseChainContext(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    aclsparseOrder_t order, aclsparseOperation_t opA, aclsparseOperation_t opB,
    int32_t num_batches, int64_t batch_stride, MatmulNpuResult& result,
    MatmulBSparseCtx& ctx,
    const sparse_test::MatmulLayoutOpts& opts = sparse_test::MatmulLayoutOpts{},
    std::vector<InT>* hCPhysOut = nullptr)
{
    using namespace sparse_test;
    constexpr aclDataType kInDtype = NpuDtypeTrait<InT>::kAclDtype;
    constexpr aclDataType kOutDtype = NpuDtypeTrait<OutT>::kAclDtype;
    constexpr aclsparseComputeType_t kComputeType = MatmulComputeTrait<InT>::kComputeType;
    InitChainCtxBase(ctx, num_batches, batch_stride, opts, m, n);
    const bool transA = (opA == ACL_SPARSE_OP_TRANSPOSE);
    const bool transB = (opB == ACL_SPARSE_OP_TRANSPOSE);
    ctx.kn = static_cast<size_t>(k) * n;
    BSparsePhysicalLayout<InT> lay = ComputeBSparsePhysicalLayout<InT>(
        hA, hB, m, k, n, order, transA, transB, num_batches, batch_stride, opts);
    AllocBSparseChainBuffers<InT, OutT>(ctx, hC, lay, m, n, num_batches, batch_stride,
                                        opts, hCPhysOut);
    ctx.matA = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), lay.physRowsA,
        lay.physColsA, lay.aLd, 16, kInDtype,
        (opts.ldDense > 0 || opts.denseOrder == ACL_SPARSE_ORDER_COL)
            ? opts.denseOrder : ACL_SPARSE_ORDER_ROW);
    ctx.matB = std::make_unique<SparseLtMatDescGuard>(ctx.handle.get(), lay.bPhysRows,
        lay.bPhysCols, lay.bLdOut, 16, kInDtype, order, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    const int64_t cdLd = CdLdOf(m, n, opts);
    ctx.matC = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, cdLd,
        16, kInDtype, opts.cdOrder);
    ctx.matD = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, cdLd,
        16, kOutDtype, opts.cdOrder);
    // Set batch attributes on all four matrix descriptors BEFORE MatmulDescriptorInit.
    SetBatchAttrsOnAllMatDescs(ctx.handle.get(), &ctx.matA->get(), &ctx.matB->get(),
        &ctx.matC->get(), &ctx.matD->get(),
        num_batches, batch_stride, result);
    ctx.matmulDesc = std::make_unique<SparseLtMatmulDescGuard>(ctx.handle.get(), opA, opB,
        ctx.matA->get(), ctx.matB->get(), ctx.matC->get(), ctx.matD->get(), kComputeType);
}

// -----------------------------------------------------------------------------
// RunBSparsePruneAndTranspose: prune B for B-sparse path.
// (b) the prune output keeps the B descriptor (order, ld) layout: matmul
// consumes that layout directly (cube ldB + isColView(transB) transpose
// flag), while golden consumption goes through the readback unpack into
// logical (k,n) row-major (CopyPrunedToHost). With transB=T the prune
// kernel reads physical (n,k) and writes back layout-preserving; with
// transB=F the physical layout is already (k,n) as-is — neither side needs
// a host transpose.
// -----------------------------------------------------------------------------
template <typename InT>
inline void RunBSparsePruneAndTranspose(MatmulNpuResult& result,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulDescriptor_t matmulDesc,
    sparse_test::DeviceBuffer& dB, sparse_test::DeviceBuffer& dBPruned,
    size_t kInElt,
    aclsparseLtPruneAlg_t pruneAlg, void*& matBPtr,
    int32_t numBatches = 1, int64_t batchStride = 0)
{
    using namespace sparse_test;
    aclrtStream stream = 0;
    const size_t totalElems = PruneOutCapacityElems(matmulDesc, numBatches, batchStride);
    dBPruned = DeviceBuffer::alloc(totalElems * kInElt);
    // prune 单调用化：描述符携带 numBatches/batchStride，一次调用处理
    // 全 batch（传 batch0 基址；切片循环会使末切片基址越界，禁止切片）。
    result.pruneRet = aclsparseLtSpMMAPrune(
        handle, &matmulDesc, dB.get(), dBPruned.get(), pruneAlg, stream);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return; }
    result.syncRet = aclrtSynchronizeStream(stream);
    if (result.syncRet != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return;
    }
    matBPtr = dBPruned.get();
    // 回读 + 解包：物理 (order,ld) batch-strided → 逻辑 (k,n) batch-strided
    //（golden 消费契约不变）。
    if (!CopyPrunedToHost(result, matmulDesc, dBPruned.get(), totalElems, kInElt,
                          numBatches, batchStride)) { return; }
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
    MatmulBSparseCtx ctx;
    PrepareBSparseChainContext<InT, OutT>(args.m, args.k, args.n,
                                          args.hA, args.hB, args.hC,
                                          args.order, args.opA, args.opB,
                                          args.numBatches, args.batchStride,
                                          result, ctx, args.layout);
    ChainSetupResult setup;
    if (!SetupVecEpiPlanWorkspace<InT, OutT>(ctx.handle, *ctx.matmulDesc, args, args.m,
                                                result, setup)) { return result; }
    void* matBPtr = nullptr;
    DeviceBuffer dBPruned;
    if (args.layout.directFeed) {
        // (b) pre-pruned direct feed (B-sparse): CPU golden logical prune of
        // B (k,n) -> (order, ldSparse) physical scatter fed to matmul
        // directly (npuPrunedA is left empty so golden takes the CPU prune
        // path, consuming the same data as the NPU).
        std::vector<InT> hPrunedLogical;
        SpMMAPruneGolden<InT>(args.hB, hPrunedLogical, args.k, args.n, 0, 1,
                              args.opB == ACL_SPARSE_OP_TRANSPOSE,
                              args.order == ACL_SPARSE_ORDER_ROW,
                              args.pruneAlg == ACLSPARSELT_PRUNE_SPMMA_TILE
                                  ? "TILE" : "STRIP");
        int64_t feedLd = 0;
        const std::vector<InT> hPrunedPhys = PreparePhysicalA<InT>(
            hPrunedLogical, args.k, args.n,
            args.opB == ACL_SPARSE_OP_TRANSPOSE, args.order, feedLd,
            args.numBatches, args.batchStride, args.layout.ldSparse);
        dBPruned = DeviceBuffer::copyFrom(hPrunedPhys.data(),
                                          hPrunedPhys.size() * kInElt);
        matBPtr = dBPruned.get();
        // npuPrunedA = CPU logical prune result (batch-strided) — the golden
        // WithNpuPrune path consumes the same data (the NPU side is fed via
        // the physical scatter).
        FeedDirectPrunedToResult(result, hPrunedLogical, args.numBatches,
                                 args.batchStride);
    } else {
        // The prune output keeps the B descriptor (order, ld) layout
        // (readback unpacks into logical for golden)
        RunBSparsePruneAndTranspose<InT>(result, ctx.handle.get(), ctx.matmulDesc->get(),
            ctx.dB, dBPruned, kInElt, args.pruneAlg, matBPtr,
            args.numBatches, args.batchStride);
        if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }
    }
    // 回读尺寸 = C/D 物理 span（batch 布局含 stride padding）
    const size_t totalOutElems = BatchTotalElems(ctx.cdSpan, ctx.numBatches, ctx.batchStride);
    ExecMatmulAndCopyResult<OutT>(result, ctx.handle.get(), setup.plan->cptr(),
        ctx.dA.get(), matBPtr, ctx.dC.get(), ctx.dD.get(), setup.wsPtr, setup.vecBufs, args.alpha, args.beta,
        args.alpha_vector_scaling, args.beta_vector_scaling, ctx.dD, args.hD, totalOutElems, kOutElt);
    return result;
}

#endif  // TEST_LTMATMUL_NPU_CHAIN_BSPARSE_H_
