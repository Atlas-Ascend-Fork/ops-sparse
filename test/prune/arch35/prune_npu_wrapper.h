/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_PRUNE_NPU_WRAPPER_H_
#define TEST_PRUNE_NPU_WRAPPER_H_

// =============================================================================
// Prune NPU wrapper (decoupled from alg_set_attribute / matmul test headers).
//
// Reuses RAII guards from matmulDescriptorInit_npu_wrapper.h (SparseLtHandleGuard,
// MatDescrGuard, MatmulDescrGuard). Provides NpuDtypeTrait and the
// PreparePhysicalA helper (extracted from the source repo's
// alg_set_attribute_npu_wrapper.h) so that prune tests do not pull in any
// alg_set_attribute or matmul code. EventGuard 复用 test/frame/event_guard.h
// 共享定义（prune / ltmatmul 计时共用，含 R6/P2 返回值检查加固语义）。
//
// Removed from the original:
//   - SparseLtAlgSelectionGuard / SparseLtPlanGuard (prune does not need
//     algSelection or plan)
//   - SetAlgAttributes (prune does not call AlgSetAttribute)
//   - algSetCfgRet / algSetSplitRet fields in PruneNpuResult
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparseLt.h"
#include "descriptor_manager.h"
#include "event_guard.h"  // shared EventGuard（prune/ltmatmul 计时共用）
#include "layout_utils.h"  // shared ld/order 布局基建（PackBatchStridedPhysical）
#include "matmulDescriptorInit_npu_wrapper.h"
#include "prune_test_util.h"
#include "transpose_utils.h"  // shared TransposeToRowMajor

namespace sparse_test {

// -----------------------------------------------------------------------------
// const-pointer accessors for the RAII guards reused from
// matmulDescriptorInit_npu_wrapper.h.
//
// The guard's ptr() returns a pointer to the underlying descr_ member
// (aclsparseLtMatDescriptor_t* = struct aclsparseLtMatDescriptor**). Several
// aclsparseLt APIs (e.g. aclsparseLtSpMMAPrune) require the const variant
// (aclsparseLtConstMatmulDescriptor_t* = const struct**). const_cast adds the
// missing const-qualification on the pointed-to struct, which is a valid,
// well-defined conversion.
// -----------------------------------------------------------------------------
inline aclsparseLtConstMatDescriptor_t* matDescCptr(MatDescrGuard& g) {
    return const_cast<aclsparseLtConstMatDescriptor_t*>(g.ptr());
}
inline aclsparseLtConstMatmulDescriptor_t* matmulDescCptr(MatmulDescrGuard& g) {
    return const_cast<aclsparseLtConstMatmulDescriptor_t*>(g.ptr());
}

// -----------------------------------------------------------------------------
// NpuDtypeTrait: map storage type T to aclDataType + element size.
// -----------------------------------------------------------------------------
template <typename T>
struct NpuDtypeTrait;

template <>
struct NpuDtypeTrait<float> {
    static constexpr aclDataType kAclDtype = ACL_FLOAT;
    static constexpr size_t kEltSize = sizeof(float);
};

template <>
struct NpuDtypeTrait<uint16_t> {
    static constexpr aclDataType kAclDtype = ACL_FLOAT16;
    static constexpr size_t kEltSize = sizeof(uint16_t);
};

template <>
struct NpuDtypeTrait<bf16_bits_t> {
    static constexpr aclDataType kAclDtype = ACL_BF16;
    static constexpr size_t kEltSize = sizeof(bf16_bits_t);
};

template <>
struct NpuDtypeTrait<int8_t> {
    static constexpr aclDataType kAclDtype = ACL_INT8;
    static constexpr size_t kEltSize = sizeof(int8_t);
};

// EventGuard 复用 test/frame/event_guard.h 共享定义（见上方 include；
// 原 11 行本地拷贝与 ltmatmul_test_utils.h 重复，codecheck 重复代码清零）。

// -----------------------------------------------------------------------------
// Physical matrix preparation (transpose logic for TRANSPOSE+ROW order).
// TransposeToRowMajor is provided by the shared header transpose_utils.h.
// -----------------------------------------------------------------------------

template <typename T>
inline std::vector<T> PreparePhysicalA(const std::vector<T>& hA,
    int32_t m, int32_t k, bool transA, aclsparseOrder_t order, int64_t& aLd)
{
    if (!transA) {
        // For NON_TRANSPOSE, the prune kernel always reads the
        // matrix as row-major (m, k) with stride ld, regardless of order.
        // The order field only selects the pruning direction (alongRow vs
        // alongCol), not the data layout. Setting aLd=k (the actual row-major
        // stride) ensures correct data access when m != k.
        aLd = static_cast<int64_t>(k);
        return hA;
    }
    aLd = (order == ACL_SPARSE_ORDER_COL)
        ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    if (order == ACL_SPARSE_ORDER_ROW) {
        return TransposeToRowMajor<T>(hA, m, k, aLd);
    }
    return hA;
}

}  // namespace sparse_test

// -----------------------------------------------------------------------------
// Prune NPU execution result (no algSet fields — prune is independent of
// algorithm selection).
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// [batch] 铺设 nb×stride 全量 host 缓冲（batch b 占 [b*stride, b*stride+span)，
// 其余 padding 置零）；四 mat 描述符设 NUM_BATCHES/BATCH_STRIDE（一致性校验
// 要求在 MatmulDescriptorInit 前设置）。输入为相同单份数据——验证输出 batch
// 布局推进与逐 batch 参考一致性（batch 正向基线）。
// -----------------------------------------------------------------------------
template <typename T>
inline std::vector<T> LayOutBatchStrided(const std::vector<T>& phys,
    int32_t nb, int64_t stride)
{
    if (nb <= 1) { return phys; }
    std::vector<T> out(static_cast<size_t>(nb) * static_cast<size_t>(stride), T{0});
    for (int32_t b = 0; b < nb; ++b) {
        std::copy(phys.begin(), phys.end(),
                  out.begin() + static_cast<size_t>(b) * static_cast<size_t>(stride));
    }
    return out;
}

inline void SetBatchAttrsOnMat(aclsparseLtConstHandle_t handle,
    aclsparseLtMatDescriptor_t* matDesc, int32_t nb, int64_t stride)
{
    if (nb <= 1) { return; }
    aclsparseLtMatDescSetAttribute(handle, matDesc,
        ACLSPARSELT_MAT_NUM_BATCHES, &nb, sizeof(int32_t));
    aclsparseLtMatDescSetAttribute(handle, matDesc,
        ACLSPARSELT_MAT_BATCH_STRIDE, &stride, sizeof(int64_t));
}

struct PruneNpuResult {
    aclsparseStatus_t pruneRet = ACL_SPARSE_STATUS_SUCCESS;
    size_t workspaceSize = 0;
    double npuMs = 0.0;
};

// Common setup for PruneNpu / PruneNpuBSparse: dtype traits + result holder +
// suppression of unused CSV-signature params (alg_config_id, split_k).
template <typename T>
struct PruneNpuSetup {
    static constexpr aclDataType kDtype = sparse_test::NpuDtypeTrait<T>::kAclDtype;
    static constexpr size_t kElt = sparse_test::NpuDtypeTrait<T>::kEltSize;
    static constexpr aclsparseComputeType_t kComputeType =
        std::is_same_v<T, int8_t> ? ACL_SPARSE_COMPUTE_32I : ACL_SPARSE_COMPUTE_32F;
    PruneNpuResult result;
    explicit PruneNpuSetup(int32_t alg_config_id, int32_t split_k) : result{} {
        (void)alg_config_id;
        (void)split_k;
    }
};

inline bool RunPruneExecution(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t matmulDesc,
    void* dA, void* dAPruned,
    aclsparseLtPruneAlg_t pruneAlg,
    aclrtStream stream,
    PruneNpuResult& result)
{
    using sparse_test::EventGuard;

    result.workspaceSize = 0;

    EventGuard evStart, evStop;
    evStart.record(stream);

    result.pruneRet = aclsparseLtSpMMAPrune(
        handle, &matmulDesc, dA, dAPruned,
        pruneAlg, stream);

    evStop.record(stream);
    auto syncRet = aclrtSynchronizeStream(stream);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] SpMMAPrune failed: " << result.pruneRet << std::endl;
        return false;
    }
    if (syncRet != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed: " << syncRet << std::endl;
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }

    result.npuMs = static_cast<double>(EventGuard::elapsedMs(evStart, evStop));
    return true;
}

// -----------------------------------------------------------------------------
// PruneDescrGroup：剪枝链描述符组（handle + matA~matD + matmulDescr）。
// RAII guards 随组存活（matmulDescr 引用其指针，析构序 md → mats → handle
// 与原函数内局部声明序一致）。Init*Descrs 助手自 PruneNpu / PruneNpuBSparse
// 的描述符创建段提取（控制函数 NBNC；codecheck 超大函数清零）。
// -----------------------------------------------------------------------------
struct PruneDescrGroup {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::MatDescrGuard matA;
    sparse_test::MatDescrGuard matB;
    sparse_test::MatDescrGuard matC;
    sparse_test::MatDescrGuard matD;
    sparse_test::MatmulDescrGuard md;
};

// A-sparse 描述符组初始化：matA structured（物理 rows/cols/ld）+ matB/C/D
// dense 占位（N=kPruneN 列；INT8 的 matB order 约束见内注）+ batch 属性
// （一致性校验要求 MatmulDescriptorInit 前设置于 4 个 mat 描述符）+
// matmulDescr（opA × N）。
template <typename T>
inline void InitPruneASparseDescrs(PruneDescrGroup& g,
    int64_t physRowsA, int64_t physColsA, int64_t aLd, int32_t m,
    aclsparseOrder_t order, aclsparseOperation_t opA, bool transA,
    bool isBatch, int32_t numBatches, int64_t batchStride)
{
    using namespace sparse_test;
    constexpr int32_t kPruneN = 128;
    constexpr aclDataType kDtype = NpuDtypeTrait<T>::kAclDtype;
    g.matA.initStructured(g.handle.ptr(), physRowsA, physColsA, aLd, 16, kDtype,
                          order, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    const int64_t logicalK = transA ? physRowsA : physColsA;
    // INT8 (low-precision) has op/layout combination constraints in
    // MatmulDescriptorInit: ROW+ROW requires N+T, but the wrapper uses opB=N.
    // Setting matB order=COL satisfies ROW+COL -> N+N (and COL+COL -> T+N),
    // which is valid for all INT8 A-sparse op/order combinations.
    // matB is a dense placeholder; its order does not affect prune results.
    const aclsparseOrder_t matBOrder = std::is_same_v<T, int8_t>
        ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW;
    g.matB.initDense(g.handle.ptr(), logicalK, kPruneN, kPruneN, 16, kDtype,
                     matBOrder);
    g.matC.initDense(g.handle.ptr(), m, kPruneN, kPruneN, 16, kDtype,
                     ACL_SPARSE_ORDER_ROW);
    g.matD.initDense(g.handle.ptr(), m, kPruneN, kPruneN, 16, kDtype,
                     ACL_SPARSE_ORDER_ROW);
    if (isBatch) {
        SetBatchAttrsOnMat(g.handle.ptr(), g.matA.ptr(), numBatches, batchStride);
        SetBatchAttrsOnMat(g.handle.ptr(), g.matB.ptr(), numBatches, batchStride);
        SetBatchAttrsOnMat(g.handle.ptr(), g.matC.ptr(), numBatches, batchStride);
        SetBatchAttrsOnMat(g.handle.ptr(), g.matD.ptr(), numBatches, batchStride);
    }
    g.md.init(g.handle.ptr(), opA, ACL_SPARSE_OP_NON_TRANSPOSE,
              g.matA.ptr(), g.matB.ptr(), g.matC.ptr(), g.matD.ptr(),
              std::is_same_v<T, int8_t> ? ACL_SPARSE_COMPUTE_32I
                                        : ACL_SPARSE_COMPUTE_32F);
}

// B-sparse 描述符组初始化：matA dense 占位（方阵 aRows×aCols，使 md->k ==
// B rows）+ matB structured（物理 rows/cols/ld）+ matC/matD dense 占位
// （mB×kB，满足 md->m=md->k=mB, md->n=kB）+ batch 属性 + matmulDescr（N × opB）。
template <typename T>
inline void InitPruneBSparseDescrs(PruneDescrGroup& g,
    int64_t aRows, int64_t aCols, int64_t aLd,
    int64_t bPhysRows, int64_t bPhysCols, int64_t bLd,
    int32_t mB, int32_t kB, aclsparseOrder_t order, aclsparseOperation_t opB,
    bool isBatch, int32_t numBatches, int64_t batchStride)
{
    using namespace sparse_test;
    constexpr aclDataType kDtype = NpuDtypeTrait<T>::kAclDtype;
    // matA: dense placeholder (sparsity=NONE via DenseDescriptorInit).
    g.matA.initDense(g.handle.ptr(), aRows, aCols, aLd, 16, kDtype,
                     ACL_SPARSE_ORDER_ROW);
    // matB: structured (sparsity=50_PERCENT).
    g.matB.initStructured(g.handle.ptr(), bPhysRows, bPhysCols, bLd, 16, kDtype,
                          order, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    // matC/matD: dense placeholder (mB, kB) — shapes satisfy md->m=md->k=mB, md->n=kB.
    g.matC.initDense(g.handle.ptr(), mB, kB, kB, 16, kDtype,
                     ACL_SPARSE_ORDER_ROW);
    g.matD.initDense(g.handle.ptr(), mB, kB, kB, 16, kDtype,
                     ACL_SPARSE_ORDER_ROW);
    if (isBatch) {
        SetBatchAttrsOnMat(g.handle.ptr(), g.matA.ptr(), numBatches, batchStride);
        SetBatchAttrsOnMat(g.handle.ptr(), g.matB.ptr(), numBatches, batchStride);
        SetBatchAttrsOnMat(g.handle.ptr(), g.matC.ptr(), numBatches, batchStride);
        SetBatchAttrsOnMat(g.handle.ptr(), g.matD.ptr(), numBatches, batchStride);
    }
    g.md.init(g.handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE, opB,
              g.matA.ptr(), g.matB.ptr(), g.matC.ptr(), g.matD.ptr(),
              std::is_same_v<T, int8_t> ? ACL_SPARSE_COMPUTE_32I
                                        : ACL_SPARSE_COMPUTE_32F);
}

// PruneNpu with transA support.
// When transA=true:
//   ROW order: physical A is (k, m) row-major. We transpose hA from (m,k) to (k,m).
//   COL order: physical A is (k, m) col-major = (m,k) row-major with stride ld=k. No transpose.
// matA descriptor is created with rows=k, cols=m (physical dimensions).
// pruneAlg: ACLSPARSELT_PRUNE_SPMMA_STRIP or ACLSPARSELT_PRUNE_SPMMA_TILE.
// alg_config_id / split_k are accepted for CSV signature compatibility but
// are NOT used by prune (prune is independent of algorithm selection).
// ld 尾参：structured A 的 ld 覆盖（0 = 紧凑 = 既有行为；>min 时
// 输入按 (order, ld) padded 物理布局摆放，pad 置零——prune 以 row*ld 读 padded
// 矩阵，输出恒紧凑 (m,k) 行主序）。
template <typename T>
inline PruneNpuResult PruneNpu(
    aclrtStream stream,
    const std::vector<T>& hA, std::vector<T>& hAPruned,
    int32_t m, int32_t k,
    int32_t alg_config_id, int32_t split_k,
    aclsparseOperation_t opA, aclsparseOrder_t order,
    aclsparseLtPruneAlg_t pruneAlg = ACLSPARSELT_PRUNE_SPMMA_STRIP,
    int32_t numBatches = 1, int64_t batchStride = 0,
    int64_t ld = 0)
{
    using namespace sparse_test;
    PruneNpuSetup<T> ctx(alg_config_id, split_k);
    auto& result = ctx.result;

    const bool transA = (opA == ACL_SPARSE_OP_TRANSPOSE);

    int64_t aLd;
    std::vector<T> hA_phys = PreparePhysicalA<T>(hA, m, k, transA, order, aLd);

    const int64_t physRowsA = transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    const int64_t physColsA = transA ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    if (ld > 0) {
        // padded 输入：物理重排到 (order, ld) 布局（pad 置零）
        aLd = ld;
        hA_phys = PackBatchStridedPhysical<T>(hA_phys,
            static_cast<int32_t>(physRowsA), static_cast<int32_t>(physColsA),
            order, aLd, numBatches, batchStride);
    }
    const size_t mk = static_cast<size_t>(m) * k;
    // For NON_TRANSPOSE, data is always row-major (m, k) with stride
    // aLd=k regardless of order (see PreparePhysicalA). Use the row-major size
    // formula (physRowsA * aLd) for NON_TRANSPOSE. The COL formula
    // (physColsA * aLd) is only valid for TRANSPOSE+COL where aLd=k and the
    // physical layout is column-major (k, m) with ld=k.
    const size_t aPhysSize = (transA && order == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(physColsA) * static_cast<size_t>(aLd)
        : static_cast<size_t>(physRowsA) * static_cast<size_t>(aLd);

    // batch：铺设 nb×stride 全量（物理变换后的单份复制）；span 取物理布局跨度
    const bool isBatch = (numBatches > 1 && batchStride > 0);
    const size_t totalElems = isBatch
        ? static_cast<size_t>(numBatches) * static_cast<size_t>(batchStride)
        : aPhysSize;
    const std::vector<T> hA_laid = isBatch
        ? LayOutBatchStrided<T>(hA_phys, numBatches, batchStride) : hA_phys;
    DeviceBuffer dA = DeviceBuffer::copyFrom(hA_laid.data(), totalElems * ctx.kElt);
    DeviceBuffer dAPruned = DeviceBuffer::alloc(totalElems * ctx.kElt);

    // 描述符组（matA structured + matB/C/D dense 占位 + batch 属性 + matmulDesc）
    PruneDescrGroup descrs;
    InitPruneASparseDescrs<T>(descrs, physRowsA, physColsA, aLd, m,
                              order, opA, transA, isBatch, numBatches, batchStride);

    if (!RunPruneExecution(descrs.handle.ptr(), descrs.md.get(),
                           dA.get(), dAPruned.get(), pruneAlg, stream, result)) {
        return result;
    }

    hAPruned.resize(totalElems);
    dAPruned.copyToHost(hAPruned.data(), totalElems * ctx.kElt);
    return result;
}

// [B-SPARSE] PruneNpu for B-sparse: B is the structured (2:4) matrix.
// Params mB/kB denote B's (rows, cols) = (logical k, logical n).
// matA is a dense placeholder so MatmulDescriptorInit can derive m/k/n;
// its shape is chosen so md->k == mB (B rows). matmulDescr implicitly
// detects matB as structured (sparsity=50_PERCENT) and prunes B.
// alg_config_id / split_k are accepted for CSV signature compatibility but
// are NOT used by prune.
template <typename T>
inline PruneNpuResult PruneNpuBSparse(
    aclrtStream stream,
    const std::vector<T>& hB, std::vector<T>& hBPruned,
    int32_t mB, int32_t kB,
    int32_t alg_config_id, int32_t split_k,
    aclsparseOperation_t opB, aclsparseOrder_t order,
    aclsparseLtPruneAlg_t pruneAlg = ACLSPARSELT_PRUNE_SPMMA_STRIP,
    int32_t numBatches = 1, int64_t batchStride = 0)
{
    using namespace sparse_test;
    PruneNpuSetup<T> ctx(alg_config_id, split_k);
    auto& result = ctx.result;

    const bool transB = (opB == ACL_SPARSE_OP_TRANSPOSE);

    // matA dense placeholder: rows=mB, cols=mB (square, so md->k = matA->cols = mB = B rows).
    const int64_t aRows = static_cast<int64_t>(mB);
    const int64_t aCols = static_cast<int64_t>(mB);
    const int64_t aLd = aCols;

    // B physical layout: same logic as PreparePhysicalA for A-sparse
    // (transB + order determine whether to physically transpose hB).
    int64_t bLd = 0;
    std::vector<T> hB_phys = PreparePhysicalA<T>(hB, mB, kB, transB, order, bLd);
    const int64_t bPhysRows = transB ? static_cast<int64_t>(kB) : static_cast<int64_t>(mB);
    const int64_t bPhysCols = transB ? static_cast<int64_t>(mB) : static_cast<int64_t>(kB);
    const size_t mk = static_cast<size_t>(mB) * static_cast<size_t>(kB);
    // Same row-major size fix as PruneNpu (see PreparePhysicalA).
    const size_t bPhysSize = (transB && order == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(bPhysCols) * static_cast<size_t>(bLd)
        : static_cast<size_t>(bPhysRows) * static_cast<size_t>(bLd);

    // batch：铺设 nb×stride 全量（物理变换后的单份复制）
    const bool isBatch = (numBatches > 1 && batchStride > 0);
    const size_t totalElems = isBatch
        ? static_cast<size_t>(numBatches) * static_cast<size_t>(batchStride)
        : bPhysSize;
    const std::vector<T> hB_laid = isBatch
        ? LayOutBatchStrided<T>(hB_phys, numBatches, batchStride) : hB_phys;
    DeviceBuffer dB = DeviceBuffer::copyFrom(hB_laid.data(), totalElems * ctx.kElt);
    DeviceBuffer dBPruned = DeviceBuffer::alloc(totalElems * ctx.kElt);

    // 描述符组（matA dense 占位 + matB structured + matC/matD dense 占位 +
    // batch 属性 + matmulDesc）
    PruneDescrGroup descrs;
    InitPruneBSparseDescrs<T>(descrs, aRows, aCols, aLd, bPhysRows, bPhysCols,
                              bLd, mB, kB, order, opB, isBatch, numBatches,
                              batchStride);

    if (!RunPruneExecution(descrs.handle.ptr(), descrs.md.get(),
                           dB.get(), dBPruned.get(), pruneAlg, stream, result)) {
        return result;
    }

    hBPruned.resize(totalElems);
    dBPruned.copyToHost(hBPruned.data(), totalElems * ctx.kElt);
    return result;
}

#endif  // TEST_PRUNE_NPU_WRAPPER_H_
