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

#ifndef TEST_MATMUL_NPU_WRAPPER_H_
#define TEST_MATMUL_NPU_WRAPPER_H_

#include <cfloat>

#include "ltmatmul_test_utils.h"  // NpuDtypeTrait, RAII guards, AlgSetAttributeNpu chain

// =============================================================================
// LT_TEST_NPU_EPILOGUE_ENABLED: compile-time gate for bias/activation
// SetAttribute calls on the NPU side.
//
// The operator side (feat/bias-activation-epilogue) has extended
// aclsparseLtMatmulDescAttribute_t with BIAS_POINTER(2), BIAS_STRIDE(3),
// ACTIVATION_RELU(4), ACTIVATION_RELU_UPPERBOUND(5), ACTIVATION_RELU_THRESHOLD(6),
// ACTIVATION_GELU(7), ACTIVATION_GELU_SCALING(8). So epilogue (stage 1:
// bias + activation) is enabled by default.
//
// LT_TEST_NPU_BATCH_ENABLED: gate for stage-2 batch attributes (NUM_BATCHES,
// BATCH_STRIDE on MatDesc). The operator side has implemented batch support
// (aclsparseLtMatDescSetAttribute with ACLSPARSELT_MAT_NUM_BATCHES /
//  ACLSPARSELT_MAT_BATCH_STRIDE); batch cases now run the full NPU path.
// =============================================================================
#ifndef LT_TEST_NPU_EPILOGUE_ENABLED
#define LT_TEST_NPU_EPILOGUE_ENABLED 1
#endif
#ifndef LT_TEST_NPU_BATCH_ENABLED
#define LT_TEST_NPU_BATCH_ENABLED 1
#endif

// =============================================================================
// Matmul NPU wrapper for the ltmatmul test binary.
//
// Provides:
//   - RunMatmulNpu: full matmul chain (4 dtype x 2 path, A-sparse / B-sparse / dense)
//     with built-in getAttr roundtrip verification
//
// Supports:
//   - 4 input dtypes: FP32(float), FP16(uint16_t), BF16(bf16_bits_t), INT8(int8_t)
//   - INT8 dual output: INT8(int8_t) or INT32(int32_t)
//   - 2 paths: sparse*dense (with prune) / dense*dense (no prune)
//   - A-sparse / B-sparse
//   - Transpose (transA, transB)
//   - split_k + split_k_mode (ONE_KERNEL / TWO_KERNELS)
//   - prune_alg (STRIP / TILE)
//   - AlgSetAttribute / AlgGetAttribute roundtrip
//
// Foundational types (traits, RAII guards, AlgSetAttributeNpu chain) live in
// ltmatmul_test_utils.h, included above.
// =============================================================================

// =============================================================================
// Matmul NPU result + chain functions (original ltmatmul_npu_wrapper.h).
// =============================================================================

// -----------------------------------------------------------------------------
// Matmul NPU result.
// -----------------------------------------------------------------------------
struct MatmulNpuResult {
    aclsparseStatus_t algSetCfgRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algSetSplitRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algSetSplitKModeRet = ACL_SPARSE_STATUS_SUCCESS;
    // getAttr roundtrip verification (setAttr -> getAttr readback).
    aclsparseStatus_t algGetCfgRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algGetSplitRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algGetSplitKModeRet = ACL_SPARSE_STATUS_SUCCESS;
    int32_t gotAlgConfigId = -1;
    int32_t gotSplitK = -1;
    int32_t gotSplitKMode = -1;
    aclsparseStatus_t descSetRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t pruneRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t wsRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t matmulRet = ACL_SPARSE_STATUS_SUCCESS;
    aclError syncRet = ACL_SUCCESS;
    aclError memcpyRet = ACL_SUCCESS;
    size_t workspaceSize = 0;
    double npuMs = 0.0;
    std::vector<int8_t> npuPrunedA;
    // Epilogue SetAttribute return codes (always SUCCESS when epilogue disabled
    // or LT_TEST_NPU_EPILOGUE_ENABLED==0).
    aclsparseStatus_t biasSetRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t actSetRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t batchSetRet = ACL_SPARSE_STATUS_SUCCESS;
    bool epilogueApplied = false;  // true iff NPU actually applied bias/activation
};

#include "ltmatmul_npu_epilogue.h"  // EpilogueBuffers, SetEpilogueAttrs

// -----------------------------------------------------------------------------
// Set alg attributes including split_k_mode (v2 extension).
// -----------------------------------------------------------------------------
inline bool SetMatmulAlgAttributes(aclsparseLtConstHandle_t handle,
                                    aclsparseLtMatmulAlgSelection_t algSel,
                                    int32_t algConfigId, int32_t splitK,
                                    int32_t splitKMode,
                                    MatmulNpuResult& result)
{
    result.algSetCfgRet = aclsparseLtMatmulAlgSetAttribute(
        handle, &algSel, ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
        &algConfigId, sizeof(int32_t));
    if (result.algSetCfgRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }

    result.algSetSplitRet = aclsparseLtMatmulAlgSetAttribute(
        handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K,
        &splitK, sizeof(int32_t));
    if (result.algSetSplitRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }

    result.algSetSplitKModeRet = aclsparseLtMatmulAlgSetAttribute(
        handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K_MODE,
        &splitKMode, sizeof(int32_t));
    if (result.algSetSplitKModeRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }

    return true;
}

// -----------------------------------------------------------------------------
// Vector scaling: set attributes on matmul descriptor + copy vectors to device.
// Extracted from RunMatmulChain / RunMatmulBSparseChainWithTrans
// to eliminate duplicate code.
// -----------------------------------------------------------------------------
struct VecScalingBuffers {
    sparse_test::DeviceBuffer dAlphaVec;
    sparse_test::DeviceBuffer dBetaVec;
};

inline VecScalingBuffers SetVecScalingAttrs(aclsparseLtConstHandle_t handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc, int32_t m,
    int32_t alpha_vector_scaling, int32_t beta_vector_scaling,
    const std::vector<float>* alphaVec, const std::vector<float>* betaVec,
    MatmulNpuResult& result)
{
    VecScalingBuffers bufs;
    // [bvec 单独] BETA_VECTOR_SCALING implies ALPHA_VECTOR_SCALING：启用 beta 向量
    // 会隐含启用 alpha 向量——alpha 未显式启用时也须设置属性并分配向量（标量值
    // 填充），否则算子把 host 标量指针当 device 向量读取。
    const bool alphaNeeded = (alpha_vector_scaling || beta_vector_scaling) != 0;
    if (alphaNeeded && alphaVec != nullptr) {
        int enable = 1;
        result.descSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
            ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &enable, sizeof(int));
        if (result.descSetRet != ACL_SPARSE_STATUS_SUCCESS) { return bufs; }
        bufs.dAlphaVec = sparse_test::DeviceBuffer::copyFrom(alphaVec->data(),
            static_cast<size_t>(m) * sizeof(float));
    }
    if (beta_vector_scaling && betaVec != nullptr) {
        int enable = 1;
        result.descSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
            ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING, &enable, sizeof(int));
        if (result.descSetRet != ACL_SPARSE_STATUS_SUCCESS) { return bufs; }
        bufs.dBetaVec = sparse_test::DeviceBuffer::copyFrom(betaVec->data(),
            static_cast<size_t>(m) * sizeof(float));
    }
    return bufs;
}

// -----------------------------------------------------------------------------
// SetBatchAttrsOnMatDesc: set NUM_BATCHES + BATCH_STRIDE on a matrix descriptor.
//
// Must be called BEFORE aclsparseLtMatmulDescriptorInit (which performs the
// A/B/C/D numBatches consistency check). All four matrices must share the same
// numBatches; batch_stride may differ per-matrix but the test uses a single
// stride from the CSV for all four.
//
// When LT_TEST_NPU_BATCH_ENABLED==0 or num_batches<=1, this is a no-op.
// -----------------------------------------------------------------------------
inline aclsparseStatus_t SetBatchAttrsOnMatDesc(aclsparseLtConstHandle_t handle,
    aclsparseLtMatDescriptor_t* matDesc, int32_t num_batches, int64_t batch_stride)
{
#if LT_TEST_NPU_BATCH_ENABLED
    if (num_batches <= 1) { return ACL_SPARSE_STATUS_SUCCESS; }
    aclsparseStatus_t st = aclsparseLtMatDescSetAttribute(handle, matDesc,
        ACLSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int32_t));
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    if (batch_stride > 0) {
        st = aclsparseLtMatDescSetAttribute(handle, matDesc,
            ACLSPARSELT_MAT_BATCH_STRIDE, &batch_stride, sizeof(int64_t));
    }
    return st;
#else
    (void)handle; (void)matDesc; (void)num_batches; (void)batch_stride;
    return ACL_SPARSE_STATUS_SUCCESS;
#endif
}

// Convenience: set batch attrs on all four matrix descriptors + record result.
inline void SetBatchAttrsOnAllMatDescs(aclsparseLtConstHandle_t handle,
    aclsparseLtMatDescriptor_t* matA, aclsparseLtMatDescriptor_t* matB,
    aclsparseLtMatDescriptor_t* matC, aclsparseLtMatDescriptor_t* matD,
    int32_t num_batches, int64_t batch_stride, MatmulNpuResult& result)
{
#if LT_TEST_NPU_BATCH_ENABLED
    if (num_batches <= 1) {
        result.batchSetRet = ACL_SPARSE_STATUS_SUCCESS;
        return;
    }
    aclsparseStatus_t st = ACL_SPARSE_STATUS_SUCCESS;
    st = SetBatchAttrsOnMatDesc(handle, matA, num_batches, batch_stride);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { result.batchSetRet = st; return; }
    st = SetBatchAttrsOnMatDesc(handle, matB, num_batches, batch_stride);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { result.batchSetRet = st; return; }
    st = SetBatchAttrsOnMatDesc(handle, matC, num_batches, batch_stride);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { result.batchSetRet = st; return; }
    st = SetBatchAttrsOnMatDesc(handle, matD, num_batches, batch_stride);
    result.batchSetRet = st;
#else
    (void)handle; (void)matA; (void)matB; (void)matC; (void)matD;
    (void)num_batches; (void)batch_stride;
    result.batchSetRet = ACL_SPARSE_STATUS_SUCCESS;
#endif
}

// -----------------------------------------------------------------------------
// BatchTotalElems: compute total element count for batch-strided buffers.
// When num_batches>1 and batch_stride>0, each matrix buffer is
// num_batches * batch_stride elements (with padding between batches).
// Otherwise, single-batch element count.
// -----------------------------------------------------------------------------
inline size_t BatchTotalElems(size_t singleBatchElems, int32_t num_batches, int64_t batch_stride)
{
    if (num_batches > 1 && batch_stride > 0) {
        return static_cast<size_t>(num_batches) * static_cast<size_t>(batch_stride);
    }
    return singleBatchElems;
}

// -----------------------------------------------------------------------------
// Execute matmul + sync + record timing + copy D back to host.
// Extracted from RunMatmulChain / RunMatmulBSparseChainWithTrans
// to eliminate duplicate code.
// -----------------------------------------------------------------------------
template <typename OutT>
inline void ExecMatmulAndCopyResult(MatmulNpuResult& result,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulPlan_t* plan,
    void* matAPtr, void* matBPtr, void* matC, void* matD, void* wsPtr,
    VecScalingBuffers& vecBufs, float alpha, float beta,
    int32_t alpha_vector_scaling, int32_t beta_vector_scaling,
    sparse_test::DeviceBuffer& dD, std::vector<OutT>& hD, size_t mn, size_t kOutElt)
{
    using namespace sparse_test;
    aclrtStream stream = 0;  // default stream
    float alphaF = alpha;
    float betaF = beta;
    const void* alphaPtr = (alpha_vector_scaling && vecBufs.dAlphaVec.get())
        ? vecBufs.dAlphaVec.get() : static_cast<const void*>(&alphaF);
    const void* betaPtr = (beta_vector_scaling && vecBufs.dBetaVec.get())
        ? vecBufs.dBetaVec.get() : static_cast<const void*>(&betaF);

    EventGuard evStart, evStop;
    evStart.record(stream);
    result.matmulRet = aclsparseLtMatmul(
        handle, plan, alphaPtr, matAPtr, matBPtr, betaPtr,
        matC, matD, wsPtr, &stream, 1);
    evStop.record(stream);
    auto syncRet = aclrtSynchronizeStream(stream);
    if (result.matmulRet != ACL_SPARSE_STATUS_SUCCESS) { return; }
    if (syncRet != ACL_SUCCESS) {
        result.matmulRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return;
    }
    result.npuMs = static_cast<double>(EventGuard::elapsedMs(evStart, evStop));
    hD.resize(mn);
    dD.copyToHost(hD.data(), mn * kOutElt);
}

#include "ltmatmul_npu_chain.h"  // chain runners (ctx, prepare, prune, RunMatmulChain/BSparse)

// -----------------------------------------------------------------------------
// Dispatch: select chain based on path (sparse A-sparse / sparse B-sparse / dense).
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline MatmulNpuResult RunMatmulNpu(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    std::vector<OutT>& hD,
    float alpha, float beta,
    int32_t algConfigId, int32_t splitK, int32_t splitKMode,
    bool isSparseA, bool isDensePath,
    aclsparseOrder_t order,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclsparseLtPruneAlg_t pruneAlg = ACLSPARSELT_PRUNE_SPMMA_STRIP,
    int32_t alpha_vector_scaling = 0,
    int32_t beta_vector_scaling = 0,
    const std::vector<float>* alphaVec = nullptr,
    const std::vector<float>* betaVec = nullptr,
    // Epilogue (backward-compatible defaults: all-off).
    int32_t biasEnabled = 0,
    int64_t biasStride = 0,
    int32_t activationType = 0,
    float reluUpperBound = FLT_MAX,
    float reluThreshold = 0.0f,
    float geluScaling = 1.0f,
    int32_t numBatches = 1,
    int64_t batchStride = 0,
    const std::vector<float>* biasVec = nullptr,
    // ld/order 布局覆盖（默认紧凑 = 既有行为）
    const sparse_test::MatmulLayoutOpts& layout = sparse_test::MatmulLayoutOpts{})
{
    MatmulRunArgs<InT, OutT> args{
        m, k, n, hA, hB, hC, hD,
        alpha, beta, algConfigId, splitK, splitKMode,
        order, opA, opB, pruneAlg,
        alpha_vector_scaling, beta_vector_scaling, alphaVec, betaVec
    };
    args.biasEnabled = biasEnabled;
    args.biasStride = biasStride;
    args.activationType = activationType;
    args.reluUpperBound = reluUpperBound;
    args.reluThreshold = reluThreshold;
    args.geluScaling = geluScaling;
    args.numBatches = numBatches;
    args.batchStride = batchStride;
    args.biasVec = biasVec;
    args.layout = layout;
    // bias dtype follows C dtype (FP16→FP16, BF16→BF16, INT8→FP32).
    args.biasEltSize = sparse_test::BiasDtypeTrait<InT>::kBiasEltSize;
    args.biasDtype = sparse_test::BiasDtypeTrait<InT>::kBiasAclDtype;
    if (isDensePath) {
        // Dense*dense: no prune, A=dense, B=dense.
        return RunMatmulChain<InT, OutT>(args, true, true);
    } else if (isSparseA) {
        // A-sparse: prune A, matmul(A_pruned, B).
        return RunMatmulChain<InT, OutT>(args, true, false);
    } else {
        // B-sparse: prune B, matmul(A, B_pruned). Use dedicated handler for transB support.
        return RunMatmulBSparseChainWithTrans<InT, OutT>(args);
    }
}

// -----------------------------------------------------------------------------
// [Int8CD 共享基建] INT8 A/B × C=D=CDT 独立链的 A/B 描述符 + 物理 B 段
// （ltmatmul RunMatmulNpuInt8CD 与 matmulSearch RunSearchChainInt8CD 共用，
// matmulSearch_npu_wrapper.h 经 include 链复用同一实体）。
// A/B 按侧（A-sparse：A structured + B dense；B-sparse：A dense + B
// structured，物理 (n,k)）；C/D 独立 kCD dtype 由调用方自建。
// 尾参（sparseOrder/lay/opts，默认 = 紧凑 ROW 基线）：structured 侧
// order、链路布局段的物理维度/ld、dense 侧 (denseOrder, ldDense) 覆盖。
// -----------------------------------------------------------------------------
struct Int8CDAbDescGroup {
    std::unique_ptr<sparse_test::SparseLtMatDescGuard> matAStruct;   // A-sparse 时有效
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matADense;  // B-sparse 时有效
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matBDense;  // A-sparse 时有效
    std::unique_ptr<sparse_test::SparseLtMatDescGuard> matBStruct;   // B-sparse 时有效
};

// transB=T 时逻辑 (k,n) B → 物理 (n,k) 行主序（prune 输入布局）。
inline std::vector<int8_t> TransposeInt8BIfNeeded(const std::vector<int8_t>& hB,
    int32_t k, int32_t n, bool transB)
{
    return transB
        ? sparse_test::TransposeToRowMajor<int8_t>(hB, k, n, static_cast<int64_t>(k)) : hB;
}

inline void MakeInt8CDAbDescriptors(sparse_test::SparseLtHandleGuard& handle,
    int32_t m, int32_t k, int32_t n, bool isSparseA, bool transB,
    Int8CDAbDescGroup& g,
    // structured 侧 order / 物理布局结果 / dense 侧布局覆盖
    //（默认 = 单 batch 紧凑 ROW 基线）
    aclsparseOrder_t sparseOrder = ACL_SPARSE_ORDER_ROW,
    const BSparsePhysicalLayout<int8_t>* lay = nullptr,
    const sparse_test::MatmulLayoutOpts& opts = sparse_test::MatmulLayoutOpts{})
{
    // 物理维度/ld：lay 非空取链路布局段结果；空则按紧凑基线计算
    //（transB 决定 B 物理，ld 紧凑，order 恒 ROW）
    const int64_t bPhysRows = (lay != nullptr) ? lay->bPhysRows
        : (transB ? static_cast<int64_t>(n) : static_cast<int64_t>(k));
    const int64_t bPhysCols = (lay != nullptr) ? lay->bPhysCols
        : (transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n));
    const int64_t bLd = (lay != nullptr) ? lay->bLdOut
        : (transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n));
    const int64_t aPhysRows = (lay != nullptr) ? lay->physRowsA : static_cast<int64_t>(m);
    const int64_t aPhysCols = (lay != nullptr) ? lay->physColsA : static_cast<int64_t>(k);
    const int64_t aLd = (lay != nullptr) ? lay->aLd : static_cast<int64_t>(k);
    // dense 侧 order：(ldDense/denseOrder) 覆盖时取 denseOrder，否则 ROW
    //（A-sparse → B dense；B-sparse → A dense）
    const aclsparseOrder_t denseOrder = (opts.ldDense > 0 ||
        opts.denseOrder == ACL_SPARSE_ORDER_COL)
        ? opts.denseOrder : ACL_SPARSE_ORDER_ROW;
    if (isSparseA) {
        g.matAStruct = std::make_unique<sparse_test::SparseLtMatDescGuard>(handle.get(),
            aPhysRows, aPhysCols, aLd, 16, ACL_INT8, sparseOrder,
            ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        g.matBDense = std::make_unique<sparse_test::SparseLtDnMatDescGuard>(handle.get(),
            bPhysRows, bPhysCols, bLd, 16, ACL_INT8, denseOrder);
    } else {
        // B-sparse：A dense（(order, ldSparse) 布局由链路段处理）、B structured（物理 (n,k)）
        g.matADense = std::make_unique<sparse_test::SparseLtDnMatDescGuard>(handle.get(),
            aPhysRows, aPhysCols, aLd, 16, ACL_INT8, denseOrder);
        g.matBStruct = std::make_unique<sparse_test::SparseLtMatDescGuard>(handle.get(),
            bPhysRows, bPhysCols, bLd, 16, ACL_INT8, sparseOrder,
            ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    }
}

// -----------------------------------------------------------------------------
// RunMatmulNpuInt8CD：INT8 A/B × C=D=CDT（CDT ∈ {int32_t,
// uint16_t, bf16_bits_t}）独立链——C/D 描述符独立 dtype（matmulSearch
// RunSearchChainInt8CD 同构，数值验证主力镜像回 ltmatmul）。transA/transB
// 透传；vec/epi 属性透传；单次 aclsparseLtMatmul。
// A/B 描述符段见 MakeInt8CDAbDescriptors，B-sparse 剪枝段见
// PruneInt8CDBSparse，铺设段见 PrepareInt8CDChainContext（物理布局 +
// 描述符 + batch 属性 + 缓冲）。
// 尾参 order/numBatches/batchStride/layout：batch（四矩阵 batch
// 属性 + batch-strided 铺设 + prune/回读按 batch 放大）与布局
// （ldSparse/ldDense/denseOrder/cdOrder/ldCd——A/B 物理布局段复用
// ComputeASparsePhysicalLayout / ComputeBSparsePhysicalLayout，C/D 接入
// (cdOrder, ldCd)）透传；默认值 = 单 batch 紧凑 ROW 基线。
// -----------------------------------------------------------------------------
// B-sparse 剪枝段：剪枝 dB（物理 (n,k)）→ dPruned（(k,n) 行主序输出）+
// 同步 + 回读 golden（prune 输出恒紧凑 (k,n)）。
// numBatches/batchStride 尾参：>1 时单调用化处理全 batch
//（描述符携带 batch 属性，prune 输出按 batchStride 逐段推进——与主链
// RunBSparsePruneAndTranspose 同范式）；kn 为 prune 缓冲的单 batch span 上界。
// 回读段（CopyPrunedToHost）与 RunPruneForASparse 共享；同步检查保持
// 本链内联判断（不记录 syncRet）。
inline bool PruneInt8CDBSparse(MatmulNpuResult& result,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulDescriptor_t* matmulDesc,
    sparse_test::DeviceBuffer& dB, sparse_test::DeviceBuffer& dPruned, size_t kn,
    aclsparseLtPruneAlg_t pruneAlg, aclrtStream stream,
    int32_t numBatches = 1, int64_t batchStride = 0)
{
    constexpr size_t kInElt = 1;  // int8
    const size_t totalElems = BatchTotalElems(kn, numBatches, batchStride);
    result.pruneRet = aclsparseLtSpMMAPrune(
        handle, matmulDesc, dB.get(), dPruned.get(), pruneAlg, stream);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }
    return CopyPrunedToHost(result, dPruned, totalElems * kInElt);
}

// [Int8CD 共享段] NPU 剪枝回读提取：A-sparse 取 A_pruned 前 m*k 元素；
// B-sparse 取 B_pruned 全量（(k,n) 行主序，剪枝已转置完成）——ltmatmul
// ComputeInt8CDCaseGolden 与 matmulSearch ComputeSearchGoldenInt8CD 共用。
inline std::vector<int8_t> ExtractInt8CDPruned(const MatmulNpuResult& npu,
    bool isSparseA, size_t mk)
{
    if (isSparseA) {
        return std::vector<int8_t>(npu.npuPrunedA.begin(), npu.npuPrunedA.begin() + mk);
    }
    return std::vector<int8_t>(npu.npuPrunedA.begin(), npu.npuPrunedA.end());
}

// vec/epi 属性段（RunMatmulNpuInt8CD / RunSearchChainInt8CD 共用；
// bias/act/batch 参数化覆盖两侧差异——ltmatmul 单 batch 基线传 0/1/0，
// matmulSearch 传 args 值）。INT8 超集链 C 为 FP 域 → bias FP32。
// 注意：epiBufs.dBias 由 BIAS_POINTER 属性持有 device 指针，matmul 执行期
// 读取——epiBufs 经输出参数交调用方持有，须存活至 matmul/exec 完成。
inline bool SetInt8CDVecEpiAttrs(sparse_test::SparseLtHandleGuard& handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc, int32_t m,
    int32_t alpha_vector_scaling, int32_t beta_vector_scaling,
    const std::vector<float>* alphaVec, const std::vector<float>* betaVec,
    int32_t biasEnabled, int64_t biasStride, int32_t activationType,
    float reluUb, float reluThr, float geluScale,
    int32_t numBatches, int64_t batchStride, const std::vector<float>* biasVec,
    VecScalingBuffers& vecBufs, EpilogueBuffers& epiBufs, MatmulNpuResult& result)
{
    vecBufs = SetVecScalingAttrs(handle.get(), matmulDesc, m,
        alpha_vector_scaling, beta_vector_scaling, alphaVec, betaVec, result);
    if (result.descSetRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    epiBufs = SetEpilogueAttrs(handle.get(), matmulDesc, m,
        biasEnabled, biasStride, activationType, reluUb, reluThr, geluScale,
        numBatches, batchStride, biasVec, result, sizeof(float), ACL_FLOAT);
    if (result.biasSetRet != ACL_SPARSE_STATUS_SUCCESS ||
        result.actSetRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    return true;
}

// 剪枝 + 输入指针选择段：A-sparse 经 RunPruneForASparse（dPruned 分配 +
// prune + npuPrunedA 回读，matAPtr ← A_pruned）；B-sparse 经
// PruneInt8CDBSparse（dPruned (k,n) 行主序，matBPtr ← B_pruned）。
// matAPtr/matBPtr：dense 侧取原输入，prune 成功后指向 dPruned。ok=false
// 时 result 已置错，调用方直接返回（dPruned 由返回结构持有，存活至
// 调用方函数尾）。
// numBatches/batchStride/bPhysSpan 尾参：B-sparse 侧 prune 缓冲
// 上界 = max(bPhysSpan, kn)（B 物理 span ≥ kn 恒成立，单 batch 紧凑时
// 相等——与主链 RunBSparsePruneAndTranspose 的 bPhysSpan 语义一致）。
struct Int8CDPrunePtrs {
    sparse_test::DeviceBuffer dPruned;
    void* matAPtr = nullptr;
    void* matBPtr = nullptr;
    bool ok = false;
};

inline Int8CDPrunePtrs PruneInt8CDSelectMatPtr(MatmulNpuResult& result,
    sparse_test::SparseLtHandleGuard& handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc,
    sparse_test::DeviceBuffer& dA, sparse_test::DeviceBuffer& dB,
    size_t mk, size_t kn,
    bool isSparseA, aclsparseLtPruneAlg_t pruneAlg, aclrtStream stream,
    int32_t numBatches = 1, int64_t batchStride = 0, size_t bPhysSpan = 0)
{
    constexpr size_t kInElt = 1;  // int8
    Int8CDPrunePtrs p;
    p.matAPtr = dA.get();
    p.matBPtr = dB.get();
    if (isSparseA) {
        p.ok = RunPruneForASparse(result, handle.get(), matmulDesc.get(), dA, p.dPruned,
                                  mk, kInElt, pruneAlg, p.matAPtr, numBatches, batchStride);
        return p;
    }
    const size_t pruneSpan = (bPhysSpan > kn) ? bPhysSpan : kn;
    p.dPruned = sparse_test::DeviceBuffer::alloc(
        BatchTotalElems(pruneSpan, numBatches, batchStride) * kInElt);
    if (!PruneInt8CDBSparse(result, handle.get(), matmulDesc.cptr(), dB, p.dPruned,
                            pruneSpan, pruneAlg, stream, numBatches, batchStride)) { return p; }
    p.matBPtr = p.dPruned.get();
    p.ok = true;
    return p;
}

// -----------------------------------------------------------------------------
// Int8CDChainCtx：RunMatmulNpuInt8CD 的聚合上下文（A/B/C/D 描述符 + 物理
// layout + 设备缓冲 + 回读尺寸）。RAII guard 无默认构造 → 按
// MatmulChainCtx / MatmulBSparseCtx 范式用 unique_ptr 持有。
// -----------------------------------------------------------------------------
template <typename CDT>
struct Int8CDChainCtx {
    sparse_test::SparseLtHandleGuard handle;
    BSparsePhysicalLayout<int8_t> lay;  // A/B 物理布局（bPhysSize 供 prune span）
    Int8CDAbDescGroup ab;               // A/B 按侧（A-sparse / B-sparse）
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matC;  // 独立 CDT dtype
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matD;
    std::unique_ptr<sparse_test::SparseLtMatmulDescGuard> matmulDesc;
    sparse_test::DeviceBuffer dA;
    sparse_test::DeviceBuffer dB;
    sparse_test::DeviceBuffer dC;
    sparse_test::DeviceBuffer dD;
    size_t totalCdElems = 0;  // C/D 回读尺寸（物理 span × batch 总量）
};

// -----------------------------------------------------------------------------
// PrepareInt8CDChainContext：Int8CD 链「A/B 物理布局 + A/B/C/D 描述符 +
// batch 属性 + 设备缓冲」铺设段（PrepareBSparseChainContext 同范式；
// ltmatmul RunMatmulNpuInt8CD 与 matmulSearch RunSearchChainInt8CD 共用）。
// batch（numBatches/batchStride）与布局（ldSparse/ldDense/denseOrder/
// cdOrder/ldCd）经参数透传；单 batch 紧凑 ROW 默认值 = 紧凑基线。
// hCPhysOut（可选输出）：pack 后的 C host 数据（batch-strided 物理
// 布局）——搜索链 rerun 重复上传同一物理布局用；nullptr = 调用方不需要。
// -----------------------------------------------------------------------------
template <typename CDT>
inline void PrepareInt8CDChainContext(Int8CDChainCtx<CDT>& ctx,
    int32_t m, int32_t k, int32_t n,
    const std::vector<int8_t>& hA, const std::vector<int8_t>& hB, const std::vector<CDT>& hC,
    aclsparseOperation_t opA, aclsparseOperation_t opB, bool isSparseA,
    aclsparseOrder_t order, int32_t numBatches, int64_t batchStride,
    const sparse_test::MatmulLayoutOpts& opts, MatmulNpuResult& result,
    std::vector<CDT>* hCPhysOut = nullptr)
{
    using namespace sparse_test;
    constexpr size_t kInElt = 1;  // int8
    constexpr size_t kOutElt = NpuDtypeTrait<CDT>::kEltSize;
    constexpr aclDataType kCD = NpuDtypeTrait<CDT>::kAclDtype;
    const bool transA = (opA == ACL_SPARSE_OP_TRANSPOSE);
    const bool transB = (opB == ACL_SPARSE_OP_TRANSPOSE);
    // ---- A/B 物理布局（batch-strided 铺设 + (order, ld) 覆盖；单 batch
    //      紧凑 ROW 时与既有 TransposeInt8BIfNeeded 基线逐位一致）----
    ctx.lay = isSparseA
        ? ComputeASparsePhysicalLayout<int8_t>(hA, hB, m, k, n, order, transA, transB,
                                               numBatches, batchStride, opts)
        : ComputeBSparsePhysicalLayout<int8_t>(hA, hB, m, k, n, order, transA, transB,
                                               numBatches, batchStride, opts);
    // ---- 描述符：A/B 按侧（共享 helper + 布局）；C/D 独立 CDT dtype
    //      接入 (cdOrder, ldCd) ----
    MakeInt8CDAbDescriptors(ctx.handle, m, k, n, isSparseA, transB, ctx.ab,
                            order, &ctx.lay, opts);
    const int64_t cdLd = CdLdOf(m, n, opts);
    ctx.matC = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, cdLd,
        16, kCD, opts.cdOrder);
    ctx.matD = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, cdLd,
        16, kCD, opts.cdOrder);
    // ---- batch 属性（四矩阵一致，须在 MatmulDescriptorInit 之前）----
    aclsparseLtMatDescriptor_t matADesc = ctx.ab.matAStruct ? ctx.ab.matAStruct->get()
        : ctx.ab.matADense->get();
    aclsparseLtMatDescriptor_t matBDesc = ctx.ab.matBStruct ? ctx.ab.matBStruct->get()
        : ctx.ab.matBDense->get();
    SetBatchAttrsOnAllMatDescs(ctx.handle.get(), &matADesc, &matBDesc,
        &ctx.matC->get(), &ctx.matD->get(), numBatches, batchStride, result);
    ctx.matmulDesc = std::make_unique<SparseLtMatmulDescGuard>(ctx.handle.get(), opA, opB,
        matADesc, matBDesc, ctx.matC->get(), ctx.matD->get(), ACL_SPARSE_COMPUTE_32I);

    // ---- buffers：A/B 按物理 span × batch 总量；C/D 按 (cdOrder, ldCd)
    //      pack 后上传（dC/dD 尺寸 = 物理 span × batch 总量）----
    const size_t totalAElems = BatchTotalElems(ctx.lay.aPhysSize, numBatches, batchStride);
    const size_t totalBElems = BatchTotalElems(ctx.lay.bPhysSize, numBatches, batchStride);
    const size_t cdSpan = CdSpanElems(m, n, opts);
    ctx.totalCdElems = BatchTotalElems(cdSpan, numBatches, batchStride);
    ctx.dA = DeviceBuffer::copyFrom(ctx.lay.hA_phys.data(), totalAElems * kInElt);
    ctx.dB = DeviceBuffer::copyFrom(ctx.lay.hB_phys.data(), totalBElems * kInElt);
    std::vector<CDT> hCPhys = PackBatchStridedPhysical<CDT>(hC, m, n,
        opts.cdOrder, cdLd, numBatches, batchStride);
    ctx.dC = DeviceBuffer::copyFrom(hCPhys.data(), ctx.totalCdElems * kOutElt);
    ctx.dD = DeviceBuffer::alloc(ctx.totalCdElems * kOutElt);
    if (hCPhysOut != nullptr) { *hCPhysOut = std::move(hCPhys); }
}

template <typename CDT>
inline MatmulNpuResult RunMatmulNpuInt8CD(
    int32_t m, int32_t k, int32_t n,
    const std::vector<int8_t>& hA, const std::vector<int8_t>& hB, const std::vector<CDT>& hC,
    std::vector<CDT>& hD,
    float alpha, float beta,
    int32_t algConfigId, int32_t splitK, int32_t splitKMode,
    bool isSparseA,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclsparseLtPruneAlg_t pruneAlg,
    int32_t alpha_vector_scaling, int32_t beta_vector_scaling,
    const std::vector<float>* alphaVec, const std::vector<float>* betaVec,
    int32_t biasEnabled, const std::vector<float>* biasVec,
    int32_t activationType, float reluUb, float reluThr, float geluScale,
    // order/batch/布局尾参（默认 = 单 batch 紧凑 ROW 基线）
    aclsparseOrder_t order = ACL_SPARSE_ORDER_ROW,
    int32_t numBatches = 1, int64_t batchStride = 0,
    const sparse_test::MatmulLayoutOpts& layout = sparse_test::MatmulLayoutOpts{})
{
    using namespace sparse_test;
    MatmulNpuResult result{};
    constexpr size_t kOutElt = NpuDtypeTrait<CDT>::kEltSize;
    const size_t mk = static_cast<size_t>(m) * k;
    const size_t kn = static_cast<size_t>(k) * n;

    aclrtStream stream = 0;
    // ---- 物理布局 + 描述符 + batch 属性 + 设备缓冲（铺设段见
    //      PrepareInt8CDChainContext）----
    Int8CDChainCtx<CDT> ctx;
    PrepareInt8CDChainContext<CDT>(ctx, m, k, n, hA, hB, hC, opA, opB, isSparseA,
                                   order, numBatches, batchStride, layout, result);

    // ---- vec / epilogue 属性（共享 helper；bias FP32——INT8 超集链 C 为 FP 域）。
    //      epiBufs.dBias 由 attr 持有 device 指针，须存活至下方 matmul 完成 ----
    VecScalingBuffers vecBufs;
    EpilogueBuffers epiBufs;
    if (!SetInt8CDVecEpiAttrs(ctx.handle, *ctx.matmulDesc, m,
        alpha_vector_scaling, beta_vector_scaling, alphaVec, betaVec,
        biasEnabled, 0, activationType, reluUb, reluThr, geluScale, numBatches, batchStride, biasVec,
        vecBufs, epiBufs, result)) { return result; }

    // ---- algSel + plan + ws ----
    std::unique_ptr<SparseLtAlgSelectionGuard> algSel;
    std::unique_ptr<SparseLtPlanGuard> plan;
    DeviceBuffer dWorkspace;
    void* wsPtr = nullptr;
    if (!PreparePlanAndWorkspace(result, ctx.handle, *ctx.matmulDesc, algConfigId, splitK,
                                 splitKMode, algSel, plan, dWorkspace, wsPtr)) {
        return result;
    }

    // ---- 剪枝（A-sparse：A_pruned；B-sparse：B_pruned (k,n)）+ 输入指针选择 ----
    Int8CDPrunePtrs pruned = PruneInt8CDSelectMatPtr(result, ctx.handle, *ctx.matmulDesc,
                                                     ctx.dA, ctx.dB, mk, kn, isSparseA,
                                                     pruneAlg, stream, numBatches,
                                                     batchStride, ctx.lay.bPhysSize);
    if (!pruned.ok) { return result; }

    // ---- matmul + 回读（回读尺寸 = C/D 物理 span × batch 总量）----
    ExecMatmulAndCopyResult<CDT>(result, ctx.handle.get(), plan->cptr(),
        pruned.matAPtr, pruned.matBPtr, ctx.dC.get(), ctx.dD.get(), wsPtr, vecBufs, alpha, beta,
        alpha_vector_scaling, beta_vector_scaling, ctx.dD, hD, ctx.totalCdElems, kOutElt);
    return result;
}

#endif  // TEST_MATMUL_NPU_WRAPPER_H_

