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

#ifndef TEST_LTMATMUL_ALG_NPU_H_
#define TEST_LTMATMUL_ALG_NPU_H_

// =============================================================================
// AlgSetAttributeNpu chain extracted from ltmatmul_test_utils.h
// to reduce header file size.
//
// Provides:
//   - AlgSetAttributeNpuResult: result struct for the alg_set chain
//   - SetupAndRunAlgMatmul: plan + prune + matmul execution
//   - AlgNpuCommonCtx / InitAlgNpuCommon: shared context for A-sparse chain
//   - RunAlgNpuASparseChain: A-sparse chain runner
//   - TransposeBPruned: transpose pruned B (shared by B-sparse path)
//   - AlgSetAttributeNpu: full A-sparse chain entry point
//
// Used by alignment-32 end-to-end exception tests in ltmatmul_test.cpp.
// =============================================================================

#include "ltmatmul_test_utils.h"  // RAII guards, PreparePhysicalA/B, traits

// =============================================================================
// Global namespace: NPU result structs + chain functions.
// =============================================================================

// -----------------------------------------------------------------------------
// AlgSetAttribute NPU result + chain functions (from alg_set_attribute).
// -----------------------------------------------------------------------------
struct AlgSetAttributeNpuResult {
    aclsparseStatus_t algSetCfgRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algSetSplitRet = ACL_SPARSE_STATUS_SUCCESS;
    // getAttr roundtrip verification (setAttr -> getAttr readback).
    aclsparseStatus_t algGetCfgRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algGetSplitRet = ACL_SPARSE_STATUS_SUCCESS;
    int32_t gotAlgConfigId = -1;
    int32_t gotSplitK = -1;
    aclsparseStatus_t pruneRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t matmulRet = ACL_SPARSE_STATUS_SUCCESS;
    size_t workspaceSize = 0;
    double npuMs = 0.0;
};

// -----------------------------------------------------------------------------
// AlgSetAttributeNpu: A-sparse chain used by alignment-32 end-to-end tests.
// Runs the full chain: setAttr → getAttr roundtrip → planInit → prune → matmul.
// (B-sparse variant removed — matmul path covers B-sparse via RunMatmulNpu.)
// -----------------------------------------------------------------------------
inline bool SetupAndRunAlgMatmul(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulDescriptor_t matmulDesc,
    aclsparseLtMatmulAlgSelection_t algSel,
    void* dA, void* dAPruned, void* dB, void* dC, void* dD,
    aclrtStream stream, float alpha, float beta,
    AlgSetAttributeNpuResult& result)
{
    using sparse_test::SparseLtPlanGuard;
    using sparse_test::EventGuard;
    using sparse_test::DeviceBuffer;

    SparseLtPlanGuard plan(handle, matmulDesc, algSel);

    auto wsRet = aclsparseLtMatmulGetWorkspace(handle, plan.cptr(), &result.workspaceSize);
    if (wsRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] MatmulGetWorkspaceSize failed: " << wsRet << std::endl;
        result.matmulRet = wsRet;
        return false;
    }
    DeviceBuffer dWorkspace = (result.workspaceSize > 0)
        ? DeviceBuffer::alloc(result.workspaceSize) : DeviceBuffer{};
    void* wsPtr = (result.workspaceSize > 0) ? dWorkspace.get() : nullptr;

    EventGuard evStart, evStop;
    evStart.record(stream);

    result.pruneRet = aclsparseLtSpMMAPrune(
        handle, const_cast<aclsparseLtConstMatmulDescriptor_t*>(&matmulDesc), dA, dAPruned,
        ACLSPARSELT_PRUNE_SPMMA_STRIP, stream);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] SpMMAPrune failed: " << result.pruneRet << std::endl;
        return false;
    }

    result.matmulRet = aclsparseLtMatmul(
        handle, plan.cptr(),
        static_cast<const void*>(&alpha),
        dAPruned, dB,
        static_cast<const void*>(&beta), dC, dD,
        wsPtr, &stream, 1);

    evStop.record(stream);
    auto syncRet = aclrtSynchronizeStream(stream);
    if (result.matmulRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] Matmul failed: " << result.matmulRet << std::endl;
        return false;
    }
    if (syncRet != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed: " << syncRet << std::endl;
        result.matmulRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }

    result.npuMs = static_cast<double>(EventGuard::elapsedMs(evStart, evStop));
    return true;
}

// Updated AlgSetAttributeNpu with transA and transB support.
// transA=true, ROW: A data is transposed from (m,k) to (k,m) row-major.
// transA=true, COL: A data stays as (m,k) row-major (= (k,m) col-major with ld=k).
// transB=true: B data is transposed from (k,n) to (n,k) row-major.

template <typename T>
struct AlgNpuCommonCtx {
    static constexpr aclDataType kDtype = sparse_test::NpuDtypeTrait<T>::kAclDtype;
    static constexpr size_t kElt = sparse_test::NpuDtypeTrait<T>::kEltSize;
    static constexpr aclsparseComputeType_t kComputeType = ACL_SPARSE_COMPUTE_32F;
    bool transA;
    bool transB;
    size_t mk;
    size_t kn;
    size_t mn;
    int64_t aLd;
    int64_t physRowsA;
    int64_t physColsA;
    size_t aPhysSize;
    sparse_test::DeviceBuffer dA;
    sparse_test::DeviceBuffer dB;
    sparse_test::DeviceBuffer dC;
    sparse_test::DeviceBuffer dD;
};

template <typename T>
inline AlgNpuCommonCtx<T> InitAlgNpuCommon(
    int32_t m, int32_t k, int32_t n,
    const std::vector<T>& hA, const std::vector<T>& hB, const std::vector<T>& hC,
    aclsparseOrder_t order, aclsparseOperation_t opA, aclsparseOperation_t opB)
{
    using namespace sparse_test;
    AlgNpuCommonCtx<T> ctx;
    ctx.transA = (opA == ACL_SPARSE_OP_TRANSPOSE);
    ctx.transB = (opB == ACL_SPARSE_OP_TRANSPOSE);
    ctx.mk = static_cast<size_t>(m) * static_cast<size_t>(k);
    ctx.kn = static_cast<size_t>(k) * static_cast<size_t>(n);
    ctx.mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    std::vector<T> hA_phys = PreparePhysicalA<T>(hA, m, k, ctx.transA, order, ctx.aLd);
    std::vector<T> hB_phys = PreparePhysicalB<T>(hB, k, n, ctx.transB);
    ctx.physRowsA = ctx.transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    ctx.physColsA = ctx.transA ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    ctx.aPhysSize = (order == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(ctx.physColsA) * static_cast<size_t>(ctx.aLd)
        : static_cast<size_t>(ctx.physRowsA) * static_cast<size_t>(ctx.aLd);
    ctx.dA = DeviceBuffer::copyFrom(hA_phys.data(), ctx.aPhysSize * ctx.kElt);
    ctx.dB = DeviceBuffer::copyFrom(hB_phys.data(),
        (ctx.transB ? static_cast<size_t>(n) * k : ctx.kn) * ctx.kElt);
    ctx.dC = DeviceBuffer::copyFrom(hC.data(), ctx.mn * ctx.kElt);
    ctx.dD = DeviceBuffer::alloc(ctx.mn * ctx.kElt);
    return ctx;
}

struct AlgNpuRunParams {
    aclrtStream stream;
    int32_t m;
    int32_t k;
    int32_t n;
    float alpha;
    float beta;
    int32_t algConfigId;
    int32_t splitK;
    uint32_t alignment;
    aclsparseOrder_t order;
    aclsparseOperation_t opA;
    aclsparseOperation_t opB;
};

template <typename T>
inline AlgSetAttributeNpuResult RunAlgNpuASparseChain(
    AlgNpuCommonCtx<T>& ctx, const AlgNpuRunParams& p, std::vector<T>& hD)
{
    using namespace sparse_test;
    AlgSetAttributeNpuResult result{};
    DeviceBuffer dAPruned = DeviceBuffer::alloc(ctx.mk * ctx.kElt);

    SparseLtMatmulChainParams chainParams{
        static_cast<int64_t>(p.m), static_cast<int64_t>(p.k), static_cast<int64_t>(p.n),
        ctx.aLd, p.alignment, ctx.kDtype, ctx.kComputeType, p.order, p.opA, p.opB,
        dAPruned.get(), ctx.dB.get(), ctx.dC.get(), ctx.dD.get()
    };
    SparseLtMatmulChainCtx chain(chainParams);

    if (!SetAlgAttributes(chain.handle.get(), chain.algSel.get(),
                          p.algConfigId, p.splitK,
                          result.algSetCfgRet, result.algSetSplitRet)) {
        return result;
    }

    // getAttr roundtrip verification: read back the values set by SetAlgAttributes.
    result.algGetCfgRet = aclsparseLtMatmulAlgGetAttribute(
        chain.handle.get(), chain.algSel.cptr(),
        ACLSPARSELT_MATMUL_ALG_CONFIG_ID, &result.gotAlgConfigId, sizeof(int32_t));
    result.algGetSplitRet = aclsparseLtMatmulAlgGetAttribute(
        chain.handle.get(), chain.algSel.cptr(),
        ACLSPARSELT_MATMUL_SPLIT_K, &result.gotSplitK, sizeof(int32_t));

    if (!SetupAndRunAlgMatmul(chain.handle.get(), chain.matmulDesc.get(),
                               chain.algSel.get(),
                               ctx.dA.get(), dAPruned.get(), ctx.dB.get(), ctx.dC.get(), ctx.dD.get(),
                               p.stream, p.alpha, p.beta, result)) {
        return result;
    }

    hD.resize(ctx.mn);
    ctx.dD.copyToHost(hD.data(), ctx.mn * ctx.kElt);
    return result;
}

// -----------------------------------------------------------------------------
// TransposeBPruned: transpose the pruned B matrix (shared by matmul B-sparse path).
// -----------------------------------------------------------------------------

template <typename T>
inline sparse_test::DeviceBuffer TransposeBPruned(
    aclrtStream stream, sparse_test::DeviceBuffer& dBPruned,
    int32_t k, int32_t n, size_t kn, size_t kElt)
{
    using namespace sparse_test;
    aclError syncRet = aclrtSynchronizeStream(stream);
    if (syncRet != ACL_SUCCESS) {
        std::cerr << "[NPU] TransposeBPruned aclrtSynchronizeStream failed: " << syncRet << std::endl;
        return sparse_test::DeviceBuffer{};
    }
    std::vector<T> hBPruned(kn);
    dBPruned.copyToHost(hBPruned.data(), kn * kElt);
    std::vector<T> hBPrunedT(kn);
    for (int32_t i = 0; i < k; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            hBPrunedT[static_cast<size_t>(j) * k + i] =
                hBPruned[static_cast<size_t>(i) * n + j];
        }
    }
    return DeviceBuffer::copyFrom(hBPrunedT.data(), kn * kElt);
}

// -----------------------------------------------------------------------------
// AlgSetAttributeNpu: A-sparse chain used by alignment-32 end-to-end tests.
// Runs the full chain: setAttr → getAttr roundtrip → planInit → prune → matmul.
// (B-sparse variant removed — matmul path covers B-sparse via RunMatmulNpu.)
// -----------------------------------------------------------------------------
template <typename T>
inline AlgSetAttributeNpuResult AlgSetAttributeNpu(
    aclrtStream stream,
    const std::vector<T>& hA, const std::vector<T>& hB, const std::vector<T>& hC,
    std::vector<T>& hD,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    int32_t algConfigId, int32_t splitK,
    uint32_t alignment = 16,
    aclsparseOrder_t order = ACL_SPARSE_ORDER_ROW,
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE,
    aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE)
{
    AlgNpuRunParams p{stream, m, k, n, alpha, beta, algConfigId, splitK,
                      alignment, order, opA, opB};
    auto ctx = InitAlgNpuCommon<T>(m, k, n, hA, hB, hC, order, opA, opB);
    return RunAlgNpuASparseChain<T>(ctx, p, hD);
}

#endif  // TEST_LTMATMUL_ALG_NPU_H_
