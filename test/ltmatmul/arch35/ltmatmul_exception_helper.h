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

#ifndef TEST_MATMUL_EXCEPTION_HELPER_H_
#define TEST_MATMUL_EXCEPTION_HELPER_H_

// =============================================================================
// Exception test helpers for aclsparseLtMatmul.
//
// Extracted from matmul_npu_wrapper.h to reduce header file size.
//
// Contents:
//   - MatmulRawParams: raw matmul call params with injectable invalid values
//   - MatmulNpuRaw: direct aclsparseLtMatmul wrapper for exception tests
//   - SparseLtMatmulCtxBase: shared RAII guard/stream/buffer skeleton +
//     common descriptor construction for negative-test matmul contexts
//     (MatmulExceptionCtx here; dtype-pair variant in ltmatmul_test_neg.cpp)
//   - MatmulExceptionCtx: RAII context for building a valid 8x8 plan
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparseLt.h"
#include "descriptor_manager.h"

#include "ltmatmul_npu_wrapper.h"

// -----------------------------------------------------------------------------
// Exception test: raw matmul call with injectable invalid parameters.
// -----------------------------------------------------------------------------
struct MatmulRawParams {
    aclsparseLtConstHandle_t handle = nullptr;
    aclsparseLtConstMatmulPlan_t plan = nullptr;
    const void* alpha = nullptr;
    const void* matA = nullptr;
    const void* matB = nullptr;
    const void* beta = nullptr;
    const void* matC = nullptr;
    void* matD = nullptr;
    void* workspace = nullptr;
    aclrtStream* streams = nullptr;
    int32_t numStreams = 1;
};

inline aclsparseStatus_t MatmulNpuRaw(const MatmulRawParams& p)
{
    aclsparseLtConstMatmulPlan_t plan = p.plan;
    return aclsparseLtMatmul(p.handle, &plan, p.alpha, p.matA, p.matB,
                             p.beta, p.matC, p.matD, p.workspace,
                             p.streams, p.numStreams);
}

// -----------------------------------------------------------------------------
// Helper: fill the common MatmulRawParams fields for exception tests.
// Individual tests override specific fields after calling this.
// (Moved from ltmatmul_test.cpp — shared by ltmatmul_test_neg.cpp.)
// -----------------------------------------------------------------------------
inline void FillMatmulRawParams(MatmulRawParams& p,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulPlan_t plan,
    const void* alpha, const void* matA, const void* matB,
    const void* beta, const void* matC, void* matD,
    void* workspace, aclrtStream* streams, int32_t numStreams)
{
    p.handle = handle;
    p.plan = plan;
    p.alpha = alpha;
    p.matA = matA;
    p.matB = matB;
    p.beta = beta;
    p.matC = matC;
    p.matD = matD;
    p.workspace = workspace;
    p.streams = streams;
    p.numStreams = numStreams;
}

// -----------------------------------------------------------------------------
// SparseLtMatmulCtxBase: shared RAII skeleton for negative-test matmul
// contexts — full guard chain (handle → A/B/C/D descriptors → matmulDesc →
// algSel → plan; A structured 50% + B/C/D dense ROW, alignment 16) plus
// stream / device buffers / workspaceSize. MatmulExceptionCtx (8x8 FP32
// baseline, this file) and MatmulDtypePairCtx (128³ dtype-pair matrix,
// ltmatmul_test_neg.cpp) derive from it; per-scenario buffer / workspace
// provisioning stays in the derived ctor bodies. Base dtor releases the
// stream before guards unwind (same sequence as the former per-struct
// member layout: dtor body first, then members in reverse order).
// -----------------------------------------------------------------------------
struct SparseLtMatmulCtxBase {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::SparseLtMatDescGuard matA;  // A structured（A-sparse 50%）
    sparse_test::SparseLtDnMatDescGuard matB;
    sparse_test::SparseLtDnMatDescGuard matC;
    sparse_test::SparseLtDnMatDescGuard matD;
    sparse_test::SparseLtMatmulDescGuard matmulDesc;
    sparse_test::SparseLtAlgSelectionGuard algSel;
    sparse_test::SparseLtPlanGuard plan;

    aclrtStream stream = nullptr;
    sparse_test::DeviceBuffer dA;
    sparse_test::DeviceBuffer dB;
    sparse_test::DeviceBuffer dC;
    sparse_test::DeviceBuffer dD;
    sparse_test::DeviceBuffer dWorkspace;
    size_t workspaceSize = 0;

    // 公共构造段：A/B/C/D 描述符 + matmulDesc + algSel + plan。A/B 共享
    // abDtype，C/D 独立 dtype（dtype 配对负向注入用）；opA 恒为 N，opB /
    // computeType 由场景注入（INT8 路径 ROW/ROW 须 (N, T)）。
    SparseLtMatmulCtxBase(int32_t m, int32_t k, int32_t n,
                          aclDataType abDtype, aclDataType cDtype, aclDataType dDtype,
                          aclsparseOperation_t opB, aclsparseComputeType_t computeType)
        : matA(handle.get(), k, m, m, 16, abDtype, ACL_SPARSE_ORDER_ROW,
               ACL_SPARSE_LT_SPARSITY_50_PERCENT),
          matB(handle.get(), k, n, n, 16, abDtype, ACL_SPARSE_ORDER_ROW),
          matC(handle.get(), m, n, n, 16, cDtype, ACL_SPARSE_ORDER_ROW),
          matD(handle.get(), m, n, n, 16, dDtype, ACL_SPARSE_ORDER_ROW),
          matmulDesc(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, opB,
                     matA.get(), matB.get(), matC.get(), matD.get(), computeType),
          algSel(handle.get(), matmulDesc.get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT),
          plan(handle.get(), matmulDesc.get(), algSel.get())
    {
    }

    // 公共 ctor 前奏：创建 stream（errMsg 保留原 per-ctx 前缀文案）。
    void CreateStream(const char* errMsg)
    {
        if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
            throw std::runtime_error(errMsg);
        }
    }

    ~SparseLtMatmulCtxBase() {
        if (stream) { aclrtDestroyStream(stream); }
    }
};

// -----------------------------------------------------------------------------
// Helper: build a valid plan for exception tests (8x8 FP32 descriptors).
// -----------------------------------------------------------------------------
struct MatmulExceptionCtx : SparseLtMatmulCtxBase {
    MatmulExceptionCtx(int32_t m = 8, int32_t k = 8, int32_t n = 8,
                       aclDataType dtype = ACL_FLOAT,
                       bool isDense = false)
        : SparseLtMatmulCtxBase(m, k, n, dtype, dtype, dtype,
                                ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_COMPUTE_32F)
    {
        CreateStream("MatmulExceptionCtx: aclrtCreateStream failed");
        const size_t mk = static_cast<size_t>(m) * k;
        const size_t kn = static_cast<size_t>(k) * n;
        const size_t mn = static_cast<size_t>(m) * n;
        std::vector<float> hA(mk, 1.0f);
        std::vector<float> hB(kn, 1.0f);
        std::vector<float> hC(mn, 0.0f);
        dA = sparse_test::DeviceBuffer::copyFrom(hA.data(), mk * sizeof(float));
        dB = sparse_test::DeviceBuffer::copyFrom(hB.data(), kn * sizeof(float));
        dC = sparse_test::DeviceBuffer::copyFrom(hC.data(), mn * sizeof(float));
        dD = sparse_test::DeviceBuffer::alloc(mn * sizeof(float));
        aclsparseStatus_t wsRet = aclsparseLtMatmulGetWorkspace(handle.get(), plan.cptr(), &workspaceSize);
        if (wsRet != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("MatmulExceptionCtx: GetWorkspaceSize failed");
        }
        if (workspaceSize > 0) {
            dWorkspace = sparse_test::DeviceBuffer::alloc(workspaceSize);
        }
    }
};

#endif  // TEST_MATMUL_EXCEPTION_HELPER_H_
