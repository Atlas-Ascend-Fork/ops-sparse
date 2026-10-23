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

#ifndef TEST_LTMATMUL_LTMATMUL_BATCH_NEG_HELPER_H_
#define TEST_LTMATMUL_LTMATMUL_BATCH_NEG_HELPER_H_

// =============================================================================
// batch 布局负向用例共享构造层（ltmatmul WB_L2_45/46/47 与 matmulSearch
// X18/X19 系的跨 target 共享段）。
//
// 提供：
//   - LtBatchMatDescKit：FP32 128³ A-sparse 四 mat 描述符 + 一致 batch
//     属性（NUM_BATCHES / BATCH_STRIDE 设置于 A/B/C/D——一致性校验要求
//     相同）+ matmulDescr 构造；numBatches / batchStride / ldB（B dense，
//     ldB<=0 时取 n）可注入，构造失败 throw（negative-ctx 范式）。
//   - MatmulBatchOverflowCtx：kit + algSel + plan + ws（原 ltmatmul_test.cpp
//     匿名 namespace 版本 kit 化重构，语义不变）。
//   - RunBatchNegMatmulRaw：WB_L2_45/46/47 共用执行段——128³ FP32
//     A/B/C/D 单 batch 尺寸缓冲 + 标量 α/β + 默认流 + MatmulNpuRaw
//     （候选执行在校验层被拒，缓冲不会被实际读取）。
// =============================================================================

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "cann_ops_sparseLt.h"

#include "ltmatmul_test_utils.h"         // SparseLt*Guard / DeviceBuffer
#include "ltmatmul_exception_helper.h"   // MatmulRawParams / MatmulNpuRaw / FillMatmulRawParams

namespace sparse_test {

// -----------------------------------------------------------------------------
// LtBatchMatDescKit：batch 域四 mat 描述符 + matmulDescr（参数化构造）。
// -----------------------------------------------------------------------------
struct LtBatchMatDescKit {
    std::unique_ptr<SparseLtMatDescGuard> matA;
    std::unique_ptr<SparseLtDnMatDescGuard> matB;
    std::unique_ptr<SparseLtDnMatDescGuard> matC;
    std::unique_ptr<SparseLtDnMatDescGuard> matD;
    std::unique_ptr<SparseLtMatmulDescGuard> matmulDesc;

    // m/k/n 固定 128³（既有 batch 负向基型一致）；ldB 作用于 B dense
    //（ldB > 0 时注入，否则取物理列数 n）。
    LtBatchMatDescKit(aclsparseLtConstHandle_t handle,
                      int32_t numBatches, int64_t batchStride,
                      int64_t ldB = 0, const char* tag = "LtBatchMatDescKit")
    {
        constexpr int32_t m = 128, k = 128, n = 128;
        matA = std::make_unique<SparseLtMatDescGuard>(
            handle, m, k, k, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW,
            ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        matB = std::make_unique<SparseLtDnMatDescGuard>(
            handle, k, n, (ldB > 0) ? ldB : n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        matC = std::make_unique<SparseLtDnMatDescGuard>(
            handle, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        matD = std::make_unique<SparseLtDnMatDescGuard>(
            handle, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        for (auto* md : {&matA->get(), &matB->get(), &matC->get(), &matD->get()}) {
            if (aclsparseLtMatDescSetAttribute(handle, md,
                    ACLSPARSELT_MAT_NUM_BATCHES, &numBatches, sizeof(int32_t))
                    != ACL_SPARSE_STATUS_SUCCESS ||
                aclsparseLtMatDescSetAttribute(handle, md,
                    ACLSPARSELT_MAT_BATCH_STRIDE, &batchStride, sizeof(int64_t))
                    != ACL_SPARSE_STATUS_SUCCESS) {
                throw std::runtime_error(std::string(tag) + ": batch attrs set failed");
            }
        }
        matmulDesc = std::make_unique<SparseLtMatmulDescGuard>(
            handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            matA->get(), matB->get(), matC->get(), matD->get(), ACL_SPARSE_COMPUTE_32F);
    }
};

// -----------------------------------------------------------------------------
// MatmulBatchOverflowCtx：batch 布局溢出基座（numBatches=2，batchStride /
// ldB 注入）——kit + algSel + plan + ws。
// -----------------------------------------------------------------------------
struct MatmulBatchOverflowCtx {
    SparseLtHandleGuard handle;
    LtBatchMatDescKit kit;
    std::unique_ptr<SparseLtAlgSelectionGuard> algSel;
    std::unique_ptr<SparseLtPlanGuard> plan;
    DeviceBuffer dWorkspace;
    size_t workspaceSize = 0;

    explicit MatmulBatchOverflowCtx(int64_t batchStride, int64_t ldB)
        : kit(handle.get(), 2, batchStride, ldB, "MatmulBatchOverflowCtx")
    {
        algSel = std::make_unique<SparseLtAlgSelectionGuard>(
            handle.get(), kit.matmulDesc->get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
        plan = std::make_unique<SparseLtPlanGuard>(
            handle.get(), kit.matmulDesc->get(), algSel->get());
        if (aclsparseLtMatmulGetWorkspace(handle.get(), plan->cptr(), &workspaceSize)
                != ACL_SPARSE_STATUS_SUCCESS || workspaceSize == 0) {
            throw std::runtime_error("MatmulBatchOverflowCtx: GetWorkspace failed");
        }
        dWorkspace = DeviceBuffer::alloc(workspaceSize);
    }
};

// -----------------------------------------------------------------------------
// RunBatchNegMatmulRaw：WB_L2_45/46/47 共用执行段。负向用例的候选执行在
// 校验层被拒，A/B/C/D 缓冲不会被实际读取——按单 batch 尺寸分配（与既有
// 用例一致）；α/β 标量、默认流。
// -----------------------------------------------------------------------------
inline aclsparseStatus_t RunBatchNegMatmulRaw(
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulPlan_t plan,
    void* workspace)
{
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = 0;
    const size_t mk = 128 * 128;
    const size_t kn = 128 * 128;
    const size_t mn = 128 * 128;
    auto dA = DeviceBuffer::alloc(mk * sizeof(float));
    auto dB = DeviceBuffer::alloc(kn * sizeof(float));
    auto dC = DeviceBuffer::alloc(mn * sizeof(float));
    auto dD = DeviceBuffer::alloc(mn * sizeof(float));
    FillMatmulRawParams(p, handle, plan, &alpha, dA.get(), dB.get(), &beta,
        dC.get(), dD.get(), workspace, &stream, 1);
    return MatmulNpuRaw(p);
}

}  // namespace sparse_test

#endif  // TEST_LTMATMUL_LTMATMUL_BATCH_NEG_HELPER_H_
