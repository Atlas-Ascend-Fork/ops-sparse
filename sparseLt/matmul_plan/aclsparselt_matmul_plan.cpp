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

/*!
 * \file aclsparselt_matmul_plan.cpp
 * \brief ops-sparseLt matmul plan 初始化与销毁函数实现。
 */

#include "cann_ops_sparseLt.h"
#include "aclsparselt_matmul_alg_selection_internal.h"
#include "aclsparselt_matmul_plan_internal.h"
#include "shared/aclsparselt_internal.h"

#include <new>

namespace {

/**
 * @brief 判断算法选择描述符引用是否有效（指针非空且句柄已初始化）。
 */
inline bool IsAlgSelectionValid(const aclsparseLtMatmulAlgSelection_t* algSelection)
{
    return algSelection != nullptr && *algSelection != nullptr;
}

/**
 * @brief 填充 plan 字段。
 */
inline void FillPlan(aclsparseLtMatmulPlan* plan,
                     const aclsparseLtMatmulDescriptor_t* matmulDescr,
                     const aclsparseLtMatmulAlgSelection_t* algSelection)
{
    plan->matmulDescr = *matmulDescr;
    plan->algSelection = *algSelection;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseLtMatmulPlanInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulPlan_t*               plan,
    const aclsparseLtMatmulDescriptor_t*   matmulDescr,
    const aclsparseLtMatmulAlgSelection_t* algSelection)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (plan == nullptr || *plan != nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (!IsMatmulDescrValid(matmulDescr)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (!IsAlgSelectionValid(algSelection)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // 校验 algSelection 内部绑定的 matmulDescr 与传入的 matmulDescr 一致
    if (ToInternal(*algSelection)->matmulDescr != *matmulDescr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto* p = new (std::nothrow) aclsparseLtMatmulPlan();
    if (p == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    FillPlan(p, matmulDescr, algSelection);

    auto* md = p->matmulDescr;
    auto* asel = p->algSelection;
    int32_t dt = dtype_from_acl(md->matA->valueType);
    int32_t effectiveSplitK = (asel->splitKMode == ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL) ? 1 : asel->splitK;
    // 读取 numBatches（A/B/C/D 一致性在 MatDescSetAttribute 时校验）
    int32_t numBatches = (md->matA->numBatches > 0) ? md->matA->numBatches : 1;
    WsLayout wsl = compute_ws_layout(md_m(md), md_n(md), md_k(md), effectiveSplitK, dt, numBatches,
                                     splt_apruned_span_elems(md, md_m(md), md_k(md)));
    uint32_t coreNum = get_cube_core_num();
    auto* td = new (std::nothrow) AclsparseltTilingData();
    if (td == nullptr) {
        delete p->tilingData;
        delete p;
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    fill_tiling_dims(*td, handle, md, asel, wsl, coreNum);
    p->tilingData = td;
    p->workspaceSize = static_cast<size_t>(wsl.totalBytes);

    *plan = p;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseLtMatmulPlanDestroy(aclsparseLtMatmulPlan_t* plan)
{
    if (plan == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (*plan == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    auto* p = *plan;
    // matmulDescr / algSelection 为非所有权引用，Destroy 不销毁它们
    delete p->tilingData;
    p->tilingData = nullptr;
    delete p;
    *plan = nullptr;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
