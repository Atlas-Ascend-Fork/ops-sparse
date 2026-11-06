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

/*!
 * \file aclsparselt_matmul_alg_selection.cpp
 * \brief ops-sparseLt matmul 算法选择描述符初始化与销毁函数实现。
 */

#include "cann_ops_sparseLt.h"
#include "aclsparselt_matmul_alg_selection_internal.h"

#include <new>

namespace {

/**
 * @brief 判断 alg 是否为合法的 aclsparseLtMatmulAlg_t 枚举值。
 */
inline bool IsValidAlg(aclsparseLtMatmulAlg_t alg)
{
    return alg == ACL_SPARSE_LT_MATMUL_ALG_DEFAULT;
}

/**
 * @brief 填充算法选择描述符字段。
 */
inline void FillAlgSelection(aclsparseLtMatmulAlgSelection* sel,
                             const aclsparseLtMatmulDescriptor_t* matmulDescr,
                             aclsparseLtMatmulAlg_t alg)
{
    sel->matmulDescr = *matmulDescr;
    sel->alg = alg;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseLtMatmulAlgSelectionInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulAlgSelection_t*       algSelection,
    const aclsparseLtMatmulDescriptor_t*   matmulDescr,
    aclsparseLtMatmulAlg_t                 alg)
{
    if (handle == nullptr || *handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (algSelection == nullptr || *algSelection != nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (!IsMatmulDescrValid(matmulDescr)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (!IsValidAlg(alg)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto* sel = new (std::nothrow) aclsparseLtMatmulAlgSelection();
    if (sel == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    FillAlgSelection(sel, matmulDescr, alg);
    *algSelection = sel;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseLtMatmulAlgSelectionDestroy(aclsparseLtMatmulAlgSelection_t* algSelection)
{
    if (algSelection == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (*algSelection == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    auto* sel = *algSelection;
    // matmulDescr 为非所有权引用，Destroy 不销毁它
    delete sel;
    *algSelection = nullptr;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
