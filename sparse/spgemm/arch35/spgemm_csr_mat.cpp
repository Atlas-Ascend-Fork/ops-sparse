/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#include <stdint.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include "acl/acl.h"
#include "spgemm.h"
#include "spgemm_csr_mat.h"
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"

/* Greedy bin-packing: sort rows by weight descending, then place each row into
 * the bin with the smallest accumulated load. This balances workload across cores. */
void SpgemmGreedyRowBinPack(
    const std::vector<int32_t> &rowWeight,
    int32_t binNum,
    SpgemmRowBinPackResult &result)
{
    if (binNum <= 0) {
        return;
    }
    int32_t m = static_cast<int32_t>(rowWeight.size());
    std::vector<int32_t> order(m);
    for (int32_t i = 0; i < m; ++i) order[i] = i;

    std::sort(order.begin(), order.end(),
              [&](int32_t a, int32_t b) { return rowWeight[a] > rowWeight[b]; });

    result.reorder.resize(m);
    result.binEdges.assign(binNum + 1, 0);

    std::vector<std::vector<int32_t>> bins(binNum);
    std::vector<int64_t> binLoad(binNum, 0);

    for (int32_t idx : order) {
        int32_t pick = 0;
        for (int32_t b = 1; b < binNum; ++b) {
            if (binLoad[b] < binLoad[pick]) { pick = b; }
        }
        bins[pick].push_back(idx);
        binLoad[pick] += rowWeight[idx];
    }

    int32_t pos = 0;
    for (int32_t b = 0; b < binNum; ++b) {
        result.binEdges[b] = pos;
        for (int32_t rowIdx : bins[b]) {
            result.reorder[pos] = rowIdx;
            ++pos;
        }
    }
    result.binEdges[binNum] = pos;
}

aclsparseStatus_t SpgemmComputeRowWeights(
    const std::vector<int32_t> &aRowPtr,
    const std::vector<int32_t> &aColInd,
    const std::vector<int32_t> &bRowPtr,
    std::vector<int32_t> &rowWeights)
{
    int32_t m = static_cast<int32_t>(aRowPtr.size()) - 1;
    rowWeights.assign(m, 0);

    for (int32_t i = 0; i < m; ++i) {
        int32_t aStart = aRowPtr[i];
        int32_t aEnd   = aRowPtr[i + 1];
        int64_t weight = 0;
        for (int32_t p = aStart; p < aEnd; ++p) {
            int32_t k = aColInd[p];
            weight += static_cast<int64_t>(bRowPtr[k + 1]) - static_cast<int64_t>(bRowPtr[k]);
        }
        if (weight > INT32_MAX) { weight = INT32_MAX; }
        rowWeights[i] = static_cast<int32_t>(weight);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

int64_t SpgemmComputeNumProds(
    const std::vector<int32_t> &aRowPtr,
    const std::vector<int32_t> &aColInd,
    const std::vector<int32_t> &bRowPtr)
{
    int32_t m = static_cast<int32_t>(aRowPtr.size()) - 1;
    int64_t totalProds = 0;
    for (int32_t i = 0; i < m; ++i) {
        int32_t aStart = aRowPtr[i];
        int32_t aEnd   = aRowPtr[i + 1];
        for (int32_t p = aStart; p < aEnd; ++p) {
            int32_t k = aColInd[p];
            totalProds += static_cast<int64_t>(bRowPtr[k + 1]) - static_cast<int64_t>(bRowPtr[k]);
        }
    }
    return totalProds;
}

aclsparseStatus_t SpgemmWriteReorderToWorkspace(
    const SpgemmRowBinPackResult &binPack,
    void *dWorkspace,
    const SpgemmWsOffsets &off,
    aclrtStream stream)
{
    if (!binPack.reorder.empty()) {
        aclError ret = aclrtMemcpyAsync(
            static_cast<uint8_t *>(dWorkspace) + off.reorderOff,
            binPack.reorder.size() * sizeof(int32_t),
            binPack.reorder.data(),
            binPack.reorder.size() * sizeof(int32_t),
            ACL_MEMCPY_HOST_TO_DEVICE, stream);
        if (ret != ACL_ERROR_NONE) {
            OP_LOGE("spgemm", "aclrtMemcpyAsync H2D for reorder failed, ret=%d", ret);
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    if (!binPack.binEdges.empty()) {
        aclError ret = aclrtMemcpyAsync(
            static_cast<uint8_t *>(dWorkspace) + off.binEdgeOff,
            binPack.binEdges.size() * sizeof(int32_t),
            binPack.binEdges.data(),
            binPack.binEdges.size() * sizeof(int32_t),
            ACL_MEMCPY_HOST_TO_DEVICE, stream);
        if (ret != ACL_ERROR_NONE) {
            OP_LOGE("spgemm", "aclrtMemcpyAsync H2D for binEdges failed, ret=%d", ret);
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t SpgemmCopyBRowPtrToWorkspace(
    aclsparseSpMatDescr *matB,
    void *dWorkspace,
    const SpgemmWsOffsets &off,
    aclrtStream stream)
{
    if (matB == nullptr || matB->ptrs == nullptr) {
        OP_LOGE("spgemm", "SpgemmCopyBRowPtrToWorkspace: matB or matB->ptrs is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int64_t k = static_cast<int64_t>(matB->rows);
    aclError ret = aclrtMemcpyAsync(
        static_cast<uint8_t *>(dWorkspace) + off.bRowPtrOff,
        (k + 1) * sizeof(int32_t),
        matB->ptrs,
        (k + 1) * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    if (ret != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "aclrtMemcpyAsync D2D for B rowPtr failed, ret=%d", ret);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t SpgemmValidateCsrSorted(
    aclsparseSpMatDescr *matDesc,
    const char *label)
{
    if (matDesc == nullptr || matDesc->ptrs == nullptr || matDesc->idxs == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;  /* empty matrix is trivially sorted */
    }
    int64_t m = static_cast<int64_t>(matDesc->rows);
    int64_t nnz = static_cast<int64_t>(matDesc->nnz);
    if (m <= 0 || nnz <= 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    /* D2H copy rowPtr + colInd for host-side sorted check */
    std::vector<int32_t> rowPtr(m + 1);
    aclError ret = aclrtMemcpy(rowPtr.data(), (m + 1) * sizeof(int32_t),
                               matDesc->ptrs, (m + 1) * sizeof(int32_t),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "%s: D2H copy rowPtr for sorted check failed, ret=%d", label, ret);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    std::vector<int32_t> colInd(nnz);
    ret = aclrtMemcpy(colInd.data(), nnz * sizeof(int32_t),
                      matDesc->idxs, nnz * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "%s: D2H copy colInd for sorted check failed, ret=%d", label, ret);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    for (int64_t i = 0; i < m; ++i) {
        int32_t rowStart = rowPtr[i];
        int32_t rowEnd = rowPtr[i + 1];
        for (int32_t p = rowStart + 1; p < rowEnd; ++p) {
            if (colInd[p] < colInd[p - 1]) {
                OP_LOGE("spgemm", "%s: colInd not sorted at row %lld, "
                        "colInd[%d]=%d < colInd[%d]=%d",
                        label, i, p, colInd[p], p - 1, colInd[p - 1]);
                return ACL_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
