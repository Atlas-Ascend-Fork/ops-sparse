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
 * \file cscsort_host.cpp
 * \brief aclsparseXcscsort Host 侧实现（Legacy API）。
 *
 * 两个 extern "C" API：
 *   - aclsparseXcscsort_bufferSizeExt：查询 pBuffer 大小（2*nnz*sizeof(int32_t)）。
 *   - aclsparseXcscsort：对 CSC 每列的行索引执行原地稳定排序，并输出排列 P。
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "cscsort_kernel.h"
#include "sort_tiling_utils.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"

namespace {

inline struct aclsparseContext *ToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

static aclsparseStatus_t ValidateCscsortCommonParams(
    const char *apiName, int m, int n, int nnz, const int *cscColPtr, const int *cscRowInd)
{
    if (m < 0 || n < 0 || nnz < 0) {
        OP_LOGE(apiName, "invalid dims: m=%d, n=%d, nnz=%d", m, n, nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if ((m == 0 || n == 0) && nnz != 0) {
        OP_LOGE(apiName, "empty matrix (m=%d, n=%d) with nnz=%d", m, n, nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n > 0 && cscColPtr == nullptr) {
        OP_LOGE(apiName, "cscColPtr is nullptr (n=%d)", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && cscRowInd == nullptr) {
        OP_LOGE(apiName, "cscRowInd is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ComputeCscsortTiling(
    int m, int n, int nnz, aclsparseIndexBase_t indexBase, CscsortTilingData &tiling)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE("aclsparseXcscsort", "GetAivCoreCount returned 0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    uint64_t ubSize = GetUbSize();
    if (ubSize == 0U) {
        OP_LOGE("aclsparseXcscsort", "GetUbSize returned 0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    uint32_t runSize = 0U;
    uint32_t sortTmpBytes = 0U;
    if (!sort_tiling::FindMaxRunSize(ubSize, runSize, sortTmpBytes)) {
        OP_LOGE("aclsparseXcscsort", "UB capacity %llu insufficient for minimum run",
                static_cast<unsigned long long>(ubSize));
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    // 每个非空 CSC 列最多形成一个独立任务。限制启动核数不超过 nnz，
    // 避免大量空列矩阵启动无实际工作的 AIV。
    uint32_t taskUpperBound = std::min(static_cast<uint32_t>(n), static_cast<uint32_t>(nnz));
    uint32_t coreNum = std::min(aivCoreNum, taskUpperBound);

    tiling.n = static_cast<uint32_t>(n);
    tiling.nnz = static_cast<uint32_t>(nnz);
    tiling.indexBase = static_cast<uint32_t>(indexBase);
    tiling.runSize = runSize;
    tiling.coreNum = coreNum;
    tiling.sortTmpBytes = sortTmpBytes;
    tiling.simtThreads = sort_tiling::CalcSimtThreads(tiling.nnz, coreNum);

    OP_LOGD("aclsparseXcscsort",
            "tiling: m=%d, n=%u, nnz=%u, runSize=%u, blockDim=%u, sortTmp=%u, simtThreads=%u",
            m, tiling.n, tiling.nnz, tiling.runSize, tiling.coreNum,
            tiling.sortTmpBytes, tiling.simtThreads);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchCscsortKernel(aclsparseHandle_t handle, int m, int n, int nnz,
                                             aclsparseIndexBase_t indexBase, const int *cscColPtr,
                                             int *cscRowInd, int *P, void *pBuffer)
{
    auto *h = ToInternalHandle(handle);
    aclrtStream stream = h->stream;
    CscsortTilingData tiling{};
    aclsparseStatus_t status = ComputeCscsortTiling(m, n, nnz, indexBase, tiling);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    OP_LOGI("aclsparseXcscsort", "launching kernel: blockDim=%u, stream=%p", tiling.coreNum, stream);
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    cscsort_kernel_do(reinterpret_cast<GM_ADDR>(const_cast<int *>(cscColPtr)),
                      reinterpret_cast<GM_ADDR>(cscRowInd),
                      reinterpret_cast<GM_ADDR>(P),
                      reinterpret_cast<GM_ADDR>(pBuffer),
                      tiling, tiling.coreNum, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

aclsparseStatus_t aclsparseXcscsort_bufferSizeExt(aclsparseHandle_t handle, int m, int n, int nnz,
                                                   const int *cscColPtr, const int *cscRowInd,
                                                   size_t *pBufferSizeInBytes)
{
    const char *api = "aclsparseXcscsort_bufferSizeExt";
    if (handle == nullptr) {
        OP_LOGE(api, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (pBufferSizeInBytes == nullptr) {
        OP_LOGE(api, "pBufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st = ValidateCscsortCommonParams(api, m, n, nnz, cscColPtr, cscRowInd);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (nnz == 0) {
        *pBufferSizeInBytes = 0U;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    constexpr size_t kFactor = 2U * sizeof(int32_t);
    if (static_cast<size_t>(nnz) > std::numeric_limits<size_t>::max() / kFactor) {
        OP_LOGE(api, "workspace size overflow: nnz=%d", nnz);
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    *pBufferSizeInBytes = static_cast<size_t>(nnz) * kFactor;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseXcscsort(aclsparseHandle_t handle, int m, int n, int nnz,
                                    const aclsparseMatDescr_t descrA, const int *cscColPtr,
                                    int *cscRowInd, int *P, void *pBuffer)
{
    const char *api = "aclsparseXcscsort";
    if (handle == nullptr) {
        OP_LOGE(api, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = ValidateCscsortCommonParams(api, m, n, nnz, cscColPtr, cscRowInd);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (descrA == nullptr) {
        OP_LOGE(api, "descrA is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseIndexBase_t indexBase = aclsparseGetMatIndexBase(descrA);
    if (indexBase != ACL_SPARSE_INDEX_BASE_ZERO && indexBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(api, "invalid indexBase: %d", static_cast<int>(indexBase));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    if (P == nullptr) {
        OP_LOGE(api, "P is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (pBuffer == nullptr) {
        OP_LOGE(api, "pBuffer is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (reinterpret_cast<uintptr_t>(pBuffer) % 128U != 0U) {
        OP_LOGE(api, "pBuffer not 128-byte aligned: %p", pBuffer);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return LaunchCscsortKernel(handle, m, n, nnz, indexBase, cscColPtr, cscRowInd, P, pBuffer);
}

}  // extern "C"
