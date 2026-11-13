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
 * \file nnz_host.cpp
 * \brief aclsparseSnnz Host 侧实现：参数校验 + Kernel launch。
 */

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "nnz_tiling_data.h"
#include "nnz_kernel.h"

inline struct aclsparseContext *ToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

namespace {

/**
 * @brief 校验 aclsparseSnnz 全部入参。
 */
static aclsparseStatus_t ValidateNnzParams(
    aclsparseHandle_t handle,
    aclsparseDirection_t dirA,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const float *A, int lda,
    int *nnzPerRowColumn,
    int *nnzTotalDevHostPtr)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSnnz", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (dirA != ACL_SPARSE_DIRECTION_ROW && dirA != ACL_SPARSE_DIRECTION_COLUMN) {
        OP_LOGE("aclsparseSnnz", "invalid dirA: %d", static_cast<int>(dirA));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (m < 0) {
        OP_LOGE("aclsparseSnnz", "invalid m: %d", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (n < 0) {
        OP_LOGE("aclsparseSnnz", "invalid n: %d", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t descrStatus = AclsparseValidateMatDescr("aclsparseSnnz", "descrA", descrA);
    if (descrStatus != ACL_SPARSE_STATUS_SUCCESS) {
        return descrStatus;
    }

    // lda >= max(1, m)：列主序下 lda 为列间距，允许 lda > m（padding / 子矩阵场景）。
    int minLda = std::max(1, m);
    if (lda < minLda) {
        OP_LOGE("aclsparseSnnz", "invalid lda: %d, expected >= %d", lda, minLda);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (m > 0 && n > 0 && A == nullptr) {
        OP_LOGE("aclsparseSnnz", "A is nullptr while m=%d, n=%d", m, n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    int32_t totalUnits = (dirA == ACL_SPARSE_DIRECTION_ROW) ? m : n;
    if (totalUnits > 0 && nnzPerRowColumn == nullptr) {
        OP_LOGE("aclsparseSnnz", "nnzPerRowColumn is nullptr while totalUnits=%d", totalUnits);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (nnzTotalDevHostPtr == nullptr) {
        OP_LOGE("aclsparseSnnz", "nnzTotalDevHostPtr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

/**
 * @brief 获取 device 指针用于存储 nnzTotal（根据 pointer mode 选择 workspace 或用户指针）
 */
static aclsparseStatus_t GetNnzTotalDevPtr(
    aclsparseContext *h, int *nnzTotalDevHostPtr, int **devTotalPtr)
{
    bool isHostPtr = (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST);
    if (isHostPtr) {
        size_t requiredBytes = sizeof(int32_t);
        size_t availableBytes = aclsparseGetEffectiveWorkspaceSize(h);
        CHECK_RET(requiredBytes <= availableBytes,
                  OP_LOGE("aclsparseSnnz", "workspace %zu > handle %zu",
                          requiredBytes, availableBytes);
                  return ACL_SPARSE_STATUS_EXECUTION_FAILED);
        *devTotalPtr = static_cast<int *>(aclsparseGetEffectiveWorkspace(h));
    } else {
        *devTotalPtr = nnzTotalDevHostPtr;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/**
 * @brief 计算 Tiling 参数并启动 kernel
 */
static aclsparseStatus_t ComputeTilingAndLaunch(
    aclsparseContext *h, aclsparseDirection_t dirA,
    int m, int n, const float *A, int lda,
    int *nnzPerRowColumn, int *devTotalPtr, aclrtStream stream)
{
    int32_t totalUnits = (dirA == ACL_SPARSE_DIRECTION_ROW) ? m : n;
    int32_t unitSize = (dirA == ACL_SPARSE_DIRECTION_ROW) ? n : m;

    NnzTilingData tiling;
    tiling.lda = static_cast<int32_t>(lda);
    tiling.dirA = static_cast<int32_t>(dirA);
    tiling.totalUnits = totalUnits;
    tiling.unitSize = unitSize;

    uint32_t maxCoreNum = GetAivCoreCount();
    CHECK_RET(maxCoreNum > 0,
              OP_LOGE("aclsparseSnnz", "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    uint32_t numBlocks = CeilDiv<uint32_t>(totalUnits, kNnzThreadsPerBlock);
    if (numBlocks > maxCoreNum) {
        numBlocks = maxCoreNum;
    }

    OP_LOGD("aclsparseSnnz",
            "Tiling: lda=%d, dirA=%d, totalUnits=%d, unitSize=%d, numBlocks=%u, ptrMode=%s",
            tiling.lda, tiling.dirA, tiling.totalUnits, tiling.unitSize, numBlocks,
            (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST) ? "HOST" : "DEVICE");

    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    nnz_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<float *>(A)),
        reinterpret_cast<GM_ADDR>(nnzPerRowColumn),
        reinterpret_cast<GM_ADDR>(devTotalPtr),
        tiling, numBlocks, stream);

    OP_LOGI("aclsparseSnnz", "Kernel launched, numBlocks=%u", numBlocks);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseSnnz", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

/**
 * @brief 计算 Tiling 参数并异步启动 kernel。
 *
 * 根据 pointer mode 处理 nnzTotalDevHostPtr：
 * - DEVICE mode：直接清零 + kernel 原子累加（全异步）
 * - HOST mode：使用 workspace 作为临时 device buffer，kernel 写入后 D2H 回写
 */
static aclsparseStatus_t LaunchNnzKernel(
    aclsparseHandle_t handle,
    aclsparseDirection_t dirA,
    int m, int n,
    const float *A, int lda,
    int *nnzPerRowColumn,
    int *nnzTotalDevHostPtr)
{
    auto *h = ToInternalHandle(handle);
    aclrtStream stream = h->stream;
    // stream 为 nullptr 时使用默认 stream（符合 aclsparseSetStream 文档约定）

    // 空矩阵提前返回：避免不必要的 workspace 操作
    if (m == 0 || n == 0) {
        OP_LOGI("aclsparseSnnz", "empty matrix (m=%d, n=%d), skip kernel launch", m, n);
        if (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
            *nnzTotalDevHostPtr = 0;
        } else {
            aclError aclRet = aclrtMemsetAsync(nnzTotalDevHostPtr, sizeof(int32_t), 0, sizeof(int32_t), stream);
            CHECK_RET(aclRet == ACL_SUCCESS,
                      OP_LOGE("aclsparseSnnz", "aclrtMemsetAsync failed, ret=%d", aclRet);
                      return ACL_SPARSE_STATUS_EXECUTION_FAILED);
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    int *devTotalPtr = nullptr;
    aclsparseStatus_t status = GetNnzTotalDevPtr(h, nnzTotalDevHostPtr, &devTotalPtr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    aclError aclRet = aclrtMemsetAsync(devTotalPtr, sizeof(int32_t), 0, sizeof(int32_t), stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              OP_LOGE("aclsparseSnnz", "aclrtMemsetAsync failed, ret=%d", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    status = ComputeTilingAndLaunch(h, dirA, m, n, A, lda, nnzPerRowColumn, devTotalPtr, stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    if (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        aclRet = aclrtMemcpyAsync(nnzTotalDevHostPtr, sizeof(int32_t),
                                   devTotalPtr, sizeof(int32_t),
                                   ACL_MEMCPY_DEVICE_TO_HOST, stream);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclsparseSnnz", "aclrtMemcpyAsync D2H failed, ret=%d", aclRet);
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

// -------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------
extern "C" {

aclsparseStatus_t aclsparseSnnz(
    aclsparseHandle_t handle,
    aclsparseDirection_t dirA,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const float *A, int lda,
    int *nnzPerRowColumn,
    int *nnzTotalDevHostPtr)
{
    aclsparseStatus_t status = ValidateNnzParams(
        handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    OP_LOGD("aclsparseSnnz",
            "params validated: m=%d, n=%d, lda=%d, dirA=%d, indexBase=%d",
            m, n, lda, static_cast<int>(dirA),
            static_cast<int>(aclsparseGetMatIndexBase(descrA)));

    return LaunchNnzKernel(handle, dirA, m, n, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
}

} // extern "C"
