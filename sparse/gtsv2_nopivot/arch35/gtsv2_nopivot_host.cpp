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
 * \file gtsv2_nopivot_host.cpp
 * \brief aclsparseSgtsv2Nopivot / bufferSizeExt Host 侧实现（SIMT, Thomas）。
 *
 * 结构：Validate + Launch 拆分；dlog 集成。
 *
 * 参考 gtsv_interleaved_batch 的 SIMT 实现模式。
 * 核间切分：numBlocks = min(ceil(n / kGtsv2ThreadsPerBlock), aivCoreNum)。
 * 每个 thread 处理一个 RHS 列，grid-stride 遍历。
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "gtsv2_nopivot_kernel.h"

namespace {

constexpr uint32_t kWorkspaceAlignment = 128u;

// ---------------------------------------------------------------------------
// Parameter validation
// ---------------------------------------------------------------------------
static aclsparseStatus_t ValidateGtsv2NopivotParams(
    aclsparseHandle_t handle,
    int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb, void *pBuffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgtsv2Nopivot", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (m < 3) {
        OP_LOGE("aclsparseSgtsv2Nopivot",
                "invalid m=%d, expected m >= 3", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (n < 1) {
        OP_LOGE("aclsparseSgtsv2Nopivot",
                "invalid n=%d, expected >= 1", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (dl == nullptr || d == nullptr || du == nullptr || B == nullptr) {
        OP_LOGE("aclsparseSgtsv2Nopivot",
                "null pointer: dl=%p, d=%p, du=%p, B=%p",
                static_cast<const void *>(dl),
                static_cast<const void *>(d),
                static_cast<const void *>(du),
                static_cast<void *>(B));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (ldb < std::max(1, m)) {
        OP_LOGE("aclsparseSgtsv2Nopivot",
                "invalid ldb=%d, expected >= max(1, m)=%d",
                ldb, std::max(1, m));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (pBuffer != nullptr &&
        reinterpret_cast<uintptr_t>(pBuffer) % kWorkspaceAlignment != 0) {
        OP_LOGE("aclsparseSgtsv2Nopivot",
                "pBuffer=%p not %u-byte aligned", pBuffer, kWorkspaceAlignment);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (m > 1 && pBuffer == nullptr) {
        OP_LOGE("aclsparseSgtsv2Nopivot",
                "pBuffer is nullptr but m=%d > 1 requires workspace", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Workspace computation (SIMT: m floats per RHS for d' values)
// ---------------------------------------------------------------------------
static size_t ComputeWorkspaceBytes(int m, int n)
{
    if (m <= 1 || n <= 0) {
        return 0;
    }

    // workspace per RHS: m floats (modified d' values from forward sweep)
    size_t wsPerRhs = static_cast<size_t>(m) * sizeof(float);
    size_t totalBytes = wsPerRhs * static_cast<size_t>(n);

    if (n > 0 && totalBytes / static_cast<size_t>(n) != wsPerRhs) {
        OP_LOGE("aclsparseSgtsv2Nopivot",
                "workspace overflow for m=%d, n=%d", m, n);
        return 0;
    }

    const size_t alignMask = static_cast<size_t>(kWorkspaceAlignment) - 1;
    totalBytes = (totalBytes + alignMask) & ~alignMask;

    return totalBytes;
}

// ---------------------------------------------------------------------------
// Kernel launch helper
// ---------------------------------------------------------------------------
static aclsparseStatus_t LaunchGtsv2NopivotKernel(
    aclsparseHandle_t handle,
    int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb,
    void *pBuffer, size_t bufferSize,
    aclrtStream stream)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0u) {
        OP_LOGE("aclsparseSgtsv2Nopivot", "GetAivCoreCount failed");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    // SIMT: numBlocks = ceil(n / kGtsv2ThreadsPerBlock), capped at aivCoreNum
    // (参考 gtsv_interleaved_batch 的 numBlocks 计算)
    uint32_t numBlocks = std::min(
        (static_cast<uint32_t>(n) + kGtsv2ThreadsPerBlock - 1u) / kGtsv2ThreadsPerBlock,
        aivCoreNum);
    if (numBlocks == 0u) {
        numBlocks = 1u;
    }

    // Compute TilingData
    Gtsv2NopivotTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.ldb = ldb;
    tiling.numBlocks = static_cast<int32_t>(numBlocks);

    OP_LOGI("aclsparseSgtsv2Nopivot",
            "launch: m=%d, n=%d, ldb=%d, numBlocks=%u (SIMT), aivCoreNum=%u, wsBytes=%zu",
            m, n, ldb, numBlocks, aivCoreNum, bufferSize);

    // clear-then-launch-then-peek（评审响应）：Peek 只读不清除，线程级 last error
    // 会保留先前已被正确处理的 ACL 调用错误（误报归属错位），且一次失败后线程内
    // 永久粘滞。下发前 aclrtGetLastError 清除，使 Peek 只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    gtsv2_nopivot_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<float *>(dl)),
        reinterpret_cast<GM_ADDR>(const_cast<float *>(d)),
        reinterpret_cast<GM_ADDR>(const_cast<float *>(du)),
        reinterpret_cast<GM_ADDR>(B),
        reinterpret_cast<GM_ADDR>(pBuffer),
        tiling,
        numBlocks,
        stream);

    OP_LOGI("aclsparseSgtsv2Nopivot",
            "Kernel launched, numBlocks=%u", numBlocks);

    // 下发失败不得静默按成功返回（对齐 ops-solver #361/#366 修复口径）：
    // PeekAtLastError 捕获 enqueue 期错误（非法启动配置等）；异步执行错误仍由
    // 调用方同步流后经既有契约观察。
    const aclError enqueueStatus = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (enqueueStatus != ACL_SUCCESS) {
        OP_LOGE("aclsolverSgtsv2Nopivot", "kernel enqueue failed: %d", static_cast<int>(enqueueStatus));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseSgtsv2Nopivot(
    aclsparseHandle_t handle,
    int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb, void *pBuffer)
{
    OP_LOGI("aclsparseSgtsv2Nopivot",
            "entry: m=%d, n=%d, ldb=%d", m, n, ldb);

    aclsparseStatus_t st = ValidateGtsv2NopivotParams(
        handle, m, n, dl, d, du, B, ldb, pBuffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    aclrtStream stream = nullptr;
    st = aclsparseGetStream(handle, &stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    size_t bufferSize = ComputeWorkspaceBytes(m, n);
    if (m > 1 && bufferSize == 0) {
        OP_LOGE("aclsparseSgtsv2Nopivot",
                "workspace size overflow for m=%d, n=%d", m, n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return LaunchGtsv2NopivotKernel(
        handle, m, n, dl, d, du,
        B, ldb, pBuffer, bufferSize, stream);
}

aclsparseStatus_t aclsparseSgtsv2Nopivot_bufferSizeExt(
    aclsparseHandle_t handle,
    int m, int n,
    const float *dl, const float *d, const float *du,
    const float *B, int ldb,
    size_t *pBufferSizeInBytes)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgtsv2Nopivot_bufferSizeExt",
                "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (pBufferSizeInBytes == nullptr) {
        OP_LOGE("aclsparseSgtsv2Nopivot_bufferSizeExt",
                "pBufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (dl == nullptr || d == nullptr || du == nullptr || B == nullptr) {
        OP_LOGE("aclsparseSgtsv2Nopivot_bufferSizeExt",
                "null pointer: dl=%p, d=%p, du=%p, B=%p",
                static_cast<const void *>(dl),
                static_cast<const void *>(d),
                static_cast<const void *>(du),
                static_cast<const void *>(B));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (m < 3 || n < 1) {
        OP_LOGE("aclsparseSgtsv2Nopivot_bufferSizeExt",
                "invalid m=%d or n=%d", m, n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (ldb < std::max(1, m)) {
        OP_LOGE("aclsparseSgtsv2Nopivot_bufferSizeExt",
                "invalid ldb=%d, expected >= max(1, m)=%d",
                ldb, std::max(1, m));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    size_t wsz = ComputeWorkspaceBytes(m, n);
    if (m > 1 && wsz == 0) {
        OP_LOGE("aclsparseSgtsv2Nopivot_bufferSizeExt",
                "workspace overflow for m=%d, n=%d", m, n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *pBufferSizeInBytes = wsz;

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
