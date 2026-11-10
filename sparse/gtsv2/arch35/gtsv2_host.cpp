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
 * \file gtsv2_host.cpp
 * \brief aclsparseSgtsv2 / bufferSizeExt Host 侧实现（SIMT, Thomas + partial pivoting）。
 *
 * 结构：Validate + ComputeWorkspaceBytes + Launch 拆分；dlog 集成。
 *
 * 参考 gtsv2_nopivot 的 SIMT 实现模式，关键区别：
 *   - workspace 为 kGtsv2WorkspaceArraysPerRhs*m floats/RHS（d' + du' + du2'），是 nopivot 的 3 倍
 *   - n == 0 时提前返回 SUCCESS（nopivot 要求 n >= 1）
 *   - bufferSizeExt 允许 B == NULL（查询接口不使用 B 的值）
 *
 * 核间切分：numBlocks = min(ceil(n / kGtsv2PivotThreadsPerBlock), aivCoreNum)。
 * 每个 thread 处理一个 RHS 列，grid-stride 遍历。
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "gtsv2_kernel.h"

namespace {

constexpr uint32_t kWorkspaceAlignment = 128u;
// Number of workspace arrays per RHS column: d' (modified main diagonal),
// du' (modified super-diagonal with fill-in), du2' (2nd super-diagonal fill-in).
constexpr size_t kGtsv2WorkspaceArraysPerRhs = 3;

// ---------------------------------------------------------------------------
// Parameter validation (for aclsparseSgtsv2 main API)
// Caller (aclsparseSgtsv2) handles handle-NULL and n == 0 early return before
// calling this; hence handle is not re-checked here (no duplicate check).
// ---------------------------------------------------------------------------
static aclsparseStatus_t ValidateGtsv2Params(
    int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb, void *pBuffer)
{
    // n < 0 is invalid; n == 0 is handled by caller (early return)
    if (n < 0) {
        OP_LOGE("aclsparseSgtsv2",
                "invalid n=%d, expected >= 0", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (m < 3) {
        OP_LOGE("aclsparseSgtsv2",
                "invalid m=%d, expected m >= 3", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (ldb < std::max(1, m)) {
        OP_LOGE("aclsparseSgtsv2",
                "invalid ldb=%d, expected >= max(1, m)=%d",
                ldb, std::max(1, m));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // When n > 0, check data pointers
    if (n > 0) {
        if (dl == nullptr || d == nullptr || du == nullptr || B == nullptr) {
            OP_LOGE("aclsparseSgtsv2",
                    "null pointer: dl=%p, d=%p, du=%p, B=%p",
                    static_cast<const void *>(dl),
                    static_cast<const void *>(d),
                    static_cast<const void *>(du),
                    static_cast<void *>(B));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }

        if (pBuffer == nullptr) {
            OP_LOGE("aclsparseSgtsv2",
                     "pBuffer is nullptr but m=%d >= 3 requires workspace", m);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }

        if (reinterpret_cast<uintptr_t>(pBuffer) % kWorkspaceAlignment != 0) {
            OP_LOGE("aclsparseSgtsv2",
                    "pBuffer=%p not %u-byte aligned", pBuffer, kWorkspaceAlignment);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Workspace computation (SIMT: kGtsv2WorkspaceArraysPerRhs*m floats per RHS for d' + du' + du2' values)
// du' and du2' include fill-in from partial pivoting swaps (3x nopivot workspace)
// ---------------------------------------------------------------------------
static size_t ComputeWorkspaceBytes(int m, int n)
{
    if (n <= 0) {
        return 0;
    }

    // workspace per RHS: d' + du' + du2' (3 arrays, partial pivoting fill-in)
    size_t wsPerRhs = kGtsv2WorkspaceArraysPerRhs * static_cast<size_t>(m) * sizeof(float);
    size_t totalBytes = wsPerRhs * static_cast<size_t>(n);

    // Overflow check
    if (totalBytes / static_cast<size_t>(n) != wsPerRhs) {
        OP_LOGE("aclsparseSgtsv2",
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
static aclsparseStatus_t LaunchGtsv2Kernel(
    aclsparseHandle_t handle,
    int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb,
    void *pBuffer, size_t bufferSize,
    aclrtStream stream)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0u) {
        OP_LOGE("aclsparseSgtsv2", "GetAivCoreCount failed");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    // SIMT: numBlocks = ceil(n / kGtsv2PivotThreadsPerBlock), capped at aivCoreNum
    uint32_t numBlocks = std::min(
        (static_cast<uint32_t>(n) + kGtsv2PivotThreadsPerBlock - 1u) / kGtsv2PivotThreadsPerBlock,
        aivCoreNum);
    if (numBlocks == 0u) {
        numBlocks = 1u;
    }

    // Compute TilingData
    Gtsv2TilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.ldb = ldb;
    tiling.numBlocks = static_cast<int32_t>(numBlocks);

    OP_LOGI("aclsparseSgtsv2",
            "launch: m=%d, n=%d, ldb=%d, numBlocks=%u (SIMT), aivCoreNum=%u, wsBytes=%zu",
            m, n, ldb, numBlocks, aivCoreNum, bufferSize);

    // clear-then-launch-then-peek（评审响应）：Peek 只读不清除，线程级 last error
    // 会保留先前已被正确处理的 ACL 调用错误（误报归属错位），且一次失败后线程内
    // 永久粘滞。下发前 aclrtGetLastError 清除，使 Peek 只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    gtsv2_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<float *>(dl)),
        reinterpret_cast<GM_ADDR>(const_cast<float *>(d)),
        reinterpret_cast<GM_ADDR>(const_cast<float *>(du)),
        reinterpret_cast<GM_ADDR>(B),
        reinterpret_cast<GM_ADDR>(pBuffer),
        tiling,
        numBlocks,
        stream);

    OP_LOGI("aclsparseSgtsv2",
            "Kernel launched, numBlocks=%u", numBlocks);

    // 下发失败不得静默按成功返回（对齐 ops-solver #361/#366 修复口径）：
    // PeekAtLastError 捕获 enqueue 期错误（非法启动配置等）；异步执行错误仍由
    // 调用方同步流后经既有契约观察。
    const aclError enqueueStatus = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (enqueueStatus != ACL_SUCCESS) {
        OP_LOGE("aclsparseSgtsv2", "kernel enqueue failed: %d", static_cast<int>(enqueueStatus));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Parameter validation (for aclsparseSgtsv2_bufferSizeExt)
// Caller checks handle-NULL first; this validates the remaining params.
// n == 0 short-circuits to *pBufferSizeInBytes = 0 (no workspace) before the
// matrix-shape checks, matching the query-API contract (B may be NULL).
// ---------------------------------------------------------------------------
static aclsparseStatus_t ValidateGtsv2BufferSizeParams(
    int m, int n,
    const float *dl, const float *d, const float *du,
    int ldb, size_t *pBufferSizeInBytes)
{
    if (pBufferSizeInBytes == nullptr) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt",
                "pBufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (n < 0) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt",
                "invalid n=%d, expected >= 0", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // n == 0: no workspace needed; short-circuit before matrix-shape checks
    if (n == 0) {
        *pBufferSizeInBytes = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    if (m < 3) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt",
                "invalid m=%d, expected m >= 3", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (ldb < std::max(1, m)) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt",
                "invalid ldb=%d, expected >= max(1, m)=%d",
                ldb, std::max(1, m));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // dl/d/du must be non-NULL when n > 0; B may be NULL (query API)
    if (dl == nullptr || d == nullptr || du == nullptr) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt",
                "null pointer: dl=%p, d=%p, du=%p",
                static_cast<const void *>(dl),
                static_cast<const void *>(d),
                static_cast<const void *>(du));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseSgtsv2(
    aclsparseHandle_t handle,
    int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb, void *pBuffer)
{
    OP_LOGI("aclsparseSgtsv2",
            "entry: m=%d, n=%d, ldb=%d", m, n, ldb);

    // Handle check first (most fundamental)
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgtsv2", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    // n == 0: early return SUCCESS, no computation
    if (n == 0) {
        OP_LOGI("aclsparseSgtsv2", "n=0, early return without computation");
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    aclsparseStatus_t st = ValidateGtsv2Params(
        m, n, dl, d, du, B, ldb, pBuffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    aclrtStream stream = nullptr;
    st = aclsparseGetStream(handle, &stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    size_t bufferSize = ComputeWorkspaceBytes(m, n);
    if (bufferSize == 0) {
        OP_LOGE("aclsparseSgtsv2",
                "workspace size overflow for m=%d, n=%d", m, n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return LaunchGtsv2Kernel(
        handle, m, n, dl, d, du,
        B, ldb, pBuffer, bufferSize, stream);
}

aclsparseStatus_t aclsparseSgtsv2_bufferSizeExt(
    aclsparseHandle_t handle,
    int m, int n,
    const float *dl, const float *d, const float *du,
    const float *B, int ldb,
    size_t *pBufferSizeInBytes)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt",
                "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = ValidateGtsv2BufferSizeParams(
        m, n, dl, d, du, ldb, pBufferSizeInBytes);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // n == 0: Validate already set *pBufferSizeInBytes = 0
    if (n == 0) {
        OP_LOGI("aclsparseSgtsv2_bufferSizeExt",
                "n=0, workspace size = 0");
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    size_t wsz = ComputeWorkspaceBytes(m, n);
    if (wsz == 0) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt",
                "workspace overflow for m=%d, n=%d", m, n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *pBufferSizeInBytes = wsz;

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
