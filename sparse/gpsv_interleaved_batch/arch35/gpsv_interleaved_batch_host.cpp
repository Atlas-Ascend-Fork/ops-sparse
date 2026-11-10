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
 * \file gpsv_interleaved_batch_host.cpp
 * \brief aclsparseSgpsvInterleavedBatch / bufferSizeExt Host-side implementation (SIMT, QR/Givens).
 *
 * QR/Givens rotation solver: workspace required for R factor storage.
 * d/dl/ds/du/dw are read-only (not modified). x holds RHS on entry, solution on exit.
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "gpsv_interleaved_batch_kernel.h"

namespace {

constexpr uint32_t kWorkspaceAlignment = 128u;

static aclsparseStatus_t ValidateGpsvCoreParams(const char *tag, int algo, int m, int batchCount)
{
    if (algo < 0 || algo > 2) {
        OP_LOGE(tag, "invalid algo=%d, expected [0, 2]", algo);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (algo != 0) {
        OP_LOGE(tag, "algo=%d not supported, only algo=0 (QR/Givens) is implemented", algo);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (m < 1) {
        OP_LOGE(tag, "invalid m=%d, expected m >= 1", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (batchCount < 1) {
        OP_LOGE(tag, "invalid batchCount=%d, expected >= 1", batchCount);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateGpsvInterleavedBatchParams(
    aclsparseHandle_t handle,
    int algo, int m,
    const float *ds, const float *dl, const float *d, const float *du,
    const float *dw, const float *x,
    int batchCount, void *pBuffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgpsvInterleavedBatch", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = ValidateGpsvCoreParams(
        "aclsparseSgpsvInterleavedBatch", algo, m, batchCount);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    if (ds == nullptr || dl == nullptr || d == nullptr ||
        du == nullptr || dw == nullptr || x == nullptr) {
        OP_LOGE("aclsparseSgpsvInterleavedBatch",
                "null pointer: ds=%p, dl=%p, d=%p, du=%p, dw=%p, x=%p",
                static_cast<const void *>(ds),
                static_cast<const void *>(dl),
                static_cast<const void *>(d),
                static_cast<const void *>(du),
                static_cast<const void *>(dw),
                static_cast<const void *>(x));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (m > 1) {
        if (pBuffer == nullptr) {
            OP_LOGE("aclsparseSgpsvInterleavedBatch",
                    "pBuffer is nullptr but workspace is required for m=%d", m);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        if (reinterpret_cast<uintptr_t>(pBuffer) % kWorkspaceAlignment != 0) {
            OP_LOGE("aclsparseSgpsvInterleavedBatch",
                    "pBuffer %p not %u-byte aligned", pBuffer, kWorkspaceAlignment);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

static size_t ComputeAlignedSegFloats(int m, int batchCount)
{
    constexpr uint32_t kSegAlignFloats = kWorkspaceAlignment / sizeof(float);
    int64_t segFloats = static_cast<int64_t>(m) * static_cast<int64_t>(batchCount);
    return static_cast<size_t>(
        ((segFloats + kSegAlignFloats - 1) / kSegAlignFloats) * kSegAlignFloats);
}

static size_t ComputeWorkspaceBytes(int m, int batchCount)
{
    if (m <= 1) {
        return 0;
    }
    size_t segAligned = ComputeAlignedSegFloats(m, batchCount);
    constexpr size_t kMultiplier = 5 * sizeof(float);
    constexpr size_t kMaxWorkspaceBytes = static_cast<size_t>(1) << 40;
    constexpr size_t kMaxSegAligned = kMaxWorkspaceBytes / kMultiplier;
    if (segAligned > kMaxSegAligned) {
        OP_LOGE("ComputeWorkspaceBytes",
                "workspace overflow: segAligned=%zu exceeds limit %zu", segAligned, kMaxSegAligned);
        return 0;
    }
    return segAligned * kMultiplier;
}

static aclsparseStatus_t LaunchGpsvInterleavedBatchKernel(
    aclsparseHandle_t handle, int m,
    float *ds, float *dl, float *d, float *du, float *dw, float *x,
    int batchCount, void *pBuffer, size_t bufferSize, aclrtStream stream)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0u) {
        OP_LOGE("aclsparseSgpsvInterleavedBatch", "GetAivCoreCount failed");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    // SIMT: numBlocks = ceil(batchCount / kGpsvBatchPerBlock), capped at AIV core count
    uint32_t numBlocks = std::min(
        (static_cast<uint32_t>(batchCount) + kGpsvBatchPerBlock - 1u) / kGpsvBatchPerBlock,
        aivCoreNum);
    // batchCount >= 1 (validated above), so numBlocks >= 1.

    OP_LOGI("aclsparseSgpsvInterleavedBatch",
            "launch: m=%d, batchCount=%d, numBlocks=%u (SIMT/QR), wsBytes=%zu, aivCoreNum=%u",
            m, batchCount, numBlocks, bufferSize, aivCoreNum);

    GpsvInterleavedBatchTilingData tiling{};
    tiling.m = m;
    tiling.batchCount = batchCount;
    tiling.wsStride = static_cast<int64_t>(ComputeAlignedSegFloats(m, batchCount));

    GM_ADDR wsAddr = (m > 1) ? reinterpret_cast<GM_ADDR>(pBuffer) : nullptr;

    // No workspace memset needed: the QR/Givens forward loop in the kernel writes
    // wsD4[0..m-3] and wsD5[0..m-3], and the backward substitution only reads
    // wsD4 at indices 0..m-4 and wsD5 at indices 0..m-5 — fully covered by the
    // writes. Positions m-2 and m-1 are never read by back-substitution for d4'/d5'.

    // clear-then-launch-then-peek（评审响应）：Peek 只读不清除，线程级 last error
    // 会保留先前已被正确处理的 ACL 调用错误（误报归属错位），且一次失败后线程内
    // 永久粘滞。下发前 aclrtGetLastError 清除，使 Peek 只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    gpsv_interleaved_batch_kernel_do(
        reinterpret_cast<GM_ADDR>(ds),
        reinterpret_cast<GM_ADDR>(dl),
        reinterpret_cast<GM_ADDR>(d),
        reinterpret_cast<GM_ADDR>(du),
        reinterpret_cast<GM_ADDR>(dw),
        reinterpret_cast<GM_ADDR>(x),
        wsAddr,
        tiling,
        numBlocks,
        stream);

    OP_LOGI("aclsparseSgpsvInterleavedBatch",
            "Kernel launched, numBlocks=%u", numBlocks);
    // 下发失败不得静默按成功返回（对齐 ops-solver #361/#366 修复口径）：
    // PeekAtLastError 捕获 enqueue 期错误（非法启动配置等）；异步执行错误仍由
    // 调用方同步流后经既有契约观察。
    const aclError enqueueStatus = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (enqueueStatus != ACL_SUCCESS) {
        OP_LOGE("aclsparseSgpsvInterleavedBatch", "kernel enqueue failed: %d", static_cast<int>(enqueueStatus));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseSgpsvInterleavedBatch(
    aclsparseHandle_t handle,
    int algo, int m,
    float *ds, float *dl, float *d, float *du, float *dw, float *x,
    int batchCount, void *pBuffer)
{
    OP_LOGI("aclsparseSgpsvInterleavedBatch",
            "entry: algo=%d, m=%d, batchCount=%d", algo, m, batchCount);

    aclsparseStatus_t st = ValidateGpsvInterleavedBatchParams(
        handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    aclrtStream stream = nullptr;
    st = aclsparseGetStream(handle, &stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    size_t bufferSize = ComputeWorkspaceBytes(m, batchCount);
    if (m > 1 && bufferSize == 0) {
        OP_LOGE("aclsparseSgpsvInterleavedBatch",
                "workspace size overflow for m=%d, batchCount=%d", m, batchCount);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return LaunchGpsvInterleavedBatchKernel(
        handle, m, ds, dl, d, du, dw, x, batchCount, pBuffer, bufferSize, stream);
}

aclsparseStatus_t aclsparseSgpsvInterleavedBatch_bufferSizeExt(
    aclsparseHandle_t handle,
    int algo, int m,
    const float *ds, const float *dl, const float *d, const float *du,
    const float *dw, const float *x,
    int batchCount, size_t *pBufferSizeInBytes)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgpsvInterleavedBatch_bufferSizeExt",
                "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (pBufferSizeInBytes == nullptr) {
        OP_LOGE("aclsparseSgpsvInterleavedBatch_bufferSizeExt",
                "pBufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st = ValidateGpsvCoreParams(
        "aclsparseSgpsvInterleavedBatch_bufferSizeExt", algo, m, batchCount);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    size_t wsz = ComputeWorkspaceBytes(m, batchCount);
    if (m > 1 && wsz == 0) {
        OP_LOGE("aclsparseSgpsvInterleavedBatch_bufferSizeExt",
                "workspace size overflow for m=%d, batchCount=%d", m, batchCount);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *pBufferSizeInBytes = wsz;

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
