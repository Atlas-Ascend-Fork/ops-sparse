/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file gtsv_interleaved_batch_host.cpp
 * \brief aclsparseSgtsvInterleavedBatch / bufferSizeExt Host 侧实现（SIMT 模型）。
 *
 * 结构：Validate + Launch 拆分；dlog 集成。
 *
 * SIMT 模型下：
 *   - blockDim = ceil(batchCount / kGtsvBatchPerBlock)（每 block 的线程数
 *     与 kernel.cpp 的 SIMT block 宽度共享同一常量；详见 gtsv_interleaved_batch_kernel.h）
 *   - workspace 仅需 m × batchCount × sizeof(float)（仅存 d'，不存 b'；b' 存入 x 原位）
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
// 提供 GetAivCoreCount()：获取 NPU 上 AI Core 的物理核数，用于把启动的
// block 数约束到硬件能力内（详见 LaunchGtsvInterleavedBatchKernel）。
#include "aclsparse_host_utils.h"
#include "gtsv_interleaved_batch_kernel.h"

namespace {

// 工作区缓冲区的字节对齐要求。pBuffer 与 bufferSizeInBytes 均遵守此值。
// 与 cuSPARSE 接口的 128 字节对齐要求保持一致。修改此处时需同步更新
// Validate 中的 pBuffer 对齐检查与 bufferSizeExt 中的 round-up。
constexpr uint32_t kWorkspaceAlignment = 128u;

static aclsparseStatus_t ValidateGtsvInterleavedBatchParams(
    aclsparseHandle_t handle,
    int algo, int m,
    const float *dl, const float *d, const float *du,
    float *x, int batchCount, void *pBuffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (algo < 0 || algo > 2) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch",
                "invalid algo=%d, expected [0, 2]", algo);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (algo != 0) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch",
                "algo=%d not supported, only algo=0 (Thomas) is implemented", algo);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    if (m < 1) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch",
                "invalid m=%d, expected m >= 1", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (batchCount < 1) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch",
                "invalid batchCount=%d, expected >= 1", batchCount);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (dl == nullptr || d == nullptr || du == nullptr || x == nullptr) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch",
                "null pointer: dl=%p, d=%p, du=%p, x=%p",
                static_cast<const void *>(dl),
                static_cast<const void *>(d),
                static_cast<const void *>(du),
                static_cast<void *>(x));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (pBuffer != nullptr &&
        reinterpret_cast<uintptr_t>(pBuffer) % kWorkspaceAlignment != 0) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch",
                "pBuffer=%p not %u-byte aligned", pBuffer, kWorkspaceAlignment);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (m > 1 && pBuffer == nullptr) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch",
                "pBuffer is null but m=%d > 1 requires workspace", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchGtsvInterleavedBatchKernel(
    aclsparseHandle_t handle, int m,
    float *dl, float *d, float *du, float *x,
    int batchCount, void *pBuffer, aclrtStream stream)
{
    // 取硬件真实 AIV 核数，避免 batch 很大时 numBlocks 超出物理 AI Core 数量，
    // 导致部分 block 永远调度不到或触发硬件调度失败。超出核数的 block 只会让
    // inactive threads 的 early-return 在同一个 core 上重入执行，没有收益。
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0u) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch", "GetAivCoreCount failed");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    // SIMT: blockDim = ceil(batchCount / kGtsvBatchPerBlock)
    // kGtsvBatchPerBlock 在 gtsv_interleaved_batch_kernel.h 中定义，
    // 与 kernel.cpp 的 SIMT block 宽度共享同一编译期常量。
    uint32_t numBlocks = std::min(
        (static_cast<uint32_t>(batchCount) + kGtsvBatchPerBlock - 1u) / kGtsvBatchPerBlock,
        aivCoreNum);
    // batchCount >= 1（Validate 已校验），所以 numBlocks 至少为 1。
    if (numBlocks == 0u) {
        numBlocks = 1u;
    }

    OP_LOGI("aclsparseSgtsvInterleavedBatch",
            "launch: m=%d, batchCount=%d, numBlocks=%u (SIMT), aivCoreNum=%u",
            m, batchCount, numBlocks, aivCoreNum);

    GtsvInterleavedBatchTilingData tiling{};
    tiling.m = m;
    tiling.batchCount = batchCount;

    // clear-then-launch-then-peek（评审响应）：Peek 只读不清除，线程级 last error
    // 会保留先前已被正确处理的 ACL 调用错误（误报归属错位），且一次失败后线程内
    // 永久粘滞。下发前 aclrtGetLastError 清除，使 Peek 只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    gtsv_interleaved_batch_kernel_do(
        reinterpret_cast<GM_ADDR>(dl),
        reinterpret_cast<GM_ADDR>(d),
        reinterpret_cast<GM_ADDR>(du),
        reinterpret_cast<GM_ADDR>(x),
        reinterpret_cast<GM_ADDR>(pBuffer),
        tiling,
        numBlocks,
        stream);

    OP_LOGI("aclsparseSgtsvInterleavedBatch",
            "Kernel launched, numBlocks=%u", numBlocks);
    // 下发失败不得静默按成功返回（对齐 ops-solver #361/#366 修复口径）：
    // PeekAtLastError 捕获 enqueue 期错误（非法启动配置等）；异步执行错误仍由
    // 调用方同步流后经既有契约观察。
    const aclError enqueueStatus = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (enqueueStatus != ACL_SUCCESS) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch", "kernel enqueue failed: %d", static_cast<int>(enqueueStatus));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseSgtsvInterleavedBatch(
    aclsparseHandle_t handle,
    int algo, int m,
    float *dl, float *d, float *du, float *x,
    int batchCount, void *pBuffer)
{
    OP_LOGI("aclsparseSgtsvInterleavedBatch",
            "entry: algo=%d, m=%d, batchCount=%d", algo, m, batchCount);

    // ValidateGtsvInterleavedBatchParams checks handle==nullptr with OP_LOGE
    // and returns ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR. Do NOT short-circuit
    // the null check here — that would silently return without any log.
    aclsparseStatus_t st = ValidateGtsvInterleavedBatchParams(
        handle, algo, m, dl, d, du, x, batchCount, pBuffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    aclrtStream stream = nullptr;
    st = aclsparseGetStream(handle, &stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    return LaunchGtsvInterleavedBatchKernel(
        handle, m, dl, d, du, x, batchCount, pBuffer, stream);
}

aclsparseStatus_t aclsparseSgtsvInterleavedBatch_bufferSizeExt(
    aclsparseHandle_t handle,
    int algo, int m,
    const float *dl, const float *d, const float *du,
    const float *x, int batchCount,
    size_t *pBufferSizeInBytes)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch_bufferSizeExt",
                "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (pBufferSizeInBytes == nullptr) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch_bufferSizeExt",
                "pBufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (algo < 0 || algo > 2) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch_bufferSizeExt",
                "invalid algo=%d", algo);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (algo != 0) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch_bufferSizeExt",
                "algo=%d not supported, only algo=0 (Thomas) is implemented",
                algo);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    if (m < 1 || batchCount < 1) {
        OP_LOGE("aclsparseSgtsvInterleavedBatch_bufferSizeExt",
                "invalid m=%d or batchCount=%d", m, batchCount);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (m == 1) {
        *pBufferSizeInBytes = 0;
    } else {
        // SIMT: workspace 只存 d' (每行一份), b' 存入 x 原位
        size_t wsBytes = static_cast<size_t>(m) *
            static_cast<size_t>(batchCount) * sizeof(float);
        const size_t roundUp = static_cast<size_t>(kWorkspaceAlignment);
        wsBytes = ((wsBytes + roundUp - 1u) / roundUp) * roundUp;
        *pBufferSizeInBytes = wsBytes;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
