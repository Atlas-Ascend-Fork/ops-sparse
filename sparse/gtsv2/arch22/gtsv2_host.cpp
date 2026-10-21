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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "gtsv2_kernel.h"

namespace {

constexpr size_t kWorkspaceAlignment = 128;

static bool ComputeWorkspaceBytes(int m, int n, size_t &workspaceBytes)
{
    if (n <= 0) {
        workspaceBytes = 0;
        return true;
    }

    constexpr size_t bytesPerRow = static_cast<size_t>(kGtsv2PackedFactorBytesPerRow);
    const size_t rows = static_cast<size_t>(m);
    constexpr size_t scratchBytesPerRow = kGtsv2RangeScratchBytesPerRow;
    if (rows > (std::numeric_limits<size_t>::max() -
                2 * (kWorkspaceAlignment - 1)) / (bytesPerRow + scratchBytesPerRow)) {
        return false;
    }
    const size_t rawBytes = rows * bytesPerRow;
    const size_t alignedFactors = (rawBytes + kWorkspaceAlignment - 1) &
                                  ~(kWorkspaceAlignment - 1);
    workspaceBytes = (alignedFactors + scratchBytesPerRow * rows + kWorkspaceAlignment - 1) &
                     ~(kWorkspaceAlignment - 1);
    return workspaceBytes >= rawBytes;
}

static bool IsBSpanRepresentable(int ldb, int n)
{
    if (n <= 0) {
        return true;
    }
    const uint64_t elements = static_cast<uint64_t>(ldb) *
                              static_cast<uint64_t>(n);
    return elements <= std::numeric_limits<size_t>::max() / sizeof(float) &&
           elements <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
}

static aclsparseStatus_t ValidateComputeParams(
    int m, int n, const float *dl, const float *d, const float *du,
    float *B, int ldb, void *workspace)
{
    if (n < 0 || m < 3 || ldb < std::max(1, m)) {
        OP_LOGE("aclsparseSgtsv2",
                "invalid shape: m=%d, n=%d, ldb=%d", m, n, ldb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (dl == nullptr || d == nullptr || du == nullptr || B == nullptr ||
        workspace == nullptr) {
        OP_LOGE("aclsparseSgtsv2",
                "null pointer: dl=%p, d=%p, du=%p, B=%p, pBuffer=%p",
                static_cast<const void *>(dl), static_cast<const void *>(d),
                static_cast<const void *>(du), static_cast<void *>(B), workspace);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if ((reinterpret_cast<uintptr_t>(workspace) &
         (kWorkspaceAlignment - 1)) != 0) {
        OP_LOGE("aclsparseSgtsv2", "pBuffer=%p is not 128-byte aligned",
                workspace);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    size_t workspaceBytes = 0;
    if (!ComputeWorkspaceBytes(m, n, workspaceBytes) ||
        !IsBSpanRepresentable(ldb, n)) {
        OP_LOGE("aclsparseSgtsv2",
                "size overflow: m=%d, n=%d, ldb=%d", m, n, ldb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateBufferSizeParams(
    int m, int n, const float *dl, const float *d, const float *du,
    int ldb, size_t *workspaceBytes)
{
    if (workspaceBytes == nullptr) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt",
                "pBufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt",
                "invalid n=%d, expected n >= 0", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *workspaceBytes = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (m < 3 || ldb < std::max(1, m) || dl == nullptr || d == nullptr ||
        du == nullptr) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt",
                "invalid parameters: m=%d, n=%d, ldb=%d, dl=%p, d=%p, du=%p",
                m, n, ldb, static_cast<const void *>(dl),
                static_cast<const void *>(d), static_cast<const void *>(du));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!ComputeWorkspaceBytes(m, n, *workspaceBytes) ||
        !IsBSpanRepresentable(ldb, n)) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt",
                "size overflow: m=%d, n=%d, ldb=%d", m, n, ldb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchKernels(
    aclsparseHandle_t handle, int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb, void *workspace)
{
    aclrtStream stream = nullptr;
    aclsparseStatus_t status = aclsparseGetStream(handle, &stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    uint32_t availableCores = GetAivCoreCount();
    if (availableCores == 0) {
        OP_LOGE("aclsparseSgtsv2", "GetAivCoreCount returned zero");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    const bool useUb = m <= kGtsv2UbMaxM;
    uint32_t blockNum = std::min(static_cast<uint32_t>(n), availableCores);
    // Non-UB scalar GM writes use DCache.  Keep that fallback single-core to
    // avoid two cores dirtying different words in the same 64-byte cache line.
    if (!useUb) {
        blockNum = 1;
    // For non-32-byte-aligned column strides, a single writer also avoids
    // overlapping non-aligned DMA boundary blocks between adjacent columns.
    } else if ((static_cast<uint32_t>(ldb) * sizeof(float)) % 32U != 0U) {
        blockNum = 1;
    }

    Gtsv2TilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.ldb = ldb;
    tiling.blockNum = static_cast<int32_t>(blockNum);
    tiling.useUb = useUb ? 1 : 0;

    OP_LOGI("aclsparseSgtsv2",
            "launch arch22: m=%d, n=%d, ldb=%d, blocks=%u, useUb=%d",
            m, n, ldb, blockNum, tiling.useUb);
    if (n == 1 && useUb) {
        // A single RHS cannot reuse the factorization across columns. Keep
        // factorization and substitution in UB within one kernel instead.
        gtsv2_single_rhs_kernel_do(
            reinterpret_cast<GM_ADDR>(const_cast<float *>(dl)),
            reinterpret_cast<GM_ADDR>(const_cast<float *>(d)),
            reinterpret_cast<GM_ADDR>(const_cast<float *>(du)),
            reinterpret_cast<GM_ADDR>(B), reinterpret_cast<GM_ADDR>(workspace), tiling, stream);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    gtsv2_factor_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<float *>(dl)),
        reinterpret_cast<GM_ADDR>(const_cast<float *>(d)),
        reinterpret_cast<GM_ADDR>(const_cast<float *>(du)),
        reinterpret_cast<GM_ADDR>(B), reinterpret_cast<GM_ADDR>(workspace), tiling, stream);
    gtsv2_solve_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<float *>(dl)),
        reinterpret_cast<GM_ADDR>(const_cast<float *>(d)),
        reinterpret_cast<GM_ADDR>(const_cast<float *>(du)),
        reinterpret_cast<GM_ADDR>(B), reinterpret_cast<GM_ADDR>(workspace),
        tiling, blockNum, stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

aclsparseStatus_t aclsparseSgtsv2(
    aclsparseHandle_t handle, int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb, void *pBuffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgtsv2", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t status = ValidateComputeParams(
        m, n, dl, d, du, B, ldb, pBuffer);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    return LaunchKernels(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

aclsparseStatus_t aclsparseSgtsv2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n,
    const float *dl, const float *d, const float *du,
    const float *B, int ldb, size_t *pBufferSizeInBytes)
{
    (void)B;
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgtsv2_bufferSizeExt", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    return ValidateBufferSizeParams(
        m, n, dl, d, du, ldb, pBufferSizeInBytes);
}

}  // extern "C"
