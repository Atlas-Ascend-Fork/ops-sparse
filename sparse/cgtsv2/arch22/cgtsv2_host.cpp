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

#include <cstddef>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "cgtsv2_kernel.h"
#include "../cgtsv2_backend.h"
#include "../cgtsv2_workspace.h"

namespace {

using cgtsv2::CheckedMultiply;

bool AppendWorkspaceRegion(size_t count, size_t elementBytes, size_t& totalBytes, uint64_t& offset)
{
    constexpr size_t alignment = static_cast<size_t>(kCgtsv2WorkspaceAlignment);
    size_t bytes = 0;
    if (!CheckedMultiply(count, elementBytes, bytes) || bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return false;
    }
    const size_t alignedBytes = (bytes + alignment - 1) & ~(alignment - 1);
    if (totalBytes > std::numeric_limits<size_t>::max() - alignedBytes) {
        return false;
    }
    offset = static_cast<uint64_t>(totalBytes);
    totalBytes += alignedBytes;
    return true;
}

bool ComputeWorkspaceLayout(int m, Cgtsv2TilingData& tiling)
{
    // The public entry points validate m >= 3 before computing m - 1/m - 2.
    const size_t rows = static_cast<size_t>(m);
    size_t bytes = 0;
    if (!AppendWorkspaceRegion(rows - 1, sizeof(aclsparseComplex_t), bytes, tiling.lOffset) ||
        !AppendWorkspaceRegion(rows, sizeof(aclsparseComplex_t), bytes, tiling.dOffset) ||
        !AppendWorkspaceRegion(rows - 1, sizeof(aclsparseComplex_t), bytes, tiling.uOffset) ||
        !AppendWorkspaceRegion(rows - 2, sizeof(aclsparseComplex_t), bytes, tiling.vOffset) ||
        !AppendWorkspaceRegion(rows, sizeof(uint32_t), bytes, tiling.pivotOffset) ||
        !AppendWorkspaceRegion(1, static_cast<size_t>(kCgtsv2StatusBytes), bytes, tiling.statusOffset) ||
        !AppendWorkspaceRegion(rows, sizeof(aclsparseComplex_t), bytes, tiling.rhsScratchOffset) ||
        !AppendWorkspaceRegion(rows - 1, sizeof(aclsparseComplex_t), bytes, tiling.lLowOffset) ||
        !AppendWorkspaceRegion(rows, sizeof(aclsparseComplex_t), bytes, tiling.dLowOffset) ||
        !AppendWorkspaceRegion(rows - 1, sizeof(aclsparseComplex_t), bytes, tiling.uLowOffset) ||
        !AppendWorkspaceRegion(rows - 1, 2 * sizeof(int64_t), bytes, tiling.lExponentOffset) ||
        !AppendWorkspaceRegion(rows, 2 * sizeof(int64_t), bytes, tiling.dExponentOffset) ||
        !AppendWorkspaceRegion(rows - 1, 2 * sizeof(int64_t), bytes, tiling.uExponentOffset) ||
        !AppendWorkspaceRegion(rows, sizeof(aclsparseComplex_t), bytes, tiling.rhsLowOffset) ||
        !AppendWorkspaceRegion(rows, 2 * sizeof(int64_t), bytes, tiling.rhsExponentOffset) ||
        !AppendWorkspaceRegion(rows, sizeof(aclsparseComplex_t), bytes, tiling.correctionHighOffset) ||
        !AppendWorkspaceRegion(rows, sizeof(aclsparseComplex_t), bytes, tiling.correctionLowOffset) ||
        !AppendWorkspaceRegion(rows, 2 * sizeof(int64_t), bytes, tiling.correctionExponentOffset) ||
        bytes > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
        return false;
    }
    tiling.workspaceBytes = static_cast<uint64_t>(bytes);
    return true;
}

aclsparseStatus_t LaunchKernels(const aclsparseComplex_t* dl, const aclsparseComplex_t* d, const aclsparseComplex_t* du,
                                aclsparseComplex_t* B, void* workspace, const Cgtsv2TilingData& tiling,
                                aclrtStream stream)
{
    const uint32_t availableCores = GetAivCoreCount();
    const uint64_t ubBytes = GetUbSize();
    if (availableCores == 0 || ubBytes == 0) {
        OP_LOGE("aclsparseCgtsv2", "platform resources unavailable: cores=%u, UB=%llu", availableCores,
                static_cast<unsigned long long>(ubBytes));
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    Cgtsv2TilingData selected = tiling;
    const uint64_t rowAlignment = kCgtsv2UbRowAlignment;
    const uint64_t rows = (static_cast<uint64_t>(tiling.m) + rowAlignment - 1) & ~(rowAlignment - 1);
    const uint64_t requiredUb = tiling.m >= kCgtsv2JacobiMinRows ?
        kCgtsv2JacobiStorageBytes : kCgtsv2SolveUbBytesPerRow * rows;
    const bool alignedBase = reinterpret_cast<uintptr_t>(B) % 32 == 0;
    const bool disjointColumns = (static_cast<uint64_t>(tiling.ldb) * sizeof(aclsparseComplex_t)) % 32 == 0;
    if (ubBytes >= kCgtsv2UbHeadroomBytes && requiredUb <= ubBytes - kCgtsv2UbHeadroomBytes) {
        selected.useUb = 1;
        // DataCopyPad accepts nonaligned GM addresses. Such columns still
        // require one physical writer, since masked boundary writes may share
        // a cacheline. The UB data path itself remains valid.
        if (alignedBase && disjointColumns) {
            selected.blockNum = static_cast<int32_t>(static_cast<uint32_t>(tiling.n) < availableCores ?
                                                       static_cast<uint32_t>(tiling.n) : availableCores);
        }
    }
    if (selected.useUb != 0 && selected.n == 1 && selected.m >= kCgtsv2JacobiMinRows &&
        reinterpret_cast<uintptr_t>(B) % 64 == 0 && disjointColumns) {
        // The factor kernel preserves the original RHS in existing scratch.
        // Each solve block owns complete center tiles; both the base
        // and every center boundary are aligned to independent cachelines.
        const uint64_t tiles = (static_cast<uint64_t>(selected.m) + kCgtsv2JacobiCenterRows - 1) /
                               kCgtsv2JacobiCenterRows;
        selected.blockNum = static_cast<int32_t>(tiles < availableCores ? tiles : availableCores);
    }
    OP_LOGI("aclsparseCgtsv2", "arch22 launch: m=%d, n=%d, ldb=%d, blocks=%d, useUb=%d", selected.m, selected.n,
            selected.ldb, selected.blockNum, selected.useUb);
    auto dlAddress = reinterpret_cast<GM_ADDR>(const_cast<aclsparseComplex_t*>(dl));
    auto dAddress = reinterpret_cast<GM_ADDR>(const_cast<aclsparseComplex_t*>(d));
    auto duAddress = reinterpret_cast<GM_ADDR>(const_cast<aclsparseComplex_t*>(du));
    auto bAddress = reinterpret_cast<GM_ADDR>(B);
    auto workspaceAddress = reinterpret_cast<GM_ADDR>(workspace);
    if (selected.n == 1 && selected.useUb != 0 && selected.blockNum == 1) {
        cgtsv2_single_rhs_kernel_do(dlAddress, dAddress, duAddress, bAddress, workspaceAddress, selected, stream);
    } else {
        cgtsv2_factor_kernel_do(dlAddress, dAddress, duAddress, bAddress, workspaceAddress, selected, stream);
        cgtsv2_solve_kernel_do(dlAddress, dAddress, duAddress, bAddress, workspaceAddress, selected, stream);
    }
    // The wrappers enqueue asynchronous work. Device failures are observed by
    // the caller when it synchronizes this stream; no hidden synchronization.
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

bool Cgtsv2WorkspaceBytes(int m, int n, int ldb, size_t &bytes)
{
    bytes = 0;
    if (m < 3 || n < 0 || ldb < m) {
        OP_LOGE("aclsparseCgtsv2", "invalid arch22 dimensions: m=%d, n=%d, ldb=%d", m, n, ldb);
        return false;
    }
    if (n == 0) { return true; }
    Cgtsv2TilingData tiling{};
    if (!ComputeWorkspaceLayout(m, tiling)) {
        OP_LOGE("aclsparseCgtsv2", "arch22 workspace layout overflows");
        return false;
    }
    bytes = static_cast<size_t>(tiling.workspaceBytes);
    return true;
}

aclsparseStatus_t Cgtsv2Dispatch(aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, aclsparseComplex_t *B, int ldb, void *workspace)
{
    aclrtStream stream = nullptr;
    const auto status = aclsparseGetStream(handle, &stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseCgtsv2", "arch22 stream lookup failed: status=%d", static_cast<int>(status));
        return status;
    }
    Cgtsv2TilingData tiling{};
    if (!ComputeWorkspaceLayout(m, tiling)) {
        OP_LOGE("aclsparseCgtsv2", "arch22 workspace layout overflows");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    tiling.m = m;
    tiling.n = n;
    tiling.ldb = ldb;
    tiling.blockNum = 1;
    tiling.useUb = 0;
    return LaunchKernels(dl, d, du, B, workspace, tiling, stream);
}
