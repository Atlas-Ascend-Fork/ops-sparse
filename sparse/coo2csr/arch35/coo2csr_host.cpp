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
 * \file coo2csr_host.cpp
 * \brief aclsparseXcoo2csr Host 侧实现：参数校验 + Kernel launch。
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "coo2csr_tiling_data.h"
#include "coo2csr_kernel.h"

namespace {

inline aclsparseContext *ToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<aclsparseContext *>(handle);
}

// ===========================================================================
// 参数校验
// ===========================================================================
static aclsparseStatus_t ValidateCoo2CsrParams(
    aclsparseHandle_t handle,
    const int *cooRowInd, int32_t nnz, int32_t m,
    int *csrRowPtr, aclsparseIndexBase_t idxBase)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseXcoo2csr", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (nnz < 0) {
        OP_LOGE("aclsparseXcoo2csr", "invalid nnz: %d", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz == INT32_MAX && idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE("aclsparseXcoo2csr", "nnz + idxBase exceeds INT32_MAX: nnz=%d, idxBase=1", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (m < 0) {
        OP_LOGE("aclsparseXcoo2csr", "invalid m: %d", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && m == 0) {
        OP_LOGE("aclsparseXcoo2csr", "nnz > 0 but m == 0: %d", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (idxBase != ACL_SPARSE_INDEX_BASE_ZERO &&
        idxBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE("aclsparseXcoo2csr", "invalid idxBase: %d",
                static_cast<int>(idxBase));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && cooRowInd == nullptr) {
        OP_LOGE("aclsparseXcoo2csr", "cooRowInd is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (csrRowPtr == nullptr) {
        OP_LOGE("aclsparseXcoo2csr", "csrRowPtr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 边界处理：nnz == 0 且 idxBase == 0 → memset 清零 csrRowPtr[0..m]
// ===========================================================================
static aclsparseStatus_t FillCsrRowPtrZero(
    aclrtStream stream, int32_t m, int *csrRowPtr)
{
    size_t sizeBytes = (static_cast<size_t>(m) + 1) * sizeof(int32_t);
    aclError aclRet = aclrtMemsetAsync(
        csrRowPtr, sizeBytes, 0, sizeBytes, stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              OP_LOGE("aclsparseXcoo2csr",
                      "aclrtMemsetAsync csrRowPtr failed, ret=%d", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    OP_LOGI("aclsparseXcoo2csr",
            "nnz=0 idxBase=0, memset csrRowPtr (%zu bytes)", sizeBytes);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 公共：K1 (CountRows) launch
// ===========================================================================
static aclsparseStatus_t LaunchCountKernel(
    aclrtStream stream,
    GM_ADDR gmRowInd, GM_ADDR gmWorkspace,
    int32_t nnz, int32_t baseVal, int32_t m,
    uint32_t aivCoreNum)
{
    uint32_t numBlocks = std::min(
        CeilDiv<uint32_t>(static_cast<uint32_t>(nnz), kCoo2CsrThreadsPerBlock),
        aivCoreNum);
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    Coo2CsrCountTilingData countTiling{};
    countTiling.nnz = nnz;
    countTiling.idxBase = baseVal;
    countTiling.m = m;

    // clear-then-launch-then-peek（issue #357 分族）
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    coo2csr_count_kernel_do(
        gmRowInd, gmWorkspace, countTiling, numBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    OP_LOGD("aclsparseXcoo2csr",
            "K1 (CountRows) launched, blocks=%u", numBlocks);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 公共：workspace 准备 + memset
// ===========================================================================
static aclsparseStatus_t PrepareWorkspace(
    aclsparseContext *h,
    int32_t m, uint32_t aivCoreNum,
    void *&workspace, size_t &workspaceSize)
{
    workspace = aclsparseGetEffectiveWorkspace(h);
    workspaceSize = aclsparseGetEffectiveWorkspaceSize(h);
    size_t rowCountSize = static_cast<size_t>(m) * sizeof(int32_t);
    size_t blockTotalsEstimate =
        static_cast<size_t>(aivCoreNum) * sizeof(int32_t);
    size_t totalNeeded = rowCountSize + blockTotalsEstimate;

    if (workspace == nullptr) {
        OP_LOGE("aclsparseXcoo2csr",
                "workspace is null (handle has no workspace)");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (totalNeeded > workspaceSize) {
        OP_LOGE("aclsparseXcoo2csr",
                "handle workspace too small: %zu < %zu (rowCount %zu + blockTotals %zu)",
                workspaceSize, totalNeeded, rowCountSize, blockTotalsEstimate);
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    aclrtStream stream = h->stream;
    if (rowCountSize > 0) {
        aclError aclRet = aclrtMemsetAsync(
            workspace, rowCountSize, 0, rowCountSize, stream);
        CHECK_RET(aclRet == ACL_SUCCESS,
                  OP_LOGE("aclsparseXcoo2csr",
                          "aclrtMemsetAsync workspace failed, ret=%d", aclRet);
                  return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 融合 kernel 路径：单 block 完成 Count + PrefixSum
// ===========================================================================
static aclsparseStatus_t LaunchFusedKernel(
    aclrtStream stream,
    GM_ADDR gmRowInd, GM_ADDR gmWorkspace, GM_ADDR gmCsrRowPtr,
    int32_t nnz, int32_t m, int32_t baseVal)
{
    Coo2CsrFusedTilingData fusedTiling{};
    fusedTiling.nnz = nnz;
    fusedTiling.m = m;
    fusedTiling.idxBase = baseVal;

    // clear-then-launch-then-peek（issue #357 分族）
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    coo2csr_fused_kernel_do(
        gmRowInd, gmWorkspace, gmCsrRowPtr, fusedTiling, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    OP_LOGD("aclsparseXcoo2csr",
            "Fused kernel launched (nnz=%d, m=%d)", nnz, m);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 多核并行前缀和 Phase A/B/C
// ===========================================================================
static aclsparseStatus_t LaunchPrefixSumPhases(
    aclrtStream stream,
    GM_ADDR gmWorkspace, GM_ADDR gmCsrRowPtr, GM_ADDR gmBlockTotals,
    int32_t m, int32_t baseVal,
    uint32_t chunkSize, uint32_t numBlocksPS)
{
    // clear-then-launch-then-peek（issue #357 分族）
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);


    // Phase A: 各 block 局部前缀和
    Coo2CsrPrefixSumLocalTilingData localTiling{};
    localTiling.m = m;
    localTiling.idxBase = baseVal;
    localTiling.chunkSize = static_cast<int32_t>(chunkSize);

    coo2csr_prefixsum_local_kernel_do(
        gmWorkspace, gmCsrRowPtr, gmBlockTotals,
        localTiling, numBlocksPS, stream);

    // Phase B: blockTotals 排他前缀和
    Coo2CsrPrefixSumBlocksTilingData blocksTiling{};
    blocksTiling.numBlocks = numBlocksPS;

    coo2csr_prefixsum_blocks_kernel_do(
        gmBlockTotals, blocksTiling, stream);

    // Phase C: 偏移修正
    Coo2CsrPrefixSumCorrectTilingData correctTiling{};
    correctTiling.m = m;
    correctTiling.chunkSize = static_cast<int32_t>(chunkSize);

    coo2csr_prefixsum_correct_kernel_do(
        gmCsrRowPtr, gmBlockTotals, correctTiling, numBlocksPS, stream);

    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 前缀和路径：K1 + (多核 Phase A/B/C | 单核单 kernel)，按 useParallel 分支
// （评审响应：合并两个同签名函数，从根源消重复）
// ===========================================================================
static aclsparseStatus_t LaunchPrefixSum(
    aclrtStream stream, GM_ADDR gmRowInd, GM_ADDR gmWorkspace, GM_ADDR gmCsrRowPtr,
    int32_t nnz, int32_t m, int32_t baseVal, uint32_t aivCoreNum, bool useParallel)
{
    aclsparseStatus_t countSt = LaunchCountKernel(stream, gmRowInd, gmWorkspace,
                      nnz, baseVal, m, aivCoreNum);
    if (countSt != ACL_SPARSE_STATUS_SUCCESS) {
        return countSt;
    }

    uint32_t chunkSize = CeilDiv<uint32_t>(
        static_cast<uint32_t>(m), aivCoreNum);
    if (chunkSize == 0) {
        chunkSize = 1;
    }
    uint32_t numBlocksPS = CeilDiv<uint32_t>(
        static_cast<uint32_t>(m), chunkSize);

    size_t rowCountSize = static_cast<size_t>(m) * sizeof(int32_t);
    auto *gmBlockTotals = reinterpret_cast<GM_ADDR>(gmWorkspace + rowCountSize);

    if (!useParallel) {
        Coo2CsrPrefixSumTilingData psTiling{};
        psTiling.m = m;
        psTiling.idxBase = baseVal;

        // clear-then-launch-then-peek（issue #357 分族，评审响应：peek 移到下发后）
        (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

        coo2csr_prefixsum_kernel_do(
            gmWorkspace, gmCsrRowPtr, psTiling, stream);
        if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        OP_LOGD("aclsparseXcoo2csr", "K2 (PrefixSum) launched");

        return ACL_SPARSE_STATUS_SUCCESS;
    }

    aclsparseStatus_t psRet = LaunchPrefixSumPhases(
        stream, gmWorkspace, gmCsrRowPtr, gmBlockTotals,
        m, baseVal, chunkSize, numBlocksPS);
    if (psRet != ACL_SPARSE_STATUS_SUCCESS) {
        return psRet;
    }

    OP_LOGD("aclsparseXcoo2csr",
            "K2 (ParallelPrefixSum) launched, blocksPS=%u, chunkSize=%u",
            numBlocksPS, chunkSize);

    return ACL_SPARSE_STATUS_SUCCESS;
}



// ===========================================================================
// Kernel launch 主入口
// ===========================================================================
static aclsparseStatus_t LaunchCoo2CsrKernel(
    aclsparseHandle_t handle,
    const int *cooRowInd, int32_t nnz, int32_t m,
    int *csrRowPtr, aclsparseIndexBase_t idxBase)
{
    auto *h = ToInternalHandle(handle);
    aclrtStream stream = h->stream;
    int32_t baseVal = (idxBase == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    if (nnz == 0) {
        if (baseVal == 0) {
            return FillCsrRowPtrZero(stream, m, csrRowPtr);
        }
        return LaunchFusedKernel(
            stream, nullptr, nullptr,
            reinterpret_cast<GM_ADDR>(csrRowPtr), nnz, m, baseVal);
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE("aclsparseXcoo2csr", "GetAivCoreCount failed");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    void *workspace = nullptr;
    size_t workspaceSize = 0;
    aclsparseStatus_t wsRet = PrepareWorkspace(h, m, aivCoreNum,
                                               workspace, workspaceSize);
    if (wsRet != ACL_SPARSE_STATUS_SUCCESS) {
        return wsRet;
    }

    bool useParallel = (m > kCoo2CsrParallelPrefixSumThreshold);
    bool useFused = (!useParallel &&
                     nnz <= static_cast<int>(kCoo2CsrThreadsPerBlock));

    OP_LOGD("aclsparseXcoo2csr",
            "path: m=%d, nnz=%d, idxBase=%d, useFused=%d, useParallel=%d",
            m, nnz, baseVal, static_cast<int>(useFused),
            static_cast<int>(useParallel));

    auto *gmRowInd = reinterpret_cast<GM_ADDR>(const_cast<int *>(cooRowInd));
    auto *gmWorkspace = reinterpret_cast<GM_ADDR>(workspace);
    auto *gmCsrRowPtr = reinterpret_cast<GM_ADDR>(csrRowPtr);

    if (useFused) {
        return LaunchFusedKernel(
            stream, gmRowInd, gmWorkspace, gmCsrRowPtr,
            nnz, m, baseVal);
    } else {
        return LaunchPrefixSum(
            stream, gmRowInd, gmWorkspace, gmCsrRowPtr,
            nnz, m, baseVal, aivCoreNum, useParallel);
    }
}

}  // namespace

// ============================================================================
// Public API
// ============================================================================
extern "C" {

aclsparseStatus_t aclsparseXcoo2csr(
    aclsparseHandle_t handle,
    const int *cooRowInd, int nnz, int m,
    int *csrRowPtr, aclsparseIndexBase_t idxBase)
{
    try {
        aclsparseStatus_t st = ValidateCoo2CsrParams(
            handle, cooRowInd, nnz, m, csrRowPtr, idxBase);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }

        OP_LOGD("aclsparseXcoo2csr",
                "params OK: m=%d, nnz=%d, idxBase=%d",
                m, nnz, static_cast<int>(idxBase));

        return LaunchCoo2CsrKernel(handle, cooRowInd, nnz, m, csrRowPtr, idxBase);
    } catch (...) {
        OP_LOGE("aclsparseXcoo2csr", "internal error: uncaught exception");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
}

}  // extern "C"
