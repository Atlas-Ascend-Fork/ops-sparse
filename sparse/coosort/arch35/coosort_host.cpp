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
 * \file coosort_host.cpp
 * \brief aclsparseXcoosort Host 侧实现（Legacy API，对标 cuSPARSE cusparseXcoosort*）。
 *
 * 三个 extern "C" API：
 *   - aclsparseXcoosort_bufferSizeExt：查询 pBuffer 大小（单核保留区或多核 GM ping-pong 区，向上对齐 128）。
 *   - aclsparseXcoosortByRow：双键 (row 主/col 次) 稳定排序，in-place 重排 row/col + 输出 P。
 *   - aclsparseXcoosortByColumn：双键 (col 主/row 次) 稳定排序。
 *
 * 算法：单核直排使用两趟 LSB radix 稳定排序；大 nnz 按真实 UB 容量切成多个 run，
 * 每核循环生成 run，再在 GM workspace 中执行多轮并行 merge-path 稳定归并。
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "coosort_kernel.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "sort/sort_tiling_intf.h"  // GetSortMaxMinTmpSize / SortConfig / SortType

namespace {

inline struct aclsparseContext *ToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

// ===========================================================================
// 公共参数校验（三个 API 共享）
// ===========================================================================
static aclsparseStatus_t ValidateCommonCoosortArgs(
    const char *apiName, int m, int n, int nnz, const int *cooRowsA, const int *cooColsA, bool checkNnzLimit)
{
    if (m <= 0) {
        OP_LOGE(apiName, "invalid m: %d (must be positive)", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        OP_LOGE(apiName, "invalid n: %d (must be positive)", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz < 0) {
        OP_LOGE(apiName, "invalid nnz: %d (must be >= 0)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && cooRowsA == nullptr) {
        OP_LOGE(apiName, "cooRowsA is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && cooColsA == nullptr) {
        OP_LOGE(apiName, "cooColsA is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (checkNnzLimit && static_cast<uint32_t>(nnz) > kCoosortSingleCoreMaxNnz) {
        OP_LOGI(apiName,
            "nnz=%d exceeds single-core dispatch threshold %u, routing to multi-core path",
            nnz,
            kCoosortSingleCoreMaxNnz);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static uint32_t CalcSortTmpSize(uint32_t nnz)
{
    std::vector<int64_t> shapeVec = {static_cast<int64_t>(nnz)};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, ge::DT_INT32, ge::DT_UINT32, false, config, maxValue, minValue);
    return maxValue;
}

static bool FindMaxPhaseOneRunSize(uint64_t ubSize, uint32_t &runSize, uint32_t &sortTmpSize)
{
    constexpr uint32_t kAlignElems = 8;
    constexpr uint32_t kPhaseOneBytesPerElem = 48;

    uint64_t maxElemsByFixedBuffer = ubSize / kPhaseOneBytesPerElem;
    uint32_t maxUnits = static_cast<uint32_t>(
        std::min<uint64_t>(maxElemsByFixedBuffer / kAlignElems, std::numeric_limits<uint32_t>::max() / kAlignElems));
    uint32_t low = 1;
    uint32_t high = maxUnits;
    uint32_t bestRunSize = 0;
    uint32_t bestSortTmpSize = 0;

    // Sort 临时空间和固定 UB 缓冲都随 runSize 单调增长。以 8 个元素为
    // 搜索单位，使每个常规 workspace run 的起点保持 32 字节对齐：
    // 8 * 3 * sizeof(int32_t) == 96 字节。
    while (low <= high) {
        uint32_t mid = low + (high - low) / 2;
        uint32_t candidate = mid * kAlignElems;
        uint32_t candidateTmpSize = CalcSortTmpSize(candidate);
        if (candidateTmpSize == 0) {
            high = mid - 1;
            continue;
        }
        uint64_t requiredBytes = static_cast<uint64_t>(candidate) * kPhaseOneBytesPerElem + candidateTmpSize;
        if (requiredBytes <= ubSize) {
            bestRunSize = candidate;
            bestSortTmpSize = candidateTmpSize;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    if (bestRunSize == 0) {
        return false;
    }
    runSize = bestRunSize;
    sortTmpSize = bestSortTmpSize;
    return true;
}

static aclsparseStatus_t ComputeMultiCoreTiling(const char *apiName, uint32_t nnz, CoosortTilingData &tiling)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE(apiName, "GetAivCoreCount returned 0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    uint64_t ubSize = GetUbSize();
    if (ubSize == 0) {
        OP_LOGE(apiName, "GetUbSize returned 0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    uint32_t runSize = 0;
    uint32_t sortTmpSize = 0;
    if (!FindMaxPhaseOneRunSize(ubSize, runSize, sortTmpSize)) {
        OP_LOGE(apiName,
            "UB capacity %llu is insufficient for the minimum coosort run",
            static_cast<unsigned long long>(ubSize));
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    uint32_t runCount = static_cast<uint32_t>((static_cast<uint64_t>(nnz) + runSize - 1U) / runSize);
    uint32_t coreNum = std::min(aivCoreNum, runCount);

    constexpr uint32_t kMergeRecordFields = 3U;
    constexpr uint32_t kMergeRecordBytes = kMergeRecordFields * sizeof(int32_t);
    constexpr uint32_t kMergeAlignRecords = 8U;
    constexpr uint32_t kMergeInputReserveRecords = 3U * kMergeAlignRecords;
    constexpr uint32_t kMergePartitionWindowRecords = 2U * kMergeAlignRecords;
    constexpr uint32_t kMergeBytesPerElem = 2U * kMergeRecordBytes;
    constexpr uint32_t kMergeFixedBufferBytes =
        (kMergeInputReserveRecords + kMergePartitionWindowRecords) * kMergeRecordBytes;
    constexpr uint32_t kMteMaxBlockBytes = 65535U;
    constexpr uint32_t kMteMaxMergeElems = kMteMaxBlockBytes / kMergeRecordBytes - (kMergeAlignRecords - 1U);

    // Phase 1 后会重置 pipe。归并阶段的线性 UB 开销是输入、输出各一份
    // COO 三元组（每元素共 24 字节）；固定开销包括输入对齐余量、两个
    // merge-path 边界窗口，以及供最终拆分输出使用的一份 runSize 三元组。
    uint64_t finalOutputBufferBytes = static_cast<uint64_t>(runSize) * kMergeRecordBytes;
    if (finalOutputBufferBytes + kMergeFixedBufferBytes >= ubSize) {
        OP_LOGE(apiName,
            "UB capacity %llu is insufficient for coosort merge buffers",
            static_cast<unsigned long long>(ubSize));
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    uint32_t onceMaxElems =
        static_cast<uint32_t>((ubSize - finalOutputBufferBytes - kMergeFixedBufferBytes) / kMergeBytesPerElem);
    onceMaxElems = std::min(onceMaxElems, kMteMaxMergeElems);
    onceMaxElems = (onceMaxElems / 32U) * 32U;
    if (onceMaxElems < 32U) {
        OP_LOGE(apiName,
            "UB capacity %llu cannot hold the minimum merge-path tile",
            static_cast<unsigned long long>(ubSize));
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    tiling.nnz = nnz;
    tiling.tileSize = runSize;
    tiling.runCount = runCount;
    tiling.coreNum = coreNum;
    tiling.mergeSize = onceMaxElems;
    tiling.sortTmpSize = sortTmpSize;
    OP_LOGI(apiName,
        "multi-run tiling: nnz=%u, runSize=%u, runCount=%u, coreNum=%u, mergeTile=%u",
        nnz,
        runSize,
        runCount,
        coreNum,
        onceMaxElems);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchCoosortKernel(
    aclsparseHandle_t handle, int nnz, bool sortByRow, int *cooRowsA, int *cooColsA, int *pOut)
{
    auto *h = ToInternalHandle(handle);
    aclrtStream stream = h->stream;

    CoosortTilingData tiling{};
    tiling.nnz = static_cast<uint32_t>(nnz);
    tiling.sortByRow = sortByRow ? 1U : 0U;
    tiling.sortTmpSize = CalcSortTmpSize(static_cast<uint32_t>(nnz));

    auto *gmRow = reinterpret_cast<GM_ADDR>(cooRowsA);
    auto *gmCol = reinterpret_cast<GM_ADDR>(cooColsA);
    auto *gmP = reinterpret_cast<GM_ADDR>(pOut);

    constexpr uint32_t numBlocks = 1;
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    coosort_kernel_do(gmRow, gmCol, gmP, tiling, numBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchCoosortMultiCoreKernel(
    aclsparseHandle_t handle, int nnz, bool sortByRow, int *cooRowsA, int *cooColsA, int *pOut, void *pBuffer)
{
    auto *h = ToInternalHandle(handle);
    aclrtStream stream = h->stream;

    CoosortTilingData tiling{};
    aclsparseStatus_t st = ComputeMultiCoreTiling(
        sortByRow ? "aclsparseXcoosortByRow" : "aclsparseXcoosortByColumn", static_cast<uint32_t>(nnz), tiling);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    tiling.sortByRow = sortByRow ? 1U : 0U;

    auto *gmRow = reinterpret_cast<GM_ADDR>(cooRowsA);
    auto *gmCol = reinterpret_cast<GM_ADDR>(cooColsA);
    auto *gmP = reinterpret_cast<GM_ADDR>(pOut);
    auto *gmWs = reinterpret_cast<GM_ADDR>(pBuffer);

    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    coosort_multi_core_kernel_do(gmRow, gmCol, gmP, gmWs, tiling, tiling.coreNum, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

// ============================================================================
// Public APIs
// ============================================================================
extern "C" {

aclsparseStatus_t aclsparseXcoosort_bufferSizeExt(aclsparseHandle_t handle, int m, int n, int nnz, const int *cooRowsA,
    const int *cooColsA, size_t *pBufferSizeInBytes)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseXcoosort_bufferSizeExt", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (pBufferSizeInBytes == nullptr) {
        OP_LOGE("aclsparseXcoosort_bufferSizeExt", "pBufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st =
        ValidateCommonCoosortArgs("aclsparseXcoosort_bufferSizeExt", m, n, nnz, cooRowsA, cooColsA, false);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (nnz == 0) {
        *pBufferSizeInBytes = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    if (static_cast<uint32_t>(nnz) > kCoosortSingleCoreMaxNnz) {
        CoosortTilingData tiling{};
        st = ComputeMultiCoreTiling("aclsparseXcoosort_bufferSizeExt", static_cast<uint32_t>(nnz), tiling);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
        uint32_t nnzU32 = static_cast<uint32_t>(nnz);
        // 两份 ping-pong workspace，每份保存 nnz 个 [row,col,P] 三元组。
        // 额外 8 个 int32 覆盖第二份 workspace 起点的 32 字节对齐间隔。
        size_t halfBytes = static_cast<size_t>(nnzU32) * 3 * sizeof(int32_t);
        size_t totalBytes =
            (halfBytes * 2 + 8 * sizeof(int32_t) + kCoosortAlign - 1) & ~static_cast<size_t>(kCoosortAlign - 1);
        *pBufferSizeInBytes = totalBytes;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    constexpr size_t kReservedBytesPerElem = 12U;
    size_t rawBytes = static_cast<size_t>(nnz) * kReservedBytesPerElem;
    size_t alignedBytes = (rawBytes + kCoosortAlign - 1) & ~static_cast<size_t>(kCoosortAlign - 1);
    *pBufferSizeInBytes = alignedBytes;

    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseXcoosortByRow(
    aclsparseHandle_t handle, int m, int n, int nnz, int *cooRowsA, int *cooColsA, int *P, void *pBuffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseXcoosortByRow", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = ValidateCommonCoosortArgs("aclsparseXcoosortByRow", m, n, nnz, cooRowsA, cooColsA, true);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    if (P == nullptr) {
        OP_LOGE("aclsparseXcoosortByRow", "P is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (pBuffer == nullptr) {
        OP_LOGE("aclsparseXcoosortByRow", "pBuffer is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (static_cast<uint32_t>(nnz) > kCoosortSingleCoreMaxNnz) {
        return LaunchCoosortMultiCoreKernel(handle, nnz, true, cooRowsA, cooColsA, P, pBuffer);
    }
    return LaunchCoosortKernel(handle, nnz, true, cooRowsA, cooColsA, P);
}

aclsparseStatus_t aclsparseXcoosortByColumn(
    aclsparseHandle_t handle, int m, int n, int nnz, int *cooRowsA, int *cooColsA, int *P, void *pBuffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseXcoosortByColumn", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = ValidateCommonCoosortArgs("aclsparseXcoosortByColumn", m, n, nnz, cooRowsA, cooColsA, true);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    if (P == nullptr) {
        OP_LOGE("aclsparseXcoosortByColumn", "P is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (pBuffer == nullptr) {
        OP_LOGE("aclsparseXcoosortByColumn", "pBuffer is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (static_cast<uint32_t>(nnz) > kCoosortSingleCoreMaxNnz) {
        return LaunchCoosortMultiCoreKernel(handle, nnz, false, cooRowsA, cooColsA, P, pBuffer);
    }
    return LaunchCoosortKernel(handle, nnz, false, cooRowsA, cooColsA, P);
}

}  // extern "C"
