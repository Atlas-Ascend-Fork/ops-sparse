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
 * \file csr2coo_host.cpp
 * \brief aclsparseXcsr2coo (Legacy API) Host side implementation:
 *        parameter validation + Tiling + Kernel launch.
 *
 * Converts CSR rowPtr[m+1] to COO cooRowInd[nnz] with flat parameters.
 * Only supports INT32 index type. No workspace, no GetBufferSize.
 *
 * Structure: ValidateInputs + ComputeTiling + LaunchKernel.
 */

#include <cstdint>
#include <algorithm>
#include <limits>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "csr2coo_kernel.h"

namespace {

// ============================================================================
// Helper: cast opaque handle to internal context (same pattern as nnz_host.cpp)
// ============================================================================
inline struct aclsparseContext *ToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

// ============================================================================
// UB budget constants (dav-3510 / Ascend950)
// ============================================================================
// SIMD 路径 UB 预算仅扣除 4KB 系统保留空间，未预留 32KB SIMT DCache。
// 原因：SIMD 与 SIMT 路径在 kernel 入口 if/else 互斥，走 SIMD 时不调用
// asc_vf_call，DCache 不激活，TPipe 分配不会侵入 DCache 区域。
// 性能验证：m=1, nnz=1M 时 UB 利用率 99.97%，输出正确无数据损坏。
// 区别于 csrsort：csrsort 同进程串行 TPipe+VF，必须预留 DCache。
constexpr uint32_t kSystemReserved = 4096u;

// ============================================================================
// Index type constants (INT32 only for Legacy API)
// ============================================================================
constexpr uint32_t kIdxByteSize = 4;          // sizeof(int32_t)
constexpr uint32_t kAlignElems = 8;           // 32 bytes / 4 bytes = 8 elements
constexpr int64_t kMinCooChunkBytes = static_cast<int64_t>(kAlignElems) * kIdxByteSize;
constexpr int64_t kRowPtrBufCount = 3;

// ============================================================================
// aclsparseXcsr2coo - parameter validation
// ============================================================================
static aclsparseStatus_t ValidateInputs(
    aclsparseHandle_t handle,
    const int32_t *csrRowPtr, int64_t nnz, int64_t m,
    int32_t *cooRowInd, aclsparseIndexBase_t idxBase)
{
    // handle 为空
    if (handle == nullptr) {
        OP_LOGE("aclsparseXcsr2coo", "handle is nullptr");
        return ACL_SPARSE_STATUS_NOT_INITIALIZED;
    }
    // nnz 为负
    if (nnz < 0) {
        OP_LOGE("aclsparseXcsr2coo", "nnz=%ld is negative", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // m 为负
    if (m < 0) {
        OP_LOGE("aclsparseXcsr2coo", "m=%ld is negative", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // m=0 但 nnz 非零
    if (m == 0 && nnz != 0) {
        OP_LOGE("aclsparseXcsr2coo", "m=0 but nnz=%ld, invalid", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // idxBase 非法
    if (idxBase != ACL_SPARSE_INDEX_BASE_ZERO && idxBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE("aclsparseXcsr2coo", "invalid idxBase: %d", static_cast<int>(idxBase));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // m>0 但 csrRowPtr 为空
    if (m > 0 && csrRowPtr == nullptr) {
        OP_LOGE("aclsparseXcsr2coo", "csrRowPtr is nullptr (m=%ld > 0)", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // nnz>0 但 cooRowInd 为空
    if (nnz > 0 && cooRowInd == nullptr) {
        OP_LOGE("aclsparseXcsr2coo", "cooRowInd is nullptr (nnz=%ld > 0)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // nnz 超过 INT32_MAX（索引类型为 int32_t）
    if (nnz > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE("aclsparseXcsr2coo", "nnz=%ld exceeds INT32_MAX", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // m 超过 INT32_MAX（COO 行索引存储为 int32_t）
    if (m > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE("aclsparseXcsr2coo", "m=%ld exceeds INT32_MAX", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// ComputeBlockSplit: multi-core block split calculation (remainder distribution)
//   blockSize = floor(m / numBlocks), remainder = m % numBlocks
//   前 remainder 个核多处理 1 行（对齐仓内 gtsv2_strided_batch 先例）。
// ============================================================================
static void ComputeBlockSplit(int64_t m, uint64_t availableUB, uint32_t maxCoreNum,
                               uint32_t &numBlocks, int64_t &blockSize, uint32_t &remainder)
{
    if (maxCoreNum == 0) {
        numBlocks = 0;
        blockSize = 0;
        remainder = 0;
        return;
    }

    // Multi-core split: each core handles at least 1 row
    numBlocks = static_cast<uint32_t>(std::min(m, static_cast<int64_t>(maxCoreNum)));
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    // 余数分配法：blockSize=floor(m/numBlocks)，前 remainder 个核多处理 1 行。
    blockSize = m / static_cast<int64_t>(numBlocks);
    remainder = static_cast<uint32_t>(m % static_cast<int64_t>(numBlocks));
    // 余数核最大行数 = blockSize + 1（remainder > 0 时），用于 UB 预算约束。
    int64_t maxRowsPerBlock = blockSize + ((remainder > 0U) ? 1 : 0);

    int64_t ubForRowPtr = static_cast<int64_t>(availableUB) - kMinCooChunkBytes;
    if (ubForRowPtr <= 0) {
        numBlocks = 0;
        blockSize = 0;
        remainder = 0;
        return;
    }

    int64_t maxAlignedCount =
        (ubForRowPtr / kIdxByteSize / static_cast<int64_t>(kAlignElems)) *
         static_cast<int64_t>(kAlignElems);
    // rowPtr + nnzBuf + startBuf 三个 buffer 各占 rowPtrBytes（Gather 从 rpLocal
    // 提取 shift，无需独立 shiftBuf）。per-core rowPtr 占用 3x rowPtrBytes。
    // maxBlockSize 为 UB 能容纳的单核最大行数（rowPtrCountAligned = CeilAlign(maxBlockSize+1, 8)）。
    int64_t maxBlockSize = 0;
    if (maxAlignedCount > 0) {
        int64_t alignedThird = (maxAlignedCount / kRowPtrBufCount)
            / static_cast<int64_t>(kAlignElems)
            * static_cast<int64_t>(kAlignElems);
        maxBlockSize = (alignedThird > 0) ? (alignedThird - 1) : 0;
    }

    // 余数核最大行数超过 UB 容量时，减少每核行数（增加 numBlocks）。
    if (maxRowsPerBlock > maxBlockSize && maxBlockSize > 0) {
        int64_t rawNumBlocks = (m + maxBlockSize - 1) / maxBlockSize;
        if (rawNumBlocks > static_cast<int64_t>(maxCoreNum)) {
            rawNumBlocks = static_cast<int64_t>(maxCoreNum);
        }
        if (rawNumBlocks <= 0) {
            rawNumBlocks = 1;
        }
        numBlocks = static_cast<uint32_t>(rawNumBlocks);
        blockSize = m / static_cast<int64_t>(numBlocks);
        remainder = static_cast<uint32_t>(m % static_cast<int64_t>(numBlocks));
    }
}

// ============================================================================
// SelectSimtPath — SIMT 路径选择 + numBlocks 计算 + tiling 填充。
// ============================================================================
static aclsparseStatus_t SelectSimtPath(
    int64_t m, uint32_t maxCoreNum,
    csr2coo::Csr2CooTilingData &tiling, uint32_t &numBlocks)
{
    // SIMT 路径: 无 UB 分配(VF 直访 GM, 不创建 TPipe)。两层分解：调度器按
    // simtRowsPerBlock 预算核间行范围，VF 内核内 grid-stride（stride=threadNum）。
    numBlocks = static_cast<uint32_t>(std::min(m, static_cast<int64_t>(maxCoreNum)));
    if (numBlocks == 0) {
        numBlocks = 1;
    }
    tiling.simtRowsPerBlock = static_cast<uint32_t>(
        (m + numBlocks - 1) / numBlocks);  // ceil(m/numBlocks)

    tiling.useSimt = 1U;
    tiling.blockSize = 0;
    tiling.remainder = 0;
    tiling.cooChunkSize = 0;
    tiling.rowPtrBytes = 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// ComputeSimdUbBudget — SIMD 路径 UB 预算计算（rowPtrBytes + cooChunkSize）。
//   maxRowsPerBlock = blockSize + (remainder > 0 ? 1 : 0)  // 余数核多 1 行
//   rowPtrCountAligned = CeilAlign(maxRowsPerBlock + 1, 8)  // rowPtr 元素数 = 行数+1
// ============================================================================
static aclsparseStatus_t ComputeSimdUbBudget(
    uint64_t availableUB, int64_t blockSize, uint32_t remainder, int64_t m,
    uint64_t &rowPtrBytes, uint32_t &cooChunkSize)
{
    int64_t maxRowsPerBlock = blockSize + ((remainder > 0U) ? 1 : 0);
    int64_t rowPtrCountAligned =
        ((maxRowsPerBlock + 1 + static_cast<int64_t>(kAlignElems) - 1) / static_cast<int64_t>(kAlignElems)) *
         static_cast<int64_t>(kAlignElems);
    // rowPtrBytes 用 uint64_t 中间计算，防止极端 blockSize 时
    // rowPtrCountAligned * kIdxByteSize 溢出 uint32_t（m 可达 INT32_MAX）。
    // TilingData.rowPtrBytes 仍为 uint32_t（由 UB 物理上限保证不超 UINT32_MAX），
    // 赋值前在 ComputeTiling 中校验。
    rowPtrBytes = static_cast<uint64_t>(rowPtrCountAligned) * static_cast<uint64_t>(kIdxByteSize);

    // rowPtr + nnzBuf + startBuf each occupy rowPtrBytes in UB (Gather from rpLocal,
    // no separate shiftBuf; kernel allocates nnzBuf_/startBuf_ with the same
    // rowPtrBytes_ for 32B-aligned coverage; startBuf_ also temporarily holds the
    // Gather offset table before Subs overwrites it).
    if (static_cast<int64_t>(rowPtrBytes) * kRowPtrBufCount >
        static_cast<int64_t>(availableUB) - kMinCooChunkBytes) {
        OP_LOGE("aclsparseXcsr2coo",
                "rowPtrBytes(%lu)*3 leaves insufficient space for cooChunk (availableUB=%lu, "
                "blockSize=%ld, m=%ld)",
                rowPtrBytes, availableUB, blockSize, m);
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    // cooQueue_ num=2 (V/MTE3 double-buffer). coo 相关 UB = accumBuf(1) + cooQueue[2](2)
    // = 3 × cooBytes。除数为 3。
    constexpr int64_t kCooBufCount = 3;
    int64_t cooInducedBytes = static_cast<int64_t>(availableUB) -
                              static_cast<int64_t>(rowPtrBytes) * kRowPtrBufCount;
    int64_t cooChunkSizeLocal =
        (cooInducedBytes / kCooBufCount / kIdxByteSize / static_cast<int64_t>(kAlignElems)) *
         static_cast<int64_t>(kAlignElems);

    if (cooChunkSizeLocal <= 0) {
        OP_LOGE("aclsparseXcsr2coo",
                "cooChunkSize=%ld <= 0, insufficient UB (availableUB=%lu, rowPtrBytes=%lu)",
                cooChunkSizeLocal, availableUB, rowPtrBytes);
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    cooChunkSize = static_cast<uint32_t>(cooChunkSizeLocal);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// QueryUbAndSplit: UB 查询 + ComputeBlockSplit + numBlocks 校验。
// ============================================================================
static aclsparseStatus_t QueryUbAndSplit(
    int64_t m, uint32_t maxCoreNum,
    uint64_t &availableUB, uint32_t &numBlocks, int64_t &blockSize, uint32_t &remainder)
{
    uint64_t ubSize = GetUbSize();
    if (ubSize == 0) {
        OP_LOGE("aclsparseXcsr2coo", "GetUbSize returned 0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    if (ubSize <= kSystemReserved) {
        OP_LOGE("aclsparseXcsr2coo",
                "GetUbSize(%lu) <= kSystemReserved(%u), insufficient UB",
                ubSize, kSystemReserved);
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    availableUB = ubSize - kSystemReserved;

    ComputeBlockSplit(m, availableUB, maxCoreNum, numBlocks, blockSize, remainder);
    if (numBlocks == 0) {
        OP_LOGE("aclsparseXcsr2coo",
                "insufficient UB for rowPtr buffers (availableUB=%lu)",
                availableUB);
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Tiling computation: path selection (SIMT/SIMD) + TilingData population
// ============================================================================
static aclsparseStatus_t ComputeTiling(
    int64_t m, int64_t nnz, aclsparseIndexBase_t idxBase,
    csr2coo::Csr2CooTilingData &tiling, uint32_t &numBlocks)
{
    uint32_t maxCoreNum = GetAivCoreCount();
    if (maxCoreNum == 0) {
        OP_LOGE("aclsparseXcsr2coo", "GetAivCoreCount returned 0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    tiling.m = m;
    tiling.nnz = nnz;
    tiling.idxBase = (idxBase == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    // -----------------------------------------------------------------------
    // 路径选择基于矩阵特征(m, nnz/m)，不依赖核数。
    //   SIMT 路径(小行多行): m > 1 且 平均每行 nnz <= 阈值 → 线程并行隐藏
    //     标量 GetValue 延迟。每线程处理1行, VF 直访 GM, 无 TPipe/TQue。
    //   SIMD 路径(大行/ultrarow): 否则 → 保留 Duplicate+DataCopyPad, 一次填大块。
    // 阈值 128: 参考 moe_init_routing_story(SIMT 适合每元素计算轻量、行数多场景);
    //   平均每行 <=128 元素时, 单行 SIMD Duplicate 启动开销相对收益不足, SIMT 更优。
    // 不依赖核数的原因: GetAivCoreCount() 返回值因芯片版本而异，依赖核数会导致
    //   小规模 multicore 场景误走 SIMD。改为纯矩阵特征判断后路径选择稳定。
    //   m=1(ultrarow) 走 SIMD; m=1 且 nnz 很小 时 nnz/m<=128 但 m>1 为 false 仍走
    //   SIMD(1 行场景 SIMD 足够)。
    // -----------------------------------------------------------------------
    constexpr int64_t kSimtAvgNnzThreshold = 128;
    bool useSimt = (m > 1) && (nnz / m <= kSimtAvgNnzThreshold);

    if (useSimt) {
        return SelectSimtPath(m, maxCoreNum, tiling, numBlocks);
    }

    // -----------------------------------------------------------------------
    // SIMD 路径: UB 预算 + ComputeBlockSplit 逻辑（余数分配法）。
    // -----------------------------------------------------------------------
    tiling.useSimt = 0U;

    uint64_t availableUB = 0;
    aclsparseStatus_t ubStatus = QueryUbAndSplit(m, maxCoreNum, availableUB,
                                                   numBlocks, tiling.blockSize, tiling.remainder);
    if (ubStatus != ACL_SPARSE_STATUS_SUCCESS) {
        return ubStatus;
    }
    int64_t blockSize = tiling.blockSize;
    uint32_t remainder = tiling.remainder;

    uint64_t rowPtrBytes = 0;
    uint32_t cooChunkSize = 0;
    aclsparseStatus_t status = ComputeSimdUbBudget(availableUB, blockSize, remainder, m,
                                                     rowPtrBytes, cooChunkSize);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    // 校验 rowPtrBytes 不超 UINT32_MAX（TilingData.rowPtrBytes 为 uint32_t）。
    // 正常情况 rowPtrBytes 由 UB 物理上限约束（远小于 UINT32_MAX），此校验防御
    // 极端 blockSize 场景下的溢出。
    if (rowPtrBytes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE("aclsparseXcsr2coo",
                "rowPtrBytes(%lu) exceeds UINT32_MAX (blockSize=%ld, m=%ld)",
                rowPtrBytes, blockSize, m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    tiling.cooChunkSize = cooChunkSize;
    tiling.rowPtrBytes = static_cast<uint32_t>(rowPtrBytes);

    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Kernel launch: extract stream from handle and dispatch to kernel_do
// ============================================================================
static aclsparseStatus_t LaunchKernel(
    aclsparseHandle_t handle,
    const int32_t *csrRowPtr, int32_t *cooRowInd,
    uint32_t numBlocks, const csr2coo::Csr2CooTilingData &tiling)
{
    // stream 为 nullptr 时使用默认 stream（符合 aclsparseSetStream 文档约定，
    // 与 cuSPARSE §4.2.9 及同仓 nnz_host.cpp:204-205 行为一致）。

    // const_cast 因 GM_ADDR 宏强制 uint8_t* 类型所致，kernel 仅读 csrRowPtr，不写。
    GM_ADDR gmRowPtr = reinterpret_cast<GM_ADDR>(const_cast<int32_t *>(csrRowPtr));
    GM_ADDR gmCooRowInd = reinterpret_cast<GM_ADDR>(cooRowInd);

    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    csr2coo_kernel_do(gmRowPtr, gmCooRowInd, tiling, numBlocks, ToInternalHandle(handle)->stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // anonymous namespace

// ----------------------------------------------------------------------------
// Public API: aclsparseXcsr2coo (Legacy)
// ----------------------------------------------------------------------------
extern "C" aclsparseStatus_t aclsparseXcsr2coo(
    aclsparseHandle_t handle,
    const int32_t *csrRowPtr,
    int64_t nnz,
    int64_t m,
    int32_t *cooRowInd,
    aclsparseIndexBase_t idxBase)
{
    // 1. Validate
    aclsparseStatus_t status = ValidateInputs(handle, csrRowPtr, nnz, m, cooRowInd, idxBase);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    if (nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    csr2coo::Csr2CooTilingData tiling{};
    uint32_t numBlocks = 0;
    status = ComputeTiling(m, nnz, idxBase, tiling, numBlocks);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    OP_LOGD("aclsparseXcsr2coo",
            "tiling: m=%ld, nnz=%ld, idxBase=%d, useSimt=%u, blockSize=%ld, "
            "remainder=%u, cooChunkSize=%u, rowPtrBytes=%u, simtRowsPerBlock=%u, numBlocks=%u",
            tiling.m, tiling.nnz,
            tiling.idxBase, tiling.useSimt, tiling.blockSize,
            tiling.remainder, tiling.cooChunkSize, tiling.rowPtrBytes,
            tiling.simtRowsPerBlock, numBlocks);

    status = LaunchKernel(handle, csrRowPtr, cooRowInd, numBlocks, tiling);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    OP_LOGI("aclsparseXcsr2coo", "Kernel launched: numBlocks=%u", numBlocks);
    return ACL_SPARSE_STATUS_SUCCESS;
}
