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
 * \file csr2gebsr_host.cpp
 * \brief csr2gebsr Host 侧实现：9 个 Legacy API 入口（三步法）。
 *
 * 1. bufferSize (×4 精度版本) - 查询 workspace 大小（纯 Host 计算）
 * 2. Nnz (×1)                 - 计算 bsrRowPtrC 和 nnzb
 * 3. Convert (×4 精度版本)     - 填充 bsrColIndC + bsrValC
 *
 * 结构：每个 API 内部拆分为 ValidateParams + LaunchKernel 两个静态函数。
 *
 * 输入数据契约（调用方必须保证，Host 侧无法校验 device 数据内容）：
 *   - CSR 行内列索引必须按升序排列（标准 CSR 约定）。Convert 阶段 BinarySearchCsrRow
 *     依赖此有序假设做二分查找，乱序输入将静默产出错误 bsrValC。
 *   - csrColIndA 每个元素（减去 baseA 后）必须落在 [0, n) 区间内。越界列索引会导致
 *     marker 数组越界访问。Kernel 侧已对 colInd 做防御性范围检查（越界元跳过），
 *     但越界元会被静默丢弃，调用方仍需保证输入合法。
 *   该契约与 cuSPARSE 同类接口（cusparseXcsr2gebsrNnz / cusparse[S|H]csr2gebsr）一致。
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "csr2gebsr.h"
#include "csr2gebsr_tiling_data.h"
#include "csr2gebsr_kernel.h"

namespace {

// ===========================================================================
// 公共辅助函数
// ===========================================================================

/// 安全向上取整除法：防御除零（rowBlockDim/colBlockDim 已在 ValidateCommonParams 中校验 >= 1）
static inline int32_t SafeCeilDiv(int32_t value, int32_t divisor) {
    return (divisor > 0) ? (value + divisor - 1) / divisor : 0;
}

/// 校验 dir 参数合法性
static aclsparseStatus_t ValidateDirection(const char *apiName, aclsparseDirection_t dir)
{
    if (dir != ACL_SPARSE_DIRECTION_ROW && dir != ACL_SPARSE_DIRECTION_COLUMN) {
        OP_LOGE(apiName, "invalid dir: %d", static_cast<int>(dir));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/// 校验维度上限，防止 mb/nb/markerBase/blockSize 计算溢出。
/// - m, n, rowBlockDim, colBlockDim 各自 ≤ INT32_MAX/2：
///   m ≤ INT32_MAX/2 且 rowBlockDim ≤ INT32_MAX/2 时，m + rowBlockDim - 1 ≤ INT32_MAX - 1，不溢出。
/// - mb×nb ≤ INT32_MAX：防止 kernel 侧 markerBase = actualBr×nb 乘积溢出 int32。
static aclsparseStatus_t ValidateDimUpperBound(
    const char *apiName, int m, int n, int rowBlockDim, int colBlockDim)
{
    constexpr int32_t kDimUpperBound = INT32_MAX / 2;
    if (m > kDimUpperBound) {
        OP_LOGE(apiName, "m too large (>%d), block count may overflow: %d",
                kDimUpperBound, m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n > kDimUpperBound) {
        OP_LOGE(apiName, "n too large (>%d), block count may overflow: %d",
                kDimUpperBound, n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // rowBlockDim/colBlockDim 上限校验，使 (m + rowBlockDim - 1) 不溢出
    if (rowBlockDim > kDimUpperBound) {
        OP_LOGE(apiName, "rowBlockDim too large (>%d): %d",
                kDimUpperBound, rowBlockDim);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (colBlockDim > kDimUpperBound) {
        OP_LOGE(apiName, "colBlockDim too large (>%d): %d",
                kDimUpperBound, colBlockDim);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // mb×nb 乘积上限校验，防止 kernel 侧 markerBase int32 溢出
    int32_t mb = SafeCeilDiv(m, rowBlockDim);
    int32_t nb = SafeCeilDiv(n, colBlockDim);
    if (static_cast<int64_t>(mb) * nb > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE(apiName, "mb*nb too large (>%d): mb=%d, nb=%d",
                INT32_MAX, mb, nb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/// 当 m=0 时填充 bsrRowPtrC 和 nnzb（仅 mb=0 场景，写入 1 个元素）
/// 同步 aclrtMemcpy：源地址 &baseC/&zero 为栈变量，函数返回后栈帧释放会导致
/// use-after-free。m=0 路径无前序 stream 操作，同步拷贝的流序代价可忽略。
static aclsparseStatus_t FillEmptyBsrRowPtrC(
    const aclsparseContext *h, aclrtStream stream,
    int32_t *bsrRowPtrC, int32_t *nnzTotalDevHostPtr,
    int32_t baseC)
{
    // 同步拷贝：源地址为栈变量，需在函数返回前完成
    (void)stream;
    aclError aclRet = aclrtMemcpy(
        bsrRowPtrC, sizeof(int32_t),
        &baseC, sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_ERROR_NONE,
              OP_LOGE("aclsparseXcsr2gebsrNnz",
                      "H2D copy bsrRowPtrC[0] failed, ret=%d", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    if (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        *nnzTotalDevHostPtr = 0;
    } else {
        int32_t zero = 0;
        // 同步拷贝：源地址为栈变量，需在函数返回前完成
        aclError aclRetNnz = aclrtMemcpy(
            nnzTotalDevHostPtr, sizeof(int32_t),
            &zero, sizeof(int32_t),
            ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRetNnz == ACL_ERROR_NONE,
                  OP_LOGE("aclsparseXcsr2gebsrNnz",
                          "H2D copy nnzb failed, ret=%d", aclRetNnz);
                  return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

/// 计算多 Block 并行切分参数（按块行 mb 切分到多个 AI Core）
static aclsparseStatus_t ComputeBlockSplits(
    const char *apiName, int32_t mb,
    uint32_t &useBlocks, uint32_t &blockRowsPerCore)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE(apiName, "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    useBlocks = std::min(aivCoreNum,
        CeilDiv<uint32_t>(static_cast<uint32_t>(mb), kCsr2gebsrMaxThreadsPerBlock));
    if (useBlocks == 0) {
        useBlocks = 1;
    }
    blockRowsPerCore = CeilDiv<uint32_t>(static_cast<uint32_t>(mb), useBlocks);
    if (blockRowsPerCore == 0) {
        blockRowsPerCore = 1;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// bufferSize - 公共实现（4 个精度版本共用）
// ===========================================================================

/// 公共基础参数校验：m/n/blockDim/dir/descrA（三步法共用）
static aclsparseStatus_t ValidateCommonParams(
    const char *apiName, int m, int n,
    int rowBlockDim, int colBlockDim,
    aclsparseDirection_t dir,
    const aclsparseMatDescr_t descrA)
{
    if (m < 0) {
        OP_LOGE(apiName, "invalid m: %d", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE(apiName, "invalid n: %d", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rowBlockDim < 1) {
        OP_LOGE(apiName, "invalid rowBlockDim: %d", rowBlockDim);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (colBlockDim < 1) {
        OP_LOGE(apiName, "invalid colBlockDim: %d", colBlockDim);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t dimSt = ValidateDimUpperBound(apiName, m, n, rowBlockDim, colBlockDim);
    if (dimSt != ACL_SPARSE_STATUS_SUCCESS) {
        return dimSt;
    }

    aclsparseStatus_t st = ValidateDirection(apiName, dir);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    return Csr2gebsrValidateMatDescr(apiName, "descrA", descrA);
}

/// bufferSize 参数校验
static aclsparseStatus_t ValidateBufferSizeParams(
    const char *apiName,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    int rowBlockDim, int colBlockDim,
    aclsparseDirection_t dir,
    size_t *pBufferSizeInBytes)
{
    aclsparseStatus_t st = ValidateCommonParams(
        apiName, m, n, rowBlockDim, colBlockDim, dir, descrA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (m > 0 && csrRowPtrA == nullptr) {
        OP_LOGE(apiName, "csrRowPtrA is nullptr (m=%d)", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (pBufferSizeInBytes == nullptr) {
        OP_LOGE(apiName, "pBufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

/// bufferSize 公共计算逻辑
static aclsparseStatus_t ComputeBufferSize(
    const char *apiName,
    int m, int n, int rowBlockDim, int colBlockDim,
    size_t *pBufferSizeInBytes)
{
    int32_t mb = SafeCeilDiv(m, rowBlockDim);
    int32_t nb = SafeCeilDiv(n, colBlockDim);

    if (m == 0) {
        *pBufferSizeInBytes = 0;
        OP_LOGD(apiName, "empty matrix (m=%d, n=%d), workspaceBytes=0", m, n);
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    size_t markerCount = static_cast<size_t>(mb) * static_cast<size_t>(nb);
    uint32_t aivCoreNum = GetAivCoreCount();
    size_t segSumCount = static_cast<size_t>(aivCoreNum);
    size_t workspaceBytes =
        (static_cast<size_t>(mb) + 1 + markerCount + segSumCount) * sizeof(int32_t);
    *pBufferSizeInBytes = workspaceBytes;

    OP_LOGD(apiName, "m=%d, n=%d, mb=%d, nb=%d, workspaceBytes=%zu",
            m, n, mb, nb, workspaceBytes);

    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// Nnz - 参数校验 + Kernel launch
// ===========================================================================

/// Nnz 参数校验
static aclsparseStatus_t ValidateNnzParams(
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    int *bsrRowPtrC, int rowBlockDim, int colBlockDim,
    aclsparseDirection_t dir,
    int *nnzTotalDevHostPtr,
    void *pBuffer)
{
    const char *apiName = "aclsparseXcsr2gebsrNnz";

    aclsparseStatus_t st = ValidateCommonParams(
        apiName, m, n, rowBlockDim, colBlockDim, dir, descrA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    st = Csr2gebsrValidateMatDescr(apiName, "descrC", descrC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (m > 0 && csrRowPtrA == nullptr) {
        OP_LOGE(apiName, "csrRowPtrA is nullptr (m=%d)", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // csrColIndA null 校验受限：Nnz API 无 nnz 输入参数，无法区分
    // "nnz=0 + null csrColIndA"（合法，kernel 循环不执行）与 "nnz>0 + null
    // csrColIndA"（非法，kernel 解引用空指针）。故 Nnz 侧不强制校验 csrColIndA
    // null，依赖调用方契约：有非零元时必须提供合法 csrColIndA。
    if (bsrRowPtrC == nullptr) {
        OP_LOGE(apiName, "bsrRowPtrC is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzTotalDevHostPtr == nullptr) {
        OP_LOGE(apiName, "nnzTotalDevHostPtr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (m > 0 && pBuffer == nullptr) {
        OP_LOGE(apiName, "pBuffer is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    OP_LOGD(apiName,
            "CSR input contract (caller-guaranteed): csrColIndA row-wise ascending, "
            "each entry in [baseA, baseA+n); host cannot validate device data content");

    return ACL_SPARSE_STATUS_SUCCESS;
}

/// marker 区初始化(-1) + 多核切分（PrepareNnzWorkspace 的 n>0 分支与
/// PrepareConvertWorkspace 共用，#372：消 13 行重复块）。
static aclsparseStatus_t InitMarkerAndSplits(
    const char *apiName,
    aclrtStream stream,
    int32_t *wsBase, int32_t mb, int32_t nb,
    uint32_t &useBlocks, uint32_t &blockRowsPerCore)
{
    // memset marker 区域为 -1
    // aclrtMemsetAsync 第 3 参数取低字节按字节填充，
    //   0xFF → 每字节 0xFF → int32_t = 0xFFFFFFFF = -1，与 marker 初值 -1 语义一致
    // marker 大小 = mb × nb（per-block-row）
    size_t markerBytes = static_cast<size_t>(mb) * static_cast<size_t>(nb) * sizeof(int32_t);
    aclError aclRet = aclrtMemsetAsync(
        wsBase + mb + 1, markerBytes,
        static_cast<int32_t>(0xFF), markerBytes,
        stream);
    CHECK_RET(aclRet == ACL_ERROR_NONE,
              OP_LOGE(apiName, "memset marker failed, ret=%d", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    return ComputeBlockSplits(apiName, mb, useBlocks, blockRowsPerCore);
}

/// Nnz workspace 准备 - memset marker(或 nnzBlocksPerRow) + 多核切分
static aclsparseStatus_t PrepareNnzWorkspace(
    aclrtStream stream,
    int32_t *wsBase, int32_t mb, int32_t nb, int32_t n,
    uint32_t &useBlocks, uint32_t &blockRowsPerCore)
{
    if (n > 0) {
        return InitMarkerAndSplits(
            "aclsparseXcsr2gebsrNnz", stream, wsBase, mb, nb,
            useBlocks, blockRowsPerCore);
    }
    // n=0: 无列块，nnzBlocksPerRow 全零
    aclError aclRet = aclrtMemsetAsync(
        wsBase, static_cast<size_t>(mb) * sizeof(int32_t),
        0, static_cast<size_t>(mb) * sizeof(int32_t), stream);
    CHECK_RET(aclRet == ACL_ERROR_NONE,
              OP_LOGE("aclsparseXcsr2gebsrNnz",
                      "memset nnzBlocksPerRow failed, ret=%d", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    return ACL_SPARSE_STATUS_SUCCESS;
}

/// Nnz TilingData 填充
static void FillNnzTiling(
    int m, int n, int32_t mb, int32_t nb,
    int rowBlockDim, int colBlockDim, int32_t baseA,
    uint32_t blockRowsPerCore,
    Csr2gebsrNnzTilingData &nnzTiling)
{
    nnzTiling.m = static_cast<int32_t>(m);
    nnzTiling.n = static_cast<int32_t>(n);
    nnzTiling.mb = mb;
    nnzTiling.nb = nb;
    nnzTiling.rowBlockDim = rowBlockDim;
    nnzTiling.colBlockDim = colBlockDim;
    nnzTiling.baseA = baseA;
    nnzTiling.blockRowsPerCore = blockRowsPerCore;
    nnzTiling.invColBlockDim = 1.0f / static_cast<float>(colBlockDim);
}

/// PrefixSum Kernel launch + nnzb 回传
///
/// useBlocks == 1（mb ≤ 128）走退化串行路径（原单 kernel，避免小 mb 多 launch 开销）。
/// useBlocks > 1 走融合单 kernel 路径（Phase1/2/3 + SyncAll，1 次 launch），
/// 消除原三阶段方案的 2 次额外 launch 开销。原 p1/p2/p3 kernel 保留但不再调用。
static aclsparseStatus_t LaunchPrefixSumAndCopyback(
    aclrtStream stream, const aclsparseContext *h,
    GM_ADDR gmNnzBlocksPerRow, GM_ADDR gmBsrRowPtrC, GM_ADDR gmNnzbDev,
    GM_ADDR gmSegSum,
    int32_t *nnzbDevPtr, int32_t mb, int32_t baseC,
    int *nnzTotalDevHostPtr)
{
    // 复用 ComputeBlockSplits 计算切分参数（与 Kernel 1 对齐）
    uint32_t useBlocks = 0;
    uint32_t blockRowsPerCore = 0;
    aclsparseStatus_t splitSt = ComputeBlockSplits(
        "aclsparseXcsr2gebsrNnz", mb, useBlocks, blockRowsPerCore);
    if (splitSt != ACL_SPARSE_STATUS_SUCCESS) {
        return splitSt;
    }

    Csr2gebsrPrefixSumTilingData psTiling{};
    psTiling.mb = mb;
    psTiling.baseC = baseC;
    psTiling.useBlocks = useBlocks;
    psTiling.blockRowsPerCore = blockRowsPerCore;

    if (useBlocks == 1) {
        // 退化路径：原单 kernel 串行 O(mb) prefix sum
        // clear-then-launch-then-peek（issue #357 分族）
        (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

        csr2gebsr_prefixsum_kernel_do(
            gmNnzBlocksPerRow, gmBsrRowPtrC, gmNnzbDev,
            psTiling, stream);
    } else {
        // 融合并行路径：单 kernel Phase1→SyncAll→Phase2→SyncAll→Phase3
        // clear-then-launch-then-peek（issue #357 分族）
        (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

        csr2gebsr_prefixsum_fused_kernel_do(
            gmNnzBlocksPerRow, gmBsrRowPtrC, gmNnzbDev, gmSegSum,
            psTiling, useBlocks, stream);
    }

    OP_LOGD("aclsparseXcsr2gebsrNnz",
            "Kernel 2 (prefix sum) launched, mb=%d, baseC=%d, useBlocks=%u",
            mb, baseC, useBlocks);

    // 异步拷贝 nnzbDev → nnzTotalDevHostPtr
    if (nnzTotalDevHostPtr != nullptr) {
        aclrtMemcpyKind kind = (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST)
                                    ? ACL_MEMCPY_DEVICE_TO_HOST
                                    : ACL_MEMCPY_DEVICE_TO_DEVICE;
        aclError aclRetCopy = aclrtMemcpyAsync(
            nnzTotalDevHostPtr, sizeof(int32_t),
            nnzbDevPtr, sizeof(int32_t),
            kind, stream);
        CHECK_RET(aclRetCopy == ACL_ERROR_NONE,
                  OP_LOGE("aclsparseXcsr2gebsrNnz",
                          "async copy nnzb failed, ret=%d", aclRetCopy);
                  return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    }
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseXcsr2gebsrNnz", "prefixsum enqueue failed");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/// Nnz Kernel launch（编排：PrepareNnzWorkspace → Kernel1 → LaunchPrefixSumAndCopyback）
static aclsparseStatus_t LaunchNnzKernel(
    aclsparseHandle_t handle,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    int *bsrRowPtrC, int rowBlockDim, int colBlockDim,
    int *nnzTotalDevHostPtr,
    void *pBuffer)
{
    auto *h = Csr2gebsrToInternalHandle(handle);
    aclrtStream stream = h->stream;

    int32_t baseA = Csr2gebsrGetBase(descrA);
    int32_t baseC = Csr2gebsrGetBase(descrC);
    int32_t mb = SafeCeilDiv(m, rowBlockDim);
    int32_t nb = SafeCeilDiv(n, colBlockDim);

    // workspace 布局: [nnzBlocksPerRow(mb) | nnzbDev(1) | marker(mb×nb) | segSum(aivCoreNum)]
    auto *wsBase = reinterpret_cast<int32_t *>(pBuffer);
    auto *gmNnzBlocksPerRow = reinterpret_cast<GM_ADDR>(wsBase);
    auto *gmNnzbDev = reinterpret_cast<GM_ADDR>(wsBase + mb);

    uint32_t useBlocks = 0;
    uint32_t blockRowsPerCore = 0;
    aclsparseStatus_t prepSt = PrepareNnzWorkspace(
        stream, wsBase, mb, nb, n, useBlocks, blockRowsPerCore);
    if (prepSt != ACL_SPARSE_STATUS_SUCCESS) {
        return prepSt;
    }

    // Kernel 1 仅在 n > 0 时运行（n=0 时无列块，nnzBlocksPerRow 已由 Prepare 清零）
    if (n > 0) {
        Csr2gebsrNnzTilingData nnzTiling{};
        FillNnzTiling(m, n, mb, nb, rowBlockDim, colBlockDim,
                      baseA, blockRowsPerCore, nnzTiling);
        auto *gmMarker = reinterpret_cast<GM_ADDR>(wsBase + mb + 1);
        auto *gmRowPtrA = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrRowPtrA));
        auto *gmColIndA = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrColIndA));
        // clear-then-launch-then-peek（issue #357 分族，评审响应：peek 移到下发后）
        (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

        csr2gebsr_nnz_kernel_do(
            gmRowPtrA, gmColIndA, gmNnzBlocksPerRow, gmMarker,
            nnzTiling, useBlocks, stream);
        if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
            OP_LOGE("aclsparseXcsr2gebsrNnz", "nnz kernel enqueue failed");
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        OP_LOGD("aclsparseXcsr2gebsrNnz",
                "Kernel 1 launched: mb=%d, nb=%d, blocks=%u", mb, nb, useBlocks);
    }

    auto *gmBsrRowPtrC = reinterpret_cast<GM_ADDR>(bsrRowPtrC);
    // segSum 位于 workspace 末尾: wsBase + mb + 1 + mb*nb
    // 用 int64_t 计算偏移，防止 mb + 1 + mb*nb 溢出 int32
    int64_t segSumOffset = static_cast<int64_t>(mb) + 1 +
                           static_cast<int64_t>(mb) * static_cast<int64_t>(nb);
    auto *gmSegSum = reinterpret_cast<GM_ADDR>(wsBase + segSumOffset);
    return LaunchPrefixSumAndCopyback(
        stream, h, gmNnzBlocksPerRow, gmBsrRowPtrC, gmNnzbDev, gmSegSum,
        wsBase + mb, mb, baseC, nnzTotalDevHostPtr);
}

// ===========================================================================
// Convert - 参数校验 + Kernel launch
// ===========================================================================

/// Convert 参数校验
static aclsparseStatus_t ValidateConvertParams(
    const char *apiName,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const void *csrValA,
    const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    const void *bsrValC,
    int *bsrRowPtrC, int *bsrColIndC,
    int rowBlockDim, int colBlockDim,
    aclsparseDirection_t dir,
    void *pBuffer)
{
    aclsparseStatus_t st = ValidateCommonParams(
        apiName, m, n, rowBlockDim, colBlockDim, dir, descrA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    st = Csr2gebsrValidateMatDescr(apiName, "descrC", descrC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (m > 0 && csrValA == nullptr) {
        OP_LOGE(apiName, "csrValA is nullptr (m=%d)", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (m > 0 && csrRowPtrA == nullptr) {
        OP_LOGE(apiName, "csrRowPtrA is nullptr (m=%d)", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (m > 0 && csrColIndA == nullptr) {
        OP_LOGE(apiName, "csrColIndA is nullptr (m=%d)", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0 && bsrValC == nullptr) {
        OP_LOGE(apiName, "bsrValC is nullptr (m=%d, n=%d)", m, n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (bsrRowPtrC == nullptr) {
        OP_LOGE(apiName, "bsrRowPtrC is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0 && bsrColIndC == nullptr) {
        OP_LOGE(apiName, "bsrColIndC is nullptr (m=%d, n=%d)", m, n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0 && pBuffer == nullptr) {
        OP_LOGE(apiName, "pBuffer is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    OP_LOGD(apiName,
            "CSR input contract (caller-guaranteed): csrColIndA row-wise ascending, "
            "each entry in [baseA, baseA+n); host cannot validate device data content");

    return ACL_SPARSE_STATUS_SUCCESS;
}

/// Convert workspace 准备 - memset marker + 多核切分
static aclsparseStatus_t PrepareConvertWorkspace(
    const char *apiName,
    aclrtStream stream,
    int32_t *wsBase, int32_t mb, int32_t nb,
    uint32_t &useBlocks, uint32_t &blockRowsPerCore)
{
    return InitMarkerAndSplits(
        apiName, stream, wsBase, mb, nb, useBlocks, blockRowsPerCore);
}

/// Convert TilingData 填充（11 字段）
static void FillConvertTiling(
    int m, int n, int32_t mb, int32_t nb,
    int rowBlockDim, int colBlockDim,
    int32_t baseA, int32_t baseC,
    aclsparseDirection_t dir, uint32_t valSize,
    uint32_t blockRowsPerCore,
    Csr2gebsrConvertTilingData &tiling)
{
    tiling.m = static_cast<int32_t>(m);
    tiling.n = static_cast<int32_t>(n);
    tiling.mb = mb;
    tiling.nb = nb;
    tiling.rowBlockDim = rowBlockDim;
    tiling.colBlockDim = colBlockDim;
    tiling.baseA = baseA;
    tiling.baseC = baseC;
    tiling.dir = (dir == ACL_SPARSE_DIRECTION_ROW) ? 0 : 1;
    tiling.valSize = valSize;
    tiling.blockRowsPerCore = blockRowsPerCore;
    tiling.invColBlockDim = 1.0f / static_cast<float>(colBlockDim);
}

/// Convert Kernel launch（编排：PrepareConvertWorkspace → FillConvertTiling → Kernel3）
static aclsparseStatus_t LaunchConvertKernel(
    aclsparseHandle_t handle,
    aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    GM_ADDR csrValA,
    const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    GM_ADDR bsrValC,
    int *bsrRowPtrC, int *bsrColIndC,
    int rowBlockDim, int colBlockDim,
    uint32_t valSize,
    void *pBuffer)
{
    auto *h = Csr2gebsrToInternalHandle(handle);
    aclrtStream stream = h->stream;

    int32_t baseA = Csr2gebsrGetBase(descrA);
    int32_t baseC = Csr2gebsrGetBase(descrC);
    int32_t mb = SafeCeilDiv(m, rowBlockDim);
    int32_t nb = SafeCeilDiv(n, colBlockDim);

    auto *wsBase = reinterpret_cast<int32_t *>(pBuffer);

    uint32_t useBlocks = 0;
    uint32_t blockRowsPerCore = 0;
    aclsparseStatus_t prepSt = PrepareConvertWorkspace(
        "aclsparseCsr2gebsr", stream, wsBase, mb, nb, useBlocks, blockRowsPerCore);
    if (prepSt != ACL_SPARSE_STATUS_SUCCESS) {
        return prepSt;
    }

    Csr2gebsrConvertTilingData tiling{};
    FillConvertTiling(m, n, mb, nb, rowBlockDim, colBlockDim,
                      baseA, baseC, dir, valSize, blockRowsPerCore, tiling);

    OP_LOGD("aclsparseCsr2gebsr",
            "tiling: m=%d, n=%d, mb=%d, nb=%d, valSize=%u, blocks=%u",
            m, n, mb, nb, valSize, useBlocks);

    auto *gmMarker = reinterpret_cast<GM_ADDR>(wsBase + mb + 1);
    auto *gmRowPtrA = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrRowPtrA));
    auto *gmColIndA = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrColIndA));
    auto *gmBsrRowPtrC = reinterpret_cast<GM_ADDR>(bsrRowPtrC);
    auto *gmBsrColIndC = reinterpret_cast<GM_ADDR>(bsrColIndC);

    // clear-then-launch-then-peek（issue #357 分族）
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    csr2gebsr_convert_kernel_do(
        csrValA, gmRowPtrA, gmColIndA,
        gmBsrRowPtrC, gmBsrColIndC, bsrValC,
        gmMarker, tiling, useBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseCsr2gebsr", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }



    OP_LOGI("aclsparseCsr2gebsr",
            "Kernel 3 (convert) launched, blocks=%u", useBlocks);

    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

// ============================================================================
// Public APIs
// ============================================================================
extern "C" {

// ===========================================================================
// bufferSize (4 个精度版本)
// ===========================================================================

static aclsparseStatus_t Csr2gebsrBufferSizeCommon(
    const char *apiName,
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n, const aclsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    int rowBlockDim, int colBlockDim, size_t *pBufferSizeInBytes)
{
    if (handle == nullptr) {
        OP_LOGE(apiName, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = ValidateBufferSizeParams(
        apiName, m, n, descrA, csrRowPtrA, rowBlockDim, colBlockDim,
        dir, pBufferSizeInBytes);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    return ComputeBufferSize(
        apiName, m, n, rowBlockDim, colBlockDim, pBufferSizeInBytes);
}

aclsparseStatus_t aclsparseScsr2gebsr_bufferSize(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n, const aclsparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    int rowBlockDim, int colBlockDim, size_t *pBufferSizeInBytes)
{
    return Csr2gebsrBufferSizeCommon(
        "aclsparseScsr2gebsr_bufferSize", handle, dir, m, n, descrA,
        csrRowPtrA, rowBlockDim, colBlockDim, pBufferSizeInBytes);
}

aclsparseStatus_t aclsparseHcsr2gebsr_bufferSize(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n, const aclsparseMatDescr_t descrA,
    const void *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    int rowBlockDim, int colBlockDim, size_t *pBufferSizeInBytes)
{
    return Csr2gebsrBufferSizeCommon(
        "aclsparseHcsr2gebsr_bufferSize", handle, dir, m, n, descrA,
        csrRowPtrA, rowBlockDim, colBlockDim, pBufferSizeInBytes);
}

aclsparseStatus_t aclsparseBhcsr2gebsr_bufferSize(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n, const aclsparseMatDescr_t descrA,
    const void *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    int rowBlockDim, int colBlockDim, size_t *pBufferSizeInBytes)
{
    return Csr2gebsrBufferSizeCommon(
        "aclsparseBhcsr2gebsr_bufferSize", handle, dir, m, n, descrA,
        csrRowPtrA, rowBlockDim, colBlockDim, pBufferSizeInBytes);
}

aclsparseStatus_t aclsparseIcsr2gebsr_bufferSize(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n, const aclsparseMatDescr_t descrA,
    const int *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    int rowBlockDim, int colBlockDim, size_t *pBufferSizeInBytes)
{
    return Csr2gebsrBufferSizeCommon(
        "aclsparseIcsr2gebsr_bufferSize", handle, dir, m, n, descrA,
        csrRowPtrA, rowBlockDim, colBlockDim, pBufferSizeInBytes);
}

// ===========================================================================
// Nnz (1 个版本，类型无关)
// ===========================================================================

/// 契约（调用方必须遵守）：
///   - pointerMode 与 nnzTotalDevHostPtr 指针类型必须一致：HOST 模式传 host 指针
///     （算子 D2H 回传 nnzb），DEVICE 模式传 device 指针（算子 D2D 回传）。
///     类型不一致会导致向 device 地址写 host 数据（或反之），属调用方契约违反。
///   - 本 API 与后续 Convert API 均异步（on stream），算子不在两步间插入同步。
///     调用方在调用 Convert 前必须同步 stream（或读回 nnzb 确认 Nnz 完成），
///     否则 Convert 读到的 bsrRowPtrC 可能为 Nnz 尚未写完的旧值。
aclsparseStatus_t aclsparseXcsr2gebsrNnz(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    int *bsrRowPtrC,
    int rowBlockDim, int colBlockDim,
    int *nnzTotalDevHostPtr,
    void *pBuffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseXcsr2gebsrNnz", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = ValidateNnzParams(
        m, n, descrA, csrRowPtrA, csrColIndA, descrC,
        bsrRowPtrC, rowBlockDim, colBlockDim, dir,
        nnzTotalDevHostPtr, pBuffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // 空矩阵提前返回（仅 m=0；n=0 由 LaunchNnzKernel 的 prefix sum 自然处理）
    if (m == 0) {
        int32_t baseC = Csr2gebsrGetBase(descrC);
        OP_LOGD("aclsparseXcsr2gebsrNnz",
                "empty matrix (m=0, n=%d), nnzb=0", n);
        auto *h = Csr2gebsrToInternalHandle(handle);
        return FillEmptyBsrRowPtrC(
            h, h->stream, bsrRowPtrC, nnzTotalDevHostPtr, baseC);
    }

    return LaunchNnzKernel(
        handle, m, n, descrA, csrRowPtrA, csrColIndA,
        descrC, bsrRowPtrC, rowBlockDim, colBlockDim,
        nnzTotalDevHostPtr, pBuffer);
}

// ===========================================================================
// Convert (4 个精度版本)
// ===========================================================================

/// 契约：本 API 依赖前置 Nnz 已在 stream 上完成。算子不在 Nnz 与 Convert 间插入
/// 同步，调用方须在调用前同步 stream（或读回 nnzb 确认 Nnz 完成）。
static aclsparseStatus_t Csr2gebsrConvertCommon(
    const char *apiName,
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const void *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    void *bsrValC, int *bsrRowPtrC, int *bsrColIndC,
    int rowBlockDim, int colBlockDim,
    uint32_t valSize, void *pBuffer)
{
    if (handle == nullptr) {
        OP_LOGE(apiName, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = ValidateConvertParams(
        apiName, m, n, descrA, csrValA, csrRowPtrA, csrColIndA,
        descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDim, colBlockDim,
        dir, pBuffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (m == 0 || n == 0) {
        OP_LOGD(apiName, "empty matrix (m=%d, n=%d), skip kernel", m, n);
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    auto *gmCsrValA = reinterpret_cast<GM_ADDR>(const_cast<void *>(csrValA));
    auto *gmBsrValC = reinterpret_cast<GM_ADDR>(bsrValC);

    return LaunchConvertKernel(
        handle, dir, m, n, descrA,
        gmCsrValA, csrRowPtrA, csrColIndA,
        descrC, gmBsrValC, bsrRowPtrC, bsrColIndC,
        rowBlockDim, colBlockDim, valSize, pBuffer);
}

/// 契约：本 API 依赖前置 Nnz 已在 stream 上完成。
aclsparseStatus_t aclsparseScsr2gebsr(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    float *bsrValC, int *bsrRowPtrC, int *bsrColIndC,
    int rowBlockDim, int colBlockDim,
    void *pBuffer)
{
    return Csr2gebsrConvertCommon(
        "aclsparseScsr2gebsr", handle, dir, m, n, descrA,
        csrValA, csrRowPtrA, csrColIndA, descrC,
        bsrValC, bsrRowPtrC, bsrColIndC,
        rowBlockDim, colBlockDim, sizeof(float), pBuffer);
}

/// 契约：本 API 依赖前置 Nnz 已在 stream 上完成。
/// H(fp16) 和 Bh(bf16) 版本签名和 valSize 完全一致（均为 void* + uint16_t），
/// 仅 API 名不同，通过宏展开避免重复代码。
#define DEFINE_CONVERT_API(NAME) \
aclsparseStatus_t NAME( \
    aclsparseHandle_t handle, aclsparseDirection_t dir, \
    int m, int n, \
    const aclsparseMatDescr_t descrA, \
    const void *csrValA, const int *csrRowPtrA, const int *csrColIndA, \
    const aclsparseMatDescr_t descrC, \
    void *bsrValC, int *bsrRowPtrC, int *bsrColIndC, \
    int rowBlockDim, int colBlockDim, \
    void *pBuffer) \
{ \
    return Csr2gebsrConvertCommon( \
        #NAME, handle, dir, m, n, descrA, \
        csrValA, csrRowPtrA, csrColIndA, descrC, \
        bsrValC, bsrRowPtrC, bsrColIndC, \
        rowBlockDim, colBlockDim, sizeof(uint16_t), pBuffer); \
}

DEFINE_CONVERT_API(aclsparseHcsr2gebsr)
DEFINE_CONVERT_API(aclsparseBhcsr2gebsr)

#undef DEFINE_CONVERT_API

/// 契约：本 API 依赖前置 Nnz 已在 stream 上完成。
aclsparseStatus_t aclsparseIcsr2gebsr(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const int *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    int *bsrValC, int *bsrRowPtrC, int *bsrColIndC,
    int rowBlockDim, int colBlockDim,
    void *pBuffer)
{
    return Csr2gebsrConvertCommon(
        "aclsparseIcsr2gebsr", handle, dir, m, n, descrA,
        csrValA, csrRowPtrA, csrColIndA, descrC,
        bsrValC, bsrRowPtrC, bsrColIndC,
        rowBlockDim, colBlockDim, sizeof(int32_t), pBuffer);
}

}  // extern "C"
