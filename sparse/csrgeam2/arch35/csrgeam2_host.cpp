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
 * \file csrgeam2_host.cpp
 * \brief csrgeam2 Host 侧实现：三组 Legacy API 入口。
 *
 * 1. aclsparseScsrgeam2_bufferSizeExt - 查询 workspace 大小
 * 2. aclsparseXcsrgeam2Nnz           - 计算 csrRowPtrC 和 nnzC
 * 3. aclsparseScsrgeam2              - 执行 C = αA + βB
 *
 * 结构：每个 API 内部拆分为 ValidateParams + LaunchKernel 两个静态函数。
 *
 * Nnz 阶段 Prefix Sum 采用 Device 方案（全异步，对齐 cuSPARSE 语义）：
 *   Kernel 1 (nnz计数) → Kernel 2 (device prefix sum → rowPtrC + nnzC)
 */

#include <algorithm>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "csrgeam2.h"
#include "csrgeam2_tiling_data.h"
#include "csrgeam2_kernel.h"

namespace {

// ===========================================================================
// 公共辅助函数
// ===========================================================================

// ValidateScalarPointers removed (#5): nullptr alpha/beta are valid per
// cuSPARSE semantics (treated as 0.0 by the kernel). All [[maybe_unused]]
// params indicated this function was a no-op shell.

/// 当 C 为空矩阵时填充 csrRowPtrC 和 nnzC（用于 m=0/n=0 或 nnzA=nnzB=0 提前返回）。
/// 使用 csrgeam2_fill_kernel 在 device 侧填充 baseC 值（全异步）。
/// HOST 模式下 *nnzTotalDevHostPtr = 0 是同步写入：这是 aclsparseXcsrgeam2Nnz
/// 的语义要求（cuSPARSE 规范），HOST 指针必须立即反映结果。
static aclsparseStatus_t FillEmptyCRowPtrC(
    const aclsparseContext *h, aclrtStream stream,
    int32_t *csrSortedRowPtrC, int32_t *nnzTotalDevHostPtr,
    int32_t m, int32_t baseC)
{
    // Callers guarantee m >= 0 (ValidateCommonCsrgeam2Args rejects m < 0)
    size_t countElems = static_cast<size_t>(m) + 1;

    Csrgeam2FillTilingData fillTiling{};
    fillTiling.count = static_cast<int32_t>(countElems);
    fillTiling.value = baseC;
    auto *gmDst = reinterpret_cast<GM_ADDR>(csrSortedRowPtrC);
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    csrgeam2_fill_kernel_do(gmDst, fillTiling, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseCsrgeam2", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }



    if (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        // Synchronous: matches cuSPARSE semantic (this API blocks until result is ready)
        *nnzTotalDevHostPtr = 0;
    } else {
        aclError aclRetNnz = aclrtMemsetAsync(nnzTotalDevHostPtr, sizeof(int32_t), 0, sizeof(int32_t), stream);
        CHECK_RET(aclRetNnz == ACL_ERROR_NONE,
                  OP_LOGE("aclsparseXcsrgeam2Nnz", "memset nnzTotal failed, ret=%d", aclRetNnz);
                  return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

/// 公共参数校验（三个 Validate* 共享）：
///   - 维度 m/n 非负
///   - nnzA/nnzB 非负
///   - descrA/B/C 非空 + type=GENERAL + indexBase 合法
///   - 当 nnz*>0 时，colInd* 必须非空
static aclsparseStatus_t ValidateCommonCsrgeam2Args(
    const char *apiName, int m, int n,
    const aclsparseMatDescr_t descrA, int nnzA, const int *csrSortedColIndA, const aclsparseMatDescr_t descrB,
    int nnzB, const int *csrSortedColIndB, const aclsparseMatDescr_t descrC)
{
    if (m < 0) {
        OP_LOGE(apiName, "invalid m: %d", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE(apiName, "invalid n: %d", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzA < 0) {
        OP_LOGE(apiName, "invalid nnzA: %d", nnzA);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzB < 0) {
        OP_LOGE(apiName, "invalid nnzB: %d", nnzB);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st;
    st = AclsparseValidateMatDescr(apiName, "descrA", descrA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = AclsparseValidateMatDescr(apiName, "descrB", descrB);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = AclsparseValidateMatDescr(apiName, "descrC", descrC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    if (nnzA > 0 && csrSortedColIndA == nullptr) {
        OP_LOGE(apiName, "csrSortedColIndA is nullptr (nnzA=%d)", nnzA);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzB > 0 && csrSortedColIndB == nullptr) {
        OP_LOGE(apiName, "csrSortedColIndB is nullptr (nnzB=%d)", nnzB);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

/// 计算多 Block 并行切分参数（两个 Launch* 共享）。
/// 基于行数和 AI Core 数量，确定 useBlocks 和 rowsPerBlock。
static aclsparseStatus_t ComputeBlockSplits(
    const char *apiName, int m,
    uint32_t &useBlocks, uint32_t &rowsPerBlock)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE(apiName, "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    useBlocks = std::min(aivCoreNum,
        CeilDiv(static_cast<uint32_t>(m), kCsrgeam2MaxThreadsPerBlock));
    if (useBlocks == 0) {
        useBlocks = 1;
    }
    rowsPerBlock = CeilDiv(static_cast<uint32_t>(m), useBlocks);
    if (rowsPerBlock == 0) {
        rowsPerBlock = 1;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

/// Launch prefix sum kernel 并异步拷贝 nnzC（从 LaunchNnzKernel 中抽出，以降低函数行数）。
/// Kernel 3 在 device 侧完成 exclusive prefix sum：nnzPerRow → rowPtrC + nnzC。
static aclsparseStatus_t LaunchPrefixSumAndCopyNnz(
    aclsparseContext *h, aclrtStream stream,
    int32_t m, int32_t baseC,
    void *workspace,
    int32_t *csrSortedRowPtrC, int *nnzTotalDevHostPtr)
{
    Csrgeam2PrefixSumTilingData psTiling{};
    psTiling.m     = m;
    psTiling.baseC = baseC;

    // nnzCDev 位于 workspace 末尾（不与 nnzPerRow[0..m-1] 重叠）
    auto *nnzPerRow = reinterpret_cast<GM_ADDR>(static_cast<int32_t *>(workspace));
    auto *nnzCDev   = reinterpret_cast<int32_t *>(workspace) + m;
    auto *gmRowPtrC = reinterpret_cast<GM_ADDR>(csrSortedRowPtrC);
    auto *gmNnzCDev = reinterpret_cast<GM_ADDR>(nnzCDev);

    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    csrgeam2_prefixsum_kernel_do(nnzPerRow, gmRowPtrC, gmNnzCDev, psTiling, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseCsrgeam2", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }



    OP_LOGD("aclsparseXcsrgeam2Nnz",
            "Kernel 3 (prefix sum) launched, m=%d, baseC=%d", m, baseC);

    // 异步将 nnzC 从 device workspace 拷贝到用户输出指针（同一 stream，无额外同步）
    // HOST 模式：nnzTotalDevHostPtr 是 host 指针 → D2H
    // DEVICE 模式：nnzTotalDevHostPtr 是 device 指针 → D2D
    if (nnzTotalDevHostPtr != nullptr) {
        aclrtMemcpyKind kind = (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST)
                                    ? ACL_MEMCPY_DEVICE_TO_HOST
                                    : ACL_MEMCPY_DEVICE_TO_DEVICE;
        aclError aclRet = aclrtMemcpyAsync(nnzTotalDevHostPtr, sizeof(int32_t),
                                           nnzCDev, sizeof(int32_t),
                                           kind, stream);
        CHECK_RET(aclRet == ACL_ERROR_NONE,
                  OP_LOGE("aclsparseXcsrgeam2Nnz",
                          "async copy nnzC failed, ret=%d", aclRet);
                  return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// bufferSizeExt - 参数校验
// ===========================================================================
static aclsparseStatus_t ValidateBufferSizeExtParams(
    aclsparseHandle_t handle, int m, int n, const float *alpha,
    const aclsparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const float *beta,
    const aclsparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const aclsparseMatDescr_t descrC, size_t *pBufferSizeInBytes)
{
    aclsparseStatus_t st = ValidateCommonCsrgeam2Args(
        "aclsparseScsrgeam2_bufferSizeExt", m, n,
        descrA, nnzA, csrSortedColIndA,
        descrB, nnzB, csrSortedColIndB, descrC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    // alpha/beta 必须非空（cuSPARSE 规范要求）
    if (alpha == nullptr) {
        OP_LOGE("aclsparseScsrgeam2_bufferSizeExt", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE("aclsparseScsrgeam2_bufferSizeExt", "beta is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // A 矩阵指针校验（rowPtrA 需要 m>0 才存在，与 Nnz/Scsrgeam2 一致）(#10)
    if (m > 0 && csrSortedRowPtrA == nullptr) {
        OP_LOGE("aclsparseScsrgeam2_bufferSizeExt",
                "csrSortedRowPtrA is nullptr (m=%d, n=%d)", m, n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // valA 校验放宽：bufferSizeExt 仅查询 workspace 大小，不依赖 valA 的值（#13,
    // cuSPARSE bufferSizeExt 通常不校验值指针）

    // B 矩阵指针校验（同 A）(#10)
    if (m > 0 && csrSortedRowPtrB == nullptr) {
        OP_LOGE("aclsparseScsrgeam2_bufferSizeExt",
                "csrSortedRowPtrB is nullptr (m=%d, n=%d)", m, n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // valB 校验放宽：理由同 valA（#13）

    if (pBufferSizeInBytes == nullptr) {
        OP_LOGE("aclsparseScsrgeam2_bufferSizeExt", "pBufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// Xcsrgeam2Nnz - 参数校验
// ===========================================================================
static aclsparseStatus_t ValidateNnzParams(
    int m, int n, const aclsparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const aclsparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const aclsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr,
    void *workspace)
{
    aclsparseStatus_t st = ValidateCommonCsrgeam2Args(
        "aclsparseXcsrgeam2Nnz", m, n,
        descrA, nnzA, csrSortedColIndA,
        descrB, nnzB, csrSortedColIndB, descrC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    if (m > 0) {
        if (csrSortedRowPtrA == nullptr) {
            OP_LOGE("aclsparseXcsrgeam2Nnz", "csrSortedRowPtrA is nullptr (m=%d)", m);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        if (csrSortedRowPtrB == nullptr) {
            OP_LOGE("aclsparseXcsrgeam2Nnz", "csrSortedRowPtrB is nullptr (m=%d)", m);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    if (csrSortedRowPtrC == nullptr) {
        OP_LOGE("aclsparseXcsrgeam2Nnz", "csrSortedRowPtrC is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzTotalDevHostPtr == nullptr) {
        OP_LOGE("aclsparseXcsrgeam2Nnz", "nnzTotalDevHostPtr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (workspace == nullptr) {
        OP_LOGE("aclsparseXcsrgeam2Nnz", "workspace is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// Xcsrgeam2Nnz - Kernel launch（Nnz + Device Prefix Sum，全异步）
// ===========================================================================
static aclsparseStatus_t LaunchNnzKernel(
    aclsparseHandle_t handle, int m, const aclsparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const aclsparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const aclsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr,
    void *workspace)
{
    auto *h = Csrgeam2ToInternalHandle(handle);
    aclrtStream stream = h->stream;

    // 提取 indexBase
    int32_t baseA = Csrgeam2GetBase(descrA);
    int32_t baseB = Csrgeam2GetBase(descrB);
    int32_t baseC = Csrgeam2GetBase(descrC);

    // 空行（nnzA=0 且 nnzB=0）：C 为空矩阵
    if (nnzA == 0 && nnzB == 0) {
        OP_LOGD("aclsparseXcsrgeam2Nnz",
                "both nnzA=0 and nnzB=0, fill csrRowPtrC and return");
        return FillEmptyCRowPtrC(h, stream, csrSortedRowPtrC, nnzTotalDevHostPtr,
                                 static_cast<int32_t>(m), baseC);
    }

    // 多 Block 并行：行级静态均匀切分到多个 AI Core
    uint32_t useBlocks = 0;
    uint32_t rowsPerBlock = 0;
    aclsparseStatus_t splitSt = ComputeBlockSplits("aclsparseXcsrgeam2Nnz",
                                                     m, useBlocks, rowsPerBlock);
    if (splitSt != ACL_SPARSE_STATUS_SUCCESS) { return splitSt; }

    // 构建 Tiling
    Csrgeam2NnzTilingData tiling{};
    tiling.m = static_cast<int32_t>(m);
    tiling.baseA = baseA;
    tiling.baseB = baseB;
    tiling.rowsPerBlock = rowsPerBlock;

    OP_LOGD("aclsparseXcsrgeam2Nnz",
            "tiling: m=%d, baseA=%d, baseB=%d, rowsPerBlock=%u, numBlocks=%u",
            tiling.m, tiling.baseA, tiling.baseB,
            tiling.rowsPerBlock, useBlocks);

    // GM_ADDR 转换
    auto *gmRowPtrA = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrSortedRowPtrA));
    auto *gmColIndA = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrSortedColIndA));
    auto *gmRowPtrB = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrSortedRowPtrB));
    auto *gmColIndB = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrSortedColIndB));
    auto *gmNnzPerRow = reinterpret_cast<GM_ADDR>(static_cast<int32_t *>(workspace));

    // Launch Kernel 1（逐行计数）
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    csrgeam2_nnz_kernel_do(
        gmRowPtrA, gmColIndA, gmRowPtrB, gmColIndB, gmNnzPerRow,
        tiling, useBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseCsrgeam2", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }



    OP_LOGD("aclsparseXcsrgeam2Nnz", "Kernel 1 (nnz) launched, blocks=%u", useBlocks);

    // Launch Kernel 3 (prefix sum) + 异步拷贝 nnzC（抽出为独立函数以降低行数）
    return LaunchPrefixSumAndCopyNnz(h, stream, static_cast<int32_t>(m), baseC,
                                      workspace, csrSortedRowPtrC, nnzTotalDevHostPtr);
}

// ===========================================================================
// Scsrgeam2 - 参数校验
// ===========================================================================
static aclsparseStatus_t ValidateScsrgeam2Params(
    int m, int n, const float *alpha, const float *beta,
    const aclsparseMatDescr_t descrA, int nnzA, const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const aclsparseMatDescr_t descrB, int nnzB, const float *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const aclsparseMatDescr_t descrC,
    float *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    // validateCommonCsrgeam2Args: m/n >= 0, nnzA/nnzB >= 0, descrA/B/C valid, colInd pointers
    aclsparseStatus_t st = ValidateCommonCsrgeam2Args(
        "aclsparseScsrgeam2", m, n,
        descrA, nnzA, csrSortedColIndA,
        descrB, nnzB, csrSortedColIndB, descrC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    // alpha / beta：必须非空，与 bufferSizeExt 行为保持一致（cuSPARSE 规范要求）
    if (alpha == nullptr) {
        OP_LOGE("aclsparseScsrgeam2", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE("aclsparseScsrgeam2", "beta is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // A 的指针（nnzA > 0 时必需）
    if (m > 0 && csrSortedRowPtrA == nullptr) {
        OP_LOGE("aclsparseScsrgeam2", "csrSortedRowPtrA is nullptr (m=%d)", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzA > 0 && csrSortedValA == nullptr) {
        OP_LOGE("aclsparseScsrgeam2", "csrSortedValA is nullptr (nnzA=%d)", nnzA);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // B 的指针（nnzB > 0 时必需）
    if (m > 0 && csrSortedRowPtrB == nullptr) {
        OP_LOGE("aclsparseScsrgeam2", "csrSortedRowPtrB is nullptr (m=%d)", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzB > 0 && csrSortedValB == nullptr) {
        OP_LOGE("aclsparseScsrgeam2", "csrSortedValB is nullptr (nnzB=%d)", nnzB);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // C 的指针（始终必需）
    if (csrSortedRowPtrC == nullptr) {
        OP_LOGE("aclsparseScsrgeam2", "csrSortedRowPtrC is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (csrSortedColIndC == nullptr) {
        OP_LOGE("aclsparseScsrgeam2", "csrSortedColIndC is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (csrSortedValC == nullptr) {
        OP_LOGE("aclsparseScsrgeam2", "csrSortedValC is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // pBuffer: compute kernel does not use workspace (workspace is consumed by Nnz only),
    // but the API contract requires a valid pointer for consistency with bufferSizeExt
    if (pBuffer == nullptr) {
        OP_LOGE("aclsparseScsrgeam2", "pBuffer is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

// 构建 compute kernel 的 tiling 并处理 alpha/beta pointer mode（评审响应：
// 自 LaunchComputeKernel 提取，压函数 NBNC 至 50 内）。
static void BuildComputeTiling(Csrgeam2TilingData &tiling,
    int m, int n, int32_t baseA, int32_t baseB, int32_t baseC,
    uint32_t rowsPerBlock, const float *alpha, const float *beta,
    aclsparsePointerMode_t pointerMode)
{
    tiling.m = static_cast<int32_t>(m);
    tiling.n = static_cast<int32_t>(n);
    tiling.baseA = baseA;
    tiling.baseB = baseB;
    tiling.baseC = baseC;
    tiling.rowsPerBlock = rowsPerBlock;
    if (pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        tiling.alpha = (alpha != nullptr) ? *alpha : 0.0f;
        tiling.beta = (beta != nullptr) ? *beta : 0.0f;
        tiling.alphaPtr = 0;
        tiling.betaPtr = 0;
    } else {
        tiling.alphaPtr = reinterpret_cast<uint64_t>(alpha);
        tiling.betaPtr = reinterpret_cast<uint64_t>(beta);
    }
}

// ===========================================================================
// Scsrgeam2 - Kernel launch
// ===========================================================================
static aclsparseStatus_t LaunchComputeKernel(
    aclsparseHandle_t handle, int m, int n, const float *alpha,
    const aclsparseMatDescr_t descrA, int nnzA, const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const float *beta,
    const aclsparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const aclsparseMatDescr_t descrC,
    float *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    auto *h = Csrgeam2ToInternalHandle(handle);
    aclrtStream stream = h->stream;

    // 提取 indexBase
    int32_t baseA = Csrgeam2GetBase(descrA);
    int32_t baseB = Csrgeam2GetBase(descrB);
    int32_t baseC = Csrgeam2GetBase(descrC);

    // 多 Block 并行：行级静态均匀切分到多个 AI Core
    uint32_t useBlocks = 0;
    uint32_t rowsPerBlock = 0;
    aclsparseStatus_t splitSt = ComputeBlockSplits("aclsparseScsrgeam2",
                                                     m, useBlocks, rowsPerBlock);
    if (splitSt != ACL_SPARSE_STATUS_SUCCESS) { return splitSt; }

    Csrgeam2TilingData tiling{};
    BuildComputeTiling(tiling, m, n, baseA, baseB, baseC, rowsPerBlock,
                       alpha, beta, h->pointerMode);

    // GM_ADDR 转换
    auto *gmRowPtrA = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrSortedRowPtrA));
    auto *gmColIndA = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrSortedColIndA));
    auto *gmValA = reinterpret_cast<GM_ADDR>(const_cast<float *>(csrSortedValA));
    auto *gmRowPtrB = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrSortedRowPtrB));
    auto *gmColIndB = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrSortedColIndB));
    auto *gmValB = reinterpret_cast<GM_ADDR>(const_cast<float *>(csrSortedValB));
    auto *gmRowPtrC = reinterpret_cast<GM_ADDR>(csrSortedRowPtrC);
    auto *gmColIndC = reinterpret_cast<GM_ADDR>(csrSortedColIndC);
    auto *gmValC = reinterpret_cast<GM_ADDR>(csrSortedValC);

    // Launch Kernel 2（逐行合并）
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    csrgeam2_compute_kernel_do(
        gmRowPtrA, gmColIndA, gmValA,
        gmRowPtrB, gmColIndB, gmValB,
        gmRowPtrC, gmColIndC, gmValC,
        tiling, useBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseCsrgeam2", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }



    OP_LOGD("aclsparseScsrgeam2", "Kernel 2 (compute) launched, blocks=%u", useBlocks);

    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

// ============================================================================
// Public APIs
// ============================================================================
extern "C" {

aclsparseStatus_t aclsparseScsrgeam2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n, const float *alpha, const aclsparseMatDescr_t descrA, int nnzA,
    const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const aclsparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const aclsparseMatDescr_t descrC, const float *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseScsrgeam2_bufferSizeExt", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = ValidateBufferSizeExtParams(
        handle, m, n, alpha,
        descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, beta,
        descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrC, pBufferSizeInBytes);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // workspace = (m + 1) * sizeof(int32_t)：
    //   前 m 个 int32_t 用于 nnzPerRow（Kernel 1 输出）
    //   第 m+1 个 int32_t 用于 nnzC output（Kernel 3 输出，供 async D2H 返回 host）
    // Use (static_cast<size_t>(m) + 1) to avoid signed overflow when m ≈ INT_MAX (#2)
    size_t workspaceBytes = (static_cast<size_t>(m) + 1) * sizeof(int32_t);
    *pBufferSizeInBytes = workspaceBytes;

    OP_LOGD("aclsparseScsrgeam2_bufferSizeExt",
            "m=%d, n=%d, workspaceBytes=%zu", m, n, workspaceBytes);

    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseXcsrgeam2Nnz(
    aclsparseHandle_t handle, int m, int n, const aclsparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const aclsparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const aclsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr,
    void *workspace)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseXcsrgeam2Nnz", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    // Validate ALL params before early returns, otherwise FillEmptyCRowPtrC
    // may dereference null pointers (#1)
    aclsparseStatus_t st = ValidateNnzParams(
        m, n,
        descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA,
        descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB,
        descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // 空矩阵提前返回（m == 0 或 n == 0）
    if (m == 0) {
        OP_LOGD("aclsparseXcsrgeam2Nnz", "empty matrix (m=0, n=%d), nnzC=0", n);
        auto *h = Csrgeam2ToInternalHandle(handle);
        int32_t baseC = Csrgeam2GetBase(descrC);
        return FillEmptyCRowPtrC(h, h->stream, csrSortedRowPtrC,
                                  nnzTotalDevHostPtr, 0, baseC);
    }
    if (n == 0) {
        OP_LOGD("aclsparseXcsrgeam2Nnz", "empty matrix (m=%d, n=0), nnzC=0", m);
        auto *h = Csrgeam2ToInternalHandle(handle);
        int32_t baseC = Csrgeam2GetBase(descrC);
        return FillEmptyCRowPtrC(h, h->stream, csrSortedRowPtrC,
                                  nnzTotalDevHostPtr, static_cast<int32_t>(m), baseC);
    }

    OP_LOGD("aclsparseXcsrgeam2Nnz",
            "params OK: m=%d, n=%d, nnzA=%d, nnzB=%d", m, n, nnzA, nnzB);

    return LaunchNnzKernel(
        handle, m,
        descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA,
        descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB,
        descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace);
}

aclsparseStatus_t aclsparseScsrgeam2(
    aclsparseHandle_t handle, int m, int n, const float *alpha, const aclsparseMatDescr_t descrA, int nnzA,
    const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const aclsparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const aclsparseMatDescr_t descrC, float *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseScsrgeam2", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    // Validate ALL params before early returns — otherwise m=-1, n=0 would silently
    // succeed without validating invalid dimensions (mirrors aclsparseXcsrgeam2Nnz pattern)
    aclsparseStatus_t st = ValidateScsrgeam2Params(m, n, alpha, beta,
        descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
        descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB,
        descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // Empty matrix: skip kernel launch.
    // Note: Unlike aclsparseXcsrgeam2Nnz (which fills rowPtrC and nnzC), this function
    // leaves csrSortedValC/RowPtrC/ColIndC unmodified. The caller must ensure rowPtrC
    // was already filled by a preceding aclsparseXcsrgeam2Nnz call (#28).
    if (m == 0 || n == 0) {
        OP_LOGD("aclsparseScsrgeam2",
                "empty matrix (m=%d, n=%d), skip kernel", m, n);
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    OP_LOGD("aclsparseScsrgeam2",
            "params OK: m=%d, n=%d, nnzA=%d, nnzB=%d", m, n, nnzA, nnzB);

    return LaunchComputeKernel(handle, m, n, alpha,
        descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
        descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB,
        descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

}  // extern "C"
