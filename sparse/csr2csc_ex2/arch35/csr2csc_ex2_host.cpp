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
 * \file csr2csc_ex2_host.cpp
 * \brief aclsparseCsr2cscEx2 Host 侧实现：参数校验 + Kernel launch。
 *
 * 两个公共 API：
 *   - aclsparseCsr2cscEx2_bufferSize: 查询 workspace 大小
 *   - aclsparseCsr2cscEx2:           执行 CSR -> CSC 格式转换
 *
 * 结构：每个 API 内部拆分为 ValidateParams + LaunchKernel 两个静态函数。
 * 五阶段 Kernel 串行执行：CountCols -> SumStripeHist -> PrefixSum -> StripeBase -> Scatter。
 */

#include <cstdint>
#include <algorithm>
#include <vector>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "csr2csc_ex2_tiling_data.h"
#include "csr2csc_ex2_kernel.h"

namespace {

inline struct aclsparseContext *ToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

/// 获取值类型的字节大小
static uint32_t GetValSize(aclDataType valType)
{
    switch (valType) {
        case ACL_INT8:
            return sizeof(int8_t);
        case ACL_FLOAT16:
        case ACL_BF16:
            return sizeof(uint16_t);
        case ACL_FLOAT:
            return sizeof(float);
        default:
            return 0;
    }
}

/// idxBase 枚举转索引基值（0 或 1）
inline int32_t ToIndexBaseValue(aclsparseIndexBase_t idxBase)
{
    return (idxBase == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
}

/// 计算 Scatter stripe 分段数（与 Kernel 1 的 numBlocks 公式一致，保证 bufferSize 与执行路径结果相同）。
///
/// stripeCount = min(⌈nnz / kCsr2CscThreadsPerBlock⌉, aivCoreNum)。
/// 每 block 单线程处理一个 stripe（连续 k 区间），按 k 升序顺序 scatter，
/// 保证列内行号升序（与 golden 逐位一致），无需 Sort kernel。
///
/// workspace 上限保护：限制 stripeHist 区总大小不超过 kCsr2CscMaxStripeWorkspaceBytes，
/// 大 n 场景自动退化为较少 stripe，避免 StripeBase kernel 复杂度 O(n × stripeCount) 过高。
/// 正确性保证：stripe 划分逻辑（CountCols 写 stripeHist、StripeBase 前缀和、Scatter 顺序游标）
/// 不依赖 stripeCount 具体取值，仅要求三者使用相同的 stripeCount/stripeSize（见 kernel.cpp 注释）。
static uint32_t ComputeStripeCount(int32_t nnz, int32_t n, uint32_t aivCoreNum)
{
    uint32_t elementsPerStripe = kCsr2CscThreadsPerBlock;
    uint32_t maxBlocks = aivCoreNum;
    uint32_t numBlocks = std::min(
        CeilDiv<uint32_t>(static_cast<uint32_t>(nnz), elementsPerStripe),
        maxBlocks);
    // workspace 上限保护：限制 stripeHist 区总大小不超过 kCsr2CscMaxStripeWorkspaceBytes
    size_t perStripeBytes = (static_cast<size_t>(n) + 1) * sizeof(int32_t);
    if (perStripeBytes > 0) {
        uint32_t stripeCap = static_cast<uint32_t>(
            kCsr2CscMaxStripeWorkspaceBytes / perStripeBytes);
        if (stripeCap == 0) {
            stripeCap = 1;
        }
        numBlocks = std::min(numBlocks, stripeCap);
    }
    if (numBlocks == 0) {
        numBlocks = 1;
    }
    return numBlocks;
}

/// 计算 workspace 字节数：固定段（colCount）+ stripeCount 段直方图/游标
static size_t ComputeWorkspaceBytes(int32_t n, uint32_t stripeCount)
{
    size_t segmentBytes = (static_cast<size_t>(n) + 1) * sizeof(int32_t);
    return (static_cast<size_t>(kCsr2CscWorkspaceSegments) + stripeCount) *
           segmentBytes;
}

// ---------------------------------------------------------------------------
// 枚举参数校验（valType / copyValues / idxBase / alg，从 CommonParams 拆分以控制 NBNC）
// ---------------------------------------------------------------------------
static aclsparseStatus_t ValidateCsr2cscEx2EnumParams(
    const char *apiName, aclDataType valType,
    aclsparseAction_t copyValues, aclsparseIndexBase_t idxBase,
    aclsparseCsr2CscAlg_t alg)
{
    uint32_t valSize = GetValSize(valType);
    if (valSize == 0) {
        OP_LOGE(apiName, "unsupported valType: %d",
                static_cast<int>(valType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (copyValues != ACL_SPARSE_ACTION_SYMBOLIC &&
        copyValues != ACL_SPARSE_ACTION_NUMERIC) {
        OP_LOGE(apiName, "invalid copyValues: %d",
                static_cast<int>(copyValues));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (idxBase != ACL_SPARSE_INDEX_BASE_ZERO &&
        idxBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(apiName, "invalid idxBase: %d",
                static_cast<int>(idxBase));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (alg != ACL_SPARSE_CSR2CSC_ALG_DEFAULT &&
        alg != ACL_SPARSE_CSR2CSC_ALG1) {
        OP_LOGE(apiName, "invalid alg: %d",
                static_cast<int>(alg));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// 公共参数校验（handle/m/n/nnz/valType/copyValues/idxBase/alg，
// bufferSize 与计算接口共用；apiName 用于日志区分调用入口）
// ---------------------------------------------------------------------------
static aclsparseStatus_t ValidateCsr2cscEx2CommonParams(
    const char *apiName,
    aclsparseHandle_t handle, int m, int n, int nnz,
    aclDataType valType, aclsparseAction_t copyValues,
    aclsparseIndexBase_t idxBase, aclsparseCsr2CscAlg_t alg)
{
    if (handle == nullptr) {
        OP_LOGE(apiName, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m < 0) {
        OP_LOGE(apiName, "invalid m: %d", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE(apiName, "invalid n: %d", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n > INT32_MAX - 1) {
        OP_LOGE(apiName, "n too large for int32 index: %d", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz < 0) {
        OP_LOGE(apiName, "invalid nnz: %d", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ValidateCsr2cscEx2EnumParams(apiName, valType, copyValues, idxBase, alg);
}

// ---------------------------------------------------------------------------
// bufferSize 参数校验
// ---------------------------------------------------------------------------
static aclsparseStatus_t ValidateBufferSizeParams(
    aclsparseHandle_t handle, int m, int n, int nnz,
    aclDataType valType, aclsparseAction_t copyValues,
    aclsparseIndexBase_t idxBase, aclsparseCsr2CscAlg_t alg,
    size_t *bufferSize)
{
    aclsparseStatus_t st = ValidateCsr2cscEx2CommonParams(
        "aclsparseCsr2cscEx2_bufferSize",
        handle, m, n, nnz, valType, copyValues, idxBase, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (bufferSize == nullptr) {
        OP_LOGE("aclsparseCsr2cscEx2_bufferSize", "bufferSize is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// 指针 nullptr 校验（从 ValidateCsr2cscEx2Params 拆分，降低圈复杂度）
// ---------------------------------------------------------------------------
static aclsparseStatus_t ValidateCsr2cscEx2Ptrs(
    int m, int nnz,
    const void *csrVal, const int *csrRowPtr, const int *csrColInd,
    void *cscVal, int *cscColPtr, int *cscRowInd,
    aclsparseAction_t copyValues)
{
    if (m > 0 && csrRowPtr == nullptr) {
        OP_LOGE("aclsparseCsr2cscEx2", "csrRowPtr is nullptr (m=%d)", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && csrColInd == nullptr) {
        OP_LOGE("aclsparseCsr2cscEx2", "csrColInd is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && copyValues == ACL_SPARSE_ACTION_NUMERIC &&
        csrVal == nullptr) {
        OP_LOGE("aclsparseCsr2cscEx2", "csrVal is nullptr (nnz=%d, NUMERIC)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (cscColPtr == nullptr) {
        OP_LOGE("aclsparseCsr2cscEx2", "cscColPtr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && cscRowInd == nullptr) {
        OP_LOGE("aclsparseCsr2cscEx2", "cscRowInd is nullptr (nnz=%d)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && copyValues == ACL_SPARSE_ACTION_NUMERIC &&
        cscVal == nullptr) {
        OP_LOGE("aclsparseCsr2cscEx2", "cscVal is nullptr (nnz=%d, NUMERIC)", nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// 计算接口参数校验
// ---------------------------------------------------------------------------
static aclsparseStatus_t ValidateCsr2cscEx2Params(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const void *csrVal, const int *csrRowPtr, const int *csrColInd,
    void *cscVal, int *cscColPtr, int *cscRowInd,
    aclDataType valType, aclsparseAction_t copyValues,
    aclsparseIndexBase_t idxBase, aclsparseCsr2CscAlg_t alg)
{
    aclsparseStatus_t st = ValidateCsr2cscEx2CommonParams(
        "aclsparseCsr2cscEx2",
        handle, m, n, nnz, valType, copyValues, idxBase, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    return ValidateCsr2cscEx2Ptrs(
        m, nnz, csrVal, csrRowPtr, csrColInd,
        cscVal, cscColPtr, cscRowInd, copyValues);
}

// ---------------------------------------------------------------------------
// Kernel launch 子函数（从 LaunchCsr2cscEx2Kernel 拆分，控制 NBNC 行数）
// ---------------------------------------------------------------------------

static aclsparseStatus_t HandleCsr2cscEx2NnzZero(
    aclrtStream stream, int n, int baseVal, int *cscColPtr)
{
    OP_LOGD("aclsparseCsr2cscEx2",
            "nnz=0, fill cscColPtr with idxBase=%d", baseVal);
    size_t count = static_cast<size_t>(n) + 1;
    size_t sizeBytes = count * sizeof(int32_t);
    if (baseVal == 0) {
        aclError aclRet = aclrtMemsetAsync(cscColPtr, sizeBytes, 0, sizeBytes, stream);
        CHECK_RET(aclRet == ACL_ERROR_NONE,
                  OP_LOGE("aclsparseCsr2cscEx2",
                          "aclrtMemsetAsync failed, ret=%d, sizeBytes=%zu",
                          aclRet, sizeBytes);
                  return ACL_SPARSE_STATUS_EXECUTION_FAILED);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    // 栈上 vector 作为 H2D 源缓冲区：填充 idxBase 后一次性拷贝到 device 的 cscColPtr。
    // 同步 aclrtMemcpy 保证栈上 hostBuf 析构前 H2D 拷贝已完成，避免异步拷贝 use-after-free。
    std::vector<int32_t> hostBuf(count, static_cast<int32_t>(baseVal));
    aclError aclRet = aclrtMemcpy(
        cscColPtr, sizeBytes, hostBuf.data(), sizeBytes,
        ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_ERROR_NONE,
              OP_LOGE("aclsparseCsr2cscEx2",
                      "aclrtMemcpy failed, ret=%d, sizeBytes=%zu",
                      aclRet, sizeBytes);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SetupCsr2cscEx2Workspace(
    aclsparseHandle_t handle, void *buffer,
    size_t requiredSize, void **outWorkspace)
{
    auto *h = ToInternalHandle(handle);
    size_t workspaceSize = 0;

    if (buffer != nullptr) {
        *outWorkspace = buffer;
        workspaceSize = requiredSize;
    } else {
        *outWorkspace = aclsparseGetEffectiveWorkspace(h);
        workspaceSize = aclsparseGetEffectiveWorkspaceSize(h);
    }

    if (*outWorkspace == nullptr) {
        OP_LOGE("aclsparseCsr2cscEx2",
                "workspace is null (buffer=nullptr and handle has no workspace)");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    if (buffer == nullptr && requiredSize > workspaceSize) {
        OP_LOGE("aclsparseCsr2cscEx2",
                "handle workspace too small: %zu < %zu (please pass a buffer "
                "of adequate size or enlarge handle workspace)",
                workspaceSize, requiredSize);
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    aclError aclRet = aclrtMemsetAsync(
        *outWorkspace, requiredSize, 0, requiredSize, h->stream);
    CHECK_RET(aclRet == ACL_ERROR_NONE,
              OP_LOGE("aclsparseCsr2cscEx2",
                      "aclrtMemsetAsync workspace failed, ret=%d, size=%zu",
                      aclRet, requiredSize);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    return ACL_SPARSE_STATUS_SUCCESS;
}

/// 计算 stripe 直方图/游标区在 workspace 中的起始 GM_ADDR
static GM_ADDR GetStripeHistPtr(void *workspace, int32_t n)
{
    return reinterpret_cast<GM_ADDR>(
        static_cast<uint8_t *>(workspace) +
        static_cast<size_t>(kCsr2CscWorkspaceSegments) *
        (static_cast<size_t>(n) + 1) * sizeof(int32_t));
}

// Kernel 1 (CountCols): 遍历 csrColInd，统计每 stripe 列直方图（不再写 colCount）
static void LaunchCountCols(
    GM_ADDR gmColInd, GM_ADDR gmStripeHist,
    int32_t nnz, int32_t n, int32_t baseVal,
    uint32_t numBlocks, int32_t stripeSize, aclrtStream stream)
{
    Csr2CscCountTilingData tiling{};
    tiling.nnz = nnz;
    tiling.n = n;
    tiling.idxBase = baseVal;
    tiling.stripeCount = static_cast<int32_t>(numBlocks);
    tiling.stripeSize = stripeSize;
    csr2csc_count_kernel_do(gmColInd, gmStripeHist, tiling, numBlocks, stream);
    OP_LOGD("aclsparseCsr2cscEx2", "Kernel 1 (CountCols) launched, blocks=%u", numBlocks);
}

// Kernel 1.5 (SumStripeHist): 对 stripeHist 按列求和重建 colCount（多线程并行）
static void LaunchSumStripeHist(
    GM_ADDR gmStripeHist, GM_ADDR gmWorkspace,
    int32_t n, uint32_t stripeCount, uint32_t numBlocks, aclrtStream stream)
{
    Csr2CscSumStripeHistTilingData tiling{};
    tiling.n = n;
    tiling.stripeCount = static_cast<int32_t>(stripeCount);
    csr2csc_sum_stripe_hist_kernel_do(gmStripeHist, gmWorkspace, tiling, numBlocks, stream);
    OP_LOGD("aclsparseCsr2cscEx2",
            "Kernel 1.5 (SumStripeHist) launched, blocks=%u", numBlocks);
}

// Kernel 2 (PrefixSum): exclusive prefix sum -> cscColPtr（单 warp 并行 scan）
static void LaunchPrefixSum(
    GM_ADDR gmWorkspace, GM_ADDR gmCscColPtr,
    int32_t n, int32_t baseVal, aclrtStream stream)
{
    Csr2CscPrefixSumTilingData tiling{};
    tiling.n = n;
    tiling.idxBase = baseVal;
    csr2csc_prefixsum_kernel_do(gmWorkspace, gmCscColPtr, tiling, stream);
    OP_LOGD("aclsparseCsr2cscEx2", "Kernel 2 (PrefixSum) launched");
}

// Kernel 3 (StripeBase): stripe 直方图按列前缀和 -> 每 stripe 写游标基址
static void LaunchStripeBase(
    GM_ADDR gmStripeHist, GM_ADDR gmCscColPtr,
    int32_t n, int32_t baseVal, uint32_t numBlocks, aclrtStream stream)
{
    Csr2CscStripeBaseTilingData tiling{};
    tiling.n = n;
    tiling.idxBase = baseVal;
    tiling.stripeCount = static_cast<int32_t>(numBlocks);
    csr2csc_stripebase_kernel_do(gmStripeHist, gmCscColPtr, tiling, numBlocks, stream);
    OP_LOGD("aclsparseCsr2cscEx2", "Kernel 3 (StripeBase) launched, blocks=%u", numBlocks);
}

// Kernel 4 (Scatter): 单线程顺序 scatter（游标私有，无原子竞争）
static void LaunchScatter(
    GM_ADDR gmRowPtr, GM_ADDR gmColInd, GM_ADDR gmCsrVal,
    GM_ADDR gmCscRowInd, GM_ADDR gmCscVal, GM_ADDR gmStripeHist,
    int32_t m, int32_t n, int32_t nnz, int32_t baseVal,
    int32_t copyVal, int32_t stripeSize, uint32_t valSize,
    uint32_t numBlocks, aclrtStream stream)
{
    Csr2CscScatterTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.nnz = nnz;
    tiling.idxBase = baseVal;
    tiling.copyValues = copyVal;
    tiling.stripeSize = stripeSize;
    tiling.valSize = valSize;
    csr2csc_scatter_kernel_do(
        gmRowPtr, gmColInd, gmCsrVal,
        gmCscRowInd, gmCscVal, gmStripeHist,
        tiling, numBlocks, stream);
    OP_LOGD("aclsparseCsr2cscEx2",
            "Kernel 4 (Scatter) launched, blocks=%u", numBlocks);
}

static aclsparseStatus_t LaunchCsr2cscEx2Kernels(
    int m, int n, int nnz,
    const void *csrVal, const int *csrRowPtr, const int *csrColInd,
    void *cscVal, int *cscColPtr, int *cscRowInd,
    void *workspace, uint32_t valSize, int32_t copyVal,
    int32_t baseVal, uint32_t numBlocks, int32_t stripeSize,
    uint32_t aivCoreNum, aclrtStream stream)
{
    auto *gmColInd = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrColInd));
    auto *gmRowPtr = reinterpret_cast<GM_ADDR>(const_cast<int *>(csrRowPtr));
    auto *gmCsrVal = reinterpret_cast<GM_ADDR>(const_cast<void *>(csrVal));
    auto *gmWorkspace = reinterpret_cast<GM_ADDR>(static_cast<uint8_t *>(workspace));
    auto *gmCscColPtr = reinterpret_cast<GM_ADDR>(cscColPtr);
    auto *gmCscRowInd = reinterpret_cast<GM_ADDR>(cscRowInd);
    auto *gmCscVal = reinterpret_cast<GM_ADDR>(cscVal);
    auto *gmStripeHist = GetStripeHistPtr(workspace, n);

    // 执行顺序：CountCols -> SumStripeHist -> PrefixSum -> StripeBase -> Scatter
    // SumStripeHist 必须在 CountCols 之后（读 stripeHist 直方图）、PrefixSum 之前
    //（写 colCount 供 PrefixSum 读）、StripeBase 之前（StripeBase 原地修改 stripeHist）。
    // Scatter 单线程按 k 升序顺序写入，列内行号自然升序，无需 Sort kernel。
    // clear-then-launch-then-peek（issue #357 分族，对齐 #369 gebsr2gebsc 样式）：
    // Peek 只读不清除，下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次编排。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    LaunchCountCols(gmColInd, gmStripeHist,
                    nnz, n, baseVal, numBlocks, stripeSize, stream);
    LaunchSumStripeHist(gmStripeHist, gmWorkspace,
                        n, numBlocks, aivCoreNum, stream);
    LaunchPrefixSum(gmWorkspace, gmCscColPtr, n, baseVal, stream);
    LaunchStripeBase(gmStripeHist, gmCscColPtr, n, baseVal, numBlocks, stream);
    LaunchScatter(gmRowPtr, gmColInd, gmCsrVal, gmCscRowInd, gmCscVal, gmStripeHist,
                  m, n, nnz, baseVal, copyVal, stripeSize, valSize, numBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseCsr2cscEx2", "kernel enqueue failed: %d",
                static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Kernel launch
// ---------------------------------------------------------------------------

/// 计算 launch 参数（从 LaunchCsr2cscEx2Kernel 拆分以控制 NBNC）。
/// numBlocks/stripeSize/valSize/copyVal/requiredSize 集中计算。
struct Csr2CscEx2LaunchParams {
    uint32_t numBlocks;
    int32_t stripeSize;
    uint32_t valSize;
    int32_t copyVal;
    size_t requiredSize;
};

static Csr2CscEx2LaunchParams ComputeCsr2cscEx2LaunchParams(
    int32_t nnz, int32_t n,
    aclDataType valType, aclsparseAction_t copyValues,
    uint32_t aivCoreNum)
{
    Csr2CscEx2LaunchParams p;
    p.numBlocks = ComputeStripeCount(nnz, n, aivCoreNum);
    p.stripeSize = static_cast<int32_t>(
        CeilDiv<uint32_t>(static_cast<uint32_t>(nnz), p.numBlocks));
    if (p.stripeSize == 0) {
        p.stripeSize = 1;
    }
    p.valSize = GetValSize(valType);
    p.copyVal = (copyValues == ACL_SPARSE_ACTION_NUMERIC)
        ? kCsr2CscNumeric : kCsr2CscSymbolic;
    p.requiredSize = ComputeWorkspaceBytes(n, p.numBlocks);
    return p;
}

static aclsparseStatus_t LaunchCsr2cscEx2Kernel(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const void *csrVal, const int *csrRowPtr, const int *csrColInd,
    void *cscVal, int *cscColPtr, int *cscRowInd,
    aclDataType valType, aclsparseAction_t copyValues,
    aclsparseIndexBase_t idxBase, void *buffer)
{
    auto *h = ToInternalHandle(handle);
    // baseVal 在本函数内唯一计算一次，供溢出校验、nnz==0 快路径与 tiling 共用
    int32_t baseVal = ToIndexBaseValue(idxBase);

    // 溢出校验：nnz + idxBase 不得溢出 int32（cscColPtr[n] = nnz + idxBase）
    if (nnz > INT32_MAX - baseVal) {
        OP_LOGE("aclsparseCsr2cscEx2",
                "nnz + idxBase exceeds int32 range: nnz=%d, baseVal=%d", nnz, baseVal);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (nnz == 0) {
        return HandleCsr2cscEx2NnzZero(h->stream, n, baseVal, cscColPtr);
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE("aclsparseCsr2cscEx2", "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    auto lp = ComputeCsr2cscEx2LaunchParams(nnz, n, valType, copyValues, aivCoreNum);

    void *workspace = nullptr;
    aclsparseStatus_t wsRet = SetupCsr2cscEx2Workspace(
        handle, buffer, lp.requiredSize, &workspace);
    if (wsRet != ACL_SPARSE_STATUS_SUCCESS) {
        return wsRet;
    }

    OP_LOGD("aclsparseCsr2cscEx2",
            "tiling: m=%d, n=%d, nnz=%d, idxBase=%d, copyValues=%d, "
            "valSize=%u, numBlocks=%u, stripeSize=%d, wsBytes=%zu",
            m, n, nnz, baseVal, lp.copyVal, lp.valSize, lp.numBlocks, lp.stripeSize,
            lp.requiredSize);

    return LaunchCsr2cscEx2Kernels(
        m, n, nnz, csrVal, csrRowPtr, csrColInd,
        cscVal, cscColPtr, cscRowInd,
        workspace, lp.valSize, lp.copyVal, baseVal, lp.numBlocks, lp.stripeSize,
        aivCoreNum, h->stream);
}

}  // namespace

// ----------------------------------------------------------------------------
// Public APIs
// ----------------------------------------------------------------------------
extern "C" {

aclsparseStatus_t aclsparseCsr2cscEx2_bufferSize(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const void *csrVal, const int *csrRowPtr, const int *csrColInd,
    void *cscVal, int *cscColPtr, int *cscRowInd,
    aclDataType valType, aclsparseAction_t copyValues,
    aclsparseIndexBase_t idxBase, aclsparseCsr2CscAlg_t alg,
    size_t *bufferSize)
{
    aclsparseStatus_t st = ValidateBufferSizeParams(
        handle, m, n, nnz, valType, copyValues, idxBase, alg, bufferSize);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (m == 0 || n == 0 || nnz == 0) {
        // 执行路径 nnz == 0 走 cscColPtr 填充快路径，不使用 workspace
        *bufferSize = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE("aclsparseCsr2cscEx2_bufferSize", "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    uint32_t stripeCount = ComputeStripeCount(nnz, n, aivCoreNum);

    size_t workspaceBytes = ComputeWorkspaceBytes(n, stripeCount);
    *bufferSize = workspaceBytes;

    OP_LOGD("aclsparseCsr2cscEx2_bufferSize",
            "m=%d, n=%d, nnz=%d, stripeCount=%u, workspaceBytes=%zu",
            m, n, nnz, stripeCount, workspaceBytes);

    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCsr2cscEx2(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const void *csrVal, const int *csrRowPtr, const int *csrColInd,
    void *cscVal, int *cscColPtr, int *cscRowInd,
    aclDataType valType, aclsparseAction_t copyValues,
    aclsparseIndexBase_t idxBase, aclsparseCsr2CscAlg_t alg,
    void *buffer)
{
    aclsparseStatus_t st = ValidateCsr2cscEx2Params(
        handle, m, n, nnz,
        csrVal, csrRowPtr, csrColInd,
        cscVal, cscColPtr, cscRowInd,
        valType, copyValues, idxBase, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // 数据契约校验：m==0 或 n==0 时 nnz 必为 0
    if (m == 0 && nnz > 0) {
        OP_LOGE("aclsparseCsr2cscEx2", "invalid input: m==0 but nnz>0 violates CSR data contract");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n == 0 && nnz > 0) {
        OP_LOGE("aclsparseCsr2cscEx2", "invalid input: n==0 but nnz>0 violates CSR data contract");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // m==0 || n==0 时 nnz 必为 0，委托给 LaunchCsr2cscEx2Kernel 的 nnz==0 路径
    // （该路径调用 HandleCsr2cscEx2NnzZero 填充 cscColPtr）。
    // baseVal 的计算与 nnz+idxBase 溢出校验统一收敛到 LaunchCsr2cscEx2Kernel 内，
    // 避免与 aclsparseCsr2cscEx2 重复计算（M-1 修复）。

    OP_LOGD("aclsparseCsr2cscEx2",
            "params OK: m=%d, n=%d, nnz=%d, valType=%d, copyValues=%d, idxBase=%d",
            m, n, nnz, static_cast<int>(valType),
            static_cast<int>(copyValues), static_cast<int>(idxBase));

    return LaunchCsr2cscEx2Kernel(
        handle, m, n, nnz,
        csrVal, csrRowPtr, csrColInd,
        cscVal, cscColPtr, cscRowInd,
        valType, copyValues, idxBase, buffer);
}

}  // extern "C"
