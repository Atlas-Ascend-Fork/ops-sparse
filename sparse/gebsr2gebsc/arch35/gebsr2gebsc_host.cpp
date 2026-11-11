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
 * \file gebsr2gebsc_host.cpp
 * \brief aclsparseGebsr2gebsc Host 侧实现：参数校验 + Kernel launch。
 *
 * 两个公共 API（type-generic，通过 valType 参数支持多种数据类型）。
 * 内部统一通过 valSize 分发，结构转换逻辑复用 csr2csc_ex2 的五阶段流水线。
 */

#include <cstdint>
#include <algorithm>
#include <vector>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "gebsr2gebsc_tiling_data.h"
#include "gebsr2gebsc_kernel.h"

namespace {

inline struct aclsparseContext *ToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

inline int32_t ToIndexBaseValue(aclsparseIndexBase_t idxBase)
{
    return (idxBase == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
}

static uint32_t GetValSize(aclDataType valType)
{
    switch (valType) {
        case ACL_INT8:
            return sizeof(int8_t);
        case ACL_FLOAT16:
        case ACL_BF16:
            return sizeof(uint16_t);
        case ACL_FLOAT:
        case ACL_INT32:
            return sizeof(uint32_t);
        default:
            return 0;
    }
}

static int32_t DetermineCopyMode(int32_t rA, int32_t cA, int32_t rC, int32_t cC)
{
    if (rC == rA && cC == cA) {
        return kGebsr2GebscBlockDirectCopy;
    }
    if (rC == cA && cC == rA) {
        return kGebsr2GebscBlockTranspose;
    }
    return -1;
}

static uint32_t ComputeStripeCount(int32_t nnzb, int32_t nb, uint32_t aivCoreNum)
{
    uint32_t elementsPerStripe = kGebsr2GebscThreadsPerBlock;
    uint32_t maxBlocks = aivCoreNum;
    uint32_t numBlocks = std::min(
        CeilDiv<uint32_t>(static_cast<uint32_t>(nnzb), elementsPerStripe),
        maxBlocks);
    size_t perStripeBytes = (static_cast<size_t>(nb) + 1) * sizeof(int32_t);
    if (perStripeBytes > 0) {
        uint32_t stripeCap = static_cast<uint32_t>(
            kGebsr2GebscMaxStripeWorkspaceBytes / perStripeBytes);
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

static size_t ComputeWorkspaceBytes(int32_t nb, uint32_t stripeCount)
{
    size_t segmentBytes = (static_cast<size_t>(nb) + 1) * sizeof(int32_t);
    return (static_cast<size_t>(kGebsr2GebscWorkspaceSegments) + stripeCount) *
           segmentBytes;
}

// ---------------------------------------------------------------------------
// 参数校验
// ---------------------------------------------------------------------------

static aclsparseStatus_t ValidateGebsr2GebscEnumParams(
    const char *apiName,
    aclDataType valType,
    aclsparseAction_t copyValues, aclsparseIndexBase_t idxBase)
{
    uint32_t valSize = GetValSize(valType);
    if (valSize == 0) {
        OP_LOGE(apiName, "unsupported valType: %d", static_cast<int>(valType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (copyValues != ACL_SPARSE_ACTION_SYMBOLIC &&
        copyValues != ACL_SPARSE_ACTION_NUMERIC) {
        OP_LOGE(apiName, "invalid copyValues: %d", static_cast<int>(copyValues));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (idxBase != ACL_SPARSE_INDEX_BASE_ZERO &&
        idxBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(apiName, "invalid idxBase: %d", static_cast<int>(idxBase));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateGebsr2GebscCommonParams(
    const char *apiName,
    aclsparseHandle_t handle, int mb, int nb, int nnzb,
    int rowBlockDimA, int colBlockDimA,
    aclsparseDirection_t dirA)
{
    if (handle == nullptr) {
        OP_LOGE(apiName, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (mb < 0) {
        OP_LOGE(apiName, "invalid mb: %d", mb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nb < 0) {
        OP_LOGE(apiName, "invalid nb: %d", nb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nb > INT32_MAX - 1) {
        OP_LOGE(apiName, "nb too large for int32 index: %d", nb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzb < 0) {
        OP_LOGE(apiName, "invalid nnzb: %d", nnzb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rowBlockDimA <= 0) {
        OP_LOGE(apiName, "invalid rowBlockDimA: %d", rowBlockDimA);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (colBlockDimA <= 0) {
        OP_LOGE(apiName, "invalid colBlockDimA: %d", colBlockDimA);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (dirA != ACL_SPARSE_DIRECTION_ROW &&
        dirA != ACL_SPARSE_DIRECTION_COLUMN) {
        OP_LOGE(apiName, "invalid dirA: %d", static_cast<int>(dirA));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateBufferSizeParams(
    const char *apiName,
    aclsparseHandle_t handle, int mb, int nb, int nnzb,
    int rowBlockDimA, int colBlockDimA,
    aclsparseDirection_t dirA,
    aclDataType valType,
    size_t *pBufferSizeInBytes)
{
    aclsparseStatus_t st = ValidateGebsr2GebscCommonParams(
        apiName, handle, mb, nb, nnzb, rowBlockDimA, colBlockDimA, dirA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateGebsr2GebscEnumParams(apiName, valType,
        ACL_SPARSE_ACTION_SYMBOLIC, ACL_SPARSE_INDEX_BASE_ZERO);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (pBufferSizeInBytes == nullptr) {
        OP_LOGE(apiName, "pBufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateGebsr2GebscPtrs(
    int mb, int nnzb,
    const void *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
    void *bscVal, int *bscColPtr, int *bscRowInd,
    aclsparseAction_t copyValues)
{
    if (mb > 0 && bsrRowPtrA == nullptr) {
        OP_LOGE("gebsr2gebsc", "bsrRowPtrA is nullptr (mb=%d)", mb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzb > 0 && bsrColIndA == nullptr) {
        OP_LOGE("gebsr2gebsc", "bsrColIndA is nullptr (nnzb=%d)", nnzb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzb > 0 && copyValues == ACL_SPARSE_ACTION_NUMERIC &&
        bsrValA == nullptr) {
        OP_LOGE("gebsr2gebsc", "bsrValA is nullptr (nnzb=%d, NUMERIC)", nnzb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (bscColPtr == nullptr) {
        OP_LOGE("gebsr2gebsc", "bscColPtr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzb > 0 && bscRowInd == nullptr) {
        OP_LOGE("gebsr2gebsc", "bscRowInd is nullptr (nnzb=%d)", nnzb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzb > 0 && copyValues == ACL_SPARSE_ACTION_NUMERIC &&
        bscVal == nullptr) {
        OP_LOGE("gebsr2gebsc", "bscVal is nullptr (nnzb=%d, NUMERIC)", nnzb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateGebsr2GebscParams(
    const char *apiName,
    aclsparseHandle_t handle, int mb, int nb, int nnzb,
    const void *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
    int rowBlockDimA, int colBlockDimA,
    void *bscVal, int *bscColPtr, int *bscRowInd,
    int rowBlockDimC, int colBlockDimC,
    aclsparseAction_t copyValues, aclsparseIndexBase_t idxBase,
    aclsparseDirection_t dirA,
    aclDataType valType)
{
    aclsparseStatus_t st = ValidateGebsr2GebscCommonParams(
        apiName, handle, mb, nb, nnzb, rowBlockDimA, colBlockDimA, dirA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (rowBlockDimC <= 0) {
        OP_LOGE(apiName, "invalid rowBlockDimC: %d", rowBlockDimC);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (colBlockDimC <= 0) {
        OP_LOGE(apiName, "invalid colBlockDimC: %d", colBlockDimC);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    int32_t copyMode = DetermineCopyMode(
        rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC);
    if (copyMode < 0) {
        OP_LOGE(apiName,
                "unsupported block dim combination: A=(%d,%d) C=(%d,%d), "
                "must be direct copy (rC==rA&&cC==cA) or transpose (rC==cA&&cC==rA)",
                rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    st = ValidateGebsr2GebscEnumParams(apiName, valType, copyValues, idxBase);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    return ValidateGebsr2GebscPtrs(
        mb, nnzb, bsrValA, bsrRowPtrA, bsrColIndA,
        bscVal, bscColPtr, bscRowInd, copyValues);
}

// ---------------------------------------------------------------------------
// nnzb==0 快路径
// ---------------------------------------------------------------------------

static aclsparseStatus_t HandleGebsr2GebscNnzbZero(
    aclrtStream stream, int nb, int baseVal, int *bscColPtr)
{
    OP_LOGD("gebsr2gebsc", "nnzb=0, fill bscColPtr with idxBase=%d", baseVal);
    size_t count = static_cast<size_t>(nb) + 1;
    size_t sizeBytes = count * sizeof(int32_t);
    if (baseVal == 0) {
        aclError aclRet = aclrtMemsetAsync(bscColPtr, sizeBytes, 0, sizeBytes, stream);
        CHECK_RET(aclRet == ACL_ERROR_NONE,
                  OP_LOGE("gebsr2gebsc", "aclrtMemsetAsync failed, ret=%d", aclRet);
                  return ACL_SPARSE_STATUS_EXECUTION_FAILED);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    std::vector<int32_t> hostBuf(count, static_cast<int32_t>(baseVal));
    aclError aclRet = aclrtMemcpy(
        bscColPtr, sizeBytes, hostBuf.data(), sizeBytes,
        ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_ERROR_NONE,
              OP_LOGE("gebsr2gebsc", "aclrtMemcpy failed, ret=%d", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Workspace 管理
// ---------------------------------------------------------------------------

static aclsparseStatus_t SetupGebsr2GebscWorkspace(
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
        OP_LOGE("gebsr2gebsc",
                "workspace is null (buffer=nullptr and handle has no workspace)");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    if (buffer == nullptr && requiredSize > workspaceSize) {
        OP_LOGE("gebsr2gebsc",
                "handle workspace too small: %zu < %zu",
                workspaceSize, requiredSize);
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    aclError aclRet = aclrtMemsetAsync(
        *outWorkspace, requiredSize, 0, requiredSize, h->stream);
    CHECK_RET(aclRet == ACL_ERROR_NONE,
              OP_LOGE("gebsr2gebsc", "aclrtMemsetAsync workspace failed, ret=%d", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static GM_ADDR GetStripeHistPtr(void *workspace, int32_t nb)
{
    return reinterpret_cast<GM_ADDR>(
        static_cast<uint8_t *>(workspace) +
        static_cast<size_t>(kGebsr2GebscWorkspaceSegments) *
        (static_cast<size_t>(nb) + 1) * sizeof(int32_t));
}

// ---------------------------------------------------------------------------
// Kernel launch 子函数
// ---------------------------------------------------------------------------

static void LaunchCountCols(
    GM_ADDR gmColInd, GM_ADDR gmStripeHist,
    int32_t nnzb, int32_t nb, int32_t baseVal,
    uint32_t numBlocks, int32_t stripeSize, aclrtStream stream)
{
    Gebsr2GebscCountTilingData tiling{};
    tiling.nnzb = nnzb;
    tiling.nb = nb;
    tiling.idxBase = baseVal;
    tiling.stripeCount = static_cast<int32_t>(numBlocks);
    tiling.stripeSize = stripeSize;
    gebsr2gebsc_count_kernel_do(gmColInd, gmStripeHist, tiling, numBlocks, stream);
    OP_LOGD("gebsr2gebsc", "Kernel 1 (CountCols) launched, blocks=%u", numBlocks);
}

static void LaunchSumStripeHist(
    GM_ADDR gmStripeHist, GM_ADDR gmWorkspace,
    int32_t nb, uint32_t stripeCount, uint32_t numBlocks, aclrtStream stream)
{
    Gebsr2GebscSumStripeHistTilingData tiling{};
    tiling.nb = nb;
    tiling.stripeCount = static_cast<int32_t>(stripeCount);
    gebsr2gebsc_sum_stripe_hist_kernel_do(gmStripeHist, gmWorkspace, tiling, numBlocks, stream);
    OP_LOGD("gebsr2gebsc", "Kernel 1.5 (SumStripeHist) launched, blocks=%u", numBlocks);
}

static void LaunchPrefixSum(
    GM_ADDR gmWorkspace, GM_ADDR gmBscColPtr,
    int32_t nb, int32_t baseVal, aclrtStream stream)
{
    Gebsr2GebscPrefixSumTilingData tiling{};
    tiling.nb = nb;
    tiling.idxBase = baseVal;
    gebsr2gebsc_prefixsum_kernel_do(gmWorkspace, gmBscColPtr, tiling, stream);
    OP_LOGD("gebsr2gebsc", "Kernel 2 (PrefixSum) launched");
}

static void LaunchStripeBase(
    GM_ADDR gmStripeHist, GM_ADDR gmBscColPtr,
    int32_t nb, int32_t baseVal, uint32_t numBlocks, aclrtStream stream)
{
    Gebsr2GebscStripeBaseTilingData tiling{};
    tiling.nb = nb;
    tiling.idxBase = baseVal;
    tiling.stripeCount = static_cast<int32_t>(numBlocks);
    gebsr2gebsc_stripebase_kernel_do(gmStripeHist, gmBscColPtr, tiling, numBlocks, stream);
    OP_LOGD("gebsr2gebsc", "Kernel 3 (StripeBase) launched, blocks=%u", numBlocks);
}

static void LaunchScatter(
    GM_ADDR gmRowPtr, GM_ADDR gmColInd, GM_ADDR gmBsrVal,
    GM_ADDR gmBscRowInd, GM_ADDR gmBscVal, GM_ADDR gmStripeHist,
    int32_t mb, int32_t nb, int32_t nnzb, int32_t baseVal,
    int32_t copyVal, int32_t stripeSize, uint32_t valSize,
    int32_t rowBlockDimA, int32_t colBlockDimA,
    int32_t rowBlockDimC, int32_t colBlockDimC,
    int32_t copyMode, int32_t dirA,
    uint32_t numBlocks, aclrtStream stream)
{
    Gebsr2GebscScatterTilingData tiling{};
    tiling.mb = mb;
    tiling.nb = nb;
    tiling.nnzb = nnzb;
    tiling.idxBase = baseVal;
    tiling.copyValues = copyVal;
    tiling.stripeSize = stripeSize;
    tiling.valSize = valSize;
    tiling.rowBlockDimA = rowBlockDimA;
    tiling.colBlockDimA = colBlockDimA;
    tiling.rowBlockDimC = rowBlockDimC;
    tiling.colBlockDimC = colBlockDimC;
    tiling.copyMode = copyMode;
    tiling.dirA = dirA;
    gebsr2gebsc_scatter_kernel_do(
        gmRowPtr, gmColInd, gmBsrVal,
        gmBscRowInd, gmBscVal, gmStripeHist,
        tiling, numBlocks, stream);
    OP_LOGD("gebsr2gebsc", "Kernel 4 (Scatter) launched, blocks=%u", numBlocks);
}

struct Gebsr2GebscLaunchParams {
    uint32_t numBlocks;
    int32_t stripeSize;
    uint32_t valSize;
    int32_t copyVal;
    int32_t copyMode;
    int32_t dirAVal;
    size_t requiredSize;
};

static Gebsr2GebscLaunchParams ComputeGebsr2GebscLaunchParams(
    int32_t nnzb, int32_t nb,
    uint32_t valSize, aclsparseAction_t copyValues,
    int32_t rowBlockDimA, int32_t colBlockDimA,
    int32_t rowBlockDimC, int32_t colBlockDimC,
    aclsparseDirection_t dirA,
    uint32_t aivCoreNum)
{
    Gebsr2GebscLaunchParams p;
    p.numBlocks = ComputeStripeCount(nnzb, nb, aivCoreNum);
    p.stripeSize = static_cast<int32_t>(
        CeilDiv<uint32_t>(static_cast<uint32_t>(nnzb), p.numBlocks));
    if (p.stripeSize == 0) {
        p.stripeSize = 1;
    }
    p.valSize = valSize;
    p.copyVal = (copyValues == ACL_SPARSE_ACTION_NUMERIC)
        ? kGebsr2GebscNumeric : kGebsr2GebscSymbolic;
    p.copyMode = DetermineCopyMode(
        rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC);
    OP_LOGD("gebsr2gebsc", "copyMode=%d (0=direct,1=transpose)", p.copyMode);
    p.dirAVal = (dirA == ACL_SPARSE_DIRECTION_COLUMN)
        ? kGebsr2GebscDirColumn : kGebsr2GebscDirRow;
    p.requiredSize = ComputeWorkspaceBytes(nb, p.numBlocks);
    return p;
}

static aclsparseStatus_t LaunchGebsr2GebscKernels(
    int mb, int nb, int nnzb,
    const void *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
    void *bscVal, int *bscColPtr, int *bscRowInd,
    void *workspace, uint32_t valSize, int32_t copyVal,
    int32_t baseVal, uint32_t numBlocks, int32_t stripeSize,
    int32_t rowBlockDimA, int32_t colBlockDimA,
    int32_t rowBlockDimC, int32_t colBlockDimC,
    int32_t copyMode, int32_t dirAVal,
    uint32_t aivCoreNum, aclrtStream stream)
{
    // clear-then-launch-then-peek（issue #357 分族）
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    auto *gmColInd = reinterpret_cast<GM_ADDR>(const_cast<int *>(bsrColIndA));
    auto *gmRowPtr = reinterpret_cast<GM_ADDR>(const_cast<int *>(bsrRowPtrA));
    auto *gmBsrVal = reinterpret_cast<GM_ADDR>(const_cast<void *>(bsrValA));
    auto *gmWorkspace = reinterpret_cast<GM_ADDR>(workspace);
    auto *gmBscColPtr = reinterpret_cast<GM_ADDR>(bscColPtr);
    auto *gmBscRowInd = reinterpret_cast<GM_ADDR>(bscRowInd);
    auto *gmBscVal = reinterpret_cast<GM_ADDR>(bscVal);
    auto *gmStripeHist = GetStripeHistPtr(workspace, nb);

    LaunchCountCols(gmColInd, gmStripeHist,
                    nnzb, nb, baseVal, numBlocks, stripeSize, stream);
    LaunchSumStripeHist(gmStripeHist, gmWorkspace,
                        nb, numBlocks, aivCoreNum, stream);
    LaunchPrefixSum(gmWorkspace, gmBscColPtr, nb, baseVal, stream);
    LaunchStripeBase(gmStripeHist, gmBscColPtr, nb, baseVal, numBlocks, stream);
    LaunchScatter(gmRowPtr, gmColInd, gmBsrVal, gmBscRowInd, gmBscVal, gmStripeHist,
                  mb, nb, nnzb, baseVal, copyVal, stripeSize, valSize,
                  rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
                  copyMode, dirAVal, numBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("gebsr2gebsc", "kernel enqueue failed");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;

}

static aclsparseStatus_t LaunchGebsr2GebscKernel(
    aclsparseHandle_t handle, int mb, int nb, int nnzb,
    const void *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
    int rowBlockDimA, int colBlockDimA,
    void *bscVal, int *bscColPtr, int *bscRowInd,
    int rowBlockDimC, int colBlockDimC,
    uint32_t valSize, aclsparseAction_t copyValues,
    aclsparseIndexBase_t idxBase, aclsparseDirection_t dirA,
    void *buffer)
{
    auto *h = ToInternalHandle(handle);
    int32_t baseVal = ToIndexBaseValue(idxBase);

    if (nnzb > INT32_MAX - baseVal) {
        OP_LOGE("gebsr2gebsc",
                "nnzb + idxBase exceeds int32 range: nnzb=%d, baseVal=%d", nnzb, baseVal);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (nnzb == 0) {
        return HandleGebsr2GebscNnzbZero(h->stream, nb, baseVal, bscColPtr);
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE("gebsr2gebsc", "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    auto lp = ComputeGebsr2GebscLaunchParams(
        nnzb, nb, valSize, copyValues,
        rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
        dirA, aivCoreNum);

    void *workspace = nullptr;
    aclsparseStatus_t wsRet = SetupGebsr2GebscWorkspace(
        handle, buffer, lp.requiredSize, &workspace);
    if (wsRet != ACL_SPARSE_STATUS_SUCCESS) {
        return wsRet;
    }

    OP_LOGD("gebsr2gebsc",
            "tiling: mb=%d, nb=%d, nnzb=%d, idxBase=%d, copyValues=%d, "
            "valSize=%u, numBlocks=%u, stripeSize=%d, wsBytes=%zu, "
            "blockDimA=(%d,%d), blockDimC=(%d,%d), copyMode=%d, dirA=%d",
            mb, nb, nnzb, baseVal, lp.copyVal, lp.valSize, lp.numBlocks,
            lp.stripeSize, lp.requiredSize,
            rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
            lp.copyMode, lp.dirAVal);

    aclsparseStatus_t launchSt = LaunchGebsr2GebscKernels(
        mb, nb, nnzb, bsrValA, bsrRowPtrA, bsrColIndA,
        bscVal, bscColPtr, bscRowInd,
        workspace, lp.valSize, lp.copyVal, baseVal, lp.numBlocks, lp.stripeSize,
        rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
        lp.copyMode, lp.dirAVal, aivCoreNum, h->stream);
    if (launchSt != ACL_SPARSE_STATUS_SUCCESS) {
        return launchSt;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

// ----------------------------------------------------------------------------
// Public APIs
// ----------------------------------------------------------------------------

extern "C" {

aclsparseStatus_t aclsparseGebsr2gebsc_bufferSize(
    aclsparseHandle_t handle, int mb, int nb, int nnzb,
    const void *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
    int rowBlockDimA, int colBlockDimA,
    aclsparseDirection_t dirA,
    aclDataType valType,
    size_t *pBufferSizeInBytes)
{
    const char *apiName = "aclsparseGebsr2gebsc_bufferSize";
    aclsparseStatus_t st = ValidateBufferSizeParams(
        apiName, handle, mb, nb, nnzb, rowBlockDimA, colBlockDimA,
        dirA, valType, pBufferSizeInBytes);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    if (mb == 0 && nnzb > 0) {
        OP_LOGE(apiName, "invalid input: mb==0 but nnzb>0");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nb == 0 && nnzb > 0) {
        OP_LOGE(apiName, "invalid input: nb==0 but nnzb>0");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (mb == 0 || nb == 0 || nnzb == 0) {
        *pBufferSizeInBytes = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE(apiName, "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    uint32_t stripeCount = ComputeStripeCount(nnzb, nb, aivCoreNum);
    *pBufferSizeInBytes = ComputeWorkspaceBytes(nb, stripeCount);
    OP_LOGD(apiName, "mb=%d, nb=%d, nnzb=%d, stripeCount=%u, wsBytes=%zu",
            mb, nb, nnzb, stripeCount, *pBufferSizeInBytes);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseGebsr2gebsc(
    aclsparseHandle_t handle, int mb, int nb, int nnzb,
    const void *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
    int rowBlockDimA, int colBlockDimA,
    void *bscVal, int *bscColPtr, int *bscRowInd,
    int rowBlockDimC, int colBlockDimC,
    aclsparseAction_t copyValues, aclsparseIndexBase_t idxBase,
    aclsparseDirection_t dirA,
    aclDataType valType,
    void *pBuffer)
{
    const char *apiName = "aclsparseGebsr2gebsc";
    aclsparseStatus_t st = ValidateGebsr2GebscParams(
        apiName, handle, mb, nb, nnzb,
        bsrValA, bsrRowPtrA, bsrColIndA,
        rowBlockDimA, colBlockDimA,
        bscVal, bscColPtr, bscRowInd,
        rowBlockDimC, colBlockDimC,
        copyValues, idxBase, dirA, valType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    if (mb == 0 && nnzb > 0) {
        OP_LOGE(apiName, "invalid input: mb==0 but nnzb>0");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nb == 0 && nnzb > 0) {
        OP_LOGE(apiName, "invalid input: nb==0 but nnzb>0");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    uint32_t valSize = GetValSize(valType);
    OP_LOGD(apiName,
            "params OK: mb=%d, nb=%d, nnzb=%d, dimA=(%d,%d), dimC=(%d,%d), "
            "copyValues=%d, idxBase=%d, dirA=%d, valType=%d",
            mb, nb, nnzb, rowBlockDimA, colBlockDimA,
            rowBlockDimC, colBlockDimC,
            static_cast<int>(copyValues), static_cast<int>(idxBase),
            static_cast<int>(dirA), static_cast<int>(valType));
    return LaunchGebsr2GebscKernel(
        handle, mb, nb, nnzb,
        bsrValA, bsrRowPtrA, bsrColIndA,
        rowBlockDimA, colBlockDimA,
        bscVal, bscColPtr, bscRowInd,
        rowBlockDimC, colBlockDimC,
        valSize, copyValues, idxBase, dirA, pBuffer);
}

}  // extern "C"
