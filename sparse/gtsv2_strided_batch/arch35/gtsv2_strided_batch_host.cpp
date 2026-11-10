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
 * \file gtsv2_strided_batch_host.cpp
 * \brief aclsparseSgtsv2StridedBatch Host 侧实现。
 *
 * 结构：ValidateGtsv2StridedBatchParams + LaunchGtsv2StridedBatchKernel 拆分。
 * 包含主接口 aclsparseSgtsv2StridedBatch 和 bufferSizeExt 接口。
 * m 无上限方案：m <= 2048 纯 UB 路径（行为不变）；2048 < m <= 2^30 外层 GM 分块 CR 路径。
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "gtsv2_strided_batch.h"
#include "gtsv2_strided_batch_tiling_data.h"
#include "gtsv2_strided_batch_kernel.h"

namespace {

constexpr int32_t kMinM = 3;
constexpr int32_t kMaxMForUBPath = 2048;
// int32 batchStride >= mPad 契约的实际天花板（mPad = 2^31 时 int32 batchStride 不可满足）
constexpr int64_t kMaxMPractical = static_cast<int64_t>(1) << 30;
constexpr int32_t kInnerUbM = 2048;     // 外层路径内层 UB CR 固定规模
constexpr int32_t kInnerUbLayers = 11;  // log2(kInnerUbM)
constexpr int32_t kOuterTileElems = 1024;
constexpr int32_t kMaxLog2MPad = 30;
constexpr int32_t kAlignFactor = 8;
constexpr size_t kWorkspaceAlign = 128;

// UB 容量估算与 kernel 侧纯 UB 路径 InitBuffers（gtsv2_strided_batch_kernel.cpp）实际分配同源：
//   5 个数据队列（aInQue/bInQue/cInQue/dInQue/dOutQue）
// + 12 个向量工作 buffer（oddA-D、aRight-dRight、k1/k2/t1/t2）
// + 2 个 Gather/Scatter 偏移表（sharedOffsetBuf1/2，mPad>8 时启用，取保守上界计入）
// = 19 x vecBytes；另有 4 路 save buffer（GTSV2_NUM_SAVE_ARRAYS x saveBufSize）
constexpr uint32_t kUbDataQueueCount = 5u;    // aInQue/bInQue/cInQue/dInQue/dOutQue
constexpr uint32_t kUbWorkBufferCount = 12u;  // oddA-D、aRight-dRight、k1/k2/t1/t2
constexpr uint32_t kUbOffsetTableCount = 2u;  // sharedOffsetBuf1/2（Gather/Scatter 偏移表）
constexpr uint32_t kUbVecBufferCount = kUbDataQueueCount + kUbWorkBufferCount + kUbOffsetTableCount;

struct Gtsv2LaunchParams {
    int64_t mPadOuter;
    int32_t outerLevels;
    int32_t innerMPad;
    int32_t innerL;
    int32_t innerAlignedM;
    int32_t saveBufSize;
    int64_t regionBByteOffset;
    int64_t saveRegionByteOffset;
    int64_t workspacePerBatchBytes;
    uint32_t numBlocks;
    uint32_t batchPerCoreBase;
    uint32_t remainder;
};

static inline int64_t CalcPaddedM64(int32_t m)
{
    int64_t mPad = 1;
    while (mPad < m) {
        mPad <<= 1;
    }
    return mPad;
}

static inline int64_t CalcAlignedM64(int64_t mPad)
{
    return static_cast<int64_t>(CeilDiv<uint64_t>(mPad, kAlignFactor)) * kAlignFactor;
}

static inline int32_t CalcAlignedM(int32_t mPad)
{
    return static_cast<int32_t>(CeilDiv<uint32_t>(mPad, kAlignFactor)) * kAlignFactor;
}

static inline int32_t CalcSaveBufSizeHost(int32_t mPad, int32_t L)
{
    int32_t saveBufSize = 0;
    int32_t cs = mPad;
    for (int32_t k = 0; k < L; k++) {
        int32_t ac = cs / 2;
        int32_t c = ((ac + kAlignFactor - 1) / kAlignFactor) * kAlignFactor;
        if (c < kAlignFactor) {
            c = kAlignFactor;
        }
        saveBufSize += c * static_cast<int32_t>(sizeof(float));
        cs = ac;
    }
    return saveBufSize;
}

/// 外层 GM 路径 workspace 布局（RegionA / RegionB / SaveRegion），Host 单一来源
static void CalcOuterWorkspaceLayout(
    int64_t mPadOuter, int64_t &regionBByteOffset,
    int64_t &saveRegionByteOffset, int64_t &workspacePerBatchBytes)
{
    int64_t sA = CalcAlignedM64(mPadOuter) + GTSV2_GUARD_FLOATS;
    int64_t sB = CalcAlignedM64(mPadOuter / 2) + GTSV2_GUARD_FLOATS;
    int64_t saveFloats = 0;
    if (mPadOuter > kInnerUbM) {
        saveFloats = GTSV2_NUM_SAVE_ARRAYS * (mPadOuter - kInnerUbM);
    }
    regionBByteOffset = GTSV2_NUM_SAVE_ARRAYS * sA * static_cast<int64_t>(sizeof(float));
    saveRegionByteOffset = regionBByteOffset +
        GTSV2_NUM_SAVE_ARRAYS * sB * static_cast<int64_t>(sizeof(float));
    int64_t perBatch = saveRegionByteOffset + saveFloats * static_cast<int64_t>(sizeof(float));
    workspacePerBatchBytes =
        ((perBatch + static_cast<int64_t>(kWorkspaceAlign) - 1) / static_cast<int64_t>(kWorkspaceAlign)) *
        static_cast<int64_t>(kWorkspaceAlign);
}

/// Legacy API：逐个校验输入矩阵指针（禁止批量 || 合并校验）
static aclsparseStatus_t ValidateGtsv2StridedBatchPointers(
    const float *dl, const float *d, const float *du, const float *x)
{
    if (dl == nullptr) {
        OP_LOGE("aclsparseSgtsv2StridedBatch", "dl is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (d == nullptr) {
        OP_LOGE("aclsparseSgtsv2StridedBatch", "d is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (du == nullptr) {
        OP_LOGE("aclsparseSgtsv2StridedBatch", "du is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        OP_LOGE("aclsparseSgtsv2StridedBatch", "x is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateGmWorkspaceBuffer(void *pBuffer, int m)
{
    if (pBuffer == nullptr) {
        OP_LOGE("aclsparseSgtsv2StridedBatch",
                "pBuffer is required for GM workspace path (m=%d > 2048)", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (reinterpret_cast<uintptr_t>(pBuffer) % kWorkspaceAlign != 0) {
        OP_LOGE("aclsparseSgtsv2StridedBatch",
                "pBuffer not 128B aligned (addr=%p)", pBuffer);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/// 参数校验（主接口）
/// @param isBufferSizeQuery true 时跳过 pBuffer null 检查（bufferSizeExt 阶段尚未分配 pBuffer）
static aclsparseStatus_t ValidateGtsv2StridedBatchParams(
    aclsparseHandle_t handle, int m,
    const float *dl, const float *d, const float *du,
    const float *x, int batchCount, int batchStride,
    void *pBuffer, bool isBufferSizeQuery = false)
{
    if (m < kMinM) {
        OP_LOGE("aclsparseSgtsv2StridedBatch", "m=%d < 3", m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (static_cast<int64_t>(m) > kMaxMPractical) {
        OP_LOGE("aclsparseSgtsv2StridedBatch",
                "m=%d > 2^30 (int32 batchStride >= mPad contract unsatisfiable)", m);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (m > kMaxMForUBPath) {
        if (!isBufferSizeQuery) {
            aclsparseStatus_t bufSt = ValidateGmWorkspaceBuffer(pBuffer, m);
            if (bufSt != ACL_SPARSE_STATUS_SUCCESS) {
                return bufSt;
            }
        }
    }
    aclsparseStatus_t ptrSt = ValidateGtsv2StridedBatchPointers(dl, d, du, x);
    if (ptrSt != ACL_SPARSE_STATUS_SUCCESS) {
        return ptrSt;
    }
    if (batchCount < 0) {
        OP_LOGE("aclsparseSgtsv2StridedBatch", "batchCount=%d < 0", batchCount);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int64_t mPad = CalcPaddedM64(m);
    int64_t alignedM = CalcAlignedM64(mPad);
    if (batchStride % kAlignFactor != 0) {
        OP_LOGE("aclsparseSgtsv2StridedBatch",
                "batchStride=%d not aligned to %d (addr offset may violate 32B DataCopy constraint)",
                batchStride, kAlignFactor);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (static_cast<int64_t>(batchStride) < alignedM) {
        OP_LOGE("aclsparseSgtsv2StridedBatch",
                "batchStride=%d < alignedM=%lld (m=%d, m_pad=%lld, DataCopy 32B alignment)",
                batchStride, static_cast<long long>(alignedM), m, static_cast<long long>(mPad));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static int32_t CalcNumLayers64(int64_t mPad)
{
    int32_t L = 0;
    int64_t tmp = mPad;
    while (tmp > 1) {
        tmp >>= 1;
        L++;
    }
    return L;
}

static aclsparseStatus_t ComputeLaunchParams(int m, int batchCount, Gtsv2LaunchParams &params)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE("aclsparseSgtsv2StridedBatch", "GetAivCoreCount failed");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    params.numBlocks = std::min(aivCoreNum, static_cast<uint32_t>(batchCount));
    if (params.numBlocks == 0) {
        params.numBlocks = 1;
    }
    params.batchPerCoreBase = static_cast<uint32_t>(batchCount) / params.numBlocks;
    params.remainder = static_cast<uint32_t>(batchCount) % params.numBlocks;

    params.mPadOuter = CalcPaddedM64(m);
    int32_t L = CalcNumLayers64(params.mPadOuter);
    if (L > kMaxLog2MPad) {
        OP_LOGE("aclsparseSgtsv2StridedBatch",
                "numLayers=%d exceeds max log2(mPad)=%d", L, kMaxLog2MPad);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    params.outerLevels = (L > kInnerUbLayers) ? (L - kInnerUbLayers) : 0;
    if (params.outerLevels > GTSV2_MAX_OUTER_CR_LEVELS) {
        OP_LOGE("aclsparseSgtsv2StridedBatch",
                "outerLevels=%d exceeds GTSV2_MAX_OUTER_CR_LEVELS=%d",
                params.outerLevels, GTSV2_MAX_OUTER_CR_LEVELS);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // 外层路径内层几何恒为 2048/11；纯 UB 路径为自然几何
    if (params.mPadOuter > kInnerUbM) {
        params.innerMPad = kInnerUbM;
        params.innerL = kInnerUbLayers;
    } else {
        params.innerMPad = static_cast<int32_t>(params.mPadOuter);
        params.innerL = L;
    }
    params.innerAlignedM = CalcAlignedM(params.innerMPad);
    params.saveBufSize = CalcSaveBufSizeHost(params.innerMPad, params.innerL);
    CalcOuterWorkspaceLayout(params.mPadOuter, params.regionBByteOffset,
                             params.saveRegionByteOffset, params.workspacePerBatchBytes);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 返回 true 表示 UB 估算超 90% 容量，纯 UB 路径不可用
static bool CheckUbCapacity(int32_t alignedM, int32_t saveBufSize)
{
    uint64_t vecBytes = static_cast<uint64_t>(alignedM) * sizeof(float);
    // saveBufSize 为单路字节数，CR 需保存 a/b/c/d 四路系数（GTSV2_NUM_SAVE_ARRAYS）
    uint64_t ubEstimate = static_cast<uint64_t>(kUbVecBufferCount) * vecBytes +
                          static_cast<uint64_t>(GTSV2_NUM_SAVE_ARRAYS) * static_cast<uint64_t>(saveBufSize);
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return false;
    }
    uint64_t ubCapacityBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCapacityBytes);
    uint64_t threshold = ubCapacityBytes * 9u / 10u;
    if (ubEstimate > threshold) {
        OP_LOGD("aclsparseSgtsv2StridedBatch",
                "UB estimate %llu > 90%% capacity %llu (threshold %llu), pure UB path unavailable",
                static_cast<unsigned long long>(ubEstimate),
                static_cast<unsigned long long>(ubCapacityBytes),
                static_cast<unsigned long long>(threshold));
        return true;
    }
    return false;
}

static void SetupTilingData(
    Gtsv2StridedBatchTilingData &tiling,
    int m, int batchCount, int batchStride,
    const Gtsv2LaunchParams &params, bool useGmWorkspace)
{
    tiling = {};
    tiling.m = static_cast<int32_t>(m);
    tiling.batchCount = static_cast<int32_t>(batchCount);
    tiling.batchStride = static_cast<int32_t>(batchStride);
    tiling.batchPerCoreBase = static_cast<int32_t>(params.batchPerCoreBase);
    tiling.remainder = static_cast<int32_t>(params.remainder);
    tiling.numLayers = params.innerL;
    tiling.saveBufSize = params.saveBufSize;
    tiling.useGmWorkspace = useGmWorkspace ? 1 : 0;
    tiling.mPad = params.innerMPad;
    tiling.alignedM = params.innerAlignedM;
    tiling.outerLevels = useGmWorkspace ? params.outerLevels : 0;
    tiling.outerTileElems = kOuterTileElems;
    tiling.mPadOuter = params.mPadOuter;
    tiling.regionBByteOffset = params.regionBByteOffset;
    tiling.saveRegionByteOffset = params.saveRegionByteOffset;
    tiling.workspacePerBatchBytes = params.workspacePerBatchBytes;
}

/// Tiling 详情日志（评审响应：自 LaunchGtsv2StridedBatchKernel 提取，压函数行数至 50 内）
static void LogStridedBatchTiling(const Gtsv2StridedBatchTilingData &tiling,
                                  const Gtsv2LaunchParams &params)
{
    OP_LOGD("aclsparseSgtsv2StridedBatch",
            "Tiling: m=%d, batchCount=%d, batchStride=%d, batchPerCoreBase=%d, remainder=%d, numBlocks=%u, "
            "numLayers=%d, saveBufSize=%d, useGmWorkspace=%d, mPad=%d, alignedM=%d, "
            "outerLevels=%d, outerTileElems=%d, mPadOuter=%lld, regionBByteOffset=%lld, "
            "saveRegionByteOffset=%lld, workspacePerBatchBytes=%lld",
            tiling.m, tiling.batchCount, tiling.batchStride, tiling.batchPerCoreBase,
            tiling.remainder, params.numBlocks,
            tiling.numLayers, tiling.saveBufSize, tiling.useGmWorkspace,
            tiling.mPad, tiling.alignedM,
            tiling.outerLevels, tiling.outerTileElems,
            static_cast<long long>(tiling.mPadOuter),
            static_cast<long long>(tiling.regionBByteOffset),
            static_cast<long long>(tiling.saveRegionByteOffset),
            static_cast<long long>(tiling.workspacePerBatchBytes));
}

/// Kernel launch（异步，不调用 aclrtSynchronizeStream）
static aclsparseStatus_t LaunchGtsv2StridedBatchKernel(
    aclsparseHandle_t handle, int m,
    const float *dl, const float *d, const float *du,
    float *x, int batchCount, int batchStride,
    void *pBuffer)
{
    auto *h = Gtsv2ToInternalHandle(handle);
    aclrtStream stream = h->stream;
    if (stream == nullptr) {
        OP_LOGE("aclsparseSgtsv2StridedBatch",
                "stream is nullptr, please call aclsparseSetStream first");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    Gtsv2LaunchParams params;
    aclsparseStatus_t calcSt = ComputeLaunchParams(m, batchCount, params);
    if (calcSt != ACL_SPARSE_STATUS_SUCCESS) {
        return calcSt;
    }

    bool useGmWorkspace = (m > kMaxMForUBPath);
    if (!useGmWorkspace && CheckUbCapacity(params.innerAlignedM, params.saveBufSize)) {
        // K=0 GM 回落路径与 bufferSizeExt（m<=2048 返回 0）存在契约缺口：
        // 回落触发时 kernel 需要 GM workspace 而调用方按契约传 nullptr，故直接拒绝（防御路径）
        OP_LOGE("aclsparseSgtsv2StridedBatch",
                "UB capacity insufficient for pure UB path (m=%d <= %d), GM fallback workspace unavailable",
                m, kMaxMForUBPath);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    Gtsv2StridedBatchTilingData tiling;
    SetupTilingData(tiling, m, batchCount, batchStride, params, useGmWorkspace);
    LogStridedBatchTiling(tiling, params);

    auto *gmDl = reinterpret_cast<GM_ADDR>(const_cast<float *>(dl));
    auto *gmD = reinterpret_cast<GM_ADDR>(const_cast<float *>(d));
    auto *gmDu = reinterpret_cast<GM_ADDR>(const_cast<float *>(du));
    auto *gmX = reinterpret_cast<GM_ADDR>(x);
    auto *gmBuf = reinterpret_cast<GM_ADDR>(pBuffer);

    // clear-then-launch-then-peek（评审响应）：Peek 只读不清除，线程级 last error
    // 会保留先前已被正确处理的 ACL 调用错误（误报归属错位），且一次失败后线程内
    // 永久粘滞。下发前 aclrtGetLastError 清除，使 Peek 只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    gtsv2_strided_batch_kernel_do(gmDl, gmD, gmDu, gmX, gmBuf, tiling, params.numBlocks, stream);

    OP_LOGI("aclsparseSgtsv2StridedBatch", "Kernel launched, numBlocks=%u", params.numBlocks);
    // 下发失败不得静默按成功返回（对齐 ops-solver #361/#366 修复口径）：
    // PeekAtLastError 捕获 enqueue 期错误（非法启动配置等）；异步执行错误仍由
    // 调用方同步流后经既有契约观察。
    const aclError enqueueStatus = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (enqueueStatus != ACL_SUCCESS) {
        OP_LOGE("aclsparseSgtsv2StridedBatch", "kernel enqueue failed: %d", static_cast<int>(enqueueStatus));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

/// bufferSizeExt GM workspace 路径：计算每 batch workspace 布局并按 batchCount 汇总（含 size_t 溢出防御）
static aclsparseStatus_t ComputeWorkspaceSizeForQuery(int m, int batchCount, size_t *bufferSizeInBytes)
{
    int64_t regionBByteOffset = 0;
    int64_t saveRegionByteOffset = 0;
    int64_t workspacePerBatchBytes = 0;
    CalcOuterWorkspaceLayout(CalcPaddedM64(m), regionBByteOffset,
                             saveRegionByteOffset, workspacePerBatchBytes);

    // Kernel 使用全局 batchIdx 索引 workspace（每 batch 段已按 128B 对齐）
    // 防御 size_t 溢出：perBatch x batchCount 极端参数下理论可回绕（物理不可达，1 行可防）；
    // 上界预留 kWorkspaceAlign-1，保证后续 128B 对齐上取整同样不溢出
    if (static_cast<uint64_t>(workspacePerBatchBytes) >
        (SIZE_MAX - (kWorkspaceAlign - 1)) / static_cast<uint64_t>(batchCount)) {
        OP_LOGE("aclsparseSgtsv2StridedBatch_bufferSizeExt",
                "workspace size overflow: perBatch=%lld x batchCount=%d exceeds size_t range",
                static_cast<long long>(workspacePerBatchBytes), batchCount);
        *bufferSizeInBytes = 0;
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    size_t totalSize = static_cast<size_t>(workspacePerBatchBytes) * static_cast<size_t>(batchCount);
    *bufferSizeInBytes = ((totalSize + kWorkspaceAlign - 1) / kWorkspaceAlign) * kWorkspaceAlign;
    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

// ============================================================================
// 主接口：aclsparseSgtsv2StridedBatch
// ============================================================================

extern "C" aclsparseStatus_t aclsparseSgtsv2StridedBatch(
    aclsparseHandle_t handle, int m,
    const float *dl, const float *d, const float *du,
    float *x, int batchCount, int batchStride,
    void *pBuffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgtsv2StridedBatch", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = ValidateGtsv2StridedBatchParams(
        handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (batchCount == 0) {
        OP_LOGD("aclsparseSgtsv2StridedBatch", "batchCount=0, skip kernel");
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    return LaunchGtsv2StridedBatchKernel(
        handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
}

// ============================================================================
// bufferSizeExt 接口：aclsparseSgtsv2StridedBatch_bufferSizeExt
// ============================================================================

extern "C" aclsparseStatus_t aclsparseSgtsv2StridedBatch_bufferSizeExt(
    aclsparseHandle_t handle, int m,
    const float *dl, const float *d, const float *du,
    const float *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSgtsv2StridedBatch_bufferSizeExt", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (bufferSizeInBytes == nullptr) {
        OP_LOGE("aclsparseSgtsv2StridedBatch_bufferSizeExt", "bufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st = ValidateGtsv2StridedBatchParams(
        handle, m, dl, d, du, x, batchCount, batchStride,
        nullptr, true);
    if (st == ACL_SPARSE_STATUS_NOT_SUPPORTED) {
        *bufferSizeInBytes = 0;
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        *bufferSizeInBytes = 0;
        return st;
    }

    if (batchCount == 0) {
        *bufferSizeInBytes = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    if (m <= kMaxMForUBPath) {
        // 与主接口回落判定保持一致（同一几何来源）：UB 不足时同样返回 NOT_SUPPORTED
        int64_t mPad = CalcPaddedM64(m);
        int32_t numLayers = CalcNumLayers64(mPad);
        int32_t alignedM = CalcAlignedM(static_cast<int32_t>(mPad));
        int32_t saveBufSize = CalcSaveBufSizeHost(static_cast<int32_t>(mPad), numLayers);
        if (CheckUbCapacity(alignedM, saveBufSize)) {
            OP_LOGE("aclsparseSgtsv2StridedBatch_bufferSizeExt",
                    "UB capacity insufficient for pure UB path (m=%d <= %d)", m, kMaxMForUBPath);
            *bufferSizeInBytes = 0;
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        }
        *bufferSizeInBytes = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    return ComputeWorkspaceSizeForQuery(m, batchCount, bufferSizeInBytes);
}
