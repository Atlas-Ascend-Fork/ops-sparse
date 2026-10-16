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
 * \file spgemm_host.cpp
 * \brief SpGEMM arch22（Atlas A2 / A3）多阶段 Host 实现。
 *
 * 硬件无关的校验、描述符与状态机在 sparse/spgemm/common/ 中，本文件只负责 arch22
 * 的 tiling 规划、workspace 布局与 Kernel 下发，从而与 A5(arch35) 实现共存。
 */

#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"
#include "../common/spgemm_common.h"
#include "spgemm.h"

#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr int64_t kInt32Max = 2147483647LL;

// Launch 形参统一为 GM_ADDR（host TU 下即 uint8_t*），此处把描述符里的
// void* 设备地址转过去，避免调用处堆叠 static_cast。
inline GM_ADDR AsGm(void *p) { return static_cast<GM_ADDR>(p); }

// 运行时查询 AIV 核数。查询失败时返回错误，避免使用错误的同步区大小。
aclsparseStatus_t GetSpgemmBlockDim(uint32_t &blockDim)
{
    static std::atomic<uint32_t> cached{0};
    static std::mutex queryMutex;
    blockDim = cached.load(std::memory_order_acquire);
    if (blockDim != 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(queryMutex);
    blockDim = cached.load(std::memory_order_relaxed);
    if (blockDim != 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    blockDim = GetAivCoreCount();
    if (blockDim == 0) {
        OP_LOGE("aclsparseSpGEMMWorkEstimation", "GetAivCoreCount failed");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    cached.store(blockDim, std::memory_order_release);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static std::mutex g_arch22CtxMutex;
static std::unordered_map<aclsparseSpGEMMDescr_t, SpgemmArch22Context> g_arch22Ctx;

static SpgemmArch22Context &GetOrCreateArch22Ctx(aclsparseSpGEMMDescr_t descr) {
    std::lock_guard<std::mutex> lock(g_arch22CtxMutex);
    return g_arch22Ctx[descr];
}

static SpgemmArch22Context *GetArch22Ctx(aclsparseSpGEMMDescr_t descr) {
    std::lock_guard<std::mutex> lock(g_arch22CtxMutex);
    auto it = g_arch22Ctx.find(descr);
    return (it != g_arch22Ctx.end()) ? &it->second : nullptr;
}

inline aclsparseSpMatDescr *ToMatInner(aclsparseConstSpMatDescr_t d)
{
    return const_cast<aclsparseSpMatDescr *>(
        reinterpret_cast<const aclsparseSpMatDescr *>(d));
}

// 把 host 侧 tiling 结构写入 buffer1 的 tiling 段。
aclsparseStatus_t UploadTiling(uint8_t *buffer1, int64_t tilingOff,
                               const SpgemmArch22TilingData &td)
{
    if (aclrtMemcpy(buffer1 + tilingOff, sizeof(td), &td, sizeof(td),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/**
 * 按每行中间乘积数 P_i 做连续区间等工作量划分，结果写入 binEdge[blockDim+1]。
 */
void PartitionRowsByWork(const std::vector<int64_t> &rowProducts, uint32_t blockDim,
                         std::vector<int32_t> &binEdge)
{
    const uint32_t M = static_cast<uint32_t>(rowProducts.size());
    binEdge.assign(static_cast<size_t>(blockDim) + 1, static_cast<int32_t>(M));
    binEdge[0] = 0;
    if (M == 0 || blockDim == 0) {
        std::fill(binEdge.begin(), binEdge.end(), 0);
        return;
    }

    int64_t total = 0;
    for (int64_t p : rowProducts) {
        // 每行至少计 1 个单位成本，保证全零行也被均匀分配。
        total += (p > 0 ? p : 1);
    }

    uint32_t curCore = 0;
    int64_t acc = 0;
    for (uint32_t r = 0; r < M && curCore < blockDim; r++) {
        acc += (rowProducts[r] > 0 ? rowProducts[r] : 1);
        // 目标：前 (curCore+1) 个核累计承担 (curCore+1)/blockDim 的总工作量
        const int64_t target = total * static_cast<int64_t>(curCore + 1) /
                               static_cast<int64_t>(blockDim);
        while (curCore < blockDim && acc >= target && r + 1 <= M) {
            curCore++;
            binEdge[curCore] = static_cast<int32_t>(r + 1);
            if (curCore >= blockDim) {
                break;
            }
            const int64_t nextTarget = total * static_cast<int64_t>(curCore + 1) /
                                       static_cast<int64_t>(blockDim);
            if (acc < nextTarget) {
                break;
            }
        }
    }
    // 尾部补齐：剩余核的区间为空，最后一个边界必须等于 M。
    for (uint32_t c = curCore + 1; c <= blockDim; c++) {
        binEdge[c] = static_cast<int32_t>(M);
    }
    binEdge[blockDim] = static_cast<int32_t>(M);
}

/**
 * nnz(C) 上界：逐行取 min(P_i + cInRowNnz, N) 之和。
 * 去重后每行最多 N 个非零，故用 min(·, N) 收紧上界。
 */
int64_t ComputeNnzUpperBound(const std::vector<int64_t> &rowProducts, uint32_t N,
                             int64_t extraCIn)
{
    int64_t ub = 0;
    const int64_t nCap = static_cast<int64_t>(N);
    for (int64_t p : rowProducts) {
        ub += std::min(p, nCap);
        if (ub > kInt32Max) {
            return ub;  // 提前返回，调用方按溢出处理
        }
    }
    ub += extraCIn;
    return ub;
}

// 三个执行阶段共用的准备工作：取 stream / pointerMode / 内部描述符指针 + 全量校验。
struct StageContext {
    aclrtStream stream = nullptr;
    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseSpMatDescr *a = nullptr;
    aclsparseSpMatDescr *b = nullptr;
    aclsparseSpMatDescr *c = nullptr;
    uint32_t dtypeId = SPGEMM_DTYPE_FP32;
};

aclsparseStatus_t PrepareStage(aclsparseHandle_t handle,
                              aclsparseConstSpMatDescr_t matA,
                              aclsparseConstSpMatDescr_t matB,
                              aclsparseSpMatDescr_t matC,
                              aclsparseOperation_t opA, aclsparseOperation_t opB,
                              aclDataType computeType, aclsparseSpGEMMAlg_t alg,
                              aclsparseSpGEMMDescr_t spgemmDescr,
                              StageContext &ctx)
{
    if (handle == nullptr || spgemmDescr == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = aclsparseGetStream(handle, &ctx.stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    st = aclsparseGetPointerMode(handle, &ctx.pointerMode);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    ctx.a = ToMatInner(matA);
    ctx.b = ToMatInner(matB);
    ctx.c = reinterpret_cast<aclsparseSpMatDescr *>(matC);
    st = ValidateSpgemmInputs(ctx.a, ctx.b, ctx.c, opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    (void)SpgemmDtypeFromAcl(computeType, ctx.dtypeId);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 填充与 arch22 Kernel 约定的 tiling 公共字段。
void FillCommonTiling(SpgemmArch22TilingData &td, const StageContext &ctx,
                     const SpgemmArch22Context *ctx22,
                     const SpgemmArch22Ws1Layout &ws1)
{
    td.M = static_cast<uint32_t>(ctx.a->rows);
    td.K = static_cast<uint32_t>(ctx.a->cols);
    td.N = static_cast<uint32_t>(ctx.b->cols);
    td.nnzA = static_cast<uint32_t>(ctx.a->nnz);
    td.nnzB = static_cast<uint32_t>(ctx.b->nnz);
    td.blockDim = ctx22->blockDim;
    td.dtypeId = ctx.dtypeId;
    td.mergeCapacity = SpgemmArch22MergeCapacity(ctx.dtypeId);
    td.chunkWidth = SpgemmArch22ChunkWidth(ctx.dtypeId);
    // 默认 0：数值阶段逐行读 rowProducts。只有在 rowProducts 已算出且
    // max ≤ mergeCapacity 时才由调用方置 1。
    td.prodFitsCapacity = 0;
    td.binEdgeOffset = ws1.binEdgeOff;
    td.rowProductsOffset = ws1.rowProductsOff;
    td.rowNnzOffset = ws1.rowNnzOff;
    td.coreSumOffset = ws1.coreSumOff;
    td.scratchBaseOffset = ws1.scratchBaseOff;
    td.bApStrideOffset = ws1.bApStrideOff;
    td.bHeadOffset = ws1.bHeadOff;
    td.bStatOffset = ws1.bStatOff;
    td.syncOffset = ws1.syncOff;
}

}  // namespace

// Validate only after the handle stream has completed the index snapshots.
// Pinned host storage is required by the asynchronous D2H copies.
struct CsrValidationSnapshot {
    const aclsparseSpMatDescr *mat = nullptr;
    size_t rowBytes = 0;
    size_t colBytes = 0;
    int32_t *rowOffsets = nullptr;
    int32_t *colIndices = nullptr;
};

static aclsparseStatus_t ValidateCsrRowOffsets(const CsrValidationSnapshot &snapshot)
{
    const auto *mat = snapshot.mat;
    if (mat == nullptr || mat->rows == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const int64_t rows = mat->rows;
    const int64_t nnz = static_cast<int64_t>(mat->nnz);
    const auto *row = snapshot.rowOffsets;
    if (row[0] != 0 || row[rows] != static_cast<int32_t>(nnz)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    for (int64_t r = 1; r <= rows; ++r) {
        if (row[r] < row[r - 1]) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateCsrColumnIndices(const CsrValidationSnapshot &snapshot)
{
    const auto *mat = snapshot.mat;
    if (mat == nullptr || mat->nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const int64_t rows = mat->rows;
    const auto *row = snapshot.rowOffsets;
    const auto *col = snapshot.colIndices;
    for (int64_t r = 0; r < rows; ++r) {
        const int32_t rowBegin = row[r];
        const int32_t rowEnd = row[r + 1];
        for (int32_t p = rowBegin; p < rowEnd; ++p) {
            // Duplicate columns are permitted; decreasing columns are not.
            if (col[p] < 0 || col[p] >= mat->cols ||
                (p > row[r] && col[p] < col[p - 1])) {
                return ACL_SPARSE_STATUS_INVALID_VALUE;
            }
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateCsrHostContents(const CsrValidationSnapshot &snapshot)
{
    aclsparseStatus_t status = ValidateCsrRowOffsets(snapshot);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    return ValidateCsrColumnIndices(snapshot);
}

static aclError QueueCsrSnapshot(const CsrValidationSnapshot &snapshot, aclrtStream stream)
{
    aclError ret = ACL_SUCCESS;
    if (snapshot.rowBytes > 0) {
        ret = aclrtMemcpyAsync(snapshot.rowOffsets, snapshot.rowBytes, snapshot.mat->ptrs,
                              snapshot.rowBytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    }
    if (ret == ACL_SUCCESS && snapshot.colBytes > 0) {
        ret = aclrtMemcpyAsync(snapshot.colIndices, snapshot.colBytes, snapshot.mat->idxs,
                              snapshot.colBytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    }
    return ret;
}

static aclsparseStatus_t PrepareCsrSnapshot(CsrValidationSnapshot &snapshot,
                                            size_t &totalBytes)
{
    if (snapshot.mat == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const auto *mat = snapshot.mat;
    if ((mat->rows > 0 && mat->ptrs == nullptr) ||
        (mat->nnz > 0 && (mat->ptrs == nullptr || mat->idxs == nullptr))) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    snapshot.rowBytes = mat->rows > 0
        ? (static_cast<size_t>(mat->rows) + 1) * sizeof(int32_t) : 0;
    snapshot.colBytes = static_cast<size_t>(mat->nnz) * sizeof(int32_t);
    totalBytes += snapshot.rowBytes + snapshot.colBytes;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static void AssignCsrSnapshotStorage(std::array<CsrValidationSnapshot, 2> &snapshots,
                                     void *storage)
{
    auto *cursor = static_cast<uint8_t *>(storage);
    for (auto &snapshot : snapshots) {
        snapshot.rowOffsets = reinterpret_cast<int32_t *>(cursor);
        cursor += snapshot.rowBytes;
        snapshot.colIndices = reinterpret_cast<int32_t *>(cursor);
        cursor += snapshot.colBytes;
    }
}

static aclError QueueCsrSnapshots(const std::array<CsrValidationSnapshot, 2> &snapshots,
                                  aclrtStream stream)
{
    for (const auto &snapshot : snapshots) {
        const aclError status = QueueCsrSnapshot(snapshot, stream);
        if (status != ACL_SUCCESS) {
            return status;
        }
    }
    return ACL_SUCCESS;
}

static aclsparseStatus_t ValidateCsrSnapshotResults(
    const std::array<CsrValidationSnapshot, 2> &snapshots)
{
    for (const auto &snapshot : snapshots) {
        const aclsparseStatus_t status = ValidateCsrHostContents(snapshot);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateCsrMatrices(const aclsparseSpMatDescr *a,
                                             const aclsparseSpMatDescr *b, aclrtStream stream)
{
    std::array<CsrValidationSnapshot, 2> snapshots{};
    snapshots[0].mat = a;
    snapshots[1].mat = b;
    size_t totalBytes = 0;
    for (auto &snapshot : snapshots) {
        const aclsparseStatus_t status = PrepareCsrSnapshot(snapshot, totalBytes);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
    }
    if (totalBytes == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    void *storage = nullptr;
    if (aclrtMallocHost(&storage, totalBytes) != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    AssignCsrSnapshotStorage(snapshots, storage);
    const aclError copyRet = QueueCsrSnapshots(snapshots, stream);
    // Also drain successfully queued copies when a later submission fails,
    // before releasing their host destinations. A/B share this single wait.
    const aclError syncRet = aclrtSynchronizeStream(stream);
    aclsparseStatus_t status = ACL_SPARSE_STATUS_EXECUTION_FAILED;
    if (copyRet == ACL_SUCCESS && syncRet == ACL_SUCCESS) {
        status = ValidateCsrSnapshotResults(snapshots);
    }
    if (aclrtFreeHost(storage) != ACL_SUCCESS && status == ACL_SPARSE_STATUS_SUCCESS) {
        status = ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return status;
}

// ===========================================================================
// 阶段共用子步骤
// ===========================================================================

// alpha / beta 的实部虚部，连同「beta 是否非零」一起带回。
struct SpgemmScalars {
    float alphaRe = 0.0f;
    float alphaIm = 0.0f;
    float betaRe = 0.0f;
    float betaIm = 0.0f;
    bool betaNonZero = false;
};

// Validate A/B together: each index array is copied once on the handle stream.
static aclsparseStatus_t ValidateCsrContents(const StageContext &ctx)
{
    return ValidateCsrMatrices(ctx.a, ctx.b, ctx.stream);
}

// beta != 0 makes matC an input; beta == 0 must not read its contents.
static aclsparseStatus_t ValidateCsrCIn(const StageContext &ctx)
{
    if (ctx.c->nnz > static_cast<uint64_t>(kInt32Max)) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (ctx.c->nnz > 0 && (ctx.c->idxs == nullptr || ctx.c->values == nullptr)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ValidateCsrMatrices(ctx.c, nullptr, ctx.stream);
}

// 校验 CSR 内容并读出 alpha/beta 标量（host 或 device pointerMode 均支持）。
static aclsparseStatus_t ValidateAndReadScalars(const StageContext &ctx,
                                                const void *alpha, const void *beta,
                                                SpgemmScalars &sc)
{
    aclsparseStatus_t st = ValidateCsrContents(ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = SpgemmReadScalar(alpha, ctx.pointerMode, ctx.dtypeId, sc.alphaRe, sc.alphaIm);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = SpgemmReadScalar(beta, ctx.pointerMode, ctx.dtypeId, sc.betaRe, sc.betaIm);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    sc.betaNonZero = (sc.betaRe != 0.0f) || (sc.betaIm != 0.0f);
    if (sc.betaNonZero) {
        st = ValidateCsrCIn(ctx);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 记录跨阶段状态：后续 EstimateMemory/Compute/Copy 都要与此比对。
static void RecordEstimatedStage(aclsparseSpGEMMDescr_t spgemmDescr,
                                 SpgemmArch22Context &arch22ctx, const StageContext &ctx,
                                 aclDataType computeType, aclsparseSpGEMMAlg_t alg,
                                 void *externalBuffer1, size_t bufferSize1,
                                 uint32_t blockDim, uint32_t M, const SpgemmScalars &sc)
{
    spgemmDescr->m = M;
    spgemmDescr->k = static_cast<uint64_t>(ctx.a->cols);
    spgemmDescr->n = static_cast<uint64_t>(ctx.b->cols);
    spgemmDescr->nnzA = static_cast<uint64_t>(ctx.a->nnz);
    spgemmDescr->nnzB = static_cast<uint64_t>(ctx.b->nnz);
    arch22ctx.cachedDtypeId = ctx.dtypeId;
    spgemmDescr->computeType = computeType;
    spgemmDescr->alg = alg;
    arch22ctx.blockDim = blockDim;
    spgemmDescr->externalBuffer1 = externalBuffer1;
    spgemmDescr->requiredBuffer1 = bufferSize1;
    // 在 matC->nnz 被 Compute 覆盖成输出规模之前，先把输入侧的 nnz(C_in) 存下来。
    arch22ctx.nnzCIn = sc.betaNonZero ? static_cast<int64_t>(ctx.c->nnz) : 0;
    arch22ctx.cachedBetaRe = sc.betaRe;
    arch22ctx.cachedBetaIm = sc.betaIm;
}

// 把 WorkEstimation 阶段的标量与规模字段填进 tiling。
static void FillEstimationTiling(SpgemmArch22TilingData &td, const SpgemmScalars &sc,
                                 const SpgemmArch22Context &arch22ctx)
{
    td.alphaRe = sc.alphaRe;
    td.alphaIm = sc.alphaIm;
    td.betaRe = sc.betaRe;
    td.betaIm = sc.betaIm;
    td.betaNonZero = sc.betaNonZero ? 1U : 0U;
    td.nnzCIn = static_cast<uint32_t>(arch22ctx.nnzCIn);
}

static aclsparseStatus_t RunCountKernel(const StageContext &ctx,
                                        SpgemmArch22Context &arch22ctx,
                                        const SpgemmArch22TilingData &td,
                                        const SpgemmArch22Ws1Layout &ws1, uint8_t *buf1,
                                        uint32_t M, uint32_t blockDim)
{
    const size_t rpBytes = static_cast<size_t>(M) * sizeof(int64_t);
    aclsparseStatus_t status = UploadTiling(buf1, ws1.tilingOff, td);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    const size_t syncBytes = static_cast<size_t>(blockDim) * 32;
    if (aclrtMemsetAsync(buf1 + ws1.syncOff, syncBytes, 0, syncBytes,
                         ctx.stream) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    spgemm_arch22_count_launch(AsGm(ctx.a->ptrs), AsGm(ctx.a->idxs),
                               AsGm(ctx.b->ptrs), AsGm(ctx.b->idxs),
                               buf1, buf1 + ws1.tilingOff, blockDim, ctx.stream);
    if (aclrtSynchronizeStream(ctx.stream) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    if (aclrtMemcpy(arch22ctx.rowProducts.data(), rpBytes,
                    buf1 + ws1.rowProductsOff, rpBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ClearCountMetadata(const SpgemmArch22Ws1Layout &ws1,
                                            uint8_t *buf1, uint32_t Kdim)
{
    if (Kdim > 0) {
        const size_t apBytes = static_cast<size_t>(Kdim) * sizeof(int32_t);
        if (aclrtMemset(buf1 + ws1.bApStrideOff, apBytes, 0, apBytes) != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        if (aclrtMemset(buf1 + ws1.bHeadOff, apBytes, 0, apBytes) != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// count Kernel 下发 + 同步回读 rowProducts。
// 退化场景不下发 Kernel，只把描述符区显式清零。
static aclsparseStatus_t RunCountStage(const StageContext &ctx,
                                       SpgemmArch22Context &arch22ctx,
                                       const SpgemmArch22TilingData &td,
                                       const SpgemmArch22Ws1Layout &ws1, uint8_t *buf1,
                                       uint32_t M, uint32_t Kdim, uint32_t blockDim,
                                       bool degenerate)
{
    arch22ctx.rowProducts.assign(static_cast<size_t>(M), 0);
    if (degenerate) {
        return ClearCountMetadata(ws1, buf1, Kdim);
    }
    return RunCountKernel(ctx, arch22ctx, td, ws1, buf1, M, blockDim);
}

// 汇总 numProducts / maxRowProduct，按工作量划分并上传 binEdge。
static aclsparseStatus_t FinishWorkEstimation(aclsparseSpGEMMDescr_t spgemmDescr,
                                              SpgemmArch22Context &arch22ctx,
                                              const SpgemmArch22Ws1Layout &ws1,
                                              uint8_t *buf1, uint32_t blockDim)
{
    int64_t numProds = 0;
    int64_t maxRowProduct = 0;
    for (int64_t p : arch22ctx.rowProducts) {
        numProds += p;
        if (p > maxRowProduct) {
            maxRowProduct = p;
        }
    }
    spgemmDescr->numProducts = numProds;
    arch22ctx.maxRowProduct = maxRowProduct;

    // 按 P_i 做连续区间等工作量划分，写回 binEdge 供符号/数值 Kernel 使用
    PartitionRowsByWork(arch22ctx.rowProducts, blockDim, arch22ctx.binEdge);
    const size_t beBytes = (static_cast<size_t>(blockDim) + 1) * sizeof(int32_t);
    if (aclrtMemcpy(buf1 + ws1.binEdgeOff, beBytes,
                    arch22ctx.binEdge.data(), beBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    // Subsequent capacity queries use the host descriptor and rowProducts.
    // Compute fills and uploads the complete tiling before launching kernels,
    // so uploading an intermediate tiling here would have no consumer.
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 填 Compute 阶段的 tiling：公共字段 + numProds + 上界判定 + 标量 + buffer2 偏移。
static void FillComputeTiling(SpgemmArch22TilingData &td, const StageContext &ctx,
                              const SpgemmArch22Context *arch22ctx,
                              const SpgemmArch22Ws1Layout &ws1,
                              const SpgemmArch22Ws2Layout &ws2,
                              aclsparseSpGEMMDescr_t spgemmDescr, const SpgemmScalars &sc)
{
    FillCommonTiling(td, ctx, arch22ctx, ws1);
    td.numProds = spgemmDescr->numProducts;
    // 若所有行的 maxRowProduct ≤ mergeCapacity，标记 prodFitsCapacity = 1，
    // 使 Kernel 跳过大行分支。
    td.prodFitsCapacity =
        (arch22ctx->maxRowProduct <= static_cast<int64_t>(td.mergeCapacity)) ? 1U : 0U;
    td.alphaRe = sc.alphaRe;
    td.alphaIm = sc.alphaIm;
    td.betaRe = sc.betaRe;
    td.betaIm = sc.betaIm;
    td.betaNonZero = sc.betaNonZero ? 1U : 0U;
    td.nnzCIn = static_cast<uint32_t>(arch22ctx->nnzCIn);
    td.cRowOffsetsOffset = ws2.cRowOffsetsOff;
    td.cColIndicesOffset = ws2.cColIndicesOff;
    td.cValuesOffset = ws2.cValuesOff;
    td.scratchColOffset = ws2.scratchColOff;
    td.scratchValOffset = ws2.scratchValOff;
}

// 计算并上传每核 T1 融合暂存区的起始槽位 scratchBase[c]。
static aclsparseStatus_t UploadScratchBase(const SpgemmArch22Context *arch22ctx,
                                           const SpgemmArch22Ws1Layout &ws1, uint8_t *buf1,
                                           uint32_t N, uint32_t blockDim)
{
    std::vector<int64_t> scratchBase(static_cast<size_t>(blockDim) + 1, 0);
    const int64_t nCap = static_cast<int64_t>(N);
    for (uint32_t c = 0; c < blockDim; c++) {
        int64_t acc = scratchBase[c];
        const uint32_t rs = static_cast<uint32_t>(arch22ctx->binEdge[c]);
        const uint32_t re = static_cast<uint32_t>(arch22ctx->binEdge[c + 1]);
        for (uint32_t r = rs; r < re; r++) {
            const int64_t p = arch22ctx->rowProducts[r];
            acc += (p < nCap) ? p : nCap;
        }
        scratchBase[c + 1] = acc;
    }
    const size_t sbBytes = (static_cast<size_t>(blockDim) + 1) * sizeof(int64_t);
    if (aclrtMemcpy(buf1 + ws1.scratchBaseOff, sbBytes, scratchBase.data(), sbBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 清空符号 Kernel 的软件栅栏区，下发符号 Kernel，同步后回读 nnz(C)。
static aclsparseStatus_t RunSymbolicStage(const StageContext &ctx,
                                          aclsparseSpGEMMDescr_t spgemmDescr,
                                          const SpgemmArch22Ws1Layout &ws1,
                                          const SpgemmArch22Ws2Layout &ws2, uint8_t *buf1,
                                          uint8_t *buf2, void *cRowPtrIn, void *cColIdxIn,
                                          uint32_t M, uint32_t dtypeId, uint32_t blockDim,
                                          int32_t &nnzC)
{
    // 符号 Kernel 的 SyncAll 要求屏障区入核前为零。Compute 可被重复调用，
    // 故这里必须重新清零。
    const size_t syncBytes = static_cast<size_t>(blockDim) * 32;
    if (aclrtMemsetAsync(buf1 + ws1.syncOff, syncBytes, 0, syncBytes,
                         ctx.stream) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    // ---- 第 1 趟：符号 Kernel（含 device 侧前缀和，直接产出 C.rowOffsets）----
    spgemm_arch22_symbolic_launch(
        AsGm(ctx.a->ptrs), AsGm(ctx.a->idxs), AsGm(ctx.a->values),
        AsGm(ctx.b->ptrs), AsGm(ctx.b->idxs), AsGm(ctx.b->values),
        AsGm(cRowPtrIn), AsGm(cColIdxIn), AsGm(spgemmDescr->externalBuffer1), buf2,
        buf1 + ws1.tilingOff, dtypeId, blockDim, ctx.stream);

    if (aclrtSynchronizeStream(ctx.stream) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    nnzC = 0;
    if (aclrtMemcpy(&nnzC, sizeof(nnzC),
                    buf2 + ws2.cRowOffsetsOff + static_cast<int64_t>(M) * sizeof(int32_t),
                    sizeof(nnzC), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 把 buffer2 中的 C 结构搬到用户输出内存（device-to-device）。
static aclsparseStatus_t CopyResultToDevice(const StageContext &ctx,
                                            const SpgemmArch22Context *arch22ctx,
                                            uint8_t *buf2, uint32_t M, int64_t nnzC,
                                            uint64_t valBytes)
{
    const size_t roBytes = (static_cast<size_t>(M) + 1) * sizeof(int32_t);
    if (aclrtMemcpyAsync(ctx.c->ptrs, roBytes, buf2 + arch22ctx->cRowOffsetsOffset,
                         roBytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                         ctx.stream) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    if (nnzC <= 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const size_t ciBytes = static_cast<size_t>(nnzC) * sizeof(int32_t);
    if (aclrtMemcpyAsync(ctx.c->idxs, ciBytes, buf2 + arch22ctx->cColIndicesOffset,
                         ciBytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                         ctx.stream) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    const size_t vBytes = static_cast<size_t>(nnzC) * static_cast<size_t>(valBytes);
    if (aclrtMemcpyAsync(ctx.c->values, vBytes, buf2 + arch22ctx->cValuesOffset, vBytes,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// CreateDescr / DestroyDescr / GetNumProducts
// ===========================================================================
aclsparseStatus_t aclsparseSpGEMMCreateDescr(aclsparseSpGEMMDescr_t *descr)
{
    if (descr == nullptr || *descr != nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *inner = new (std::nothrow) aclsparseSpGEMMDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    inner->signature = kSpGemmSignature;
    *descr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpGEMMDestroyDescr(aclsparseSpGEMMDescr_t descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (descr->signature != kSpGemmSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    {
        std::lock_guard<std::mutex> lock(g_arch22CtxMutex);
        g_arch22Ctx.erase(descr);
    }
    delete descr;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpGEMMGetNumProducts(
    aclsparseSpGEMMDescr_t descr, int64_t *numProds)
{
    if (descr == nullptr || descr->signature != kSpGemmSignature || numProds == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (descr->state < AclsparseSpGemmState::WORK_ESTIMATED) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *numProds = descr->numProducts;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 阶段 1：WorkEstimation
// ===========================================================================
struct WorkEstimationPlan {
    StageContext ctx{};
    uint32_t blockDim = 0;
    uint32_t M = 0;
    uint32_t Kdim = 0;
    SpgemmArch22Ws1Layout ws1{};
};

static aclsparseStatus_t PrepareWorkEstimationPlan(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    aclsparseSpMatDescr_t matC, aclDataType computeType, aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr, WorkEstimationPlan &plan)
{
    aclsparseStatus_t status = PrepareStage(handle, matA, matB, matC, opA, opB,
                                            computeType, alg, spgemmDescr, plan.ctx);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = GetSpgemmBlockDim(plan.blockDim);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    plan.M = static_cast<uint32_t>(plan.ctx.a->rows);
    plan.Kdim = static_cast<uint32_t>(plan.ctx.a->cols);
    plan.ws1 = SpgemmArch22ComputeWs1(plan.M, plan.Kdim, plan.blockDim);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t RunWorkEstimation(const WorkEstimationPlan &plan,
                                           const void *alpha, const void *beta,
                                           aclDataType computeType, aclsparseSpGEMMAlg_t alg,
                                           aclsparseSpGEMMDescr_t spgemmDescr,
                                           SpgemmArch22Context &arch22ctx,
                                           size_t bufferSize1, void *buffer1)
{
    SpgemmScalars scalars;
    aclsparseStatus_t status = ValidateAndReadScalars(plan.ctx, alpha, beta, scalars);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    spgemmDescr->state = AclsparseSpGemmState::CREATED;
    RecordEstimatedStage(spgemmDescr, arch22ctx, plan.ctx, computeType, alg, buffer1,
                         bufferSize1, plan.blockDim, plan.M, scalars);
    SpgemmArch22TilingData tiling{};
    FillCommonTiling(tiling, plan.ctx, &arch22ctx, plan.ws1);
    FillEstimationTiling(tiling, scalars, arch22ctx);
    const bool degenerate = (plan.M == 0) || (plan.ctx.a->cols == 0) ||
                            (plan.ctx.b->cols == 0) || (plan.ctx.a->nnz == 0) ||
                            (plan.ctx.b->nnz == 0);
    auto *buffer = static_cast<uint8_t *>(buffer1);
    status = RunCountStage(plan.ctx, arch22ctx, tiling, plan.ws1, buffer, plan.M,
                           plan.Kdim, plan.blockDim, degenerate);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = FinishWorkEstimation(spgemmDescr, arch22ctx, plan.ws1, buffer, plan.blockDim);
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        spgemmDescr->state = AclsparseSpGemmState::WORK_ESTIMATED;
    }
    return status;
}

aclsparseStatus_t aclsparseSpGEMMWorkEstimation(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize1, void *externalBuffer1)
{
    if (bufferSize1 == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    WorkEstimationPlan plan;
    aclsparseStatus_t status = PrepareWorkEstimationPlan(handle, opA, opB, matA, matB,
                                                          matC, computeType, alg,
                                                          spgemmDescr, plan);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    SpgemmArch22Context &arch22ctx = GetOrCreateArch22Ctx(spgemmDescr);
    if (externalBuffer1 == nullptr) {
        *bufferSize1 = static_cast<size_t>(plan.ws1.totalBytes);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (*bufferSize1 < plan.ws1.totalBytes) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    return RunWorkEstimation(plan, alpha, beta, computeType, alg, spgemmDescr,
                             arch22ctx, *bufferSize1, externalBuffer1);
}

// ===========================================================================
// 阶段 2：EstimateMemory（仅 ALG2 / ALG3）
// ===========================================================================
aclsparseStatus_t aclsparseSpGEMMEstimateMemory(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t spgemmDescr,
    float chunkFraction, size_t *bufferSize3, void *externalBuffer3,
    size_t *bufferSize2)
{
    StageContext ctx;
    aclsparseStatus_t st = PrepareStage(handle, matA, matB, matC, opA, opB,
                                       computeType, alg, spgemmDescr, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    SpgemmArch22Context *arch22ctx = GetArch22Ctx(spgemmDescr);

    // DEFAULT / ALG1 不需要本阶段，调用即为用法错误。
    if (!SpgemmAlgNeedsMemEstimate(alg)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // 阶段顺序：必须先完成 WorkEstimation。
    if (spgemmDescr->state < AclsparseSpGemmState::WORK_ESTIMATED) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    st = ValidateSpgemmStageConsistency(spgemmDescr, ctx.a, ctx.b, computeType,
                                       opA, opB, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (bufferSize3 == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!(chunkFraction > 0.0f) || chunkFraction > 1.0f) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // arch22 的 T1/T3 模板 UB 占用与 N 无关，分块由 Kernel 内的 chunkWidth 完成，
    // 不需要额外的 device 侧临时缓冲，故 bufferSize3 为 0。
    if (externalBuffer3 == nullptr) {
        *bufferSize3 = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (bufferSize2 == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    spgemmDescr->chunkFraction = chunkFraction;

    // buffer2 定容须计入 beta != 0 时并入的 C_in 结构。
    int64_t nnzUB = ComputeNnzUpperBound(arch22ctx->rowProducts,
                                         static_cast<uint32_t>(spgemmDescr->n), arch22ctx->nnzCIn);
    if (nnzUB > kInt32Max) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    const SpgemmArch22Ws2Layout ws2 =
        SpgemmArch22ComputeWs2(static_cast<uint32_t>(spgemmDescr->m), nnzUB, arch22ctx->cachedDtypeId);
    *bufferSize2 = static_cast<size_t>(ws2.totalBytes);

    spgemmDescr->state = AclsparseSpGemmState::MEMORY_ESTIMATED;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateComputeStage(aclsparseSpGEMMDescr_t spgemmDescr,
                                              const StageContext &ctx,
                                              aclDataType computeType,
                                              aclsparseOperation_t opA,
                                              aclsparseOperation_t opB,
                                              aclsparseSpGEMMAlg_t alg)
{
    // 阶段状态机：未 WorkEstimation 直接 Compute 必须报错，不能静默算错。
    if (spgemmDescr->state < AclsparseSpGemmState::WORK_ESTIMATED) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // ALG2/ALG3 声明了分块流程，必须先走 EstimateMemory。
    if (SpgemmAlgNeedsMemEstimate(alg) &&
        spgemmDescr->state < AclsparseSpGemmState::MEMORY_ESTIMATED) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ValidateSpgemmStageConsistency(spgemmDescr, ctx.a, ctx.b, computeType, opA,
                                          opB, alg);
}

// Compute 阶段的规划产物：查询模式只用到 nnzUB / ws2 即早退，其余字段供执行模式使用。
struct ComputePlan {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t dtypeId = SPGEMM_DTYPE_FP32;
    uint32_t blockDim = 0;
    int64_t nnzUB = 0;
    SpgemmArch22Ws1Layout ws1{};
    SpgemmArch22Ws2Layout ws2{};
};

// 算出 nnz 上界与两块 buffer 的布局，并校验调用方给出的容量。
// 查询模式（externalBuffer2 == nullptr）只回填 *bufferSize2，不改动任何描述符状态；
// 调用方据此早退。执行模式才把 buffer2 与其段偏移记入描述符。
static aclsparseStatus_t PlanComputeBuffers(aclsparseSpGEMMDescr_t spgemmDescr,
                                            SpgemmArch22Context *arch22ctx,
                                            const StageContext &ctx, const void *beta,
                                            void *externalBuffer2, size_t *bufferSize2,
                                            SpgemmScalars &sc, ComputePlan &plan)
{
    plan.M = static_cast<uint32_t>(spgemmDescr->m);
    plan.N = static_cast<uint32_t>(spgemmDescr->n);
    plan.dtypeId = arch22ctx->cachedDtypeId;
    plan.blockDim = arch22ctx->blockDim;

    // beta 必须在查询模式早退之前读：nnz 上界依赖它决定是否计入 C_in。
    aclsparseStatus_t st =
        SpgemmReadScalar(beta, ctx.pointerMode, plan.dtypeId, sc.betaRe, sc.betaIm);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    sc.betaNonZero = (sc.betaRe != 0.0f) || (sc.betaIm != 0.0f);
    if (sc.betaRe != arch22ctx->cachedBetaRe || sc.betaIm != arch22ctx->cachedBetaIm) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // C_in 的规模使用 WorkEstimation 时缓存的 nnzCIn，因为 matC->nnz 可能
    // 已被上一轮 Compute 覆盖为输出规模。
    const int64_t extraCIn = sc.betaNonZero ? arch22ctx->nnzCIn : 0;

    plan.nnzUB = ComputeNnzUpperBound(arch22ctx->rowProducts, plan.N, extraCIn);
    if (plan.nnzUB > kInt32Max) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    plan.ws2 = SpgemmArch22ComputeWs2(plan.M, plan.nnzUB, plan.dtypeId);

    // ---- 查询模式 ----
    if (externalBuffer2 == nullptr) {
        *bufferSize2 = static_cast<size_t>(plan.ws2.totalBytes);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (*bufferSize2 < plan.ws2.totalBytes) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    // buffer1 必须仍然是 WorkEstimation 时那一块（内含 binEdge/rowProducts/tiling）。
    if (spgemmDescr->externalBuffer1 == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    spgemmDescr->externalBuffer2 = externalBuffer2;
    spgemmDescr->requiredBuffer2 = *bufferSize2;
    arch22ctx->cRowOffsetsOffset = plan.ws2.cRowOffsetsOff;
    arch22ctx->cColIndicesOffset = plan.ws2.cColIndicesOff;
    arch22ctx->cValuesOffset = plan.ws2.cValuesOff;
    plan.ws1 = SpgemmArch22ComputeWs1(plan.M, static_cast<uint32_t>(ctx.a->cols),
                                      plan.blockDim);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 读 alpha、填 Compute 阶段 tiling、上传 tiling 与 scratchBase。
static aclsparseStatus_t PrepareComputeTiling(const StageContext &ctx,
                                              SpgemmArch22Context *arch22ctx,
                                              aclsparseSpGEMMDescr_t spgemmDescr,
                                              const ComputePlan &plan, SpgemmScalars &sc,
                                              const void *alpha, uint8_t *buf1)
{
    // alpha/beta 可能在 WorkEstimation 之后被调用方改写，故按当前值重填 tiling。
    aclsparseStatus_t st =
        SpgemmReadScalar(alpha, ctx.pointerMode, plan.dtypeId, sc.alphaRe, sc.alphaIm);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    SpgemmArch22TilingData td{};
    FillComputeTiling(td, ctx, arch22ctx, plan.ws1, plan.ws2, spgemmDescr, sc);
    st = UploadTiling(buf1, plan.ws1.tilingOff, td);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return UploadScratchBase(arch22ctx, plan.ws1, buf1, plan.N, plan.blockDim);
}

// 空结构判定：M/N 为 0，或（A/B 无非零元且 beta 为 0）→ nnz(C) 必为 0。
// K 为 0 时，beta != 0 仍须保留并缩放 C_in，不能走清零路径。
static bool IsComputeDegenerate(const StageContext &ctx, const ComputePlan &plan,
                                bool betaNonZero)
{
    return (plan.M == 0) || (plan.N == 0) ||
           ((ctx.a->nnz == 0 || ctx.b->nnz == 0) && !betaNonZero);
}

// 退化场景收尾：rowOffsets 全零、nnz(C)=0，不下发任何 Kernel。
static aclsparseStatus_t FinishDegenerateCompute(const StageContext &ctx,
                                                 aclsparseSpGEMMDescr_t spgemmDescr,
                                                 uint8_t *buf2, const ComputePlan &plan)
{
    const size_t roBytes = (static_cast<size_t>(plan.M) + 1) * sizeof(int32_t);
    if (aclrtMemsetAsync(buf2 + plan.ws2.cRowOffsetsOff, roBytes, 0, roBytes,
                         ctx.stream) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    spgemmDescr->nnzC = 0;
    ctx.c->nnz = 0;
    spgemmDescr->state = AclsparseSpGemmState::COMPUTED;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 两趟 Kernel：符号（含 device 侧前缀和）→ 回读 nnz(C) → 数值（按 dtype 特化）。
static aclsparseStatus_t RunSpgemmKernels(const StageContext &ctx,
                                          aclsparseSpGEMMDescr_t spgemmDescr,
                                          const ComputePlan &plan, uint8_t *buf1,
                                          uint8_t *buf2, void *cRowPtrIn, void *cColIdxIn,
                                          void *cValuesIn)
{
    int32_t nnzC = 0;
    aclsparseStatus_t st =
        RunSymbolicStage(ctx, spgemmDescr, plan.ws1, plan.ws2, buf1, buf2, cRowPtrIn,
                         cColIdxIn, plan.M, plan.dtypeId, plan.blockDim, nnzC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (nnzC < 0 || static_cast<int64_t>(nnzC) > plan.nnzUB) {
        // 符号阶段结果超出预估上界，说明上界模型有误，必须显式报错而不是越界写。
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    spgemmDescr->nnzC = nnzC;
    ctx.c->nnz = static_cast<uint64_t>(nnzC);

    // ---- 第 2 趟：数值 Kernel（按 dtype 特化）----
    if (nnzC > 0) {
        spgemm_arch22_numeric_launch(
            AsGm(ctx.a->ptrs), AsGm(ctx.a->idxs), AsGm(ctx.a->values),
            AsGm(ctx.b->ptrs), AsGm(ctx.b->idxs), AsGm(ctx.b->values),
            AsGm(cRowPtrIn), AsGm(cColIdxIn), AsGm(cValuesIn),
            AsGm(spgemmDescr->externalBuffer1), buf2, buf1 + plan.ws1.tilingOff,
            plan.dtypeId, plan.blockDim, ctx.stream);
    }

    spgemmDescr->state = AclsparseSpGemmState::COMPUTED;
    return ACL_SPARSE_STATUS_SUCCESS;
}


// ===========================================================================
// 阶段 3：Compute（符号 + 数值）
// ===========================================================================
aclsparseStatus_t aclsparseSpGEMMCompute(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize2, void *externalBuffer2)
{
    if (bufferSize2 == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    StageContext ctx;
    aclsparseStatus_t st = PrepareStage(handle, matA, matB, matC, opA, opB,
                                       computeType, alg, spgemmDescr, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    SpgemmArch22Context *arch22ctx = GetArch22Ctx(spgemmDescr);
    st = ValidateComputeStage(spgemmDescr, ctx, computeType, opA, opB, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    SpgemmScalars sc;
    ComputePlan plan;
    st = PlanComputeBuffers(spgemmDescr, arch22ctx, ctx, beta, externalBuffer2,
                            bufferSize2, sc, plan);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // 查询模式：PlanComputeBuffers 已回填 *bufferSize2，不改动任何描述符状态。
    if (externalBuffer2 == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    auto *buf1 = static_cast<uint8_t *>(spgemmDescr->externalBuffer1);
    auto *buf2 = static_cast<uint8_t *>(externalBuffer2);
    st = PrepareComputeTiling(ctx, arch22ctx, spgemmDescr, plan, sc, alpha, buf1);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    const bool betaNonZero = sc.betaNonZero;
    void *cRowPtrIn = betaNonZero ? ctx.c->ptrs : nullptr;
    void *cColIdxIn = betaNonZero ? ctx.c->idxs : nullptr;
    void *cValuesIn = betaNonZero ? ctx.c->values : nullptr;

    // 退化场景：M/N 为 0，或（A/B 无非零元且 beta 为 0）→ nnz(C) 必为 0。
    if (IsComputeDegenerate(ctx, plan, betaNonZero)) {
        return FinishDegenerateCompute(ctx, spgemmDescr, buf2, plan);
    }
    return RunSpgemmKernels(ctx, spgemmDescr, plan, buf1, buf2, cRowPtrIn, cColIdxIn,
                            cValuesIn);
}

// ===========================================================================
// 阶段 4：Copy（把 buffer2 中的结果搬到 matC 的用户内存）
// ===========================================================================
aclsparseStatus_t aclsparseSpGEMMCopy(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t spgemmDescr)
{
    StageContext ctx;
    aclsparseStatus_t st = PrepareStage(handle, matA, matB, matC, opA, opB,
                                       computeType, alg, spgemmDescr, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    SpgemmArch22Context *arch22ctx = GetArch22Ctx(spgemmDescr);

    // 阶段状态机：必须先 Compute。
    if (spgemmDescr->state < AclsparseSpGemmState::COMPUTED) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    st = ValidateSpgemmStageConsistency(spgemmDescr, ctx.a, ctx.b, computeType,
                                       opA, opB, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    const uint32_t M = static_cast<uint32_t>(spgemmDescr->m);
    const int64_t nnzC = spgemmDescr->nnzC;
    const uint64_t valBytes = SpgemmArch22ValueBytes(arch22ctx->cachedDtypeId);

    // rowOffsets 长度 M+1，即使 nnz(C)=0 也必须写出（全零的合法空结构）。
    if (M > 0 && ctx.c->ptrs == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnzC > 0 && (ctx.c->idxs == nullptr || ctx.c->values == nullptr)) {
        // 调用方未按 nnz(C) 回填输出指针。
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (M == 0) {
        spgemmDescr->state = AclsparseSpGemmState::COPIED;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (spgemmDescr->externalBuffer2 == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *buf2 = static_cast<uint8_t *>(spgemmDescr->externalBuffer2);

    st = CopyResultToDevice(ctx, arch22ctx, buf2, M, nnzC, valBytes);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    spgemmDescr->state = AclsparseSpGemmState::COPIED;
    return ACL_SPARSE_STATUS_SUCCESS;
}
