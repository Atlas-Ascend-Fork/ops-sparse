/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file prune_host.cpp
 * \brief LtMatmulSpMMAPrune host-side API implementation.
 *
 * Launches the prune kernel: A (d_in) -> A_pruned (d_out).
 *
 * Tiling is passed by value through the kernel launch args block
 * (the AclsparseltTilingData POD struct is forwarded by the launcher into the
 * <<<>>> invocation). This aligns Prune with the cuSPARSELt SpMMAPrune
 * properties:
 *   - D5 "requires no extra storage": no aclrtMalloc'd tilingGm buffer.
 *   - D4 "supports asynchronous execution with respect to stream": no
 *     aclrtSynchronizeStream; the API returns immediately after launching the
 *     kernel on the stream (the args block is managed by the ACL runtime for
 *     the lifetime of the launched task, so freeing a host-owned buffer is not
 *     a concern).
 * The matmul path conveys tiling through the user-provided workspace; Prune has
 * no workspace parameter (matching the NVIDIA signature), so the args block is
 * the natural channel for the small (~84 B) tiling struct.
 */

// Include order: C standard → C++ → CANN → local.
#include <cstdint>

#include "acl/acl_rt.h"
#include "log/log.h"
#include "cann_ops_sparseLt.h"
#include "shared/aclsparselt_internal.h"
#include "prune/arch35/prune_kernel.h"

// ============================================================================
// SpMMAPrune — launch prune kernel: A (d_in) -> A_pruned (d_out).
//
// Split into three focused helpers:
//   1. validate_prune_params  — null/enum checks + core param derivation
//   2. resolve_prune_buffers  — UB capacity + buffer pointer/alignment checks
//   3. compute_prune_tiling   — AclsparseltTilingData struct construction
// The main function is a thin orchestration: validate → resolve → tile → launch.
// ============================================================================

// ----------------------------------------------------------------------------
// validate_prune_params — validate API inputs and derive core parameters.
// On success, outputs md / dt / m / k / pruneAlongRow / pruneAlgOut /
// sparseMat (the structured-side mat descriptor carrying the batch layout).
// ----------------------------------------------------------------------------
static aclsparseStatus_t derive_bsparse_params(
    aclsparseLtMatmulDescriptor_t md, int32_t& dt,
    int32_t& m, int32_t& k, int32_t& pruneAlongRow,
    const aclsparseLtMatDescriptor*& sparseMat)
{
    // B-sparse: B is the (k, n) structured matrix; prune operates on B.
    // Map B's (rows=k, cols=n) to the kernel's (m, k) naming convention.
    dt = dtype_from_acl(md->matB->valueType);
    if (dt == SPLT_DTYPE_INVALID) {
        OP_LOGE(kSparseLtLogTag, "Prune: unsupported dtype (only FP32/FP16/BF16/INT8)");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    m = md_k(md);   // B rows = logical k
    k = md_n(md);   // B cols = logical n
    const bool isRowOrder = (md->matB->order == ACL_SPARSE_ORDER_ROW);
    pruneAlongRow = (md_transB(md) != isRowOrder) ? 1 : 0;
    sparseMat = md->matB;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t validate_prune_params(
    const aclsparseLtHandle_t* handle, aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    aclsparseLtPruneAlg_t pruneAlg, aclrtStream stream,
    aclsparseLtMatmulDescriptor_t& md, int32_t& dt,
    int32_t& m, int32_t& k, int32_t& pruneAlongRow, int32_t& pruneAlgOut,
    const aclsparseLtMatDescriptor*& sparseMat)
{
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Prune: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matmulDescr == nullptr || *matmulDescr == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Prune: matmulDescr is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // stream 可为 nullptr（ACL 默认流），不做非空校验。
    if (pruneAlg != ACLSPARSELT_PRUNE_SPMMA_STRIP && pruneAlg != ACLSPARSELT_PRUNE_SPMMA_TILE) {
        OP_LOGE(kSparseLtLogTag, "Prune: unsupported pruneAlg %d",
                static_cast<int>(pruneAlg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    pruneAlgOut = static_cast<int32_t>(pruneAlg);


    md = to_matmul_internal(*matmulDescr);
    if (md->matA == nullptr || md->matB == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Prune: matA/matB is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // Derive prune parameters from the matmul descriptor (no plan needed).
    // isSparseA: A structured -> prune A (existing path).
    //            B structured -> prune B (matmulDescr implicit detection).
    if (md_isSparseA(md)) {
        dt = dtype_from_acl(md->matA->valueType);
        if (dt == SPLT_DTYPE_INVALID) {
            OP_LOGE(kSparseLtLogTag, "Prune: unsupported dtype (only FP32/FP16/BF16/INT8)");
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        }
        m = md_m(md);
        k = md_k(md);
        const bool isRowOrder = (md->matA->order == ACL_SPARSE_ORDER_ROW);
        pruneAlongRow = (md_transA(md) != isRowOrder) ? 1 : 0;
        sparseMat = md->matA;
    } else {
        return derive_bsparse_params(md, dt, m, k, pruneAlongRow, sparseMat);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// checked_mul_add — shared/aclsparselt_internal.h 共享 helper（本 TU 经
// include 引入，原本地重复定义已消除）。
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// validate_prune_batch — 读取 structured 侧 mat（A-sparse 为 matA、
// B-sparse 为 matB）的 numBatches / batchStride（元素单位，与
// BATCH_STRIDE 属性语义一致）并校验 batch 布局：
//   numBatches == 1：直通返回（单 batch 路径行为与返回码完全不变）；
//   numBatches > 1：
//     V1  batchStride > 0 且 ≥ 输入物理访问 span（按 kernel 路径推导，
//         与 pruneAlongRow/sparseTrans 分派同源——见下方 inSpan 段）且
//         ≥ 输出单 batch 大小（prune 输出为 (m,k) 行主序紧凑布局，A/B
//         两侧在函数参数命名约定下均为 m×k）——输入输出共用
//         batchStride 推进，逐 batch 基址不越界、batch 区不重叠；
//     V2  batchStride 为 16/elemSize 的倍数（⟺ batchStride × elemSize
//         为 16 字节倍数——elemSize ∈ {1,2,4} 均为 2 的幂，纯取模
//         等价判断，不计算乘积；配合 d_in/d_out 基址的 16B 校验，逐
//         batch 基址恒 16B 对齐）；
//     V3  (numBatches-1)×batchStride + span 与 ×elemSize 按 int64 防溢出
//         （最终防线：V2 放行的对齐巨大 stride 由本层拦截）。
// pruneAlongRow / sparseTrans 与 kernel 路径分派（SpltPruneImpl）同源，
// 由调用方自 md 推导传入。
// ----------------------------------------------------------------------------
static aclsparseStatus_t validate_prune_batch(
    const aclsparseLtMatDescriptor* sparseMat, int32_t dt, int32_t m, int32_t k,
    int32_t pruneAlongRow, bool sparseTrans,
    int32_t& numBatches, int64_t& batchStride)
{
    numBatches = (sparseMat->numBatches > 0) ? sparseMat->numBatches : 1;
    batchStride = sparseMat->batchStride;
    if (numBatches <= 1) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const int64_t elemSize =
        (dt == SPLT_DTYPE_FP32) ? SPLT_FP32_BYTES :
        (dt == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES : SPLT_HALF_BYTES;
    // V1: inSpan（元素单位）。按 kernel 实际访问路径推导（与
    // SpltPruneImpl 的 pruneAlongRow/sparseTrans 分派同源）：
    //   TransRowOrder（pruneAlongRow==0 && sparseTrans）：物理按 (k,m)
    //     行主序寻址（SpltPruneLoadRowBlockChunk 的 rowStart*ld+colStart，
    //     行距 ld），单 batch 跨度 (k-1)·ld+m；
    //   其余（AlongRow / AlongCol）：物理按 (m,k) 行主序寻址
    //     （SpltPruneAlongRowUnified / SpltPruneLoadRowBlock 的 rowStart*ld，
    //     行长 k），单 batch 跨度 (m-1)·ld+k。
    // 此前按描述符 order 的列主序跨度（COL 序 (cols-1)·ld+rows）推导，
    // 与 COL+NT 的 AlongCol 路径实际访问不一致：m>k 时低估（batch 串读
    // + 末批越界）、m<k 时误拒（紧凑布局被误判）。rows/cols/ld 为 int64
    // 描述符字段且 setter 无上限校验，(rows/cols-1)·ld 乘加链先做受检查
    // 计算，溢出即布局非法（物理 span 超 int64 无法被 kernel 寻址）。
    // 输出为 (m,k) 行主序单 batch 紧凑布局（outSpan 为 int32 域乘积，
    // ≤ 2^62，无溢出面）。setter 保证 rows/cols/ld > 0，乘子非负，
    // checked_mul_add 前置条件成立。
    int64_t inSpan = 0;
    const bool isTransRowPath = (pruneAlongRow == 0 && sparseTrans);
    const bool inSpanOk = isTransRowPath
        ? checked_mul_add(k - 1, sparseMat->ld, m, inSpan)
        : checked_mul_add(m - 1, sparseMat->ld, k, inSpan);
    if (!inSpanOk) {
        OP_LOGE(kSparseLtLogTag,
                "Prune: input physical span overflow (rows=%lld, cols=%lld, ld=%lld)",
                static_cast<long long>(sparseMat->rows),
                static_cast<long long>(sparseMat->cols),
                static_cast<long long>(sparseMat->ld));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const int64_t outSpan = static_cast<int64_t>(m) * static_cast<int64_t>(k);
    const int64_t span = (inSpan > outSpan) ? inSpan : outSpan;
    if (batchStride <= 0 || batchStride < span) {
        OP_LOGE(kSparseLtLogTag,
                "Prune: batchStride=%lld < span=%lld (inSpan=%lld, outSpan=%lld, numBatches=%d)",
                static_cast<long long>(batchStride), static_cast<long long>(span),
                static_cast<long long>(inSpan), static_cast<long long>(outSpan), numBatches);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // V2: 逐 batch 基址 16B 对齐。无乘法等价判断：
    // elemSize ∈ {1,2,4} 均为 2 的幂，故
    //   batchStride·elemSize ≡ 0 (mod 16) ⟺ batchStride ≡ 0 (mod 16/elemSize)
    // （elemSize=1→16、FP16/BF16→8、FP32→4）——纯取模，不计算可溢出的
    // batchStride·elemSize 乘积：stride=2^61 × elemSize=4 的输入先乘后检
    // 会溢出 int64（UB），无法保证按预期返回 INVALID_VALUE；本判断在
    // 无溢出域与原逐位等价，溢出域从 UB 变为确定行为（对齐者放行，由
    // V3 最终拦截）。V1 已保证 batchStride > 0，取模安全；日志亦只打印
    // 操作数（打印乘积同样构成溢出 UB）。
    if (batchStride % (16 / elemSize) != 0) {
        OP_LOGE(kSparseLtLogTag,
                "Prune: batchStride=%lld not 16-byte aligned (elemSize=%lld, requires multiple of %lld)",
                static_cast<long long>(batchStride), static_cast<long long>(elemSize),
                static_cast<long long>(16 / elemSize));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // V3: 溢出防护（乘加链，最终防线）。
    int64_t totalElems = 0;
    int64_t totalBytes = 0;
    if (!checked_mul_add(numBatches - 1, batchStride, span, totalElems) ||
        !checked_mul_add(totalElems, elemSize, 0, totalBytes)) {
        OP_LOGE(kSparseLtLogTag, "Prune: batch layout overflow (numBatches=%d, batchStride=%lld)",
                numBatches, static_cast<long long>(batchStride));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// compute_prune_chunk_m — compute the maximum chunkM (columns per chunk) for
// the trans path so that the per-chunk UB usage fits within the UB capacity.
//
// When m <= chunkM, the kernel uses a single chunk (= m), identical to the
// original behavior. When m > chunkM, the kernel splits the m-dimension into
// chunks of chunkM columns each, loading TRANS_TILE(16) rows × chunkM cols
// per chunk.
//
// chunkM is aligned to 32B/elemSize so that the UB row stride is exact
// (alignedChunkM == chunkM). Since 32B/elemSize is always a multiple of both
// groupSize and TS, this alignment also ensures groups/tiles don't span chunk
// boundaries.
//
// Returns: chunkM (>= 1 if supported), or 0 if UB too small for even one
// minimum tile/group.
// ----------------------------------------------------------------------------
static int32_t compute_prune_chunk_m(int32_t dt, int32_t m, int32_t pruneAlg)
{
    const int64_t elemSize =
        (dt == SPLT_DTYPE_FP32) ? SPLT_FP32_BYTES :
        (dt == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES : SPLT_HALF_BYTES;
    const bool isTile = (pruneAlg == ACLSPARSELT_PRUNE_SPMMA_TILE);
    const int32_t groupSize = (dt == SPLT_DTYPE_FP32) ? 2 : 4;
    // Alignment granularity = 32B / elemSize elements (always a multiple of TS/groupSize).
    const int32_t alignUnit = SPLT_UB_ALIGN_BYTES / static_cast<int32_t>(elemSize);
    const int32_t TS = isTile
        ? ((dt == SPLT_DTYPE_FP32) ? SPLT_TILE_SIZE_FP32 : SPLT_TILE_SIZE_FP16)
        : groupSize;
    // transUB stride: INT8 uses 32, others use 16 (TRANS_TILE).
    const int32_t transUbStride = (elemSize == 1) ? 32 : 16;
    constexpr int32_t TRANS_TILE = 16;
    const int64_t ubBytes = static_cast<int64_t>(get_ub_size());

    // UB budget (exact, since chunkM is aligned to 32B/elemSize → alignedChunkM == chunkM):
    //   blkUB      = TRANS_TILE * chunkM * elemSize + TRANS_TILE * elemSize  (row padding)
    //   transUB    = TRANS_TILE * transUbStride * elemSize
    //   absBuf     = (chunkM + 64) * sizeof(float)   [STRIP only]
    //   keepMaskBuf= (chunkM + 64) * sizeof(float)   [STRIP only]
    const int64_t blkPadding = static_cast<int64_t>(TRANS_TILE) * elemSize;
    const int64_t transBytes = static_cast<int64_t>(TRANS_TILE) * transUbStride * elemSize;
    const int64_t vecOverhead = isTile ? 0 : 2 * 64 * static_cast<int64_t>(sizeof(float));
    const int64_t overhead = blkPadding + transBytes + vecOverhead;
    if (overhead >= ubBytes) {
        return 0;
    }
    const int64_t available = ubBytes - overhead;

    // per-chunkM bytes: blkUB row data + (absBuf + keepMaskBuf for STRIP)
    const int64_t perChunkBytes = isTile
        ? static_cast<int64_t>(TRANS_TILE) * elemSize
        : static_cast<int64_t>(TRANS_TILE) * elemSize + 2 * static_cast<int64_t>(sizeof(float));

    int64_t maxChunkM = available / perChunkBytes;
    // Align down to alignUnit (32B/elemSize).
    maxChunkM = (maxChunkM / alignUnit) * alignUnit;
    if (maxChunkM < TS) {
        return 0;
    }

    // chunkM = min(maxChunkM, m). When m <= maxChunkM, single chunk (= m).
    int32_t chunkM = (maxChunkM < m) ? static_cast<int32_t>(maxChunkM) : m;
    return chunkM;
}

static aclsparseStatus_t check_ub_capacity(
    aclsparseLtMatmulDescriptor_t md, int32_t dt,
    int32_t m, int32_t k, int32_t pruneAlongRow, int32_t pruneAlg)
{
    const int64_t elemSize =
        (dt == SPLT_DTYPE_FP32) ? SPLT_FP32_BYTES :
        (dt == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES : SPLT_HALF_BYTES;
    const bool isTile = (pruneAlg == ACLSPARSELT_PRUNE_SPMMA_TILE);
    const bool sparseTrans = md_isSparseA(md) ? md_transA(md) : md_transB(md);
    const bool isTransRowPath = (pruneAlongRow == 0 && sparseTrans);
    const int64_t SPLT_UB_BYTES = static_cast<int64_t>(get_ub_size());

    // Trans path supports chunked loading: when full m exceeds UB,
    // the kernel splits m into chunks of chunkM columns. Return NOT_SUPPORTED
    // only if even a single minimum tile/group cannot fit in UB.
    if (isTransRowPath) {
        const int32_t chunkM = compute_prune_chunk_m(dt, m, pruneAlg);
        const int32_t TS = isTile
            ? ((dt == SPLT_DTYPE_FP32) ? SPLT_TILE_SIZE_FP32 : SPLT_TILE_SIZE_FP16)
            : ((dt == SPLT_DTYPE_FP32) ? 2 : 4);
        if (chunkM < TS) {
            OP_LOGE(kSparseLtLogTag, "Prune: trans path chunkM=%d < TS=%d (m=%d, dt=%d, alg=%d)",
                    chunkM, TS, m, dt, pruneAlg);
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    // Non-trans paths: keep existing full-load UB check (rowDim = k).
    const int32_t rowDim = k;
    const int64_t rowBytes = static_cast<int64_t>(rowDim) * elemSize;
    const int32_t tileSize = isTile
        ? ((dt == SPLT_DTYPE_FP32) ? SPLT_TILE_SIZE_FP32 : SPLT_TILE_SIZE_FP16)
        : 1;
    const int64_t alignedRowBytes = (rowBytes + (SPLT_UB_ALIGN_BYTES - 1)) /
                                    SPLT_UB_ALIGN_BYTES * SPLT_UB_ALIGN_BYTES;
    const int64_t dataBytes = (pruneAlongRow != 0)
        ? alignedRowBytes * static_cast<int64_t>(tileSize)
        : alignedRowBytes * static_cast<int64_t>(SPLT_PRUNE_ROW_TILE);
    const bool hasVecBuf = (pruneAlongRow != 0);
    const int64_t vecBufBytes = hasVecBuf
        ? 2 * static_cast<int64_t>(rowDim + 64) * static_cast<int64_t>(sizeof(float))
        : 0;
    const int64_t ubBytes = dataBytes + vecBufBytes;
    if (ubBytes > SPLT_UB_BYTES) {
        OP_LOGE(kSparseLtLogTag, "Prune: k=%d exceeds UB capacity (ubBytes=%lld > %lld, alongRow=%d)",
                rowDim,
                static_cast<long long>(ubBytes), static_cast<long long>(SPLT_UB_BYTES),
                pruneAlongRow);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// resolve_prune_buffers — validate UB capacity and resolve data pointers.
// On success, outputs aIn (source) / aPruned (destination).
// ----------------------------------------------------------------------------
static aclsparseStatus_t resolve_prune_buffers(
    const void* d_in, void* d_out, aclsparseLtMatmulDescriptor_t md,
    int32_t dt, int32_t m, int32_t k, int32_t pruneAlongRow, int32_t pruneAlg,
    const void*& aIn, void*& aPruned)
{
    aclsparseStatus_t ubSt = check_ub_capacity(md, dt, m, k, pruneAlongRow, pruneAlg);
    if (ubSt != ACL_SPARSE_STATUS_SUCCESS) {
        return ubSt;
    }

    if (d_in == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Prune: d_in is null (must be passed explicitly)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aIn = d_in;
    if (d_out == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Prune: d_out is null (no workspace fallback)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 转置路径（sparseTrans=1, pruneAlongRow=0）不支持 in-place：输出布局 (m,k) 与
    // 输入布局 (k,m) 不同，多核并行写入会覆盖后续 block 的读取区域。
    const bool sparseTrans = md_isSparseA(md) ? md_transA(md) : md_transB(md);
    const bool isTransRowPath = (pruneAlongRow == 0 && sparseTrans);
    if (isTransRowPath && d_in == d_out) {
        OP_LOGE(kSparseLtLogTag, "Prune: in-place (d_in == d_out) not supported for transpose path");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if ((reinterpret_cast<uintptr_t>(aIn) % 16) != 0 ||
        (reinterpret_cast<uintptr_t>(d_out) % 16) != 0) {
        OP_LOGE(kSparseLtLogTag, "Prune: d_in/d_out not 16-byte aligned");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aPruned = d_out;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// compute_prune_tiling — build the AclsparseltTilingData struct for the kernel.
// The prune kernel consumes m, k, pruneAlongRow, pruneAlg from tiling.
// Other fields are zeroed — they are irrelevant to the prune path.
// ----------------------------------------------------------------------------
static AclsparseltTilingData compute_prune_tiling(
    aclsparseLtMatmulDescriptor_t md, int32_t dt, int32_t m,
    int32_t k, int32_t pruneAlongRow, int32_t pruneAlg)
{
    AclsparseltTilingData td{};
    td.m = m;
    td.k = k;
    // Pass leading dimension to kernel so row offsets use row*ld
    // instead of row*k. When ld > k (padded matrix), this prevents incorrect
    // memory access from the 2nd row onward. md->matA->ld is int64_t; truncate
    // to int32_t (same width as m/k — matrices fit in int32_t per existing code).
    // 描述符层（StructuredDescriptorInit）已显式拒绝 ld > INT32_MAX，收窄
    // 无截断（合法使用域由上游校验保证）。
    // ld from the structured matrix: A-sparse -> matA->ld; B-sparse -> matB->ld.
    const int64_t sparseLd = md_isSparseA(md) ? md->matA->ld : md->matB->ld;
    td.ld = static_cast<int32_t>(sparseLd);
    // The minimum ld depends on which kernel path runs:
    //   pruneAlongRow=0, sparseTrans=1 → SpltPruneTransRowOrder: physical
    //     matrix is (k, m) row-major, each physical row has m elements → ld ≥ m.
    //   All other paths: physical matrix has rows of k elements → ld ≥ k.
    const bool sparseTrans = md_isSparseA(md) ? md_transA(md) : md_transB(md);
    const bool isTransRowPath = (pruneAlongRow == 0 && sparseTrans);
    const int32_t minLd = isTransRowPath ? td.m : td.k;
    if (td.ld < minLd) {
        td.ld = minLd;
    }
    td.pruneAlongRow = pruneAlongRow;
    td.pruneAlg = pruneAlg;
    td.sparseTrans = sparseTrans ? 1 : 0;
    td.dataType = dt;
    td.coreNum = static_cast<int32_t>(get_cube_core_num());
    // Compute chunkM for the trans path. When m > chunkM, the kernel
    // splits the m-dimension into chunks. Non-trans paths set chunkM = 0
    // (kernel ignores it). When m <= chunkM (single chunk), chunkM = m.
    td.chunkM = isTransRowPath ? compute_prune_chunk_m(dt, m, pruneAlg) : 0;
    // [TILE] TILE alongRow uses TS rows/block (vs 1 for STRIP alongRow).
    const int32_t tileSize = (pruneAlg == ACLSPARSELT_PRUNE_SPMMA_TILE)
        ? ((dt == SPLT_DTYPE_FP32) ? SPLT_TILE_SIZE_FP32 : SPLT_TILE_SIZE_FP16)
        : 1;
    // For alongCol paths, physical row count = sparseTrans ? k : m.
    // When sparseTrans=1, the physical matrix is (k, m) row-major, so k is the
    // physical row count (m is logical row count, used for alongRow paths).
    const int32_t pruneTiles = (pruneAlongRow != 0)
        ? ((m + tileSize - 1) / tileSize)
        : ((td.sparseTrans ? k : m) + SPLT_PRUNE_ROW_TILE - 1) / SPLT_PRUNE_ROW_TILE;
    td.usedCoreNum = (pruneTiles < static_cast<int32_t>(get_cube_core_num()))
        ? pruneTiles
        : static_cast<int32_t>(get_cube_core_num());
    if (td.usedCoreNum <= 0) {
        td.usedCoreNum = 1;
    }
    return td;
}

// ============================================================================
// Public API entry point.
// ============================================================================
extern "C" aclsparseStatus_t aclsparseLtSpMMAPrune(
    const aclsparseLtHandle_t* handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    void* d_out,
    aclsparseLtPruneAlg_t pruneAlg,
    aclrtStream stream)
{
    aclsparseLtMatmulDescriptor_t md = nullptr;
    const aclsparseLtMatDescriptor* sparseMat = nullptr;
    int32_t dt = 0;
    int32_t m = 0;
    int32_t k = 0;
    int32_t pruneAlongRow = 0;
    int32_t pruneAlgInt = 0;
    aclsparseStatus_t st = validate_prune_params(handle, matmulDescr, pruneAlg, stream,
                                                    md, dt, m, k, pruneAlongRow, pruneAlgInt,
                                                    sparseMat);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // Batch 布局读取与校验（numBatches == 1 时直通，单 batch 路径零变更）。
    // pruneAlongRow / sparseTrans 与 kernel 路径分派同源（validate_prune_params
    // 的推导 + md 的转置位），供 V1 span 按 kernel 实际访问路径选取公式。
    int32_t numBatches = 1;
    int64_t batchStride = 0;
    const bool sparseTransForSpan = md_isSparseA(md) ? md_transA(md) : md_transB(md);
    st = validate_prune_batch(sparseMat, dt, m, k, pruneAlongRow, sparseTransForSpan,
                              numBatches, batchStride);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    const void* aIn = nullptr;
    void* aPruned = nullptr;
    st = resolve_prune_buffers(d_in, d_out, md, dt, m, k, pruneAlongRow, pruneAlgInt, aIn, aPruned);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    AclsparseltTilingData td = compute_prune_tiling(md, dt, m, k, pruneAlongRow, pruneAlgInt);

    // Launch the prune kernel with tiling passed by value through
    // the kernel launch args block. No aclrtMalloc'd tilingGm buffer (D5: no
    // extra storage) and no aclrtSynchronizeStream (D4: async execution). The
    // ACL runtime owns the args block for the lifetime of the launched task,
    // so the host may return immediately after enqueueing the kernel.
    //
    // Batch 处理（matmulDescr 携带，对齐 cusparseLtSpMMAPrune 模式）：
    // 一次调用处理全 batch——host 逐 batch launch，第 b 次处理
    // d_in + b·batchStride·elemSize → d_out + b·batchStride·elemSize。
    // 同一 td 复用（per-batch m/k/pruneAlongRow/pruneAlg/chunkM/
    // usedCoreNum 相同，仅 GM 基址推进），同 stream 串行提交，返回前
    // 不同步（异步语义不变）。numBatches == 1 时循环退化为一次偏移 0
    // 的调用，与单 batch 行为完全一致。
    // batchOff 乘法安全性：validate_prune_batch 的 V3 已按 int64 受检查
    // 证明 (numBatches-1)·batchStride·elemSize + span·elemSize ≤
    // INT64_MAX，而 b ≤ numBatches-1、span ≥ 0，故
    // b·batchStride·elemSize ≤ totalBytes 不溢出（左结合中间值
    // b·batchStride ≤ totalElems 同理），无需在此重复受检查乘法。
    const int64_t elemSize =
        (dt == SPLT_DTYPE_FP32) ? SPLT_FP32_BYTES :
        (dt == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES : SPLT_HALF_BYTES;
    uint8_t* inBase = reinterpret_cast<uint8_t*>(const_cast<void*>(aIn));
    uint8_t* outBase = reinterpret_cast<uint8_t*>(aPruned);
    for (int32_t b = 0; b < numBatches; ++b) {
        const int64_t batchOff = static_cast<int64_t>(b) * batchStride * elemSize;
        splt_prune_kernel_launch(reinterpret_cast<GM_ADDR>(inBase + batchOff),
                                 reinterpret_cast<GM_ADDR>(outBase + batchOff), &td,
                                 td.dataType, static_cast<uint32_t>(td.usedCoreNum), stream);
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}
