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
//     V1  batchStride > 0 and >= the single-batch access span ((b) output
//         keeps the input layout, so in/out share one span: ColView
//         (k-1)·ld+m / RowView (m-1)·ld+k, same as the kernel SpltPruneImpl
//         access path) — input and output advance together by
//         batchStride 推进，逐 batch 基址不越界、batch 区不重叠；
//     V2  batchStride 为 16/elemSize 的倍数（⟺ batchStride × elemSize
//         为 16 字节倍数——elemSize ∈ {1,2,4} 均为 2 的幂，纯取模
//         等价判断，不计算乘积；配合 d_in/d_out 基址的 16B 校验，逐
//         batch 基址恒 16B 对齐）；
//     V3  (numBatches-1)×batchStride + span 与 ×elemSize 按 int64 防溢出
//         （最终防线：V2 放行的对齐巨大 stride 由本层拦截）。
// pruneAlongRow 与 kernel 路径分派（SpltPruneImpl）同源：pruneAlongRow==0
// ⟺ ColView（(N,COL)/(T,ROW)，物理 (k,m) 行主序 ld≥m）；非零 ⟺ RowView。
// ----------------------------------------------------------------------------
// prune_batch_span —— validate_prune_batch 的 V1 span 段（自该函数拆出，
// codecheck 超大函数项，语句与返回语义逐位等价）：
// V1: span (in elements, same formula for in/out — the output keeps the
// input layout). Derived from the kernel's actual access path (same
// source as SpltPruneImpl's RowView/ColView dispatch):
//   ColView (pruneAlongRow==0): physical addressing is (k,m) row-major
//     (SpltPruneLoadRowBlockChunk's rowStart*ld+colStart), single-batch
//     span (k-1)·ld+m;
//   RowView (pruneAlongRow!=0): physical addressing is (m,k) row-major
//     (SpltPruneAlongRowUnified's rowStart*ld), row length k,
//     single-batch span (m-1)·ld+k.
// rows/cols/ld are int64 descriptor fields with no upper-bound checks in
// the setter; the (rows/cols-1)·ld multiply-add chain is computed with
// checked arithmetic first — overflow means an illegal layout (a physical
// span beyond int64 cannot be addressed by the kernel). The setter
// guarantees rows/cols/ld > 0, so the multipliers are non-negative and
// the checked_mul_add preconditions hold. Overflow logs and returns false
// (span valid only on true).
static bool prune_batch_span(const aclsparseLtMatDescriptor* sparseMat,
                             int32_t m, int32_t k, int32_t pruneAlongRow,
                             int64_t& span)
{
    const bool isColView = (pruneAlongRow == 0);
    const bool spanOk = isColView
        ? checked_mul_add(k - 1, sparseMat->ld, m, span)
        : checked_mul_add(m - 1, sparseMat->ld, k, span);
    if (!spanOk) {
        OP_LOGE(kSparseLtLogTag,
                "Prune: physical span overflow (rows=%lld, cols=%lld, ld=%lld)",
                static_cast<long long>(sparseMat->rows),
                static_cast<long long>(sparseMat->cols),
                static_cast<long long>(sparseMat->ld));
    }
    return spanOk;
}

static aclsparseStatus_t validate_prune_batch(
    const aclsparseLtMatDescriptor* sparseMat, int32_t dt, int32_t m, int32_t k,
    int32_t pruneAlongRow,
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
    // V1: span (in elements, same formula for in/out) — 见 prune_batch_span。
    int64_t span = 0;
    if (!prune_batch_span(sparseMat, m, k, pruneAlongRow, span)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // stride < 0 is always illegal (the kernel advances per base + b*stride;
    // a negative stride walks backwards out of bounds). stride == 0 is
    // cross-batch broadcast (cuSPARSELt batch_stride=0 semantics: every batch
    // reads/writes batch 0, b*0 stays at the base) — the buffer only needs
    // single-batch capacity, so the V1 span check is skipped for this side
    // (V2 alignment 0%x==0 passes trivially; V3 has no overflow face with
    // (nb-1)*0+span — both degenerate naturally for the broadcast side).
    // out-of-place: repeated identical writes of the prune result (prune is
    // idempotent per batch); in-place: same. Equivalent to single-matrix
    // pruning.
    if (batchStride < 0) {
        OP_LOGE(kSparseLtLogTag,
                "Prune: batchStride=%lld < 0 (numBatches=%d)",
                static_cast<long long>(batchStride), numBatches);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (batchStride == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;  // broadcast: all batches use batch 0
    }
    if (batchStride < span) {
        OP_LOGE(kSparseLtLogTag,
                "Prune: batchStride=%lld < span=%lld (numBatches=%d)",
                static_cast<long long>(batchStride), static_cast<long long>(span), numBatches);
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
    constexpr int32_t TRANS_TILE = 16;
    const int64_t ubBytes = static_cast<int64_t>(get_ub_size());

    // UB budget (exact; chunkM is aligned to 32B/elemSize, so
    // alignedChunkM == chunkM):
    // After (b) layout preserving, the ColView path only occupies blkUB
    // (the transpose write-back transBuf and the STRIP absBuf/keepMaskBuf
    // are both removed — ColView grouping runs along physical columns and
    // is a scalar path anyway):
    //   blkUB = TRANS_TILE * chunkM * elemSize + TRANS_TILE * elemSize
    //           (row-tail prefetch safety margin, see
    //            SpltPruneTransInitBuffers).
    const int64_t blkPadding = static_cast<int64_t>(TRANS_TILE) * elemSize;
    if (blkPadding >= ubBytes) {
        return 0;
    }
    const int64_t available = ubBytes - blkPadding;

    // per-chunkM bytes: one row of elements per blkUB column.
    const int64_t perChunkBytes = static_cast<int64_t>(TRANS_TILE) * elemSize;

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

static aclsparseStatus_t check_ub_capacity(int32_t dt,
    int32_t m, int32_t k, int32_t pruneAlongRow, int32_t pruneAlg)
{
    const int64_t elemSize =
        (dt == SPLT_DTYPE_FP32) ? SPLT_FP32_BYTES :
        (dt == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES : SPLT_HALF_BYTES;
    const bool isTile = (pruneAlg == ACLSPARSELT_PRUNE_SPMMA_TILE);
    const int64_t SPLT_UB_BYTES = static_cast<int64_t>(get_ub_size());

    // ColView ((N,COL)/(T,ROW), pruneAlongRow==0) supports chunked loading:
    // when m exceeds UB it is chunked into chunkM column blocks. Only when a
    // single minimal tile/group cannot fit is NOT_SUPPORTED returned.
    // ((N,COL) has gone through chunkM chunking since this fix — large-k
    // (N,COL) moves from the NOT_SUPPORTED domain of the whole-row k full-UB
    // check into the supported domain, symmetric with (T,ROW).)
    const bool isColView = (pruneAlongRow == 0);
    if (isColView) {
        const int32_t chunkM = compute_prune_chunk_m(dt, m, pruneAlg);
        const int32_t TS = isTile
            ? ((dt == SPLT_DTYPE_FP32) ? SPLT_TILE_SIZE_FP32 : SPLT_TILE_SIZE_FP16)
            : ((dt == SPLT_DTYPE_FP32) ? 2 : 4);
        if (chunkM < TS) {
            OP_LOGE(kSparseLtLogTag, "Prune: ColView chunkM=%d < TS=%d (m=%d, dt=%d, alg=%d)",
                    chunkM, TS, m, dt, pruneAlg);
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    // RowView path: keeps the existing whole-row full-UB check (rowDim = k).
    const int32_t rowDim = k;
    const int64_t rowBytes = static_cast<int64_t>(rowDim) * elemSize;
    const int32_t tileSize = isTile
        ? ((dt == SPLT_DTYPE_FP32) ? SPLT_TILE_SIZE_FP32 : SPLT_TILE_SIZE_FP16)
        : 1;
    const int64_t alignedRowBytes = (rowBytes + (SPLT_UB_ALIGN_BYTES - 1)) /
                                    SPLT_UB_ALIGN_BYTES * SPLT_UB_ALIGN_BYTES;
    const int64_t dataBytes = alignedRowBytes * static_cast<int64_t>(tileSize);
    const int64_t vecBufBytes = 2 * static_cast<int64_t>(rowDim + 64) *
                                static_cast<int64_t>(sizeof(float));
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
    aclsparseStatus_t ubSt = check_ub_capacity(dt, m, k, pruneAlongRow, pruneAlg);
    if (ubSt != ACL_SPARSE_STATUS_SUCCESS) {
        return ubSt;
    }

    // Defensive explicit ld rejection (B3 root fix, replacing the old silent
    // clamp): physical row length = RowView k / ColView m. Valid descriptors
    // always satisfy it (ROW order ld >= cols and COL order ld >= rows exactly
    // cover both views' physical row lengths); only forged descriptors can
    // trigger this.
    const bool isColView = (pruneAlongRow == 0);
    const int32_t minLd = isColView ? m : k;
    const int64_t sparseLd = md_isSparseA(md) ? md->matA->ld : md->matB->ld;
    if (sparseLd < minLd) {
        OP_LOGE(kSparseLtLogTag, "Prune: ld=%lld < physical row length=%d (colView=%d)",
                static_cast<long long>(sparseLd), minLd, static_cast<int>(isColView));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
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
    // in-place (d_in == d_out) is supported for all combinations: (b) the
    // output keeps the input layout; RowView per row-block / ColView per
    // physical block does read-then-write with overlapping read/write
    // regions; the per-core block-grid partitioning is disjoint; and the
    // padding regions are skipped symmetrically by load srcStride / store
    // dstStride (the input padding is preserved as-is automatically) — no
    // in-place rejection is applied.
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
    // instead of row*k. When ld > the physical row length (padded matrix), this prevents
    // incorrect memory access from the 2nd row onward. md->matA->ld is int64_t;
    // truncate to int32_t (same width as m/k — matrices fit in int32_t per
    // existing code). The descriptor layer (StructuredDescriptorInit) already
    // rejects ld > INT32_MAX explicitly, so the narrowing never truncates
    // (the legal usage domain is guaranteed by upstream validation).
    // ld from the structured matrix: A-sparse -> matA->ld; B-sparse -> matB->ld.
    // The defensive ld lower-bound rejection lives in resolve_prune_buffers
    // (always satisfied by valid descriptors, B3).
    const int64_t sparseLd = md_isSparseA(md) ? md->matA->ld : md->matB->ld;
    td.ld = static_cast<int32_t>(sparseLd);
    td.pruneAlongRow = pruneAlongRow;
    td.pruneAlg = pruneAlg;
    // sparseTrans is kept for logging/debugging only ((b) dispatch looks only
    // at pruneAlongRow).
    const bool sparseTrans = md_isSparseA(md) ? md_transA(md) : md_transB(md);
    td.sparseTrans = sparseTrans ? 1 : 0;
    td.dataType = dt;
    td.coreNum = static_cast<int32_t>(get_cube_core_num());
    // Compute chunkM for the ColView path (pruneAlongRow==0). When m > chunkM,
    // the kernel splits the m-dimension into chunks. RowView sets chunkM = 0
    // (kernel ignores it). When m <= chunkM (single chunk), chunkM = m.
    td.chunkM = (pruneAlongRow == 0) ? compute_prune_chunk_m(dt, m, pruneAlg) : 0;
    // [TILE] TILE alongRow uses TS rows/block (vs 1 for STRIP alongRow).
    const int32_t tileSize = (pruneAlg == ACLSPARSELT_PRUNE_SPMMA_TILE)
        ? ((dt == SPLT_DTYPE_FP32) ? SPLT_TILE_SIZE_FP32 : SPLT_TILE_SIZE_FP16)
        : 1;
    // ColView physical row count = k (physical (k,m) row-major); RowView
    // physical row count = m.
    const int32_t pruneTiles = (pruneAlongRow != 0)
        ? ((m + tileSize - 1) / tileSize)
        : ((k + SPLT_PRUNE_ROW_TILE - 1) / SPLT_PRUNE_ROW_TILE);
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
    // pruneAlongRow 与 kernel 路径分派同源（validate_prune_params 的推导），
    // 供 V1 span 按 kernel 实际访问路径选取公式。
    int32_t numBatches = 1;
    int64_t batchStride = 0;
    st = validate_prune_batch(sparseMat, dt, m, k, pruneAlongRow,
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
