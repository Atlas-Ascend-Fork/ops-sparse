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
 * \file aclsparselt_internal.h
 * \brief sparseLt shared internal header: TilingData struct + constants
 *        (shared CXX/ASC), host-only helper functions (CXX only), and
 *        host-only descriptor internal structs (CXX only).
 * Included by both host.cpp (CXX) and kernel.cpp (ASC). The TilingData struct,
 * constants and splt_load_tiling above the __CCE_AICORE__ guard are compiled by
 * both; the host-only helpers and descriptor structs below the guard are skipped
 * by the ASC compiler (host-only, pull in acl/acl.h + cann_ops_sparse*.h).
 * Merged from the former single-folder header + the anonymous-namespace
 * helpers in the former host.cpp + splt_load_tiling from the former kernel.cpp so the
 * old single-folder layout can be split into shared/matmul/prune/alg_set_attr
 * sub-directories that all reuse this common internal header.
 */

#ifndef ACLSPARSELT_INTERNAL_H
#define ACLSPARSELT_INTERNAL_H


#include <cstdint>
#include <cfloat>
#include <algorithm>  // std::min (validSplitK computation in fill_splitk_tiling_dims)

#ifndef __gm__
#define __gm__
#endif

// Data type encoding for host-side kernel dispatch (matches SpmmDtype convention).
#define SPLT_DTYPE_INVALID (-1)
#define SPLT_DTYPE_FP32 0
#define SPLT_DTYPE_FP16 1
// BF16 / INT8 input dtypes; INT32 only used for outDataType field
// (INT8 input -> INT32 output path), never appears in dataType field.
#define SPLT_DTYPE_BF16 2
#define SPLT_DTYPE_INT8 3
#define SPLT_DTYPE_INT32 4

// Element byte sizes by dtype (used in workspace / UB capacity calculations).
#define SPLT_FP32_BYTES 4
#define SPLT_HALF_BYTES 2   // FP16 / BF16
#define SPLT_INT8_BYTES 1
// L1 budget (bytes) used for kL1 segment computation: kL1Size = SPLT_KL1_BUDGET_BYTES / elemSize.
#define SPLT_KL1_BUDGET_BYTES 512
// UB 32-byte alignment boundary (DataCopyPad requirement).
#define SPLT_UB_ALIGN_BYTES 32

// Workspace layout (all regions 64B aligned):
//   [ tilingOff .. aPrunedOff )  : AclsparseltTilingData
//   [ aPrunedOff .. tempOff )    : A_pruned ((b) A-sparse uses the sparse-side
//                                  physical span sparseSpan*elemSize — prune
//                                  output keeps the descriptor (order, ld)
//                                  layout including padding; non-A-sparse stays
//                                  m * k * sizeof(T), this region is unused)
//   [ tempOff .. end )           : temp_result (splitK * m * n * sizeof(float))
#define SPLT_WS_HEADER_BYTES 64
#define SPLT_WS_ALIGN 64

// Hardware constants (DAV-3510).
#define SPLT_CORE_NUM 32
// L0C cube edge is always 16 on DAV-3510.
#define SPLT_L0C_C0 16
// L0A/L0B buffer size (64KB each on DAV-3510). Used for L0 double-buffer offset.
#define SPLT_L0A_SIZE (64 * 1024)
// Half L0A for ping-pong double buffer (32KB per slot).
#define SPLT_HALF_L0_SIZE (SPLT_L0A_SIZE / 2)
// GM output (temp, FP32) C0 must match L0C C0 (=16) for correct Fixpipe
// ND output conversion. Previously used 8 (32/sizeof(float)), but the
// Fixpipe converts L0C NZ fractals (C0=16) to GM NDExt — a C0 mismatch
// causes incorrect fractal splitting that scrambles ~0.005% of elements.
#define SPLT_GM_FLOAT_C0 16

#define SPLT_PRUNE_ROW_TILE 16
// TILE mode constants.
// Tile size: FP16=4 (4x4 tile), FP32=2 (2x2 tile).
#define SPLT_TILE_SIZE_FP16 4
#define SPLT_TILE_SIZE_FP32 2
// Number of valid configurations per tile.
#define SPLT_TILE_CONFIGS_4X4 90   // 4x4, 2-per-row, 2-per-col
#define SPLT_TILE_CONFIGS_2X2 2    // 2x2, 1-per-row, 1-per-col
// Prune algorithm constants (mirror aclsparseLtPruneAlg_t enum for kernel use).
// Defined here because cann_ops_sparseLt.h is host-only (not ASC-compilable).
#define SPLT_PRUNE_ALG_TILE 0
#define SPLT_PRUNE_ALG_STRIP 1
// unitFlag values for Mmad (matching SDK blaze common_utils.h:67-68).
// FINAL_ACCUMULATION = 3: last K iteration, triggers M->FIX sync internally.
// NON_FINAL_ACCUMULATION = 2: intermediate K iterations, tells hardware more
// accumulations are coming. Using 0 (as before) is undefined behavior and
// causes occasional pipeline sync failures producing outlier elements.
#define SPLT_FINAL_ACCUMULATION 3
#define SPLT_NON_FINAL_ACCUMULATION 2

// TilingData read by every kernel from workspace[tilingOff].
// POD struct — copied H2D by host before each kernel launch.
typedef struct AclsparseltTilingData {
    // matrix dimensions
    int32_t m;
    int32_t n;
    int32_t k;
    // leading dimension of matrix A (may be > k when padded).
    // Prune kernel uses ld for row offsets (row*ld) instead of row*k to handle
    // padded matrices correctly. After the A_pruned layout-preserving change,
    // matmul addresses the sparse side via ldA/ldB; td.ld is still consumed
    // only by prune (the matmul path ignores it).
    int32_t ld;
    // cube tiling parameters
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    // K-dimension L1 tile size for the outer kL1 loop.
    // The K-loop is split into two layers: outer kL1 (GM→L1, kL1Size chunk)
    // and inner kL0 (L1→L0, baseK chunk). kL1Size = 512/sizeof(T) aligned to
    // baseK (256 for FP16, 128 for FP32). L1 capacity constraint:
    // 2*(baseM*kL1Size + kL1Size*baseN)*sizeof(T) <= 512KB (DAV-3510 L1 total).
    int32_t kL1Size;
    // split-k
    int32_t splitK;
    int32_t kSegLen;        // ceil(k / splitK) aligned up to baseK
    // algorithm config
    int32_t algConfigId;    // 0 / 1
    // data type (SPLT_DTYPE_*)
    int32_t dataType;
    // output dtype (SPLT_DTYPE_*), drives INT8 epilogue dispatch.
    //   FP32/FP16/BF16: outDataType == dataType.
    //   INT8: outDataType == SPLT_DTYPE_INT8(3) or SPLT_DTYPE_INT32(4).
    int32_t outDataType;
    int32_t pruneAlongRow;
    // pruneAlg: 0=TILE (ACLSPARSELT_PRUNE_SPMMA_TILE), 1=STRIP (ACLSPARSELT_PRUNE_SPMMA_STRIP).
    // Consumed by the prune kernel to dispatch between TILE and STRIP paths.
    // Default 0 (TILE) when zero-initialized; prune host explicitly sets this.
    int32_t pruneAlg;
    // sparseTrans/transB flags (consumed by cube): PlanInit snapshot = the op
    // component base value (transA/transB); at execution time
    // splt_apply_gm_layout flips on top for order==COL -> the final value =
    // isColView(side). 0 -> NDExt (row-major view); 1 -> DNExt (column-major
    // view, CopyGM2L1 automatically selects the transposing fractal
    // conversion). The prune kernel no longer consumes sparseTrans (dispatch
    // looks only at pruneAlongRow; the field is kept for logging/debugging).
    int32_t sparseTrans;
    int32_t transB;
    // scalars (filled at Matmul time; prune/compress ignore them)
    float alpha;
    float beta;
    // vector scaling flags (0=scalar, 1=per-row device pointer)
    int32_t alphaVectorScaling;
    int32_t betaVectorScaling;
    // device pointers to alpha/beta vectors (float[M]), valid when scaling==1
    uint64_t alphaDevPtr;
    uint64_t betaDevPtr;
    // multi-core
    int32_t coreNum;        // 32
    int32_t usedCoreNum;
    // workspace byte offsets
    // Use int64_t for offsets to prevent overflow on large
    // workspaces (m*k*sizeof(T) can exceed 2GB for large FP32 shapes).
    int64_t tilingOffset;
    int64_t aPrunedOffset;
    int64_t tempResultOffset;
    // —— bias + activation epilogue 字段（追加在末尾，向后兼容）——
    uint64_t biasDevPtr;       // bias device 指针，0=无 bias
    int64_t  biasStride;       // batch 间 bias 步长，0=广播
    int32_t  activationType;   // 0=无, 1=ReLU, 2=GeLU（统一编码，替代分离的 relu/gelu）
    float    reluUpperBound;   // 默认 FLT_MAX
    float    reluThreshold;    // 默认 0.0f
    float    geluScaling;      // 默认 1.0f
    int32_t  biasChunkMode;    // 0=全量加载, 1=per-chunk 降级（UB 超限时）
    // —— batch 字段（追加在末尾，向后兼容）——
    int32_t  numBatches;       // batch 数量，默认 1
    int64_t  batchStrideA;     // A 的 batch 步长（元素数）
    int64_t  batchStrideB;     // B 的 batch 步长
    int64_t  batchStrideC;     // C 的 batch 步长
    int64_t  batchStrideD;     // D 的 batch 步长
    // Prune trans path chunk size (columns per chunk).
    // Computed by host based on UB capacity. When m > chunkM, the trans path
    // splits the m-dimension into chunks of chunkM columns each. When m <= chunkM,
    // single chunk (= m), behavior unchanged from original. chunkM is aligned to
    // 32B/elemSize so that alignedChunkM == chunkM (exact UB row stride).
    // 0 = not set (non-trans paths or default); kernel treats 0 as "use m".
    int32_t  chunkM;
    // C 矩阵的实际 dtype（SPLT_DTYPE_*）。epilogue 读取 C 时按此类型
    // 实例化（CType 模板贯通）：A/B 为 INT8 时 C 可为 INT8/INT32/FP16/BF16
    // （描述符合法性矩阵的超集行），非 INT8 时 C 与 A/B 同型（恒等于
    // dataType 字段）。默认 0（SPLT_DTYPE_FP32）——仅当 dataType != 0 且
    // cDataType 与 dataType 不同时才有意义；host 侧 fill_tiling_dims
    // 恒填充，kernel 侧经分派 switch 消费。
    int32_t  cDataType;
    // —— X1 布局接入字段（追加在末尾，向后兼容）——
    // C/D 的物理 leading dimension（元素单位）与 order。kernel 侧 C 读 /
    // D 写经 SpltCdOffset 统一寻址：ROW 序行步长 = ld（ld == n 时与旧
    // row*n + col 寻址逐位等价，零回归锚点）；COL 序为转置视图寻址
    // （元素 (i,j) 位于 i + j*ld）。C/D order 一致由 MatmulDescriptorInit
    // 的 ValidateCDConsistency 保证。
    int32_t  ldC;           // C 物理 ld（COL 时为列主序 ld）
    int32_t  ldD;           // D 物理 ld
    int32_t  cOrderIsCol;   // C/D order == ACL_SPARSE_ORDER_COL ? 1 : 0
    // —— cube GM 布局直通 ld（ld padding 直读接入）——
    // cube A/B 输入的物理行距（元素单位），直通 GM 布局 NDExt/DNExt 的
    // 行距参数（NDExt 第二参数 / DNExt 第一参数），MTE2 搬运按行距跳过
    // 行尾 padding，带 ld 的输入无需任何预处理：
    //   ldA / ldB：两侧恒直读描述符 ld（(b) 布局保持——稀疏侧 A_pruned /
    //        B_pruned 为 prune 的描述符布局输出（含 padding），dense 侧为
    //        描述符布局输入，两者同规则）。COL 的转置视图翻转由执行期
    //        splt_apply_gm_layout 翻转 cube 转置标志承载（最终标志 =
    //        isColView(该侧)，ld 口径不变；ldA/ldB 为 PlanInit 快照，
    //        与 ldC/ldD 同口径）。
    // 紧凑场景（ld == 物理列数）与既有紧凑布局参数逐位一致（零回归锚点）。
    // ld 为 int64 描述符字段，此处收窄为 int32——描述符层
    // （Dense/StructuredDescriptorInit）已显式拒绝 ld > INT32_MAX，收窄无
    // 截断（与 ldC/ldD 同口径）。
    int32_t  ldA;
    int32_t  ldB;
    // —— B4 splitK 空段治理（末尾追加，向后兼容）——
    // 有效段数 = 起点落在 [0,k) 内的段数 = ceil(k/kSegLen)（kSeg 对齐 baseK
    // 使其 ≤ splitK；effectiveSplitK==1 时 kSeg 已 clamp 到 k，恒为 1）。
    // epilogue 归约上界（temp [validSplitK, splitK) 空段不读）；temp memset
    // 已完全去除，依赖「每个有效段的每个 (m,n) 元素必被 cube 写入」不变量
    //（证明链见 matmul_host memset 删除处注释）。
    int32_t  validSplitK;
} AclsparseltTilingData;


// ============================================================================
// Device-side tiling loader (ASC only).
//
// Reads the TilingData POD struct from GM workspace[tilingOff] into a local
// copy. Marked __aicore__ inline so it is only usable from kernel code; the
// host compiler (CXX) skips this section via the __CCE_AICORE__ guard below.
// 逐字段拷贝拆为两段（原单函数 nbnc 51 > 50）：
//   splt_load_tiling_core_fields          —— 维度 / cube tiling / splitK /
//                                            dtype / 算法与布局标志 + X1
//                                            布局接入字段（chunkM/cDataType/
//                                            ldC/ldD/cOrderIsCol/ldA/ldB）
//   splt_load_tiling_scalar_batch_fields  —— 标量缩放 / 核数 / workspace
//                                            偏移 / bias+activation / batch
//                                            字段
// 各字段为独立 POD 字段拷贝、无顺序依赖。
// ============================================================================
#ifdef __CCE_AICORE__
__aicore__ inline void splt_load_tiling_core_fields(
    const __gm__ AclsparseltTilingData* gmTd, AclsparseltTilingData& td)
{
    td.m = gmTd->m;
    td.n = gmTd->n;
    td.k = gmTd->k;
    td.ld = gmTd->ld;  // load leading dimension for prune kernel
    td.baseM = gmTd->baseM;
    td.baseN = gmTd->baseN;
    td.baseK = gmTd->baseK;
    td.kL1Size = gmTd->kL1Size;
    td.splitK = gmTd->splitK;
    td.kSegLen = gmTd->kSegLen;
    td.algConfigId = gmTd->algConfigId;
    td.dataType = gmTd->dataType;
    td.outDataType = gmTd->outDataType;  // output dtype for epilogue dispatch
    td.pruneAlongRow = gmTd->pruneAlongRow;
    td.pruneAlg = gmTd->pruneAlg;
    td.sparseTrans = gmTd->sparseTrans;
    td.transB = gmTd->transB;
    td.chunkM = gmTd->chunkM;
    td.cDataType = gmTd->cDataType;
    // X1 布局接入字段
    td.ldC = gmTd->ldC;
    td.ldD = gmTd->ldD;
    td.cOrderIsCol = gmTd->cOrderIsCol;
    // cube GM 布局直通 ld
    td.ldA = gmTd->ldA;
    td.ldB = gmTd->ldB;
    td.validSplitK = gmTd->validSplitK;
}

__aicore__ inline void splt_load_tiling_scalar_batch_fields(
    const __gm__ AclsparseltTilingData* gmTd, AclsparseltTilingData& td)
{
    td.alpha = gmTd->alpha;
    td.beta = gmTd->beta;
    td.alphaVectorScaling = gmTd->alphaVectorScaling;
    td.betaVectorScaling = gmTd->betaVectorScaling;
    td.alphaDevPtr = gmTd->alphaDevPtr;
    td.betaDevPtr = gmTd->betaDevPtr;
    td.coreNum = gmTd->coreNum;
    td.usedCoreNum = gmTd->usedCoreNum;
    td.tilingOffset = gmTd->tilingOffset;
    td.aPrunedOffset = gmTd->aPrunedOffset;
    td.tempResultOffset = gmTd->tempResultOffset;
    // bias + activation 字段
    td.biasDevPtr = gmTd->biasDevPtr;
    td.biasStride = gmTd->biasStride;
    td.activationType = gmTd->activationType;
    td.reluUpperBound = gmTd->reluUpperBound;
    td.reluThreshold = gmTd->reluThreshold;
    td.geluScaling = gmTd->geluScaling;
    td.biasChunkMode = gmTd->biasChunkMode;
    // batch 字段
    td.numBatches = gmTd->numBatches;
    td.batchStrideA = gmTd->batchStrideA;
    td.batchStrideB = gmTd->batchStrideB;
    td.batchStrideC = gmTd->batchStrideC;
    td.batchStrideD = gmTd->batchStrideD;
}

__aicore__ inline AclsparseltTilingData splt_load_tiling(GM_ADDR tilingGm)
{
    __gm__ AclsparseltTilingData* gmTd = reinterpret_cast<__gm__ AclsparseltTilingData*>(tilingGm);
    AclsparseltTilingData td;
    splt_load_tiling_core_fields(gmTd, td);
    splt_load_tiling_scalar_batch_fields(gmTd, td);
    return td;
}
#endif // __CCE_AICORE__

// ============================================================================
// Host-side descriptor internal structs + helpers (not exposed publicly).
//
// Mirrors cuSPARSELt opaque-pointer pattern: external headers
// (cann_ops_sparseLt.h) only forward-declare the structs; internal layout lives
// here and is visible only to the library impl (host.cpp).
//
// Guarded by __CCE_AICORE__: kernel.cpp (ASC) includes this header for the
// TilingData struct above; the descriptor section is host-only (CXX) and must
// not be compiled by the ASC compiler (it pulls in acl/acl.h +
// cann_ops_sparse*.h which are not available / not compatible on the device
// side).
// ============================================================================
#ifndef __CCE_AICORE__
#include "aclsparselt_mat_descriptor_internal.h"
#include "aclsparselt_matmul_descriptor_internal.h"
#include "aclsparselt_matmul_alg_selection_internal.h"
#include "aclsparselt_matmul_plan_internal.h"
#include <acl/acl.h>
#include "cann_ops_sparse.h"
#include "cann_ops_sparseLt.h"
#include "log/log.h"

// Centralized log tag for all OP_LOGE calls in the sparseLt
// library. Previously the string literal "aclsparseLt" was hardcoded in every
// OP_LOGE call site (79 occurrences across 5 files). Defining it once here as
// a compile-time constant ensures consistency and makes future tag changes a
// single-line edit.
constexpr const char* kSparseLtLogTag = "aclsparseLt";

// Named constant for ALG_CONFIG_MAX_ID, shared between
// alg_set_attribute_host.cpp (set/get) and aclsparselt_host.cpp (validate).
// algConfigId 0 and 1 are valid (2 configs total). Platform-driven; future
// versions may expand — change this single constant instead of magic 0/1.
constexpr int32_t kAlgConfigMaxId = 2;

// aclsparseLtMatDescriptor / aclsparseLtMatmulDescriptor structs are defined
// in upstream's common/aclsparselt_mat_descriptor_internal.h and
// common/aclsparselt_matmul_descriptor_internal.h (included above). This
// avoids duplicate definitions and ensures binary compatibility with
// upstream's DenseDescriptorInit / StructuredDescriptorInit / MatmulDescriptorInit.

// ============================================================================
// Inline helper functions — compute PR-specific derived fields from the
// upstream-compatible matmul descriptor. These replace the former cached
// fields (transA, transB, isSparseA, hasStructuredSparsity, m, n, k) that
// were removed when the struct was aligned to upstream's layout.
// ============================================================================
inline bool md_transA(const aclsparseLtMatmulDescriptor* md) {
    return md->opA == ACL_SPARSE_OP_TRANSPOSE;
}
inline bool md_transB(const aclsparseLtMatmulDescriptor* md) {
    return md->opB == ACL_SPARSE_OP_TRANSPOSE;
}
inline bool md_isSparseA(const aclsparseLtMatmulDescriptor* md) {
    return md->matA->isStructured;
}
inline bool md_hasStructuredSparsity(const aclsparseLtMatmulDescriptor* md) {
    return md->matA->isStructured || md->matB->isStructured;
}
inline int32_t md_m(const aclsparseLtMatmulDescriptor* md) {
    return md_transA(md) ? static_cast<int32_t>(md->matA->cols)
                         : static_cast<int32_t>(md->matA->rows);
}
inline int32_t md_k(const aclsparseLtMatmulDescriptor* md) {
    return md_transA(md) ? static_cast<int32_t>(md->matA->rows)
                         : static_cast<int32_t>(md->matA->cols);
}
inline int32_t md_n(const aclsparseLtMatmulDescriptor* md) {
    return md_transB(md) ? static_cast<int32_t>(md->matB->rows)
                         : static_cast<int32_t>(md->matB->cols);
}

// 16-byte alignment predicate for all matmul data pointers, shared by
// aclsparseLtMatmul / aclsparseLtMatmulSearch validation (each host keeps its
// own error-logging wrapper; nullptr matrices are skipped — matA=nullptr has
// its own A-sparse fallback validation in MatmulSearch). Condition order
// (workspace → matA → matB → matC → matD) matches the former inline lambdas.
inline bool splt_data_ptrs_aligned16(const void* workspace, const void* matA,
                                     const void* matB, const void* matC,
                                     const void* matD)
{
    auto aligned16 = [](const void* p) -> bool {
        return (reinterpret_cast<uintptr_t>(p) % 16) == 0;
    };
    return aligned16(workspace) &&
           (matA == nullptr || aligned16(matA)) &&
           (matB == nullptr || aligned16(matB)) &&
           (matC == nullptr || aligned16(matC)) &&
           (matD == nullptr || aligned16(matD));
}

// ============================================================================
// Vector scaling flag encoding into matD->alignment high bits.
//
// alphaVectorScaling and betaVectorScaling are set via aclsparseLtMatmulDescSetAttribute
// (which receives the matmul descriptor, not the algSelection). The upstream-compatible
// matmul descriptor struct has no field for these, so we encode them into the high 2 bits
// of matD->alignment (uint32_t).
//
// Rationale: alignment is validated as a non-zero multiple of 16 (low 4 bits = 0) and
// practical values are small (16..4096), so bits 30-31 are always free. Encoding:
//   bit 31 (0x80000000): alphaVectorScaling
//   bit 30 (0x40000000): betaVectorScaling
//
// This replaces the former global side-map (get_matmul_desc_ext_map) which was keyed by
// raw descriptor pointer and suffered from address-reuse pollution: after Destroy+realloc
// the new descriptor could inherit the previous occupant's flags. With alignment encoding,
// FillMatDescriptor resets alignment to the user-provided value (clearing high bits) on
// every DescriptorInit, so stale flags are naturally cleared.
//
// NOTE: This is thread-unsafe (single-threaded C API library, like cuSPARSELt).
// ============================================================================
inline void encode_scaling_flags(aclsparseLtMatmulDescriptor* md, bool alphaVec, bool betaVec)
{
    if (md == nullptr || md->matD == nullptr) { return; }
    uint32_t& a = md->matD->alignment;
    a &= 0x3FFFFFFFu;  // clear top 2 bits
    if (alphaVec) { a |= 0x80000000u; }
    if (betaVec)  { a |= 0x40000000u; }
}

inline bool decode_alpha_scaling(const aclsparseLtMatmulDescriptor* md)
{
    return (md != nullptr && md->matD != nullptr) && ((md->matD->alignment & 0x80000000u) != 0u);
}

inline bool decode_beta_scaling(const aclsparseLtMatmulDescriptor* md)
{
    return (md != nullptr && md->matD != nullptr) && ((md->matD->alignment & 0x40000000u) != 0u);
}

// Safe handle -> internal struct casts. The external API exposes const
// descriptor types (aclsparseLtConstMatmulDescriptor_t etc.); internal code
// needs mutable access. to_xxx_internal accepts the const type and
// performs the const_cast centrally, so business code does not scatter
// const_cast calls.
inline aclsparseLtMatDescriptor_t to_mat_internal(aclsparseLtConstMatDescriptor_t d) {
    return const_cast<aclsparseLtMatDescriptor_t>(d);
}
inline aclsparseLtMatmulDescriptor_t to_matmul_internal(aclsparseLtConstMatmulDescriptor_t d) {
    return const_cast<aclsparseLtMatmulDescriptor_t>(d);
}

// ============================================================================
// Host-only helper functions (formerly anonymous-namespace helpers in the
// former host.cpp). Declared inline/static to avoid ODR issues
// when multiple host TUs include this header.
// ============================================================================

inline int32_t dtype_from_acl(aclDataType t)
{
    if (t == ACL_FLOAT) { return SPLT_DTYPE_FP32; }
    if (t == ACL_FLOAT16) { return SPLT_DTYPE_FP16; }
    // BF16 / INT8 input dtypes; INT32 only used for matD output type (INT8 path).
    if (t == ACL_BF16) { return SPLT_DTYPE_BF16; }
    if (t == ACL_INT8) { return SPLT_DTYPE_INT8; }
    if (t == ACL_INT32) { return SPLT_DTYPE_INT32; }
    return SPLT_DTYPE_INVALID;
}

inline int64_t align_up(int64_t v, int64_t a)
{
    if (a == 0) {
        return v;
    }
    return ((v + a - 1) / a) * a;
}

// checked_mul_add — int64 非负乘加溢出检查（a/b/c 均非负）。溢出返回
// false，否则写入 result = a*b + c。prune/matmul host 的 batch 布局
// V1/V3 校验共用（span / totalBytes 受检查计算）。
inline bool checked_mul_add(int64_t a, int64_t b, int64_t c, int64_t& result)
{
    if (b != 0 && a > (INT64_MAX - c) / b) {
        return false;
    }
    result = a * b + c;
    return true;
}

// Resolve cube tiling (baseM, baseN, baseK) from dtype + algConfigId.
struct CubeTiling { int32_t baseM; int32_t baseN; int32_t baseK; };
inline CubeTiling resolve_cube_tiling(int32_t dataType, int32_t algConfigId)
{
    if (dataType == SPLT_DTYPE_FP32) {
        if (algConfigId == 1) { return {32, 128, 16}; }
        // baseK=64: fewer K-loop iterations reduce MTE2→MTE1 sync overhead
        // and improve L1 double-buffer utilization. Validated by the full
        // ltmatmul test suite (algConfigId 0/1 × all dtype paths).
        return {64, 64, 64};
    }
    // BF16 reuses FP16 tiling (sizeof=2, C0=16).
    if (dataType == SPLT_DTYPE_BF16) {
        if (algConfigId == 1) { return {64, 256, 32}; }
        return {128, 128, 64};
    }
    // INT8 tiling (C0=32, sizeof=1; L1 usage verified).
    if (dataType == SPLT_DTYPE_INT8) {
        if (algConfigId == 1) { return {64, 256, 32}; }
        return {128, 128, 64};
    }
    // FP16
    if (algConfigId == 1) { return {64, 256, 32}; }
    return {128, 128, 64};
}

// get_cube_core_num is the only consumer of
// tiling/platform/platform_ascendc.h. Moving the implementation to
// aclsparselt_host.cpp (which already includes the platform header) keeps
// this header lightweight — other host TUs (alg_set_attribute_host.cpp,
// prune_host.cpp) that include this header no longer transitively pull in
// the heavy platform_ascendc.h. Only the declaration lives here.
uint32_t get_cube_core_num();

// get_ub_size queries the platform UB capacity via
// PlatformAscendC::GetCoreMemSize(CoreMemType::UB). Implemented in
// aclsparselt_host.cpp (which includes platform_ascendc.h). Returns the
// per-core Unified Buffer size in bytes, or a hardcoded fallback if the
// platform API is unavailable.
uint64_t get_ub_size();

// Compute workspace byte offsets.
struct WsLayout {
    int64_t tilingOff;
    int64_t aPrunedOff;
    int64_t tempOff;
    int64_t totalBytes;
};

// (b) sparse-side physical span (in elements): order==ROW -> rows*ld;
// COL -> cols*ld.
// Verified over the four combinations (A-sparse): (N,ROW) m*ld / (T,COL)
// m*ld / (N,COL) k*ld / (T,ROW) k*ld — i.e. the full-rectangle buffer
// capacity of the descriptor physical matrix (rows, cols, ld) (>= the
// kernel access span (maxDim-1)*ld+minDim). Once prune outputs preserve
// the input layout, the buffer capacity of A_pruned/B_pruned is exactly
// this span (the header's d_out contract: "same shape and ld as d_in").
// Dimensions are in the int32 domain per the md_* caliber, so the product
// is <= 2^62 with no overflow surface.
inline int64_t splt_sparse_span_elems(const aclsparseLtMatDescriptor* sparse)
{
    return (sparse->order == ACL_SPARSE_ORDER_ROW)
        ? sparse->rows * sparse->ld : sparse->cols * sparse->ld;
}

// (b) workspace A_pruned fallback-region capacity (in elements): under
// A-sparse this region carries the prune output (descriptor layout with
// padding) -> it takes the sparse-side physical span; under non-A-sparse
// (B-sparse / dense x dense) the region is never consumed (matA is always
// passed explicitly), keeping the existing m*k capacity (zero disturbance
// to the workspace size).
inline int64_t splt_apruned_span_elems(const aclsparseLtMatmulDescriptor* md,
                                       int32_t m, int32_t k)
{
    return md_isSparseA(md)
        ? splt_sparse_span_elems(md->matA)
        : static_cast<int64_t>(m) * static_cast<int64_t>(k);
}

inline WsLayout compute_ws_layout(int32_t m, int32_t n, int32_t k, int32_t splitK,
                                  int32_t dataType, int32_t numBatches,
                                  int64_t aPrunedSpanElems)
{
    // four-branch elemSize (FP32=4, INT8=1, INT32=4, FP16/BF16=2).
    const int64_t elemSize =
        (dataType == SPLT_DTYPE_FP32) ? SPLT_FP32_BYTES :
        (dataType == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES :
        (dataType == SPLT_DTYPE_INT32) ? SPLT_FP32_BYTES : SPLT_HALF_BYTES;  // FP16/BF16
    // numBatches 至少为 1（防御性，fill_tiling_dims 保证 >= 1）
    const int64_t batches = (numBatches > 0) ? static_cast<int64_t>(numBatches) : 1;
    WsLayout w;
    w.tilingOff = SPLT_WS_HEADER_BYTES;
    w.aPrunedOff = align_up(w.tilingOff + sizeof(AclsparseltTilingData), SPLT_WS_ALIGN);
    // (b) A_pruned 区容量 = aPrunedSpanElems（A-sparse 为稀疏侧物理跨度——
    // prune 输出保持描述符布局（含 padding），matA=null 回退时用户须先以
    // d_out=该区调用 SpMMAPrune 填充描述符布局；非 A-sparse 维持 m*k）。
    w.tempOff = align_up(w.aPrunedOff + aPrunedSpanElems * elemSize, SPLT_WS_ALIGN);
    // When splitK==1, the fused kernel path (SpltFusedMatmulCubeImpl +
    // SpltFusedEpilogueImpl) writes directly L0C->UB->GM, bypassing the temp
    // buffer entirely. Skip the temp_result allocation (splitK*m*n*4 bytes) to
    // avoid a large block of unused workspace memory. tempOff is still computed
    // (points past A_pruned) so td.tempResultOffset remains a valid offset, but
    // totalBytes stops at tempOff — the host never allocates or memsets temp.
    // The fused path never reads td.tempResultOffset, so this is safe.
    if (splitK == 1) {
        w.totalBytes = w.tempOff;
    } else {
        // temp buffer 须包含所有 batch 的部分和
        w.totalBytes = align_up(w.tempOff + batches * static_cast<int64_t>(splitK) * static_cast<int64_t>(m) *
                                static_cast<int64_t>(n) * SPLT_FP32_BYTES, SPLT_WS_ALIGN);
    }
    // A_pruned buffer is always allocated here because compute_ws_layout
    // runs at tiling compute time, before the caller invokes aclsparseLtMatmul (where
    // matA is provided). When the Matmul caller passes a non-null matA (already
    // pruned A, e.g. caller's d_out from SpMMAPrune), the host routes matA
    // directly to the kernel and the workspace A_pruned region is unused. We
    // cannot avoid this allocation at tiling compute time since matA availability is
    // unknown until Matmul. The wasted memory is m*k*sizeof(T), modest relative
    // to the temp buffer (splitK*m*n*4) that is already saved by D7 when
    // splitK==1.
    return w;
}

// Copy host TilingData to device workspace[tilingOff].
inline aclsparseStatus_t push_tiling_to_device(void* workspace, int64_t tilingOff,
                                               const AclsparseltTilingData& td)
{
    if (workspace == nullptr) {
        OP_LOGE(kSparseLtLogTag, "workspace is null when pushing tiling");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    void* devTiling = static_cast<uint8_t*>(workspace) + tilingOff;
    aclError ret = aclrtMemcpy(devTiling, sizeof(td), &td, sizeof(td),
                               ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(kSparseLtLogTag, "aclrtMemcpy tiling failed, aclErr=%d", (int)ret);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// 超大头文件整改：tiling 填充链（fill_tiling_dims 族 +
// query_alg_attributes + compute_bias_chunk_mode）拆至同目录
// aclsparselt_tiling_fill.h（按主题分头；拆分后另有 4 处语义变更，
// 清单见 tiling_fill.h 头注释）。
// 此处转发 include：既有引用点（include 本头的全部 TU）零改动、
// 零编译破坏；两头的 include guard 解开相互引用（本头先展开完毕，
// tiling_fill.h 再引用本头时空展开）。
// ============================================================================
#include "aclsparselt_tiling_fill.h"

#endif // __CCE_AICORE__

#endif // ACLSPARSELT_INTERNAL_H
