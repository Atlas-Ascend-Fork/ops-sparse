/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file aclsparselt_tiling_fill.h
 * \brief sparseLt host 侧 tiling 填充链 helper
 *        （超大头文件整改：原 shared/aclsparselt_internal.h
 *        nbnc 545 > 500，按主题拆分——本头承载 PlanInit/Search 共用的
 *        fill_tiling_dims 族（cube/splitK/sparse 维度填充、alg 属性查询、
 *        biasChunkMode 降级）；internal.h 保留 TilingData/常量/基础
 *        helper 并在 host 段尾部 include 本头转发——既有引用点
 *        （include internal.h 的全部 TU）零改动零编译破坏）。
 *        相对拆分前版本的语义变更（非逐字一致，回溯/检视以本清单为准）：
 *        ① compute_bias_chunk_mode 触发条件由 biasEnabled &&
 *          needVecScale 放宽为 (biasEnabled || needVecScale)——vec-only /
 *          bias-only 大 m 场景同样纳入 UB 预算检查与降级；
 *        ② compute_bias_chunk_mode 预算新增 colWideBytes 项（COL 布局
 *          C/D 中转缓冲），并为行级小缓冲逐项精确化；
 *        ③ fill_sparse_tiling_dims 新增 ldA/ldB 填充（dense 侧 ld 直通，
 *          见字段注释）；
 *        ④ fill_tiling_dims 新增 cDataType/ldC/ldD/cOrderIsCol 字段
 *          （C/D 布局贯通）。
 * Host-only：依赖 aclsparselt_internal.h 的 host 段（描述符结构、
 * md_* helpers、CubeTiling、kSparseLtLogTag、get_ub_size 等），经
 * 顶部 include 获取；单独 include 本头同样自洽（幂等 guard 解开与
 * internal.h 的相互引用）。
 */

#ifndef ACLSPARSELT_TILING_FILL_H
#define ACLSPARSELT_TILING_FILL_H

#include "aclsparselt_internal.h"

#ifndef __CCE_AICORE__

inline void fill_cube_tiling_dims(AclsparseltTilingData& td,
                                  const aclsparseLtMatmulAlgSelection* asel,
                                  const CubeTiling& ct, int32_t dt)
{
    const auto* md = asel->matmulDescr;
    td.m = md_m(md);
    td.n = md_n(md);
    td.k = md_k(md);
    // matmul path: A_pruned is stored contiguously (ld == k).
    // Prune path sets td.ld = matA->ld separately in prune_host.cpp.
    td.ld = md_k(md);
    td.baseM = ct.baseM;
    td.baseN = ct.baseN;
    td.baseK = ct.baseK;
    // kL1Size: SPLT_KL1_BUDGET_BYTES/elemSize, aligned up to baseK.
    // four-branch elemSize (FP32=4, INT8=1, FP16/BF16=2).
    const int32_t elemSize =
        (dt == SPLT_DTYPE_FP32) ? SPLT_FP32_BYTES :
        (dt == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES : SPLT_HALF_BYTES;  // FP16/BF16
    int32_t kL1Sz = SPLT_KL1_BUDGET_BYTES / elemSize;
    kL1Sz = ((kL1Sz + ct.baseK - 1) / ct.baseK) * ct.baseK;
    td.kL1Size = kL1Sz;
}

// splitK / splitKMode are passed in (queried via AlgGetAttribute by the
// caller fill_tiling_dims) rather than read from asel directly, so that the
// tiling path consumes alg attributes exclusively through the public API.
inline void fill_splitk_tiling_dims(AclsparseltTilingData& td,
                                     const aclsparseLtMatmulAlgSelection* asel,
                                     const CubeTiling& ct,
                                     int32_t splitK, int32_t splitKMode)
{
    const auto* md = asel->matmulDescr;
    const int32_t k = md_k(md);
    // [SPLIT_K_MODE] ONE_KERNEL absorbs splitK as 1 (fused GEMM+reduction);
    // TWO_KERNELS preserves user splitK (matmul+epilogue two-kernel path).
    const int32_t effectiveSplitK =
        (splitKMode == ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL) ? 1 : splitK;
    // Warn when ONE_KERNEL mode silently ignores splitK > 1.
    if (splitKMode == ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL && splitK > 1) {
        OP_LOGI(kSparseLtLogTag, "Tiling: splitKMode=ONE_KERNEL, splitK=%d ignored (effectiveSplitK=1)",
                splitK);
    }
    td.splitK = effectiveSplitK;
    // kSegLen = ceil(k / effectiveSplitK), aligned up to baseK for cube efficiency.
    // kernel clamps to td.k at segment tail.
    int32_t kSeg = (k + effectiveSplitK - 1) / effectiveSplitK;
    kSeg = ((kSeg + ct.baseK - 1) / ct.baseK) * ct.baseK;
    if (effectiveSplitK == 1 && kSeg > k) { kSeg = k; }
    td.kSegLen = kSeg;
}

inline void fill_sparse_tiling_dims(AclsparseltTilingData& td,
                                     const aclsparseLtMatmulDescriptor* md)
{
    const bool hasStructuredSparsity = md_hasStructuredSparsity(md);
    const bool isSparseA = md_isSparseA(md);
    const bool transA = md_transA(md);
    const bool transB = md_transB(md);
    // pruneAlongRow: dispatch by isSparseA to use the sparse
    // matrix's order/trans. dense×dense sets pruneAlongRow=0.
    if (!hasStructuredSparsity) {
        td.pruneAlongRow = 0;
    } else {
        const bool isRowOrder = (isSparseA ? md->matA->order : md->matB->order)
                                == ACL_SPARSE_ORDER_ROW;
        td.pruneAlongRow = ((isSparseA ? transA : transB) != isRowOrder) ? 1 : 0;
    }
    td.sparseTrans = transA ? 1 : 0;
    // A-sparse: prune transposes A, matmul must not.
    // B-sparse: A is dense (not pruned), matmul must handle transA via sparseTrans.
    if (hasStructuredSparsity && isSparseA) {
        td.sparseTrans = 0;
    }
    // B-sparse: prune always outputs B_pruned in (k, n) row-major layout
    // (TransRowOrder path transposes during prune; AlongRow/AlongCol paths
    // read B in (k,n) and output (k,n) unchanged). The matmul must treat
    // B_pruned as NDExt(k, n) — no additional transpose.
    if (hasStructuredSparsity && !isSparseA) {
        td.transB = 0;
    } else {
        td.transB = transB ? 1 : 0;
    }
    // cube A/B 输入的物理行距：稀疏侧为 prune 紧凑输出（ldA=k / ldB=n）；
    // dense 侧从描述符 ld 直读（ROW/COL 的 ld 同为物理 ROW 视图的行步长，
    // IsLdValid 保证 >= 紧凑值）。COL 的转置视图翻转由执行期
    // splt_apply_dense_layout 翻转转置标志承载（ld 口径不变，ldA/ldB 为
    // PlanInit 快照，与 ldC/ldD 同口径）。
    td.ldA = (hasStructuredSparsity && isSparseA)
        ? md_k(md) : static_cast<int32_t>(md->matA->ld);
    td.ldB = (hasStructuredSparsity && !isSparseA)
        ? md_n(md) : static_cast<int32_t>(md->matB->ld);
}

// Fill tiling dimension fields (shared by PlanInit).
// Derived fields are computed from md (matmulDescr) at call time, not cached in asel.
//
// algConfigId/splitK/splitKMode are queried via AlgGetAttribute API;
// other fields (m/n/k/transA etc.) are derived from md.
//
// The AlgGetAttribute calls are extracted into query_alg_attributes().

struct AlgAttrValues {
    int32_t algConfigId = 0;
    int32_t splitK = 1;
    int32_t splitKMode = 0;
};

// Query algConfigId / splitK / splitKMode via the public AlgGetAttribute API.
// On failure, fall back to safe defaults and log the error.
inline AlgAttrValues query_alg_attributes(aclsparseLtConstHandle_t handle,
                                          const aclsparseLtMatmulAlgSelection* asel)
{
    AlgAttrValues v;
    aclsparseLtConstMatmulAlgSelection_t algSelForApi = asel;
    aclsparseStatus_t st = aclsparseLtMatmulAlgGetAttribute(
        handle, &algSelForApi, ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
        &v.algConfigId, sizeof(int32_t));
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kSparseLtLogTag, "fill_tiling_dims: get algConfigId failed (%d), using default 0",
                static_cast<int32_t>(st));
        v.algConfigId = 0;
    }
    st = aclsparseLtMatmulAlgGetAttribute(
        handle, &algSelForApi, ACLSPARSELT_MATMUL_SPLIT_K,
        &v.splitK, sizeof(int32_t));
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kSparseLtLogTag, "fill_tiling_dims: get splitK failed (%d), using default 1",
                static_cast<int32_t>(st));
        v.splitK = 1;
    }
    st = aclsparseLtMatmulAlgGetAttribute(
        handle, &algSelForApi, ACLSPARSELT_MATMUL_SPLIT_K_MODE,
        &v.splitKMode, sizeof(int32_t));
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kSparseLtLogTag, "fill_tiling_dims: get splitKMode failed (%d), using default 0",
                static_cast<int32_t>(st));
        v.splitKMode = 0;
    }
    return v;
}

/**
 * @brief 填充 bias + activation 默认值到 tiling data。
 * prepare_matmul_pointers 会用 descriptor 实际值覆盖这些默认值。
 */
inline void fill_bias_activation_defaults(AclsparseltTilingData& td)
{
    td.biasDevPtr = 0;           // 默认无 bias
    td.biasStride = 0;
    td.activationType = 0;      // 默认无 activation
    td.reluUpperBound = FLT_MAX;
    td.reluThreshold = 0.0f;
    td.geluScaling = 1.0f;
    td.biasChunkMode = 0;       // 默认全量加载（下方根据 UB 预算计算降级标志）
}

/**
 * @brief dtype → 元素字节数（FP32/INT32=4、INT8=1、FP16/BF16=2）。
 * compute_bias_chunk_mode 的 dtype 收窄段（与 compute_ws_layout 的
 * elemSize 四分支同值域）。
 */
inline uint64_t splt_dtype_elem_size(int32_t dtype)
{
    return (dtype == SPLT_DTYPE_FP32 || dtype == SPLT_DTYPE_INT32)
        ? SPLT_FP32_BYTES
        : (dtype == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES : SPLT_HALF_BYTES;
}

/**
 * @brief 融合 general 路径行级小缓冲逐项上界（compute_bias_chunk_mode
 * 组成段）。
 * SpltFusedEpilogueInitBuffers / SpltFusedEpilogueInitBias /
 * SpltLoadScalingUB / splt_init_relu_mask 的逐项上界，替换原 2048B 固定
 * 补贴——固定值低于实际组成（各档 baseN=128/256 下最坏 ~2.6-5.2KB），
 * 边界带 biasChunkMode 不降级会导致 kernel InitBuffer 超 UB：
 *   dTBuf     ：OutType != FP32 时 maxNAlign × outElemSize；
 *   halfBuf   ：OutType == INT8 时 maxNAlign × 2（float→half→int8 两步
 *               Cast 中间量）；
 *   tempBuf   ：maxNAlign × 4（beta*C 中间量 / bias chunk 加载 / GeLU
 *               temp / ReLU thr 向量，chunk+bias 时恒需）；
 *   cTBuf     ：maxNAlign × max(sizeof(A dtype), sizeof(C dtype))
 *               （beta≠0 时 C 行加载；beta 为执行期标量，按最坏恒计，
 *               宁保守）；
 *   cFp32Buf  ：C dtype != FP32 时 maxNAlign × 4（beta≠0 的 C Cast
 *               中间量，同上最坏恒计）；
 *   biasRawBuf：bias 启用且 bias dtype != FP32 时 maxMAlign ×
 *               biasElemSize（chunk 模式的 per-tile Cast 中间量；bias
 *               dtype 键于 A/B valueType，INT8 路恒 FP32）；
 *   vec 槽位  ：alpha/beta 向量 chunk 模式下各 maxMAlign × 4；
 *   reluMask  ：64B（行级 b8 位掩码，固定）。
 * 行级/块级缓冲上界（与 kernel 侧 SpltFusedEpilogueImpl 的
 * maxMAlign/maxNAlign 同式：各 tiling 档 baseM 恒为偶数、baseN 恒为
 * SPLT_L0C_C0 的倍数，故上界 == baseM/baseN 本值）。
 * COL 布局 C/D 中转缓冲（wideCdBuf：maxNAlign 个 32B 槽位，与 ld 无关
 * 的 tile 级常量）经 colWideBytes 输出——原公式漏计该项，COL + 大 m
 * 边界域存在 ~maxNAlign*32B 的预算盲区（阈值漂移）。
 */
inline uint64_t compute_fused_epilogue_small_buffer_bytes(
    const AclsparseltTilingData& td, int32_t dt, const CubeTiling& ct,
    bool biasEnabled, uint64_t& colWideBytes)
{
    const uint64_t maxMAlign = static_cast<uint64_t>((ct.baseM + 1) & ~1);
    const uint64_t maxNAlign =
        static_cast<uint64_t>((ct.baseN + SPLT_L0C_C0 - 1) / SPLT_L0C_C0) * SPLT_L0C_C0;
    colWideBytes = (td.cOrderIsCol != 0) ? maxNAlign * 32 : 0;
    const uint64_t outElemSize = splt_dtype_elem_size(td.outDataType);
    const uint64_t aElemSize = splt_dtype_elem_size(dt);
    const uint64_t cElemSize = splt_dtype_elem_size(td.cDataType);
    const uint64_t cElemSizeMax = (aElemSize > cElemSize) ? aElemSize : cElemSize;
    const uint64_t biasElemSize = (dt == SPLT_DTYPE_INT8 || dt == SPLT_DTYPE_FP32) ? 4 : 2;
    return (td.outDataType != SPLT_DTYPE_FP32 ? maxNAlign * outElemSize : 0) +  // dTBuf
        (td.outDataType == SPLT_DTYPE_INT8 ? maxNAlign * 2 : 0) +               // halfBuf
        maxNAlign * 4 +                                                          // tempBuf
        maxNAlign * cElemSizeMax +                                               // cTBuf
        (td.cDataType != SPLT_DTYPE_FP32 ? maxNAlign * 4 : 0) +                  // cFp32Buf
        (biasEnabled && biasElemSize != 4 ? maxMAlign * biasElemSize : 0) +      // biasRawBuf
        (td.alphaVectorScaling == 1 ? maxMAlign * 4 : 0) +                       // alpha chunk 槽位
        (td.betaVectorScaling == 1 ? maxMAlign * 4 : 0) +                        // beta chunk 槽位
        64;                                                                      // reluMask
}

/**
 * @brief 向量族全量加载字节（compute_bias_chunk_mode 组成段）。
 * biasVecBuf (FP32) + biasLoadBuf (FP16/BF16 only, not FP32/INT8)。
 */
inline uint64_t compute_vec_fullload_bytes(
    const AclsparseltTilingData& td, int32_t dt, bool biasEnabled, int32_t m)
{
    const bool needBiasLoadBuf = (dt != SPLT_DTYPE_FP32 && dt != SPLT_DTYPE_INT8);
    return (td.alphaVectorScaling ? static_cast<uint64_t>(m) * 4 : 0) +
        (td.betaVectorScaling ? static_cast<uint64_t>(m) * 4 : 0) +
        (biasEnabled ? static_cast<uint64_t>(m) * 4 : 0) +
        (biasEnabled && needBiasLoadBuf
            ? static_cast<uint64_t>(m) * static_cast<uint64_t>(SPLT_HALF_BYTES)
            : 0);
}

/**
 * @brief 计算 biasChunkMode 降级标志。
 * 向量族（alpha/beta 缩放向量 + bias）per-tile chunk 降级标志：
 * 三向量全量加载 + tile buffer 超 UB 时降级为 per-chunk——kernel 侧
 * bias 与 alpha/beta 缩放向量同步切换为逐 tile 惰性加载（见
 * matmul_kernel.cpp 的 SpltLoadScalingUB chunk 分支与
 * SpltFusedEpilogueGeneralPath 的 tile 段加载，chunk 模式下三向量同策略，
 * 预算公式与降级动作对齐）。
 * 仅融合路径（splitK==1）走降级（降级后 UB 需求为 tile 级常量，与 m
 * 无关，恒可容纳）；非融合路径（splitK>1）CHUNK=256 固定小 buffer，
 * 其向量族全量加载的 UB 预算由执行期 validate_scaling_ub_budget 校验。
 * 条件覆盖向量族任一启用：原条件 biasEnabled && needVecScale 使
 * vec-only / bias-only 大 m 场景绕过预算检查（预算公式已含 vec
 * 字节，但触发条件与降级动作均须覆盖 vec 自身）。
 * 组成段：小缓冲/中转缓冲上界见 compute_fused_epilogue_small_buffer_bytes，
 * 向量族字节见 compute_vec_fullload_bytes（阈值/条件逐位不变）。
 */
inline void compute_bias_chunk_mode(AclsparseltTilingData& td,
                                     const aclsparseLtMatmulDescriptor* md,
                                     int32_t dt, const CubeTiling& ct, int32_t m)
{
    const bool biasEnabled = (md->biasPointer != nullptr);
    const bool needVecScale = (td.alphaVectorScaling == 1 || td.betaVectorScaling == 1);
    if ((biasEnabled || needVecScale) && td.splitK == 1) {
        const uint64_t ubSize = get_ub_size();
        uint64_t colWideBytes = 0;
        const uint64_t smallBufBytes = compute_fused_epilogue_small_buffer_bytes(
            td, dt, ct, biasEnabled, colWideBytes);
        // accBuf + dFp32Buf（AccType 恒 4 字节；isFastPath 在向量族任一
        // 启用时恒 false，两者恒分配）。
        const uint64_t maxTileBytes =
            2 * static_cast<uint64_t>(ct.baseM) * static_cast<uint64_t>(ct.baseN) * sizeof(float)
            + smallBufBytes
            + colWideBytes;
        const uint64_t vecBytes = compute_vec_fullload_bytes(td, dt, biasEnabled, m);
        if (vecBytes + maxTileBytes > ubSize) {
            td.biasChunkMode = 1;
        }
    }
}

inline void fill_tiling_dims(AclsparseltTilingData& td,
                             aclsparseLtConstHandle_t handle,
                             const aclsparseLtMatmulDescriptor* md,
                             const aclsparseLtMatmulAlgSelection* asel,
                             const WsLayout& wsl, int32_t coreNum)
{
    const int32_t dt = dtype_from_acl(md->matA->valueType);
    const AlgAttrValues v = query_alg_attributes(handle, asel);
    const CubeTiling ct = resolve_cube_tiling(dt, v.algConfigId);
    fill_cube_tiling_dims(td, asel, ct, dt);
    fill_splitk_tiling_dims(td, asel, ct, v.splitK, v.splitKMode);
    td.algConfigId = v.algConfigId;
    td.dataType = dt;
    // outDataType derived from matD->valueType (INT8 path: INT8 or INT32).
    td.outDataType = dtype_from_acl(md->matD->valueType);
    // C 矩阵实际 dtype（CType 贯通：INT8 A/B 的 C 超集行 / 非 INT8 同型）
    td.cDataType = dtype_from_acl(md->matC->valueType);
    fill_sparse_tiling_dims(td, md);
    td.alpha = 1.0f;
    td.beta = 0.0f;
    // vec 编码位此处仅作 PlanInit 时点快照（信息性字段）；Matmul/
    // MatmulSearch 执行期由 fill_matmul_tiling_fields /
    // validate_vector_scaling_pointers 从描述符活读覆盖（PlanInit 后
    // SetAttribute 开/关向量缩放即时生效——统一活读的宽容时序语义）。
    // 派生字段 biasChunkMode 同样在执行期按活读状态重算。
    td.alphaVectorScaling = decode_alpha_scaling(md) ? 1 : 0;
    td.betaVectorScaling = decode_beta_scaling(md) ? 1 : 0;
    td.alphaDevPtr = 0;
    td.betaDevPtr = 0;
    td.coreNum = static_cast<int32_t>(coreNum);
    const int32_t m = md_m(md);
    const int32_t n = md_n(md);
    // 从 mat descriptor 读取 batch 配置（SetAttribute 在 PlanInit 之前调用）
    const int32_t numBatches = (md->matA != nullptr && md->matA->numBatches > 0) ? md->matA->numBatches : 1;
    td.numBatches = numBatches;
    td.batchStrideA = (md->matA != nullptr) ? md->matA->batchStride : 0;
    td.batchStrideB = (md->matB != nullptr) ? md->matB->batchStride : 0;
    td.batchStrideC = (md->matC != nullptr) ? md->matC->batchStride : 0;
    td.batchStrideD = (md->matD != nullptr) ? md->matD->batchStride : 0;
    const int64_t totalMNTiles = static_cast<int64_t>((m + ct.baseM - 1) / ct.baseM) *
                                  static_cast<int64_t>((n + ct.baseN - 1) / ct.baseN);
    // usedCoreNum 基于 per-batch tile 数（不含 numBatches）。
    // batch 循环在 kernel 内部，每个 block 处理所有 batch 的同一组 tile。
    // 此前 totalTiles 乘以 numBatches 导致 usedCoreNum 被放大，多余 block 在
    // 每 batch 的 tile 循环中空转（rawBlockId >= totalTiles 直接 continue）。
    // 改为 per-batch tile 数后，核数与 kernel 侧 totalTiles 一致，无空转。
    const int64_t perBatchTotalTiles = totalMNTiles * static_cast<int64_t>(td.splitK);
    td.usedCoreNum = (perBatchTotalTiles < static_cast<int64_t>(coreNum))
                         ? static_cast<int32_t>(perBatchTotalTiles)
                         : static_cast<int32_t>(coreNum);
    if (td.usedCoreNum <= 0) { td.usedCoreNum = 1; }
    td.tilingOffset = wsl.tilingOff;
    td.aPrunedOffset = wsl.aPrunedOff;
    td.tempResultOffset = wsl.tempOff;
    // X1 布局接入：C/D 的 ld/order 从描述符填充（元素单位）。
    // ld 为 int64 描述符字段，此处收窄为 int32——描述符层
    // （Dense/StructuredDescriptorInit）已显式拒绝 ld > INT32_MAX，
    // 收窄无截断（合法使用域由上游校验保证）。
    td.ldC = static_cast<int32_t>(md->matC->ld);
    td.ldD = static_cast<int32_t>(md->matD->ld);
    td.cOrderIsCol = (md->matC->order == ACL_SPARSE_ORDER_COL) ? 1 : 0;
    fill_bias_activation_defaults(td);
    compute_bias_chunk_mode(td, md, dt, ct, m);
}

#endif // __CCE_AICORE__

#endif // ACLSPARSELT_TILING_FILL_H
