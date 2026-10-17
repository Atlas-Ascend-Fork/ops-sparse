/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef SPGEMM_H
#define SPGEMM_H

#include <stdint.h>
#include <unistd.h>
#include <acl/acl_base_rt.h>

#ifndef __gm__
#define __gm__
#endif

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

/* 常量定义，与 SpMM 保持一致 */
constexpr uint32_t kSpgemmMaxThreadsPerBlock = 512u;
constexpr uint32_t kSpgemmWarpSize           = 32u;
constexpr uint32_t kSpgemmMinRowsPerBlock    = 1u;
constexpr uint32_t kSpgemmWsHeaderBytes      = 64u;
constexpr uint32_t kSpgemmWsAlign            = 64u;

/* 分块累加阈值。
 * 当 n > kSpgemmTileN 时，将 n 维度按 kSpgemmTileN 列分块。
 * 每块使用 local 累加器（kSpgemmTileN floats = 512 字节）
 * + bitmask（kSpgemmTileN/8 = 16 字节）= 528 字节，完全放在寄存器中不溢出。
 *
 * 替代了之前的 GM-backed 累加器路径（n > kMaxLocalAccumN），
 * 该路径因 bisheng 编译器寄存器溢出 → GM 缓存一致性问题，
 * 导致符号/数值阶段列集不一致。
 *
 * 代价：分块路径每块需遍历一次 A.row × B.row（ceil(n / kSpgemmTileN) 次），
 * 但每次遍历无溢出且结果正确。 */
constexpr int32_t kSpgemmTileN              = 128;   /* tile width for n>128 path */

/* 仅支持 fp32 / fp16 / bf16 */
constexpr int32_t SPGEMM_DTYPE_INVALID = -1;
constexpr int32_t SPGEMM_DTYPE_FP32    = 0;
constexpr int32_t SPGEMM_DTYPE_FP16    = 1;
constexpr int32_t SPGEMM_DTYPE_BF16    = 2;

static inline int32_t SpgemmDataTypeFromAcl(aclDataType dt) {
    switch (dt) {
        case ACL_FLOAT:   return SPGEMM_DTYPE_FP32;
        case ACL_FLOAT16: return SPGEMM_DTYPE_FP16;
        case ACL_BF16:    return SPGEMM_DTYPE_BF16;
        default:          return SPGEMM_DTYPE_INVALID;
    }
}

/* SpGEMM 内部描述符 — 多阶段 API 状态机。
 *
 * 跟踪 7 接口生命周期的推进：
 *   Created → WorkEstimated → SymbolicDone → Computed → Copied
 *
 * Fields:
 *   numProds     — 总乘积对数 (Σ_i Σ_{k∈A.row(i)} nnz(B.row(k)))
 *   nnzC         — 符号阶段输出 nnz（符号阶段完成前为 0）
 *   maxNnzC      — colIndC/valuesC 的 workspace 容量
 *   phase        — 当前生命周期状态
 *   cInDataValid — 用户标记：matC->values 包含有效的 C_in 数据（β≠0 路径）
 *   buffer1Size  — buffer1 字节大小（WorkEstimation + Compute）
 *   buffer2Size  — buffer2 字节大小（Compute + Copy）
 *   buffer3Size  — buffer3 字节大小（EstimateMemory，仅 ALG2/3）
 */
typedef enum SpgemmPhase_t {
    SPGEMM_PHASE_CREATED       = 0,
    SPGEMM_PHASE_WORK_ESTIMATE = 1,
    SPGEMM_PHASE_SYMBOLIC_DONE = 2,
    SPGEMM_PHASE_COMPUTED      = 3,
    SPGEMM_PHASE_COPIED        = 4,
} SpgemmPhase_t;

struct SpgemmDescr {
    int64_t numProds;
    int64_t nnzC;
    int64_t maxNnzC;
    SpgemmPhase_t phase;
    int32_t cInDataValid;
    int64_t buffer1Size;
    int64_t buffer2Size;
    int64_t buffer3Size;
};

/* TilingData 结构体 — kernel 从 GM 读取到 scratch */
#pragma pack(push, 4)

struct SpgemmSymbolicTilingData {
    int32_t m;
    int32_t n;
    int32_t blockDim;
    int32_t baseA;
    int32_t baseB;
    int32_t baseC;
    int32_t reorderOffset;
    int32_t binEdgeOffset;
    int64_t symBitmapOffset;   /* GM-backed per-block bitmap for n>64 dedup */
};

struct SpgemmPrefixSumTilingData {
    int32_t m;
    int32_t baseC;
};

struct SpgemmNumericTilingData {
    int32_t m;
    int32_t n;
    int32_t blockDim;
    int32_t baseA;
    int32_t baseB;
    int32_t baseC;
    int32_t reorderOffset;
    int32_t binEdgeOffset;
    int32_t rowPtrCOffset;
    int32_t nnzPerRowOffset;
    int64_t accumOffset;
    int64_t numAccumOffset;     /* n>64 时 GM-backed 每 block 的 dense float 累加器 */
    int64_t symBitmapOffset;    /* shared with symbolic phase (same workspace region) */
    float   alphaHost;
    float   betaHost;
    uint64_t alphaPtr;
    uint64_t betaPtr;
};

struct SpgemmFillTilingData {
    int32_t count;
    int32_t value;
};

#pragma pack(pop)

/* Workspace 布局（全部 64B 对齐）：
 *   [ 0           , tilingOff    )  : 64B header
 *   [ tilingOff   , bRowPtrOff   )  : SpgemmTilingData (symbolic/numeric 取较大者)
 *   [ bRowPtrOff  , nnzPerRowOff )  : int32 B rowPtr copy [k+1]
 *   [ nnzPerRowOff, rowPtrCOff   )  : int32 nnzPerRow [m]
 *   [ rowPtrCOff  , reorderOff   )  : int32 rowPtrC [m+1] + int32 nnzC [1]
 *   [ reorderOff  , binEdgeOff   )  : int32 reorder [m]
 *   [ binEdgeOff  , colIndCOff   )  : int32 binEdge [blockDim+1]
 *   [ colIndCOff  , valuesCOff   )  : int32 colIndC [maxNnzC]
 *   [ valuesCOff  , gmAccumOff   )  : dtype valuesC [maxNnzC]
 *   [ gmAccumOff  , end          )  : n>64 时 GM-backed 每 block workspace：
 *                                        符号阶段: bitmap (ceil(n/8) per block)
 *                                        数值阶段:  float accumulator (n*4 per block)
 *                                        Total: blockDim * n * sizeof(float)
 */
struct SpgemmWsOffsets {
    int64_t headerOff;
    int64_t tilingOff;
    int64_t bRowPtrOff;
    int64_t nnzPerRowOff;
    int64_t rowPtrCOff;
    int64_t reorderOff;
    int64_t binEdgeOff;
    int64_t colIndCOff;
    int64_t valuesCOff;
    int64_t gmAccumOff;     /* GM-backed per-block accum+touched region */
    int64_t totalBytes;
};

static inline int64_t spgemm_align_up(int64_t v, int64_t a) {
    /* G.EXP.22: 除法仅在 a > 0 时执行，消除静态分析除零告警。 */
    if (a > 0) {
        return ((v + a - 1) / a) * a;
    }
    return v;
}

static inline SpgemmWsOffsets ComputeSpgemmWsOffsets(int64_t m, int64_t n, int64_t k,
                                                        int64_t nnzA,
                                                        int32_t blockDim,
                                                        int32_t computeDtypeSize) {
    SpgemmWsOffsets o;
    o.headerOff      = kSpgemmWsHeaderBytes;
    o.tilingOff      = spgemm_align_up(o.headerOff, kSpgemmWsAlign);
    int64_t tilingSz = (int64_t)sizeof(SpgemmNumericTilingData);
    if ((int64_t)sizeof(SpgemmSymbolicTilingData) > tilingSz) {
        tilingSz = (int64_t)sizeof(SpgemmSymbolicTilingData);
    }

    o.bRowPtrOff     = spgemm_align_up(o.tilingOff + tilingSz, kSpgemmWsAlign);
    int64_t bRowPtrLen = k + 1;
    o.nnzPerRowOff   = spgemm_align_up(o.bRowPtrOff + bRowPtrLen * (int64_t)sizeof(int32_t), kSpgemmWsAlign);
    o.rowPtrCOff     = spgemm_align_up(o.nnzPerRowOff + m * (int64_t)sizeof(int32_t), kSpgemmWsAlign);
    o.reorderOff     = spgemm_align_up(o.rowPtrCOff + (m + 1) * (int64_t)sizeof(int32_t), kSpgemmWsAlign);
    o.binEdgeOff     = spgemm_align_up(o.reorderOff + m * (int64_t)sizeof(int32_t), kSpgemmWsAlign);
    o.colIndCOff     = spgemm_align_up(o.binEdgeOff + (blockDim + 1) * (int64_t)sizeof(int32_t), kSpgemmWsAlign);

    /* maxNnzC: use m*n as upper bound (absolute max for any SpGEMM result),
     * capped at 5e8 for memory safety. The previous nnzA*16 heuristic was
     * insufficient for high-density matrices where nnzC >> nnzA*16. */
    int64_t maxNnzC  = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    if (maxNnzC > 500000000) { maxNnzC = 500000000; }
    if (maxNnzC < 1024) { maxNnzC = 1024; }
    o.valuesCOff     = spgemm_align_up(o.colIndCOff + maxNnzC * (int64_t)sizeof(int32_t), kSpgemmWsAlign);

    /* GM-backed per-block workspace for n>64 path.
     * Each block gets n*sizeof(float) bytes for the GM float accumulator.
     * The local maskBuf (ceil(n/8) bytes, max 128) stays in register file.
     * Total: blockDim * n * sizeof(float) */
    int64_t gmAccumPerBlock = n * (int64_t)sizeof(float);
    int64_t gmAccumTotal = static_cast<int64_t>(blockDim) * gmAccumPerBlock;
    o.gmAccumOff     = spgemm_align_up(o.valuesCOff + maxNnzC * (int64_t)computeDtypeSize, kSpgemmWsAlign);
    o.totalBytes     = spgemm_align_up(o.gmAccumOff + gmAccumTotal, kSpgemmWsAlign);
    return o;
}

/* Host 侧 launch dispatcher — C++ 链接。
 * tiling 通过 GM 传递（对齐 SpMM 风格），host 写 tiling 到 workspace，kernel 从 GM 读。
 */
void spgemm_symbolic_kernel_do(
    GM_ADDR aRowPtr, GM_ADDR aColInd,
    GM_ADDR bRowPtr, GM_ADDR bColInd,
    GM_ADDR nnzPerRow,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM,
    uint32_t numBlocks, void *stream);

void spgemm_prefixsum_kernel_do(
    GM_ADDR nnzPerRow,
    GM_ADDR rowPtrC,
    GM_ADDR nnzCDev,
    GM_ADDR tilingGM,
    void *stream);

void spgemm_numeric_kernel_do(
    GM_ADDR aRowPtr, GM_ADDR aColInd, GM_ADDR aValues,
    GM_ADDR bRowPtr, GM_ADDR bColInd, GM_ADDR bValues,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valuesC,
    GM_ADDR nnzPerRow,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM,
    int32_t dataType, uint32_t numBlocks, void *stream);

void spgemm_fill_kernel_do(
    GM_ADDR dst,
    GM_ADDR tilingGM,
    void *stream);

#endif /* SPGEMM_H */
