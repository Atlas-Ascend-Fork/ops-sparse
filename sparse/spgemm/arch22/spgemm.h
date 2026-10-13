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
 * \file spgemm.h
 * \brief SpGEMM arch22（Atlas A2 / A3，dav-2201）TilingData 与 workspace 布局。
 *
 * 本文件同时被 host（spgemm_host.cpp）与 kernel（spgemm_*_kernel.cpp）包含，
 * 故只能使用 stdint 与纯 POD，不得引入 acl/AscendC 头文件。
 *
 * 三个 Kernel 分工：
 *   count     —— 统计每行中间乘积数 P_i（只读 A.rowPtr/A.colIdx/B.rowPtr），dtype 无关
 *   symbolic  —— 按 k 路归并去重求每行 nnz（只读结构，不读 values），dtype 无关
 *   numeric   —— 按 dtype 模板特化，归并累加写出 colIndices / values
 *
 * 行内模板（T1 归并 / T3 列分块）在 Kernel 内按 da 与 P_i 判定。
 */

#ifndef SPGEMM_ARCH22_H
#define SPGEMM_ARCH22_H

#include <stdint.h>

#ifndef __gm__
#define __gm__
#endif

// A2/A3 单核 UB 192KB，预留部分给编译器临时与栈。
#define SPGEMM_ARCH22_UB_SIZE      (192 * 1024)
#define SPGEMM_ARCH22_UB_OVERHEAD  (24 * 1024)
// T1 归并路径支持的最大段数（= A 单行 nnz 上限）。超过则退化到 T3。
#define SPGEMM_ARCH22_MAX_WAYS     64
// 段首列索引描述符（bHead）在 SpgemmSegments 里的缓存长度。
// 只有编译期特化的快路径（da ∈ {2,3,4,6,7,8}）消费它，故取 8 即可覆盖；
// da > 8 的通用归并路径照旧从 GM 读段首。
#define SPGEMM_ARCH22_FIRST_WAYS   8
// rowNnz 回写的行分块大小（限制 UB 占用，避免 M 很大时缓冲爆掉）。
#define SPGEMM_ARCH22_ROW_TILE     256
// A 行列索引搬入 UB 的分块长度（T3 路径下单行 nnz 可能很大）。
#define SPGEMM_ARCH22_COL_TILE     2048
// 符号阶段 T3 列分块宽度（每列仅需 4 字节命中标记，故可比数值阶段更宽）。
#define SPGEMM_ARCH22_SYM_CHUNK    8192

#ifndef SPGEMM_ROUNDUP
#define SPGEMM_ROUNDUP(a, b) (((a) + (b) - 1) / (b) * (b))
#endif

/* dtype ID，与 common/spgemm_common.h 的 SpgemmDtypeId 保持一致 */
#define SPGEMM_KERNEL_DTYPE_FP32   0
#define SPGEMM_KERNEL_DTYPE_FP16   1
#define SPGEMM_KERNEL_DTYPE_BF16   2
#define SPGEMM_KERNEL_DTYPE_C64    3

/**
 * TilingData：host 填充后拷贝到 device，kernel 只读。
 * 所有 *Offset 均为相对对应 workspace 基址的字节偏移，64B 对齐。
 */
typedef struct {
    uint32_t M;              // A 的行数 / C 的行数
    uint32_t K;              // A 的列数 == B 的行数
    uint32_t N;              // B 的列数 / C 的列数
    uint32_t nnzA;
    uint32_t nnzB;
    uint32_t nnzCIn;         // beta != 0 时 matC 原有 nnz；否则 0
    uint32_t blockDim;       // 实际使用核数（运行时查询）
    uint32_t dtypeId;        // SPGEMM_KERNEL_DTYPE_*
    uint32_t mergeCapacity;  // T1 单行归并缓冲容量（元素数，按 dtype 动态算）
    uint32_t chunkWidth;     // T3 列分块宽度（元素数，按 dtype 动态算）
    uint32_t betaNonZero;    // beta != 0 时需并入 matC 原结构
    // `max_r rowProducts[r] <= mergeCapacity` 时置 1。
    // 数值阶段据此跳过每行的 rowProducts 标量 GM 读。
    // 保守方向：置 0 永远安全（退回逐行读），只有置 1 才需要那个不等式成立。
    uint32_t prodFitsCapacity;
    float    alphaRe;
    float    alphaIm;        // complex64 用；实数类型恒 0
    float    betaRe;
    float    betaIm;
    int64_t  numProds;       // 中间乘积总数（int64，避免溢出）

    // buffer1 内偏移
    int64_t  binEdgeOffset;      // int32  × (blockDim+1)：每核负责的起始行（contiguous 分区）
    int64_t  rowProductsOffset;  // int64  × M：每行中间乘积数 P_i
    int64_t  rowNnzOffset;       // int32  × M：符号阶段产出的每行 nnz
    int64_t  coreSumOffset;      // int32  × (blockDim+1)：两级前缀和的每核局部总和
    int64_t  scratchBaseOffset;  // int64  × (blockDim+1)：T1 融合暂存区每核起始槽位
    // int32 × K：B 每行的等差公差描述符。
    //   bApStride[k] = d   若 nnz(B.row k) >= 2 且该行列索引为公差 d 的等差数列
    //                = 0   否则
    // 由 count Kernel 一次性算出，供符号阶段的轮转快路径判定复用。
    int64_t  bApStrideOffset;
    // int32 × K：B 每行的首列索引 bColIndices[bRowPtr[k]]。
    //   bHead[k] = bColIdx[bRowPtr[k]]  若 nnz(B.row k) >= 1
    //            = 0                    否则（空行的段在建段时即被跳过）
    // 与 bApStride 一同由 count Kernel 算出。
    int64_t  bHeadOffset;
    // int32 × (blockDim × 16)：count Kernel 内部的跨核 B 行长统计交换区。
    // int32 × (blockDim × 8)：紧随其后的 SyncAll 屏障区。
    int64_t  bStatOffset;
    int64_t  syncOffset;

    // buffer2 内偏移
    int64_t  cRowOffsetsOffset;  // int32  × (M+1)
    int64_t  cColIndicesOffset;  // int32  × nnzUpperBound
    int64_t  cValuesOffset;      // sizeof(T) × nnzUpperBound
    int64_t  scratchColOffset;   // int32  × nnzUpperBound：T1 融合暂存列索引
    int64_t  scratchValOffset;   // sizeof(T) × nnzUpperBound：T1 融合暂存值
} SpgemmArch22TilingData;

/**
 * T1 归并容量：按 dtype 字节宽动态计算，complex64 自动减半。
 *
 * 单行归并所需 UB（元素数 cap）：
 *   segCol   cap × 4                 B 各段列索引（连续拼接）
 *   segVal   cap × valBytes          B 各段值（数值阶段）
 *   outCol   cap × 4                 归并输出列索引
 *   outVal   cap × accBytes          fp32 累加器（complex 实虚双路 → 8B）
 */
static inline uint32_t SpgemmArch22MergeCapacity(uint32_t dtypeId)
{
    uint64_t availUB = SPGEMM_ARCH22_UB_SIZE - SPGEMM_ARCH22_UB_OVERHEAD;
    // 固定开销：A 行列索引/值、各段游标与长度、rowNnz 分块缓冲
    uint64_t fixedCost = (uint64_t)SPGEMM_ARCH22_MAX_WAYS * (4 + 8 + 4 + 4 + 4) +
                         (uint64_t)SPGEMM_ARCH22_ROW_TILE * 4;
    if (availUB <= fixedCost) {
        return 64;
    }
    availUB -= fixedCost;

    uint64_t valBytes = (dtypeId == SPGEMM_KERNEL_DTYPE_C64) ? 8 : 4;
    uint64_t accBytes = (dtypeId == SPGEMM_KERNEL_DTYPE_C64) ? 8 : 4;
    uint64_t perElem = 4 + valBytes + 4 + accBytes;
    uint64_t cap = availUB / perElem;
    cap = (cap / 32) * 32;  // 32 元素对齐，便于 DataCopyPad
    if (cap < 64) {
        cap = 64;
    }
    if (cap > 4096) {
        cap = 4096;
    }
    return (uint32_t)cap;
}

/**
 * T3 列分块宽度：稠密累加器每列需 accBytes 值 + 4 字节命中标记。
 * complex64 的 acc 为 8 字节，chunkWidth 自动缩小。
 */
static inline uint32_t SpgemmArch22ChunkWidth(uint32_t dtypeId)
{
    uint64_t availUB = SPGEMM_ARCH22_UB_SIZE - SPGEMM_ARCH22_UB_OVERHEAD;
    availUB = availUB / 2;  // 另一半留给输出暂存与段缓冲
    uint64_t accBytes = (dtypeId == SPGEMM_KERNEL_DTYPE_C64) ? 8 : 4;
    uint64_t perCol = accBytes + 4;  // 值 + 命中标记
    uint64_t width = availUB / perCol;
    width = (width / 32) * 32;
    if (width < 32) {
        width = 32;
    }
    return (uint32_t)width;
}

static inline uint64_t SpgemmArch22ValueBytes(uint32_t dtypeId)
{
    if (dtypeId == SPGEMM_KERNEL_DTYPE_C64) {
        return 8;
    }
    if (dtypeId == SPGEMM_KERNEL_DTYPE_FP32) {
        return 4;
    }
    return 2;  // fp16 / bf16
}

/* ---- buffer1 布局 ---- */
typedef struct {
    int64_t  tilingOff;
    int64_t  binEdgeOff;
    int64_t  rowProductsOff;
    int64_t  rowNnzOff;
    int64_t  coreSumOff;
    int64_t  scratchBaseOff;
    int64_t  bApStrideOff;
    int64_t  bHeadOff;
    int64_t  bStatOff;   // int32 × (blockDim × 16)：count 内部跨核 B 行长统计
    int64_t  syncOff;    // int32 × (blockDim × 8)：SyncAll 屏障区
    uint64_t totalBytes;
} SpgemmArch22Ws1Layout;

static inline SpgemmArch22Ws1Layout SpgemmArch22ComputeWs1(uint32_t M, uint32_t K,
                                                          uint32_t blockDim)
{
    SpgemmArch22Ws1Layout off;
    off.tilingOff      = 64;
    off.binEdgeOff     = off.tilingOff + (int64_t)SPGEMM_ROUNDUP(sizeof(SpgemmArch22TilingData), 64);
    off.rowProductsOff = off.binEdgeOff + (int64_t)SPGEMM_ROUNDUP(((uint64_t)blockDim + 1) * 4, 64);
    off.rowNnzOff      = off.rowProductsOff + (int64_t)SPGEMM_ROUNDUP((uint64_t)M * 8, 64);
    off.coreSumOff     = off.rowNnzOff + (int64_t)SPGEMM_ROUNDUP((uint64_t)M * 4, 64);
    off.scratchBaseOff = off.coreSumOff + (int64_t)SPGEMM_ROUNDUP(((uint64_t)blockDim + 1) * 4, 64);
    off.bApStrideOff   = off.scratchBaseOff + (int64_t)SPGEMM_ROUNDUP(((uint64_t)blockDim + 1) * 8, 64);
    off.bHeadOff       = off.bApStrideOff +
                         (int64_t)SPGEMM_ROUNDUP((uint64_t)((K > 0) ? K : 1) * 4, 64);
    // bStat：int32 × (blockDim × 16)，count Kernel 内部跨核 B 行长统计交换区。
    // 每核跨距 16 个 int32 = 64B = 一整条 cacheline，防伪共享。
    off.bStatOff       = off.bHeadOff +
                         (int64_t)SPGEMM_ROUNDUP((uint64_t)((K > 0) ? K : 1) * 4, 64);
    // syncOff：SyncAll 的显式 GM 屏障区。每核 32B（8 个 int32）是该原语的约定。
    off.syncOff        = off.bStatOff + (int64_t)SPGEMM_ROUNDUP((uint64_t)blockDim * 64, 64);
    off.totalBytes     = (uint64_t)off.syncOff +
                         SPGEMM_ROUNDUP((uint64_t)blockDim * 32, 64);
    return off;
}

/* ---- buffer2 布局：C 的 rowOffsets + colIndices + values + T1 融合暂存区 ---- */
typedef struct {
    int64_t  cRowOffsetsOff;
    int64_t  cColIndicesOff;
    int64_t  cValuesOff;
    int64_t  scratchColOff;    // int32    × nnzUpperBound：T1 融合归并暂存列索引
    int64_t  scratchValOff;    // valBytes × nnzUpperBound：T1 融合归并暂存值
    uint64_t totalBytes;
} SpgemmArch22Ws2Layout;

static inline SpgemmArch22Ws2Layout SpgemmArch22ComputeWs2(
    uint32_t M, int64_t nnzUpperBound, uint32_t dtypeId)
{
    // nnz=0 时仍需 64B 占位，避免 aclrtMalloc(0) 与偏移退化。
    uint64_t nnzUB = (nnzUpperBound > 0) ? (uint64_t)nnzUpperBound : 1;
    uint64_t valBytes = SpgemmArch22ValueBytes(dtypeId);
    SpgemmArch22Ws2Layout off;
    off.cRowOffsetsOff = 64;
    off.cColIndicesOff = off.cRowOffsetsOff + (int64_t)SPGEMM_ROUNDUP(((uint64_t)M + 1) * 4, 64);
    off.cValuesOff     = off.cColIndicesOff + (int64_t)SPGEMM_ROUNDUP(nnzUB * 4, 64);
    // T1 融合暂存区：符号阶段一次归并即产出 (col, val)，数值阶段压实到最终位置。
    // 容量按 nnzUB：暂存的是归并后的输出（每行 rowNnz 个），与 C 规模同阶。
    off.scratchColOff  = off.cValuesOff    + (int64_t)SPGEMM_ROUNDUP(nnzUB * valBytes, 64);
    off.scratchValOff  = off.scratchColOff + (int64_t)SPGEMM_ROUNDUP(nnzUB * 4, 64);
    off.totalBytes     = (uint64_t)off.scratchValOff + SPGEMM_ROUNDUP(nnzUB * valBytes, 64);
    return off;
}

/* ---------------------------------------------------------------------------
 * Kernel launch 入口
 * --------------------------------------------------------------------------- */

/**
 * 中间乘积计数 Kernel（WorkEstimation 阶段）：
 * rowProducts[r] = Σ_{k ∈ A.cols(r)} nnz(B.row(k))，int64 计数避免溢出。
 * 只读结构，与 dtype 无关。
 * 同趟还产出 bApStride[k]（B 每行是否为等差数列及其公差）和 bHead[k]（B 每行首列索引）。
 */
void spgemm_arch22_count_launch(
    void *aRowPtr, void *aColIdx, void *bRowPtr, void *bColIdx,
    void *buffer1, void *tiling, uint32_t blockDim, void *stream);

/**
 * 符号阶段 Kernel（Compute 阶段第 1 趟）：
 * 每核处理 binEdge 划定的连续行区间，k 路归并去重求 rowNnz[r]，
 * 随后做两级前缀和（核内串行扫描 + 核间偏移修正）直接产出 C.rowOffsets，
 * nnz(C) 即 C.rowOffsets[M]。
 *
 * T1 融合（Variant B）：对走 T1 归并的行，本 Kernel 在同一趟归并中直接算出
 * (colIdx, value) 并写入暂存区，数值阶段仅做压实搬运。
 */
void spgemm_arch22_symbolic_launch(
    void *aRowPtr, void *aColIdx, void *aValues,
    void *bRowPtr, void *bColIdx, void *bValues,
    void *cRowPtrIn, void *cColIdxIn,
    void *buffer1, void *buffer2, void *tiling,
    uint32_t dtypeId, uint32_t blockDim, void *stream);

/**
 * 数值阶段 Kernel（Compute 阶段第 2 趟）：按 dtype 分派模板特化。
 * fp16/bf16 一律 fp32 累加后 Cast 回目标类型；complex64 实虚双路 fp32 累加。
 * 累加顺序固定为「段序（= A 行内列序）× 段内 B 列序」，故 bit-wise 可复现。
 */
void spgemm_arch22_numeric_launch(
    void *aRowPtr, void *aColIdx, void *aValues,
    void *bRowPtr, void *bColIdx, void *bValues,
    void *cRowPtrIn, void *cColIdxIn, void *cValuesIn,
    void *buffer1, void *buffer2, void *tiling,
    uint32_t dtypeId, uint32_t blockDim, void *stream);

#endif  // SPGEMM_ARCH22_H
