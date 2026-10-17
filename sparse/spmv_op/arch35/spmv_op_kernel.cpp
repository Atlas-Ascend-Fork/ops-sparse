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
 * \file spmv_op_kernel.cpp
 * \brief spmv_op SIMT kernel 实现（仅 arch35/DAV-3510 可用）。
 *
 * 三层结构：
 * Layer 1: __simt_vf__ 计算函数（线程级并行，grid-stride loop，Priest 双重补偿求和）
 *   Layer 2: __global__ 调度器（读 tiling → asc_vf_call 分发）
 *   Layer 3: kernel_do 启动器（<<<>>> 异步 launch）
 *
 * 两个 kernel：
 *   Kernel 1 (Compute):  Z = alpha * A * X + beta * Y，Priest 双重补偿求和
 *   Kernel 2 (BetaY):    nnz==0 时，Z = beta * Y
 *
 * 模板实例化矩阵：
 *   [i32/ALG1] [i32/ALG2] [i64/ALG1] [i64/ALG2]
 * 由 dispatcher 按 tiling.rowOffsetType + tiling.algType 选择。
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "spmv_op_kernel.h"

// ===========================================================================
// Dispatcher 基类：提取公共逻辑（block→行范围映射、线程数计算）
// ===========================================================================

struct SpmvOpDispatcherBase {
    int32_t m_{0};
    int32_t indexBase_{0};
    int32_t algType_{SPMV_OP_ALG_TYPE_1};
    int32_t rowOffsetType_{SPMV_OP_IDX_RT_I32};
    float alpha_{1.0f};
    float beta_{0.0f};
    uint32_t rowsPerBlock_{0};

    __aicore__ inline static uint32_t AlignToWarp(uint32_t simtThreadNum)
    {
        return (simtThreadNum + kSpmvOpWarpSize - 1u) & ~(kSpmvOpWarpSize - 1u);
    }

    // 根据 GetBlockIdx() 计算本 core 行范围，并对齐 SIMT 线程数到 warp 整数倍
    __aicore__ inline void ComputeRowRangeAndThreads(
        int32_t &rowStart, int32_t &rowEnd, uint32_t &simtThreadNum)
    {
        uint32_t blockId = AscendC::GetBlockIdx();
        rowStart = static_cast<int32_t>(blockId) * static_cast<int32_t>(rowsPerBlock_);
        rowEnd = rowStart + static_cast<int32_t>(rowsPerBlock_);
        if (rowEnd > m_) {
            rowEnd = m_;
        }
        int32_t coreRangeLen = rowEnd - rowStart;
        simtThreadNum = (coreRangeLen < static_cast<int32_t>(kSpmvOpMaxThreadsPerBlock))
            ? static_cast<uint32_t>(coreRangeLen)
            : kSpmvOpMaxThreadsPerBlock;
        if (simtThreadNum == 0) {
            simtThreadNum = 1;
        }
        simtThreadNum = AlignToWarp(simtThreadNum);
    }
};

// ===========================================================================
// Kernel 1: 主计算 SpMVOp SIMT VF (Priest 双重补偿点积)
// ===========================================================================

// Priest 双重补偿点积辅助函数：对单行 CSR 数据完成 dot(A[r], X) 的高精度计算。
//
// 使用三精度累加器 (dotHi, dotMd, dotLo)：dot ≈ dotHi + dotMd + dotLo
//  - 第一层 (dotHi): TwoProductFMA 产生 (prod, pi), 再用 TwoSum 将 prod 累入 dotHi, 得误差 e1
//  - 第二层 (dotMd): 将 (e1 + pi) 合为 ee, 经 TwoSum 累入 dotMd, 得误差 e2
//  - 第三层 (dotLo): 直接累计 e2
// 误差界: |dot - exact| <= 2 * eps * |exact_dot|, 条件: 12*n*eps < 1
// 参考: Priest (2003) "A Doubly Compensated Sum and Dot Product Algorithm"
template <typename RowT>
__aicore__ __simt_callee__ inline void PriestDotProduct(
    __gm__ const RowT *rowOffsets,
    __gm__ const int32_t *colInd,
    __gm__ const float *values,
    __gm__ const float *xVec,
    RowT rowStart, RowT rowEnd,
    int32_t indexBase,
    float &outHi, float &outMd, float &outLo)
{
    float dotHi = 0.0f;
    float dotMd = 0.0f;
    float dotLo = 0.0f;
    for (RowT p = rowStart; p < rowEnd; ++p) {
        const int32_t col = colInd[p] - indexBase;
        const float val   = values[p];
        const float xVal  = xVec[col];

        // TwoProductFMA: val * xVal = prod + pi
        const float prod = val * xVal;
        const float pi   = __fmaf_rn(val, xVal, -prod);

        // 第一层: TwoSum(dotHi, prod) → (dotHi_new, e1)
        const float s1 = dotHi + prod;
        const float v1 = s1 - dotHi;
        const float e1 = (dotHi - (s1 - v1)) + (prod - v1);
        dotHi = s1;

        // 第二层: TwoSum(dotMd, e1 + pi) → (dotMd_new, e2)
        const float ee = e1 + pi;
        const float s2 = dotMd + ee;
        const float v2 = s2 - dotMd;
        const float e2 = (dotMd - (s2 - v2)) + (ee - v2);
        dotMd = s2;

        // 第三层: 将 e2 累入 dotLo
        dotLo = dotLo + e2;
    }
    outHi = dotHi;
    outMd = dotMd;
    outLo = dotLo;
}

// SIMT VF 计算函数：Z = alpha * A * X + beta * Y，含 Priest 双重补偿求和。
//
// 模板参数：
//   RowT       int32_t 或 int64_t（csrRowOffsets 类型）
//   UseReorder ALG1: false（直接用行号 r），ALG2: true（reorder[r] 重映射）
//
// 参数说明：
//   rowStart, rowEnd: 本 core 负责的行范围
//   threadsPerCore: 本 core 的 SIMT 线程数（用于 grid-stride 步长）
template <typename RowT, bool UseReorder>
__simt_vf__ __aicore__ __launch_bounds__(kSpmvOpMaxThreadsPerBlock) inline void
SpmvOpSimtCompute(
    __gm__ const RowT *rowOffsets,
    __gm__ const int32_t *colInd,
    __gm__ const float *values,
    __gm__ const float *xVec,
    __gm__ const float *yVec,
    __gm__ float *zVec,
    __gm__ const int32_t *reorder,
    float alpha, float beta,
    int32_t indexBase,
    int32_t rowStart, int32_t rowEnd,
    int32_t threadsPerCore)
{
    const int32_t rangeLen = rowEnd - rowStart;
    for (int32_t row = static_cast<int32_t>(threadIdx.x);
         row < rangeLen;
         row += threadsPerCore) {
        const int32_t r = rowStart + row;
        // UseReorder: ALG2 将逻辑行号 r 映射到 reorder 表中的原始行号
        const int32_t origRow = UseReorder ? reorder[r] : r;
        const int32_t outRow = UseReorder ? origRow : r;

        const RowT rStart = rowOffsets[origRow] - static_cast<RowT>(indexBase);
        const RowT rEnd   = rowOffsets[origRow + 1] - static_cast<RowT>(indexBase);

        // beta=0  短路：跳过 Y[outRow] 的 GM 读
        float dotHi = 0.0f, dotMd = 0.0f, dotLo = 0.0f;
        PriestDotProduct(rowOffsets, colInd, values, xVec, rStart, rEnd, indexBase,
                         dotHi, dotMd, dotLo);
        const float yVal = (beta == 0.0f) ? 0.0f : yVec[outRow];

        // 最终合并: z = alpha * (dotHi + dotMd + dotLo) + beta * yVal
        // 使用 FMA 链保持精度（避免中间结果先合并为一个 float 再乘 alpha 的精度损失）
        const float dotCorr = dotMd + dotLo;
        const float step1 = __fmaf_rn(alpha, dotCorr, beta * yVal);
        const float zVal  = __fmaf_rn(alpha, dotHi, step1);

        // UseReorder 时使用原行号 outRow 写入（ALG2 按原始行号写 Z）
        zVec[outRow] = zVal;
    }
}

// ===========================================================================
// Kernel 1: 主计算 Dispatcher
// ===========================================================================

class SpmvOpDispatcher : public SpmvOpDispatcherBase {
public:
    __aicore__ inline void Init(
        GM_ADDR gmRowOffsets, GM_ADDR gmColInd, GM_ADDR gmValues,
        GM_ADDR gmXVec, GM_ADDR gmYVec, GM_ADDR gmZVec,
        GM_ADDR gmReorder,
        GM_ADDR gmBinEdge,
        const SpmvOpTilingData *tiling)
    {
        rowOffsetsAddr_ = gmRowOffsets;
        colInd_ = (__gm__ const int32_t *)gmColInd;
        values_ = (__gm__ const float *)gmValues;
        xVec_ = (__gm__ const float *)gmXVec;
        yVec_ = (__gm__ const float *)gmYVec;
        zVec_ = (__gm__ float *)gmZVec;
        reorder_ = (__gm__ const int32_t *)gmReorder;
        binEdge_ = (__gm__ const int32_t *)gmBinEdge;
        m_ = tiling->m;
        indexBase_ = tiling->indexBase;
        rowOffsetType_ = tiling->rowOffsetType;
        algType_ = tiling->algType;
        rowsPerBlock_ = tiling->rowsPerBlock;
        // DEVICE mode: alpha/beta 是 device 指针，kernel 直接从 device 内存读取
        // 同时检查 alphaPtr 和 betaPtr，防止其中任一为 0 时解引用空指针
        if (tiling->alphaPtr != 0ULL && tiling->betaPtr != 0ULL) {
            alpha_ = *(__gm__ const float *)tiling->alphaPtr;
            beta_  = *(__gm__ const float *)tiling->betaPtr;
        } else {
            alpha_ = tiling->alpha;
            beta_  = tiling->beta;
        }
    }

    __aicore__ inline void Process()
    {
        int32_t rowStart{}, rowEnd{};
        uint32_t simtThreadNum{};
        ComputeRowRangeAndThreads(rowStart, rowEnd, simtThreadNum);
        if (binEdge_ != nullptr) {
            uint32_t blockId = AscendC::GetBlockIdx();
            rowStart = binEdge_[blockId];
            rowEnd   = binEdge_[blockId + 1];
            int32_t coreRangeLen = rowEnd - rowStart;
            simtThreadNum = (coreRangeLen < static_cast<int32_t>(kSpmvOpMaxThreadsPerBlock))
                ? static_cast<uint32_t>(coreRangeLen) : kSpmvOpMaxThreadsPerBlock;
            if (simtThreadNum == 0) { simtThreadNum = 1; }
            simtThreadNum = AlignToWarp(simtThreadNum);
        }
        const int32_t threadsPerCore = static_cast<int32_t>(simtThreadNum);
        const bool useReorder = (algType_ == SPMV_OP_ALG_TYPE_2);
        DispatchCompute(simtThreadNum, useReorder, rowStart, rowEnd, threadsPerCore);
    }

    __aicore__ inline void DispatchCompute(
        uint32_t simtThreadNum, bool useReorder,
        int32_t rowStart, int32_t rowEnd, int32_t threadsPerCore)
    {
        if (rowOffsetType_ == SPMV_OP_IDX_RT_I32) {
            DispatchI32(simtThreadNum, useReorder, rowStart, rowEnd, threadsPerCore);
        } else {
            DispatchI64(simtThreadNum, useReorder, rowStart, rowEnd, threadsPerCore);
        }
    }

    __aicore__ inline void DispatchI32(
        uint32_t simtThreadNum, bool useReorder,
        int32_t rowStart, int32_t rowEnd, int32_t threadsPerCore)
    {
        auto *roff = (__gm__ const int32_t *)rowOffsetsAddr_;
        if (useReorder) {
            asc_vf_call<SpmvOpSimtCompute<int32_t, true>>(
                dim3{simtThreadNum}, roff, colInd_, values_,
                xVec_, yVec_, zVec_, reorder_,
                alpha_, beta_, indexBase_, rowStart, rowEnd, threadsPerCore);
        } else {
            asc_vf_call<SpmvOpSimtCompute<int32_t, false>>(
                dim3{simtThreadNum}, roff, colInd_, values_,
                xVec_, yVec_, zVec_, nullptr,
                alpha_, beta_, indexBase_, rowStart, rowEnd, threadsPerCore);
        }
    }

    __aicore__ inline void DispatchI64(
        uint32_t simtThreadNum, bool useReorder,
        int32_t rowStart, int32_t rowEnd, int32_t threadsPerCore)
    {
        auto *roff = (__gm__ const int64_t *)rowOffsetsAddr_;
        if (useReorder) {
            asc_vf_call<SpmvOpSimtCompute<int64_t, true>>(
                dim3{simtThreadNum}, roff, colInd_, values_,
                xVec_, yVec_, zVec_, reorder_,
                alpha_, beta_, indexBase_, rowStart, rowEnd, threadsPerCore);
        } else {
            asc_vf_call<SpmvOpSimtCompute<int64_t, false>>(
                dim3{simtThreadNum}, roff, colInd_, values_,
                xVec_, yVec_, zVec_, nullptr,
                alpha_, beta_, indexBase_, rowStart, rowEnd, threadsPerCore);
        }
    }

private:
    GM_ADDR rowOffsetsAddr_{nullptr};  // 按 rowOffsetType_ 在 Process() 中按需 cast
    __gm__ const int32_t *colInd_{nullptr};
    __gm__ const float   *values_{nullptr};
    __gm__ const float   *xVec_{nullptr};
    __gm__ const float   *yVec_{nullptr};
    __gm__ float         *zVec_{nullptr};
    __gm__ const int32_t *reorder_{nullptr};
    __gm__ const int32_t *binEdge_{nullptr};
};

// ===========================================================================
// Kernel 1: __global__ 调度器（主计算）
// ===========================================================================

extern "C" __global__ __aicore__ void spmv_op_kernel(
    GM_ADDR gmRowOffsets, GM_ADDR gmColInd, GM_ADDR gmValues,
    GM_ADDR gmXVec, GM_ADDR gmYVec, GM_ADDR gmZVec,
    GM_ADDR gmReorder,
    GM_ADDR gmBinEdge,
    const SpmvOpTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpmvOpDispatcher dispatcher;
    dispatcher.Init(gmRowOffsets, gmColInd, gmValues,
                    gmXVec, gmYVec, gmZVec,
                    gmReorder, gmBinEdge, &tiling);
    dispatcher.Process();
}

// kernel_do 启动器（主计算）
extern "C" void spmv_op_kernel_do(
    GM_ADDR rowOffsets, GM_ADDR colInd, GM_ADDR values,
    GM_ADDR xVec, GM_ADDR yVec, GM_ADDR zVec,
    GM_ADDR reorder,
    GM_ADDR binEdge,
    const SpmvOpTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    spmv_op_kernel<<<numBlocks, nullptr, stream>>>(
        rowOffsets, colInd, values, xVec, yVec, zVec,
        reorder, binEdge, tiling);
}

// ===========================================================================
// Kernel 2: BetaOnly SIMT VF（nnz == 0 时：Z = beta * Y）
// ===========================================================================

__simt_vf__ __aicore__ __launch_bounds__(kSpmvOpMaxThreadsPerBlock) inline void
SpmvOpBetaYSimtCompute(
    __gm__ const float *yVec,
    __gm__ float *zVec,
    float beta,
    int32_t m,
    int32_t rowStart, int32_t rowEnd,
    int32_t threadsPerCore)
{
    const int32_t rangeLen = rowEnd - rowStart;
    for (int32_t row = static_cast<int32_t>(threadIdx.x);
         row < rangeLen;
         row += threadsPerCore) {
        const int32_t r = rowStart + row;
        const float yVal = (beta == 0.0f) ? 0.0f : yVec[r];
        zVec[r] = beta * yVal;
    }
}

// BetaOnly Dispatcher
struct SpmvOpBetaYDispatcher : public SpmvOpDispatcherBase {
    __gm__ const float *yVec_{nullptr};
    __gm__ float       *zVec_{nullptr};

    __aicore__ inline void Init(
        GM_ADDR gmYVec, GM_ADDR gmZVec,
        const SpmvOpTilingData *tiling)
    {
        yVec_ = (__gm__ const float *)gmYVec;
        zVec_ = (__gm__ float *)gmZVec;
        m_ = tiling->m;
        rowsPerBlock_ = tiling->rowsPerBlock;
        // 与主 Dispatcher 保持一致：检查 alphaPtr 和 betaPtr 均非 0 时从 GM 读
        // BetaY 不使用 alpha 值，但 host 侧已设置 alphaPtr，故检查条件一致
        if (tiling->alphaPtr != 0ULL && tiling->betaPtr != 0ULL) {
            beta_ = *(__gm__ const float *)tiling->betaPtr;
        } else {
            beta_ = tiling->beta;
        }
    }

    __aicore__ inline void Process()
    {
        int32_t rowStart{};
        int32_t rowEnd{};
        uint32_t simtThreadNum{};
        ComputeRowRangeAndThreads(rowStart, rowEnd, simtThreadNum);
        asc_vf_call<SpmvOpBetaYSimtCompute>(
            dim3{simtThreadNum},
            yVec_, zVec_, beta_, m_,
            rowStart, rowEnd,
            static_cast<int32_t>(simtThreadNum));
    }
};

// __global__ 调度器（BetaOnly）
extern "C" __global__ __aicore__ void spmv_op_beta_y_kernel(
    GM_ADDR gmYVec, GM_ADDR gmZVec,
    const SpmvOpTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpmvOpBetaYDispatcher dispatcher;
    dispatcher.Init(gmYVec, gmZVec, &tiling);
    dispatcher.Process();
}

// kernel_do 启动器（BetaOnly：nnz==0 快捷路径）
extern "C" void spmv_op_beta_y_kernel_do(
    GM_ADDR yVec, GM_ADDR zVec,
    const SpmvOpTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    spmv_op_beta_y_kernel<<<numBlocks, nullptr, stream>>>(yVec, zVec, tiling);
}

// ===========================================================================
// Kernel 3: ALG2 预处理（device-side，单线程）
//
// 在 device 上完成：
//   1. 读取 rowOffsets，计算每行 nnz，写入 scratch[m]
//   2. 初始化 reorder[i] = i，对 (nnz, row_index) 做降序归并排序 O(m log m)
//   3. 用排序后的 nnz 计算 bin_edge[numBlocks+1]（按 nnz 总量均匀切分）
//   4. 输出 reorder 和 bin_edge 到 user buffer
//
// 使用 workspace 内的 tmpReorder 和 tmpScratch 区作为归并排序的临时缓冲区。
// 单线程顺序执行，复杂度 O(m log m)，支持任意规模矩阵。
// ===========================================================================

__aicore__ inline void SpmvOpPreprocessPass1_I32(
    __gm__ const int32_t *rowOffsets, int32_t m,
    __gm__ int32_t *reorder, __gm__ int32_t *scratch)
{
    for (int32_t i = 0; i < m; ++i) {
        scratch[i] = static_cast<int32_t>(rowOffsets[i + 1] - rowOffsets[i]);
        reorder[i] = i;
    }
}

__aicore__ inline void SpmvOpPreprocessPass1_I64(
    __gm__ const int64_t *rowOffsets, int32_t m,
    __gm__ int32_t *reorder, __gm__ int32_t *scratch)
{
    for (int32_t i = 0; i < m; ++i) {
        scratch[i] = static_cast<int32_t>(rowOffsets[i + 1] - rowOffsets[i]);
        reorder[i] = i;
    }
}

// 内部归并函数：合并两个有序区间 [left, mid) 和 [mid, right)
__aicore__ inline void SpmvOpMergeInternal(
    int32_t left, int32_t mid, int32_t right,
    __gm__ int32_t *reorder, __gm__ int32_t *scratch,
    __gm__ int32_t *tmpReorder, __gm__ int32_t *tmpNnz)
{
    int32_t i = left, j = mid, k = left;
    while (i < mid && j < right) {
        if (scratch[i] >= scratch[j]) {
            tmpReorder[k] = reorder[i];
            tmpNnz[k] = scratch[i];
            i++; k++;
        } else {
            tmpReorder[k] = reorder[j];
            tmpNnz[k] = scratch[j];
            j++; k++;
        }
    }
    while (i < mid) {
        tmpReorder[k] = reorder[i];
        tmpNnz[k] = scratch[i];
        i++; k++;
    }
    while (j < right) {
        tmpReorder[k] = reorder[j];
        tmpNnz[k] = scratch[j];
        j++; k++;
    }
    for (int32_t idx = left; idx < right; ++idx) {
        reorder[idx] = tmpReorder[idx];
        scratch[idx] = tmpNnz[idx];
    }
}

// 自底向上归并排序（O(m log m)），使用外部提供的临时缓冲区
__aicore__ inline void SpmvOpPreprocessPass2_Sort(
    int32_t m, __gm__ int32_t *reorder, __gm__ int32_t *scratch,
    __gm__ int32_t *tmpReorder, __gm__ int32_t *tmpNnz)
{
    for (int32_t width = 1; width < m; width *= 2) {
        for (int32_t i = 0; i < m; i += 2 * width) {
            int32_t left = i;
            int32_t mid = (i + width < m) ? (i + width) : m;
            int32_t right = (i + 2 * width < m) ? (i + 2 * width) : m;
            SpmvOpMergeInternal(left, mid, right, reorder, scratch, tmpReorder, tmpNnz);
        }
    }
}

__aicore__ inline void SpmvOpPreprocessPass3_BinEdge(
    int32_t m, uint32_t numBlocks,
    __gm__ int32_t *scratch, __gm__ int32_t *binEdge)
{
    int64_t totalNnz = 0;
    for (int32_t i = 0; i < m; ++i) {
        totalNnz += scratch[i];
    }
    const int64_t nnzPerBlock = (totalNnz > 0 && numBlocks > 0)
        ? (totalNnz + static_cast<int64_t>(numBlocks) - 1) / static_cast<int64_t>(numBlocks)
        : 0;
    binEdge[0] = 0;
    uint32_t binIdx = 1;
    int64_t runningNnz = 0;
    for (int32_t i = 0; i < m; ++i) {
        runningNnz += scratch[i];
        if (binIdx < numBlocks && runningNnz >= nnzPerBlock * static_cast<int64_t>(binIdx)) {
            binEdge[binIdx] = i + 1;
            binIdx++;
        }
    }
    while (binIdx <= numBlocks) {
        binEdge[binIdx++] = m;
    }
}

__aicore__ inline void SpmvOpPreprocessCompute(
    __gm__ const int32_t *rowOffsets32,
    __gm__ const int64_t *rowOffsets64,
    int32_t rowOffsetType,
    int32_t m,
    uint32_t numBlocks,
    __gm__ int32_t *reorder,
    __gm__ int32_t *tmpReorder,
    __gm__ int32_t *scratch,
    __gm__ int32_t *tmpScratch,
    __gm__ int32_t *binEdge)
{
    if (rowOffsetType == SPMV_OP_IDX_RT_I32) {
        SpmvOpPreprocessPass1_I32(rowOffsets32, m, reorder, scratch);
    } else {
        SpmvOpPreprocessPass1_I64(rowOffsets64, m, reorder, scratch);
    }
    // O(m log m) 归并排序，支持任意规模矩阵，无需 m 上限保护
    SpmvOpPreprocessPass2_Sort(m, reorder, scratch, tmpReorder, tmpScratch);
    SpmvOpPreprocessPass3_BinEdge(m, numBlocks, scratch, binEdge);
}

// Dispatcher（单线程，不使用 SIMT 多线程模型）
class SpmvOpPreprocessDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmRowOffsets,
        GM_ADDR gmReorder,
        GM_ADDR gmTmpReorder,
        GM_ADDR gmScratch,
        GM_ADDR gmTmpScratch,
        GM_ADDR gmBinEdge,
        int32_t m, uint32_t numBlocks, int32_t rowOffsetType)
    {
        rowOffsets32_ = (__gm__ const int32_t *)gmRowOffsets;
        rowOffsets64_ = (__gm__ const int64_t *)gmRowOffsets;
        reorder_      = (__gm__ int32_t *)gmReorder;
        tmpReorder_   = (__gm__ int32_t *)gmTmpReorder;
        scratch_      = (__gm__ int32_t *)gmScratch;
        tmpScratch_   = (__gm__ int32_t *)gmTmpScratch;
        binEdge_      = (__gm__ int32_t *)gmBinEdge;
        m_            = m;
        numBlocks_    = numBlocks;
        rowOffsetType_ = rowOffsetType;
    }

    __aicore__ inline void Process()
    {
        SpmvOpPreprocessCompute(
            rowOffsets32_, rowOffsets64_, rowOffsetType_,
            m_, numBlocks_,
            reorder_, tmpReorder_, scratch_, tmpScratch_, binEdge_);
    }

private:
    __gm__ const int32_t *rowOffsets32_{nullptr};
    __gm__ const int64_t *rowOffsets64_{nullptr};
    __gm__ int32_t *reorder_{nullptr};
    __gm__ int32_t *tmpReorder_{nullptr};
    __gm__ int32_t *scratch_{nullptr};
    __gm__ int32_t *tmpScratch_{nullptr};
    __gm__ int32_t *binEdge_{nullptr};
    int32_t m_{0};
    uint32_t numBlocks_{0};
    int32_t rowOffsetType_{0};
};

// __global__ 调度器（预处理，单 block 单线程）
extern "C" __global__ __aicore__ void spmv_op_preprocess_kernel(
    GM_ADDR gmRowOffsets,
    GM_ADDR gmReorder,
    GM_ADDR gmTmpReorder,
    GM_ADDR gmScratch,
    GM_ADDR gmTmpScratch,
    GM_ADDR gmBinEdge,
    int32_t m,
    uint32_t numBlocks,
    int32_t rowOffsetType)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpmvOpPreprocessDispatcher dispatcher;
    dispatcher.Init(
        gmRowOffsets, gmReorder, gmTmpReorder, gmScratch, gmTmpScratch, gmBinEdge,
        m, numBlocks, rowOffsetType);
    dispatcher.Process();
}

// kernel_do 启动器（预处理：ALG2 异步预处理，单 block）
extern "C" void spmv_op_preprocess_kernel_do(
    GM_ADDR rowOffsets,
    GM_ADDR reorder,
    GM_ADDR tmpReorder,
    GM_ADDR scratch,
    GM_ADDR tmpScratch,
    GM_ADDR binEdge,
    int32_t m,
    uint32_t numBlocks,
    int32_t rowOffsetType,
    void *stream)
{
    spmv_op_preprocess_kernel<<<1, nullptr, stream>>>(
        rowOffsets, reorder, tmpReorder, scratch, tmpScratch, binEdge,
        m, numBlocks, rowOffsetType);
}
