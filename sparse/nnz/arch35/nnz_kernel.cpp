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
 * \file nnz_kernel.cpp
 * \brief aclsparseSnnz kernel 实现（SIMD 与 SIMT 混合编程）。
 *
 * 层次结构：
 *   nnz_kernel_do  →  nnz_kernel<<<>>>  →  NnzSimtLauncher::Process()  →  asc_vf_call
 *
 * 参考 spmm_kernel.cpp 的 class-based 混合编程模式。
 */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "nnz_kernel.h"

namespace {
constexpr int32_t kWarpSize = 32;
constexpr int32_t kDirRow = 0;
} // namespace

// ---------------------------------------------------------------------------
// SIMT VF 函数
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ __launch_bounds__(kNnzThreadsPerBlock) inline void
NnzSimtVfCompute(
    __gm__ float *A,
    __gm__ int32_t *nnzPerRowColumn,
    __gm__ int32_t *nnzTotal,
    int32_t lda, int32_t dirA, int32_t totalUnits, int32_t unitSize,
    int32_t numBlocks)
{
    int32_t globalTid = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t gridStride = blockDim.x * numBlocks;
    int32_t localCount = 0;
    int32_t laneId = threadIdx.x % kWarpSize;
    int64_t lda64 = static_cast<int64_t>(lda);

    if (dirA == kDirRow) {
        // ROW: 一线程一行，warp 内线程访问连续行地址实现合并访问
        for (int32_t unitIdx = globalTid; unitIdx < totalUnits; unitIdx += gridStride) {
            int32_t count = 0;
            int64_t base = static_cast<int64_t>(unitIdx);
            for (int32_t j = 0; j < unitSize; j++) {
                if (A[j * lda64 + base] != 0.0f) { count++; }
            }
            nnzPerRowColumn[unitIdx] = count;
            localCount += count;
        }
    } else {
        // COLUMN: warp 协作，kWarpSize 线程处理同一列的不同行，最后 warp 归约
        int32_t warpIdx = globalTid / kWarpSize;
        int32_t numWarpsPerGrid = gridStride / kWarpSize;
        for (int32_t colIdx = warpIdx; colIdx < totalUnits; colIdx += numWarpsPerGrid) {
            int32_t count = 0;
            int64_t base = static_cast<int64_t>(colIdx) * lda64;
            for (int32_t i = laneId; i < unitSize; i += kWarpSize) {
                if (A[base + i] != 0.0f) { count++; }
            }
            int32_t colNnz = asc_reduce_add(count);
            if (laneId == 0) { nnzPerRowColumn[colIdx] = colNnz; }
            localCount += count;
        }
    }

    int32_t warpSum = asc_reduce_add(localCount);
    if (laneId == 0 && warpSum != 0) {
        asc_atomic_add(nnzTotal, warpSum);
    }
}

// ---------------------------------------------------------------------------
// SIMD 侧封装类：管理 GM 指针 + 调用 asc_vf_call（对齐 spmm 模式）
// ---------------------------------------------------------------------------
class NnzSimtLauncher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmA, GM_ADDR gmNnzPerRow, GM_ADDR gmNnzTotal,
        int32_t lda, int32_t dirA, int32_t totalUnits, int32_t unitSize,
        int32_t numBlocks)
    {
        A_ = (__gm__ float *)gmA;
        nnzPerRowColumn_ = (__gm__ int32_t *)gmNnzPerRow;
        nnzTotal_ = (__gm__ int32_t *)gmNnzTotal;
        lda_ = lda;
        dirA_ = dirA;
        totalUnits_ = totalUnits;
        unitSize_ = unitSize;
        numBlocks_ = numBlocks;
    }

    __aicore__ inline void Process()
    {
        // 线程数自适应：totalUnits < kNnzThreadsPerBlock 时缩减线程数。
        // grid-stride 保证 totalUnits 较大时全部 unit 被覆盖。
        // simtThreadNum=0 的防御性保护：正常路径 Host 已拦截 totalUnits=0，
        // 此处兜底确保 asc_vf_call 的 dim3 参数 ≥ 1。
        // 向上取整到 kWarpSize 的倍数：确保 warp 内所有线程均为 active 状态，
        // 避免 asc_reduce_add 在部分 warp 下行为不确定。取整后可能有多余线程
        // 空转（localCount=0），asc_reduce_add 对这些线程贡献 0 值，不影响结果。
        uint32_t simtThreadNum = kNnzThreadsPerBlock;
        if (static_cast<uint32_t>(totalUnits_) < simtThreadNum) {
            simtThreadNum = static_cast<uint32_t>(totalUnits_);
        }
        if (simtThreadNum == 0) {
            simtThreadNum = 1;
        }
        simtThreadNum = (simtThreadNum + static_cast<uint32_t>(kWarpSize) - 1u) & ~(static_cast<uint32_t>(kWarpSize) - 1u);

        asc_vf_call<NnzSimtVfCompute>(dim3{simtThreadNum},
            A_, nnzPerRowColumn_, nnzTotal_,
            lda_, dirA_, totalUnits_, unitSize_, numBlocks_);
    }

private:
    __gm__ float *A_{nullptr};
    __gm__ int32_t *nnzPerRowColumn_{nullptr};
    __gm__ int32_t *nnzTotal_{nullptr};
    int32_t lda_{0};
    int32_t dirA_{0};
    int32_t totalUnits_{0};
    int32_t unitSize_{0};
    int32_t numBlocks_{0};
};

// ---------------------------------------------------------------------------
// __global__ 入口：SIMD 侧 kernel 入口，通过 NnzSimtLauncher 类管理
// GM 指针和参数，在 Process() 内调用 asc_vf_call 启动 SIMT VF 函数。
// 对齐 spmm 的 class-based 混合编程模式。
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void nnz_kernel(
    GM_ADDR gmA, GM_ADDR gmNnzPerRow, GM_ADDR gmNnzTotal,
    int32_t lda, int32_t dirA, int32_t totalUnits, int32_t unitSize,
    int32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    NnzSimtLauncher launcher;
    launcher.Init(gmA, gmNnzPerRow, gmNnzTotal, lda, dirA, totalUnits, unitSize, numBlocks);
    launcher.Process();
}

// ---------------------------------------------------------------------------
// kernel_do：Host 侧调用入口
// ---------------------------------------------------------------------------
extern "C" void nnz_kernel_do(
    GM_ADDR A,
    GM_ADDR nnzPerRowColumn,
    GM_ADDR nnzTotal,
    NnzTilingData tiling,
    uint32_t numBlocks,
    void *stream)
{
    nnz_kernel<<<numBlocks, 0, stream>>>(
        A, nnzPerRowColumn, nnzTotal,
        tiling.lda, tiling.dirA, tiling.totalUnits, tiling.unitSize,
        static_cast<int32_t>(numBlocks));
}
