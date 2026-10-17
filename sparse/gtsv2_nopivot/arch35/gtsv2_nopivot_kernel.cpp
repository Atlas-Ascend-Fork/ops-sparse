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
 * \file gtsv2_nopivot_kernel.cpp
 * \brief aclsparseSgtsv2Nopivot SIMT kernel (Thomas algorithm).
 *
 * 参考 gtsv_interleaved_batch 的 SIMT 实现模式。
 * 每个 thread 处理一个 RHS 列，直接读写 GM。
 *
 * Launcher 层: Init → Process → asc_vf_call<Gtsv2SimtCompute>
 * SIMT 层:     Gtsv2SimtCompute → Gtsv2ProcessOneRhs
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "gtsv2_nopivot_kernel.h"

// ---------------------------------------------------------------------------
// IEEE-754 division that keeps Inf/NaN on a singular pivot (issues #154/#155).
// Writing 0 on zero pivot would silently yield a wrong "solution"; nopivot API
// documents Inf/NaN instead (caller must supply a nonsingular system).
// ---------------------------------------------------------------------------
__simt_callee__ inline float GtsvIeeeDiv(float num, float den)
{
    if (den != 0.0f) {
        return num / den;
    }
    // 0/±0 -> NaN; nonzero/±0 -> ±Inf with IEEE sign(num) XOR sign(den).
    if (num == 0.0f) {
        return __builtin_nanf("");
    }
    const bool neg =
        static_cast<bool>(__builtin_signbit(num)) != static_cast<bool>(__builtin_signbit(den));
    return neg ? -__builtin_inff() : __builtin_inff();
}

// ---------------------------------------------------------------------------
// 每个 RHS 列的 Thomas 算法 (参考 gtsv_interleaved_batch 的 GtsvProcessOneBatch)
//
// Forward:  读取 GM 上的 dl/d/du/b，d'/b' 写回 ws/b 供 backward 用
// Backward: 从 ws/b 读取 d'/b'，解写回 B 的对应列
//
// GM layout:
//   dl/d/du: [m] — 所有 RHS 列共享
//   B:        [ldb * n] column-major — 第 rhsIdx 列 = B[rhsIdx * ldb : rhsIdx * ldb + m]
//   ws:       [m] per RHS column — forward 中间 d' 值
// ---------------------------------------------------------------------------
__simt_callee__ inline void Gtsv2ProcessOneRhs(
    __gm__ const float *dlGm,
    __gm__ const float *dGm,
    __gm__ const float *duGm,
    __gm__ float *BGm,
    __gm__ float *wsGm,
    int32_t m, int32_t ldb, int32_t rhsIdx)
{
    if (m < 1) { return; }

    // B is column-major (ldb x n): column j starts at B[j * ldb]
    int64_t colOff = static_cast<int64_t>(rhsIdx) * ldb;
    // Workspace is per-RHS m floats: RHS j starts at ws[j * m]
    int64_t wsOff  = static_cast<int64_t>(rhsIdx) * m;

    // ---- Phase 1: CopyIn (读 GM 到寄存器) + Thomas Forward ----
    // d_prev = d'[0], b_prev = b'[0], 在寄存器中传递
    float d_prev = dGm[0];
    float b_prev = BGm[colOff];

    // 行 1..m-1: forward sweep
    for (int32_t i = 1; i < m; i++) {
        int64_t idx = colOff + i;
        float dl_i = dlGm[i];
        float d_i  = dGm[i];
        float du_p = duGm[i - 1];
        float b_i  = BGm[idx];

        const float w = GtsvIeeeDiv(dl_i, d_prev);
        d_prev = d_i - w * du_p;
        b_prev = b_i - w * b_prev;

        // Phase 1.5: 写中间值到 ws (d') 和 B (b')，供 backward 用
        // 注: B 使用 ldb-stride, workspace 使用 m-stride（两者在 ldb>m 时可能不同）
        wsGm[wsOff + i] = d_prev;
        BGm[idx]        = b_prev;
    }

    // ---- Phase 2: Compute (Thomas Backward) ----
    // 行 m-1: x[m-1] = b'[m-1] / d'[m-1]
    {
        int64_t idxLast = colOff + (m - 1);
        float x_next = GtsvIeeeDiv(b_prev, d_prev);
        BGm[idxLast] = x_next;

        // 行 m-2..0: x[i] = (b'[i] - du[i] * x[i+1]) / d'[i]
        for (int32_t i = m - 2; i >= 0; i--) {
            int64_t idx = colOff + i;
            float b_prime = BGm[idx];           // b'[i] 在 forward 中写入 B
            float d_prime = (i > 0) ? wsGm[wsOff + i] : dGm[0];  // d'[0] = d[0] (未写入 ws)
            float du_i    = duGm[i];
            float x_i = GtsvIeeeDiv(b_prime - du_i * x_next, d_prime);
            x_next = x_i;

            // Phase 3: CopyOut (写解 x[i] 回 B)
            BGm[idx] = x_i;
        }
    }
}

// ---------------------------------------------------------------------------
// SIMT VF 入口: 每个 thread 处理一个 RHS 列 (grid-stride)
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ __launch_bounds__(kGtsv2ThreadsPerBlock) inline void
Gtsv2SimtCompute(
    __gm__ float *dlGm,
    __gm__ float *dGm,
    __gm__ float *duGm,
    __gm__ float *BGm,
    __gm__ float *wsGm,
    int32_t m, int32_t n, int32_t ldb, int32_t gridStrideInRhs)
{
    for (uint32_t rhsIdx = blockIdx.x * kGtsv2ThreadsPerBlock + threadIdx.x;
         rhsIdx < static_cast<uint32_t>(n);
         rhsIdx += static_cast<uint32_t>(gridStrideInRhs)) {
        Gtsv2ProcessOneRhs(dlGm, dGm, duGm, BGm, wsGm, m, ldb,
                           static_cast<int32_t>(rhsIdx));
    }
}

// ---------------------------------------------------------------------------
// Launcher: 管理 GM 指针，调用 asc_vf_call 启动 VF
// ---------------------------------------------------------------------------
class Gtsv2NopivotLauncher {
public:
    __aicore__ inline void Init(
        GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
        GM_ADDR workspace,
        const Gtsv2NopivotTilingData &tiling)
    {
        dlGm_ = reinterpret_cast<__gm__ float *>(dl);
        dGm_  = reinterpret_cast<__gm__ float *>(d);
        duGm_ = reinterpret_cast<__gm__ float *>(du);
        BGm_  = reinterpret_cast<__gm__ float *>(B);
        wsGm_ = reinterpret_cast<__gm__ float *>(workspace);
        m_ = tiling.m;
        n_ = tiling.n;
        ldb_ = tiling.ldb;
        gridStrideInRhs_ = static_cast<int32_t>(
            static_cast<uint32_t>(tiling.numBlocks) * kGtsv2ThreadsPerBlock);
    }

    __aicore__ inline void Process()
    {
        asc_vf_call<Gtsv2SimtCompute>(dim3{kGtsv2ThreadsPerBlock},
            dlGm_, dGm_, duGm_, BGm_, wsGm_, m_, n_, ldb_, gridStrideInRhs_);
    }

private:
    __gm__ float *dlGm_{nullptr};
    __gm__ float *dGm_{nullptr};
    __gm__ float *duGm_{nullptr};
    __gm__ float *BGm_{nullptr};
    __gm__ float *wsGm_{nullptr};
    int32_t m_{0};
    int32_t n_{0};
    int32_t ldb_{0};
    int32_t gridStrideInRhs_{0};
};

// ---------------------------------------------------------------------------
// __global__ 入口
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void gtsv2_nopivot_kernel(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
    GM_ADDR workspace,
    const Gtsv2NopivotTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Gtsv2NopivotLauncher launcher;
    launcher.Init(dl, d, du, B, workspace, tiling);
    launcher.Process();
}

void gtsv2_nopivot_kernel_do(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
    GM_ADDR workspace,
    const Gtsv2NopivotTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    gtsv2_nopivot_kernel<<<numBlocks, nullptr, stream>>>(
        dl, d, du, B, workspace, tiling);
}
