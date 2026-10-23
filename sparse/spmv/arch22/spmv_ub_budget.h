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

// SpMV kernel 的 UB 预算常量单一事实源（GEN-4.2）：host 侧 aclsparseSpMV 用
// SPMV_UB_BUDGET_BYTES 与 GetCoreMemSize(UB) 做一次性校验，kernel 侧直接用各常量。
// 只含 constexpr，双侧（host/device 编译）均可包含，不得 include kernel_operator.h。
#pragma once

#include <cstdint>

constexpr uint32_t BUFFER_NUM = 1;
// 行内/块内分块大小：nnz 按 tile 循环，支持任意行长（含长尾行）
constexpr uint32_t SPMV_TILE_ELEMS = 4096;     // 常规类型（f32/i32/f16/bf16/i8）
constexpr uint32_t SPMV_TILE_ELEMS_C64 = 1024; // complex64（元素为交错 float2）
// 每组处理的行数：行指针/累加器/y 读写均按组批量化，避免逐行 DMA 与流水线同步。
// 取 768 为 bufAccLo（Dot2 补偿累加器）腾出 UB 空间，保持 UB 总占用不增。
constexpr uint32_t SPMV_ROW_GROUP = 768;
// 行段长度达到该阈值时改用矢量 ReduceSum 树形归约（长尾行精度 + 提速）；
// 短于该阈值的行段走标量 Dot2 补偿累加（短行/消减场景的严格精度）
constexpr uint32_t SPMV_LONG_ROW_MIN = 128;
// UB 中 x 窗口/y 累加窗口的字节预算（910B UB 192KB，需为其它 tile 预留空间）
constexpr uint32_t SPMV_XWIN_BYTES = 96 * 1024;

// kernel UB 总占用上界（按各 kernel Init 的 TBuf 实际分配口径逐项列出，取三条
// 路径的最大值；各 sizeof 按 4B 上限计）。最坏为非转置 SpmvKernel 的
// f32 计算 + 2B 原始值（f16/bf16->f32）档：
constexpr uint32_t SPMV_UB_BUDGET_BYTES =
    (SPMV_ROW_GROUP + 1) * sizeof(int32_t) // bufRowPtr：行指针组缓冲
    + SPMV_ROW_GROUP * 4 * 5   // bufAcc/bufAccLo/bufYRaw/bufYC/bufYOut（累加器 + y 三份，按 4B 上限）
    + SPMV_TILE_ELEMS * 4 * 3  // bufColIdx/bufValsC/bufProd（索引 + 转换后值 + 积缓冲）
    + SPMV_TILE_ELEMS * 2      // bufValsRaw（2B 原始值上限；同类型档仅 32B 但 XWin 升到 128KB，总额更低）
    + 112 * 1024               // bufXWin（XWIN_ELEMS 上界：f32 计算 + 2B vals 档 112KB）
    + 2048 * 2                 // bufXRaw（XRAW_CHUNK=2048 × 2B 原始值）
    + 1024;                    // bufReduceWk（归约工作区）
// 其余两路均低于该值：转置 SpmvKernelTrans 最坏（f32 DS 双平面）≈149.5KB；
// SpmvKernelC64（INPUT_COMPLEX）≈172KB。arch22 UB 容量 192KB。
static_assert(SPMV_UB_BUDGET_BYTES == 195588, "UB budget formula out of sync with kernel Init");
