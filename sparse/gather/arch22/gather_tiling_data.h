/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file gather_tiling_data.h
 * \brief gather (arch22 / Atlas A2-A3) TilingData 结构体定义（Host / Kernel 共用）。
 *
 * Gather:  X.values[i] = Y[X.indices[i] - idxBase]  for i = 0 .. nnz-1.
 *
 * 与 arch22 scatter 保持一致：按 Vector Core 切分 nnz（coreNnzOffset / coreNnzCount），
 * 不额外分配 workspace，tiling 随 kernel 启动参数以 const 引用下发。
 */

#pragma once

#include <cstdint>

// 最大核数（与 ScatterTilingData.coreNnzOffset 数组容量一致，覆盖 Atlas A2/A3 的 Vector Core 数）。
constexpr uint32_t GATHER_MAX_CORE_NUM = 64;

// 单 tile 最大元素数（UB 容量约束；核内切分的实际 tileNn 为 <= 该值且为 2 的幂）。
constexpr uint32_t GATHER_TILE_NN_MAX = 4096;

// UB 总预算（字节）：Kernel 内**所有** InitBuffer 的合计容量不得超过该值。
// A3 (DAV-2201) 实测可用 UB ≈ 191.75 KiB，此处预留安全余量取 176 KiB。
// 最坏组合（complex64 + tileNn = GATHER_TILE_NN_MAX）：
//   idx 队列 2*tileNn*4 = 32 KiB + val 队列 tileNn*8 = 32 KiB = 64 KiB，
//   仍可留出 >= 112 KiB 给 Y 分块缓冲，不会撑爆 UB。
constexpr uint32_t GATHER_UB_BUDGET_BYTES = 176 * 1024;

// DataCopyPad 单块最大字节数：uint16 上限 65535 向下取到 32 字节对齐。
constexpr uint32_t GATHER_DC_BLOCK_BYTES_MAX = 65504;

// Gather TilingData（Host / Kernel 共用）
struct GatherTilingData {
    uint32_t nnz;        // 稀疏向量非零元个数
    uint32_t blockNum;   // 实际使用的核数（grid dim）
    uint32_t tileNn;     // 核内切分的元素数（2 的幂，min 8）
    uint32_t tileNnAligned8; // 值缓冲字节数 = tileNn * elemBytes（天然 8 字节对齐）
    uint32_t tileNnAligned4; // 索引缓冲字节数 = tileNn * sizeof(int32_t)（4 字节对齐）
    uint32_t elemBytes;  // 值元素字节宽：f16/bf16=2，f32=4，c64=8
    uint64_t yElems;     // 稠密向量 Y 的元素个数（用于 GM 边界）
    uint32_t idxBase;    // 索引基址：0 或 1
    uint32_t coreNnzOffset[GATHER_MAX_CORE_NUM]; // 每核在 nnz 上的起始偏移
    uint32_t coreNnzCount[GATHER_MAX_CORE_NUM];  // 每核处理的元素个数
};
