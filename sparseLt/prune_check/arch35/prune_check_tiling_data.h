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

#ifndef SPARSELT_PRUNE_CHECK_TILING_DATA_H_
#define SPARSELT_PRUNE_CHECK_TILING_DATA_H_

#include <cstdint>

// PruneCheck kernel 的 tiling 参数：host 侧构造，by-value 经 kernel launch args
// 传递（无 aclrtMalloc 的 tilingGm，满足 no extra storage / async 语义）。
// 尺寸/偏移使用 64 位整数；UB 内部尺寸受 UB 容量限制，使用 32 位。
// 分组大小不落入 tiling：由 kernel 按 sizeof(U) 编译期推导
//（4 字节 → g=2 即 1:2，其余 → g=4 即 2:4），与 host 侧 dtype 白名单一一对应。
struct PruneCheckTilingData {
    // 扫描几何（物理视角，d_in 一律为行主序物理存储、行步长 ld 元素）：
    //   alongRow=true（RowScan）：rows×cols，每行内 g 个相邻元素为一组；
    //   alongRow=false（ColScan）：rows×cols，同列相邻 g 行为一组（步长 ld）。
    //   pruneAlongRow=0 且稀疏侧 trans=1 的派生组合退化为 RowScan（rows=k, cols=m），
    //   由 host 在构造 tiling 前完成映射，kernel 只见两种扫描几何。
    // 批量语义：d_valid = OR(全部 batch 的判定)，每 batch 基址偏移
    //   batchIdx×batchStride 元素；单 batch 判定逻辑与批量支持前完全一致。
    uint64_t rows = 0;             // 扫描行数（RowScan: m 或 k；ColScan: m）
    uint64_t cols = 0;             // 每行有效列数（RowScan: k 或 m；ColScan: k）
    uint64_t ld = 0;               // 物理行步长（元素数）
    uint64_t batchStride = 0;      // batch 步长（元素数）；numBatches==1 时恒 0（不消费）
    uint64_t numBatches = 1;       // batch 数（≥ 1，host 校验保证）
    uint64_t tilesPerMatrix = 0;   // 单 batch tile 总数 = ceil(rows/tileRows) × tilesPerRowBand
    uint64_t totalTiles = 0;       // batch × 行带 × 列块总数（host 侧 blockDim 上界）
    uint32_t dataBytes = 0;        // 元素字节宽：4/2/1 → u32/u16/u8 kernel 分派
    bool alongRow = false;         // true=RowScan（行内连续分组），false=ColScan（列向跨行分组）
    uint32_t tileRows = 0;         // 行带行数（固定 16，保证 ColScan 垂直组不跨行带）
    uint32_t tileCols = 0;         // 列块列数（lanes 整数倍，lanes = 256B/dataBytes）
    uint32_t inputPitch = 0;       // UB 行 pitch（字节，256B 对齐）
    uint32_t inputBufferBytes = 0; // tileRows × inputPitch（单 tile 搬运槽大小）
    uint32_t tilesPerRowBand = 0;  // ceil(cols / tileCols)
};

#endif // SPARSELT_PRUNE_CHECK_TILING_DATA_H_
