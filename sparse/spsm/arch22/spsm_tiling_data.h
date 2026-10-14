/**
 * Copyright (c) 2026 liutaw.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSM_ARCH22_TILING_DATA_H
#define SPSM_ARCH22_TILING_DATA_H
#include <cstdint>
struct SpsmA2TilingData {
    int64_t ldb;
    int64_t ldc;
    int32_t rows;
    int32_t rhs;
    int32_t nnz;
    int32_t base;
    int32_t upper;
    int32_t unit;
    int32_t deviceAlpha;
    // Status slots written by the preprocessing pipeline (format/canonical),
    // scanned by the solve kernel. Same count as PlanConfig::blocks.
    int32_t statusBlocks;
    float alpha;
    float alphaImag{0};
    int32_t complex{0};
    // 1 pre-solves independent rows; 2 skips them; 3 parallel level; 4 serial level range.
    int32_t independentMode{0};
    int32_t level{0};
    // Same-stream dispatch condition; values are documented in kernel/spsm_dispatch.h.
    int32_t packedFallback{0};
};
constexpr int32_t kSpsmA2RhsTile = 64;
constexpr int32_t kSpsmA2SolveColumns = 8;
constexpr int32_t kSpsmA2LevelRows = 8192;
constexpr int32_t kSpsmA2MaxLevels = 64;
constexpr int32_t kSpsmA2NnzChunk = 512;
constexpr int32_t kSpsmA2StatusStride = 16;

struct SpsmA2FormatTilingData {
    int32_t rows;
    int32_t nnz;
    int32_t base;
    int32_t format; // CSR=0, CSC=1, COO=2
    int32_t operation; // N=0, T=1, H=2
    int32_t upper; // Original A triangle, before operation.
    int32_t complex;
    int32_t blocks;
};

struct SpsmA2SortTilingData {
    int64_t run;
    int32_t nnz;
    int32_t complex;
    int32_t blocks;
};

struct SpsmA2CanonicalTilingData {
    int32_t rows;
    int32_t nnz;
    int32_t complex;
    int32_t unit;
    int32_t blocks;
};

struct SpsmA2RefreshTilingData {
    int32_t nnz;
    int32_t complex;
    int32_t diagonal;
    int32_t blocks;
};

struct SpsmA2DenseTilingData {
    int64_t srcRowStride;
    int64_t srcColStride;
    int64_t dstRowStride;
    int64_t dstColStride;
    int32_t rows;
    int32_t cols;
    int32_t complex;
    int32_t conjugate;
    int32_t statusBlocks;
    int32_t packedFallback{0};
};

struct SpsmA2GatherTilingData {
    int32_t rows;
    int32_t nnz;
    int32_t complex;
    int32_t unit;
    int32_t statusBlocks;
};
#endif
