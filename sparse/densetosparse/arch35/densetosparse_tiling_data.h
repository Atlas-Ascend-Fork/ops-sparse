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

#ifndef DENSETOSPARSE_TILING_DATA_H_
#define DENSETOSPARSE_TILING_DATA_H_

#include <cstdint>

// Re-measured after the scans became warp-cooperative, which reversed the
// earlier result: more warps per block now buy memory parallelism.
// P-01 COO float32: 256 -> 3602us, 384 -> 3158us, 512 -> 2829us,
// 640 -> 3565us, 768 -> 3485us.
constexpr uint32_t kDenseToSparseThreads = 512;

struct DenseToSparseTilingData {
    uint64_t rows;
    uint64_t cols;
    uint64_t ld;
    uint64_t nnz;
    uint64_t ellBlockSize;
    uint64_t ellCols;
    uint64_t unitCount;
    // Elements of the minor dimension per CSR/CSC count unit. Chosen on the
    // host from the matrix shape: wide matrices want large chunks, narrow ones
    // need small chunks to produce enough units to fill the cores.
    uint64_t majorChunk;
    uint64_t statusOffset;
    uint64_t nnzOffset;
    uint64_t level0Offset;
    // Convert's count pass records, per 32-byte window, whether it holds any
    // non-zero, so the emit pass only has to load the windows that do. At the
    // target sparsity that turns the second read of the dense input into a
    // read of the bitmap plus a couple of percent of the data.
    uint64_t bitmapOffset;
    uint64_t bitmapWordsPerUnit;
    uint32_t format;
    uint32_t order;
    uint32_t base;
    uint32_t offsetType;
    uint32_t indexType;
    uint32_t elementBytes;
    uint32_t numBlocks;
    // When set, Convert derives every output position straight from `offsets`
    // and therefore skips the count+scan pass over the dense input. Only valid
    // for CSR/CSC with one unit per major, chosen by the host.
    uint32_t singlePassConvert;
    /// 0 = ignore the bitmap, 1 = count writes it, 2 = emit consumes it.
    uint32_t bitmapMode;
};

#endif // DENSETOSPARSE_TILING_DATA_H_
