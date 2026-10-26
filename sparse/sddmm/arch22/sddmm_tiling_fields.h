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
#ifndef SDDMM_TILING_FIELDS_H_
#define SDDMM_TILING_FIELDS_H_

#include <cstdint>

// Shared element and byte budgets used by Host dispatch and Kernel buffers.
namespace SddmmBudget {
constexpr uint32_t kPatternChunk = 1024;
constexpr uint32_t kFlagStrideBytes = 32;
constexpr uint32_t kComplexSplitChunk = 512;
constexpr uint32_t kComplexIndexChunk = 256;
constexpr uint32_t kComplexOutputChunk = 1024;
constexpr uint32_t kComplexTileElements = 16u * 1024u;
constexpr uint32_t kComplexWindowElements = 32u * 1024u;
constexpr uint32_t kComplexDirectKLimit = 512;
constexpr uint32_t kRowHintsKLimit = 512;
constexpr uint32_t kCubeGatherElements = 512;
constexpr uint32_t kCubeEpilogueValues = 1024;
constexpr uint32_t kCubeWaveBytes = 16u * 1024u * 1024u;
constexpr uint32_t kCubeGroupMetaBytes = 256u * 1024u;
} // namespace SddmmBudget

enum SddmmDtypeCode : uint32_t {
    SDDMM_DTYPE_FP32 = 0,
    SDDMM_DTYPE_FP16 = 1,
    SDDMM_DTYPE_BF16 = 2,
    SDDMM_DTYPE_COMPLEX64 = 3,
};

// Shared wire fields. Host and Kernel retain distinct semantic types.
struct SddmmTilingFields {
    uint32_t nnz;
    uint32_t rowCount;
    uint32_t matARows;
    uint32_t matACols;
    uint32_t matBRows;
    uint32_t matBCols;
    uint32_t blockDim;
    uint32_t rowsPerCore;
    uint32_t remainderRows;
    uint32_t kBlockLen;
    uint32_t nnzPartition;
    uint32_t nnzPerCore;
    uint32_t remainderNnz;
    float alphaHost;
    float betaHost;
    // Imaginary parts of alpha/beta. Only the complex64 rows read them; the real
    // rows leave both at zero, so kernels that ignore them stay correct.
    float alphaImagHost;
    float betaImagHost;
    uint32_t opX;
    uint32_t opY;
    uint32_t ldx;
    uint32_t ldy;
    uint32_t orderX;
    uint32_t orderY;
    // Storage dtype of matX/matY and of the matC values. The two differ in the
    // mixed-precision rows of the support matrix (X/Y FP16 or BF16 against an
    // FP32 C); compute always accumulates in FP32 apart from the complex64
    // rows, so this pair alone selects the kernel entry point.
    uint32_t xType;
    uint32_t cType;
    // 0- or 1-based rowOffsets / colIndices. The kernel subtracts it before
    // using either array as a 0-based offset or column id.
    uint32_t idxBase;
    uint64_t rowHints;
    // Optional complete-pattern validation flags. Numeric execution may overlap
    // the host's validation wait, but must not write C if any core found a change.
    uint64_t patternFlags;
    uint32_t patternFlagWords;
};

#endif
