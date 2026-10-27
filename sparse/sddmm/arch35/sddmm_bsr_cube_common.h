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
 * \file sddmm_bsr_cube_common.h
 * \brief Ascend 950 BSR SDDMM Cube fast-path predicates and launchers.
 */

#ifndef SDDMM_BSR_CUBE_COMMON_H_
#define SDDMM_BSR_CUBE_COMMON_H_

#include <cstdint>
#include <limits>

#include "sddmm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

namespace sddmm_arch35 {

constexpr uint64_t kSddmmBsrCubeScratchAlignment = 64ULL;
constexpr uint64_t kSddmmBsrCubeScratchLimit = 64ULL * 1024ULL * 1024ULL;

// The Cube stage writes one logical row-major FP32 b x b tile for every
// physical BSR block and batch. A return value of zero denotes an empty shape
// or an arithmetic overflow and must not be used as a workspace allocation.
inline uint64_t SddmmBsrCubeScratchBytes(const SddmmTilingData &tiling)
{
    if (tiling.batchCount <= 0 || tiling.nnzUnits <= 0 || tiling.blockDim <= 0) {
        return 0ULL;
    }
    const uint64_t batchCount = static_cast<uint64_t>(tiling.batchCount);
    const uint64_t blockCount = static_cast<uint64_t>(tiling.nnzUnits);
    const uint64_t blockDim = static_cast<uint64_t>(tiling.blockDim);
    constexpr uint64_t kElementBytes = sizeof(float);
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    if (blockDim > kMax / blockDim) {
        return 0ULL;
    }
    const uint64_t blockArea = blockDim * blockDim;
    if (blockCount > kMax / blockArea) {
        return 0ULL;
    }
    const uint64_t valuesPerBatch = blockCount * blockArea;
    if (batchCount > kMax / valuesPerBatch) {
        return 0ULL;
    }
    const uint64_t values = batchCount * valuesPerBatch;
    if (values > kMax / kElementBytes) {
        return 0ULL;
    }
    return values * kElementBytes;
}

// Cube tiles require a full 16-element output dimension and a reduction
// aligned to every supported input C0. Unsupported layouts, dimensions and
// complex64 calls remain on the generic SDDMM path.
inline bool SddmmBsrCubeCanRun(const SddmmTilingData &tiling)
{
    const bool supportedBlock = tiling.blockDim == 16 ||
        tiling.blockDim == 32 || tiling.blockDim == 64;
    const bool supportedReduction = tiling.k == 32 ||
        tiling.k == 64 || tiling.k == 128;
    const bool supportedInput = tiling.inputType == SDDMM_VALUE_FP16 ||
        tiling.inputType == SDDMM_VALUE_BF16 || tiling.inputType == SDDMM_VALUE_FP32;
    const bool supportedOutput =
        (tiling.inputType == SDDMM_VALUE_FP32 && tiling.outputType == SDDMM_VALUE_FP32) ||
        (tiling.inputType == SDDMM_VALUE_FP16 &&
            (tiling.outputType == SDDMM_VALUE_FP16 || tiling.outputType == SDDMM_VALUE_FP32)) ||
        (tiling.inputType == SDDMM_VALUE_BF16 &&
            (tiling.outputType == SDDMM_VALUE_BF16 || tiling.outputType == SDDMM_VALUE_FP32));
    const bool validShape = tiling.m > 0 && tiling.n > 0 &&
        tiling.blockRows > 0 && tiling.blockCols > 0 && tiling.blockDim > 0 &&
        tiling.m % tiling.blockDim == 0 && tiling.n % tiling.blockDim == 0 &&
        tiling.blockRows == tiling.m / tiling.blockDim &&
        tiling.blockCols == tiling.n / tiling.blockDim;
    const bool validBatch = tiling.batchCount > 0 &&
        (tiling.batchCountX == 1 || tiling.batchCountX == tiling.batchCount) &&
        (tiling.batchCountY == 1 || tiling.batchCountY == tiling.batchCount);
    return tiling.format == SDDMM_FORMAT_BSR && supportedBlock &&
        supportedInput && supportedOutput && tiling.computeType == SDDMM_VALUE_FP32 &&
        supportedReduction && tiling.opX == 0 && tiling.opY == 0 &&
        tiling.orderX == 0 && tiling.orderY == 0 && tiling.ldX == tiling.k &&
        tiling.ldY == tiling.n && validShape && validBatch &&
        (tiling.indexBase == 0 || tiling.indexBase == 1) &&
        (tiling.blockOrder == 0 || tiling.blockOrder == 1) &&
        tiling.pointerMode == 0 &&
        tiling.usePackedY == 0 && tiling.nnzUnits > 0 &&
        SddmmBsrCubeScratchBytes(tiling) != 0ULL &&
        SddmmBsrCubeScratchBytes(tiling) <= kSddmmBsrCubeScratchLimit;
}

} // namespace sddmm_arch35

extern "C" {

// Stage 1: BSR block GEMMs, FP32 scratch output. cubeBlockDim must not exceed
// the available AIC count for the selected device.
void sddmm_bsr_cube_compute_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns,
    GM_ADDR matX, GM_ADDR matY, GM_ADDR scratch,
    const SddmmTilingData &tiling, uint32_t cubeBlockDim, void *stream);

// Stage 2: alpha/beta epilogue, block ROW/COL conversion, and output cast.
void sddmm_bsr_cube_epilogue_kernel_do(
    GM_ADDR sparseValues, GM_ADDR scratch,
    const SddmmTilingData &tiling, uint32_t vectorBlockDim, void *stream);

// Convenience launcher. Both kernels are submitted to the same stream, so no
// host-side synchronization is required between the stages.
void sddmm_bsr_cube_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns, GM_ADDR sparseValues,
    GM_ADDR matX, GM_ADDR matY, GM_ADDR scratch,
    const SddmmTilingData &tiling,
    uint32_t cubeBlockDim, uint32_t vectorBlockDim, void *stream);

} // extern "C"

#endif // SDDMM_BSR_CUBE_COMMON_H_
