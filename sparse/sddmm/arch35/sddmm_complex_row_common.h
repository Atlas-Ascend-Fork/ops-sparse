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
 * \file sddmm_complex_row_common.h
 * \brief Ascend 950 complex64 CSR/BSR row-tile fast-path predicate and launcher.
 */

#ifndef SDDMM_ARCH35_COMPLEX_ROW_COMMON_H_
#define SDDMM_ARCH35_COMPLEX_ROW_COMMON_H_

#include <cstdint>
#include <limits>

#include "sddmm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

namespace sddmm_arch35 {

constexpr int32_t kSddmmComplexReduction = 128;
constexpr int32_t kSddmmComplexSamplesPerRow = 64;
constexpr int32_t kSddmmComplexLanesPerRow = 8;
constexpr int32_t kSddmmComplexOutputsPerLane =
    kSddmmComplexSamplesPerRow / kSddmmComplexLanesPerRow;

/*
 * The implementation intentionally accepts only the task's stable performance
 * layouts. Calls rejected here continue through the generic kernel.  The CSR
 * flag is produced by Preprocess after checking every row, so Execute never
 * guesses that a 64-entry row is safe from dimensions alone.
 */
inline bool SddmmComplexRowDenseLayout(const SddmmTilingData &tiling)
{
    return
        tiling.inputType == SDDMM_VALUE_COMPLEX64 &&
        tiling.outputType == SDDMM_VALUE_COMPLEX64 &&
        tiling.computeType == SDDMM_VALUE_COMPLEX64 &&
        tiling.k == kSddmmComplexReduction &&
        tiling.rowsX == tiling.m &&
        tiling.colsX == kSddmmComplexReduction &&
        tiling.rowsY == kSddmmComplexReduction &&
        tiling.colsY == tiling.n &&
        tiling.opX == 0 && tiling.opY == 0 &&
        tiling.orderX == 0 && tiling.orderY == 0 &&
        tiling.ldX == kSddmmComplexReduction &&
        tiling.ldY == tiling.n &&
        tiling.pointerMode == 0;
}

inline bool SddmmComplexRowBounds(const SddmmTilingData &tiling)
{
    return tiling.batchCount == 1 &&
        tiling.batchCountX == 1 && tiling.batchCountY == 1 &&
        tiling.offsetsBatchStride == 0 &&
        tiling.columnsBatchStride == 0 &&
        tiling.valuesBatchStride == 0 &&
        tiling.usePackedY == 0 &&
        (tiling.indexBase == 0 || tiling.indexBase == 1) &&
        tiling.m > 0 && tiling.n > 0 && tiling.nnzUnits > 0 &&
        tiling.m <= std::numeric_limits<int32_t>::max() &&
        tiling.n <= std::numeric_limits<int32_t>::max() /
            (2 * kSddmmComplexReduction) &&
        tiling.nnzUnits <= std::numeric_limits<int32_t>::max();
}

inline bool SddmmComplexRowCanRun(const SddmmTilingData &tiling)
{
    if (!SddmmComplexRowDenseLayout(tiling) || !SddmmComplexRowBounds(tiling)) {
        return false;
    }

    if (tiling.format == SDDMM_FORMAT_CSR) {
        return tiling.outputsPerBatch == tiling.nnzUnits &&
            tiling.nnzUnits == tiling.m * kSddmmComplexSamplesPerRow &&
            (tiling.patternFlags &
                SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64) != 0U;
    }

    const bool supportedBlock = tiling.blockDim == 16 ||
        tiling.blockDim == 32 || tiling.blockDim == 64;
    const bool validShape = supportedBlock && tiling.blockRows > 0 &&
        tiling.blockCols > 0 && tiling.m % tiling.blockDim == 0 &&
        tiling.n % tiling.blockDim == 0 &&
        tiling.blockRows == tiling.m / tiling.blockDim &&
        tiling.blockCols == tiling.n / tiling.blockDim;
    const int64_t blockArea = static_cast<int64_t>(tiling.blockDim) *
        tiling.blockDim;
    return tiling.format == SDDMM_FORMAT_BSR && validShape &&
        (tiling.blockOrder == 0 || tiling.blockOrder == 1) &&
        tiling.nnzUnits <= std::numeric_limits<int32_t>::max() /
            tiling.blockDim &&
        tiling.outputsPerBatch == tiling.nnzUnits * blockArea;
}

} // namespace sddmm_arch35

extern "C" {

/*
 * vectorBlockDim must be nonzero and no greater than the available AIV core
 * count. The launch is asynchronous on stream.
 */
void sddmm_complex_row_kernel_do(
    GM_ADDR sparseOffsets,
    GM_ADDR sparseColumns,
    GM_ADDR sparseValues,
    GM_ADDR matX,
    GM_ADDR matY,
    const SddmmTilingData &tiling,
    uint32_t vectorBlockDim,
    void *stream);

} // extern "C"

#endif // SDDMM_ARCH35_COMPLEX_ROW_COMMON_H_
