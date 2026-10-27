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

#ifndef SDDMM_ARCH35_PATTERN_HOST_H_
#define SDDMM_ARCH35_PATTERN_HOST_H_

#include <algorithm>
#include <cstdint>
#include <limits>

#include "../sddmm_tiling_data.h"

namespace sddmm_arch35 {

constexpr uint32_t kSddmmAllPatternFlags = SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64 |
    SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17;
constexpr uint64_t kSddmmValidationUnitsPerCore = 4096ULL;

inline int64_t PatternBatchCount(const SddmmTilingData &tiling)
{
    return tiling.offsetsBatchStride == 0 && tiling.columnsBatchStride == 0 ?
        1 : tiling.batchCount;
}

inline bool PatternCheckedMultiply(int64_t lhs, int64_t rhs, int64_t &result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline bool PlanPatternValidation(
    const SddmmTilingData &tiling, uint32_t coreCount, uint32_t &numBlocks)
{
    numBlocks = 0U;
    if ((tiling.format != SDDMM_FORMAT_CSR && tiling.format != SDDMM_FORMAT_BSR) ||
        tiling.batchCount <= 0 || tiling.offsetsBatchStride < 0 || tiling.columnsBatchStride < 0) {
        return false;
    }
    const int64_t majorCount = tiling.format == SDDMM_FORMAT_CSR ? tiling.m : tiling.blockRows;
    const int64_t batches = PatternBatchCount(tiling);
    int64_t flatRows = 0;
    int64_t flatColumns = 0;
    if (!PatternCheckedMultiply(batches, majorCount, flatRows) ||
        !PatternCheckedMultiply(batches, tiling.nnzUnits, flatColumns)) {
        return false;
    }
    const uint64_t units = static_cast<uint64_t>(std::max(flatRows, flatColumns));
    if (units == 0U) {
        return true;
    }
    // Keep the initial scheduling grain on Host, outside the Device scan.
    const uint64_t blocksNeeded = (units - 1U) / kSddmmValidationUnitsPerCore + 1U;
    numBlocks = static_cast<uint32_t>(std::min<uint64_t>(coreCount, blocksNeeded));
    return coreCount != 0U;
}

inline bool ValidateQueryPartial(const SddmmValidationResult &partial, int64_t columnCount)
{
    const uint32_t flags = partial.patternFlags;
    return !(partial.magic != kSddmmValidationMagic ||
        partial.status < SDDMM_PATTERN_VALID || partial.status > SDDMM_PATTERN_BAD_COLUMN ||
        (flags & ~kSddmmAllPatternFlags) != 0U ||
        flags == SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17 ||
        (partial.status != SDDMM_PATTERN_VALID && flags != SDDMM_PATTERN_FLAG_NONE) ||
        partial.reserved != 0 ||
        (flags == SDDMM_PATTERN_FLAG_NONE && partial.logicalFirstColumn != -1) ||
        (flags != SDDMM_PATTERN_FLAG_NONE &&
            (partial.logicalFirstColumn < 0 || partial.logicalFirstColumn >= columnCount)));
}

inline bool MergePatternFlags(const SddmmValidationResult &partial, int32_t &firstColumn,
    SddmmValidationResult &validation)
{
    if (partial.patternFlags != SDDMM_PATTERN_FLAG_NONE) {
        if (firstColumn >= 0 && firstColumn != partial.logicalFirstColumn) {
            return false;
        }
        firstColumn = partial.logicalFirstColumn;
    }
    validation.patternFlags &= partial.patternFlags;
    return true;
}

inline bool ReducePatternQuery(const SddmmQueryPartial *partials, uint32_t numBlocks,
    int64_t columnCount, SddmmValidationResult &validation)
{
    if (partials == nullptr || numBlocks == 0U) {
        return false;
    }
    validation = {};
    validation.magic = kSddmmValidationMagic;
    validation.status = SDDMM_PATTERN_VALID;
    validation.patternFlags = kSddmmAllPatternFlags;
    int32_t firstColumn = -1;
    for (uint32_t block = 0; block < numBlocks; ++block) {
        const auto &partial = partials[block].validation;
        if (!ValidateQueryPartial(partial, columnCount) ||
            !MergePatternFlags(partial, firstColumn, validation)) {
            return false;
        }
        if (partial.status != SDDMM_PATTERN_VALID &&
            (validation.status == SDDMM_PATTERN_VALID || partial.status < validation.status)) {
            validation.status = partial.status;
        }
    }
    validation.logicalFirstColumn = validation.patternFlags != SDDMM_PATTERN_FLAG_NONE ? firstColumn : -1;
    return true;
}

inline bool ValidatePatternValidationPartial(const SddmmQueryPartial &slot,
    int64_t batches, int64_t columnCount, bool canClassify)
{
    const auto &partial = slot.validation;
    return ValidateQueryPartial(partial, columnCount) &&
        (partial.patternFlags == SDDMM_PATTERN_FLAG_NONE || canClassify) &&
        (partial.status == SDDMM_PATTERN_VALID ? slot.firstInvalidBatch == -1 :
            (slot.firstInvalidBatch >= 0 && slot.firstInvalidBatch < batches));
}

inline bool IsEarlierPatternError(const SddmmQueryPartial &slot,
    int64_t firstInvalidBatch, int32_t status)
{
    return slot.validation.status != SDDMM_PATTERN_VALID &&
        (firstInvalidBatch < 0 || slot.firstInvalidBatch < firstInvalidBatch ||
            (slot.firstInvalidBatch == firstInvalidBatch && slot.validation.status < status));
}

inline bool ReducePatternValidation(const SddmmQueryPartial *partials, uint32_t numBlocks,
    const SddmmTilingData &tiling, SddmmValidationResult &validation)
{
    if (partials == nullptr || numBlocks == 0U ||
        (tiling.format != SDDMM_FORMAT_CSR && tiling.format != SDDMM_FORMAT_BSR)) {
        return false;
    }
    const int64_t batches = PatternBatchCount(tiling);
    const int64_t columnCount = tiling.format == SDDMM_FORMAT_CSR ? tiling.n : tiling.blockCols;
    const bool canClassify = tiling.format == SDDMM_FORMAT_CSR && batches == 1 &&
        tiling.m > 0 && columnCount >= 64;
    validation = {};
    validation.magic = kSddmmValidationMagic;
    validation.status = SDDMM_PATTERN_VALID;
    validation.patternFlags = kSddmmAllPatternFlags;
    int32_t firstColumn = -1;
    int64_t firstInvalidBatch = -1;
    for (uint32_t block = 0; block < numBlocks; ++block) {
        const auto &slot = partials[block];
        const auto &partial = slot.validation;
        if (!ValidatePatternValidationPartial(slot, batches, columnCount, canClassify) ||
            !MergePatternFlags(partial, firstColumn, validation)) {
            return false;
        }
        // Preserve the serial scan: first invalid batch wins, then endpoint,
        // monotonic offsets, and finally column bounds within that batch.
        if (IsEarlierPatternError(slot, firstInvalidBatch, validation.status)) {
            firstInvalidBatch = slot.firstInvalidBatch;
            validation.status = partial.status;
        }
    }
    if (validation.status != SDDMM_PATTERN_VALID) {
        validation.patternFlags = SDDMM_PATTERN_FLAG_NONE;
    }
    validation.logicalFirstColumn = validation.patternFlags != SDDMM_PATTERN_FLAG_NONE ? firstColumn : -1;
    return true;
}

} // namespace sddmm_arch35

#endif // SDDMM_ARCH35_PATTERN_HOST_H_
