/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SDDMM_ARCH35_PATTERN_VALIDATOR_H_
#define SDDMM_ARCH35_PATTERN_VALIDATOR_H_

#include "kernel_operator.h"
#include "../../sddmm_tiling_data.h"

namespace sddmm_arch35 {

__aicore__ inline uint32_t BuildSddmmPatternFlags(int32_t status, bool sortedCyclic64, bool stride17)
{
    if (status != SDDMM_PATTERN_VALID || !sortedCyclic64) {
        return SDDMM_PATTERN_FLAG_NONE;
    }
    return SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64 |
        (stride17 ? SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17 : SDDMM_PATTERN_FLAG_NONE);
}

// The stored columns must be strictly sorted, with at most one cyclic gap.
__aicore__ inline int32_t RecoverSddmmSortedCyclic64Start(
    __gm__ const int32_t *rowColumns, int64_t minorCount, int32_t indexBase)
{
    constexpr int32_t kPatternColumns = 64;
    if (minorCount < kPatternColumns) {
        return -1;
    }
    const int64_t first = static_cast<int64_t>(rowColumns[0]) - indexBase;
    if (first < 0 || first >= minorCount) {
        return -1;
    }
    int64_t previous = first;
    int32_t gaps = 0;
    int32_t wrappedStart = -1;
    for (int32_t index = 1; index < kPatternColumns; ++index) {
        const int64_t column = static_cast<int64_t>(rowColumns[index]) - indexBase;
        if (column <= previous || column >= minorCount) {
            return -1;
        }
        if (column != previous + 1) {
            ++gaps;
            if (gaps > 1) {
                return -1;
            }
            wrappedStart = static_cast<int32_t>(column);
        }
        previous = column;
    }
    if (gaps == 0) {
        return static_cast<int32_t>(first);
    }
    return first == 0 && previous == minorCount - 1 ? wrappedStart : -1;
}

class SddmmPatternValidator {
public:
    __aicore__ inline void Init(GM_ADDR offsets, GM_ADDR columns, GM_ADDR workspace,
        const SddmmTilingData *tiling, uint32_t numBlocks)
    {
        offsets_ = reinterpret_cast<__gm__ const int32_t *>(offsets);
        columns_ = reinterpret_cast<__gm__ const int32_t *>(columns);
        block_ = AscendC::GetBlockIdx();
        numBlocks_ = numBlocks;
        partial_ = reinterpret_cast<__gm__ SddmmQueryPartial *>(
            workspace + static_cast<uint64_t>(block_) * kSddmmWorkspaceBytes);
        resultGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(partial_),
            kSddmmWorkspaceBytes / sizeof(uint64_t));
        tiling_ = *tiling;
    }

    __aicore__ inline void Process()
    {
        status_ = SDDMM_PATTERN_VALID;
        firstInvalidBatch_ = -1;
        majorCount_ = tiling_.format == SDDMM_FORMAT_CSR ? tiling_.m : tiling_.blockRows;
        minorCount_ = tiling_.format == SDDMM_FORMAT_CSR ? tiling_.n : tiling_.blockCols;
        patternBatches_ = (tiling_.offsetsBatchStride == 0 &&
            tiling_.columnsBatchStride == 0) ? 1 : tiling_.batchCount;
        expectedEnd_ = tiling_.nnzUnits + tiling_.indexBase;
        sortedCyclic64_ = tiling_.format == SDDMM_FORMAT_CSR && majorCount_ > 0 && patternBatches_ == 1 &&
            minorCount_ >= 64;
        logicalFirstColumn_ = sortedCyclic64_ ? RecoverRow(offsets_[0], offsets_[1], 0) : -1;
        sortedCyclic64_ = sortedCyclic64_ && logicalFirstColumn_ >= 0;
        stride17_ = sortedCyclic64_;
        ValidateRows();
        ValidateColumns();
        StoreResult();
    }

private:
    __aicore__ inline int64_t PartitionBegin(int64_t count, uint32_t block) const
    {
        const int64_t quotient = count / numBlocks_;
        const uint32_t remainder = static_cast<uint32_t>(count % numBlocks_);
        return quotient * block + (block < remainder ? block : remainder);
    }

    __aicore__ inline void RecordError(int64_t batch, int32_t status)
    {
        // Preserve the serial validator's first batch, then endpoint/row/column
        // ordering even though rows and columns now execute independently.
        if (firstInvalidBatch_ < 0 || batch < firstInvalidBatch_ ||
            (batch == firstInvalidBatch_ && status < status_)) {
            firstInvalidBatch_ = batch;
            status_ = status;
        }
    }

    __aicore__ inline int32_t RecoverRow(int64_t current, int64_t next,
        int64_t columnsBase) const
    {
        const int64_t begin = current - tiling_.indexBase;
        const int64_t end = next - tiling_.indexBase;
        if (begin < 0 || end > tiling_.nnzUnits || end - begin != 64) {
            return -1;
        }
        return RecoverSddmmSortedCyclic64Start(
            columns_ + columnsBase + begin, minorCount_, tiling_.indexBase);
    }

    __aicore__ inline void CheckCyclicRow(int64_t current, int64_t next,
        int64_t row, int64_t columnsBase)
    {
        const int32_t start = RecoverRow(current, next, columnsBase);
        if (start < 0) {
            sortedCyclic64_ = false;
            stride17_ = false;
        } else if (stride17_) {
            const int64_t expected = (logicalFirstColumn_ +
                static_cast<int64_t>(kSddmmCsrCyclic64RowStride) * row) % minorCount_;
            stride17_ = start == expected;
        }
    }

    __aicore__ inline void ValidateRows()
    {
        // Host checks both flattened products before the launch. A row's pair
        // of offsets includes its trailing halo at a partition boundary.
        const int64_t count = patternBatches_ * majorCount_;
        int64_t task = PartitionBegin(count, block_);
        const int64_t endTask = PartitionBegin(count, block_ + 1U);
        while (task < endTask) {
            const int64_t batch = task / majorCount_;
            const int64_t beginRow = task % majorCount_;
            const int64_t available = endTask - task;
            const int64_t rows = available < majorCount_ - beginRow ?
                available : majorCount_ - beginRow;
            const int64_t offsetsBase = tiling_.offsetsBatchStride == 0 ?
                0 : batch * tiling_.offsetsBatchStride;
            const int64_t columnsBase = tiling_.columnsBatchStride == 0 ?
                0 : batch * tiling_.columnsBatchStride;
            const auto *offsets = offsets_ + offsetsBase;
            if (beginRow == 0 && (offsets[0] != tiling_.indexBase ||
                static_cast<int64_t>(offsets[majorCount_]) != expectedEnd_)) {
                RecordError(batch, SDDMM_PATTERN_BAD_ENDPOINT);
            }
            for (int64_t row = beginRow; row < beginRow + rows; ++row) {
                const int64_t current = offsets[row];
                const int64_t next = offsets[row + 1];
                if (current < tiling_.indexBase || next < current || next > expectedEnd_) {
                    RecordError(batch, SDDMM_PATTERN_NOT_MONOTONIC);
                }
                if (sortedCyclic64_) {
                    CheckCyclicRow(current, next, row, columnsBase);
                }
            }
            task += rows;
        }
    }

    __aicore__ inline void ValidateColumns()
    {
        // Ranges never depend on potentially malformed offsets. A zero-nnz
        // partition is empty, so the loop never divides by nnzUnits in it.
        const int64_t count = patternBatches_ * tiling_.nnzUnits;
        int64_t task = PartitionBegin(count, block_);
        const int64_t endTask = PartitionBegin(count, block_ + 1U);
        while (task < endTask) {
            const int64_t batch = task / tiling_.nnzUnits;
            const int64_t begin = task % tiling_.nnzUnits;
            const int64_t available = endTask - task;
            const int64_t columns = available < tiling_.nnzUnits - begin ?
                available : tiling_.nnzUnits - begin;
            const int64_t base = tiling_.columnsBatchStride == 0 ?
                0 : batch * tiling_.columnsBatchStride;
            for (int64_t index = begin; index < begin + columns; ++index) {
                const int64_t column = columns_[base + index];
                if (column < tiling_.indexBase || column >= minorCount_ + tiling_.indexBase) {
                    RecordError(batch, SDDMM_PATTERN_BAD_COLUMN);
                }
            }
            task += columns;
        }
    }

    __aicore__ inline void StoreResult()
    {
        const uint32_t flags = BuildSddmmPatternFlags(status_, sortedCyclic64_, stride17_);
        partial_->validation.magic = kSddmmValidationMagic;
        partial_->validation.status = status_;
        partial_->validation.patternFlags = flags;
        partial_->validation.logicalFirstColumn = flags != SDDMM_PATTERN_FLAG_NONE ?
            logicalFirstColumn_ : -1;
        partial_->validation.reserved = 0;
        partial_->firstInvalidBatch = firstInvalidBatch_;
        // Each block owns one whole cache line; Host waits before reading it.
        AscendC::DataCacheCleanAndInvalid<uint64_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(resultGlobal_);
    }

    __gm__ const int32_t *offsets_{nullptr};
    __gm__ const int32_t *columns_{nullptr};
    __gm__ SddmmQueryPartial *partial_{nullptr};
    AscendC::GlobalTensor<uint64_t> resultGlobal_;
    SddmmTilingData tiling_{};
    uint32_t block_{};
    uint32_t numBlocks_{};
    int64_t patternBatches_{};
    int64_t majorCount_{};
    int64_t minorCount_{};
    int64_t expectedEnd_{};
    int64_t firstInvalidBatch_{};
    int32_t status_{};
    int32_t logicalFirstColumn_{};
    bool sortedCyclic64_{};
    bool stride17_{};
};

} // namespace sddmm_arch35

#endif // SDDMM_ARCH35_PATTERN_VALIDATOR_H_
