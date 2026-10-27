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
 * \file sddmm_column_vector_kernel.h
 * \brief FP16 CSR SDDMM SIMD kernel for column-major Y and K=128 on Ascend 950.
 */

#ifndef SDDMM_ARCH35_COLUMN_VECTOR_KERNEL_H_
#define SDDMM_ARCH35_COLUMN_VECTOR_KERNEL_H_

#include <cstdint>

#include "kernel_operator.h"
#include "../shared/sddmm_device_math.h"
#include "../../sddmm_tiling_data.h"

namespace sddmm_arch35 {

constexpr int32_t kColumnVectorReductionSize = 128;
constexpr int32_t kColumnVectorReductionHalf = 64;
constexpr int32_t kColumnVectorFp32ElementsPerBlock = 8;
constexpr int32_t kColumnVectorSamplesPerRow = 64;

template <int32_t ColumnsPerTile>
class SddmmColumnVectorKernelFp16 {
    static_assert(ColumnsPerTile == 8 || ColumnsPerTile == 16 ||
        ColumnsPerTile == 32 || ColumnsPerTile == 64,
        "Column-major SDDMM supports an output-column tile of 8, 16, 32, or 64");

public:
    __aicore__ inline void Init(
        GM_ADDR,
        GM_ADDR,
        GM_ADDR sparseValues,
        GM_ADDR matX,
        GM_ADDR matY,
        const SddmmTilingData *tiling,
        AscendC::TPipe *pipe)
    {
        tiling_ = *tiling;
        pipe_ = pipe;

        values_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(sparseValues),
            static_cast<uint64_t>(tiling_.nnzUnits));
        matX_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(matX),
            static_cast<uint64_t>(tiling_.m) * static_cast<uint64_t>(tiling_.ldX));
        matY_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(matY),
            static_cast<uint64_t>(tiling_.n) * static_cast<uint64_t>(tiling_.ldY));

        pipe_->InitBuffer(xQueue_, 1, kColumnVectorReductionSize * sizeof(half));
        pipe_->InitBuffer(yQueue_, 1,
            ColumnsPerTile * kColumnVectorReductionSize * sizeof(half));
        pipe_->InitBuffer(outQueue_, 1, ColumnsPerTile * sizeof(half));
        pipe_->InitBuffer(productBuffer_,
            ColumnsPerTile * kColumnVectorReductionHalf * sizeof(float));
        pipe_->InitBuffer(accumulatorBuffer_, ColumnsPerTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (!CanRun(tiling_) || tiling_.m == 0 || tiling_.nnzUnits == 0) {
            return;
        }
        const int64_t block = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
        if (blockCount <= 0) {
            return;
        }
        for (int64_t row = block; row < tiling_.m; row += blockCount) {
            ProcessRow(row);
        }
    }

    __aicore__ static inline bool CanRun(const SddmmTilingData &tiling)
    {
        return tiling.format == SDDMM_FORMAT_CSR &&
            tiling.usePackedY == 0 &&
            tiling.k == kColumnVectorReductionSize &&
            tiling.rowsX == tiling.m &&
            tiling.colsX == kColumnVectorReductionSize &&
            tiling.rowsY == kColumnVectorReductionSize &&
            tiling.colsY == tiling.n &&
            tiling.opX == 0 && tiling.opY == 0 &&
            tiling.orderX == 0 && tiling.orderY == 1 &&
            tiling.inputType == SDDMM_VALUE_FP16 &&
            tiling.outputType == SDDMM_VALUE_FP16 &&
            tiling.computeType == SDDMM_VALUE_FP32 &&
            tiling.pointerMode == 0 &&
            tiling.alphaReal == 1.0F && tiling.betaReal == 0.0F &&
            tiling.batchCount == 1 &&
            tiling.batchCountX == 1 && tiling.batchCountY == 1 &&
            tiling.offsetsBatchStride == 0 &&
            tiling.columnsBatchStride == 0 &&
            tiling.valuesBatchStride == 0 &&
            tiling.ldX == kColumnVectorReductionSize &&
            tiling.ldY == kColumnVectorReductionSize &&
            tiling.outputsPerBatch == tiling.nnzUnits &&
            (tiling.patternFlags &
                SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17) != 0U &&
            tiling.patternLogicalFirstColumn >= 0 &&
            static_cast<int64_t>(tiling.patternLogicalFirstColumn) < tiling.n &&
            (tiling.indexBase == 0 || tiling.indexBase == 1) &&
            tiling.m >= 0 && tiling.n >= 0 && tiling.nnzUnits >= 0;
    }

private:
    __aicore__ inline void ProcessRow(int64_t row)
    {
        // The preprocess-owned cyclic-64 flag proves that every row contains
        // exactly 64 entries.  Derive the CSR value range instead of reading
        // two scalar offsets on every row.
        const int32_t rowStart = static_cast<int32_t>(
            row * kColumnVectorSamplesPerRow);
        const int32_t rowEnd = rowStart + kColumnVectorSamplesPerRow;
        const int64_t logicalFirstColumn =
            (static_cast<int64_t>(tiling_.patternLogicalFirstColumn) +
                static_cast<int64_t>(kSddmmCsrCyclic64RowStride) * row) %
            tiling_.n;

        AscendC::LocalTensor<half> xLocal = xQueue_.AllocTensor<half>();
        const uint64_t xOffset = static_cast<uint64_t>(row) *
            static_cast<uint64_t>(tiling_.ldX);
        AscendC::DataCopyExtParams xCopy{1,
            static_cast<uint32_t>(kColumnVectorReductionSize * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> xPad{false, 0, 0, static_cast<half>(0.0F)};
        AscendC::DataCopyPad(xLocal, matX_[xOffset], xCopy, xPad);
        xQueue_.EnQue(xLocal);
        xLocal = xQueue_.DeQue<half>();

        for (int32_t segment = rowStart; segment < rowEnd;
             segment += ColumnsPerTile) {
            const int32_t active = SddmmDeviceMin(ColumnsPerTile, rowEnd - segment);
            ProcessSegment(
                segment, active, rowStart, logicalFirstColumn, xLocal);
        }
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessSegment(
        int32_t segment,
        int32_t active,
        int32_t rowStart,
        int64_t logicalFirstColumn,
        AscendC::LocalTensor<half> &xLocal)
    {
        AscendC::LocalTensor<half> yLocal = yQueue_.AllocTensor<half>();
        AscendC::DataCopyPadExtParams<half> yPad{false, 0, 0, static_cast<half>(0.0F)};
        int64_t rowPosition = static_cast<int64_t>(segment - rowStart);
        int64_t destinationPosition = 0;
        int64_t remaining = active;
        const int64_t wrappedColumns = logicalFirstColumn +
            kColumnVectorSamplesPerRow > tiling_.n ?
            logicalFirstColumn + kColumnVectorSamplesPerRow - tiling_.n : 0;
        // The validator proves the logical cyclic run and CSR stores each row
        // in sorted order.  Reconstruct its one or two contiguous runs instead
        // of reading a scalar column index for every output.
        while (remaining > 0) {
            int64_t firstColumn = 0;
            int64_t run = remaining;
            if (rowPosition < wrappedColumns) {
                firstColumn = rowPosition;
                run = SddmmDeviceMin(remaining, wrappedColumns - rowPosition);
            } else if (wrappedColumns == 0) {
                firstColumn = logicalFirstColumn + rowPosition;
            } else {
                firstColumn = tiling_.n - kColumnVectorSamplesPerRow +
                    rowPosition;
            }
            AscendC::DataCopyExtParams yCopy{1,
                static_cast<uint32_t>(
                    run * kColumnVectorReductionSize * sizeof(half)), 0, 0, 0};
            const uint64_t yOffset = static_cast<uint64_t>(firstColumn) *
                static_cast<uint64_t>(tiling_.ldY);
            AscendC::DataCopyPad(
                yLocal[destinationPosition * kColumnVectorReductionSize],
                matY_[yOffset], yCopy, yPad);
            rowPosition += run;
            destinationPosition += run;
            remaining -= run;
        }
        yQueue_.EnQue(yLocal);
        yLocal = yQueue_.DeQue<half>();

        ComputeDotProducts(xLocal, yLocal, active);
        yQueue_.FreeTensor(yLocal);
        StoreSegment(segment, active);
    }

    __aicore__ inline void ComputeDotProducts(
        AscendC::LocalTensor<half> &xLocal,
        AscendC::LocalTensor<half> &yLocal,
        int32_t active)
    {
        AscendC::LocalTensor<float> product = productBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuffer_.Get<float>();
        AscendC::BinaryRepeatParams multiplyParams;
        multiplyParams.dstBlkStride = 1;
        multiplyParams.src0BlkStride = 1;
        multiplyParams.src1BlkStride = 1;
        multiplyParams.dstRepStride = static_cast<uint8_t>(
            kColumnVectorReductionHalf / kColumnVectorFp32ElementsPerBlock);
        multiplyParams.src0RepStride = 0;
        multiplyParams.src1RepStride = static_cast<uint8_t>(
            kColumnVectorReductionSize / kColumnVectorFp32ElementsPerBlock / 2);

        const uint8_t repeat = static_cast<uint8_t>(active);
        constexpr int32_t productRepeatStride =
            kColumnVectorReductionHalf / kColumnVectorFp32ElementsPerBlock;
        AscendC::Duplicate(product, 0.0F, active * kColumnVectorReductionHalf);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst(product, xLocal, yLocal,
            static_cast<uint64_t>(kColumnVectorReductionHalf), repeat, multiplyParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst(product, xLocal[kColumnVectorReductionHalf],
            yLocal[kColumnVectorReductionHalf],
            static_cast<uint64_t>(kColumnVectorReductionHalf), repeat, multiplyParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(accumulator, product,
            kColumnVectorReductionHalf, active, 1, 1, productRepeatStride);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreSegment(int32_t segment, int32_t active)
    {
        AscendC::LocalTensor<float> accumulator = accumulatorBuffer_.Get<float>();
        AscendC::LocalTensor<half> output = outQueue_.AllocTensor<half>();
        AscendC::Cast(output, accumulator, AscendC::RoundMode::CAST_ROUND, active);
        outQueue_.EnQue(output);
        output = outQueue_.DeQue<half>();
        AscendC::DataCopyExtParams outputCopy{1,
            static_cast<uint32_t>(active * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPad(values_[segment], output, outputCopy);
        outQueue_.FreeTensor(output);
    }

    AscendC::TPipe *pipe_{nullptr};
    AscendC::TQue<AscendC::TPosition::VECIN, 1> xQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> yQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> productBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulatorBuffer_;

    AscendC::GlobalTensor<half> values_;
    AscendC::GlobalTensor<half> matX_;
    AscendC::GlobalTensor<half> matY_;
    SddmmTilingData tiling_{};
};

} // namespace sddmm_arch35

#endif // SDDMM_ARCH35_COLUMN_VECTOR_KERNEL_H_
