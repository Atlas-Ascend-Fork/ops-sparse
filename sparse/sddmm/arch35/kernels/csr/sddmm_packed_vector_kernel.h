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
 * \file sddmm_packed_vector_kernel.h
 * \brief FP32 CSR SDDMM SIMD kernel for packed Y and K=128 on Ascend 950.
 */

#ifndef SDDMM_ARCH35_PACKED_VECTOR_KERNEL_H_
#define SDDMM_ARCH35_PACKED_VECTOR_KERNEL_H_

#include <cstdint>

#include "kernel_operator.h"
#include "../shared/sddmm_device_math.h"
#include "../../sddmm_tiling_data.h"

namespace sddmm_arch35 {

constexpr int32_t kPackedVectorReductionSize = 128;
constexpr int32_t kPackedVectorReductionHalf = 64;
constexpr int32_t kPackedVectorFp32ElementsPerBlock = 8;
constexpr int32_t kPackedVectorSamplesPerRow = 64;

template <int32_t ColumnsPerTile>
class SddmmPackedVectorKernelFp32 {
    static_assert(ColumnsPerTile == 8 || ColumnsPerTile == 16,
        "Packed SDDMM supports an output-column tile of 8 or 16");

public:
    __aicore__ inline void Init(
        GM_ADDR,
        GM_ADDR,
        GM_ADDR sparseValues,
        GM_ADDR matX,
        GM_ADDR packedY,
        const SddmmTilingData *tiling,
        AscendC::TPipe *pipe)
    {
        tiling_ = *tiling;
        pipe_ = pipe;

        values_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sparseValues),
            static_cast<uint64_t>(tiling_.nnzUnits));
        matX_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matX),
            static_cast<uint64_t>(tiling_.m) * static_cast<uint64_t>(tiling_.ldX));
        packedY_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(packedY),
            static_cast<uint64_t>(tiling_.n) * kPackedVectorReductionSize);

        pipe_->InitBuffer(xQueue_, 1, kPackedVectorReductionSize * sizeof(float));
        pipe_->InitBuffer(yQueue_, 1,
            ColumnsPerTile * kPackedVectorReductionSize * sizeof(float));
        pipe_->InitBuffer(cQueue_, 1, ColumnsPerTile * sizeof(float));
        pipe_->InitBuffer(outQueue_, 1, ColumnsPerTile * sizeof(float));
        pipe_->InitBuffer(productBuffer_,
            ColumnsPerTile * kPackedVectorReductionHalf * sizeof(float));
        pipe_->InitBuffer(accumulatorBuffer_, ColumnsPerTile * sizeof(float));
        pipe_->InitBuffer(partialBuffer_, ColumnsPerTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (!CanRun(tiling_) || tiling_.m == 0 || tiling_.nnzUnits == 0) {
            return;
        }

        const float alpha = tiling_.alphaReal;
        const float beta = tiling_.betaReal;
        if (alpha == 0.0F && beta == 1.0F) {
            return;
        }

        const int64_t block = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
        if (blockCount <= 0) {
            return;
        }
        for (int64_t row = block; row < tiling_.m; row += blockCount) {
            ProcessRow(row, alpha, beta);
        }
    }

    __aicore__ static inline bool CanRun(const SddmmTilingData &tiling)
    {
        return tiling.format == SDDMM_FORMAT_CSR &&
            tiling.usePackedY != 0 &&
            tiling.k == kPackedVectorReductionSize &&
            tiling.rowsX == tiling.m &&
            tiling.colsX == kPackedVectorReductionSize &&
            tiling.rowsY == kPackedVectorReductionSize &&
            tiling.colsY == tiling.n &&
            tiling.opX == 0 && tiling.opY == 0 &&
            tiling.orderX == 0 && tiling.orderY == 0 &&
            tiling.inputType == SDDMM_VALUE_FP32 &&
            tiling.outputType == SDDMM_VALUE_FP32 &&
            tiling.computeType == SDDMM_VALUE_FP32 &&
            tiling.pointerMode == 0 &&
            tiling.batchCount == 1 &&
            tiling.batchCountX == 1 && tiling.batchCountY == 1 &&
            tiling.offsetsBatchStride == 0 &&
            tiling.columnsBatchStride == 0 &&
            tiling.valuesBatchStride == 0 &&
            tiling.ldX >= kPackedVectorReductionSize &&
            tiling.ldY >= tiling.n &&
            tiling.outputsPerBatch == tiling.nnzUnits &&
            (tiling.patternFlags &
                SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17) != 0U &&
            tiling.patternLogicalFirstColumn >= 0 &&
            static_cast<int64_t>(tiling.patternLogicalFirstColumn) < tiling.n &&
            (tiling.indexBase == 0 || tiling.indexBase == 1) &&
            tiling.m >= 0 && tiling.n >= 0 && tiling.nnzUnits >= 0;
    }

private:
    __aicore__ inline void ProcessRow(int64_t row, float alpha, float beta)
    {
        // The strong preprocess flag proves both the row length and the
        // stride-17 cyclic column sequence, so neither CSR array is needed.
        const int32_t rowStart = static_cast<int32_t>(
            row * kPackedVectorSamplesPerRow);
        const int32_t rowEnd = rowStart + kPackedVectorSamplesPerRow;
        const int64_t logicalFirstColumn =
            (static_cast<int64_t>(tiling_.patternLogicalFirstColumn) +
                static_cast<int64_t>(kSddmmCsrCyclic64RowStride) * row) %
            tiling_.n;

        if (alpha == 0.0F) {
            ScaleOnly(rowStart, rowEnd, beta);
            return;
        }

        AscendC::LocalTensor<float> xLocal = xQueue_.AllocTensor<float>();
        const uint64_t xOffset = static_cast<uint64_t>(row) *
            static_cast<uint64_t>(tiling_.ldX);
        AscendC::DataCopyExtParams xCopy{1,
            static_cast<uint32_t>(kPackedVectorReductionSize * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> xPad{false, 0, 0, 0.0F};
        AscendC::DataCopyPad(xLocal, matX_[xOffset], xCopy, xPad);
        xQueue_.EnQue(xLocal);
        xLocal = xQueue_.DeQue<float>();

        for (int32_t segment = rowStart; segment < rowEnd; segment += ColumnsPerTile) {
            const int32_t active = SddmmDeviceMin(ColumnsPerTile, rowEnd - segment);
            ProcessSegment(segment, active, rowStart, logicalFirstColumn,
                xLocal, alpha, beta);
        }
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessSegment(
        int32_t segment,
        int32_t active,
        int32_t rowStart,
        int64_t logicalFirstColumn,
        AscendC::LocalTensor<float> &xLocal,
        float alpha,
        float beta)
    {
        AscendC::LocalTensor<float> yLocal = yQueue_.AllocTensor<float>();
        AscendC::DataCopyPadExtParams<float> yPad{false, 0, 0, 0.0F};
        int64_t rowPosition = static_cast<int64_t>(segment - rowStart);
        int64_t destinationPosition = 0;
        int64_t remaining = active;
        const int64_t wrappedColumns = logicalFirstColumn +
            kPackedVectorSamplesPerRow > tiling_.n ?
            logicalFirstColumn + kPackedVectorSamplesPerRow - tiling_.n : 0;
        while (remaining > 0) {
            int64_t firstColumn = 0;
            int64_t run = remaining;
            if (rowPosition < wrappedColumns) {
                firstColumn = rowPosition;
                run = SddmmDeviceMin(remaining, wrappedColumns - rowPosition);
            } else if (wrappedColumns == 0) {
                firstColumn = logicalFirstColumn + rowPosition;
            } else {
                firstColumn = tiling_.n - kPackedVectorSamplesPerRow +
                    rowPosition;
            }
            AscendC::DataCopyExtParams yCopy{1,
                static_cast<uint32_t>(
                    run * kPackedVectorReductionSize * sizeof(float)), 0, 0, 0};
            const uint64_t yOffset = static_cast<uint64_t>(firstColumn) *
                kPackedVectorReductionSize;
            AscendC::DataCopyPad(
                yLocal[destinationPosition * kPackedVectorReductionSize],
                packedY_[yOffset], yCopy, yPad);
            rowPosition += run;
            destinationPosition += run;
            remaining -= run;
        }
        yQueue_.EnQue(yLocal);
        yLocal = yQueue_.DeQue<float>();

        ComputeDotProducts(xLocal, yLocal, active);
        yQueue_.FreeTensor(yLocal);
        WriteSegment(segment, active, alpha, beta);
    }

    __aicore__ inline void ComputeDotProducts(
        AscendC::LocalTensor<float> &xLocal,
        AscendC::LocalTensor<float> &yLocal,
        int32_t active)
    {
        AscendC::LocalTensor<float> product = productBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuffer_.Get<float>();
        AscendC::LocalTensor<float> partial = partialBuffer_.Get<float>();

        AscendC::BinaryRepeatParams multiplyParams;
        multiplyParams.dstBlkStride = 1;
        multiplyParams.src0BlkStride = 1;
        multiplyParams.src1BlkStride = 1;
        multiplyParams.dstRepStride = static_cast<uint8_t>(
            kPackedVectorReductionHalf / kPackedVectorFp32ElementsPerBlock);
        multiplyParams.src0RepStride = 0;
        multiplyParams.src1RepStride = static_cast<uint8_t>(
            kPackedVectorReductionSize / kPackedVectorFp32ElementsPerBlock);

        const uint8_t repeat = static_cast<uint8_t>(active);
        constexpr int32_t productRepeatStride =
            kPackedVectorReductionHalf / kPackedVectorFp32ElementsPerBlock;
        AscendC::Mul(product, xLocal, yLocal,
            static_cast<uint64_t>(kPackedVectorReductionHalf), repeat, multiplyParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(accumulator, product,
            kPackedVectorReductionHalf, active, 1, 1, productRepeatStride);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(product, xLocal[kPackedVectorReductionHalf],
            yLocal[kPackedVectorReductionHalf],
            static_cast<uint64_t>(kPackedVectorReductionHalf), repeat, multiplyParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(partial, product,
            kPackedVectorReductionHalf, active, 1, 1, productRepeatStride);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(accumulator, accumulator, partial, active);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline AscendC::LocalTensor<float> LoadOldValues(int32_t segment, int32_t active)
    {
        AscendC::LocalTensor<float> oldValues = cQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams oldCopy{1,
            static_cast<uint32_t>(active * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> oldPad{false, 0, 0, 0.0F};
        AscendC::DataCopyPad(oldValues, values_[segment], oldCopy, oldPad);
        cQueue_.EnQue(oldValues);
        return cQueue_.DeQue<float>();
    }

    __aicore__ inline void StoreOutput(
        AscendC::LocalTensor<float> &output, int32_t segment, int32_t active)
    {
        outQueue_.EnQue(output);
        output = outQueue_.DeQue<float>();
        AscendC::DataCopyExtParams outputCopy{1,
            static_cast<uint32_t>(active * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(values_[segment], output, outputCopy);
        outQueue_.FreeTensor(output);
    }

    __aicore__ inline void WriteSegment(
        int32_t segment, int32_t active, float alpha, float beta)
    {
        AscendC::LocalTensor<float> accumulator = accumulatorBuffer_.Get<float>();
        AscendC::LocalTensor<float> output = outQueue_.AllocTensor<float>();
        AscendC::Muls(output, accumulator, alpha, active);

        if (beta != 0.0F) {
            AscendC::LocalTensor<float> oldValues = LoadOldValues(segment, active);
            AscendC::Muls(oldValues, oldValues, beta, active);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(output, output, oldValues, active);
            AscendC::PipeBarrier<PIPE_V>();
            cQueue_.FreeTensor(oldValues);
        }

        StoreOutput(output, segment, active);
    }

    __aicore__ inline void ScaleOnly(int32_t rowStart, int32_t rowEnd, float beta)
    {
        for (int32_t segment = rowStart; segment < rowEnd; segment += ColumnsPerTile) {
            const int32_t active = SddmmDeviceMin(ColumnsPerTile, rowEnd - segment);
            AscendC::LocalTensor<float> output = outQueue_.AllocTensor<float>();
            if (beta == 0.0F) {
                AscendC::Duplicate(output, 0.0F, active);
            } else {
                AscendC::LocalTensor<float> oldValues = LoadOldValues(segment, active);
                AscendC::Muls(output, oldValues, beta, active);
                AscendC::PipeBarrier<PIPE_V>();
                cQueue_.FreeTensor(oldValues);
            }
            StoreOutput(output, segment, active);
        }
    }

    AscendC::TPipe *pipe_{nullptr};
    AscendC::TQue<AscendC::TPosition::VECIN, 1> xQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> yQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> cQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> productBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulatorBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partialBuffer_;

    AscendC::GlobalTensor<float> values_;
    AscendC::GlobalTensor<float> matX_;
    AscendC::GlobalTensor<float> packedY_;
    SddmmTilingData tiling_{};
};

} // namespace sddmm_arch35

#endif // SDDMM_ARCH35_PACKED_VECTOR_KERNEL_H_
