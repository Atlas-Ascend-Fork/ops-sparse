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
 * \file sddmm_bsr8_vector_kernel.h
 * \brief FP32-accumulating BSR b8 SIMD kernel with a reusable packed-Y tile.
 */

#ifndef SDDMM_ARCH35_BSR8_VECTOR_KERNEL_H_
#define SDDMM_ARCH35_BSR8_VECTOR_KERNEL_H_

#include <cstdint>

#include "kernel_operator.h"
#include "../../sddmm_bsr8_common.h"
#include "../../sddmm_tiling_data.h"

namespace sddmm_arch35 {

constexpr int32_t kSddmmBsr8HalfK = 64;
constexpr int32_t kSddmmBsr8InputElements = kSddmmBsr8Side * kSddmmBsr8K;
constexpr int32_t kSddmmBsr8OutputElements = kSddmmBsr8Side * kSddmmBsr8Side;

template <typename T>
struct SddmmBsr8ValueType;

template <>
struct SddmmBsr8ValueType<float> {
    static constexpr int32_t code = SDDMM_VALUE_FP32;
};

template <>
struct SddmmBsr8ValueType<half> {
    static constexpr int32_t code = SDDMM_VALUE_FP16;
};

// Keep products and reductions in registers for the whole block. Two separate
// half-K reductions retain the established arithmetic grouping without UB
// product spills or an AIV scalar/vector barrier between every instruction.
__simd_vf__ inline void SddmmBsr8DotBlockVf(
    __ubuf__ float *dst, __ubuf__ float *matX, __ubuf__ float *matY)
{
    using namespace AscendC::MicroAPI;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    MaskReg first = CreateMask<float, MaskPattern::VL1>();
    RegTensor<float> xLow;
    RegTensor<float> xHigh;
    RegTensor<float> yLow;
    RegTensor<float> yHigh;
    RegTensor<float> productLow;
    RegTensor<float> productHigh;
    RegTensor<float> sumLow;
    RegTensor<float> sumHigh;
    for (uint16_t row = 0; row < kSddmmBsr8Side; ++row) {
        LoadAlign(xLow, matX + row * kSddmmBsr8K);
        LoadAlign(xHigh, matX + row * kSddmmBsr8K + kSddmmBsr8HalfK);
        for (uint16_t column = 0; column < kSddmmBsr8Side; ++column) {
            LoadAlign(yLow, matY + column * kSddmmBsr8K);
            LoadAlign(yHigh, matY + column * kSddmmBsr8K + kSddmmBsr8HalfK);
            Mul(productLow, xLow, yLow, full);
            Mul(productHigh, xHigh, yHigh, full);
            ReduceSum(sumLow, productLow, full);
            ReduceSum(sumHigh, productHigh, full);
            Add(sumLow, sumLow, sumHigh, first);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                dst + row * kSddmmBsr8Side + column, sumLow, first);
        }
    }
}

// One register covers the complete 8x8 block; both UB bases are 32-byte aligned.
__simd_vf__ inline void SddmmBsr8TransposeVf(__ubuf__ float *dst, __ubuf__ float *src)
{
    using namespace AscendC::MicroAPI;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<int32_t> index;
    RegTensor<int32_t> moduloMask;
    RegTensor<int32_t> row;
    RegTensor<int32_t> column;
    RegTensor<float> values;
    Arange(index, 0);
    Duplicate(moduloMask, kSddmmBsr8Side - 1, mask);
    And(column, index, moduloMask, mask);
    ShiftLefts(column, column, static_cast<int16_t>(3), mask);
    ShiftRights(row, index, static_cast<int16_t>(3), mask);
    Or(index, row, column, mask);
    Gather(values, src, (RegTensor<uint32_t> &)index, mask);
    StoreAlign(dst, values, mask);
}

template <typename InputT, typename OutputT>
class SddmmBsr8VectorKernel {
public:
    __aicore__ inline void Init(GM_ADDR offsets, GM_ADDR columns, GM_ADDR values,
        GM_ADDR matX, GM_ADDR packedY, const SddmmTilingData *tiling, AscendC::TPipe *pipe)
    {
        tiling_ = *tiling;
        pipe_ = pipe;
        offsets_ = reinterpret_cast<__gm__ const int32_t *>(offsets);
        columns_ = reinterpret_cast<__gm__ const int32_t *>(columns);
        values_.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT *>(values),
            static_cast<uint64_t>(tiling_.outputsPerBatch));
        matX_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(matX),
            static_cast<uint64_t>(tiling_.m) * kSddmmBsr8K);
        packedY_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(packedY),
            static_cast<uint64_t>(tiling_.n) * kSddmmBsr8K);
        InitBuffers();
    }

    __aicore__ static inline bool CanRun(const SddmmTilingData &tiling)
    {
        return HasSupportedShape(tiling) && HasSupportedStructure(tiling) &&
            HasIdentityScalarsAndSingleBatch(tiling) &&
            tiling.inputType == SddmmBsr8ValueType<InputT>::code &&
            tiling.outputType == SddmmBsr8ValueType<OutputT>::code &&
            (tiling.outputType == tiling.inputType || tiling.outputType == SDDMM_VALUE_FP32) &&
            tiling.computeType == SDDMM_VALUE_FP32;
    }

    __aicore__ inline void Process()
    {
        const int64_t coreCount = static_cast<int64_t>(AscendC::GetBlockNum());
        if (!CanRun(tiling_) || coreCount <= 0) {
            return;
        }
        const int64_t core = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t begin = tiling_.nnzUnits * core / coreCount;
        const int64_t end = tiling_.nnzUnits * (core + 1) / coreCount;
        AscendC::LocalTensor<InputT> xLocal;
        int64_t rowEnd = 0;
        bool xLoaded = false;
        for (int64_t block = begin; block < end; ++block) {
            if (block >= rowEnd) {
                if (xLoaded) {
                    xQueue_.FreeTensor(xLocal);
                }
                const int64_t row = FindBlockRow(block);
                xLocal = LoadX(row);
                rowEnd = static_cast<int64_t>(offsets_[row + 1]) - tiling_.indexBase;
                xLoaded = true;
            }
            ProcessBlock(block, xLocal);
        }
        if (xLoaded) {
            xQueue_.FreeTensor(xLocal);
        }
    }

private:
    __aicore__ static inline bool HasSupportedShape(const SddmmTilingData &tiling)
    {
        return tiling.format == SDDMM_FORMAT_BSR && tiling.blockDim == kSddmmBsr8Side &&
            tiling.usePackedY != 0 && tiling.k == kSddmmBsr8K &&
            tiling.opX == 0 && tiling.opY == 0 && tiling.orderX == 0 && tiling.orderY == 0 &&
            tiling.rowsX == tiling.m && tiling.colsX == kSddmmBsr8K &&
            tiling.rowsY == kSddmmBsr8K && tiling.colsY == tiling.n &&
            tiling.ldX == kSddmmBsr8K && tiling.ldY == tiling.n;
    }

    __aicore__ static inline bool HasSupportedStructure(const SddmmTilingData &tiling)
    {
        return (tiling.indexBase == 0 || tiling.indexBase == 1) &&
            (tiling.blockOrder == 0 || tiling.blockOrder == 1) &&
            tiling.m > 0 && tiling.n > 0 && tiling.nnzUnits > 0 &&
            tiling.nnzUnits <= INT32_MAX - tiling.indexBase &&
            tiling.m % kSddmmBsr8Side == 0 && tiling.n % kSddmmBsr8Side == 0 &&
            tiling.blockRows == tiling.m / kSddmmBsr8Side &&
            tiling.blockCols == tiling.n / kSddmmBsr8Side &&
            tiling.blockRows <= INT32_MAX && tiling.blockCols <= INT32_MAX &&
            tiling.outputsPerBatch == tiling.nnzUnits * kSddmmBsr8OutputElements;
    }

    __aicore__ static inline bool HasIdentityScalarsAndSingleBatch(const SddmmTilingData &tiling)
    {
        return tiling.pointerMode == 0 && tiling.alphaReal == 1.0F && tiling.alphaImag == 0.0F &&
            tiling.betaReal == 0.0F && tiling.betaImag == 0.0F &&
            tiling.batchCount == 1 && tiling.batchCountX == 1 && tiling.batchCountY == 1 &&
            tiling.offsetsBatchStride == 0 && tiling.columnsBatchStride == 0 &&
            tiling.valuesBatchStride == 0;
    }

    __aicore__ inline void InitBuffers()
    {
        pipe_->InitBuffer(xQueue_, 1, kSddmmBsr8InputElements * sizeof(InputT));
        pipe_->InitBuffer(yQueue_, 1, kSddmmBsr8InputElements * sizeof(InputT));
        pipe_->InitBuffer(outputQueue_, 1, kSddmmBsr8OutputElements * sizeof(OutputT));
        pipe_->InitBuffer(accumulatorBuffer_, kSddmmBsr8OutputElements * sizeof(float));
        pipe_->InitBuffer(reorderedBuffer_, kSddmmBsr8OutputElements * sizeof(float));
        if constexpr (SddmmBsr8ValueType<InputT>::code == SDDMM_VALUE_FP16) {
            pipe_->InitBuffer(xFp32Buffer_, kSddmmBsr8InputElements * sizeof(float));
            pipe_->InitBuffer(yFp32Buffer_, kSddmmBsr8InputElements * sizeof(float));
        }
    }

    __aicore__ inline int64_t FindBlockRow(int64_t block) const
    {
        int64_t low = 0;
        int64_t high = tiling_.blockRows;
        while (low < high) {
            const int64_t middle = low + (high - low) / 2;
            const int64_t next = static_cast<int64_t>(offsets_[middle + 1]) - tiling_.indexBase;
            if (next <= block) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low;
    }

    __aicore__ inline AscendC::LocalTensor<InputT> LoadX(int64_t row)
    {
        AscendC::LocalTensor<InputT> xLocal = xQueue_.template AllocTensor<InputT>();
        AscendC::DataCopyExtParams copy{1,
            static_cast<uint32_t>(kSddmmBsr8InputElements * sizeof(InputT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<InputT> pad{false, 0, 0, static_cast<InputT>(0)};
        AscendC::DataCopyPad(xLocal, matX_[row * kSddmmBsr8InputElements], copy, pad);
        xQueue_.EnQue(xLocal);
        xLocal = xQueue_.template DeQue<InputT>();
        if constexpr (SddmmBsr8ValueType<InputT>::code == SDDMM_VALUE_FP16) {
            AscendC::LocalTensor<float> xFp32 = xFp32Buffer_.template Get<float>();
            AscendC::Cast(xFp32, xLocal, AscendC::RoundMode::CAST_NONE, kSddmmBsr8InputElements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        return xLocal;
    }

    __aicore__ inline void LoadY(int64_t block, AscendC::LocalTensor<InputT> &yLocal)
    {
        const int64_t column = static_cast<int64_t>(columns_[block]) - tiling_.indexBase;
        AscendC::DataCopyExtParams copy{1,
            static_cast<uint32_t>(kSddmmBsr8InputElements * sizeof(InputT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<InputT> pad{false, 0, 0, static_cast<InputT>(0)};
        AscendC::DataCopyPad(yLocal, packedY_[column * kSddmmBsr8InputElements], copy, pad);
        yQueue_.EnQue(yLocal);
        yLocal = yQueue_.template DeQue<InputT>();
    }

    __aicore__ inline void ProcessBlock(int64_t block, AscendC::LocalTensor<InputT> &xLocal)
    {
        AscendC::LocalTensor<InputT> yLocal = yQueue_.template AllocTensor<InputT>();
        LoadY(block, yLocal);
        if constexpr (SddmmBsr8ValueType<InputT>::code == SDDMM_VALUE_FP16) {
            AscendC::LocalTensor<float> xFp32 = xFp32Buffer_.template Get<float>();
            AscendC::LocalTensor<float> yFp32 = yFp32Buffer_.template Get<float>();
            AscendC::Cast(yFp32, yLocal, AscendC::RoundMode::CAST_NONE, kSddmmBsr8InputElements);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeBlock(xFp32, yFp32);
        } else {
            ComputeBlock(xLocal, yLocal);
        }
        yQueue_.FreeTensor(yLocal);
        StoreBlock(block);
    }

    __aicore__ inline void ComputeBlock(AscendC::LocalTensor<float> &xLocal,
        AscendC::LocalTensor<float> &yLocal)
    {
        AscendC::LocalTensor<float> accumulator = accumulatorBuffer_.template Get<float>();
        auto *dst = reinterpret_cast<__ubuf__ float *>(accumulator.GetPhyAddr());
        auto *x = reinterpret_cast<__ubuf__ float *>(xLocal.GetPhyAddr());
        auto *y = reinterpret_cast<__ubuf__ float *>(yLocal.GetPhyAddr());
        SddmmBsr8DotBlockVf(dst, x, y);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline AscendC::LocalTensor<float> OrderedResult()
    {
        AscendC::LocalTensor<float> result = accumulatorBuffer_.template Get<float>();
        if (tiling_.blockOrder == 0) {
            return result;
        }
        AscendC::LocalTensor<float> reordered = reorderedBuffer_.template Get<float>();
        auto *dst = reinterpret_cast<__ubuf__ float *>(reordered.GetPhyAddr());
        auto *src = reinterpret_cast<__ubuf__ float *>(result.GetPhyAddr());
        SddmmBsr8TransposeVf(dst, src);
        AscendC::PipeBarrier<PIPE_V>();
        return reordered;
    }

    __aicore__ inline void StoreBlock(int64_t block)
    {
        AscendC::LocalTensor<float> result = OrderedResult();
        AscendC::LocalTensor<OutputT> output = outputQueue_.template AllocTensor<OutputT>();
        if constexpr (SddmmBsr8ValueType<OutputT>::code == SDDMM_VALUE_FP16) {
            AscendC::Cast(output, result, AscendC::RoundMode::CAST_ROUND, kSddmmBsr8OutputElements);
        } else {
            AscendC::Muls(output, result, 1.0F, kSddmmBsr8OutputElements);
        }
        AscendC::PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(output);
        output = outputQueue_.template DeQue<OutputT>();
        AscendC::DataCopyExtParams copy{1,
            static_cast<uint32_t>(kSddmmBsr8OutputElements * sizeof(OutputT)), 0, 0, 0};
        AscendC::DataCopyPad(values_[block * kSddmmBsr8OutputElements], output, copy);
        outputQueue_.FreeTensor(output);
    }

    SddmmTilingData tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    __gm__ const int32_t *offsets_{nullptr};
    __gm__ const int32_t *columns_{nullptr};
    AscendC::GlobalTensor<InputT> matX_;
    AscendC::GlobalTensor<InputT> packedY_;
    AscendC::GlobalTensor<OutputT> values_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> yQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xFp32Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> yFp32Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accumulatorBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reorderedBuffer_;
};

} // namespace sddmm_arch35

#endif // SDDMM_ARCH35_BSR8_VECTOR_KERNEL_H_
