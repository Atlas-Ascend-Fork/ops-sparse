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

#ifndef SDDMM_ARCH35_CSR_CUBE_GATHER_KERNEL_H_
#define SDDMM_ARCH35_CSR_CUBE_GATHER_KERNEL_H_

#include "sddmm_csr_cube_traits.h"

namespace sddmm_arch35 {

/*
 * A wrapped row needs an arbitrary cyclic rotation of one 64-element vector.
 * Level-2 tensor operations require 32-byte-aligned slice addresses, so a
 * two-slice Muls is not valid for every split point.  Build the cyclic index
 * in one B32 register and gather from the aligned UB base instead.  The
 * register path keeps the complete reorder and scale operation on PIPE_V.
 */
__simd_vf__ inline void SddmmCsrCubeRotateAndScaleVf(
    __ubuf__ float *dst,
    __ubuf__ float *src,
    int32_t highCount,
    float scale)
{
    using namespace AscendC::MicroAPI;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<int32_t> logicalIndex;
    RegTensor<int32_t> moduloMask;
    RegTensor<float> gathered;

    Arange(logicalIndex, highCount);
    Duplicate(moduloMask, kSddmmCsrCubeSamplesPerRow - 1, mask);
    And(logicalIndex, logicalIndex, moduloMask, mask);
    Gather(gathered, src,
        (RegTensor<uint32_t> &)logicalIndex, mask);
    Muls(gathered, gathered, scale, mask);
    StoreAlign(dst, gathered, mask);
}

/*
 * AIV epilogue for the dense-band producer.
 *
 * The CSR validator guarantees that every row has exactly 64 sorted samples
 * and that their circular starts follow the plan.  The useful values for row
 * r are contiguous in the group's logical workspace band.  If the sampled
 * interval wraps at N, the CSR columns are sorted as [0, head), [start, N),
 * so the two portions are rotated while building the output vector.
 */
template <typename T>
class SddmmCsrCubeGatherKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR sparseValues,
        GM_ADDR denseWorkspace,
        const SddmmTilingData *tiling,
        const SddmmCsrCubePlan *plan,
        AscendC::TPipe *pipe)
    {
        tiling_ = *tiling;
        plan_ = *plan;
        pipe_ = pipe;
        values_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(sparseValues),
            static_cast<uint64_t>(tiling_.nnzUnits));
        dense_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(denseWorkspace),
            static_cast<uint64_t>(plan_.groupCount) *
                kSddmmCsrCubeRows * kSddmmCsrCubeWorkspaceBand);

        // The identity FP16/BF16 epilogue gathers a complete 16-row group
        // with one strided DMA.  The same queues remain valid for the
        // row-at-a-time fallback because those operations use only the first
        // 64 elements.
        pipe_->InitBuffer(denseQueue_, 1,
            kSddmmCsrCubeSamplesPerGroup * sizeof(float));
        pipe_->InitBuffer(oldQueue_, 1,
            kSddmmCsrCubeSamplesPerRow * sizeof(T));
        pipe_->InitBuffer(outputQueue_, 1,
            kSddmmCsrCubeSamplesPerGroup * sizeof(T));
        pipe_->InitBuffer(resultBuffer_,
            kSddmmCsrCubeSamplesPerRow * sizeof(float));
        pipe_->InitBuffer(oldFp32Buffer_,
            kSddmmCsrCubeSamplesPerRow * sizeof(float));
    }

    __aicore__ static inline bool CanRun(
        const SddmmTilingData &tiling,
        const SddmmCsrCubePlan &plan)
    {
        return SddmmCsrCubeCommonCanRun(tiling, plan) &&
            SddmmCsrCubeSupportedTypes(tiling) &&
            tiling.outputType == SddmmCsrCubeTypeCode<T>::value &&
            tiling.pointerMode == 0;
    }

    __aicore__ inline void Process()
    {
        if (!CanRun(tiling_, plan_) || plan_.groupCount == 0) {
            return;
        }
        const int64_t block = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t blockCount =
            static_cast<int64_t>(AscendC::GetBlockNum());
        if (blockCount <= 0) {
            return;
        }
        if constexpr (SddmmCsrCubeTypeCode<T>::value !=
            SDDMM_VALUE_FP32) {
            if (tiling_.alphaReal == 1.0F && tiling_.betaReal == 0.0F) {
                ProcessIdentityGroups(block, blockCount);
                return;
            }
        }

        const int64_t rows = static_cast<int64_t>(plan_.groupCount) *
            kSddmmCsrCubeRows;
        for (int64_t row = block; row < rows; row += blockCount) {
            ProcessRow(row);
        }
    }

private:
    /*
     * Most task groups do not wrap at N.  Their 16 useful 64-value intervals
     * are separated by a fixed 273-float gap in the 16x320 workspace:
     *
     *   (next row * 320 + next row * 17) - (current start + 64)
     *       = 320 + 17 - 64 = 273.
     *
     * DataCopyPad can therefore compact all 16 intervals into one UB tensor.
     * A single vector cast and contiguous store replace sixteen queue/DMA/cast
     * cycles.  Groups touching the physical wrap retain ProcessRow, including
     * its CSR sorted-order rotation.
     */
    __aicore__ inline void ProcessIdentityGroups(
        int64_t block,
        int64_t blockCount)
    {
        for (int64_t group = block; group < plan_.groupCount;
             group += blockCount) {
            if (CanGatherIdentityGroup(group)) {
                GatherIdentityGroup(group);
                continue;
            }
            const int64_t rowStart = group * kSddmmCsrCubeRows;
            for (int32_t rowInGroup = 0;
                 rowInGroup < kSddmmCsrCubeRows; ++rowInGroup) {
                ProcessRow(rowStart + rowInGroup);
            }
        }
    }

    __aicore__ inline bool CanGatherIdentityGroup(int64_t group) const
    {
        const int64_t rowStart = group * kSddmmCsrCubeRows;
        const int64_t logicalStart =
            (static_cast<int64_t>(plan_.logicalFirstColumn) +
                rowStart * plan_.rowColumnStride) % tiling_.n;
        return logicalStart + kSddmmCsrCubeLogicalBand <= tiling_.n;
    }

    __aicore__ inline void GatherIdentityGroup(int64_t group)
    {
        constexpr uint32_t denseBlockBytes =
            kSddmmCsrCubeSamplesPerRow * sizeof(float);
        constexpr uint32_t denseSourceGapBytes =
            (kSddmmCsrCubeWorkspaceBand +
                kSddmmCsrCubeRowColumnStride -
                kSddmmCsrCubeSamplesPerRow) * sizeof(float);
        const uint64_t denseOffset = static_cast<uint64_t>(group) *
            kSddmmCsrCubeRows * kSddmmCsrCubeWorkspaceBand;
        const int64_t valueOffset = group *
            static_cast<int64_t>(kSddmmCsrCubeSamplesPerGroup);

        AscendC::LocalTensor<float> denseLocal =
            denseQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams denseCopy{
            static_cast<uint16_t>(kSddmmCsrCubeRows),
            denseBlockBytes, denseSourceGapBytes, 0, 0};
        AscendC::DataCopyPadExtParams<float> densePad{
            false, 0, 0, 0.0F};
        AscendC::DataCopyPad(denseLocal, dense_[denseOffset],
            denseCopy, densePad);
        denseQueue_.EnQue(denseLocal);
        denseLocal = denseQueue_.DeQue<float>();

        AscendC::LocalTensor<T> output = outputQueue_.AllocTensor<T>();
        AscendC::Cast(output, denseLocal,
            AscendC::RoundMode::CAST_ROUND,
            kSddmmCsrCubeSamplesPerGroup);
        AscendC::PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<T>();
        AscendC::DataCopyExtParams outputCopy{1,
            static_cast<uint32_t>(
                kSddmmCsrCubeSamplesPerGroup * sizeof(T)),
            0, 0, 0};
        AscendC::DataCopyPad(values_[valueOffset], output, outputCopy);
        outputQueue_.FreeTensor(output);
        denseQueue_.FreeTensor(denseLocal);
    }

    __aicore__ inline AscendC::LocalTensor<float> LoadDenseRow(uint64_t denseOffset)
    {
        AscendC::LocalTensor<float> denseLocal =
            denseQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams denseCopy{1,
            static_cast<uint32_t>(
                kSddmmCsrCubeSamplesPerRow * sizeof(float)),
            0, 0, 0};
        AscendC::DataCopyPadExtParams<float> densePad{
            false, 0, 0, 0.0F};
        AscendC::DataCopyPad(denseLocal, dense_[denseOffset],
            denseCopy, densePad);
        denseQueue_.EnQue(denseLocal);
        return denseQueue_.DeQue<float>();
    }

    __aicore__ inline void ProcessRow(int64_t row)
    {
        const int64_t group = row / kSddmmCsrCubeRows;
        const int32_t rowInGroup = static_cast<int32_t>(
            row - group * kSddmmCsrCubeRows);
        const uint64_t denseOffset = static_cast<uint64_t>(group) *
                kSddmmCsrCubeRows * kSddmmCsrCubeWorkspaceBand +
            static_cast<uint64_t>(rowInGroup) *
                kSddmmCsrCubeWorkspaceBand +
            static_cast<uint64_t>(rowInGroup) *
                kSddmmCsrCubeRowColumnStride;

        AscendC::LocalTensor<float> denseLocal = LoadDenseRow(denseOffset);

        const int32_t rowLogicalStart = static_cast<int32_t>(
            (static_cast<int64_t>(plan_.logicalFirstColumn) +
                row * plan_.rowColumnStride) % tiling_.n);
        const int32_t highCount = Min(kSddmmCsrCubeSamplesPerRow,
            static_cast<int32_t>(tiling_.n - rowLogicalStart));
        const int32_t lowCount =
            kSddmmCsrCubeSamplesPerRow - highCount;
        AscendC::LocalTensor<float> result = resultBuffer_.Get<float>();

        if constexpr (SddmmCsrCubeTypeCode<T>::value !=
            SDDMM_VALUE_FP32) {
            if (lowCount == 0 && tiling_.alphaReal == 1.0F &&
                tiling_.betaReal == 0.0F) {
                StoreResult(denseLocal,
                    row * kSddmmCsrCubeSamplesPerRow);
                denseQueue_.FreeTensor(denseLocal);
                return;
            }
        }

        if (tiling_.betaReal == 0.0F) {
            BuildResultWithoutBeta(denseLocal, result,
                highCount, lowCount);
        } else {
            BuildResultWithBeta(denseLocal, result,
                row * kSddmmCsrCubeSamplesPerRow,
                highCount, lowCount);
        }
        denseQueue_.FreeTensor(denseLocal);
        StoreResult(result, row * kSddmmCsrCubeSamplesPerRow);
    }

    __aicore__ inline void BuildResultWithoutBeta(
        AscendC::LocalTensor<float> &denseLocal,
        AscendC::LocalTensor<float> &result,
        int32_t highCount,
        int32_t lowCount)
    {
        RotateAndScale(result, denseLocal, highCount, lowCount,
            tiling_.alphaReal);
    }

    __aicore__ inline void BuildResultWithBeta(
        AscendC::LocalTensor<float> &denseLocal,
        AscendC::LocalTensor<float> &result,
        int64_t valueOffset,
        int32_t highCount,
        int32_t lowCount)
    {
        AscendC::LocalTensor<T> oldValues = oldQueue_.AllocTensor<T>();
        AscendC::DataCopyExtParams oldCopy{1,
            static_cast<uint32_t>(
                kSddmmCsrCubeSamplesPerRow * sizeof(T)),
            0, 0, 0};
        AscendC::DataCopyPadExtParams<T> oldPad{
            false, 0, 0, static_cast<T>(0.0F)};
        AscendC::DataCopyPad(oldValues, values_[valueOffset],
            oldCopy, oldPad);
        oldQueue_.EnQue(oldValues);
        oldValues = oldQueue_.DeQue<T>();

        AscendC::LocalTensor<float> oldFp32 = oldFp32Buffer_.Get<float>();
        if constexpr (SddmmCsrCubeTypeCode<T>::value != SDDMM_VALUE_FP32) {
            AscendC::Cast(oldFp32, oldValues,
                AscendC::RoundMode::CAST_NONE,
                kSddmmCsrCubeSamplesPerRow);
            AscendC::PipeBarrier<PIPE_V>();
        }

        RotateAndScale(result, denseLocal, highCount, lowCount,
            tiling_.alphaReal);
        if constexpr (SddmmCsrCubeTypeCode<T>::value ==
            SDDMM_VALUE_FP32) {
            AscendC::Muls(oldFp32, oldValues, tiling_.betaReal,
                kSddmmCsrCubeSamplesPerRow);
        } else {
            AscendC::Muls(oldFp32, oldFp32, tiling_.betaReal,
                kSddmmCsrCubeSamplesPerRow);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(result, result, oldFp32,
            kSddmmCsrCubeSamplesPerRow);
        AscendC::PipeBarrier<PIPE_V>();
        oldQueue_.FreeTensor(oldValues);
    }

    __aicore__ inline void RotateAndScale(
        AscendC::LocalTensor<float> &result,
        AscendC::LocalTensor<float> &denseLocal,
        int32_t highCount,
        int32_t lowCount,
        float scale)
    {
        if (lowCount == 0) {
            AscendC::Muls(result, denseLocal, scale,
                kSddmmCsrCubeSamplesPerRow);
        } else {
            auto *dst = reinterpret_cast<__ubuf__ float *>(
                result.GetPhyAddr());
            auto *src = reinterpret_cast<__ubuf__ float *>(
                denseLocal.GetPhyAddr());
            SddmmCsrCubeRotateAndScaleVf(
                dst, src, highCount, scale);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreResult(
        AscendC::LocalTensor<float> &result,
        int64_t valueOffset)
    {
        AscendC::LocalTensor<T> output = outputQueue_.AllocTensor<T>();
        if constexpr (SddmmCsrCubeTypeCode<T>::value ==
            SDDMM_VALUE_FP32) {
            AscendC::Muls(output, result, 1.0F,
                kSddmmCsrCubeSamplesPerRow);
        } else {
            AscendC::Cast(output, result,
                AscendC::RoundMode::CAST_ROUND,
                kSddmmCsrCubeSamplesPerRow);
        }
        AscendC::PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<T>();
        AscendC::DataCopyExtParams outputCopy{1,
            static_cast<uint32_t>(
                kSddmmCsrCubeSamplesPerRow * sizeof(T)),
            0, 0, 0};
        AscendC::DataCopyPad(values_[valueOffset], output, outputCopy);
        outputQueue_.FreeTensor(output);
    }

    __aicore__ static inline int32_t Min(int32_t lhs, int32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    AscendC::TPipe *pipe_{nullptr};
    AscendC::TQue<AscendC::TPosition::VECIN, 1> denseQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> oldQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oldFp32Buffer_;
    AscendC::GlobalTensor<T> values_;
    AscendC::GlobalTensor<float> dense_;
    SddmmTilingData tiling_{};
    SddmmCsrCubePlan plan_{};
};

} // namespace sddmm_arch35

#endif // SDDMM_ARCH35_CSR_CUBE_GATHER_KERNEL_H_
