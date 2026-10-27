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
 * \file sddmm_csr_cube_kernel.h
 * \brief DAV-3510 Cube path for the fixed CSR performance pattern.
 *
 * The performance cases contain 64 samples in every row.  Their logical first
 * column advances by 17 between adjacent rows.  Sixteen rows therefore fit in
 * one circular column band of
 *
 *     64 + (16 - 1) * 17 = 319 columns.
 *
 * This kernel computes a dense 16 x 320 band with Cube and stores it in a
 * 16 x 320 FP32 workspace slot.  The first 319 columns cover the logical
 * sample band and the final column is padding.  A later AIV epilogue gathers
 * the 64 sampled values for each row, applies alpha/beta, and casts to the
 * requested output type.  The padded 320-column stride keeps every row and
 * group 64-byte aligned.
 *
 * This header deliberately contains no public launcher.  Integration requires
 * a dedicated AIC launch followed by an AIV gather/fallback launch; the
 * existing generic sddmm_kernel is AIV-only.  Keeping the implementation in a
 * header lets the launch translation unit select FP16, BF16, or FP32 without
 * duplicating the Tensor API pipeline.
 */

#ifndef SDDMM_ARCH35_CSR_CUBE_KERNEL_H_
#define SDDMM_ARCH35_CSR_CUBE_KERNEL_H_

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "sddmm_csr_cube_traits.h"
#include "sddmm_csr_cube_gather_kernel.h"

namespace sddmm_arch35 {

/*
 * AIC-only dense-band producer.
 *
 * Each AIC processes whole 16-row groups in grid-stride order.  L1 holds one
 * A tile (16x128) and one B segment (128x<=64).  Each N segment uses one
 * K=128 Mmad operation.  A physical wrap of the circular 320-column band
 * creates at most one extra N segment; no Y duplication is required.
 */
template <typename T>
class SddmmCsrCubeBandKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR matX,
        GM_ADDR matY,
        GM_ADDR denseWorkspace,
        const SddmmTilingData *tiling,
        const SddmmCsrCubePlan *plan)
    {
        matX_ = matX;
        matY_ = matY;
        denseWorkspace_ = denseWorkspace;
        tiling_ = *tiling;
        plan_ = *plan;
    }

    __aicore__ static inline bool CanRun(
        const SddmmTilingData &tiling,
        const SddmmCsrCubePlan &plan)
    {
        return SddmmCsrCubeCommonCanRun(tiling, plan) &&
            SddmmCsrCubeSupportedTypes(tiling) &&
            tiling.inputType == SddmmCsrCubeTypeCode<T>::value;
    }

    __aicore__ inline void Process()
    {
        if (!CanRun(tiling_, plan_) || plan_.groupCount == 0) {
            return;
        }
        const int32_t blockId = static_cast<int32_t>(AscendC::GetBlockIdx());
        const int32_t blockCount = static_cast<int32_t>(AscendC::GetBlockNum());
        if (blockCount <= 0) {
            return;
        }

        auto gmX = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ T *>(matX_)),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn,
                AscendC::Te::LayoutTraitDefault<T>>(tiling_.m, tiling_.k));

        if (plan_.yPhysicalOrder == 0) {
            auto gmY = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                    reinterpret_cast<__gm__ T *>(matY_)),
                AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn,
                    AscendC::Te::LayoutTraitDefault<T>>(tiling_.k, tiling_.n));
            ProcessGroups(gmX, gmY, blockId, blockCount);
        } else {
            auto gmY = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                    reinterpret_cast<__gm__ T *>(matY_)),
                AscendC::Te::MakeFrameLayout<AscendC::Te::DNExtLayoutPtn,
                    AscendC::Te::LayoutTraitDefault<T>>(tiling_.k, tiling_.n));
            ProcessGroups(gmX, gmY, blockId, blockCount);
        }
    }

private:
    template <typename GmTensorX, typename GmTensorY>
    __aicore__ inline void ProcessGroups(
        GmTensorX &gmX,
        GmTensorY &gmY,
        int32_t blockId,
        int32_t blockCount)
    {
        AscendC::SetMMLayoutTransform(true);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);

        for (int32_t group = blockId; group < plan_.groupCount;
             group += blockCount) {
            ProcessGroup(gmX, gmY, group);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetMMLayoutTransform(false);
    }

    template <typename GmTensorX, typename GmTensorY>
    __aicore__ inline void ProcessGroup(
        GmTensorX &gmX,
        GmTensorY &gmY,
        int32_t group)
    {
        const int32_t rowStart = group * kSddmmCsrCubeRows;
        const int64_t logicalStart64 =
            (static_cast<int64_t>(plan_.logicalFirstColumn) +
                static_cast<int64_t>(rowStart) * plan_.rowColumnStride) %
            tiling_.n;
        const int32_t logicalStart = static_cast<int32_t>(logicalStart64);

        auto gmA = gmX.Slice(
            AscendC::Te::MakeCoord(rowStart, 0),
            AscendC::Te::MakeShape(kSddmmCsrCubeRows,
                kSddmmCsrCubeReduction));
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, T>(0),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn,
                AscendC::Te::LayoutTraitDefault<T>>(
                    kSddmmCsrCubeRows, kSddmmCsrCubeReduction));

        __gm__ float *groupOutput =
            reinterpret_cast<__gm__ float *>(denseWorkspace_) +
            static_cast<int64_t>(group) * kSddmmCsrCubeRows *
                kSddmmCsrCubeWorkspaceBand;
        auto gmC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                groupOutput),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn,
                AscendC::Std::Int<kSddmmCsrCubeL0cC0>>(
                    kSddmmCsrCubeRows, kSddmmCsrCubeWorkspaceBand));

        bool loadA = true;
        int32_t logicalColumn = 0;
        while (logicalColumn < kSddmmCsrCubeWorkspaceBand) {
            const int32_t physicalColumn = static_cast<int32_t>(
                (static_cast<int64_t>(logicalStart) + logicalColumn) %
                tiling_.n);
            int32_t currentN = Min(SddmmCsrCubeNtile<T>::value,
                kSddmmCsrCubeWorkspaceBand - logicalColumn);
            currentN = Min(currentN,
                static_cast<int32_t>(tiling_.n - physicalColumn));

            LoadAndComputeSegment(gmA, gmY, gmC, tensorAL1,
                physicalColumn, logicalColumn, currentN, loadA);
            loadA = false;
            logicalColumn += currentN;
        }
    }

    template <typename L1TensorA, typename L1TensorB, typename L0TensorC>
    __aicore__ inline void ComputeReductionTile(
        L1TensorA &tensorAL1, L1TensorB &tensorBL1, L0TensorC &tensorL0C,
        int32_t kOffset, int32_t currentN, bool initializeC)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        auto tensorAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, T>(0),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn,
                AscendC::Te::LayoutTraitDefault<T>>(
                    kSddmmCsrCubeRows, kSddmmCsrCubeKtile));
        auto tensorBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, T>(0),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn,
                AscendC::Te::LayoutTraitDefault<T>>(
                    kSddmmCsrCubeKtile, currentN));
        auto tensorAL1Tile = tensorAL1.Slice(
            AscendC::Te::MakeCoord(0, kOffset),
            AscendC::Te::MakeShape(kSddmmCsrCubeRows,
                kSddmmCsrCubeKtile));
        auto tensorBL1Tile = tensorBL1.Slice(
            AscendC::Te::MakeCoord(kOffset, 0),
            AscendC::Te::MakeShape(kSddmmCsrCubeKtile, currentN));

        AscendC::Te::Copy(
            AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{}),
            tensorAL0, tensorAL1Tile);
        AscendC::Te::Copy(
            AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{}),
            tensorBL0, tensorBL1Tile);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);

        const bool finalK =
            kOffset + kSddmmCsrCubeKtile == kSddmmCsrCubeReduction;
        AscendC::Te::MmadParams params{
            static_cast<uint16_t>(kSddmmCsrCubeRows),
            static_cast<uint16_t>(currentN),
            static_cast<uint16_t>(kSddmmCsrCubeKtile),
            finalK ? kSddmmCsrCubeFinalAccumulation :
                kSddmmCsrCubeNonFinalAccumulation,
            initializeC};
        auto mmad = AscendC::Te::MakeMmad(
            AscendC::Te::MmadOperation{},
            AscendC::Te::MmadTraitDefault{});
        AscendC::Te::Mmad(mmad.with(params), tensorL0C,
            tensorAL0, tensorBL0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    }

    template <typename GmTensorC, typename L0TensorC>
    __aicore__ inline void StoreSegment(
        GmTensorC &gmC, L0TensorC &tensorL0C, int32_t logicalColumn, int32_t currentN)
    {
        auto gmCBlock = gmC.Slice(
            AscendC::Te::MakeCoord(0, logicalColumn),
            AscendC::Te::MakeShape(kSddmmCsrCubeRows, currentN));
        AscendC::Te::FixpipeParams fixpipeParams;
        fixpipeParams.unitFlag = kSddmmCsrCubeFinalAccumulation;
        auto copyL0C2GM = AscendC::Te::MakeCopy(
            AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(copyL0C2GM.with(fixpipeParams),
            gmCBlock, tensorL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    }

    template <typename GmTensorA, typename GmTensorY, typename GmTensorC,
        typename L1TensorA>
    __aicore__ inline void LoadAndComputeSegment(
        GmTensorA &gmA,
        GmTensorY &gmY,
        GmTensorC &gmC,
        L1TensorA &tensorAL1,
        int32_t physicalColumn,
        int32_t logicalColumn,
        int32_t currentN,
        bool loadA)
    {
        constexpr int64_t aL1Bytes =
            static_cast<int64_t>(kSddmmCsrCubeRows) *
            kSddmmCsrCubeReduction * sizeof(T);
        auto gmB = gmY.Slice(
            AscendC::Te::MakeCoord(0, physicalColumn),
            AscendC::Te::MakeShape(kSddmmCsrCubeReduction, currentN));
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, T>(aL1Bytes),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn,
                AscendC::Te::LayoutTraitDefault<T>>(
                    kSddmmCsrCubeReduction, currentN));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        if (loadA) {
            AscendC::Te::Copy(
                AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{}),
                tensorAL1, gmA);
        }
        AscendC::Te::Copy(
            AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{}),
            tensorBL1, gmB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        auto tensorL0C = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn,
                AscendC::Std::Int<kSddmmCsrCubeL0cC0>>(
                    kSddmmCsrCubeRows, currentN));

        bool initializeC = true;
        for (int32_t kOffset = 0; kOffset < kSddmmCsrCubeReduction;
             kOffset += kSddmmCsrCubeKtile) {
            ComputeReductionTile(tensorAL1, tensorBL1, tensorL0C,
                kOffset, currentN, initializeC);
            initializeC = false;
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        StoreSegment(gmC, tensorL0C, logicalColumn, currentN);
    }

    __aicore__ static inline int32_t Min(int32_t lhs, int32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    GM_ADDR matX_{nullptr};
    GM_ADDR matY_{nullptr};
    GM_ADDR denseWorkspace_{nullptr};
    SddmmTilingData tiling_{};
    SddmmCsrCubePlan plan_{};
};

} // namespace sddmm_arch35

#endif // SDDMM_ARCH35_CSR_CUBE_KERNEL_H_
