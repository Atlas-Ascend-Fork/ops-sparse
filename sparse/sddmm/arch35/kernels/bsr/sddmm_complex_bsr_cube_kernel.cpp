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
 * \file sddmm_complex_bsr_cube_kernel.cpp
 * \brief Split-real Cube fast path for complex64 BSR SDDMM.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "tensor_api/tensor.h"

#include "../../sddmm_complex_bsr_cube_common.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
#error "SDDMM complex64 BSR Cube kernel is only available on DAV-3510 / Ascend 950."
#endif

using namespace AscendC;

namespace {

constexpr uint32_t kComplexBsrVectorThreads = 256U;
constexpr int32_t kComplexBsrPackTileComplex = 8192;
constexpr int32_t kComplexBsrEpilogueTileComponents = 4096;
constexpr int32_t kComplexBsrL0cC0 = 16;
constexpr uint8_t kComplexBsrNonFinalAccumulation = 2;
constexpr uint8_t kComplexBsrFinalAccumulation = 3;

using ComplexBsrL0cLayoutTrait =
    AscendC::Te::LayoutTrait<float, AscendC::Std::Int<kComplexBsrL0cC0>>;

template <int32_t Reduction>
class SddmmComplexBsrPackKernel {
    static_assert(Reduction == 64 || Reduction == 128,
        "unsupported complex BSR Cube reduction");
    static constexpr int32_t kComplexBsrK = Reduction;
    static constexpr int32_t kComplexBsrExpandedK = Reduction * 2;

public:
    __aicore__ inline void Init(
        GM_ADDR sparseOffsets,
        GM_ADDR sparseColumns,
        GM_ADDR matX,
        GM_ADDR matY,
        GM_ADDR packedX,
        GM_ADDR packedY,
        const SddmmTilingData *tiling,
        AscendC::TPipe *pipe)
    {
        tiling_ = *tiling;
        pipe_ = pipe;
        offsets_ = reinterpret_cast<__gm__ const int32_t *>(sparseOffsets);
        columns_ = reinterpret_cast<__gm__ const int32_t *>(sparseColumns);
        xParts_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matX),
            static_cast<uint64_t>(tiling_.m) * kComplexBsrK * 2ULL);
        yParts_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matY),
            static_cast<uint64_t>(kComplexBsrK) * tiling_.n * 2ULL);
        packedX_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(packedX),
            static_cast<uint64_t>(tiling_.m) * kComplexBsrExpandedK * 2ULL);
        packedY_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(packedY),
            static_cast<uint64_t>(kComplexBsrExpandedK) * tiling_.n);

        // 160 KiB leaves more than 80 KiB of DAV-3510 UB available for the
        // GatherMask temporary allocation and runtime bookkeeping.
        pipe_->InitBuffer(sourceBuffer_,
            kComplexBsrPackTileComplex * 2 * sizeof(float));
        pipe_->InitBuffer(realBuffer_,
            kComplexBsrPackTileComplex * sizeof(float));
        pipe_->InitBuffer(imagBuffer_,
            kComplexBsrPackTileComplex * sizeof(float));
        pipe_->InitBuffer(negativeImagBuffer_,
            kComplexBsrPackTileComplex * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t block = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t blockCount =
            static_cast<int64_t>(AscendC::GetBlockNum());
        for (int64_t blockRow = block; blockRow < tiling_.blockRows;
             blockRow += blockCount) {
            // Cube reads X only for rows owning at least one physical block.
            // Empty packed rows may stay unwritten; comparing endpoints works
            // for either index base and does not require unique columns.
            if (offsets_[blockRow] != offsets_[blockRow + 1]) {
                PackXBlockRow(blockRow);
            }
        }

        if (sddmm_arch35::SddmmComplexBsrCubeUseSparseYPack(tiling_)) {
            for (int64_t slot = block; slot < tiling_.nnzUnits; slot += blockCount) {
                const int32_t blockColumn = columns_[slot];
                bool firstOccurrence = true;
                for (int64_t previous = 0; previous < slot; ++previous) {
                    if (columns_[previous] == blockColumn) {
                        firstOccurrence = false;
                        break;
                    }
                }
                if (firstOccurrence) {
                    PackSparseYBlock(static_cast<int64_t>(blockColumn) -
                        tiling_.indexBase);
                }
            }
            return;
        }
        const int64_t tilesPerRow =
            (tiling_.n + kComplexBsrPackTileComplex - 1) /
            kComplexBsrPackTileComplex;
        const int64_t tileCount =
            static_cast<int64_t>(kComplexBsrK) * tilesPerRow;
        for (int64_t tile = block; tile < tileCount; tile += blockCount) {
            const int64_t reduction = tile / tilesPerRow;
            const int64_t columnTile = tile - reduction * tilesPerRow;
            const int64_t columnStart =
                columnTile * kComplexBsrPackTileComplex;
            const int32_t current = static_cast<int32_t>(
                tiling_.n - columnStart < kComplexBsrPackTileComplex ?
                    tiling_.n - columnStart : kComplexBsrPackTileComplex);
            PackYTile(reduction, columnStart, current);
        }
    }

private:
    __aicore__ inline void Deinterleave(int32_t complexCount)
    {
        AscendC::LocalTensor<float> source = sourceBuffer_.Get<float>();
        AscendC::LocalTensor<float> real = realBuffer_.Get<float>();
        AscendC::LocalTensor<float> imag = imagBuffer_.Get<float>();
        uint64_t selected = 0;
        const uint32_t partCount =
            static_cast<uint32_t>(complexCount) * 2U;
        const AscendC::GatherMaskParams params{1, 1, 0, 0};
        AscendC::GatherMask(real, source, static_cast<uint8_t>(1), true,
            partCount, params, selected);
        AscendC::GatherMask(imag, source, static_cast<uint8_t>(2), true,
            partCount, params, selected);
        // All X/Y pack variants reuse sourceBuffer_. Its next MTE2 load must
        // wait for both vector gathers; output-buffer MTE3_V is a separate edge.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    }

    __aicore__ inline void PackXBlockRow(int64_t blockRow)
    {
        const int32_t complexCount = tiling_.blockDim * kComplexBsrK;
        const int64_t sourceOffset =
            blockRow * static_cast<int64_t>(complexCount) * 2;
        AscendC::LocalTensor<float> source = sourceBuffer_.Get<float>();
        AscendC::LocalTensor<float> real = realBuffer_.Get<float>();
        AscendC::LocalTensor<float> imag = imagBuffer_.Get<float>();
        AscendC::LocalTensor<float> negativeImag =
            negativeImagBuffer_.Get<float>();

        AscendC::DataCopy(source, xParts_[sourceOffset], complexCount * 2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        Deinterleave(complexCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(negativeImag, imag, -1.0F, complexCount);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const uint32_t rowBytes = kComplexBsrK * sizeof(float);
        AscendC::DataCopyExtParams stridedStore{
            static_cast<uint16_t>(tiling_.blockDim), rowBytes,
            0, rowBytes, 0};
        const int64_t packedBlock = blockRow *
            static_cast<int64_t>(tiling_.blockDim) * 2 *
            kComplexBsrExpandedK;
        const int64_t imaginaryRows =
            static_cast<int64_t>(tiling_.blockDim) *
            kComplexBsrExpandedK;
        AscendC::DataCopyPad(packedX_[packedBlock], real, stridedStore);
        AscendC::DataCopyPad(
            packedX_[packedBlock + kComplexBsrK],
            negativeImag, stridedStore);
        AscendC::DataCopyPad(
            packedX_[packedBlock + imaginaryRows], imag, stridedStore);
        AscendC::DataCopyPad(
            packedX_[packedBlock + imaginaryRows + kComplexBsrK],
            real, stridedStore);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void PackYTile(
        int64_t reduction, int64_t columnStart, int32_t complexCount)
    {
        const int64_t sourceOffset =
            (reduction * tiling_.n + columnStart) * 2;
        AscendC::LocalTensor<float> source = sourceBuffer_.Get<float>();
        AscendC::LocalTensor<float> real = realBuffer_.Get<float>();
        AscendC::LocalTensor<float> imag = imagBuffer_.Get<float>();
        AscendC::DataCopy(source, yParts_[sourceOffset], complexCount * 2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        Deinterleave(complexCount);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(
            packedY_[reduction * tiling_.n + columnStart],
            real, complexCount);
        AscendC::DataCopy(
            packedY_[(kComplexBsrK + reduction) * tiling_.n + columnStart],
            imag, complexCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void PackSparseYBlock(int64_t blockColumn)
    {
        constexpr int32_t kBlockColumns = 64;
        const int64_t columnStart = blockColumn * kBlockColumns;
        AscendC::LocalTensor<float> source = sourceBuffer_.Get<float>();
        AscendC::LocalTensor<float> real = realBuffer_.Get<float>();
        AscendC::LocalTensor<float> imag = imagBuffer_.Get<float>();
        // Gather one Kx64 complex column block into the existing 64 KiB
        // source buffer, then write its two planes into the original layout.
        AscendC::DataCopyExtParams load{
            static_cast<uint16_t>(kComplexBsrK),
            kBlockColumns * 2U * sizeof(float),
            static_cast<uint32_t>((tiling_.n - kBlockColumns) * 2 * sizeof(float)),
            0, 0};
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0F};
        AscendC::DataCopyPad(source, yParts_[columnStart * 2], load, noPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        Deinterleave(kComplexBsrK * kBlockColumns);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams store{
            static_cast<uint16_t>(kComplexBsrK),
            kBlockColumns * sizeof(float), 0,
            static_cast<uint32_t>((tiling_.n - kBlockColumns) * sizeof(float)), 0};
        AscendC::DataCopyPad(packedY_[columnStart], real, store);
        AscendC::DataCopyPad(
            packedY_[static_cast<int64_t>(kComplexBsrK) * tiling_.n + columnStart],
            imag, store);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    AscendC::TPipe *pipe_{nullptr};
    __gm__ const int32_t *offsets_{nullptr};
    __gm__ const int32_t *columns_{nullptr};
    AscendC::TBuf<AscendC::TPosition::VECCALC> sourceBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> realBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> imagBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> negativeImagBuffer_;
    AscendC::GlobalTensor<float> xParts_;
    AscendC::GlobalTensor<float> yParts_;
    AscendC::GlobalTensor<float> packedX_;
    AscendC::GlobalTensor<float> packedY_;
    SddmmTilingData tiling_{};
};

// Four adjacent BSR16 columns share one M32xN64xK256 product. The A tile
// has the same layout as the single-block path and remains resident per row.
template <typename TensorAL1, typename TensorBL1, typename TensorAL0,
    typename TensorBL0, typename TensorL0C, typename GmPackedY>
__aicore__ inline void SddmmComplexBsrComputeFourB16(
    TensorAL1 &tensorAL1, TensorBL1 &tensorBL1, TensorAL0 &tensorAL0,
    TensorBL0 &tensorBL0, TensorL0C &tensorL0C, GmPackedY &gmPackedY,
    int64_t blockColumn, bool loadA)
{
    using Tile = sddmm_arch35::SddmmComplexBsrCubeGroupedTileConfig;
    constexpr int32_t kRows = Tile::kRows;
    constexpr int32_t kColumns = Tile::kColumns;
    constexpr int32_t kReduction = Tile::kReduction;
    constexpr int32_t kBlockColumns = Tile::kBlockColumns;
    WaitFlag<HardEvent::MTE1_MTE2>(1);
    auto gmYGroup = gmPackedY.Slice(Te::MakeCoord(0, blockColumn * kBlockColumns),
        Te::MakeShape(kReduction, kColumns));
    Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}), tensorBL1, gmYGroup);
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::FIX_M>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    if (loadA) {
        Te::Copy(Te::MakeCopy(Te::CopyL12L0A{}), tensorAL0, tensorAL1);
    }
    Te::Copy(Te::MakeCopy(Te::CopyL12L0B{}), tensorBL0, tensorBL1);
    SetFlag<HardEvent::MTE1_MTE2>(1);
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Te::MmadParams params{kRows, kColumns, kReduction,
        kComplexBsrFinalAccumulation, true};
    auto kMmad = Te::MakeMmad(Te::MmadOperation{}, Te::MmadTraitDefault{});
    Te::Mmad(kMmad.with(params), tensorL0C, tensorAL0, tensorBL0);
    SetFlag<HardEvent::M_MTE1>(0);

}

template <typename TensorL0C, typename GmScratch>
__aicore__ inline void SddmmComplexBsrStoreFourB16(
    TensorL0C &tensorL0C, GmScratch &gmScratch, int64_t blockSlot)
{
    using Tile = sddmm_arch35::SddmmComplexBsrCubeGroupedTileConfig;
    constexpr int32_t kRows = Tile::kRows;
    constexpr int32_t kBlockColumns = Tile::kBlockColumns;
    Te::FixpipeParams fixpipeParams;
    fixpipeParams.unitFlag = kComplexBsrFinalAccumulation;
    auto kFixpipe = Te::MakeCopy(Te::CopyL0C2GM{});
    for (int32_t part = 0; part < Tile::kBlockCount; ++part) {
        auto source = tensorL0C.Slice(Te::MakeCoord(0, part * kBlockColumns),
            Te::MakeShape(kRows, kBlockColumns));
        auto destination = gmScratch.Slice(Te::MakeCoord((blockSlot + part) * kRows, 0),
            Te::MakeShape(kRows, kBlockColumns));
        // Unit flags belong to 512-byte L0C regions. These disjoint 2 KiB
        // slices each consume their region exactly once with final flag 3.
        Te::Copy(kFixpipe.with(fixpipeParams), destination, source);
    }
    SetFlag<HardEvent::FIX_M>(0);
}

__aicore__ inline void SddmmComplexBsrCubeFourB16(
    GM_ADDR packedY, GM_ADDR scratch, const SddmmTilingData &tiling,
    int64_t blockSlot, int64_t blockColumn, bool loadA)
{
    using Tile = sddmm_arch35::SddmmComplexBsrCubeGroupedTileConfig;
    constexpr int32_t kRows = Tile::kRows;
    constexpr int32_t kColumns = Tile::kColumns;
    constexpr int32_t kReduction = Tile::kReduction;
    constexpr int32_t kBlockColumns = Tile::kBlockColumns;
    constexpr int32_t kABytes = static_cast<int32_t>(Tile::kABytes);
    auto tensorAL1 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::L1, float>(0),
        Te::MakeFrameLayout<Te::NZLayoutPtn, Te::LayoutTraitDefault<float>>(
            kRows, kReduction));
    auto tensorBL1 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::L1, float>(kABytes),
        Te::MakeFrameLayout<Te::ZNLayoutPtn, Te::LayoutTraitDefault<float>>(
            kReduction, kColumns));
    auto tensorAL0 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::L0A, float>(0),
        Te::MakeFrameLayout<Te::NZLayoutPtn, Te::LayoutTraitDefault<float>>(
            kRows, kReduction));
    auto tensorBL0 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::L0B, float>(0),
        Te::MakeFrameLayout<Te::ZNLayoutPtn, Te::LayoutTraitDefault<float>>(
            kReduction, kColumns));
    auto tensorL0C = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::L0C, float>(0),
        Te::MakeFrameLayout<Te::NZLayoutPtn, ComplexBsrL0cLayoutTrait>(kRows, kColumns));
    auto gmPackedY = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ float *>(packedY)),
        Te::MakeFrameLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<float>>(
            kReduction, tiling.n));
    auto gmScratch = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ float *>(scratch)),
        Te::MakeFrameLayout<Te::NDExtLayoutPtn, ComplexBsrL0cLayoutTrait>(
            tiling.nnzUnits * kRows, kBlockColumns));

    SddmmComplexBsrComputeFourB16(tensorAL1, tensorBL1, tensorAL0,
        tensorBL0, tensorL0C, gmPackedY, blockColumn, loadA);
    SddmmComplexBsrStoreFourB16(tensorL0C, gmScratch, blockSlot);
}

template <int32_t BlockDim, int32_t Reduction>
class SddmmComplexBsrCubeCompute {
    using Tile = sddmm_arch35::SddmmComplexBsrCubeTileConfig<BlockDim, Reduction>;
    static constexpr int32_t kExpandedK = Tile::kExpandedK;
    static constexpr int32_t kExpandedRows = Tile::kExpandedRows;
    // Keep the original K=128 tiles.  With K=64 the complete expanded
    // reduction is one tile, allowing A to stay resident across all sparse
    // blocks assigned from the same row. Host checks the required L0 capacity.
    static constexpr int32_t kKtile = Tile::kKtile;
    static constexpr bool kReuseFullAL0 =
        kKtile == kExpandedK;

public:
    __aicore__ inline SddmmComplexBsrCubeCompute(
        GM_ADDR sparseOffsets, GM_ADDR sparseColumns, GM_ADDR packedX,
        GM_ADDR packedY, GM_ADDR scratch, const SddmmTilingData &tiling)
        : offsets_(reinterpret_cast<__gm__ const int32_t *>(sparseOffsets)),
          columns_(reinterpret_cast<__gm__ const int32_t *>(sparseColumns)),
          packedX_(packedX), packedY_(packedY), scratch_(scratch), tiling_(tiling)
    {}

    __aicore__ inline void Process()
    {
        const int32_t blockId = static_cast<int32_t>(GetBlockIdx());
        const int32_t blockCount = static_cast<int32_t>(GetBlockNum());
        // Divide physical blocks, not rows, so a single long BSR row can use
        // all Cube cores.  Quotient/remainder partitioning avoids multiplying
        // nnzUnits by the grid size and assigns each physical block exactly once.
        const int64_t blocksPerCore = tiling_.nnzUnits / blockCount;
        const int64_t remainingBlocks = tiling_.nnzUnits % blockCount;
        ownedBegin_ = blockId * blocksPerCore +
            (blockId < remainingBlocks ? blockId : remainingBlocks);
        ownedEnd_ = ownedBegin_ + blocksPerCore +
            (blockId < remainingBlocks ? 1 : 0);
        if (ownedBegin_ == ownedEnd_) {
            return;
        }
        auto gmPackedX = MakeGmPackedX();
        auto gmPackedY = MakeGmPackedY();
        auto gmScratch = MakeGmScratch();
        auto tensorAL1 = MakeTensorAL1();
        auto tensorBL1 = MakeTensorBL1();
        SetFlag<HardEvent::MTE1_MTE2>(0);
        SetFlag<HardEvent::MTE1_MTE2>(1);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::FIX_M>(0);
        SetMMLayoutTransform(true);

        ProcessRows(gmPackedX, gmPackedY, gmScratch, tensorAL1, tensorBL1);

        WaitFlag<HardEvent::MTE1_MTE2>(0);
        WaitFlag<HardEvent::MTE1_MTE2>(1);
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::FIX_M>(0);
        SetMMLayoutTransform(false);
    }

private:
    __aicore__ inline auto MakeGmPackedX()
    {
        return Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(
                reinterpret_cast<__gm__ float *>(packedX_)),
            Te::MakeFrameLayout<Te::NDExtLayoutPtn,
                Te::LayoutTraitDefault<float>>(
                    tiling_.blockRows * kExpandedRows, kExpandedK));
    }

    __aicore__ inline auto MakeGmPackedY()
    {
        return Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(
                reinterpret_cast<__gm__ float *>(packedY_)),
            Te::MakeFrameLayout<Te::NDExtLayoutPtn,
                Te::LayoutTraitDefault<float>>(
                    kExpandedK, tiling_.n));
    }

    __aicore__ inline auto MakeGmScratch()
    {
        return Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(
                reinterpret_cast<__gm__ float *>(scratch_)),
            Te::MakeFrameLayout<Te::NDExtLayoutPtn,
                ComplexBsrL0cLayoutTrait>(tiling_.nnzUnits * kExpandedRows, BlockDim));
    }

    __aicore__ inline auto MakeTensorAL1()
    {
        return Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::L1, float>(0),
            Te::MakeFrameLayout<Te::NZLayoutPtn,
                Te::LayoutTraitDefault<float>>(
                    kExpandedRows, kExpandedK));
    }

    __aicore__ inline auto MakeTensorBL1()
    {
        return Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::L1, float>(
                static_cast<int64_t>(kExpandedRows) * kExpandedK * sizeof(float)),
            Te::MakeFrameLayout<Te::ZNLayoutPtn,
                Te::LayoutTraitDefault<float>>(
                    kExpandedK, BlockDim));
    }


    __aicore__ inline int64_t FindFirstRow()
    {
        int64_t firstRow = 0;
        int64_t lastRow = tiling_.blockRows;
        while (firstRow < lastRow) {
            const int64_t middle = firstRow + (lastRow - firstRow) / 2;
            if (static_cast<int64_t>(offsets_[middle + 1]) -
                    tiling_.indexBase <= ownedBegin_) {
                firstRow = middle + 1;
            } else {
                lastRow = middle;
            }
        }
        return firstRow;
    }

    template <typename TensorAL1, typename TensorBL1, typename TensorL0C>
    __aicore__ inline void ComputeReductionTile(
        TensorAL1 &tensorAL1, TensorBL1 &tensorBL1, TensorL0C &tensorL0C,
        int32_t kOffset, bool loadA, bool initializeC)
    {
        WaitFlag<HardEvent::M_MTE1>(0);
        auto tensorAL0 = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::L0A, float>(0),
            Te::MakeFrameLayout<Te::NZLayoutPtn,
                Te::LayoutTraitDefault<float>>(
                    kExpandedRows, kKtile));
        auto tensorBL0 = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::L0B, float>(0),
            Te::MakeFrameLayout<Te::ZNLayoutPtn,
                Te::LayoutTraitDefault<float>>(
                    kKtile, BlockDim));
        auto tensorAL1Tile = tensorAL1.Slice(
            Te::MakeCoord(0, kOffset),
            Te::MakeShape(kExpandedRows, kKtile));
        auto tensorBL1Tile = tensorBL1.Slice(
            Te::MakeCoord(kOffset, 0),
            Te::MakeShape(kKtile, BlockDim));
        if constexpr (!kReuseFullAL0) {
            Te::Copy(Te::MakeCopy(Te::CopyL12L0A{}),
                tensorAL0, tensorAL1Tile);
        } else if (loadA) {
            Te::Copy(Te::MakeCopy(Te::CopyL12L0A{}),
                tensorAL0, tensorAL1Tile);
        }
        Te::Copy(Te::MakeCopy(Te::CopyL12L0B{}),
            tensorBL0, tensorBL1Tile);
        const bool finalK =
            kOffset + kKtile == kExpandedK;
        if (finalK) {
            // B-L1 may be overwritten only after its final L0 copy.
            SetFlag<HardEvent::MTE1_MTE2>(1);
        }
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);

        Te::MmadParams params{
            static_cast<uint16_t>(kExpandedRows),
            static_cast<uint16_t>(BlockDim),
            static_cast<uint16_t>(kKtile),
            finalK ? kComplexBsrFinalAccumulation :
                kComplexBsrNonFinalAccumulation,
            initializeC};
        auto kMmad = Te::MakeMmad(
            Te::MmadOperation{}, Te::MmadTraitDefault{});
        Te::Mmad(kMmad.with(params), tensorL0C,
            tensorAL0, tensorBL0);
        SetFlag<HardEvent::M_MTE1>(0);
    }

    template <typename GmPackedY, typename GmScratch,
        typename TensorAL1, typename TensorBL1>
    __aicore__ inline void ComputeBlock(
        GmPackedY &gmPackedY, GmScratch &gmScratch,
        TensorAL1 &tensorAL1, TensorBL1 &tensorBL1,
        int64_t blockSlot, int64_t blockColumn, bool loadA)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(1);
        auto gmYBlock = gmPackedY.Slice(
            Te::MakeCoord(0, blockColumn * BlockDim),
            Te::MakeShape(kExpandedK, BlockDim));
        Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}), tensorBL1, gmYBlock);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);

        WaitFlag<HardEvent::FIX_M>(0);
        auto tensorL0C = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::L0C, float>(0),
            Te::MakeFrameLayout<Te::NZLayoutPtn,
                ComplexBsrL0cLayoutTrait>(kExpandedRows, BlockDim));
        bool initializeC = true;
        for (int32_t kOffset = 0;
             kOffset < kExpandedK;
             kOffset += kKtile) {
            ComputeReductionTile(tensorAL1, tensorBL1, tensorL0C,
                kOffset, loadA, initializeC);
            initializeC = false;
        }

        auto gmScratchTile = gmScratch.Slice(
            Te::MakeCoord(blockSlot * kExpandedRows, 0),
            Te::MakeShape(kExpandedRows, BlockDim));
        Te::FixpipeParams fixpipeParams;
        fixpipeParams.unitFlag = kComplexBsrFinalAccumulation;
        auto kFixpipe =
            Te::MakeCopy(Te::CopyL0C2GM{});
        Te::Copy(kFixpipe.with(fixpipeParams),
            gmScratchTile, tensorL0C);
        SetFlag<HardEvent::FIX_M>(0);
    }

    template <typename GmPackedY, typename GmScratch,
        typename TensorAL1, typename TensorBL1>
    __aicore__ inline void ProcessBlocks(
        GmPackedY &gmPackedY, GmScratch &gmScratch,
        TensorAL1 &tensorAL1, TensorBL1 &tensorBL1,
        int64_t rowStart, int64_t rowEnd)
    {
        for (int64_t blockSlot = rowStart; blockSlot < rowEnd;) {
            const int64_t blockColumn =
                static_cast<int64_t>(columns_[blockSlot]) - tiling_.indexBase;
            if constexpr (BlockDim == 16 && Reduction == 128) {
                if (rowEnd - blockSlot >= 4 &&
                    static_cast<int64_t>(columns_[blockSlot + 1]) - tiling_.indexBase == blockColumn + 1 &&
                    static_cast<int64_t>(columns_[blockSlot + 2]) - tiling_.indexBase == blockColumn + 2 &&
                    static_cast<int64_t>(columns_[blockSlot + 3]) - tiling_.indexBase == blockColumn + 3) {
                    SddmmComplexBsrCubeFourB16(packedY_, scratch_, tiling_,
                        blockSlot, blockColumn, blockSlot == rowStart);
                    blockSlot += 4;
                    continue;
                }
            }
            ComputeBlock(gmPackedY, gmScratch, tensorAL1, tensorBL1,
                blockSlot, blockColumn, blockSlot == rowStart);
            ++blockSlot;
        }
    }

    template <typename GmPackedX, typename GmPackedY, typename GmScratch,
        typename TensorAL1, typename TensorBL1>
    __aicore__ inline void ProcessRows(
        GmPackedX &gmPackedX, GmPackedY &gmPackedY, GmScratch &gmScratch,
        TensorAL1 &tensorAL1, TensorBL1 &tensorBL1)
    {
        for (int64_t blockRow = FindFirstRow(); blockRow < tiling_.blockRows;
             ++blockRow) {
            const int64_t physicalRowStart =
                static_cast<int64_t>(offsets_[blockRow]) - tiling_.indexBase;
            const int64_t physicalRowEnd =
                static_cast<int64_t>(offsets_[blockRow + 1]) - tiling_.indexBase;
            if (physicalRowStart >= ownedEnd_) {
                break;
            }
            const int64_t rowStart = physicalRowStart > ownedBegin_ ?
                physicalRowStart : ownedBegin_;
            const int64_t rowEnd = physicalRowEnd < ownedEnd_ ?
                physicalRowEnd : ownedEnd_;
            if (rowStart >= rowEnd) {
                continue;
            }

            WaitFlag<HardEvent::MTE1_MTE2>(0);
            auto gmXBlock = gmPackedX.Slice(
                Te::MakeCoord(blockRow * kExpandedRows, 0),
                Te::MakeShape(kExpandedRows, kExpandedK));
            Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}), tensorAL1, gmXBlock);
            SetFlag<HardEvent::MTE2_MTE1>(0);
            WaitFlag<HardEvent::MTE2_MTE1>(0);

            ProcessBlocks(gmPackedY, gmScratch, tensorAL1, tensorBL1, rowStart, rowEnd);
            // The full-tile path copies A once per assigned row segment; the
            // sliced path keeps A-L1 live through its last A-L0 copy.  This event
            // releases A-L1 only after those copies finish.
            SetFlag<HardEvent::MTE1_MTE2>(0);
        }
    }

    __gm__ const int32_t *offsets_;
    __gm__ const int32_t *columns_;
    GM_ADDR packedX_;
    GM_ADDR packedY_;
    GM_ADDR scratch_;
    const SddmmTilingData &tiling_;
    int64_t ownedBegin_ = 0;
    int64_t ownedEnd_ = 0;
};

template <int32_t BlockDim, int32_t Reduction>
__aicore__ inline void SddmmComplexBsrCubeImpl(
    GM_ADDR sparseOffsets,
    GM_ADDR sparseColumns,
    GM_ADDR packedX,
    GM_ADDR packedY,
    GM_ADDR scratch,
    const SddmmTilingData &tiling)
{
    SddmmComplexBsrCubeCompute<BlockDim, Reduction> kernel(
        sparseOffsets, sparseColumns, packedX, packedY, scratch, tiling);
    kernel.Process();
}

struct SddmmComplexBsrValue {
    float real;
    float imag;
};

// Reorder row-major real/imaginary planes into physical COL complex values.
// B32 payloads preserve every bit; this path performs no floating-point
// arithmetic and uses the copy epilogue's existing UB buffers.
__simd_vf__ inline void SddmmComplexBsrColumn16InterleaveVf(
    __ubuf__ uint32_t *output, __ubuf__ uint32_t *real,
    __ubuf__ uint32_t *imag, uint16_t blockCount)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t kBlockElements = 16U * 16U;
    constexpr uint32_t kVectorElements = 64U;
    constexpr uint16_t kVectorsPerBlock = kBlockElements / kVectorElements;
    MaskReg mask = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<int32_t> lanes;
    RegTensor<int32_t> rowMask;
    RegTensor<int32_t> rows;
    RegTensor<int32_t> columns;
    RegTensor<int32_t> firstIndices;
    RegTensor<int32_t> sourceIndices;
    RegTensor<uint32_t> realValues;
    RegTensor<uint32_t> imagValues;
    RegTensor<uint32_t> low;
    RegTensor<uint32_t> high;
    Arange(lanes, int32_t{0});
    Duplicate(rowMask, int32_t{15}, mask);
    And(rows, lanes, rowMask, mask);
    ShiftLefts(rows, rows, int16_t{4}, mask);
    ShiftRights(columns, lanes, int16_t{4}, mask);
    Add(firstIndices, rows, columns, mask);
    for (uint16_t block = 0; block < blockCount; ++block) {
        const uint32_t planeOffset = block * kBlockElements;
        for (uint16_t vector = 0; vector < kVectorsPerBlock; ++vector) {
            Adds(sourceIndices, firstIndices, static_cast<int32_t>(vector * 4U), mask);
            Gather(realValues, real + planeOffset,
                (RegTensor<uint32_t> &)sourceIndices, mask);
            Gather(imagValues, imag + planeOffset,
                (RegTensor<uint32_t> &)sourceIndices, mask);
            AscendC::MicroAPI::Interleave(low, high, realValues, imagValues);
            const uint32_t outputOffset =
                planeOffset * 2U + vector * kVectorElements * 2U;
            StoreAlign(output + outputOffset, low, mask);
            StoreAlign(output + outputOffset + kVectorElements, high, mask);
        }
    }
}

class SddmmComplexBsrCopyEpilogueKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR sparseValues,
        GM_ADDR scratch,
        const SddmmTilingData *tiling,
        AscendC::TPipe *pipe)
    {
        tiling_ = *tiling;
        pipe_ = pipe;
        valueParts_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(sparseValues),
            static_cast<uint64_t>(tiling_.outputsPerBatch) * 2ULL);
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch),
            static_cast<uint64_t>(tiling_.outputsPerBatch) * 2ULL);
        pipe_->InitBuffer(realBuffer_,
            kComplexBsrEpilogueTileComponents * sizeof(float));
        pipe_->InitBuffer(imagBuffer_,
            kComplexBsrEpilogueTileComponents * sizeof(float));
        pipe_->InitBuffer(outputBuffer_,
            kComplexBsrEpilogueTileComponents * 2 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int32_t blockArea = tiling_.blockDim * tiling_.blockDim;
        const int32_t blocksPerTile =
            kComplexBsrEpilogueTileComponents / blockArea;
        const int64_t tileCount =
            (tiling_.nnzUnits + blocksPerTile - 1) / blocksPerTile;
        const int64_t core = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t coreCount =
            static_cast<int64_t>(AscendC::GetBlockNum());
        for (int64_t tile = core; tile < tileCount; tile += coreCount) {
            const int64_t blockStart = tile * blocksPerTile;
            const int32_t currentBlocks = static_cast<int32_t>(
                tiling_.nnzUnits - blockStart < blocksPerTile ?
                    tiling_.nnzUnits - blockStart : blocksPerTile);
            CopyTile(blockStart, currentBlocks, blockArea);
        }
    }

private:
    __aicore__ inline void CopyTile(
        int64_t blockStart, int32_t blockCount, int32_t blockArea)
    {
        const int32_t componentCount = blockCount * blockArea;
        const int64_t scratchOffset =
            blockStart * static_cast<int64_t>(blockArea) * 2;
        AscendC::LocalTensor<float> real = realBuffer_.Get<float>();
        AscendC::LocalTensor<float> imag = imagBuffer_.Get<float>();
        AscendC::LocalTensor<float> output = outputBuffer_.Get<float>();
        const uint32_t planeBytes = blockArea * sizeof(float);
        AscendC::DataCopyExtParams planeCopy{
            static_cast<uint16_t>(blockCount), planeBytes,
            planeBytes, 0, 0};
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0F};
        AscendC::DataCopyPad(
            real, scratch_[scratchOffset], planeCopy, noPad);
        AscendC::DataCopyPad(
            imag, scratch_[scratchOffset + blockArea], planeCopy, noPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        if (tiling_.blockOrder == 1 && tiling_.blockDim == 16) {
            SddmmComplexBsrColumn16InterleaveVf(
                reinterpret_cast<__ubuf__ uint32_t *>(output.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(real.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(imag.GetPhyAddr()),
                static_cast<uint16_t>(blockCount));
        } else {
            AscendC::Interleave(output, output[componentCount],
                real, imag, componentCount);
        }
        // The next tile's MTE2 loads may overwrite these input planes
        // only after this vector operation has finished reading them.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams outputCopy{
            1, static_cast<uint32_t>(componentCount * 2 * sizeof(float)),
            0, 0, 0};
        AscendC::DataCopyPad(
            valueParts_[scratchOffset], output, outputCopy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    AscendC::TPipe *pipe_{nullptr};
    AscendC::TBuf<AscendC::TPosition::VECCALC> realBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> imagBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuffer_;
    AscendC::GlobalTensor<float> valueParts_;
    AscendC::GlobalTensor<float> scratch_;
    SddmmTilingData tiling_{};
};

static __simt_callee__ __aicore__ inline SddmmComplexBsrValue
SddmmComplexBsrMultiply(
    const SddmmComplexBsrValue &lhs,
    const SddmmComplexBsrValue &rhs)
{
    return SddmmComplexBsrValue{
        __fmaf_rn(lhs.real, rhs.real, -lhs.imag * rhs.imag),
        __fmaf_rn(lhs.real, rhs.imag, lhs.imag * rhs.real)};
}

__simt_vf__ __aicore__ __launch_bounds__(kComplexBsrVectorThreads) inline void
SddmmComplexBsrEpilogueSimt(
    __gm__ float *valueParts,
    __gm__ const float *scratch,
    SddmmTilingData tiling,
    int32_t threadsPerCore,
    int32_t coreCount)
{
    const uint64_t globalThread =
        static_cast<uint64_t>(blockIdx.x) * threadsPerCore + threadIdx.x;
    const uint64_t stride =
        static_cast<uint64_t>(coreCount) * threadsPerCore;
    const uint64_t blockArea =
        static_cast<uint64_t>(tiling.blockDim) * tiling.blockDim;
    const uint64_t total = static_cast<uint64_t>(tiling.outputsPerBatch);
    const SddmmComplexBsrValue alpha{
        tiling.alphaReal, tiling.alphaImag};
    const SddmmComplexBsrValue beta{
        tiling.betaReal, tiling.betaImag};
    const bool alphaIsZero = alpha.real == 0.0F && alpha.imag == 0.0F;
    const bool alphaIsOne = alpha.real == 1.0F && alpha.imag == 0.0F;
    const bool betaIsZero = beta.real == 0.0F && beta.imag == 0.0F;
    const bool betaIsOne = beta.real == 1.0F && beta.imag == 0.0F;

    for (uint64_t output = globalThread; output < total; output += stride) {
        const uint64_t blockSlot = output / blockArea;
        const uint64_t inner = output - blockSlot * blockArea;
        // Cube scratch always stores row-major real/imaginary planes.
        // Values retain the descriptor's physical order, including beta*C.
        const uint64_t scratchInner = tiling.blockOrder == 0 ? inner :
            (inner % tiling.blockDim) * tiling.blockDim +
                inner / tiling.blockDim;
        const uint64_t scratchBase = blockSlot * blockArea * 2ULL;
        SddmmComplexBsrValue result{0.0F, 0.0F};
        if (!alphaIsZero) {
            const SddmmComplexBsrValue dot{
                scratch[scratchBase + scratchInner],
                scratch[scratchBase + blockArea + scratchInner]};
            result = alphaIsOne ? dot :
                SddmmComplexBsrMultiply(alpha, dot);
        }
        const uint64_t part = output * 2ULL;
        if (!betaIsZero) {
            const SddmmComplexBsrValue oldValue{
                valueParts[part], valueParts[part + 1ULL]};
            // Preserve components instead of introducing 0*Inf cross terms.
            const SddmmComplexBsrValue scaledOld = betaIsOne ? oldValue :
                SddmmComplexBsrMultiply(beta, oldValue);
            result.real += scaledOld.real;
            result.imag += scaledOld.imag;
        }
        valueParts[part] = result.real;
        valueParts[part + 1ULL] = result.imag;
    }
}

template <int32_t BlockDim>
__aicore__ inline void SddmmComplexBsrCubeDispatch(
    GM_ADDR sparseOffsets,
    GM_ADDR sparseColumns,
    GM_ADDR packedX,
    GM_ADDR packedY,
    GM_ADDR scratch,
    const SddmmTilingData &tiling)
{
    if (tiling.k == sddmm_arch35::kSddmmComplexBsrCubeSmallReduction) {
        SddmmComplexBsrCubeImpl<BlockDim, 64>(sparseOffsets, sparseColumns,
            packedX, packedY, scratch, tiling);
    } else {
        SddmmComplexBsrCubeImpl<BlockDim, 128>(sparseOffsets, sparseColumns,
            packedX, packedY, scratch, tiling);
    }
}

} // namespace

extern "C" __global__ __aicore__ void sddmm_complex_bsr_pack_kernel(
    GM_ADDR sparseOffsets,
    GM_ADDR sparseColumns,
    GM_ADDR matX,
    GM_ADDR matY,
    GM_ADDR packedX,
    GM_ADDR packedY,
    const SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    if (tiling.k == sddmm_arch35::kSddmmComplexBsrCubeSmallReduction) {
        SddmmComplexBsrPackKernel<64> packer;
        packer.Init(sparseOffsets, sparseColumns, matX, matY, packedX, packedY, &tiling, &pipe);
        packer.Process();
    } else {
        SddmmComplexBsrPackKernel<128> packer;
        packer.Init(sparseOffsets, sparseColumns, matX, matY, packedX, packedY, &tiling, &pipe);
        packer.Process();
    }
}

extern "C" __global__ __aicore__ void sddmm_complex_bsr_cube_kernel(
    GM_ADDR sparseOffsets,
    GM_ADDR sparseColumns,
    GM_ADDR packedX,
    GM_ADDR packedY,
    GM_ADDR scratch,
    const SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (tiling.blockDim == 16) {
        SddmmComplexBsrCubeDispatch<16>(sparseOffsets, sparseColumns,
            packedX, packedY, scratch, tiling);
    } else if (tiling.blockDim == 32) {
        SddmmComplexBsrCubeDispatch<32>(sparseOffsets, sparseColumns,
            packedX, packedY, scratch, tiling);
    } else if (tiling.blockDim == 64) {
        SddmmComplexBsrCubeDispatch<64>(sparseOffsets, sparseColumns,
            packedX, packedY, scratch, tiling);
    }
}

extern "C" __global__ __aicore__ void sddmm_complex_bsr_epilogue_kernel(
    GM_ADDR sparseValues,
    GM_ADDR scratch,
    const SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ((tiling.blockOrder == 0 ||
            (tiling.blockOrder == 1 && tiling.blockDim == 16)) &&
        tiling.alphaReal == 1.0F && tiling.alphaImag == 0.0F &&
        tiling.betaReal == 0.0F && tiling.betaImag == 0.0F) {
        AscendC::TPipe pipe;
        SddmmComplexBsrCopyEpilogueKernel epilogue;
        epilogue.Init(sparseValues, scratch, &tiling, &pipe);
        epilogue.Process();
        return;
    }
    const int32_t coreCount = static_cast<int32_t>(GetBlockNum());
    asc_vf_call<SddmmComplexBsrEpilogueSimt>(
        dim3{kComplexBsrVectorThreads},
        reinterpret_cast<__gm__ float *>(sparseValues),
        reinterpret_cast<__gm__ const float *>(scratch),
        tiling, static_cast<int32_t>(kComplexBsrVectorThreads), coreCount);
}

extern "C" void sddmm_complex_bsr_cube_pack_kernel_do(
    GM_ADDR sparseOffsets,
    GM_ADDR sparseColumns,
    GM_ADDR matX,
    GM_ADDR matY,
    GM_ADDR packedX,
    GM_ADDR packedY,
    const SddmmTilingData &tiling,
    uint32_t vectorBlockDim,
    void *stream)
{
    if (vectorBlockDim == 0U ||
        !sddmm_arch35::SddmmComplexBsrCubeCanRun(tiling)) {
        return;
    }
    sddmm_complex_bsr_pack_kernel<<<vectorBlockDim, nullptr, stream>>>(
        sparseOffsets, sparseColumns, matX, matY, packedX, packedY, tiling);
}

extern "C" void sddmm_complex_bsr_cube_compute_kernel_do(
    GM_ADDR sparseOffsets,
    GM_ADDR sparseColumns,
    GM_ADDR packedX,
    GM_ADDR packedY,
    GM_ADDR scratch,
    const SddmmTilingData &tiling,
    uint32_t cubeBlockDim,
    void *stream)
{
    if (cubeBlockDim == 0U ||
        !sddmm_arch35::SddmmComplexBsrCubeCanRun(tiling)) {
        return;
    }
    sddmm_complex_bsr_cube_kernel<<<cubeBlockDim, nullptr, stream>>>(
        sparseOffsets, sparseColumns, packedX, packedY, scratch, tiling);
}

extern "C" void sddmm_complex_bsr_cube_epilogue_kernel_do(
    GM_ADDR sparseValues,
    GM_ADDR scratch,
    const SddmmTilingData &tiling,
    uint32_t vectorBlockDim,
    void *stream)
{
    if (vectorBlockDim == 0U ||
        !sddmm_arch35::SddmmComplexBsrCubeCanRun(tiling)) {
        return;
    }
    sddmm_complex_bsr_epilogue_kernel<<<vectorBlockDim, nullptr, stream>>>(
        sparseValues, scratch, tiling);
}

extern "C" void sddmm_complex_bsr_cube_kernel_do(
    GM_ADDR sparseOffsets,
    GM_ADDR sparseColumns,
    GM_ADDR sparseValues,
    GM_ADDR matX,
    GM_ADDR matY,
    GM_ADDR workspace,
    const SddmmTilingData &tiling,
    uint32_t cubeBlockDim,
    uint32_t vectorBlockDim,
    void *stream)
{
    sddmm_arch35::SddmmComplexBsrCubeWorkspaceLayout layout{};
    if (cubeBlockDim == 0U || vectorBlockDim == 0U || workspace == nullptr ||
        !sddmm_arch35::SddmmComplexBsrCubeCanRun(tiling) ||
        !sddmm_arch35::SddmmComplexBsrCubeMakeWorkspaceLayout(
            tiling, layout)) {
        return;
    }
    GM_ADDR packedX = workspace + layout.packedXOffset;
    GM_ADDR packedY = workspace + layout.packedYOffset;
    GM_ADDR scratch = workspace + layout.scratchOffset;
    sddmm_complex_bsr_cube_pack_kernel_do(
        sparseOffsets, sparseColumns, matX, matY, packedX, packedY,
        tiling, vectorBlockDim, stream);
    sddmm_complex_bsr_cube_compute_kernel_do(
        sparseOffsets, sparseColumns, packedX, packedY, scratch,
        tiling, cubeBlockDim, stream);
    sddmm_complex_bsr_cube_epilogue_kernel_do(
        sparseValues, scratch, tiling, vectorBlockDim, stream);
}
