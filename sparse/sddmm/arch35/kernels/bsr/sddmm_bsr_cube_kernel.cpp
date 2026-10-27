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
 * \file sddmm_bsr_cube_kernel.cpp
 * \brief Ascend 950 BSR SDDMM Cube fast path for b=16/32/64 and K=32/64/128.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "tensor_api/tensor.h"

#include "../../sddmm_bsr_cube_common.h"
#include "../shared/sddmm_simt_convert.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
#error "SDDMM BSR Cube kernel is only available on DAV-3510 / Ascend 950."
#endif

using namespace AscendC;

namespace {

constexpr int32_t kBsrCubeL0cC0 = 16;
constexpr uint8_t kBsrCubeFinalAccumulation = 3;
constexpr uint32_t kBsrCubeVectorThreads = 256U;

using BsrCubeL0cLayoutTrait =
    AscendC::Te::LayoutTrait<float, AscendC::Std::Int<kBsrCubeL0cC0>>;

// GM/L1 input layouts use the dtype-derived C0. In particular, the FP32
// specialization must use C0=8; the FP16/BF16 specializations use C0=16.
template <typename T>
struct SddmmBsrCubeInputC0 {
    static constexpr int32_t value = 16;
};

template <>
struct SddmmBsrCubeInputC0<float> {
    static constexpr int32_t value = 8;
};

struct SddmmBsrCubeInputs {
    GM_ADDR sparseOffsets;
    GM_ADDR sparseColumns;
    GM_ADDR matX;
    GM_ADDR matY;
    const SddmmTilingData &tiling;
};

struct SddmmBsrCubeRowCache {
    int64_t batch = -1;
    int64_t blockRow = 0;
    int64_t rowEnd = 0;
};

template <typename AL1, typename BL1, typename AL0, typename BL0, typename Scratch>
struct SddmmBsrCubeTiles {
    AL1 &aL1;
    BL1 &bL1;
    AL0 &aL0;
    BL0 &bL0;
    Scratch &scratch;
};

__aicore__ inline bool SddmmBsrCubeUpdateRow(
    const SddmmBsrCubeInputs &inputs, int64_t batch, int64_t blockSlot,
    SddmmBsrCubeRowCache &cache)
{
    const bool loadA = batch != cache.batch || blockSlot >= cache.rowEnd;
    if (!loadA) {
        return false;
    }
    const auto &tiling = inputs.tiling;
    const int64_t offsetsBase = tiling.offsetsBatchStride == 0 ?
        0 : batch * tiling.offsetsBatchStride;
    __gm__ const int32_t *offsets =
        reinterpret_cast<__gm__ const int32_t *>(inputs.sparseOffsets);
    int64_t lower = 0;
    int64_t upper = tiling.blockRows;
    while (lower < upper) {
        const int64_t middle = lower + (upper - lower) / 2;
        const int64_t rowEnd =
            static_cast<int64_t>(offsets[offsetsBase + middle + 1]) - tiling.indexBase;
        if (rowEnd <= blockSlot) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    cache.batch = batch;
    cache.blockRow = lower;
    cache.rowEnd = static_cast<int64_t>(
        offsets[offsetsBase + cache.blockRow + 1]) - tiling.indexBase;
    return true;
}

template <typename T, int32_t BlockDim, int32_t ReductionK, typename Tiles>
__aicore__ inline void SddmmBsrCubeLoadTiles(
    const SddmmBsrCubeInputs &inputs, const SddmmBsrCubeRowCache &cache,
    int64_t blockSlot, bool loadA, Tiles &tiles)
{
    const auto &tiling = inputs.tiling;
    const int64_t batchX = tiling.batchCountX == 1 ? 0 : cache.batch;
    const int64_t batchY = tiling.batchCountY == 1 ? 0 : cache.batch;
    __gm__ T *xBatch = reinterpret_cast<__gm__ T *>(inputs.matX) + batchX * tiling.xBatchStride;
    __gm__ T *yBatch = reinterpret_cast<__gm__ T *>(inputs.matY) + batchY * tiling.yBatchStride;
    auto gmX = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(xBatch),
        Te::MakeFrameLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>(tiling.m, ReductionK));
    auto gmY = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(yBatch),
        Te::MakeFrameLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>(ReductionK, tiling.n));
    if (loadA) {
        WaitFlag<HardEvent::MTE1_MTE2>(0);
        auto gmXBlock = gmX.Slice(
            Te::MakeCoord(cache.blockRow * BlockDim, 0),
            Te::MakeShape(BlockDim, ReductionK));
        Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}), tiles.aL1, gmXBlock);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
    }
    const int64_t columnsBase = tiling.columnsBatchStride == 0 ?
        0 : cache.batch * tiling.columnsBatchStride;
    __gm__ const int32_t *columns =
        reinterpret_cast<__gm__ const int32_t *>(inputs.sparseColumns);
    const int64_t blockColumn =
        static_cast<int64_t>(columns[columnsBase + blockSlot]) - tiling.indexBase;
    WaitFlag<HardEvent::MTE1_MTE2>(1);
    auto gmYBlock = gmY.Slice(
        Te::MakeCoord(0, blockColumn * BlockDim),
        Te::MakeShape(ReductionK, BlockDim));
    Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}), tiles.bL1, gmYBlock);
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
}

template <int32_t BlockDim, int32_t ReductionK, typename Tiles>
__aicore__ inline void SddmmBsrCubeMultiplyTile(
    int64_t globalBlockSlot, bool loadA, Tiles &tiles)
{
    WaitFlag<HardEvent::M_MTE1>(0);
    if (loadA) {
        // Every supported specialization already stores the complete A tile
        // in L0A. MMAD and the B copies leave it available for row reuse.
        Te::Copy(Te::MakeCopy(Te::CopyL12L0A{}), tiles.aL0, tiles.aL1);
    }
    Te::Copy(Te::MakeCopy(Te::CopyL12L0B{}), tiles.bL0, tiles.bL1);
    SetFlag<HardEvent::MTE1_MTE2>(1);
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::FIX_M>(0);
    auto tensorL0C = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::L0C, float>(0),
        Te::MakeFrameLayout<Te::NZLayoutPtn,
            BsrCubeL0cLayoutTrait>(BlockDim, BlockDim));
    Te::MmadParams params{
        static_cast<uint16_t>(BlockDim), static_cast<uint16_t>(BlockDim),
        static_cast<uint16_t>(ReductionK), kBsrCubeFinalAccumulation, true};
    auto kMmad = Te::MakeMmad(Te::MmadOperation{}, Te::MmadTraitDefault{});
    Te::Mmad(kMmad.with(params), tensorL0C, tiles.aL0, tiles.bL0);
    SetFlag<HardEvent::M_MTE1>(0);
    auto gmScratchBlock = tiles.scratch.Slice(
        Te::MakeCoord(globalBlockSlot * BlockDim, 0),
        Te::MakeShape(BlockDim, BlockDim));
    Te::FixpipeParams fixpipeParams;
    fixpipeParams.unitFlag = kBsrCubeFinalAccumulation;
    auto copyL0C2GM = Te::MakeCopy(Te::CopyL0C2GM{});
    Te::Copy(copyL0C2GM.with(fixpipeParams), gmScratchBlock, tensorL0C);
    SetFlag<HardEvent::FIX_M>(0);
}

template <typename T, int32_t BlockDim, int32_t ReductionK, typename Tiles>
__aicore__ inline void SddmmBsrCubeRun(const SddmmBsrCubeInputs &inputs, Tiles &tiles)
{
    const auto &tiling = inputs.tiling;
    const int32_t blockId = static_cast<int32_t>(GetBlockIdx());
    const int32_t blockCount = static_cast<int32_t>(GetBlockNum());
    const int64_t totalBlocks = tiling.batchCount * tiling.nnzUnits;
    // The workspace limit bounds totalBlocks to at most 65,536 (b=16),
    // so these products remain within int64_t for every valid core count.
    const int64_t firstBlock = totalBlocks * blockId / blockCount;
    const int64_t lastBlock = totalBlocks * (blockId + 1) / blockCount;
    // Slot 0 guards the reused A-L1 tile, slot 1 guards B-L1, M_MTE1 guards
    // L0A/L0B reuse, and FIX_M guards L0C until its previous tile is written.
    SetFlag<HardEvent::MTE1_MTE2>(0);
    SetFlag<HardEvent::MTE1_MTE2>(1);
    SetFlag<HardEvent::M_MTE1>(0);
    SetFlag<HardEvent::FIX_M>(0);
    SetMMLayoutTransform(true);
    // Contiguous physical sparse blocks balance arbitrary row lengths.
    // UpdateRow skips empty rows and searches only at row/batch boundaries.
    SddmmBsrCubeRowCache cache;
    for (int64_t globalBlockSlot = firstBlock; globalBlockSlot < lastBlock;
         ++globalBlockSlot) {
        const int64_t batch = globalBlockSlot / tiling.nnzUnits;
        const int64_t blockSlot = globalBlockSlot - batch * tiling.nnzUnits;
        const bool loadA = SddmmBsrCubeUpdateRow(inputs, batch, blockSlot, cache);
        SddmmBsrCubeLoadTiles<T, BlockDim, ReductionK>(inputs, cache, blockSlot, loadA, tiles);
        SddmmBsrCubeMultiplyTile<BlockDim, ReductionK>(globalBlockSlot, loadA, tiles);
        // Release A-L1 at a row boundary. M_MTE1 protects resident A-L0
        // until the previous row's MMAD is complete.
        if (blockSlot + 1 == cache.rowEnd || globalBlockSlot + 1 == lastBlock) {
            SetFlag<HardEvent::MTE1_MTE2>(0);
        }
    }
    WaitFlag<HardEvent::MTE1_MTE2>(0);
    WaitFlag<HardEvent::MTE1_MTE2>(1);
    WaitFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::FIX_M>(0);
    SetMMLayoutTransform(false);
}

template <typename T, int32_t BlockDim, int32_t ReductionK>
__aicore__ inline void SddmmBsrCubeImpl(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns,
    GM_ADDR matX, GM_ADDR matY, GM_ADDR scratch,
    const SddmmTilingData &tiling)
{
    static_assert(BlockDim == 16 || BlockDim == 32 || BlockDim == 64,
        "unsupported BSR Cube block size");
    static_assert(ReductionK == 32 || ReductionK == 64 || ReductionK == 128,
        "unsupported BSR Cube reduction size");
    static_assert(SddmmBsrCubeInputC0<T>::value == (32 / sizeof(T)),
        "input C0 must match the input dtype");
    static_assert(ReductionK % SddmmBsrCubeInputC0<T>::value == 0,
        "reduction size must be aligned to the input C0");
    const int64_t scratchRows = tiling.batchCount * tiling.nnzUnits * BlockDim;
    const int64_t aL1Bytes = static_cast<int64_t>(BlockDim) * ReductionK * sizeof(T);
    auto scratchTensor = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ float *>(scratch)),
        Te::MakeFrameLayout<Te::NDExtLayoutPtn,
            BsrCubeL0cLayoutTrait>(scratchRows, BlockDim));
    auto tensorAL1 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::L1, T>(0),
        Te::MakeFrameLayout<Te::NZLayoutPtn, Te::LayoutTraitDefault<T>>(BlockDim, ReductionK));
    auto tensorBL1 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::L1, T>(aL1Bytes),
        Te::MakeFrameLayout<Te::ZNLayoutPtn, Te::LayoutTraitDefault<T>>(ReductionK, BlockDim));
    auto tensorAL0 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::L0A, T>(0),
        Te::MakeFrameLayout<Te::NZLayoutPtn, Te::LayoutTraitDefault<T>>(BlockDim, ReductionK));
    auto tensorBL0 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::L0B, T>(0),
        Te::MakeFrameLayout<Te::ZNLayoutPtn, Te::LayoutTraitDefault<T>>(ReductionK, BlockDim));
    SddmmBsrCubeTiles<decltype(tensorAL1), decltype(tensorBL1),
        decltype(tensorAL0), decltype(tensorBL0), decltype(scratchTensor)> tiles{
        tensorAL1, tensorBL1, tensorAL0, tensorBL0, scratchTensor};
    const SddmmBsrCubeInputs inputs{sparseOffsets, sparseColumns, matX, matY, tiling};
    SddmmBsrCubeRun<T, BlockDim, ReductionK>(inputs, tiles);
}

static __simt_callee__ __aicore__ inline float SddmmBsrHalfToFloat(uint16_t value)
{
    const uint32_t sign = (static_cast<uint32_t>(value) >> 15) & 1U;
    const uint32_t exponent = (static_cast<uint32_t>(value) >> 10) & 0x1FU;
    const uint32_t mantissa = static_cast<uint32_t>(value) & 0x3FFU;
    uint32_t bits = 0U;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign << 31;
        } else {
            int32_t shift = -1;
            uint32_t normalized = mantissa;
            do {
                ++shift;
                normalized <<= 1;
            } while ((normalized & 0x400U) == 0U);
            bits = (sign << 31) |
                (static_cast<uint32_t>(127 - 15 - shift) << 23) |
                ((normalized & 0x3FFU) << 13);
        }
    } else if (exponent == 0x1FU) {
        bits = (sign << 31) | (0xFFU << 23) | (mantissa << 13);
    } else {
        bits = (sign << 31) | ((exponent + 127U - 15U) << 23) | (mantissa << 13);
    }
    float result = 0.0F;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

template <int32_t OutputType>
static __simt_callee__ __aicore__ inline float SddmmBsrLoadOld(
    __gm__ const uint8_t *values, int64_t index)
{
    if constexpr (OutputType == SDDMM_VALUE_FP32) {
        return reinterpret_cast<__gm__ const float *>(values)[index];
    } else if constexpr (OutputType == SDDMM_VALUE_FP16) {
        return SddmmBsrHalfToFloat(reinterpret_cast<__gm__ const uint16_t *>(values)[index]);
    } else {
        return sddmm_arch35::SddmmBf16ToFloat(reinterpret_cast<__gm__ const uint16_t *>(values)[index]);
    }
}

template <int32_t OutputType>
static __simt_callee__ __aicore__ inline void SddmmBsrStore(
    __gm__ uint8_t *values, int64_t index, float value)
{
    if constexpr (OutputType == SDDMM_VALUE_FP32) {
        reinterpret_cast<__gm__ float *>(values)[index] = value;
    } else if constexpr (OutputType == SDDMM_VALUE_FP16) {
        reinterpret_cast<__gm__ uint16_t *>(values)[index] = sddmm_arch35::SddmmFloatToHalf(value);
    } else {
        reinterpret_cast<__gm__ uint16_t *>(values)[index] = sddmm_arch35::SddmmFloatToBf16(value);
    }
}

template <int32_t OutputType>
__simt_vf__ __aicore__ __launch_bounds__(kBsrCubeVectorThreads) inline void
SddmmBsrCubeEpilogueSimt(
    __gm__ uint8_t *values,
    __gm__ const float *scratch,
    SddmmTilingData tiling,
    float alpha,
    float beta,
    int32_t threadsPerCore,
    int32_t coreCount)
{
    const int64_t blockArea = static_cast<int64_t>(tiling.blockDim) * tiling.blockDim;
    const uint64_t total = static_cast<uint64_t>(tiling.batchCount) *
        static_cast<uint64_t>(tiling.nnzUnits) * static_cast<uint64_t>(blockArea);
    const uint64_t globalThread = static_cast<uint64_t>(blockIdx.x) * threadsPerCore + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(coreCount) * threadsPerCore;
    const uint64_t valuesPerBatch = static_cast<uint64_t>(tiling.nnzUnits) * blockArea;

    for (uint64_t scratchIndex = globalThread; scratchIndex < total; scratchIndex += stride) {
        const int64_t batch = static_cast<int64_t>(scratchIndex / valuesPerBatch);
        const int64_t local = static_cast<int64_t>(scratchIndex % valuesPerBatch);
        const int64_t blockSlot = local / blockArea;
        const int64_t logicalInner = local - blockSlot * blockArea;
        const int64_t innerRow = logicalInner / tiling.blockDim;
        const int64_t innerColumn = logicalInner - innerRow * tiling.blockDim;
        const int64_t physicalInner = tiling.blockOrder == 0 ? logicalInner :
            innerColumn * tiling.blockDim + innerRow;
        const int64_t valueIndex = batch * tiling.valuesBatchStride +
            blockSlot * blockArea + physicalInner;
        float result = 0.0F;
        if (alpha == 0.0F) {
            // IEEE 0*NaN is NaN, but SDDMM scalars define a zero coefficient as
            // suppressing that operand.  Do not read the Cube product here.
            result = beta == 0.0F ? 0.0F :
                beta * SddmmBsrLoadOld<OutputType>(values, valueIndex);
        } else if (beta == 0.0F) {
            result = alpha * scratch[scratchIndex];
        } else {
            const float oldValue =
                SddmmBsrLoadOld<OutputType>(values, valueIndex);
            result = __fmaf_rn(
                alpha, scratch[scratchIndex], beta * oldValue);
        }
        SddmmBsrStore<OutputType>(values, valueIndex, result);
    }
}

// Reorder each FP32 block entirely in UB. The data registers carry B32
// payloads, so NaNs, signed zeros and every other bit pattern are copied
// without floating-point arithmetic or conversion.
template <uint32_t BlockDim>
__simd_vf__ inline void SddmmBsrCubeTransposeFp32Vf(
    __ubuf__ uint32_t *dst, __ubuf__ uint32_t *src)
{
    using namespace AscendC::MicroAPI;
    static_assert(BlockDim == 16 || BlockDim == 32 || BlockDim == 64,
        "unsupported BSR transpose block size");
    constexpr uint32_t kVectorElements = 64;
    constexpr uint16_t kVectorCount = BlockDim * BlockDim / kVectorElements;
    constexpr int16_t kBlockShift = BlockDim == 16 ? 4 : (BlockDim == 32 ? 5 : 6);
    constexpr int32_t kColumnsPerVector = kVectorElements / BlockDim;
    MaskReg mask = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<int32_t> logicalIndex;
    RegTensor<int32_t> rowMask;
    RegTensor<int32_t> rowIndex;
    RegTensor<int32_t> columnIndex;
    RegTensor<int32_t> sourceIndex;
    RegTensor<uint32_t> gathered;
    Arange(logicalIndex, int32_t{0});
    Duplicate(rowMask, static_cast<int32_t>(BlockDim - 1), mask);
    And(rowIndex, logicalIndex, rowMask, mask);
    ShiftLefts(rowIndex, rowIndex, kBlockShift, mask);
    ShiftRights(columnIndex, logicalIndex, kBlockShift, mask);
    Add(sourceIndex, rowIndex, columnIndex, mask);
    for (uint16_t vector = 0; vector < kVectorCount; ++vector) {
        Gather(gathered, src, (RegTensor<uint32_t> &)sourceIndex, mask);
        StoreAlign(dst + vector * kVectorElements, gathered, mask);
        // A 64-element output vector spans an integral number of columns
        // for every supported block size; only this column offset changes.
        Adds(sourceIndex, sourceIndex, kColumnsPerVector, mask);
    }
}

template <uint32_t BlockDim>
class SddmmBsrCubeTransposeFp32Kernel {
public:
    __aicore__ inline void Init(
        GM_ADDR values, GM_ADDR scratch,
        const SddmmTilingData &tiling, AscendC::TPipe *pipe)
    {
        physicalBlocks_ = tiling.nnzUnits;
        values_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(values),
            static_cast<uint64_t>(tiling.outputsPerBatch));
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(scratch),
            static_cast<uint64_t>(tiling.outputsPerBatch));
        pipe->InitBuffer(sourceBuffer_, kBlockBytes);
        pipe->InitBuffer(outputBuffer_, kBlockBytes);
    }

    __aicore__ inline void Process()
    {
        const int64_t core = static_cast<int64_t>(GetBlockIdx());
        const int64_t coreCount = static_cast<int64_t>(GetBlockNum());
        auto source = sourceBuffer_.Get<uint32_t>();
        auto output = outputBuffer_.Get<uint32_t>();
        AscendC::DataCopyExtParams blockCopy{1, kBlockBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> noPad{false, 0, 0, 0};
        for (int64_t block = core; block < physicalBlocks_; block += coreCount) {
            const int64_t offset = block * kBlockElements;
            AscendC::DataCopyPad(source, scratch_[offset], blockCopy, noPad);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            SddmmBsrCubeTransposeFp32Vf<BlockDim>(
                reinterpret_cast<__ubuf__ uint32_t *>(output.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(source.GetPhyAddr()));
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            // Exactly one block payload is written, including for legal
            // FP32 descriptor addresses that are only four-byte aligned.
            AscendC::DataCopyPad(values_[offset], output, blockCopy);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
    }

private:
    static constexpr uint32_t kBlockElements = BlockDim * BlockDim;
    static constexpr uint32_t kBlockBytes = kBlockElements * sizeof(uint32_t);
    int64_t physicalBlocks_{0};
    AscendC::TBuf<AscendC::TPosition::VECCALC> sourceBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuffer_;
    AscendC::GlobalTensor<uint32_t> values_;
    AscendC::GlobalTensor<uint32_t> scratch_;
};

template <uint32_t BlockDim>
__aicore__ inline void SddmmBsrCubeTransposeFp32Run(
    GM_ADDR values, GM_ADDR scratch,
    const SddmmTilingData &tiling, AscendC::TPipe *pipe)
{
    SddmmBsrCubeTransposeFp32Kernel<BlockDim> epilogue;
    epilogue.Init(values, scratch, tiling, pipe);
    epilogue.Process();
}

// The public scratch limit bounds a single batch to 16,777,216 FP32
// elements. Identity epilogues can therefore use 32-bit output ownership;
// constant block dimensions turn the COL permutation into shifts and masks.
// General scalars and strided batches keep the full epilogue above.
template <int32_t OutputType, uint32_t BlockDim, bool ColumnMajor>
__simt_vf__ __aicore__ __launch_bounds__(kBsrCubeVectorThreads) inline void
SddmmBsrCubeIdentityEpilogueSimt(
    __gm__ uint8_t *values,
    __gm__ const float *scratch,
    uint32_t total,
    uint32_t coreCount)
{
    constexpr uint32_t kBlockArea = BlockDim * BlockDim;
    const uint32_t globalThread = blockIdx.x * kBsrCubeVectorThreads + threadIdx.x;
    const uint32_t stride = coreCount * kBsrCubeVectorThreads;
    for (uint32_t logicalIndex = globalThread; logicalIndex < total;
         logicalIndex += stride) {
        uint32_t valueIndex = logicalIndex;
        if constexpr (ColumnMajor) {
            const uint32_t inner = logicalIndex & (kBlockArea - 1U);
            valueIndex = (logicalIndex & ~(kBlockArea - 1U)) +
                (inner % BlockDim) * BlockDim + inner / BlockDim;
        }
        if constexpr (OutputType == SDDMM_VALUE_FP16) {
            // Native conversion replaces the per-element software encoder.
            // No saturation is applied: finite overflow retains IEEE Inf.
            reinterpret_cast<__gm__ __fp16 *>(values)[valueIndex] =
                static_cast<__fp16>(scratch[logicalIndex]);
        } else {
            SddmmBsrStore<OutputType>(values, valueIndex, scratch[logicalIndex]);
        }
    }
}

template <int32_t OutputType, uint32_t BlockDim>
__aicore__ inline void SddmmBsrCubeDispatchIdentityOrder(
    __gm__ uint8_t *values, __gm__ const float *scratch,
    const SddmmTilingData &tiling, uint32_t coreCount)
{
    const uint32_t total = static_cast<uint32_t>(tiling.outputsPerBatch);
    if (tiling.blockOrder == 0) {
        asc_vf_call<SddmmBsrCubeIdentityEpilogueSimt<OutputType, BlockDim, false>>(
            dim3{kBsrCubeVectorThreads}, values, scratch, total, coreCount);
    } else {
        asc_vf_call<SddmmBsrCubeIdentityEpilogueSimt<OutputType, BlockDim, true>>(
            dim3{kBsrCubeVectorThreads}, values, scratch, total, coreCount);
    }
}

template <int32_t OutputType>
__aicore__ inline void SddmmBsrCubeDispatchIdentityBlock(
    __gm__ uint8_t *values, __gm__ const float *scratch,
    const SddmmTilingData &tiling, uint32_t coreCount)
{
    if (tiling.blockDim == 16) {
        SddmmBsrCubeDispatchIdentityOrder<OutputType, 16>(values, scratch, tiling, coreCount);
    } else if (tiling.blockDim == 32) {
        SddmmBsrCubeDispatchIdentityOrder<OutputType, 32>(values, scratch, tiling, coreCount);
    } else if (tiling.blockDim == 64) {
        SddmmBsrCubeDispatchIdentityOrder<OutputType, 64>(values, scratch, tiling, coreCount);
    }
}

template <typename T, int32_t ReductionK>
__aicore__ inline void SddmmBsrCubeDispatchBlockDim(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns,
    GM_ADDR matX, GM_ADDR matY, GM_ADDR scratch,
    const SddmmTilingData &tiling)
{
    if (tiling.blockDim == 16) {
        SddmmBsrCubeImpl<T, 16, ReductionK>(sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
    } else if (tiling.blockDim == 32) {
        SddmmBsrCubeImpl<T, 32, ReductionK>(sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
    } else if (tiling.blockDim == 64) {
        SddmmBsrCubeImpl<T, 64, ReductionK>(sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
    }
}

template <typename T>
__aicore__ inline void SddmmBsrCubeDispatchReduction(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns,
    GM_ADDR matX, GM_ADDR matY, GM_ADDR scratch,
    const SddmmTilingData &tiling)
{
    if (tiling.k == 32) {
        SddmmBsrCubeDispatchBlockDim<T, 32>(
            sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
    } else if (tiling.k == 64) {
        SddmmBsrCubeDispatchBlockDim<T, 64>(
            sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
    } else if (tiling.k == 128) {
        SddmmBsrCubeDispatchBlockDim<T, 128>(
            sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
    }
}

__aicore__ inline void SddmmBsrCubeDispatchIdentity(
    GM_ADDR sparseValues, GM_ADDR scratch, const SddmmTilingData &tiling,
    uint32_t coreCount)
{
    __gm__ uint8_t *values = reinterpret_cast<__gm__ uint8_t *>(sparseValues);
    __gm__ const float *scratchValues = reinterpret_cast<__gm__ const float *>(scratch);
    if (tiling.outputType == SDDMM_VALUE_FP32) {
        if (tiling.blockOrder == 1) {
            AscendC::TPipe pipe;
            if (tiling.blockDim == 16) {
                SddmmBsrCubeTransposeFp32Run<16>(sparseValues, scratch, tiling, &pipe);
            } else if (tiling.blockDim == 32) {
                SddmmBsrCubeTransposeFp32Run<32>(sparseValues, scratch, tiling, &pipe);
            } else if (tiling.blockDim == 64) {
                SddmmBsrCubeTransposeFp32Run<64>(sparseValues, scratch, tiling, &pipe);
            }
        } else {
            SddmmBsrCubeDispatchIdentityBlock<SDDMM_VALUE_FP32>(
                values, scratchValues, tiling, coreCount);
        }
    } else if (tiling.outputType == SDDMM_VALUE_FP16) {
        SddmmBsrCubeDispatchIdentityBlock<SDDMM_VALUE_FP16>(
            values, scratchValues, tiling, coreCount);
    } else {
        SddmmBsrCubeDispatchIdentityBlock<SDDMM_VALUE_BF16>(
            values, scratchValues, tiling, coreCount);
    }
}

} // namespace

extern "C" __global__ __aicore__ void sddmm_bsr_cube_fp16_kernel(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns,
    GM_ADDR matX, GM_ADDR matY, GM_ADDR scratch,
    const SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SddmmBsrCubeDispatchReduction<half>(
        sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
}

extern "C" __global__ __aicore__ void sddmm_bsr_cube_bf16_kernel(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns,
    GM_ADDR matX, GM_ADDR matY, GM_ADDR scratch,
    const SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SddmmBsrCubeDispatchReduction<__bf16>(
        sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
}

extern "C" __global__ __aicore__ void sddmm_bsr_cube_fp32_kernel(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns,
    GM_ADDR matX, GM_ADDR matY, GM_ADDR scratch,
    const SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SddmmBsrCubeDispatchReduction<float>(
        sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
}

extern "C" __global__ __aicore__ void sddmm_bsr_cube_epilogue_kernel(
    GM_ADDR sparseValues, GM_ADDR scratch, const SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __gm__ uint8_t *values = reinterpret_cast<__gm__ uint8_t *>(sparseValues);
    __gm__ const float *scratchValues = reinterpret_cast<__gm__ const float *>(scratch);
    float alpha = tiling.alphaReal;
    float beta = tiling.betaReal;
    if (tiling.pointerMode != 0) {
        alpha = reinterpret_cast<__gm__ const float *>(tiling.alphaPtr)[0];
        beta = reinterpret_cast<__gm__ const float *>(tiling.betaPtr)[0];
    }
    const int32_t coreCount = static_cast<int32_t>(GetBlockNum());
    if (tiling.pointerMode == 0 && tiling.batchCount == 1 &&
        alpha == 1.0F && beta == 0.0F && tiling.outputsPerBatch > 0 &&
        static_cast<uint64_t>(tiling.outputsPerBatch) <=
            sddmm_arch35::kSddmmBsrCubeScratchLimit / sizeof(float)) {
        SddmmBsrCubeDispatchIdentity(
            sparseValues, scratch, tiling, static_cast<uint32_t>(coreCount));
        return;
    }
    if (tiling.outputType == SDDMM_VALUE_FP32) {
        asc_vf_call<SddmmBsrCubeEpilogueSimt<SDDMM_VALUE_FP32>>(
            dim3{kBsrCubeVectorThreads}, values, scratchValues, tiling,
            alpha, beta, static_cast<int32_t>(kBsrCubeVectorThreads), coreCount);
    } else if (tiling.outputType == SDDMM_VALUE_FP16) {
        asc_vf_call<SddmmBsrCubeEpilogueSimt<SDDMM_VALUE_FP16>>(
            dim3{kBsrCubeVectorThreads}, values, scratchValues, tiling,
            alpha, beta, static_cast<int32_t>(kBsrCubeVectorThreads), coreCount);
    } else {
        asc_vf_call<SddmmBsrCubeEpilogueSimt<SDDMM_VALUE_BF16>>(
            dim3{kBsrCubeVectorThreads}, values, scratchValues, tiling,
            alpha, beta, static_cast<int32_t>(kBsrCubeVectorThreads), coreCount);
    }
}

extern "C" void sddmm_bsr_cube_compute_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns,
    GM_ADDR matX, GM_ADDR matY, GM_ADDR scratch,
    const SddmmTilingData &tiling, uint32_t cubeBlockDim, void *stream)
{
    if (tiling.inputType == SDDMM_VALUE_FP16) {
        sddmm_bsr_cube_fp16_kernel<<<cubeBlockDim, nullptr, stream>>>(
            sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
    } else if (tiling.inputType == SDDMM_VALUE_BF16) {
        sddmm_bsr_cube_bf16_kernel<<<cubeBlockDim, nullptr, stream>>>(
            sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
    } else if (tiling.inputType == SDDMM_VALUE_FP32) {
        sddmm_bsr_cube_fp32_kernel<<<cubeBlockDim, nullptr, stream>>>(
            sparseOffsets, sparseColumns, matX, matY, scratch, tiling);
    }
}

extern "C" void sddmm_bsr_cube_epilogue_kernel_do(
    GM_ADDR sparseValues, GM_ADDR scratch,
    const SddmmTilingData &tiling, uint32_t vectorBlockDim, void *stream)
{
    sddmm_bsr_cube_epilogue_kernel<<<vectorBlockDim, nullptr, stream>>>(
        sparseValues, scratch, tiling);
}

extern "C" void sddmm_bsr_cube_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns, GM_ADDR sparseValues,
    GM_ADDR matX, GM_ADDR matY, GM_ADDR scratch,
    const SddmmTilingData &tiling,
    uint32_t cubeBlockDim, uint32_t vectorBlockDim, void *stream)
{
    // For every supported FP32 block size, Fixpipe's logical ROW tile is
    // already the requested identity output. Write it to C directly to avoid
    // a second kernel and scratch traffic. Batching, COL blocks, mixed output
    // and general alpha/beta retain the scratch epilogue.
    const bool directFp32Output =
        (tiling.blockDim == 16 || tiling.blockDim == 32 || tiling.blockDim == 64) &&
        (tiling.k == 32 || tiling.k == 64 || tiling.k == 128) &&
        tiling.inputType == SDDMM_VALUE_FP32 &&
        tiling.outputType == SDDMM_VALUE_FP32 &&
        tiling.computeType == SDDMM_VALUE_FP32 &&
        tiling.pointerMode == 0 && tiling.batchCount == 1 &&
        tiling.valuesBatchStride == 0 && tiling.blockOrder == 0 &&
        (reinterpret_cast<uintptr_t>(sparseValues) & 63U) == 0U &&
        tiling.alphaReal == 1.0F && tiling.betaReal == 0.0F;
    GM_ADDR cubeOutput = directFp32Output ? sparseValues : scratch;
    sddmm_bsr_cube_compute_kernel_do(
        sparseOffsets, sparseColumns, matX, matY, cubeOutput,
        tiling, cubeBlockDim, stream);
    if (!directFp32Output) {
        sddmm_bsr_cube_epilogue_kernel_do(
            sparseValues, scratch, tiling, vectorBlockDim, stream);
    }
}
