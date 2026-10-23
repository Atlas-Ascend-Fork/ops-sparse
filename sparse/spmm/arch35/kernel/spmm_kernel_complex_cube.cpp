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

#include "spmm_kernel_device_common.h"

using namespace AscendC;
using namespace matmul;


class KernelSpmmComplex64Cube {
public:
    __aicore__ inline KernelSpmmComplex64Cube() {}

    __aicore__ inline void Init(GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        td_ = LoadSpmmKernelTilingData(tilingGM);
        workspace_ = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        meta_ = reinterpret_cast<__gm__ const SpmmCubeGroupMeta *>(
            workspace_ + td_.cube_meta_offset);
        const uint64_t aPlaneElements = static_cast<uint64_t>(td_.cube_group_count) *
            SPMM_CUBE_MIN_ROWS * SPMM_CUBE_MAX_K;
        const uint64_t bPlaneElements = td_.cube_complex_b_plane_elements;
        arHigh_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace_ + td_.cube_packed_a_offset));
        aiHigh_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace_ + td_.cube_packed_a_offset + aPlaneElements * sizeof(half)));
        negAiHigh_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace_ + td_.cube_packed_a_offset + 2 * aPlaneElements * sizeof(half)));
        arLow_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace_ + td_.cube_packed_a_offset + 3 * aPlaneElements * sizeof(half)));
        aiLow_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace_ + td_.cube_packed_a_offset + 4 * aPlaneElements * sizeof(half)));
        negAiLow_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace_ + td_.cube_packed_a_offset + 5 * aPlaneElements * sizeof(half)));
        brHigh_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace_ + td_.cube_complex_b_offset));
        biHigh_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace_ + td_.cube_complex_b_offset + bPlaneElements * sizeof(half)));
        brLow_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace_ + td_.cube_complex_b_offset + 2 * bPlaneElements * sizeof(half)));
        biLow_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace_ + td_.cube_complex_b_offset + 3 * bPlaneElements * sizeof(half)));
        const uint64_t scratchPlaneElements = static_cast<uint64_t>(td_.cube_group_count) *
            SPMM_CUBE_MIN_ROWS * static_cast<uint32_t>(td_.n);
        realScratch_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace_ + td_.cube_scratch_offset));
        imagScratch_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace_ + td_.cube_scratch_offset + scratchPlaneElements * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = GetBlockIdx();
        const uint32_t blocks = GetBlockNum();
        const uint32_t groups = static_cast<uint32_t>(td_.cube_group_count);
        if (blocks == 0u) {
            return;
        }
        const uint32_t firstGroup = static_cast<uint32_t>(
            static_cast<uint64_t>(groups) * block / blocks);
        const uint32_t lastGroup = static_cast<uint32_t>(
            static_cast<uint64_t>(groups) * (block + 1) / blocks);
        for (uint32_t group = firstGroup; group < lastGroup; ++group) {
            RunGroup(group);
        }
    }

private:
    static constexpr uint32_t kTileM = SPMM_CUBE_MIN_ROWS;
    static constexpr uint32_t kTileK = SPMM_CUBE_MAX_K;
    static constexpr uint32_t kTileN = 16;
    static constexpr uint32_t kAElements = kTileM * kTileK;
    static constexpr uint32_t kBElements = kTileK * SPMM_CUBE_MAX_N;
    static constexpr uint32_t kCElements = kTileM * SPMM_CUBE_MAX_N;
    // LocalTensor offsets are byte offsets. Keep B1 after all six A1 planes.
    static constexpr uint32_t kB1OffsetBytes =
        6 * kAElements * sizeof(half);
    static constexpr uint32_t kBrSlot = 0;
    static constexpr uint32_t kBiSlot = 1;

    __aicore__ inline void Product(LocalTensor<float> output,
                                   LocalTensor<half> a, uint32_t aOffset,
                                   LocalTensor<half> b, uint32_t bOffset,
                                   uint32_t packedK, uint32_t tileN,
                                   bool initialize)
    {
        MmadParams mm{};
        mm.m = kTileM;
        mm.n = kTileN;
        mm.k = packedK;
        mm.cmatrixInitVal = initialize;
        mm.cmatrixSource = false;
        mm.disableGemv = true;
        mm.unitFlag = 0;
        const uint32_t tileCount = (tileN + kTileN - 1) / kTileN;
        const uint32_t bTileElements = packedK * kTileN;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            Mmad(output[tile * kTileM * kTileN], a[aOffset],
                 b[bOffset + tile * bTileElements], mm);
        }
    }

    __aicore__ inline void LoadAPlanes(uint32_t group, uint32_t packedK)
    {
        const uint64_t offset = static_cast<uint64_t>(group) * kAElements;
        Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = kTileM;
        copy.dValue = packedK;
        copy.srcDValue = kTileK;
        copy.dstNzC0Stride = kTileM;
        copy.dstNzNStride = 1;
        DataCopy(aL1_[0 * kAElements], arHigh_[offset], copy);
        DataCopy(aL1_[1 * kAElements], aiHigh_[offset], copy);
        DataCopy(aL1_[2 * kAElements], negAiHigh_[offset], copy);
        DataCopy(aL1_[3 * kAElements], arLow_[offset], copy);
        DataCopy(aL1_[4 * kAElements], aiLow_[offset], copy);
        DataCopy(aL1_[5 * kAElements], negAiLow_[offset], copy);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        LoadData2DParamsV2 load{};
        load.mStep = 1;
        load.kStep = packedK / 16;
        load.srcStride = 1;
        load.dstStride = 1;
        load.ifTranspose = false;
        for (uint32_t plane = 0; plane < 6; ++plane) {
            LoadData(aL0_[plane * kAElements], aL1_[plane * kAElements], load);
        }
    }

    __aicore__ inline void LoadBPlanes(GlobalTensor<half> &real,
                                       GlobalTensor<half> &imag,
                                       uint32_t group, uint32_t packedK,
                                       uint32_t columnBegin,
                                       uint32_t tileN)
    {
        const int32_t colBase = meta_[group].col_base;
        const uint64_t offset = static_cast<uint64_t>(colBase) *
            static_cast<uint32_t>(td_.n) + columnBegin;
        Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = packedK;
        copy.dValue = tileN;
        copy.srcDValue = td_.n;
        copy.dstNzC0Stride = packedK;
        copy.dstNzNStride = 1;
        DataCopy(bL1_[kBrSlot * kBElements], real[offset], copy);
        DataCopy(bL1_[kBiSlot * kBElements], imag[offset], copy);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        LoadData2DParamsV2 load{};
        load.mStep = 1;
        load.kStep = packedK / 16;
        load.srcStride = 1;
        load.dstStride = 1;
        load.ifTranspose = true;
        const uint32_t tileCount = (tileN + kTileN - 1) / kTileN;
        const uint32_t bTileElements = packedK * kTileN;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            LoadData(bL0_[kBrSlot * kBElements + tile * bTileElements],
                     bL1_[kBrSlot * kBElements + tile * bTileElements], load);
            LoadData(bL0_[kBiSlot * kBElements + tile * bTileElements],
                     bL1_[kBiSlot * kBElements + tile * bTileElements], load);
        }
    }

    __aicore__ inline void RunGroup(uint32_t group)
    {
        const uint32_t packedK = static_cast<uint32_t>(meta_[group].packed_k);
        if (packedK != 32 && packedK != kTileK) {
            return;
        }
        LoadAPlanes(group, packedK);
        const uint32_t n = static_cast<uint32_t>(td_.n);
        for (uint32_t columnBegin = 0; columnBegin < n;
             columnBegin += SPMM_CUBE_MAX_N) {
            if (columnBegin != 0) {
                WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            }
            const uint32_t tileN = n - columnBegin < SPMM_CUBE_MAX_N ?
                n - columnBegin : SPMM_CUBE_MAX_N;
            RunTile(group, packedK, columnBegin, tileN);
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    __aicore__ inline void RunTile(uint32_t group, uint32_t packedK,
                                   uint32_t columnBegin,
                                   uint32_t tileN)
    {
        LoadBPlanes(brHigh_, biHigh_, group, packedK, columnBegin, tileN);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        RunHighProducts(packedK, tileN);

        // Do not overwrite the two B2 slots until all high-B MMADs retire.
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        LoadBPlanes(brLow_, biLow_, group, packedK, columnBegin, tileN);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        RunLowProducts(packedK, tileN);

        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fix{};
        fix.params.ndNum = 1;
        fix.mSize = kTileM;
        fix.nSize = tileN;
        fix.srcStride = kTileM;
        fix.dstStride = td_.n;
        fix.quantPre = QuantMode_t::NoQuant;
        fix.unitFlag = 0;
        const uint64_t scratchOffset = static_cast<uint64_t>(group) * kTileM *
            static_cast<uint32_t>(td_.n) + columnBegin;
        Fixpipe<float, float, CFG_ROW_MAJOR>(realScratch_[scratchOffset], realLocal_, fix);
        Fixpipe<float, float, CFG_ROW_MAJOR>(imagScratch_[scratchOffset], imagLocal_, fix);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    __aicore__ inline void RunHighProducts(uint32_t packedK, uint32_t tileN)
    {
        Product(realLocal_, aL0_, 0 * kAElements, bL0_, kBrSlot * kBElements,
                packedK, tileN, true);
        Product(realLocal_, aL0_, 2 * kAElements, bL0_, kBiSlot * kBElements,
                packedK, tileN, false);
        Product(realLocal_, aL0_, 3 * kAElements, bL0_, kBrSlot * kBElements,
                packedK, tileN, false);
        Product(realLocal_, aL0_, 5 * kAElements, bL0_, kBiSlot * kBElements,
                packedK, tileN, false);
        Product(imagLocal_, aL0_, 1 * kAElements, bL0_, kBrSlot * kBElements,
                packedK, tileN, true);
        Product(imagLocal_, aL0_, 0 * kAElements, bL0_, kBiSlot * kBElements,
                packedK, tileN, false);
        Product(imagLocal_, aL0_, 4 * kAElements, bL0_, kBrSlot * kBElements,
                packedK, tileN, false);
        Product(imagLocal_, aL0_, 3 * kAElements, bL0_, kBiSlot * kBElements,
                packedK, tileN, false);
    }

    __aicore__ inline void RunLowProducts(uint32_t packedK, uint32_t tileN)
    {
        Product(realLocal_, aL0_, 0 * kAElements, bL0_, kBrSlot * kBElements,
                packedK, tileN, false);
        Product(realLocal_, aL0_, 2 * kAElements, bL0_, kBiSlot * kBElements,
                packedK, tileN, false);
        Product(imagLocal_, aL0_, 1 * kAElements, bL0_, kBrSlot * kBElements,
                packedK, tileN, false);
        Product(imagLocal_, aL0_, 0 * kAElements, bL0_, kBiSlot * kBElements,
                packedK, tileN, false);
    }

    SpmmKernelTilingData td_{};
    __gm__ uint8_t *workspace_{nullptr};
    __gm__ const SpmmCubeGroupMeta *meta_{nullptr};
    GlobalTensor<half> arHigh_;
    GlobalTensor<half> aiHigh_;
    GlobalTensor<half> negAiHigh_;
    GlobalTensor<half> arLow_;
    GlobalTensor<half> aiLow_;
    GlobalTensor<half> negAiLow_;
    GlobalTensor<half> brHigh_;
    GlobalTensor<half> biHigh_;
    GlobalTensor<half> brLow_;
    GlobalTensor<half> biLow_;
    GlobalTensor<float> realScratch_;
    GlobalTensor<float> imagScratch_;
    LocalTensor<half> aL1_{TPosition::A1, 0, 6 * kAElements};
    LocalTensor<half> bL1_{TPosition::B1, kB1OffsetBytes, 2 * kBElements};
    LocalTensor<half> aL0_{TPosition::A2, 0, 6 * kAElements};
    LocalTensor<half> bL0_{TPosition::B2, 0, 2 * kBElements};
    LocalTensor<float> realLocal_{TPosition::CO1, 0, kCElements};
    LocalTensor<float> imagLocal_{TPosition::CO1, kCElements * sizeof(float), kCElements};
};

constexpr uint32_t kComplexSplitTileElements = 4096;
constexpr uint32_t kComplexSplitAlignment = 32;
constexpr float kComplexPackingRelativeError = 0x1p-20f;

__simt_callee__ __aicore__ inline bool IsComplexHalfPairAccurate(
    float value, half high, half low)
{
    const float bound = (value < 0.0f ? -value : value) * kComplexPackingRelativeError;
    const float loss = value - (static_cast<float>(high) + static_cast<float>(low));
    return loss >= -bound && loss <= bound;
}

__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmComplexRefreshA(__gm__ const float *input, __gm__ const uint32_t *packedMap,
                    __gm__ half *arHigh, __gm__ half *aiHigh,
                    __gm__ half *negAiHigh, __gm__ half *arLow,
                    __gm__ half *aiLow, __gm__ half *negAiLow,
                    __gm__ int32_t *status, uint64_t first, uint64_t end)
{
    constexpr float kFp16Max = 65504.0f;
    for (uint64_t index = first + static_cast<uint64_t>(threadIdx.x);
         index < end; index += static_cast<uint64_t>(blockDim.x)) {
        const uint32_t packed = packedMap[index];
        if (packed == 0xffffffffu) {
            continue;
        }
        const float real = input[index * 2];
        const float imag = input[index * 2 + 1];
        if (!Simt::IsFinite(real) || !Simt::IsFinite(imag) ||
            real > kFp16Max || real < -kFp16Max ||
            imag > kFp16Max || imag < -kFp16Max) {
            (void)asc_atomic_add(status, 1);
            continue;
        }
        const half realHigh = static_cast<half>(real);
        const half imagHigh = static_cast<half>(imag);
        const half negativeImagHigh = static_cast<half>(-imag);
        const half realLow = static_cast<half>(real - static_cast<float>(realHigh));
        const half imagLow = static_cast<half>(imag - static_cast<float>(imagHigh));
        const half negativeImagLow = static_cast<half>(
            -imag - static_cast<float>(negativeImagHigh));
        if (!IsComplexHalfPairAccurate(real, realHigh, realLow) ||
            !IsComplexHalfPairAccurate(imag, imagHigh, imagLow)) {
            (void)asc_atomic_add(status, 1);
            continue;
        }
        arHigh[packed] = realHigh;
        aiHigh[packed] = imagHigh;
        negAiHigh[packed] = negativeImagHigh;
        arLow[packed] = realLow;
        aiLow[packed] = imagLow;
        negAiLow[packed] = negativeImagLow;
    }
}

struct SpmmComplexPairAcc {
    float real0;
    float imag0;
    float real1;
    float imag1;
};

__simt_callee__ __aicore__ inline void AccumulateSpmmComplexFallbackPair(
    __gm__ const int32_t *rowOff, __gm__ const int32_t *colInd,
    __gm__ const float *values, __gm__ const float *matB, int32_t row,
    uint32_t column, int32_t ldb, int32_t indexBase, bool fullTile,
    SpmmComplexPairAcc *acc)
{
    const int32_t begin = rowOff[row] - indexBase;
    const int32_t rowEnd = rowOff[row + 1] - indexBase;
    for (int32_t p = begin; p < rowEnd; ++p) {
        const int32_t sparseColumn = colInd[p] - indexBase;
        const __gm__ float *value = values + (static_cast<uint64_t>(p) << 1);
        const __gm__ float *b = matB +
            ((static_cast<uint64_t>(sparseColumn) * static_cast<uint64_t>(ldb) +
              column) << 1);
        const float valueReal = value[0];
        const float valueImag = value[1];
        acc->real0 += valueReal * b[0] - valueImag * b[1];
        acc->imag0 += valueReal * b[1] + valueImag * b[0];
        if (fullTile) {
            acc->real1 += valueReal * b[2] - valueImag * b[3];
            acc->imag1 += valueReal * b[3] + valueImag * b[2];
        }
    }
}

template <bool BetaOne>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmComplexCubeFallbackRows(
    __gm__ const int32_t *rowOff, __gm__ const int32_t *colInd,
    __gm__ const float *values, __gm__ const float *matB,
    __gm__ const float *initialC, __gm__ float *matC,
    __gm__ const int32_t *fallbackRows, uint64_t first, uint64_t end,
    int32_t n, int32_t ldb, int32_t ldc, int32_t indexBase)
{
    const uint32_t columnTiles = (static_cast<uint32_t>(n) + 1u) / 2u;
    for (uint64_t work = first + static_cast<uint64_t>(threadIdx.x);
         work < end; work += static_cast<uint64_t>(blockDim.x)) {
        const uint64_t fallbackIndex = work / columnTiles;
        const uint32_t column = static_cast<uint32_t>(work % columnTiles) * 2u;
        const bool fullTile = column + 1u < static_cast<uint32_t>(n);
        const int32_t row = fallbackRows[fallbackIndex];
        SpmmComplexPairAcc acc{0.0f, 0.0f, 0.0f, 0.0f};
        AccumulateSpmmComplexFallbackPair(rowOff, colInd, values, matB, row,
            column, ldb, indexBase, fullTile, &acc);

        const uint64_t outputOffset =
            ((static_cast<uint64_t>(row) * static_cast<uint64_t>(ldc) + column) << 1);
        if constexpr (BetaOne) {
            acc.real0 += initialC[outputOffset];
            acc.imag0 += initialC[outputOffset + 1];
            if (fullTile) {
                acc.real1 += initialC[outputOffset + 2];
                acc.imag1 += initialC[outputOffset + 3];
            }
        }
        matC[outputOffset] = acc.real0;
        matC[outputOffset + 1] = acc.imag0;
        if (fullTile) {
            matC[outputOffset + 2] = acc.real1;
            matC[outputOffset + 3] = acc.imag1;
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmComplexSplitTail(__gm__ const float *input, __gm__ half *realHigh,
                     __gm__ half *imagHigh, __gm__ half *realLow,
                     __gm__ half *imagLow, __gm__ int32_t *status,
                     uint64_t first, uint64_t end)
{
    constexpr float kFp16Max = 65504.0f;
    for (uint64_t index = first + static_cast<uint64_t>(threadIdx.x);
         index < end; index += static_cast<uint64_t>(blockDim.x)) {
        const float real = input[index * 2];
        const float imag = input[index * 2 + 1];
        if (!Simt::IsFinite(real) || !Simt::IsFinite(imag) ||
            real > kFp16Max || real < -kFp16Max ||
            imag > kFp16Max || imag < -kFp16Max) {
            (void)asc_atomic_add(status, 1);
            continue;
        }
        const half rh = static_cast<half>(real);
        const half ih = static_cast<half>(imag);
        const half rl = static_cast<half>(real - static_cast<float>(rh));
        const half il = static_cast<half>(imag - static_cast<float>(ih));
        if (!IsComplexHalfPairAccurate(real, rh, rl) ||
            !IsComplexHalfPairAccurate(imag, ih, il)) {
            (void)asc_atomic_add(status, 1);
            continue;
        }
        realHigh[index] = rh;
        imagHigh[index] = ih;
        realLow[index] = rl;
        imagLow[index] = il;
    }
}

class KernelSpmmComplex64Split {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR workspaceGM,
                                GM_ADDR tilingGM, uint64_t alignedElements)
    {
        const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
        __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        const uint64_t planeElements = td.cube_complex_b_plane_elements;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
        realHigh_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace + td.cube_complex_b_offset));
        imagHigh_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace + td.cube_complex_b_offset + planeElements * sizeof(half)));
        realLow_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace + td.cube_complex_b_offset + 2 * planeElements * sizeof(half)));
        imagLow_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            workspace + td.cube_complex_b_offset + 3 * planeElements * sizeof(half)));
        status_ = reinterpret_cast<__gm__ int32_t *>(
            workspace + td.cube_complex_status_offset);
        alignedElements_ = alignedElements;

        pipe_.InitBuffer(inputQueue_, 1, kComplexSplitTileElements * 2 * sizeof(float));
        pipe_.InitBuffer(realBuffer_, kComplexSplitTileElements * sizeof(float));
        pipe_.InitBuffer(imagBuffer_, kComplexSplitTileElements * sizeof(float));
        pipe_.InitBuffer(floatTemp_, kComplexSplitTileElements * sizeof(float));
        pipe_.InitBuffer(safetyMask_, kComplexSplitTileElements / 8);
        pipe_.InitBuffer(safetyReduce_, kComplexSplitTileElements * sizeof(float));
        pipe_.InitBuffer(outputQueue_, 2, kComplexSplitTileElements * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        if (status_[0] != 0) {
            return;
        }
        const uint64_t chunks =
            (alignedElements_ + kComplexSplitTileElements - 1) / kComplexSplitTileElements;
        const uint64_t firstChunk = chunks * GetBlockIdx() / GetBlockNum();
        const uint64_t lastChunk = chunks * (GetBlockIdx() + 1) / GetBlockNum();
        for (uint64_t chunk = firstChunk; chunk < lastChunk; ++chunk) {
            const uint64_t offset = chunk * kComplexSplitTileElements;
            const uint32_t count = static_cast<uint32_t>(
                alignedElements_ - offset < kComplexSplitTileElements ?
                    alignedElements_ - offset : kComplexSplitTileElements);
            ProcessTile(offset, count);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint64_t offset, uint32_t count)
    {
        LocalTensor<float> input = inputQueue_.AllocTensor<float>();
        DataCopyExtParams copyIn{1, static_cast<uint32_t>(count * 2 * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> inputPad{false, 0, 0, 0};
        DataCopyPad(input, input_[offset * 2], copyIn, inputPad);
        inputQueue_.EnQue(input);

        input = inputQueue_.DeQue<float>();
        LocalTensor<float> real = realBuffer_.Get<float>();
        LocalTensor<float> imag = imagBuffer_.Get<float>();
        uint64_t reserved = 0;
        const uint16_t repeats = static_cast<uint16_t>(count / kComplexSplitAlignment);
        const GatherMaskParams gather{1, repeats, 8, 0};
        GatherMask(real, input, static_cast<uint8_t>(1), false, 0, gather, reserved);
        GatherMask(imag, input, static_cast<uint8_t>(2), false, 0, gather, reserved);
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);

        CheckSafety(real, imag, count);
        ConvertAndStore(real, realHigh_, realLow_, offset, count);
        ConvertAndStore(imag, imagHigh_, imagLow_, offset, count);
    }

    __aicore__ inline void CheckSafety(const LocalTensor<float> &real,
                                       const LocalTensor<float> &imag,
                                       uint32_t count)
    {
        LocalTensor<float> temp = floatTemp_.Get<float>();
        LocalTensor<uint8_t> safetyMask = safetyMask_.Get<uint8_t>();
        LocalTensor<float> safetyReduce = safetyReduce_.Get<float>();

        Mins(temp, real, 65504.0f, count);
        PipeBarrier<PIPE_V>();
        Maxs(temp, temp, -65504.0f, count);
        PipeBarrier<PIPE_V>();
        Compare(safetyMask, real, temp, CMPMODE::EQ, count);
        PipeBarrier<PIPE_V>();
        Duplicate(safetyReduce, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Select(safetyReduce, safetyMask, safetyReduce, 0.0f,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        PipeBarrier<PIPE_V>();

        Mins(temp, imag, 65504.0f, count);
        PipeBarrier<PIPE_V>();
        Maxs(temp, temp, -65504.0f, count);
        PipeBarrier<PIPE_V>();
        Compare(safetyMask, imag, temp, CMPMODE::EQ, count);
        PipeBarrier<PIPE_V>();
        Duplicate(temp, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Select(temp, safetyMask, temp, 0.0f,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        PipeBarrier<PIPE_V>();
        Add(temp, temp, safetyReduce, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(safetyReduce, temp, safetyReduce, static_cast<int32_t>(count));
        event_t eventIdVToS = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        if (safetyReduce.GetValue(0) != static_cast<float>(count * 2)) {
            AtomicAdd<int32_t>(status_, 1);
        }
    }

    __aicore__ inline void CheckReconstruction(
        const LocalTensor<float> &residual, const LocalTensor<half> &low,
        uint32_t count)
    {
        LocalTensor<float> temp = floatTemp_.Get<float>();
        LocalTensor<float> bound = safetyReduce_.Get<float>();
        LocalTensor<uint8_t> mask = safetyMask_.Get<uint8_t>();
        Cast<float, half>(temp, low, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Sub(temp, residual, temp, count);
        PipeBarrier<PIPE_V>();
        Abs(temp, temp, count);
        PipeBarrier<PIPE_V>();
        Compare(mask, temp, bound, CMPMODE::LE, count);
        PipeBarrier<PIPE_V>();
        Duplicate(temp, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Select(temp, mask, temp, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(bound, temp, bound, static_cast<int32_t>(count));
        const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event);
        WaitFlag<HardEvent::V_S>(event);
        if (bound.GetValue(0) != static_cast<float>(count)) AtomicAdd<int32_t>(status_, 1);
    }

    __aicore__ inline void ConvertAndStore(const LocalTensor<float> &source,
                                           GlobalTensor<half> &highGm,
                                           GlobalTensor<half> &lowGm,
                                           uint64_t offset, uint32_t count)
    {
        LocalTensor<float> temp = floatTemp_.Get<float>();
        LocalTensor<float> bound = safetyReduce_.Get<float>();
        Abs(bound, source, count);
        PipeBarrier<PIPE_V>();
        Muls(bound, bound, kComplexPackingRelativeError, count);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> high = outputQueue_.AllocTensor<half>();
        Cast<half, float>(high, source, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Cast<float, half>(temp, high, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Sub<float>(source, source, temp, count);
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(high);

        LocalTensor<half> low = outputQueue_.AllocTensor<half>();
        Cast<half, float>(low, source, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        CheckReconstruction(source, low, count);
        outputQueue_.EnQue(low);

        DataCopyExtParams copyOut{1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
        high = outputQueue_.DeQue<half>();
        DataCopyPad(highGm[offset], high, copyOut);
        outputQueue_.FreeTensor(high);
        low = outputQueue_.DeQue<half>();
        DataCopyPad(lowGm[offset], low, copyOut);
        outputQueue_.FreeTensor(low);
    }

    TPipe pipe_;
    GlobalTensor<float> input_;
    GlobalTensor<half> realHigh_;
    GlobalTensor<half> imagHigh_;
    GlobalTensor<half> realLow_;
    GlobalTensor<half> imagLow_;
    __gm__ int32_t *status_{nullptr};
    uint64_t alignedElements_{0};
    TQue<TPosition::VECIN, 1> inputQueue_;
    TBuf<TPosition::VECCALC> realBuffer_;
    TBuf<TPosition::VECCALC> imagBuffer_;
    TBuf<TPosition::VECCALC> floatTemp_;
    TBuf<TPosition::VECCALC> safetyMask_;
    TBuf<TPosition::VECCALC> safetyReduce_;
    TQue<TPosition::VECOUT, 2> outputQueue_;
};

template <uint32_t ColumnsPerThread>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmComplexCubeScatter(__gm__ const float *realScratch,
                       __gm__ const float *imagScratch,
                       __gm__ const int32_t *slotToRow,
                       __gm__ const float *initial, __gm__ float *output,
                       uint32_t firstSlot,
                       uint32_t slotStride, uint32_t slotCount,
                       int32_t columns, int32_t leadingDimension, int32_t betaOne)
{
    const uint32_t workPerRow =
        (static_cast<uint32_t>(columns) + ColumnsPerThread - 1) /
        ColumnsPerThread;
    for (uint32_t slot = firstSlot; slot < slotCount; slot += slotStride) {
        const int32_t row = slotToRow[slot];
        if (row < 0) {
            continue;
        }
        const uint64_t scratchRow = static_cast<uint64_t>(slot) *
            static_cast<uint32_t>(columns);
        const uint64_t outputRow = static_cast<uint64_t>(row) *
            static_cast<uint32_t>(leadingDimension);
        for (uint32_t work = static_cast<uint32_t>(threadIdx.x);
             work < workPerRow; work += static_cast<uint32_t>(blockDim.x)) {
            const uint32_t column = work * ColumnsPerThread;
            const uint64_t scratchBase = scratchRow + column;
            const uint64_t outputBase = outputRow + column;
#pragma unroll
            for (uint32_t lane = 0; lane < ColumnsPerThread; ++lane) {
                if (column + lane >= static_cast<uint32_t>(columns)) {
                    break;
                }
                float real = realScratch[scratchBase + lane];
                float imag = imagScratch[scratchBase + lane];
                if (betaOne != 0) {
                    const uint64_t old =
                        reinterpret_cast<__gm__ const uint64_t *>(initial)[outputBase + lane];
                    real += __builtin_bit_cast(float, static_cast<uint32_t>(old));
                    imag += __builtin_bit_cast(float, static_cast<uint32_t>(old >> 32u));
                }
                const uint64_t packed =
                    static_cast<uint64_t>(__builtin_bit_cast(uint32_t, real)) |
                    (static_cast<uint64_t>(__builtin_bit_cast(uint32_t, imag)) << 32u);
                reinterpret_cast<__gm__ uint64_t *>(output)[outputBase + lane] = packed;
            }
        }
    }
}

template <uint32_t ColumnsPerThread>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmComplexCubeScatterFlat(__gm__ const float *realScratch,
                           __gm__ const float *imagScratch,
                           __gm__ const int32_t *slotToRow,
                           __gm__ const float *initial, __gm__ float *output,
                           uint64_t first, uint64_t end,
                           int32_t columns, int32_t leadingDimension,
                           int32_t betaOne)
{
    const uint32_t workPerSlot =
        (static_cast<uint32_t>(columns) + ColumnsPerThread - 1) /
        ColumnsPerThread;
    for (uint64_t task = first + static_cast<uint64_t>(threadIdx.x);
         task < end; task += static_cast<uint64_t>(blockDim.x)) {
        const uint32_t slot = static_cast<uint32_t>(task / workPerSlot);
        const uint32_t column =
            static_cast<uint32_t>(task % workPerSlot) * ColumnsPerThread;
        const int32_t row = slotToRow[slot];
        if (row < 0) {
            continue;
        }
        const uint64_t scratchBase = static_cast<uint64_t>(slot) *
            static_cast<uint32_t>(columns) + column;
        const uint64_t outputBase = static_cast<uint64_t>(row) *
            static_cast<uint32_t>(leadingDimension) + column;
#pragma unroll
        for (uint32_t lane = 0; lane < ColumnsPerThread; ++lane) {
            if (column + lane >= static_cast<uint32_t>(columns)) {
                break;
            }
            float real = realScratch[scratchBase + lane];
            float imag = imagScratch[scratchBase + lane];
            if (betaOne != 0) {
                const uint64_t old =
                    reinterpret_cast<__gm__ const uint64_t *>(initial)[
                        outputBase + lane];
                real += __builtin_bit_cast(
                    float, static_cast<uint32_t>(old));
                imag += __builtin_bit_cast(
                    float, static_cast<uint32_t>(old >> 32u));
            }
            const uint64_t packed =
                static_cast<uint64_t>(__builtin_bit_cast(uint32_t, real)) |
                (static_cast<uint64_t>(
                     __builtin_bit_cast(uint32_t, imag)) << 32u);
            reinterpret_cast<__gm__ uint64_t *>(output)[outputBase + lane] =
                packed;
        }
    }
}

constexpr uint32_t kComplexScatterTileElements = 1024;

#define SPMM_COMPLEX_SCATTER_ACCUMULATE(REAL, IMAG) \
    Reg::LoadAlign(extraReal, (REAL) + offset); \
    Reg::LoadAlign(extraImag, (IMAG) + offset); \
    Reg::Add(realReg, realReg, extraReal, currentMask); \
    Reg::Add(imagReg, imagReg, extraImag, currentMask)

template <bool BetaOne, uint32_t Contributions>
__simd_vf__ inline void SpmmComplexCubeScatterVector(
    __ubuf__ float *output, __ubuf__ float *real,
    __ubuf__ float *imag, __ubuf__ float *real1,
    __ubuf__ float *imag1, __ubuf__ float *real2,
    __ubuf__ float *imag2, __ubuf__ float *initial,
    uint32_t count)
{
    constexpr uint32_t kLanes = 64;
    Reg::RegTensor<float> realReg;
    Reg::RegTensor<float> imagReg;
    Reg::RegTensor<float> extraReal;
    Reg::RegTensor<float> extraImag;
    Reg::RegTensor<float> initial0;
    Reg::RegTensor<float> initial1;
    Reg::RegTensor<float> oldReal;
    Reg::RegTensor<float> oldImag;
    Reg::RegTensor<float> output0;
    Reg::RegTensor<float> output1;
    for (uint32_t offset = 0; offset < count; offset += kLanes) {
        const uint32_t current = count - offset < kLanes ?
            count - offset : kLanes;
        uint32_t currentMaskCount = current;
        Reg::MaskReg currentMask = Reg::UpdateMask<float>(currentMaskCount);
        Reg::LoadAlign(realReg, real + offset);
        Reg::LoadAlign(imagReg, imag + offset);
        if constexpr (Contributions > 1) {
            SPMM_COMPLEX_SCATTER_ACCUMULATE(real1, imag1);
        }
        if constexpr (Contributions > 2) {
            SPMM_COMPLEX_SCATTER_ACCUMULATE(real2, imag2);
        }
        if constexpr (BetaOne) {
            Reg::LoadAlign(initial0, initial + 2 * offset);
            Reg::LoadAlign(initial1, initial + 2 * offset + kLanes);
            Reg::DeInterleave(oldReal, oldImag, initial0, initial1);
            Reg::Add(realReg, realReg, oldReal, currentMask);
            Reg::Add(imagReg, imagReg, oldImag, currentMask);
        }
        Reg::Interleave(output0, output1, realReg, imagReg);

        const uint32_t firstCount = current * 2 < kLanes ?
            current * 2 : kLanes;
        uint32_t firstMaskCount = firstCount;
        Reg::MaskReg firstMask = Reg::UpdateMask<float>(firstMaskCount);
        Reg::StoreAlign(output + 2 * offset, output0, firstMask);
        if (current * 2 > kLanes) {
            uint32_t secondMaskCount = current * 2 - kLanes;
            Reg::MaskReg secondMask =
                Reg::UpdateMask<float>(secondMaskCount);
            Reg::StoreAlign(output + 2 * offset + kLanes, output1, secondMask);
        }
    }
}

#undef SPMM_COMPLEX_SCATTER_ACCUMULATE

class KernelSpmmComplex64CubeScatterVector {
public:
    __aicore__ inline void Init(GM_ADDR initialC, GM_ADDR matC,
                                GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        td_ = LoadSpmmKernelTilingData(tilingGM);
        __gm__ uint8_t *workspace =
            reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        status_ = reinterpret_cast<__gm__ const int32_t *>(
            workspace + td_.cube_complex_status_offset);
        const uint64_t scratchPlaneElements =
            static_cast<uint64_t>(td_.cube_group_count) *
            SPMM_CUBE_MIN_ROWS * static_cast<uint32_t>(td_.n);
        realScratch_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + td_.cube_scratch_offset));
        imagScratch_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + td_.cube_scratch_offset +
            scratchPlaneElements * sizeof(float)));
        slotToRow_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            workspace + td_.cube_row_ids_offset));
        rowSlots_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            workspace + td_.cube_slot1_offset +
            static_cast<uint64_t>(td_.m) * sizeof(int32_t)));
        initial_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initialC));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matC));

        pipe_.InitBuffer(realQueue_, 1,
            kComplexScatterTileElements * sizeof(float));
        pipe_.InitBuffer(imagQueue_, 1,
            kComplexScatterTileElements * sizeof(float));
        pipe_.InitBuffer(initialQueue_, 1,
            2 * kComplexScatterTileElements * sizeof(float));
        pipe_.InitBuffer(outputQueue_, 1,
            2 * kComplexScatterTileElements * sizeof(float));
        if (td_.cube_rows == SPMM_COMPLEX64_SEGMENTED_CUBE_MODE) {
            pipe_.InitBuffer(real1Queue_, 1,
                kComplexScatterTileElements * sizeof(float));
            pipe_.InitBuffer(imag1Queue_, 1,
                kComplexScatterTileElements * sizeof(float));
            pipe_.InitBuffer(real2Queue_, 1,
                kComplexScatterTileElements * sizeof(float));
            pipe_.InitBuffer(imag2Queue_, 1,
                kComplexScatterTileElements * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (status_[0] != 0) {
            return;
        }
        if (td_.cube_rows == SPMM_COMPLEX64_SEGMENTED_CUBE_MODE) {
            ProcessSegmented();
            return;
        }
        const uint32_t block = GetBlockIdx();
        const uint32_t blocks = GetBlockNum();
        const uint32_t slotCount =
            static_cast<uint32_t>(td_.cube_group_count) * SPMM_CUBE_MIN_ROWS;
        for (uint32_t slot = block; slot < slotCount; slot += blocks) {
            const int32_t row = slotToRow_.GetValue(slot);
            if (row < 0) {
                continue;
            }
            const uint64_t scratchRow = static_cast<uint64_t>(slot) *
                static_cast<uint32_t>(td_.n);
            const uint64_t outputRow = static_cast<uint64_t>(row) *
                static_cast<uint32_t>(td_.ldc);
            for (uint32_t column = 0; column < static_cast<uint32_t>(td_.n);
                 column += kComplexScatterTileElements) {
                const uint32_t count =
                    static_cast<uint32_t>(td_.n) - column <
                            kComplexScatterTileElements ?
                        static_cast<uint32_t>(td_.n) - column :
                        kComplexScatterTileElements;
                if (td_.cube_beta_one != 0) {
                    ProcessTile<true>(scratchRow + column,
                                      outputRow + column, count);
                } else {
                    ProcessTile<false>(scratchRow + column,
                                       outputRow + column, count);
                }
            }
        }
    }

private:
    template <bool BetaOne>
    __aicore__ inline void ProcessTile(uint64_t scratchOffset,
                                       uint64_t outputOffset,
                                       uint32_t count)
    {
        LocalTensor<float> real = realQueue_.AllocTensor<float>();
        LocalTensor<float> imag = imagQueue_.AllocTensor<float>();
        const DataCopyExtParams scratchCopy{
            1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
        DataCopyPad(real, realScratch_[scratchOffset], scratchCopy, noPad);
        DataCopyPad(imag, imagScratch_[scratchOffset], scratchCopy, noPad);
        realQueue_.EnQue(real);
        imagQueue_.EnQue(imag);

        LocalTensor<float> initial = initialQueue_.AllocTensor<float>();
        if constexpr (BetaOne) {
            const DataCopyExtParams initialCopy{
                1, static_cast<uint32_t>(count * 2 * sizeof(float)), 0, 0, 0};
            DataCopyPad(initial, initial_[outputOffset * 2],
                        initialCopy, noPad);
        }
        initialQueue_.EnQue(initial);

        real = realQueue_.DeQue<float>();
        imag = imagQueue_.DeQue<float>();
        initial = initialQueue_.DeQue<float>();
        LocalTensor<float> output = outputQueue_.AllocTensor<float>();
        asc_vf_call<SpmmComplexCubeScatterVector<BetaOne, 1>>(
            reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(real.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(imag.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(real.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(imag.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(real.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(imag.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(initial.GetPhyAddr()),
            count);
        outputQueue_.EnQue(output);
        realQueue_.FreeTensor(real);
        imagQueue_.FreeTensor(imag);
        initialQueue_.FreeTensor(initial);

        output = outputQueue_.DeQue<float>();
        const DataCopyExtParams outputCopy{
            1, static_cast<uint32_t>(count * 2 * sizeof(float)), 0, 0, 0};
        DataCopyPad(output_[outputOffset * 2], output, outputCopy);
        outputQueue_.FreeTensor(output);
    }

    __aicore__ inline void ProcessSegmented()
    {
        const uint32_t block = GetBlockIdx();
        const uint32_t blocks = GetBlockNum();
        for (uint32_t row = block; row < static_cast<uint32_t>(td_.m);
             row += blocks) {
            const uint32_t slotBase =
                row * SPMM_COMPLEX64_SEGMENTED_MAX_SLOTS;
            const int32_t slot0 = rowSlots_.GetValue(slotBase);
            const int32_t slot1 = rowSlots_.GetValue(slotBase + 1);
            const int32_t slot2 = rowSlots_.GetValue(slotBase + 2);
            if (slot0 < 0 || slot1 < 0) {
                continue;
            }
            const uint64_t outputRow = static_cast<uint64_t>(row) *
                static_cast<uint32_t>(td_.ldc);
            for (uint32_t column = 0; column < static_cast<uint32_t>(td_.n);
                 column += kComplexScatterTileElements) {
                const uint32_t count =
                    static_cast<uint32_t>(td_.n) - column <
                            kComplexScatterTileElements ?
                        static_cast<uint32_t>(td_.n) - column :
                        kComplexScatterTileElements;
                if (slot2 >= 0) {
                    ProcessSegmentedTile<3>(slot0, slot1, slot2, column,
                                            outputRow + column, count);
                } else {
                    ProcessSegmentedTile<2>(slot0, slot1, slot1, column,
                                            outputRow + column, count);
                }
            }
        }
    }

    template <bool CopyInput>
    __aicore__ inline void QueueSegmentedPair(
        TQue<TPosition::VECIN, 1> &realQueue,
        TQue<TPosition::VECIN, 1> &imagQueue,
        LocalTensor<float> &real, LocalTensor<float> &imag, int32_t slot,
        uint64_t rowStride, uint32_t column,
        const DataCopyExtParams &copy,
        const DataCopyPadExtParams<float> &noPad)
    {
        if constexpr (CopyInput) {
            DataCopyPad(real, realScratch_[static_cast<uint64_t>(slot) * rowStride + column],
                copy, noPad);
            DataCopyPad(imag, imagScratch_[static_cast<uint64_t>(slot) * rowStride + column],
                copy, noPad);
        }
        realQueue.EnQue(real);
        imagQueue.EnQue(imag);
    }

    __aicore__ inline void QueueSegmentedInitial(
        LocalTensor<float> &initial, uint64_t outputOffset, uint32_t count,
        const DataCopyPadExtParams<float> &noPad)
    {
        if (td_.cube_beta_one != 0) {
            const DataCopyExtParams copy{
                1, static_cast<uint32_t>(count * 2 * sizeof(float)), 0, 0, 0};
            DataCopyPad(initial, initial_[outputOffset * 2], copy, noPad);
        }
        initialQueue_.EnQue(initial);
    }

    template <bool BetaOne, uint32_t Contributions>
    __aicore__ inline void RunSegmentedVector(
        LocalTensor<float> &real0, LocalTensor<float> &imag0,
        LocalTensor<float> &real1, LocalTensor<float> &imag1,
        LocalTensor<float> &real2, LocalTensor<float> &imag2,
        LocalTensor<float> &initial, uint32_t count)
    {
        LocalTensor<float> output = outputQueue_.AllocTensor<float>();
        asc_vf_call<SpmmComplexCubeScatterVector<BetaOne, Contributions>>(
            reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(real0.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(imag0.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(real1.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(imag1.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(real2.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(imag2.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(initial.GetPhyAddr()), count);
        outputQueue_.EnQue(output);
        realQueue_.FreeTensor(real0);
        imagQueue_.FreeTensor(imag0);
        real1Queue_.FreeTensor(real1);
        imag1Queue_.FreeTensor(imag1);
        real2Queue_.FreeTensor(real2);
        imag2Queue_.FreeTensor(imag2);
        initialQueue_.FreeTensor(initial);
    }

    __aicore__ inline void StoreSegmentedOutput(
        uint64_t outputOffset, uint32_t count)
    {
        LocalTensor<float> output = outputQueue_.DeQue<float>();
        const DataCopyExtParams copy{
            1, static_cast<uint32_t>(count * 2 * sizeof(float)), 0, 0, 0};
        DataCopyPad(output_[outputOffset * 2], output, copy);
        outputQueue_.FreeTensor(output);
    }

    template <uint32_t Contributions>
    __aicore__ inline void ProcessSegmentedTile(
        int32_t slot0, int32_t slot1, int32_t slot2, uint32_t column,
        uint64_t outputOffset, uint32_t count)
    {
        const uint64_t rowStride = static_cast<uint32_t>(td_.n);
        const DataCopyExtParams scratchCopy{
            1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
        LocalTensor<float> real0 = realQueue_.AllocTensor<float>();
        LocalTensor<float> imag0 = imagQueue_.AllocTensor<float>();
        LocalTensor<float> real1 = real1Queue_.AllocTensor<float>();
        LocalTensor<float> imag1 = imag1Queue_.AllocTensor<float>();
        QueueSegmentedPair<true>(realQueue_, imagQueue_, real0, imag0, slot0,
            rowStride, column, scratchCopy, noPad);
        QueueSegmentedPair<true>(real1Queue_, imag1Queue_, real1, imag1, slot1,
            rowStride, column, scratchCopy, noPad);
        LocalTensor<float> real2 = real2Queue_.AllocTensor<float>();
        LocalTensor<float> imag2 = imag2Queue_.AllocTensor<float>();
        QueueSegmentedPair<(Contributions > 2)>(real2Queue_, imag2Queue_, real2,
            imag2, slot2, rowStride, column, scratchCopy, noPad);

        LocalTensor<float> initial = initialQueue_.AllocTensor<float>();
        QueueSegmentedInitial(initial, outputOffset, count, noPad);

        real0 = realQueue_.DeQue<float>();
        imag0 = imagQueue_.DeQue<float>();
        real1 = real1Queue_.DeQue<float>();
        imag1 = imag1Queue_.DeQue<float>();
        real2 = real2Queue_.DeQue<float>();
        imag2 = imag2Queue_.DeQue<float>();
        initial = initialQueue_.DeQue<float>();
        if (td_.cube_beta_one != 0) {
            RunSegmentedVector<true, Contributions>(real0, imag0, real1, imag1,
                real2, imag2, initial, count);
        } else {
            RunSegmentedVector<false, Contributions>(real0, imag0, real1, imag1,
                real2, imag2, initial, count);
        }
        StoreSegmentedOutput(outputOffset, count);
    }

    SpmmKernelTilingData td_{};
    __gm__ const int32_t *status_{nullptr};
    TPipe pipe_;
    GlobalTensor<float> realScratch_;
    GlobalTensor<float> imagScratch_;
    GlobalTensor<int32_t> slotToRow_;
    GlobalTensor<int32_t> rowSlots_;
    GlobalTensor<float> initial_;
    GlobalTensor<float> output_;
    TQue<TPosition::VECIN, 1> realQueue_;
    TQue<TPosition::VECIN, 1> imagQueue_;
    TQue<TPosition::VECIN, 1> initialQueue_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    TQue<TPosition::VECIN, 1> real1Queue_;
    TQue<TPosition::VECIN, 1> imag1Queue_;
    TQue<TPosition::VECIN, 1> real2Queue_;
    TQue<TPosition::VECIN, 1> imag2Queue_;
};

extern "C" __global__ __aicore__ void spmm_custom_complex64_refresh_a(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
    __gm__ int32_t *status = reinterpret_cast<__gm__ int32_t *>(
        workspace + td.cube_complex_status_offset);
    const uint64_t planeBytes = td.cube_complex_a_plane_elements * sizeof(half);
    __gm__ half *packedA = reinterpret_cast<__gm__ half *>(
        workspace + td.cube_packed_a_offset);
    const uint64_t first = td.cube_a_value_nnz * GetBlockIdx() / GetBlockNum();
    const uint64_t end = td.cube_a_value_nnz * (GetBlockIdx() + 1) / GetBlockNum();
    asc_vf_call<SpmmComplexRefreshA>(dim3{kMaxSimtThreadsPerBlock},
        reinterpret_cast<__gm__ const float *>(csrValues),
        reinterpret_cast<__gm__ const uint32_t *>(
            workspace + td.cube_a_value_map_offset),
        packedA,
        reinterpret_cast<__gm__ half *>(
            workspace + td.cube_packed_a_offset + planeBytes),
        reinterpret_cast<__gm__ half *>(
            workspace + td.cube_packed_a_offset + 2 * planeBytes),
        reinterpret_cast<__gm__ half *>(
            workspace + td.cube_packed_a_offset + 3 * planeBytes),
        reinterpret_cast<__gm__ half *>(
            workspace + td.cube_packed_a_offset + 4 * planeBytes),
        reinterpret_cast<__gm__ half *>(
            workspace + td.cube_packed_a_offset + 5 * planeBytes),
        status, first, end);
}
extern "C" __global__ __aicore__ void spmm_custom_complex64_split_vector(
    GM_ADDR matB, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    const uint64_t aligned = td.cube_complex_b_plane_elements /
        kComplexSplitAlignment * kComplexSplitAlignment;
    if (aligned == 0) {
        return;
    }
    KernelSpmmComplex64Split op;
    op.Init(matB, workspaceGM, tilingGM, aligned);
    op.Process();
}
extern "C" __global__ __aicore__ void spmm_custom_complex64_split_tail(
    GM_ADDR matB, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    const uint64_t elements = td.cube_complex_b_plane_elements;
    const uint64_t aligned = elements / kComplexSplitAlignment * kComplexSplitAlignment;
    if (aligned == elements) {
        return;
    }
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
    __gm__ int32_t *status = reinterpret_cast<__gm__ int32_t *>(
        workspace + td.cube_complex_status_offset);
    if (status[0] != 0) {
        return;
    }
    const uint64_t planeBytes = elements * sizeof(half);
    asc_vf_call<SpmmComplexSplitTail>(dim3{kMaxSimtThreadsPerBlock},
        reinterpret_cast<__gm__ const float *>(matB),
        reinterpret_cast<__gm__ half *>(workspace + td.cube_complex_b_offset),
        reinterpret_cast<__gm__ half *>(workspace + td.cube_complex_b_offset + planeBytes),
        reinterpret_cast<__gm__ half *>(workspace + td.cube_complex_b_offset + 2 * planeBytes),
        reinterpret_cast<__gm__ half *>(workspace + td.cube_complex_b_offset + 3 * planeBytes),
        status, aligned, elements);
}
extern "C" __global__ __aicore__ void spmm_custom_complex64_cube(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    const __gm__ int32_t *status = reinterpret_cast<__gm__ const int32_t *>(
        reinterpret_cast<__gm__ uint8_t *>(workspaceGM) +
        td.cube_complex_status_offset);
    if (status[0] != 0) {
        return;
    }
    // CSR and original dense pointers are intentionally not consumed here:
    // preprocess has packed the guarded Complex64 input into workspace.
    (void)csrRowOffsets;
    (void)csrColInd;
    (void)csrValues;
    (void)matB;
    (void)matC;
    KernelSpmmComplex64Cube op;
    op.Init(workspaceGM, tilingGM);
    op.Process();
}
extern "C" __global__ __aicore__ void spmm_custom_complex64_cube_fallback(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
    const __gm__ int32_t *status = reinterpret_cast<__gm__ const int32_t *>(
        workspace + td.cube_complex_status_offset);
    if (status[0] != 0 || td.cube_complex_fallback_count <= 0) {
        return;
    }
    const uint64_t total = static_cast<uint64_t>(td.cube_complex_fallback_count) *
        ((static_cast<uint32_t>(td.n) + 1u) / 2u);
    const uint64_t first = total * GetBlockIdx() / GetBlockNum();
    const uint64_t end = total * (GetBlockIdx() + 1) / GetBlockNum();
    const __gm__ int32_t *fallbackRows = reinterpret_cast<__gm__ const int32_t *>(
        workspace + td.cube_slot1_offset);
    if (td.cube_beta_one != 0) {
        asc_vf_call<SpmmComplexCubeFallbackRows<true>>(
            dim3{kMaxSimtThreadsPerBlock},
            reinterpret_cast<__gm__ const int32_t *>(csrRowOffsets),
            reinterpret_cast<__gm__ const int32_t *>(csrColInd),
            reinterpret_cast<__gm__ const float *>(csrValues),
            reinterpret_cast<__gm__ const float *>(matB),
            reinterpret_cast<__gm__ const float *>(initialC),
            reinterpret_cast<__gm__ float *>(matC), fallbackRows,
            first, end, td.n, td.ldb, td.ldc, td.index_base);
    } else {
        asc_vf_call<SpmmComplexCubeFallbackRows<false>>(
            dim3{kMaxSimtThreadsPerBlock},
            reinterpret_cast<__gm__ const int32_t *>(csrRowOffsets),
            reinterpret_cast<__gm__ const int32_t *>(csrColInd),
            reinterpret_cast<__gm__ const float *>(csrValues),
            reinterpret_cast<__gm__ const float *>(matB),
            reinterpret_cast<__gm__ const float *>(initialC),
            reinterpret_cast<__gm__ float *>(matC), fallbackRows,
            first, end, td.n, td.ldb, td.ldc, td.index_base);
    }
}

__aicore__ inline void LaunchComplexFlatScatter(
    GM_ADDR initialC, GM_ADDR matC, GM_ADDR workspaceGM,
    const SpmmKernelTilingData &td)
{
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
    const __gm__ int32_t *status = reinterpret_cast<__gm__ const int32_t *>(
        workspace + td.cube_complex_status_offset);
    if (status[0] != 0) return;
    const uint64_t scratchPlaneElements = static_cast<uint64_t>(td.cube_group_count) *
        SPMM_CUBE_MIN_ROWS * static_cast<uint32_t>(td.n);
    __gm__ const float *realScratch = reinterpret_cast<__gm__ const float *>(
        workspace + td.cube_scratch_offset);
    __gm__ const float *imagScratch = reinterpret_cast<__gm__ const float *>(
        workspace + td.cube_scratch_offset + scratchPlaneElements * sizeof(float));
    __gm__ const int32_t *slotToRow = reinterpret_cast<__gm__ const int32_t *>(
        workspace + td.cube_row_ids_offset);
    constexpr uint32_t kColumnsPerThread = 4;
    const uint64_t workPerSlot = (static_cast<uint32_t>(td.n) +
        kColumnsPerThread - 1) / kColumnsPerThread;
    const uint64_t total = static_cast<uint64_t>(td.cube_group_count) *
        SPMM_CUBE_MIN_ROWS * workPerSlot;
    const SpmmBlockWorkRange range = GetSpmmBlockWorkRange(total);
    asc_vf_call<SpmmComplexCubeScatterFlat<kColumnsPerThread>>(
        dim3{kMaxSimtThreadsPerBlock}, realScratch, imagScratch, slotToRow,
        reinterpret_cast<__gm__ const float *>(initialC),
        reinterpret_cast<__gm__ float *>(matC), range.first,
        range.end, td.n, td.ldc, td.cube_beta_one);
}

extern "C" __global__ __aicore__ void spmm_custom_complex64_cube_scatter(
    GM_ADDR initialC, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    if (td.cube_rows == SPMM_COMPLEX64_SEGMENTED_CUBE_MODE) {
        KernelSpmmComplex64CubeScatterVector op;
        op.Init(initialC, matC, workspaceGM, tilingGM);
        op.Process();
        return;
    }

    // For moderate output sizes, the UB vector scatter amortizes the complex
    // interleave and GM stores.  Keep the flat SIMT path for very large row
    // counts such as the official P-03 case, where per-row queue setup would
    // dominate the finalization pass.
    if (td.m <= 10000 && td.n <= 2048) {
        KernelSpmmComplex64CubeScatterVector op;
        op.Init(initialC, matC, workspaceGM, tilingGM);
        op.Process();
        return;
    }

    LaunchComplexFlatScatter(initialC, matC, workspaceGM, td);
}
