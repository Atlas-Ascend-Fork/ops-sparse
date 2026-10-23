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

// This entry executes on AICs; Matmul must not start the MIX message service.
#define ASCENDC_CUBE_ONLY
#include "spmm_kernel_device_common.h"

using namespace AscendC;
using namespace matmul;

class KernelSpmmFp32Cube {
public:
    using Fp32Matmul = Matmul<
        MatmulType<TPosition::GM, CubeFormat::ND, float>,
        MatmulType<TPosition::GM, CubeFormat::ND, float>,
        MatmulType<TPosition::GM, CubeFormat::ND, float>>;

    __aicore__ inline void Init(GM_ADDR matB, GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        td_ = LoadSpmmKernelTilingData(tilingGM);
        workspace_ = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        matB_ = reinterpret_cast<__gm__ float *>(matB);
        meta_ = reinterpret_cast<__gm__ const SpmmCubeGroupMeta *>(
            workspace_ + td_.cube_meta_offset);
        packedA_ = reinterpret_cast<__gm__ float *>(workspace_ + td_.cube_packed_a_offset);
        scratch_ = reinterpret_cast<__gm__ float *>(workspace_ + td_.cube_scratch_offset);
        CopySpmmCubeTiling(cubeTiling_, workspace_ + td_.cube_tiling_offset);
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = GetBlockIdx();
        const uint32_t blocks = GetBlockNum();
        const int32_t tileN = td_.cube_tile_n > 0 ? td_.cube_tile_n : SPMM_CUBE_MAX_N;
        const uint32_t tileCount = static_cast<uint32_t>(
            (static_cast<uint32_t>(td_.n) + static_cast<uint32_t>(tileN) - 1u) /
            static_cast<uint32_t>(tileN));
        const uint64_t totalTasks = static_cast<uint64_t>(td_.cube_group_count) * tileCount;
        GlobalTensor<float> aGlobal;
        GlobalTensor<float> bGlobal;
        GlobalTensor<float> cGlobal;
        aGlobal.SetGlobalBuffer(packedA_, static_cast<uint64_t>(td_.cube_group_count) *
            static_cast<uint32_t>(td_.cube_rows) * static_cast<uint32_t>(td_.cube_max_k));
        bGlobal.SetGlobalBuffer(matB_);
        cGlobal.SetGlobalBuffer(scratch_, static_cast<uint64_t>(td_.cube_group_count) *
            static_cast<uint32_t>(td_.cube_rows) * static_cast<uint32_t>(td_.n));

        TPipe pipe;
        Fp32Matmul mm;
        REGIST_MATMUL_OBJ(&pipe, nullptr, mm, &cubeTiling_);
        for (uint64_t task = block; task < totalTasks; task += blocks) {
            const int32_t group = static_cast<int32_t>(task / tileCount);
            const int32_t tile = static_cast<int32_t>(task % tileCount);
            const int32_t column = tile * tileN;
            const int32_t remainingN = td_.n - column;
            const int32_t tailN = remainingN < tileN ? remainingN : tileN;
            if (tailN <= 0) {
                continue;
            }
            const int32_t columnBase = meta_[group].col_base;
            const uint64_t aOffset = static_cast<uint64_t>(group) *
                static_cast<uint32_t>(td_.cube_rows) * static_cast<uint32_t>(td_.cube_max_k);
            const uint64_t bOffset = static_cast<uint64_t>(columnBase) *
                static_cast<uint64_t>(td_.ldb) + column;
            const uint64_t cOffset = static_cast<uint64_t>(group) *
                static_cast<uint32_t>(td_.cube_rows) * static_cast<uint32_t>(td_.n) + column;
            mm.SetOrgShape(td_.cube_rows, td_.n, td_.cube_max_k);
            mm.SetSingleShape(td_.cube_rows, tailN, td_.cube_max_k);
            mm.SetTensorA(aGlobal[aOffset], false);
            mm.SetTensorB(bGlobal[bOffset], false);
            mm.IterateAll(cGlobal[cOffset]);
            mm.End();
        }
    }

private:
    SpmmKernelTilingData td_{};
    TCubeTiling cubeTiling_{};
    __gm__ uint8_t *workspace_{nullptr};
    __gm__ float *matB_{nullptr};
    __gm__ const SpmmCubeGroupMeta *meta_{nullptr};
    __gm__ float *packedA_{nullptr};
    __gm__ float *scratch_{nullptr};
};

constexpr int32_t kFp32CubeScatterColumnsPerThread = 8;

__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmFp32CubeScatter(__gm__ const float *scratch, __gm__ const int32_t *firstSlot,
                    __gm__ const int32_t *secondSlot, __gm__ float *matC,
                    uint64_t first, uint64_t end, int32_t n, int32_t ldc,
                    int32_t betaOne)
{
    for (uint64_t index = first + static_cast<uint64_t>(threadIdx.x);
         index < end; index += static_cast<uint64_t>(blockDim.x)) {
        const int32_t columnTiles =
            (n + kFp32CubeScatterColumnsPerThread - 1) /
            kFp32CubeScatterColumnsPerThread;
        const int32_t row = static_cast<int32_t>(
            index / static_cast<uint64_t>(columnTiles));
        const int32_t column = static_cast<int32_t>(
            index % static_cast<uint64_t>(columnTiles)) *
            kFp32CubeScatterColumnsPerThread;
        const int32_t slot0 = firstSlot[row];
        const int32_t slot1 = secondSlot[row];
        const uint64_t outputBase = static_cast<uint64_t>(row) * ldc + column;
        if (betaOne == 0 && slot0 >= 0 && slot1 < 0 &&
            (column & 1) == 0 && (n & 1) == 0 && (ldc & 1) == 0) {
            const int32_t pairCount =
                (n - column) / 2 < kFp32CubeScatterColumnsPerThread / 2 ?
                    (n - column) / 2 : kFp32CubeScatterColumnsPerThread / 2;
            const uint64_t scratchPairBase =
                (static_cast<uint64_t>(slot0) * n + column) / 2;
            const uint64_t outputPairBase = outputBase / 2;
#pragma unroll
            for (int32_t pair = 0;
                 pair < kFp32CubeScatterColumnsPerThread / 2; ++pair) {
                if (pair >= pairCount) {
                    break;
                }
                reinterpret_cast<__gm__ uint64_t *>(matC)[outputPairBase + pair] =
                    reinterpret_cast<__gm__ const uint64_t *>(scratch)[scratchPairBase + pair];
            }
            continue;
        }
#pragma unroll
        for (int32_t lane = 0; lane < kFp32CubeScatterColumnsPerThread; ++lane) {
            if (column + lane >= n) {
                break;
            }
            float value = betaOne != 0 ? matC[outputBase + lane] : 0.0f;
            if (slot0 >= 0) {
                value += scratch[static_cast<uint64_t>(slot0) * n + column + lane];
            }
            if (slot1 >= 0) {
                value += scratch[static_cast<uint64_t>(slot1) * n + column + lane];
            }
            matC[outputBase + lane] = value;
        }
    }
}

extern "C" __global__ __aicore__ void spmm_custom_fp32_cube(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    if ASCEND_IS_AIV {
        return;
    }
    KernelSpmmFp32Cube op;
    op.Init(matB, workspaceGM, tilingGM);
    op.Process();
}
extern "C" __global__ __aicore__ void spmm_custom_fp32_cube_scatter(
    GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
    __gm__ const float *scratch = reinterpret_cast<__gm__ const float *>(
        workspace + td.cube_scratch_offset);
    __gm__ const int32_t *firstSlot = reinterpret_cast<__gm__ const int32_t *>(
        workspace + td.cube_row_ids_offset);
    __gm__ const int32_t *secondSlot = reinterpret_cast<__gm__ const int32_t *>(
        workspace + td.cube_slot1_offset);
    const uint64_t total = static_cast<uint64_t>(td.m) *
        static_cast<uint32_t>((td.n + kFp32CubeScatterColumnsPerThread - 1) /
                              kFp32CubeScatterColumnsPerThread);
    const uint32_t block = GetBlockIdx();
    const uint32_t blocks = GetBlockNum();
    const uint64_t first = total * block / blocks;
    const uint64_t end = total * (block + 1) / blocks;
    asc_vf_call<SpmmFp32CubeScatter>(dim3{kMaxSimtThreadsPerBlock},
        scratch, firstSlot, secondSlot, reinterpret_cast<__gm__ float *>(matC),
        first, end, td.n, td.ldc, td.cube_beta_one);
}

template <typename InputT, uint32_t TileM, uint32_t TileN>
class KernelSpmmCube {
public:
    __aicore__ inline KernelSpmmCube()
    {
#pragma unroll
        for (uint16_t slot = 0; slot < kBufferCount; ++slot) {
            SetFlag<HardEvent::MTE1_MTE2>(slot);
            SetFlag<HardEvent::MTE1_MTE2>(slot + kBufferCount);
            SetFlag<HardEvent::M_MTE1>(slot + kMte1EventOffset);
            SetFlag<HardEvent::FIX_M>(slot);
        }
    }

    __aicore__ inline void Init(GM_ADDR csrRowOffsets, GM_ADDR csrColInd,
        GM_ADDR csrValues, GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM,
        GM_ADDR tilingGM)
    {
        td_ = LoadSpmmKernelTilingData(tilingGM);
        matB_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(matB));
        __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        meta_ = reinterpret_cast<__gm__ const SpmmCubeGroupMeta *>(workspace + td_.cube_meta_offset);
        packedA_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(
            workspace + td_.cube_packed_a_offset));
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + td_.cube_scratch_offset));
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = GetBlockIdx();
        const uint32_t blocks = GetBlockNum();
        const uint32_t groupCount = static_cast<uint32_t>(td_.cube_group_count);
        if (block < groupCount) {
            uint32_t group = block;
            uint16_t slot = 0;
            uint32_t packedK = CopyGroupToL1(group, slot);
            while (group < groupCount) {
                WaitFlag<HardEvent::MTE2_MTE1>(slot);
                LoadL1ToL0(packedK, slot);
                SetFlag<HardEvent::MTE1_MTE2>(slot);
                SetFlag<HardEvent::MTE1_MTE2>(slot + kBufferCount);

                const uint32_t nextGroup = group + blocks;
                const uint16_t nextSlot = slot ^ 1u;
                uint32_t nextPackedK = 0;
                if (nextGroup < groupCount) {
                    nextPackedK = CopyGroupToL1(nextGroup, nextSlot);
                }

                ComputeAndStore(group, packedK, slot);
                if (nextGroup < groupCount) {
                    group = nextGroup;
                    slot = nextSlot;
                    packedK = nextPackedK;
                } else {
                    group = groupCount;
                }
            }
        }
#pragma unroll
        for (uint16_t slot = 0; slot < kBufferCount; ++slot) {
            WaitFlag<HardEvent::MTE1_MTE2>(slot);
            WaitFlag<HardEvent::MTE1_MTE2>(slot + kBufferCount);
            WaitFlag<HardEvent::M_MTE1>(slot + kMte1EventOffset);
            WaitFlag<HardEvent::FIX_M>(slot);
        }
    }

private:
    static constexpr uint16_t kBufferCount = 2;
    static constexpr uint16_t kMte1EventOffset = 3;
    static constexpr uint32_t kA1Elements = 2 * TileM * kCubeMaxK;
    // LocalTensor offsets are byte offsets. Keep B1 after both A1 slots.
    static constexpr uint32_t kB1OffsetBytes =
        kA1Elements * sizeof(InputT);

    __aicore__ inline void LoadL1ToL0(uint32_t packedK, uint16_t slot)
    {
        WaitFlag<HardEvent::M_MTE1>(slot + kMte1EventOffset);
        LoadData2DParamsV2 aLoad{};
        aLoad.mStep = TileM / 16;
        aLoad.kStep = packedK / 16;
        aLoad.srcStride = aLoad.mStep;
        aLoad.dstStride = aLoad.mStep;
        LoadData(aL0_[slot * TileM * kCubeMaxK],
                 aL1_[slot * TileM * kCubeMaxK], aLoad);

        LoadData2DParamsV2 bLoad{};
        bLoad.mStep = packedK / 16;
        bLoad.kStep = TileN / 16;
        bLoad.srcStride = bLoad.mStep;
        bLoad.dstStride = bLoad.kStep;
        bLoad.ifTranspose = true;
        const uint32_t nTiles = static_cast<uint32_t>(td_.n) / TileN;
        const uint32_t tileElements = packedK * TileN;
        for (uint32_t tile = 0; tile < nTiles; ++tile) {
            LoadData(bL0_[slot * kCubeMaxK * kCubeMaxN + tile * tileElements],
                     bL1_[slot * kCubeMaxK * kCubeMaxN + tile * tileElements], bLoad);
        }
        SetFlag<HardEvent::MTE1_M>(slot);
        WaitFlag<HardEvent::MTE1_M>(slot);
    }

    __aicore__ inline void Compute(uint32_t packedK, uint16_t cSlot)
    {
        MmadParams mm{};
        mm.m = TileM;
        mm.n = TileN;
        mm.k = packedK;
        mm.cmatrixInitVal = true;
        mm.disableGemv = true;
        const uint32_t nTiles = static_cast<uint32_t>(td_.n) / TileN;
        const uint32_t tileElements = packedK * TileN;
        for (uint32_t tile = 0; tile < nTiles; ++tile) {
            Mmad(cL0_[cSlot * TileM * kCubeMaxN + tile * TileM * TileN],
                 aL0_[cSlot * TileM * kCubeMaxK],
                 bL0_[cSlot * kCubeMaxK * kCubeMaxN + tile * tileElements], mm);
        }
        SetFlag<HardEvent::M_MTE1>(cSlot + kMte1EventOffset);
    }

    __aicore__ inline uint32_t CopyGroupToL1(uint32_t group, uint16_t slot)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(slot);
        WaitFlag<HardEvent::MTE1_MTE2>(slot + kBufferCount);
        const int32_t colBase = meta_[group].col_base;
        const int32_t packedK = meta_[group].packed_k;
        Nd2NzParams copyA{};
        copyA.ndNum = 1;
        copyA.nValue = TileM;
        copyA.dValue = packedK;
        copyA.srcDValue = td_.cube_max_k;
        copyA.dstNzC0Stride = TileM;
        copyA.dstNzNStride = 1;
        const uint64_t aOffset = static_cast<uint64_t>(group) * TileM * td_.cube_max_k;
        DataCopy(aL1_[slot * TileM * kCubeMaxK], packedA_[aOffset], copyA);

        Nd2NzParams copyB{};
        copyB.ndNum = 1;
        copyB.nValue = packedK;
        copyB.dValue = td_.n;
        copyB.srcDValue = td_.ldb;
        copyB.dstNzC0Stride = packedK;
        copyB.dstNzNStride = 1;
        DataCopy(bL1_[slot * kCubeMaxK * kCubeMaxN],
                 matB_[static_cast<uint64_t>(colBase) * td_.ldb], copyB);
        SetFlag<HardEvent::MTE2_MTE1>(slot);
        return static_cast<uint32_t>(packedK);
    }

    __aicore__ inline void ComputeAndStore(uint32_t group, uint32_t packedK, uint16_t slot)
    {
        WaitFlag<HardEvent::FIX_M>(slot);
        Compute(packedK, slot);

        SetFlag<HardEvent::M_FIX>(slot);
        WaitFlag<HardEvent::M_FIX>(slot);
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fix{};
        fix.params.ndNum = 1;
        fix.mSize = TileM;
        fix.nSize = td_.n;
        fix.srcStride = TileM;
        fix.dstStride = td_.n;
        fix.quantPre = QuantMode_t::NoQuant;
        Fixpipe<float, float, CFG_ROW_MAJOR>(
            scratch_[static_cast<uint64_t>(group) * TileM * td_.n],
            cL0_[slot * TileM * kCubeMaxN], fix);
        SetFlag<HardEvent::FIX_M>(slot);
    }

    GlobalTensor<InputT> packedA_;
    GlobalTensor<InputT> matB_;
    GlobalTensor<float> scratch_;
    __gm__ const SpmmCubeGroupMeta *meta_{nullptr};
    LocalTensor<InputT> aL1_{TPosition::A1, 0, kA1Elements};
    LocalTensor<InputT> bL1_{TPosition::B1, kB1OffsetBytes,
                             2 * kCubeMaxK * kCubeMaxN};
    LocalTensor<InputT> aL0_{TPosition::A2, 0, 2 * TileM * kCubeMaxK};
    LocalTensor<InputT> bL0_{TPosition::B2, 0, 2 * kCubeMaxK * kCubeMaxN};
    LocalTensor<float> cL0_{TPosition::CO1, 0, 2 * TileM * kCubeMaxN};
    SpmmKernelTilingData td_{};
};

// Complex64 is not accepted by the dav-3510 Cube LoadData/Fixpipe type
// routes.  This kernel mirrors the measured high/low v2 schedule with six
// packed FP16 A planes, four FP16 B planes and two FP32 component accumulators.
// It is selected only by the narrow host gate; generic Complex64 remains the
// SIMT implementation.
template <typename InputT, bool BetaOne>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmRealCubeFallbackRows(
    __gm__ const int32_t *rowOff, __gm__ const int32_t *colInd,
    __gm__ const InputT *values, __gm__ const InputT *matB,
    __gm__ const InputT *initialC, __gm__ InputT *matC,
    __gm__ const int32_t *fallbackRows, uint64_t first, uint64_t end,
    int32_t n, int32_t ldb, int32_t ldc, int32_t indexBase)
{
    constexpr uint32_t kColumnsPerThread = 4;
    const uint32_t columnTiles =
        (static_cast<uint32_t>(n) + kColumnsPerThread - 1u) /
        kColumnsPerThread;
    for (uint64_t work = first + static_cast<uint64_t>(threadIdx.x);
         work < end; work += static_cast<uint64_t>(blockDim.x)) {
        const uint64_t fallbackIndex = work / columnTiles;
        const uint32_t column =
            static_cast<uint32_t>(work % columnTiles) * kColumnsPerThread;
        const int32_t row = fallbackRows[fallbackIndex];
        const int32_t begin = rowOff[row] - indexBase;
        const int32_t rowEnd = rowOff[row + 1] - indexBase;
        float output0 = 0.0f;
        float output1 = 0.0f;
        float output2 = 0.0f;
        float output3 = 0.0f;
        for (int32_t p = begin; p < rowEnd; ++p) {
            const int32_t sparseColumn = colInd[p] - indexBase;
            const __gm__ InputT *b = matB +
                static_cast<uint64_t>(sparseColumn) * ldb + column;
            const float value = static_cast<float>(values[p]);
            output0 += value * static_cast<float>(b[0]);
            output1 += value * static_cast<float>(b[1]);
            output2 += value * static_cast<float>(b[2]);
            output3 += value * static_cast<float>(b[3]);
        }
        const uint64_t outputOffset =
            static_cast<uint64_t>(row) * ldc + column;
        if constexpr (BetaOne) {
            output0 += static_cast<float>(initialC[outputOffset]);
            output1 += static_cast<float>(initialC[outputOffset + 1]);
            output2 += static_cast<float>(initialC[outputOffset + 2]);
            output3 += static_cast<float>(initialC[outputOffset + 3]);
        }
        matC[outputOffset] = static_cast<InputT>(output0);
        matC[outputOffset + 1] = static_cast<InputT>(output1);
        matC[outputOffset + 2] = static_cast<InputT>(output2);
        matC[outputOffset + 3] = static_cast<InputT>(output3);
    }
}

struct SpmmCubeScatterWork {
    uint64_t scratchIndex;
    uint64_t outputIndex;
};

struct SpmmCubeScatterValues {
    float value0;
    float value1;
    float value2;
    float value3;
};

__simt_callee__ __aicore__ inline void GetSpmmCubeScatterWork(
    __gm__ const int32_t *rowIds, uint64_t work, int32_t n, int32_t ldc,
    SpmmCubeScatterWork *scatter)
{
    constexpr uint32_t kColumnsPerThread = 4;
    const uint32_t workPerRow = static_cast<uint32_t>(n) / kColumnsPerThread;
    const uint64_t groupedRow = work / workPerRow;
    const uint32_t column = static_cast<uint32_t>(work % workPerRow) *
        kColumnsPerThread;
    const int32_t row = rowIds[groupedRow];
    scatter->scratchIndex = groupedRow * static_cast<uint32_t>(n) + column;
    scatter->outputIndex = static_cast<uint64_t>(row) * ldc + column;
}

__simt_callee__ __aicore__ inline void LoadSpmmCubeScatterInput(
    __gm__ const float *scratch, __gm__ const int32_t *rowIds, uint64_t work,
    int32_t n, int32_t ldc, SpmmCubeScatterWork *scatter,
    SpmmCubeScatterValues *values)
{
    GetSpmmCubeScatterWork(rowIds, work, n, ldc, scatter);
    values->value0 = scratch[scatter->scratchIndex];
    values->value1 = scratch[scatter->scratchIndex + 1];
    values->value2 = scratch[scatter->scratchIndex + 2];
    values->value3 = scratch[scatter->scratchIndex + 3];
}

__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmCubeScatterFp16(__gm__ const float *scratch, __gm__ const int32_t *rowIds,
                    __gm__ half *matC, uint64_t first, uint64_t end,
                    int32_t n, int32_t ldc, int32_t betaOne)
{
    for (uint64_t work = first + static_cast<uint64_t>(threadIdx.x);
         work < end; work += static_cast<uint64_t>(blockDim.x)) {
        SpmmCubeScatterWork scatter{};
        SpmmCubeScatterValues values{};
        LoadSpmmCubeScatterInput(scratch, rowIds, work, n, ldc, &scatter,
            &values);
        if (betaOne != 0) {
            const half2 old01 = *reinterpret_cast<__gm__ const half2 *>(
                matC + scatter.outputIndex);
            const half2 old23 = *reinterpret_cast<__gm__ const half2 *>(
                matC + scatter.outputIndex + 2);
            values.value0 += __low2float(old01);
            values.value1 += __high2float(old01);
            values.value2 += __low2float(old23);
            values.value3 += __high2float(old23);
        }
        *reinterpret_cast<__gm__ half2 *>(matC + scatter.outputIndex) =
            __floats2half2_rn(values.value0, values.value1);
        *reinterpret_cast<__gm__ half2 *>(matC + scatter.outputIndex + 2) =
            __floats2half2_rn(values.value2, values.value3);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmCubeScatterBf16(__gm__ const float *scratch, __gm__ const int32_t *rowIds,
                    __gm__ bfloat16_t *matC, uint64_t first, uint64_t end,
                    int32_t n, int32_t ldc, int32_t betaOne)
{
    for (uint64_t work = first + static_cast<uint64_t>(threadIdx.x);
         work < end; work += static_cast<uint64_t>(blockDim.x)) {
        SpmmCubeScatterWork scatter{};
        SpmmCubeScatterValues values{};
        LoadSpmmCubeScatterInput(scratch, rowIds, work, n, ldc, &scatter,
            &values);
        if (betaOne != 0) {
            values.value0 += static_cast<float>(matC[scatter.outputIndex]);
            values.value1 += static_cast<float>(matC[scatter.outputIndex + 1]);
            values.value2 += static_cast<float>(matC[scatter.outputIndex + 2]);
            values.value3 += static_cast<float>(matC[scatter.outputIndex + 3]);
        }
        matC[scatter.outputIndex] = static_cast<bfloat16_t>(values.value0);
        matC[scatter.outputIndex + 1] = static_cast<bfloat16_t>(values.value1);
        matC[scatter.outputIndex + 2] = static_cast<bfloat16_t>(values.value2);
        matC[scatter.outputIndex + 3] = static_cast<bfloat16_t>(values.value3);
    }
}

// spmm_custom_fp16_cube and spmm_custom_bf16_cube only differ by InputT; the
// tiling-driven KernelSpmmCube instantiation is shared here so the two
// extern "C" kernel entries stay thin, matching LaunchRealCubeScatter and
// LaunchRealCubeFallbackEntry below.
template <typename InputT>
__aicore__ inline void LaunchRealCube(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    if (td.cube_rows == SPMM_CUBE_ROWS && td.n == 128) {
        KernelSpmmCube<InputT, SPMM_CUBE_ROWS, 128> op;
        op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
        op.Process();
    } else if (td.cube_rows == SPMM_CUBE_MIN_ROWS && td.n == 128) {
        KernelSpmmCube<InputT, SPMM_CUBE_MIN_ROWS, 128> op;
        op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
        op.Process();
    } else if (td.cube_rows == SPMM_CUBE_MIN_ROWS) {
        KernelSpmmCube<InputT, SPMM_CUBE_MIN_ROWS, 16> op;
        op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
        op.Process();
    } else {
        KernelSpmmCube<InputT, SPMM_CUBE_ROWS, 16> op;
        op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void spmm_custom_fp16_cube(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    LaunchRealCube<half>(csrRowOffsets, csrColInd, csrValues, matB, matC,
        workspaceGM, tilingGM);
}

struct SpmmRealCubeScatterArgs {
    GM_ADDR csrRowOffsets;
    GM_ADDR csrColInd;
    GM_ADDR csrValues;
    GM_ADDR matB;
    GM_ADDR initialC;
    GM_ADDR matC;
    GM_ADDR workspaceGM;
    GM_ADDR tilingGM;
};

template <typename InputT>
__aicore__ inline void LaunchRealCubeFallback(
    const SpmmRealCubeScatterArgs &args, const SpmmKernelTilingData &td,
    __gm__ const int32_t *fallbackRows, uint64_t fallbackTotal,
    uint64_t fallbackFirst, uint64_t fallbackEnd, uint32_t blocks)
{
    if (fallbackTotal == 0) return;
    const uint64_t maxWorkPerBlock = (fallbackTotal + blocks - 1u) / blocks;
    const uint32_t fallbackThreads = static_cast<uint32_t>(
        maxWorkPerBlock < kMaxSimtThreadsPerBlock ? maxWorkPerBlock :
        kMaxSimtThreadsPerBlock);
    if (td.cube_beta_one != 0) {
        asc_vf_call<SpmmRealCubeFallbackRows<InputT, true>>(dim3{fallbackThreads},
            reinterpret_cast<__gm__ const int32_t *>(args.csrRowOffsets),
            reinterpret_cast<__gm__ const int32_t *>(args.csrColInd),
            reinterpret_cast<__gm__ const InputT *>(args.csrValues),
            reinterpret_cast<__gm__ const InputT *>(args.matB),
            reinterpret_cast<__gm__ const InputT *>(args.initialC),
            reinterpret_cast<__gm__ InputT *>(args.matC), fallbackRows,
            fallbackFirst, fallbackEnd, td.n, td.ldb, td.ldc, td.index_base);
    } else {
        asc_vf_call<SpmmRealCubeFallbackRows<InputT, false>>(dim3{fallbackThreads},
            reinterpret_cast<__gm__ const int32_t *>(args.csrRowOffsets),
            reinterpret_cast<__gm__ const int32_t *>(args.csrColInd),
            reinterpret_cast<__gm__ const InputT *>(args.csrValues),
            reinterpret_cast<__gm__ const InputT *>(args.matB),
            reinterpret_cast<__gm__ const InputT *>(args.initialC),
            reinterpret_cast<__gm__ InputT *>(args.matC), fallbackRows,
            fallbackFirst, fallbackEnd, td.n, td.ldb, td.ldc, td.index_base);
    }
}

template <typename OutputT>
__aicore__ inline void LaunchRealCubeScatter(const SpmmRealCubeScatterArgs &args)
{
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(args.tilingGM);
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(args.workspaceGM);
    __gm__ const float *scratch = reinterpret_cast<__gm__ const float *>(
        workspace + td.cube_scratch_offset);
    __gm__ const int32_t *rowIds = reinterpret_cast<__gm__ const int32_t *>(
        workspace + td.cube_row_ids_offset);
    __gm__ const int32_t *fallbackRows = reinterpret_cast<__gm__ const int32_t *>(
        workspace + td.cube_slot1_offset);
    constexpr uint32_t kColumnsPerThread = 4;
    const uint32_t mappedRows = td.cube_complex_fallback_count < td.m ?
        static_cast<uint32_t>(td.m - td.cube_complex_fallback_count) : 0;
    const uint64_t scatterTotal = static_cast<uint64_t>(mappedRows) *
        (static_cast<uint32_t>(td.n) / kColumnsPerThread);
    const uint64_t fallbackTotal = static_cast<uint64_t>(td.cube_complex_fallback_count) *
        (static_cast<uint32_t>(td.n) / kColumnsPerThread);
    const uint32_t block = GetBlockIdx();
    const uint32_t blocks = GetBlockNum();
    const uint64_t scatterFirst = scatterTotal * block / blocks;
    const uint64_t scatterEnd = scatterTotal * (block + 1) / blocks;
    if constexpr (SpmmIsFp16<OutputT>::value) {
        asc_vf_call<SpmmCubeScatterFp16>(dim3{kMaxSimtThreadsPerBlock}, scratch,
            rowIds, reinterpret_cast<__gm__ half *>(args.matC), scatterFirst,
            scatterEnd, td.n, td.ldc, td.cube_beta_one);
    } else {
        asc_vf_call<SpmmCubeScatterBf16>(dim3{kMaxSimtThreadsPerBlock}, scratch,
            rowIds, reinterpret_cast<__gm__ bfloat16_t *>(args.matC), scatterFirst,
            scatterEnd, td.n, td.ldc, td.cube_beta_one);
    }
    const uint64_t fallbackFirst = fallbackTotal * block / blocks;
    const uint64_t fallbackEnd = fallbackTotal * (block + 1) / blocks;
    LaunchRealCubeFallback<OutputT>(args, td, fallbackRows, fallbackTotal,
        fallbackFirst, fallbackEnd, blocks);
}

extern "C" __global__ __aicore__ void spmm_custom_fp16_cube_scatter(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    LaunchRealCubeScatter<half>({csrRowOffsets, csrColInd, csrValues, matB,
        initialC, matC, workspaceGM, tilingGM});
}
extern "C" __global__ __aicore__ void spmm_custom_bf16_cube_scatter(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    LaunchRealCubeScatter<bfloat16_t>({csrRowOffsets, csrColInd, csrValues,
        matB, initialC, matC, workspaceGM, tilingGM});
}

template<typename InputT, bool BetaOne>
__aicore__ inline void LaunchRealCubeFallbackEntryRows(
    const SpmmRealCubeScatterArgs &args, const SpmmKernelTilingData &td,
    __gm__ const int32_t *fallbackRows, uint64_t first, uint64_t end)
{
    asc_vf_call<SpmmRealCubeFallbackRows<InputT, BetaOne>>(
        dim3{kMaxSimtThreadsPerBlock},
        reinterpret_cast<__gm__ const int32_t *>(args.csrRowOffsets),
        reinterpret_cast<__gm__ const int32_t *>(args.csrColInd),
        reinterpret_cast<__gm__ const InputT *>(args.csrValues),
        reinterpret_cast<__gm__ const InputT *>(args.matB),
        reinterpret_cast<__gm__ const InputT *>(args.initialC),
        reinterpret_cast<__gm__ InputT *>(args.matC), fallbackRows, first, end,
        td.n, td.ldb, td.ldc, td.index_base);
}

template<typename InputT>
__aicore__ inline void LaunchRealCubeFallbackEntry(
    const SpmmRealCubeScatterArgs &args)
{
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(args.tilingGM);
    if (td.cube_complex_fallback_count <= 0) {
        return;
    }
    __gm__ uint8_t *workspace =
        reinterpret_cast<__gm__ uint8_t *>(args.workspaceGM);
    const __gm__ int32_t *fallbackRows =
        reinterpret_cast<__gm__ const int32_t *>(workspace + td.cube_slot1_offset);
    const uint64_t total = static_cast<uint64_t>(td.cube_complex_fallback_count) *
        ((static_cast<uint32_t>(td.n) + 3u) / 4u);
    const SpmmBlockWorkRange range = GetSpmmBlockWorkRange(total);
    if (td.cube_beta_one != 0) {
        LaunchRealCubeFallbackEntryRows<InputT, true>(
            args, td, fallbackRows, range.first, range.end);
    } else {
        LaunchRealCubeFallbackEntryRows<InputT, false>(
            args, td, fallbackRows, range.first, range.end);
    }
}

extern "C" __global__ __aicore__ void spmm_custom_fp16_cube_fallback(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    LaunchRealCubeFallbackEntry<half>({csrRowOffsets, csrColInd, csrValues,
        matB, initialC, matC, workspaceGM, tilingGM});
}
extern "C" __global__ __aicore__ void spmm_custom_bf16_cube_fallback(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    LaunchRealCubeFallbackEntry<bfloat16_t>({csrRowOffsets, csrColInd,
        csrValues, matB, initialC, matC, workspaceGM, tilingGM});
}
extern "C" __global__ __aicore__ void spmm_custom_bf16_cube(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    LaunchRealCube<bfloat16_t>(csrRowOffsets, csrColInd, csrValues, matB, matC,
        workspaceGM, tilingGM);
}
template <typename InputT>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmRefreshPackedValues(__gm__ const InputT *values, __gm__ const uint32_t *valueMap,
                        __gm__ InputT *packedA, uint64_t first, uint64_t end)
{
    for (uint64_t position = first + static_cast<uint64_t>(threadIdx.x);
         position < end; position += static_cast<uint64_t>(blockDim.x)) {
        const uint32_t packed = valueMap[position];
        if (packed != 0xffffffffu) packedA[packed] = values[position];
    }
}

template <typename InputT>
__aicore__ inline void LaunchPackedValueRefresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
    const uint64_t first = td.cube_a_value_nnz * GetBlockIdx() / GetBlockNum();
    const uint64_t end = td.cube_a_value_nnz * (GetBlockIdx() + 1) / GetBlockNum();
    asc_vf_call<SpmmRefreshPackedValues<InputT>>(dim3{kMaxSimtThreadsPerBlock},
        reinterpret_cast<__gm__ const InputT *>(csrValues),
        reinterpret_cast<__gm__ const uint32_t *>(workspace + td.cube_a_value_map_offset),
        reinterpret_cast<__gm__ InputT *>(workspace + td.cube_packed_a_offset), first, end);
}

extern "C" __global__ __aicore__ void spmm_custom_fp32_cube_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    LaunchPackedValueRefresh<float>(csrValues, workspaceGM, tilingGM);
}

extern "C" __global__ __aicore__ void spmm_custom_fp16_cube_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    LaunchPackedValueRefresh<half>(csrValues, workspaceGM, tilingGM);
}

extern "C" __global__ __aicore__ void spmm_custom_bf16_cube_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    LaunchPackedValueRefresh<bfloat16_t>(csrValues, workspaceGM, tilingGM);
}
