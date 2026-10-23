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

#include "spmm_kernel_regbase_common.h"

using namespace AscendC;
using namespace matmul;

namespace {

struct SpmmRegBaseTileContext {
    int32_t begin;
    int32_t elements;
    bool wrap;
    int32_t highStart;
    int32_t lowRows;
    int32_t highRows;
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
using SpmmRegComplex64 = complex64;
#else
using SpmmRegComplex64 = uint64_t;
#endif

} // namespace

#define SPMM_COMPLEX_PREPARE_COLUMN \
        const uint32_t remaining = count - column; \
        uint32_t maskCount = 32; \
        if (remaining < maskCount) { \
            maskCount = remaining; \
        } \
        Reg::MaskReg mask = \
            Reg::UpdateMask<SpmmRegComplex64, Reg::RegTraitNumTwo>( \
                maskCount); \
        Reg::LoadAlign(sum, accumulatorComplex + column); \
        Reg::LoadAlign( \
            dense, windowComplex + localStart * TileN + column)

template<int32_t TileN>
__simd_vf__ inline void SpmmComplex64Degree2RowMac(
    float value0Real, float value0Imag,
    float value1Real, float value1Imag,
    __ubuf__ float *accumulator, __ubuf__ float *window,
    int32_t localStart, uint16_t count)
{
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> value0;
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> value1;
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> dense;
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> product0;
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> product1;
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> sum;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    const SpmmRegComplex64 scalar0(value0Real, value0Imag);
    const SpmmRegComplex64 scalar1(value1Real, value1Imag);
#else
    const SpmmRegComplex64 scalar0 = 0;
    const SpmmRegComplex64 scalar1 = 0;
#endif
    auto accumulatorComplex =
        reinterpret_cast<__ubuf__ SpmmRegComplex64 *>(accumulator);
    auto windowComplex =
        reinterpret_cast<__ubuf__ SpmmRegComplex64 *>(window);
    uint32_t fullMaskCount = 32;
    Reg::MaskReg fullMask =
        Reg::UpdateMask<SpmmRegComplex64, Reg::RegTraitNumTwo>(
            fullMaskCount);
    Reg::Duplicate(value0, scalar0, fullMask);
    Reg::Duplicate(value1, scalar1, fullMask);
    for (uint16_t column = 0; column < count; column += 32) {
        SPMM_COMPLEX_PREPARE_COLUMN;
        Reg::Mul(product0, value0, dense, mask);
        Reg::Add(sum, sum, product0, mask);
        Reg::LoadAlign(
            dense, windowComplex + (localStart + 1) * TileN + column);
        Reg::Mul(product1, value1, dense, mask);
        Reg::Add(sum, sum, product1, mask);
        Reg::StoreAlign(accumulatorComplex + column, sum, mask);
    }
}

#define SPMM_COMPLEX_ARGUMENTS_4 \
    float v0r, float v0i, float v1r, float v1i, \
    float v2r, float v2i, float v3r, float v3i

#define SPMM_COMPLEX_ARGUMENTS_8 \
    SPMM_COMPLEX_ARGUMENTS_4, \
    float v4r, float v4i, float v5r, float v5i, \
    float v6r, float v6i, float v7r, float v7i

#define SPMM_COMPLEX_DECLARE_VALUE(ID) \
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> value##ID

#define SPMM_COMPLEX_DECLARE_VALUES_4 \
    SPMM_COMPLEX_DECLARE_VALUE(0); \
    SPMM_COMPLEX_DECLARE_VALUE(1); \
    SPMM_COMPLEX_DECLARE_VALUE(2); \
    SPMM_COMPLEX_DECLARE_VALUE(3)

#define SPMM_COMPLEX_DECLARE_VALUES_8 \
    SPMM_COMPLEX_DECLARE_VALUES_4; \
    SPMM_COMPLEX_DECLARE_VALUE(4); \
    SPMM_COMPLEX_DECLARE_VALUE(5); \
    SPMM_COMPLEX_DECLARE_VALUE(6); \
    SPMM_COMPLEX_DECLARE_VALUE(7)

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define SPMM_COMPLEX_DECLARE_SCALAR(ID) \
    const SpmmRegComplex64 scalar##ID(v##ID##r, v##ID##i)
#else
#define SPMM_COMPLEX_DECLARE_SCALAR(ID) \
    const SpmmRegComplex64 scalar##ID = 0
#endif

#define SPMM_COMPLEX_DECLARE_SCALARS_4 \
    SPMM_COMPLEX_DECLARE_SCALAR(0); \
    SPMM_COMPLEX_DECLARE_SCALAR(1); \
    SPMM_COMPLEX_DECLARE_SCALAR(2); \
    SPMM_COMPLEX_DECLARE_SCALAR(3)

#define SPMM_COMPLEX_DECLARE_SCALARS_8 \
    SPMM_COMPLEX_DECLARE_SCALARS_4; \
    SPMM_COMPLEX_DECLARE_SCALAR(4); \
    SPMM_COMPLEX_DECLARE_SCALAR(5); \
    SPMM_COMPLEX_DECLARE_SCALAR(6); \
    SPMM_COMPLEX_DECLARE_SCALAR(7)

#define SPMM_COMPLEX_DUPLICATE_VALUE(ID) \
    Reg::Duplicate(value##ID, scalar##ID, fullMask)

#define SPMM_COMPLEX_DUPLICATE_VALUES_4 \
    SPMM_COMPLEX_DUPLICATE_VALUE(0); \
    SPMM_COMPLEX_DUPLICATE_VALUE(1); \
    SPMM_COMPLEX_DUPLICATE_VALUE(2); \
    SPMM_COMPLEX_DUPLICATE_VALUE(3)

#define SPMM_COMPLEX_DUPLICATE_VALUES_8 \
    SPMM_COMPLEX_DUPLICATE_VALUES_4; \
    SPMM_COMPLEX_DUPLICATE_VALUE(4); \
    SPMM_COMPLEX_DUPLICATE_VALUE(5); \
    SPMM_COMPLEX_DUPLICATE_VALUE(6); \
    SPMM_COMPLEX_DUPLICATE_VALUE(7)

#define SPMM_COMPLEX_DECLARE_COMPONENT_STATE \
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> dense; \
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> sum; \
    auto &denseReal = (Reg::RegTensor<float> &)dense.reg[0]; \
    auto &denseImag = (Reg::RegTensor<float> &)dense.reg[1]; \
    auto &sumReal = (Reg::RegTensor<float> &)sum.reg[0]; \
    auto &sumImag = (Reg::RegTensor<float> &)sum.reg[1]; \
    Reg::RegTensor<float> realPositive; \
    Reg::RegTensor<float> realNegative; \
    Reg::RegTensor<float> imagFromReal; \
    Reg::RegTensor<float> imagFromImag

template<int32_t TileN>
__simd_vf__ inline void SpmmComplex64Degree4RowMac(
    SPMM_COMPLEX_ARGUMENTS_4,
    __ubuf__ float *accumulator, __ubuf__ float *window,
    int32_t localStart, uint16_t count)
{
    SPMM_COMPLEX_DECLARE_VALUES_4;
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> dense;
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> product;
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> sum;
    SPMM_COMPLEX_DECLARE_SCALARS_4;
    auto accumulatorComplex =
        reinterpret_cast<__ubuf__ SpmmRegComplex64 *>(accumulator);
    auto windowComplex =
        reinterpret_cast<__ubuf__ SpmmRegComplex64 *>(window);
    uint32_t fullMaskCount = 32;
    Reg::MaskReg fullMask =
        Reg::UpdateMask<SpmmRegComplex64, Reg::RegTraitNumTwo>(
            fullMaskCount);
    SPMM_COMPLEX_DUPLICATE_VALUES_4;
    for (uint16_t column = 0; column < count; column += 32) {
        SPMM_COMPLEX_PREPARE_COLUMN;
        Reg::Mul(product, value0, dense, mask);
        Reg::Add(sum, sum, product, mask);
        Reg::LoadAlign(
            dense, windowComplex + (localStart + 1) * TileN + column);
        Reg::Mul(product, value1, dense, mask);
        Reg::Add(sum, sum, product, mask);
        Reg::LoadAlign(
            dense, windowComplex + (localStart + 2) * TileN + column);
        Reg::Mul(product, value2, dense, mask);
        Reg::Add(sum, sum, product, mask);
        Reg::LoadAlign(
            dense, windowComplex + (localStart + 3) * TileN + column);
        Reg::Mul(product, value3, dense, mask);
        Reg::Add(sum, sum, product, mask);
        Reg::StoreAlign(accumulatorComplex + column, sum, mask);
    }
}

template<int32_t TileN>
__simd_vf__ inline void SpmmComplex64Degree8RowMac(
    SPMM_COMPLEX_ARGUMENTS_8,
    __ubuf__ float *accumulator, __ubuf__ float *window,
    int32_t localStart, uint16_t count)
{
    SPMM_COMPLEX_DECLARE_VALUES_8;
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> dense;
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> product;
    Reg::RegTensor<SpmmRegComplex64, Reg::RegTraitNumTwo> sum;
    SPMM_COMPLEX_DECLARE_SCALARS_8;
    auto accumulatorComplex =
        reinterpret_cast<__ubuf__ SpmmRegComplex64 *>(accumulator);
    auto windowComplex =
        reinterpret_cast<__ubuf__ SpmmRegComplex64 *>(window);
    uint32_t fullMaskCount = 32;
    Reg::MaskReg fullMask =
        Reg::UpdateMask<SpmmRegComplex64, Reg::RegTraitNumTwo>(
            fullMaskCount);
    SPMM_COMPLEX_DUPLICATE_VALUES_8;
    for (uint16_t column = 0; column < count; column += 32) {
        SPMM_COMPLEX_PREPARE_COLUMN;
#define SPMM_COMPLEX_ACCUMULATE(ID) \
        Reg::LoadAlign( \
            dense, windowComplex + \
                (localStart + (ID)) * TileN + column); \
        Reg::Mul(product, value##ID, dense, mask); \
        Reg::Add(sum, sum, product, mask)
        Reg::LoadAlign(
            dense, windowComplex + localStart * TileN + column);
        Reg::Mul(product, value0, dense, mask);
        Reg::Add(sum, sum, product, mask);
        SPMM_COMPLEX_ACCUMULATE(1);
        SPMM_COMPLEX_ACCUMULATE(2);
        SPMM_COMPLEX_ACCUMULATE(3);
        SPMM_COMPLEX_ACCUMULATE(4);
        SPMM_COMPLEX_ACCUMULATE(5);
        SPMM_COMPLEX_ACCUMULATE(6);
        SPMM_COMPLEX_ACCUMULATE(7);
        Reg::StoreAlign(accumulatorComplex + column, sum, mask);
#undef SPMM_COMPLEX_ACCUMULATE
    }
}

template<int32_t TileN>
__simd_vf__ inline void SpmmComplex64Degree8ComponentIlpRowMac(
    SPMM_COMPLEX_ARGUMENTS_8,
    __ubuf__ float *accumulator, __ubuf__ float *window,
    int32_t localStart, uint16_t count)
{
    SPMM_COMPLEX_DECLARE_COMPONENT_STATE;
    auto accumulatorComplex =
        reinterpret_cast<__ubuf__ SpmmRegComplex64 *>(accumulator);
    auto windowComplex =
        reinterpret_cast<__ubuf__ SpmmRegComplex64 *>(window);

#define SPMM_COMPLEX_COMPONENT_ACCUMULATE(ID) \
        Reg::LoadAlign( \
            dense, windowComplex + \
                (localStart + (ID)) * TileN + column); \
        Reg::Axpy(realPositive, denseReal, v##ID##r, mask); \
        Reg::Axpy(realNegative, denseImag, v##ID##i, mask); \
        Reg::Axpy(imagFromReal, denseImag, v##ID##r, mask); \
        Reg::Axpy(imagFromImag, denseReal, v##ID##i, mask)
    for (uint16_t column = 0; column < count; column += 32) {
        const uint32_t remaining = count - column;
        uint32_t maskCount = 32;
        if (remaining < maskCount) {
            maskCount = remaining;
        }
        Reg::MaskReg mask =
            Reg::UpdateMask<SpmmRegComplex64, Reg::RegTraitNumTwo>(
                maskCount);
        Reg::Duplicate(realPositive, 0.0f, mask);
        Reg::Duplicate(realNegative, 0.0f, mask);
        Reg::Duplicate(imagFromReal, 0.0f, mask);
        Reg::Duplicate(imagFromImag, 0.0f, mask);
        SPMM_COMPLEX_COMPONENT_ACCUMULATE(0);
        SPMM_COMPLEX_COMPONENT_ACCUMULATE(1);
        SPMM_COMPLEX_COMPONENT_ACCUMULATE(2);
        SPMM_COMPLEX_COMPONENT_ACCUMULATE(3);
        SPMM_COMPLEX_COMPONENT_ACCUMULATE(4);
        SPMM_COMPLEX_COMPONENT_ACCUMULATE(5);
        SPMM_COMPLEX_COMPONENT_ACCUMULATE(6);
        SPMM_COMPLEX_COMPONENT_ACCUMULATE(7);
        Reg::LoadAlign(sum, accumulatorComplex + column);
        Reg::Add(sumReal, sumReal, realPositive, mask);
        Reg::Sub(sumReal, sumReal, realNegative, mask);
        Reg::Add(sumImag, sumImag, imagFromReal, mask);
        Reg::Add(sumImag, sumImag, imagFromImag, mask);
        Reg::StoreAlign(accumulatorComplex + column, sum, mask);
    }
#undef SPMM_COMPLEX_COMPONENT_ACCUMULATE
}

#undef SPMM_COMPLEX_ARGUMENTS_4
#undef SPMM_COMPLEX_ARGUMENTS_8
#undef SPMM_COMPLEX_DECLARE_VALUE
#undef SPMM_COMPLEX_DECLARE_VALUES_4
#undef SPMM_COMPLEX_DECLARE_VALUES_8
#undef SPMM_COMPLEX_DECLARE_SCALAR
#undef SPMM_COMPLEX_DECLARE_SCALARS_4
#undef SPMM_COMPLEX_DECLARE_SCALARS_8
#undef SPMM_COMPLEX_DUPLICATE_VALUE
#undef SPMM_COMPLEX_DUPLICATE_VALUES_4
#undef SPMM_COMPLEX_DUPLICATE_VALUES_8
#undef SPMM_COMPLEX_DECLARE_COMPONENT_STATE
#undef SPMM_COMPLEX_PREPARE_COLUMN

template<int32_t Degree>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmComplex64RegBaseRefresh(
    __gm__ const int32_t *rowMeta, __gm__ const uint64_t *values,
    __gm__ uint64_t *packedValues, uint64_t first, uint64_t end)
{
    for (uint64_t work = first + static_cast<uint64_t>(threadIdx.x);
         work < end; work += static_cast<uint64_t>(blockDim.x)) {
        const uint64_t row = work / Degree;
        const uint64_t ordinal = work % Degree;
        const int32_t valueOffset = rowMeta[row * 2 + 1];
        packedValues[work] = values[valueOffset + ordinal];
    }
}

template<int32_t Degree, int32_t Rows, int32_t TileN,
         int32_t WindowRows, int32_t OutputRows, int32_t OutputDepth>
class KernelSpmmComplex64RegBaseCompute : public SpmmRegBaseTensorState {
public:
    __aicore__ inline void Init(
        GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
        GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        InitializeSpmmRegBaseTensors(matB, initialC, matC, workspaceGM,
            tilingGM, &td_, matB_, initialC_, matC_, &batches_, rowIds_,
            rowMeta_, packedValues_);
        tileCount_ = (td_.n + TileN - 1) / TileN;

        pipe_.InitBuffer(windowQueue_, 1,
            WindowRows * TileN *
                SPMM_COMPLEX64_COMPONENTS * sizeof(float));
        pipe_.InitBuffer(accumulatorQueue_, OutputDepth,
            OutputRows * TileN *
                SPMM_COMPLEX64_COMPONENTS * sizeof(float));
        pipe_.InitBuffer(valueBuffer_,
            Rows * Degree *
                SPMM_COMPLEX64_COMPONENTS * sizeof(float));
        pipe_.InitBuffer(rowMetaBuffer_,
            Rows * 2 * sizeof(int32_t));
        pipe_.InitBuffer(rowIdsBuffer_,
            Rows * sizeof(int32_t));
        metadataReady_ = static_cast<event_t>(
            pipe_.FetchEventID(HardEvent::MTE2_S));
        valuesReady_ = static_cast<event_t>(
            pipe_.FetchEventID(HardEvent::MTE2_S));
        accumulatorVectorToMte3_ = static_cast<event_t>(
            pipe_.FetchEventID(HardEvent::V_MTE3));
        for (int32_t slot = 0; slot < OutputDepth; ++slot) {
            accumulatorDone_[slot] = static_cast<event_t>(
                pipe_.FetchEventID(HardEvent::MTE3_V));
            accumulatorPending_[slot] = false;
        }
        nextAccumulatorSlot_ = 0;
    }

    __aicore__ inline void Process()
    {
        if (td_.uniform_degree != Degree) {
            return;
        }
        const int32_t block = static_cast<int32_t>(GetBlockIdx());
        const int32_t blocks = static_cast<int32_t>(GetBlockNum());
        const int32_t totalTasks = td_.cube_group_count * tileCount_;
        for (int32_t task = block; task < totalTasks; task += blocks) {
            ProcessTile(task / tileCount_, task % tileCount_);
        }
        DrainAccumulators();
    }

private:
    static_assert(Degree == SPMM_COMPLEX64_DEGREE2_DEGREE ||
                  Degree == SPMM_COMPLEX64_DEGREE4_DEGREE ||
                  Degree == SPMM_COMPLEX64_DEGREE8_DEGREE ||
                  Degree == SPMM_COMPLEX64_DEGREE64_DEGREE,
                  "unsupported Complex64 RegBase degree");
    static_assert(OutputDepth == 1 || OutputDepth == 2,
                  "unsupported Complex64 output depth");

    __aicore__ inline void WaitAccumulator(uint32_t slot)
    {
        if (!accumulatorPending_[slot]) {
            return;
        }
        WaitFlag<HardEvent::MTE3_V>(accumulatorDone_[slot]);
        accumulatorQueue_.FreeTensor(pendingAccumulator_[slot]);
        accumulatorPending_[slot] = false;
    }

    __aicore__ inline void DrainAccumulators()
    {
        for (int32_t slot = 0; slot < OutputDepth; ++slot) {
            WaitAccumulator(static_cast<uint32_t>(slot));
        }
    }

    __aicore__ inline void ComputeDegree2Row(
        __ubuf__ float *accumulator, __ubuf__ float *window,
        __ubuf__ float *values, int32_t valueOffset,
        int32_t localStart, uint16_t count)
    {
        asc_vf_call<SpmmComplex64Degree2RowMac<TileN>>(
            values[valueOffset], values[valueOffset + 1],
            values[valueOffset + 2], values[valueOffset + 3], accumulator,
            window, localStart, count);
    }

    __aicore__ inline void ComputeDegree4Row(
        __ubuf__ float *accumulator, __ubuf__ float *window,
        __ubuf__ float *values, int32_t valueOffset,
        int32_t localStart, uint16_t count)
    {
        asc_vf_call<SpmmComplex64Degree4RowMac<TileN>>(
            values[valueOffset], values[valueOffset + 1],
            values[valueOffset + 2], values[valueOffset + 3],
            values[valueOffset + 4], values[valueOffset + 5],
            values[valueOffset + 6], values[valueOffset + 7], accumulator,
            window, localStart, count);
    }

    __aicore__ inline void ComputeDegree8Row(
        __ubuf__ float *accumulator, __ubuf__ float *window,
        __ubuf__ float *values, int32_t valueOffset,
        int32_t localStart, uint16_t count)
    {
        asc_vf_call<SpmmComplex64Degree8RowMac<TileN>>(
            values[valueOffset], values[valueOffset + 1],
            values[valueOffset + 2], values[valueOffset + 3],
            values[valueOffset + 4], values[valueOffset + 5],
            values[valueOffset + 6], values[valueOffset + 7],
            values[valueOffset + 8], values[valueOffset + 9],
            values[valueOffset + 10], values[valueOffset + 11],
            values[valueOffset + 12], values[valueOffset + 13],
            values[valueOffset + 14], values[valueOffset + 15], accumulator,
            window, localStart, count);
    }

    __aicore__ inline void ComputeDegree64Row(
        __ubuf__ float *accumulator, __ubuf__ float *window,
        __ubuf__ float *values, int32_t valueOffset,
        int32_t localStart, uint16_t count)
    {
        for (int32_t group = 0; group < SPMM_COMPLEX64_DEGREE64_DEGREE /
             SPMM_COMPLEX64_DEGREE8_DEGREE; ++group) {
            const int32_t groupOffset = valueOffset + group *
                SPMM_COMPLEX64_DEGREE8_DEGREE * SPMM_COMPLEX64_COMPONENTS;
            asc_vf_call<SpmmComplex64Degree8ComponentIlpRowMac<TileN>>(
                values[groupOffset], values[groupOffset + 1],
                values[groupOffset + 2], values[groupOffset + 3],
                values[groupOffset + 4], values[groupOffset + 5],
                values[groupOffset + 6], values[groupOffset + 7],
                values[groupOffset + 8], values[groupOffset + 9],
                values[groupOffset + 10], values[groupOffset + 11],
                values[groupOffset + 12], values[groupOffset + 13],
                values[groupOffset + 14], values[groupOffset + 15], accumulator,
                window, localStart + group * SPMM_COMPLEX64_DEGREE8_DEGREE,
                count);
        }
    }

    __aicore__ inline void ComputeRow(
        __ubuf__ float *accumulator, __ubuf__ float *window,
        __ubuf__ float *values, int32_t valueOffset,
        int32_t localStart, uint16_t count)
    {
        if constexpr (Degree == SPMM_COMPLEX64_DEGREE2_DEGREE) {
            ComputeDegree2Row(accumulator, window, values, valueOffset,
                localStart, count);
        } else if constexpr (Degree == SPMM_COMPLEX64_DEGREE4_DEGREE) {
            ComputeDegree4Row(accumulator, window, values, valueOffset,
                localStart, count);
        } else if constexpr (Degree == SPMM_COMPLEX64_DEGREE8_DEGREE) {
            ComputeDegree8Row(accumulator, window, values, valueOffset,
                localStart, count);
        } else {
            ComputeDegree64Row(accumulator, window, values, valueOffset,
                localStart, count);
        }
    }

    __aicore__ inline bool IsValidBatch(
        const SpmmFp32Degree2BatchMeta &batch, bool wrapBatch,
        int32_t wrapHighStart, int32_t wrapLowRows, int32_t wrapHighRows)
    {
        return (!wrapBatch || (wrapHighStart > 0 && wrapLowRows > 0 &&
                               wrapHighRows > 0)) &&
            batch.row_count > 0 && batch.row_count <= Rows &&
            batch.row_meta_begin >= 0 && batch.window_rows > 0 &&
            (wrapBatch || batch.window_rows <= WindowRows);
    }

    __aicore__ inline void LoadTileMetadata(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<int32_t> &rowMeta,
        LocalTensor<int32_t> &rowIds)
    {
        DataCopyPadExtParams<int32_t> noPad{false, 0, 0, 0};
        DataCopyExtParams metaCopy{1, static_cast<uint32_t>(
            batch.row_count * 2 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(rowMeta,
            rowMeta_[static_cast<uint64_t>(batch.row_meta_begin) * 2],
            metaCopy, noPad);
        DataCopyExtParams idCopy{1, static_cast<uint32_t>(
            batch.row_count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(rowIds, rowIds_[batch.row_meta_begin], idCopy, noPad);
        SetFlag<HardEvent::MTE2_S>(metadataReady_);
    }

    __aicore__ inline void LoadTileValues(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<float> &packedValues,
        const DataCopyPadExtParams<float> &noPad)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(batch.row_count *
            Degree * SPMM_COMPLEX64_COMPONENTS * sizeof(float)), 0, 0, 0};
        DataCopyPad(packedValues,
            packedValues_[static_cast<uint64_t>(batch.row_meta_begin) * Degree *
                          SPMM_COMPLEX64_COMPONENTS], copy, noPad);
        SetFlag<HardEvent::MTE2_S>(valuesReady_);
    }

    __aicore__ inline void QueueTileWindow(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<float> &window,
        int32_t tileBegin, int32_t tileElements, bool wrapBatch,
        int32_t wrapHighStart, int32_t wrapLowRows, int32_t wrapHighRows,
        const DataCopyPadExtParams<float> &noPad)
    {
        const uint32_t bytes = static_cast<uint32_t>(tileElements *
            SPMM_COMPLEX64_COMPONENTS * sizeof(float));
        const uint32_t padded = (bytes + 31U) / 32U * 32U;
        const int64_t sourceGap = static_cast<int64_t>((td_.ldb - tileElements) *
            SPMM_COMPLEX64_COMPONENTS * sizeof(float));
        const int64_t destinationGap = static_cast<int64_t>((TileN *
            SPMM_COMPLEX64_COMPONENTS * sizeof(float) - padded) / 32U);
        if (wrapBatch) {
            DataCopyExtParams low{static_cast<uint16_t>(wrapLowRows), bytes,
                sourceGap, destinationGap, 0};
            DataCopyPad(window,
                matB_[static_cast<uint64_t>(tileBegin) *
                    SPMM_COMPLEX64_COMPONENTS], low, noPad);
            DataCopyExtParams high{static_cast<uint16_t>(wrapHighRows), bytes,
                sourceGap, destinationGap, 0};
            DataCopyPad(window[wrapLowRows * TileN * SPMM_COMPLEX64_COMPONENTS],
                matB_[(static_cast<uint64_t>(wrapHighStart) * td_.ldb +
                    tileBegin) * SPMM_COMPLEX64_COMPONENTS], high, noPad);
        } else {
            DataCopyExtParams copy{static_cast<uint16_t>(batch.window_rows),
                bytes, sourceGap, destinationGap, 0};
            DataCopyPad(window,
                matB_[(static_cast<uint64_t>(batch.window_begin) * td_.ldb +
                    tileBegin) * SPMM_COMPLEX64_COMPONENTS], copy, noPad);
        }
        windowQueue_.EnQue(window);
    }

    __aicore__ inline void QueueAccumulatorInput(
        LocalTensor<int32_t> &rowIds,
        LocalTensor<float> &accumulator, int32_t rowBegin, int32_t rowEnd,
        int32_t tileBegin, int32_t tileElements,
        const DataCopyPadExtParams<float> &noPad)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(tileElements *
            SPMM_COMPLEX64_COMPONENTS * sizeof(float)), 0, 0, 0};
        for (int32_t row = rowBegin; row < rowEnd; ++row) {
            const int32_t outputRow = rowIds.GetValue(row);
            DataCopyPad(accumulator[(row - rowBegin) * TileN *
                    SPMM_COMPLEX64_COMPONENTS],
                initialC_[(static_cast<uint64_t>(outputRow) * td_.ldc +
                    tileBegin) * SPMM_COMPLEX64_COMPONENTS], copy, noPad);
        }
        accumulatorQueue_.EnQue(accumulator);
    }

    __aicore__ inline void ComputeOutputRows(
        LocalTensor<int32_t> &rowMeta, LocalTensor<float> &accumulator,
        __ubuf__ float *windowAddress, __ubuf__ float *valueAddress,
        int32_t rowBegin, int32_t rowEnd, int32_t tileElements)
    {
        __ubuf__ float *accumulatorAddress =
            reinterpret_cast<__ubuf__ float *>(accumulator.GetPhyAddr());
        for (int32_t row = rowBegin; row < rowEnd; ++row) {
            const int32_t valueOffset = row * Degree * SPMM_COMPLEX64_COMPONENTS;
            ComputeRow(accumulatorAddress + (row - rowBegin) * TileN *
                SPMM_COMPLEX64_COMPONENTS, windowAddress, valueAddress,
                valueOffset, rowMeta.GetValue(row * 2),
                static_cast<uint16_t>(tileElements));
        }
    }

    __aicore__ inline void StoreOutputRows(
        LocalTensor<int32_t> &rowIds, LocalTensor<float> &accumulator,
        int32_t rowBegin, int32_t rowEnd, int32_t tileBegin,
        int32_t tileElements)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(tileElements *
            SPMM_COMPLEX64_COMPONENTS * sizeof(float)), 0, 0, 0};
        for (int32_t row = rowBegin; row < rowEnd; ++row) {
            const int32_t outputRow = rowIds.GetValue(row);
            DataCopyPad(matC_[(static_cast<uint64_t>(outputRow) * td_.ldc +
                tileBegin) * SPMM_COMPLEX64_COMPONENTS],
                accumulator[(row - rowBegin) * TileN *
                    SPMM_COMPLEX64_COMPONENTS], copy);
        }
    }

    __aicore__ inline void ProcessOutputRows(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<int32_t> &rowMeta,
        LocalTensor<int32_t> &rowIds, __ubuf__ float *windowAddress,
        __ubuf__ float *valueAddress, int32_t tileBegin, int32_t tileElements)
    {
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
        for (int32_t rowBegin = 0; rowBegin < batch.row_count;
             rowBegin += OutputRows) {
            const uint32_t slot = nextAccumulatorSlot_;
            WaitAccumulator(slot);
            const int32_t rowEnd = rowBegin + OutputRows < batch.row_count ?
                rowBegin + OutputRows : batch.row_count;
            LocalTensor<float> accumulator =
                accumulatorQueue_.template AllocTensor<float>();
            QueueAccumulatorInput(rowIds, accumulator, rowBegin, rowEnd,
                tileBegin, tileElements, noPad);
            accumulator = accumulatorQueue_.template DeQue<float>();
            ComputeOutputRows(rowMeta, accumulator, windowAddress, valueAddress,
                rowBegin, rowEnd, tileElements);
            SetFlag<HardEvent::V_MTE3>(accumulatorVectorToMte3_);
            WaitFlag<HardEvent::V_MTE3>(accumulatorVectorToMte3_);
            StoreOutputRows(rowIds, accumulator, rowBegin, rowEnd, tileBegin,
                tileElements);
            SetFlag<HardEvent::MTE3_V>(accumulatorDone_[slot]);
            pendingAccumulator_[slot] = accumulator;
            accumulatorPending_[slot] = true;
            nextAccumulatorSlot_ = OutputDepth == 2 ? 1U - slot : 0;
        }
    }

    __aicore__ inline void ProcessValidatedTile(
        const SpmmFp32Degree2BatchMeta &batch,
        const SpmmRegBaseTileContext &tile)
    {
        LocalTensor<int32_t> rowMeta = rowMetaBuffer_.Get<int32_t>();
        LocalTensor<int32_t> rowIds = rowIdsBuffer_.Get<int32_t>();
        LoadTileMetadata(batch, rowMeta, rowIds);
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
        LocalTensor<float> packedValues = valueBuffer_.Get<float>();
        LoadTileValues(batch, packedValues, noPad);
        LocalTensor<float> window = windowQueue_.AllocTensor<float>();
        QueueTileWindow(batch, window, tile.begin, tile.elements, tile.wrap,
            tile.highStart, tile.lowRows, tile.highRows, noPad);
        window = windowQueue_.DeQue<float>();
        WaitFlag<HardEvent::MTE2_S>(metadataReady_);
        WaitFlag<HardEvent::MTE2_S>(valuesReady_);
        ProcessOutputRows(batch, rowMeta, rowIds,
            reinterpret_cast<__ubuf__ float *>(window.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(packedValues.GetPhyAddr()),
            tile.begin, tile.elements);
        windowQueue_.FreeTensor(window);
    }

    __aicore__ inline void ProcessTile(
        int32_t batchIndex, int32_t tileIndex)
    {
        const SpmmFp32Degree2BatchMeta batch =
            LoadSpmmRegBaseBatch(batches_, batchIndex);
        const bool wrapBatch = batch.window_begin < 0;
        const int32_t wrapHighStart =
            wrapBatch ? -batch.window_begin - 1 : 0;
        const int32_t wrapLowRows = wrapBatch ? batch.window_rows : 0;
        const int32_t wrapHighRows = wrapBatch ? Degree - wrapLowRows : 0;
        if (!IsValidBatch(batch, wrapBatch, wrapHighStart, wrapLowRows,
                          wrapHighRows)) {
            return;
        }
        const int32_t tileBegin =
            tileIndex * TileN;
        const int32_t tileElements = td_.n - tileBegin <
            TileN ? td_.n - tileBegin : TileN;
        if (tileElements <= 0) {
            return;
        }

        ProcessValidatedTile(batch, {tileBegin, tileElements, wrapBatch,
            wrapHighStart, wrapLowRows, wrapHighRows});
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> windowQueue_;
    TQue<TPosition::VECIN, OutputDepth> accumulatorQueue_;
    TBuf<TPosition::VECCALC> valueBuffer_;
    TBuf<TPosition::VECCALC> rowMetaBuffer_;
    TBuf<TPosition::VECCALC> rowIdsBuffer_;
    event_t metadataReady_;
    event_t valuesReady_;
    event_t accumulatorVectorToMte3_;
    event_t accumulatorDone_[OutputDepth];
    LocalTensor<float> pendingAccumulator_[OutputDepth];
    bool accumulatorPending_[OutputDepth];
    uint32_t nextAccumulatorSlot_{0};
};

template<int32_t Degree>
__aicore__ inline void LaunchComplex64RegBaseRefresh(
    __gm__ const int32_t *rowMeta, GM_ADDR csrValues,
    __gm__ uint64_t *packedValues, int32_t refreshRunCount)
{
    const uint64_t total = static_cast<uint64_t>(refreshRunCount) * Degree;
    const SpmmBlockWorkRange range = GetSpmmBlockWorkRange(total);
    asc_vf_call<SpmmComplex64RegBaseRefresh<Degree>>(
        dim3{kMaxSimtThreadsPerBlock}, rowMeta,
        reinterpret_cast<__gm__ const uint64_t *>(csrValues), packedValues,
        range.first, range.end);
}

extern "C" __global__ __aicore__ void spmm_custom_complex64_regbase_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    __gm__ const SpmmFp32RegBaseHeader *header = nullptr;
    __gm__ const int32_t *rowMeta = nullptr;
    __gm__ uint64_t *packedValues = nullptr;
    const SpmmKernelTilingData td = InitializeSpmmRegBaseRefreshData(
        workspaceGM, tilingGM, &header, &rowMeta, &packedValues);
    if (td.uniform_degree == SPMM_COMPLEX64_DEGREE2_DEGREE) {
        LaunchComplex64RegBaseRefresh<SPMM_COMPLEX64_DEGREE2_DEGREE>(
            rowMeta, csrValues, packedValues, header->refresh_run_count);
    } else if (td.uniform_degree == SPMM_COMPLEX64_DEGREE4_DEGREE) {
        LaunchComplex64RegBaseRefresh<SPMM_COMPLEX64_DEGREE4_DEGREE>(
            rowMeta, csrValues, packedValues, header->refresh_run_count);
    } else if (td.uniform_degree == SPMM_COMPLEX64_DEGREE8_DEGREE) {
        LaunchComplex64RegBaseRefresh<SPMM_COMPLEX64_DEGREE8_DEGREE>(
            rowMeta, csrValues, packedValues, header->refresh_run_count);
    } else if (td.uniform_degree == SPMM_COMPLEX64_DEGREE64_DEGREE) {
        LaunchComplex64RegBaseRefresh<SPMM_COMPLEX64_DEGREE64_DEGREE>(
            rowMeta, csrValues, packedValues, header->refresh_run_count);
    }
}
extern "C" __global__ __aicore__ void spmm_custom_complex64_regbase_compute(
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    if (td.uniform_degree == SPMM_COMPLEX64_DEGREE2_DEGREE) {
        KernelSpmmComplex64RegBaseCompute<
            SPMM_COMPLEX64_DEGREE2_DEGREE, SPMM_COMPLEX64_DEGREE2_ROWS,
            SPMM_COMPLEX64_DEGREE2_TILE_N,
            SPMM_COMPLEX64_DEGREE2_WINDOW_ROWS,
            SPMM_COMPLEX64_DEGREE2_OUTPUT_ROWS,
            SPMM_COMPLEX64_DEGREE2_OUTPUT_DEPTH> op;
        op.Init(matB, initialC, matC, workspaceGM, tilingGM);
        op.Process();
    } else if (td.uniform_degree == SPMM_COMPLEX64_DEGREE4_DEGREE) {
        KernelSpmmComplex64RegBaseCompute<
            SPMM_COMPLEX64_DEGREE4_DEGREE, SPMM_COMPLEX64_DEGREE4_ROWS,
            SPMM_COMPLEX64_DEGREE4_TILE_N,
            SPMM_COMPLEX64_DEGREE4_WINDOW_ROWS,
            SPMM_COMPLEX64_DEGREE4_OUTPUT_ROWS,
            SPMM_COMPLEX64_DEGREE4_OUTPUT_DEPTH> op;
        op.Init(matB, initialC, matC, workspaceGM, tilingGM);
        op.Process();
    } else if (td.uniform_degree == SPMM_COMPLEX64_DEGREE8_DEGREE) {
        KernelSpmmComplex64RegBaseCompute<
            SPMM_COMPLEX64_DEGREE8_DEGREE, SPMM_COMPLEX64_DEGREE8_ROWS,
            SPMM_COMPLEX64_DEGREE8_TILE_N,
            SPMM_COMPLEX64_DEGREE8_WINDOW_ROWS,
            SPMM_COMPLEX64_DEGREE8_OUTPUT_ROWS,
            SPMM_COMPLEX64_DEGREE8_OUTPUT_DEPTH> op;
        op.Init(matB, initialC, matC, workspaceGM, tilingGM);
        op.Process();
    } else if (td.uniform_degree == SPMM_COMPLEX64_DEGREE64_DEGREE) {
        KernelSpmmComplex64RegBaseCompute<
            SPMM_COMPLEX64_DEGREE64_DEGREE,
            SPMM_COMPLEX64_DEGREE64_ROWS,
            SPMM_COMPLEX64_DEGREE64_TILE_N,
            SPMM_COMPLEX64_DEGREE64_WINDOW_ROWS,
            SPMM_COMPLEX64_DEGREE64_OUTPUT_ROWS,
            SPMM_COMPLEX64_DEGREE64_OUTPUT_DEPTH> op;
        op.Init(matB, initialC, matC, workspaceGM, tilingGM);
        op.Process();
    }
}
