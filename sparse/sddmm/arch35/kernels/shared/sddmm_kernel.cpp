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
 * \file sddmm_kernel.cpp
 * \brief Ascend 950 SDDMM pattern validation and generic CSR/BSR SIMT kernels.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "../bsr/sddmm_bsr8_vector_kernel.h"
#include "../csr/sddmm_column_vector_kernel.h"
#include "../../sddmm_kernel.h"
#include "sddmm_simt_convert.h"
#include "sddmm_pattern_validator.h"
#include "../csr/sddmm_packed_vector_kernel.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
#error "SDDMM arch35 kernel is only available on DAV-3510 / Ascend 950."
#endif

namespace {

struct SddmmSimtBuffers {
    __gm__ const int32_t *offsets;
    __gm__ const int32_t *columns;
    __gm__ uint8_t *sparseValues;
    __gm__ const uint8_t *matX;
    __gm__ const uint8_t *matY;
};

struct SddmmComplexValue {
    float real;
    float imag;
};

using sddmm_arch35::RecoverSddmmSortedCyclic64Start;
using sddmm_arch35::SddmmPatternValidator;
using sddmm_arch35::SddmmFloatToHalf;
using sddmm_arch35::SddmmFloatToBf16;
using sddmm_arch35::SddmmBf16ToFloat;

template <int32_t ValueType>
static __simt_callee__ __aicore__ inline float SddmmLoadReal(
    __gm__ const uint8_t *values, int64_t index)
{
    if constexpr (ValueType == SDDMM_VALUE_FP32) {
        return reinterpret_cast<__gm__ const float *>(values)[index];
    } else if constexpr (ValueType == SDDMM_VALUE_FP16) {
        return static_cast<float>(reinterpret_cast<__gm__ const __fp16 *>(values)[index]);
    } else {
        return SddmmBf16ToFloat(reinterpret_cast<__gm__ const uint16_t *>(values)[index]);
    }
}

template <int32_t ValueType>
static __simt_callee__ __aicore__ inline void SddmmStoreReal(
    __gm__ uint8_t *values, int64_t index, float value)
{
    if constexpr (ValueType == SDDMM_VALUE_FP32) {
        reinterpret_cast<__gm__ float *>(values)[index] = value;
    } else if constexpr (ValueType == SDDMM_VALUE_FP16) {
        reinterpret_cast<__gm__ uint16_t *>(values)[index] = SddmmFloatToHalf(value);
    } else {
        reinterpret_cast<__gm__ uint16_t *>(values)[index] = SddmmFloatToBf16(value);
    }
}

static __simt_callee__ __aicore__ inline SddmmComplexValue SddmmLoadComplex(
    __gm__ const uint8_t *values, int64_t index)
{
    __gm__ const float *parts = reinterpret_cast<__gm__ const float *>(values) + index * 2;
    return SddmmComplexValue{parts[0], parts[1]};
}

static __simt_callee__ __aicore__ inline void SddmmStoreComplex(
    __gm__ uint8_t *values, int64_t index, const SddmmComplexValue &value)
{
    __gm__ float *parts = reinterpret_cast<__gm__ float *>(values) + index * 2;
    parts[0] = value.real;
    parts[1] = value.imag;
}

static __simt_callee__ __aicore__ inline int64_t SddmmFindMajor(
    __gm__ const int32_t *offsets, int64_t majorCount, int64_t sparseIndex, int32_t indexBase)
{
    int64_t low = 0;
    int64_t high = majorCount;
    while (low < high) {
        const int64_t middle = low + (high - low) / 2;
        const int64_t end = static_cast<int64_t>(offsets[middle + 1]) - indexBase;
        if (end <= sparseIndex) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

static __simt_callee__ __aicore__ inline void SddmmResolveOutput(
    __gm__ const int32_t *offsets,
    __gm__ const int32_t *columns,
    const SddmmTilingData &tiling,
    int64_t batch,
    int64_t localOutput,
    int64_t &row,
    int64_t &column,
    int64_t &valueIndex)
{
    const int64_t offsetsBase = tiling.offsetsBatchStride == 0 ? 0 : batch * tiling.offsetsBatchStride;
    const int64_t columnsBase = tiling.columnsBatchStride == 0 ? 0 : batch * tiling.columnsBatchStride;
    const int64_t valuesBase = batch * tiling.valuesBatchStride;
    __gm__ const int32_t *batchOffsets = offsets + offsetsBase;
    if (tiling.format == SDDMM_FORMAT_CSR) {
        row = SddmmFindMajor(batchOffsets, tiling.m, localOutput, tiling.indexBase);
        column = static_cast<int64_t>(columns[columnsBase + localOutput]) - tiling.indexBase;
        valueIndex = valuesBase + localOutput;
        return;
    }

    const int64_t blockArea = static_cast<int64_t>(tiling.blockDim) * tiling.blockDim;
    const int64_t blockSlot = localOutput / blockArea;
    const int64_t inner = localOutput - blockSlot * blockArea;
    const int64_t blockRow = SddmmFindMajor(batchOffsets, tiling.blockRows, blockSlot, tiling.indexBase);
    const int64_t blockColumn = static_cast<int64_t>(columns[columnsBase + blockSlot]) - tiling.indexBase;
    int64_t innerRow = 0;
    int64_t innerColumn = 0;
    if (tiling.blockOrder == 0) {
        innerRow = inner / tiling.blockDim;
        innerColumn = inner - innerRow * tiling.blockDim;
    } else {
        innerColumn = inner / tiling.blockDim;
        innerRow = inner - innerColumn * tiling.blockDim;
    }
    row = blockRow * tiling.blockDim + innerRow;
    column = blockColumn * tiling.blockDim + innerColumn;
    valueIndex = valuesBase + localOutput;
}

static __simt_callee__ __aicore__ inline void SddmmResolveLogicalOutput(
    __gm__ const int32_t *offsets, __gm__ const int32_t *columns,
    const SddmmTilingData &tiling, uint64_t output, int64_t &batch,
    int64_t &row, int64_t &column, int64_t &valueIndex)
{
    batch = static_cast<int64_t>(output / static_cast<uint64_t>(tiling.outputsPerBatch));
    int64_t localOutput = static_cast<int64_t>(output % static_cast<uint64_t>(tiling.outputsPerBatch));
    // Adjacent lanes sample adjacent logical columns even when the BSR
    // values are stored column-major. ResolveOutput still supplies the
    // physical value index, so the public block order is preserved.
    if (tiling.format == SDDMM_FORMAT_BSR && tiling.blockOrder != 0) {
        const int64_t area = static_cast<int64_t>(tiling.blockDim) * tiling.blockDim;
        const int64_t inner = localOutput % area;
        localOutput = localOutput - inner +
            (inner % tiling.blockDim) * tiling.blockDim + inner / tiling.blockDim;
    }
    SddmmResolveOutput(offsets, columns, tiling, batch, localOutput, row, column, valueIndex);
}

template <int32_t InputType, int32_t OutputType>
__simt_vf__ __aicore__ __launch_bounds__(kSddmmThreadsPerBlock) inline void SddmmRealSimtCompute(
    SddmmSimtBuffers buffers,
    SddmmTilingData tiling,
    float alpha,
    float beta,
    int32_t threadsPerCore)
{
    const uint64_t totalOutputs = static_cast<uint64_t>(tiling.outputsPerBatch) *
        static_cast<uint64_t>(tiling.batchCount);
    const uint64_t globalThread = static_cast<uint64_t>(threadIdx.x) +
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(threadsPerCore);
    const uint64_t gridStride = static_cast<uint64_t>(threadsPerCore) *
        static_cast<uint64_t>(tiling.numBlocks);
    for (uint64_t output = globalThread; output < totalOutputs; output += gridStride) {
        int64_t batch = 0;
        int64_t row = 0;
        int64_t column = 0;
        int64_t valueIndex = 0;
        SddmmResolveLogicalOutput(buffers.offsets, buffers.columns, tiling, output,
            batch, row, column, valueIndex);

        float accumulator = 0.0F;
        if (alpha != 0.0F) {
            const int64_t batchX = tiling.batchCountX == 1 ? 0 : batch;
            const int64_t batchY = tiling.batchCountY == 1 ? 0 : batch;
            const int64_t xBase = batchX * tiling.xBatchStride;
            const int64_t yBase = tiling.usePackedY != 0 ?
                batchY * tiling.packedYBatchStride : batchY * tiling.yBatchStride;
            const bool xReductionContiguous = (tiling.opX == 0) == (tiling.orderX == 0);
            const bool yReductionContiguous = tiling.usePackedY != 0 ||
                ((tiling.opY == 0) != (tiling.orderY == 0));
            const int64_t xStep = xReductionContiguous ? 1 : tiling.ldX;
            const int64_t yStep = yReductionContiguous ? 1 : tiling.ldY;
            const int64_t xStart = xBase + row * (xReductionContiguous ? tiling.ldX : 1);
            const int64_t yStart = yBase + column * (tiling.usePackedY != 0 ? tiling.k :
                (yReductionContiguous ? tiling.ldY : 1));
#pragma unroll 4
            for (int64_t reduction = 0; reduction < tiling.k; ++reduction) {
                const int64_t xIndex = xStart + reduction * xStep;
                const int64_t yIndex = yStart + reduction * yStep;
                const float xValue = SddmmLoadReal<InputType>(buffers.matX, xIndex);
                const float yValue = SddmmLoadReal<InputType>(buffers.matY, yIndex);
                accumulator = __fmaf_rn(xValue, yValue, accumulator);
            }
        }
        const float oldValue = beta == 0.0F ? 0.0F : SddmmLoadReal<OutputType>(buffers.sparseValues, valueIndex);
        const float outputValue = __fmaf_rn(alpha, accumulator, beta * oldValue);
        SddmmStoreReal<OutputType>(buffers.sparseValues, valueIndex, outputValue);
    }
}

// CSR fast path for the dominant sampled_addmm layout:
//   X[m, k] row-major, Y[k, n] row-major, opX = opY = N.
//
// Eight SIMT lanes cooperate on one CSR row.  Each lane keeps up to eight
// output accumulators, so one load of X[row, reduction] feeds eight sampled
// columns instead of being reloaded once per nonzero.  Mapping work from CSR
// rows also removes the per-output binary search used by the fully generic
// CSR/BSR fallback.  A 64-entry tile matches the task performance shapes while
// the outer tile loop retains correct behavior for arbitrary row lengths.
constexpr int32_t kSddmmCsrThreadsPerRow = 8;
constexpr int32_t kSddmmCsrOutputsPerThread = 8;
constexpr int32_t kSddmmCsrOutputsPerTile =
    kSddmmCsrThreadsPerRow * kSddmmCsrOutputsPerThread;

struct SddmmCsrRowState {
    int64_t columnsBase;
    int64_t valuesBase;
    int64_t rowStart;
    int64_t rowEnd;
    int64_t xRowBase;
    int64_t yBase;
};

static __simt_callee__ __aicore__ inline SddmmCsrRowState SddmmCsrResolveRow(
    SddmmSimtBuffers buffers, const SddmmTilingData &tiling, int64_t linearRow)
{
    SddmmCsrRowState state;
    const int64_t batch = linearRow / tiling.m;
    const int64_t row = linearRow - batch * tiling.m;
    const int64_t offsetsBase = tiling.offsetsBatchStride == 0 ?
        0 : batch * tiling.offsetsBatchStride;
    state.columnsBase = tiling.columnsBatchStride == 0 ?
        0 : batch * tiling.columnsBatchStride;
    state.valuesBase = batch * tiling.valuesBatchStride;
    state.rowStart = static_cast<int64_t>(buffers.offsets[offsetsBase + row]) - tiling.indexBase;
    state.rowEnd = static_cast<int64_t>(buffers.offsets[offsetsBase + row + 1]) - tiling.indexBase;
    const int64_t batchX = tiling.batchCountX == 1 ? 0 : batch;
    const int64_t batchY = tiling.batchCountY == 1 ? 0 : batch;
    state.xRowBase = batchX * tiling.xBatchStride + row * tiling.ldX;
    state.yBase = tiling.usePackedY != 0 ?
        batchY * tiling.packedYBatchStride : batchY * tiling.yBatchStride;
    return state;
}

template <int32_t OutputType>
static __simt_callee__ __aicore__ inline void SddmmCsrPackedRow(
    SddmmSimtBuffers buffers, const SddmmTilingData &tiling,
    const SddmmCsrRowState &state, int32_t lane, float alpha, float beta)
{
    int32_t packedBase4[kSddmmCsrOutputsPerThread];
    float accumulator[kSddmmCsrOutputsPerThread];
#pragma unroll
    for (int32_t output = 0; output < kSddmmCsrOutputsPerThread; ++output) {
        const int64_t sparseIndex = state.rowStart + lane +
            static_cast<int64_t>(output) * kSddmmCsrThreadsPerRow;
        const int32_t column = buffers.columns[state.columnsBase + sparseIndex] - tiling.indexBase;
        packedBase4[output] = column * 32;
        accumulator[output] = 0.0F;
    }

    if (alpha != 0.0F) {
        __gm__ const float4 *xValues =
            reinterpret_cast<__gm__ const float4 *>(buffers.matX) + state.xRowBase / 4;
        __gm__ const float4 *packedValues =
            reinterpret_cast<__gm__ const float4 *>(buffers.matY) + state.yBase / 4;
#pragma unroll 2
        for (int32_t chunk = 0; chunk < 32; ++chunk) {
            const float4 x = xValues[chunk];
#pragma unroll
            for (int32_t output = 0; output < kSddmmCsrOutputsPerThread; ++output) {
                const float4 y = packedValues[packedBase4[output] + chunk];
                accumulator[output] = __fmaf_rn(x.x, y.x, accumulator[output]);
                accumulator[output] = __fmaf_rn(x.y, y.y, accumulator[output]);
                accumulator[output] = __fmaf_rn(x.z, y.z, accumulator[output]);
                accumulator[output] = __fmaf_rn(x.w, y.w, accumulator[output]);
            }
        }
    }

#pragma unroll
    for (int32_t output = 0; output < kSddmmCsrOutputsPerThread; ++output) {
        const int64_t sparseIndex = state.rowStart + lane +
            static_cast<int64_t>(output) * kSddmmCsrThreadsPerRow;
        const int64_t valueIndex = state.valuesBase + sparseIndex;
        const float oldValue = beta == 0.0F ? 0.0F :
            SddmmLoadReal<OutputType>(buffers.sparseValues, valueIndex);
        const float result = __fmaf_rn(alpha, accumulator[output], beta * oldValue);
        SddmmStoreReal<OutputType>(buffers.sparseValues, valueIndex, result);
    }
}

template <int32_t InputType>
static __simt_callee__ __aicore__ inline void SddmmCsrAccumulateTile(
    SddmmSimtBuffers buffers, const SddmmTilingData &tiling,
    const SddmmCsrRowState &state, const int32_t *column,
    const bool *valid, float *accumulator)
{
    for (int64_t reduction = 0; reduction < tiling.k; ++reduction) {
        const float xValue = SddmmLoadReal<InputType>(buffers.matX, state.xRowBase + reduction);
#pragma unroll
        for (int32_t output = 0; output < kSddmmCsrOutputsPerThread; ++output) {
            if (valid[output]) {
                const int64_t yIndex = tiling.usePackedY != 0 ?
                    state.yBase + static_cast<int64_t>(column[output]) * tiling.k + reduction :
                    state.yBase + reduction * tiling.ldY + column[output];
                const float yValue = SddmmLoadReal<InputType>(buffers.matY, yIndex);
                accumulator[output] = __fmaf_rn(xValue, yValue, accumulator[output]);
            }
        }
    }
}

template <int32_t InputType, int32_t OutputType>
static __simt_callee__ __aicore__ inline void SddmmCsrGeneralRow(
    SddmmSimtBuffers buffers, const SddmmTilingData &tiling,
    const SddmmCsrRowState &state, int32_t lane, float alpha, float beta)
{
    for (int64_t tileStart = state.rowStart; tileStart < state.rowEnd;
         tileStart += kSddmmCsrOutputsPerTile) {
        int64_t sparseIndex[kSddmmCsrOutputsPerThread];
        int32_t column[kSddmmCsrOutputsPerThread];
        bool valid[kSddmmCsrOutputsPerThread];
        float accumulator[kSddmmCsrOutputsPerThread];
#pragma unroll
        for (int32_t output = 0; output < kSddmmCsrOutputsPerThread; ++output) {
            const int64_t index = tileStart + lane +
                static_cast<int64_t>(output) * kSddmmCsrThreadsPerRow;
            sparseIndex[output] = index;
            valid[output] = index < state.rowEnd;
            column[output] = valid[output] ?
                buffers.columns[state.columnsBase + index] - tiling.indexBase : 0;
            accumulator[output] = 0.0F;
        }

        if (alpha != 0.0F) {
            SddmmCsrAccumulateTile<InputType>(buffers, tiling, state,
                column, valid, accumulator);
        }

#pragma unroll
        for (int32_t output = 0; output < kSddmmCsrOutputsPerThread; ++output) {
            if (valid[output]) {
                const int64_t valueIndex = state.valuesBase + sparseIndex[output];
                const float oldValue = beta == 0.0F ? 0.0F :
                    SddmmLoadReal<OutputType>(buffers.sparseValues, valueIndex);
                const float result = __fmaf_rn(alpha, accumulator[output], beta * oldValue);
                SddmmStoreReal<OutputType>(buffers.sparseValues, valueIndex, result);
            }
        }
    }
}

template <int32_t InputType, int32_t OutputType>
__simt_vf__ __aicore__ __launch_bounds__(kSddmmThreadsPerBlock) inline void
SddmmCsrRowTileSimtCompute(
    SddmmSimtBuffers buffers,
    SddmmTilingData tiling,
    float alpha,
    float beta,
    int32_t threadsPerCore)
{
    const int32_t localThread = static_cast<int32_t>(threadIdx.x);
    const int32_t localRowGroup = localThread / kSddmmCsrThreadsPerRow;
    const int32_t lane = localThread % kSddmmCsrThreadsPerRow;
    const int32_t rowGroupsPerCore = threadsPerCore / kSddmmCsrThreadsPerRow;
    if (rowGroupsPerCore <= 0) {
        return;
    }

    const int64_t totalRows = tiling.batchCount * tiling.m;
    int64_t linearRow = static_cast<int64_t>(blockIdx.x) * rowGroupsPerCore + localRowGroup;
    const int64_t rowStride = static_cast<int64_t>(tiling.numBlocks) * rowGroupsPerCore;
    for (; linearRow < totalRows; linearRow += rowStride) {
        const SddmmCsrRowState state = SddmmCsrResolveRow(buffers, tiling, linearRow);
        // Packed FP32 rows retain the ordered float4 FMA fast path.
        if constexpr (InputType == SDDMM_VALUE_FP32 && OutputType == SDDMM_VALUE_FP32) {
            if (tiling.usePackedY != 0 && tiling.k == 128 &&
                state.rowEnd - state.rowStart == kSddmmCsrOutputsPerTile &&
                (reinterpret_cast<uintptr_t>(buffers.matX) & (sizeof(float4) - 1U)) == 0U) {
                SddmmCsrPackedRow<OutputType>(buffers, tiling, state, lane, alpha, beta);
                continue;
            }
        }
        SddmmCsrGeneralRow<InputType, OutputType>(buffers, tiling, state, lane, alpha, beta);
    }
}

constexpr int32_t kSddmmPackTile = 16;

template <typename T>
class SddmmPackYKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR matY, GM_ADDR packedY, const SddmmTilingData *tiling,
        AscendC::TPipe *pipe)
    {
        tiling_ = *tiling;
        pipe_ = pipe;
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(matY),
            static_cast<uint64_t>(tiling_.rowsY) * static_cast<uint64_t>(tiling_.ldY));
        packed_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(packedY),
            static_cast<uint64_t>(tiling_.n) * static_cast<uint64_t>(tiling_.k));
        pipe_->InitBuffer(sourceBuffer_, kSddmmPackTile * kSddmmPackTile * sizeof(T));
        pipe_->InitBuffer(transposedBuffer_, kSddmmPackTile * kSddmmPackTile * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        const int64_t tilesK = (tiling_.k + kSddmmPackTile - 1) / kSddmmPackTile;
        const int64_t tilesN = (tiling_.n + kSddmmPackTile - 1) / kSddmmPackTile;
        const int64_t tileCount = tilesK * tilesN;
        const int64_t block = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
        for (int64_t tile = block; tile < tileCount; tile += blockCount) {
            const int64_t tileK = tile / tilesN;
            const int64_t tileN = tile - tileK * tilesN;
            const int64_t kStart = tileK * kSddmmPackTile;
            const int64_t nStart = tileN * kSddmmPackTile;
            const int32_t currentK = static_cast<int32_t>(
                tiling_.k - kStart < kSddmmPackTile ? tiling_.k - kStart : kSddmmPackTile);
            const int32_t currentN = static_cast<int32_t>(
                tiling_.n - nStart < kSddmmPackTile ? tiling_.n - nStart : kSddmmPackTile);
            ProcessTile(kStart, nStart, currentK, currentN);
        }
    }

private:
    __aicore__ inline void ProcessTile(
        int64_t kStart, int64_t nStart, int32_t currentK, int32_t currentN)
    {
        AscendC::LocalTensor<T> source = sourceBuffer_.Get<T>();
        AscendC::LocalTensor<T> transposed = transposedBuffer_.Get<T>();
        AscendC::Duplicate(source, static_cast<T>(0), kSddmmPackTile * kSddmmPackTile);
        AscendC::Duplicate(transposed, static_cast<T>(0), kSddmmPackTile * kSddmmPackTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);

        AscendC::DataCopyExtParams load{1,
            static_cast<uint32_t>(currentN * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        for (int32_t row = 0; row < currentK; ++row) {
            const int64_t sourceOffset = (kStart + row) * tiling_.ldY + nStart;
            AscendC::DataCopyPad(source[row * kSddmmPackTile], source_[sourceOffset], load, pad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        TransposeTile(source, transposed);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams store{1,
            static_cast<uint32_t>(currentK * sizeof(T)), 0, 0, 0};
        for (int32_t column = 0; column < currentN; ++column) {
            const int64_t packedOffset = (nStart + column) * tiling_.k + kStart;
            AscendC::DataCopyPad(packed_[packedOffset],
                transposed[column * kSddmmPackTile], store);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void TransposeTile(
        AscendC::LocalTensor<T> &source, AscendC::LocalTensor<T> &transposed)
    {
        AscendC::LocalTensor<T> sourceList[kSddmmPackTile];
        AscendC::LocalTensor<T> destinationList[kSddmmPackTile];
        if constexpr (sizeof(T) == sizeof(float)) {
            constexpr int32_t elementsPerBlock = 8;
            for (int32_t group = 0; group < 2; ++group) {
                for (int32_t row = 0; row < kSddmmPackTile; ++row) {
                    sourceList[row] = source[row * kSddmmPackTile + group * elementsPerBlock];
                }
                for (int32_t row = 0; row < elementsPerBlock; ++row) {
                    const int32_t outputRow = group * elementsPerBlock + row;
                    destinationList[2 * row] = transposed[outputRow * kSddmmPackTile];
                    destinationList[2 * row + 1] =
                        transposed[outputRow * kSddmmPackTile + elementsPerBlock];
                }
                AscendC::TransDataTo5HDParams params(false, false, 1, 0, 0);
                AscendC::TransDataTo5HD<T>(destinationList, sourceList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            for (int32_t row = 0; row < kSddmmPackTile; ++row) {
                sourceList[row] = source[row * kSddmmPackTile];
                destinationList[row] = transposed[row * kSddmmPackTile];
            }
            AscendC::TransDataTo5HDParams params(false, false, 1, 0, 0);
            AscendC::TransDataTo5HD<T>(destinationList, sourceList, params);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::TPipe *pipe_{nullptr};
    AscendC::TBuf<AscendC::TPosition::VECCALC> sourceBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> transposedBuffer_;
    AscendC::GlobalTensor<T> source_;
    AscendC::GlobalTensor<T> packed_;
    SddmmTilingData tiling_{};
};

static __simt_callee__ __aicore__ inline void SddmmResolveBsr4Output(
    __gm__ const int32_t *offsets, __gm__ const int32_t *columns,
    const SddmmTilingData &tiling, uint32_t output,
    int32_t &row, int32_t &column, uint32_t &valueIndex)
{
    const int32_t blockSlot = static_cast<int32_t>(output >> 4U);
    const uint32_t innerRow = (output >> 2U) & 3U;
    const uint32_t innerColumn = output & 3U;
    int32_t low = 0;
    int32_t high = static_cast<int32_t>(tiling.blockRows);
    while (low < high) {
        const int32_t middle = low + (high - low) / 2;
        if (offsets[middle + 1] - tiling.indexBase <= blockSlot) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    row = low * 4 + static_cast<int32_t>(innerRow);
    column = (columns[blockSlot] - tiling.indexBase) * 4 +
        static_cast<int32_t>(innerColumn);
    valueIndex = tiling.blockOrder == 0 ? output :
        (output & ~15U) + innerColumn * 4U + innerRow;
}

// Fixed coordinates remove per-output batch/layout divisions. Dense offsets
// stay 64-bit and the FMA and output conversion order match the generic path.
template <int32_t InputType, int32_t OutputType>
__simt_vf__ __aicore__ __launch_bounds__(kSddmmThreadsPerBlock) inline void
SddmmRealBsr4K32SimtCompute(
    SddmmSimtBuffers buffers,
    SddmmTilingData tiling,
    float alpha,
    float beta,
    int32_t threadsPerCore)
{
    const uint32_t totalOutputs = static_cast<uint32_t>(tiling.outputsPerBatch);
    const uint32_t globalThread = static_cast<uint32_t>(threadIdx.x) +
        static_cast<uint32_t>(blockIdx.x) * static_cast<uint32_t>(threadsPerCore);
    const uint32_t gridStride = static_cast<uint32_t>(threadsPerCore) *
        static_cast<uint32_t>(tiling.numBlocks);
    for (uint32_t output = globalThread; output < totalOutputs; output += gridStride) {
        int32_t row = 0;
        int32_t column = 0;
        uint32_t valueIndex = 0;
        SddmmResolveBsr4Output(buffers.offsets, buffers.columns, tiling, output, row, column, valueIndex);
        float accumulator = 0.0F;
        if (alpha != 0.0F) {
            const int64_t xStart = static_cast<int64_t>(row) * 32;
            int64_t yOffset = column;
#pragma unroll 32
            for (int32_t reduction = 0; reduction < 32; ++reduction) {
                const float xValue = SddmmLoadReal<InputType>(buffers.matX, xStart + reduction);
                const float yValue = SddmmLoadReal<InputType>(buffers.matY, yOffset);
                accumulator = __fmaf_rn(xValue, yValue, accumulator);
                yOffset += tiling.ldY;
            }
        }
        const float oldValue = beta == 0.0F ? 0.0F :
            SddmmLoadReal<OutputType>(buffers.sparseValues, valueIndex);
        const float outputValue = __fmaf_rn(alpha, accumulator, beta * oldValue);
        SddmmStoreReal<OutputType>(buffers.sparseValues, valueIndex, outputValue);
    }
}

static __simt_callee__ __aicore__ inline void SddmmComplexEpilogue(
    __gm__ uint8_t *sparseValues, int64_t valueIndex,
    const SddmmComplexValue &accumulator,
    const SddmmComplexValue &alpha, const SddmmComplexValue &beta)
{
    const bool betaIsZero = beta.real == 0.0F && beta.imag == 0.0F;
    const SddmmComplexValue oldValue = betaIsZero ? SddmmComplexValue{0.0F, 0.0F} :
        SddmmLoadComplex(sparseValues, valueIndex);
    const SddmmComplexValue scaledDot{
        alpha.real * accumulator.real - alpha.imag * accumulator.imag,
        alpha.real * accumulator.imag + alpha.imag * accumulator.real};
    // Multiplication by complex one would introduce 0*Inf cross terms.
    const bool betaIsOne = beta.real == 1.0F && beta.imag == 0.0F;
    const SddmmComplexValue scaledOld = betaIsOne ? oldValue : SddmmComplexValue{
        beta.real * oldValue.real - beta.imag * oldValue.imag,
        beta.real * oldValue.imag + beta.imag * oldValue.real};
    SddmmStoreComplex(sparseValues, valueIndex,
        SddmmComplexValue{scaledDot.real + scaledOld.real, scaledDot.imag + scaledOld.imag});
}

// The small-block path keeps the generic complex arithmetic order, while
// resolving fixed BSR4 coordinates without dynamic divisions or batch work.
__simt_vf__ __aicore__ __launch_bounds__(kSddmmThreadsPerBlock) inline void
SddmmComplexBsr4K32SimtCompute(
    SddmmSimtBuffers buffers,
    SddmmTilingData tiling,
    SddmmComplexValue alpha,
    SddmmComplexValue beta,
    int32_t threadsPerCore)
{
    const uint32_t totalOutputs = static_cast<uint32_t>(tiling.outputsPerBatch);
    const uint32_t globalThread = static_cast<uint32_t>(threadIdx.x) +
        static_cast<uint32_t>(blockIdx.x) * static_cast<uint32_t>(threadsPerCore);
    const uint32_t gridStride = static_cast<uint32_t>(threadsPerCore) *
        static_cast<uint32_t>(tiling.numBlocks);
    const bool alphaIsZero = alpha.real == 0.0F && alpha.imag == 0.0F;
    const int64_t yStep = tiling.ldY * 2;
    for (uint32_t output = globalThread; output < totalOutputs; output += gridStride) {
        int32_t row = 0;
        int32_t column = 0;
        uint32_t valueIndex = 0;
        SddmmResolveBsr4Output(buffers.offsets, buffers.columns, tiling, output, row, column, valueIndex);

        SddmmComplexValue accumulator{0.0F, 0.0F};
        if (!alphaIsZero) {
            __gm__ const float *x = reinterpret_cast<__gm__ const float *>(buffers.matX) +
                static_cast<int64_t>(row) * 64;
            __gm__ const float *y = reinterpret_cast<__gm__ const float *>(buffers.matY);
            int64_t yOffset = static_cast<int64_t>(column) * 2;
#pragma unroll 4
            for (int32_t reduction = 0; reduction < 32; ++reduction) {
                const SddmmComplexValue xValue{x[0], x[1]};
                const SddmmComplexValue yValue{y[yOffset], y[yOffset + 1]};
                accumulator.real = __fmaf_rn(xValue.real, yValue.real, accumulator.real);
                accumulator.real = __fmaf_rn(-xValue.imag, yValue.imag, accumulator.real);
                accumulator.imag = __fmaf_rn(xValue.real, yValue.imag, accumulator.imag);
                accumulator.imag = __fmaf_rn(xValue.imag, yValue.real, accumulator.imag);
                x += 2;
                yOffset += yStep;
            }
        }

        SddmmComplexEpilogue(buffers.sparseValues, valueIndex, accumulator, alpha, beta);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSddmmThreadsPerBlock) inline void SddmmComplexSimtCompute(
    SddmmSimtBuffers buffers,
    SddmmTilingData tiling,
    SddmmComplexValue alpha,
    SddmmComplexValue beta,
    int32_t threadsPerCore)
{
    const uint64_t totalOutputs = static_cast<uint64_t>(tiling.outputsPerBatch) *
        static_cast<uint64_t>(tiling.batchCount);
    const uint64_t globalThread = static_cast<uint64_t>(threadIdx.x) +
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(threadsPerCore);
    const uint64_t gridStride = static_cast<uint64_t>(threadsPerCore) *
        static_cast<uint64_t>(tiling.numBlocks);
    const bool alphaIsZero = alpha.real == 0.0F && alpha.imag == 0.0F;
    for (uint64_t output = globalThread; output < totalOutputs; output += gridStride) {
        int64_t batch = 0;
        int64_t row = 0;
        int64_t column = 0;
        int64_t valueIndex = 0;
        SddmmResolveLogicalOutput(buffers.offsets, buffers.columns, tiling, output,
            batch, row, column, valueIndex);

        SddmmComplexValue accumulator{0.0F, 0.0F};
        if (!alphaIsZero) {
            const int64_t batchX = tiling.batchCountX == 1 ? 0 : batch;
            const int64_t batchY = tiling.batchCountY == 1 ? 0 : batch;
            const int64_t xBase = batchX * tiling.xBatchStride;
            const int64_t yBase = batchY * tiling.yBatchStride;
            const bool xReductionContiguous = (tiling.opX == 0) == (tiling.orderX == 0);
            const bool yReductionContiguous = (tiling.opY == 0) != (tiling.orderY == 0);
            const int64_t xStep = xReductionContiguous ? 1 : tiling.ldX;
            const int64_t yStep = yReductionContiguous ? 1 : tiling.ldY;
            const int64_t xStart = xBase + row * (xReductionContiguous ? tiling.ldX : 1);
            const int64_t yStart = yBase + column * (yReductionContiguous ? tiling.ldY : 1);
#pragma unroll 4
            for (int64_t reduction = 0; reduction < tiling.k; ++reduction) {
                const int64_t xIndex = xStart + reduction * xStep;
                const int64_t yIndex = yStart + reduction * yStep;
                const SddmmComplexValue xValue = SddmmLoadComplex(buffers.matX, xIndex);
                const SddmmComplexValue yValue = SddmmLoadComplex(buffers.matY, yIndex);
                accumulator.real = __fmaf_rn(xValue.real, yValue.real, accumulator.real);
                accumulator.real = __fmaf_rn(-xValue.imag, yValue.imag, accumulator.real);
                accumulator.imag = __fmaf_rn(xValue.real, yValue.imag, accumulator.imag);
                accumulator.imag = __fmaf_rn(xValue.imag, yValue.real, accumulator.imag);
            }
        }

        SddmmComplexEpilogue(buffers.sparseValues, valueIndex, accumulator, alpha, beta);
    }
}

template <typename InputT, typename OutputT>
__aicore__ inline bool TrySddmmBsr8Vector(
    GM_ADDR offsets, GM_ADDR columns, GM_ADDR values,
    GM_ADDR matX, GM_ADDR matY, const SddmmTilingData &tiling)
{
    using Kernel = sddmm_arch35::SddmmBsr8VectorKernel<InputT, OutputT>;
    if (!Kernel::CanRun(tiling)) {
        return false;
    }
    AscendC::TPipe pipe;
    Kernel kernel;
    kernel.Init(offsets, columns, values, matX, matY, &tiling, &pipe);
    kernel.Process();
    return true;
}

class SddmmComputeDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR offsets,
        GM_ADDR columns,
        GM_ADDR sparseValues,
        GM_ADDR matX,
        GM_ADDR matY,
        const SddmmTilingData *tiling)
    {
        buffers_.offsets = reinterpret_cast<__gm__ const int32_t *>(offsets);
        buffers_.columns = reinterpret_cast<__gm__ const int32_t *>(columns);
        buffers_.sparseValues = reinterpret_cast<__gm__ uint8_t *>(sparseValues);
        buffers_.matX = reinterpret_cast<__gm__ const uint8_t *>(matX);
        buffers_.matY = reinterpret_cast<__gm__ const uint8_t *>(matY);
        tiling_ = *tiling;
        alpha_ = SddmmComplexValue{tiling_.alphaReal, tiling_.alphaImag};
        beta_ = SddmmComplexValue{tiling_.betaReal, tiling_.betaImag};
        if (tiling_.pointerMode != 0) {
            __gm__ const float *alphaParts = reinterpret_cast<__gm__ const float *>(tiling_.alphaPtr);
            __gm__ const float *betaParts = reinterpret_cast<__gm__ const float *>(tiling_.betaPtr);
            alpha_.real = alphaParts[0];
            beta_.real = betaParts[0];
            if (tiling_.computeType == SDDMM_VALUE_COMPLEX64) {
                alpha_.imag = alphaParts[1];
                beta_.imag = betaParts[1];
            } else {
                alpha_.imag = 0.0F;
                beta_.imag = 0.0F;
            }
        }
    }

    __aicore__ inline void Process()
    {
        // Device scalars are resolved in Init. Identity must preserve the C
        // payload exactly, including signed zeros and nonfinite components;
        // multiplying by one would introduce 0*Inf cross terms for complex C.
        if (alpha_.real == 0.0F && alpha_.imag == 0.0F &&
            beta_.real == 1.0F && beta_.imag == 0.0F) {
            return;
        }
        const uint64_t totalOutputs = static_cast<uint64_t>(tiling_.outputsPerBatch) *
            static_cast<uint64_t>(tiling_.batchCount);
        uint32_t threads = totalOutputs < kSddmmThreadsPerBlock ?
            static_cast<uint32_t>(totalOutputs) : kSddmmThreadsPerBlock;
        if (threads == 0U) {
            return;
        }
        threads = (threads + kSddmmWarpSize - 1U) & ~(kSddmmWarpSize - 1U);
        if (tiling_.inputType == SDDMM_VALUE_COMPLEX64) {
            if (UseBsr4K32Path()) {
                asc_vf_call<SddmmComplexBsr4K32SimtCompute>(dim3{threads},
                    buffers_, tiling_, alpha_, beta_,
                    static_cast<int32_t>(threads));
            } else {
                asc_vf_call<SddmmComplexSimtCompute>(dim3{threads},
                    buffers_, tiling_, alpha_, beta_,
                    static_cast<int32_t>(threads));
            }
            return;
        }
        DispatchReal(threads);
    }

private:
    __aicore__ inline bool UseBsr4K32Path() const
    {
        return tiling_.format == SDDMM_FORMAT_BSR && tiling_.blockDim == 4 &&
            tiling_.k == 32 && tiling_.opX == 0 && tiling_.opY == 0 &&
            tiling_.orderX == 0 && tiling_.orderY == 0 &&
            tiling_.ldX == 32 && tiling_.ldY == tiling_.n &&
            tiling_.batchCount == 1 && tiling_.batchCountX == 1 &&
            tiling_.batchCountY == 1 && tiling_.usePackedY == 0 &&
            tiling_.m <= INT32_MAX && tiling_.n <= INT32_MAX &&
            tiling_.outputsPerBatch <= INT32_MAX;
    }

    __aicore__ inline void DispatchReal(uint32_t threads)
    {
#define SDDMM_DISPATCH_REAL(INPUT_TYPE, OUTPUT_TYPE) do { \
        if (UseBsr4K32Path()) { \
            asc_vf_call<SddmmRealBsr4K32SimtCompute<INPUT_TYPE, OUTPUT_TYPE>>(dim3{threads}, \
                buffers_, tiling_, alpha_.real, beta_.real, \
                static_cast<int32_t>(threads)); \
        } else if (UseCsrRowTilePath()) { \
            asc_vf_call<SddmmCsrRowTileSimtCompute<INPUT_TYPE, OUTPUT_TYPE>>(dim3{threads}, \
                buffers_, tiling_, alpha_.real, beta_.real, \
                static_cast<int32_t>(threads)); \
        } else { \
            asc_vf_call<SddmmRealSimtCompute<INPUT_TYPE, OUTPUT_TYPE>>(dim3{threads}, \
                buffers_, tiling_, alpha_.real, beta_.real, \
                static_cast<int32_t>(threads)); \
        } \
    } while (0)
        if (tiling_.inputType == SDDMM_VALUE_FP32) {
            SDDMM_DISPATCH_REAL(SDDMM_VALUE_FP32, SDDMM_VALUE_FP32);
        } else if (tiling_.inputType == SDDMM_VALUE_FP16 && tiling_.outputType == SDDMM_VALUE_FP32) {
            SDDMM_DISPATCH_REAL(SDDMM_VALUE_FP16, SDDMM_VALUE_FP32);
        } else if (tiling_.inputType == SDDMM_VALUE_FP16) {
            SDDMM_DISPATCH_REAL(SDDMM_VALUE_FP16, SDDMM_VALUE_FP16);
        } else if (tiling_.outputType == SDDMM_VALUE_FP32) {
            SDDMM_DISPATCH_REAL(SDDMM_VALUE_BF16, SDDMM_VALUE_FP32);
        } else {
            SDDMM_DISPATCH_REAL(SDDMM_VALUE_BF16, SDDMM_VALUE_BF16);
        }
#undef SDDMM_DISPATCH_REAL
    }

    __aicore__ inline bool UseCsrRowTilePath() const
    {
        // Row ownership is balanced only for a validated fixed-row pattern.
        // General sparse distributions use independent output ownership so a
        // long row can employ every AIV instead of one eight-lane group.
        return tiling_.format == SDDMM_FORMAT_CSR &&
            (tiling_.patternFlags & SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64) != 0U &&
            tiling_.opX == 0 && tiling_.opY == 0 &&
            tiling_.orderX == 0 && tiling_.orderY == 0;
    }

    SddmmSimtBuffers buffers_{};
    SddmmTilingData tiling_{};
    SddmmComplexValue alpha_{0.0F, 0.0F};
    SddmmComplexValue beta_{0.0F, 0.0F};
};

// BufferSize reaches this kernel only for the single-pattern CSR Cube shape.
// Each core validates independent row and column ranges without sharing writable
// state. The Host combines the partial results after one wait.
class SddmmPatternQuery {
public:
    __aicore__ inline void Init(
        GM_ADDR offsets, GM_ADDR columns, GM_ADDR workspace,
        const SddmmTilingData *tiling, uint32_t numBlocks)
    {
        offsets_ = reinterpret_cast<__gm__ const int32_t *>(offsets);
        columns_ = reinterpret_cast<__gm__ const int32_t *>(columns);
        block_ = AscendC::GetBlockIdx();
        numBlocks_ = numBlocks;
        // Scalar stores from different cores must never share a cache line.
        result_ = reinterpret_cast<__gm__ SddmmValidationResult *>(
            workspace + static_cast<uint64_t>(block_) * kSddmmWorkspaceBytes);
        resultGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(result_),
            kSddmmWorkspaceBytes / sizeof(uint64_t));
        tiling_ = *tiling;
    }

    __aicore__ inline void Process()
    {
        InitializeStatus();
        ValidateRows();
        ValidateColumns();
        StoreResult();
    }

private:
    __aicore__ inline void InitializeStatus()
    {
        expectedEnd_ = tiling_.nnzUnits + tiling_.indexBase;
        const int64_t first = offsets_[0];
        status_ = SDDMM_PATTERN_VALID;
        if (first != tiling_.indexBase || offsets_[tiling_.m] != expectedEnd_) {
            status_ = SDDMM_PATTERN_BAD_ENDPOINT;
        }
        // All cores use the same safe row-zero reference. This small repeated
        // read avoids a second kernel or any cross-core communication.
        logicalFirst_ = RecoverRow(first, offsets_[1]);
        sortedCyclic64_ = logicalFirst_ >= 0;
        stride17_ = sortedCyclic64_;
    }

    __aicore__ inline void ValidateRows()
    {
        const int64_t rowBegin = PartitionBegin(tiling_.m, block_);
        const int64_t rowEnd = PartitionBegin(tiling_.m, block_ + 1U);
        for (int64_t row = rowBegin; row < rowEnd; ++row) {
            const int64_t current = offsets_[row];
            const int64_t next = offsets_[row + 1];
            if (current < tiling_.indexBase || next < current || next > expectedEnd_) {
                if (status_ == SDDMM_PATTERN_VALID) {
                    status_ = SDDMM_PATTERN_NOT_MONOTONIC;
                }
            }
            if (sortedCyclic64_) {
                const int32_t rowStart = RecoverRow(current, next);
                if (rowStart < 0) {
                    sortedCyclic64_ = false;
                    stride17_ = false;
                } else if (stride17_) {
                    const int64_t expectedStart =
                        (logicalFirst_ + static_cast<int64_t>(kSddmmCsrCyclic64RowStride) * row) % tiling_.n;
                    stride17_ = rowStart == expectedStart;
                }
            }
        }
    }

    __aicore__ inline void ValidateColumns()
    {
        // Do not derive this range from offsets: malformed offsets must not
        // skip stored columns or cause an out-of-bounds column read.
        const int64_t columnBegin = PartitionBegin(tiling_.nnzUnits, block_);
        const int64_t columnEnd = PartitionBegin(tiling_.nnzUnits, block_ + 1U);
        for (int64_t index = columnBegin; index < columnEnd; ++index) {
            const int64_t column = columns_[index];
            if (column < tiling_.indexBase || column >= tiling_.n + tiling_.indexBase) {
                if (status_ == SDDMM_PATTERN_VALID) {
                    status_ = SDDMM_PATTERN_BAD_COLUMN;
                }
            }
        }
    }

    __aicore__ inline void StoreResult()
    {
        const uint32_t flags = sddmm_arch35::BuildSddmmPatternFlags(status_, sortedCyclic64_, stride17_);
        result_->magic = kSddmmValidationMagic;
        result_->status = status_;
        result_->patternFlags = flags;
        result_->logicalFirstColumn = flags != SDDMM_PATTERN_FLAG_NONE ? logicalFirst_ : -1;
        result_->reserved = 0;
        AscendC::DataCacheCleanAndInvalid<uint64_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(resultGlobal_);
    }

    __aicore__ inline int64_t PartitionBegin(int64_t count, uint32_t block) const
    {
        const int64_t quotient = count / numBlocks_;
        const uint32_t remainder = static_cast<uint32_t>(count % numBlocks_);
        return quotient * block + (block < remainder ? block : remainder);
    }

    __aicore__ inline int32_t RecoverRow(int64_t current, int64_t next) const
    {
        const int64_t begin = current - tiling_.indexBase;
        const int64_t end = next - tiling_.indexBase;
        if (begin < 0 || end > tiling_.nnzUnits || end - begin != 64) {
            return -1;
        }
        return RecoverSddmmSortedCyclic64Start(columns_ + begin, tiling_.n, tiling_.indexBase);
    }

    __gm__ const int32_t *offsets_{nullptr};
    __gm__ const int32_t *columns_{nullptr};
    __gm__ SddmmValidationResult *result_{nullptr};
    AscendC::GlobalTensor<uint64_t> resultGlobal_;
    SddmmTilingData tiling_{};
    uint32_t block_{};
    uint32_t numBlocks_{};
    int64_t expectedEnd_{};
    int32_t status_{};
    int32_t logicalFirst_{};
    bool sortedCyclic64_{};
    bool stride17_{};
};

} // namespace

extern "C" __global__ __aicore__ void sddmm_pattern_validate_kernel(
    GM_ADDR offsets, GM_ADDR columns, GM_ADDR workspace,
    const SddmmTilingData tiling, uint32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SddmmPatternValidator validator;
    validator.Init(offsets, columns, workspace, &tiling, numBlocks);
    validator.Process();
}

extern "C" __global__ __aicore__ void sddmm_pattern_query_kernel(
    GM_ADDR offsets, GM_ADDR columns, GM_ADDR workspace,
    const SddmmTilingData tiling, uint32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SddmmPatternQuery query;
    query.Init(offsets, columns, workspace, &tiling, numBlocks);
    query.Process();
}

extern "C" __global__ __aicore__ void sddmm_pack_y_kernel(
    GM_ADDR matY, GM_ADDR packedY, const SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    if (tiling.inputType == SDDMM_VALUE_FP16) {
        SddmmPackYKernel<half> packer;
        packer.Init(matY, packedY, &tiling, &pipe);
        packer.Process();
    } else {
        SddmmPackYKernel<float> packer;
        packer.Init(matY, packedY, &tiling, &pipe);
        packer.Process();
    }
}

extern "C" __global__ __aicore__ void sddmm_kernel(
    GM_ADDR offsets, GM_ADDR columns, GM_ADDR sparseValues,
    GM_ADDR matX, GM_ADDR matY, const SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TrySddmmBsr8Vector<float, float>(offsets, columns, sparseValues, matX, matY, tiling) ||
        TrySddmmBsr8Vector<half, half>(offsets, columns, sparseValues, matX, matY, tiling) ||
        TrySddmmBsr8Vector<half, float>(offsets, columns, sparseValues, matX, matY, tiling)) {
        return;
    }
    using PackedVectorKernel = sddmm_arch35::SddmmPackedVectorKernelFp32<16>;
    if (PackedVectorKernel::CanRun(tiling)) {
        AscendC::TPipe pipe;
        PackedVectorKernel kernel;
        kernel.Init(offsets, columns, sparseValues, matX, matY, &tiling, &pipe);
        kernel.Process();
        return;
    }
    using ColumnVectorKernel = sddmm_arch35::SddmmColumnVectorKernelFp16<64>;
    if (ColumnVectorKernel::CanRun(tiling)) {
        AscendC::TPipe pipe;
        ColumnVectorKernel kernel;
        kernel.Init(offsets, columns, sparseValues, matX, matY, &tiling, &pipe);
        kernel.Process();
        return;
    }
    SddmmComputeDispatcher dispatcher;
    dispatcher.Init(offsets, columns, sparseValues, matX, matY, &tiling);
    dispatcher.Process();
}

extern "C" void sddmm_pattern_validate_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns, GM_ADDR workspace,
    const SddmmTilingData &tiling, uint32_t numBlocks, void *stream)
{
    sddmm_pattern_validate_kernel<<<numBlocks, nullptr, stream>>>(
        sparseOffsets, sparseColumns, workspace, tiling, numBlocks);
}

extern "C" void sddmm_pattern_query_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns, GM_ADDR workspace,
    const SddmmTilingData &tiling, uint32_t numBlocks, void *stream)
{
    sddmm_pattern_query_kernel<<<numBlocks, nullptr, stream>>>(
        sparseOffsets, sparseColumns, workspace, tiling, numBlocks);
}

extern "C" void sddmm_pack_y_kernel_do(
    GM_ADDR matY, GM_ADDR packedY,
    const SddmmTilingData &tiling, uint32_t numBlocks, void *stream)
{
    sddmm_pack_y_kernel<<<numBlocks, nullptr, stream>>>(matY, packedY, tiling);
}

extern "C" void sddmm_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns, GM_ADDR sparseValues,
    GM_ADDR matX, GM_ADDR matY,
    const SddmmTilingData &tiling, uint32_t numBlocks, void *stream)
{
    sddmm_kernel<<<numBlocks, nullptr, stream>>>(
        sparseOffsets, sparseColumns, sparseValues, matX, matY, tiling);
}
