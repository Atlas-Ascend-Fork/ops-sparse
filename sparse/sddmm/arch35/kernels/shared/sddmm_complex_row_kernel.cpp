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
 * \file sddmm_complex_row_kernel.cpp
 * \brief Ascend 950 complex64 CSR/BSR row-tile fast path.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "../../sddmm_complex_row_common.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
#error "SDDMM complex64 row kernel is only available on DAV-3510 / Ascend 950."
#endif

namespace {

constexpr uint32_t kSddmmComplexThreads = 256U;

struct SddmmComplexFastValue {
    float real;
    float imag;
};

static __simt_callee__ __aicore__ inline SddmmComplexFastValue
SddmmComplexFastLoad(__gm__ const float *parts, int64_t complexIndex)
{
    const int64_t part = complexIndex * 2;
    return SddmmComplexFastValue{parts[part], parts[part + 1]};
}

static __simt_callee__ __aicore__ inline void SddmmComplexFastStore(
    __gm__ float *parts, int64_t complexIndex,
    const SddmmComplexFastValue &value)
{
    const int64_t part = complexIndex * 2;
    parts[part] = value.real;
    parts[part + 1] = value.imag;
}

static __simt_callee__ __aicore__ inline SddmmComplexFastValue
SddmmComplexFastEpilogue(
    const SddmmComplexFastValue &dot,
    const SddmmComplexFastValue &oldValue,
    const SddmmComplexFastValue &alpha,
    const SddmmComplexFastValue &beta)
{
    const float dotReal = __fmaf_rn(alpha.real, dot.real,
        -alpha.imag * dot.imag);
    const float dotImag = __fmaf_rn(alpha.real, dot.imag,
        alpha.imag * dot.real);
    // Multiplication by complex one would introduce 0*Inf cross terms.
    const bool betaIsOne = beta.real == 1.0F && beta.imag == 0.0F;
    const float oldReal = betaIsOne ? oldValue.real : __fmaf_rn(beta.real, oldValue.real,
        -beta.imag * oldValue.imag);
    const float oldImag = betaIsOne ? oldValue.imag : __fmaf_rn(beta.real, oldValue.imag,
        beta.imag * oldValue.real);
    return SddmmComplexFastValue{dotReal + oldReal, dotImag + oldImag};
}

constexpr int32_t kLanes = sddmm_arch35::kSddmmComplexLanesPerRow;
constexpr int32_t kOutputs = sddmm_arch35::kSddmmComplexOutputsPerLane;
constexpr int32_t kTile = sddmm_arch35::kSddmmComplexSamplesPerRow;

struct SddmmComplexRowTile {
    int64_t valueIndex[kOutputs];
    int32_t yPartOffset[kOutputs];
    bool valid[kOutputs];
    float accumulatorReal[kOutputs];
    float accumulatorImag[kOutputs];
};

template <bool IsBsr>
static __simt_callee__ __aicore__ inline void SddmmComplexRowMapOutput(
    __gm__ const int32_t *columns, const SddmmTilingData &tiling,
    int32_t sparseRowStart, int32_t innerRow, int32_t scalarSample,
    int32_t output, SddmmComplexRowTile &tile)
{
    if constexpr (IsBsr) {
        const int32_t blockOrdinal = scalarSample / tiling.blockDim;
        const int32_t innerColumn =
            scalarSample - blockOrdinal * tiling.blockDim;
        const int32_t blockSlot = sparseRowStart + blockOrdinal;
        const int32_t blockColumn =
            columns[blockSlot] - tiling.indexBase;
        const int64_t blockArea =
            static_cast<int64_t>(tiling.blockDim) * tiling.blockDim;
        const int64_t physicalInner = tiling.blockOrder == 0 ?
            static_cast<int64_t>(innerRow) * tiling.blockDim + innerColumn :
            static_cast<int64_t>(innerColumn) * tiling.blockDim + innerRow;
        tile.valueIndex[output] =
            static_cast<int64_t>(blockSlot) * blockArea + physicalInner;
        tile.yPartOffset[output] =
            (blockColumn * tiling.blockDim + innerColumn) * 2;
    } else {
        const int32_t sparseIndex = sparseRowStart + scalarSample;
        tile.valueIndex[output] = sparseIndex;
        tile.yPartOffset[output] =
            (columns[sparseIndex] - tiling.indexBase) * 2;
    }
}

template <bool IsBsr>
static __simt_callee__ __aicore__ inline void SddmmComplexRowInitTile(
    __gm__ const int32_t *columns, const SddmmTilingData &tiling,
    int32_t sparseRowStart, int32_t innerRow, int32_t scalarSamples,
    int32_t tileStart, int32_t lane, SddmmComplexRowTile &tile)
{
#pragma unroll
    for (int32_t output = 0; output < kOutputs; ++output) {
        const int32_t scalarSample = tileStart + lane + output * kLanes;
        tile.valid[output] = scalarSample < scalarSamples;
        tile.accumulatorReal[output] = 0.0F;
        tile.accumulatorImag[output] = 0.0F;
        if (!tile.valid[output]) {
            tile.valueIndex[output] = 0;
            tile.yPartOffset[output] = 0;
            continue;
        }
        SddmmComplexRowMapOutput<IsBsr>(columns, tiling,
            sparseRowStart, innerRow, scalarSample, output, tile);
    }
}

static __simt_callee__ __aicore__ inline void SddmmComplexRowAccumulate(
    __gm__ const float *xParts, __gm__ const float *yParts,
    const SddmmTilingData &tiling, int32_t row, SddmmComplexRowTile &tile)
{
    __gm__ const float *xRow = xParts +
        static_cast<int64_t>(row) * tiling.ldX * 2;
    const int32_t yRowStrideParts =
        static_cast<int32_t>(tiling.ldY) * 2;
#pragma unroll 2
    for (int32_t reduction = 0;
         reduction < sddmm_arch35::kSddmmComplexReduction;
         ++reduction) {
        const float xReal = xRow[reduction * 2];
        const float xImag = xRow[reduction * 2 + 1];
#pragma unroll
        for (int32_t output = 0; output < kOutputs; ++output) {
            if (tile.valid[output]) {
                const float yReal = yParts[tile.yPartOffset[output]];
                const float yImag = yParts[tile.yPartOffset[output] + 1];
                tile.accumulatorReal[output] = __fmaf_rn(
                    xReal, yReal, tile.accumulatorReal[output]);
                tile.accumulatorReal[output] = __fmaf_rn(
                    -xImag, yImag, tile.accumulatorReal[output]);
                tile.accumulatorImag[output] = __fmaf_rn(
                    xReal, yImag, tile.accumulatorImag[output]);
                tile.accumulatorImag[output] = __fmaf_rn(
                    xImag, yReal, tile.accumulatorImag[output]);
                tile.yPartOffset[output] += yRowStrideParts;
            }
        }
    }
}

static __simt_callee__ __aicore__ inline void SddmmComplexRowStoreTile(
    __gm__ float *sparseParts, const SddmmTilingData &tiling,
    const SddmmComplexRowTile &tile)
{
    const SddmmComplexFastValue alpha{tiling.alphaReal, tiling.alphaImag};
    const SddmmComplexFastValue beta{tiling.betaReal, tiling.betaImag};
    const bool alphaIsOne = alpha.real == 1.0F && alpha.imag == 0.0F;
    const bool betaIsZero = beta.real == 0.0F && beta.imag == 0.0F;
#pragma unroll
    for (int32_t output = 0; output < kOutputs; ++output) {
        if (!tile.valid[output]) {
            continue;
        }
        const SddmmComplexFastValue dot{
            tile.accumulatorReal[output], tile.accumulatorImag[output]};
        if (alphaIsOne && betaIsZero) {
            SddmmComplexFastStore(sparseParts, tile.valueIndex[output], dot);
        } else {
            const SddmmComplexFastValue oldValue = betaIsZero ?
                SddmmComplexFastValue{0.0F, 0.0F} :
                SddmmComplexFastLoad(sparseParts, tile.valueIndex[output]);
            SddmmComplexFastStore(sparseParts, tile.valueIndex[output],
                SddmmComplexFastEpilogue(
                    dot, oldValue, alpha, beta));
        }
    }
}

template <bool IsBsr>
static __simt_callee__ __aicore__ inline void SddmmComplexProcessRow(
    __gm__ const int32_t *offsets, __gm__ const int32_t *columns,
    __gm__ float *sparseParts, __gm__ const float *xParts,
    __gm__ const float *yParts, const SddmmTilingData &tiling,
    int32_t row, int32_t lane)
{
    int32_t sparseRowStart = 0;
    int32_t sparseRowEnd = 0;
    int32_t innerRow = 0;
    int32_t samplesPerSparseUnit = 1;
    if constexpr (IsBsr) {
        const int32_t blockRow = row / tiling.blockDim;
        innerRow = row - blockRow * tiling.blockDim;
        sparseRowStart = offsets[blockRow] - tiling.indexBase;
        sparseRowEnd = offsets[blockRow + 1] - tiling.indexBase;
        samplesPerSparseUnit = tiling.blockDim;
    } else {
        sparseRowStart = offsets[row] - tiling.indexBase;
        sparseRowEnd = offsets[row + 1] - tiling.indexBase;
    }
    const int32_t scalarSamples =
        (sparseRowEnd - sparseRowStart) * samplesPerSparseUnit;
    const bool alphaIsZero = tiling.alphaReal == 0.0F && tiling.alphaImag == 0.0F;
    for (int32_t tileStart = 0; tileStart < scalarSamples; tileStart += kTile) {
        SddmmComplexRowTile tile;
        SddmmComplexRowInitTile<IsBsr>(columns, tiling, sparseRowStart,
            innerRow, scalarSamples, tileStart, lane, tile);
        if (!alphaIsZero) {
            SddmmComplexRowAccumulate(xParts, yParts, tiling, row, tile);
        }
        SddmmComplexRowStoreTile(sparseParts, tiling, tile);
    }
}

template <bool IsBsr>
__simt_vf__ __aicore__ __launch_bounds__(kSddmmComplexThreads) inline void
SddmmComplexRowSimt(
    __gm__ const int32_t *offsets,
    __gm__ const int32_t *columns,
    __gm__ float *sparseParts,
    __gm__ const float *xParts,
    __gm__ const float *yParts,
    SddmmTilingData tiling,
    int32_t threadsPerCore,
    int32_t coreCount)
{
    constexpr int32_t kLanes = sddmm_arch35::kSddmmComplexLanesPerRow;
    const int32_t localThread = static_cast<int32_t>(threadIdx.x);
    const int32_t localRowGroup = localThread / kLanes;
    const int32_t lane = localThread % kLanes;
    const int32_t rowGroupsPerCore = threadsPerCore / kLanes;
    if (rowGroupsPerCore <= 0 || coreCount <= 0) {
        return;
    }

    const int32_t rowStride = coreCount * rowGroupsPerCore;
    int32_t row = static_cast<int32_t>(blockIdx.x) * rowGroupsPerCore + localRowGroup;
    for (; static_cast<int64_t>(row) < tiling.m; row += rowStride) {
        SddmmComplexProcessRow<IsBsr>(offsets, columns, sparseParts,
            xParts, yParts, tiling, row, lane);
    }
}

} // namespace

extern "C" __global__ __aicore__ void sddmm_complex_row_kernel(
    GM_ADDR sparseOffsets,
    GM_ADDR sparseColumns,
    GM_ADDR sparseValues,
    GM_ADDR matX,
    GM_ADDR matY,
    const SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr int32_t kThreads = static_cast<int32_t>(kSddmmComplexThreads);
    const int32_t coreCount = static_cast<int32_t>(AscendC::GetBlockNum());
    __gm__ const int32_t *offsets =
        reinterpret_cast<__gm__ const int32_t *>(sparseOffsets);
    __gm__ const int32_t *columns =
        reinterpret_cast<__gm__ const int32_t *>(sparseColumns);
    __gm__ float *values = reinterpret_cast<__gm__ float *>(sparseValues);
    __gm__ const float *xParts = reinterpret_cast<__gm__ const float *>(matX);
    __gm__ const float *yParts = reinterpret_cast<__gm__ const float *>(matY);
    if (tiling.format == SDDMM_FORMAT_CSR) {
        asc_vf_call<SddmmComplexRowSimt<false>>(dim3{kSddmmComplexThreads},
            offsets, columns, values, xParts, yParts, tiling,
            kThreads, coreCount);
    } else {
        asc_vf_call<SddmmComplexRowSimt<true>>(dim3{kSddmmComplexThreads},
            offsets, columns, values, xParts, yParts, tiling,
            kThreads, coreCount);
    }
}

extern "C" void sddmm_complex_row_kernel_do(
    GM_ADDR sparseOffsets,
    GM_ADDR sparseColumns,
    GM_ADDR sparseValues,
    GM_ADDR matX,
    GM_ADDR matY,
    const SddmmTilingData &tiling,
    uint32_t vectorBlockDim,
    void *stream)
{
    if (vectorBlockDim == 0U ||
        !sddmm_arch35::SddmmComplexRowCanRun(tiling)) {
        return;
    }
    sddmm_complex_row_kernel<<<vectorBlockDim, nullptr, stream>>>(
        sparseOffsets, sparseColumns, sparseValues, matX, matY, tiling);
}
