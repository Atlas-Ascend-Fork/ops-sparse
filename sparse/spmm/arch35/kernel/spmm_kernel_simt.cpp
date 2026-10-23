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

namespace {

// Column chunks for the real, half-pair, complex and integer SIMT paths.
constexpr int32_t kRealChunk = 4;
constexpr int32_t kFp16PairChunk = 2;
// Complex64 keeps separate real/imag FP32 accumulators.
constexpr int32_t kComplexChunk = 2;
constexpr int32_t kIntChunk = 8;

struct SpmmSimtWork {
    int32_t logicalRow;
    int32_t colTile;
    int32_t rowAdvance;
    int32_t tileAdvance;
};

struct SpmmSimtLaunchInfo {
    int32_t rowStart;
    int32_t rowEnd;
    uint32_t threadCount;
};

__aicore__ inline SpmmSimtLaunchInfo GetSpmmSimtLaunchInfo(
    const SpmmKernelTilingData &tiling, __gm__ const int32_t *binEdge,
    int32_t colTiles)
{
    const int32_t blockCount = static_cast<int32_t>(GetBlockNum());
    const int32_t blockIndex = static_cast<int32_t>(GetBlockIdx());
    SpmmSimtLaunchInfo info{0, 0, 1u};
    if (blockIndex >= 0 && blockIndex < blockCount) {
        info.rowStart = binEdge[blockIndex];
        info.rowEnd = binEdge[blockIndex + 1];
    }
    const uint64_t totalWork = static_cast<uint64_t>(tiling.m) *
        static_cast<uint64_t>(colTiles);
    if (totalWork > 0u && blockCount > 0) {
        const uint64_t workPerBlock = (totalWork +
            static_cast<uint64_t>(blockCount) - 1u) /
            static_cast<uint64_t>(blockCount);
        info.threadCount = workPerBlock > kMaxSimtThreadsPerBlock
            ? kMaxSimtThreadsPerBlock : static_cast<uint32_t>(workPerBlock);
    }
    return info;
}

__simt_callee__ __aicore__ inline bool GetSpmmSimtWork(
    uint64_t totalWork, int32_t colTiles, int32_t logicalRowStart,
    SpmmSimtWork *work)
{
    const uint64_t threadIndex = static_cast<uint64_t>(threadIdx.x);
    if (threadIndex >= totalWork || colTiles <= 0 || work == nullptr) {
        return false;
    }
    const uint32_t threadCount = blockDim.x;
    work->logicalRow = logicalRowStart + static_cast<int32_t>(
        threadIndex / static_cast<uint64_t>(colTiles));
    work->colTile = static_cast<int32_t>(
        threadIndex % static_cast<uint64_t>(colTiles));
    work->rowAdvance = static_cast<int32_t>(
        threadCount / static_cast<uint32_t>(colTiles));
    work->tileAdvance = static_cast<int32_t>(
        threadCount % static_cast<uint32_t>(colTiles));
    return true;
}

__simt_callee__ __aicore__ inline bool GetSpmmCsrTileWork(
    int32_t logicalRowStart, int32_t logicalRowEnd,
    int32_t colTiles, SpmmSimtWork *work)
{
    const int32_t numRows = logicalRowEnd - logicalRowStart;
    if (numRows <= 0 || colTiles <= 0) return false;
    const uint64_t totalWork = static_cast<uint64_t>(numRows) * colTiles;
    return GetSpmmSimtWork(totalWork, colTiles, logicalRowStart, work);
}

template<typename T, typename U>
struct SpmmIsSame { static constexpr bool value = false; };

template<typename T>
struct SpmmIsSame<T, T> { static constexpr bool value = true; };

struct SpmmComplex64 {
    float real;
    float imag;
};

static_assert(sizeof(SpmmComplex64) == 8, "complex64 device ABI must be 8 bytes");

__simt_callee__ __aicore__ inline float ClampSpmmFp16(float value)
{
    constexpr float kFp16Max = 65504.0f;
    return value > kFp16Max ? kFp16Max :
        (value < -kFp16Max ? -kFp16Max : value);
}

template<int32_t PairChunk>
__simt_callee__ __aicore__ inline void AccumulateSpmmFp16PairProduct(
    __gm__ const int32_t *colInd, __gm__ const half *values,
    __gm__ const half *matB, int32_t start, int32_t end, int32_t indexBase,
    int32_t n, int32_t ldb, int32_t colFirst, float *accLow, float *accHigh)
{
    for (int32_t position = start; position < end; ++position) {
        const int32_t column = colInd[position] - indexBase;
        const uint64_t bBase = static_cast<uint64_t>(column) * ldb + colFirst;
        const float value = __half2float(values[position]);
#pragma unroll
        for (int j = 0; j < PairChunk; ++j) {
            const int32_t first = colFirst + j * 2;
            if (first >= n) {
                break;
            }
            if (first + 1 < n) {
                const half2 packedB = *reinterpret_cast<__gm__ const half2 *>(
                    matB + bBase + static_cast<uint64_t>(j * 2));
                accLow[j] += value * __low2float(packedB);
                accHigh[j] += value * __high2float(packedB);
            } else {
                accLow[j] += value * __half2float(matB[bBase + j * 2]);
            }
        }
    }
}

template<int32_t PairChunk>
__simt_callee__ __aicore__ inline void StoreSpmmFp16PairProduct(
    __gm__ half *matC, uint64_t cBase, int32_t colFirst, int32_t n,
    float alpha, float beta, bool alphaOne, bool betaZero, bool betaOne,
    const float *accLow, const float *accHigh)
{
#pragma unroll
    for (int j = 0; j < PairChunk; ++j) {
        const int32_t first = colFirst + j * 2;
        if (first >= n) {
            break;
        }
        float low = alphaOne ? accLow[j] : alpha * accLow[j];
        float high = alphaOne ? accHigh[j] : alpha * accHigh[j];
        if (!betaZero) {
            if (first + 1 < n) {
                const half2 old = *reinterpret_cast<__gm__ const half2 *>(
                    matC + cBase + static_cast<uint64_t>(j * 2));
                low += betaOne ? __low2float(old) : beta * __low2float(old);
                high += betaOne ? __high2float(old) : beta * __high2float(old);
            } else {
                const float old = static_cast<float>(matC[cBase + j * 2]);
                low += betaOne ? old : beta * old;
            }
        }
        low = ClampSpmmFp16(low);
        high = ClampSpmmFp16(high);
        if (first + 1 < n) {
            *reinterpret_cast<__gm__ half2 *>(
                matC + cBase + static_cast<uint64_t>(j * 2)) =
                __floats2half2_rn(low, high);
        } else {
            matC[cBase + j * 2] = static_cast<half>(low);
        }
    }
}

template<int32_t PairChunk>
__simt_callee__ __aicore__ inline void ProcessSpmmFp16PairTile(
    __gm__ const int32_t *rowOff, __gm__ const int32_t *colInd,
    __gm__ const half *values, __gm__ const half *matB, __gm__ half *matC,
    __gm__ const int32_t *reorder, int32_t logicalRow, int32_t colTile,
    int32_t n, int32_t ldb, int32_t ldc, int32_t indexBase, float alpha,
    float beta, bool skipProduct, bool alphaOne, bool betaZero, bool betaOne)
{
    const int32_t colFirst = colTile * PairChunk * 2;
    const int32_t row = skipProduct ? logicalRow : reorder[logicalRow];
    const int32_t start = skipProduct ? 0 : rowOff[row] - indexBase;
    const int32_t end = skipProduct ? 0 : rowOff[row + 1] - indexBase;
    float accLow[PairChunk];
    float accHigh[PairChunk];
#pragma unroll
    for (int j = 0; j < PairChunk; ++j) {
        accLow[j] = 0.0f;
        accHigh[j] = 0.0f;
    }
    AccumulateSpmmFp16PairProduct<PairChunk>(colInd, values, matB, start, end,
        indexBase, n, ldb, colFirst, accLow, accHigh);
    const uint64_t cBase = static_cast<uint64_t>(row) * ldc + colFirst;
    StoreSpmmFp16PairProduct<PairChunk>(matC, cBase, colFirst, n, alpha, beta,
        alphaOne, betaZero, betaOne, accLow, accHigh);
}

template<int32_t PairChunk>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmCsrSimtComputeFp16PairProduct(
    __gm__ const int32_t *rowOff, __gm__ const int32_t *colInd,
    __gm__ const half *values, __gm__ const half *matB, __gm__ half *matC,
    __gm__ const int32_t *reorder, int32_t n, int32_t ldb, int32_t ldc,
    int32_t colTiles, int32_t logicalRowStart, int32_t logicalRowEnd,
    float alpha, float beta, int32_t indexBase, int32_t sparseEmpty)
{
    SpmmSimtWork work;
    if (!GetSpmmCsrTileWork(logicalRowStart, logicalRowEnd, colTiles, &work)) {
        return;
    }

    int32_t logicalRow = work.logicalRow;
    int32_t colTile = work.colTile;
    const int32_t rowAdvance = work.rowAdvance;
    const int32_t tileAdvance = work.tileAdvance;
    const bool skipProduct = alpha == 0.0f || sparseEmpty != 0;
    const bool alphaOne = alpha == 1.0f;
    const bool betaZero = beta == 0.0f;
    const bool betaOne = beta == 1.0f;

    while (logicalRow < logicalRowEnd) {
        ProcessSpmmFp16PairTile<PairChunk>(rowOff, colInd, values, matB, matC,
            reorder, logicalRow, colTile, n, ldb, ldc, indexBase, alpha, beta,
            skipProduct, alphaOne, betaZero, betaOne);
        logicalRow += rowAdvance;
        colTile += tileAdvance;
        if (colTile >= colTiles) {
            colTile -= colTiles;
            ++logicalRow;
        }
    }
}

__simt_callee__ __aicore__ inline SpmmComplex64 SpmmComplexMul(SpmmComplex64 lhs, SpmmComplex64 rhs)
{
    return SpmmComplex64{
        lhs.real * rhs.real - lhs.imag * rhs.imag,
        lhs.real * rhs.imag + lhs.imag * rhs.real};
}

__simt_callee__ __aicore__ inline SpmmComplex64 SpmmComplexConj(SpmmComplex64 value)
{
    return SpmmComplex64{value.real, -value.imag};
}

template<int32_t Chunk>
__simt_callee__ __aicore__ inline void AccumulateSpmmComplexProduct(
    __gm__ const int32_t *colInd, __gm__ const float *values,
    __gm__ const float *matB, int32_t start, int32_t end, int32_t indexBase,
    int32_t n, int32_t ldb, int32_t colFirst, bool bContiguous,
    uint64_t bStride, int32_t opB, float *accReal, float *accImag)
{
    for (int32_t position = start; position < end; ++position) {
        const int32_t column = colInd[position] - indexBase;
        const uint64_t valueOffset = static_cast<uint64_t>(position) * 2u;
        const float valueReal = values[valueOffset];
        const float valueImag = values[valueOffset + 1u];
        const uint64_t bBase = bContiguous
            ? static_cast<uint64_t>(column) * static_cast<uint64_t>(ldb) +
                  static_cast<uint64_t>(colFirst)
            : static_cast<uint64_t>(colFirst) * static_cast<uint64_t>(ldb) +
                  static_cast<uint64_t>(column);
#pragma unroll
        for (int j = 0; j < Chunk; ++j) {
            if (colFirst + j < n) {
                const uint64_t bOffset =
                    (bBase + static_cast<uint64_t>(j) * bStride) * 2u;
                const float bReal = matB[bOffset];
                const float bImag = opB == 2 ? -matB[bOffset + 1u] :
                    matB[bOffset + 1u];
                accReal[j] += valueReal * bReal - valueImag * bImag;
                accImag[j] += valueReal * bImag + valueImag * bReal;
            }
        }
    }
}

template<int32_t Chunk>
__simt_callee__ __aicore__ inline void StoreSpmmComplexProduct(
    __gm__ const float *initialC, __gm__ float *matC, uint64_t cBase,
    uint64_t cStride, int32_t colFirst, int32_t n, float alphaReal,
    float alphaImag, float betaReal, float betaImag, bool alphaOne,
    bool betaZero, bool betaOne, const float *accReal, const float *accImag)
{
#pragma unroll
    for (int j = 0; j < Chunk; ++j) {
        if (colFirst + j >= n) {
            break;
        }
        float outReal = alphaOne ? accReal[j] :
            alphaReal * accReal[j] - alphaImag * accImag[j];
        float outImag = alphaOne ? accImag[j] :
            alphaReal * accImag[j] + alphaImag * accReal[j];
        const uint64_t cOffset =
            (cBase + static_cast<uint64_t>(j) * cStride) * 2u;
        if (!betaZero) {
            const float oldReal = initialC[cOffset];
            const float oldImag = initialC[cOffset + 1u];
            if (betaOne) {
                outReal += oldReal;
                outImag += oldImag;
            } else {
                outReal += betaReal * oldReal - betaImag * oldImag;
                outImag += betaReal * oldImag + betaImag * oldReal;
            }
        }
        matC[cOffset] = outReal;
        matC[cOffset + 1u] = outImag;
    }
}



template<typename ScalarT>
__aicore__ inline void LoadSpmmScalarPair(const SpmmKernelTilingData &tiling,
    ScalarT *alpha, ScalarT *beta)
{
    if (tiling.pointer_mode != 0) {
        if constexpr (SpmmIsSame<ScalarT, SpmmComplex64>::value) {
            const __gm__ SpmmComplex64 *alphaPtr =
                reinterpret_cast<__gm__ SpmmComplex64 *>(
                    static_cast<uintptr_t>(tiling.alpha_ptr));
            const __gm__ SpmmComplex64 *betaPtr =
                reinterpret_cast<__gm__ SpmmComplex64 *>(
                    static_cast<uintptr_t>(tiling.beta_ptr));
            *alpha = SpmmComplex64{alphaPtr->real, alphaPtr->imag};
            *beta = SpmmComplex64{betaPtr->real, betaPtr->imag};
        } else if constexpr (SpmmIsSame<ScalarT, int32_t>::value) {
            const __gm__ int32_t *alphaPtr = reinterpret_cast<__gm__ int32_t *>(
                static_cast<uintptr_t>(tiling.alpha_ptr));
            const __gm__ int32_t *betaPtr = reinterpret_cast<__gm__ int32_t *>(
                static_cast<uintptr_t>(tiling.beta_ptr));
            *alpha = *alphaPtr;
            *beta = *betaPtr;
        } else {
            const __gm__ float *alphaPtr = reinterpret_cast<__gm__ float *>(
                static_cast<uintptr_t>(tiling.alpha_ptr));
            const __gm__ float *betaPtr = reinterpret_cast<__gm__ float *>(
                static_cast<uintptr_t>(tiling.beta_ptr));
            *alpha = static_cast<ScalarT>(*alphaPtr);
            *beta = static_cast<ScalarT>(*betaPtr);
        }
    } else if constexpr (SpmmIsSame<ScalarT, SpmmComplex64>::value) {
        *alpha = SpmmComplex64{tiling.alpha_host, tiling.alpha_imag};
        *beta = SpmmComplex64{tiling.beta_host, tiling.beta_imag};
    } else if constexpr (SpmmIsSame<ScalarT, int32_t>::value) {
        *alpha = static_cast<ScalarT>(tiling.alpha_int);
        *beta = static_cast<ScalarT>(tiling.beta_int);
    } else {
        *alpha = static_cast<ScalarT>(tiling.alpha_host);
        *beta = static_cast<ScalarT>(tiling.beta_host);
    }
}

__simt_callee__ __aicore__ inline SpmmComplex64 SpmmComplexAdd(SpmmComplex64 lhs, SpmmComplex64 rhs)
{
    return SpmmComplex64{lhs.real + rhs.real, lhs.imag + rhs.imag};
}

__aicore__ inline bool SpmmComplexIsZero(SpmmComplex64 value)
{
    return value.real == 0.0f && value.imag == 0.0f;
}

} // namespace

// ============================================================================
// Templated SIMT compute: SpmmCsrSimtCompute<ValT, CT, ScalarT, UseKahan>
//
//   ValT    = float / half / bfloat16_t / int8_t / SpmmComplex64
//   CT      = float / half / bfloat16_t / int32_t / SpmmComplex64
//   ScalarT = float / int32_t / SpmmComplex64
//
// Uses if constexpr to select type-specific logic at compile time:
//   - ScalarT==float  => fp32 accumulation, direct or cast output
//   - ScalarT==int32_t => int64 accumulation with int32 saturated output
// ============================================================================
struct SpmmScalarFlags {
    bool alphaZero;
    bool alphaOne;
    bool betaZero;
    bool betaOne;
};

template<typename ScalarT>
__simt_callee__ __aicore__ inline SpmmScalarFlags GetSpmmScalarFlags(
    ScalarT alpha, ScalarT beta)
{
    if constexpr (SpmmIsSame<ScalarT, SpmmComplex64>::value) {
        return {SpmmComplexIsZero(alpha),
            alpha.real == 1.0f && alpha.imag == 0.0f,
            SpmmComplexIsZero(beta),
            beta.real == 1.0f && beta.imag == 0.0f};
    } else {
        return {alpha == static_cast<ScalarT>(0), alpha == static_cast<ScalarT>(1),
            beta == static_cast<ScalarT>(0), beta == static_cast<ScalarT>(1)};
    }
}

template<typename ValT, bool UseKahan, int32_t Chunk>
__simt_callee__ __aicore__ inline void AccumulateSpmmFloatCsr(
    __gm__ const int32_t *colInd, __gm__ const ValT *values,
    __gm__ const ValT *matB, int32_t start, int32_t end, int32_t indexBase,
    int32_t colFirst, int32_t n, bool bContiguous, uint64_t bStride,
    int32_t ldb, float *acc, float *comp)
{
    for (int32_t position = start; position < end; ++position) {
        const int32_t column = colInd[position] - indexBase;
        const uint64_t bBase = bContiguous ?
            static_cast<uint64_t>(column) * ldb + colFirst :
            static_cast<uint64_t>(colFirst) * ldb + column;
        const float value = static_cast<float>(values[position]);
#pragma unroll
        for (int j = 0; j < Chunk; ++j) {
            if (colFirst + j >= n) continue;
            const float product = value * static_cast<float>(
                matB[bBase + static_cast<uint64_t>(j) * bStride]);
            if constexpr (UseKahan) {
                const float y = product - comp[j];
                const float sum = acc[j] + y;
                comp[j] = (sum - acc[j]) - y;
                acc[j] = sum;
            } else {
                acc[j] += product;
            }
        }
    }
}

struct SpmmCsrTile {
    int32_t row;
    int32_t start;
    int32_t end;
    int32_t colFirst;
    uint64_t cBase;
};

template<bool IdentityReorder>
__simt_callee__ __aicore__ inline SpmmCsrTile GetSpmmCsrTile(
    __gm__ const int32_t *rowOff, __gm__ const int32_t *reorder,
    int32_t logicalRow, int32_t colTile, int32_t indexBase, int32_t ldc,
    int32_t chunk, bool cRowMajor, bool skipProduct)
{
    const int32_t row = (skipProduct || IdentityReorder) ? logicalRow : reorder[logicalRow];
    const int32_t colFirst = colTile * chunk;
    const uint64_t cBase = cRowMajor ?
        static_cast<uint64_t>(row) * static_cast<uint64_t>(ldc) + colFirst :
        static_cast<uint64_t>(colFirst) * static_cast<uint64_t>(ldc) + row;
    return {row, skipProduct ? 0 : rowOff[row] - indexBase,
        skipProduct ? 0 : rowOff[row + 1] - indexBase, colFirst, cBase};
}

template<typename ScalarT, bool UseKahan, int32_t Chunk>
__simt_callee__ __aicore__ inline void InitializeSpmmAccumulators(
    ScalarT *acc, int64_t *accWide, ScalarT *comp)
{
#pragma unroll
    for (int j = 0; j < Chunk; ++j) {
        if constexpr (SpmmIsSame<ScalarT, SpmmComplex64>::value) {
            acc[j] = SpmmComplex64{0.0f, 0.0f};
        } else if constexpr (SpmmIsSame<ScalarT, float>::value) {
            acc[j] = 0.0f;
            if constexpr (UseKahan) comp[j] = 0.0f;
        } else {
            acc[j] = 0;
            accWide[j] = 0;
        }
    }
}

template<typename ValT, int32_t Chunk>
__simt_callee__ __aicore__ inline void AccumulateSpmmComplexCsr(
    __gm__ const int32_t *colInd, __gm__ const ValT *values,
    __gm__ const ValT *matB, const SpmmCsrTile &tile, int32_t indexBase,
    int32_t n, bool bContiguous, uint64_t bStride, int32_t ldb, int32_t opB,
    SpmmComplex64 *acc)
{
    for (int32_t p = tile.start; p < tile.end; ++p) {
        const int32_t column = colInd[p] - indexBase;
        const uint64_t bBase = bContiguous ?
            static_cast<uint64_t>(column) * ldb + tile.colFirst :
            static_cast<uint64_t>(tile.colFirst) * ldb + column;
        const SpmmComplex64 value{values[p].real, values[p].imag};
#pragma unroll
        for (int j = 0; j < Chunk; ++j) {
            if (tile.colFirst + j >= n) continue;
            SpmmComplex64 bValue{matB[bBase + static_cast<uint64_t>(j) * bStride].real,
                matB[bBase + static_cast<uint64_t>(j) * bStride].imag};
            if (opB == 2) bValue = SpmmComplexConj(bValue);
            acc[j] = SpmmComplexAdd(acc[j], SpmmComplexMul(value, bValue));
        }
    }
}

template<typename ValT, int32_t Chunk>
__simt_callee__ __aicore__ inline void AccumulateSpmmIntCsr(
    __gm__ const int32_t *colInd, __gm__ const ValT *values,
    __gm__ const ValT *matB, const SpmmCsrTile &tile, int32_t indexBase,
    int32_t n, bool bContiguous, uint64_t bStride, int32_t ldb, int64_t *acc)
{
    for (int32_t p = tile.start; p < tile.end; ++p) {
        const int32_t column = colInd[p] - indexBase;
        const uint64_t bBase = bContiguous ?
            static_cast<uint64_t>(column) * ldb + tile.colFirst :
            static_cast<uint64_t>(tile.colFirst) * ldb + column;
        const int32_t value = static_cast<int32_t>(values[p]);
#pragma unroll
        for (int j = 0; j < Chunk; ++j) {
            if (tile.colFirst + j < n) {
                acc[j] += static_cast<int64_t>(value) * static_cast<int64_t>(
                    matB[bBase + static_cast<uint64_t>(j) * bStride]);
            }
        }
    }
}

template<typename CT, int32_t Chunk>
__simt_callee__ __aicore__ inline void StoreSpmmComplexCsr(
    __gm__ const CT *initialC, __gm__ CT *matC, const SpmmCsrTile &tile,
    int32_t n, uint64_t cStride, SpmmComplex64 alpha, SpmmComplex64 beta,
    const SpmmScalarFlags &flags, const SpmmComplex64 *acc)
{
#pragma unroll
    for (int j = 0; j < Chunk; ++j) {
        if (tile.colFirst + j >= n) break;
        const uint64_t cIdx = tile.cBase + static_cast<uint64_t>(j) * cStride;
        SpmmComplex64 value = flags.alphaOne ? acc[j] : SpmmComplexMul(alpha, acc[j]);
        if (!flags.betaZero) {
            const SpmmComplex64 oldValue{initialC[cIdx].real, initialC[cIdx].imag};
            value = SpmmComplexAdd(value, flags.betaOne ? oldValue : SpmmComplexMul(beta, oldValue));
        }
        matC[cIdx].real = value.real;
        matC[cIdx].imag = value.imag;
    }
}

template<typename ValT, typename CT, int32_t Chunk>
__simt_callee__ __aicore__ inline void StoreSpmmFloatCsr(
    __gm__ const CT *initialC, __gm__ CT *matC, const SpmmCsrTile &tile,
    int32_t n, uint64_t cStride, float alpha, float beta,
    const SpmmScalarFlags &flags, const float *acc)
{
#pragma unroll
    for (int j = 0; j < Chunk; ++j) {
        if (tile.colFirst + j >= n) break;
        const uint64_t cIdx = tile.cBase + static_cast<uint64_t>(j) * cStride;
        float value = flags.alphaOne ? acc[j] : alpha * acc[j];
        if (!flags.betaZero) {
            const float oldValue = static_cast<float>(initialC[cIdx]);
            value += flags.betaOne ? oldValue : beta * oldValue;
        }
        if constexpr (SpmmIsSame<CT, float>::value) {
            matC[cIdx] = value;
        } else if constexpr (SpmmIsSame<CT, bfloat16_t>::value) {
            matC[cIdx] = static_cast<bfloat16_t>(value);
        } else {
            constexpr float kFp16Max = 65504.0f;
            const float clamped = value > kFp16Max ? kFp16Max :
                (value < -kFp16Max ? -kFp16Max : value);
            matC[cIdx] = static_cast<ValT>(clamped);
        }
    }
}

template<int32_t Chunk>
__simt_callee__ __aicore__ inline void StoreSpmmIntCsr(
    __gm__ const int32_t *initialC, __gm__ int32_t *matC,
    const SpmmCsrTile &tile, int32_t n, uint64_t cStride, int32_t alpha,
    int32_t beta, const SpmmScalarFlags &flags, const int64_t *acc)
{
#pragma unroll
    for (int j = 0; j < Chunk; ++j) {
        if (tile.colFirst + j >= n) break;
        const uint64_t cIdx = tile.cBase + static_cast<uint64_t>(j) * cStride;
        int64_t value = alpha == 1 ? acc[j] : static_cast<int64_t>(alpha) * acc[j];
        if (!flags.betaZero) {
            const int64_t oldValue = static_cast<int64_t>(initialC[cIdx]);
            value += flags.betaOne ? oldValue : static_cast<int64_t>(beta) * oldValue;
        }
        constexpr int64_t kInt32Max = 2147483647LL;
        constexpr int64_t kInt32Min = -2147483648LL;
        value = value > kInt32Max ? kInt32Max : (value < kInt32Min ? kInt32Min : value);
        matC[cIdx] = static_cast<int32_t>(value);
    }
}

__simt_callee__ __aicore__ inline void AdvanceSpmmCsrTile(
    int32_t *logicalRow, int32_t *colTile, int32_t rowAdvance,
    int32_t tileAdvance, int32_t colTiles)
{
    *logicalRow += rowAdvance;
    *colTile += tileAdvance;
    if (*colTile >= colTiles) {
        *colTile -= colTiles;
        ++*logicalRow;
    }
}

template<typename ValT, typename CT, typename ScalarT, bool UseKahan,
         int32_t Chunk, bool IdentityReorder>
__simt_callee__ __aicore__ inline void ProcessSpmmCsrTile(
    __gm__ const int32_t *rowOff, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ const ValT *matB,
    __gm__ const CT *initialC, __gm__ CT *matC,
    __gm__ const int32_t *reorder, int32_t logicalRow, int32_t colTile,
    int32_t n, int32_t ldb, int32_t ldc, ScalarT alpha, ScalarT beta,
    int32_t opB, int32_t indexBase, int32_t sparseEmpty, bool cRowMajor,
    bool bContiguous, uint64_t bStride, uint64_t cStride,
    const SpmmScalarFlags &flags)
{
    const bool skipProduct = flags.alphaZero || sparseEmpty != 0;
    const SpmmCsrTile tile = GetSpmmCsrTile<IdentityReorder>(
        rowOff, reorder, logicalRow, colTile, indexBase, ldc, Chunk,
        cRowMajor, skipProduct);
    ScalarT acc[Chunk];
    int64_t accWide[Chunk];
    ScalarT comp[Chunk];
    InitializeSpmmAccumulators<ScalarT, UseKahan, Chunk>(acc, accWide, comp);
    if constexpr (SpmmIsSame<ScalarT, SpmmComplex64>::value) {
        AccumulateSpmmComplexCsr<ValT, Chunk>(colInd, values, matB, tile,
            indexBase, n, bContiguous, bStride, ldb, opB, acc);
        StoreSpmmComplexCsr<CT, Chunk>(initialC, matC, tile, n, cStride,
            alpha, beta, flags, acc);
    } else if constexpr (SpmmIsSame<ScalarT, float>::value) {
        AccumulateSpmmFloatCsr<ValT, UseKahan, Chunk>(colInd, values, matB,
            tile.start, tile.end, indexBase, tile.colFirst, n, bContiguous,
            bStride, ldb, acc, comp);
        StoreSpmmFloatCsr<ValT, CT, Chunk>(initialC, matC, tile, n, cStride,
            alpha, beta, flags, acc);
    } else {
        AccumulateSpmmIntCsr<ValT, Chunk>(colInd, values, matB, tile,
            indexBase, n, bContiguous, bStride, ldb, accWide);
        StoreSpmmIntCsr<Chunk>(initialC, matC, tile, n, cStride, alpha,
            beta, flags, accWide);
    }
}

template<typename ValT, typename CT, typename ScalarT, bool UseKahan,
         int32_t Chunk, bool IdentityReorder>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void SpmmCsrSimtCompute(
    __gm__ const int32_t *rowOff,
    __gm__ const int32_t *colInd,
    __gm__ const ValT *values,
    __gm__ const ValT *matB,
    __gm__ const CT *initialC,
    __gm__ CT *matC,
    __gm__ const int32_t *reorder,
    int32_t n,
    int32_t ldb,
    int32_t ldc,
    int32_t colTiles,
    int32_t logicalRowStart,
    int32_t logicalRowEnd,
    ScalarT alpha,
    ScalarT beta,
    int32_t opB,
    int32_t orderPair,
    int32_t indexBase,
    int32_t sparseEmpty)
{
    const int32_t numRows = logicalRowEnd - logicalRowStart;
    if (numRows <= 0 || colTiles <= 0) {
        return;
    }
    const uint64_t totalWork =
        static_cast<uint64_t>(numRows) * static_cast<uint64_t>(colTiles);
    const SpmmScalarFlags flags = GetSpmmScalarFlags(alpha, beta);
    const bool bRowMajor = (orderPair == SPMM_ORDER_RR || orderPair == SPMM_ORDER_RC);
    const bool cRowMajor = (orderPair == SPMM_ORDER_RR || orderPair == SPMM_ORDER_CR);
    const bool bContiguous = ((opB == 0) == bRowMajor);
    const uint64_t bStride = bContiguous ? 1u : static_cast<uint64_t>(ldb);
    const uint64_t cStride = cRowMajor ? 1u : static_cast<uint64_t>(ldc);

    SpmmSimtWork work;
    if (!GetSpmmSimtWork(totalWork, colTiles, logicalRowStart, &work)) {
        return;
    }

    // Decode the linear work id once per SIMT thread, then advance the
    // row/tile coordinates arithmetically. Repeating a 64-bit divide and
    // modulo is avoidable address-generation overhead (P-03 has about
    // 39 million fp32 and 78 million complex64 work items).
    int32_t logicalRow = work.logicalRow;
    int32_t colTile = work.colTile;
    const int32_t rowAdvance = work.rowAdvance;
    const int32_t tileAdvance = work.tileAdvance;

    while (logicalRow < logicalRowEnd) {
        ProcessSpmmCsrTile<ValT, CT, ScalarT, UseKahan, Chunk, IdentityReorder>(
            rowOff, colInd, values, matB, initialC, matC, reorder, logicalRow,
            colTile, n, ldb, ldc, alpha, beta, opB, indexBase, sparseEmpty,
            cRowMajor, bContiguous, bStride, cStride, flags);

        AdvanceSpmmCsrTile(&logicalRow, &colTile, rowAdvance, tileAdvance, colTiles);
    }
}

template<int32_t Chunk>
__simt_callee__ __aicore__ inline void ProcessSpmmComplexCsrTile(
    __gm__ const int32_t *rowOff, __gm__ const int32_t *colInd,
    __gm__ const float *values, __gm__ const float *matB,
    __gm__ const float *initialC, __gm__ float *matC,
    __gm__ const int32_t *reorder, int32_t logicalRow, int32_t colTile,
    int32_t n, int32_t ldb, int32_t ldc, int32_t indexBase, int32_t opB,
    int32_t sparseEmpty, float alphaReal, float alphaImag, float betaReal,
    float betaImag, bool cRowMajor, bool bContiguous, uint64_t bStride,
    uint64_t cStride, const SpmmScalarFlags &flags)
{
    const SpmmCsrTile tile = GetSpmmCsrTile<false>(rowOff, reorder, logicalRow,
        colTile, indexBase, ldc, Chunk, cRowMajor,
        flags.alphaZero || sparseEmpty != 0);
    float accReal[Chunk];
    float accImag[Chunk];
#pragma unroll
    for (int j = 0; j < Chunk; ++j) {
        accReal[j] = 0.0f;
        accImag[j] = 0.0f;
    }
    AccumulateSpmmComplexProduct<Chunk>(colInd, values, matB, tile.start,
        tile.end, indexBase, n, ldb, tile.colFirst, bContiguous, bStride,
        opB, accReal, accImag);
    StoreSpmmComplexProduct<Chunk>(initialC, matC, tile.cBase, cStride,
        tile.colFirst, n, alphaReal, alphaImag, betaReal, betaImag,
        flags.alphaOne, flags.betaZero, flags.betaOne, accReal, accImag);
}

// Complex64 is kept out of the generic ScalarT path. On dav-3510, arrays of a
// user-defined two-float type can be materialized in the SIMT DCache stack and
// trigger an out-of-range stack access. Treating complex tensors as interleaved
// fp32 GM data keeps the accumulator state in two ordinary fp32 arrays.
template<int32_t Chunk>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void SpmmCsrSimtComputeComplex(
    __gm__ const int32_t *rowOff,
    __gm__ const int32_t *colInd,
    __gm__ const float *values,
    __gm__ const float *matB,
    __gm__ const float *initialC,
    __gm__ float *matC,
    __gm__ const int32_t *reorder,
    int32_t n,
    int32_t ldb,
    int32_t ldc,
    int32_t colTiles,
    int32_t logicalRowStart,
    int32_t logicalRowEnd,
    float alphaReal,
    float alphaImag,
    float betaReal,
    float betaImag,
    int32_t opB,
    int32_t orderPair,
    int32_t indexBase,
    int32_t sparseEmpty)
{
    SpmmSimtWork work;
    if (!GetSpmmCsrTileWork(logicalRowStart, logicalRowEnd, colTiles, &work)) {
        return;
    }

    const SpmmScalarFlags flags{alphaReal == 0.0f && alphaImag == 0.0f,
        alphaReal == 1.0f && alphaImag == 0.0f,
        betaReal == 0.0f && betaImag == 0.0f,
        betaReal == 1.0f && betaImag == 0.0f};
    const bool bRowMajor = (orderPair == SPMM_ORDER_RR || orderPair == SPMM_ORDER_RC);
    const bool cRowMajor = (orderPair == SPMM_ORDER_RR || orderPair == SPMM_ORDER_CR);
    const bool bContiguous = ((opB == 0) == bRowMajor);
    const uint64_t bStride = bContiguous ? 1u : static_cast<uint64_t>(ldb);
    const uint64_t cStride = cRowMajor ? 1u : static_cast<uint64_t>(ldc);

    int32_t logicalRow = work.logicalRow;
    int32_t colTile = work.colTile;
    const int32_t rowAdvance = work.rowAdvance;
    const int32_t tileAdvance = work.tileAdvance;

    while (logicalRow < logicalRowEnd) {
        ProcessSpmmComplexCsrTile<Chunk>(rowOff, colInd, values, matB, initialC,
            matC, reorder, logicalRow, colTile, n, ldb, ldc, indexBase, opB,
            sparseEmpty, alphaReal, alphaImag, betaReal, betaImag, cRowMajor,
            bContiguous, bStride, cStride, flags);
        AdvanceSpmmCsrTile(&logicalRow, &colTile, rowAdvance, tileAdvance, colTiles);
    }
}

// The official large P-03 Complex64 case is row-major, non-transposed and
// alpha=1/beta in {0,1}. Keep this path deliberately narrow: with two adjacent
// output columns, every CSR value reuses one contiguous pair of complex B
// values and the address arithmetic is reduced to one row-base calculation.
// All other layouts and scalar combinations continue through the generic path
// above, which is the semantic fallback for the public API.
struct SpmmComplexPairKernelArgs {
    __gm__ const int32_t *rowOff;
    __gm__ const int32_t *colInd;
    __gm__ const float *values;
    __gm__ const float *matB;
    __gm__ const float *initialC;
    __gm__ float *matC;
    __gm__ const int32_t *reorder;
    int32_t n;
    int32_t ldb;
    int32_t ldc;
    int32_t indexBase;
};

struct SpmmComplexPairShape {
    int32_t n;
    int32_t ldb;
    int32_t ldc;
};

template<bool BetaOne, bool IdentityReorder>
__simt_callee__ __aicore__ inline void ComputeSpmmComplexPairTile(
    const SpmmComplexPairKernelArgs &args, int32_t logicalRow,
    int32_t colFirst)
{
    const bool fullTile = colFirst + 1 < args.n;
    const int32_t row = IdentityReorder ? logicalRow : args.reorder[logicalRow];
    const int32_t start = args.rowOff[row] - args.indexBase;
    const int32_t end = args.rowOff[row + 1] - args.indexBase;
    float acc0Real = 0.0f;
    float acc0Imag = 0.0f;
    float acc1Real = 0.0f;
    float acc1Imag = 0.0f;
    for (int32_t position = start; position < end; ++position) {
        const int32_t column = args.colInd[position] - args.indexBase;
        const __gm__ float *value = args.values + (static_cast<uint64_t>(position) << 1);
        const __gm__ float *b = args.matB +
            ((static_cast<uint64_t>(column) * args.ldb + colFirst) << 1);
        const float valueReal = value[0];
        const float valueImag = value[1];
        acc0Real += valueReal * b[0] - valueImag * b[1];
        acc0Imag += valueReal * b[1] + valueImag * b[0];
        if (fullTile) {
            acc1Real += valueReal * b[2] - valueImag * b[3];
            acc1Imag += valueReal * b[3] + valueImag * b[2];
        }
    }
    const uint64_t outputOffset =
        ((static_cast<uint64_t>(row) * args.ldc + colFirst) << 1);
    __gm__ float *output = args.matC + outputOffset;
    if constexpr (BetaOne) {
        const __gm__ float *input = args.initialC + outputOffset;
        acc0Real += input[0];
        acc0Imag += input[1];
        if (fullTile) {
            acc1Real += input[2];
            acc1Imag += input[3];
        }
    }
    output[0] = acc0Real;
    output[1] = acc0Imag;
    if (fullTile) {
        output[2] = acc1Real;
        output[3] = acc1Imag;
    }
}

template<bool BetaOne, bool IdentityReorder>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmCsrSimtComputeComplexRowMajor2(
    __gm__ const int32_t *rowOff,
    __gm__ const int32_t *colInd,
    __gm__ const float *values,
    __gm__ const float *matB,
    __gm__ const float *initialC,
    __gm__ float *matC,
    __gm__ const int32_t *reorder,
    SpmmComplexPairShape shape,
    int32_t logicalRowStart,
    int32_t logicalRowEnd,
    int32_t indexBase)
{
    const int32_t numRows = logicalRowEnd - logicalRowStart;
    if (numRows <= 0 || shape.n <= 0) {
        return;
    }
    const int32_t colTiles = (shape.n + 1) / 2;
    const uint64_t totalWork = static_cast<uint64_t>(numRows) *
        static_cast<uint64_t>(colTiles);
    SpmmSimtWork work;
    if (!GetSpmmSimtWork(totalWork, colTiles, logicalRowStart, &work)) {
        return;
    }
    int32_t logicalRow = work.logicalRow;
    int32_t colTile = work.colTile;

    const SpmmComplexPairKernelArgs args{rowOff, colInd, values, matB,
        initialC, matC, reorder, shape.n, shape.ldb, shape.ldc, indexBase};
    while (logicalRow < logicalRowEnd) {
        ComputeSpmmComplexPairTile<BetaOne, IdentityReorder>(
            args, logicalRow, colTile * 2);

        AdvanceSpmmCsrTile(&logicalRow, &colTile, work.rowAdvance,
            work.tileAdvance, colTiles);
    }
}

__simt_callee__ __aicore__ inline void
ComputeSpmmFp32RowMajor4IdentityTile(__gm__ const int32_t *colInd,
    __gm__ const float *values, __gm__ const float *matB,
    __gm__ const float *initialC, __gm__ float *matC, int32_t row,
    int32_t colFirst, int32_t n, int32_t ldb, int32_t ldc,
    int32_t uniformDegree, int32_t indexBase)
{
    const int32_t start = row * uniformDegree;
    const int32_t end = start + uniformDegree;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int32_t p = start; p < end; ++p) {
        const int32_t column = colInd[p] - indexBase;
        const __gm__ float *b = matB + static_cast<uint64_t>(column) * ldb + colFirst;
        const float value = values[p];
        if (colFirst + 3 < n) {
            acc0 += value * b[0]; acc1 += value * b[1];
            acc2 += value * b[2]; acc3 += value * b[3];
        } else {
            if (colFirst < n) acc0 += value * b[0];
            if (colFirst + 1 < n) acc1 += value * b[1];
            if (colFirst + 2 < n) acc2 += value * b[2];
            if (colFirst + 3 < n) acc3 += value * b[3];
        }
    }
    const uint64_t output = static_cast<uint64_t>(row) * ldc + colFirst;
    if (colFirst < n) matC[output] = acc0 + initialC[output];
    if (colFirst + 1 < n) matC[output + 1] = acc1 + initialC[output + 1];
    if (colFirst + 2 < n) matC[output + 2] = acc2 + initialC[output + 2];
    if (colFirst + 3 < n) matC[output + 3] = acc3 + initialC[output + 3];
}

__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmCsrSimtComputeFp32RowMajor4Identity(
    __gm__ const int32_t *colInd,
    __gm__ const float *values,
    __gm__ const float *matB,
    __gm__ const float *initialC,
    __gm__ float *matC,
    int32_t n,
    int32_t ldb,
    int32_t ldc,
    int32_t colTiles,
    int32_t logicalRowStart,
    int32_t logicalRowEnd,
    int32_t uniformDegree,
    int32_t indexBase)
{
    SpmmSimtWork work;
    if (!GetSpmmCsrTileWork(logicalRowStart, logicalRowEnd, colTiles, &work)) {
        return;
    }
    int32_t row = work.logicalRow;
    int32_t colTile = work.colTile;

    while (row < logicalRowEnd) {
        const int32_t colFirst = colTile * 4;
        ComputeSpmmFp32RowMajor4IdentityTile(
            colInd, values, matB, initialC, matC, row, colFirst, n, ldb, ldc,
            uniformDegree, indexBase);

        AdvanceSpmmCsrTile(&row, &colTile, work.rowAdvance, work.tileAdvance,
            colTiles);
    }
}

// The fixed P-01 path has beta=0 and near-uniform 3/4-entry rows.  Keep its
// write-only case separate from the beta=1 helper: the generic kernel still
// carries scalar and initial-C branches in every output tile.  This helper
// uses the CSR row offsets directly, so it also handles the short remainder
// rows without assuming one uniform degree.
__simt_callee__ __aicore__ inline void ComputeSpmmFp32BetaZeroTile(
    __gm__ const int32_t *rowOff, __gm__ const int32_t *colInd,
    __gm__ const float *values, __gm__ const float *matB, __gm__ float *matC,
    int32_t row, int32_t colFirst, int32_t n, int32_t ldb, int32_t ldc,
    int32_t indexBase)
{
    const int32_t start = rowOff[row] - indexBase;
    const int32_t end = rowOff[row + 1] - indexBase;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int32_t p = start; p < end; ++p) {
        const __gm__ float *b = matB +
            static_cast<uint64_t>(colInd[p] - indexBase) * ldb + colFirst;
        const float value = values[p];
        if (colFirst < n) acc0 += value * b[0];
        if (colFirst + 1 < n) acc1 += value * b[1];
        if (colFirst + 2 < n) acc2 += value * b[2];
        if (colFirst + 3 < n) acc3 += value * b[3];
    }
    const uint64_t output = static_cast<uint64_t>(row) * ldc + colFirst;
    if (colFirst < n) matC[output] = acc0;
    if (colFirst + 1 < n) matC[output + 1] = acc1;
    if (colFirst + 2 < n) matC[output + 2] = acc2;
    if (colFirst + 3 < n) matC[output + 3] = acc3;
}

__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmCsrSimtComputeFp32RowMajor4BetaZero(
    __gm__ const int32_t *rowOff,
    __gm__ const int32_t *colInd,
    __gm__ const float *values,
    __gm__ const float *matB,
    __gm__ float *matC,
    __gm__ const int32_t *reorder,
    int32_t n,
    int32_t ldb,
    int32_t ldc,
    int32_t colTiles,
    int32_t logicalRowStart,
    int32_t logicalRowEnd,
    int32_t indexBase)
{
    const int32_t numRows = logicalRowEnd - logicalRowStart;
    if (numRows <= 0 || colTiles <= 0) {
        return;
    }
    const uint64_t totalWork = static_cast<uint64_t>(numRows) *
        static_cast<uint64_t>(colTiles);
    SpmmSimtWork work;
    if (!GetSpmmSimtWork(totalWork, colTiles, logicalRowStart, &work)) {
        return;
    }
    int32_t logicalRow = work.logicalRow;
    int32_t colTile = work.colTile;

    while (logicalRow < logicalRowEnd) {
        const int32_t row = reorder[logicalRow];
        const int32_t colFirst = colTile * 4;
        ComputeSpmmFp32BetaZeroTile(rowOff, colInd, values, matB, matC, row,
            colFirst, n, ldb, ldc, indexBase);

        AdvanceSpmmCsrTile(&logicalRow, &colTile, work.rowAdvance,
            work.tileAdvance, colTiles);
    }
}

// ============================================================================
// Templated kernel class: KernelSpmmSimt<ValT, CT, ScalarT, UseKahan>
//
// UseKahan=true  => fp32 high-precision (Kahan compensated summation) path.
// UseKahan=false => default high-performance (standard accumulation) path.
//
// Each template instantiation is a fully typed, compile-time resolved class.
// No runtime dataType dispatch — the correct instantiation is selected by
// the per-type kernel entry point (spmm_custom_fp32 / fp16 / int8).
// ============================================================================
template<typename ValT, typename CT, typename ScalarT, bool UseKahan, int32_t Chunk>
class KernelSpmmSimt {
public:
    __aicore__ inline KernelSpmmSimt() {}

    __aicore__ inline void Init(GM_ADDR csrRowOffsets,
        GM_ADDR csrColInd,
        GM_ADDR csrValues,
        GM_ADDR matB,
        GM_ADDR matC,
        GM_ADDR workspaceGM,
        GM_ADDR tilingGM)
    {
        Init(csrRowOffsets, csrColInd, csrValues, matB, matC, matC,
             workspaceGM, tilingGM);
    }

    __aicore__ inline void Init(GM_ADDR csrRowOffsets,
        GM_ADDR csrColInd,
        GM_ADDR csrValues,
        GM_ADDR matB,
        GM_ADDR initialC,
        GM_ADDR matC,
        GM_ADDR workspaceGM,
        GM_ADDR tilingGM)
    {
        tilingData_ = LoadSpmmKernelTilingData(tilingGM);
        wsBase_ = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        rowOff_ = reinterpret_cast<__gm__ const int32_t *>(csrRowOffsets);
        colInd_ = reinterpret_cast<__gm__ const int32_t *>(csrColInd);
        values_ = reinterpret_cast<__gm__ const ValT *>(csrValues);
        matB_ = reinterpret_cast<__gm__ const ValT *>(matB);
        initialC_ = reinterpret_cast<__gm__ const CT *>(initialC);
        matC_ = reinterpret_cast<__gm__ CT *>(matC);
        reorder_ = reinterpret_cast<__gm__ const int32_t *>(
            wsBase_ + static_cast<uint64_t>(tilingData_.reorder_offset));
        binEdge_ = reinterpret_cast<__gm__ int32_t *>(
            wsBase_ + static_cast<uint64_t>(tilingData_.bin_edge_offset));
        n_ = tilingData_.n;
        ldb_ = tilingData_.ldb;
        ldc_ = tilingData_.ldc;
        opB_ = tilingData_.opB;
        indexBase_ = tilingData_.index_base;
        orderPair_ = tilingData_.order_pair;
        colTiles_ = static_cast<int32_t>(
            (static_cast<uint64_t>(tilingData_.n) + static_cast<uint64_t>(Chunk) - 1u) /
            static_cast<uint64_t>(Chunk));
    }

    __aicore__ inline void Process()
    {
        ScalarT alpha;
        ScalarT beta;
        LoadSpmmScalarPair(tilingData_, &alpha, &beta);
        // IMPORTANT: do NOT early-return for out-of-range / empty bins here.
        //
        // On dav-3510 (split AIC/AIV core), asc_vf_call -> cce::async_invoke
        // dispatches the inner SIMT function from each outer AIC core to its
        // paired AIV vector core. That cube->vector dispatch must be issued
        // UNIFORMLY by every outer core. When m < blockDim (e.g. m=2, blockDim=56)
        // most row bins are empty; if those cores returned before asc_vf_call the
        // cube/vector handshake would deadlock and aclrtSynchronizeStream would
        // hang forever (observed as a random mid-batch freeze, all dtypes).
        //
        // Instead, every core falls through to asc_vf_call. Empty bins pass
        // rowStart == rowEnd, which makes SpmmCsrSimtCompute return immediately
        // (numRows <= 0) without touching memory, so they do no work but still
        // participate in the dispatch.
        const SpmmSimtLaunchInfo launch = GetSpmmSimtLaunchInfo(
            tilingData_, binEdge_, colTiles_);
        if (TryRunFp16Pair(launch, alpha, beta)) return;
        if (TryRunFp32FastPath(launch, alpha, beta)) return;
        RunGeneric(launch, alpha, beta);
    }

private:
    __aicore__ inline bool TryRunFp16Pair(
        const SpmmSimtLaunchInfo &launch, ScalarT alpha, ScalarT beta)
    {
        if constexpr (SpmmIsFp16<ValT>::value && SpmmIsFp16<CT>::value) {
            if (opB_ == 0 && orderPair_ == SPMM_ORDER_RR &&
                (ldb_ & 1) == 0 && (ldc_ & 1) == 0) {
                asc_vf_call<SpmmCsrSimtComputeFp16PairProduct<kFp16PairChunk>>(
                    dim3{launch.threadCount}, rowOff_, colInd_,
                    reinterpret_cast<__gm__ const half *>(values_),
                    reinterpret_cast<__gm__ const half *>(matB_),
                    reinterpret_cast<__gm__ half *>(matC_), reorder_, n_, ldb_, ldc_,
                    colTiles_, launch.rowStart, launch.rowEnd, alpha, beta, indexBase_,
                    tilingData_.sparse_empty);
                return true;
            }
        }
        return false;
    }

    __aicore__ inline bool TryRunFp32FastPath(
        const SpmmSimtLaunchInfo &launch, ScalarT alpha, ScalarT beta)
    {
        if constexpr (!UseKahan && Chunk == 4 &&
                      SpmmIsSame<ValT, float>::value &&
                      SpmmIsSame<CT, float>::value &&
                      SpmmIsSame<ScalarT, float>::value) {
            const int64_t nnz = tilingData_.m > 0 ?
                static_cast<int64_t>(rowOff_[tilingData_.m]) - indexBase_ : 0;
            if (CanRunFp32IdentityFastPath(nnz, alpha, beta)) {
                asc_vf_call<SpmmCsrSimtComputeFp32RowMajor4Identity>(
                    dim3{launch.threadCount}, colInd_, values_, matB_, initialC_,
                    matC_, n_, ldb_, ldc_, colTiles_, launch.rowStart, launch.rowEnd,
                    tilingData_.uniform_degree, indexBase_);
                return true;
            }
            if (CanRunFp32BetaZeroFastPath(alpha, beta)) {
                asc_vf_call<SpmmCsrSimtComputeFp32RowMajor4BetaZero>(
                    dim3{launch.threadCount}, rowOff_, colInd_, values_, matB_,
                    matC_, reorder_, n_, ldb_, ldc_, colTiles_, launch.rowStart,
                    launch.rowEnd, indexBase_);
                return true;
            }
        }
        return false;
    }

    __aicore__ inline bool CanRunFp32IdentityFastPath(
        int64_t nnz, float alpha, float beta) const
    {
        return tilingData_.row_reorder_identity != 0 &&
            tilingData_.uniform_degree >= 0 && tilingData_.uniform_degree <= 8 &&
            nnz >= 0 && nnz <= static_cast<int64_t>(tilingData_.m) * 8 &&
            tilingData_.sparse_empty == 0 && opB_ == 0 &&
            orderPair_ == SPMM_ORDER_RR && alpha == 1.0f && beta == 1.0f;
    }

    __aicore__ inline bool CanRunFp32BetaZeroFastPath(
        float alpha, float beta) const
    {
        return tilingData_.sparse_empty == 0 && opB_ == 0 &&
            orderPair_ == SPMM_ORDER_RR && alpha == 1.0f && beta == 0.0f;
    }

    __aicore__ inline void RunGeneric(
        const SpmmSimtLaunchInfo &launch, ScalarT alpha, ScalarT beta)
    {
        if constexpr (SpmmIsSame<ValT, float>::value &&
                      SpmmIsSame<CT, float>::value &&
                      SpmmIsSame<ScalarT, float>::value) {
            if (tilingData_.row_reorder_identity != 0) {
                asc_vf_call<SpmmCsrSimtCompute<ValT, CT, ScalarT, UseKahan,
                    Chunk, true>>(dim3{launch.threadCount}, rowOff_, colInd_, values_,
                    matB_, initialC_, matC_, reorder_, n_, ldb_, ldc_, colTiles_,
                    launch.rowStart, launch.rowEnd, alpha, beta, opB_, orderPair_,
                    indexBase_, tilingData_.sparse_empty);
                return;
            }
        }
        asc_vf_call<SpmmCsrSimtCompute<ValT, CT, ScalarT, UseKahan,
            Chunk, false>>(dim3{launch.threadCount}, rowOff_, colInd_, values_,
            matB_, initialC_, matC_, reorder_, n_, ldb_, ldc_, colTiles_,
            launch.rowStart, launch.rowEnd, alpha, beta, opB_, orderPair_, indexBase_,
            tilingData_.sparse_empty);
    }

    __gm__ uint8_t *wsBase_{nullptr};
    __gm__ const int32_t *rowOff_{nullptr};
    __gm__ const int32_t *colInd_{nullptr};
    __gm__ const ValT *values_{nullptr};
    __gm__ const ValT *matB_{nullptr};
    __gm__ const CT *initialC_{nullptr};
    __gm__ CT *matC_{nullptr};
    __gm__ const int32_t *reorder_{nullptr};
    __gm__ int32_t *binEdge_{nullptr};
    int32_t n_{0};
    int32_t ldb_{0};
    int32_t ldc_{0};
    int32_t opB_{0};
    int32_t indexBase_{0};
    int32_t orderPair_{0};
    int32_t colTiles_{0};
    SpmmKernelTilingData tilingData_{};
};

struct SpmmSimtComplexInitArgs {
    GM_ADDR csrRowOffsets;
    GM_ADDR csrColInd;
    GM_ADDR csrValues;
    GM_ADDR matB;
    GM_ADDR initialC;
    GM_ADDR matC;
    GM_ADDR workspaceGM;
    GM_ADDR tilingGM;
};

template<int32_t Chunk>
class KernelSpmmSimtComplex {
public:
    __aicore__ inline KernelSpmmSimtComplex() {}

    __aicore__ inline void Init(const SpmmSimtComplexInitArgs &args)
    {
        tilingData_ = LoadSpmmKernelTilingData(args.tilingGM);
        wsBase_ = reinterpret_cast<__gm__ uint8_t *>(args.workspaceGM);
        rowOff_ = reinterpret_cast<__gm__ const int32_t *>(args.csrRowOffsets);
        colInd_ = reinterpret_cast<__gm__ const int32_t *>(args.csrColInd);
        values_ = reinterpret_cast<__gm__ const float *>(args.csrValues);
        matB_ = reinterpret_cast<__gm__ const float *>(args.matB);
        initialC_ = reinterpret_cast<__gm__ const float *>(args.initialC);
        matC_ = reinterpret_cast<__gm__ float *>(args.matC);
        reorder_ = reinterpret_cast<__gm__ const int32_t *>(
            wsBase_ + static_cast<uint64_t>(tilingData_.reorder_offset));
        binEdge_ = reinterpret_cast<__gm__ int32_t *>(
            wsBase_ + static_cast<uint64_t>(tilingData_.bin_edge_offset));
        colTiles_ = static_cast<int32_t>(
            (static_cast<uint64_t>(tilingData_.n) + static_cast<uint64_t>(Chunk) - 1u) /
            static_cast<uint64_t>(Chunk));
    }

    __aicore__ inline void Process()
    {
        SpmmComplex64 alpha;
        SpmmComplex64 beta;
        LoadSpmmScalarPair(tilingData_, &alpha, &beta);
        const SpmmSimtLaunchInfo launch = GetSpmmSimtLaunchInfo(
            tilingData_, binEdge_, colTiles_);

        if (TryProcessFastLayout(launch, alpha, beta)) {
            return;
        }

        asc_vf_call<SpmmCsrSimtComputeComplex<Chunk>>(dim3{launch.threadCount},
            rowOff_, colInd_, values_, matB_, initialC_, matC_, reorder_, tilingData_.n,
            tilingData_.ldb, tilingData_.ldc, colTiles_, launch.rowStart, launch.rowEnd,
            alpha.real, alpha.imag, beta.real, beta.imag, tilingData_.opB,
            tilingData_.order_pair, tilingData_.index_base, tilingData_.sparse_empty);
    }

private:
    __aicore__ inline bool TryProcessFastLayout(
        const SpmmSimtLaunchInfo &launch, SpmmComplex64 alpha,
        SpmmComplex64 beta)
    {
        const bool evenColumns = (tilingData_.n & 1) == 0;
        const bool oddTailEligible = IsOddTailEligible(evenColumns);
        const bool fastLayout = tilingData_.opB == 0 &&
            tilingData_.order_pair == SPMM_ORDER_RR && tilingData_.n > 0 &&
            tilingData_.sparse_empty == 0 &&
            (evenColumns || oddTailEligible);
        const bool alphaOne = alpha.real == 1.0f && alpha.imag == 0.0f;
        const bool betaZero = SpmmComplexIsZero(beta);
        const bool betaOne = beta.real == 1.0f && beta.imag == 0.0f;
        if (fastLayout && alphaOne && (betaZero || betaOne)) {
            if (betaZero) {
                if (tilingData_.row_reorder_identity != 0) {
                    asc_vf_call<SpmmCsrSimtComputeComplexRowMajor2<false, true>>(
                        dim3{launch.threadCount}, rowOff_, colInd_, values_, matB_, initialC_,
                        matC_, reorder_, SpmmComplexPairShape{tilingData_.n,
                            tilingData_.ldb, tilingData_.ldc},
                        launch.rowStart, launch.rowEnd, tilingData_.index_base);
                } else {
                    asc_vf_call<SpmmCsrSimtComputeComplexRowMajor2<false, false>>(
                        dim3{launch.threadCount}, rowOff_, colInd_, values_, matB_, initialC_,
                        matC_, reorder_, SpmmComplexPairShape{tilingData_.n,
                            tilingData_.ldb, tilingData_.ldc},
                        launch.rowStart, launch.rowEnd, tilingData_.index_base);
                }
            } else {
                if (tilingData_.row_reorder_identity != 0) {
                    asc_vf_call<SpmmCsrSimtComputeComplexRowMajor2<true, true>>(
                        dim3{launch.threadCount}, rowOff_, colInd_, values_, matB_, initialC_,
                        matC_, reorder_, SpmmComplexPairShape{tilingData_.n,
                            tilingData_.ldb, tilingData_.ldc},
                        launch.rowStart, launch.rowEnd, tilingData_.index_base);
                } else {
                    asc_vf_call<SpmmCsrSimtComputeComplexRowMajor2<true, false>>(
                        dim3{launch.threadCount}, rowOff_, colInd_, values_, matB_, initialC_,
                        matC_, reorder_, SpmmComplexPairShape{tilingData_.n,
                            tilingData_.ldb, tilingData_.ldc},
                        launch.rowStart, launch.rowEnd, tilingData_.index_base);
                }
            }
            return true;
        }
        return false;
    }

    __aicore__ inline bool IsOddTailEligible(bool evenColumns) const
    {
        if (evenColumns || tilingData_.m <= 0) {
            return false;
        }
        const int64_t nnz = tilingData_.sparse_empty != 0
            ? 0
            : static_cast<int64_t>(rowOff_[tilingData_.m]) -
                  static_cast<int64_t>(tilingData_.index_base);
        return nnz >= 0 && nnz <= static_cast<int64_t>(tilingData_.m) * 8;
    }

    __gm__ uint8_t *wsBase_{nullptr};
    __gm__ const int32_t *rowOff_{nullptr};
    __gm__ const int32_t *colInd_{nullptr};
    __gm__ const float *values_{nullptr};
    __gm__ const float *matB_{nullptr};
    __gm__ const float *initialC_{nullptr};
    __gm__ float *matC_{nullptr};
    __gm__ const int32_t *reorder_{nullptr};
    __gm__ int32_t *binEdge_{nullptr};
    int32_t colTiles_{0};
    SpmmKernelTilingData tilingData_{};
};

extern "C" __global__ __aicore__ void spmm_custom_fp32(
    GM_ADDR csrRowOffsets,
    GM_ADDR csrColInd,
    GM_ADDR csrValues,
    GM_ADDR matB,
    GM_ADDR initialC,
    GM_ADDR matC,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData tiling = LoadSpmmKernelTilingData(tilingGM);
    if (tiling.high_precision != 0) {
        KernelSpmmSimt<float, float, float, true, kRealChunk> op;
        op.Init(csrRowOffsets, csrColInd, csrValues, matB, initialC, matC,
                workspaceGM, tilingGM);
        op.Process();
    } else {
        KernelSpmmSimt<float, float, float, false, kRealChunk> op;
        op.Init(csrRowOffsets, csrColInd, csrValues, matB, initialC, matC,
                workspaceGM, tilingGM);
        op.Process();
    }
}
extern "C" __global__ __aicore__ void spmm_custom_fp16(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    KernelSpmmSimt<__fp16, __fp16, float, false, kRealChunk> op;
    op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
    op.Process();
}
extern "C" __global__ __aicore__ void spmm_custom_int8(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    KernelSpmmSimt<int8_t, int32_t, int32_t, false, kIntChunk> op;
    op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
    op.Process();
}
extern "C" __global__ __aicore__ void spmm_custom_bf16(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    KernelSpmmSimt<bfloat16_t, bfloat16_t, float, false, kRealChunk> op;
    op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
    op.Process();
}
extern "C" __global__ __aicore__ void spmm_custom_complex64(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    KernelSpmmSimtComplex<kComplexChunk> op;
    op.Init({csrRowOffsets, csrColInd, csrValues, matB, initialC, matC,
        workspaceGM, tilingGM});
    op.Process();
}
extern "C" __global__ __aicore__ void spmm_custom_fp16_fp32(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    KernelSpmmSimt<__fp16, float, float, false, kRealChunk> op;
    op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
    op.Process();
}
extern "C" __global__ __aicore__ void spmm_custom_bf16_fp32(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    KernelSpmmSimt<bfloat16_t, float, float, false, kRealChunk> op;
    op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
    op.Process();
}
extern "C" __global__ __aicore__ void spmm_custom_complex64_conditional_fallback(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    const __gm__ int32_t *status = reinterpret_cast<__gm__ const int32_t *>(
        reinterpret_cast<__gm__ uint8_t *>(workspaceGM) +
        td.cube_complex_status_offset);
    if (status[0] == 0) {
        return;
    }
    KernelSpmmSimtComplex<kComplexChunk> op;
    op.Init({csrRowOffsets, csrColInd, csrValues, matB, initialC, matC,
        workspaceGM, tilingGM});
    op.Process();
}
