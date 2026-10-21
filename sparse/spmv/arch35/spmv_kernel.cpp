/*
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

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/math_functions.h"
#include "c_api/cache_ctrl/cache_ctrl.h"
#include "c_api/sync/sync.h"
#include "spmv_kernel.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
#error "A5 SpMV kernel requires dav-3510 / Ascend 950 (__NPU_ARCH__==3510)."
#endif

using namespace AscendC;

namespace {

constexpr uint32_t kMaxThreads = 128U;
constexpr uint32_t kGatherThreads = 256U;
constexpr uint32_t kValidatorThreads = 512U;
// Keep the device TU independent of the host-only public ACL header. These
// values are frozen by aclsparseOperation_t/aclsparseSpMVAlg_t in the public
// ABI and are copied into the POD tiling payload by Host code.
constexpr uint32_t kSpmvOpNonTranspose = 0U;
constexpr uint32_t kSpmvOpTranspose = 1U;
constexpr uint32_t kSpmvOpConjugateTranspose = 2U;
constexpr uint32_t kSpmvAlgCsrAlg2 = 4U;

// SIMT VF return values must be Ascend C built-in types.  float2 is the
// built-in two-lane scalar used here for complex64 (real=x, imag=y); half and
// bfloat16_t identify the public GM element formats.
using SpmvComplex64 = float2;
using SpmvHalfBits = half;
using SpmvBf16Bits = bfloat16_t;

inline bool SpmvIsTransposeOp(uint32_t op)
{
    return op == kSpmvOpTranspose || op == kSpmvOpConjugateTranspose;
}

// In the A5 SIMT path, direct GM loads/stores of half/bfloat16 are known to
// be compiler/runtime-sensitive. Keep the public tensor representation
// unchanged, but move those accesses through uint16_t and explicit IEEE
// conversion helpers. This mirrors the robust representation used by the
// arch35 SpGEMM kernel.
__aicore__ __simt_callee__ inline float SpmvHalfToFloat(uint16_t h)
{
    const uint32_t sign = (static_cast<uint32_t>(h) >> 15U) & 1U;
    const uint32_t exp = (static_cast<uint32_t>(h) >> 10U) & 0x1FU;
    const uint32_t mant = static_cast<uint32_t>(h) & 0x3FFU;
    uint32_t bits = 0U;
    if (exp == 0U) {
        if (mant == 0U) {
            bits = sign << 31U;
        } else {
            // A normalized subnormal mantissa has an implicit leading bit;
            // subtract one from the shift count when forming its exponent.
            int32_t shift = -1;
            uint32_t normalized = mant;
            while ((normalized & 0x400U) == 0U) {
                normalized <<= 1U;
                ++shift;
            }
            bits = (sign << 31U) |
                   (static_cast<uint32_t>(127 - 15 - shift) << 23U) |
                   ((normalized & 0x3FFU) << 13U);
        }
    } else if (exp == 0x1FU) {
        bits = (sign << 31U) | (0xFFU << 23U) | (mant << 13U);
    } else {
        bits = (sign << 31U) |
               ((exp + 127U - 15U) << 23U) | (mant << 13U);
    }
    float result = 0.0F;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

__aicore__ __simt_callee__ inline float SpmvBf16ToFloat(uint16_t bits)
{
    const uint32_t fp32Bits = static_cast<uint32_t>(bits) << 16U;
    float result = 0.0F;
    __builtin_memcpy(&result, &fp32Bits, sizeof(result));
    return result;
}

__aicore__ __simt_callee__ inline uint16_t SpmvFloatToHalf(float value)
{
    uint32_t bits = 0U;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 31U) & 1U;
    const int32_t exp = static_cast<int32_t>((bits >> 23U) & 0xFFU);
    const uint32_t mant = bits & 0x7FFFFFU;
    if (exp == 0xFF) {
        return static_cast<uint16_t>((sign << 15U) | 0x7C00U | (mant != 0U ? 1U : 0U));
    }
    int32_t halfExp = exp - 127 + 15;
    if (halfExp >= 0x1F) {
        return static_cast<uint16_t>((sign << 15U) | 0x7C00U);
    }
    if (halfExp <= 0) {
        if (halfExp < -10) {
            return static_cast<uint16_t>(sign << 15U);
        }
        const uint32_t normalized = mant | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - halfExp);
        const uint32_t roundBit = 1U << (shift - 1U);
        const uint32_t rounded = (normalized + roundBit - 1U +
                                  ((normalized >> shift) & 1U)) >> shift;
        // Rounding a subnormal up across the boundary produces the smallest
        // normal value (exponent 1, zero mantissa).
        if (rounded >= 0x400U) {
            return static_cast<uint16_t>((sign << 15U) | 0x0400U);
        }
        return static_cast<uint16_t>((sign << 15U) | rounded);
    }
    const uint32_t roundLsb = (mant >> 13U) & 1U;
    uint32_t rounded = (mant + 0x0FFFU + roundLsb) >> 13U;
    if (rounded > 0x3FFU) {
        rounded = 0U;
        ++halfExp;
    }
    if (halfExp >= 0x1F) {
        return static_cast<uint16_t>((sign << 15U) | 0x7C00U);
    }
    return static_cast<uint16_t>((sign << 15U) |
                                 (static_cast<uint32_t>(halfExp) << 10U) | rounded);
}

__aicore__ __simt_callee__ inline uint16_t SpmvFloatToBf16(float value)
{
    uint32_t bits = 0U;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + rounding) >> 16U);
}

template <typename T>
__simt_callee__ inline T SpmvZero()
{
    return static_cast<T>(0);
}

template <>
__simt_callee__ inline SpmvComplex64 SpmvZero<SpmvComplex64>()
{
    return make_float2(0.0f, 0.0f);
}

template <typename T>
__simt_callee__ inline T SpmvAdd(T a, T b)
{
    return a + b;
}

template <typename T>
__simt_callee__ inline bool SpmvIsZero(T value)
{
    return value == static_cast<T>(0);
}

template <>
__simt_callee__ inline bool SpmvIsZero<SpmvComplex64>(SpmvComplex64 value)
{
    return value.x == 0.0f && value.y == 0.0f;
}

template <>
__simt_callee__ inline SpmvComplex64 SpmvAdd(SpmvComplex64 a, SpmvComplex64 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

template <typename T>
__simt_callee__ inline T SpmvMul(T a, T b)
{
    return a * b;
}

template <>
__simt_callee__ inline SpmvComplex64 SpmvMul(SpmvComplex64 a, SpmvComplex64 b)
{
    return make_float2(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
}

template <typename T>
__simt_callee__ inline T SpmvMaybeConjugate(T value, uint32_t op)
{
    (void)op;
    return value;
}

template <>
__simt_callee__ inline SpmvComplex64 SpmvMaybeConjugate(
    SpmvComplex64 value, uint32_t op)
{
    return op == kSpmvOpConjugateTranspose
        ? make_float2(value.x, -value.y) : value;
}

template <typename AccT, typename ValT>
__simt_callee__ inline AccT SpmvToAcc(ValT value)
{
    return static_cast<AccT>(value);
}

template <>
__simt_callee__ inline SpmvComplex64 SpmvToAcc<SpmvComplex64, float>(float value)
{
    return make_float2(value, 0.0f);
}

__simt_callee__ inline uint64_t SpmvPhysicalIndex(uint64_t index, int64_t stride)
{
    return index * static_cast<uint64_t>(stride);
}

__simt_callee__ inline int64_t SpmvLoadRowOffset(
    __gm__ const int32_t *rowPtr, __gm__ const SpmvArch35TilingData *t,
    uint64_t row)
{
    return static_cast<int64_t>(rowPtr[SpmvPhysicalIndex(row, t->rowOffsetsStride)]) -
           static_cast<int64_t>(t->rowPtrBase);
}

template <typename AccT, typename ValT>
__simt_callee__ inline AccT SpmvLoadValue(__gm__ const ValT *values, uint64_t index)
{
    return SpmvToAcc<AccT>(values[index]);
}

template <>
__simt_callee__ inline float SpmvLoadValue<float, SpmvHalfBits>(
    __gm__ const SpmvHalfBits *values, uint64_t index)
{
    const auto *raw = reinterpret_cast<__gm__ const uint16_t *>(values);
    return SpmvHalfToFloat(raw[index]);
}

template <>
__simt_callee__ inline float SpmvLoadValue<float, SpmvBf16Bits>(
    __gm__ const SpmvBf16Bits *values, uint64_t index)
{
    const auto *raw = reinterpret_cast<__gm__ const uint16_t *>(values);
    return SpmvBf16ToFloat(raw[index]);
}

template <typename AccT, typename OutT>
__simt_callee__ inline AccT SpmvLoadOutput(__gm__ const OutT *values, uint64_t index)
{
    return SpmvLoadValue<AccT, OutT>(values, index);
}

template <>
__simt_callee__ inline float SpmvLoadOutput<float, SpmvHalfBits>(
    __gm__ const SpmvHalfBits *values, uint64_t index)
{
    const auto *raw = reinterpret_cast<__gm__ const uint16_t *>(values);
    return SpmvHalfToFloat(raw[index]);
}

template <>
__simt_callee__ inline float SpmvLoadOutput<float, SpmvBf16Bits>(
    __gm__ const SpmvBf16Bits *values, uint64_t index)
{
    const auto *raw = reinterpret_cast<__gm__ const uint16_t *>(values);
    return SpmvBf16ToFloat(raw[index]);
}

template <typename AccT>
__simt_callee__ inline AccT SpmvLoadAlpha(
    __gm__ const SpmvArch35TilingData *t)
{
    if constexpr (std::is_same_v<AccT, int64_t>) {
        return static_cast<int64_t>(t->alphaPtr != 0U
            ? *reinterpret_cast<__gm__ const int32_t *>(t->alphaPtr) : t->alphaInt);
    } else if constexpr (std::is_same_v<AccT, SpmvComplex64>) {
        if (t->alphaPtr != 0U) {
            const auto *src = reinterpret_cast<__gm__ const float *>(t->alphaPtr);
            return make_float2(src[0], src[1]);
        }
        return make_float2(t->alphaReal, t->alphaImag);
    } else {
        return t->alphaPtr != 0U
            ? *reinterpret_cast<__gm__ const float *>(t->alphaPtr) : t->alphaReal;
    }
}

template <typename AccT>
__simt_callee__ inline AccT SpmvLoadBeta(
    __gm__ const SpmvArch35TilingData *t)
{
    if constexpr (std::is_same_v<AccT, int64_t>) {
        return static_cast<int64_t>(t->betaPtr != 0U
            ? *reinterpret_cast<__gm__ const int32_t *>(t->betaPtr) : t->betaInt);
    } else if constexpr (std::is_same_v<AccT, SpmvComplex64>) {
        if (t->betaPtr != 0U) {
            const auto *src = reinterpret_cast<__gm__ const float *>(t->betaPtr);
            return make_float2(src[0], src[1]);
        }
        return make_float2(t->betaReal, t->betaImag);
    } else {
        return t->betaPtr != 0U
            ? *reinterpret_cast<__gm__ const float *>(t->betaPtr) : t->betaReal;
    }
}

template <typename OutT, typename AccT>
__simt_callee__ inline OutT SpmvFromAcc(AccT value)
{
    return static_cast<OutT>(value);
}

template <>
__simt_callee__ inline int32_t SpmvFromAcc<int32_t, int64_t>(int64_t value)
{
    constexpr int64_t maxValue = 2147483647LL;
    constexpr int64_t minValue = -2147483648LL;
    return static_cast<int32_t>(value > maxValue ? maxValue : value < minValue ? minValue : value);
}

template <typename AccT>
__simt_callee__ inline AccT SpmvScaledResult(AccT sum, AccT alpha, AccT beta, AccT oldY)
{
    return SpmvIsZero(beta) ? SpmvMul(alpha, sum)
        : SpmvAdd(SpmvMul(alpha, sum), SpmvMul(beta, oldY));
}

template <>
__simt_callee__ inline int64_t SpmvScaledResult<int64_t>(
    int64_t sum, int64_t alpha, int64_t beta, int64_t oldY)
{
    constexpr int64_t maxValue = 2147483647LL;
    constexpr int64_t minValue = -2147483648LL;
    constexpr int64_t maxInt64 = 9223372036854775807LL;
    const int64_t betaTerm = beta * oldY;
    if (alpha == 0) {
        return SpmvFromAcc<int32_t>(betaTerm);
    }
    // INT8 dot products have at most INT32_MAX entries: |sum| < 2^45.
    const int64_t magnitude = sum < 0 ? -sum : sum;
    const int64_t scale = alpha < 0 ? -alpha : alpha;
    const bool negative = (sum < 0) != (alpha < 0);
    if (magnitude > maxInt64 / scale) {
        // |beta * oldY| <= 2^62 cannot cancel this into INT32 range.
        return negative ? minValue : maxValue;
    }
    const int64_t product = negative ? -(magnitude * scale) : magnitude * scale;
    // Shift the bounds to preserve cancellation without an overflowing add.
    if (product > maxValue - betaTerm) {
        return maxValue;
    }
    if (product < minValue - betaTerm) {
        return minValue;
    }
    return product + betaTerm;
}

template <>
__simt_callee__ inline SpmvComplex64 SpmvFromAcc<SpmvComplex64, SpmvComplex64>(SpmvComplex64 value)
{
    return value;
}

template <typename OutT, typename AccT>
__simt_callee__ inline void SpmvStoreValue(__gm__ OutT *values, uint64_t index, AccT value)
{
    values[index] = SpmvFromAcc<OutT>(value);
}

template <>
__simt_callee__ inline void SpmvStoreValue<SpmvHalfBits, float>(
    __gm__ SpmvHalfBits *values, uint64_t index, float value)
{
    auto *raw = reinterpret_cast<__gm__ uint16_t *>(values);
    raw[index] = SpmvFloatToHalf(value);
}

template <>
__simt_callee__ inline void SpmvStoreValue<SpmvBf16Bits, float>(
    __gm__ SpmvBf16Bits *values, uint64_t index, float value)
{
    auto *raw = reinterpret_cast<__gm__ uint16_t *>(values);
    raw[index] = SpmvFloatToBf16(value);
}

template <typename AccT>
__simt_callee__ inline void SpmvStoreTransposeAtomic(
    __gm__ uint8_t *accBase, uint32_t index, AccT value)
{
    if constexpr (std::is_same_v<AccT, SpmvComplex64>) {
        auto *dst = reinterpret_cast<__gm__ float *>(accBase) + index * 2U;
        asc_atomic_add(dst, value.x);
        asc_atomic_add(dst + 1, value.y);
    } else if constexpr (std::is_same_v<AccT, int64_t>) {
        asc_atomic_add(reinterpret_cast<__gm__ int64_t *>(accBase) + index, value);
    } else {
        asc_atomic_add(reinterpret_cast<__gm__ float *>(accBase) + index, value);
    }
}

template <typename AccT>
__simt_callee__ inline AccT SpmvWarpReduceSum(AccT value)
{
    for (uint32_t offset = 16U; offset > 0U; offset >>= 1U) {
        value = SpmvAdd(value, asc_shfl_down(value, offset));
    }
    return value;
}

template <>
__simt_callee__ inline SpmvComplex64 SpmvWarpReduceSum(SpmvComplex64 value)
{
    for (uint32_t offset = 16U; offset > 0U; offset >>= 1U) {
        value.x += asc_shfl_down(value.x, offset);
        value.y += asc_shfl_down(value.y, offset);
    }
    return value;
}

__simt_callee__ inline void SpmvInvalidateTiling(
    __gm__ SpmvArch35TilingData *tiling)
{
    if (threadIdx.x == 0U) {
        asc_dcci_entire(reinterpret_cast<__gm__ void *>(tiling));
    }
}

__simt_callee__ inline float SpmvSmallFloatRow(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const float *values, __gm__ const float *x, __gm__ const float *y,
    __gm__ const SpmvArch35TilingData *t, uint32_t row, float alpha, float beta)
{
    const int64_t begin = SpmvLoadRowOffset(rowPtr, t, row);
    const int64_t end = SpmvLoadRowOffset(rowPtr, t, static_cast<uint64_t>(row) + 1U);
    float sum = 0.0f;
    float ordinarySum = 0.0f;
    float compensation = 0.0f;
    float productResidual = 0.0f;
    bool special = false;
    if (begin >= 0 && end >= begin) {
        for (int64_t p = begin; p < end && static_cast<uint64_t>(p) < t->nnz; ++p) {
            const int64_t col = static_cast<int64_t>(colInd[SpmvPhysicalIndex(p, t->colIndStride)]) - t->indexBase;
            if (col < 0 || static_cast<uint64_t>(col) >= t->cols) {
                continue;
            }
            const float a = values[SpmvPhysicalIndex(p, t->valuesStride)];
            const float b = x[SpmvPhysicalIndex(col, t->xStride)];
            const float product = a * b;
            ordinarySum += product;
            special = special || !isfinite(product) || !isfinite(ordinarySum);
            productResidual += fmaf(a, b, -product);
            const float adjusted = product - compensation;
            const float next = sum + adjusted;
            compensation = (next - sum) - adjusted;
            sum = next;
        }
    }
    if (special) {
        return SpmvIsZero(beta) ? alpha * ordinarySum
                                : alpha * ordinarySum + beta * y[SpmvPhysicalIndex(row, t->yStride)];
    }
    const float correction = alpha * (productResidual - compensation);
    return SpmvIsZero(beta) ? fmaf(alpha, sum, correction)
                            : fmaf(alpha, sum, beta * y[SpmvPhysicalIndex(row, t->yStride)]) + correction;
}

__simt_callee__ inline void SpmvNFloatSmall(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const float *values, __gm__ const float *x, __gm__ float *y,
    __gm__ const SpmvArch35TilingData *t, uint32_t coreIndex,
    uint32_t coreCount, float alpha, float beta)
{
    for (uint64_t row64 = static_cast<uint64_t>(coreIndex) * kMaxThreads + threadIdx.x;
         row64 < t->rows;
         row64 += static_cast<uint64_t>(coreCount) * kMaxThreads) {
        const uint32_t row = static_cast<uint32_t>(row64);
        y[SpmvPhysicalIndex(row, t->yStride)] = SpmvSmallFloatRow(rowPtr, colInd, values, x, y, t, row, alpha, beta);
    }
}

template <typename ValT, typename AccT>
__simt_callee__ inline AccT SpmvNWarpRow(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ const ValT *x,
    __gm__ const SpmvArch35TilingData *t, uint32_t row, uint32_t lane)
{
    constexpr uint32_t kWarpSize = 32U;
    const int64_t begin = SpmvLoadRowOffset(rowPtr, t, row);
    const int64_t end = SpmvLoadRowOffset(rowPtr, t, static_cast<uint64_t>(row) + 1U);
    AccT sum = SpmvZero<AccT>();
    if (begin < 0 || end < begin) {
        return sum;
    }
    for (int64_t p = begin + lane; p < end && static_cast<uint64_t>(p) < t->nnz;
         p += kWarpSize) {
        const int64_t rawCol = static_cast<int64_t>(colInd[SpmvPhysicalIndex(p, t->colIndStride)]) - t->indexBase;
        if (rawCol >= 0 && static_cast<uint64_t>(rawCol) < t->cols) {
            sum = SpmvAdd(sum, SpmvMul(
                SpmvLoadValue<AccT, ValT>(values, SpmvPhysicalIndex(p, t->valuesStride)),
                SpmvLoadValue<AccT, ValT>(x, SpmvPhysicalIndex(rawCol, t->xStride))));
        }
    }
    return sum;
}

template <typename ValT, typename OutT, typename AccT>
__simt_callee__ inline void SpmvNWarp(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ const ValT *x,
    __gm__ OutT *y, __gm__ const SpmvArch35TilingData *t,
    uint32_t coreIndex, uint32_t coreCount, AccT alpha, AccT beta)
{
    constexpr uint32_t kWarpSize = 32U;
    constexpr uint32_t kWarpsPerBlock = kMaxThreads / kWarpSize;
    const uint32_t lane = threadIdx.x & (kWarpSize - 1U);
    const uint32_t warp = threadIdx.x / kWarpSize;
    for (uint64_t row64 = static_cast<uint64_t>(coreIndex) * kWarpsPerBlock + warp;
         row64 < t->rows;
         row64 += static_cast<uint64_t>(coreCount) * kWarpsPerBlock) {
        const uint32_t row = static_cast<uint32_t>(row64);
        AccT sum = SpmvNWarpRow<ValT, AccT>(rowPtr, colInd, values, x, t, row, lane);
        sum = SpmvWarpReduceSum(sum);
        if (lane != 0U) {
            continue;
        }
        // Preserve the beta==0 contract: do not read the previous Y value in
        // this case (it may intentionally contain NaN or be an invalid GM
        // address in a scale-only call).
        const AccT oldY = SpmvIsZero(beta) ? SpmvZero<AccT>()
            : SpmvLoadOutput<AccT, OutT>(y, SpmvPhysicalIndex(row, t->yStride));
        const AccT result = SpmvScaledResult(sum, alpha, beta, oldY);
        SpmvStoreValue<OutT, AccT>(y, SpmvPhysicalIndex(row, t->yStride), result);
    }
}

template <typename ValT, typename OutT, typename AccT>
__simt_vf__ __aicore__ __launch_bounds__(kMaxThreads) inline void SpmvN(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ const ValT *x,
    __gm__ OutT *y, __gm__ SpmvArch35TilingData *t,
    uint32_t coreIndex, uint32_t coreCount)
{
    SpmvInvalidateTiling(t);
    asc_syncthreads();
    const AccT alpha = SpmvLoadAlpha<AccT>(t);
    const AccT beta = SpmvLoadBeta<AccT>(t);
    if constexpr (std::is_same_v<ValT, float> && std::is_same_v<OutT, float> &&
                  std::is_same_v<AccT, float>) {
        if (t->rows <= kSpmvFloatSmallRowsMax) {
            SpmvNFloatSmall(rowPtr, colInd, values, x, y, t,
                            coreIndex, coreCount, alpha, beta);
            return;
        }
    }
    SpmvNWarp<ValT, OutT, AccT>(rowPtr, colInd, values, x, y, t,
                                coreIndex, coreCount, alpha, beta);
}

template <typename ValT, typename AccT>
__simt_vf__ __aicore__ __launch_bounds__(kMaxThreads) inline void SpmvTransposeAtomic(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ const ValT *x,
    __gm__ uint8_t *workspace, __gm__ SpmvArch35TilingData *t,
    uint32_t coreIndex, uint32_t coreCount)
{
    SpmvInvalidateTiling(t);
    asc_syncthreads();
    // A zero-row CSR may legitimately have no row-pointer allocation while a
    // transpose still has a non-empty output vector.  Do not dereference the
    // absent row pointer in that case; the finalize pass will apply beta to Y.
    if (t->rows == 0U) {
        return;
    }
    const AccT alpha = SpmvLoadAlpha<AccT>(t);
    // rowsPerBlock is based on the output columns for transpose.  It cannot
    // be used to partition input rows when M != K; use a grid-stride walk over
    // the actual CSR row domain instead.
    for (uint64_t row64 = static_cast<uint64_t>(coreIndex) * kMaxThreads + threadIdx.x;
         row64 < t->rows;
         row64 += static_cast<uint64_t>(coreCount) * kMaxThreads) {
        const uint32_t row = static_cast<uint32_t>(row64);
        const int64_t begin = SpmvLoadRowOffset(rowPtr, t, row);
        const int64_t end = SpmvLoadRowOffset(rowPtr, t, static_cast<uint64_t>(row) + 1U);
        if (begin < 0 || end < begin) {
            continue;
        }
        const AccT xValue = SpmvLoadValue<AccT, ValT>(x, SpmvPhysicalIndex(row, t->xStride));
        for (int64_t p = begin; p < end &&
             static_cast<uint64_t>(p) < static_cast<uint64_t>(t->nnz); ++p) {
            const int64_t rawCol = static_cast<int64_t>(colInd[SpmvPhysicalIndex(p, t->colIndStride)]) -
                static_cast<int64_t>(t->indexBase);
            if (rawCol >= 0 && static_cast<uint64_t>(rawCol) < t->cols) {
                const AccT value = SpmvMaybeConjugate(
                    SpmvLoadValue<AccT, ValT>(values, SpmvPhysicalIndex(p, t->valuesStride)), t->opA);
                AccT contribution = SpmvMul(value, xValue);
                if constexpr (!std::is_same_v<AccT, int64_t>) {
                    contribution = SpmvMul(alpha, contribution);
                }
                SpmvStoreTransposeAtomic(workspace + t->accumulatorOffset,
                                          static_cast<uint32_t>(rawCol), contribution);
            }
        }
    }
}

template <typename ValT, typename OutT, typename AccT>
__simt_vf__ __aicore__ __launch_bounds__(kMaxThreads) inline void SpmvTransposeFinalize(
    __gm__ uint8_t *workspace, __gm__ OutT *y,
    __gm__ SpmvArch35TilingData *t, uint32_t coreIndex)
{
    // Atomic updates bypass the SIMT Data Cache.  Invalidate this block's
    // cache before ordinary loads, then make the result visible to all
    // threads in the block.  The preceding atomic kernel is on the same
    // stream, so no device-wide barrier is needed here.
    if (threadIdx.x == 0) {
        asc_dcci_entire(reinterpret_cast<__gm__ void *>(workspace));
    }
    asc_syncthreads();
    const auto *accBase = workspace + t->accumulatorOffset;
    const AccT beta = SpmvLoadBeta<AccT>(t);
    const uint64_t colBegin = static_cast<uint64_t>(coreIndex) * t->rowsPerBlock;
    const uint64_t colEnd = colBegin + t->rowsPerBlock;
    for (uint64_t col64 = colBegin + threadIdx.x;
         col64 < colEnd && col64 < t->cols;
         col64 += blockDim.x) {
        const uint32_t col = static_cast<uint32_t>(col64);
        AccT sum;
        if constexpr (std::is_same_v<AccT, SpmvComplex64>) {
            const auto *src = reinterpret_cast<__gm__ const float *>(accBase) + col * 2U;
            sum = make_float2(src[0], src[1]);
        } else if constexpr (std::is_same_v<AccT, int64_t>) {
            sum = reinterpret_cast<__gm__ const int64_t *>(accBase)[col];
        } else {
            sum = reinterpret_cast<__gm__ const float *>(accBase)[col];
        }
        // Integer contributions are unscaled to keep the accumulator in range.
        // Floating/complex contributions already include alpha.
        const AccT oldY = SpmvIsZero(beta) ? SpmvZero<AccT>()
            : SpmvLoadOutput<AccT, OutT>(y, SpmvPhysicalIndex(col, t->yStride));
        AccT result;
        if constexpr (std::is_same_v<AccT, int64_t>) {
            result = SpmvScaledResult(sum, SpmvLoadAlpha<AccT>(t), beta, oldY);
        } else {
            result = SpmvIsZero(beta) ? sum : SpmvAdd(sum, SpmvMul(beta, oldY));
        }
        SpmvStoreValue<OutT, AccT>(y, SpmvPhysicalIndex(col, t->yStride), result);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kValidatorThreads) inline void SpmvValidateSorted(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ int32_t *sortedFlag, uint32_t rows, uint32_t nnz,
    uint32_t rowPtrBase, int64_t rowOffsetsStride, int64_t colIndStride,
    uint32_t coreIndex, uint32_t coreCount)
{
    if (threadIdx.x == 0U) {
        asc_dcci_entire(reinterpret_cast<__gm__ void *>(sortedFlag));
    }
    asc_syncthreads();
    constexpr uint32_t kWarpSize = 32U;
    constexpr uint32_t kWarpsPerBlock = kValidatorThreads / kWarpSize;
    const uint32_t lane = threadIdx.x & (kWarpSize - 1U);
    const uint32_t warp = threadIdx.x / kWarpSize;
    for (uint64_t row64 = static_cast<uint64_t>(coreIndex) * kWarpsPerBlock + warp;
         row64 < rows; row64 += static_cast<uint64_t>(coreCount) * kWarpsPerBlock) {
        const uint32_t row = static_cast<uint32_t>(row64);
        const int64_t begin = static_cast<int64_t>(rowPtr[SpmvPhysicalIndex(row, rowOffsetsStride)]) - rowPtrBase;
        const uint64_t endIndex = SpmvPhysicalIndex(static_cast<uint64_t>(row) + 1U, rowOffsetsStride);
        const int64_t end = static_cast<int64_t>(rowPtr[endIndex]) - rowPtrBase;
        if (begin < 0 || end < begin) {
            (void)asc_atomic_exch(sortedFlag, 0);
            continue;
        }
        for (int64_t p = begin + 1 + lane; p < end &&
             static_cast<uint64_t>(p) < nnz; p += kWarpSize) {
            if (colInd[SpmvPhysicalIndex(p, colIndStride)] < colInd[SpmvPhysicalIndex(p - 1, colIndStride)]) {
                (void)asc_atomic_exch(sortedFlag, 0);
            }
        }
    }
}

template <typename ValT, typename AccT>
__simt_callee__ inline void SpmvAccumulateSerialCsr(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ const ValT *x,
    __gm__ AccT *acc, __gm__ const SpmvArch35TilingData *t)
{
    for (uint32_t row = 0; row < t->rows; ++row) {
        const int64_t begin = SpmvLoadRowOffset(rowPtr, t, row);
        const int64_t end = SpmvLoadRowOffset(rowPtr, t, static_cast<uint64_t>(row) + 1U);
        if (begin < 0 || end < begin) {
            continue;
        }
        const AccT xValue = SpmvLoadValue<AccT, ValT>(x, SpmvPhysicalIndex(row, t->xStride));
        for (int64_t p = begin; p < end && static_cast<uint64_t>(p) < t->nnz; ++p) {
            const int64_t rawCol = static_cast<int64_t>(colInd[SpmvPhysicalIndex(p, t->colIndStride)]) - t->indexBase;
            if (rawCol < 0 || static_cast<uint64_t>(rawCol) >= t->cols) {
                continue;
            }
            const AccT value = SpmvMaybeConjugate(
                SpmvLoadValue<AccT, ValT>(values, SpmvPhysicalIndex(p, t->valuesStride)), t->opA);
            acc[rawCol] = SpmvAdd(acc[rawCol], SpmvMul(value, xValue));
        }
    }
}

template <typename OutT, typename AccT>
__simt_callee__ inline void SpmvFinalizeSerialCsr(
    __gm__ const AccT *acc, __gm__ OutT *y,
    __gm__ const SpmvArch35TilingData *t)
{
    const AccT alpha = SpmvLoadAlpha<AccT>(t);
    const AccT beta = SpmvLoadBeta<AccT>(t);
    for (uint32_t col = 0; col < t->cols; ++col) {
        const AccT oldY = SpmvIsZero(beta) ? SpmvZero<AccT>()
            : SpmvLoadOutput<AccT, OutT>(y, SpmvPhysicalIndex(col, t->yStride));
        const AccT result = SpmvScaledResult(acc[col], alpha, beta, oldY);
        SpmvStoreValue<OutT, AccT>(y, SpmvPhysicalIndex(col, t->yStride), result);
    }
}

template <typename ValT, typename OutT, typename AccT>
__simt_callee__ inline void SpmvTransposeSerial(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ const ValT *x, __gm__ OutT *y,
    __gm__ uint8_t *workspace, __gm__ const SpmvArch35TilingData *t)
{
    __gm__ AccT *acc = reinterpret_cast<__gm__ AccT *>(workspace + t->accumulatorOffset);
    for (uint32_t col = 0; col < t->cols; ++col) {
        acc[col] = SpmvZero<AccT>();
    }
    SpmvAccumulateSerialCsr<ValT, AccT>(rowPtr, colInd, values, x, acc, t);
    SpmvFinalizeSerialCsr<OutT, AccT>(acc, y, t);
}

__simt_callee__ inline int64_t SpmvLowerBoundColumn(
    __gm__ const int32_t *colInd, int64_t begin, int64_t end, int64_t target,
    int64_t colIndStride)
{
    while (begin < end) {
        const int64_t mid = begin + ((end - begin) >> 1U);
        if (static_cast<int64_t>(colInd[SpmvPhysicalIndex(mid, colIndStride)]) < target) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return begin;
}

template <typename ValT, typename AccT>
__simt_callee__ inline AccT SpmvGatherRowContribution(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ const ValT *x,
    __gm__ const SpmvArch35TilingData *t, uint32_t row, int64_t target)
{
    const int64_t begin = SpmvLoadRowOffset(rowPtr, t, row);
    int64_t end = SpmvLoadRowOffset(rowPtr, t, static_cast<uint64_t>(row) + 1U);
    if (begin < 0 || end <= begin) {
        return SpmvZero<AccT>();
    }
    if (static_cast<uint64_t>(end) > t->nnz) {
        end = t->nnz;
    }
    if (end <= begin) {
        return SpmvZero<AccT>();
    }
    if (target < static_cast<int64_t>(colInd[SpmvPhysicalIndex(begin, t->colIndStride)]) ||
        target > static_cast<int64_t>(colInd[SpmvPhysicalIndex(end - 1, t->colIndStride)])) {
        return SpmvZero<AccT>();
    }
    int64_t pos = SpmvLowerBoundColumn(colInd, begin, end, target, t->colIndStride);
    AccT sum = SpmvZero<AccT>();
    const AccT xValue = SpmvLoadValue<AccT, ValT>(x, SpmvPhysicalIndex(row, t->xStride));
    while (pos < end && static_cast<int64_t>(colInd[SpmvPhysicalIndex(pos, t->colIndStride)]) == target) {
        const AccT value = SpmvMaybeConjugate(
            SpmvLoadValue<AccT, ValT>(values, SpmvPhysicalIndex(pos, t->valuesStride)), t->opA);
        sum = SpmvAdd(sum, SpmvMul(value, xValue));
        ++pos;
    }
    return sum;
}

template <typename ValT, typename OutT, typename AccT>
__simt_callee__ inline void SpmvTransposeGatherSorted(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ const ValT *x,
    __gm__ OutT *y, __gm__ const SpmvArch35TilingData *t,
    uint32_t coreIndex, uint32_t coreCount)
{
    constexpr uint32_t kWarpSize = 32U;
    constexpr uint32_t kWarpsPerBlock = kGatherThreads / kWarpSize;
    const uint32_t lane = threadIdx.x & (kWarpSize - 1U);
    const uint32_t warp = threadIdx.x / kWarpSize;
    const AccT alpha = SpmvLoadAlpha<AccT>(t);
    const AccT beta = SpmvLoadBeta<AccT>(t);
    for (uint64_t col64 = static_cast<uint64_t>(coreIndex) * kWarpsPerBlock + warp;
         col64 < t->cols;
         col64 += static_cast<uint64_t>(coreCount) * kWarpsPerBlock) {
        const uint32_t col = static_cast<uint32_t>(col64);
        const int64_t target = static_cast<int64_t>(col) + t->indexBase;
        AccT sum = SpmvZero<AccT>();
        for (uint32_t row = lane; row < t->rows; row += kWarpSize) {
            sum = SpmvAdd(sum, SpmvGatherRowContribution<ValT, AccT>(
                rowPtr, colInd, values, x, t, row, target));
        }
        sum = SpmvWarpReduceSum(sum);
        if (lane == 0U) {
            const AccT oldY = SpmvIsZero(beta) ? SpmvZero<AccT>()
                : SpmvLoadOutput<AccT, OutT>(y, SpmvPhysicalIndex(col, t->yStride));
            const AccT result = SpmvScaledResult(sum, alpha, beta, oldY);
            SpmvStoreValue<OutT, AccT>(y, SpmvPhysicalIndex(col, t->yStride), result);
        }
    }
}

template <typename ValT, typename OutT, typename AccT>
__simt_vf__ __aicore__ __launch_bounds__(kGatherThreads) inline void SpmvTransposeGather(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ const ValT *x,
    __gm__ OutT *y, __gm__ SpmvArch35TilingData *t,
    uint32_t coreIndex, uint32_t coreCount)
{
    SpmvInvalidateTiling(t);
    asc_syncthreads();
    const auto *sortedFlag = reinterpret_cast<__gm__ const int32_t *>(
        reinterpret_cast<__gm__ const uint8_t *>(t) + kSpmvArch35SortedFlagOffset);
    if (*sortedFlag == 0) {
        if (coreIndex == 0U && threadIdx.x == 0U) {
            auto *workspace = reinterpret_cast<__gm__ uint8_t *>(t);
            SpmvTransposeSerial<ValT, OutT, AccT>(
                rowPtr, colInd, values, x, y, workspace, t);
        }
        return;
    }
    SpmvTransposeGatherSorted<ValT, OutT, AccT>(
        rowPtr, colInd, values, x, y, t, coreIndex, coreCount);
}

template <typename ValT, typename OutT, typename AccT>
__simt_vf__ __aicore__ __launch_bounds__(kMaxThreads) inline void SpmvTransposeDeterministic(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ const ValT *x, __gm__ OutT *y,
    __gm__ uint8_t *workspace, __gm__ SpmvArch35TilingData *t,
    uint32_t coreIndex)
{
    SpmvInvalidateTiling(t);
    asc_syncthreads();
    // ALG2 is the deterministic fallback. A single logical worker walks the
    // original CSR row/entry order once, so complexity is O(nnz + cols), not
    // O(cols * nnz) as in a column-by-column rescan. The accumulator lives in
    // GM because the public workspace contract already reserves this region.
    if (coreIndex != 0U || threadIdx.x != 0U) {
        return;
    }
    SpmvTransposeSerial<ValT, OutT, AccT>(
        rowPtr, colInd, values, x, y, workspace, t);
}

template <typename AccT>
__simt_vf__ __aicore__ __launch_bounds__(kMaxThreads) inline void SpmvClearAccumulator(
    __gm__ uint8_t *accBase, uint32_t length,
    uint32_t coreIndex, uint32_t coreCount)
{
    for (uint32_t i = coreIndex * kMaxThreads + threadIdx.x;
         i < length; i += coreCount * kMaxThreads) {
        if constexpr (std::is_same_v<AccT, SpmvComplex64>) {
            auto *dst = reinterpret_cast<__gm__ float *>(accBase) + i * 2U;
            // The following accumulation kernel uses GM atomics, which
            // bypass the Data Cache.  Atomic exchange keeps initialization in
            // the same visibility domain and avoids stale GM values.
            (void)asc_atomic_exch(dst, 0.0f);
            (void)asc_atomic_exch(dst + 1, 0.0f);
        } else if constexpr (std::is_same_v<AccT, int64_t>) {
            (void)asc_atomic_exch(reinterpret_cast<__gm__ int64_t *>(accBase) + i,
                                  static_cast<int64_t>(0));
        } else {
            (void)asc_atomic_exch(reinterpret_cast<__gm__ float *>(accBase) + i, 0.0f);
        }
    }
}

template <typename ValT, typename OutT, typename AccT>
__aicore__ inline void SpmvDispatch(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR x, GM_ADDR y,
    GM_ADDR workspace, uint32_t opA, uint32_t alg,
    uint32_t rows, uint32_t coreIndex, uint32_t coreCount)
{
    auto *rowOffsets = reinterpret_cast<__gm__ const int32_t *>(rowPtr);
    auto *cols = reinterpret_cast<__gm__ const int32_t *>(colInd);
    auto *vals = reinterpret_cast<__gm__ const ValT *>(values);
    auto *xPtr = reinterpret_cast<__gm__ const ValT *>(x);
    auto *yPtr = reinterpret_cast<__gm__ OutT *>(y);
    auto *tilingGm = reinterpret_cast<__gm__ SpmvArch35TilingData *>(workspace);
    if (opA == kSpmvOpNonTranspose) {
        asc_vf_call<SpmvN<ValT, OutT, AccT>>(dim3{kMaxThreads}, rowOffsets, cols, vals, xPtr, yPtr, tilingGm,
                                             coreIndex, coreCount);
    } else if (alg == kSpmvAlgCsrAlg2) {
        asc_vf_call<SpmvTransposeDeterministic<ValT, OutT, AccT>>(
            dim3{kMaxThreads}, rowOffsets, cols, vals, xPtr, yPtr, workspace, tilingGm, coreIndex);
    } else if (rows > 0U && rows <= kSpmvTransposeGatherRowsMax) {
        asc_vf_call<SpmvTransposeGather<ValT, OutT, AccT>>(
            dim3{kGatherThreads}, rowOffsets, cols, vals, xPtr, yPtr, tilingGm,
            coreIndex, coreCount);
    } else {
        // The accumulator is finalized after this entry-wise atomic pass.
        asc_vf_call<SpmvTransposeAtomic<ValT, AccT>>(
            dim3{kMaxThreads}, rowOffsets, cols, vals, xPtr, workspace, tilingGm, coreIndex, coreCount);
    }
}

// Publish the POD tiling payload once before the SIMT compute pass.  Keeping
// this as a one-block kernel gives all later kernels a stream-ordered,
// device-visible header without passing a struct through a SIMT VF (which the
// A5 ABI forbids).
__global__ __aicore__ void SpmvArch35InitTilingKernel(
    GM_ADDR workspace, const SpmvArch35TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto *dst = reinterpret_cast<__gm__ SpmvArch35TilingData *>(workspace);
    if (AscendC::GetBlockIdx() != 0U) {
        return;
    }
    dst->rows = tiling.rows;
    dst->cols = tiling.cols;
    dst->nnz = tiling.nnz;
    dst->indexBase = tiling.indexBase;
    dst->rowPtrBase = tiling.rowPtrBase;
    dst->rowsPerBlock = tiling.rowsPerBlock;
    dst->valueType = tiling.valueType;
    dst->outputType = tiling.outputType;
    dst->computeType = tiling.computeType;
    dst->opA = tiling.opA;
    dst->alg = tiling.alg;
    dst->useNormalizedRowPtr = tiling.useNormalizedRowPtr;
    dst->reserved = tiling.reserved;
    dst->alphaInt = tiling.alphaInt;
    dst->betaInt = tiling.betaInt;
    dst->alphaReal = tiling.alphaReal;
    dst->betaReal = tiling.betaReal;
    dst->alphaImag = tiling.alphaImag;
    dst->betaImag = tiling.betaImag;
    dst->alphaPtr = tiling.alphaPtr;
    dst->betaPtr = tiling.betaPtr;
    dst->accumulatorOffset = tiling.accumulatorOffset;
    dst->rowOffsetsStride = tiling.rowOffsetsStride;
    dst->colIndStride = tiling.colIndStride;
    dst->valuesStride = tiling.valuesStride;
    dst->xStride = tiling.xStride;
    dst->yStride = tiling.yStride;
    *reinterpret_cast<__gm__ int32_t *>(workspace + kSpmvArch35SortedFlagOffset) = 1;
    // Scalar stores can remain dirty in this AIV's DCache. The following
    // stream-ordered SIMT kernels may run on other AIVs, so publish the whole
    // header to GM and wait for that writeback before this kernel completes.
    asc_dcci_entire_out();
    asc_sync_data_barrier(mem_dsb_t::DSB_DDR);
}

template <typename ValT, typename OutT, typename AccT>
__global__ __aicore__ void SpmvArch35Kernel(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR x, GM_ADDR y,
    GM_ADDR workspace, const SpmvArch35TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto *tilingGm = reinterpret_cast<__gm__ SpmvArch35TilingData *>(workspace);
    const uint32_t coreIndex = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t coreCount = static_cast<uint32_t>(AscendC::GetBlockNum());
    GM_ADDR effectiveRowPtr = rowPtr;
    if (tiling.useNormalizedRowPtr != 0U) {
        effectiveRowPtr = workspace + kSpmvArch35HeaderBytes;
    }
    SpmvDispatch<ValT, OutT, AccT>(effectiveRowPtr, colInd, values, x, y,
                                   workspace, tiling.opA, tiling.alg,
                                   tiling.rows, coreIndex, coreCount);
}

template <typename AccT>
__global__ __aicore__ void SpmvArch35ClearKernel(GM_ADDR workspace, uint32_t length,
                                                   uint64_t accumulatorOffset)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t coreIndex = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t coreCount = static_cast<uint32_t>(AscendC::GetBlockNum());
    asc_vf_call<SpmvClearAccumulator<AccT>>(dim3{kMaxThreads},
        workspace + accumulatorOffset, length, coreIndex, coreCount);
}

__global__ __aicore__ void SpmvArch35ValidateKernel(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR workspace,
    uint32_t rows, uint32_t nnz, uint32_t rowPtrBase, uint32_t useNormalizedRowPtr,
    int64_t rowOffsetsStride, int64_t colIndStride)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto *effectiveRowPtr = reinterpret_cast<__gm__ const int32_t *>(
        useNormalizedRowPtr != 0U ? workspace + kSpmvArch35HeaderBytes : rowPtr);
    auto *sortedFlag = reinterpret_cast<__gm__ int32_t *>(
        workspace + kSpmvArch35SortedFlagOffset);
    asc_vf_call<SpmvValidateSorted>(dim3{kValidatorThreads}, effectiveRowPtr,
        reinterpret_cast<__gm__ const int32_t *>(colInd), sortedFlag,
        rows, nnz, rowPtrBase, rowOffsetsStride, colIndStride,
        static_cast<uint32_t>(AscendC::GetBlockIdx()),
        static_cast<uint32_t>(AscendC::GetBlockNum()));
}

template <typename OutT, typename AccT>
__global__ __aicore__ void SpmvArch35FinalizeKernel(GM_ADDR workspace, GM_ADDR y)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t coreIndex = static_cast<uint32_t>(AscendC::GetBlockIdx());
    auto *tilingGm = reinterpret_cast<__gm__ SpmvArch35TilingData *>(workspace);
    asc_vf_call<SpmvTransposeFinalize<uint8_t, OutT, AccT>>(
        dim3{kMaxThreads}, reinterpret_cast<__gm__ uint8_t *>(workspace),
        reinterpret_cast<__gm__ OutT *>(y), tilingGm, coreIndex);
}

template <typename ValT, typename OutT, typename AccT>
void LaunchTyped(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR x, GM_ADDR y,
                 GM_ADDR workspace,
                 const SpmvArch35TilingData &tiling,
                 uint32_t blockDim, void *stream)
{
    SpmvArch35Kernel<ValT, OutT, AccT><<<blockDim, nullptr, stream>>>(
        rowPtr, colInd, values, x, y, workspace, tiling);
}

template <typename AccT>
void LaunchClear(GM_ADDR workspace, uint32_t length, uint64_t accumulatorOffset,
                 uint32_t blockDim, void *stream)
{
    SpmvArch35ClearKernel<AccT><<<blockDim, nullptr, stream>>>(workspace, length, accumulatorOffset);
}

template <typename OutT, typename AccT>
void LaunchFinalize(GM_ADDR workspace, GM_ADDR y, uint32_t blockDim, void *stream)
{
    SpmvArch35FinalizeKernel<OutT, AccT><<<blockDim, nullptr, stream>>>(workspace, y);
}

inline bool UsesAtomicTranspose(const SpmvArch35TilingData &t)
{
    return SpmvIsTransposeOp(t.opA) && t.alg != kSpmvAlgCsrAlg2 &&
           (t.rows == 0U || t.rows > kSpmvTransposeGatherRowsMax);
}

void PrepareTranspose(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR workspace,
                      const SpmvArch35TilingData &t,
                      uint32_t blockDim, void *stream)
{
    if (!SpmvIsTransposeOp(t.opA) || t.alg == kSpmvAlgCsrAlg2) {
        return;
    }
    if (t.rows > 0U && t.rows <= kSpmvTransposeGatherRowsMax) {
        SpmvArch35ValidateKernel<<<blockDim, nullptr, stream>>>(
            rowPtr, colInd, workspace, t.rows, t.nnz,
            t.rowPtrBase, t.useNormalizedRowPtr, t.rowOffsetsStride, t.colIndStride);
        return;
    }
    if (t.computeType == SPMV35_COMPUTE_I32) {
        LaunchClear<int64_t>(workspace, t.cols, t.accumulatorOffset, blockDim, stream);
    } else if (t.computeType == SPMV35_COMPUTE_C64) {
        LaunchClear<SpmvComplex64>(workspace, t.cols, t.accumulatorOffset, blockDim, stream);
    } else {
        LaunchClear<float>(workspace, t.cols, t.accumulatorOffset, blockDim, stream);
    }
}

void LaunchSpmvByType(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
                      GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                      const SpmvArch35TilingData &t,
                      uint32_t blockDim, void *stream)
{
#define SPMV35_LAUNCH(V, O, A) LaunchTyped<V, O, A>(rowPtr, colInd, values, x, y, workspace, t, blockDim, stream)
    if (t.computeType == SPMV35_COMPUTE_I32) {
        SPMV35_LAUNCH(int8_t, int32_t, int64_t);
    } else if (t.computeType == SPMV35_COMPUTE_C64) {
        if (t.valueType == SPMV35_VALUE_F32) {
            SPMV35_LAUNCH(float, SpmvComplex64, SpmvComplex64);
        } else {
            SPMV35_LAUNCH(SpmvComplex64, SpmvComplex64, SpmvComplex64);
        }
    } else if (t.valueType == SPMV35_VALUE_I8) {
        SPMV35_LAUNCH(int8_t, float, float);
    } else if (t.valueType == SPMV35_VALUE_F16) {
        if (t.outputType == SPMV35_OUTPUT_F16) {
            SPMV35_LAUNCH(SpmvHalfBits, SpmvHalfBits, float);
        } else {
            SPMV35_LAUNCH(SpmvHalfBits, float, float);
        }
    } else if (t.valueType == SPMV35_VALUE_BF16) {
        if (t.outputType == SPMV35_OUTPUT_BF16) {
            SPMV35_LAUNCH(SpmvBf16Bits, SpmvBf16Bits, float);
        } else {
            SPMV35_LAUNCH(SpmvBf16Bits, float, float);
        }
    } else {
        SPMV35_LAUNCH(float, float, float);
    }
#undef SPMV35_LAUNCH
}

void FinalizeTranspose(GM_ADDR workspace, GM_ADDR y,
                       const SpmvArch35TilingData &t,
                       uint32_t blockDim, void *stream)
{
    if (!UsesAtomicTranspose(t)) {
        return;
    }
    if (t.computeType == SPMV35_COMPUTE_C64) {
        LaunchFinalize<SpmvComplex64, SpmvComplex64>(workspace, y, blockDim, stream);
    } else if (t.computeType == SPMV35_COMPUTE_I32) {
        LaunchFinalize<int32_t, int64_t>(workspace, y, blockDim, stream);
    } else if (t.valueType == SPMV35_VALUE_F16 && t.outputType == SPMV35_OUTPUT_F16) {
        LaunchFinalize<SpmvHalfBits, float>(workspace, y, blockDim, stream);
    } else if (t.valueType == SPMV35_VALUE_BF16 && t.outputType == SPMV35_OUTPUT_BF16) {
        LaunchFinalize<SpmvBf16Bits, float>(workspace, y, blockDim, stream);
    } else {
        LaunchFinalize<float, float>(workspace, y, blockDim, stream);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void SpmvArch35PreprocessKernel(
    GM_ADDR rowPtr, GM_ADDR workspace, uint32_t rows, int32_t indexBase,
    int64_t rowOffsetsStride)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto *src = reinterpret_cast<__gm__ const int32_t *>(rowPtr);
    auto *dst = reinterpret_cast<__gm__ int32_t *>(workspace) +
        kSpmvArch35HeaderBytes / sizeof(int32_t);
    // This kernel is deliberately scalar at the outer AIV level. It avoids
    // relying on SIMT builtins in the preprocessing entry and works for the
    // same block count used by the subsequent compute launch.
    for (uint32_t i = AscendC::GetBlockIdx();
         i <= rows; i += AscendC::GetBlockNum()) {
        dst[i] = src[static_cast<uint64_t>(i) * rowOffsetsStride] - indexBase;
    }
    // Preprocess writes the normalized CSR row offsets through Scalar DCache;
    // the compute pass reads them on SIMT and potentially on different AIVs.
    asc_dcci_entire_out();
    asc_sync_data_barrier(mem_dsb_t::DSB_DDR);
}

extern "C" void spmv_arch35_preprocess_kernel_do(
    GM_ADDR rowPtr, GM_ADDR workspace, uint32_t rows, int32_t indexBase,
    int64_t rowOffsetsStride, uint32_t blockDim, void *stream)
{
    SpmvArch35PreprocessKernel<<<blockDim, nullptr, stream>>>(
        rowPtr, workspace, rows, indexBase, rowOffsetsStride);
}

extern "C" void spmv_arch35_kernel_do(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR x, GM_ADDR y,
    GM_ADDR workspace, const SpmvArch35TilingData &t,
    uint32_t blockDim, void *stream)
{
    // The POD tiling argument is legal for this ordinary Kernel entry.  A
    // single block publishes it before all pointer-only SIMT VF calls below.
    SpmvArch35InitTilingKernel<<<1, nullptr, stream>>>(workspace, t);
    PrepareTranspose(rowPtr, colInd, workspace, t, blockDim, stream);
    LaunchSpmvByType(rowPtr, colInd, values, x, y, workspace, t, blockDim, stream);
    FinalizeTranspose(workspace, y, t, blockDim, stream);
}
