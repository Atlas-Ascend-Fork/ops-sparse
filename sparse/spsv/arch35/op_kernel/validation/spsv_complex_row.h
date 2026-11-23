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

#ifndef OP_KERNEL_VALIDATION_SPSV_COMPLEX_ROW_H_
#define OP_KERNEL_VALIDATION_SPSV_COMPLEX_ROW_H_

__simt_callee__ __attribute__((always_inline)) inline void SpsvMergeComplexBlock(int32_t lane, SpsvTriple& re,
                                                                                 SpsvTriple& im, float& discardedRe,
                                                                                 float& discardedIm, uint32_t& rejected)
{
    __ubuf__ float partials[64 * 9];
    int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
    int32_t warps = static_cast<int32_t>(blockDim.x) / 32;
    if (lane == 0) {
        int32_t p = 9 * warp;
        partials[p] = re.hi;
        partials[p + 1] = re.mid;
        partials[p + 2] = re.lo;
        partials[p + 3] = im.hi;
        partials[p + 4] = im.mid;
        partials[p + 5] = im.lo;
        partials[p + 6] = discardedRe;
        partials[p + 7] = discardedIm;
        partials[p + 8] = static_cast<float>(rejected);
    }
    asc_syncthreads();
    re = {};
    im = {};
    discardedRe = discardedIm = 0.0f;
    rejected = 0;
    for (int32_t w = lane; w < warps; w += 32) {
        int32_t p = 9 * w;
        SpsvTripleAdd(re, partials[p], discardedRe);
        SpsvTripleAdd(re, partials[p + 1], discardedRe);
        SpsvTripleAdd(re, partials[p + 2], discardedRe);
        SpsvTripleAdd(im, partials[p + 3], discardedIm);
        SpsvTripleAdd(im, partials[p + 4], discardedIm);
        SpsvTripleAdd(im, partials[p + 5], discardedIm);
        discardedRe += partials[p + 6];
        discardedIm += partials[p + 7];
        rejected |= static_cast<uint32_t>(partials[p + 8]);
    }
    SpsvTripleWarpMerge<true>(re, discardedRe, rejected);
    SpsvTripleWarpMerge<true>(im, discardedIm, rejected);
}

template <typename RowPtrT, typename ColIndT>
__simt_callee__ __attribute__((always_inline)) inline void SpsvAccumulateComplexRedo(
    __gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values, __gm__ const float* yWide, RowPtrT start,
    RowPtrT end, int32_t first, int32_t stride, int32_t row, bool forward, SpsvTriple& re, SpsvTriple& im,
    float& discardedRe, float& discardedIm, uint32_t& rejected)
{
    for (SpsvCsrCursor<RowPtrT> p = static_cast<SpsvCsrCursor<RowPtrT>>(start) + first; p < end; p += stride) {
        int32_t column = static_cast<int32_t>(colInd[p]);
        if (forward ? column < row : column > row) {
            for (int32_t word = 0; word < 2; ++word) {
                float yr = yWide[4 * static_cast<int64_t>(column) + word];
                float yi = yWide[4 * static_cast<int64_t>(column) + 2 + word];
                rejected |= !SpsvTripleTerm(re, discardedRe, -values[p].real, yr);
                rejected |= !SpsvTripleTerm(re, discardedRe, values[p].imag, yi);
                rejected |= !SpsvTripleTerm(im, discardedIm, -values[p].real, yi);
                rejected |= !SpsvTripleTerm(im, discardedIm, -values[p].imag, yr);
            }
        }
    }
}

template <typename RowPtrT, typename ColIndT, bool WARP, bool BLOCK = false>
__simt_callee__ __attribute__((always_inline)) inline bool SpsvTryCertifiedComplexRedo(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const volatile SpsvComplex64* vecX, __gm__ const float* yWide, SpsvComplex64 alpha, int32_t row,
    bool forward, bool nonUnit, SpsvComplex64 diagonal, SpsvDFComplex& answer)
{
    RowPtrT start = rowPtr[row], end = rowPtr[row + 1];
    if (end - start > (1 << 20)) {
        return false;
    }
    int32_t lane = WARP ? static_cast<int32_t>(threadIdx.x) & 31 : 0;
    int32_t first = BLOCK ? static_cast<int32_t>(threadIdx.x) : lane;
    int32_t stride = BLOCK ? static_cast<int32_t>(blockDim.x) : WARP ? 32 : 1;
    SpsvTriple re, im;
    float discardedRe = 0.0f, discardedIm = 0.0f;
    uint32_t rejected = 0;
    if (first == 0) {
        rejected |= !SpsvTripleTerm(re, discardedRe, alpha.real, vecX[row].real);
        rejected |= !SpsvTripleTerm(re, discardedRe, -alpha.imag, vecX[row].imag);
        rejected |= !SpsvTripleTerm(im, discardedIm, alpha.real, vecX[row].imag);
        rejected |= !SpsvTripleTerm(im, discardedIm, alpha.imag, vecX[row].real);
    }
    SpsvAccumulateComplexRedo(colInd, values, yWide, start, end, first, stride, row, forward, re, im, discardedRe,
                              discardedIm, rejected);
    SpsvTripleWarpMerge<WARP>(re, discardedRe, rejected);
    SpsvTripleWarpMerge<WARP>(im, discardedIm, rejected);
    if constexpr (BLOCK) {
        SpsvMergeComplexBlock(lane, re, im, discardedRe, discardedIm, rejected);
    }
    uint32_t accepted = 0;
    if (lane == 0 && !rejected) {
        SpsvDFComplex numerator{SpsvTripleFinish(re, discardedRe), SpsvTripleFinish(im, discardedIm)};
        accepted = SpsvCertifiedComplexQuotient(numerator, discardedRe, discardedIm, diagonal, answer, false);
    }
    if constexpr (WARP) {
        accepted = asc_shfl(accepted, 0);
        answer.re.hi = asc_shfl(answer.re.hi, 0);
        answer.re.lo = asc_shfl(answer.re.lo, 0);
        answer.im.hi = asc_shfl(answer.im.hi, 0);
        answer.im.lo = asc_shfl(answer.im.lo, 0);
    }
    return accepted != 0;
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, bool WARP, bool BLOCK>
__simt_callee__ __attribute__((always_inline)) inline void SpsvComplexColdBounds(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const volatile SpsvComplex64* vecX, __gm__ SpsvComplex64* vecY, __gm__ float* yWide, SpsvComplex64 alpha,
    int32_t row, __gm__ const float* rowPrep, int32_t lane, uint32_t& nonfinite, float& dependencyError, float& norm)
{
    bool deferred = rowPrep[18 * static_cast<int64_t>(row) + 7] == kSpsvDeferredComplexCertificate;
    const int32_t stride = WARP ? 32 : 1;
    if (lane != 0) {
        norm = 0.0f;
    }
    for (SpsvCsrCursor<RowPtrT> p = static_cast<SpsvCsrCursor<RowPtrT>>(rowPtr[row]) + lane;
         p < rowPtr[row + 1] && (!BLOCK || static_cast<int32_t>(threadIdx.x) < 32); p += stride) {
        int32_t col = static_cast<int32_t>(colInd[p]);
        if (FORWARD ? col < row : col > row) {
            int64_t at = 4 * static_cast<int64_t>(col);
            if (!deferred) {
                float coefficient = SpsvAbsF(values[p].real) + SpsvAbsF(values[p].imag);
                float pe = rowPrep[18 * static_cast<int64_t>(col) + 9];
                float pi = rowPrep[18 * static_cast<int64_t>(col) + 10];
                dependencyError += coefficient * (pe > pi ? pe : pi);
                norm += coefficient * (SpsvAbsF(yWide[at]) + SpsvAbsF(yWide[at + 1]) + SpsvAbsF(yWide[at + 2]) +
                                       SpsvAbsF(yWide[at + 3]));
            }
            nonfinite |= !SpsvIsFiniteF(values[p].real) || !SpsvIsFiniteF(values[p].imag) ||
                         !SpsvIsFiniteF(yWide[at]) || !SpsvIsFiniteF(yWide[at + 1]) || !SpsvIsFiniteF(yWide[at + 2]) ||
                         !SpsvIsFiniteF(yWide[at + 3]);
        }
    }
    if constexpr (WARP) {
        nonfinite = asc_reduce_max(nonfinite);
        dependencyError = asc_reduce_add(dependencyError);
        norm = asc_reduce_add(norm);
    }
    if constexpr (BLOCK) {
        __ubuf__ uint32_t invalid[1];
        __ubuf__ float bounds[2];
        if (static_cast<int32_t>(threadIdx.x) == 0) {
            invalid[0] = nonfinite;
            bounds[0] = dependencyError;
            bounds[1] = norm;
        }
        asc_syncthreads();
        nonfinite = invalid[0];
        dependencyError = bounds[0];
        norm = bounds[1];
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, bool WARP, bool BLOCK>
__simt_callee__ __attribute__((always_inline)) inline void SpsvPublishComplexColdAnswer(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const volatile SpsvComplex64* vecX, __gm__ SpsvComplex64* vecY, __gm__ float* yWide, SpsvComplex64 alpha,
    int32_t row, __gm__ const float* rowPrep, SpsvComplex64 diagonal, SpsvDFComplex answer, float dependencyError,
    float norm)
{
    int64_t at = 4 * static_cast<int64_t>(row);
    yWide[at] = answer.re.hi;
    yWide[at + 1] = answer.re.lo;
    yWide[at + 2] = answer.im.hi;
    yWide[at + 3] = answer.im.lo;
    vecY[row].real = answer.re.hi;
    vecY[row].imag = answer.im.hi;
    if (rowPrep[18 * static_cast<int64_t>(row) + 7] == kSpsvDeferredComplexCertificate) {
        ((__gm__ float*)rowPrep)[18 * static_cast<int64_t>(row) + 9] = __builtin_inff();
        ((__gm__ float*)rowPrep)[18 * static_cast<int64_t>(row) + 10] = __builtin_inff();
        return;
    }
    float dr = SpsvAbsF(diagonal.real), di = SpsvAbsF(diagonal.imag);
    float inverseNorm = rowPrep[18 * static_cast<int64_t>(row) + 8];
    if (!(inverseNorm > 0.0f)) {
        inverseNorm = SpsvFloatBoundDiv(2.0f, dr > di ? dr : di);
    }
    float inherited = SpsvFloatBoundMul(
        SpsvFloatDependencyBound(dependencyError, static_cast<int64_t>(rowPtr[row + 1]) - rowPtr[row]), inverseNorm);
    float radius = SpsvFloatBoundDiv(
        SpsvFloatBoundUp(norm * 0x1p-38f +
                         static_cast<float>(static_cast<int64_t>(rowPtr[row + 1]) - rowPtr[row] + 1) * 0x1p-118f),
        dr > di ? dr : di);
    radius = SpsvFloatBoundAdd(radius, inherited);
    if (static_cast<int64_t>(rowPtr[row + 1]) - rowPtr[row] > (1 << 20)) {
        radius = __builtin_inff();
    }
    ((__gm__ float*)rowPrep)[18 * static_cast<int64_t>(row) + 9] = SpsvFloatBoundAdd(radius,
                                                                                     SpsvAbsF(answer.re.hi) * 0x1p-40f);
    ((__gm__ float*)rowPrep)[18 * static_cast<int64_t>(row) + 10] = SpsvFloatBoundAdd(
        radius, SpsvAbsF(answer.im.hi) * 0x1p-40f);
    return;
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, bool WARP, bool BLOCK>
__simt_callee__ __attribute__((always_inline)) inline void SpsvComplexColdPropagate(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const volatile SpsvComplex64* vecX, __gm__ SpsvComplex64* vecY, __gm__ float* yWide, SpsvComplex64 alpha,
    int32_t row, __gm__ const float* rowPrep)
{
    ((__gm__ float*)rowPrep)[18 * static_cast<int64_t>(row) + 9] = __builtin_inff();
    ((__gm__ float*)rowPrep)[18 * static_cast<int64_t>(row) + 10] = __builtin_inff();
    if constexpr (WARP) {
        if (!BLOCK || static_cast<int32_t>(threadIdx.x) < 32) {
            SpsvSolveRowComplexScaled<RowPtrT, ColIndT, FORWARD, NON_UNIT>(rowPtr, colInd, values, vecY, yWide, rowPrep,
                                                                           row);
        }
        if constexpr (BLOCK) {
            asc_syncthreads();
        }
    } else {
        SpsvSolveRowComplexScaledThread<RowPtrT, ColIndT, FORWARD, NON_UNIT>(rowPtr, colInd, values, vecY, yWide,
                                                                             rowPrep, row);
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, bool WARP, bool BLOCK>
__simt_callee__ __attribute__((always_inline)) inline void SpsvRecoverComplexColdAnswer(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const volatile SpsvComplex64* vecX, __gm__ const float* yWide, SpsvComplex64 alpha, int32_t row,
    int32_t lane, SpsvComplex64 diagonal, SpsvDFComplex& answer)
{
    if ((BLOCK ? static_cast<int32_t>(threadIdx.x) : lane) == 0) {
        answer = SpsvComplexExactRow<RowPtrT, ColIndT>(rowPtr, colInd, values, vecX, yWide, alpha, row, FORWARD,
                                                       NON_UNIT, diagonal);
    }
    if constexpr (BLOCK) {
        __ubuf__ float recovered[4];
        if (static_cast<int32_t>(threadIdx.x) == 0) {
            recovered[0] = answer.re.hi;
            recovered[1] = answer.re.lo;
            recovered[2] = answer.im.hi;
            recovered[3] = answer.im.lo;
        }
        asc_syncthreads();
        answer.re = {recovered[0], recovered[1]};
        answer.im = {recovered[2], recovered[3]};
    } else if constexpr (WARP) {
        answer.re.hi = asc_shfl(answer.re.hi, 0);
        answer.re.lo = asc_shfl(answer.re.lo, 0);
        answer.im.hi = asc_shfl(answer.im.hi, 0);
        answer.im.lo = asc_shfl(answer.im.lo, 0);
    }
}

// Keep the shadow result compensated on the cold path as well. A finite
// FP32 quotient alone does not certify its low word for downstream rows.
template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, bool WARP, bool BLOCK = false>
__simt_callee__ __attribute__((noinline)) void SpsvFinishComplexColdRow(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const volatile SpsvComplex64* vecX, __gm__ SpsvComplex64* vecY, __gm__ float* yWide, SpsvComplex64 alpha,
    int32_t row, __gm__ const float* rowPrep)
{
    int32_t lane = WARP ? static_cast<int32_t>(threadIdx.x) & 31 : 0;
    SpsvComplex64 diagonal{1.0f, 0.0f};
    if constexpr (NON_UNIT) {
        int64_t meta = 18 * static_cast<int64_t>(row);
        diagonal = {rowPrep[meta + 16], rowPrep[meta + 17]};
    }
    uint32_t nonfinite = !SpsvIsFiniteF(alpha.real) || !SpsvIsFiniteF(alpha.imag) || !SpsvIsFiniteF(vecX[row].real) ||
                         !SpsvIsFiniteF(vecX[row].imag);
    float dependencyError = 0.0f;
    float norm = (SpsvAbsF(alpha.real) + SpsvAbsF(alpha.imag)) * (SpsvAbsF(vecX[row].real) + SpsvAbsF(vecX[row].imag));
    SpsvComplexColdBounds<RowPtrT, ColIndT, FORWARD, NON_UNIT, WARP, BLOCK>(
        rowPtr, colInd, values, vecX, vecY, yWide, alpha, row, rowPrep, lane, nonfinite, dependencyError, norm);
    if (nonfinite || !SpsvIsFiniteF(diagonal.real) || !SpsvIsFiniteF(diagonal.imag) ||
        (diagonal.real == 0.0f && diagonal.imag == 0.0f)) {
        SpsvComplexColdPropagate<RowPtrT, ColIndT, FORWARD, NON_UNIT, WARP, BLOCK>(rowPtr, colInd, values, vecX, vecY,
                                                                                   yWide, alpha, row, rowPrep);
        return;
    }
    SpsvDFComplex answer{};
    bool accepted = SpsvTryCertifiedComplexRedo<RowPtrT, ColIndT, WARP, BLOCK>(
        rowPtr, colInd, values, vecX, yWide, alpha, row, FORWARD, NON_UNIT, diagonal, answer);
    if (!accepted) {
        SpsvRecoverComplexColdAnswer<RowPtrT, ColIndT, FORWARD, NON_UNIT, WARP, BLOCK>(
            rowPtr, colInd, values, vecX, yWide, alpha, row, lane, diagonal, answer);
    }
    if (rowPrep[18 * static_cast<int64_t>(row) + 7] != kSpsvDeferredComplexCertificate) {
        ((__gm__ float*)rowPrep)[18 * static_cast<int64_t>(row) + 7] = accepted ? 2.0f : 3.0f;
    }
    SpsvPublishComplexColdAnswer<RowPtrT, ColIndT, FORWARD, NON_UNIT, WARP, BLOCK>(
        rowPtr, colInd, values, vecX, vecY, yWide, alpha, row, rowPrep, diagonal, answer, dependencyError, norm);
}

#endif // OP_KERNEL_VALIDATION_SPSV_COMPLEX_ROW_H_
