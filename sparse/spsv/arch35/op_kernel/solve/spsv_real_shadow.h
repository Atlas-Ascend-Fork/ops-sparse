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

#ifndef OP_KERNEL_SOLVE_SPSV_REAL_SHADOW_H_
#define OP_KERNEL_SOLVE_SPSV_REAL_SHADOW_H_

// Every scheduled FP32 path carries a low solution word. A row-local retry
// cannot recover a predecessor that was already rounded to FP32.
// Retain the fast accumulator's two words, but enclose the actual
// discarded residuals of its low-word operations. A factor of four after
// the existing 2^-21 conversion encloses positive-bound accumulation error
// for the supported (<2^20 term) certificate; the row floor covers FTZ.
__simt_callee__ inline void SpsvAccFloatEnclosed(float& hi, float& lo, float& error, float a, float yh, float yl)
{
    float ph = a * yh;
    float pl = a * yl;
    SpsvDF productLow = SpsvTwoSum(fmaf(a, yh, -ph), pl);
    SpsvDF head = SpsvTwoSum(hi, ph);
    SpsvDF correction = SpsvTwoSum(head.lo, productLow.hi);
    SpsvDF low = SpsvTwoSum(lo, correction.hi);
    float discarded = SpsvAbsF(fmaf(a, yl, -pl)) + SpsvAbsF(productLow.lo) + SpsvAbsF(correction.lo) + SpsvAbsF(low.lo);
    error += discarded * 0x1p23f;
    hi = head.hi;
    lo = low.hi;
}

__simt_callee__ inline SpsvDF SpsvMergeFloatEnclosed(SpsvDF a, SpsvDF b, float& error)
{
    SpsvDF high = SpsvTwoSum(a.hi, b.hi);
    SpsvDF low = SpsvTwoSum(a.lo, b.lo);
    SpsvDF middle = SpsvTwoSum(high.lo, low.hi);
    error += (SpsvAbsF(low.lo) + SpsvAbsF(middle.lo)) * 0x1p23f;
    return SpsvTwoSum(high.hi, middle.hi);
}

template <typename RowPtrT, typename ColIndT, bool FORWARD>
__simt_callee__ __attribute__((noinline)) SpsvDF SpsvFloatShadowExact(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ const float* values, __gm__ const float* vecX,
    __gm__ const float* vecY, __gm__ const float* yLow, int32_t row, float alpha, float diagonal)
{
    RowPtrT start = rowPtr[row], end = rowPtr[row + 1];
    SpsvTriple triple;
    float discarded = 0.0f;
    bool rejected = end - start > (1 << 20);
    rejected |= !SpsvTripleTerm(triple, discarded, alpha, vecX[row]);
    for (SpsvCsrCursor<RowPtrT> p = static_cast<SpsvCsrCursor<RowPtrT>>(start); p < end && !rejected; ++p) {
        int32_t col = static_cast<int32_t>(colInd[p]);
        if (FORWARD ? col < row : col > row) {
            rejected |= !SpsvTripleTerm(triple, discarded, -values[p], vecY[col]);
            rejected |= !SpsvTripleTerm(triple, discarded, -values[p], yLow[2 * static_cast<int64_t>(col)]);
        }
    }
    SpsvDF certified;
    SpsvDF numerator = SpsvTripleFinish(triple, discarded);
    if (!rejected && SpsvRealWideQuotient(numerator, discarded, diagonal, certified)) {
        return certified;
    }
    SpsvExactSum exact;
    SpsvExactClear(exact);
    SpsvExactProduct(exact, alpha, vecX[row], 1.0f, false);
    for (SpsvCsrCursor<RowPtrT> p = static_cast<SpsvCsrCursor<RowPtrT>>(start); p < end; ++p) {
        int32_t col = static_cast<int32_t>(colInd[p]);
        if (FORWARD ? col < row : col > row) {
            SpsvExactProduct(exact, values[p], vecY[col], 1.0f, true);
            SpsvExactProduct(exact, values[p], yLow[2 * static_cast<int64_t>(col)], 1.0f, true);
        }
    }
    return SpsvExactQuotient(SpsvExactFinish(exact), SpsvDFExpProd(diagonal, 1.0f));
}

// A cancellation retry distributes both predecessor words over the same
// lane group as the fast row. Only an uncertifiable result needs the exact
// integer accumulator on lane zero.
template <typename RowPtrT, typename ColIndT, bool FORWARD, int LANES>
__simt_callee__ __attribute__((noinline)) bool SpsvTryFloatShadowRedo(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ const float* values, __gm__ const float* vecX,
    __gm__ const float* vecY, __gm__ const float* yLow, float alpha, int32_t row, float diagonal, SpsvDF& answer)
{
    int32_t lane = static_cast<int32_t>(threadIdx.x) % LANES;
    RowPtrT start = rowPtr[row], end = rowPtr[row + 1];
    SpsvTriple sum;
    float discarded = 0.0f;
    uint32_t rejected = end - start > (1 << 20);
    if (lane == 0) {
        rejected |= !SpsvTripleTerm(sum, discarded, alpha, vecX[row]);
    }
    for (SpsvCsrCursor<RowPtrT> p = static_cast<SpsvCsrCursor<RowPtrT>>(start) + lane; p < end; p += LANES) {
        int32_t col = static_cast<int32_t>(colInd[p]);
        if (FORWARD ? col < row : col > row) {
            rejected |= !SpsvTripleTerm(sum, discarded, -values[p], vecY[col]);
            rejected |= !SpsvTripleTerm(sum, discarded, -values[p], yLow[2 * static_cast<int64_t>(col)]);
        }
    }
    for (int32_t offset = LANES / 2; offset > 0; offset >>= 1) {
        SpsvTriple other{asc_shfl_xor(sum.hi, offset, LANES), asc_shfl_xor(sum.mid, offset, LANES),
                         asc_shfl_xor(sum.lo, offset, LANES)};
        float otherDiscarded = asc_shfl_xor(discarded, offset, LANES);
        uint32_t otherRejected = asc_shfl_xor(rejected, offset, LANES);
        if (lane < offset) {
            SpsvTripleAdd(sum, other.hi, discarded);
            SpsvTripleAdd(sum, other.mid, discarded);
            SpsvTripleAdd(sum, other.lo, discarded);
            discarded += otherDiscarded;
            rejected |= otherRejected;
        }
    }
    uint32_t accepted = 0;
    if (lane == 0 && !rejected) {
        SpsvDF numerator = SpsvTripleFinish(sum, discarded);
        accepted = SpsvRealWideQuotient(numerator, discarded, diagonal, answer);
    }
    if constexpr (LANES > 1) {
        accepted = asc_shfl(accepted, 0, LANES);
    }
    return accepted != 0;
}

template <typename RowPtrT, int LANES>
__simt_callee__ __attribute__((always_inline)) inline void SpsvPublishFloatShadow(
    __gm__ const RowPtrT* rowPtr, __gm__ float* vecY, __gm__ float* yLow, int32_t row, float diagonal,
    SpsvDF originalAnswer, SpsvDF answer, float bound, float predecessorError, uint32_t flags)
{
    if (static_cast<int32_t>(threadIdx.x) % LANES == 0) {
        vecY[row] = answer.hi;
        if (yLow != nullptr) {
            yLow[2 * static_cast<int64_t>(row)] = answer.lo;
            float displacement = SpsvFloatBoundAdd(SpsvAbsF(originalAnswer.hi - answer.hi),
                                                   SpsvAbsF(originalAnswer.lo - answer.lo));
            float enclosure = SpsvFloatBoundAdd(
                SpsvFloatBoundAdd(bound, displacement),
                SpsvFloatBoundDiv(
                    SpsvFloatDependencyBound(predecessorError, static_cast<int64_t>(rowPtr[row + 1]) - rowPtr[row]),
                    SpsvAbsF(diagonal)));
            yLow[2 * static_cast<int64_t>(row) + 1] = (flags & 2U) == 0 && SpsvIsFiniteF(enclosure) &&
                                                              static_cast<int64_t>(rowPtr[row + 1]) - rowPtr[row] <=
                                                                  (1 << 20) ?
                                                          enclosure :
                                                          __builtin_inff();
        }
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD>
__simt_callee__ __attribute__((always_inline)) inline float SpsvPropagateFloatRow(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ const float* values, __gm__ const float* vecX,
    __gm__ const float* vecY, int32_t row, float alpha, float diagonal)
{
    float propagated = alpha * vecX[row];
    for (SpsvCsrCursor<RowPtrT> p = static_cast<SpsvCsrCursor<RowPtrT>>(rowPtr[row]); p < rowPtr[row + 1]; ++p) {
        int32_t col = static_cast<int32_t>(colInd[p]);
        if (FORWARD ? col < row : col > row) {
            propagated -= values[p] * vecY[col];
        }
    }
    return propagated / diagonal;
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, int LANES>
__simt_callee__ inline void SpsvFinishFloatShadow(__gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd,
                                                  __gm__ const float* values, __gm__ const float* vecX,
                                                  __gm__ float* vecY, __gm__ float* yLow, float alpha, int32_t row,
                                                  float diagonal, SpsvDF sum, float error, float predecessorError,
                                                  uint32_t flags)
{
    SpsvDF answer = SpsvDFDivF(SpsvTwoSum(sum.hi, sum.lo), diagonal);
    float bound = (error * 0x1p-21f +
                   static_cast<float>(static_cast<int64_t>(rowPtr[row + 1]) - rowPtr[row] + 1) * 0x1p-120f) /
                      SpsvAbsF(diagonal) +
                  SpsvAbsF(answer.hi) * 0x1p-45f;
    SpsvDF originalAnswer = answer;
    float budget = SpsvAbsF(answer.hi) * 0x1p-32f;
    budget = budget > 0x1p-45f ? budget : 0x1p-45f;
    bool risk = flags || static_cast<int64_t>(rowPtr[row + 1]) - rowPtr[row] > (1 << 20) || !SpsvIsFiniteF(bound) ||
                !SpsvIsFiniteF(answer.hi) || !SpsvIsFiniteF(answer.lo) || bound > budget;
    if (risk) {
        int32_t lane = static_cast<int32_t>(threadIdx.x) % LANES;
        if ((flags & 1U) || !SpsvIsFiniteF(diagonal) || diagonal == 0.0f) {
            if (lane == 0) {
                answer = {SpsvPropagateFloatRow<RowPtrT, ColIndT, FORWARD>(rowPtr, colInd, values, vecX, vecY, row,
                                                                           alpha, diagonal),
                          0.0f};
            }
        } else if (!SpsvTryFloatShadowRedo<RowPtrT, ColIndT, FORWARD, LANES>(rowPtr, colInd, values, vecX, vecY, yLow,
                                                                             alpha, row, diagonal, answer)) {
            if (lane == 0) {
                answer = SpsvFloatShadowExact<RowPtrT, ColIndT, FORWARD>(rowPtr, colInd, values, vecX, vecY, yLow, row,
                                                                         alpha, diagonal);
            }
        }
        if constexpr (LANES > 1) {
            answer.hi = asc_shfl(answer.hi, 0, LANES);
            answer.lo = asc_shfl(answer.lo, 0, LANES);
        }
    }
    SpsvPublishFloatShadow<RowPtrT, LANES>(rowPtr, vecY, yLow, row, diagonal, originalAnswer, answer, bound,
                                           predecessorError, flags);
}

__simt_callee__ inline uint32_t SpsvFloatOperandFlags(float a, float b, float c)
{
    uint32_t aa, bb, cc;
    __builtin_memcpy(&aa, &a, sizeof(aa));
    __builtin_memcpy(&bb, &b, sizeof(bb));
    __builtin_memcpy(&cc, &c, sizeof(cc));
    aa &= 0x7fffffffU;
    bb &= 0x7fffffffU;
    cc &= 0x7fffffffU;
    uint32_t flags = ((aa - 1U < 0x007fffffU) || (bb - 1U < 0x007fffffU) || (cc - 1U < 0x007fffffU)) ? 2U : 0U;
    return flags | ((aa >= 0x7f800000U || bb >= 0x7f800000U || cc >= 0x7f800000U) ? 3U : 0U);
}

template <int LANES, bool TIGHT = true>
__simt_callee__ inline SpsvDF SpsvReduceFloatShadow(SpsvDF sum, float& error, float& predecessorError, uint32_t& flags)
{
    for (int32_t mask = 1; mask < LANES; mask <<= 1) {
        SpsvDF other{asc_shfl_xor(sum.hi, mask, LANES), asc_shfl_xor(sum.lo, mask, LANES)};
        float otherError = asc_shfl_xor(error, mask, LANES);
        if constexpr (TIGHT) {
            sum = SpsvMergeFloatEnclosed(sum, other, error);
        } else {
            sum = SpsvMergeEnclosed(sum, other, error);
        }
        error += otherError;
        predecessorError += asc_shfl_xor(predecessorError, mask, LANES);
        flags |= asc_shfl_xor(flags, mask, LANES);
    }
    return sum;
}

template <int LANES>
__simt_callee__ __attribute__((always_inline)) inline void SpsvPublishFloatDiagonalOnly(__gm__ float* vecY,
                                                                                        __gm__ float* yLow, int32_t row,
                                                                                        float alpha, float x,
                                                                                        float diagonal)
{
    if (static_cast<int32_t>(threadIdx.x) % LANES == 0) {
        if (yLow == nullptr) {
            vecY[row] = (alpha * x) / diagonal;
        } else {
            SpsvDF answer = SpsvDFDivF(SpsvTwoProd(alpha, x), diagonal);
            vecY[row] = answer.hi;
            yLow[2 * static_cast<int64_t>(row)] = answer.lo;
            yLow[2 * static_cast<int64_t>(row) + 1] = SpsvAbsF(answer.hi) * 0x1p-44f + 0x1p-119f / SpsvAbsF(diagonal);
        }
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT, int LANES, bool TIGHT = true>
__simt_callee__ inline void SpsvSolveFloatShadowRow(__gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd,
                                                    __gm__ const float* values, __gm__ const float* vecX,
                                                    __gm__ float* vecY, __gm__ float* yLow, __gm__ const PermT* diagPtr,
                                                    float alpha, int32_t row)
{
    RowPtrT start = rowPtr[row], end = rowPtr[row + 1];
    int32_t lane = static_cast<int32_t>(threadIdx.x) % LANES;
    float diagonal = NON_UNIT ? (diagPtr[row] >= 0 ? values[diagPtr[row]] : 0.0f) : 1.0f;
    bool diagonalOnly = NON_UNIT ? (end - start == 1 && diagPtr[row] == start) :
                                   (end == start || (end - start == 1 && static_cast<int32_t>(colInd[start]) == row));
    float x = vecX[row];
    if (diagonalOnly && diagonal != 0.0f && SpsvFastComponent(alpha) && SpsvFastComponent(x) &&
        SpsvFastComponent(diagonal)) {
        SpsvPublishFloatDiagonalOnly<LANES>(vecY, yLow, row, alpha, x, diagonal);
        return;
    }
    SpsvDF sum{0.0f, 0.0f};
    float error = 0.0f, predecessorError = 0.0f;
    uint32_t flags = 0;
    if (lane == 0) {
        if constexpr (TIGHT) {
            SpsvAccFloatEnclosed(sum.hi, sum.lo, error, alpha, x, 0.0f);
        } else {
            SpsvAccEnclosed(sum.hi, sum.lo, error, alpha, x, 0.0f);
        }
        flags |= SpsvFloatOperandFlags(alpha, x, diagonal);
    }
    for (SpsvCsrCursor<RowPtrT> q = static_cast<SpsvCsrCursor<RowPtrT>>(start) + lane; q < end; q += LANES) {
        int32_t col = static_cast<int32_t>(colInd[q]);
        if (FORWARD ? col < row : col > row) {
            float a = values[q], y = vecY[col], low = yLow[2 * static_cast<int64_t>(col)];
            flags |= SpsvFloatOperandFlags(a, y, low);
            predecessorError = SpsvFloatDependencyAcc(predecessorError, SpsvAbsF(a),
                                                      yLow[2 * static_cast<int64_t>(col) + 1]);
            if constexpr (TIGHT) {
                SpsvAccFloatEnclosed(sum.hi, sum.lo, error, -a, y, low);
            } else {
                SpsvAccEnclosed(sum.hi, sum.lo, error, -a, y, low);
            }
        }
    }
    if constexpr (LANES > 1) {
        sum = SpsvReduceFloatShadow<LANES, TIGHT>(sum, error, predecessorError, flags);
    }
    SpsvFinishFloatShadow<RowPtrT, ColIndT, FORWARD, LANES>(rowPtr, colInd, values, vecX, vecY, yLow, alpha, row,
                                                            diagonal, sum, error, predecessorError, flags);
}

#endif // OP_KERNEL_SOLVE_SPSV_REAL_SHADOW_H_
