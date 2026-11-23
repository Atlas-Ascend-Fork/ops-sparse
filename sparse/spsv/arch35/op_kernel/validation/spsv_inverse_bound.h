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

#ifndef SPSV_INVERSE_BOUND_H_
#define SPSV_INVERSE_BOUND_H_

#include "spsv_block_bound.h"

constexpr int32_t kSpsvInverseWindow = 64;
constexpr int32_t kSpsvInverseScratch = kSpsvInverseWindow + kSpsvBoundBandwidth;
constexpr int32_t kSpsvInverseRowWords = kSpsvBoundBandwidth + 2;

// Interleave each row's scratch by participating thread. This keeps the
// verification workspace coalesced and avoids a large per-thread stack.
__simt_callee__ inline void SpsvInverseResidualAcc(__gm__ float* scratch, int32_t pitch, int32_t tid, int32_t index,
                                                   float q, float a)
{
    SpsvDF sum{scratch[(kSpsvInverseScratch + index) * pitch + tid],
               scratch[(2 * kSpsvInverseScratch + index) * pitch + tid]};
    float error = scratch[(3 * kSpsvInverseScratch + index) * pitch + tid];
    SpsvBlockAcc(sum, error, q, {a, 0.0f});
    scratch[(kSpsvInverseScratch + index) * pitch + tid] = sum.hi;
    scratch[(2 * kSpsvInverseScratch + index) * pitch + tid] = sum.lo;
    scratch[(3 * kSpsvInverseScratch + index) * pitch + tid] = error;
}

__simt_callee__ inline bool SpsvBuildApproximateInverse(const SpsvRationalIndexArray& rowPtr,
                                                        const SpsvRationalIndexArray& colInd,
                                                        __gm__ const float* values,
                                                        const SpsvRationalIndexArray& diagPtr, int64_t m, bool forward,
                                                        bool nonUnit, int64_t step, int64_t first, int64_t lower,
                                                        __gm__ float* scratch, int32_t pitch, int32_t tid)
{
    for (int64_t at = step; at >= first; --at) {
        int64_t r = forward ? at : m - 1 - at;
        float d = nonUnit ? (diagPtr[r] >= 0 ? values[diagPtr[r]] : 0.0f) : 1.0f;
        if (!SpsvBlockFiniteNormal(d) || d == 0.0f) {
            return false;
        }
        float q = scratch[(at - lower) * pitch + tid] / d;
        if (!SpsvBlockFiniteNormal(q)) {
            return false;
        }
        scratch[(at - lower) * pitch + tid] = q;
        for (int64_t p = rowPtr[r]; p < rowPtr[r + 1]; ++p) {
            int64_t c = colInd[p], logical = forward ? c : m - 1 - c;
            if (logical >= at) {
                continue;
            }
            float a = values[p];
            if (!SpsvBlockFiniteNormal(a) || logical < at - kSpsvBoundBandwidth) {
                return false;
            }
            scratch[(logical - lower) * pitch + tid] = fmaf(-q, a, scratch[(logical - lower) * pitch + tid]);
        }
    }
    return true;
}

__simt_callee__ inline bool SpsvTransformInverseDefect(const SpsvRationalIndexArray& rowPtr,
                                                       const SpsvRationalIndexArray& colInd, __gm__ const float* values,
                                                       const SpsvRationalIndexArray& diagPtr, int64_t m, bool forward,
                                                       bool nonUnit, int64_t step, int64_t first, int64_t lower,
                                                       __gm__ float* scratch, int32_t pitch, int32_t tid,
                                                       __gm__ const float* rhs, float alpha, __gm__ const float* vecY,
                                                       __gm__ const float* state, SpsvDF& transformed,
                                                       float& transformedError, float& transformedRadius)
{
    for (int64_t at = first; at <= step; ++at) {
        int64_t r = forward ? at : m - 1 - at;
        float q = scratch[(at - lower) * pitch + tid];
        float d = nonUnit ? values[diagPtr[r]] : 1.0f;
        if (!SpsvBlockFiniteNormal(alpha) || !SpsvBlockFiniteNormal(rhs[r]) || !SpsvBlockFiniteNormal(vecY[r]) ||
            !SpsvBlockFiniteNormal(state[2 * r])) {
            return false;
        }
        SpsvDF defect{0.0f, 0.0f};
        float de = 0.0f;
        SpsvBlockAcc(defect, de, alpha, {rhs[r], 0.0f});
        SpsvBlockAcc(defect, de, -d, {vecY[r], state[2 * r]});
        SpsvInverseResidualAcc(scratch, pitch, tid, at - lower, -q, d);
        for (int64_t p = rowPtr[r]; p < rowPtr[r + 1]; ++p) {
            int64_t c = colInd[p], logical = forward ? c : m - 1 - c;
            if (logical >= at) {
                continue;
            }
            if (!SpsvBlockFiniteNormal(vecY[c]) || !SpsvBlockFiniteNormal(state[2 * c])) {
                return false;
            }
            float a = values[p];
            SpsvBlockAcc(defect, de, -a, {vecY[c], state[2 * c]});
            SpsvInverseResidualAcc(scratch, pitch, tid, logical - lower, -q, a);
        }
        SpsvBlockAcc(transformed, transformedError, q, defect);
        float floor = static_cast<float>(rowPtr[r + 1] - rowPtr[r] + 3) * 0x1p-118f;
        transformedRadius = SpsvFloatBoundAdd(transformedRadius,
                                              SpsvFloatBoundMul(SpsvAbsF(q), SpsvFloatBoundUp(de * 0x1p-20f + floor)));
    }
    return true;
}

__simt_callee__ inline void SpsvReduceInverseBounds(int64_t m, bool forward, int64_t step, int64_t first, int64_t lower,
                                                    int32_t count, __gm__ const float* scratch, int32_t pitch,
                                                    int32_t tid, __gm__ const float* vecY, __gm__ const float* state,
                                                    float transformedRadius, SpsvDF transformed, float transformedError,
                                                    float allowed, float& residualNorm, float& defectNorm,
                                                    __gm__ float* rowBound)
{
    float unknown = 0.0f, interior = 0.0f, diagonal = 0.0f;
    float known = SpsvFloatBoundAdd(transformedRadius, SpsvBlockMagnitude(transformed, transformedError, 0x1p-110f));
    for (int32_t j = 0; j < count; ++j) {
        int64_t logical = lower + j, r = forward ? logical : m - 1 - logical;
        SpsvDF residual{scratch[(kSpsvInverseScratch + j) * pitch + tid],
                        scratch[(2 * kSpsvInverseScratch + j) * pitch + tid]};
        float coefficient = SpsvBlockMagnitude(residual, scratch[(3 * kSpsvInverseScratch + j) * pitch + tid],
                                               0x1p-110f);
        float radius = state[2 * r + 1];
        float margin = SpsvFloatAllowedRadius(vecY[r], state[2 * r]);
        if (radius < 0.0f || !(margin > 0.0f)) {
            return;
        }
        if (radius <= margin) {
            known = SpsvFloatBoundAdd(known, SpsvFloatBoundMul(coefficient, radius));
        } else {
            float term = SpsvFloatBoundMul(coefficient, margin);
            unknown = SpsvFloatBoundAdd(unknown, term);
            if (logical == step) {
                diagonal = coefficient;
            } else if (logical >= first) {
                interior = SpsvFloatBoundAdd(interior, term);
            } else {
                rowBound[2 + j] = SpsvFloatBoundDiv(term, allowed);
            }
        }
    }
    residualNorm = SpsvFloatBoundDiv(unknown, allowed);
    defectNorm = SpsvFloatBoundDiv(known, allowed);
    rowBound[0] = SpsvFloatBoundDiv(SpsvFloatBoundAdd(known, interior), allowed);
    rowBound[1] = diagonal;
}

// One row of a sparse approximate inverse P. Construction may round freely:
// the separately enclosed residual R=I-P*A validates the actual stored P.
// Certified rows supply their known radii; only uncertified components
// participate in the weighted infinity-norm contraction.
__simt_callee__ inline void SpsvFloatInverseRowBound(const SpsvRationalIndexArray& rowPtr,
                                                     const SpsvRationalIndexArray& colInd, __gm__ const float* values,
                                                     __gm__ const float* rhs, const SpsvRationalIndexArray& diagPtr,
                                                     float alpha, int64_t m, bool forward, bool nonUnit,
                                                     __gm__ const float* vecY, __gm__ const float* state, int64_t row,
                                                     __gm__ float* scratch, int32_t pitch, int32_t tid,
                                                     float& residualNorm, float& defectNorm, __gm__ float* rowBound)
{
    residualNorm = defectNorm = __builtin_inff();
    for (int32_t j = 0; j < kSpsvInverseRowWords; ++j) {
        rowBound[j] = 0.0f;
    }
    rowBound[0] = __builtin_inff();
    float allowed = SpsvFloatAllowedRadius(vecY[row], state[2 * row]);
    if (!(allowed > 0.0f) || state[2 * row + 1] < 0.0f) {
        return;
    }
    if (state[2 * row + 1] <= allowed) {
        residualNorm = defectNorm = 0.0f;
        return;
    }
    int64_t step = forward ? row : m - 1 - row;
    int64_t first = step >= kSpsvInverseWindow ? step - kSpsvInverseWindow + 1 : 0;
    int64_t lower = first > kSpsvBoundBandwidth ? first - kSpsvBoundBandwidth : 0;
    int32_t count = static_cast<int32_t>(step - lower + 1);
    for (int32_t j = 0; j < 4 * kSpsvInverseScratch; ++j) {
        scratch[j * pitch + tid] = 0.0f;
    }
    scratch[(count - 1) * pitch + tid] = 1.0f;
    if (!SpsvBuildApproximateInverse(rowPtr, colInd, values, diagPtr, m, forward, nonUnit, step, first, lower, scratch,
                                     pitch, tid)) {
        return;
    }
    scratch[(kSpsvInverseScratch + count - 1) * pitch + tid] = 1.0f;
    SpsvDF transformed{0.0f, 0.0f};
    float transformedError = 0.0f, transformedRadius = 0.0f;
    if (!SpsvTransformInverseDefect(rowPtr, colInd, values, diagPtr, m, forward, nonUnit, step, first, lower, scratch,
                                    pitch, tid, rhs, alpha, vecY, state, transformed, transformedError,
                                    transformedRadius)) {
        return;
    }
    SpsvReduceInverseBounds(m, forward, step, first, lower, count, scratch, pitch, tid, vecY, state, transformedRadius,
                            transformed, transformedError, allowed, residualNorm, defectNorm, rowBound);
}

// R=I-P*A is triangular. Previously certified rows bound the small interior
// defect; propagate the remaining, at most bandwidth-wide tail in solve
// order. This avoids requiring a global contraction when only a few rows
// have a tail norm above one. Each new radius is checked against its own
// original acceptance margin, including the diagonal defect's denominator.
__simt_callee__ inline bool SpsvFloatInverseTailRowBound(__gm__ const float* vecY, __gm__ const float* state, int64_t m,
                                                         bool forward, int32_t blocks, __gm__ const float* normalized,
                                                         int64_t r, float& radius)
{
    __gm__ const float* rows = state + 4 * m + 2 * blocks;
    float allowed = SpsvFloatAllowedRadius(vecY[r], state[2 * r]);
    float error = state[2 * r + 1];
    if (!(allowed > 0.0f) || !(error >= 0.0f)) {
        return false;
    }
    if (error <= allowed) {
        radius = SpsvFloatBoundDiv(error, allowed);
        return true;
    }
    __gm__ const float* row = rows + kSpsvInverseRowWords * r;
    if (!(row[1] >= 0.0f && row[1] < 1.0f)) {
        return false;
    }
    radius = row[0];
    int64_t step = forward ? r : m - 1 - r;
    int64_t first = step >= kSpsvInverseWindow ? step - kSpsvInverseWindow + 1 : 0;
    int64_t lower = first > kSpsvBoundBandwidth ? first - kSpsvBoundBandwidth : 0;
    for (int64_t at = lower; at < first; ++at) {
        int64_t c = forward ? at : m - 1 - at;
        radius = SpsvFloatBoundAdd(radius, SpsvFloatBoundMul(row[2 + at - lower], normalized[c]));
    }
    radius = SpsvFloatBoundDiv(radius, SpsvBoundDown(1.0f - row[1]));
    return radius < 1.0f;
}

#endif // SPSV_INVERSE_BOUND_H_
