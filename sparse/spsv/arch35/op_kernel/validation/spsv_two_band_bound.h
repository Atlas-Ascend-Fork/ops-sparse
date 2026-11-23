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
#ifndef SPSV_TWO_BAND_BOUND_H_
#define SPSV_TWO_BAND_BOUND_H_

#include "spsv_block_bound.h"

constexpr int32_t kSpsvTwoBandWords = 8;

__simt_callee__ inline float SpsvTwoBandBoundaryResidual(__gm__ const float* scratch, int64_t step,
                                                         float diagonal, float firstSubdiag, float h0,
                                                         float h1)
{
    SpsvDF jump{-1.0f, 0.0f};
    float jumpError = 0.0f;
    SpsvBlockAcc(jump, jumpError, diagonal, {step == 0 ? h0 : h1, 0.0f});
    float residual = SpsvBlockMagnitude(jump, jumpError, 0x1p-116f);
    if (step == 1) {
        SpsvDF offDiagonal{0.0f, 0.0f};
        float offDiagonalError = 0.0f;
        SpsvBlockAcc(offDiagonal, offDiagonalError, diagonal, {scratch[3], 0.0f});
        SpsvBlockAcc(offDiagonal, offDiagonalError, firstSubdiag, {scratch[2], 0.0f});
        residual = SpsvFloatBoundAdd(residual, SpsvBlockMagnitude(offDiagonal, offDiagonalError, 0x1p-116f));
    }
    return residual;
}

__simt_callee__ inline float SpsvTwoBandInteriorResidual(__gm__ const float* row, float d, float a, float b,
                                                         float f0, float f1, float h0, float h1)
{
    __gm__ const float* previous = row - kSpsvTwoBandWords;
    __gm__ const float* second = previous - kSpsvTwoBandWords;
    SpsvDF ef0{0.0f, 0.0f}, ef1{0.0f, 0.0f};
    float e0 = 0.0f, e1 = 0.0f;
    SpsvBlockAcc(ef0, e0, d, {f0, 0.0f});
    SpsvBlockAcc(ef0, e0, a, {previous[0], 0.0f});
    SpsvBlockAcc(ef0, e0, b, {second[0], 0.0f});
    SpsvBlockAcc(ef1, e1, d, {f1, 0.0f});
    SpsvBlockAcc(ef1, e1, a, {previous[1], 0.0f});
    SpsvBlockAcc(ef1, e1, b, {second[1], 0.0f});
    float ef =
        SpsvFloatBoundAdd(SpsvBlockMagnitude(ef0, e0, 0x1p-116f), SpsvBlockMagnitude(ef1, e1, 0x1p-116f));
    SpsvDF missing{0.0f, 0.0f}, dot{0.0f, 0.0f};
    float me = 0.0f, qe = 0.0f;
    SpsvBlockAcc(missing, me, second[0], {previous[2], 0.0f});
    SpsvBlockAcc(missing, me, second[1], {previous[3], 0.0f});
    SpsvBlockAcc(dot, qe, f0, {h0, 0.0f});
    SpsvBlockAcc(dot, qe, f1, {h1, 0.0f});
    SpsvDF jump{-1.0f, 0.0f};
    float je = 0.0f;
    SpsvBlockAcc(jump, je, d, dot);
    float jumpBound =
        SpsvFloatBoundAdd(SpsvBlockMagnitude(jump, je, 0x1p-116f),
                          SpsvFloatBoundMul(SpsvAbsF(d), SpsvFloatBoundUp(qe * 0x1p-20f + 0x1p-116f)));
    float normResidual = SpsvFloatBoundAdd(
        SpsvFloatBoundMul(ef, previous[4]),
        SpsvFloatBoundAdd(SpsvFloatBoundMul(SpsvAbsF(b), SpsvBlockMagnitude(missing, me, 0x1p-116f)),
                          jumpBound));
    return normResidual;
}

__simt_callee__ inline void SpsvTwoBandRowBounds(__gm__ const float* scratch, __gm__ const float* rhs,
                                                 float alpha, int64_t step, int64_t m, bool forward,
                                                 __gm__ const float* vecY, __gm__ const float* state,
                                                 float& normP, float& normResidual, float& defectNorm,
                                                 float& allowed)
{
    int64_t r = forward ? step : m - 1 - step;
    normP = normResidual = defectNorm = __builtin_inff();
    allowed = 0.0f;
    if (!SpsvBlockFiniteNormal(rhs[r]) || !SpsvBlockFiniteNormal(vecY[r]) ||
        !SpsvBlockFiniteNormal(state[2 * r])) {
        return;
    }
    __gm__ const float* row = scratch + kSpsvTwoBandWords * step;
    float f0 = row[0], f1 = row[1], h0 = row[2], h1 = row[3];
    if (!SpsvBlockFiniteNormal(f0) || !SpsvBlockFiniteNormal(f1) || !SpsvBlockFiniteNormal(h0) ||
        !SpsvBlockFiniteNormal(h1)) {
        return;
    }
    float d = row[5], a = row[6], b = row[7];
    SpsvDF defect{0.0f, 0.0f};
    float de = 0.0f;
    SpsvBlockAcc(defect, de, alpha, {rhs[r], 0.0f});
    SpsvBlockAcc(defect, de, -d, {vecY[r], state[2 * r]});
    if (step >= 1) {
        int64_t c = forward ? r - 1 : r + 1;
        SpsvBlockAcc(defect, de, -a, {vecY[c], state[2 * c]});
    }
    if (step >= 2) {
        int64_t c = forward ? r - 2 : r + 2;
        SpsvBlockAcc(defect, de, -b, {vecY[c], state[2 * c]});
    }
    defectNorm = SpsvBlockMagnitude(defect, de, 0x1p-116f);
    allowed = SpsvFloatAllowedRadius(vecY[r], state[2 * r]);
    normP = SpsvFloatBoundMul(SpsvFloatBoundAdd(SpsvAbsF(f0), SpsvAbsF(f1)), row[4]);
    if (step < 2) {
        normResidual = SpsvTwoBandBoundaryResidual(scratch, step, d, a, h0, h1);
        return;
    }
    normResidual = SpsvTwoBandInteriorResidual(row, d, a, b, f0, f1, h0, h1);
}

__simt_callee__ inline bool SpsvTwoBandPrefixError(int64_t completed, float normResidual, float normP,
                                                   float defectNorm, float allowed, float& error)
{
    if (completed < 0 || !(normResidual < 0.5f) || !SpsvIsFiniteF(normP)) {
        return false;
    }
    error = SpsvFloatBoundDiv(SpsvFloatBoundMul(normP, defectNorm), SpsvBoundDown(1.0f - normResidual));
    if (!(error < allowed)) {
        return false;
    }
    return true;
}

__simt_callee__ inline bool SpsvTwoBandTailCertified(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    __gm__ const float* rhs, const SpsvRationalIndexArray& diagPtr, float alpha, int64_t m, bool forward,
    bool nonUnit, __gm__ const float* vecY, __gm__ const float* state, int64_t completed, float normP,
    float normResidual, float defectNorm, float allowed)
{
    float error;
    if (!SpsvTwoBandPrefixError(completed, normResidual, normP, defectNorm, allowed, error)) {
        return false;
    }
    float tail[kSpsvBoundBlockRows];
    for (int64_t step = completed; step < m; ++step) {
        int64_t r = forward ? step : m - 1 - step;
        float d = nonUnit ? (diagPtr[r] >= 0 ? values[diagPtr[r]] : 0.0f) : 1.0f;
        if (!SpsvBlockFiniteNormal(d) || d == 0.0f || !SpsvBlockFiniteNormal(rhs[r]) ||
            !SpsvBlockFiniteNormal(vecY[r]) || !SpsvBlockFiniteNormal(state[2 * r])) {
            return false;
        }
        SpsvDF defect{0.0f, 0.0f};
        float de = 0.0f, propagated = 0.0f;
        SpsvBlockAcc(defect, de, alpha, {rhs[r], 0.0f});
        SpsvBlockAcc(defect, de, -d, {vecY[r], state[2 * r]});
        for (int64_t p = rowPtr[r]; p < rowPtr[r + 1]; ++p) {
            int64_t c = colInd[p], at = forward ? c : m - 1 - c;
            if (at >= step) {
                continue;
            }
            if (!SpsvBlockFiniteNormal(values[p])) {
                return false;
            }
            SpsvBlockAcc(defect, de, -values[p], {vecY[c], state[2 * c]});
            float previous = at < completed ? error : tail[at - completed];
            propagated = SpsvFloatBoundAdd(propagated, SpsvFloatBoundMul(SpsvAbsF(values[p]), previous));
        }
        float bound = SpsvFloatBoundDiv(
            SpsvFloatBoundAdd(propagated,
                              SpsvBlockMagnitude(
                                  defect, de, static_cast<float>(rowPtr[r + 1] - rowPtr[r] + 3) * 0x1p-118f)),
            SpsvAbsF(d));
        if (!(bound < SpsvFloatAllowedRadius(vecY[r], state[2 * r]))) {
            return false;
        }
        tail[step - completed] = bound;
    }
    return true;
}

#endif  // SPSV_TWO_BAND_BOUND_H_
