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

#ifndef SPSV_COMPLEX_DYADIC_DIVISION_H_
#define SPSV_COMPLEX_DYADIC_DIVISION_H_

__simt_callee__ inline void SpsvComplexDyadicCompactQuotient(__gm__ uint16_t* qr, __gm__ uint16_t* qi,
                                                             int32_t quotientCount, int32_t precision,
                                                             int32_t& rg, SpsvDyadicBound& error,
                                                             uint16_t* re, uint16_t* im);

__simt_callee__ inline SpsvDyadicBound SpsvComplexDyadicDivideError(SpsvComplex64 diagonal, float inverseNorm,
                                                                    SpsvDyadicBound error)
{
    uint32_t realBits, imagBits;
    __builtin_memcpy(&realBits, &diagonal.real, sizeof(realBits));
    __builtin_memcpy(&imagBits, &diagonal.imag, sizeof(imagBits));
    float dominant = (realBits & 0x7fffffffU) > (imagBits & 0x7fffffffU) ? diagonal.real : diagonal.imag;
    int32_t dominantExponent;
    bool dominantNegative;
    uint32_t dominantMantissa = SpsvRationalParts(dominant, dominantExponent, dominantNegative);
    if (inverseNorm > 0.0f && SpsvIsFiniteF(inverseNorm)) {
        int32_t inverseExponent;
        bool inverseNegative;
        uint32_t inverseMantissa = SpsvRationalParts(inverseNorm, inverseExponent, inverseNegative);
        error = SpsvDyadicBoundMul(error, inverseMantissa, inverseExponent, false);
    } else {
        error = SpsvDyadicBoundMul(error, dominantMantissa, dominantExponent, true);
        if (error.mantissa != 0.0f) {
            ++error.exponent;
        }
    }
    return error;
}

__simt_callee__ inline void SpsvComplexDyadicDivideCenter(
    SpsvComplex64 diagonal, int32_t ed, int32_t diagonalBits, float inverseNorm, int32_t precision,
    bool nonUnit, int32_t& rg, uint16_t* re, uint16_t* im, __gm__ uint16_t* qr, __gm__ uint16_t* qi,
    __gm__ uint16_t* norm, int32_t sr, int32_t si, int32_t& qrs, int32_t& qis, SpsvDyadicBound& error)
{
    if (!nonUnit) {
        return;
    }
    for (int32_t j = 0; j < kSpsvComplexQuotientLimbs; ++j) {
        qr[j] = qi[j] = 0;
    }
    for (int32_t j = 0; j < 2 * kSpsvComplexDiagonalLimbs + 1; ++j) {
        norm[j] = 0;
    }
    int32_t quotientCount = precision + 2 * kSpsvComplexDiagonalLimbs + 2;
    int32_t normCount = 2 * kSpsvComplexDiagonalLimbs + 1;
    qrs = 0;
    qis = 0;
    SpsvComplexDyadicScale(qr, qrs, re, sr, diagonal.real, ed, false, precision);
    SpsvComplexDyadicScale(qr, qrs, im, si, diagonal.imag, ed, false, precision);
    SpsvComplexDyadicScale(qi, qis, im, si, diagonal.real, ed, false, precision);
    SpsvComplexDyadicScale(qi, qis, re, sr, diagonal.imag, ed, true, precision);
    SpsvComplexDiagonalSquare(norm, diagonal.real, ed, normCount);
    SpsvComplexDiagonalSquare(norm, diagonal.imag, ed, normCount);
    rg -= ed + diagonalBits;
    for (int32_t j = quotientCount - 1; j >= 0; --j) {
        qr[j] = SpsvComplexShiftedWordGM(qr, quotientCount, diagonalBits, j);
        qi[j] = SpsvComplexShiftedWordGM(qi, quotientCount, diagonalBits, j);
    }
    bool remainder = SpsvComplexIntegerDivide(qr, quotientCount, norm, normCount);
    remainder |= SpsvComplexIntegerDivide(qi, quotientCount, norm, normCount);
    error = SpsvComplexDyadicDivideError(diagonal, inverseNorm, error);
    if (remainder) {
        error = SpsvDyadicBoundAdd(error, {1.0f, rg});
    }
    SpsvComplexDyadicCompactQuotient(qr, qi, quotientCount, precision, rg, error, re, im);
}

__simt_callee__ inline void SpsvComplexDyadicCompactQuotient(__gm__ uint16_t* qr, __gm__ uint16_t* qi,
                                                             int32_t quotientCount, int32_t precision,
                                                             int32_t& rg, SpsvDyadicBound& error,
                                                             uint16_t* re, uint16_t* im)
{
    int32_t br = SpsvComplexDyadicBits(qr, quotientCount);
    int32_t bi = SpsvComplexDyadicBits(qi, quotientCount);
    int32_t shift = ((br > bi ? br : bi) + 15) / 16 - precision;
    shift = shift > 0 ? shift : 0;
    bool discarded = false;
    for (int32_t j = 0; j < shift; ++j) {
        discarded |= qr[j] != 0 || qi[j] != 0;
    }
    rg += shift * 16;
    if (discarded) {
        error = SpsvDyadicBoundAdd(error, {1.0f, rg});
    }
    for (int32_t j = 0; j < precision; ++j) {
        re[j] = qr[j + shift];
        im[j] = qi[j + shift];
    }
}

__simt_callee__ inline void SpsvComplexDyadicStoreAndCertify(
    __gm__ SpsvComplex64* vecY, __gm__ float* yWide, __gm__ float* rowPrep, __gm__ int32_t* grid,
    __gm__ int32_t* signs, __gm__ int32_t* boundExponent, __gm__ float* boundMantissa,
    __gm__ uint16_t* digits, int64_t row, int32_t precision, int32_t rg, int32_t qrs, int32_t qis,
    SpsvDyadicBound error, uint16_t* re, uint16_t* im)
{
    grid[row] = rg;
    signs[2 * row] = qrs;
    signs[2 * row + 1] = qis;
    boundMantissa[row] = error.mantissa;
    boundExponent[row] = error.exponent;
    for (int32_t j = 0; j < kSpsvDyadicLimbs; ++j) {
        digits[2 * row * kSpsvDyadicLimbs + j] = j < precision ? re[j] : 0;
        digits[(2 * row + 1) * kSpsvDyadicLimbs + j] = j < precision ? im[j] : 0;
    }
    for (int32_t component = 0; component < 2; ++component) {
        float current = component ? vecY[row].imag : vecY[row].real;
        float low = yWide[4 * row + 2 * component + 1];
        float prior = rowPrep[18 * row + 9 + component];
        if (!SpsvFloatNeedsDependencyRetry(current, low, prior)) {
            continue;
        }
        const uint16_t* center = component ? im : re;
        int32_t sign = component ? qis : qrs;
        float answer = SpsvDyadicRound(center, precision, rg, sign);
        if (SpsvDyadicCertified(answer, error, center, precision, rg)) {
            if (component) {
                vecY[row].imag = answer;
            } else {
                vecY[row].real = answer;
            }
            yWide[4 * row + 2 * component] = answer;
            yWide[4 * row + 2 * component + 1] = 0.0f;
            rowPrep[18 * row + 9 + component] = SpsvIsFiniteF(answer) ? 0.0f : kSpsvCertifiedOverflowRadius;
        }
    }
}

__simt_callee__ inline bool SpsvComplexDyadicAnyRetry(__gm__ const SpsvComplex64* vecY,
                                                      __gm__ const float* yWide, __gm__ const float* rowPrep,
                                                      int64_t m)
{
    for (int64_t r = 0; r < m; ++r) {
        if (SpsvFloatNeedsDependencyRetry(vecY[r].real, yWide[4 * r + 1], rowPrep[18 * r + 9]) ||
            SpsvFloatNeedsDependencyRetry(vecY[r].imag, yWide[4 * r + 3], rowPrep[18 * r + 10])) {
            return true;
        }
    }
    return false;
}

__simt_callee__ inline bool SpsvComplexDyadicPrepareRow(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ const int32_t* diagPtr, __gm__ int32_t* grid,
    __gm__ uint16_t* digits, SpsvComplex64 alpha, int64_t r, bool forward, bool nonUnit, int32_t precision,
    SpsvComplex64& diagonal, int32_t& ed, int32_t& diagonalBits, int64_t& wideGrid)
{
    diagonal = {1.0f, 0.0f};
    if (nonUnit) {
        if (diagPtr[r] < 0) {
            grid[r] = kSpsvDyadicInvalid;
            return false;
        }
        diagonal = {values[diagPtr[r]].real, values[diagPtr[r]].imag};
    }
    if (!SpsvComplexDyadicExponentBounds(rowPtr, colInd, values, rhs, grid, digits, alpha, r, forward,
                                         precision, diagonal, ed, diagonalBits, wideGrid)) {
        grid[r] = kSpsvDyadicInvalid;
        return false;
    }
    return true;
}

// Inline the integer filter so callees share registers and non-overlapping
// array lifetimes instead of stacking their private frames at each call.
__simt_callee__ __attribute__((always_inline)) inline bool SpsvComplexDyadicRetry(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ const int32_t* diagPtr, SpsvComplex64 alpha, int64_t m,
    bool forward, bool nonUnit, __gm__ SpsvComplex64* vecY, __gm__ float* yWide, __gm__ float* rowPrep,
    __gm__ uint16_t* scratch)
{
    __gm__ int32_t* grid = (__gm__ int32_t*)scratch;
    __gm__ int32_t* signs = grid + m;
    __gm__ int32_t* boundExponent = signs + 2 * m;
    __gm__ float* boundMantissa = (__gm__ float*)(boundExponent + m);
    __gm__ uint16_t* digits = (__gm__ uint16_t*)(boundMantissa + m);
    __gm__ uint16_t* qr = digits + 2 * m * kSpsvDyadicLimbs;
    __gm__ uint16_t* qi = qr + kSpsvComplexQuotientLimbs;
    __gm__ uint16_t* norm = qi + kSpsvComplexQuotientLimbs;
    if (!SpsvComplexDyadicMarkRetryClosure(rowPtr, colInd, values, vecY, yWide, rowPrep, m, forward, grid)) {
        return false;
    }
    for (int32_t precision = 8; precision <= kSpsvDyadicLimbs; precision *= 2) {
        for (int64_t step = 0; step < m; ++step) {
            int64_t r = forward ? step : m - 1 - step;
            if (grid[r] == kSpsvDyadicInvalid) {
                continue;
            }
            SpsvComplex64 diagonal;
            int32_t ed, diagonalBits;
            int64_t wideGrid;
            if (!SpsvComplexDyadicPrepareRow(rowPtr, colInd, values, rhs, diagPtr, grid, digits, alpha, r,
                                             forward, nonUnit, precision, diagonal, ed, diagonalBits,
                                             wideGrid)) {
                continue;
            }
            int32_t rg = static_cast<int32_t>(wideGrid);
            uint16_t re[kSpsvDyadicLimbs] = {}, im[kSpsvDyadicLimbs] = {};
            int32_t sr = 0, si = 0;
            SpsvDyadicBound er{0.0f, 0}, ei{0.0f, 0};
            SpsvDyadicBound error = SpsvComplexDyadicBuildCenter(
                rowPtr, colInd, values, rhs, grid, signs, boundExponent, boundMantissa, digits, alpha, r,
                forward, precision, rg, re, im, sr, si, er, ei);
            int32_t qrs = sr, qis = si;
            // A unit diagonal remains the exact identity and bypasses division.
            SpsvComplexDyadicDivideCenter(diagonal, ed, diagonalBits, rowPrep[18 * r + 8], precision, nonUnit,
                                          rg, re, im, qr, qi, norm, sr, si, qrs, qis, error);
            SpsvComplexDyadicStoreAndCertify(vecY, yWide, rowPrep, grid, signs, boundExponent, boundMantissa,
                                             digits, r, precision, rg, qrs, qis, error, re, im);
        }
        if (!SpsvComplexDyadicAnyRetry(vecY, yWide, rowPrep, m)) {
            return false;
        }
    }
    return true;
}

#endif  // SPSV_COMPLEX_DYADIC_DIVISION_H_
