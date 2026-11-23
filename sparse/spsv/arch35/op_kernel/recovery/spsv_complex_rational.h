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
#ifndef SPSV_COMPLEX_RATIONAL_H_
#define SPSV_COMPLEX_RATIONAL_H_

#include "spsv_rational.h"

__simt_callee__ inline uint32_t SpsvModAdd(uint32_t a, uint32_t b, uint32_t p)
{
    uint32_t sum = a + b;
    return sum >= p ? sum - p : sum;
}

__simt_callee__ inline uint32_t SpsvModSub(uint32_t a, uint32_t b, uint32_t p) { return a >= b ? a - b : p - (b - a); }

__simt_callee__ inline void SpsvBigCopy(SpsvBigUInt& a, const SpsvBigUInt& b)
{
    a.size = b.size;
    for (int64_t i = 0; i < b.size; ++i) {
        a.digit[i] = b.digit[i];
    }
}

__simt_callee__ inline void SpsvBigAdd(SpsvBigUInt& a, const SpsvBigUInt& b)
{
    int64_t size = a.size > b.size ? a.size : b.size;
    uint32_t carry = 0;
    for (int64_t i = 0; i < size; ++i) {
        uint32_t value = (i < a.size ? a.digit[i] : 0) + (i < b.size ? b.digit[i] : 0) + carry;
        a.digit[i] = static_cast<uint16_t>(value);
        carry = value >> 16;
    }
    a.size = size;
    if (carry != 0) {
        a.digit[a.size++] = static_cast<uint16_t>(carry);
    }
}

// The real 2m-by-2m representation has diagonal blocks [dr,-di;di,dr].
// After integer row scaling its determinant is the product of dr^2+di^2.
// Reconstruct each requested real/imaginary numerator against this positive
// denominator. No rounded predecessor is used in either the modular solve
// or its determinant bound.
__simt_callee__ inline void SpsvComplexRationalMarkTargets(__gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd,
                                                           __gm__ const SpsvComplex64* values, int64_t m,
                                                           const int64_t* targets, int32_t targetCount, bool forward,
                                                           __gm__ int32_t* scale)
{
    for (int64_t r = 0; r < m; ++r) {
        scale[r] = kSpsvRationalUnmarked;
    }
    for (int32_t i = 0; i < targetCount; ++i) {
        scale[targets[i] / 2] = 0;
    }
    for (int64_t step = 0; step < m; ++step) {
        int64_t r = forward ? m - 1 - step : step;
        if (scale[r] == kSpsvRationalUnmarked) {
            continue;
        }
        for (int64_t p = rowPtr[r]; p < rowPtr[r + 1]; ++p) {
            int32_t c = colInd[p];
            bool nonzero = SpsvRationalNonzero(values[p].real) || SpsvRationalNonzero(values[p].imag);
            if ((forward ? c < r : c > r) && nonzero) {
                scale[c] = 0;
            }
        }
    }
}

__simt_callee__ inline void SpsvComplexRationalProductBounds(SpsvComplex64 alpha, float valueReal, float valueImag,
                                                             int32_t& lowest, int32_t& highest)
{
    for (int32_t a = 0; a < 2; ++a) {
        for (int32_t x = 0; x < 2; ++x) {
            int32_t ea, ex;
            bool na, nx;
            uint32_t ma = SpsvRationalParts(a == 0 ? alpha.real : alpha.imag, ea, na);
            uint32_t mx = SpsvRationalParts(x == 0 ? valueReal : valueImag, ex, nx);
            if (ma != 0 && mx != 0) {
                lowest = ea + ex < lowest ? ea + ex : lowest;
                int32_t top = ea + ex + SpsvUIntBits(ma) + SpsvUIntBits(mx);
                highest = top > highest ? top : highest;
            }
        }
    }
}

__simt_callee__ inline bool SpsvComplexRationalDiagonalBounds(SpsvComplex64 diagonal, int32_t& lowest, int32_t& highest)
{
    int32_t er, ei;
    bool nr, ni;
    uint32_t mr = SpsvRationalParts(diagonal.real, er, nr);
    uint32_t mi = SpsvRationalParts(diagonal.imag, ei, ni);
    if (mr == 0 && mi == 0) {
        return false;
    }
    lowest = mr != 0 ? er : ei;
    highest = lowest;
    for (int32_t part = 0; part < 2; ++part) {
        float component = part == 0 ? diagonal.real : diagonal.imag;
        int32_t exponent;
        bool negative;
        uint32_t mantissa = SpsvRationalParts(component, exponent, negative);
        if (mantissa != 0) {
            lowest = exponent < lowest ? exponent : lowest;
            int32_t top = exponent + SpsvUIntBits(mantissa);
            highest = top > highest ? top : highest;
        }
    }
    return true;
}

__simt_callee__ inline void SpsvComplexRationalValueBounds(__gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd,
                                                           __gm__ const SpsvComplex64* values, int64_t row,
                                                           bool forward, int32_t& lowest, int32_t& highest,
                                                           uint64_t& terms)
{
    for (int64_t p = rowPtr[row]; p < rowPtr[row + 1]; ++p) {
        int32_t c = colInd[p];
        if (!(forward ? c < row : c > row)) {
            continue;
        }
        for (int32_t part = 0; part < 2; ++part) {
            int32_t exponent;
            bool negative;
            uint32_t mantissa = SpsvRationalParts(part == 0 ? values[p].real : values[p].imag, exponent, negative);
            if (mantissa == 0) {
                continue;
            }
            lowest = exponent < lowest ? exponent : lowest;
            int32_t top = exponent + SpsvUIntBits(mantissa);
            highest = top > highest ? top : highest;
            ++terms;
        }
    }
}

__simt_callee__ inline bool SpsvComplexRationalExponentBounds(__gm__ const int32_t* rowPtr,
                                                              __gm__ const int32_t* colInd,
                                                              __gm__ const SpsvComplex64* values,
                                                              __gm__ const SpsvComplex64* rhs, SpsvComplex64 alpha,
                                                              int64_t row, bool forward, SpsvComplex64 diagonal,
                                                              int32_t& lowest, int32_t& highest, int32_t& termBits)
{
    if (!SpsvComplexRationalDiagonalBounds(diagonal, lowest, highest)) {
        return false;
    }
    SpsvComplexRationalProductBounds(alpha, rhs[row].real, rhs[row].imag, lowest, highest);
    uint64_t terms = 6;
    SpsvComplexRationalValueBounds(rowPtr, colInd, values, row, forward, lowest, highest, terms);
    termBits = 0;
    for (uint64_t n = terms - 1; n != 0; n >>= 1) {
        ++termBits;
    }
    return true;
}

__simt_callee__ inline bool SpsvComplexRationalAccumulateDeterminantRow(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ const int32_t* diagPtr, SpsvComplex64 alpha, int64_t row, bool forward,
    bool nonUnit, __gm__ int32_t* scale, int64_t& determinantBoundBits, SpsvBigUInt& denominator,
    SpsvBigUInt& temporary)
{
    SpsvComplex64 diagonal{1.0f, 0.0f};
    if (nonUnit) {
        if (diagPtr[row] < 0) {
            return false;
        }
        diagonal = {values[diagPtr[row]].real, values[diagPtr[row]].imag};
    }
    int32_t lowest, highest, termBits;
    if (!SpsvComplexRationalExponentBounds(rowPtr, colInd, values, rhs, alpha, row, forward, diagonal, lowest, highest,
                                           termBits)) {
        return false;
    }
    int32_t er, ei;
    bool nr, ni;
    uint32_t mr = SpsvRationalParts(diagonal.real, er, nr);
    uint32_t mi = SpsvRationalParts(diagonal.imag, ei, ni);
    determinantBoundBits += 2 * static_cast<int64_t>(highest - lowest + termBits);
    scale[row] = lowest;
    SpsvBigCopy(temporary, denominator);
    SpsvBigMul(denominator, mr);
    SpsvBigMul(denominator, mr);
    if (mr != 0) {
        SpsvBigShift(denominator, 2 * static_cast<int64_t>(er - lowest));
    }
    SpsvBigMul(temporary, mi);
    SpsvBigMul(temporary, mi);
    if (mi != 0) {
        SpsvBigShift(temporary, 2 * static_cast<int64_t>(ei - lowest));
    }
    SpsvBigAdd(denominator, temporary);
    return true;
}

__simt_callee__ inline bool SpsvComplexRationalBuildDenominator(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ const int32_t* diagPtr, SpsvComplex64 alpha, int64_t m, bool forward,
    bool nonUnit, __gm__ int32_t* scale, int64_t& determinantBoundBits, SpsvBigUInt& denominator,
    SpsvBigUInt& temporary)
{
    SpsvBigSet(denominator, 1);
    determinantBoundBits = 0;
    for (int64_t row = 0; row < m; ++row) {
        if (scale[row] == kSpsvRationalUnmarked) {
            continue;
        }
        if (!SpsvComplexRationalAccumulateDeterminantRow(rowPtr, colInd, values, rhs, diagPtr, alpha, row, forward,
                                                         nonUnit, scale, determinantBoundBits, denominator,
                                                         temporary)) {
            return false;
        }
    }
    return true;
}

__simt_callee__ inline bool SpsvComplexRationalSolvePrimeRow(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ const int32_t* diagPtr, SpsvComplex64 alpha, int64_t row, bool forward,
    bool nonUnit, __gm__ const int32_t* scale, uint32_t prime, __gm__ uint32_t* residue, uint32_t& determinant)
{
    SpsvComplex64 diagonal{1.0f, 0.0f};
    if (nonUnit) {
        diagonal = {values[diagPtr[row]].real, values[diagPtr[row]].imag};
    }
    uint32_t dr = SpsvRationalMod(diagonal.real, 1.0f, scale[row], prime);
    uint32_t di = SpsvRationalMod(diagonal.imag, 1.0f, scale[row], prime);
    uint32_t norm = SpsvModAdd(SpsvModMul(dr, dr, prime), SpsvModMul(di, di, prime), prime);
    if (norm == 0) {
        return false;
    }
    uint32_t re = SpsvModSub(SpsvRationalMod(alpha.real, rhs[row].real, scale[row], prime),
                             SpsvRationalMod(alpha.imag, rhs[row].imag, scale[row], prime), prime);
    uint32_t im = SpsvModAdd(SpsvRationalMod(alpha.real, rhs[row].imag, scale[row], prime),
                             SpsvRationalMod(alpha.imag, rhs[row].real, scale[row], prime), prime);
    for (int64_t p = rowPtr[row]; p < rowPtr[row + 1]; ++p) {
        int32_t c = colInd[p];
        if (!(forward ? c < row : c > row)) {
            continue;
        }
        if (!SpsvRationalNonzero(values[p].real) && !SpsvRationalNonzero(values[p].imag)) {
            continue;
        }
        uint32_t ar = SpsvRationalMod(values[p].real, 1.0f, scale[row], prime);
        uint32_t ai = SpsvRationalMod(values[p].imag, 1.0f, scale[row], prime);
        int64_t component = 2 * static_cast<int64_t>(c);
        uint32_t yr = residue[component], yi = residue[component + 1];
        re = SpsvModSub(re, SpsvModSub(SpsvModMul(ar, yr, prime), SpsvModMul(ai, yi, prime), prime), prime);
        im = SpsvModSub(im, SpsvModAdd(SpsvModMul(ar, yi, prime), SpsvModMul(ai, yr, prime), prime), prime);
    }
    uint32_t inverse = SpsvModPow(norm, prime - 2, prime);
    residue[2 * row] = SpsvModMul(SpsvModAdd(SpsvModMul(re, dr, prime), SpsvModMul(im, di, prime), prime), inverse,
                                  prime);
    residue[2 * row + 1] = SpsvModMul(SpsvModSub(SpsvModMul(im, dr, prime), SpsvModMul(re, di, prime), prime), inverse,
                                      prime);
    determinant = SpsvModMul(determinant, norm, prime);
    return true;
}

__simt_callee__ inline bool SpsvComplexRationalSolvePrime(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ const int32_t* diagPtr, SpsvComplex64 alpha, int64_t m, bool forward,
    bool nonUnit, __gm__ const int32_t* scale, uint32_t prime, __gm__ uint32_t* residue, uint32_t& determinant)
{
    determinant = 1;
    for (int64_t step = 0; step < m; ++step) {
        int64_t row = forward ? step : m - 1 - step;
        if (scale[row] == kSpsvRationalUnmarked) {
            continue;
        }
        if (!SpsvComplexRationalSolvePrimeRow(rowPtr, colInd, values, rhs, diagPtr, alpha, row, forward, nonUnit, scale,
                                              prime, residue, determinant)) {
            return false;
        }
    }
    return true;
}

struct SpsvComplexPrimeSystem {
    __gm__ const int32_t* rowPtr;
    __gm__ const int32_t* colInd;
    __gm__ const SpsvComplex64* values;
    __gm__ const SpsvComplex64* rhs;
    __gm__ const int32_t* diagPtr;
    SpsvComplex64 alpha;
    int64_t m;
    bool forward;
    bool nonUnit;
    __gm__ const int32_t* scale;
    __gm__ uint32_t* residue;

    __simt_callee__ inline bool Solve(uint32_t prime, uint32_t& determinant) const
    {
        return SpsvComplexRationalSolvePrime(rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, scale,
                                             prime, residue, determinant);
    }
};

__simt_callee__ inline bool SpsvComplexRationalSolveBatch(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ const int32_t* diagPtr, SpsvComplex64 alpha, int64_t m,
    const int64_t* targets, int32_t targetCount, bool forward, bool nonUnit, __gm__ int32_t* scale,
    __gm__ uint32_t* residue, SpsvBigUInt* numerators, SpsvBigUInt& modulus, SpsvBigUInt& denominator,
    SpsvBigUInt& temporary, float* answers, uint32_t* answerValid)
{
    if (targetCount <= 0 || targetCount > kSpsvRationalTargetBatch) {
        return false;
    }
    for (int32_t i = 0; i < targetCount; ++i) {
        answerValid[i] = 0;
    }
    SpsvComplexRationalMarkTargets(rowPtr, colInd, values, m, targets, targetCount, forward, scale);
    int64_t determinantBoundBits = 0;
    if (!SpsvComplexRationalBuildDenominator(rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, scale,
                                             determinantBoundBits, denominator, temporary)) {
        return false;
    }
    SpsvComplexPrimeSystem system{rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, scale, residue};
    SpsvRationalReconstructPrimes(targets, targetCount, residue, numerators, modulus, determinantBoundBits, system);
    for (int32_t i = 0; i < targetCount; ++i) {
        bool negative = SpsvBigCompareShift(modulus, numerators[i], 1) < 0;
        if (negative) {
            SpsvBigComplement(numerators[i], modulus);
        }
        answers[i] = SpsvRationalRound(numerators[i], denominator, negative);
        answerValid[i] = 1;
    }
    return true;
}

#endif // SPSV_COMPLEX_RATIONAL_H_
