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

#ifndef SPSV_RATIONAL_H_
#define SPSV_RATIONAL_H_

#include "../../shared/spsv_tiling_data.h"
#include "../math/spsv_big_uint.h"

// Rare, exact dependency-subgraph retry. Binary32 inputs are dyadic rationals.
// Scale each marked row by a power of two to get an INTEGER triangular system.
// Its solution is N/D, where D is the product of the integer diagonals. Compute
// N modulo distinct primes, then reconstruct it with the Chinese remainder
// theorem. Stop only when the modulus exceeds twice a proved determinant
// bound, not after a fixed number of floating point words or refinement steps.
// No approximate predecessor solution participates in this calculation.
//
// A finite coefficient occupies at most 277 bits after scaling all binary32
// coefficients to integers. A product alpha*x extends the row exponent span
// to 554 bits. At most INT64_MAX entries add 63 bits to the row 1-norm bound.
// Thus 620 bits/row, plus the quotient's 149-bit subnormal shift, suffice for
// each numerator. Batched targets share the denominator and modulus, so scratch
// stays O(m), not O(m^2).
constexpr int32_t kSpsvRationalUnmarked = (-2147483647 - 1);

__simt_callee__ inline uint32_t SpsvModMul(uint32_t a, uint32_t b, uint32_t p)
{
    if (p < 2) {
        return 0;
    }
    a %= p;
    b %= p;
    if (p <= 65535U) {
        return a * b % p;
    }
    uint32_t result = 0;
    while (b != 0) {
        if ((b & 1U) != 0) {
            uint32_t sum = result + a;
            result = sum >= p ? sum - p : sum;
        }
        b >>= 1;
        if (b != 0) {
            uint32_t doubled = a + a;
            a = doubled >= p ? doubled - p : doubled;
        }
    }
    return result;
}

__simt_callee__ inline uint32_t SpsvModPow(uint32_t a, uint32_t exponent, uint32_t p)
{
    uint32_t result = 1;
    while (exponent != 0) {
        if ((exponent & 1U) != 0) {
            result = SpsvModMul(result, a, p);
        }
        exponent >>= 1;
        if (exponent != 0) {
            a = SpsvModMul(a, a, p);
        }
    }
    return result;
}

// Deterministic Miller-Rabin for p < 2^31 (bases 2, 3, 5, 7, 11).
__simt_callee__ inline bool SpsvRationalPrime(uint32_t p)
{
    const uint32_t bases[5] = {2, 3, 5, 7, 11};
    uint32_t d = p - 1;
    int32_t s = 0;
    while ((d & 1U) == 0) {
        d >>= 1;
        ++s;
    }
    for (int32_t i = 0; i < 5; ++i) {
        uint32_t x = SpsvModPow(bases[i], d, p);
        if (x == 1 || x == p - 1) {
            continue;
        }
        bool witness = true;
        for (int32_t j = 1; j < s; ++j) {
            x = SpsvModMul(x, x, p);
            if (x == p - 1) {
                witness = false;
                break;
            }
        }
        if (witness) {
            return false;
        }
    }
    return true;
}

__simt_callee__ inline void SpsvRationalNextPrime(uint32_t& prime, bool& ascending)
{
    if (!ascending && prime <= 13U) {
        prime = 65537U;
        ascending = true;
    } else {
        prime = ascending ? prime + 2U : prime - 2U;
    }
}

// Remove exact powers of two from the significand. Zero has no exponent
// constraint and must not lower a row's integer scaling exponent.
__simt_callee__ inline uint32_t SpsvRationalParts(float value, int32_t& exponent, bool& negative)
{
    uint32_t mantissa = SpsvFloatMantissa(value, exponent, negative);
    if (mantissa != 0) {
        while ((mantissa & 1U) == 0) {
            mantissa >>= 1;
            ++exponent;
        }
    }
    return mantissa;
}

__simt_callee__ inline bool SpsvRationalNonzero(float value)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7fffffffU) != 0;
}

__simt_callee__ inline uint32_t SpsvRationalMod(float a, float b, int32_t scale, uint32_t p)
{
    if (p < 2) {
        return 0;
    }
    int32_t ea, eb;
    bool na, nb;
    uint32_t ma = SpsvRationalParts(a, ea, na), mb = SpsvRationalParts(b, eb, nb);
    if (ma == 0 || mb == 0) {
        return 0;
    }
    uint32_t magnitude = SpsvModMul(SpsvModMul(ma % p, mb % p, p),
                                    SpsvModPow(2, static_cast<uint32_t>(ea + eb - scale), p), p);
    return na != nb && magnitude != 0 ? p - magnitude : magnitude;
}

// The cold retry has one VF for every supported index-width combination.
// Runtime index views avoid cloning the large integer algorithm six times.
struct SpsvRationalIndexArray {
    __gm__ const int32_t* small;
    __gm__ const int64_t* large;
    bool wide;

    __simt_callee__ inline int64_t operator[](int64_t index) const { return wide ? large[index] : small[index]; }
};

__simt_callee__ inline void SpsvRationalValidateRows(const SpsvRationalIndexArray& rowPtr,
                                                     const SpsvRationalIndexArray& colInd, __gm__ const float* values,
                                                     __gm__ const float* rhs, const SpsvRationalIndexArray& diagPtr,
                                                     float alpha, int64_t m, bool forward, bool nonUnit,
                                                     __gm__ uint32_t* residue)
{
    // Validate finite, nonsingular dependency chains once and share the
    // result across all target rows in this batch.
    for (int64_t step = 0; step < m; ++step) {
        int64_t r = forward ? step : m - 1 - step;
        float diagonal = nonUnit ? (diagPtr[r] >= 0 ? values[diagPtr[r]] : 0.0f) : 1.0f;
        bool valid = SpsvIsFiniteF(alpha) && SpsvIsFiniteF(rhs[r]) && SpsvIsFiniteF(diagonal) &&
                     SpsvRationalNonzero(diagonal);
        for (int64_t p = rowPtr[r]; p < rowPtr[r + 1]; ++p) {
            int32_t c = static_cast<int32_t>(colInd[p]);
            if (forward ? c < r : c > r) {
                valid &= SpsvIsFiniteF(values[p]);
                if (SpsvRationalNonzero(values[p])) {
                    valid &= residue[c] != 0;
                }
            }
        }
        residue[r] = valid ? 1U : 0U;
    }
}

__simt_callee__ inline void SpsvRationalMarkTargets(const SpsvRationalIndexArray& rowPtr,
                                                    const SpsvRationalIndexArray& colInd, __gm__ const float* values,
                                                    int64_t m, bool forward, const int32_t* active, int32_t activeCount,
                                                    __gm__ int32_t* scale)
{
    for (int64_t r = 0; r < m; ++r) {
        scale[r] = kSpsvRationalUnmarked;
    }
    for (int32_t i = 0; i < activeCount; ++i) {
        scale[active[i]] = 0;
    }
    // The row order is topological. Walk backwards to mark the union of the
    // targets' nonzero dependency subgraphs.
    for (int64_t step = 0; step < m; ++step) {
        int64_t r = forward ? m - 1 - step : step;
        if (scale[r] == kSpsvRationalUnmarked) {
            continue;
        }
        for (int64_t p = rowPtr[r]; p < rowPtr[r + 1]; ++p) {
            int32_t c = static_cast<int32_t>(colInd[p]);
            if ((forward ? c < r : c > r) && SpsvRationalNonzero(values[p])) {
                scale[c] = 0;
            }
        }
    }
}

__simt_callee__ inline void SpsvRationalBuildDenominator(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    __gm__ const float* rhs, const SpsvRationalIndexArray& diagPtr, float alpha, int64_t m, bool forward, bool nonUnit,
    __gm__ int32_t* scale, SpsvBigUInt& denominator, bool& negativeDenominator, int64_t& determinantBoundBits)
{
    for (int64_t r = 0; r < m; ++r) {
        if (scale[r] == kSpsvRationalUnmarked) {
            continue;
        }
        float diagonal = nonUnit ? values[diagPtr[r]] : 1.0f;
        int32_t ed, ea, ex;
        bool nd, na, nx;
        uint32_t md = SpsvRationalParts(diagonal, ed, nd);
        uint32_t ma = SpsvRationalParts(alpha, ea, na), mx = SpsvRationalParts(rhs[r], ex, nx);
        int32_t lowest = ed;
        int32_t highest = ed + SpsvUIntBits(md);
        if (ma != 0 && mx != 0) {
            lowest = ea + ex < lowest ? ea + ex : lowest;
            int32_t rhsTop = ea + ex + SpsvUIntBits(ma) + SpsvUIntBits(mx);
            highest = rhsTop > highest ? rhsTop : highest;
        }
        uint64_t terms = 2;
        for (int64_t p = rowPtr[r]; p < rowPtr[r + 1]; ++p) {
            int32_t c = static_cast<int32_t>(colInd[p]);
            if ((forward ? c < r : c > r) && SpsvRationalNonzero(values[p])) {
                int32_t e;
                bool negative;
                uint32_t mantissa = SpsvRationalParts(values[p], e, negative);
                lowest = e < lowest ? e : lowest;
                int32_t top = e + SpsvUIntBits(mantissa);
                highest = top > highest ? top : highest;
                ++terms;
            }
        }
        int32_t termBits = 0;
        for (uint64_t count = terms - 1; count != 0; count >>= 1) {
            ++termBits;
        }
        // Each determinant term selects one entry per row. The product of
        // (sum |a_rc| + |b_r|) bounds BOTH the numerator determinant and D.
        determinantBoundBits += highest - lowest + termBits;
        scale[r] = lowest;
        SpsvBigMul(denominator, md);
        SpsvBigShift(denominator, ed - lowest);
        negativeDenominator ^= nd;
    }
}

__simt_callee__ inline bool SpsvRationalSolvePrime(const SpsvRationalIndexArray& rowPtr,
                                                   const SpsvRationalIndexArray& colInd, __gm__ const float* values,
                                                   __gm__ const float* rhs, const SpsvRationalIndexArray& diagPtr,
                                                   float alpha, int64_t m, bool forward, bool nonUnit,
                                                   __gm__ const int32_t* scale, __gm__ uint32_t* residue,
                                                   uint32_t prime, uint32_t& det)
{
    det = 1;
    bool usable = true;
    for (int64_t step = 0; step < m; ++step) {
        int64_t r = forward ? step : m - 1 - step;
        if (scale[r] == kSpsvRationalUnmarked) {
            continue;
        }
        float diagonal = nonUnit ? values[diagPtr[r]] : 1.0f;
        uint32_t dm = SpsvRationalMod(diagonal, 1.0f, scale[r], prime);
        if (dm == 0) {
            usable = false;
            break;
        }
        uint32_t sum = SpsvRationalMod(alpha, rhs[r], scale[r], prime);
        for (int64_t p = rowPtr[r]; p < rowPtr[r + 1]; ++p) {
            int32_t c = static_cast<int32_t>(colInd[p]);
            if ((forward ? c < r : c > r) && SpsvRationalNonzero(values[p])) {
                uint32_t product = SpsvModMul(SpsvRationalMod(values[p], 1.0f, scale[r], prime), residue[c], prime);
                sum = sum >= product ? sum - product : prime - (product - sum);
            }
        }
        residue[r] = SpsvModMul(sum, SpsvModPow(dm, prime - 2, prime), prime);
        det = SpsvModMul(det, dm, prime);
    }
    return usable;
}

#include "spsv_rational_reconstruction.h"

struct SpsvRealPrimeSystem {
    const SpsvRationalIndexArray& rowPtr;
    const SpsvRationalIndexArray& colInd;
    __gm__ const float* values;
    __gm__ const float* rhs;
    const SpsvRationalIndexArray& diagPtr;
    float alpha;
    int64_t m;
    bool forward;
    bool nonUnit;
    __gm__ const int32_t* scale;
    __gm__ uint32_t* residue;

    __simt_callee__ inline bool Solve(uint32_t prime, uint32_t& determinant) const
    {
        return SpsvRationalSolvePrime(rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, scale, residue,
                                      prime, determinant);
    }
};

__simt_callee__ inline void SpsvRationalReconstruct(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    __gm__ const float* rhs, const SpsvRationalIndexArray& diagPtr, float alpha, int64_t m, bool forward, bool nonUnit,
    const int32_t* active, int32_t activeCount, __gm__ const int32_t* scale, __gm__ uint32_t* residue,
    SpsvBigUInt* numerators, SpsvBigUInt& modulus, int64_t determinantBoundBits)
{
    SpsvRealPrimeSystem system{rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, scale, residue};
    SpsvRationalReconstructPrimes(active, activeCount, residue, numerators, modulus, determinantBoundBits, system);
}

__simt_callee__ inline bool SpsvRationalSolveBatch(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    __gm__ const float* rhs, const SpsvRationalIndexArray& diagPtr, float alpha, int64_t m, const int32_t* targets,
    int32_t targetCount, bool forward, bool nonUnit, __gm__ int32_t* scale, __gm__ uint32_t* residue,
    SpsvBigUInt* numerators, SpsvBigUInt& modulus, SpsvBigUInt& denominator, float* answers, uint32_t* answerValid)
{
    if (targetCount <= 0 || targetCount > kSpsvRationalTargetBatch) {
        return false;
    }
    for (int32_t i = 0; i < targetCount; ++i) {
        answerValid[i] = 0;
    }

    SpsvRationalValidateRows(rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, residue);

    int32_t active[kSpsvRationalTargetBatch];
    int32_t activeInput[kSpsvRationalTargetBatch];
    int32_t activeCount = 0;
    for (int32_t i = 0; i < targetCount; ++i) {
        if (residue[targets[i]] != 0) {
            active[activeCount] = targets[i];
            activeInput[activeCount++] = i;
        }
    }
    if (activeCount == 0) {
        return true;
    }

    SpsvRationalMarkTargets(rowPtr, colInd, values, m, forward, active, activeCount, scale);

    SpsvBigSet(denominator, 1);
    bool negativeDenominator = false;
    int64_t determinantBoundBits = 0;
    SpsvRationalBuildDenominator(rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, scale, denominator,
                                 negativeDenominator, determinantBoundBits);

    SpsvRationalReconstruct(rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, active, activeCount,
                            scale, residue, numerators, modulus, determinantBoundBits);
    for (int32_t i = 0; i < activeCount; ++i) {
        bool negativeNumerator = SpsvBigCompareShift(modulus, numerators[i], 1) < 0;
        if (negativeNumerator) {
            SpsvBigComplement(numerators[i], modulus);
        }
        answers[activeInput[i]] = SpsvRationalRound(numerators[i], denominator,
                                                    negativeNumerator != negativeDenominator);
        answerValid[activeInput[i]] = 1;
    }
    return true;
}

#endif // SPSV_RATIONAL_H_
