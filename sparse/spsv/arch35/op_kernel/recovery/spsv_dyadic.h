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
#ifndef SPSV_DYADIC_H_
#define SPSV_DYADIC_H_

#include "../common/spsv_simt_common.h"

#include "../validation/spsv_float_bound.h"
#include "spsv_rational.h"
#include "spsv_dyadic_layout.h"

// Certified bounded-precision filter before exact CRT reconstruction. Each
// value is a signed integer times 2^exponent; discarded integer digits enter
// a separately scaled, outward-rounded radius. Thus FTZ cannot erase a low
// solution word. Every tier starts from original inputs, never rounded Y.
constexpr int32_t kSpsvDyadicInvalid = (-2147483647 - 1);

struct SpsvDyadicBound {
    float mantissa;
    int32_t exponent;
};

__simt_callee__ inline SpsvDyadicBound SpsvDyadicNormalize(float value, int32_t exponent)
{
    if (!SpsvIsFiniteF(value)) {
        return {__builtin_inff(), 0x7fffffff};
    }
    if (value == 0.0f) {
        return {0.0f, 0};
    }
    int32_t e = SpsvExpOf(value);
    return {ldexpf(value, -e), exponent + e};
}

__simt_callee__ inline SpsvDyadicBound SpsvDyadicBoundAdd(SpsvDyadicBound a, SpsvDyadicBound b)
{
    if (!SpsvIsFiniteF(a.mantissa) || !SpsvIsFiniteF(b.mantissa)) {
        return {__builtin_inff(), 0x7fffffff};
    }
    if (a.mantissa == 0.0f) {
        return b;
    }
    if (b.mantissa == 0.0f) {
        return a;
    }
    if (a.exponent < b.exponent) {
        SpsvDyadicBound swap = a;
        a = b;
        b = swap;
    }
    int64_t shift = static_cast<int64_t>(b.exponent) - a.exponent;
    float low = shift < -64 ? 0.0f : ldexpf(b.mantissa, static_cast<int32_t>(shift));
    return SpsvDyadicNormalize(SpsvFloatBoundUp(a.mantissa + low), a.exponent);
}

__simt_callee__ inline SpsvDyadicBound SpsvDyadicBoundMul(SpsvDyadicBound bound, uint32_t mantissa, int32_t exponent,
                                                          bool divide)
{
    if (bound.mantissa == 0.0f || mantissa == 0) {
        return {0.0f, 0};
    }
    float factor = static_cast<float>(mantissa);
    float value = divide ? bound.mantissa / factor : bound.mantissa * factor;
    return SpsvDyadicNormalize(SpsvFloatBoundUp(value), divide ? bound.exponent - exponent : bound.exponent + exponent);
}

__simt_callee__ inline int32_t SpsvDyadicBits(const uint16_t* words, int32_t count)
{
    while (count > 0 && words[count - 1] == 0) {
        --count;
    }
    return count == 0 ? 0 : (count - 1) * 16 + SpsvUIntBits(words[count - 1]);
}

// Align an exact integer product onto the accumulator grid. Truncation is
// toward zero and its magnitude is strictly below one accumulator unit.
__simt_callee__ inline bool SpsvDyadicAlign(uint16_t* aligned, const uint16_t* source, int32_t sourceCount,
                                            int32_t shift, int32_t count)
{
    bool lost = false;
    for (int32_t i = 0; i < count; ++i) {
        aligned[i] = 0;
    }
    for (int32_t i = 0; i < sourceCount; ++i) {
        uint32_t word = source[i];
        int64_t bit = static_cast<int64_t>(i) * 16 + shift;
        if (bit <= -16) {
            lost |= word != 0;
            continue;
        }
        if (bit < 0) {
            int32_t down = static_cast<int32_t>(-bit);
            lost |= (word & ((1U << down) - 1U)) != 0;
            aligned[0] |= static_cast<uint16_t>(word >> down);
            continue;
        }
        int32_t index = static_cast<int32_t>(bit / 16);
        int32_t offset = static_cast<int32_t>(bit % 16);
        if (index < count) {
            aligned[index] |= static_cast<uint16_t>(word << offset);
        }
        if (offset != 0 && index + 1 < count) {
            aligned[index + 1] |= static_cast<uint16_t>(word >> (16 - offset));
        }
    }
    return lost;
}

__simt_callee__ inline void SpsvDyadicAccumulate(uint16_t* sum, int32_t& sign, const uint16_t* term, int32_t termSign,
                                                 int32_t count)
{
    if (sign == 0 || sign == termSign) {
        uint32_t carry = 0;
        for (int32_t i = 0; i < count; ++i) {
            uint32_t value = static_cast<uint32_t>(sum[i]) + term[i] + carry;
            sum[i] = static_cast<uint16_t>(value);
            carry = value >> 16;
        }
        sign = termSign;
        return;
    }
    int32_t comparison = 0;
    for (int32_t i = count - 1; i >= 0 && comparison == 0; --i) {
        comparison = sum[i] > term[i] ? 1 : sum[i] < term[i] ? -1 : 0;
    }
    uint32_t borrow = 0;
    for (int32_t i = 0; i < count; ++i) {
        uint32_t a = comparison >= 0 ? sum[i] : term[i];
        uint32_t b = (comparison >= 0 ? term[i] : sum[i]) + borrow;
        sum[i] = static_cast<uint16_t>(a - b);
        borrow = a < b;
    }
    if (comparison <= 0) {
        sign = comparison == 0 ? 0 : termSign;
    }
}

// Integer division by a binary32 significand. Two base-256 steps keep each
// dividend below 2^32, avoiding a software uint64 division in every digit.
__simt_callee__ inline uint32_t SpsvDyadicDivide(uint16_t* words, int32_t count, uint32_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    uint32_t remainder = 0;
    for (int32_t i = count - 1; i >= 0; --i) {
        uint32_t high = (remainder << 8) | (words[i] >> 8);
        uint32_t q = high / divisor;
        remainder = high % divisor;
        uint32_t low = (remainder << 8) | (words[i] & 255U);
        words[i] = static_cast<uint16_t>((q << 8) | (low / divisor));
        remainder = low % divisor;
    }
    return remainder;
}

__simt_callee__ inline uint32_t SpsvDyadicBit(const uint16_t* words, int32_t count, int32_t bit)
{
    return bit < 0 || bit / 16 >= count ? 0U : (words[bit / 16] >> (bit % 16)) & 1U;
}

__simt_callee__ inline float SpsvDyadicRound(const uint16_t* words, int32_t count, int32_t grid, int32_t sign)
{
    int32_t bits = SpsvDyadicBits(words, count);
    int32_t exponent = bits == 0 ? -150 : bits - 1 + grid;
    uint32_t result = sign < 0 ? 0x80000000U : 0U;
    if (exponent > 127) {
        result |= 0x7f800000U;
    } else if (bits != 0 && exponent >= -150) {
        int32_t shift = exponent >= -126 ? bits - 24 : -149 - grid;
        uint32_t q = 0;
        for (int32_t b = 0; b < 24; ++b) {
            q |= SpsvDyadicBit(words, count, shift + b) << b;
        }
        uint32_t half = SpsvDyadicBit(words, count, shift - 1);
        bool tail = false;
        int32_t tailEnd = shift - 1 < count * 16 ? shift - 1 : count * 16;
        for (int32_t b = 0; b < tailEnd && !tail; ++b) {
            tail |= SpsvDyadicBit(words, count, b) != 0;
        }
        q += half != 0 && (tail || (q & 1U) != 0);
        if (exponent < -126) {
            result |= q;
        } else {
            if (q >= 0x1000000U) {
                q >>= 1;
                ++exponent;
            }
            result |= exponent > 127 ? 0x7f800000U : (static_cast<uint32_t>(exponent + 127) << 23) | (q & 0x7fffffU);
        }
    }
    float answer;
    __builtin_memcpy(&answer, &result, sizeof(answer));
    return answer;
}

// A finite rational may round to infinity. Certify its distance above
// 2^128 (a conservative bound above the binary32 overflow midpoint).
// Integer subtraction keeps this test independent of hardware FTZ/range.
__simt_callee__ inline bool SpsvDyadicOverflowCertified(SpsvDyadicBound error, const uint16_t* words, int32_t count,
                                                        int32_t grid)
{
    if (error.mantissa == 0.0f) {
        return true;
    }
    if (!SpsvIsFiniteF(error.mantissa)) {
        return false;
    }
    int32_t bits = SpsvDyadicBits(words, count);
    int64_t exponent = static_cast<int64_t>(grid) + bits - 1;
    if (exponent > 128) {
        return error.exponent < exponent - 1;
    }
    if (exponent < 128 || grid > 128) {
        return false;
    }
    uint16_t difference[kSpsvDyadicLimbs + 2] = {};
    for (int32_t i = 0; i < count; ++i) {
        difference[i] = words[i];
    }
    int32_t bit = 128 - grid;
    difference[bit / 16] -= static_cast<uint16_t>(1U << (bit % 16));
    bits = SpsvDyadicBits(difference, count);
    if (bits == 0) {
        return false;
    }
    uint32_t top = 0;
    for (int32_t i = 0; i < 24; ++i) {
        top |= SpsvDyadicBit(difference, count, bits - 24 + i) << i;
    }
    int64_t marginExponent = static_cast<int64_t>(grid) + bits - 1;
    return marginExponent > error.exponent ||
           (marginExponent == error.exponent && static_cast<float>(top) * 0x1p-23f > error.mantissa);
}

__simt_callee__ inline bool SpsvDyadicCertified(float answer, SpsvDyadicBound error, const uint16_t* words,
                                                int32_t count, int32_t grid)
{
    if (!SpsvIsFiniteF(answer)) {
        return SpsvDyadicOverflowCertified(error, words, count, grid);
    }
    if (error.mantissa == 0.0f) {
        return true;
    }
    // Certify the rounding cell, rather than stopping a severely amplified
    // cancellation merely because its error fits the task tolerance. Form
    // the center-to-answer distance with integers, so subnormal distances
    // and a tiny radius remain visible on devices with floating-point FTZ.
    int32_t exponent;
    bool negative;
    uint32_t mantissa = SpsvRationalParts(answer, exponent, negative);
    uint16_t rounded[2] = {static_cast<uint16_t>(mantissa), static_cast<uint16_t>(mantissa >> 16)};
    uint16_t aligned[kSpsvDyadicLimbs + 2];
    uint16_t difference[kSpsvDyadicLimbs + 2] = {};
    if (SpsvDyadicAlign(aligned, rounded, 2, exponent - grid, count + 2)) {
        return false;
    }
    for (int32_t i = 0; i < count; ++i) {
        difference[i] = words[i];
    }
    int32_t sign = 1;
    SpsvDyadicAccumulate(difference, sign, aligned, -1, count + 2);
    int32_t bits = SpsvDyadicBits(difference, count + 2);
    if (bits != 0) {
        uint32_t top = 0;
        for (int32_t i = 0; i < 24; ++i) {
            top |= SpsvDyadicBit(difference, count + 2, bits - 24 + i) << i;
        }
        error = SpsvDyadicBoundAdd(error,
                                   SpsvDyadicNormalize(SpsvFloatBoundUp(static_cast<float>(top)), grid + bits - 24));
    }
    uint32_t answerBits;
    __builtin_memcpy(&answerBits, &answer, sizeof(answerBits));
    answerBits &= 0x7fffffffU;
    int32_t halfGap = -150;
    if (answerBits >= 0x00800000U) {
        halfGap = static_cast<int32_t>(answerBits >> 23) - 127 - 24;
        if (answerBits > 0x00800000U && (answerBits & 0x7fffffU) == 0) {
            --halfGap;
        }
    }
    return SpsvIsFiniteF(error.mantissa) &&
           (error.exponent < halfGap || (error.exponent == halfGap && error.mantissa < 1.0f));
}

__simt_callee__ inline bool SpsvDyadicCoefficientNonzero(__gm__ const float* values, int64_t index)
{
    return SpsvRationalNonzero(values[index]);
}

__simt_callee__ inline bool SpsvDyadicCoefficientNonzero(__gm__ const SpsvComplex64* values, int64_t index)
{
    return SpsvRationalNonzero(values[index].real) || SpsvRationalNonzero(values[index].imag);
}

template <bool COMPLEX, typename RowPtr, typename ColInd, typename Value>
__simt_callee__ __attribute__((always_inline)) inline bool SpsvMarkDyadicPredecessors(
    const RowPtr& rowPtr, const ColInd& colInd, __gm__ const Value* values, int64_t m,
    bool forward, __gm__ int32_t* grid, bool anyTarget)
{
    if (!anyTarget) {
        return false;
    }
    for (int64_t step = 0; step < m; ++step) {
        int64_t row = SpsvDyadicActiveRow(step, m, forward, grid, kSpsvDyadicInvalid);
        if (row < 0) {
            continue;
        }
        for (int64_t p = rowPtr[row]; p < rowPtr[row + 1]; ++p) {
            int64_t column = colInd[p];
            if constexpr (!COMPLEX) {
                column = static_cast<int32_t>(column);
            }
            if ((forward ? column < row : column > row) && SpsvDyadicCoefficientNonzero(values, p)) {
                grid[column] = 0;
            }
        }
    }
    return true;
}

__simt_callee__ inline bool SpsvDyadicMarkRetryClosure(const SpsvRationalIndexArray& rowPtr,
                                                       const SpsvRationalIndexArray& colInd, __gm__ const float* values,
                                                       __gm__ float* vecY, __gm__ float* state, int64_t m, bool forward,
                                                       __gm__ int32_t* grid)
{
    bool anyTarget = false;
    for (int64_t r = 0; r < m; ++r) {
        grid[r] = kSpsvDyadicInvalid;
        if (SpsvFloatNeedsDependencyRetry(vecY[r], state[2 * r], state[2 * r + 1])) {
            grid[r] = 0;
            anyTarget = true;
        }
    }
    return SpsvMarkDyadicPredecessors<false>(rowPtr, colInd, values, m, forward, grid, anyTarget);
}

__simt_callee__ inline int32_t SpsvDyadicDependencyBits(__gm__ const uint16_t* digits, int64_t row, int32_t precision)
{
    int32_t bits = 0;
    for (int32_t j = precision - 1; j >= 0 && bits == 0; --j) {
        uint16_t word = digits[SpsvDyadicDigitOffset(row, j)];
        if (word != 0) {
            bits = j * 16 + SpsvUIntBits(word);
        }
    }
    return bits;
}

__simt_callee__ inline bool SpsvDyadicScanDependencies(const SpsvRationalIndexArray& rowPtr,
                                                       const SpsvRationalIndexArray& colInd, __gm__ const float* values,
                                                       __gm__ const int32_t* grid, __gm__ const uint16_t* digits,
                                                       int64_t row, bool forward, int32_t precision, int64_t& top,
                                                       uint64_t& terms)
{
    for (int64_t p = rowPtr[row]; p < rowPtr[row + 1]; ++p) {
        int32_t c = static_cast<int32_t>(colInd[p]);
        if (!(forward ? c < row : c > row) || !SpsvRationalNonzero(values[p])) {
            continue;
        }
        if (!SpsvIsFiniteF(values[p]) || grid[c] == kSpsvDyadicInvalid) {
            return false;
        }
        int32_t exponent;
        bool negative;
        uint32_t mantissa = SpsvRationalParts(values[p], exponent, negative);
        int32_t bits = SpsvDyadicDependencyBits(digits, c, precision);
        if (bits != 0) {
            int64_t termTop = static_cast<int64_t>(grid[c]) + bits + exponent + SpsvUIntBits(mantissa);
            top = termTop > top ? termTop : top;
        }
        ++terms;
    }
    return true;
}

__simt_callee__ inline bool SpsvDyadicExponentBounds(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    __gm__ const float* rhs, const SpsvRationalIndexArray& diagPtr, float alpha, int64_t row, bool forward,
    bool nonUnit, int32_t precision, __gm__ const int32_t* grid, __gm__ const uint16_t* digits, int32_t& ed, bool& nd,
    uint32_t& ma, uint32_t& mx, int32_t& ea, int32_t& ex, bool& na, bool& nx, uint32_t& md, int64_t& rowGrid)
{
    float diagonal = nonUnit ? (diagPtr[row] >= 0 ? values[diagPtr[row]] : 0.0f) : 1.0f;
    md = SpsvRationalParts(diagonal, ed, nd);
    ma = SpsvRationalParts(alpha, ea, na);
    mx = SpsvRationalParts(rhs[row], ex, nx);
    bool valid = SpsvIsFiniteF(alpha) && SpsvIsFiniteF(rhs[row]) && SpsvIsFiniteF(diagonal) && md != 0;
    int64_t top = ma != 0 && mx != 0 ? static_cast<int64_t>(ea) + ex + SpsvUIntBits(ma) + SpsvUIntBits(mx) : -4096;
    uint64_t terms = 2;
    if (valid) {
        valid = SpsvDyadicScanDependencies(rowPtr, colInd, values, grid, digits, row, forward, precision, top, terms);
    }
    int32_t termBits = 0;
    for (uint64_t n = terms; n != 0; n >>= 1) {
        ++termBits;
    }
    rowGrid = top + termBits - precision * 16;
    return valid && rowGrid > -1000000000 && rowGrid < 1000000000;
}

__simt_callee__ inline SpsvDyadicBound SpsvDyadicBuildCenter(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    __gm__ const int32_t* grid, __gm__ const int32_t* signs, __gm__ const int32_t* boundExponent,
    __gm__ const float* boundMantissa, __gm__ const uint16_t* digits, int64_t row, bool forward, int32_t precision,
    int32_t rowGrid, uint32_t ma, uint32_t mx, int32_t ea, int32_t ex, bool na, bool nx, uint16_t* sum,
    uint16_t* product, uint16_t* aligned, int32_t& sign)
{
    SpsvDyadicBound error{0.0f, 0};
    uint64_t right = static_cast<uint64_t>(ma) * mx;
    for (int32_t j = 0; j < 3; ++j) {
        product[j] = static_cast<uint16_t>(right >> (16 * j));
    }
    if (SpsvDyadicAlign(aligned, product, 3, ea + ex - static_cast<int32_t>(rowGrid), precision)) {
        error = SpsvDyadicBoundAdd(error, {1.0f, static_cast<int32_t>(rowGrid)});
    }
    SpsvDyadicAccumulate(sum, sign, aligned, na != nx ? -1 : 1, precision);
    for (int64_t p = rowPtr[row]; p < rowPtr[row + 1]; ++p) {
        int32_t c = static_cast<int32_t>(colInd[p]);
        if (!(forward ? c < row : c > row) || !SpsvRationalNonzero(values[p])) {
            continue;
        }
        int32_t exponent;
        bool negative;
        uint32_t mantissa = SpsvRationalParts(values[p], exponent, negative);
        uint64_t carry = 0;
        for (int32_t j = 0; j < precision; ++j) {
            uint64_t value = static_cast<uint64_t>(digits[SpsvDyadicDigitOffset(static_cast<int64_t>(c), j)]) *
                                 mantissa +
                             carry;
            product[j] = static_cast<uint16_t>(value);
            carry = value >> 16;
        }
        product[precision] = static_cast<uint16_t>(carry);
        product[precision + 1] = static_cast<uint16_t>(carry >> 16);
        if (SpsvDyadicAlign(aligned, product, precision + 2, grid[c] + exponent - static_cast<int32_t>(rowGrid),
                            precision)) {
            error = SpsvDyadicBoundAdd(error, {1.0f, static_cast<int32_t>(rowGrid)});
        }
        int32_t termSign = negative ? signs[c] : -signs[c];
        if (termSign != 0) {
            SpsvDyadicAccumulate(sum, sign, aligned, termSign, precision);
        }
        error = SpsvDyadicBoundAdd(error,
                                   SpsvDyadicBoundMul({boundMantissa[c], boundExponent[c]}, mantissa, exponent, false));
    }
    return error;
}

__simt_callee__ inline void SpsvDyadicFinishCenter(__gm__ float* vecY, __gm__ float* state, __gm__ int32_t* grid,
                                                   __gm__ int32_t* signs, __gm__ int32_t* boundExponent,
                                                   __gm__ float* boundMantissa, __gm__ uint16_t* digits, int64_t row,
                                                   int32_t precision, int32_t ed, bool nd, uint32_t md, int64_t rowGrid,
                                                   uint16_t* sum, int32_t sign, SpsvDyadicBound error)
{
    grid[row] = static_cast<int32_t>(rowGrid) - ed;
    error = SpsvDyadicBoundMul(error, md, ed, true);
    if (SpsvDyadicDivide(sum, precision, md) != 0) {
        error = SpsvDyadicBoundAdd(error, {1.0f, grid[row]});
    }
    signs[row] = nd ? -sign : sign;
    boundMantissa[row] = error.mantissa;
    boundExponent[row] = error.exponent;
    for (int32_t j = 0; j < kSpsvDyadicLimbs; ++j) {
        digits[row * kSpsvDyadicLimbs + j] = j < precision ? sum[j] : 0;
    }
    if (SpsvFloatNeedsDependencyRetry(vecY[row], state[2 * row], state[2 * row + 1])) {
        float answer = SpsvDyadicRound(sum, precision, grid[row], signs[row]);
        if (SpsvDyadicCertified(answer, error, sum, precision, grid[row])) {
            vecY[row] = answer;
            state[2 * row] = 0.0f;
            state[2 * row + 1] = SpsvIsFiniteF(answer) ? 0.0f : kSpsvCertifiedOverflowRadius;
        }
    }
}

__simt_callee__ inline void SpsvDyadicProcessRows(const SpsvRationalIndexArray& rowPtr,
                                                  const SpsvRationalIndexArray& colInd, __gm__ const float* values,
                                                  __gm__ const float* rhs, const SpsvRationalIndexArray& diagPtr,
                                                  float alpha, int64_t m, bool forward, bool nonUnit, int32_t precision,
                                                  __gm__ float* vecY, __gm__ float* state, __gm__ int32_t* grid,
                                                  __gm__ int32_t* signs, __gm__ int32_t* boundExponent,
                                                  __gm__ float* boundMantissa, __gm__ uint16_t* digits)
{
    for (int64_t step = 0; step < m; ++step) {
        int64_t row = forward ? step : m - 1 - step;
        if (grid[row] == kSpsvDyadicInvalid) {
            continue;
        }
        int32_t ed, ea, ex;
        bool nd, na, nx;
        uint32_t ma, mx, md;
        int64_t rowGrid;
        if (!SpsvDyadicExponentBounds(rowPtr, colInd, values, rhs, diagPtr, alpha, row, forward, nonUnit, precision,
                                      grid, digits, ed, nd, ma, mx, ea, ex, na, nx, md, rowGrid)) {
            grid[row] = kSpsvDyadicInvalid;
            continue;
        }
        uint16_t sum[kSpsvDyadicLimbs] = {};
        uint16_t product[kSpsvDyadicLimbs + 2] = {};
        uint16_t aligned[kSpsvDyadicLimbs];
        int32_t sign = 0;
        SpsvDyadicBound error = SpsvDyadicBuildCenter(rowPtr, colInd, values, grid, signs, boundExponent, boundMantissa,
                                                      digits, row, forward, precision, static_cast<int32_t>(rowGrid),
                                                      ma, mx, ea, ex, na, nx, sum, product, aligned, sign);
        SpsvDyadicFinishCenter(vecY, state, grid, signs, boundExponent, boundMantissa, digits, row, precision, ed, nd,
                               md, rowGrid, sum, sign, error);
    }
}

__simt_callee__ inline bool SpsvDyadicAnyRetry(__gm__ const float* vecY, __gm__ const float* state, int64_t m)
{
    for (int64_t row = 0; row < m; ++row) {
        if (SpsvFloatNeedsDependencyRetry(vecY[row], state[2 * row], state[2 * row + 1])) {
            return true;
        }
    }
    return false;
}

// Scratch overlaps the otherwise idle CRT digits. 32 uint16 digits and four
// metadata words per row require 80*m bytes, below the reserved integer area.
__simt_callee__ inline void SpsvDyadicRetry(const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd,
                                            __gm__ const float* values, __gm__ const float* rhs,
                                            const SpsvRationalIndexArray& diagPtr, float alpha, int64_t m, bool forward,
                                            bool nonUnit, __gm__ float* vecY, __gm__ float* state,
                                            __gm__ uint16_t* scratch)
{
    __gm__ int32_t* grid = (__gm__ int32_t*)scratch;
    __gm__ int32_t* signs = grid + m;
    __gm__ int32_t* boundExponent = signs + m;
    __gm__ float* boundMantissa = (__gm__ float*)(boundExponent + m);
    __gm__ uint16_t* digits = (__gm__ uint16_t*)(boundMantissa + m);
    if (!SpsvDyadicMarkRetryClosure(rowPtr, colInd, values, vecY, state, m, forward, grid)) {
        return;
    }
    for (int32_t precision = 8; precision <= kSpsvDyadicLimbs; precision *= 2) {
        SpsvDyadicProcessRows(rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, precision, vecY, state,
                              grid, signs, boundExponent, boundMantissa, digits);
        if (!SpsvDyadicAnyRetry(vecY, state, m)) {
            return;
        }
    }
}
#endif // SPSV_DYADIC_H_
