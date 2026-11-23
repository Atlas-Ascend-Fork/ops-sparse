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

#ifndef SPSV_COMPLEX_DYADIC_CORE_H_
#define SPSV_COMPLEX_DYADIC_CORE_H_

#include "../common/spsv_simt_common.h"

__simt_callee__ __attribute__((always_inline)) inline void SpsvComplexDyadicRhs(uint16_t* sum, int32_t& sign,
                                                                                SpsvDyadicBound& error, int32_t count,
                                                                                int32_t grid, float a, float b,
                                                                                bool subtract)
{
    int32_t ea, eb;
    bool na, nb;
    uint32_t ma = SpsvRationalParts(a, ea, na), mb = SpsvRationalParts(b, eb, nb);
    uint64_t value = static_cast<uint64_t>(ma) * mb;
    uint16_t product[3], aligned[kSpsvDyadicLimbs];
    for (int32_t j = 0; j < 3; ++j) {
        product[j] = static_cast<uint16_t>(value >> (16 * j));
    }
    if (SpsvDyadicAlign(aligned, product, 3, ea + eb - grid, count)) {
        error = SpsvDyadicBoundAdd(error, {1.0f, grid});
    }
    SpsvDyadicAccumulate(sum, sign, aligned, (na != nb) != subtract ? -1 : 1, count);
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvComplexDyadicTerm(
    uint16_t* sum, int32_t& sign, SpsvDyadicBound& error, int32_t count, int32_t grid, float coefficient, bool subtract,
    __gm__ const uint16_t* source, int32_t sourceGrid, int32_t sourceSign, SpsvDyadicBound sourceError)
{
    int32_t exponent;
    bool negative;
    uint32_t mantissa = SpsvRationalParts(coefficient, exponent, negative);
    uint16_t product[kSpsvDyadicLimbs + 2], aligned[kSpsvDyadicLimbs];
    uint64_t carry = 0;
    for (int32_t j = 0; j < count; ++j) {
        uint64_t value = static_cast<uint64_t>(source[j]) * mantissa + carry;
        product[j] = static_cast<uint16_t>(value);
        carry = value >> 16;
    }
    product[count] = static_cast<uint16_t>(carry);
    product[count + 1] = static_cast<uint16_t>(carry >> 16);
    if (SpsvDyadicAlign(aligned, product, count + 2, sourceGrid + exponent - grid, count)) {
        error = SpsvDyadicBoundAdd(error, {1.0f, grid});
    }
    int32_t termSign = negative != subtract ? -sourceSign : sourceSign;
    if (termSign != 0) {
        SpsvDyadicAccumulate(sum, sign, aligned, termSign, count);
    }
    error = SpsvDyadicBoundAdd(error, SpsvDyadicBoundMul(sourceError, mantissa, exponent, false));
}

// Two binary32 components have at most 277 bits on their common grid.
constexpr int32_t kSpsvComplexDiagonalLimbs = 18;
constexpr int32_t kSpsvComplexQuotientLimbs = kSpsvDyadicLimbs + 2 * kSpsvComplexDiagonalLimbs + 2;

__simt_callee__ inline uint16_t SpsvComplexShiftedWord(const uint16_t* words, int32_t count, int32_t shift,
                                                       int32_t index)
{
    int32_t word = index - shift / 16, bits = shift % 16;
    uint32_t value = word >= 0 && word < count ? static_cast<uint32_t>(words[word]) << bits : 0;
    if (bits && word > 0 && word <= count) {
        value |= words[word - 1] >> (16 - bits);
    }
    return static_cast<uint16_t>(value);
}

__simt_callee__ inline uint16_t SpsvComplexShiftedWordGM(__gm__ const uint16_t* words, int32_t count, int32_t shift,
                                                         int32_t index)
{
    int32_t word = index - shift / 16, bits = shift % 16;
    uint32_t value = word >= 0 && word < count ? static_cast<uint32_t>(words[word]) << bits : 0;
    if (bits && word > 0 && word <= count) {
        value |= words[word - 1] >> (16 - bits);
    }
    return static_cast<uint16_t>(value);
}

__simt_callee__ inline int32_t SpsvComplexDyadicBits(__gm__ const uint16_t* words, int32_t count)
{
    while (count > 0 && words[count - 1] == 0) {
        --count;
    }
    return count == 0 ? 0 : (count - 1) * 16 + SpsvUIntBits(words[count - 1]);
}

// Add an exactly shifted short product without materializing a second
// quotient-sized array on the default 1152-byte SIMT thread stack.
__simt_callee__ inline void SpsvComplexShiftedAccumulate(__gm__ uint16_t* sum, int32_t& sign, const uint16_t* term,
                                                         int32_t termCount, int32_t shift, int32_t termSign,
                                                         int32_t count)
{
    if (termSign == 0 || SpsvDyadicBits(term, termCount) == 0) {
        return;
    }
    if (sign == 0 || sign == termSign) {
        uint32_t carry = 0;
        for (int32_t j = 0; j < count; ++j) {
            uint32_t value = static_cast<uint32_t>(sum[j]) + SpsvComplexShiftedWord(term, termCount, shift, j) + carry;
            sum[j] = static_cast<uint16_t>(value);
            carry = value >> 16;
        }
        sign = termSign;
        return;
    }
    int32_t order = 0;
    for (int32_t j = count - 1; j >= 0 && order == 0; --j) {
        uint16_t value = SpsvComplexShiftedWord(term, termCount, shift, j);
        order = sum[j] > value ? 1 : sum[j] < value ? -1 : 0;
    }
    uint32_t borrow = 0;
    for (int32_t j = 0; j < count; ++j) {
        uint32_t value = SpsvComplexShiftedWord(term, termCount, shift, j);
        uint32_t a = order >= 0 ? sum[j] : value;
        uint32_t b = (order >= 0 ? value : sum[j]) + borrow;
        sum[j] = static_cast<uint16_t>(a - b);
        borrow = a < b;
    }
    if (order <= 0) {
        sign = order == 0 ? 0 : termSign;
    }
}

__simt_callee__ __attribute__((always_inline)) inline bool SpsvComplexDividePowerOfTwo(__gm__ uint16_t* words,
                                                                                       int32_t count, int32_t wordCount,
                                                                                       int32_t divisorWords,
                                                                                       int32_t topBits)
{
    int32_t shift = (divisorWords - 1) * 16 + topBits - 1;
    int32_t whole = shift / 16, part = shift % 16;
    bool remainder = false;
    for (int32_t j = 0; j < whole; ++j) {
        remainder |= words[j] != 0;
    }
    if (part != 0) {
        remainder |= (words[whole] & ((1U << part) - 1U)) != 0;
    }
    for (int32_t j = 0; j < count; ++j) {
        int32_t source = j + whole;
        uint32_t value = source < wordCount ? words[source] >> part : 0;
        if (part != 0 && source + 1 < wordCount) {
            value |= static_cast<uint32_t>(words[source + 1]) << (16 - part);
        }
        words[j] = static_cast<uint16_t>(value);
    }
    return remainder;
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvComplexDivideDigits(
    __gm__ uint16_t* words, uint16_t* numerator, const uint16_t* denominator, int32_t wordCount, int32_t divisorWords,
    uint32_t leading)
{
    for (int32_t j = wordCount - divisorWords; j >= 0; --j) {
        uint32_t head = (static_cast<uint32_t>(numerator[j + divisorWords]) << 16) | numerator[j + divisorWords - 1];
        uint32_t digit = head / leading, residual = head % leading;
        if (digit >= 0x10000U) {
            digit = 0xffffU;
            residual = head - digit * leading;
        }
        if (divisorWords > 1) {
            while (residual < 0x10000U &&
                   digit * denominator[divisorWords - 2] > (residual << 16) + numerator[j + divisorWords - 2]) {
                --digit;
                residual += leading;
            }
        }
        uint32_t borrow = 0;
        uint32_t carry = 0;
        for (int32_t k = 0; k < divisorWords; ++k) {
            uint32_t product = digit * denominator[k] + carry;
            carry = product >> 16;
            uint32_t subtrahend = (product & 0xffffU) + borrow;
            uint32_t value = numerator[j + k];
            numerator[j + k] = static_cast<uint16_t>(value - subtrahend);
            borrow = value < subtrahend;
        }
        uint32_t high = numerator[j + divisorWords];
        bool negative = high < carry + borrow;
        numerator[j + divisorWords] = static_cast<uint16_t>(high - carry - borrow);
        if (negative) {
            --digit;
            carry = 0;
            for (int32_t k = 0; k < divisorWords; ++k) {
                uint32_t value = static_cast<uint32_t>(numerator[j + k]) + denominator[k] + carry;
                numerator[j + k] = static_cast<uint16_t>(value);
                carry = value >> 16;
            }
            numerator[j + divisorWords] = static_cast<uint16_t>(numerator[j + divisorWords] + carry);
        }
        words[j] = static_cast<uint16_t>(digit);
    }
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvComplexNormalizeDivision(
    __gm__ const uint16_t* words, __gm__ const uint16_t* divisor, uint16_t* numerator, uint16_t* denominator,
    int32_t wordCount, int32_t divisorWords, int32_t topBits)
{
    int32_t normalize = 16 - topBits;
    uint32_t carry = 0;
    for (int32_t j = 0; j < wordCount; ++j) {
        uint32_t value = (static_cast<uint32_t>(words[j]) << normalize) | carry;
        numerator[j] = static_cast<uint16_t>(value);
        carry = value >> 16;
    }
    numerator[wordCount] = static_cast<uint16_t>(carry);
    carry = 0;
    for (int32_t j = 0; j < divisorWords; ++j) {
        uint32_t value = (static_cast<uint32_t>(divisor[j]) << normalize) | carry;
        denominator[j] = static_cast<uint16_t>(value);
        carry = value >> 16;
    }
}

// Exact base-2^16 division. Keep the short normalized operands in private
// memory; scanning every quotient bit against a full GM array makes even
// division by the unit diagonal expensive. No floating estimate is used.
__simt_callee__ __attribute__((always_inline)) inline bool SpsvComplexIntegerDivide(__gm__ uint16_t* words,
                                                                                    int32_t count,
                                                                                    __gm__ const uint16_t* divisor,
                                                                                    int32_t divisorCount)
{
    int32_t divisorWords = divisorCount;
    while (divisorWords > 0 && divisor[divisorWords - 1] == 0) {
        --divisorWords;
    }
    int32_t wordCount = count;
    while (wordCount > 0 && words[wordCount - 1] == 0) {
        --wordCount;
    }
    // A finite nonzero diagonal always has a positive squared norm.
    if (divisorWords == 0) {
        return true;
    }
    if (wordCount < divisorWords) {
        for (int32_t j = 0; j < count; ++j) {
            words[j] = 0;
        }
        return wordCount != 0;
    }
    int32_t topBits = SpsvUIntBits(divisor[divisorWords - 1]);
    uint16_t top = divisor[divisorWords - 1];
    bool powerOfTwo = (top & (top - 1)) == 0;
    for (int32_t j = 0; j + 1 < divisorWords; ++j) {
        powerOfTwo &= divisor[j] == 0;
    }
    if (powerOfTwo) {
        return SpsvComplexDividePowerOfTwo(words, count, wordCount, divisorWords, topBits);
    }
    uint16_t numerator[kSpsvComplexQuotientLimbs + 1] = {};
    uint16_t denominator[2 * kSpsvComplexDiagonalLimbs + 1] = {};
    SpsvComplexNormalizeDivision(words, divisor, numerator, denominator, wordCount, divisorWords, topBits);
    uint32_t leading = denominator[divisorWords - 1];
    // The normalized copy owns the dividend from here; write quotient digits
    // directly to GM instead of retaining a third private array.
    for (int32_t j = 0; j < count; ++j) {
        words[j] = 0;
    }
    SpsvComplexDivideDigits(words, numerator, denominator, wordCount, divisorWords, leading);
    bool remainder = false;
    for (int32_t j = 0; j < divisorWords; ++j) {
        remainder |= numerator[j] != 0;
    }
    return remainder;
}

__simt_callee__ inline void SpsvComplexDyadicScale(__gm__ uint16_t* sum, int32_t& sign, const uint16_t* source,
                                                   int32_t sourceSign, float factor, int32_t diagonalGrid,
                                                   bool subtract, int32_t count)
{
    int32_t exponent;
    bool negative;
    uint32_t mantissa = SpsvRationalParts(factor, exponent, negative);
    uint16_t product[kSpsvDyadicLimbs + 2];
    uint64_t carry = 0;
    for (int32_t j = 0; j < count; ++j) {
        uint64_t value = static_cast<uint64_t>(source[j]) * mantissa + carry;
        product[j] = static_cast<uint16_t>(value);
        carry = value >> 16;
    }
    product[count] = static_cast<uint16_t>(carry);
    product[count + 1] = static_cast<uint16_t>(carry >> 16);
    if (mantissa != 0) {
        SpsvComplexShiftedAccumulate(sum, sign, product, count + 2, exponent - diagonalGrid,
                                     negative != subtract ? -sourceSign : sourceSign,
                                     count + 2 * kSpsvComplexDiagonalLimbs + 2);
    }
}

__simt_callee__ inline void SpsvComplexDiagonalSquare(__gm__ uint16_t* norm, float component, int32_t diagonalGrid,
                                                      int32_t count)
{
    int32_t exponent;
    bool negative;
    uint32_t mantissa = SpsvRationalParts(component, exponent, negative);
    if (mantissa == 0) {
        return;
    }
    uint64_t square = static_cast<uint64_t>(mantissa) * mantissa;
    uint16_t product[3];
    for (int32_t j = 0; j < 3; ++j) {
        product[j] = static_cast<uint16_t>(square >> (16 * j));
    }
    int32_t sign = 1;
    SpsvComplexShiftedAccumulate(norm, sign, product, 3, 2 * (exponent - diagonalGrid), 1, count);
}

__simt_callee__ inline bool SpsvComplexDyadicDiagonal(SpsvComplex64 diagonal, int32_t& exponent, int32_t& bits)
{
    int32_t er, ei;
    bool nr, ni;
    uint32_t mr = SpsvRationalParts(diagonal.real, er, nr);
    uint32_t mi = SpsvRationalParts(diagonal.imag, ei, ni);
    if ((!mr && !mi) || !SpsvIsFiniteF(diagonal.real) || !SpsvIsFiniteF(diagonal.imag)) {
        return false;
    }
    exponent = mr == 0 ? ei : mi == 0 ? er : (er < ei ? er : ei);
    int32_t br = mr ? SpsvUIntBits(mr) + er - exponent : 0;
    int32_t bi = mi ? SpsvUIntBits(mi) + ei - exponent : 0;
    bits = br > bi ? br : bi;
    return true;
}

__simt_callee__ inline bool SpsvComplexDyadicMarkRetryClosure(__gm__ const int32_t* rowPtr,
                                                              __gm__ const int32_t* colInd,
                                                              __gm__ const SpsvComplex64* values,
                                                              __gm__ SpsvComplex64* vecY, __gm__ float* yWide,
                                                              __gm__ float* rowPrep, int64_t m, bool forward,
                                                              __gm__ int32_t* grid)
{
    bool anyTarget = false;
    for (int64_t r = 0; r < m; ++r) {
        bool retry = SpsvFloatNeedsDependencyRetry(vecY[r].real, yWide[4 * r + 1], rowPrep[18 * r + 9]) ||
                     SpsvFloatNeedsDependencyRetry(vecY[r].imag, yWide[4 * r + 3], rowPrep[18 * r + 10]);
        grid[r] = retry ? 0 : kSpsvDyadicInvalid;
        anyTarget |= retry;
    }
    return SpsvMarkDyadicPredecessors<true>(rowPtr, colInd, values, m, forward, grid, anyTarget);
}

__simt_callee__ inline int64_t SpsvComplexDyadicRhsTop(SpsvComplex64 alpha, float rhsReal, float rhsImag, int64_t top)
{
    for (int32_t a = 0; a < 2; ++a)
        for (int32_t x = 0; x < 2; ++x) {
            int32_t ea, ex;
            bool na, nx;
            uint32_t ma = SpsvRationalParts(a ? alpha.imag : alpha.real, ea, na);
            uint32_t mx = SpsvRationalParts(x ? rhsImag : rhsReal, ex, nx);
            if (ma && mx) {
                int64_t t = static_cast<int64_t>(ea) + ex + SpsvUIntBits(ma) + SpsvUIntBits(mx);
                top = t > top ? t : top;
            }
        }
    return top;
}

__simt_callee__ inline int32_t SpsvComplexDyadicSourceBits(__gm__ const uint16_t* digits, int64_t row,
                                                           int32_t precision)
{
    int32_t bits = 0;
    for (int32_t component = 0; component < 2; ++component) {
        for (int32_t j = precision - 1; j >= 0; --j) {
            uint16_t word = digits[(2 * row + component) * kSpsvDyadicLimbs + j];
            if (word != 0) {
                int32_t current = j * 16 + SpsvUIntBits(word);
                bits = current > bits ? current : bits;
                break;
            }
        }
    }
    return bits;
}

__simt_callee__ inline int64_t SpsvComplexDyadicDependencyTop(float real, float imag, int64_t sourceGrid,
                                                              int32_t sourceBits, int64_t top)
{
    for (int32_t component = 0; component < 2; ++component) {
        int32_t exponent;
        bool negative;
        uint32_t mantissa = SpsvRationalParts(component ? imag : real, exponent, negative);
        if (sourceBits && mantissa) {
            int64_t current = sourceGrid + sourceBits + exponent + SpsvUIntBits(mantissa);
            top = current > top ? current : top;
        }
    }
    return top;
}

__simt_callee__ inline bool SpsvComplexDyadicScanDependencies(__gm__ const int32_t* rowPtr,
                                                              __gm__ const int32_t* colInd,
                                                              __gm__ const SpsvComplex64* values,
                                                              __gm__ const int32_t* grid, __gm__ const uint16_t* digits,
                                                              int64_t row, bool forward, int32_t precision,
                                                              int64_t& top, uint64_t& terms)
{
    for (int64_t p = rowPtr[row]; p < rowPtr[row + 1]; ++p) {
        int64_t c = colInd[p];
        if (!(forward ? c < row : c > row) ||
            (!SpsvRationalNonzero(values[p].real) && !SpsvRationalNonzero(values[p].imag))) {
            continue;
        }
        if (grid[c] == kSpsvDyadicInvalid || !SpsvIsFiniteF(values[p].real) || !SpsvIsFiniteF(values[p].imag)) {
            return false;
        }
        int32_t bits = SpsvComplexDyadicSourceBits(digits, c, precision);
        top = SpsvComplexDyadicDependencyTop(values[p].real, values[p].imag, grid[c], bits, top);
        terms += 4;
    }
    return true;
}

__simt_callee__ inline bool SpsvComplexDyadicExponentBounds(__gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd,
                                                            __gm__ const SpsvComplex64* values,
                                                            __gm__ const SpsvComplex64* rhs, __gm__ const int32_t* grid,
                                                            __gm__ const uint16_t* digits, SpsvComplex64 alpha,
                                                            int64_t row, bool forward, int32_t precision,
                                                            SpsvComplex64 diagonal, int32_t& ed, int32_t& diagonalBits,
                                                            int64_t& wideGrid)
{
    bool valid = SpsvComplexDyadicDiagonal(diagonal, ed, diagonalBits) && SpsvIsFiniteF(alpha.real) &&
                 SpsvIsFiniteF(alpha.imag) && SpsvIsFiniteF(rhs[row].real) && SpsvIsFiniteF(rhs[row].imag);
    int64_t top = SpsvComplexDyadicRhsTop(alpha, rhs[row].real, rhs[row].imag, -4096);
    uint64_t terms = 8;
    if (valid) {
        valid = SpsvComplexDyadicScanDependencies(rowPtr, colInd, values, grid, digits, row, forward, precision, top,
                                                  terms);
    }
    int32_t termBits = 0;
    for (uint64_t n = terms; n != 0; n >>= 1) {
        ++termBits;
    }
    wideGrid = top + termBits - precision * 16;
    return valid && wideGrid > -1000000000 && wideGrid < 1000000000;
}

__simt_callee__ inline SpsvDyadicBound SpsvComplexDyadicBuildCenter(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ const int32_t* grid, __gm__ const int32_t* signs,
    __gm__ const int32_t* boundExponent, __gm__ const float* boundMantissa, __gm__ const uint16_t* digits,
    SpsvComplex64 alpha, int64_t row, bool forward, int32_t precision, int32_t rg, uint16_t* re, uint16_t* im,
    int32_t& sr, int32_t& si, SpsvDyadicBound& er, SpsvDyadicBound& ei)
{
    SpsvComplexDyadicRhs(re, sr, er, precision, rg, alpha.real, rhs[row].real, false);
    SpsvComplexDyadicRhs(re, sr, er, precision, rg, alpha.imag, rhs[row].imag, true);
    SpsvComplexDyadicRhs(im, si, ei, precision, rg, alpha.real, rhs[row].imag, false);
    SpsvComplexDyadicRhs(im, si, ei, precision, rg, alpha.imag, rhs[row].real, false);
    for (int64_t p = rowPtr[row]; p < rowPtr[row + 1]; ++p) {
        int64_t c = colInd[p];
        if (!(forward ? c < row : c > row) ||
            (!SpsvRationalNonzero(values[p].real) && !SpsvRationalNonzero(values[p].imag))) {
            continue;
        }
        __gm__ const uint16_t* cr = digits + 2 * c * kSpsvDyadicLimbs;
        __gm__ const uint16_t* ci = cr + kSpsvDyadicLimbs;
        SpsvDyadicBound error{boundMantissa[c], boundExponent[c]};
        SpsvComplexDyadicTerm(re, sr, er, precision, rg, values[p].real, true, cr, grid[c], signs[2 * c], error);
        SpsvComplexDyadicTerm(re, sr, er, precision, rg, values[p].imag, false, ci, grid[c], signs[2 * c + 1], error);
        SpsvComplexDyadicTerm(im, si, ei, precision, rg, values[p].real, true, ci, grid[c], signs[2 * c + 1], error);
        SpsvComplexDyadicTerm(im, si, ei, precision, rg, values[p].imag, true, cr, grid[c], signs[2 * c], error);
    }
    return er.mantissa == 0.0f ?
               ei :
           ei.mantissa == 0.0f ?
               er :
               (er.exponent > ei.exponent || (er.exponent == ei.exponent && er.mantissa > ei.mantissa) ? er : ei);
}

#endif // SPSV_COMPLEX_DYADIC_CORE_H_
