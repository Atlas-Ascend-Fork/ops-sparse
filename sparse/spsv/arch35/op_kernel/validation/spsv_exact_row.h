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

#ifndef OP_KERNEL_VALIDATION_SPSV_EXACT_ROW_H_
#define OP_KERNEL_VALIDATION_SPSV_EXACT_ROW_H_

// A component may cancel across arbitrarily many exponent tiers. The retry
// rebuilds (alpha*x - A*y)*conj(diag) as exact integer products; components
// retain separate exponents until the final division by |diag|^2.
template <typename RowPtrT, typename ColIndT>
__simt_callee__ inline void SpsvComplexExactComponent(__gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd,
                                                      __gm__ const SpsvComplex64* values,
                                                      __gm__ const volatile SpsvComplex64* vecX,
                                                      __gm__ const float* yWide, SpsvComplex64 alpha, int32_t row,
                                                      bool forward, bool nonUnit, SpsvComplex64 diag, bool imaginary,
                                                      SpsvDFExp& answer)
{
    SpsvExactSum sum;
    SpsvExactClear(sum);
    // Real output: nr*dr + ni*di. Imaginary: ni*dr - nr*di.
    float rr = imaginary ? -diag.imag : diag.real;
    float ri = imaginary ? diag.real : diag.imag;
    SpsvExactProduct(sum, alpha.real, vecX[row].real, rr, false);
    SpsvExactProduct(sum, alpha.imag, vecX[row].imag, rr, true);
    SpsvExactProduct(sum, alpha.real, vecX[row].imag, ri, false);
    SpsvExactProduct(sum, alpha.imag, vecX[row].real, ri, false);
    for (SpsvCsrCursor<RowPtrT> p = static_cast<SpsvCsrCursor<RowPtrT>>(rowPtr[row]); p < rowPtr[row + 1]; ++p) {
        int32_t col = static_cast<int32_t>(colInd[p]);
        if (forward ? col < row : col > row) {
            for (int32_t word = 0; word < 2; ++word) {
                float yr = yWide[4 * static_cast<int64_t>(col) + word];
                float yi = yWide[4 * static_cast<int64_t>(col) + 2 + word];
                SpsvExactProduct(sum, values[p].real, yr, rr, true);
                SpsvExactProduct(sum, values[p].imag, yi, rr, false);
                SpsvExactProduct(sum, values[p].real, yi, ri, true);
                SpsvExactProduct(sum, values[p].imag, yr, ri, true);
            }
        }
    }
    answer = SpsvExactFinish(sum);
}

// Keep the cold retry's divisions and live temporaries out of the row loop.
template <typename RowPtrT, typename ColIndT>
__simt_callee__ inline SpsvDFComplex SpsvComplexExactRow(__gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd,
                                                         __gm__ const SpsvComplex64* values,
                                                         __gm__ const volatile SpsvComplex64* vecX,
                                                         __gm__ const float* yWide, SpsvComplex64 alpha, int32_t row,
                                                         bool forward, bool nonUnit, SpsvComplex64 diag)
{
    SpsvDFExp denominator = SpsvDFExpAdd(SpsvDFExpProd(diag.real, diag.real), SpsvDFExpProd(diag.imag, diag.imag));
    SpsvDFExp numerator;
    SpsvComplexExactComponent<RowPtrT, ColIndT>(rowPtr, colInd, values, vecX, yWide, alpha, row, forward, nonUnit, diag,
                                                false, numerator);
    SpsvDF re = SpsvExactQuotient(numerator, denominator);
    SpsvComplexExactComponent<RowPtrT, ColIndT>(rowPtr, colInd, values, vecX, yWide, alpha, row, forward, nonUnit, diag,
                                                true, numerator);
    return {re, SpsvExactQuotient(numerator, denominator)};
}

__simt_callee__ __attribute__((always_inline)) inline SpsvDF SpsvDFDivF(SpsvDF value, float divisor)
{
    if (divisor == 0.0f) {
        uint32_t bits = 0x7fc00000U;
        float nanValue;
        __builtin_memcpy(&nanValue, &bits, sizeof(bits));
        return {nanValue, 0.0f, value.lost | kSpsvRangeLoss};
    }
    // Outside the bounded fast domain the reciprocal can flush to zero (for
    // example 1 / 2^127), even when value / divisor is a normal finite result.
    // Preserve direct division there so later rows receive the small solution.
    if (!SpsvFastComponent(divisor)) {
        float q = value.hi / divisor;
        float residual = fmaf(-q, divisor, value.hi) + value.lo;
        SpsvDF result = SpsvTwoSum(q, residual / divisor);
        result.lost = value.lost;
        return result;
    }
    float inverse = 1.0f / divisor;
    float q = value.hi * inverse;
    float residual = fmaf(-q, divisor, value.hi) + value.lo;
    SpsvDF result = SpsvTwoSum(q, residual * inverse);
    // Refine once more after normalizing the high word. The two residuals
    // keep a float-float quotient while sharing one reciprocal division.
    residual = fmaf(-result.hi, divisor, value.hi) + value.lo;
    result = SpsvTwoSum(result.hi, residual * inverse);
    result.lost = value.lost;
    return result;
}

#endif // OP_KERNEL_VALIDATION_SPSV_EXACT_ROW_H_
