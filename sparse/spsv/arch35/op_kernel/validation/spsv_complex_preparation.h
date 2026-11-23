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

#ifndef OP_KERNEL_VALIDATION_SPSV_COMPLEX_PREPARATION_H_
#define OP_KERNEL_VALIDATION_SPSV_COMPLEX_PREPARATION_H_

__simt_callee__ __attribute__((always_inline)) inline void SpsvPrepareComplexProduct(SpsvComplex64 alpha,
                                                                                     SpsvComplex64 x, SpsvDF& a0,
                                                                                     SpsvDF& a1)
{
    if (SpsvFastComponent(alpha.real) && SpsvFastComponent(alpha.imag) && SpsvFastComponent(x.real) &&
        SpsvFastComponent(x.imag)) {
        a0 = SpsvFastAdd(SpsvFastMul({alpha.real, 0.0f}, x.real), SpsvDFNeg(SpsvFastMul({alpha.imag, 0.0f}, x.imag)));
        a1 = SpsvFastAdd(SpsvFastMul({alpha.real, 0.0f}, x.imag), SpsvFastMul({alpha.imag, 0.0f}, x.real));
        a0.lost = a1.lost = kSpsvRoundLoss;
    } else {
        a0 = SpsvDFAdd(SpsvTwoProd(alpha.real, x.real), SpsvDFNeg(SpsvTwoProd(alpha.imag, x.imag)));
        a1 = SpsvDFAdd(SpsvTwoProd(alpha.real, x.imag), SpsvTwoProd(alpha.imag, x.real));
    }
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvPrepareComplexInverseBounds(
    SpsvComplex64 alpha, SpsvComplex64 x, SpsvComplex64 diag, float storedScale, float dScale, bool hasDiagonal,
    __gm__ float* rowPrep, int64_t i)
{
    bool bounded = storedScale > 0.0f && SpsvFastComponent(dScale) && SpsvFastComponent(alpha.real) &&
                   SpsvFastComponent(alpha.imag) && SpsvFastComponent(x.real) && SpsvFastComponent(x.imag);
    rowPrep[18 * i + 9] = SpsvAbsF(alpha.real * x.real) + SpsvAbsF(alpha.imag * x.imag);
    rowPrep[18 * i + 10] = SpsvAbsF(alpha.real * x.imag) + SpsvAbsF(alpha.imag * x.real);
    SpsvDFComplex inverse{{1.0f, 0.0f}, {0.0f, 0.0f}};
    float inverseError = 0.0f;
    if (hasDiagonal) {
        inverseError = kSpsvUncertifiedInverseError;
        if (bounded && SpsvCertifiedComplexQuotient({{1.0f, 0.0f}, {0.0f, 0.0f}}, 0.0f, 0.0f, diag, inverse)) {
            SpsvDFComplex residual;
            float er = 0.0f, ei = 0.0f;
            if (SpsvCertifiedComplexResidual({{1.0f, 0.0f}, {0.0f, 0.0f}}, inverse, diag, er, ei, residual)) {
                inverseError = 16.0f *
                                   (SpsvAbsF(residual.re.hi) + SpsvAbsF(residual.re.lo) + SpsvAbsF(residual.im.hi) +
                                    SpsvAbsF(residual.im.lo) + er + ei) /
                                   dScale +
                               0x1p-90f;
            }
        }
    }
    rowPrep[18 * i + 11] = inverse.re.hi;
    rowPrep[18 * i + 12] = inverse.re.lo;
    rowPrep[18 * i + 13] = inverse.im.hi;
    rowPrep[18 * i + 14] = inverse.im.lo;
    rowPrep[18 * i + 15] = inverseError;
    rowPrep[18 * i + 16] = diag.real;
    rowPrep[18 * i + 17] = diag.imag;
    // The certified reciprocal supplies the actual component infinity
    // norm of division. A fixed 4/max(|d|) multiplies its slack at every
    // dependency level and can manufacture exponential uncertainty.
    float inverseNorm = SpsvFloatBoundAdd(SpsvFloatBoundAdd(SpsvAbsF(inverse.re.hi), SpsvAbsF(inverse.re.lo)),
                                          SpsvFloatBoundAdd(SpsvAbsF(inverse.im.hi), SpsvAbsF(inverse.im.lo)));
    rowPrep[18 * i + 8] = bounded && inverseError >= 0.0f ?
                              SpsvFloatBoundAdd(inverseNorm, SpsvFloatBoundMul(2.0f, inverseError)) :
                              0.0f;
}

template <typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvComplexDiagRecipCompute(
    __gm__ const SpsvComplex64* values, __gm__ const PermT* diagPtr, __gm__ const volatile SpsvComplex64* vecX,
    SpsvComplex64 alpha, __gm__ float* rowPrep, int64_t m, int32_t blockId, int32_t numBlocks, bool deferredCertificate)
{
    int64_t start = static_cast<int64_t>(blockId) * static_cast<int64_t>(blockDim.x) +
                    static_cast<int64_t>(threadIdx.x);
    int64_t stride = static_cast<int64_t>(numBlocks) * static_cast<int64_t>(blockDim.x);
    for (int64_t i = start; i < m; i += stride) {
        SpsvComplex64 diag{1.0f, 0.0f};
        if (diagPtr != nullptr) {
            diag = {0.0f, 0.0f};
            PermT dp = diagPtr[i];
            if (dp >= 0) {
                diag = {values[dp].real, values[dp].imag};
            }
        }
        // Normalising by the larger component bounds the only sum that
        // could overflow: dU*dU + dV*dV is always in [1, 2].  What is stored
        // is the reciprocal of that normalised diagonal, never the reciprocal
        // itself -- above |diag| of about 3.4e37 the true reciprocal is
        // subnormal, and this part flushes subnormals to zero, so storing it
        // would lose the row.  The scale goes alongside and the row divides
        // by it.  A zero diagonal gives a NaN scale and a non-finite result,
        // which is what a singular row owes.
        float dScale = SpsvMaxAbs2(diag.real, diag.imag);
        float dU = diag.real / dScale;
        float dV = diag.imag / dScale;
        float dT = 1.0f / (dU * dU + dV * dV);
        SpsvComplex64 recip{dU * dT, -dV * dT};
        // alpha * x for this row.  When the scale is at least one it comes
        // off here rather than in the row: alpha * x can leave fp32 range
        // while alpha * x / diag sits well inside it (alpha = 2e38, x = 2,
        // diag = 4).  It is applied to whichever operand is larger, because
        // dividing the smaller one can send it subnormal and this part
        // flushes subnormals to zero -- alpha = 1 over a diagonal of 2^127
        // would become exactly zero and lose the row.
        SpsvComplex64 x{vecX[i].real, vecX[i].imag};
        SpsvDF a0, a1;
        SpsvPrepareComplexProduct(alpha, x, a0, a1);
        // Only when alpha * x itself leaves fp32 range is anything done
        // here (alpha = 2e38, x = 2, diag = 4: the product is 4e38 and the
        // quotient 1e38).  The test is on the product, not on the scale:
        // dScale is the larger component, while what decides whether
        // dividing by the diagonal shrinks is |diag| -- for 0.75+0.75i the
        // component is 0.75 and the modulus 1.06, and dividing by 0.75 alone
        // would inflate the product rather than reduce it.
        //
        // So the recovery absorbs the whole 1/diag -- the normalised
        // reciprocal and the scale together -- into whichever operand is
        // larger.  The smaller one would go subnormal, and this part flushes
        // subnormals to zero: alpha = 1 over a diagonal of 2^127 would become
        // exactly zero and lose the row.
        //
        // The scale is stored negated to say that the right-hand side already
        // carries the whole division.  That matters because the row seeds its
        // accumulator with the right-hand side, and may only do that while
        // the two are in the same units; seeding is what keeps a dot product
        // that cancels against the right-hand side finite.
        float storedScale = dScale;
        if (!(SpsvIsFiniteF(a0.hi) && SpsvIsFiniteF(a1.hi))) {
            // Dividing whichever operand is larger is not enough: with
            // A = 1+0.5i, X = 0.65-0.65i and alpha = 3e38+3e38i, alpha/diag
            // is 3.6e38+1.2e38i and its real part is already out of range,
            // while (alpha * X)/diag is 3.12e38-1.56e38i and is not.  So the
            // product and the division are formed as one expression, every
            // operand normalised to a power of two first and a single rescale
            // at the end -- see SpsvComplexScaledProductDivByDiag.  This runs
            // once per row in a kernel of its own, off the dependency chain.
            SpsvComplexScaledProductDivByDiag(alpha, x, recip, dScale, a0, a1);
            // Unit rows never consume a pre-divided RHS. Keep it in the
            // numerator so exceptional alpha*x cannot be silently omitted.
            storedScale = diagPtr != nullptr ? -dScale : dScale;
        }
        rowPrep[18 * i] = recip.real;
        rowPrep[18 * i + 1] = recip.imag;
        rowPrep[18 * i + 2] = a0.hi;
        rowPrep[18 * i + 3] = a0.lo;
        rowPrep[18 * i + 4] = a1.hi;
        rowPrep[18 * i + 5] = a1.lo;
        rowPrep[18 * i + 6] = storedScale;
        rowPrep[18 * i + 7] = deferredCertificate ? kSpsvDeferredComplexCertificate :
                                                    static_cast<float>(a0.lost | a1.lost);
        SpsvPrepareComplexInverseBounds(alpha, x, diag, storedScale, dScale, diagPtr != nullptr, rowPrep, i);
    }
}

#endif // OP_KERNEL_VALIDATION_SPSV_COMPLEX_PREPARATION_H_
