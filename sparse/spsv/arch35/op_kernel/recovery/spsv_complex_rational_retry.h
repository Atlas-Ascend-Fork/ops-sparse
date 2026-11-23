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

#ifndef SPSV_COMPLEX_RATIONAL_RETRY_H_
#define SPSV_COMPLEX_RATIONAL_RETRY_H_

#include "../common/spsv_complex_workspace.h"

__simt_callee__ __attribute__((always_inline)) inline void SpsvComplexRationalDependencyRetry(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ SpsvComplex64* vecY, __gm__ float* yWide, __gm__ const int32_t* diagPtr,
    SpsvComplex64 alpha, int64_t m, bool forward, bool nonUnit)
{
    SpsvComplexScratchView scratch{yWide, {static_cast<uint64_t>(m)}};
    __gm__ float* rowPrep = scratch.rowPrep;
    __gm__ int32_t* scale = scratch.scale;
    __gm__ uint32_t* residue = scratch.residue;
    __gm__ uint16_t* digits = scratch.digits;
    int64_t capacity = static_cast<int64_t>(scratch.geometry.rationalDigits);
    SpsvBigUInt numerators[kSpsvRationalTargetBatch];
    for (int32_t i = 0; i < kSpsvRationalTargetBatch; ++i) {
        numerators[i] = {digits + static_cast<int64_t>(i) * capacity, 0};
    }
    SpsvBigUInt modulus{digits + kSpsvRationalTargetBatch * capacity, 0};
    SpsvBigUInt denominator{digits + (kSpsvRationalTargetBatch + 1) * capacity, 0};
    SpsvBigUInt temporary{digits + (kSpsvRationalTargetBatch + 2) * capacity, 0};
    for (int64_t first = 0; first < 2 * m;) {
        int64_t targets[kSpsvRationalTargetBatch];
        int32_t count = 0;
        for (; first < 2 * m && count < kSpsvRationalTargetBatch; ++first) {
            int64_t r = first / 2;
            bool imaginary = (first & 1) != 0;
            if (SpsvFloatNeedsDependencyRetry(imaginary ? vecY[r].imag : vecY[r].real,
                                              yWide[4 * r + (imaginary ? 3 : 1)],
                                              rowPrep[18 * r + (imaginary ? 10 : 9)])) {
                targets[count++] = first;
            }
        }
        if (count == 0) {
            continue;
        }
        float answers[kSpsvRationalTargetBatch];
        uint32_t answerValid[kSpsvRationalTargetBatch];
        if (SpsvComplexRationalSolveBatch(rowPtr, colInd, values, rhs, diagPtr, alpha, m, targets, count, forward,
                                          nonUnit, scale, residue, numerators, modulus, denominator, temporary, answers,
                                          answerValid)) {
            for (int32_t i = 0; i < count; ++i) {
                if (answerValid[i] == 0) {
                    continue;
                }
                if ((targets[i] & 1) == 0) {
                    vecY[targets[i] / 2].real = answers[i];
                } else {
                    vecY[targets[i] / 2].imag = answers[i];
                }
            }
        }
    }
}

#endif // SPSV_COMPLEX_RATIONAL_RETRY_H_
