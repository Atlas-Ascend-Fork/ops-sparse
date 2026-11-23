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

#ifndef SPSV_SCALED_DATA_H_
#define SPSV_SCALED_DATA_H_

#include "../validation/spsv_complex_prefix.h"

struct SpsvRealScaledData {
    __gm__ const int32_t* rowPtr;
    __gm__ const int32_t* colInd;
    __gm__ const float* values;
    __gm__ const float* rhs;
    __gm__ const int32_t* diagPtr;
    __gm__ const float* vecY;
    __gm__ const float* wide;
    SpsvComplex64 alpha;
    bool forward;
    bool nonUnit;
};

__simt_callee__ inline int32_t SpsvScaledComponentCount(const SpsvComplexCertificateData&) { return 2; }

__simt_callee__ inline int32_t SpsvScaledComponentCount(const SpsvRealScaledData&) { return 1; }

__simt_callee__ inline SpsvComplex64 SpsvScaledCoefficient(const SpsvComplexCertificateData& data, int64_t p)
{
    return {data.values[p].real, data.values[p].imag};
}

__simt_callee__ inline SpsvComplex64 SpsvScaledCoefficient(const SpsvRealScaledData& data, int64_t p)
{
    return {data.values[p], 0.f};
}

__simt_callee__ inline SpsvComplex64 SpsvScaledRhs(const SpsvComplexCertificateData& data, int64_t row)
{
    return {data.rhs[row].real, data.rhs[row].imag};
}

__simt_callee__ inline SpsvComplex64 SpsvScaledRhs(const SpsvRealScaledData& data, int64_t row)
{
    return {data.rhs[row], 0.f};
}

__simt_callee__ inline bool SpsvScaledColumnOrder(const SpsvComplexCertificateData& data, int64_t row, int64_t p)
{
    return p == data.rowPtr[row] || data.colInd[p] > data.colInd[p - 1];
}

__simt_callee__ inline bool SpsvScaledColumnOrder(const SpsvRealScaledData& data, int64_t row, int64_t p)
{
    // Transpose scatter preserves all terms but need not sort them. Reject
    // duplicate diagonals; the affine sum explicitly visits every other term.
    return data.colInd[p] != row || p == data.diagPtr[row];
}

__simt_callee__ inline void SpsvScaledWriteComponent(const SpsvComplexCertificateData& data, int64_t m, int64_t row,
                                                     int32_t component, float high, float low, float bound)
{
    auto wide = (__gm__ float*)data.wide;
    auto prep = wide + 6 * m;
    auto y = (__gm__ SpsvComplex64*)data.vecY;
    if (component == 0) {
        y[row].real = high;
    } else {
        y[row].imag = high;
    }
    wide[4 * row + 2 * component] = high;
    wide[4 * row + 2 * component + 1] = low;
    prep[18 * row + 9 + component] = bound;
}

__simt_callee__ inline void SpsvScaledWriteComponent(const SpsvRealScaledData& data, int64_t m, int64_t row,
                                                     int32_t component, float high, float low, float bound)
{
    if (component == 0) {
        ((__gm__ float*)data.vecY)[row] = high;
    }
}

#endif // SPSV_SCALED_DATA_H_
