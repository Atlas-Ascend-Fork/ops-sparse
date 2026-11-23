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
#ifndef SPSV_COMPLEX_DEFECT_H_
#define SPSV_COMPLEX_DEFECT_H_

// Four-word expansion of the original-equation defect. This changes only
// the enclosure cache; the ordinary two-word solution is left untouched.
struct SpsvDefectExpansion {
    float word[4];
    float discarded;
};

__simt_callee__ inline void SpsvDefectExpansionAdd(SpsvDefectExpansion& sum, float term)
{
    for (int32_t i = 0; i < 4; ++i) {
        SpsvDF next = SpsvTwoSum(sum.word[i], term);
        sum.word[i] = next.hi;
        term = next.lo;
    }
    sum.discarded += SpsvAbsF(term);
}

__simt_callee__ inline bool SpsvDefectExpansionProduct(SpsvDefectExpansion& sum, float a, float b)
{
    if (!SpsvBlockFiniteNormal(a) || !SpsvBlockFiniteNormal(b)) {
        return false;
    }
    float product = a * b;
    if (!SpsvIsFiniteF(product)) {
        return false;
    }
    SpsvDefectExpansionAdd(sum, product);
    SpsvDefectExpansionAdd(sum, fmaf(a, b, -product));
    return true;
}

__simt_callee__ inline bool SpsvComplexDefectAccumulate(SpsvDefectExpansion& re, SpsvDefectExpansion& im,
                                                        const SpsvComplexCertificateData& data, int32_t c,
                                                        SpsvComplex64 a)
{
    bool valid = true;
    float yr = data.vecY[c].real, yi = data.vecY[c].imag;
    float lr = data.wide[4 * static_cast<int64_t>(c) + 1];
    float li = data.wide[4 * static_cast<int64_t>(c) + 3];
    valid = SpsvDefectExpansionProduct(re, -a.real, yr) && valid;
    valid = SpsvDefectExpansionProduct(re, -a.real, lr) && valid;
    valid = SpsvDefectExpansionProduct(re, a.imag, yi) && valid;
    valid = SpsvDefectExpansionProduct(re, a.imag, li) && valid;
    valid = SpsvDefectExpansionProduct(im, -a.real, yi) && valid;
    valid = SpsvDefectExpansionProduct(im, -a.real, li) && valid;
    valid = SpsvDefectExpansionProduct(im, -a.imag, yr) && valid;
    valid = SpsvDefectExpansionProduct(im, -a.imag, lr) && valid;
    return valid;
}

__simt_callee__ inline bool SpsvComplexTightDefectTerms(const SpsvComplexCertificateData& data, int64_t row,
                                                        SpsvComplex64& diagonal, SpsvDefectExpansion& re,
                                                        SpsvDefectExpansion& im)
{
    bool valid = SpsvDefectExpansionProduct(re, data.alpha.real, data.rhs[row].real) &&
                 SpsvDefectExpansionProduct(re, -data.alpha.imag, data.rhs[row].imag) &&
                 SpsvDefectExpansionProduct(im, data.alpha.real, data.rhs[row].imag) &&
                 SpsvDefectExpansionProduct(im, data.alpha.imag, data.rhs[row].real);
    for (int64_t p = data.rowPtr[row] - 1; p < data.rowPtr[row + 1]; ++p) {
        int32_t c = p < data.rowPtr[row] ? static_cast<int32_t>(row) : data.colInd[p];
        if (p >= data.rowPtr[row] && !(data.forward ? c < row : c > row)) {
            continue;
        }
        SpsvComplex64 a = p < data.rowPtr[row] ? diagonal : SpsvComplex64{data.values[p].real, data.values[p].imag};
        valid = SpsvComplexDefectAccumulate(re, im, data, c, a) && valid;
    }
    return valid;
}

__simt_callee__ inline bool SpsvComplexTightDefect(const SpsvComplexCertificateData& data, int64_t row, float* out)
{
    if (data.rowPtr[row + 1] - data.rowPtr[row] > 64 || (data.nonUnit && data.diagPtr[row] < 0)) {
        return false;
    }
    SpsvDefectExpansion re{{0.f, 0.f, 0.f, 0.f}, 0.f};
    SpsvDefectExpansion im{{0.f, 0.f, 0.f, 0.f}, 0.f};
    SpsvComplex64 diagonal = data.nonUnit ? SpsvComplex64{data.values[data.diagPtr[row]].real,
                                                          data.values[data.diagPtr[row]].imag} :
                                            SpsvComplex64{1.f, 0.f};
    if (!SpsvComplexTightDefectTerms(data, row, diagonal, re, im)) {
        return false;
    }
    SpsvDF r{0.f, 0.f}, i{0.f, 0.f};
    float er = 0.f, ei = 0.f;
    for (int32_t j = 3; j >= 0; --j) {
        SpsvBlockAcc(r, er, 1.f, {re.word[j], 0.f});
        SpsvBlockAcc(i, ei, 1.f, {im.word[j], 0.f});
    }
    float discarded = re.discarded + im.discarded;
    float n = static_cast<float>(data.rowPtr[row + 1] - data.rowPtr[row] + 4);
    float radius = SpsvFloatBoundUp(discarded * (1.f + n * 0x1p-16f) + (er + ei) * 0x1p-20f + n * 0x1p-110f);
    if (!SpsvIsFiniteF(r.hi) || !SpsvIsFiniteF(r.lo) || !SpsvIsFiniteF(i.hi) || !SpsvIsFiniteF(i.lo) ||
        !SpsvIsFiniteF(radius)) {
        return false;
    }
    out[0] = r.hi;
    out[1] = r.lo;
    out[2] = i.hi;
    out[3] = i.lo;
    out[4] = radius;
    return true;
}

#endif // SPSV_COMPLEX_DEFECT_H_
