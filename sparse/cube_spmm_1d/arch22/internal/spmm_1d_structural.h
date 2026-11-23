/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSPARSE_SPMM_1D_STRUCTURAL_H_
#define ACLSPARSE_SPMM_1D_STRUCTURAL_H_
#include "spmm_1d_catalog.h"
struct SpmmStructuralDecision { int key; bool minimalPadding; int reason; };
struct SpmmStructuralParameters {
    int directMaxAt64 = 24;
    int directMaxAt256 = 20;
    int directMaxAt2048 = 40;
    int narrowColumnDomain = 262144;
    int smallRowsPerCore = 4;
    int coreCount = 20;
};
inline const SpmmStructuralParameters &SpmmDefaultStructuralParameters()
{
    static const SpmmStructuralParameters parameters {};
    return parameters;
}

struct SpmmStructuralShape {
    int m, n, k;
    int64_t p, totalVectors, nonemptyRows;
    bool fewRows;
    const SpmmStructuralParameters& parameters;
};

inline SpmmStructuralDecision SpmmStructuralSmall(const SpmmStructuralShape& shape)
{
    if (shape.p > 32) {
        if (shape.fewRows && shape.totalVectors <= shape.nonemptyRows * 32) return {0, false, 1};
        return {36, true, 1};
    }
    if (shape.p <= (shape.n <= 64 ? shape.parameters.directMaxAt64 : shape.parameters.directMaxAt256) &&
        (shape.n <= 128 || shape.k <= shape.parameters.narrowColumnDomain)) return {9, true, 2};
    return {0, false, 3};
}

inline SpmmStructuralDecision SpmmStructuralOnePanel(const SpmmStructuralShape& shape)
{
    if (shape.p <= 32 && shape.fewRows) return {9, true, 4};
    if (shape.p > 32 && shape.p <= 128 && shape.fewRows) return {34, false, 5};
    if (shape.p <= 32 && shape.k <= shape.parameters.narrowColumnDomain) return {36, false, 4};
    if (shape.p <= 16) return {37, false, 4};
    if (shape.p <= 32) return {33, false, 4};
    if (shape.p <= 64) return {15, false, 5};
    if (shape.p <= 128) return {36, false, 5};
    if (shape.p <= 256) return {37, false, 6};
    return {shape.p <= 2048 ? 34 : 37, false, 7};
}

inline SpmmStructuralDecision SpmmStructuralTwoPanels(const SpmmStructuralShape& shape)
{
    if (shape.p <= 32) return {9, shape.fewRows, 8};
    if (shape.p <= 128) return {shape.totalVectors > shape.nonemptyRows * 64 ? 34 : 37, false, 9};
    return {shape.p <= 2048 ? 34 : 37, false, 7};
}

inline SpmmStructuralDecision SpmmStructuralMedium(const SpmmStructuralShape& shape)
{
    if (shape.p <= 16 && shape.m <= 4096) return {39, false, 8};
    if (shape.p <= shape.parameters.directMaxAt2048) return {9, false, 8};
    if (shape.p <= 64) return {10, false, 10};
    return {shape.p <= 2048 ? 34 : 37, false, 7};
}

inline SpmmStructuralDecision SpmmStructuralLarge(const SpmmStructuralShape& shape)
{
    if (shape.n <= 4096 && shape.p <= 16 && shape.m <= 4096) return {39, false, 11};
    if (shape.p <= 32 || (shape.n <= 4096 && shape.p <= 40)) return {shape.n <= 4096 ? 9 : 10, false, 11};
    if (shape.p <= 160) return {23, false, 12};
    if (shape.p <= 1024) return {34, false, 12};
    return {5, false, 13};
}

inline SpmmStructuralDecision SpmmStructuralWide(const SpmmStructuralShape& shape)
{
    if (shape.p <= 32) return {1, false, 14};
    if (shape.p <= 256) return {shape.totalVectors > shape.nonemptyRows * 128 ? 34 : 8, false, 14};
    if (shape.p <= 512) return {6, false, 14};
    if (shape.p <= 2048) return {7, false, 14};
    return {5, false, 13};
}

// Width classes partition the complete rule table; no thresholds or ordering change.
inline SpmmStructuralDecision SpmmSelectStructural(
    int m, int n, int k, int64_t p, int64_t totalVectors, int64_t emptyRows,
    const SpmmStructuralParameters& parameters = SpmmDefaultStructuralParameters())
{
    const SpmmStructuralShape shape{m, n, k, p, totalVectors, static_cast<int64_t>(m) / 16 - emptyRows,
                                    m / 16 <= parameters.smallRowsPerCore * parameters.coreCount, parameters};
    if (n <= 256) return SpmmStructuralSmall(shape);
    if (n <= 512) return SpmmStructuralOnePanel(shape);
    if (n <= 1024) return SpmmStructuralTwoPanels(shape);
    if (n <= 2048) return SpmmStructuralMedium(shape);
    if (n <= 8192) return SpmmStructuralLarge(shape);
    return SpmmStructuralWide(shape);
}
#endif
