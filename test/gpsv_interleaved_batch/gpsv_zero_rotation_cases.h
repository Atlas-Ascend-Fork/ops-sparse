/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GPSV_ZERO_ROTATION_CASES_H
#define GPSV_ZERO_ROTATION_CASES_H

#include <numeric>
#include "fill.h"

namespace sparse_test {
struct GpsvKnownSolutionCase {
    PentadiagMatrix matrix;
    std::vector<float> rhs;
    std::vector<float> expected;
};

// A uniformly scaled permutation has condition number one; only bandwidth <= 2 is retained.
inline GpsvKnownSolutionCase MakeGpsvPermutationCase(int m, float scale)
{
    std::vector<int> permutation(m);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<std::vector<int>> permutations;
    do {
        bool banded = true;
        for (int row = 0; row < m; ++row) {
            banded = banded && std::abs(permutation[row] - row) <= 2;
        }
        if (banded) {
            permutations.push_back(permutation);
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    const int batches = static_cast<int>(permutations.size());
    GpsvKnownSolutionCase result{makeZeroPentadiag(m, batches), std::vector<float>(m * batches),
                                 std::vector<float>(m * batches)};
    std::vector<float>* diagonals[] = {&result.matrix.ds, &result.matrix.dl, &result.matrix.d, &result.matrix.du,
                                       &result.matrix.dw};
    for (int batch = 0; batch < batches; ++batch) {
        for (int row = 0; row < m; ++row) {
            const int column = permutations[batch][row];
            const int index = row * batches + batch;
            (*diagonals[column - row + 2])[index] = scale;
            result.rhs[index] = scale * static_cast<float>(column + 1);
            result.expected[index] = static_cast<float>(row + 1);
        }
    }
    return result;
}

// Cover both FP32 exponent limits while keeping every input and the solution finite.
inline std::vector<GpsvKnownSolutionCase> MakeGpsvExtremeCases()
{
    std::vector<GpsvKnownSolutionCase> cases;
    for (float scale : {0x1p-126F, -0x1p-126F, 0x1p127F, -0x1p127F}) {
        GpsvKnownSolutionCase diagonal{makeZeroPentadiag(2, 1), {scale, scale}, {1.0F, 1.0F}};
        diagonal.matrix.d = {scale, scale};
        cases.push_back(diagonal);
    }
    return cases;
}

// Uniform scaling preserves condition number one; the solution remains independent of scale.
inline std::vector<GpsvKnownSolutionCase> MakeGpsvRangeCases()
{
    auto cases = MakeGpsvExtremeCases();
    for (float scale : {0x1p-100F, -0x1p-100F, 1.0F, -1.0F, 0x1p100F, -0x1p100F}) {
        cases.push_back(MakeGpsvPermutationCase(3, scale));
        // A = scale * [[3, 4], [-4, 3]], b = A * [1, 2]. Both rotation inputs are nonzero.
        GpsvKnownSolutionCase rotation{makeZeroPentadiag(2, 1), {11.0F * scale, 2.0F * scale}, {1.0F, 2.0F}};
        rotation.matrix.d = {3.0F * scale, 3.0F * scale};
        rotation.matrix.dl[1] = -4.0F * scale;
        rotation.matrix.du[0] = 4.0F * scale;
        cases.push_back(rotation);
    }
    return cases;
}
} // namespace sparse_test
#endif
