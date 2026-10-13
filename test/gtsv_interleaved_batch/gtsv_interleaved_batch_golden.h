/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GTSV_INTERLEAVED_BATCH_GOLDEN_H_
#define TEST_GTSV_INTERLEAVED_BATCH_GOLDEN_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "fill.h"

namespace sparse_test {

/**
 * Thomas algorithm (FP64 double) golden reference for GTSV interleaved batch.
 *
 * Solves batch of tridiagonal systems A^(j) * x^(j) = b^(j), j = 0..batchCount-1
 * Data layout: row-major interleaved, data[row * batchCount + batch].
 *
 * Forward elimination:
 *   c'[0] = du[0] / d[0]
 *   b'[0] = b[0] / d[0]
 *   c'[i] = du[i] / (d[i] - dl[i] * c'[i-1])   for i = 1..m-2
 *   b'[i] = (b[i] - dl[i] * b'[i-1]) / (d[i] - dl[i] * c'[i-1])
 *
 * Back substitution:
 *   x[m-1] = b'[m-1]
 *   x[i]   = b'[i] - c'[i] * x[i+1]             for i = m-2..0
 */
inline std::vector<float> ThomasSolveGolden(
    const std::vector<float>& dl,
    const std::vector<float>& d,
    const std::vector<float>& du,
    const std::vector<float>& b,
    int m, int batchCount)
{
    int total = m * batchCount;
    std::vector<float> x(total, 0.0f);
    if (m <= 0 || batchCount <= 0) return x;

    // Each batch is solved independently in FP64.
    // No singularity guards: matches kernel and cuSPARSE behavior.
    // For singular inputs, IEEE-754 division by zero produces Inf/NaN,
    // which propagates through the output naturally.
    for (int j = 0; j < batchCount; j++) {
        std::vector<double> c_prime(m, 0.0);
        std::vector<double> b_prime(m, 0.0);
        std::vector<double> x_batch(m, 0.0);

        // Forward elimination: row 0
        {
            double d0  = static_cast<double>(d[0 * batchCount + j]);
            double du0 = static_cast<double>(du[0 * batchCount + j]);
            double b0  = static_cast<double>(b[0 * batchCount + j]);
            c_prime[0] = du0 / d0;
            b_prime[0] = b0 / d0;
        }

        // Forward elimination: rows 1..m-1
        for (int i = 1; i < m; i++) {
            double dl_i = static_cast<double>(dl[i * batchCount + j]);
            double d_i  = static_cast<double>(d[i * batchCount + j]);
            double du_i = static_cast<double>(du[i * batchCount + j]);
            double b_i  = static_cast<double>(b[i * batchCount + j]);

            double denom = d_i - dl_i * c_prime[i - 1];
            if (i < m - 1) {
                c_prime[i] = du_i / denom;
            }
            b_prime[i] = (b_i - dl_i * b_prime[i - 1]) / denom;
        }

        // Back substitution
        x_batch[m - 1] = b_prime[m - 1];
        for (int i = m - 2; i >= 0; i--) {
            x_batch[i] = b_prime[i] - c_prime[i] * x_batch[i + 1];
        }

        // Write results to interleaved output
        for (int i = 0; i < m; i++) {
            x[i * batchCount + j] = static_cast<float>(x_batch[i]);
        }
    }

    return x;
}

/**
 * Verify Thomas golden self-consistency: generate known solution, compute b=Ax,
 * solve golden, check solution matches. Returns true if all errors < fp64_tol.
 */
inline bool ThomasGoldenSelfTest(int m, int batchCount, uint32_t seed, double fp64_tol = 1e-4) {
    // Use diagonal-dominant matrix for stability
    auto tri = makeDiagDominantTridiag(m, batchCount, seed);

    // Generate known solution
    std::mt19937 rng(seed + 100);
    std::uniform_real_distribution<float> xDist(-5.0f, 5.0f);
    std::vector<float> xTrue(m * batchCount, 0.0f);
    for (int i = 0; i < m * batchCount; i++) xTrue[i] = xDist(rng);

    // Compute b = A * x_true
    auto b = makeKnownSolutionTridiag(tri, xTrue);

    // Golden solve
    auto xSolved = ThomasSolveGolden(tri.dl, tri.d, tri.du, b, m, batchCount);

    // Compare
    double maxErr = 0.0;
    for (int i = 0; i < (int)xSolved.size(); i++) {
        double err = std::abs(static_cast<double>(xSolved[i]) - static_cast<double>(xTrue[i]));
        if (!std::isfinite(err)) return false;
        if (err > maxErr) maxErr = err;
    }
    return (maxErr < fp64_tol);
}

}  // namespace sparse_test

#endif
