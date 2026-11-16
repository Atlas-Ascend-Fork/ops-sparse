/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GPSV_INTERLEAVED_BATCH_GOLDEN_H_
#define TEST_GPSV_INTERLEAVED_BATCH_GOLDEN_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "fill.h"

namespace sparse_test {

/**
 * Build full m×m pentadiagonal matrix A and RHS bb for batch j (interleaved layout).
 * Row index `i` maps to position `i*batchCount + j` in the interleaved arrays.
 */
inline void BuildPentadiagSingleBatch(int m, int batchCount, int j,
                                      const std::vector<float>& ds_in,
                                      const std::vector<float>& dl_in,
                                      const std::vector<float>& d_in,
                                      const std::vector<float>& du_in,
                                      const std::vector<float>& dw_in,
                                      const std::vector<float>& b_in,
                                      std::vector<double>& A, std::vector<double>& bb)
{
    std::fill(A.begin(), A.end(), 0.0);
    for (int i = 0; i < m; i++) {
        int idx = i * batchCount + j;
        A[static_cast<int64_t>(i) * m + i] = static_cast<double>(d_in[idx]);
        if (i >= 1) A[static_cast<int64_t>(i) * m + (i - 1)] = static_cast<double>(dl_in[idx]);
        if (i >= 2) A[static_cast<int64_t>(i) * m + (i - 2)] = static_cast<double>(ds_in[idx]);
        if (i < m - 1) A[static_cast<int64_t>(i) * m + (i + 1)] = static_cast<double>(du_in[idx]);
        if (i < m - 2) A[static_cast<int64_t>(i) * m + (i + 2)] = static_cast<double>(dw_in[idx]);
        bb[i] = static_cast<double>(b_in[idx]);
    }
}

/**
 * Apply a Givens rotation on rows (r1, r2) of matrix A, affecting columns [colStart, m).
 * Also applies the same rotation to bb entries at rows r1 and r2.
 * A zero pivot and target require the identity rotation, even for nonsingular systems.
 */
inline void ApplyGivensRotation(
    std::vector<double>& A, std::vector<double>& bb,
    int m, int r1, int r2, int colStart, double pivot, double target)
{
    if (pivot == 0.0 && target == 0.0) {
        return;
    }
    double r = std::hypot(pivot, target);
    double cs = pivot / r;
    double sn = target / r;
    for (int c = colStart; c < m; c++) {
        double a = A[static_cast<int64_t>(r1) * m + c];
        double b = A[static_cast<int64_t>(r2) * m + c];
        A[static_cast<int64_t>(r1) * m + c] = cs * a + sn * b;
        A[static_cast<int64_t>(r2) * m + c] = -sn * a + cs * b;
    }
    double tb1 = bb[r1], tb2 = bb[r2];
    bb[r1] = cs * tb1 + sn * tb2;
    bb[r2] = -sn * tb1 + cs * tb2;
}

/**
 * QR forward sweep: for rows i = 1..m-1, apply:
 *   Step 1 (i >= 2): Givens on rows (i-2, i) to eliminate A[i, i-2]
 *   Step 2 (i >= 1): Givens on rows (i-1, i) to eliminate A[i, i-1]
 * Zero rotations leave both rows unchanged; singular final diagonals remain unguarded.
 */
inline void QrPentadiagForward(std::vector<double>& A, std::vector<double>& bb, int m)
{
    for (int i = 1; i < m; i++) {
        if (i >= 2) {
            double target = A[static_cast<int64_t>(i) * m + (i - 2)];
            double pivot  = A[static_cast<int64_t>(i - 2) * m + (i - 2)];
            ApplyGivensRotation(A, bb, m, i - 2, i, i - 2, pivot, target);
        }
        {
            double target = A[static_cast<int64_t>(i) * m + (i - 1)];
            double pivot  = A[static_cast<int64_t>(i - 1) * m + (i - 1)];
            ApplyGivensRotation(A, bb, m, i - 1, i, i - 1, pivot, target);
        }
    }
}

/**
 * Back-substitution: solve upper-triangular system A x = bb (x stored in output param).
 * No singularity guard: if diag == 0, IEEE-754 yields Inf/NaN — matches kernel behavior.
 */
inline void QrPentadiagBackward(const std::vector<double>& A, const std::vector<double>& bb,
                                int m, std::vector<double>& x)
{
    std::fill(x.begin(), x.end(), 0.0);
    for (int i = m - 1; i >= 0; i--) {
        double diag = A[static_cast<int64_t>(i) * m + i];
        double sum = bb[i];
        for (int c = i + 1; c < m; c++) {
            sum -= A[static_cast<int64_t>(i) * m + c] * x[c];
        }
        x[i] = sum / diag;
    }
}

/**
 * QR / Givens rotation algorithm (FP64) golden reference for GPSV interleaved batch.
 * Dispatches to BuildPentadiagSingleBatch / QrPentadiagForward / QrPentadiagBackward.
 */
inline std::vector<float> QrPentadiagSolveGolden(
    const std::vector<float>& ds_in, const std::vector<float>& dl_in,
    const std::vector<float>& d_in, const std::vector<float>& du_in,
    const std::vector<float>& dw_in, const std::vector<float>& b_in,
    int m, int batchCount)
{
    int total = m * batchCount;
    std::vector<float> result(total, 0.0f);
    if (m <= 0 || batchCount <= 0) return result;

    std::vector<double> A(static_cast<int64_t>(m) * m, 0.0);
    std::vector<double> bb(m, 0.0);
    std::vector<double> x(m, 0.0);

    for (int j = 0; j < batchCount; j++) {
        BuildPentadiagSingleBatch(m, batchCount, j, ds_in, dl_in, d_in, du_in, dw_in, b_in, A, bb);
        QrPentadiagForward(A, bb, m);
        QrPentadiagBackward(A, bb, m, x);
        for (int i = 0; i < m; i++) {
            result[i * batchCount + j] = static_cast<float>(x[i]);
        }
    }
    return result;
}

/**
 * Verify QR pentadiagonal golden self-consistency: generate known solution,
 * compute b = Ax, solve golden, check solution matches.
 */
inline bool QrPentadiagGoldenSelfTest(int m, int batchCount, uint32_t seed, double fp64_tol = 1e-4) {
    if (m <= 0 || batchCount <= 0) return false;
    auto pent = makeDiagDominantPentadiag(m, batchCount, seed);
    std::mt19937 rng(seed + 100);
    std::uniform_real_distribution<float> xDist(-5.0f, 5.0f);
    std::vector<float> xTrue(m * batchCount, 0.0f);
    for (int i = 0; i < m * batchCount; i++) xTrue[i] = xDist(rng);
    auto b = makeKnownSolutionPentadiag(pent, xTrue);
    auto xSolved = QrPentadiagSolveGolden(pent.ds, pent.dl, pent.d, pent.du, pent.dw, b, m, batchCount);
    double maxErr = 0.0;
    for (size_t i = 0; i < xSolved.size(); i++) {
        double err = std::abs(static_cast<double>(xSolved[i]) - static_cast<double>(xTrue[i]));
        if (err > maxErr) maxErr = err;
    }
    return (maxErr < fp64_tol);
}

}  // namespace sparse_test

#endif
