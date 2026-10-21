/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GTSV2_GTSV2_GOLDEN_H_
#define TEST_GTSV2_GTSV2_GOLDEN_H_

#include <algorithm>
#include <cmath>
#include <vector>

#include "fill.h"

namespace sparse_test {

/**
 * FP64 Thomas algorithm with partial pivoting — golden reference for gtsv2
 * (column-major layout).
 *
 * Solves A * X = B where A is a tridiagonal matrix defined by diagonals (dl, d, du):
 *   - dl: [m] sub-diagonal,  dl[0] is unused (0), dl[1..m-1] are the sub-diagonal entries
 *   - d:  [m] main diagonal
 *   - du: [m] super-diagonal, du[m-1] is unused (0), du[0..m-2] are the super-diagonal entries
 *
 * B is an m x n dense matrix in column-major layout (ldb x n).
 * Output X overwrites B in-place (also column-major, ldb x n).
 *
 * Partial pivoting (the key difference from gtsv2_nopivot):
 *   At each forward-elimination step i (i = 1..m-1), the sub-diagonal element
 *   dl[i] is compared with the current pivot d[i-1]. If |dl[i]| > |d[i-1]|,
 *   rows i-1 and i are swapped (including the RHS). A row swap may push the
 *   super-diagonal of row i onto the second super-diagonal (du2) of row i-1;
 *   this fill-in is tracked explicitly via a du2 scratch array.
 *
 * Forward elimination (step i = 1..m-1):
 *   if |dl[i]| > |d[i-1]|:
 *       swap(d[i-1], dl[i])      // pivot becomes the old sub-diagonal
 *       swap(du[i-1], d[i])      // old super-diagonal becomes new diagonal of row i
 *       swap(du2[i-1], du[i])    // old super-diagonal of row i becomes 2nd super-diag of row i-1
 *       swap(B[i-1], B[i])
 *   mult = dl[i] / d[i-1]
 *   d[i]  -= mult * du[i-1]
 *   du[i] -= mult * du2[i-1]     // only when i < m-1 (du[i] unused at last row)
 *   B[i]  -= mult * B[i-1]
 *   dl[i] = 0
 *
 * Back substitution (from last row upward):
 *   x[m-1] = B[m-1] / d[m-1]
 *   x[i]   = (B[i] - du[i]*x[i+1] - du2[i]*x[i+2]) / d[i]
 *            (the du2 term is applied only when i+2 < m)
 *
 * Each RHS column is solved independently using FP64 arithmetic.
 * No singular protection is applied: a zero pivot produces Inf/NaN naturally.
 * Output is truncated to float.
 */
inline void Gtsv2Golden(
    const std::vector<float>& dl,
    const std::vector<float>& d,
    const std::vector<float>& du,
    std::vector<float>& B,  // in-place: input RHS, output solution
    int m, int n, int ldb)
{
    if (m <= 0 || n <= 0) return;

    // Pre-allocate FP64 scratch buffers (reused across RHS columns)
    std::vector<double> dl_w(m, 0.0);
    std::vector<double> d_w(m, 0.0);
    std::vector<double> du_w(m, 0.0);
    std::vector<double> du2_w(m, 0.0);  // second super-diagonal (fill-in from pivoting)
    std::vector<double> b_w(m, 0.0);

    // Solve each RHS column independently using FP64 Thomas with partial pivoting
    for (int j = 0; j < n; j++) {
        // Load diagonals and RHS column into FP64 working buffers
        for (int i = 0; i < m; i++) {
            dl_w[i]  = static_cast<double>(dl[i]);
            d_w[i]   = static_cast<double>(d[i]);
            du_w[i]  = static_cast<double>(du[i]);
            du2_w[i] = 0.0;
            b_w[i]   = static_cast<double>(B[i + j * ldb]);
        }

        // ---- Forward elimination with partial pivoting ----
        for (int i = 1; i < m; i++) {
            // Partial pivot: compare |dl[i]| with |d[i-1]|
            if (std::abs(dl_w[i]) > std::abs(d_w[i - 1])) {
                // Swap rows i-1 and i (coefficients + RHS)
                std::swap(d_w[i - 1],   dl_w[i]);
                std::swap(du_w[i - 1],  d_w[i]);
                std::swap(du2_w[i - 1], du_w[i]);
                std::swap(b_w[i - 1],   b_w[i]);
            }

            // Eliminate sub-diagonal dl_w[i]
            double mult = dl_w[i] / d_w[i - 1];
            d_w[i] -= mult * du_w[i - 1];
            if (i < m - 1) {
                du_w[i] -= mult * du2_w[i - 1];
            }
            b_w[i] -= mult * b_w[i - 1];
            dl_w[i] = 0.0;
        }

        // ---- Back substitution (last row upward) ----
        // Keep the complete reference solve in FP64.  Casting an already
        // solved x[i] to FP32 before using it for x[i-1] would make this an
        // FP32/FP64 hybrid instead of the task-required FP64 CPU golden.
        b_w[m - 1] /= d_w[m - 1];
        for (int i = m - 2; i >= 0; i--) {
            double xi = b_w[i] - du_w[i] * b_w[i + 1];
            if (i + 2 < m) {
                xi -= du2_w[i] * b_w[i + 2];
            }
            b_w[i] = xi / d_w[i];
        }
        for (int i = 0; i < m; ++i) {
            B[i + j * ldb] = static_cast<float>(b_w[i]);
        }
    }
}

}  // namespace sparse_test

#endif  // TEST_GTSV2_GTSV2_GOLDEN_H_
