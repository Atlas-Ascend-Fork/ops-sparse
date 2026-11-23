/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPSV_GOLDEN_H_
#define TEST_SPSV_GOLDEN_H_

#include <Eigen/Eigen>
#include <Eigen/SparseCore>
#include <complex>
#include <vector>
#include "../frame/fill.h"

namespace sparse_test {

template <typename T>
std::vector<double> ComputeRhs(const std::vector<T>& x, T alpha) {
    std::vector<double> rhs(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        rhs[i] = static_cast<double>(alpha) * static_cast<double>(x[i]);
    }
    return rhs;
}

inline Eigen::SparseMatrix<double, Eigen::RowMajor> BuildEigenMatrix(const CsrMatrix& csr) {
    int m = static_cast<int>(csr.rows);
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(m, m);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(csr.nnz));
    for (int i = 0; i < m; i++) {
        for (int j = static_cast<int>(csr.rowOffsets[i]); j < static_cast<int>(csr.rowOffsets[i + 1]); j++) {
            triplets.emplace_back(i, static_cast<int>(csr.colIndices[j]),
                                  static_cast<double>(csr.values[j]));
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

template <bool FORWARD>
void SolveTriangularLoop(const CsrMatrix& csr, const std::vector<double>& rhs,
                         bool unitDiag, std::vector<double>& y) {
    int m = static_cast<int>(csr.rows);
    if constexpr (FORWARD) {
        for (int i = 0; i < m; i++) {
            double sum = rhs[i];
            double diagVal = 0.0;
            for (int k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
                int col = csr.colIndices[k];
                if (col < i) {
                    sum -= static_cast<double>(csr.values[k]) * y[col];
                } else if (col == i) {
                    diagVal = static_cast<double>(csr.values[k]);
                }
            }
            y[i] = unitDiag ? sum : sum / diagVal;
        }
    } else {
        for (int i = m - 1; i >= 0; i--) {
            double sum = rhs[i];
            double diagVal = 0.0;
            for (int k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
                int col = csr.colIndices[k];
                if (col > i) {
                    sum -= static_cast<double>(csr.values[k]) * y[col];
                } else if (col == i) {
                    diagVal = static_cast<double>(csr.values[k]);
                }
            }
            y[i] = unitDiag ? sum : sum / diagVal;
        }
    }
}

inline bool DetermineForward(bool lower, bool transpose) {
    int fm = lower ? 0 : 1;
    int op = transpose ? 1 : 0;
    if (fm == 0 && op == 0) return true;
    if (fm == 1 && op != 0) return true;
    return false;
}

template <typename T>
std::vector<T> SpSVGolden(
    const CsrMatrix& csr,
    const std::vector<T>& x,
    T alpha,
    bool lower,
    bool unitDiag,
    bool transpose)
{
    int m = static_cast<int>(csr.rows);
    if (m <= 0) return {};

    auto A = BuildEigenMatrix(csr);
    auto rhsVec = ComputeRhs(x, alpha);
    Eigen::VectorXd rhs(m);
    for (int i = 0; i < m; i++) rhs(i) = rhsVec[i];

    Eigen::VectorXd result(m);
    if (!transpose) {
        if (lower && !unitDiag) {
            result = A.triangularView<Eigen::Lower>().solve(rhs);
        } else if (lower && unitDiag) {
            result = A.triangularView<Eigen::Lower | Eigen::UnitDiag>().solve(rhs);
        } else if (!lower && !unitDiag) {
            result = A.triangularView<Eigen::Upper>().solve(rhs);
        } else {
            result = A.triangularView<Eigen::Upper | Eigen::UnitDiag>().solve(rhs);
        }
    } else {
        if (lower && !unitDiag) {
            result = A.triangularView<Eigen::Lower>().transpose().solve(rhs);
        } else if (lower && unitDiag) {
            result = A.triangularView<Eigen::Lower | Eigen::UnitDiag>().transpose().solve(rhs);
        } else if (!lower && !unitDiag) {
            result = A.triangularView<Eigen::Upper>().transpose().solve(rhs);
        } else {
            result = A.triangularView<Eigen::Upper | Eigen::UnitDiag>().transpose().solve(rhs);
        }
    }

    std::vector<T> output(m);
    for (int i = 0; i < m; i++) {
        output[i] = static_cast<T>(result(i));
    }
    return output;
}

// One triangular solve with op applied.  Eigen takes the view mode as a
// compile-time parameter, so the four fill/diag combinations have to be
// spelled out by the caller; applying op in here means they are spelled out
// once rather than once per op.
template <int MODE, typename M, typename V>
inline V SpSVTriSolveC(const M& A, const V& rhs, int op) {
    if (op == 1) return A.template triangularView<MODE>().transpose().solve(rhs);
    if (op == 2) return A.template triangularView<MODE>().adjoint().solve(rhs);
    return A.template triangularView<MODE>().solve(rhs);
}

// complex64 golden.  The structure still comes from CsrMatrix, which is
// shared with the other operators and holds floats; the complex values travel
// beside it.  op is 0 = N, 1 = T, 2 = H -- T and H are the same thing for a
// real matrix but not for this one, so the conjugate case solves against the
// adjoint rather than the transpose.
inline std::vector<std::complex<double>> SpSVGoldenComplex(
    const CsrMatrix& csr,
    const std::vector<std::complex<float>>& values,
    const std::vector<std::complex<float>>& x,
    std::complex<float> alpha,
    bool lower,
    bool unitDiag,
    int op)
{
    using cd = std::complex<double>;
    int m = static_cast<int>(csr.rows);
    if (m <= 0) return {};

    Eigen::SparseMatrix<cd, Eigen::RowMajor> A(m, m);
    std::vector<Eigen::Triplet<cd>> triplets;
    triplets.reserve(static_cast<size_t>(csr.nnz));
    for (int i = 0; i < m; i++) {
        for (int j = static_cast<int>(csr.rowOffsets[i]); j < static_cast<int>(csr.rowOffsets[i + 1]); j++) {
            triplets.emplace_back(i, static_cast<int>(csr.colIndices[j]),
                                  cd(values[j].real(), values[j].imag()));
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::Matrix<cd, Eigen::Dynamic, 1> rhs(m);
    cd a(alpha.real(), alpha.imag());
    for (int i = 0; i < m; i++) {
        rhs(i) = a * cd(x[i].real(), x[i].imag());
    }

    Eigen::Matrix<cd, Eigen::Dynamic, 1> result(m);
    if (lower && !unitDiag) result = SpSVTriSolveC<Eigen::Lower>(A, rhs, op);
    else if (lower)         result = SpSVTriSolveC<Eigen::Lower | Eigen::UnitDiag>(A, rhs, op);
    else if (!unitDiag)     result = SpSVTriSolveC<Eigen::Upper>(A, rhs, op);
    else                    result = SpSVTriSolveC<Eigen::Upper | Eigen::UnitDiag>(A, rhs, op);

    std::vector<cd> output(m);
    for (int i = 0; i < m; i++) output[i] = result(i);
    return output;
}

template <typename T>
std::vector<T> SpSVGoldenManual(
    const CsrMatrix& csr,
    const std::vector<T>& x,
    T alpha,
    bool lower,
    bool unitDiag,
    bool transpose)
{
    int m = static_cast<int>(csr.rows);
    if (m <= 0) return {};

    auto rhs = ComputeRhs(x, alpha);
    std::vector<double> y(m, 0.0);

    bool forward = DetermineForward(lower, transpose);
    if (forward) {
        SolveTriangularLoop<true>(csr, rhs, unitDiag, y);
    } else {
        SolveTriangularLoop<false>(csr, rhs, unitDiag, y);
    }

    std::vector<T> output(m);
    for (int i = 0; i < m; i++) {
        output[i] = static_cast<T>(y[i]);
    }
    return output;
}

}

#endif
