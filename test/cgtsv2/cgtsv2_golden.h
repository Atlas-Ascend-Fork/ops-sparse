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
#ifndef TEST_CGTSV2_GOLDEN_H_
#define TEST_CGTSV2_GOLDEN_H_

#include <complex>
#include <cmath>
#include <cstddef>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>
#include "cann_ops_sparse.h"

namespace cgtsv2_test {
using Complex = aclsparseComplex_t;
using High = std::complex<double>;
inline High Promote(Complex x) { return {x.x, x.y}; }

struct Input {
    int m;
    int n;
    int ldb;
    std::vector<Complex> dl;
    std::vector<Complex> d;
    std::vector<Complex> du;
    std::vector<Complex> b;
};

// Near the Jacobi admission boundary, construct bounded RHS with strong
// cancellation and tiny solution components. Golden solves the actual rounded
// FP32 inputs; manufactured targets are only used to generate those inputs.
inline Input MakeJacobiBoundaryInput(int m, int n, int ldb, const std::string &kind)
{
    float ratio = 0.0f;
    if (kind == "jacobi_boundary_126") { ratio = 1.26f; }
    else if (kind == "jacobi_boundary_135") { ratio = 1.35f; }
    else if (kind == "jacobi_boundary_136") { ratio = 1.36f; }
    else if (kind == "jacobi_boundary_150") { ratio = 1.50f; }
    else if (kind == "jacobi_boundary_200") { ratio = 2.00f; }
    else if (kind == "jacobi_boundary_201") { ratio = 2.01f; }
    else { throw std::invalid_argument("unknown Jacobi boundary fixture: " + kind); }
    const float lower = 1.0f / ratio;
    Input x {m, n, ldb, std::vector<Complex>(m, {lower, 0.0f}),
        std::vector<Complex>(m, {1.0f, 0.0f}), std::vector<Complex>(m, {0.0f, 0.0f}),
        std::vector<Complex>(static_cast<size_t>(ldb) * n, {-123.25f, 67.5f})};
    x.dl.front() = {0.0f, 0.0f};
    for (int column = 0; column < n; ++column) {
        High previous{};
        const double scale = column % 2 == 0 ? 1.0 : 0.5;
        for (int row = 0; row < m; ++row) {
            const double sign = row % 2 == 0 ? 1.0 : -1.0;
            const High target = row % 257 == 128 ? High{0x1p-20, -0x1p-21} : High{6 * sign, -3 * sign};
            const High value = scale * target;
            const High rhs = value + static_cast<double>(lower) * previous;
            x.b[static_cast<size_t>(column) * ldb + row] = {
                static_cast<float>(rhs.real()), static_cast<float>(rhs.imag())};
            previous = value;
        }
    }
    return x;
}

// Supplemental deterministic ST fixtures. These never replace official ATK inputs.
inline Input MakeInput(int m, int n, int ldb, const std::string &kind)
{
    if (kind.rfind("jacobi_boundary_", 0) == 0) { return MakeJacobiBoundaryInput(m, n, ldb, kind); }
    Input x {m, n, ldb, std::vector<Complex>(m, {-0.3f, 0.1f}),
        std::vector<Complex>(m, {4.0f, 0.25f}), std::vector<Complex>(m, {0.2f, -0.15f}),
        std::vector<Complex>(static_cast<size_t>(ldb) * n, {-123.25f, 67.5f})};
    x.dl.front() = {0.0f, 0.0f};
    x.du.back() = {0.0f, 0.0f};
    if (kind == "pivot") {
        for (int i = 0; i < m; ++i) {
            x.d[i] = {0.1f, 0.05f};
            if (i) { x.dl[i] = {2.0f, 0.25f}; }
            if (i + 1 < m) { x.du[i] = {1.0f, -0.125f}; }
        }
    } else if (kind == "zero_pivot") {
        x.d[0] = {0.0f, 0.0f};
        x.dl[1] = {1.4f, -0.2f};
        x.du[0] = {0.8f, 0.3f};
    } else if (kind == "modulus_tie") {
        x.d[0] = {3.0f, 4.0f};
        x.dl[1] = {0.0f, 5.0f};
    } else if (kind == "imaginary") {
        for (auto &v : x.d) { v = {0.0f, 4.0f}; }
    } else if (kind == "scaled_large" || kind == "scaled_tiny") {
        const float scale = kind == "scaled_large" ? 1e30f : 1e-30f;
        for (int i = 0; i < m; ++i) {
            x.d[i] = {scale, scale};
            x.dl[i] = x.du[i] = {0.0f, 0.0f};
        }
    } else if (kind != "dominant") {
        throw std::invalid_argument("unknown Cgtsv2 ST fixture: " + kind);
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            x.b[static_cast<size_t>(j) * ldb + i] = {
                static_cast<float>((i * 7 + j * 3) % 17 - 8) / 8.0f,
                static_cast<float>((i * 3 + j * 5) % 13 - 6) / 8.0f};
            if (kind == "scaled_large" || kind == "scaled_tiny") {
                const float scale = kind == "scaled_large" ? 1e30f : 1e-30f;
                auto &v = x.b[static_cast<size_t>(j) * ldb + i];
                v.x *= scale;
                v.y *= scale;
            }
        }
    }
    return x;
}

// Independent host complex128 tridiagonal Gaussian elimination with partial
// pivoting. No device arithmetic or optional third-party library is reused.
// Official acceptance still uses the supplied SciPy/LAPACK Golden unchanged.
inline std::vector<High> Golden(const Input &x)
{
    std::vector<High> result;
    result.reserve(x.b.size());
    for (const auto &v : x.b) { result.push_back(Promote(v)); }
    if (!x.n) { return result; }
    std::vector<High> diagonal(x.m), upper(x.m), lower(x.m), second(x.m);
    std::vector<bool> swapped(x.m, false);
    for (int i = 0; i < x.m; ++i) {
        diagonal[i] = Promote(x.d[i]);
        upper[i] = Promote(x.du[i]);
        lower[i] = Promote(x.dl[i]);
    }
    for (int i = 0; i + 1 < x.m; ++i) {
        if (std::abs(diagonal[i]) >= std::abs(lower[i + 1])) {
            if (diagonal[i] == High{}) { throw std::runtime_error("ST Golden singular matrix"); }
            lower[i + 1] /= diagonal[i];
            diagonal[i + 1] -= lower[i + 1] * upper[i];
        } else {
            const High multiplier = diagonal[i] / lower[i + 1];
            diagonal[i] = lower[i + 1];
            lower[i + 1] = multiplier;
            const High nextDiagonal = diagonal[i + 1];
            diagonal[i + 1] = upper[i] - multiplier * nextDiagonal;
            upper[i] = nextDiagonal;
            if (i + 2 < x.m) {
                second[i] = upper[i + 1];
                upper[i + 1] *= -multiplier;
            }
            swapped[i] = true;
        }
    }
    if (diagonal.back() == High{}) { throw std::runtime_error("ST Golden singular matrix"); }
    for (int j = 0; j < x.n; ++j) {
        const size_t offset = static_cast<size_t>(j) * x.ldb;
        for (int i = 0; i + 1 < x.m; ++i) {
            if (swapped[i]) { std::swap(result[offset + i], result[offset + i + 1]); }
            result[offset + i + 1] -= lower[i + 1] * result[offset + i];
        }
        result[offset + x.m - 1] /= diagonal.back();
        for (int i = x.m - 2; i >= 0; --i) {
            High numerator = result[offset + i] - upper[i] * result[offset + i + 1];
            if (i + 2 < x.m) { numerator -= second[i] * result[offset + i + 2]; }
            result[offset + i] = numerator / diagonal[i];
        }
    }
    return result;
}
// Independent host reference: adjacent-row partial pivoting with an explicit
// second upper diagonal. Storage and work are O(m*n), without a dense matrix.
inline std::vector<High> SolveGolden(int m, int n, int ldb,
    const std::vector<High>& lower, const std::vector<High>& diagonal,
    const std::vector<High>& upper, const std::vector<High>& b)
{
    if (m < 3 || n < 0 || ldb < m || lower.size() != static_cast<size_t>(m) ||
        diagonal.size() != static_cast<size_t>(m) || upper.size() != static_cast<size_t>(m) ||
        b.size() != static_cast<size_t>(ldb) * n) {
        throw std::invalid_argument("invalid Cgtsv2 Golden input");
    }
    std::vector<High> x(static_cast<size_t>(m) * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            x[static_cast<size_t>(j) * m + i] = b[static_cast<size_t>(j) * ldb + i];
        }
    }
    if (n == 0) {
        return x;
    }
    auto d = diagonal;
    auto u = upper;
    std::vector<High> second(m, High{});
    for (int i = 0; i < m - 1; ++i) {
        const High candidate = lower[i + 1];
        if (std::abs(d[i]) >= std::abs(candidate)) {
            if (d[i] == High{}) {
                throw std::runtime_error("singular Cgtsv2 Golden");
            }
            const High factor = candidate / d[i];
            d[i + 1] -= factor * u[i];
            for (int j = 0; j < n; ++j) {
                const size_t offset = static_cast<size_t>(j) * m + i;
                x[offset + 1] -= factor * x[offset];
            }
        } else {
            const High factor = d[i] / candidate;
            const High nextDiagonal = d[i + 1];
            d[i] = candidate;
            d[i + 1] = u[i] - factor * nextDiagonal;
            u[i] = nextDiagonal;
            if (i + 2 < m) {
                second[i] = u[i + 1];
                u[i + 1] *= -factor;
            }
            for (int j = 0; j < n; ++j) {
                const size_t offset = static_cast<size_t>(j) * m + i;
                const High old = x[offset];
                x[offset] = x[offset + 1];
                x[offset + 1] = old - factor * x[offset + 1];
            }
        }
    }
    if (d.back() == High{}) {
        throw std::runtime_error("singular Cgtsv2 Golden");
    }
    for (int j = 0; j < n; ++j) {
        const size_t offset = static_cast<size_t>(j) * m;
        x[offset + m - 1] /= d[m - 1];
        x[offset + m - 2] = (x[offset + m - 2] - u[m - 2] * x[offset + m - 1]) / d[m - 2];
        for (int i = m - 3; i >= 0; --i) {
            x[offset + i] = (x[offset + i] - u[i] * x[offset + i + 1] -
                second[i] * x[offset + i + 2]) / d[i];
        }
    }
    return x;
}

} // namespace cgtsv2_test
#endif
