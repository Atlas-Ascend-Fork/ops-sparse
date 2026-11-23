/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUBE_SPMM_1D_GOLDEN_H_
#define CUBE_SPMM_1D_GOLDEN_H_
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <map>
#include <random>
#include <vector>
#include "cube_spmm_1d_param.h"

inline uint16_t CubeHalf(double value)
{
    Eigen::half h(static_cast<float>(value));
    uint16_t bits = 0;
    static_assert(sizeof(h) == sizeof(bits), "binary16 storage");
    const auto* source = reinterpret_cast<const unsigned char*>(&h);
    auto* destination = reinterpret_cast<unsigned char*>(&bits);
    std::copy_n(source, sizeof(bits), destination);
    return bits;
}
inline double CubeFloat(uint16_t bits)
{
    Eigen::half h;
    const auto* source = reinterpret_cast<const unsigned char*>(&bits);
    auto* destination = reinterpret_cast<unsigned char*>(&h);
    std::copy_n(source, sizeof(bits), destination);
    return static_cast<float>(h);
}
struct CubeInput {
    std::vector<int64_t> offsets{0}, columns;
    std::vector<uint16_t> values;
};
inline CubeInput CubeMakeInput(const CubeSpmm1dParam& p)
{
    CubeInput a;
    std::mt19937 gen(p.seed);
    for (int row = 0; row < p.m; ++row) {
        const int limit = p.mode == 6 ? p.k : (p.mode == 7 ? 40 : 7);
        const int count = p.mode == 0 || (p.mode == 1 && row / 16 % 2 == 0) ? 0 : std::min(p.k, limit);
        for (int j = 0; j < count; ++j) {
            // modes 3/4 exercise row-dependent patterns and unsorted duplicate entries.
            int col = (j * 13 + (p.mode >= 3 ? row * 11 : row / 16 * 7)) % p.k;
            if (p.mode == 6) {
                col = j;
            } else if (p.mode == 7) {
                col = (j + row % 16 * count + row / 16 * 7) % p.k;
            }
            a.columns.push_back(col);
            const double value = p.mode >= 6 ? (1 + int(gen() % 8)) * 0.0625 :
                                 (int(gen() % 17) - 8) * (p.mode == 5 ? 0.1 : 0.0625);
            a.values.push_back(CubeHalf(value));
            if (p.mode == 4) {
                a.columns.push_back(col);
                a.values.push_back(CubeHalf(0.125));
            }
        }
        a.offsets.push_back(a.columns.size());
    }
    return a;
}
inline Eigen::SparseMatrix<double, Eigen::RowMajor> CubeCanonical(const CubeInput& a, int m, int k)
{
    std::vector<Eigen::Triplet<double>> entries;
    for (int row = 0; row < m; ++row) {
        std::map<int64_t, double> canonical;
        for (int64_t i = a.offsets[row]; i < a.offsets[row + 1]; ++i) {
            canonical[a.columns[i]] += CubeFloat(a.values[i]);
        }
        for (auto entry : canonical) {
            const double value = CubeFloat(CubeHalf(entry.second));
            if (value != 0) {
                entries.emplace_back(row, entry.first, value);
            }
        }
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor> out(m, k);
    out.setFromTriplets(entries.begin(), entries.end());
    return out;
}
inline std::vector<double> CubeGolden(const CubeInput& a, int m, int n, int k, const std::vector<uint16_t>& b,
                                      int paddedN)
{
    const auto canonical = CubeCanonical(a, m, k);
    std::vector<double> out(size_t(m) * n);
    // Bound the double-precision temporary even when logical FP16 B exceeds L2.
    for (int first = 0; first < n; first += 64) {
        const int width = std::min(64, n - first);
        Eigen::MatrixXd dense(k, width);
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < width; ++j) {
                dense(i, j) = CubeFloat(b[size_t(i) * paddedN + first + j]);
            }
        }
        const Eigen::MatrixXd result = canonical * dense;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < width; ++j) {
                out[size_t(i) * n + first + j] = result(i, j);
            }
        }
    }
    return out;
}
#endif
