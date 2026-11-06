/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_FILL_H_
#define TEST_FRAME_FILL_H_
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <vector>
namespace sparse_test {

struct CsrMatrix {
    std::vector<int32_t> rowOffsets;
    std::vector<int32_t> colIndices;
    std::vector<float> values;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;

    template <typename T>
    std::vector<T> valuesAs() const {
        std::vector<T> out(values.size());
        for (size_t i = 0; i < values.size(); i++) out[i] = static_cast<T>(values[i]);
        return out;
    }
};

struct CooMatrix {
    std::vector<int32_t> rowIndices;
    std::vector<int32_t> colIndices;
    std::vector<float> values;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
};

struct CscMatrix {
    std::vector<int32_t> colOffsets;
    std::vector<int32_t> rowIndices;
    std::vector<float> values;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
};

class SparseFillGenerator {
public:
    explicit SparseFillGenerator(uint32_t seed = 42) : rng_(seed) {}
    void setSparsity(double sparsity) {
        sparsity_ = std::clamp(sparsity, 0.0, 1.0);
    }
    void setEmptyRowProb(double p) {
        emptyRowProb_ = std::clamp(p, 0.0, 1.0);
    }
    void setValueRange(double lo, double hi) {
        valueLo_ = lo;
        valueHi_ = hi;
    }
    void setSeed(uint32_t seed) {
        rng_.seed(seed);
    }
    CsrMatrix generateCsr(int64_t rows, int64_t cols, int64_t nnzHint = -1) {
        CsrMatrix out;
        out.rows = rows;
        out.cols = cols;
        out.rowOffsets.assign(rows + 1, 0);
        if (rows <= 0 || cols <= 0) return out;
        std::uniform_real_distribution<float> emptyDist(0.0f, 1.0f);
        std::uniform_real_distribution<float> valDist(static_cast<float>(valueLo_), static_cast<float>(valueHi_));
        double density = 1.0 - sparsity_;
        std::binomial_distribution<int> nnzDist(static_cast<int>(cols), density);
        int64_t currentNnz = 0;
        std::vector<int> visited(cols, -1);
        std::vector<int> rowCols;
        for (int64_t i = 0; i < rows; i++) {
            out.rowOffsets[i] = static_cast<int32_t>(currentNnz);
            if (emptyRowProb_ > 0.0 && emptyDist(rng_) < static_cast<float>(emptyRowProb_)) continue;
            int targetNnz = (nnzHint > 0) ? static_cast<int>(nnzHint / std::max<int64_t>(1, rows)) : nnzDist(rng_);
            targetNnz = std::min(targetNnz, static_cast<int>(cols));
            if (targetNnz <= 0) continue;
            int attempt = 0;
            while (static_cast<int>(rowCols.size()) < targetNnz && attempt < targetNnz * 4) {
                std::uniform_int_distribution<int32_t> colDist(0, static_cast<int32_t>(cols) - 1);
                int32_t c = colDist(rng_);
                if (visited[c] != static_cast<int>(i)) {
                    visited[c] = static_cast<int>(i);
                    rowCols.push_back(c);
                }
                attempt++;
            }
            // Repeated draws may exhaust the budget before the row reaches
            // its requested size, especially for dense rows. Complete from
            // the unused columns, sampling them without replacement.
            if (static_cast<int>(rowCols.size()) < targetNnz) {
                const auto sampledNnz = rowCols.size();
                for (int32_t c = 0; c < cols; c++) {
                    if (visited[c] != static_cast<int>(i))
                        rowCols.push_back(c);
                }
                std::shuffle(rowCols.begin() + sampledNnz, rowCols.end(), rng_);
                rowCols.resize(targetNnz);
            }
            std::sort(rowCols.begin(), rowCols.end());
            for (int32_t c : rowCols) {
                out.colIndices.push_back(c);
                out.values.push_back(valDist(rng_));
                currentNnz++;
            }
            rowCols.clear();
        }
        out.rowOffsets[rows] = static_cast<int32_t>(currentNnz);
        out.nnz = currentNnz;
        return out;
    }
    CooMatrix generateCoo(int64_t rows, int64_t cols, int64_t nnzHint) {
        CsrMatrix csr = generateCsr(rows, cols, nnzHint);
        return csrToCoo(csr, rows, cols);
    }
    CscMatrix generateCsc(int64_t rows, int64_t cols, int64_t nnzHint) {
        CsrMatrix csr = generateCsr(rows, cols, nnzHint);
        return csrToCsc(csr, rows, cols);
    }

    template <typename T>
    std::vector<T> generateDenseVector(int64_t size) {
        std::uniform_real_distribution<float> dist(static_cast<float>(valueLo_), static_cast<float>(valueHi_));
        std::vector<T> out(size);
        for (int64_t i = 0; i < size; i++) out[i] = static_cast<T>(dist(rng_));
        return out;
    }
    std::vector<float> generateDenseFloat(int64_t size) {
        return generateDenseVector<float>(size);
    }

    template <typename T>
    std::vector<T> generateConstant(int64_t size, T val) {
        return std::vector<T>(size, val);
    }

private:
    std::mt19937 rng_;
    double sparsity_ = 0.9;
    double emptyRowProb_ = 0.0;
    double valueLo_ = -2.0;
    double valueHi_ = 2.0;
    CooMatrix csrToCoo(const CsrMatrix& csr, int64_t rows, int64_t cols) {
        CooMatrix out;
        out.rows = rows;
        out.cols = cols;
        out.nnz = csr.nnz;
        out.rowIndices.reserve(csr.nnz);
        out.colIndices.reserve(csr.nnz);
        out.values.reserve(csr.nnz);
        for (int64_t i = 0; i < rows; i++) {
            for (int32_t j = csr.rowOffsets[i]; j < csr.rowOffsets[i + 1]; j++) {
                out.rowIndices.push_back(static_cast<int32_t>(i));
                out.colIndices.push_back(csr.colIndices[j]);
                out.values.push_back(csr.values[j]);
            }
        }
        return out;
    }
    CscMatrix csrToCsc(const CsrMatrix& csr, int64_t rows, int64_t cols) {
        CscMatrix out;
        out.rows = rows;
        out.cols = cols;
        out.nnz = csr.nnz;
        out.colOffsets.assign(cols + 1, 0);
        out.rowIndices.assign(csr.nnz, 0);
        out.values.assign(csr.nnz, 0.0f);
        for (int32_t c : csr.colIndices) out.colOffsets[c + 1]++;
        for (int64_t c = 0; c < cols; c++) out.colOffsets[c + 1] += out.colOffsets[c];
        for (int64_t i = 0; i < rows; i++) {
            for (int32_t j = csr.rowOffsets[i]; j < csr.rowOffsets[i + 1]; j++) {
                int32_t c = csr.colIndices[j];
                int32_t dest = out.colOffsets[c];
                out.rowIndices[dest] = static_cast<int32_t>(i);
                out.values[dest] = csr.values[j];
                out.colOffsets[c]++;
            }
        }
        for (int64_t c = cols; c > 0; c--) out.colOffsets[c] = out.colOffsets[c - 1];
        out.colOffsets[0] = 0;
        return out;
    }
};
inline CsrMatrix makeSparseCsr(int64_t rows, int64_t cols, double sparsity = 0.9, uint32_t seed = 42) {
    SparseFillGenerator gen(seed);
    gen.setSparsity(sparsity);
    return gen.generateCsr(rows, cols);
}
inline CooMatrix makeSparseCoo(int64_t rows, int64_t cols, double sparsity = 0.9, uint32_t seed = 42) {
    SparseFillGenerator gen(seed);
    gen.setSparsity(sparsity);
    return gen.generateCoo(rows, cols, -1);
}
inline CscMatrix makeSparseCsc(int64_t rows, int64_t cols, double sparsity = 0.9, uint32_t seed = 42) {
    SparseFillGenerator gen(seed);
    gen.setSparsity(sparsity);
    return gen.generateCsc(rows, cols, -1);
}
inline std::vector<float> makeDenseFloat(int64_t size, double lo = -2.0, double hi = 2.0, uint32_t seed = 42) {
    SparseFillGenerator gen(seed);
    gen.setValueRange(lo, hi);
    return gen.generateDenseFloat(size);
}

template <typename T>
inline std::vector<T> makeDense(int64_t size, double lo = -2.0, double hi = 2.0, uint32_t seed = 42) {
    SparseFillGenerator gen(seed);
    gen.setValueRange(lo, hi);
    return gen.generateDenseVector<T>(size);
}
inline CsrMatrix makeDiagCsr(int64_t n, float diagValue = 1.0f) {
    CsrMatrix out;
    out.rows = n;
    out.cols = n;
    out.nnz = n;
    out.rowOffsets.resize(n + 1);
    out.colIndices.resize(n);
    out.values.resize(n, diagValue);
    for (int64_t i = 0; i < n; i++) {
        out.rowOffsets[i] = static_cast<int32_t>(i);
        out.colIndices[i] = static_cast<int32_t>(i);
    }
    out.rowOffsets[n] = static_cast<int32_t>(n);
    return out;
}
inline CsrMatrix makeEmptyCsr(int64_t rows, int64_t cols) {
    CsrMatrix out;
    out.rows = rows;
    out.cols = cols;
    out.nnz = 0;
    out.rowOffsets.assign(rows + 1, 0);
    return out;
}
// Dense column-major matrix generators
inline std::vector<float> makeDenseColMajor(int m, int n, int lda,
                                             double density, uint32_t seed = 42,
                                             double lo = -5.0, double hi = 5.0) {
    std::vector<float> A(static_cast<int64_t>(lda) * n, 0.0f);
    if (m <= 0 || n <= 0) return A;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> valDist(static_cast<float>(lo), static_cast<float>(hi));
    std::uniform_real_distribution<float> zeroDist(0.0f, 1.0f);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (zeroDist(rng) < static_cast<float>(density)) {
                float v = valDist(rng);
                if (v == 0.0f) v = 1.0f;
                A[static_cast<int64_t>(j) * lda + i] = v;
            }
        }
    }
    return A;
}
inline std::vector<float> makeZeroColMajor(int m, int n, int lda) {
    return std::vector<float>(static_cast<int64_t>(lda) * n, 0.0f);
}
inline std::vector<float> makeFullColMajor(int m, int n, int lda, uint32_t seed = 42,
                                            double lo = 1.0, double hi = 10.0) {
    std::vector<float> A(static_cast<int64_t>(lda) * n, 0.0f);
    if (m <= 0 || n <= 0) return A;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> valDist(static_cast<float>(lo), static_cast<float>(hi));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A[static_cast<int64_t>(j) * lda + i] = valDist(rng);
        }
    }
    return A;
}
inline std::vector<float> makeDiagColMajor(int m, int n, int lda, uint32_t seed = 42,
                                            double lo = 1.0, double hi = 10.0,
                                            double offDiagDensity = 0.05) {
    std::vector<float> A(static_cast<int64_t>(lda) * n, 0.0f);
    if (m <= 0 || n <= 0) return A;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> valDist(static_cast<float>(lo), static_cast<float>(hi));
    std::uniform_real_distribution<float> zeroDist(0.0f, 1.0f);
    int minDim = std::min(m, n);
    for (int i = 0; i < minDim; i++) {
        A[static_cast<int64_t>(i) * lda + i] = valDist(rng);
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (i == j) continue;
            if (zeroDist(rng) < static_cast<float>(offDiagDensity)) {
                float v = valDist(rng);
                if (v == 0.0f) v = 1.0f;
                A[static_cast<int64_t>(j) * lda + i] = v;
            }
        }
    }
    return A;
}
inline std::vector<float> makeExtremelySparseColMajor(int m, int n, int lda,
                                                       uint32_t seed = 42, float value = 42.0f) {
    std::vector<float> A(static_cast<int64_t>(lda) * n, 0.0f);
    if (m <= 0 || n <= 0) return A;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> rowDist(0, m - 1);
    std::uniform_int_distribution<int> colDist(0, n - 1);
    int r = rowDist(rng);
    int c = colDist(rng);
    A[static_cast<int64_t>(c) * lda + r] = value;
    return A;
}
// Tridiagonal matrix data generators (row-major interleaved layout)
// Layout: data[row * batchCount + batch]

struct TridiagMatrix {
    std::vector<float> dl;       // [m * batchCount] sub-diagonal (dl[0][*]=0)
    std::vector<float> d;        // [m * batchCount] main diagonal
    std::vector<float> du;       // [m * batchCount] super-diagonal (du[m-1][*]=0)
    int m = 0;
    int batchCount = 0;
};
inline TridiagMatrix makeDiagDominantTridiag(int m, int batchCount, uint32_t seed) {
    TridiagMatrix out;
    out.m = m;
    out.batchCount = batchCount;
    int total = m * batchCount;
    out.dl.resize(total, 0.0f);
    out.d.resize(total, 0.0f);
    out.du.resize(total, 0.0f);
    if (m <= 0 || batchCount <= 0) return out;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> offDist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> diagDist(8.0f, 12.0f);
    for (int j = 0; j < batchCount; j++) {
        for (int i = 0; i < m; i++) {
            int idx = i * batchCount + j;
            out.d[idx] = diagDist(rng);
            if (i > 0)     out.dl[idx] = offDist(rng);         // dl[0][*]=0
            if (i < m - 1) out.du[idx] = offDist(rng);         // du[m-1][*]=0
        }
    }
    return out;
}
inline TridiagMatrix makeRandomTridiag(int m, int batchCount, double lo, double hi, uint32_t seed) {
    TridiagMatrix out;
    out.m = m;
    out.batchCount = batchCount;
    int total = m * batchCount;
    out.dl.resize(total, 0.0f);
    out.d.resize(total, 0.0f);
    out.du.resize(total, 0.0f);
    if (m <= 0 || batchCount <= 0) return out;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(static_cast<float>(lo), static_cast<float>(hi));
    for (int j = 0; j < batchCount; j++) {
        for (int i = 0; i < m; i++) {
            int idx = i * batchCount + j;
            out.d[idx] = dist(rng);
            if (i > 0)     out.dl[idx] = dist(rng);            // dl[0][*]=0
            if (i < m - 1) out.du[idx] = dist(rng);            // du[m-1][*]=0
        }
    }
    return out;
}
inline TridiagMatrix makeConstantDiagTridiag(int m, int batchCount, float diagVal) {
    TridiagMatrix out;
    out.m = m; out.batchCount = batchCount; int total = m * batchCount;
    out.dl.resize(total, 0.0f); out.d.resize(total, diagVal); out.du.resize(total, 0.0f);
    return out;
}
inline TridiagMatrix makeIdentityTridiag(int m, int batchCount) {
    return makeConstantDiagTridiag(m, batchCount, 1.0f);
}
inline std::vector<float> makeTridiagRHS(int m, int batchCount, double lo, double hi, uint32_t seed) {
    std::vector<float> b(m * batchCount, 0.0f);
    if (m <= 0 || batchCount <= 0) return b;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(static_cast<float>(lo), static_cast<float>(hi));
    for (int j = 0; j < batchCount; j++) {
        for (int i = 0; i < m; i++) {
            b[i * batchCount + j] = dist(rng);
        }
    }
    return b;
}
// makePentadiagRHS shares the exact same interleaved-layout RNG pattern
// as makeTridiagRHS, so it is implemented as an alias.
inline std::vector<float> makePentadiagRHS(int m, int bc, double lo, double hi, uint32_t seed) {
    return makeTridiagRHS(m, bc, lo, hi, seed);
}
inline std::vector<float> makeKnownSolutionTridiag(
    const TridiagMatrix& tri,
    const std::vector<float>& xTrue)
{
    // b = A * xTrue  (tridiagonal multiply)
    int m = tri.m;
    int batchCount = tri.batchCount;
    std::vector<float> b(m * batchCount, 0.0f);
    for (int j = 0; j < batchCount; j++) {
        for (int i = 0; i < m; i++) {
            int idx = i * batchCount + j;
            float val = tri.d[idx] * xTrue[idx];
            if (i > 0)     val += tri.dl[idx] * xTrue[(i - 1) * batchCount + j];
            if (i < m - 1) val += tri.du[idx] * xTrue[(i + 1) * batchCount + j];
            b[idx] = val;
        }
    }
    return b;
}
// Pentadiagonal matrix data generators (row-major interleaved layout)
// Layout: data[row * batchCount + batch]
// Boundary conditions: ds[0][*]=ds[1][*]=0, dl[0][*]=0,
//                      du[m-1][*]=0, dw[m-2][*]=dw[m-1][*]=0

struct PentadiagMatrix {
    std::vector<float> ds;     // [m * batchCount] distance-2 sub-diagonal
    std::vector<float> dl;     // [m * batchCount] sub-diagonal
    std::vector<float> d;      // [m * batchCount] main diagonal
    std::vector<float> du;     // [m * batchCount] super-diagonal
    std::vector<float> dw;     // [m * batchCount] distance-2 super-diagonal
    int m = 0;
    int batchCount = 0;
};
inline PentadiagMatrix makeZeroPentadiag(int m, int batchCount) {
    PentadiagMatrix out;
    out.m = m; out.batchCount = batchCount;
    int total = m * batchCount;
    out.ds.resize(total, 0.0f); out.dl.resize(total, 0.0f);
    out.d.resize(total, 0.0f);  out.du.resize(total, 0.0f); out.dw.resize(total, 0.0f);
    return out;
}
inline PentadiagMatrix makeDiagDominantPentadiag(int m, int batchCount, uint32_t seed,
                                                   double diagLo = 8.0, double diagHi = 12.0) {
    PentadiagMatrix out = makeZeroPentadiag(m, batchCount);
    if (m <= 0 || batchCount <= 0) return out;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> offDist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> diagDist(static_cast<float>(diagLo), static_cast<float>(diagHi));
    for (int j = 0; j < batchCount; j++) {
        for (int i = 0; i < m; i++) {
            int idx = i * batchCount + j;
            out.d[idx] = diagDist(rng);
            if (i >= 2)     out.ds[idx] = offDist(rng);         // ds[0..1][*]=0
            if (i >= 1)     out.dl[idx] = offDist(rng);         // dl[0][*]=0
            if (i < m - 1)  out.du[idx] = offDist(rng);         // du[m-1][*]=0
            if (i < m - 2)  out.dw[idx] = offDist(rng);         // dw[m-2..m-1][*]=0
        }
    }
    return out;
}
inline PentadiagMatrix makeRandomPentadiag(int m, int batchCount, double lo, double hi, uint32_t seed) {
    PentadiagMatrix out = makeZeroPentadiag(m, batchCount);
    if (m <= 0 || batchCount <= 0) return out;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(static_cast<float>(lo), static_cast<float>(hi));
    for (int j = 0; j < batchCount; j++) {
        for (int i = 0; i < m; i++) {
            int idx = i * batchCount + j;
            out.d[idx] = dist(rng);
            if (i >= 2)     out.ds[idx] = dist(rng);
            if (i >= 1)     out.dl[idx] = dist(rng);
            if (i < m - 1)  out.du[idx] = dist(rng);
            if (i < m - 2)  out.dw[idx] = dist(rng);
        }
    }
    return out;
}
inline PentadiagMatrix makeIdentityPentadiag(int m, int batchCount) {
    PentadiagMatrix out = makeZeroPentadiag(m, batchCount);
    if (m <= 0 || batchCount <= 0) return out;
    std::fill(out.d.begin(), out.d.end(), 1.0f);
    return out;
}
inline PentadiagMatrix makeConstantDiagPentadiag(int m, int batchCount, float diagVal) {
    PentadiagMatrix out = makeZeroPentadiag(m, batchCount);
    if (m <= 0 || batchCount <= 0) return out;
    std::fill(out.d.begin(), out.d.end(), diagVal);
    return out;
}
inline std::vector<float> makeKnownSolutionPentadiag(
    const PentadiagMatrix& pent,
    const std::vector<float>& xTrue)
{
    // b = A * xTrue  (pentadiagonal multiply)
    int m = pent.m;
    int batchCount = pent.batchCount;
    std::vector<float> b(m * batchCount, 0.0f);
    for (int j = 0; j < batchCount; j++) {
        for (int i = 0; i < m; i++) {
            int idx = i * batchCount + j;
            float val = pent.d[idx] * xTrue[idx];
            if (i >= 2)     val += pent.ds[idx] * xTrue[(i - 2) * batchCount + j];
            if (i >= 1)     val += pent.dl[idx] * xTrue[(i - 1) * batchCount + j];
            if (i < m - 1)  val += pent.du[idx] * xTrue[(i + 1) * batchCount + j];
            if (i < m - 2)  val += pent.dw[idx] * xTrue[(i + 2) * batchCount + j];
            b[idx] = val;
        }
    }
    return b;
}
}

#include "fill_sparse.h"

#endif
