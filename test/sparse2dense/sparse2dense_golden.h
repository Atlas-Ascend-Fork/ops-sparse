/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SPARSE2DENSE_GOLDEN_H_
#define TEST_SPARSE2DENSE_GOLDEN_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SPARSE_TEST_USE_EIGEN
#include <Eigen/Sparse>
#endif

namespace sparse_test {

struct CsrData {
    std::vector<int32_t> rowOff;
    std::vector<int32_t> colInd;
    std::vector<uint8_t> values;
    int64_t nnz = 0;
};

struct CscData {
    std::vector<int32_t> colOff;
    std::vector<int32_t> rowInd;
    std::vector<uint8_t> values;
    int64_t nnz = 0;
};

struct CooData {
    std::vector<int32_t> rowInd;
    std::vector<int32_t> colInd;
    std::vector<uint8_t> values;
    int64_t nnz = 0;
};

inline size_t Sparse2DenseElemWidth(aclDataType type) {
    if (type == ACL_INT8) return 1;
    if (type == ACL_FLOAT16 || type == ACL_BF16) return 2;
    if (type == ACL_FLOAT || type == ACL_INT32) return 4;
    if (type == ACL_COMPLEX64) return 8;
    return 0;
}

inline uint32_t Sparse2DenseInputBits(aclDataType type, uint64_t linear,
                                       uint64_t seed) {
    if (type == ACL_INT8) {
        uint8_t bits =
            static_cast<int8_t>((linear + seed) % 255 - 127);
        return bits == 0 ? 1 : bits;
    }
    if (type == ACL_FLOAT)
        return 0x3f000001u + static_cast<uint32_t>(linear % 127);
    if (type == ACL_FLOAT16)
        return 0x3801u + static_cast<uint16_t>(linear % 63);
    if (type == ACL_INT32)
        return static_cast<uint32_t>(linear + seed + 1);
    if (type == ACL_COMPLEX64) {
        // 仅返回实部 pattern；GenerateCsrGolden 对 width=8 按字节展开时需特殊处理
        return 0x3f000001u + static_cast<uint32_t>(linear % 127);
    }
    return 0x3f01u + static_cast<uint16_t>(linear % 63);
}

inline bool Sparse2DenseIsNonzero(const std::string &dist, uint64_t linear,
                                   uint64_t logical, uint64_t seed) {
    if (dist == "Z100" || dist == "EMPTY")
        return false;
    if (dist == "Z99" || dist == "SINGLE")
        return linear == seed % std::max<uint64_t>(logical, 1);
    if (dist == "Z80")
        return linear % 5 == 0;
    if (dist == "Z75")
        return linear % 4 == 0;
    if (dist == "Z50")
        return linear % 2 == 0;
    if (dist == "FULL")
        return true;
    return linear % 4 != 0;
}

inline size_t DensePosition(int64_t row, int64_t col, int64_t ld,
                             bool rowMajor) {
    return static_cast<size_t>(rowMajor ? row * ld + col : col * ld + row);
}

inline CsrData GenerateCsrGolden(int64_t m, int64_t n, int64_t ld,
                                  bool rowMajor, int64_t base,
                                  aclDataType valueType,
                                  const std::string &dist, uint32_t seed) {
    const size_t width = Sparse2DenseElemWidth(valueType);
    if (width == 0) throw std::invalid_argument("unsupported value type");
    CsrData c;
    c.rowOff.assign(static_cast<size_t>(m) + 1, static_cast<int32_t>(base));
    const uint64_t logical = static_cast<uint64_t>(m) * n;
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            const uint64_t linear = static_cast<uint64_t>(row) * n + col;
            if (!Sparse2DenseIsNonzero(dist, linear, logical, seed))
                continue;
            c.colInd.push_back(static_cast<int32_t>(col + base));
            if (valueType == ACL_COMPLEX64) {
                const uint32_t realBits =
                    0x3f000001u + static_cast<uint32_t>(linear % 127);
                const uint32_t imagBits =
                    0x3e800001u + static_cast<uint32_t>((linear + seed) % 127);
                for (size_t b = 0; b < 4; ++b)
                    c.values.push_back(
                        static_cast<uint8_t>((realBits >> (b * 8)) & 0xffu));
                for (size_t b = 0; b < 4; ++b)
                    c.values.push_back(
                        static_cast<uint8_t>((imagBits >> (b * 8)) & 0xffu));
            } else {
                const uint32_t bits =
                    Sparse2DenseInputBits(valueType, linear, seed);
                for (size_t b = 0; b < width; ++b)
                    c.values.push_back(
                        static_cast<uint8_t>((bits >> (b * 8)) & 0xffu));
            }
            ++c.nnz;
        }
        c.rowOff[static_cast<size_t>(row + 1)] =
            static_cast<int32_t>(c.nnz + base);
    }
    return c;
}

inline CscData CsrToCscGolden(const CsrData &csr, int64_t m, int64_t n,
                               int64_t base, size_t width) {
    CscData csc;
    csc.nnz = csr.nnz;
    // colOff starts at 0 (counts per column), then prefix-sum adds base at the end.
    csc.colOff.assign(static_cast<size_t>(n) + 1, 0);
    for (int64_t row = 0; row < m; ++row) {
        for (int32_t p = csr.rowOff[row] - static_cast<int32_t>(base);
             p < csr.rowOff[row + 1] - static_cast<int32_t>(base); ++p) {
            int32_t col = csr.colInd[p] - static_cast<int32_t>(base);
            ++csc.colOff[col + 1];
        }
    }
    for (int64_t i = 0; i < n; ++i)
        csc.colOff[i + 1] += csc.colOff[i];
    for (int64_t i = 0; i <= n; ++i)
        csc.colOff[i] += static_cast<int32_t>(base);
    csc.rowInd.resize(csr.nnz);
    csc.values.resize(csr.nnz * width);
    std::vector<int32_t> cursor(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < n; ++i)
        cursor[i] = csc.colOff[i] - static_cast<int32_t>(base);
    for (int64_t row = 0; row < m; ++row) {
        for (int32_t p = csr.rowOff[row] - static_cast<int32_t>(base);
             p < csr.rowOff[row + 1] - static_cast<int32_t>(base); ++p) {
            int32_t col = csr.colInd[p] - static_cast<int32_t>(base);
            int32_t pos = cursor[col]++;
            csc.rowInd[pos] = static_cast<int32_t>(row + base);
            std::copy_n(csr.values.begin() + p * width, width,
                        csc.values.begin() + pos * width);
        }
    }
    return csc;
}

inline CooData CsrToCooGolden(const CsrData &csr, int64_t m, int64_t base) {
    CooData coo;
    coo.nnz = csr.nnz;
    coo.values = csr.values;
    coo.colInd = csr.colInd;
    coo.rowInd.resize(csr.nnz);
    for (int64_t row = 0; row < m; ++row) {
        for (int32_t p = csr.rowOff[row] - static_cast<int32_t>(base);
             p < csr.rowOff[row + 1] - static_cast<int32_t>(base); ++p) {
            coo.rowInd[p] = static_cast<int32_t>(row + base);
        }
    }
    return coo;
}

struct Sparse2DenseGoldenResult {
    std::vector<uint8_t> dense;
};

inline void ScatterCsrToDense(const CsrData &csr, int64_t m, int64_t base,
                               size_t width, int64_t ld, bool rowMajor,
                               Sparse2DenseGoldenResult &result)
{
    auto writeDense = [&](int64_t row, int64_t col, const uint8_t *val) {
        const size_t pos = DensePosition(row, col, ld, rowMajor);
        std::copy_n(val, width, result.dense.begin() + pos * width);
    };
    for (int64_t row = 0; row < m; ++row) {
        for (int32_t p = csr.rowOff[row] - static_cast<int32_t>(base);
             p < csr.rowOff[row + 1] - static_cast<int32_t>(base); ++p) {
            int32_t col = csr.colInd[p] - static_cast<int32_t>(base);
            writeDense(row, col, csr.values.data() + p * width);
        }
    }
}

inline void ScatterCscToDense(const CscData &csc, int64_t n, int64_t base,
                               size_t width, int64_t ld, bool rowMajor,
                               Sparse2DenseGoldenResult &result)
{
    auto writeDense = [&](int64_t row, int64_t col, const uint8_t *val) {
        const size_t pos = DensePosition(row, col, ld, rowMajor);
        std::copy_n(val, width, result.dense.begin() + pos * width);
    };
    for (int64_t col = 0; col < n; ++col) {
        for (int32_t p = csc.colOff[col] - static_cast<int32_t>(base);
             p < csc.colOff[col + 1] - static_cast<int32_t>(base); ++p) {
            int32_t row = csc.rowInd[p] - static_cast<int32_t>(base);
            writeDense(row, col, csc.values.data() + p * width);
        }
    }
}

inline void ScatterCooToDense(const CooData &coo, int64_t base,
                               size_t width, int64_t ld, bool rowMajor,
                               Sparse2DenseGoldenResult &result)
{
    auto writeDense = [&](int64_t row, int64_t col, const uint8_t *val) {
        const size_t pos = DensePosition(row, col, ld, rowMajor);
        std::copy_n(val, width, result.dense.begin() + pos * width);
    };
    for (int64_t p = 0; p < coo.nnz; ++p) {
        int32_t row = coo.rowInd[p] - static_cast<int32_t>(base);
        int32_t col = coo.colInd[p] - static_cast<int32_t>(base);
        writeDense(row, col, coo.values.data() + p * width);
    }
}

#ifdef SPARSE_TEST_USE_EIGEN
inline double DecodeDenseValue(const uint8_t *ptr, aclDataType valueType)
{
    if (valueType == ACL_FLOAT) {
        float f;
        std::copy_n(ptr, 4, reinterpret_cast<uint8_t *>(&f));
        return static_cast<double>(f);
    }
    if (valueType == ACL_INT32) {
        int32_t i;
        std::copy_n(ptr, 4, reinterpret_cast<uint8_t *>(&i));
        return static_cast<double>(i);
    }
    return 0.0;
}

inline void VerifyEigenDense(const Eigen::MatrixXd &eigenDense,
    const Sparse2DenseGoldenResult &result, int64_t m, int64_t n,
    int64_t ld, bool rowMajor, aclDataType valueType, size_t width)
{
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            if (!rowMajor || col >= ld) continue;
            const size_t pos = DensePosition(row, col, ld, true);
            double got = DecodeDenseValue(result.dense.data() + pos * width,
                                          valueType);
            if (got != eigenDense(row, col)) {
                throw std::logic_error(
                    "Eigen cross-check: dense mismatch at (" +
                    std::to_string(row) + "," + std::to_string(col) + ")");
            }
        }
    }
}

inline void EigenCrossCheck(const CsrData &csr, int64_t m, int64_t n, int64_t ld,
                             bool rowMajor, int64_t base, aclDataType valueType,
                             size_t width, const Sparse2DenseGoldenResult &result)
{
    using Triplet = Eigen::Triplet<double, int64_t>;
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<size_t>(csr.nnz));
    for (int64_t row = 0; row < m; ++row) {
        for (int32_t p = csr.rowOff[row] - static_cast<int32_t>(base);
             p < csr.rowOff[row + 1] - static_cast<int32_t>(base); ++p) {
            int32_t col = csr.colInd[p] - static_cast<int32_t>(base);
            double val = DecodeDenseValue(csr.values.data() + p * width, valueType);
            if (valueType != ACL_FLOAT && valueType != ACL_INT32)
                val = static_cast<double>(p + 1);
            triplets.emplace_back(row, col, val);
        }
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor, int64_t> eigenSparse(m, n);
    eigenSparse.setFromTriplets(triplets.begin(), triplets.end());
    if (static_cast<int64_t>(eigenSparse.nonZeros()) != csr.nnz) {
        throw std::logic_error("Eigen cross-check: nnz mismatch");
    }
    VerifyEigenDense(Eigen::MatrixXd(eigenSparse), result, m, n, ld,
                     rowMajor, valueType, width);
}
#endif

inline Sparse2DenseGoldenResult
Sparse2DenseGolden(const std::string &format, int64_t m, int64_t n, int64_t ld,
                   bool rowMajor, int64_t base, aclDataType valueType,
                   const std::string &dist, uint32_t seed) {
    const size_t width = Sparse2DenseElemWidth(valueType);
    if (width == 0) throw std::invalid_argument("unsupported value type");
    const CsrData csr =
        GenerateCsrGolden(m, n, ld, rowMajor, base, valueType, dist, seed);
    Sparse2DenseGoldenResult result;
    const size_t physical =
        static_cast<size_t>(rowMajor ? m : n) * static_cast<size_t>(ld);
    result.dense.assign(physical * width, 0);

    if (format == "CSC") {
        const CscData csc = CsrToCscGolden(csr, m, n, base, width);
        ScatterCscToDense(csc, n, base, width, ld, rowMajor, result);
    } else if (format == "COO") {
        const CooData coo = CsrToCooGolden(csr, m, base);
        ScatterCooToDense(coo, base, width, ld, rowMajor, result);
    } else {
        ScatterCsrToDense(csr, m, base, width, ld, rowMajor, result);
    }

#ifdef SPARSE_TEST_USE_EIGEN
    EigenCrossCheck(csr, m, n, ld, rowMajor, base, valueType, width, result);
#endif

    return result;
}

} // namespace sparse_test
#endif
