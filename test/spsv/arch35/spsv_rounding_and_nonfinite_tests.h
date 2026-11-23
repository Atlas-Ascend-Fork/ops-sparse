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

#ifndef SPSV_ROUNDING_AND_NONFINITE_TESTS_H_
#define SPSV_ROUNDING_AND_NONFINITE_TESTS_H_

TEST_F(SpSVExceptionTest, ReviewFp32ProductRounding) {
    for (int m : {2, 40, 300}) {
        CsrMatrix a;
        a.rows = a.cols = m;
        a.rowOffsets = {0};
        std::vector<float> x(m, 1.0f), expected(m, 1.0f);
        x[0] = expected[0] = 1.0f + 0x1p-12f;
        for (int row = 0; row < m; ++row) {
            if (row == m - 1) {
                for (int col = 0; col < row; ++col) {
                    a.colIndices.push_back(col);
                    a.values.push_back(col == 0 ? 1074003968.0f : 0.0f);
                }
                x[row] = 1074298880.0f;
                expected[row] = 32704.0f;
            }
            a.colIndices.push_back(row);
            a.values.push_back(1.0f);
            a.rowOffsets.push_back(static_cast<int32_t>(a.values.size()));
        }
        a.nnz = static_cast<int64_t>(a.values.size());
        VerifyRealProductRounding(spHandle_->get(), env_->stream(), a, x, expected, m);
    }
}

TEST_F(SpSVExceptionTest, ReviewFp32GroupedProductRounding) {
    constexpr int width = 32;
    constexpr int m = 5 * width;
    for (int dependencies : {1, 3, 8, 31}) {
        CsrMatrix a;
        a.rows = a.cols = m;
        a.rowOffsets = {0};
        std::vector<float> x(m, 1.0f), expected(m, 1.0f);
        for (int row = 0; row < m; ++row) {
            int group = row / width;
            if (group == 0) {
                x[row] = expected[row] = 1.0f + 0x1p-12f;
            } else {
                int previous = (group - 1) * width;
                for (int dep = 0; dep < dependencies; ++dep) {
                    a.colIndices.push_back(previous + dep);
                    a.values.push_back(dep == 0 ? (group == 1 ? 1074003968.0f : 1.0f) : 0.0f);
                }
                if (group == 1) {
                    x[row] = 1074298880.0f;
                    expected[row] = 32704.0f;
                } else {
                    x[row] = expected[previous] + 1.0f;
                }
            }
            a.colIndices.push_back(row);
            a.values.push_back(1.0f);
            a.rowOffsets.push_back(static_cast<int32_t>(a.values.size()));
        }
        a.nnz = static_cast<int64_t>(a.values.size());
        VerifyRealProductRounding(spHandle_->get(), env_->stream(),
            a, x, expected, dependencies);
    }
}

// Ordinary nonzero complex components must retain accuracy without treating
// every low-word rounding residual as a reason to rebuild the entire row.
// The last two cases also check both broadcast components after a warp retry.
static void BuildComplexRoundingRow(
    CsrMatrix &a, std::vector<std::complex<float>> &values,
    std::vector<std::complex<float>> &x, std::vector<std::complex<float>> &expected,
    std::complex<float> alpha, int kind, int row, int m)
{
    using cf = std::complex<float>;
    auto add = [&](int col, cf v) {
        a.colIndices.push_back(col);
        a.values.push_back(v.real());
        values.push_back(v);
    };
    if (kind == 0) {
        cf diag(1.25f, 0.125f);
        std::complex<double> rhs = std::complex<double>(diag) *
            std::complex<double>(expected[row]);
        int end = (row / 64) * 64;
        for (int col = std::max(0, end - 8); col < end; ++col) {
            cf v(0.002f + (col % 5) * 0.0001f, 0.0007f);
            add(col, v);
            rhs += std::complex<double>(v) * std::complex<double>(expected[col]);
        }
        add(row, diag);
        rhs /= std::complex<double>(alpha);
        x.emplace_back(rhs.real(), rhs.imag());
    } else {
        x.push_back(expected[row]);
        if (row == m - 2) {
            for (int col = 0; col < row; ++col) {
                float v = col == 0 ? 0x1p100f : col == 1 ? 0x1p50f :
                    col == 2 ? 1.0f : col == 3 ? -0x1p100f :
                    col == 4 ? -0x1p50f : 0.0f;
                add(col, cf(v, 0.0f));
            }
            x[row] = cf(0.0f, 0.0f);
            expected[row] = cf(-1.0f, -2.0f);
        } else if (row == m - 1) {
            add(row - 1, cf(-1.0f, 0.0f));
            x[row] = cf(0.0f, 0.0f);
            expected[row] = cf(-1.0f, -2.0f);
        }
        add(row, cf(1.0f, 0.0f));
    }
}

TEST_F(SpSVExceptionTest, ReviewComplexRoundingAndRetry) {
    using cf = std::complex<float>;
    for (int kind = 0; kind < 3; ++kind) {
        int m = kind == 0 ? 256 : kind == 1 ? 40 : 300;
        CsrMatrix a;
        a.rows = a.cols = m;
        a.rowOffsets = {0};
        std::vector<cf> values, x, expected(m, cf(1.0f, kind == 0 ? 0.5f : 2.0f));
        cf alpha = kind == 0 ? cf(0.9f, 0.15f) : cf(1.0f, 0.0f);
        for (int row = 0; row < m; ++row) {
            BuildComplexRoundingRow(a, values, x, expected, alpha, kind, row, m);
            a.rowOffsets.push_back(static_cast<int32_t>(values.size()));
        }
        a.nnz = static_cast<int64_t>(values.size());
        for (bool inplace : {false, true}) {
            SpSVNpuConfig cfg;
            cfg.inPlace = inplace;
            auto result = SpSVNpuWrapperComplex(spHandle_->get(), env_->stream(),
                a, values, x, alpha, cfg);
            for (int row = 0; row < m; ++row) {
                EXPECT_NEAR(result.y[row].real(), expected[row].real(), 1.0e-6f)
                    << kind << ":" << inplace << ":" << row;
                EXPECT_NEAR(result.y[row].imag(), expected[row].imag(), 1.0e-6f)
                    << kind << ":" << inplace << ":" << row;
            }
        }
    }
}

static void BuildComplexNaNDependencyRows(
    std::vector<std::vector<std::pair<int, std::complex<float>>>> &rows,
    std::vector<std::complex<float>> &x, std::vector<bool> &affected,
    std::complex<float> diag, int m, int layout, bool forward, aclsparseOperation_t op)
{
    using cf = std::complex<float>;
    const float nan = std::nanf("");
    auto index = [&](int row) { return forward ? row : m - 1 - row; };
    auto add = [&](int row, int col, cf value) {
        if (op != ACL_SPARSE_OP_NON_TRANSPOSE) {
            std::swap(row, col);
            if (op == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) value = std::conj(value);
        }
        rows[row].emplace_back(col, value);
    };
    for (int r = 0; r < m; ++r) {
        int row = index(r);
        add(row, row, diag);
        x[row] = diag * cf(1.0f, 0.5f);
        int end = layout == 2 ? (r / 32) * 32 : r;
        int count = layout == 0 ? 1 : layout == 3 ? r : 8;
        if (layout == 3 && r != m - 1) end = 0;
        if (layout == 2 && end > count) {
            add(row, index(0), cf(0.0f, 0.0f));
            affected[row] = true;
        }
        for (int c = std::max(0, end - count); c < end; ++c) {
            add(row, index(c), cf(0.0f, 0.0f));
            affected[row] = affected[row] || affected[index(c)];
        }
        if (r == 0) {
            x[row] = cf(nan, 0.5f);
            affected[row] = true;
        }
    }
}

static void RunComplexNaNDependencyCase(
    aclsparseHandle_t handle, aclrtStream stream, int layout, bool forward,
    aclsparseOperation_t op, bool unit)
{
    using cf = std::complex<float>;
    int m = layout == 3 ? 300 : 128;
    CsrMatrix a;
    a.rows = a.cols = m;
    a.rowOffsets = {0};
    std::vector<std::vector<std::pair<int, cf>>> rows(m);
    std::vector<cf> x(m, cf(1.0f, 0.5f)), values;
    std::vector<bool> affected(m, false);
    cf diag = unit ? cf(1.0f, 0.0f) : cf(2.0f, 0.25f);
    BuildComplexNaNDependencyRows(rows, x, affected, diag, m, layout, forward, op);
    for (auto& entries : rows) {
        std::sort(entries.begin(), entries.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
        for (const auto& entry : entries) {
            a.colIndices.push_back(entry.first);
            a.values.push_back(entry.second.real());
            values.push_back(entry.second);
        }
        a.rowOffsets.push_back(static_cast<int32_t>(values.size()));
    }
    a.nnz = static_cast<int64_t>(values.size());
    for (bool inplace : {false, true}) {
        SpSVNpuConfig cfg;
        cfg.lower = forward == (op == ACL_SPARSE_OP_NON_TRANSPOSE);
        cfg.unitDiag = unit;
        cfg.opA = op;
        cfg.inPlace = inplace;
        auto result = SpSVNpuWrapperComplex(handle, stream,
            a, values, x, cf(1.0f, 0.0f), cfg);
        for (int row = 0; row < m; ++row) {
            SCOPED_TRACE(::testing::Message() << layout << ":" << forward << ":"
                << op << ":" << unit << ":" << inplace << ":" << row);
            if (affected[row]) {
                EXPECT_TRUE(std::isnan(result.y[row].real()));
                EXPECT_TRUE(std::isnan(result.y[row].imag()));
            } else {
                EXPECT_NEAR(result.y[row].real(), 1.0f, 1.0e-6f);
                EXPECT_NEAR(result.y[row].imag(), 0.5f, 1.0e-6f);
            }
        }
    }
}

TEST_F(SpSVExceptionTest, ReviewComplexNaNDependency) {
    // Explicit zero entries are still IEEE products: zero times NaN is NaN.
    // Exercise chains, independent rows in wide levels, and a cooperative row.
    for (int layout = 0; layout < 4; ++layout) {
        for (bool forward : {false, true}) {
            for (auto op : {ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
                            ACL_SPARSE_OP_CONJUGATE_TRANSPOSE}) {
                for (bool unit : {false, true}) {
                    ASSERT_NO_FATAL_FAILURE(RunComplexNaNDependencyCase(
                        spHandle_->get(), env_->stream(), layout, forward, op, unit));
                }
            }
        }
    }
}

// The exact last solution is one. Two rounded predecessor words are
// insufficient: their remaining 1/3 error is amplified by 3*2^exponent.
TEST_F(SpSVExceptionTest, ReviewFp32AmplifiedPredecessorError) {
    for (int exponent : {50, 70, 100, 120}) {
        for (bool forward : {false, true}) {
            for (bool transpose : {false, true}) {
                CsrMatrix a;
                a.rows = a.cols = 3;
                a.rowOffsets = {0};
                std::vector<std::vector<std::pair<int, float>>> rows(3);
                auto index = [&](int row) { return forward ? row : 2 - row; };
                auto add = [&](int r, int c, float value) {
                    r = index(r); c = index(c);
                    if (transpose) { std::swap(r, c); }
                    rows[r].emplace_back(c, value);
                };
                add(0, 0, 3.0f); add(1, 1, 1.0f);
                add(2, 0, std::ldexp(3.0f, exponent));
                add(2, 1, -1.0f); add(2, 2, 1.0f);
                for (auto &entries : rows) {
                    std::sort(entries.begin(), entries.end());
                    for (auto entry : entries) {
                        a.colIndices.push_back(entry.first); a.values.push_back(entry.second);
                    }
                    a.rowOffsets.push_back(static_cast<int32_t>(a.values.size()));
                }
                a.nnz = static_cast<int64_t>(a.values.size());
                std::vector<float> x(3, 1.0f);
                x[index(2)] = std::ldexp(1.0f, exponent);
                for (int layout = 0; layout < 3; ++layout) {
                    for (bool inplace : {false, true}) {
                        SpSVNpuConfig cfg;
                        cfg.isRowPtrI64 = layout != 0; cfg.isColIndI64 = layout == 2;
                        cfg.lower = forward != transpose;
                        cfg.opA = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
                        cfg.inPlace = inplace;
                        auto result = SpSVNpuWrapper(spHandle_->get(), env_->stream(), a, x, 1.0f, cfg);
                        ASSERT_FLOAT_EQ(result.y[index(0)], 1.0f / 3.0f);
                        ASSERT_EQ(result.y[index(1)], 1.0f);
                        EXPECT_EQ(result.y[index(2)], 1.0f)
                            << exponent << ":" << forward << ":" << transpose << ":" << layout << ":" << inplace;
                    }
                }
            }
        }
    }
}

TEST_F(SpSVExceptionTest, ReviewFp32NonfiniteSuccessor) {
    CsrMatrix a;
    a.rows = a.cols = 4;
    a.nnz = 8;
    a.rowOffsets = {0, 1, 2, 5, 8};
    a.colIndices = {0, 1, 0, 1, 2, 1, 2, 3};
    a.values = {3.0f, 1.0f, 0x1.8p71f, -1.0f, 1.0f, -1.0f, 0x1p120f, 1.0f};
    const std::vector<float> x = {1.0f, 1.0f, 0x1p70f, 0x1p120f};
    for (int layout = 0; layout < 3; ++layout) {
        for (bool inplace : {false, true}) {
            for (int repeat = 0; repeat < 3; ++repeat) {
                SpSVNpuConfig cfg;
                cfg.isRowPtrI64 = layout != 0;
                cfg.isColIndI64 = layout == 2;
                cfg.inPlace = inplace;
                const auto result = SpSVNpuWrapper(spHandle_->get(), env_->stream(), a, x, 1.0f, cfg);
                SCOPED_TRACE(::testing::Message() << layout << ":" << inplace << ":" << repeat);
                EXPECT_FLOAT_EQ(result.y[0], 1.0f / 3.0f);
                EXPECT_EQ(result.y[1], 1.0f);
                EXPECT_EQ(result.y[2], 1.0f);
                EXPECT_EQ(result.y[3], 1.0f);
            }
        }
    }
}

TEST_F(SpSVExceptionTest, ReviewComplexExactMidpointRecovery) {
    using Complex = std::complex<float>;
    // The last exact component is 1 + 2^-24, precisely halfway between
    // two binary32 values. A finite dyadic enclosure cannot discharge it;
    // CRT must remain reachable after the final dyadic precision attempt.
    for (int32_t m : {4, 64, 257}) {
        CsrMatrix matrix;
        matrix.rows = matrix.cols = m;
        matrix.rowOffsets = {0};
        std::vector<Complex> values;
        for (int32_t row = 0; row < m; ++row) {
            if (row == m - 1) {
                matrix.colIndices.insert(matrix.colIndices.end(), {0, 1, 2});
                matrix.values.insert(matrix.values.end(), {0x1.8p71f, -1.0f, -1.0f});
                values.insert(values.end(), {Complex(0x1.8p71f, 0.0f),
                    Complex(-1.0f, 0.0f), Complex(-1.0f, 0.0f)});
            }
            const float diagonal = row == 0 ? 3.0f : 1.0f;
            matrix.colIndices.push_back(row);
            matrix.values.push_back(diagonal);
            values.emplace_back(diagonal, 0.0f);
            matrix.rowOffsets.push_back(static_cast<int32_t>(values.size()));
        }
        matrix.nnz = static_cast<int64_t>(values.size());
        std::vector<Complex> input(m, Complex(0.0f, 0.0f));
        input[0] = input[1] = Complex(1.0f, 0.0f);
        input[2] = Complex(0x1p-24f, 0.0f);
        input[m - 1] = Complex(0x1p70f, 0.0f);
        for (bool inplace : {false, true}) {
            for (int32_t repeat = 0; repeat < 2; ++repeat) {
                SpSVNpuConfig cfg;
                cfg.inPlace = inplace;
                cfg.analysisCacheState = repeat;
                const auto result = SpSVNpuWrapperComplex(spHandle_->get(), env_->stream(),
                    matrix, values, input, Complex(1.0f, 0.0f), cfg);
                SCOPED_TRACE(::testing::Message() << m << ":" << inplace << ":" << repeat);
                ASSERT_EQ(result.analysisLevels, 2);
                EXPECT_FLOAT_EQ(result.y[0].real(), 1.0f / 3.0f);
                EXPECT_EQ(result.y[1].real(), 1.0f);
                EXPECT_EQ(result.y[2].real(), 0x1p-24f);
                const float answer = result.y[m - 1].real();
                EXPECT_EQ(FloatBits(answer), 0x3f800000U);
                for (int32_t row = 0; row < m; ++row) {
                    EXPECT_EQ(result.y[row].imag(), 0.0f);
                    if (row > 2 && row < m - 1) { EXPECT_EQ(result.y[row].real(), 0.0f); }
                }
            }
        }
    }
}

TEST_F(SpSVExceptionTest, ReviewComplexModerateAmplifiedCancellation) {
    using Complex = std::complex<float>;
    // Moderate coefficients amplify the one-third rounding residue down a
    // dependency chain; the final RHS nearly cancels the predecessor.
    constexpr int32_t m = 67;
    CsrMatrix matrix;
    matrix.rows = matrix.cols = m;
    matrix.rowOffsets = {0};
    std::vector<Complex> values;
    std::vector<Complex> input(m, Complex(0.0f, 0.0f));
    input[0] = Complex(1.0f, 0.0f);
    input[m - 1] = Complex(93070188544.0f, 0.0f);
    auto append = [&](int32_t column, float value) {
        matrix.colIndices.push_back(column);
        matrix.values.push_back(value);
        values.emplace_back(value, 0.0f);
    };
    for (int32_t row = 0; row < m; ++row) {
        if (row == 0) {
            append(row, 3.0f);
        } else if (row < m - 1) {
            append(row - 1, -1.5f);
            append(row, 1.0f);
        } else {
            append(row - 1, 1.0f);
            append(row, 1.0f);
        }
        matrix.rowOffsets.push_back(static_cast<int32_t>(values.size()));
    }
    matrix.nnz = static_cast<int64_t>(values.size());
    for (bool inplace : {false, true}) {
        SpSVNpuConfig cfg;
        cfg.inPlace = inplace;
        const auto result = SpSVNpuWrapperComplex(spHandle_->get(), env_->stream(),
            matrix, values, input, Complex(1.0f, 0.0f), cfg);
        EXPECT_EQ(result.y.back(), Complex(2104.26318359375f, 0.0f));
    }
}

TEST_F(SpSVExceptionTest, ReviewComplexStableDeepDependency) {
    using Complex = std::complex<float>;
    constexpr int32_t m = 1025;
    CsrMatrix matrix;
    matrix.rows = matrix.cols = m;
    matrix.rowOffsets = {0};
    std::vector<Complex> values;
    std::vector<Complex> input(m, Complex(1.3125f, 0.8125f));
    input[0] = Complex(1.0f, 0.5f);
    for (int32_t row = 0; row < m; ++row) {
        if (row != 0) {
            matrix.colIndices.push_back(row - 1);
            matrix.values.push_back(0.375f);
            values.emplace_back(0.375f, 0.125f);
        }
        matrix.colIndices.push_back(row);
        matrix.values.push_back(1.0f);
        values.emplace_back(1.0f, 0.0f);
        matrix.rowOffsets.push_back(static_cast<int32_t>(values.size()));
    }
    matrix.nnz = static_cast<int64_t>(values.size());
    // |a|_1 = 1/2 makes the true dependency-radius recurrence contractive.
    // A fixed factor of four per level incorrectly makes it expansive.
    for (bool inplace : {false, true}) {
        SpSVNpuConfig cfg;
        cfg.inPlace = inplace;
        const auto result = SpSVNpuWrapperComplex(spHandle_->get(), env_->stream(),
            matrix, values, input, Complex(1.0f, 0.0f), cfg);
        for (int32_t row = 0; row < m; ++row) {
            SCOPED_TRACE(::testing::Message() << inplace << ":" << row);
            EXPECT_EQ(result.y[row], Complex(1.0f, 0.5f));
        }
    }
}


#endif // SPSV_ROUNDING_AND_NONFINITE_TESTS_H_
