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

#ifndef SPSV_DEPENDENCY_AND_ALIASING_TESTS_H_
#define SPSV_DEPENDENCY_AND_ALIASING_TESTS_H_

TEST_F(SpSVExceptionTest, ReviewComplexLargeFiniteRounding) {
    using Complex = std::complex<float>;
    using WideComplex = std::complex<double>;
    // The local two-word enclosure exceeds 0.0005 for these large values,
    // but remains inside the output rounding cell. Keep the dependency chain
    // contractive so this also exercises the ordinary no-recovery path.
    for (int32_t m : {65, 257}) {
        CsrMatrix matrix;
        matrix.rows = matrix.cols = m;
        matrix.rowOffsets = {0};
        std::vector<Complex> values;
        std::vector<Complex> input(m);
        std::vector<WideComplex> expected(m);
        const Complex diagonal(0.375f, 0.125f);
        const Complex coefficient(0.125f, -0.0625f);
        for (int32_t row = 0; row < m; ++row) {
            input[row] = Complex(0x1p40f + static_cast<float>(row % 7) * 0x1p36f,
                -0x1.8p38f + static_cast<float>(row % 5) * 0x1p34f);
            WideComplex rhs(input[row].real(), input[row].imag());
            if (row != 0) {
                matrix.colIndices.push_back(row - 1);
                matrix.values.push_back(coefficient.real());
                values.push_back(coefficient);
                rhs -= WideComplex(coefficient.real(), coefficient.imag()) * expected[row - 1];
            }
            matrix.colIndices.push_back(row);
            matrix.values.push_back(diagonal.real());
            values.push_back(diagonal);
            matrix.rowOffsets.push_back(static_cast<int32_t>(values.size()));
            expected[row] = rhs / WideComplex(diagonal.real(), diagonal.imag());
        }
        matrix.nnz = static_cast<int64_t>(values.size());
        for (bool inplace : {false, true}) {
            SpSVNpuConfig cfg;
            cfg.inPlace = inplace;
            const auto result = SpSVNpuWrapperComplex(spHandle_->get(), env_->stream(),
                matrix, values, input, Complex(1.0f, 0.0f), cfg);
            for (int32_t row = 0; row < m; ++row) {
                SCOPED_TRACE(::testing::Message() << m << ":" << inplace << ":" << row);
                EXPECT_EQ(result.y[row].real(), static_cast<float>(expected[row].real()));
                EXPECT_EQ(result.y[row].imag(), static_cast<float>(expected[row].imag()));
            }
        }
    }
}

TEST_F(SpSVExceptionTest, ReviewComplexAmplifiedPredecessorError) {
    using cf = std::complex<float>;
    for (int exponent : {60, 70, 90}) {
        CsrMatrix a;
        a.rows = a.cols = 3;
        a.nnz = 5;
        a.rowOffsets = {0, 1, 2, 5};
        a.colIndices = {0, 1, 0, 1, 2};
        a.values = {3.0f, 1.0f, std::ldexp(3.0f, exponent), -1.0f, 1.0f};
        std::vector<cf> values;
        for (float value : a.values) { values.emplace_back(value, 0.0f); }
        const std::vector<cf> x = {cf(1.0f, 0.0f), cf(1.0f, 0.0f), cf(std::ldexp(1.0f, exponent), 0.0f)};
        for (bool inplace : {false, true}) {
            for (int repeat = 0; repeat < 3; ++repeat) {
                SpSVNpuConfig cfg;
                cfg.inPlace = inplace;
                const auto result = SpSVNpuWrapperComplex(spHandle_->get(), env_->stream(), a, values, x, cf(1.0f, 0.0f), cfg);
                SCOPED_TRACE(::testing::Message() << exponent << ":" << inplace << ":" << repeat);
                EXPECT_FLOAT_EQ(result.y[0].real(), 1.0f / 3.0f);
                EXPECT_EQ(result.y[1].real(), 1.0f);
                EXPECT_EQ(result.y[2].real(), 1.0f);
                EXPECT_EQ(result.y[0].imag(), 0.0f);
                EXPECT_EQ(result.y[1].imag(), 0.0f);
                EXPECT_EQ(result.y[2].imag(), 0.0f);
            }
        }
    }
}

// Exercise the additional whole-system certificate above its row threshold.
// Independent padding must not turn either amplified P1 system into a proof.
TEST_F(SpSVExceptionTest, ReviewLargeFp32NonfiniteSuccessor) {
    for (int m : {513, 1025}) {
        CsrMatrix a;
        a.rows = a.cols = m;
        a.rowOffsets = {0};
        std::vector<float> x(m, 1.0f);
        for (int r = 0; r < m; ++r) {
            if (r == m - 2) {
                a.colIndices.insert(a.colIndices.end(), {0, 1});
                a.values.insert(a.values.end(), {0x1.8p71f, -1.0f});
                x[r] = 0x1p70f;
            } else if (r == m - 1) {
                a.colIndices.insert(a.colIndices.end(), {1, m - 2});
                a.values.insert(a.values.end(), {-1.0f, 0x1p120f});
                x[r] = 0x1p120f;
            }
            a.colIndices.push_back(r);
            a.values.push_back(r == 0 ? 3.0f : 1.0f);
            a.rowOffsets.push_back(static_cast<int32_t>(a.values.size()));
        }
        a.nnz = static_cast<int64_t>(a.values.size());
        for (int layout = 0; layout < 3; ++layout) {
            for (bool inplace : {false, true}) {
                SpSVNpuConfig cfg;
                cfg.isRowPtrI64 = layout != 0;
                cfg.isColIndI64 = layout == 2;
                cfg.inPlace = inplace;
                const auto result = SpSVNpuWrapper(spHandle_->get(), env_->stream(), a, x, 1.0f, cfg);
                SCOPED_TRACE(::testing::Message() << m << ":" << layout << ":" << inplace);
                ASSERT_EQ(result.y.size(), static_cast<size_t>(m));
                EXPECT_FLOAT_EQ(result.y[0], 1.0f / 3.0f);
                for (int r = 1; r < m; ++r) { EXPECT_EQ(result.y[r], 1.0f) << r; }
            }
        }
    }
}

TEST_F(SpSVExceptionTest, ReviewLargeComplexAmplifiedPredecessorError) {
    using cf = std::complex<float>;
    for (int m : {513, 1025}) {
        for (int exponent : {60, 70, 90}) {
            for (bool forward : {false, true}) {
                CsrMatrix a;
                a.rows = a.cols = m;
                a.rowOffsets = {0};
                const auto index = [&](int r) { return forward ? r : m - 1 - r; };
                std::vector<std::vector<std::pair<int, float>>> rows(m);
                for (int r = 0; r < m; ++r) { rows[index(r)].emplace_back(index(r), r == 0 ? 3.0f : 1.0f); }
                rows[index(m - 1)].emplace_back(index(0), std::ldexp(3.0f, exponent));
                rows[index(m - 1)].emplace_back(index(1), -1.0f);
                for (auto &entries : rows) {
                    std::sort(entries.begin(), entries.end());
                    for (const auto &entry : entries) {
                        a.colIndices.push_back(entry.first);
                        a.values.push_back(entry.second);
                    }
                    a.rowOffsets.push_back(static_cast<int32_t>(a.values.size()));
                }
                a.nnz = static_cast<int64_t>(a.values.size());
                std::vector<cf> values;
                for (float value : a.values) { values.emplace_back(value, 0.0f); }
                std::vector<cf> x(m, cf(1.0f, 0.0f));
                x[index(m - 1)] = cf(std::ldexp(1.0f, exponent), 0.0f);
                for (bool inplace : {false, true}) {
                    SpSVNpuConfig cfg;
                    cfg.lower = forward;
                    cfg.inPlace = inplace;
                    const auto result = SpSVNpuWrapperComplex(spHandle_->get(), env_->stream(), a, values, x, cf(1.0f, 0.0f), cfg);
                    SCOPED_TRACE(::testing::Message() << m << ":" << exponent << ":" << forward << ":" << inplace);
                    ASSERT_EQ(result.y.size(), static_cast<size_t>(m));
                    EXPECT_FLOAT_EQ(result.y[index(0)].real(), 1.0f / 3.0f);
                    for (int r = 1; r < m; ++r) { EXPECT_EQ(result.y[index(r)].real(), 1.0f) << r; }
                    for (int r = 0; r < m; ++r) { EXPECT_EQ(result.y[r].imag(), 0.0f) << r; }
                }
            }
        }
    }
}

static void RunComplexDependencyDirection(
    aclsparseHandle_t handle, aclrtStream stream, int exponent, bool forward, aclsparseOperation_t op)
{
    using cf = std::complex<float>;
    CsrMatrix a;
    a.rows = a.cols = 3;
    a.nnz = 5;
    a.rowOffsets = {0};
    std::vector<std::vector<std::pair<int, cf>>> rows(3);
    auto index = [&](int r) { return forward ? r : 2 - r; };
    auto add = [&](int r, int c, cf value) {
        r = index(r); c = index(c);
        if (op != ACL_SPARSE_OP_NON_TRANSPOSE) {
            std::swap(r, c);
            if (op == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) { value = std::conj(value); }
        }
        rows[r].emplace_back(c, value);
    };
    float b = std::ldexp(1.0f, exponent);
    add(0, 0, cf(3.0f, -3.0f)); add(1, 1, cf(-1.0f, -1.0f));
    add(2, 0, cf(3.0f * b, -3.0f * b));
    add(2, 1, cf(1.0f, 1.0f)); add(2, 2, cf(1.0f, 1.0f));
    std::vector<cf> values, rhs(3, cf(0.0f, 2.0f));
    rhs[index(2)] = cf(0.0f, 2.0f * b);
    for (auto &entries : rows) {
        std::sort(entries.begin(), entries.end(),
            [](const auto &left, const auto &right) { return left.first < right.first; });
        for (const auto &entry : entries) {
            a.colIndices.push_back(entry.first); a.values.push_back(entry.second.real());
            values.push_back(entry.second);
        }
        a.rowOffsets.push_back(static_cast<int32_t>(values.size()));
    }
    for (bool inplace : {false, true}) {
        SpSVNpuConfig cfg;
        cfg.lower = forward == (op == ACL_SPARSE_OP_NON_TRANSPOSE);
        cfg.opA = op; cfg.inPlace = inplace;
        const auto result = SpSVNpuWrapperComplex(handle, stream,
            a, values, rhs, cf(1.0f, 0.0f), cfg);
        SCOPED_TRACE(::testing::Message() << exponent << ":" << forward << ":" << op << ":" << inplace);
        EXPECT_FLOAT_EQ(result.y[index(0)].real(), -1.0f / 3.0f);
        EXPECT_FLOAT_EQ(result.y[index(0)].imag(), 1.0f / 3.0f);
        EXPECT_EQ(result.y[index(1)], cf(-1.0f, -1.0f));
        EXPECT_EQ(result.y[index(2)], cf(1.0f, 1.0f));
    }
}

TEST_F(SpSVExceptionTest, ReviewComplexDependencyDirections) {
    for (int exponent : {60, 70, 90}) {
        for (bool forward : {false, true}) {
            for (auto op : {ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
                           ACL_SPARSE_OP_CONJUGATE_TRANSPOSE}) {
                RunComplexDependencyDirection(spHandle_->get(), env_->stream(), exponent, forward, op);
            }
        }
    }
}

static void AddAmplifiedPredecessorDependencies(
    std::vector<std::vector<std::pair<int, float>>> &rows, std::vector<float> &rhs,
    int width, int dependencies, int m, bool forward, int r, int level)
{
    auto at = [&](int row) { return forward ? row : m - 1 - row; };
    auto add = [&](int row, int col, float v) { rows[at(row)].emplace_back(at(col), v); };
    if ((width == 1 && r == 2) || (width != 1 && level == 1)) {
        add(r, 0, 0x1.8p71f); add(r, 1, -1.0f);
        rhs[at(r)] = 0x1p70f;
        for (int c = 2; c < dependencies && c < width; ++c) { add(r, c, 0.0f); }
    } else if ((width == 1 && r > 2) || (width != 1 && level > 1)) {
        int previous = width == 1 ? r - 1 : (level - 1) * width + (r + 127) % width;
        add(r, previous, 1.0f);
        rhs[at(r)] = 2.0f;
        for (int k = 1; k < dependencies; ++k) {
            add(r, (level - 1) * width + (r + 127 + k) % width, 0.0f);
        }
    }
}

static void BuildAmplifiedPredecessorRows(
    std::vector<std::vector<std::pair<int, float>>> &rows,
    std::vector<float> &rhs, std::vector<float> &expected,
    int width, int dependencies, int m, bool forward)
{
    auto at = [&](int r) { return forward ? r : m - 1 - r; };
    auto add = [&](int r, int c, float v) { rows[at(r)].emplace_back(at(c), v); };
    for (int r = 0; r < m; ++r) {
        int level = width == 1 ? r : r / width;
        bool first = width == 1 ? r == 0 : level == 0 && r % 2 == 0;
        add(r, r, first ? 3.0f : 1.0f);
        if (first) { expected[at(r)] = 1.0f / 3.0f; }
        if (width == 1 && r == 1) { add(r, 0, 0.0f); }
        AddAmplifiedPredecessorDependencies(rows, rhs, width, dependencies, m, forward, r, level);
    }
}

TEST_F(SpSVExceptionTest, ReviewFp32AmplifiedPredecessorSchedules) {
    // A stored zero supplies the chain schedule's second root. Wider levels
    // rotate ownership; stored padding selects grouped/warp/wide-row paths.
    for (auto shape : std::vector<std::pair<int, int>>{
            {1, 1}, {2, 1}, {32, 8}, {32, 32}, {128, 1}, {256, 8}, {257, 256}}) {
        int width = shape.first, dependencies = shape.second;
        int m = width == 1 ? 5 : 5 * width;
        for (bool forward : {false, true}) {
            CsrMatrix a;
            a.rows = a.cols = m;
            a.rowOffsets = {0};
            std::vector<std::vector<std::pair<int, float>>> rows(m);
            std::vector<float> rhs(m, 1.0f), expected(m, 1.0f);
            BuildAmplifiedPredecessorRows(rows, rhs, expected, width, dependencies, m, forward);
            for (auto &entries : rows) {
                std::sort(entries.begin(), entries.end());
                for (auto e : entries) { a.colIndices.push_back(e.first); a.values.push_back(e.second); }
                a.rowOffsets.push_back(static_cast<int32_t>(a.values.size()));
            }
            a.nnz = static_cast<int64_t>(a.values.size());
            for (bool inplace : {false, true}) {
                SpSVNpuConfig cfg;
                cfg.lower = forward; cfg.inPlace = inplace;
                auto result = SpSVNpuWrapper(spHandle_->get(), env_->stream(), a, rhs, 1.0f, cfg);
                for (int r = 0; r < m; ++r) {
                    ASSERT_FLOAT_EQ(result.y[r], expected[r])
                        << width << ":" << dependencies << ":" << forward << ":" << inplace << ":" << r;
                }
            }
        }
    }
}

TEST_F(SpSVExceptionTest, ReviewFp32SmallPredecessorLowWord) {
    CsrMatrix a;
    a.rows = a.cols = 3;
    a.nnz = 4;
    a.rowOffsets = {0, 1, 3, 4};
    a.colIndices = {0, 0, 1, 2};
    a.values = {3.0f, 402653184.0f, 1.0f, 1.0f};
    const std::vector<float> x = {1.0f, 134217744.0f, 1.0f};
    for (bool inplace : {false, true}) {
        SpSVNpuConfig cfg;
        cfg.inPlace = inplace;
        auto result = SpSVNpuWrapper(spHandle_->get(), env_->stream(), a, x, 1.0f, cfg);
        ASSERT_FLOAT_EQ(result.y[0], 1.0f / 3.0f);
        ASSERT_FLOAT_EQ(result.y[1], 16.0f) << "inplace=" << inplace;
        ASSERT_FLOAT_EQ(result.y[2], 1.0f);
    }
}

static void BuildPredecessorLowWordRows(
    std::vector<std::vector<std::pair<int, float>>> &rows,
    std::vector<float> &x, std::vector<float> &expected,
    int width, int dependencies, int m, bool forward, bool transpose)
{
    if (width <= 0) {
        ADD_FAILURE() << "Predecessor test width must be positive.";
        return;
    }
    const int rowWidth = width > 0 ? width : 1;
    auto index = [forward, m](int r) { return forward ? r : m - 1 - r; };
    auto add = [transpose, &rows](int r, int c, float v) {
        if (transpose) { std::swap(r, c); }
        rows[r].emplace_back(c, v);
    };
    for (int r = 0; r < m; ++r) {
        int row = index(r), level = r / rowWidth;
        add(row, row, level == 0 ? 3.0f : 1.0f);
        if (level == 0) {
            x[row] = 1.0f;
            expected[row] = 1.0f / 3.0f;
        } else {
            // Change ownership across levels rather than mapping
            // each dependency back to the same block/thread.
            int previous = (level - 1) * rowWidth + (r + 127) % rowWidth;
            add(row, index(previous), level == 1 ? 402653184.0f : 1.0f);
            for (int k = 1; k < dependencies; ++k) {
                int padding = (level - 1) * rowWidth + (r + 127 + k) % rowWidth;
                add(row, index(padding), 0.0f);
            }
            x[row] = level == 1 ? 134217744.0f : level == 2 ? 17.0f : 2.0f;
            expected[row] = level == 1 ? 16.0f : 1.0f;
        }
    }
}

static void VerifyPredecessorLowWordCase(
    aclsparseHandle_t handle, aclrtStream stream, const CsrMatrix &a,
    const std::vector<float> &x, const std::vector<float> &expected,
    int m, int width, int dependencies, bool forward, bool transpose)
{
    for (bool inplace : {false, true}) {
        SpSVNpuConfig cfg;
        cfg.lower = forward != transpose;
        cfg.opA = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
        cfg.inPlace = inplace;
        auto result = SpSVNpuWrapper(handle, stream, a, x, 1.0f, cfg);
        for (int row = 0; row < m; ++row) {
            ASSERT_NEAR(result.y[row], expected[row], 0x1p-18f)
                << width << ":" << dependencies << ":" << forward << ":" << transpose
                << ":" << inplace << ":" << row;
        }
    }
}

TEST_F(SpSVExceptionTest, ReviewFp32PredecessorLowWord) {
    // Include short and longer rows at identical level widths. Density controls
    // whether the scheduler uses one lane, a small lane group, or a full warp.
    const std::vector<std::pair<int, int>> shapes = {
        {1, 1}, {2, 1}, {32, 1}, {32, 8}, {32, 32}, {64, 8},
        {127, 8}, {128, 1}, {128, 32}, {129, 8},
        {255, 1}, {256, 1}, {256, 8}, {256, 64}, {257, 256}, {512, 1}, {512, 32},
        {513, 1}, {768, 8}, {1024, 1}, {1024, 32}};
    for (const auto &shape : shapes) {
        int width = shape.first, dependencies = shape.second;
        constexpr int levels = 5;
        int m = width * levels;
        for (bool forward : {false, true}) {
            for (bool transpose : {false, true}) {
                CsrMatrix a;
                a.rows = a.cols = m;
                a.rowOffsets = {0};
                std::vector<std::vector<std::pair<int, float>>> rows(m);
                std::vector<float> x(m), expected(m);
                BuildPredecessorLowWordRows(rows, x, expected,
                    width, dependencies, m, forward, transpose);
                for (auto &entries : rows) {
                    std::sort(entries.begin(), entries.end());
                    for (auto entry : entries) {
                        a.colIndices.push_back(entry.first);
                        a.values.push_back(entry.second);
                    }
                    a.rowOffsets.push_back(static_cast<int32_t>(a.values.size()));
                }
                a.nnz = static_cast<int64_t>(a.values.size());
                ASSERT_NO_FATAL_FAILURE(VerifyPredecessorLowWordCase(spHandle_->get(), env_->stream(),
                    a, x, expected, m, width, dependencies, forward, transpose));
            }
        }
    }
}

static void BuildDescriptorReuseMatrix(
    CsrMatrix &a, std::vector<float> &x, std::vector<float> &expected, int32_t width, int32_t m)
{
    if (width <= 0) {
        ADD_FAILURE() << "Descriptor reuse test width must be positive.";
        return;
    }
    const int32_t rowWidth = width > 0 ? width : 1;
    for (int32_t row = 0; row < m; ++row) {
        expected[row] = row % 2 ? 5.0f : -5.0f;
        x[row] = expected[row];
        if (row >= rowWidth) {
            int32_t previous = (row / rowWidth - 1) * rowWidth + (row + 127) % rowWidth;
            a.colIndices.push_back(previous);
            a.values.push_back(1.0f);
            x[row] += expected[previous];
        }
        a.colIndices.push_back(row);
        a.values.push_back(1.0f);
        a.rowOffsets.push_back(static_cast<int32_t>(a.values.size()));
    }
}


#endif // SPSV_DEPENDENCY_AND_ALIASING_TESTS_H_
