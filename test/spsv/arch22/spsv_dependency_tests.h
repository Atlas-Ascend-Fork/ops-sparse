/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_DEPENDENCY_TESTS_H_
#define SPSV_ARCH22_DEPENDENCY_TESTS_H_

// Included after spsv_range_tests.h inside the test anonymous namespace.
Complex SensitiveDependencyResidual(double power, aclDataType type)
{
    if (type == ACL_FLOAT)
        return {std::fma(-3 * power, 1.0 / 3.0, power), 0.0};
    return Complex{power, 0.0} - Complex{3 * power, 0.0} * Complex{1.0 / 3.0, 0.0};
}

void RunSensitiveDependencyRegression(const RangePathCase& test, int exponent)
{
    const auto& c = test.config;
    const bool forward = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? c.lower : !c.lower;
    const int first = forward ? 0 : test.rows - 1;
    const int second = forward ? 1 : test.rows - 2;
    const double power = std::ldexp(1.0, exponent);
    std::vector<Entry> entries;
    for (int row = 0; row < test.rows; ++row)
        entries.push_back(RangeStoredEntry(row, row, {row == first ? 3.0 : 1.0, 0.0}, c));
    entries.push_back(RangeStoredEntry(second, first, {3 * power, 0.0}, c));
    const auto matrix = EncodeEntries(test.rows, entries, c);
    std::vector<Complex> x(test.rows, {0.0, 0.0});
    std::vector<Complex> expected(test.rows, {0.0, 0.0});
    x[first] = {1.0, 0.0};
    x[second] = {power, 0.0};
    expected[first] = {1.0 / 3.0, 0.0};
    expected[second] = SensitiveDependencyResidual(power, c.type);
    Plan plan(matrix, c);
    plan.input = Pack(x, c.type);
    plan.alpha = Pack({{1.0, 0.0}}, c.type);
    plan.dX.Put(plan.input);
    plan.dAlpha.Put(plan.alpha);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    VerifyRangeTwice(plan, expected, 1);
    ASSERT_EQ(aclsparseSpSV_updateMatrix(SpSVArch22Test::handle, plan.state, plan.dValues.get(),
                                         ACL_SPARSE_SPSV_UPDATE_GENERAL),
              kSuccess);
    VerifyRangeTwice(plan, expected, 1);
    std::vector<Complex> diagonal(test.rows, {1.0, 0.0});
    diagonal[first] = {3.0, 0.0};
    DeviceBuffer updated(Pack(diagonal, c.type).size() * sizeof(float));
    updated.Put(Pack(diagonal, c.type));
    ASSERT_EQ(
        aclsparseSpSV_updateMatrix(SpSVArch22Test::handle, plan.state, updated.get(), ACL_SPARSE_SPSV_UPDATE_DIAGONAL),
        kSuccess);
    VerifyRangeTwice(plan, expected, 1);
    CheckAcl(aclrtSynchronizeStream(SpSVArch22Test::env->stream()), "sensitive dependency complete");
    DestroyChecked(plan);
}

void AppendSensitiveVariants(std::vector<RangePathCase>& tests, aclDataType type, int rows,
                             aclsparseOperation_t operation)
{
    for (int variant = 0; variant < 4; ++variant) {
        auto c = RangePathConfig(type, operation, variant);
        c.inPlace = (variant & 1) != 0;
        c.base = c.inPlace ? 1 : 0;
        if ((variant & 2) != 0)
            c.lower = !c.lower;
        tests.push_back({c, rows, false});
    }
}

std::vector<RangePathCase> SensitiveDependencyCases()
{
    std::vector<RangePathCase> tests;
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        const int rows[] = {2, type == ACL_FLOAT ? 513 : 257, type == ACL_FLOAT ? 24577 : 12289};
        for (int path = 0; path < 3; ++path) {
            for (auto operation : kOperations) {
                AppendSensitiveVariants(tests, type, rows[path], operation);
            }
        }
    }
    return tests;
}

TEST_F(SpSVArch22Range, RoundedDependenciesReplayFromOriginalInputs)
{
    for (const auto& test : SensitiveDependencyCases()) {
        SCOPED_TRACE(RangePathName(test) + (test.config.lower ? "_Lower" : "_Upper"));
        RunSensitiveDependencyRegression(test, test.rows == 2 ? 44 : 30);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

void RunDependencyMagnitudeRegression(const RangePathCase& test, float coefficient)
{
    const auto& c = test.config;
    const bool forward = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? c.lower : !c.lower;
    const int first = forward ? 0 : test.rows - 1;
    const float source = static_cast<float>(coefficient / 3.0 + std::abs(coefficient) * 0x1p-23);
    const auto residual = c.type == ACL_FLOAT ?
                              Complex{std::fma(-static_cast<double>(coefficient), 1.0 / 3.0, source), 0} :
                              Complex{source, 0} - Complex{coefficient, 0} * Complex{1.0 / 3.0, 0};
    std::vector<Entry> entries;
    std::vector<Complex> x(test.rows, {0, 0});
    std::vector<Complex> expected(test.rows, {0, 0});
    x[first] = {1, 0};
    expected[first] = {1.0 / 3.0, 0};
    for (int step = 0; step < test.rows; ++step) {
        const int row = forward ? step : test.rows - 1 - step;
        entries.push_back(RangeStoredEntry(row, row, {step == 0 ? 3.0 : 1.0, 0}, c));
        if (step > 0 && step <= 512) {
            entries.push_back(RangeStoredEntry(row, first, {coefficient, 0}, c));
            x[row] = {source, 0};
            expected[row] = residual;
        }
    }
    RunRangeRegression(EncodeEntries(test.rows, entries, c), c, x, {1, 0}, expected, 1);
}

TEST_F(SpSVArch22Range, DependencyReplayCoversMagnitudeBoundary)
{
    Case c;
    c.type = ACL_FLOAT;
    // Every dependent row must retain the unrounded 1/3 prefix.
    const int rows = 513;
    std::vector<Entry> entries{{0, 0, {3, 0}}};
    std::vector<Complex> x(rows, {4096 + 0x1p-11, 0});
    std::vector<Complex> expected(rows, {0x1p-11, 0});
    x[0] = {1, 0};
    expected[0] = {1.0 / 3.0, 0};
    for (int row = 1; row < rows; ++row) {
        entries.push_back({row, 0, {12288, 0}});
        entries.push_back({row, row, {1, 0}});
    }
    RunRangeRegression(EncodeEntries(rows, entries, c), c, x, {1, 0}, expected, 1);
}

TEST_F(SpSVArch22Range, DependencyReplayCoversMagnitudeNeighbors)
{
    for (auto test : SensitiveDependencyCases()) {
        if (test.rows == 2)
            test.rows = test.config.type == ACL_FLOAT ? 512 : 256;
        for (float coefficient : {96.0f, 1536.0f, 6144.0f, std::nextafter(12288.0f, 0.0f), 12288.0f,
                                  std::nextafter(12288.0f, std::numeric_limits<float>::infinity()), 24576.0f}) {
            SCOPED_TRACE(RangePathName(test) + "_coefficient=" + std::to_string(coefficient));
            RunDependencyMagnitudeRegression(test, coefficient);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

void RunLaterAmplificationRegression(aclDataType type, bool inPlace)
{
    Case c;
    c.type = type;
    c.inPlace = inPlace;
    c.deviceAlpha = inPlace;
    c.base = inPlace ? 1 : 0;
    const int rows = type == ACL_FLOAT ? 513 : 257;
    const double power = std::ldexp(1.0, 44);
    std::vector<Entry> entries{{0, 0, {3, 0}}};
    std::vector<Complex> x(rows, {power, 0});
    std::vector<Complex> expected(rows, SensitiveDependencyResidual(power, type));
    x[0] = {1, 0};
    x[1] = {4096 + 0x1p-11, 0};
    expected[0] = {1.0 / 3.0, 0};
    expected[1] = {0x1p-11, 0};
    for (int row = 1; row < rows; ++row) {
        entries.push_back({row, 0, {row == 1 ? 12288 : 3 * power, 0}});
        entries.push_back({row, row, {1, 0}});
    }
    RunRangeRegression(EncodeEntries(rows, entries, c), c, x, {1, 0}, expected, 1);
}

TEST_F(SpSVArch22Range, PrefixReplayRetainsPrecisionUnderLaterAmplification)
{
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        for (bool inPlace : {false, true}) {
            RunLaterAmplificationRegression(type, inPlace);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

void RunWideDuplicateRegression(const Case& c, bool repeatedDiagonal)
{
    const double large = std::ldexp(1.0, 40);
    const double small = std::ldexp(1.0, -20);
    const double power = std::ldexp(1.0, 30);
    std::vector<Entry> entries;
    std::vector<Complex> x;
    std::vector<Complex> expected;
    if (repeatedDiagonal) {
        entries = {{0, 0, {large, 0}}, {0, 0, {small, 0}},     {0, 0, {-large, 0}},
                   {1, 1, {3, 0}},     {2, 1, {3 * power, 0}}, {2, 2, {1, 0}}};
        x = {{small, 0}, {1, 0}, {power, 0}};
        expected = {{1, 0}, {1.0 / 3.0, 0}, SensitiveDependencyResidual(power, c.type)};
        // Select resident solve so the 2^30 dependency must replay the
        // complete prefix, including row zero's repeated diagonal.
        const int rows = c.type == ACL_FLOAT ? 513 : 257;
        for (int row = 3; row < rows; ++row)
            entries.push_back({row, row, {1, 0}});
        x.resize(rows, {0, 0});
        expected.resize(rows, {0, 0});
    } else {
        entries = {{0, 0, {1, 0}}, {1, 0, {large, 0}}, {1, 0, {small, 0}}, {1, 0, {-large, 0}}, {1, 1, {small, 0}}};
        x = {{1, 0}, {0, 0}};
        expected = {{1, 0}, {-1, 0}};
    }
    for (auto& entry : entries)
        entry = RangeStoredEntry(entry.row, entry.col, entry.value, c);
    RunRangeRegression(EncodeEntries(static_cast<int>(x.size()), entries, c), c, x, {1, 0}, expected, 1);
}

TEST_F(SpSVArch22Range, WideReplayRetainsExactDuplicateCancellation)
{
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        for (auto operation : kOperations) {
            for (int variant = 0; variant < 2; ++variant) {
                auto c = RangePathConfig(type, operation, variant);
                c.inPlace = variant != 0;
                c.base = variant;
                RunWideDuplicateRegression(c, true);
                RunWideDuplicateRegression(c, false);
                if (::testing::Test::HasFatalFailure())
                    return;
            }
        }
    }
}

void VerifyWideZeroPrefix(const std::vector<float>& actual, aclDataType type, bool unit)
{
    if (unit) {
        const int components = type == ACL_FLOAT ? 1 : 2;
        for (int part = 0; part < components; ++part) {
            const auto bits = spsv_test::BitCopy<uint32_t>(actual[part]);
            EXPECT_EQ(bits, 0u);
        }
    } else if (type == ACL_FLOAT) {
        EXPECT_TRUE(std::isinf(actual[0]));
        EXPECT_GT(actual[0], 0);
    } else {
        EXPECT_TRUE(std::isnan(actual[0]));
        EXPECT_TRUE(std::isnan(actual[1]));
    }
}

void RunWideSingleZeroRegression(aclDataType type, bool unit, bool inPlace)
{
    Case c;
    c.type = type;
    c.unit = unit;
    c.inPlace = inPlace;
    c.deviceAlpha = inPlace;
    const int rows = type == ACL_FLOAT ? 513 : 257;
    const double power = std::ldexp(1.0, 30);
    std::vector<Entry> entries;
    for (int row = 0; row < rows; ++row) {
        double diagonal = row == 0 ? (unit ? std::numeric_limits<double>::quiet_NaN() : -0.0) : row == 1 ? 3.0 : 1.0;
        entries.push_back({row, row, {diagonal, 0}});
    }
    entries.push_back({2, 1, {(unit ? 1 : 3) * power, 0}});
    Plan plan(EncodeEntries(rows, entries, c), c);
    std::vector<Complex> x(rows, {0, 0});
    x[0] = unit ? Complex{-0.0, -0.0} : Complex{1, 0};
    const double alpha = unit ? 1 + std::ldexp(1.0, -23) : 1;
    x[1] = {alpha, 0};
    x[2] = {power * alpha, 0};
    plan.input = Pack(x, type);
    plan.alpha = Pack({{alpha, 0}}, type);
    plan.dAlpha.Put(plan.alpha);
    plan.dX.Put(plan.input);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    for (int repeat = 0; repeat < 2; ++repeat) {
        plan.dX.Put(plan.input);
        ASSERT_EQ(plan.Solve(c.operation), kSuccess);
        CheckAcl(aclrtSynchronizeStream(SpSVArch22Test::env->stream()), "wide zero solve complete");
        auto actual = (inPlace ? plan.dX : plan.dY).Get<float>(plan.input.size());
        VerifyWideZeroPrefix(actual, type, unit);
        const int components = type == ACL_FLOAT ? 1 : 2;
        const auto expected = unit ? Complex{0, 0} : SensitiveDependencyResidual(power, type);
        VerifyRangeComponents(std::vector<float>(actual.begin() + 2 * components, actual.begin() + 3 * components),
                              {expected}, type, 1);
    }
    DestroyChecked(plan);
}

TEST_F(SpSVArch22Range, WideSingleDiagonalPreservesCanonicalZero)
{
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        for (int variant = 0; variant < 4; ++variant) {
            RunWideSingleZeroRegression(type, (variant & 2) != 0, (variant & 1) != 0);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

bool DuplicateDependencyActive(int step, int rows)
{
    return step > 0 && (rows <= 513 || step == 1 || step == rows / 2 || step >= rows - 2);
}

Matrix MakeDuplicateDependencyMatrix(const RangePathCase& test)
{
    const auto& c = test.config;
    const bool forward = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? c.lower : !c.lower;
    const int first = forward ? 0 : test.rows - 1;
    std::vector<Entry> entries;
    for (int step = 0; step < test.rows; ++step) {
        const int row = forward ? step : test.rows - 1 - step;
        const bool active = DuplicateDependencyActive(step, test.rows);
        entries.push_back(RangeStoredEntry(row, row, {step == 0 ? 3.0 : active ? 0.5 : 1.0, 0}, c));
        if (active) {
            entries.push_back(RangeStoredEntry(row, row, {0.5, 0}, c));
            entries.push_back(RangeStoredEntry(row, first, Value(12288, 6144, c), c));
        }
    }
    return EncodeEntries(test.rows, entries, c);
}

void VerifyDuplicateDependencyUpdates(Plan& plan, int first, std::vector<Complex> expected)
{
    auto values = StoredValues(plan.a, plan.c, false);
    const auto original = values;
    for (size_t slot = 0; slot < values.size(); ++slot) {
        const auto& e = plan.a.entries[slot];
        if (e.row >= 0 && e.row == e.col && e.row != first)
            values[slot] *= 0.5;
    }
    ASSERT_NE(values, original);
    UpdateInput general(values, plan.c.type);
    ASSERT_EQ(general.Apply(plan, ACL_SPARSE_SPSV_UPDATE_GENERAL), kSuccess);
    for (size_t row = 0; row < expected.size(); ++row)
        if (static_cast<int>(row) != first)
            expected[row] *= 2.0;
    VerifyRangeTwice(plan, expected, 1);
    if (::testing::Test::HasFatalFailure())
        return;
    general.ExpectUnchanged();
    std::vector<Complex> diagonal(plan.a.m, {2, 0});
    diagonal[first] = {3, 0};
    // DIAGONAL writes 2 into the first duplicate and zero into its siblings;
    // the stored duplicates still require dependency replay after this update.
    UpdateInput replacement(diagonal, plan.c.type);
    ASSERT_EQ(replacement.Apply(plan, ACL_SPARSE_SPSV_UPDATE_DIAGONAL), kSuccess);
    for (size_t row = 0; row < expected.size(); ++row)
        if (static_cast<int>(row) != first)
            expected[row] *= 0.25;
    VerifyRangeTwice(plan, expected, 1);
    if (::testing::Test::HasFatalFailure())
        return;
    general.ExpectUnchanged();
    replacement.ExpectUnchanged();
}

void RunDuplicateDependencyRegression(const Matrix& matrix, const Case& c)
{
    const bool forward = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? c.lower : !c.lower;
    const int first = forward ? 0 : matrix.m - 1;
    std::vector<Complex> x(matrix.m, {0, 0});
    std::vector<Complex> expected(matrix.m, {0, 0});
    x[first] = {1, 0};
    expected[first] = {1.0 / 3.0, 0};
    // Exact binary residuals: 12288/3=4096, 6144/3=2048. Avoid using
    // FP32 y[first] (or subtracting its rounded product) as the oracle.
    for (int step = 1; step < matrix.m; ++step) {
        if (!DuplicateDependencyActive(step, matrix.m))
            continue;
        const int row = forward ? step : matrix.m - 1 - step;
        x[row] = Value(4096 + 0x1p-11, 2048 + 0x1p-12, c);
        expected[row] = Value(0x1p-11, 0x1p-12, c);
    }
    Plan plan(matrix, c);
    plan.input = Pack(x, c.type);
    plan.alpha = Pack({{1, 0}}, c.type);
    plan.dX.Put(plan.input);
    plan.dAlpha.Put(plan.alpha);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    VerifyRangeTwice(plan, expected, 1);
    if (::testing::Test::HasFatalFailure())
        return;
    VerifyDuplicateDependencyUpdates(plan, first, expected);
    if (::testing::Test::HasFatalFailure())
        return;
    DestroyChecked(plan);
}

TEST_F(SpSVArch22Range, DuplicateDiagonalDependencyLiteralCsr)
{
    Case c;
    Matrix matrix;
    matrix.m = 513;
    matrix.ptr = {0, 1};
    matrix.idx = {0};
    matrix.entries = {{0, 0, {3, 0}}};
    // Construct CSR directly: EncodeEntries reverses each row's storage order.
    for (int row = 1; row < matrix.m; ++row) {
        matrix.idx.insert(matrix.idx.end(), {0, row, row});
        matrix.entries.push_back({row, 0, {12288, 0}});
        matrix.entries.push_back({row, row, {0.5, 0}});
        matrix.entries.push_back({row, row, {0.5, 0}});
        matrix.ptr.push_back(static_cast<int32_t>(matrix.entries.size()));
    }
    ASSERT_EQ(matrix.entries.size(), 1537u);
    ASSERT_EQ(matrix.idx.size(), 1537u);
    ASSERT_EQ(matrix.ptr.size(), 514u);
    ASSERT_EQ(matrix.ptr[0], 0);
    ASSERT_EQ(matrix.entries[0].value, Complex(3, 0));
    for (int row = 1; row < matrix.m; ++row) {
        const int begin = 1 + 3 * (row - 1);
        ASSERT_EQ(matrix.ptr[row], begin);
        ASSERT_EQ(matrix.ptr[row + 1], begin + 3);
        for (int part = 0; part < 3; ++part) {
            ASSERT_EQ(matrix.idx[begin + part], part == 0 ? 0 : row);
            ASSERT_EQ(matrix.entries[begin + part].row, row);
            ASSERT_EQ(matrix.entries[begin + part].col, matrix.idx[begin + part]);
            ASSERT_EQ(matrix.entries[begin + part].value, Complex(part == 0 ? 12288 : 0.5, 0));
        }
    }
    RunDuplicateDependencyRegression(matrix, c);
}

TEST_F(SpSVArch22Range, DuplicateDiagonalDependencyReplayAcrossPathsAndUpdates)
{
    const aclsparseFormat_t formats[] = {ACL_SPARSE_FORMAT_CSR, ACL_SPARSE_FORMAT_CSC, ACL_SPARSE_FORMAT_COO,
                                         ACL_SPARSE_FORMAT_SLICED_ELL};
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        const int components = type == ACL_FLOAT ? 1 : 2;
        const int rows[] = {512 / components, 512 / components + 1, 24576 / components, 24576 / components + 1};
        for (int path = 0; path < 4; ++path) {
            auto c = RangePathConfig(type, static_cast<aclsparseOperation_t>(path % 3), path);
            c.format = formats[path];
            c.base = path & 1;
            if (path == 2)
                c.lower = !c.lower;
            const RangePathCase test{c, rows[path], false};
            SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(c.format));
            RunDuplicateDependencyRegression(MakeDuplicateDependencyMatrix(test), c);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

#endif // SPSV_ARCH22_DEPENDENCY_TESTS_H_
