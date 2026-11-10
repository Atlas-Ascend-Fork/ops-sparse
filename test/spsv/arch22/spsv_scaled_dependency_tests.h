/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_SCALED_DEPENDENCY_TESTS_H_
#define SPSV_ARCH22_SCALED_DEPENDENCY_TESTS_H_

// Included after the range/dependency helpers inside the test anonymous namespace.
void VerifyScaledDependency(Plan& plan, const std::vector<Complex>& expected, bool exactAll = true)
{
    plan.VerifyTwice(expected);
    if (::testing::Test::HasFatalFailure())
        return;
    const auto actual = (plan.c.inPlace ? plan.dX : plan.dY).Get<float>(plan.input.size());
    const auto rounded = Pack(expected, plan.c.type);
    ASSERT_EQ(actual.size(), rounded.size());
    const size_t components = plan.c.type == ACL_COMPLEX64 ? 2 : 1;
    const bool forward = plan.c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? plan.c.lower : !plan.c.lower;
    const size_t finalRow = forward ? expected.size() - 1 : 0;
    for (size_t part = 0; part < rounded.size(); ++part) {
        // VerifyTwice checks every public output against the task tolerance.
        // Residual recovery additionally requires an exact amplified endpoint;
        // tiny public intermediates may differ while their internal tails survive.
        if (!exactAll && part / components != finalRow)
            continue;
        SCOPED_TRACE("component=" + std::to_string(part));
        // Includes the sign of Inf; finite dependent rows must remain finite.
        if (std::isnan(rounded[part]))
            EXPECT_TRUE(std::isnan(actual[part]));
        else
            EXPECT_EQ(actual[part], rounded[part]);
    }
}

void RunScaledDependencyStages(const Matrix& matrix, const Case& c, const std::vector<Complex>& x, Complex alpha,
                               const std::vector<Complex>& diagonal, const std::vector<Complex>& expected,
                               bool exactAll = true)
{
    Plan plan(matrix, c);
    plan.input = Pack(x, c.type);
    plan.alpha = Pack({alpha}, c.type);
    plan.dX.Put(plan.input);
    plan.dAlpha.Put(plan.alpha);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    VerifyScaledDependency(plan, expected, exactAll);
    if (::testing::Test::HasFatalFailure())
        return;
    ASSERT_EQ(aclsparseSpSV_updateMatrix(SpSVArch22Test::handle, plan.state, plan.dValues.get(),
                                         ACL_SPARSE_SPSV_UPDATE_GENERAL),
              kSuccess);
    VerifyScaledDependency(plan, expected, exactAll);
    if (::testing::Test::HasFatalFailure())
        return;
    const auto packed = Pack(diagonal, c.type);
    DeviceBuffer values(packed.size() * sizeof(float));
    values.Put(packed);
    ASSERT_EQ(
        aclsparseSpSV_updateMatrix(SpSVArch22Test::handle, plan.state, values.get(), ACL_SPARSE_SPSV_UPDATE_DIAGONAL),
        kSuccess);
    VerifyScaledDependency(plan, expected, exactAll);
    CheckAcl(aclrtSynchronizeStream(SpSVArch22Test::env->stream()), "scaled dependency stages complete");
}

void RunScaledZeroDependency(const RangePathCase& test, int exponent)
{
    const auto& c = test.config;
    const bool forward = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? c.lower : !c.lower;
    const int first = forward ? 0 : test.rows - 1;
    const int second = forward ? 1 : test.rows - 2;
    const double power = std::ldexp(1.0, exponent);
    const double source = exponent == 12 ? power + 0x1p-11 : power;
    const Complex residual = exponent == 12 ? Complex{0x1p-11, 0} : SensitiveDependencyResidual(power, c.type);
    std::vector<Entry> entries;
    std::vector<Complex> x(test.rows, {0, 0});
    std::vector<Complex> expected(test.rows, {0, 0});
    std::vector<Complex> diagonal(test.rows, {1, 0});
    diagonal[first] = {3, 0};
    x[first] = {1, 0};
    expected[first] = {1.0 / 3.0, 0};
    for (int step = 0; step < test.rows; ++step) {
        const int row = forward ? step : test.rows - 1 - step;
        entries.push_back(RangeStoredEntry(row, row, diagonal[row], c));
        if (step >= 2 && step <= 512) {
            entries.push_back(RangeStoredEntry(row, first, {3 * power, 0}, c));
            entries.push_back(RangeStoredEntry(row, second, {0x1p-60, 0}, c));
            x[row] = {source, 0};
            expected[row] = residual;
        }
    }
    RunScaledDependencyStages(EncodeEntries(test.rows, entries, c), c, x, {1, 0}, diagonal, expected);
}

void RunOverflowDependency(const RangePathCase& test, int exponent, bool negative)
{
    const auto& c = test.config;
    const bool forward = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? c.lower : !c.lower;
    const int first = forward ? 0 : test.rows - 1;
    const double sign = negative ? -1.0 : 1.0;
    const bool imaginary = c.type == ACL_COMPLEX64 && negative;
    const auto component = [imaginary](double value) { return imaginary ? Complex{0, value} : Complex{value, 0}; };
    std::vector<Entry> entries;
    std::vector<Complex> diagonal(test.rows, {1, 0});
    diagonal[first] = {std::ldexp(1.0, 100 - exponent), 0};
    std::vector<Complex> x(test.rows, {0, 0});
    std::vector<Complex> expected(test.rows, {0, 0});
    x[first] = component(0x1p50);
    const double publicFirst = exponent > 127 ? std::numeric_limits<double>::infinity() : std::ldexp(1.0, exponent);
    expected[first] = component(sign * publicFirst);
    for (int step = 0; step < test.rows; ++step) {
        const int row = forward ? step : test.rows - 1 - step;
        entries.push_back(RangeStoredEntry(row, row, diagonal[row], c));
        if (step == 1 || step == 2) {
            entries.push_back(RangeStoredEntry(row, first, {0x1p-100, 0}, c));
            expected[row] = component(-sign * std::ldexp(1.0, exponent - 100));
        }
    }
    RunScaledDependencyStages(EncodeEntries(test.rows, entries, c), c, x, {sign * 0x1p50, 0}, diagonal, expected);
}

TEST_F(SpSVArch22Range, RangeProtectionDoesNotSkipDependencyBudget)
{
    Case c;
    RunScaledZeroDependency({c, 513, false}, 12);
}

TEST_F(SpSVArch22Range, FiniteOverflowDependenciesKeepExponent)
{
    Case c;
    RunOverflowDependency({c, 3, false}, 150, false);
}

void AppendScaledDependencyPaths(std::vector<RangePathCase>& tests, aclDataType type, int rows, int path,
                                 aclsparseOperation_t op)
{
    constexpr aclsparseFormat_t formats[] = {ACL_SPARSE_FORMAT_CSR, ACL_SPARSE_FORMAT_CSC, ACL_SPARSE_FORMAT_COO,
                                             ACL_SPARSE_FORMAT_SLICED_ELL};
    for (int variant = 0; variant < 2; ++variant) {
        auto c = RangePathConfig(type, op, path + variant);
        c.format = formats[(path + static_cast<int>(op) + variant) % 4];
        c.base = variant;
        c.inPlace = variant != 0;
        c.deviceAlpha = variant != 0;
        if (variant != 0)
            c.lower = !c.lower;
        tests.push_back({c, rows, false});
    }
}

std::vector<RangePathCase> ScaledDependencyPaths()
{
    std::vector<RangePathCase> tests;
    for (const auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        const int rows[] = {3, type == ACL_FLOAT ? 513 : 257, type == ACL_FLOAT ? 24577 : 12289};
        for (int path = 0; path < 3; ++path) {
            for (const auto op : kOperations) {
                AppendScaledDependencyPaths(tests, type, rows[path], path, op);
            }
        }
    }
    return tests;
}

TEST_F(SpSVArch22Range, ScaledDependencyBudgetCoversStoragePaths)
{
    for (const auto& test : ScaledDependencyPaths()) {
        SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(test.config.format));
        RunScaledZeroDependency(test, test.rows == 3 ? 44 : 12);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

TEST_F(SpSVArch22Range, FiniteOverflowDependenciesCoverStoragePaths)
{
    for (const auto& test : ScaledDependencyPaths()) {
        SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(test.config.format));
        for (const int exponent : {127, 128, 150}) {
            SCOPED_TRACE("exponent=" + std::to_string(exponent));
            RunOverflowDependency(test, exponent, test.config.inPlace);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

void RunSmallScaledStages(const Case& c, std::vector<Entry> entries, std::vector<Complex> x, Complex alpha,
                          std::vector<Complex> diagonal, std::vector<Complex> expected, bool exactAll = true)
{
    const int rows = static_cast<int>(x.size());
    const bool forward = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? c.lower : !c.lower;
    if (!forward) {
        for (auto& entry : entries) {
            entry.row = rows - 1 - entry.row;
            entry.col = rows - 1 - entry.col;
        }
        std::reverse(x.begin(), x.end());
        std::reverse(diagonal.begin(), diagonal.end());
        std::reverse(expected.begin(), expected.end());
    }
    for (auto& entry : entries)
        entry = RangeStoredEntry(entry.row, entry.col, entry.value, c);
    RunScaledDependencyStages(EncodeEntries(rows, entries, c), c, x, alpha, diagonal, expected, exactAll);
}

void RunScaledCancellationRegression(const Case& c)
{
    const Complex first = c.type == ACL_FLOAT ? Complex{0x1p150, 0} : Complex{0x1p150, 0x1p150};
    const Complex residual = c.type == ACL_FLOAT ? Complex{1, 0} : Complex{1, -1};
    const Complex coefficient = c.type == ACL_FLOAT ? Complex{1, 0} : Complex{1, 1};
    std::vector<Entry> entries{{0, 0, {0x1p-50, 0}}, {1, 0, {-1, 0}},      {1, 1, {1, 0}},
                               {2, 0, coefficient},  {2, 1, -coefficient}, {2, 2, {1, 0}}};
    std::vector<Complex> x{first * 0x1p-100, {0, 0}, residual * 0x1p-50};
    const double infinity = std::numeric_limits<double>::infinity();
    const Complex overflow = c.type == ACL_FLOAT ? Complex{infinity, 0} : Complex{infinity, infinity};
    RunSmallScaledStages(c, entries, x, {0x1p50, 0}, {{0x1p-50, 0}, {1, 0}, {1, 0}}, {overflow, overflow, residual});
}

void RunUnitOverflowRegression(Case c)
{
    c.unit = true;
    const Complex value = c.type == ACL_FLOAT ? Complex{1, 0} : Complex{1, 1};
    std::vector<Entry> entries{
        {0, 0, {5, 0}}, {1, 0, {-0x1p50, 0}}, {1, 1, {7, 0}}, {2, 1, {0x1p-100, 0}}, {2, 2, {-3, 0}}};
    const double infinity = std::numeric_limits<double>::infinity();
    const Complex overflow = c.type == ACL_FLOAT ? Complex{infinity, 0} : Complex{infinity, infinity};
    RunSmallScaledStages(c, entries, {value * 0x1p50, {0, 0}, {0, 0}}, {0x1p50, 0}, {{2, 0}, {-7, 0}, {3, 0}},
                         {value * 0x1p100, overflow, -value * 0x1p50});
}

TEST_F(SpSVArch22Range, ScaledFastResidualRetainsCancellation)
{
    for (const auto& test : ScaledDependencyPaths()) {
        if (test.rows != 3)
            continue;
        SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(test.config.format));
        RunScaledCancellationRegression(test.config);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

TEST_F(SpSVArch22Range, ScaledUnitOverflowKeepsFiniteDependents)
{
    for (const auto& test : ScaledDependencyPaths()) {
        if (test.rows != 3)
            continue;
        SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(test.config.format));
        RunUnitOverflowRegression(test.config);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

void RunScaledLowBitRegression(const RangePathCase& test)
{
    const int rows = std::max(4, test.rows);
    std::vector<Entry> entries{{1, 1, {1, 0}},        {2, 1, {-1, 0}},      {2, 1, {-0x1p-25, 0}},
                               {2, 1, {-0x1p-50, 0}}, {2, 1, {0x1p-25, 0}}, {2, 2, {1, 0}},
                               {3, 1, {1, 0}},        {3, 2, {-1, 0}},      {3, 3, {1, 0}}};
    std::vector<Complex> diagonal(rows, {1, 0}), x(rows, {0, 0}), expected(rows, {0, 0});
    diagonal[0] = {0x1p50, 0};
    entries.push_back({0, 0, diagonal[0]});
    for (int row = 4; row < rows; ++row)
        entries.push_back({row, row, diagonal[row]});
    const double sign = test.config.inPlace ? -1 : 1;
    const Complex value = test.config.type == ACL_FLOAT ? Complex{sign, 0} : Complex{sign, -sign};
    x[0] = value * 0x1p-50;
    x[1] = value * 0x1p100;
    expected[1] = expected[2] = value * 0x1p50;
    expected[3] = value;
    RunSmallScaledStages(test.config, entries, x, {0x1p-50, 0}, diagonal, expected);
}

void RunWideExponentRegression(const RangePathCase& test)
{
    const int rows = std::max(24, test.rows);
    std::vector<Entry> entries{{1, 0, {0x1p60, 0}}};
    std::vector<Complex> diagonal(rows, {1, 0}), x(rows, {0, 0}), expected(rows, {0, 0});
    const double sign = test.config.inPlace ? -1 : 1;
    const Complex value = test.config.type == ACL_FLOAT ? Complex{sign, 0} : Complex{sign, -sign};
    x[0] = expected[0] = x[2] = value;
    x[1] = value * 0x1p60;
    for (int row = 0; row < rows; ++row) {
        if (row >= 2 && row <= 23) {
            diagonal[row] = {row <= 12 ? 0x1p100 : 0x1p-100, 0};
            const int exponent = row <= 12 ? -100 * (row - 1) : -1100 + 100 * (row - 12);
            expected[row] = value * (exponent < -149 ? 0 : std::ldexp(1.0, exponent));
            if (row >= 3)
                entries.push_back({row, row - 1, {-1, 0}});
        }
        entries.push_back({row, row, diagonal[row]});
    }
    // The analytic tail is exactly one; a binary64 reference would itself underflow.
    RunSmallScaledStages(test.config, entries, x, {1, 0}, diagonal, expected);
}

TEST_F(SpSVArch22Range, ScaledDependencyReplayRetainsLowBit) { RunScaledLowBitRegression({Case{}, 4, false}); }

TEST_F(SpSVArch22Range, WideReplayRetainsExponentBeyondBinary64) { RunWideExponentRegression({Case{}, 24, false}); }

TEST_F(SpSVArch22Range, ScaledLowBitReplayCoversStoragePaths)
{
    for (const auto& test : ScaledDependencyPaths()) {
        SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(test.config.format));
        RunScaledLowBitRegression(test);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

TEST_F(SpSVArch22Range, WideExponentReplayCoversStoragePaths)
{
    for (const auto& test : ScaledDependencyPaths()) {
        SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(test.config.format));
        RunWideExponentRegression(test);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

#include "spsv_tail_dependency_tests.h"
#endif // SPSV_ARCH22_SCALED_DEPENDENCY_TESTS_H_
