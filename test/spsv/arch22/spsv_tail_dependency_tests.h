/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_TAIL_DEPENDENCY_TESTS_H_
#define SPSV_ARCH22_TAIL_DEPENDENCY_TESTS_H_

// Uses the range and lifecycle fixtures in the enclosing anonymous namespace.
void RunAmplifiedTail(const RangePathCase& test, int gap, bool scaled)
{
    const int rows = std::max(6, test.rows);
    const double small = std::ldexp(1.0, -gap), large = std::ldexp(1.0, gap);
    const double sign = test.config.inPlace ? -1 : 1;
    const Complex value = test.config.type == ACL_FLOAT ? Complex{sign, 0} : Complex{sign, -sign};
    std::vector<Complex> diagonal(rows, {1, 0}), x(rows, {0, 0}), expected(rows, {0, 0});
    std::vector<Entry> entries{
        {2, 1, {-small, 0}}, {3, 1, {1, 0}}, {3, 2, {-1, 0}}, {4, 3, {-large, 0}}, {5, 4, {-1, 0}}};
    diagonal[0] = scaled ? Complex{0x1p50, 0} : Complex{1, 0};
    x[0] = scaled ? value * 0x1p-100 : Complex{0, 0};
    expected[0] = test.config.unit ? x[0] : Complex{0, 0};
    x[1] = x[2] = expected[1] = expected[2] = value;
    expected[3] = value * small;
    expected[4] = expected[5] = value;
    // 2^-150 rounds to zero publicly, but remains a live internal dependency.
    for (int row = 0; row < rows; ++row)
        entries.push_back({row, row, diagonal[row]});
    RunSmallScaledStages(test.config, entries, x, {1, 0}, diagonal, expected);
}

TEST_F(SpSVArch22Range, ScaledAlignmentTailSurvivesDownstreamAmplification)
{
    for (const int gap : {99, 100, 101, 110, 125}) {
        SCOPED_TRACE("gap=" + std::to_string(gap));
        RunAmplifiedTail({Case{}, 6, false}, gap, true);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

TEST_F(SpSVArch22Range, AmplifiedTailCoversStorageAndLifecyclePaths)
{
    for (const auto& test : ScaledDependencyPaths()) {
        SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(test.config.format));
        RunAmplifiedTail(test, 110, true);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

TEST_F(SpSVArch22Range, OrdinaryCancellationReplaysUnroundedPrefix)
{
    for (const auto& test : ScaledDependencyPaths()) {
        SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(test.config.format));
        RunAmplifiedTail(test, 25, false);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

void RunSeparatedExponentTail(const Case& c, int levels)
{
    const int combined = levels + 1, residual = levels + 2, rows = 2 * levels + 3;
    const Complex value = c.type == ACL_FLOAT ? Complex{1, 0} : Complex{1, -1};
    std::vector<Complex> diagonal(rows, {1, 0}), x(rows, {0, 0}), expected(rows, {0, 0});
    std::vector<Entry> entries{{combined, levels, {-1, 0}}, {residual, combined, {-1, 0}}, {residual, 0, {1, 0}}};
    x[0] = expected[0] = x[combined] = expected[combined] = value;
    for (int row = 0; row < rows; ++row) {
        if (row > 0 && row <= levels) {
            diagonal[row] = {0x1p100, 0};
            entries.push_back({row, row - 1, {-1, 0}});
            expected[row] = row == 1 ? value * 0x1p-100 : Complex{0, 0};
        } else if (row > residual) {
            diagonal[row] = {0x1p-100, 0};
            entries.push_back({row, row - 1, {-1, 0}});
            const int remaining = rows - 1 - row;
            expected[row] = remaining <= 1 ? value * std::ldexp(1.0, -100 * remaining) : Complex{0, 0};
        }
        entries.push_back({row, row, diagonal[row]});
    }
    RunSmallScaledStages(c, entries, x, {1, 0}, diagonal, expected);
}

TEST_F(SpSVArch22Range, IndependentTailExponentSurvivesMultipleWindows)
{
    for (const auto& test : ScaledDependencyPaths()) {
        if (test.rows != 3)
            continue;
        SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(test.config.format));
        for (const int levels : {7, 11, 19, 20, 21, 79, 80, 81, 82}) {
            RunSeparatedExponentTail(test.config, levels);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

TEST_F(SpSVArch22Range, UnitIterationFallsBackBeforeAmplifiedCancellation)
{
    for (auto test : ScaledDependencyPaths()) {
        if (test.rows != 3)
            continue;
        test.rows = 33;
        test.config.unit = true;
        SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(test.config.format));
        RunAmplifiedTail(test, 25, false);
        RunAmplifiedTail(test, 110, true);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}
void RunComplexWindowCarry(const Case& c)
{
    constexpr int rows = 37, combined = 35, last = 36;
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<Complex> diagonal(rows, {0x1p-100, 0}), x(rows, {0, 0}), expected(rows, {infinity, infinity});
    std::vector<Entry> entries;
    diagonal[5] = {0x1p-63, 0};
    diagonal[combined] = diagonal[last] = {1, 0};
    x[0] = {1.5, 1.5};
    expected[0] = {1.5 * 0x1p100, 1.5 * 0x1p100};
    for (int row = 0; row < rows; ++row) {
        entries.push_back({row, row, diagonal[row]});
        if (row > 0 && row < combined)
            entries.push_back({row, row == 6 ? 0 : row - 1, {-1, 0}});
    }
    entries.push_back({combined, 5, {1.5, -1.5}});
    entries.push_back({combined, 34, {1, 0}});
    entries.push_back({combined, 34, {-1, 0}});
    entries.push_back({last, combined, {-1, 0}});
    entries.push_back({last, 5, {-1.5, 1.5}});
    x[combined] = {1, 0};
    expected[combined] = {-infinity, 0};
    expected[last] = {1, 0};
    RunSmallScaledStages(c, entries, x, {1, 0}, diagonal, expected);
}

TEST_F(SpSVArch22Range, ComplexProductCarryCrossesIntegerWindow)
{
    for (const auto& test : ScaledDependencyPaths()) {
        if (test.rows != 3 || test.config.type != ACL_COMPLEX64)
            continue;
        SCOPED_TRACE(RangePathName(test) + "_format=" + std::to_string(test.config.format));
        RunComplexWindowCarry(test.config);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}
void RunGrowingTail(const Case& c, int levels)
{
    const int rows = 2 * levels + 1, last = rows - 1;
    const double infinity = std::numeric_limits<double>::infinity();
    const Complex value = c.type == ACL_FLOAT ? Complex{1, 0} : Complex{1, -1};
    std::vector<Complex> diagonal(rows, {1, 0}), x(rows, {0, 0}), expected(rows, {0, 0});
    std::vector<Entry> entries;
    for (int row = 0; row < rows; ++row) {
        entries.push_back({row, row, {1, 0}});
        if (row == last)
            continue;
        const int power = row < levels ? row + 1 : row - levels + 1;
        const double result = power < 128 ? std::ldexp(1.0, power) - (row < levels ? 1 : 0) : infinity;
        expected[row] = c.type == ACL_FLOAT ? Complex{result, 0} : Complex{result, -result};
        if (row < levels)
            x[row] = value;
        if (row != 0 && row != levels)
            entries.push_back({row, row - 1, {-2, 0}});
    }
    x[levels] = value * 2.0;
    entries.push_back({last, levels - 1, {1, 0}});
    entries.push_back({last, last - 1, {-1, 0}});
    expected[last] = value;
    RunSmallScaledStages(c, entries, x, {1, 0}, diagonal, expected);
}

TEST_F(SpSVArch22Range, ExponentialGrowthTailSurvivesEmptyWindows)
{
    RunGrowingTail(Case{}, 4096);
    Case c;
    c.type = ACL_COMPLEX64;
    c.operation = ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    c.base = 1;
    c.inPlace = true;
    c.deviceAlpha = true;
    RunGrowingTail(c, 1024);
}
struct LayeredDependency {
    std::vector<Entry> entries;
    std::vector<Complex> x, diagonal, expected;
    LayeredDependency(const Case& c, int tails)
        : x(3 * tails + 3, {0, 0}), diagonal(x.size(), {1, 0}), expected(x.size(), {0, 0})
    {
        const int combined = tails + 2;
        const Complex value = c.type == ACL_FLOAT ? Complex{1, 0} : Complex{1, -1};
        x[0] = value * 0x1p-100;
        diagonal[0] = {0x1p50, 0};
        x[1] = x[combined] = expected[1] = expected[combined] = value;
        for (int row = 0; row < static_cast<int>(x.size()); ++row) {
            AddLeadingRow(row, combined, value);
            if (row == combined)
                AddCombinedRow(row);
            if (row > combined)
                AddFollowingRow(row, combined, tails, value);
            entries.push_back({row, row, diagonal[row]});
        }
    }

private:
    void AddLeadingRow(int row, int combined, const Complex& value)
    {
        if (row < 2 || row >= combined)
            return;
        entries.push_back({row, row - 1, {-1, 0}});
        diagonal[row] = {0x1p100, 0};
        if (row == 2)
            expected[row] = value * 0x1p-100;
    }
    void AddCombinedRow(int row)
    {
        for (int col = 2; col < row; ++col)
            entries.push_back({row, col, {-1, 0}});
    }
    void AddFollowingRow(int row, int combined, int tails, const Complex& value)
    {
        entries.push_back({row, row - 1, {-1, 0}});
        if (row <= combined + tails) {
            entries.push_back({row, row - combined, {1, 0}});
            if (row == combined + 1)
                expected[row] = value * 0x1p-100;
            return;
        }
        diagonal[row] = {0x1p-100, 0};
        const int remaining = static_cast<int>(x.size()) - 1 - row;
        if (remaining <= 1)
            expected[row] = value * std::ldexp(1.0, -100 * remaining);
    }
};
void VerifyLayeredDependency(const Case& c, const LayeredDependency& data)
{
    RunSmallScaledStages(c, data.entries, data.x, {1, 0}, data.diagonal, data.expected);
}
TEST_F(SpSVArch22Range, RepeatedCancellationRecoversSeparatedResidualLayers)
{
    for (const auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        Case c;
        c.type = type;
        c.format = ACL_SPARSE_FORMAT_COO;
        c.operation = type == ACL_FLOAT ? ACL_SPARSE_OP_NON_TRANSPOSE : ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
        c.base = type == ACL_COMPLEX64 ? 1 : 0;
        c.inPlace = type == ACL_COMPLEX64;
        c.deviceAlpha = c.inPlace;
        for (const int tails : {2, 3, 4, 6, 8, 12}) {
            SCOPED_TRACE("tails=" + std::to_string(tails) + "_dtype=" + std::to_string(type));
            VerifyLayeredDependency(c, LayeredDependency(c, tails));
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}
TEST_F(SpSVArch22Range, DependencyRecoveryPreservesUnrelatedRoundedAndSingularRows)
{
    for (const auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        Case c;
        c.type = type;
        LayeredDependency data(c, 4);
        for (auto& entry : data.entries) {
            entry.row += 2;
            entry.col += 2;
        }
        data.entries.insert(data.entries.end(), {{0, 0, {3, 0}}, {1, 0, {3 * 0x1p30, 0}}, {1, 1, {1, 0}}});
        data.x.insert(data.x.begin(), {{1, 0}, {0x1p30, 0}});
        data.diagonal.insert(data.diagonal.begin(), {{3, 0}, {1, 0}});
        data.expected.insert(data.expected.begin(), {{1.0 / 3.0, 0}, SensitiveDependencyResidual(0x1p30, type)});
        VerifyLayeredDependency(c, data);
    }
    Case c;
    LayeredDependency data(c, 4);
    data.x[0] = data.diagonal[0] = {0, 0};
    data.entries[0].value = {0, 0};
    data.expected[0] = {std::numeric_limits<double>::quiet_NaN(), 0};
    VerifyLayeredDependency(c, data);
}
std::vector<float> CoupledRecoveryOutput(Complex firstDiagonal, Complex lastDiagonal, int power, bool extraTail)
{
    Case c;
    c.type = ACL_COMPLEX64;
    const double scale = std::ldexp(1.0, power);
    std::vector<Entry> entries{{0, 0, {0x1p50, 0}},           {1, 1, firstDiagonal}, {2, 2, {1, 0}},
                               {6, 1, firstDiagonal * scale}, {6, 3, {-1, 0}},       {6, 4, {-1, 0}},
                               {6, 6, lastDiagonal}};
    for (int row = 3; row <= 5; ++row) {
        entries.push_back({row, row - 1, {-1, 0}});
        entries.push_back({row, row, {0x1p100, 0}});
    }
    if (extraTail)
        entries.push_back({6, 5, {-1, 0}});
    const auto matrix = EncodeEntries(7, entries, c);
    Plan plan(matrix, c);
    plan.input = Pack({{0x1p-100, 0}, {1, 1}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {scale, scale}}, c.type);
    plan.alpha = Pack({{1, 0}}, c.type);
    plan.dX.Put(plan.input);
    plan.dAlpha.Put(plan.alpha);
    CheckSparse(plan.Analyze(true), "coupled recovery analysis");
    CheckSparse(plan.Solve(c.operation), "coupled recovery solve");
    CheckAcl(aclrtSynchronizeStream(SpSVArch22Test::env->stream()), "coupled recovery complete");
    return plan.dY.Get<float>(plan.input.size());
}
TEST_F(SpSVArch22Range, ResidualRecoveryPreservesComplexComponentRounding)
{
    struct Scenario {
        Complex first;
        Complex last;
        int power;
    };
    const Scenario scenarios[] = {{{3, 1}, {2, -3}, 44}, {{3, 1}, {2, -3}, 60}, {{5, 2}, {3, 1}, 44},
                                  {{5, 2}, {3, 1}, 60},  {{7, 4}, {5, 2}, 44},  {{7, 4}, {5, 2}, 60}};
    for (const auto& scenario : scenarios) {
        SCOPED_TRACE("power=" + std::to_string(scenario.power));
        const auto baseline = CoupledRecoveryOutput(scenario.first, scenario.last, scenario.power, false);
        const auto recovered = CoupledRecoveryOutput(scenario.first, scenario.last, scenario.power, true);
        ASSERT_EQ(baseline.size(), recovered.size());
        for (size_t component = 0; component < baseline.size(); ++component) {
            const double tolerance = 0x1p-16 + 0x1p-10 * std::abs(baseline[component]);
            EXPECT_LE(std::abs(recovered[component] - baseline[component]), tolerance);
        }
    }
}

float UnitChainValue(uint32_t& seed)
{
    seed = seed * 1664525u + 1013904223u;
    return static_cast<float>(seed & 0xffffffu) * 0x1p-23f - 1;
}
struct UnitChainData {
    std::vector<Entry> entries;
    std::vector<Complex> x, diagonal, expected;
    UnitChainData(aclDataType type, int rows) : x(rows), diagonal(rows, {1, 0}), expected(rows)
    {
        uint32_t seed = 0x53505631u;
        const Complex alpha = type == ACL_FLOAT ? Complex{1.25, 0} : Complex{1.25, 0.5};
        std::vector<Complex> unrounded(rows);
        for (int row = 0; row < rows; ++row) {
            const float real = UnitChainValue(seed);
            x[row] = {real, type == ACL_FLOAT ? 0 : UnitChainValue(seed)};
            unrounded[row] = alpha * x[row];
            for (int col = std::max(0, row - 6); col < row; ++col) {
                const float value = UnitChainValue(seed);
                const Complex coefficient{value, type == ACL_FLOAT ? 0 : UnitChainValue(seed)};
                entries.push_back({row, col, coefficient});
                unrounded[row] -= coefficient * unrounded[col];
            }
            entries.push_back({row, row, {1, 0}});
            expected[row] = {static_cast<float>(unrounded[row].real()), static_cast<float>(unrounded[row].imag())};
        }
    }
};
TEST_F(SpSVArch22Range, BoundedUnitPrecisionHandlesLongSignedDependencyChains)
{
    for (const auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        Case c;
        c.type = type;
        c.unit = true;
        c.format = ACL_SPARSE_FORMAT_COO;
        c.operation = type == ACL_FLOAT ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
        c.lower = false;
        c.base = 1;
        c.inPlace = c.deviceAlpha = true;
        const UnitChainData data(type, type == ACL_FLOAT ? 3584 : 1280);
        const Complex alpha = type == ACL_FLOAT ? Complex{1.25, 0} : Complex{1.25, 0.5};
        RunSmallScaledStages(c, data.entries, data.x, alpha, data.diagonal, data.expected, false);
    }
}

void RunUnitSourceBits(const Case& c, double alpha)
{
    const int rows = c.type == ACL_FLOAT ? 513 : 257;
    const Complex value = c.type == ACL_FLOAT ? Complex{1, 0} : Complex{1, -1};
    std::vector<Complex> x(rows, {0, 0}), diagonal(rows, {1, 0}), expected(rows, {0, 0});
    std::vector<Entry> entries{{2, 0, {-1, 0}}, {3, 1, {1, 0}}, {3, 2, {-1, 0}}, {4, 3, {-0x1p100, 0}}};
    x[0] = value * 0x1p-100;
    x[1] = x[2] = value;
    x[5] = value * (alpha == 1 ? 0x1p-149 : 0x1p-100);
    expected[0] = expected[3] = value * (alpha * 0x1p-100);
    expected[1] = expected[2] = expected[4] = value * alpha;
    expected[5] = value * 0x1p-149;
    for (int row = 0; row < rows; ++row)
        entries.push_back({row, row, {1, 0}});
    RunSmallScaledStages(c, entries, x, {alpha, 0}, diagonal, expected);
}

TEST_F(SpSVArch22Range, UnitSourceProductsRetainSubnormalAndAmplifiedBits)
{
    for (const auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        Case c;
        c.type = type;
        c.unit = true;
        c.inPlace = c.deviceAlpha = true;
        c.format = ACL_SPARSE_FORMAT_COO;
        c.operation = type == ACL_FLOAT ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
        for (double alpha : {1.0, 0x1p-49}) {
            RunUnitSourceBits(c, alpha);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}
#endif // SPSV_ARCH22_TAIL_DEPENDENCY_TESTS_H_
