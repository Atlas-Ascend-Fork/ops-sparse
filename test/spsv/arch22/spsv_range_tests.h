/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_RANGE_TESTS_H_
#define SPSV_ARCH22_RANGE_TESTS_H_

// Included inside spsv_test.cpp's anonymous namespace to reuse its device Plan.
void VerifyRangeComponents(const std::vector<float> &actual, const std::vector<Complex> &expected,
    aclDataType type, int maxUlps)
{
    const auto rounded = Pack(expected, type);
    ASSERT_EQ(actual.size(), rounded.size());
    for (size_t component = 0; component < rounded.size(); ++component) {
        SCOPED_TRACE("scalar_component=" + std::to_string(component));
        ASSERT_TRUE(std::isfinite(rounded[component]));
        ASSERT_TRUE(std::isfinite(actual[component]));
        if (maxUlps == 0 || rounded[component] == 0.0f) {
            EXPECT_EQ(actual[component], rounded[component]);
            continue;
        }
        const float next = std::nextafter(rounded[component], std::numeric_limits<float>::infinity());
        const double ulp = std::abs(static_cast<double>(next) - rounded[component]);
        EXPECT_LE(std::abs(static_cast<double>(actual[component]) - rounded[component]), maxUlps * ulp);
    }
}

void VerifyRangeTwice(Plan &plan, const std::vector<Complex> &expected, int maxUlps)
{
    std::vector<float> first;
    for (int repeat = 0; repeat < 2; ++repeat) {
        plan.dX.Put(plan.input);
        const auto solved = plan.Solve(plan.c.operation);
        const auto synced = aclrtSynchronizeStream(SpSVArch22Test::env->stream());
        ASSERT_EQ(solved, kSuccess);
        ASSERT_EQ(synced, ACL_SUCCESS);
        const auto actual = (plan.c.inPlace ? plan.dX : plan.dY).Get<float>(plan.input.size());
        VerifyRangeComponents(actual, expected, plan.c.type, maxUlps);
        if (::testing::Test::HasFatalFailure()) return;
        if (repeat == 0) first = actual;
        else ExpectBitwiseFloats(actual, first);
        if (!plan.c.inPlace) ExpectBitwiseFloats(plan.dX.Get<float>(plan.input.size()), plan.input);
    }
    EXPECT_EQ(plan.dPtr.Get<int32_t>(plan.a.ptr.size()), plan.a.ptr);
    EXPECT_EQ(plan.dIdx.Get<int32_t>(plan.a.idx.size()), plan.a.idx);
    ExpectBitwiseFloats(plan.dValues.Get<float>(plan.initial.size()), plan.initial);
    ExpectBitwiseFloats(plan.dAlpha.Get<float>(plan.alpha.size()), plan.alpha);
}

void RunRangeRegression(const Matrix &matrix, const Case &c, const std::vector<Complex> &x,
    Complex alpha, const std::vector<Complex> &expected, int maxUlps = 0)
{
    Plan plan(matrix, c);
    plan.input = Pack(x, c.type);
    plan.alpha = Pack({alpha}, c.type);
    plan.dX.Put(plan.input);
    plan.dAlpha.Put(plan.alpha);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    VerifyRangeTwice(plan, expected, maxUlps);
}

Entry RangeStoredEntry(int row, int col, Complex effectiveValue, const Case &c)
{
    if (c.operation == ACL_SPARSE_OP_NON_TRANSPOSE) return {row, col, effectiveValue};
    if (c.operation == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) effectiveValue = std::conj(effectiveValue);
    return {col, row, effectiveValue};
}

class SpSVArch22Range : public SpSVArch22Test {};

void RunSmithRangeRegression(int branch, int signs, aclsparseOperation_t operation, bool deviceAlpha)
{
    Case c;
    c.type = ACL_COMPLEX64;
    c.operation = operation;
    c.deviceAlpha = deviceAlpha;
    c.inPlace = (signs & 1) != 0;
    const double large = branch == 0 ? std::ldexp(1.0, 127) :
        static_cast<double>(std::numeric_limits<float>::max());
    const double real = (signs & 1 ? -1.0 : 1.0) * (branch == 2 ? large / 2.0 : large);
    const double imag = (signs & 2 ? -1.0 : 1.0) * (branch == 1 ? large / 2.0 : large);
    const Complex diagonal{real, imag};
    const Complex expected = operation == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE ?
        diagonal / std::conj(diagonal) : Complex{1.0, 0.0};
    SCOPED_TRACE("Smith branch=" + std::to_string(branch) + " signs=" + std::to_string(signs) +
        " operation=" + std::to_string(operation) + " device_alpha=" + std::to_string(deviceAlpha));
    // A=X=2^127+i*2^127 gives N/T=1 and H=i. Using FLT_MAX for
    // the strict-dominance branches also overflows their unscaled denominator.
    const auto matrix = EncodeEntries(1, {{0, 0, diagonal}}, c);
    RunRangeRegression(matrix, c, {diagonal}, {1.0, 0.0}, {expected}, 2);
}

TEST_F(SpSVArch22Range, ComplexSmithBranchesKeepFiniteHighExponentQuotients)
{
    for (int branch = 0; branch < 3; ++branch) {
        // One equal-magnitude anchor, then both Smith branches in all quadrants.
        for (int signs = 0; signs < (branch == 0 ? 1 : 4); ++signs) {
            for (auto operation : kOperations) {
                for (bool deviceAlpha : {false, true}) {
                    RunSmithRangeRegression(branch, signs, operation, deviceAlpha);
                    if (::testing::Test::HasFatalFailure()) return;
                }
            }
        }
    }
}

struct RangePathCase {
    Case config;
    int rows;
    bool diagonal;
};

Case RangePathConfig(aclDataType type, aclsparseOperation_t operation, int variant)
{
    Case c;
    c.type = type;
    c.operation = operation;
    // All fixtures have an effective lower triangle after N/T/H normalization.
    c.lower = operation == ACL_SPARSE_OP_NON_TRANSPOSE;
    c.deviceAlpha = (variant & 1) != 0;
    c.inPlace = (variant & 2) != 0;
    return c;
}

std::string RangePathName(const RangePathCase &test)
{
    const auto &c = test.config;
    return std::string(c.type == ACL_FLOAT ? "F32_" : "C64_") + "M" + std::to_string(test.rows) +
        "_" + "NTH"[static_cast<int>(c.operation)] + (test.diagonal ? "_Diagonal" : "_Triangular") +
        (c.deviceAlpha ? "_DeviceAlpha" : "_HostAlpha") + (c.inPlace ? "_InPlace" : "_Separate");
}

std::vector<RangePathCase> RangeScaleCases()
{
    std::vector<RangePathCase> tests;
    for (int type = 0; type < 2; ++type) {
        const int rows[] = {1, 2, type == 0 ? 513 : 257, type == 0 ? 24577 : 12289};
        for (int path = 0; path < 4; ++path) {
            const auto operation = static_cast<aclsparseOperation_t>((path + type) % 3);
            auto c = RangePathConfig(type == 0 ? ACL_FLOAT : ACL_COMPLEX64, operation, path + type);
            tests.push_back({c, rows[path], path == 0});
        }
    }
    return tests;
}

struct RangeScaleValues {
    const char *name;
    double alpha;
    double x;
    double diagonal;
    double expected;
};

std::vector<RangeScaleValues> RangeScaleInputs()
{
    const double large = std::ldexp(1.0, 127);
    const double small = std::ldexp(1.0, -100);
    return {
        {"alpha_times_x_overflows", 2.0, large, large, 2.0},
        {"large_product_other_exponents", -std::ldexp(1.0, 100), std::ldexp(1.0, 100),
            std::ldexp(1.0, 100), -std::ldexp(1.0, 100)},
        {"normal_inputs_product_underflows", small, small, small, small},
        {"small_product_other_exponents", -std::ldexp(1.0, -80), small, small, -std::ldexp(1.0, -80)},
        {"finite_subnormal_result", small, small, std::ldexp(1.0, -60), std::ldexp(1.0, -140)}
    };
}

Matrix MakeRangeScaleMatrix(const RangePathCase &test, double diagonal)
{
    std::vector<Entry> source;
    source.reserve(test.rows + 1);
    for (int row = 0; row < test.rows; ++row) {
        const double value = row == test.rows - 1 ? diagonal : 1.0;
        source.push_back(RangeStoredEntry(row, row, {value, 0.0}, test.config));
    }
    // A genuine stored dependency selects triangular solve; Y[0]=0 makes its
    // mathematical contribution zero without changing the range counterexample.
    if (!test.diagonal) source.push_back(RangeStoredEntry(test.rows - 1, 0, {1.0, 0.0}, test.config));
    return EncodeEntries(test.rows, source, test.config);
}

void RunScaledInputRegression(const RangePathCase &test, const RangeScaleValues &values)
{
    const auto &c = test.config;
    SCOPED_TRACE(values.name);
    const auto matrix = MakeRangeScaleMatrix(test, values.diagonal);
    std::vector<Complex> x(test.rows, Complex{1.0, 0.0});
    if (!test.diagonal) x[0] = {0.0, 0.0};
    x.back() = {values.x, 0.0};
    const auto alpha = Value(values.alpha, -values.alpha / 2.0, c);
    const auto expected = SparseChainGolden(matrix, c, StoredValues(matrix, c, false), x, alpha);
    ASSERT_EQ(expected.back(), Value(values.expected, -values.expected / 2.0, c));
    // Every nonzero input component is normal. The final subnormal is 2^-140
    // (and -2^-141 for complex imaginary); an erroneous zero must fail exactly.
    RunRangeRegression(matrix, c, x, alpha, expected);
}

void RunScaledDependencyRegression(const RangePathCase &test)
{
    if (test.diagonal) return;
    const auto &c = test.config;
    const double large = std::ldexp(1.0, 127);
    std::vector<Entry> source;
    for (int row = 0; row < test.rows; ++row)
        source.push_back(RangeStoredEntry(row, row, {row == test.rows - 1 ? large : 1.0, 0.0}, c));
    source.push_back(RangeStoredEntry(test.rows - 1, 0, Value(large / 2, large / 4, c), c));
    const auto matrix = EncodeEntries(test.rows, source, c);
    std::vector<Complex> x(test.rows, {1.0, 0.0});
    x[0] = Value(1.0, -1.0, c);
    x.back() = Value(large, large / 2, c);
    const auto expected = SparseChainGolden(matrix, c, StoredValues(matrix, c, false), x, {2.0, 0.0});
    ASSERT_EQ(expected.back(), Value(c.type == ACL_FLOAT ? 1.0 : 0.5, 1.5, c));
    // The scaled replay must use an already solved nonzero dependency in both
    // resident and windowed Y caches, not the original input or stale GM.
    RunRangeRegression(matrix, c, x, {2.0, 0.0}, expected);
}

class SpSVArch22RangeScale : public SpSVArch22Test, public ::testing::WithParamInterface<RangePathCase> {};
TEST_P(SpSVArch22RangeScale, FiniteResultsSurviveOverflowAndUnderflowOfIntermediateProducts)
{
    const auto test = GetParam();
    for (const auto &values : RangeScaleInputs()) {
        RunScaledInputRegression(test, values);
        if (::testing::Test::HasFatalFailure()) return;
    }
    RunScaledDependencyRegression(test);
}
INSTANTIATE_TEST_SUITE_P(NumericalRange, SpSVArch22RangeScale, ::testing::ValuesIn(RangeScaleCases()),
    [](const ::testing::TestParamInfo<RangePathCase> &info) { return RangePathName(info.param); });

std::vector<RangePathCase> RangeCancellationCases()
{
    std::vector<RangePathCase> tests;
    for (int type = 0; type < 2; ++type) {
        const int rows[] = {type == 0 ? 512 : 256, type == 0 ? 513 : 257,
            type == 0 ? 24576 : 12288, type == 0 ? 24577 : 12289};
        for (int path = 0; path < 4; ++path) {
            for (auto operation : kOperations) {
                const int variant = path + type + static_cast<int>(operation);
                auto c = RangePathConfig(type == 0 ? ACL_FLOAT : ACL_COMPLEX64, operation, variant);
                tests.push_back({c, rows[path], false});
            }
        }
    }
    return tests;
}

Matrix MakeRangeCancellationMatrix(const RangePathCase &test)
{
    std::vector<Entry> source;
    source.reserve(test.rows + 2);
    for (int row = 0; row < test.rows; ++row) {
        source.push_back(RangeStoredEntry(row, row, {1.0, 0.0}, test.config));
    }
    // EncodeEntries reverses each CSR group. Insert these in reverse so the
    // effective dependency order is 2^24, then 1 for both N and T/H.
    source.push_back(RangeStoredEntry(test.rows - 1, 1, {1.0, 0.0}, test.config));
    source.push_back(RangeStoredEntry(test.rows - 1, 0, {std::ldexp(1.0, 24), 0.0}, test.config));
    return EncodeEntries(test.rows, source, test.config);
}

class SpSVArch22RangeCancellation : public SpSVArch22Test,
    public ::testing::WithParamInterface<RangePathCase> {};
TEST_P(SpSVArch22RangeCancellation, NeumaierCorrectionSurvivesCancellationAtBothCacheBoundaries)
{
    const auto test = GetParam();
    const auto &c = test.config;
    const auto matrix = MakeRangeCancellationMatrix(test);
    std::vector<Complex> x(test.rows, Value(1.0, 1.0, c));
    x.back() = Value(std::ldexp(1.0, 24), std::ldexp(1.0, 24), c);
    const auto expected = SparseChainGolden(matrix, c, StoredValues(matrix, c, false), x, {1.0, 0.0});
    ASSERT_EQ(expected.back(), Value(-1.0, -1.0, c));
    // M=513 FP32 is the original counterexample. Do not form sum+correction
    // in FP32 before subtracting it from the equally large RHS.
    RunRangeRegression(matrix, c, x, {1.0, 0.0}, expected);
}
INSTANTIATE_TEST_SUITE_P(NumericalRange, SpSVArch22RangeCancellation,
    ::testing::ValuesIn(RangeCancellationCases()),
    [](const ::testing::TestParamInfo<RangePathCase> &info) { return RangePathName(info.param); });

void RunGeneralUpdateDmaRegression(int rows, aclDataType type,
    aclsparseOperation_t operation, bool diagonal)
{
    Case c = RangePathConfig(type, operation, static_cast<int>(operation) + (type == ACL_COMPLEX64));
    c.base = rows == 17 ? 1 : 0;
    std::vector<Entry> source;
    for (int row = 0; row < rows; ++row)
        source.push_back(RangeStoredEntry(row, row, Value(4.0, 0.5, c), c));
    if (!diagonal) source.push_back(RangeStoredEntry(rows - 1, 0, Value(0.125, -0.0625, c), c));
    Plan plan(EncodeEntries(rows, source, c), c);
    ASSERT_EQ(plan.Analyze(), kSuccess);
    const auto values = StoredValues(plan.a, c, true);
    UpdateInput updated(values, c.type);
    ASSERT_EQ(updated.Apply(plan, ACL_SPARSE_SPSV_UPDATE_GENERAL), kSuccess);
    plan.VerifyTwice(SparseChainGolden(plan.a, c, values, Input(plan.a, c), plan.AlphaValue()));
    updated.ExpectUnchanged();
}

TEST_F(SpSVArch22Range, GeneralUpdateCopyAndPermutationPathsPreserveDmaTails)
{
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        for (auto operation : kOperations) {
            for (int rows : {17, 4097}) {
                for (bool diagonal : {false, true}) {
                    SCOPED_TRACE(std::to_string(type) + ":" + std::to_string(operation) +
                        ":m=" + std::to_string(rows) + ":diagonal=" + std::to_string(diagonal));
                    RunGeneralUpdateDmaRegression(rows, type, operation, diagonal);
                    if (::testing::Test::HasFatalFailure()) return;
                }
            }
        }
    }
}

void RunResidentExtremeDependencyRegression(aclDataType type, aclsparseOperation_t operation, bool unit)
{
    Case c = RangePathConfig(type, operation, static_cast<int>(operation) + (type == ACL_COMPLEX64));
    c.unit = unit;
    int rows = type == ACL_FLOAT ? 513 : 257;
    int dependentRow = unit ? 2 : 1;
    double large = std::ldexp(1.0, 127);
    double coefficient = std::ldexp(1.0, 50);
    std::vector<Entry> source;
    for (int row = 0; row < rows; ++row)
        source.push_back(RangeStoredEntry(row, row, {!unit && row == 1 ? coefficient : 1.0, 0.0}, c));
    source.push_back(RangeStoredEntry(dependentRow, 0, {coefficient, 0.0}, c));
    if (unit) source.push_back(RangeStoredEntry(dependentRow, 1, {coefficient, 0.0}, c));
    const auto matrix = EncodeEntries(rows, source, c);
    std::vector<Complex> x(rows, {1.0, 0.0});
    x[0] = Value(large, large, c);
    x[1] = unit ? -x[0] : Complex{0.0, 0.0};
    x[dependentRow] = {0.0, 0.0};
    const auto expected = SparseChainGolden(matrix, c, StoredValues(matrix, c, false), x, {1.0, 0.0});
    ASSERT_EQ(expected[dependentRow], Value(unit ? 0.0 : -large, unit ? 0.0 : -large, c));
    // The dependent row has safe-range A/X/alpha, but previous Y needs scaling.
    // UNIT cancels two overflowing products; NON_UNIT divides the large product.
    RunRangeRegression(matrix, c, x, {1.0, 0.0}, expected);
}

TEST_F(SpSVArch22Range, ResidentRangeGuardEnablesAfterAnExtremeSolvedDependency)
{
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        for (auto operation : kOperations) {
            SCOPED_TRACE(std::to_string(type) + ":" + std::to_string(operation));
            for (bool unit : {false, true}) {
                RunResidentExtremeDependencyRegression(type, operation, unit);
                if (::testing::Test::HasFatalFailure()) return;
            }
        }
    }
}

Matrix MakeRoundedProductMatrix(int rows, const Case& c)
{
    std::vector<Entry> entries;
    const double delta = std::ldexp(1.0, -12);
    for (int pair = 0; pair < 8192; ++pair) {
        entries.push_back(RangeStoredEntry(rows - 1, 0, {1.0 + delta, 0.0}, c));
        entries.push_back(RangeStoredEntry(rows - 1, 1, {-(1.0 + 2.0 * delta), 0.0}, c));
    }
    return EncodeEntries(rows, entries, c);
}

TEST_P(SpSVArch22RangeCancellation, RepeatedProductErrorsSurviveBothCapacityBoundaries)
{
    const auto test = GetParam();
    auto c = test.config;
    c.unit = true;
    const auto matrix = MakeRoundedProductMatrix(test.rows, c);
    std::vector<Complex> x(test.rows, {0.0, 0.0});
    x[0] = Value(1.0 + std::ldexp(1.0, -12), 1.0 + std::ldexp(1.0, -12), c);
    x[1] = Value(1.0, 1.0, c);
    x.back() = Value(std::ldexp(1.0, -9), std::ldexp(1.0, -9), c);
    const auto expected = SparseChainGolden(matrix, c, StoredValues(matrix, c, false), x, {1.0, 0.0});
    ASSERT_EQ(expected.back(), Value(0.00146484375, 0.00146484375, c));
    RunRangeRegression(matrix, c, x, {1.0, 0.0}, expected);
}

std::vector<RangePathCase> ScaledDependencyCases()
{
    std::vector<RangePathCase> tests;
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        const int rows[] = {3, type == ACL_FLOAT ? 513 : 257, type == ACL_FLOAT ? 24577 : 12289};
        for (int count : rows) {
            for (auto operation : kOperations) {
                auto c = RangePathConfig(type, operation, count + static_cast<int>(operation));
                tests.push_back({c, count, false});
            }
        }
    }
    return tests;
}

class SpSVArch22ScaledDependency : public SpSVArch22Test, public ::testing::WithParamInterface<RangePathCase> {};
TEST_P(SpSVArch22ScaledDependency, SubnormalIntermediateRetainsFractionAndExponent)
{
    const auto test = GetParam();
    const auto& c = test.config;
    std::vector<Entry> entries;
    for (int row = 0; row < test.rows; ++row) {
        double diagonal = row == test.rows - 1 ? std::ldexp(1.0, -50) : 1.0;
        entries.push_back(RangeStoredEntry(row, row, {diagonal, 0.0}, c));
    }
    entries.push_back(RangeStoredEntry(1, 0, {-std::ldexp(1.0, -25), 0.0}, c));
    entries.push_back(RangeStoredEntry(test.rows - 1, 1, {std::ldexp(1.0, 127), 0.0}, c));
    const auto matrix = EncodeEntries(test.rows, entries, c);
    std::vector<Complex> x(test.rows, {0.0, 0.0});
    x[0] = Value(std::ldexp(1.0, -148), std::ldexp(1.0, -148), c);
    x[1] = Value(std::ldexp(1.0, -149), std::ldexp(1.0, -149), c);
    const auto expected = SparseChainGolden(matrix, c, StoredValues(matrix, c, false), x, {0.5, 0.0});
    ASSERT_EQ(expected.back(), Value(-134217736.0, -134217736.0, c));
    RunRangeRegression(matrix, c, x, {0.5, 0.0}, expected);
}
INSTANTIATE_TEST_SUITE_P(NumericalRange, SpSVArch22ScaledDependency, ::testing::ValuesIn(ScaledDependencyCases()),
                         [](const ::testing::TestParamInfo<RangePathCase>& info) { return RangePathName(info.param); });

TEST_P(SpSVArch22ScaledDependency, SafeRangeInputsKeepUnderflowedQuotientForLaterRows)
{
    const auto test = GetParam();
    const auto& c = test.config;
    std::vector<Entry> entries;
    for (int row = 0; row < test.rows; ++row) {
        double diagonal = row == 1 ? std::ldexp(1.0, 50) : 1.0;
        if (row == test.rows - 1)
            diagonal = std::ldexp(1.0, -50);
        entries.push_back(RangeStoredEntry(row, row, {diagonal, 0.0}, c));
    }
    entries.push_back(RangeStoredEntry(test.rows - 1, 1, {std::ldexp(1.0, 127), 0.0}, c));
    const auto matrix = EncodeEntries(test.rows, entries, c);
    std::vector<Complex> x(test.rows, {0.0, 0.0});
    x[0] = Value(1.0, 1.0, c);
    x[1] = Value(std::ldexp(1.0, -50), std::ldexp(1.0, -50), c);
    Complex alpha{std::ldexp(1.0, -50), 0.0};
    const auto expected = SparseChainGolden(matrix, c, StoredValues(matrix, c, false), x, alpha);
    ASSERT_EQ(expected.back(), Value(-std::ldexp(1.0, 27), -std::ldexp(1.0, 27), c));
    RunRangeRegression(matrix, c, x, alpha, expected);
}

void RunWideExponentDependencyRegression(aclDataType type, aclsparseOperation_t operation)
{
    auto c = RangePathConfig(type, operation, static_cast<int>(operation) + (type == ACL_COMPLEX64));
    std::vector<Entry> entries;
    for (int row = 0; row < 17; ++row) {
        double diagonal = row < 8 ? std::ldexp(1.0, 100) : std::ldexp(1.0, -100);
        if (row == 8)
            diagonal = 1.0;
        entries.push_back(RangeStoredEntry(row, row, {diagonal, 0.0}, c));
        if (row != 0 && row != 8)
            entries.push_back(RangeStoredEntry(row, row - 1, {-1.0, 0.0}, c));
    }
    entries.push_back(RangeStoredEntry(8, 0, {std::ldexp(1.0, 100), 0.0}, c));
    entries.push_back(RangeStoredEntry(8, 7, {1.0, 0.0}, c));
    const auto matrix = EncodeEntries(17, entries, c);
    std::vector<Complex> x(17, {0.0, 0.0});
    x[0] = x[8] = Value(1.0, -1.0, c);
    // The closed-form powers retain the 2^-800 term that an ordinary
    // double-precision sequential subtraction would itself cancel to zero.
    std::vector<Complex> expected(17);
    for (int row = 0; row < 17; ++row) {
        double real = row < 8 ? std::ldexp(1.0, -100 * (row + 1)) : -std::ldexp(1.0, -800 + 100 * (row - 8));
        expected[row] = Value(real, -real, c);
    }
    RunRangeRegression(matrix, c, x, {1.0, 0.0}, expected);
}

TEST_F(SpSVArch22Range, ScaledResidualKeepsCancellationAcrossMultipleIntegerWindows)
{
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        for (auto operation : kOperations) {
            RunWideExponentDependencyRegression(type, operation);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

#endif
