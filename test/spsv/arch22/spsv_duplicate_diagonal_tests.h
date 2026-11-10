/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_DUPLICATE_DIAGONAL_TESTS_H_
#define SPSV_ARCH22_DUPLICATE_DIAGONAL_TESTS_H_

// Included inside spsv_test.cpp's anonymous namespace after spsv_range_tests.h.
struct DuplicateDiagonalCase {
    Case config;
    int rows;
    bool diagonal;
    std::string path;
};

std::vector<DuplicateDiagonalCase> DuplicateDiagonalCases()
{
    std::vector<DuplicateDiagonalCase> tests;
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        const int components = type == ACL_COMPLEX64 ? 2 : 1;
        const int rows[] = {1, 129, 2, 512 / components + 1, 24576 / components + 1};
        const char *paths[] = {"SingleDiagonal", "MulticoreDiagonal", "Compact", "Resident", "Windowed"};
        for (int path = 0; path < 5; ++path) {
            for (auto operation : kOperations) {
                const int variant = path + static_cast<int>(operation) + (components - 1);
                Case c;
                c.type = type;
                c.operation = operation;
                const bool effectiveLower = (variant % 2) == 0;
                c.lower = operation == ACL_SPARSE_OP_NON_TRANSPOSE ? effectiveLower : !effectiveLower;
                c.base = path % 2;
                c.deviceAlpha = (variant & 1) != 0;
                c.inPlace = (variant & 2) != 0;
                tests.push_back({c, rows[path], path < 2, paths[path]});
            }
        }
    }
    return tests;
}

int DuplicateDiagonalTarget(const DuplicateDiagonalCase &test)
{
    const auto &c = test.config;
    const bool forward = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? c.lower : !c.lower;
    return forward ? test.rows - 1 : 0;
}

Matrix EncodeDuplicateDiagonalMatrix(const DuplicateDiagonalCase &test, const std::vector<Complex> &terms)
{
    const auto &c = test.config;
    const int target = DuplicateDiagonalTarget(test);
    std::vector<Entry> source;
    source.reserve(static_cast<size_t>(test.rows) * (test.diagonal ? terms.size() : 1) + terms.size());
    for (int row = 0; row < test.rows; ++row) {
        if (test.diagonal || row == target) {
            for (const auto &value : terms) source.push_back(RangeStoredEntry(row, row, value, c));
        } else {
            source.push_back(RangeStoredEntry(row, row, {1.0, 0.0}, c));
        }
    }
    if (!test.diagonal) {
        const int previous = target == 0 ? test.rows - 1 : 0;
        // A nonzero stored dependency selects the triangular path. It also
        // verifies that diagonal compensation does not alter RHS accumulation.
        source.push_back(RangeStoredEntry(target, previous, Value(0.25, 0.125, c), c));
    }
    return EncodeEntries(test.rows, source, c);
}

Matrix MakeDuplicateCancellationMatrix(const DuplicateDiagonalCase &test, int update)
{
    const auto &c = test.config;
    const double large = std::ldexp(1.0, update == 0 ? 24 : 25);
    const auto largeValue = Value(large, large, c);
    const auto residual = update == 0 ? Value(1.0, 1.0, c) : Value(2.0, -1.0, c);
    // EncodeEntries reverses CSR row groups. The normalized diagonal order
    // is large, residual, -large, whose plain FP32 sum loses the residual.
    return EncodeDuplicateDiagonalMatrix(test, {-largeValue, residual, largeValue});
}

void RunDuplicateDiagonalRegression(DuplicateDiagonalCase test, bool unit)
{
    test.config.unit = unit;
    const auto &c = test.config;
    SCOPED_TRACE(unit ? "UNIT" : "NON_UNIT");
    Plan plan(MakeDuplicateCancellationMatrix(test, 0), c);
    std::vector<Complex> x(test.rows, Value(1.0, 0.5, c));
    plan.input = Pack(x, c.type);
    plan.alpha = Pack({{1.0, 0.0}}, c.type);
    plan.dX.Put(plan.input);
    plan.dAlpha.Put(plan.alpha);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    const auto initial = StoredValues(plan.a, c, false);
    const auto expected = SparseChainGolden(plan.a, c, initial, x, {1.0, 0.0});
    if (test.rows == 1 && c.type == ACL_FLOAT) ASSERT_EQ(expected[0], Complex(1.0, 0.0));
    VerifyRangeTwice(plan, expected, 4);
    if (::testing::Test::HasFatalFailure()) return;

    const auto updatedMatrix = MakeDuplicateCancellationMatrix(test, 1);
    ASSERT_EQ(updatedMatrix.ptr, plan.a.ptr);
    ASSERT_EQ(updatedMatrix.idx, plan.a.idx);
    const auto updated = StoredValues(updatedMatrix, c, false);
    UpdateInput update(updated, c.type);
    ASSERT_EQ(update.Apply(plan, ACL_SPARSE_SPSV_UPDATE_GENERAL), kSuccess);
    const auto updatedExpected = SparseChainGolden(plan.a, c, updated, x, {1.0, 0.0});
    if (test.rows == 1 && c.type == ACL_FLOAT && !unit)
        ASSERT_EQ(updatedExpected[0], Complex(0.5, 0.0));
    VerifyRangeTwice(plan, updatedExpected, 4);
    update.ExpectUnchanged();
}

class SpSVArch22DuplicateDiagonal : public SpSVArch22Test,
    public ::testing::WithParamInterface<DuplicateDiagonalCase> {};
TEST_P(SpSVArch22DuplicateDiagonal, CancellationSurvivesAnalysisAndGeneralUpdate)
{
    for (bool unit : {false, true}) {
        RunDuplicateDiagonalRegression(GetParam(), unit);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

std::string DuplicateDiagonalName(const DuplicateDiagonalCase &test)
{
    const auto &c = test.config;
    return std::string(c.type == ACL_FLOAT ? "F32_" : "C64_") + test.path + "_" +
        "NTH"[static_cast<int>(c.operation)] + (c.lower ? "_Lower" : "_Upper") +
        (c.deviceAlpha ? "_DeviceAlpha" : "_HostAlpha") + (c.inPlace ? "_InPlace" : "_Separate");
}

INSTANTIATE_TEST_SUITE_P(DuplicateCancellation, SpSVArch22DuplicateDiagonal,
    ::testing::ValuesIn(DuplicateDiagonalCases()),
    [](const ::testing::TestParamInfo<DuplicateDiagonalCase> &info) { return DuplicateDiagonalName(info.param); });

void RunDuplicateLowPartRegression(aclDataType type, bool diagonal)
{
    Case c;
    c.type = type;
    const int rows = diagonal ? 1 : 512 / (type == ACL_COMPLEX64 ? 2 : 1) + 1;
    const double epsilon = std::ldexp(1.0, -24);
    std::vector<Entry> source;
    for (int row = 0; row < rows - 1; ++row) source.push_back({row, row, {1.0, 0.0}});
    // Both components retain a nonzero low part after the sum. Rounding the
    // denominator first gives 1, whereas the correctly rounded quotient is
    // the FP32 value immediately below 1. Symmetric complex X/A makes Im(Y)=0.
    source.push_back({rows - 1, rows - 1, Value(epsilon, epsilon, c)});
    source.push_back({rows - 1, rows - 1, Value(1.0, 1.0, c)});
    if (!diagonal) source.push_back({rows - 1, 0, {1.0, 0.0}});
    const auto matrix = EncodeEntries(rows, source, c);
    std::vector<Complex> x(rows, {0.0, 0.0});
    x.back() = Value(1.0, 1.0, c);
    const auto expected = SparseChainGolden(matrix, c, StoredValues(matrix, c, false), x, {1.0, 0.0});
    ASSERT_EQ(static_cast<float>(expected.back().real()), std::nextafter(1.0f, 0.0f));
    ASSERT_EQ(expected.back().imag(), 0.0);
    RunRangeRegression(matrix, c, x, {1.0, 0.0}, expected, 0);
}

class SpSVArch22DuplicateLowPart : public SpSVArch22Test {};
TEST_F(SpSVArch22DuplicateLowPart, DiagonalResidualParticipatesInDivisionWithoutEarlyRounding)
{
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        for (bool diagonal : {true, false}) {
            SCOPED_TRACE(std::to_string(type) + (diagonal ? ":diagonal" : ":resident"));
            RunDuplicateLowPartRegression(type, diagonal);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

Complex ExtremeDiagonalResidual(const Case &c, int scenario)
{
    const double small = std::ldexp(1.0, scenario == 1 ? -149 : -50);
    return Value(small, scenario == 1 ? -small : small, c);
}

Matrix MakeExtremeDuplicateMatrix(const DuplicateDiagonalCase &test, int scenario, int update, int order)
{
    const auto &c = test.config;
    const auto large = Value(std::ldexp(1.0, 127), std::ldexp(1.0, 127), c);
    const auto residual = ExtremeDiagonalResidual(c, scenario) * (update == 0 ? 1.0 : 2.0);
    std::vector<Complex> terms{large, residual, -large};
    if (scenario == 1) terms = {-large, residual, large};
    if (scenario == 2) terms = {large, large, residual, -large, -large};
    std::rotate(terms.begin(), terms.begin() + order % 3, terms.end());
    if (order >= 3) std::reverse(terms.begin(), terms.end());
    return EncodeDuplicateDiagonalMatrix(test, terms);
}

void RunAnalyticDuplicateRegression(const DuplicateDiagonalCase &test, const Matrix &initial,
    const Matrix &updated, Complex residual)
{
    const auto &c = test.config;
    Plan plan(initial, c);
    const int target = DuplicateDiagonalTarget(test);
    std::vector<Complex> x(test.rows, {0.0, 0.0});
    std::vector<Complex> expected(test.rows, {0.0, 0.0});
    for (int row = 0; row < test.rows; ++row) {
        if (test.diagonal || row == target) {
            x[row] = residual;
            expected[row] = {1.0, 0.0};
        }
    }
    // Construct the exact solution analytically; even sequential FP64 sums
    // can lose a small term before the larger entries cancel.
    plan.input = Pack(x, c.type);
    plan.alpha = Pack({{1.0, 0.0}}, c.type);
    plan.dX.Put(plan.input);
    plan.dAlpha.Put(plan.alpha);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    VerifyRangeTwice(plan, expected, 4);
    if (::testing::Test::HasFatalFailure()) return;
    ASSERT_EQ(updated.ptr, plan.a.ptr);
    ASSERT_EQ(updated.idx, plan.a.idx);
    UpdateInput update(StoredValues(updated, c, false), c.type);
    ASSERT_EQ(update.Apply(plan, ACL_SPARSE_SPSV_UPDATE_GENERAL), kSuccess);
    for (auto &value : expected) value *= 0.5;
    VerifyRangeTwice(plan, expected, 4);
    update.ExpectUnchanged();
}

void RunExtremeDuplicateRegression(const DuplicateDiagonalCase &test, int scenario, int order = 0)
{
    SCOPED_TRACE("scenario=" + std::to_string(scenario) + " order=" + std::to_string(order));
    RunAnalyticDuplicateRegression(test, MakeExtremeDuplicateMatrix(test, scenario, 0, order),
        MakeExtremeDuplicateMatrix(test, scenario, 1, order), ExtremeDiagonalResidual(test.config, scenario));
}

TEST_P(SpSVArch22DuplicateDiagonal, ExtremeExponentGapsSurviveAnalysisAndGeneralUpdate)
{
    for (int scenario = 0; scenario < 3; ++scenario) {
        RunExtremeDuplicateRegression(GetParam(), scenario);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

TEST_F(SpSVArch22DuplicateLowPart, ExtremeCancellationCoversFormatsAndStoredOrders)
{
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        for (auto format : {ACL_SPARSE_FORMAT_CSR, ACL_SPARSE_FORMAT_CSC,
            ACL_SPARSE_FORMAT_COO, ACL_SPARSE_FORMAT_SLICED_ELL}) {
            for (int order = 0; order < 6; ++order) {
                Case c;
                c.type = type;
                c.format = format;
                c.base = order % 2;
                c.deviceAlpha = (order & 1) != 0;
                c.inPlace = (order & 2) != 0;
                SCOPED_TRACE(std::to_string(type) + ":format=" + std::to_string(format));
                RunExtremeDuplicateRegression({c, 1, true, "SingleDiagonal"}, 0, order);
                if (::testing::Test::HasFatalFailure()) return;
            }
        }
    }
}

Matrix MakeNormalDuplicateMatrix(const DuplicateDiagonalCase &test, int scenario, int update, int order)
{
    const auto &c = test.config;
    const auto large = Value(std::ldexp(1.0, 24), std::ldexp(1.0, 24), c);
    const auto one = Value(1.0, 1.0, c);
    const auto residual = Value(std::ldexp(1.0, -25), -std::ldexp(1.0, -25), c) * (update ? 2.0 : 1.0);
    std::vector<Complex> terms{large, one, residual, -large, -one};
    // The second small term leaves a nonzero but inaccurate DD result,
    // so checking only a zero approximate diagonal would be insufficient.
    if (scenario != 0) terms.push_back(residual);
    std::rotate(terms.begin(), terms.begin() + order % terms.size(), terms.end());
    if (static_cast<size_t>(order) < terms.size()) std::reverse(terms.begin(), terms.end());
    return EncodeDuplicateDiagonalMatrix(test, terms);
}

void RunNormalDuplicateRegression(const DuplicateDiagonalCase &test, int scenario, int order)
{
    SCOPED_TRACE("normal scenario=" + std::to_string(scenario) + " order=" + std::to_string(order));
    const double small = std::ldexp(1.0, scenario == 0 ? -25 : -24);
    RunAnalyticDuplicateRegression(test, MakeNormalDuplicateMatrix(test, scenario, 0, order),
        MakeNormalDuplicateMatrix(test, scenario, 1, order), Value(small, -small, test.config));
}

void RunDuplicateZeroRegression(DuplicateDiagonalCase test, bool unit)
{
    test.config.unit = unit;
    const auto &c = test.config;
    SCOPED_TRACE(unit ? "unit duplicate diagonal" : "singular duplicate diagonal");
    const auto large = Value(std::ldexp(1.0, 24), std::ldexp(1.0, 24), c);
    const auto one = Value(1.0, 1.0, c);
    Plan plan(EncodeDuplicateDiagonalMatrix(test, {large, one, -large, -one}), c);
    const auto input = c.type == ACL_COMPLEX64 && !unit ? Complex{0.0, 0.0} : Value(1.0, 0.5, c);
    const std::vector<Complex> x(test.rows, input);
    plan.input = Pack(x, c.type);
    plan.dX.Put(plan.input);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    const auto initial = StoredValues(plan.a, c, false);
    plan.VerifyTwice(SparseChainGolden(plan.a, c, initial, x, plan.AlphaValue()));
    if (::testing::Test::HasFatalFailure()) return;
    const std::vector<Complex> replacement(test.rows, {2.0, 0.0});
    UpdateInput update(replacement, c.type);
    ASSERT_EQ(update.Apply(plan, ACL_SPARSE_SPSV_UPDATE_DIAGONAL), kSuccess);
    plan.VerifyTwice(SparseChainGolden(plan.a, c, initial, x, plan.AlphaValue(), &replacement));
    update.ExpectUnchanged();
}

TEST_P(SpSVArch22DuplicateDiagonal, NormalRangeCancellationAndSingularDiagonals)
{
    for (int scenario = 0; scenario < 2; ++scenario) {
        for (int order : {0, 2, 5}) {
            RunNormalDuplicateRegression(GetParam(), scenario, order);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
    for (bool unit : {false, true}) {
        RunDuplicateZeroRegression(GetParam(), unit);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

TEST_F(SpSVArch22DuplicateLowPart, NormalRangeExactStoredOrderAndFormats)
{
    Case original;
    const auto matrix = MakeNormalDuplicateMatrix({original, 1, true, "SingleDiagonal"}, 0, 0, 0);
    ASSERT_EQ(matrix.ptr, (std::vector<int32_t>{0, 5}));
    ASSERT_EQ(matrix.idx, (std::vector<int32_t>{0, 0, 0, 0, 0}));
    ASSERT_EQ(Pack(StoredValues(matrix, original, false), original.type),
        (std::vector<float>{std::ldexp(1.0f, 24), 1.0f, std::ldexp(1.0f, -25), -std::ldexp(1.0f, 24), -1.0f}));
    const aclsparseFormat_t formats[] = {ACL_SPARSE_FORMAT_CSR, ACL_SPARSE_FORMAT_CSC,
        ACL_SPARSE_FORMAT_COO, ACL_SPARSE_FORMAT_SLICED_ELL};
    for (int variant = 0; variant < 96; ++variant) {
        Case c;
        c.type = variant < 48 ? ACL_FLOAT : ACL_COMPLEX64;
        c.format = formats[(variant / 12) % 4];
        c.operation = static_cast<aclsparseOperation_t>(variant % 3);
        c.base = variant % 2;
        c.deviceAlpha = (variant & 1) != 0;
        c.inPlace = (variant & 2) != 0;
        for (int scenario = 0; scenario < 2; ++scenario) {
            RunNormalDuplicateRegression({c, 1, true, "SingleDiagonal"}, scenario, variant % 12);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

void RunResidualCancellationRegression(Case c, int rows, bool distinct, bool unit, int extra)
{
    c.unit = unit;
    c.lower = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE;
    const double small = std::ldexp(1.0, -25);
    const double residual = small * (extra + 1);
    std::vector<double> terms{std::ldexp(1.0, 24), 1, small, -std::ldexp(1.0, 24), -1};
    if (extra != 0) terms.push_back(small);
    const int target = rows - 1;
    std::vector<Entry> source;
    for (int row = 0; row < rows; ++row)
        source.push_back(RangeStoredEntry(row, row, {row == target && !unit ? residual : 1, 0}, c));
    for (int i = static_cast<int>(terms.size()) - 1; i >= 0; --i)
        source.push_back(RangeStoredEntry(target, distinct ? i : 0, Value(terms[i], terms[i], c), c));
    Plan plan(EncodeEntries(rows, source, c), c);
    const Complex dependency = Value(1, -1, c);
    std::vector<Complex> rhs(rows, dependency);
    std::vector<Complex> expected(rows, dependency);
    rhs[target] = {0, 0};
    const double targetValue = -(c.type == ACL_COMPLEX64 ? 2 : 1) * (unit ? residual : 1);
    expected[target] = {targetValue, 0};
    plan.input = Pack(rhs, c.type);
    plan.dX.Put(plan.input);
    plan.alpha = Pack({{1, 0}}, c.type);
    plan.dAlpha.Put(plan.alpha);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    VerifyRangeTwice(plan, expected, 0);
    if (::testing::Test::HasFatalFailure()) return;
    auto updated = StoredValues(plan.a, c, false);
    for (size_t i = 0; i < updated.size(); ++i) {
        const auto &entry = plan.a.entries[i];
        if (entry.row >= 0 && entry.row != entry.col) updated[i] *= 2;
    }
    UpdateInput general(updated, c.type);
    ASSERT_EQ(general.Apply(plan, ACL_SPARSE_SPSV_UPDATE_GENERAL), kSuccess);
    expected[target] *= 2;
    VerifyRangeTwice(plan, expected, 0);
    general.ExpectUnchanged();
    std::vector<Complex> diagonal(rows, {1, 0});
    diagonal[target] = {2 * residual, 0};
    UpdateInput replacement(diagonal, c.type);
    ASSERT_EQ(replacement.Apply(plan, ACL_SPARSE_SPSV_UPDATE_DIAGONAL), kSuccess);
    if (!unit) expected[target] *= 0.5;
    VerifyRangeTwice(plan, expected, 0);
    replacement.ExpectUnchanged();
}

TEST_F(SpSVArch22DuplicateLowPart, OffDiagonalCancellationRetainsResidualAcrossSolvePaths)
{
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        const int components = type == ACL_COMPLEX64 ? 2 : 1;
        for (int rows : {8, 512 / components + 1, 24576 / components + 1}) {
            for (auto format : {ACL_SPARSE_FORMAT_CSR, ACL_SPARSE_FORMAT_CSC,
                ACL_SPARSE_FORMAT_COO, ACL_SPARSE_FORMAT_SLICED_ELL}) {
                for (auto op : kOperations) {
                    for (int variant = 0; variant < 4; ++variant) {
                        Case c;
                        c.type = type;
                        c.format = format;
                        c.operation = op;
                        c.base = variant % 2;
                        c.deviceAlpha = (variant & 1) != 0;
                        c.inPlace = (variant & 2) != 0;
                        SCOPED_TRACE(std::to_string(type) + ":" + std::to_string(rows) + ":" +
                            std::to_string(format) + ":" + std::to_string(op) + ":" + std::to_string(variant));
                        RunResidualCancellationRegression(c, rows, (variant & 1) != 0,
                            (variant & 2) != 0, variant % 2);
                        if (::testing::Test::HasFatalFailure()) return;
                    }
                }
            }
        }
    }
}
TEST_F(SpSVArch22DuplicateLowPart, UnitResidentAccumulationChecksEntryCountBoundary)
{
    constexpr int rows = 513;
    constexpr int limit = 1 << 20;
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        for (int entries : {limit, limit + 1}) {
            SCOPED_TRACE(std::to_string(type) + ":" + std::to_string(entries));
            Case c;
            c.type = type;
            c.unit = true;
            c.inPlace = (entries & 1) != 0;
            c.deviceAlpha = type == ACL_COMPLEX64;
            std::vector<Entry> source;
            source.reserve(entries);
            for (int row = 0; row < rows; ++row) {
                source.push_back({row, row, {1, 0}});
            }
            for (int p = rows; p < entries; ++p) {
                source.push_back({rows - 1, 0, {p == rows ? 1.0 : 0.0, 0}});
            }
            std::vector<Complex> rhs(rows, {1, 0});
            std::vector<Complex> expected(rows, {1, 0});
            rhs.back() = {2, 0};
            RunRangeRegression(EncodeEntries(rows, source, c), c, rhs, {1, 0}, expected, 0);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

void VerifyResidentMixedTwice(Plan& plan, const std::vector<Complex>& expected)
{
    const auto rounded = Pack(expected, plan.c.type);
    std::vector<float> first;
    for (int repeat = 0; repeat < 2; ++repeat) {
        plan.dX.Put(plan.input);
        ASSERT_EQ(plan.Solve(plan.c.operation), kSuccess);
        ASSERT_EQ(aclrtSynchronizeStream(SpSVArch22Test::env->stream()), ACL_SUCCESS);
        const auto actual = (plan.c.inPlace ? plan.dX : plan.dY).Get<float>(plan.input.size());
        Verify(actual, expected, plan.c.type);
        for (size_t i = 0; i < rounded.size(); ++i) {
            if (std::isfinite(rounded[i]))
                EXPECT_EQ(actual[i], rounded[i]);
        }
        if (repeat == 0)
            first = actual;
        else
            ExpectBitwiseFloats(actual, first);
        if (!plan.c.inPlace)
            ExpectBitwiseFloats(plan.dX.Get<float>(plan.input.size()), plan.input);
    }
    ExpectBitwiseFloats(plan.dValues.Get<float>(plan.initial.size()), plan.initial);
    ExpectBitwiseFloats(plan.dAlpha.Get<float>(plan.alpha.size()), plan.alpha);
    EXPECT_EQ(plan.dPtr.Get<int32_t>(plan.a.ptr.size()), plan.a.ptr);
    EXPECT_EQ(plan.dIdx.Get<int32_t>(plan.a.idx.size()), plan.a.idx);
}

void RunResidentCheckedSpecial(Case c, int scenario)
{
    constexpr int rows = 513;
    const int head = c.lower ? 0 : rows - 1;
    const int next = head + (c.lower ? 1 : -1);
    const int third = next + (c.lower ? 1 : -1);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<Entry> source;
    for (int row = 0; row < rows; ++row) {
        source.push_back({row, row, {1, 0}});
    }
    source.push_back({next, head, {0, 0}});
    std::vector<Complex> rhs(rows, {1, 0});
    std::vector<Complex> expected(rows, {1, 0});
    if (scenario == 3) {
        source.back().value = {std::numeric_limits<double>::infinity(), 0};
        source.push_back({third, next, {0, 0}});
        expected[next] = {-std::numeric_limits<double>::infinity(), 0};
        expected[third] = {nan, 0};
    } else {
        const double value = scenario == 0 ? nan : std::ldexp(1.0, scenario == 1 ? -70 : 70);
        rhs[head] = Value(value, -value, c);
        expected[head] = rhs[head];
        if (scenario == 0)
            expected[next] = Value(nan, nan, c);
    }
    Plan plan(EncodeEntries(rows, source, c), c);
    plan.input = Pack(rhs, c.type);
    plan.alpha = Pack({{1, 0}}, c.type);
    plan.dX.Put(plan.input);
    plan.dAlpha.Put(plan.alpha);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    VerifyResidentMixedTwice(plan, expected);
}

TEST_F(SpSVArch22DuplicateLowPart, UnitResidentDependenciesRetainSpecialAndExtremePropagation)
{
    for (auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
        for (int variant = 0; variant < 2; ++variant) {
            Case c;
            c.type = type;
            c.unit = true;
            c.lower = variant == 0;
            c.base = variant;
            c.inPlace = variant != 0;
            c.deviceAlpha = type == ACL_COMPLEX64;
            for (int scenario = 0; scenario < (type == ACL_COMPLEX64 ? 3 : 4); ++scenario) {
                SCOPED_TRACE(std::to_string(type) + ":" + std::to_string(variant) + ":" + std::to_string(scenario));
                RunResidentCheckedSpecial(c, scenario);
                if (::testing::Test::HasFatalFailure())
                    return;
            }
        }
    }
}
#endif
