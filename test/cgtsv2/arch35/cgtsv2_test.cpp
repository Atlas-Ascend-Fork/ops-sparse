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

#include "test_common.h"
#include "cgtsv2_golden.h"
#include "cgtsv2_param.h"
#include "cgtsv2_npu_wrapper.h"
#include "arch35/cgtsv2_workspace.h"
#include "gtest/gtest-spi.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace {
using Complex = cgtsv2_test::High;
using cgtsv2_test::Inputs;
using cgtsv2_test::GuardedDeviceBuffer;
using Value = aclsparseComplex_t;

static_assert(std::is_same<Value, aclsparseComplex>::value, "reuse public complex64 type");
static_assert(sizeof(Value) == 8, "complex64 ABI must occupy two float32 components");
static_assert(alignof(Value) == alignof(float), "complex64 ABI float alignment");
static_assert(std::is_standard_layout<Value>::value, "complex64 ABI must be standard layout");
static_assert(offsetof(Value, x) == 0 && offsetof(Value, y) == sizeof(float), "complex64 ABI order");

enum class SolvePath {
    ANY,
    PCR,
    TWOFOLD
};

void ExpectDispatch(const cgtsv2_test::DispatchFlags& flags, SolvePath expected)
{
    if (expected == SolvePath::ANY) {
        return;
    }
    ASSERT_TRUE(std::all_of(flags.begin(), flags.end(), [](uint32_t value) { return value <= 1U; }));
    const bool stable = std::all_of(flags.begin(), flags.end(), [](uint32_t value) { return value == 1U; });
    EXPECT_EQ(stable, expected == SolvePath::PCR);
}

std::vector<Complex> Promote(const std::vector<Value>& values)
{
    std::vector<Complex> result;
    result.reserve(values.size());
    for (const auto& value : values) { result.emplace_back(value.x, value.y); }
    return result;
}

void ExpectBytesEqual(const std::vector<Value>& actual, const std::vector<Value>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    if (actual.empty()) { return; }
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(Value)), 0);
}

double FiniteUlp(float anchor)
{
    float adjacent = std::nextafter(anchor, std::numeric_limits<float>::infinity());
    if (!std::isfinite(adjacent)) {
        adjacent = std::nextafter(anchor, -std::numeric_limits<float>::infinity());
    }
    return std::abs(static_cast<double>(adjacent) - static_cast<double>(anchor));
}

void ExpectAccuracy(const std::vector<Value>& actual, const std::vector<Complex>& expected, int m, int n, int ldb)
{
    const size_t count = static_cast<size_t>(m) * n;
    ASSERT_EQ(expected.size(), count);
    if (count == 0) { return; }
    constexpr double tolerance = 1.0 / 8192.0;
    for (int component = 0; component < 2; ++component) {
        size_t matched = 0;
        double maxError = -1.0;
        float anchor = 0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                const auto& got = actual[static_cast<size_t>(j) * ldb + i];
                const auto& ref = expected[static_cast<size_t>(j) * m + i];
                const double value = component == 0 ? got.x : got.y;
                const double golden = component == 0 ? ref.real() : ref.imag();
                ASSERT_FALSE(std::isnan(value) || std::isnan(golden));
                if (std::isinf(value)) {
                    ASSERT_EQ(value, static_cast<double>(static_cast<float>(golden)));
                    ++matched;
                    continue;
                }
                ASSERT_TRUE(std::isfinite(value) && std::isfinite(golden));
                const double error = std::abs(value - golden);
                matched += error <= tolerance + tolerance * std::abs(golden);
                if (error > maxError) {
                    maxError = error;
                    anchor = static_cast<float>(golden);
                }
            }
        }
        EXPECT_GE(static_cast<double>(matched) / count, 0.99) << "component=" << component;
        if (maxError >= 0.0) {
            ASSERT_TRUE(std::isfinite(anchor));
            EXPECT_LE(maxError, std::max(0.01, 32 * FiniteUlp(anchor))) << "component=" << component;
        }
    }
}

// Nonfinite inputs have a separate classification check. Ordinary accuracy
// tests continue to use ExpectAccuracy, which rejects every NaN.
void ExpectSpecialAccuracy(const std::vector<Value>& actual, const std::vector<Complex>& expected,
    int m, int n, int ldb)
{
    ASSERT_EQ(expected.size(), static_cast<size_t>(m) * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const auto& got = actual[static_cast<size_t>(j) * ldb + i];
            const auto& ref = expected[static_cast<size_t>(j) * m + i];
            for (int component = 0; component < 2; ++component) {
                const float value = component == 0 ? got.x : got.y;
                const float golden = static_cast<float>(component == 0 ? ref.real() : ref.imag());
                ASSERT_EQ(std::isnan(value), std::isnan(golden));
                ASSERT_EQ(std::isinf(value), std::isinf(golden));
                if (!std::isnan(golden)) {
                    EXPECT_EQ(value, golden);
                }
            }
        }
    }
}

using AccuracyCheck = void (*)(const std::vector<Value>&, const std::vector<Complex>&, int, int, int);

TEST(Cgtsv2Accuracy, FiniteEndpointsRejectOutlier)
{
    const float largest = std::numeric_limits<float>::max();
    const double ulp = std::ldexp(1.0, 104);
    for (int component = 0; component < 2; ++component) {
        for (float sign : {-1.0f, 1.0f}) {
            SCOPED_TRACE(testing::Message() << "component=" << component << ",sign=" << sign);
            std::vector<Value> actual(100, Value{0, 0});
            std::vector<Complex> expected(100, Complex{});
            expected.back() = component == 0 ? Complex{sign * largest, 0} : Complex{0, sign * largest};
            // Exactly 99% match. The independent hard error bound must reject.
            EXPECT_NONFATAL_FAILURE(ExpectAccuracy(actual, expected, 100, 1, 100), "maxError");
            const float near = static_cast<float>(static_cast<double>(largest) - 32 * ulp);
            actual.back() = component == 0 ? Value{sign * near, 0} : Value{0, sign * near};
            ExpectAccuracy(actual, expected, 100, 1, 100);
            const float far = static_cast<float>(static_cast<double>(largest) - 33 * ulp);
            actual.back() = component == 0 ? Value{sign * far, 0} : Value{0, sign * far};
            EXPECT_NONFATAL_FAILURE(ExpectAccuracy(actual, expected, 100, 1, 100), "maxError");
            actual.back() = component == 0 ? Value{sign * largest, 0} : Value{0, sign * largest};
            ExpectAccuracy(actual, expected, 100, 1, 100);
        }
    }
}

void ExpectPaddingUnchanged(const std::vector<Value>& actual, const std::vector<Value>& expected,
    int m, int n, int ldb)
{
    for (int j = 0; j < n; ++j) {
        for (int i = m; i < ldb; ++i) {
            const size_t index = static_cast<size_t>(j) * ldb + i;
            EXPECT_EQ(std::memcmp(&actual[index], &expected[index], sizeof(Value)), 0);
        }
    }
}

Inputs MakeInputs(int m, int n, int ldb, bool normal, bool dominant, uint32_t seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> uniform(-5.0F, 5.0F);
    std::uniform_real_distribution<float> mean(-5.0F, 5.0F);
    std::uniform_real_distribution<float> sigma(0.1F, 2.0F);
    std::normal_distribution<float> gaussian(mean(generator), sigma(generator));
    auto sample = [&]() -> Value {
        return normal ? Value{gaussian(generator), gaussian(generator)} : Value{uniform(generator), uniform(generator)};
    };
    Inputs inputs{std::vector<Value>(m), std::vector<Value>(m), std::vector<Value>(m),
        std::vector<Value>(static_cast<size_t>(ldb) * n)};
    for (int i = 0; i < m; ++i) {
        inputs.dl[i] = sample();
        inputs.d[i] = sample();
        inputs.du[i] = sample();
    }
    inputs.dl.front() = {0, 0};
    inputs.du.back() = {0, 0};
    if (dominant) {
        for (int i = 0; i < m; ++i) {
            const float offDiagonal = std::hypot(inputs.dl[i].x, inputs.dl[i].y) +
                std::hypot(inputs.du[i].x, inputs.du[i].y);
            inputs.d[i].x = std::abs(inputs.d[i].x) + 1.25F * offDiagonal + 1.0F;
        }
    }
    for (auto& value : inputs.b) { value = sample(); }
    return inputs;
}

Inputs MakeInputs(int m, int n, int ldb, bool normal, bool dominant)
{
    return MakeInputs(m, n, ldb, normal, dominant, 20261028U + m * 17U + n * 31U + ldb);
}

TEST(Cgtsv2Golden, PartialPivotingResidual)
{
    for (int m : {3, 7, 31, 128, 1024}) {
        for (bool zeroPivot : {false, true}) {
            SCOPED_TRACE(testing::Message() << "m=" << m << ",zero_pivot=" << zeroPivot);
            auto inputs = MakeInputs(m, 3, m + 1, false, false);
            if (zeroPivot) {
                inputs.d[0] = {0, 0};
                inputs.dl[1] = {1, -0.25F};
                inputs.du[0] = {1, 0.25F};
                inputs.dl[2] = {0, 0};
                inputs.du[1] = {0, 0};
            }
            const auto dl = Promote(inputs.dl);
            const auto d = Promote(inputs.d);
            const auto du = Promote(inputs.du);
            const auto b = Promote(inputs.b);
            const auto answer = cgtsv2_test::SolveGolden(m, 3, m + 1, dl, d, du, b);
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < m; ++i) {
                    const size_t index = static_cast<size_t>(j) * m + i;
                    const auto rhs = b[static_cast<size_t>(j) * (m + 1) + i];
                    Complex product = d[i] * answer[index];
                    double scale = std::abs(rhs) + std::abs(product);
                    if (i > 0) {
                        product += dl[i] * answer[index - 1];
                        scale += std::abs(dl[i] * answer[index - 1]);
                    }
                    if (i + 1 < m) {
                        product += du[i] * answer[index + 1];
                        scale += std::abs(du[i] * answer[index + 1]);
                    }
                    EXPECT_LE(std::abs(product - rhs) / std::max(1.0, scale), 1e-11);
                }
            }
        }
    }
}

class Cgtsv2Test : public testing::Test {
protected:
    static void SetUpTestSuite() { env_ = std::make_unique<sparse_test::AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }
    void SetUp() override
    {
        handleOwner_ = std::make_unique<sparse_test::HandleManager>();
        handle_ = handleOwner_->get();
        ASSERT_EQ(aclsparseSetStream(handle_, env_->stream()), ACL_SPARSE_STATUS_SUCCESS);
    }
    void TearDown() override
    {
        EXPECT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
        handleOwner_.reset();
        handle_ = nullptr;
    }
    void Run(int m, int n, int padding, const Inputs& inputs, aclrtStream stream,
        SolvePath expectedPath = SolvePath::ANY, AccuracyCheck check = ExpectAccuracy)
    {
        const int ldb = m + padding;
        SCOPED_TRACE(testing::Message() << "m=" << m << ",n=" << n << ",ldb=" << ldb);
        const auto expected = cgtsv2_test::SolveGolden(m, n, ldb, Promote(inputs.dl), Promote(inputs.d),
            Promote(inputs.du), Promote(inputs.b));
        cgtsv2_test::NpuResult result;
        ASSERT_NO_FATAL_FAILURE(cgtsv2_test::RunOnNpu(handle_, m, n, ldb, inputs, stream, result));
        ASSERT_EQ(result.outputs.size(), 3U);
        ASSERT_EQ(result.dispatchFlags.size(), result.outputs.size());
        for (size_t repeat = 0; repeat < result.outputs.size(); ++repeat) {
            ExpectDispatch(result.dispatchFlags[repeat], expectedPath);
            check(result.outputs[repeat], expected, m, n, ldb);
            ExpectBytesEqual(result.outputs[repeat], result.outputs.front());
            ExpectPaddingUnchanged(result.outputs[repeat], inputs.b, m, n, ldb);
        }
        ASSERT_EQ(aclsparseSetStream(handle_, env_->stream()), ACL_SPARSE_STATUS_SUCCESS);
    }
    inline static std::unique_ptr<sparse_test::AclEnvScope> env_;
    aclsparseHandle_t handle_ = nullptr;
    std::unique_ptr<sparse_test::HandleManager> handleOwner_;
};

class Cgtsv2CsvTest : public Cgtsv2Test, public testing::WithParamInterface<cgtsv2_test::Cgtsv2Param> {};

TEST_P(Cgtsv2CsvTest, AccuracyAndInterface)
{
    const auto& p = GetParam();
    std::cout << "Cgtsv2 CSV ";
    cgtsv2_test::PrintTo(p, &std::cout);
    std::cout << '\n';
    SCOPED_TRACE(testing::Message() << p.caseId() << ",seed=" << p.seed);
    const SolvePath path = p.expectedPath == "pcr" ? SolvePath::PCR :
        (p.expectedPath == "twofold" ? SolvePath::TWOFOLD : SolvePath::ANY);
    Run(p.m, p.n, p.ldb - p.m,
        MakeInputs(p.m, p.n, p.ldb, p.distribution == "normal", p.dominant, p.seed), env_->stream(), path);
}

INSTANTIATE_TEST_SUITE_P(Cgtsv2Cases, Cgtsv2CsvTest,
    testing::ValuesIn(cgtsv2_test::LoadCases("cgtsv2_test.csv")),
    [](const testing::TestParamInfo<cgtsv2_test::Cgtsv2Param>& info) { return info.param.caseId(); });

TEST_F(Cgtsv2Test, ExtendedRangeBidiagonal)
{
    // The solution grows exponentially beyond FP32 while the complex128
    // Golden remains finite. Narrow only the final output, never intermediates.
    for (int m : {255, 511}) {
        auto inputs = MakeInputs(m, 2, m + 1, false, false);
        for (int i = 0; i < m; ++i) {
            inputs.d[i] = {1.0f, 0.1f};
            inputs.dl[i] = i == 0 ? Value{0, 0} : Value{2.0f, 0.2f};
            inputs.du[i] = {0, 0};
        }
        Run(m, 2, 1, inputs, env_->stream());
        if (HasFatalFailure()) { return; }
    }
}

TEST_F(Cgtsv2Test, SingleRowBreaksPcrDispatch)
{
    constexpr int m = 1025;
    const auto stableInputs = MakeInputs(m, 3, m + 1, false, true);
    Run(m, 3, 1, stableInputs, env_->stream(), SolvePath::PCR);
    if (HasFatalFailure()) { return; }
    // Fail early and late rows, including adjacent Probe reduction partitions.
    for (int row : {0, 31, 32, m - 1}) {
        SCOPED_TRACE(testing::Message() << "broken_row=" << row);
        auto inputs = stableInputs;
        inputs.d[row] = {0.0F, 0.0F};
        Run(m, 3, 1, inputs, env_->stream(), SolvePath::TWOFOLD);
        if (HasFatalFailure()) { return; }
    }
}

TEST_F(Cgtsv2Test, PivotComparisonBeyondFloatRange)
{
    const float high = std::ldexp(1.0f, 127);
    const float low = std::ldexp(1.0f, -120);
    for (int m : {3, 7}) {
        Inputs inputs{std::vector<Value>(m, Value{0, 0}), std::vector<Value>(m, Value{1, 0}),
            std::vector<Value>(m, Value{0, 0}), std::vector<Value>(m + 1, Value{0, 0})};
        inputs.d[0] = {high, 0};
        inputs.d[1] = {high, 0};
        inputs.d[2] = {low, 0};
        inputs.dl[1] = {high, 0};
        inputs.dl[2] = {low, 0};
        inputs.du[0] = {-high, 0};
        inputs.du[1] = {1, 0};
        inputs.b[2] = {1, 0};
        for (int i = 3; i < m; ++i) {
            inputs.d[i] = {static_cast<float>(i - 1), 0};
            inputs.b[i] = {static_cast<float>((i - 2) * (i - 1)), 0};
        }
        // The first elimination produces 2^128; compare before output narrowing.
        // Every input and final solution in this regression remains finite.
        Run(m, 1, 1, inputs, env_->stream());
        if (HasFatalFailure()) { return; }
    }
    auto inputs = MakeInputs(3, 2, 4, false, false);
    inputs.d.assign(3, Value{3e38f, 1e38f});
    inputs.dl = {{0, 0}, {3e38f, 1e38f}, {1e-30f, -1e-31f}};
    inputs.du = {{-3e38f, -1e38f}, {1e38f, 0}, {0, 0}};
    inputs.b.assign(8, Value{1e38f, 1e38f});
    Run(3, 2, 1, inputs, env_->stream());
}

TEST_F(Cgtsv2Test, DominantExtremeRhsUsesExtendedRange)
{
    const float high = std::ldexp(3.0f, 126);
    for (int n : {1, 129}) {
        for (bool imaginary : {false, true}) {
            for (float sign : {-1.0f, 1.0f}) {
                Inputs inputs{std::vector<Value>(3, Value{0, 0}), std::vector<Value>(3, Value{2, 0}),
                    {{-1, 0}, {0, 0}, {0, 0}}, std::vector<Value>(static_cast<size_t>(4) * n, Value{7, -3})};
                const Value rhs = imaginary ? Value{0, sign * high} : Value{sign * high, 0};
                for (int column = 0; column < n; ++column) {
                    inputs.b[4 * column] = rhs;
                    inputs.b[4 * column + 1] = rhs;
                    inputs.b[4 * column + 2] = {0, 0};
                }
                Run(3, n, 1, inputs, env_->stream(), SolvePath::TWOFOLD);
                if (HasFatalFailure()) { return; }
            }
        }
    }
}

TEST_F(Cgtsv2Test, IndependentRealImaginaryRange)
{
    const float high = std::ldexp(1.0f, 126);
    const float small = std::ldexp(1.0f, -30);
    for (int n : {1, 129}) {
        for (bool rotated : {false, true}) {
            for (float sign : {-1.0f, 1.0f}) {
                Inputs inputs{std::vector<Value>(3, Value{0, 0}), {{std::ldexp(1.0f, -40), 0}, {1, 0}, {1, 0}},
                    {{1, 0}, {0, 0}, {0, 0}}, std::vector<Value>(static_cast<size_t>(4) * n, Value{7, -3})};
                for (int column = 0; column < n; ++column) {
                    inputs.b[4 * column] = rotated ? Value{0, sign * high} : Value{sign * high, 0};
                    inputs.b[4 * column + 1] = rotated ? Value{-sign * small, sign * high} :
                        Value{sign * high, sign * small};
                    inputs.b[4 * column + 2] = {0, 0};
                }
                Run(3, n, 1, inputs, env_->stream(), SolvePath::TWOFOLD);
                if (HasFatalFailure()) { return; }
            }
        }
    }
}

TEST_F(Cgtsv2Test, ManyRhsFallback)
{
    for (int n : {129, 131072}) {
        Inputs inputs{std::vector<Value>(3, Value{0, 0}), std::vector<Value>(3, Value{1, 0}),
            {{2, 0}, {0, 0}, {0, 0}}, std::vector<Value>(static_cast<size_t>(4) * n, Value{2, -1})};
        Run(3, n, 1, inputs, env_->stream(), SolvePath::TWOFOLD);
        if (HasFatalFailure()) { return; }
    }
}

TEST_F(Cgtsv2Test, NonfiniteRhsClassification)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (int n : {1, 129}) {
        for (Value rhs : {Value{nan, 1}, Value{inf, inf}, Value{-inf, inf}}) {
            Inputs inputs{std::vector<Value>(3, Value{0, 0}), std::vector<Value>(3, Value{1, 0}),
                std::vector<Value>(3, Value{0, 0}), std::vector<Value>(static_cast<size_t>(4) * n, Value{7, -3})};
            for (int column = 0; column < n; ++column) {
                for (int row = 0; row < 3; ++row) {
                    inputs.b[4 * column + row] = rhs;
                }
            }
            Run(3, n, 1, inputs, env_->stream(), SolvePath::TWOFOLD, ExpectSpecialAccuracy);
            if (HasFatalFailure()) { return; }
        }
    }
}

TEST_F(Cgtsv2Test, ForcedPivotAndNonsingularZeroDiagonal)
{
    for (int pattern = 0; pattern < 7; ++pattern) {
        SCOPED_TRACE(pattern);
        auto inputs = MakeInputs(7, 3, 8, false, true);
        if (pattern <= 1) {
            inputs.d[0] = pattern == 0 ? Value{0, 0} : Value{1e-8F, -1e-8F};
            inputs.du[0] = {1, 0.25F};
            inputs.dl[1] = {1, -0.25F};
            inputs.du[1] = {0, 0};
            inputs.dl[2] = {0, 0};
        } else if (pattern == 2) {
            for (int i = 0; i < 7; ++i) {
                inputs.d[i] = {1, 0.1F};
                inputs.dl[i] = i == 0 ? Value{0, 0} : Value{2, -0.25F};
                inputs.du[i] = i == 6 ? Value{0, 0} : Value{1, 0.25F};
            }
        } else {
            for (int i = 0; i < 7; ++i) {
                inputs.d[i] = {10, 0.1F};
                inputs.dl[i] = i == 0 ? Value{0, 0} : Value{1, -0.25F};
                inputs.du[i] = i == 6 ? Value{0, 0} : Value{1, 0.25F};
            }
            if (pattern == 3) { // First-row swap only.
                inputs.d[0] = {1, 0.1F};
                inputs.dl[1] = {100, -0.25F};
                inputs.du[0] = {100, 0.25F};
            } else if (pattern == 4) { // Last-row swap only.
                inputs.d[6] = {1, 0.1F};
                inputs.dl[6] = {100, -0.25F};
            } else { // Extreme ratios in both pivot directions.
                for (int i = 0; i < 7; ++i) {
                    if (pattern == 5 && i > 0) { inputs.dl[i] = {1e6F, -1e5F}; }
                    if (pattern == 6) { inputs.d[i] = {1e6F, 1e5F}; }
                }
            }
        }
        Run(7, 3, 1, inputs, env_->stream());
        if (HasFatalFailure()) { return; }
    }
}

TEST_F(Cgtsv2Test, AlternateStream)
{
    aclrtStream stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    const auto releaseStream = [this](void* ownedStream) {
        EXPECT_EQ(aclrtSynchronizeStream(ownedStream), ACL_SUCCESS);
        EXPECT_EQ(aclsparseSetStream(handle_, env_->stream()), ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(aclrtDestroyStream(ownedStream), ACL_SUCCESS);
    };
    std::unique_ptr<void, decltype(releaseStream)> ownedStream(stream, releaseStream);
    Run(128, 3, 32, MakeInputs(128, 3, 160, true, false), stream);
    // An input update enqueued on the bound stream must precede the solver.
    // Identity coefficients turn this into an observable stream-ordering check.
    Inputs identity{std::vector<Value>(7, Value{0, 0}), std::vector<Value>(7, Value{1, 0}),
        std::vector<Value>(7, Value{0, 0}), std::vector<Value>(7, Value{5, -3})};
    GuardedDeviceBuffer dl(7 * sizeof(Value));
    GuardedDeviceBuffer d(7 * sizeof(Value));
    GuardedDeviceBuffer du(7 * sizeof(Value));
    GuardedDeviceBuffer b(7 * sizeof(Value));
    dl.Upload(identity.dl);
    d.Upload(identity.d);
    du.Upload(identity.du);
    b.Upload(identity.b);
    size_t bytes = 0;
    ASSERT_EQ(aclsparseCgtsv2_bufferSizeExt(handle_, 7, 1, dl.values(), d.values(), du.values(), b.values(),
        7, &bytes), ACL_SPARSE_STATUS_SUCCESS);
    GuardedDeviceBuffer workspace(bytes);
    EXPECT_EQ(aclsparseSetStream(handle_, stream), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclrtMemsetAsync(b.data(), b.size(), 0, b.size(), stream), ACL_SUCCESS);
    EXPECT_EQ(aclsparseCgtsv2(handle_, 7, 1, dl.values(), d.values(), du.values(), b.values(), 7,
        workspace.data()), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    ExpectAccuracy(b.Download(), std::vector<Complex>(7, Complex{}), 7, 1, 7);
    dl.CheckGuards();
    d.CheckGuards();
    du.CheckGuards();
    b.CheckGuards();
    workspace.CheckGuards();
}

TEST_F(Cgtsv2Test, StaleRuntimeErrorDoesNotFailLaunch)
{
    for (int n : {3, 129}) {
        (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
        ASSERT_NE(aclrtSetDevice(-1), ACL_SUCCESS);
        ASSERT_NE(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL), ACL_SUCCESS);
        Run(31, n, 1, MakeInputs(31, n, 32, false, true), env_->stream(), SolvePath::PCR);
        EXPECT_EQ(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL), ACL_SUCCESS);
    }
}

TEST_F(Cgtsv2Test, EmptyRhsAndDimensionValidation)
{
    size_t bytes = 999;
    EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle_, 3, 0, nullptr, nullptr, nullptr, nullptr, 3, &bytes),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bytes, 0U);
    EXPECT_EQ(aclsparseCgtsv2(handle_, 3, 0, nullptr, nullptr, nullptr, nullptr, 3, nullptr),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle_, 3, 0, nullptr, nullptr, nullptr, nullptr, 3, nullptr),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(nullptr, 3, 0, nullptr, nullptr, nullptr, nullptr, 3, &bytes),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseCgtsv2(nullptr, 3, 0, nullptr, nullptr, nullptr, nullptr, 3, nullptr),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    for (int padding : {0, 1, 32}) {
        Run(7, 0, padding, MakeInputs(7, 0, 7 + padding, false, false), env_->stream());
        if (HasFatalFailure()) { return; }
    }
    for (const auto& dimensions : {std::array<int, 3>{2, 0, 3}, {3, -1, 3}, {3, 0, 2}, {-1, 0, 3}}) {
        EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle_, dimensions[0], dimensions[1], nullptr, nullptr,
            nullptr, nullptr, dimensions[2], &bytes), ACL_SPARSE_STATUS_INVALID_VALUE);
        EXPECT_EQ(aclsparseCgtsv2(handle_, dimensions[0], dimensions[1], nullptr, nullptr,
            nullptr, nullptr, dimensions[2], nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    }
}

TEST_F(Cgtsv2Test, NullPointersAlignmentAndOverflow)
{
    auto inputs = MakeInputs(3, 1, 3, false, true);
    GuardedDeviceBuffer dl(3 * sizeof(Value));
    GuardedDeviceBuffer d(3 * sizeof(Value));
    GuardedDeviceBuffer du(3 * sizeof(Value));
    GuardedDeviceBuffer b(3 * sizeof(Value));
    dl.Upload(inputs.dl);
    d.Upload(inputs.d);
    du.Upload(inputs.du);
    b.Upload(inputs.b);
    size_t bytes = 0;
    EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(nullptr, 3, 1, dl.values(), d.values(), du.values(), b.values(),
        3, &bytes), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseCgtsv2(nullptr, 3, 1, dl.values(), d.values(), du.values(), b.values(), 3, nullptr),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle_, 3, 1, dl.values(), d.values(), du.values(), b.values(),
        3, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(aclsparseCgtsv2_bufferSizeExt(handle_, 3, 1, dl.values(), d.values(), du.values(), b.values(),
        3, &bytes), ACL_SPARSE_STATUS_SUCCESS);
    GuardedDeviceBuffer workspace(bytes + 128);
    for (int missing = 0; missing < 4; ++missing) {
        SCOPED_TRACE(missing);
        std::array<Value*, 4> pointers{dl.values(), d.values(), du.values(), b.values()};
        pointers[missing] = nullptr;
        EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle_, 3, 1, pointers[0], pointers[1], pointers[2],
            pointers[3], 3, &bytes), ACL_SPARSE_STATUS_INVALID_VALUE);
        EXPECT_EQ(aclsparseCgtsv2(handle_, 3, 1, pointers[0], pointers[1], pointers[2], pointers[3],
            3, workspace.data()), ACL_SPARSE_STATUS_INVALID_VALUE);
    }
    EXPECT_EQ(aclsparseCgtsv2(handle_, 3, 1, dl.values(), d.values(), du.values(), b.values(), 3, nullptr),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCgtsv2(handle_, 3, 1, dl.values(), d.values(), du.values(), b.values(), 3,
        static_cast<unsigned char*>(workspace.data()) + 1), ACL_SPARSE_STATUS_INVALID_VALUE);
    // All dimensions are positive, but complex64 B bytes exceed the addressable
    // size at INT_MAX squared. The host must reject before any kernel dispatch.
    EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle_, INT_MAX, INT_MAX, dl.values(), d.values(), du.values(),
        b.values(), INT_MAX, &bytes), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCgtsv2(handle_, INT_MAX, INT_MAX, dl.values(), d.values(), du.values(), b.values(),
        INT_MAX, workspace.data()), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
    ExpectBytesEqual(b.Download(), inputs.b);
    dl.CheckGuards();
    d.CheckGuards();
    du.CheckGuards();
    b.CheckGuards();
    workspace.CheckGuards();
}

TEST_F(Cgtsv2Test, SingularSystemRecordsStatusAndOutput)
{
    Inputs inputs{std::vector<Value>(7, Value{0, 0}), std::vector<Value>(7, Value{0, 0}),
        std::vector<Value>(7, Value{0, 0}), std::vector<Value>(8 * 3, Value{1, -1})};
    GuardedDeviceBuffer dl(7 * sizeof(Value));
    GuardedDeviceBuffer d(7 * sizeof(Value));
    GuardedDeviceBuffer du(7 * sizeof(Value));
    GuardedDeviceBuffer b(24 * sizeof(Value));
    dl.Upload(inputs.dl);
    d.Upload(inputs.d);
    du.Upload(inputs.du);
    b.Upload(inputs.b);
    size_t bytes = 0;
    ASSERT_EQ(aclsparseCgtsv2_bufferSizeExt(handle_, 7, 3, dl.values(), d.values(), du.values(), b.values(),
        8, &bytes), ACL_SPARSE_STATUS_SUCCESS);
    GuardedDeviceBuffer workspace(bytes);
    const auto status = aclsparseCgtsv2(handle_, 7, 3, dl.values(), d.values(), du.values(), b.values(),
        8, workspace.data());
    RecordProperty("singular_status", static_cast<int>(status));
    std::cout << "Cgtsv2 singular status=" << static_cast<int>(status) << '\n';
    EXPECT_TRUE(status == ACL_SPARSE_STATUS_SUCCESS || status == ACL_SPARSE_STATUS_INTERNAL_ERROR);
    ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
    const auto output = b.Download();
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 7; ++i) {
            const auto& value = output[j * 8 + i];
            std::cout << "singular B[" << i << ',' << j << "]=" << value.x << ',' << value.y << '\n';
        }
        EXPECT_EQ(std::memcmp(&output[j * 8 + 7], &inputs.b[j * 8 + 7], sizeof(Value)), 0);
    }
    ExpectBytesEqual(dl.Download(), inputs.dl);
    ExpectBytesEqual(d.Download(), inputs.d);
    ExpectBytesEqual(du.Download(), inputs.du);
    dl.CheckGuards();
    d.CheckGuards();
    du.CheckGuards();
    b.CheckGuards();
    workspace.CheckGuards();
}

} // namespace
