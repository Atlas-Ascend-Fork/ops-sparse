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
#include "verify.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <limits>
#include <type_traits>

namespace cgtsv2_test {
using namespace sparse_test;
static_assert(std::is_same_v<Complex, aclsparseComplex>);
static_assert(sizeof(Complex) == 8);
static_assert(alignof(Complex) == alignof(float));
static_assert(offsetof(Complex, x) == 0 && offsetof(Complex, y) == 4);
static_assert(std::is_standard_layout_v<Complex> && std::is_trivially_copyable_v<Complex>);

void CheckReadOnly(const DeviceBuffer &device, const std::vector<Complex> &expected)
{
    std::vector<Complex> actual(expected.size());
    device.copyToHost(actual.data(), actual.size() * sizeof(Complex));
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(Complex)), 0);
}

void CheckResult(const Input &x, const std::vector<Complex> &actual, const std::vector<High> &expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (int component = 0; component < 2; ++component) {
        std::vector<float> got;
        std::vector<double> golden;
        for (int j = 0; j < x.n; ++j) {
            for (int i = 0; i < x.m; ++i) {
                const size_t k = static_cast<size_t>(j) * x.ldb + i;
                const float a = component ? actual[k].y : actual[k].x;
                const double g = component ? expected[k].imag() : expected[k].real();
                got.push_back(a);
                golden.push_back(g);
            }
            for (int i = x.m; i < x.ldb; ++i) {
                const size_t k = static_cast<size_t>(j) * x.ldb + i;
                ASSERT_EQ(std::memcmp(&actual[k], &x.b[k], sizeof(Complex)), 0);
            }
        }
        // Keep this deterministic ST's stricter 100% mixed-tolerance requirement.
        // The maximum-error anchor cap and nonfinite rules follow the official
        // task exactly, using the ORIGINAL unrounded complex128 reference.
        VerifyConfig config;
        config.SetMode(PrecisionMode::MIXED_TOLERANCE_ANCHOR).SetMixedTol(0x1p-13, 0x1p-13);
        config.SetMixedRequiredRatio(1.0);
        MixedAnchorResult verdict;
        Verifier::verifyVector(got, golden, config, component ? "cgtsv2_imag" : "cgtsv2_real", &verdict);
        const auto flags = std::cout.flags();
        const auto precision = std::cout.precision();
        std::cout << std::scientific << std::setprecision(17)
            << (component ? "cgtsv2_imag" : "cgtsv2_real")
            << " matched=" << verdict.matched << '/' << verdict.total
            << " required_ratio=1.0 max_abs_error=" << verdict.maxAbsError
            << " anchor_flat=" << verdict.maxErrorIndex
            << " anchor_golden128=" << verdict.anchorGolden
            << " anchor_low32=" << verdict.anchorLow << " anchor_ulp=" << verdict.anchorUlp
            << " anchor_limit=" << verdict.absoluteLimit
            << " verdict=" << verdict.passed << " reason=" << verdict.reason << '\n';
        // Capture every active value for small fixtures, so a future disagreement
        // can be regraded from original DUT bytes rather than a rounded summary.
        if (got.size() <= 4096) {
            for (size_t k = 0; k < got.size(); ++k) {
                std::cout << "cgtsv2_raw component=" << component << " index=" << k
                    << " actual=" << got[k] << " golden128=" << golden[k] << '\n';
            }
        }
        std::cout.flags(flags);
        std::cout.precision(precision);
        EXPECT_TRUE(verdict.passed) << verdict.reason;
    }
}

// Inspect the arch22 private workspace prefix without changing operator dispatch.
size_t InternalStatusOffset(int m)
{
    const auto aligned = [](size_t bytes) { return (bytes + 127U) & ~size_t(127U); };
    return aligned((m - 1) * sizeof(Complex)) + aligned(m * sizeof(Complex)) +
        aligned((m - 1) * sizeof(Complex)) + aligned((m - 2) * sizeof(Complex)) +
        aligned(m * sizeof(uint32_t));
}

void ReadInternalStatus(const Input &x, void *workspace, std::array<uint32_t, 4> &status)
{
    const auto address = static_cast<uint8_t *>(workspace) + InternalStatusOffset(x.m);
    ASSERT_EQ(aclrtMemcpy(status.data(), sizeof(status), address, sizeof(status),
        ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
}

void CheckParallelJacobiSnapshot(const Input &x, void *workspace)
{
    constexpr uint32_t jacobiFlag = 16U;
    std::array<uint32_t, 4> status{};
    ASSERT_NO_FATAL_FAILURE(ReadInternalStatus(x, workspace, status));
    ASSERT_EQ(status[0], jacobiFlag);
    std::vector<Complex> snapshot(x.m);
    // The status occupies one aligned 128-byte region, followed by RHS scratch.
    const auto address = static_cast<uint8_t *>(workspace) + InternalStatusOffset(x.m) + 128U;
    ASSERT_EQ(aclrtMemcpy(snapshot.data(), snapshot.size() * sizeof(Complex), address,
        snapshot.size() * sizeof(Complex), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    // Only parallelSingleRhs publishes this snapshot on the Jacobi route.
    ASSERT_EQ(std::memcmp(snapshot.data(), x.b.data(), snapshot.size() * sizeof(Complex)), 0);
    std::cout << "parallelSingleRhs Jacobi m=" << x.m << " ldb=" << x.ldb
        << " status=" << status[0] << " original_B_snapshot=verified\n";
}

class Cgtsv2Test : public testing::Test {
protected:
    inline static std::unique_ptr<AclEnvScope> env;
    static void SetUpTestSuite() { env = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env.reset(); }
    std::unique_ptr<HandleManager> handle;
    void SetUp() override
    {
        handle = std::make_unique<HandleManager>();
        handle->setStream(env->stream());
    }
    void TearDown() override { handle.reset(); }
    void CheckFiniteSolution(const Input &x, std::vector<Complex> &actual,
        const std::vector<High> *expected = nullptr)
    {
        const auto golden = expected == nullptr ? Golden(x) : *expected;
        DeviceInput device(x);
        size_t bytes = 0;
        ASSERT_EQ(Query(*handle, device, x, &bytes), ACL_SPARSE_STATUS_SUCCESS);
        auto workspace = DeviceBuffer::alloc(bytes);
        ASSERT_EQ(Solve(*handle, device, x, workspace.get()), ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
        actual.resize(x.b.size());
        device.b.copyToHost(actual.data(), actual.size() * sizeof(Complex));
        for (int i = 0; i < x.m; ++i) {
            ASSERT_TRUE(std::isfinite(actual[i].x)) << i;
            ASSERT_TRUE(std::isfinite(actual[i].y)) << i;
        }
        ASSERT_NO_FATAL_FAILURE(CheckResult(x, actual, golden));
        CheckReadOnly(device.dl, x.dl);
        CheckReadOnly(device.d, x.d);
        CheckReadOnly(device.du, x.du);
    }
};

class Cgtsv2CsvTest : public Cgtsv2Test, public testing::WithParamInterface<Param> {};

void CheckWorkspaceGuards(const DeviceBuffer &guarded, size_t capacity, size_t begin, size_t bytes)
{
    std::vector<uint8_t> after(capacity);
    guarded.copyToHost(after.data(), after.size());
    for (size_t i = 0; i < after.size(); ++i) {
        if (i < begin || i >= begin + bytes) { ASSERT_EQ(after[i], 0xA5); }
    }
}

void CheckCsvNumericalResult(const std::string &kind, const Input &x,
    const std::vector<Complex> &actual, const std::vector<High> &golden)
{
    ASSERT_NO_FATAL_FAILURE(CheckResult(x, actual, golden));
    if (kind.rfind("jacobi_boundary_", 0) != 0) { return; }
    // Additional reviewer boundary check, separate from the unchanged task comparator.
    for (int column = 0; column < x.n; ++column) {
        for (int row = 0; row < x.m; ++row) {
            const size_t index = static_cast<size_t>(column) * x.ldb + row;
            EXPECT_LE(std::abs(actual[index].x - golden[index].real()),
                1e-8 + 1e-5 * std::abs(golden[index].real()));
            EXPECT_LE(std::abs(actual[index].y - golden[index].imag()),
                1e-8 + 1e-5 * std::abs(golden[index].imag()));
        }
    }
}

TEST_P(Cgtsv2CsvTest, NumericalReadOnlyPaddingWorkspaceAndDeterminism)
{
    const auto p = GetParam();
    auto x = MakeInput(p.m, p.n, p.ldb, p.kind);
    const auto golden = Golden(x);
    DeviceInput d(x);
    const bool jacobiBoundary = p.kind.rfind("jacobi_boundary_", 0) == 0;
    const bool boundaryAccepted = jacobiBoundary && p.kind != "jacobi_boundary_126" && p.kind != "jacobi_boundary_135";
    const bool parallelJacobi = x.n == 1 && x.m >= 1024 && (p.kind == "dominant" || boundaryAccepted);
    if (parallelJacobi || jacobiBoundary) {
        ASSERT_EQ(reinterpret_cast<uintptr_t>(d.b.get()) % 64U, 0U);
        ASSERT_EQ(x.ldb % 4, 0);
    }
    size_t bytes = 0;
    ASSERT_EQ(Query(*handle, d, x, &bytes), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_GT(bytes, 0U);
    std::vector<uint8_t> sentinel(bytes + 384, 0xA5);
    auto guarded = DeviceBuffer::copyFrom(sentinel.data(), sentinel.size());
    const uintptr_t base = reinterpret_cast<uintptr_t>(guarded.get());
    const uintptr_t aligned = ((base + 127U) & ~uintptr_t(127U)) + 128U;
    std::vector<Complex> previous;
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_EQ(aclrtMemcpy(d.b.get(), x.b.size() * sizeof(Complex), x.b.data(),
            x.b.size() * sizeof(Complex), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        ASSERT_EQ(Solve(*handle, d, x, reinterpret_cast<void *>(aligned)), ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
        if (jacobiBoundary) {
            std::array<uint32_t, 4> status{};
            ASSERT_NO_FATAL_FAILURE(ReadInternalStatus(x, reinterpret_cast<void *>(aligned), status));
            // Ratios 1.26/1.35 fall back; 1.36/1.5/2/2.01 use refined Jacobi.
            ASSERT_EQ(status[0], boundaryAccepted ? 16U : 8U);
        }
        if (parallelJacobi) {
            ASSERT_NO_FATAL_FAILURE(CheckParallelJacobiSnapshot(x, reinterpret_cast<void *>(aligned)));
        }
        std::vector<Complex> actual(x.b.size());
        d.b.copyToHost(actual.data(), actual.size() * sizeof(Complex));
        ASSERT_NO_FATAL_FAILURE(CheckCsvNumericalResult(p.kind, x, actual, golden));
        if (repeat) {
            EXPECT_EQ(std::memcmp(actual.data(), previous.data(), actual.size() * sizeof(Complex)), 0);
        }
        previous = actual;
        CheckReadOnly(d.dl, x.dl);
        CheckReadOnly(d.d, x.d);
        CheckReadOnly(d.du, x.du);
        ASSERT_NO_FATAL_FAILURE(CheckWorkspaceGuards(guarded, sentinel.size(), aligned - base, bytes));
    }
}

INSTANTIATE_TEST_SUITE_P(Supplemental, Cgtsv2CsvTest,
    testing::ValuesIn(GetCasesFromCsvChecked<Param>("cgtsv2_test.csv", 26)),
    [](const testing::TestParamInfo<Param> &info) { return info.param.caseId(); });

TEST_F(Cgtsv2Test, ExpandedCorrectionRecoversCancelledLowPart)
{
    for (const int m : {3, 8193}) {
        for (const int power : {79, 80, 81, 100, 126, 149}) {
            SCOPED_TRACE(testing::Message() << "m=" << m << " power=" << power);
            auto x = MakeInput(m, 1, m + 1, "dominant");
            const float t = std::ldexp(1.0f, -power);
            std::fill(x.dl.begin(), x.dl.end(), Complex{0.0f, 0.0f});
            std::fill(x.d.begin(), x.d.end(), Complex{1.0f, 0.0f});
            std::fill(x.du.begin(), x.du.end(), Complex{0.0f, 0.0f});
            std::fill(x.b.begin(), x.b.begin() + m, Complex{1.0f, 0.0f});
            x.d[0] = {t, 0.0f};
            x.du[0] = {1.0f, 0.0f};
            x.du[1] = {-t, 0.0f};
            // The analytic internal second component is 1+t, which rounds to 1.
            // A complex128 solve can lose t before recovering the first component.
            std::vector<High> expected(x.b.size(), High{1.0, 0.0});
            expected[0] = High{-1.0, 0.0};
            std::vector<Complex> actual;
            ASSERT_NO_FATAL_FAILURE(CheckFiniteSolution(x, actual, &expected));
            for (int row = 0; row < 3; ++row) {
                EXPECT_EQ(actual[row].x, row == 0 ? -1.0f : 1.0f);
                EXPECT_EQ(actual[row].y, 0.0f);
            }
        }
    }
}

TEST_F(Cgtsv2Test, FiniteSolutionWithOverflowingBackSubstitutionNumerator)
{
    // Exercise small/large dispatch with an RHS that requires expanded-range solving.
    const float k = std::ldexp(3.0f, 126);
    for (const int m : {3, 8193}) {
        SCOPED_TRACE(m);
        auto x = MakeInput(m, 1, m + 1, "dominant");
        std::fill(x.dl.begin(), x.dl.end(), Complex{0.0f, 0.0f});
        std::fill(x.d.begin(), x.d.end(), Complex{2.0f, 0.0f});
        std::fill(x.du.begin(), x.du.end(), Complex{1.0f, 0.0f});
        x.du.back() = {0.0f, 0.0f};
        std::fill(x.b.begin(), x.b.begin() + m, Complex{0.0f, 0.0f});
        x.b[m - 2] = {k, 0.0f};
        x.b[m - 1] = {-k, 0.0f};
        std::vector<Complex> actual;
        ASSERT_NO_FATAL_FAILURE(CheckFiniteSolution(x, actual));
        // Compute the exact analytical tail in double to avoid test-side overflow.
        EXPECT_EQ(actual[m - 3].x, static_cast<float>(-3.0 * k / 8.0));
        EXPECT_EQ(actual[m - 2].x, static_cast<float>(3.0 * k / 4.0));
        EXPECT_EQ(actual[m - 1].x, -k / 2.0f);
    }
}

TEST_F(Cgtsv2Test, FiniteSolutionWithOverflowingForwardSubstitutionRhs)
{
    const float k = std::ldexp(3.0f, 126);
    for (const int m : {3, 8193}) {
        SCOPED_TRACE(m);
        auto x = MakeInput(m, 1, m + 1, "dominant");
        std::fill(x.dl.begin(), x.dl.end(), Complex{0.0f, 0.0f});
        std::fill(x.d.begin(), x.d.end(), Complex{2.0f, 0.0f});
        std::fill(x.du.begin(), x.du.end(), Complex{0.0f, 0.0f});
        std::fill(x.b.begin(), x.b.begin() + m, Complex{0.0f, 0.0f});
        x.dl[1] = {1.0f, 0.0f};
        x.b[0] = {-k, 0.0f};
        x.b[1] = {k, 0.0f};
        std::vector<Complex> actual;
        ASSERT_NO_FATAL_FAILURE(CheckFiniteSolution(x, actual));
        EXPECT_EQ(actual[0].x, -k / 2.0f);
        EXPECT_EQ(actual[1].x, static_cast<float>(3.0 * k / 4.0));
        EXPECT_EQ(actual[2].x, 0.0f);
    }
}

TEST_F(Cgtsv2Test, ExpandedBackSubstitutionPreservesComplexCrossTerms)
{
    for (const int m : {3, 8193}) {
        for (const int scenario : {0, 1, 2}) {
            SCOPED_TRACE(m);
            SCOPED_TRACE(scenario);
            const float h = std::ldexp(1.0f, scenario == 2 ? 25 : 100);
            const float t = scenario == 2 ? 1.0f : std::ldexp(1.0f, -20);
            auto x = MakeInput(m, 1, m == 3 ? m : m + 1, "dominant");
            std::fill(x.dl.begin(), x.dl.end(), Complex{0.0f, 0.0f});
            std::fill(x.d.begin(), x.d.end(), Complex{1.0f, 0.0f});
            std::fill(x.du.begin(), x.du.end(), Complex{0.0f, 0.0f});
            std::fill(x.b.begin(), x.b.begin() + m, Complex{1.0f, 0.0f});
            x.d[0] = {scenario == 2 ? 2.0f : t, 0.0f};
            x.du[0] = {1.0f, 1.0f};
            x.b[0] = {h, h};
            x.b[1] = {h, t};
            // Use the analytic solution: a complex128 multiply-then-subtract
            // reference can also erase t when the component exponents differ by 120.
            std::vector<High> expected(x.b.size(), High{1.0, 0.0});
            const double first = scenario == 2 ? 0.5 : 1.0;
            expected[0] = {first, -first};
            expected[1] = {h, t};
            if (scenario == 1) {
                x.d[0] = {0.0f, 0.0f};
                x.dl[1] = {t, 0.0f};
                x.d[1] = {1.0f, 1.0f};
                x.du[0] = {1.0f, 0.0f};
                x.du[1] = {1.0f, 1.0f};
                x.b[0] = {h, t};
                x.b[1] = {2.0f * h, 2.0f * h};
                x.b[2] = {h, 0.0f};
                expected[2] = {h, 0.0f};
            }
            std::vector<Complex> actual;
            ASSERT_NO_FATAL_FAILURE(CheckFiniteSolution(x, actual, &expected));
            EXPECT_EQ(actual[0].x, static_cast<float>(first));
            EXPECT_EQ(actual[0].y, static_cast<float>(-first));
        }
    }
}

TEST_F(Cgtsv2Test, LargeNonfiniteRhsUsesExpandedRoute)
{
    constexpr uint32_t expandedFlag = 8U;
    for (const int m : {4097, 8193}) {
        for (const int row : {0, 2047, m - 1}) {
            for (const bool imaginary : {false, true}) {
                SCOPED_TRACE(m);
                SCOPED_TRACE(row);
                SCOPED_TRACE(imaginary);
                auto x = MakeInput(m, 1, (m + 3) & ~3, "dominant");
                auto &component = imaginary ? x.b[row].y : x.b[row].x;
                component = std::numeric_limits<float>::quiet_NaN();
                DeviceInput device(x);
                size_t bytes = 0;
                ASSERT_EQ(Query(*handle, device, x, &bytes), ACL_SPARSE_STATUS_SUCCESS);
                auto workspace = DeviceBuffer::alloc(bytes);
                ASSERT_EQ(Solve(*handle, device, x, workspace.get()), ACL_SPARSE_STATUS_SUCCESS);
                ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
                std::array<uint32_t, 4> status{};
                ASSERT_NO_FATAL_FAILURE(ReadInternalStatus(x, workspace.get(), status));
                ASSERT_EQ(status[0], expandedFlag);
                std::vector<Complex> actual(x.b.size());
                device.b.copyToHost(actual.data(), actual.size() * sizeof(Complex));
                const auto golden = Golden(x);
                // The finite-value accuracy checker deliberately rejects NaN.
                // This special-value regression checks classification separately.
                for (int i = 0; i < x.m; ++i) {
                    ASSERT_TRUE(std::isnan(golden[i].real()));
                    ASSERT_TRUE(std::isnan(golden[i].imag()));
                    ASSERT_TRUE(std::isnan(actual[i].x));
                    ASSERT_TRUE(std::isnan(actual[i].y));
                }
                for (int i = x.m; i < x.ldb; ++i) {
                    ASSERT_EQ(std::memcmp(&actual[i], &x.b[i], sizeof(Complex)), 0);
                }
                CheckReadOnly(device.dl, x.dl);
                CheckReadOnly(device.d, x.d);
                CheckReadOnly(device.du, x.du);
            }
        }
    }
}

TEST_F(Cgtsv2Test, ReusedWorkspaceSameAddressesChangedValuesAndStreamOrder)
{
    auto x = MakeInput(17, 3, 20, "zero_pivot");
    DeviceInput d(x);
    size_t bytes = 0;
    ASSERT_EQ(Query(*handle, d, x, &bytes), ACL_SPARSE_STATUS_SUCCESS);
    auto workspace = DeviceBuffer::alloc(bytes);
    for (int run = 0; run < 3; ++run) {
        x.d[2].x += 0.25f;
        x.b[0].y += 0.5f;
        const auto golden = Golden(x);
        auto source = DeviceBuffer::copyFrom(x.b.data(), x.b.size() * sizeof(Complex));
        auto destination = DeviceBuffer::alloc(x.b.size() * sizeof(Complex));
        ASSERT_EQ(aclrtMemcpy(d.d.get(), x.d.size() * sizeof(Complex), x.d.data(),
            x.d.size() * sizeof(Complex), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        const size_t bBytes = x.b.size() * sizeof(Complex);
        ASSERT_EQ(aclrtMemcpyAsync(d.b.get(), bBytes, source.raw(), bBytes,
            ACL_MEMCPY_DEVICE_TO_DEVICE, env->stream()), ACL_SUCCESS);
        ASSERT_EQ(Solve(*handle, d, x, workspace.get()), ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclrtMemcpyAsync(destination.get(), bBytes, d.b.raw(), bBytes,
            ACL_MEMCPY_DEVICE_TO_DEVICE, env->stream()), ACL_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
        std::vector<Complex> actual(x.b.size());
        destination.copyToHost(actual.data(), bBytes);
        ASSERT_NO_FATAL_FAILURE(CheckResult(x, actual, golden));
        CheckReadOnly(d.d, x.d);
    }
}

TEST_F(Cgtsv2Test, EmptyRhsAndValidationPriority)
{
    size_t bytes = 123;
    EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle->get(), 3, 0, nullptr, nullptr, nullptr,
        nullptr, 3, &bytes), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bytes, 0U);
    EXPECT_EQ(aclsparseCgtsv2(handle->get(), 3, 0, nullptr, nullptr, nullptr,
        nullptr, 3, nullptr), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(nullptr, 3, 0, nullptr, nullptr, nullptr,
        nullptr, 3, &bytes), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseCgtsv2(nullptr, 3, 0, nullptr, nullptr, nullptr,
        nullptr, 3, nullptr), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle->get(), 3, 0, nullptr, nullptr, nullptr,
        nullptr, 3, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    // Empty RHS retains shape validation in the common Host.
    for (const int m : {INT_MIN, -1, 0, 2, 3, 5}) {
        for (const int ldb : {INT_MIN, -1, 0, 2, 3, 5}) {
            const bool valid = m >= 3 && ldb >= m;
            const auto expected = valid ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_INVALID_VALUE;
            bytes = 123;
            EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle->get(), m, 0, nullptr,
                nullptr, nullptr, nullptr, ldb, &bytes), expected);
            EXPECT_EQ(bytes, valid ? 0U : 123U);
            EXPECT_EQ(aclsparseCgtsv2(handle->get(), m, 0, nullptr,
                nullptr, nullptr, nullptr, ldb, nullptr), expected);
        }
    }
    for (const auto shape : {std::array<int, 3>{2, 0, 3}, {3, -1, 3}, {3, 0, 2}}) {
        EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle->get(), shape[0], shape[1], nullptr,
            nullptr, nullptr, nullptr, shape[2], &bytes), ACL_SPARSE_STATUS_INVALID_VALUE);
        EXPECT_EQ(aclsparseCgtsv2(handle->get(), shape[0], shape[1], nullptr,
            nullptr, nullptr, nullptr, shape[2], nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    }
}

TEST_F(Cgtsv2Test, NullPointersAlignmentAndOverflow)
{
    auto x = MakeInput(3, 1, 3, "dominant");
    DeviceInput d(x);
    size_t bytes = 0;
    ASSERT_EQ(Query(*handle, d, x, &bytes), ACL_SPARSE_STATUS_SUCCESS);
    auto workspace = DeviceBuffer::alloc(bytes + 128);
    const Complex *data[] = {static_cast<const Complex *>(d.dl.raw()),
        static_cast<const Complex *>(d.d.raw()), static_cast<const Complex *>(d.du.raw()),
        static_cast<const Complex *>(d.b.raw())};
    for (int missing = 0; missing < 4; ++missing) {
        const Complex *args[] = {data[0], data[1], data[2], data[3]};
        args[missing] = nullptr;
        EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle->get(), 3, 1, args[0], args[1], args[2],
            args[3], 3, &bytes), ACL_SPARSE_STATUS_INVALID_VALUE);
        EXPECT_EQ(aclsparseCgtsv2(handle->get(), 3, 1, args[0], args[1], args[2],
            const_cast<Complex *>(args[3]), 3, workspace.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    }
    EXPECT_EQ(Solve(*handle, d, x, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(Solve(*handle, d, x, static_cast<uint8_t *>(workspace.get()) + 1),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCgtsv2_bufferSizeExt(handle->get(), INT_MAX, INT_MAX, data[0], data[1],
        data[2], data[3], INT_MAX, &bytes), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCgtsv2(handle->get(), INT_MAX, INT_MAX, data[0], data[1], data[2],
        const_cast<Complex *>(data[3]), INT_MAX, workspace.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Cgtsv2Test, SingularSystemRecordsStatusWithoutMemoryCorruption)
{
    auto x = MakeInput(3, 2, 4, "dominant");
    std::fill(x.d.begin(), x.d.end(), Complex {0, 0});
    std::fill(x.dl.begin(), x.dl.end(), Complex {0, 0});
    std::fill(x.du.begin(), x.du.end(), Complex {0, 0});
    DeviceInput d(x);
    size_t bytes = 0;
    ASSERT_EQ(Query(*handle, d, x, &bytes), ACL_SPARSE_STATUS_SUCCESS);
    auto workspace = DeviceBuffer::alloc(bytes);
    const auto status = Solve(*handle, d, x, workspace.get());
    std::cout << "Cgtsv2 singular API status=" << static_cast<int>(status) << std::endl;
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    std::vector<Complex> actual(x.b.size());
    d.b.copyToHost(actual.data(), actual.size() * sizeof(Complex));
    for (int j = 0; j < x.n; ++j) {
        for (int i = 0; i < x.m; ++i) {
            const auto v = actual[static_cast<size_t>(j) * x.ldb + i];
            std::cout << "singular[" << i << ',' << j << "]=" << v.x << ',' << v.y << '\n';
        }
        const size_t k = static_cast<size_t>(j) * x.ldb + x.m;
        EXPECT_EQ(std::memcmp(&actual[k], &x.b[k], sizeof(Complex)), 0);
    }
    CheckReadOnly(d.dl, x.dl);
    CheckReadOnly(d.d, x.d);
    CheckReadOnly(d.du, x.du);
}

// These host-only controls lock the documented task/frame comparison difference.
TEST(Cgtsv2AccuracyContractTest, MaximumErrorAnchorIsNotAPerElementCap)
{
    std::vector<float> actual(100, 0.0f), rounded(100, 0.0f);
    std::vector<double> golden(100, 0.0);
    golden[0] = rounded[0] = 100000000.0f;
    actual[0] = 100000256.0f;
    golden[1] = rounded[1] = 1.0f;
    actual[1] = 1.02f;
    VerifyConfig anchorConfig;
    anchorConfig.SetMode(PrecisionMode::MIXED_TOLERANCE_ANCHOR).SetMixedTol(0x1p-13, 0x1p-13);
    MixedAnchorResult task;
    EXPECT_TRUE(Verifier::verifyVector(actual, golden, anchorConfig, "shared_anchor_cap", &task));
    EXPECT_TRUE(task.passed);
    EXPECT_EQ(task.matched, 99U);
    EXPECT_EQ(task.maxErrorIndex, 0U);
    EXPECT_DOUBLE_EQ(task.absoluteLimit, 256.0);
    VerifyConfig config;
    config.SetMode(PrecisionMode::MIXED_TOLERANCE).SetMixedTol(0x1p-13, 0x1p-13);
    config.SetMixedRequiredRatio(0.99);
    // A deliberate negative control: frame rejects the second element's 0.02 error.
    EXPECT_FALSE(Verifier::verifyVector(actual, rounded, config, "task_frame_cap_difference"));
}

TEST(Cgtsv2AccuracyContractTest, PreservesGoldenPrecisionAndSignedAnchorUlp)
{
    VerifyConfig anchorConfig;
    anchorConfig.SetMode(PrecisionMode::MIXED_TOLERANCE_ANCHOR).SetMixedTol(0x1p-13, 0x1p-13);
    MixedAnchorResult unrounded;
    EXPECT_TRUE(Verifier::verifyVector(std::vector<float>{1.0f}, std::vector<double>{1.0 + 0x1p-30},
        anchorConfig, "shared_unrounded_golden", &unrounded));
    EXPECT_TRUE(unrounded.passed);
    EXPECT_DOUBLE_EQ(unrounded.maxAbsError, 0x1p-30);
    MixedAnchorResult negative;
    EXPECT_FALSE(Verifier::verifyVector(std::vector<float>{-16777168.0f}, std::vector<double>{-16777216.0},
        anchorConfig, "shared_signed_anchor", &negative));
    EXPECT_FALSE(negative.passed);
    EXPECT_DOUBLE_EQ(negative.anchorUlp, 1.0);
    EXPECT_DOUBLE_EQ(negative.absoluteLimit, 32.0);
    VerifyConfig config;
    config.SetMode(PrecisionMode::MIXED_TOLERANCE).SetMixedTol(0x1p-13, 0x1p-13);
    EXPECT_TRUE(Verifier::verifyVector(std::vector<float>{-16777168.0f},
        std::vector<float>{-16777216.0f}, config, "signed_anchor_ulp_difference"));
}
TEST(Cgtsv2AccuracyContractTest, SharedAnchorRejectsInvalidShapeAndConfiguration)
{
    VerifyConfig config;
    config.SetMode(PrecisionMode::MIXED_TOLERANCE_ANCHOR).SetMixedTol(0x1p-13, 0x1p-13);
    EXPECT_FALSE(Verifier::verifyVector(std::vector<float>{1}, std::vector<double>{}, config, "shape"));
    config.SetMixedRequiredRatio(std::numeric_limits<double>::quiet_NaN());
    EXPECT_FALSE(Verifier::verifyVector(std::vector<float>{1}, std::vector<double>{1}, config, "config"));
    config.SetMixedRequiredRatio(1);
    EXPECT_FALSE(Verifier::verifyVector(std::vector<float>{1}, std::vector<float>{1}, config, "rounded"));
}

TEST(Cgtsv2AccuracyContractTest, SharedAnchorHandlesEmptyAndNonfiniteValues)
{
    VerifyConfig config;
    config.SetMode(PrecisionMode::MIXED_TOLERANCE_ANCHOR).SetMixedTol(0x1p-13, 0x1p-13);
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(Verifier::verifyVector(std::vector<float>{}, std::vector<double>{}, config, "empty"));
    EXPECT_TRUE(Verifier::verifyVector(std::vector<float>{inf}, std::vector<double>{inf}, config, "inf"));
    EXPECT_FALSE(Verifier::verifyVector(std::vector<float>{inf}, std::vector<double>{-inf}, config, "sign"));
    EXPECT_FALSE(Verifier::verifyVector(std::vector<float>{inf}, std::vector<double>{1}, config, "finite"));
    EXPECT_FALSE(Verifier::verifyVector(std::vector<float>{NAN}, std::vector<double>{NAN}, config, "nan"));
}

TEST(Cgtsv2AccuracyContractTest, SharedAnchorUsesConfiguredToleranceAndRatio)
{
    VerifyConfig config;
    config.SetMode(PrecisionMode::MIXED_TOLERANCE_ANCHOR).SetMixedTol(0, 0);
    config.SetMixedRequiredRatio(0.5).SetMixedMaxAbsErrLim(0.1);
    const std::vector<float> output{0, 0.02f};
    const std::vector<double> golden{0, 0};
    EXPECT_TRUE(Verifier::verifyVector(output, golden, config, "half_matched"));
    config.SetMixedRequiredRatio(1);
    EXPECT_FALSE(Verifier::verifyVector(output, golden, config, "all_required"));
    config.SetMixedTol(0.03, 0);
    EXPECT_TRUE(Verifier::verifyVector(output, golden, config, "configured_atol"));
}
} // namespace cgtsv2_test
