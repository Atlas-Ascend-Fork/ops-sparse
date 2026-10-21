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

#include <limits>

#include "../gtsv2_golden.h"
#include "../arch35/gtsv2_npu_wrapper.h"
#include "gtsv2_precision_fixture.h"
#include "gtsv2_jacobi_fixture.h"
#include "gtsv2_range_fixture.h"

using namespace sparse_test;

namespace {

constexpr size_t kArch22BytesPerRow = 4 * sizeof(float) + sizeof(uint8_t);

size_t ExpectedArch22Workspace(int m)
{
    const size_t raw = static_cast<size_t>(m) * kArch22BytesPerRow;
    const size_t factors = (raw + kAlignMask) & ~kAlignMask;
    return (factors + 85 * static_cast<size_t>(m) + kAlignMask) & ~kAlignMask;
}

aclsparseStatus_t QueryDeviceWorkspace(
    HandleManager &handle, Gtsv2DeviceData &device, int m, int n, int ldb, size_t &bytes)
{
    return aclsparseSgtsv2_bufferSizeExt(handle.get(), m, n,
        static_cast<const float *>(device.dDl.raw()), static_cast<const float *>(device.dD.raw()),
        static_cast<const float *>(device.dDu.raw()), static_cast<const float *>(device.dB.raw()),
        ldb, &bytes);
}

void CheckActiveRows(const std::vector<float> &actual, const std::vector<float> &expected,
    int m, int n, int ldb)
{
    std::vector<float> actualValues;
    std::vector<float> expectedValues;
    actualValues.reserve(static_cast<size_t>(m) * n);
    expectedValues.reserve(static_cast<size_t>(m) * n);
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < m; ++row) {
            const size_t index = static_cast<size_t>(column) * ldb + row;
            actualValues.push_back(actual[index]);
            expectedValues.push_back(expected[index]);
        }
    }
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, expectedValues.data(), expectedValues.size());
    EXPECT_TRUE(Verifier::verifyVector(
        actualValues, expectedValues, config, "arch22_async_stream_order"));
}

void CheckSingleRhsValues(const std::vector<float> &actual,
    const std::vector<float> &expected, int m, int ldb, float padding)
{
    for (int row = 0; row < m; ++row) {
        EXPECT_NEAR(actual[row], expected[row],
                    1e-4f * (1.0f + std::abs(expected[row])));
    }
    for (int row = m; row < ldb; ++row) {
        EXPECT_EQ(actual[row], padding);
    }
}

// Only the final copy is followed by a synchronization. The test must still
// detect missing ordering between the initial copy, solve, and final copy.
void RunAsyncSolve(HandleManager &handle, aclrtStream stream, Gtsv2DeviceData &device,
    DeviceBuffer &input, DeviceBuffer &output, int m, int n, int ldb, void *workspace)
{
    const size_t bytes = static_cast<size_t>(ldb) * n * sizeof(float);
    ASSERT_EQ(aclrtMemcpyAsync(device.dB.get(), bytes, input.raw(), bytes,
        ACL_MEMCPY_DEVICE_TO_DEVICE, stream), ACL_SUCCESS);
    ASSERT_EQ(aclsparseSgtsv2(handle.get(), m, n,
        static_cast<const float *>(device.dDl.raw()), static_cast<const float *>(device.dD.raw()),
        static_cast<const float *>(device.dDu.raw()), static_cast<float *>(device.dB.get()),
        ldb, workspace), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtMemcpyAsync(output.get(), bytes, device.dB.raw(), bytes,
        ACL_MEMCPY_DEVICE_TO_DEVICE, stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
}

class Gtsv2Arch22Test : public testing::Test {
public:
    static void SetUpTestSuite()
    {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite()
    {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;
    std::vector<float> dl_;
    std::vector<float> d_;
    std::vector<float> du_;

    void SetUp() override
    {
        stream_ = env_->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);
        auto tri = makeDiagDominantTridiag(4, 1, 100);
        dl_ = tri.dl;
        d_ = tri.d;
        du_ = tri.du;
    }

    void TearDown() override
    {
        handle_.reset();
    }
};

}  // namespace

// These systems have finite FP32 inputs and small finite solutions, even
// though their FP32 elimination intermediates exceed FLT_MAX. Embed the same
// independent block to cover fused, multi-core UB, wide UB and streaming GM.
TEST_F(Gtsv2Arch22Test, FiniteInputsSurviveEliminationOverflowAcrossRoutes)
{
    constexpr float large = 0x1.8p127f;
    const float maximum = std::numeric_limits<float>::max();
    for (int m : {3, 127, 128, 129, 6144, 6145, 8192, 8193}) {
        for (int n : {1, 3}) {
            for (int kind : {0, 1, 2}) {
                SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n) +
                    " kind=" + std::to_string(kind));
                const int ldb = (m + 7) & ~7;
                std::vector<float> dl(m, 0.0f), d(m, 1.0f), du(m, 0.0f);
                std::vector<float> b(static_cast<size_t>(ldb) * n, -123.0f);
                d[0] = large; d[1] = large; d[2] = large;
                dl[1] = kind == 1 ? large / 4 : large;
                du[0] = kind == 1 ? 0.0f : -large / 2;
                if (kind == 2) { d[0] = large / 2; }
                for (int column = 0; column < n; ++column) {
                    const float sign = column == 1 ? -1.0f : 1.0f;
                    const size_t offset = static_cast<size_t>(column) * ldb;
                    for (int row = 0; row < m; ++row) { b[offset + row] = 0.0f; }
                    b[offset] = sign * (kind == 1 ? maximum : large * (kind == 2 ? 0.125f : 0.375f));
                    b[offset + 1] = sign * (kind == 1 ? -maximum : large * 0.75f);
                    b[offset + 2] = sign * large * 0.5f;
                }
                auto result = Gtsv2Npu(*handle_, stream_, dl, d, du, b, m, n, ldb);
                ASSERT_EQ(result.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS);
                ASSERT_EQ(result.computeRet, ACL_SPARSE_STATUS_SUCCESS);
                ASSERT_EQ(result.X.size(), b.size());
                for (int column = 0; column < n; ++column) {
                    const double sign = column == 1 ? -1.0 : 1.0;
                    const size_t offset = static_cast<size_t>(column) * ldb;
                    EXPECT_NEAR(result.X[offset], sign * (kind == 1 ? double(maximum) / large : 0.5), 1e-6);
                    EXPECT_NEAR(result.X[offset + 1], sign * (kind == 1 ? -1.25 * double(maximum) / large : 0.25), 1e-6);
                    EXPECT_EQ(result.X[offset + 2], sign * 0.5);
                    for (int row = 3; row < m; ++row) { EXPECT_EQ(result.X[offset + row], 0.0f); }
                    for (int row = m; row < ldb; ++row) { EXPECT_EQ(result.X[offset + row], -123.0f); }
                }
            }
        }
    }
}

TEST_F(Gtsv2Arch22Test, IndependentExponentsPreserveTinyCoefficients)
{
    const float tiny = std::numeric_limits<float>::denorm_min();
    const float maximum = std::numeric_limits<float>::max();
    // A single scale would erase dl[1] and d[1], changing the solution.
    for (int m : {3, 129, 8193}) {
        for (int n : {1, 3}) {
            const int ldb = m + 3;
            std::vector<float> dl(m, 0.0f), d(m, 1.0f), du(m, 0.0f);
            std::vector<float> b(static_cast<size_t>(ldb) * n, -321.0f);
            d[0] = maximum; d[1] = 2 * tiny; dl[1] = tiny; du[0] = maximum / 2;
            for (int column = 0; column < n; ++column) {
                size_t offset = static_cast<size_t>(column) * ldb;
                for (int row = 0; row < m; ++row) { b[offset + row] = 1.0f; }
                b[offset] = maximum; b[offset + 1] = tiny;
            }
            auto result = Gtsv2Npu(*handle_, stream_, dl, d, du, b, m, n, ldb);
            ASSERT_EQ(result.computeRet, ACL_SPARSE_STATUS_SUCCESS);
            for (int column = 0; column < n; ++column) {
                size_t offset = static_cast<size_t>(column) * ldb;
                for (int row = 0; row < m; ++row) {
                    EXPECT_NEAR(result.X[offset + row], row == 1 ? 0.0f : 1.0f, 1e-6f);
                }
                for (int row = m; row < ldb; ++row) { EXPECT_EQ(result.X[offset + row], -321.0f); }
            }
        }
    }
}

TEST_F(Gtsv2Arch22Test, HugeRhsPreservesWorkspaceCoefficientsAndLaterColumns)
{
    for (int m : {3, 129, 6145, 8193}) {
        for (int n : {1, 5}) {
            ASSERT_NO_FATAL_FAILURE(CheckRangeStorage(*handle_, stream_, m, n));
        }
    }
}

TEST_F(Gtsv2Arch22Test, ExpandedRangeRoundsSubnormalOutputs)
{
    constexpr float large = 0x1.8p127f;
    std::vector<float> dl(3, 0.0f), d(3, large), du(3, 0.0f);
    std::vector<float> b{1.0f, -1.0f, 0.0f};
    auto result = Gtsv2Npu(*handle_, stream_, dl, d, du, b, 3, 1, 3);
    ASSERT_EQ(result.computeRet, ACL_SPARSE_STATUS_SUCCESS);
    const float expected = static_cast<float>(1.0 / static_cast<double>(large));
    EXPECT_EQ(result.X[0], expected);
    EXPECT_EQ(result.X[1], -expected);
    EXPECT_EQ(result.X[2], 0.0f);
    EXPECT_FALSE(std::signbit(result.X[2]));
}

TEST_F(Gtsv2Arch22Test, MixedRowScalesPreserveSolutionAndEquationResidual)
{
    namespace fixture = gtsv2_mixed_scale_fixture;
    for (int m : {12, 127, 128, 129, 6144, 6145, 8192, 8193}) {
        for (int n : {1, 3}) {
            for (bool mixed : {false, true}) {
                SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n) +
                    " mixed=" + std::to_string(mixed));
                const int ldb = m + 3;
                std::vector<float> lower(m, 0.0f), diagonal(m, 1.0f), upper(m, 0.0f);
                std::vector<float> rhs(static_cast<size_t>(ldb) * n, -12345.0f);
                for (size_t row = 0; row < fixture::d.size(); ++row) {
                    const int shift = mixed ? 0 : -fixture::rowExponent[row];
                    lower[row] = std::ldexp(fixture::dl[row], shift);
                    diagonal[row] = std::ldexp(fixture::d[row], shift);
                    upper[row] = std::ldexp(fixture::du[row], shift);
                }
                for (int col = 0; col < n; ++col) {
                    const float scale = col == 1 ? -1.0f : (col == 2 ? 0.5f : 1.0f);
                    const size_t offset = static_cast<size_t>(col) * ldb;
                    for (int row = 0; row < m; ++row) {
                        rhs[offset + row] = row < 12 ? scale * std::ldexp(fixture::rhs[row],
                            mixed ? 0 : -fixture::rowExponent[row]) : scale;
                    }
                }
                const auto input = rhs;
                auto result = Gtsv2Npu(*handle_, stream_, lower, diagonal, upper, rhs, m, n, ldb);
                ASSERT_EQ(result.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS);
                ASSERT_EQ(result.computeRet, ACL_SPARSE_STATUS_SUCCESS);
                ASSERT_EQ(result.X.size(), input.size());
                std::vector<double> expected(input.size(), -12345.0);
                for (int col = 0; col < n; ++col) {
                    const double scale = col == 1 ? -1.0 : (col == 2 ? 0.5 : 1.0);
                    for (int row = 0; row < m; ++row) {
                        expected[static_cast<size_t>(col) * ldb + row] =
                            scale * (row < 12 ? fixture::expected[row] : 1.0);
                    }
                }
                CheckScaledSolution(result.X, expected, lower, diagonal, upper, input, m, n, ldb);
            }
        }
    }
}

TEST_F(Gtsv2Arch22Test, ZeroCorrectionPreservesSignedZero)
{
    const std::vector<float> zero(3, 0.0f);
    const std::vector<float> diagonal(3, 0x1p30f);
    const std::vector<float> rhs{0.0f, 0.0f, -0.0f};
    auto result = Gtsv2Npu(*handle_, stream_, zero, diagonal, zero, rhs, 3, 1, 3);
    ASSERT_EQ(result.computeRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(result.X.size(), rhs.size());
    for (size_t i = 0; i < rhs.size(); ++i) {
        EXPECT_EQ(result.X[i], 0.0f);
        EXPECT_EQ(std::signbit(result.X[i]), std::signbit(rhs[i]));
    }
}

TEST_F(Gtsv2Arch22Test, SmallBackwardErrorStillRequiresForwardAccuracy)
{
    // Exact FP32 inputs. A 500-digit independent solve gives the four goldens
    // below. The old result [1, 2^40, 0, 1] has backward error below 1e-27 but
    // catastrophic forward error, so a small residual must not skip correction.
    const std::vector<float> lower{0.0f, 0x1.8p11f, 0x1p-65f, -0x1p-82f};
    const std::vector<float> diagonal{0x1p62f, -0x1p-38f, -0x1p-112f, -0x1p35f};
    const std::vector<float> upper{-0x1p-112f, -0x1p2f, -0x1p64f, 0.0f};
    const std::vector<float> input{0x1p62f, 3068.0f, -0x1p64f, -0x1p35f};
    const std::vector<double> golden{1.0, -4096.000015258789112238, 1.00000000372529031234, 1.0};
    for (int m : {4, 129, 8193}) {
        for (int n : {1, 3}) {
            const int ldb = m + 3;
            SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n));
            std::vector<float> dl(m, 0.0f), d(m, 1.0f), du(m, 0.0f);
            std::copy(lower.begin(), lower.end(), dl.begin());
            std::copy(diagonal.begin(), diagonal.end(), d.begin());
            std::copy(upper.begin(), upper.end(), du.begin());
            std::vector<float> rhs(static_cast<size_t>(ldb) * n, -12345.0f);
            std::vector<double> expected(rhs.size(), -12345.0);
            for (int col = 0; col < n; ++col) {
                const float scale = col == 1 ? -1.0f : (col == 2 ? 0.5f : 1.0f);
                for (int row = 0; row < m; ++row) {
                    const size_t index = static_cast<size_t>(col) * ldb + row;
                    rhs[index] = scale * (row < 4 ? input[row] : 1.0f);
                    expected[index] = scale * (row < 4 ? golden[row] : 1.0);
                }
            }
            auto result = Gtsv2Npu(*handle_, stream_, dl, d, du, rhs, m, n, ldb);
            ASSERT_EQ(result.computeRet, ACL_SPARSE_STATUS_SUCCESS);
            CheckScaledSolution(result.X, expected, dl, d, du, rhs, m, n, ldb);
        }
    }
}

TEST_F(Gtsv2Arch22Test, RowEquilibrationRetainsTinyCoefficientWithinOneRow)
{
    const float tiny = std::numeric_limits<float>::denorm_min();
    const float maximum = std::numeric_limits<float>::max();
    std::vector<float> lower{0.0f, 0.0f, 0.0f}, diagonal{maximum, tiny, 1.0f};
    std::vector<float> upper{tiny, 0.0f, 0.0f};
    const float product = maximum * tiny;
    std::vector<float> rhs{2 * product, product, 1.0f};
    auto result = Gtsv2Npu(*handle_, stream_, lower, diagonal, upper, rhs, 3, 1, 3);
    ASSERT_EQ(result.computeRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(result.X.size(), rhs.size());
    // Erasing the tiny upper coefficient during row scaling doubles x[0].
    // An absolute tolerance cannot detect that subnormal error; compare exactly.
    EXPECT_EQ(result.X[0], tiny);
    EXPECT_EQ(result.X[1], maximum);
    EXPECT_EQ(result.X[2], 1.0f);
}

TEST_F(Gtsv2Arch22Test, MixedCoefficientScalesPreserveSmallSolutionComponents)
{
    for (const auto &fixture : gtsv2_pivot_scale_fixture::cases) {
        const int m = static_cast<int>(fixture.d.size());
        const int ldb = m + 3;
        for (int n : {1, 3}) {
            SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n));
            std::vector<float> rhs(static_cast<size_t>(ldb) * n, -12345.0f);
            std::vector<double> expected(rhs.size(), -12345.0);
            for (int col = 0; col < n; ++col) {
                const float scale = col == 1 ? 0.0f : (col == 2 ? -1.0f : 1.0f);
                for (int row = 0; row < m; ++row) {
                    const size_t index = static_cast<size_t>(col) * ldb + row;
                    rhs[index] = scale * fixture.rhs[row];
                    expected[index] = scale * fixture.expected[row];
                }
            }
            auto result = Gtsv2Npu(*handle_, stream_, fixture.dl, fixture.d, fixture.du, rhs, m, n, ldb);
            ASSERT_EQ(result.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS);
            ASSERT_EQ(result.computeRet, ACL_SPARSE_STATUS_SUCCESS);
            CheckScaledSolution(result.X, expected, fixture.dl, fixture.d, fixture.du, rhs, m, n, ldb);
        }
    }
}

TEST_F(Gtsv2Arch22Test, VectorJacobiBoundaryShapes)
{
    for (int m : {127, 128, 129, 6143, 6144, 6145, 8191, 8192, 8193}) {
        for (int n : {1, 3}) {
            CheckJacobiRegression(*handle_, stream_, m, n, 1, 1.0f, false, false);
        }
    }
}

TEST_F(Gtsv2Arch22Test, VectorJacobiSignedCoefficients)
{
    for (int m : {129, 4096, 6145, 8192}) {
        CheckJacobiRegression(*handle_, stream_, m, 3, 3, 1.0f, true, false);
    }
}

TEST_F(Gtsv2Arch22Test, VectorJacobiRhsEligibilityBoundary)
{
    for (int m : {257, 6144, 6145, 8192}) {
        for (float scale : {16.0f, std::nextafter(16.0f, INFINITY), 4096.0f}) {
            CheckJacobiRegression(*handle_, stream_, m, 2, 8, scale, false, false);
        }
    }
}

TEST_F(Gtsv2Arch22Test, VectorJacobiReusesEventsForManyUnalignedRhs)
{
    for (int m : {129, 6144, 6145}) {
        CheckJacobiRegression(*handle_, stream_, m, 17, 1, 1.0f, false, false);
    }
}

TEST_F(Gtsv2Arch22Test, VectorJacobiPreservesNonfiniteFallbackAndNextColumn)
{
    for (int m : {128, 6144, 6145, 8192}) {
        CheckJacobiRegression(*handle_, stream_, m, 7, 1, 1.0f, false, true);
    }
}

TEST_F(Gtsv2Arch22Test, VectorJacobiRepeatedMulticoreWaveAndWorkspaceGuards)
{
    for (int m : {4096, 7168, 8192}) {
        CheckJacobiRegression(*handle_, stream_, m, 65, 16, 1.0f, false, false);
    }
}

TEST_F(Gtsv2Arch22Test, WorkspaceIsCompactAlignedAndIndependentOfN)
{
    for (int m : {3, 7, 8, 15, 16, 8192, 8193, 16384}) {
        size_t oneRhsBytes = 0;
        size_t manyRhsBytes = 0;
        ASSERT_EQ(aclsparseSgtsv2_bufferSizeExt(
            handle_->get(), m, 1, dl_.data(), d_.data(), du_.data(),
            nullptr, m, &oneRhsBytes), ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclsparseSgtsv2_bufferSizeExt(
            handle_->get(), m, 100000, dl_.data(), d_.data(), du_.data(),
            nullptr, m, &manyRhsBytes), ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(oneRhsBytes, ExpectedArch22Workspace(m));
        if (m == 3 || m == 7 || m == 8) {
            const size_t documentedBytes = m == 3 ? 384u : (m == 7 ? 768u : 1024u);
            EXPECT_EQ(oneRhsBytes, documentedBytes);
        }
        EXPECT_EQ(manyRhsBytes, oneRhsBytes);
        EXPECT_EQ(oneRhsBytes % kAlignSize, 0u);
    }
}

TEST_F(Gtsv2Arch22Test, MaximumDimensionsUseWideSizeArithmetic)
{
    constexpr int maximum = std::numeric_limits<int>::max();
    const size_t expectedBytes = ExpectedArch22Workspace(maximum);
    size_t workspaceBytes = 0;

    // The largest positive int values still fit the target's uint64_t/size_t
    // arithmetic.  This guards against accidental int32 multiplication in
    // both workspace and B-span validation without launching a kernel.
    EXPECT_EQ(aclsparseSgtsv2_bufferSizeExt(
        handle_->get(), maximum, maximum, dl_.data(), d_.data(),
        du_.data(), nullptr, maximum, &workspaceBytes),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(workspaceBytes, expectedBytes);
}

TEST_F(Gtsv2Arch22Test, RefinesHistoricalCancellationAcrossLaunchAndStorageRoutes)
{
    namespace fixture = gtsv2_precision_fixture;
    for (int m : {17, 129, 8193}) {
        for (int n : {1, 3}) {
            SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n));
            const int ldb = m + 3;
            std::vector<float> dl(m, 0.0f), d(m, 1.0f), du(m, 0.0f), b(ldb * n, 0.0f);
            std::copy(fixture::dl.begin(), fixture::dl.end(), dl.begin());
            std::copy(fixture::d.begin(), fixture::d.end(), d.begin());
            std::copy(fixture::du.begin(), fixture::du.end(), du.begin());
            for (int column = 0; column < n; ++column) {
                for (int row = 0; row < 17; ++row) {
                    b[column * ldb + row] = std::ldexp(fixture::rhs[row], column);
                }
            }
            auto result = Gtsv2Npu(*handle_, stream_, dl, d, du, b, m, n, ldb);
            ASSERT_EQ(result.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS);
            ASSERT_EQ(result.computeRet, ACL_SPARSE_STATUS_SUCCESS);
            ASSERT_EQ(result.X.size(), b.size());
            for (int column = 0; column < n; ++column) {
                for (int row = 0; row < ldb; ++row) {
                    double expected = row < 17 ? std::ldexp(fixture::expected[row], column) : 0.0;
                    float rounded = static_cast<float>(std::abs(expected));
                    double ulp = static_cast<double>(std::nextafter(rounded, INFINITY)) - rounded;
                    double error = std::abs(static_cast<double>(result.X[column * ldb + row]) - expected);
                    EXPECT_LE(error, std::max(0.01, 32.0 * ulp));
                    EXPECT_LE(error, 0x1p-16 + 0x1p-10 * std::abs(expected));
                }
            }
        }
    }
}

TEST_F(Gtsv2Arch22Test, KeepsDiagonalsAndLdbPaddingUnchanged)
{
    constexpr int m = 7;
    constexpr int n = 3;
    constexpr int ldb = 9;
    constexpr float paddingSentinel = -12345.25f;

    auto tri = makeDiagDominantTridiag(m, 1, 20260903);
    auto inputB = makeFullColMajor(m, n, ldb, 20260904, -5.0, 5.0);
    for (int column = 0; column < n; ++column) {
        for (int row = m; row < ldb; ++row) {
            inputB[static_cast<int64_t>(column) * ldb + row] = paddingSentinel;
        }
    }

    auto device = PrepareDeviceData(tri.dl, tri.d, tri.du, inputB, m, n, ldb);

    size_t workspaceBytes = 0;
    ASSERT_EQ(QueryDeviceWorkspace(*handle_, device, m, n, ldb, workspaceBytes),
        ACL_SPARSE_STATUS_SUCCESS);
    AlignedDeviceBuffer workspace =
        AlignedDeviceBuffer::allocAligned(workspaceBytes);
    ASSERT_EQ(aclsparseSgtsv2(
        handle_->get(), m, n,
        reinterpret_cast<const float *>(device.dDl.raw()),
        reinterpret_cast<const float *>(device.dD.raw()),
        reinterpret_cast<const float *>(device.dDu.raw()),
        reinterpret_cast<float *>(device.dB.get()), ldb, workspace.get()),
        ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

    CheckDeviceDiagonals(device, tri.dl, tri.d, tri.du);
    std::vector<float> actualB(static_cast<size_t>(ldb) * n);
    device.dB.copyToHost(actualB.data(), actualB.size() * sizeof(float));
    for (int column = 0; column < n; ++column) {
        for (int row = m; row < ldb; ++row) {
            EXPECT_EQ(actualB[static_cast<int64_t>(column) * ldb + row],
                      paddingSentinel);
        }
    }
}

TEST_F(Gtsv2Arch22Test, HonorsQueriedWorkspaceBounds)
{
    constexpr int m = 17;
    constexpr int n = 5;
    constexpr int ldb = 17;
    constexpr uint8_t guard = 0xA5;

    auto tri = makeDiagDominantTridiag(m, 1, 20260905);
    auto inputB = makeFullColMajor(m, n, ldb, 20260906, -5.0, 5.0);
    auto device = PrepareDeviceData(tri.dl, tri.d, tri.du, inputB, m, n, ldb);

    size_t workspaceBytes = 0;
    ASSERT_EQ(QueryDeviceWorkspace(*handle_, device, m, n, ldb, workspaceBytes),
        ACL_SPARSE_STATUS_SUCCESS);

    const size_t backingBytes = workspaceBytes + 3 * kAlignSize;
    std::vector<uint8_t> initial(backingBytes, guard);
    auto backing = DeviceBuffer::copyFrom(initial.data(), backingBytes);
    const uintptr_t base = reinterpret_cast<uintptr_t>(backing.get());
    const uintptr_t aligned =
        (base + kAlignSize + kAlignMask) & ~static_cast<uintptr_t>(kAlignMask);
    const size_t prefixBytes = aligned - base;
    ASSERT_GE(prefixBytes, kAlignSize);
    ASSERT_LE(prefixBytes + workspaceBytes + kAlignSize, backingBytes);

    ASSERT_EQ(aclsparseSgtsv2(
        handle_->get(), m, n,
        reinterpret_cast<const float *>(device.dDl.raw()),
        reinterpret_cast<const float *>(device.dD.raw()),
        reinterpret_cast<const float *>(device.dDu.raw()),
        reinterpret_cast<float *>(device.dB.get()), ldb,
        reinterpret_cast<void *>(aligned)), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

    std::vector<uint8_t> actual(backingBytes);
    backing.copyToHost(actual.data(), actual.size());
    EXPECT_TRUE(std::all_of(actual.begin(), actual.begin() + prefixBytes,
                            [](uint8_t value) { return value == guard; }));
    EXPECT_TRUE(std::all_of(
        actual.begin() + prefixBytes + workspaceBytes, actual.end(),
        [](uint8_t value) { return value == guard; }));
}

TEST_F(Gtsv2Arch22Test, RespectsSameStreamAsyncOrdering)
{
    constexpr int m = 31;
    constexpr int n = 5;
    constexpr int ldb = 32;

    auto tri = makeDiagDominantTridiag(m, 1, 20260907);
    auto inputB = makeFullColMajor(m, n, ldb, 20260908, -5.0, 5.0);
    auto expected = inputB;
    Gtsv2Golden(tri.dl, tri.d, tri.du, expected, m, n, ldb);

    Gtsv2DeviceData device;
    device.dDl = DeviceBuffer::copyFrom(tri.dl.data(), m * sizeof(float));
    device.dD = DeviceBuffer::copyFrom(tri.d.data(), m * sizeof(float));
    device.dDu = DeviceBuffer::copyFrom(tri.du.data(), m * sizeof(float));
    const size_t bBytes = static_cast<size_t>(ldb) * n * sizeof(float);
    auto dInputB = DeviceBuffer::copyFrom(inputB.data(), bBytes);
    device.dB = DeviceBuffer::alloc(bBytes);
    auto dOutputB = DeviceBuffer::alloc(bBytes);

    size_t workspaceBytes = 0;
    ASSERT_EQ(aclsparseSgtsv2_bufferSizeExt(
        handle_->get(), m, n,
        reinterpret_cast<const float *>(device.dDl.raw()),
        reinterpret_cast<const float *>(device.dD.raw()),
        reinterpret_cast<const float *>(device.dDu.raw()), nullptr, ldb,
        &workspaceBytes), ACL_SPARSE_STATUS_SUCCESS);
    AlignedDeviceBuffer workspace =
        AlignedDeviceBuffer::allocAligned(workspaceBytes);

    ASSERT_NO_FATAL_FAILURE(RunAsyncSolve(*handle_, stream_, device,
        dInputB, dOutputB, m, n, ldb, workspace.get()));

    std::vector<float> actual(static_cast<size_t>(ldb) * n);
    dOutputB.copyToHost(actual.data(), bBytes);
    CheckActiveRows(actual, expected, m, n, ldb);
}

TEST_F(Gtsv2Arch22Test, SingleRhsPreservesMemoryAndAsyncDeterminism)
{
    constexpr int m = 17;
    constexpr int ldb = 20;
    constexpr float padding = -12345.25f;
    auto tri = makeDiagDominantTridiag(m, 1, 20260909);
    tri.d[0] = 0.0f;
    tri.dl[1] = 1.0f;
    tri.du[0] = 1.0f;
    auto inputB = makeFullColMajor(m, 1, ldb, 20260910, -1.0, 1.0);
    std::fill(inputB.begin() + m, inputB.end(), padding);
    auto expected = inputB;
    Gtsv2Golden(tri.dl, tri.d, tri.du, expected, m, 1, ldb);
    Gtsv2DeviceData device;
    device.dDl = DeviceBuffer::copyFrom(tri.dl.data(), m * sizeof(float));
    device.dD = DeviceBuffer::copyFrom(tri.d.data(), m * sizeof(float));
    device.dDu = DeviceBuffer::copyFrom(tri.du.data(), m * sizeof(float));
    const size_t bBytes = ldb * sizeof(float);
    auto dInput = DeviceBuffer::copyFrom(inputB.data(), bBytes);
    device.dB = DeviceBuffer::alloc(bBytes);
    auto dOutput = DeviceBuffer::alloc(bBytes);
    size_t workspaceBytes = 0;
    ASSERT_EQ(aclsparseSgtsv2_bufferSizeExt(
        handle_->get(), m, 1,
        reinterpret_cast<const float *>(device.dDl.raw()),
        reinterpret_cast<const float *>(device.dD.raw()),
        reinterpret_cast<const float *>(device.dDu.raw()), nullptr, ldb,
        &workspaceBytes), ACL_SPARSE_STATUS_SUCCESS);
    std::vector<uint8_t> guard(workspaceBytes + 3 * kAlignSize, 0xA5);
    auto backing = DeviceBuffer::copyFrom(guard.data(), guard.size());
    const uintptr_t base = reinterpret_cast<uintptr_t>(backing.get());
    const uintptr_t aligned =
        (base + kAlignSize + kAlignMask) & ~static_cast<uintptr_t>(kAlignMask);
    const size_t prefix = aligned - base;
    std::vector<float> first;
    for (int repetition = 0; repetition < 3; ++repetition) {
        ASSERT_NO_FATAL_FAILURE(RunAsyncSolve(*handle_, stream_, device,
            dInput, dOutput, m, 1, ldb, reinterpret_cast<void *>(aligned)));
        std::vector<float> actual(ldb);
        dOutput.copyToHost(actual.data(), bBytes);
        CheckSingleRhsValues(actual, expected, m, ldb, padding);
        if (repetition == 0) {
            first = actual;
        } else {
            EXPECT_EQ(actual, first);
        }
    }
    CheckDeviceDiagonals(device, tri.dl, tri.d, tri.du);
    ASSERT_NO_FATAL_FAILURE(CheckJacobiWorkspaceGuards(backing, guard.size(), prefix, workspaceBytes));
}
