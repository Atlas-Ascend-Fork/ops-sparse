/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSV2_ARCH22_JACOBI_FIXTURE_H_
#define GTSV2_ARCH22_JACOBI_FIXTURE_H_

#include <cstring>
#include <limits>

inline void CheckDeviceDiagonals(Gtsv2DeviceData &device,
    const std::vector<float> &lower, const std::vector<float> &diagonal,
    const std::vector<float> &upper)
{
    std::vector<float> actual(lower.size());
    device.dDl.copyToHost(actual.data(), actual.size() * sizeof(float));
    EXPECT_EQ(actual, lower);
    device.dD.copyToHost(actual.data(), actual.size() * sizeof(float));
    EXPECT_EQ(actual, diagonal);
    device.dDu.copyToHost(actual.data(), actual.size() * sizeof(float));
    EXPECT_EQ(actual, upper);
}

inline std::vector<float> MakeJacobiRegressionRhs(
    int m, int n, int ldb, float rhsScale, bool nonfiniteRhs)
{
    std::vector<float> input(static_cast<size_t>(ldb) * n, -12345.0f);
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < m; ++row) {
            input[column * ldb + row] = rhsScale * static_cast<float>((row * 37 + column * 13) % 129 - 64) / 64.0f;
        }
        input[column * ldb] = rhsScale;
        if (nonfiniteRhs && column % 3 == 0) {
            input[column * ldb + m / 2] = column % 2 ?
                std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
        }
    }
    return input;
}

inline void CheckJacobiRegressionValues(
    const std::vector<float> &actual, const std::vector<float> &input,
    const std::vector<float> &golden, int m, int ldb)
{
    if (ldb <= 0) {
        ADD_FAILURE() << "Jacobi regression requires a positive leading dimension";
        return;
    }
    int row = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (row >= m) {
            ASSERT_EQ(actual[i], input[i]);
        } else if (std::isnan(golden[i])) {
            ASSERT_TRUE(std::isnan(actual[i]));
        } else if (std::isinf(golden[i])) {
            ASSERT_EQ(actual[i], golden[i]);
        } else {
            ASSERT_TRUE(std::isfinite(actual[i]));
            const double magnitude = std::abs(static_cast<double>(golden[i]));
            const double ulp = std::nextafter(static_cast<float>(magnitude), INFINITY) - magnitude;
            const double error = std::abs(static_cast<double>(actual[i]) - golden[i]);
            ASSERT_LE(error, 0x1p-16 + 0x1p-10 * magnitude);
            ASSERT_LE(error, std::max(0.01, 32.0 * ulp));
        }
        if (++row == ldb) {
            row = 0;
        }
    }
}

inline void CheckJacobiWorkspaceGuards(
    sparse_test::DeviceBuffer &workspace, size_t totalBytes, size_t begin, size_t workspaceBytes)
{
    std::vector<uint8_t> after(totalBytes);
    workspace.copyToHost(after.data(), after.size());
    for (size_t i = 0; i < after.size(); ++i) {
        if (i < begin || i >= begin + workspaceBytes) {
            ASSERT_EQ(after[i], 0xA5);
        }
    }
}

inline void RunGtsv2StorageSolve(sparse_test::HandleManager &handle, aclrtStream stream,
    Gtsv2DeviceData &device, int m, int n, int ldb, void *workspace,
    const std::vector<float> &input, std::vector<float> &actual)
{
    ASSERT_EQ(aclrtMemcpy(device.dB.get(), input.size() * sizeof(float), input.data(),
        input.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclsparseSgtsv2(handle.get(), m, n,
        static_cast<const float *>(device.dDl.raw()), static_cast<const float *>(device.dD.raw()),
        static_cast<const float *>(device.dDu.raw()), static_cast<float *>(device.dB.get()),
        ldb, workspace), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    actual.resize(input.size());
    device.dB.copyToHost(actual.data(), actual.size() * sizeof(float));
}

// Includes FP32-rounded FP64 direct golden plus explicit storage guards.
// The external ATK and stress suites retain the full unrounded FP64 golden.
inline void CheckJacobiRegression(sparse_test::HandleManager &handle, aclrtStream stream,
    int m, int n, int padding, float rhsScale, bool signedDiagonal, bool nonfiniteRhs)
{
    using namespace sparse_test;
    SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n) +
                 " padding=" + std::to_string(padding) + " scale=" + std::to_string(rhsScale));
    const int ldb = m + padding;
    std::vector<float> lower(m, -0.99999988f), diagonal(m, 4.0f), upper(m, 0.99999988f);
    lower[0] = 0.0f; upper[m - 1] = 0.0f;
    if (signedDiagonal) {
        for (int row = 1; row < m; row += 2) {
            diagonal[row] = -4.0f;
        }
    }
    auto input = MakeJacobiRegressionRhs(m, n, ldb, rhsScale, nonfiniteRhs);
    auto golden = input;
    Gtsv2Golden(lower, diagonal, upper, golden, m, n, ldb);
    auto device = PrepareDeviceData(lower, diagonal, upper, input, m, n, ldb);
    size_t workspaceBytes = 0;
    ASSERT_EQ(aclsparseSgtsv2_bufferSizeExt(handle.get(), m, n,
        static_cast<const float *>(device.dDl.raw()), static_cast<const float *>(device.dD.raw()),
        static_cast<const float *>(device.dDu.raw()), static_cast<const float *>(device.dB.raw()),
        ldb, &workspaceBytes), ACL_SPARSE_STATUS_SUCCESS);
    std::vector<uint8_t> sentinel(workspaceBytes + 3 * 128, 0xA5);
    auto workspace = DeviceBuffer::copyFrom(sentinel.data(), sentinel.size());
    uintptr_t address = reinterpret_cast<uintptr_t>(workspace.get());
    uintptr_t aligned = ((address + 127) & ~static_cast<uintptr_t>(127)) + 128;
    const size_t begin = aligned - address;
    std::vector<float> previous;
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::vector<float> actual;
        ASSERT_NO_FATAL_FAILURE(RunGtsv2StorageSolve(handle, stream, device, m, n, ldb,
            reinterpret_cast<void *>(aligned), input, actual));
        if (!previous.empty()) {
            EXPECT_EQ(std::memcmp(previous.data(), actual.data(), actual.size() * sizeof(float)), 0);
        }
        previous = actual;
        ASSERT_NO_FATAL_FAILURE(CheckJacobiRegressionValues(actual, input, golden, m, ldb));
        ASSERT_NO_FATAL_FAILURE(CheckJacobiWorkspaceGuards(workspace, sentinel.size(), begin, workspaceBytes));
    }
    CheckDeviceDiagonals(device, lower, diagonal, upper);
}

#endif
