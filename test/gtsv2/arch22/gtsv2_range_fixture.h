/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GTSV2_ARCH22_RANGE_FIXTURE_H_
#define GTSV2_ARCH22_RANGE_FIXTURE_H_

inline void CheckScaledSolution(const std::vector<float> &actual, const std::vector<double> &expected,
    const std::vector<float> &lower, const std::vector<float> &diagonal, const std::vector<float> &upper,
    const std::vector<float> &input, int m, int n, int ldb)
{
    ASSERT_EQ(actual.size(), input.size());
    for (int col = 0; col < n; ++col) {
        const size_t offset = static_cast<size_t>(col) * ldb;
        for (int row = 0; row < m; ++row) {
            const double x = actual[offset + row];
            const double want = expected[offset + row];
            ASSERT_TRUE(std::isfinite(x));
            EXPECT_NEAR(x, want, 1e-4 * (1.0 + std::abs(want)));
            const double left = row == 0 ? 0.0 : double(lower[row]) * actual[offset + row - 1];
            const double center = double(diagonal[row]) * x;
            const double right = row + 1 == m ? 0.0 : double(upper[row]) * actual[offset + row + 1];
            const double b = input[offset + row];
            const double denominator = std::abs(b) + std::abs(left) + std::abs(center) + std::abs(right);
            // Each original row is checked at its own scale, so large rows
            // cannot hide a small row's error. This is additional to solution accuracy.
            EXPECT_LE(std::abs(left + center + right - b), 1e-5 * denominator);
        }
        for (int row = m; row < ldb; ++row) { EXPECT_EQ(actual[offset + row], input[offset + row]); }
    }
}

inline std::vector<float> MakeRangeRhs(int m, int n, int ldb)
{
    std::vector<float> rhs(static_cast<size_t>(ldb) * n, -12345.0f);
    for (int column = 0; column < n; ++column) {
        const size_t offset = static_cast<size_t>(column) * ldb;
        for (int row = 0; row < m; ++row) { rhs[offset + row] = 0.0f; }
        if (column == 2) {
            rhs[offset + m / 2] = std::numeric_limits<float>::infinity();
        } else if (column != 4) {
            const float sign = column == 3 ? -1.0f : 1.0f;
            rhs[offset] = sign * std::numeric_limits<float>::max();
            rhs[offset + 1] = -rhs[offset];
        }
    }
    return rhs;
}

inline void CheckRangeStorage(sparse_test::HandleManager &handle, aclrtStream stream, int m, int n)
{
    using namespace sparse_test;
    const int ldb = m + 3;
    std::vector<float> lower(m, 0.0f), diagonal(m, 4.0f), upper(m, 0.0f);
    lower[1] = 1.0f;
    auto input = MakeRangeRhs(m, n, ldb);
    auto golden = input;
    Gtsv2Golden(lower, diagonal, upper, golden, m, n, ldb);
    auto device = PrepareDeviceData(lower, diagonal, upper, input, m, n, ldb);
    size_t bytes = 0;
    ASSERT_EQ(aclsparseSgtsv2_bufferSizeExt(handle.get(), m, n,
        static_cast<const float *>(device.dDl.raw()), static_cast<const float *>(device.dD.raw()),
        static_cast<const float *>(device.dDu.raw()), nullptr, ldb, &bytes), ACL_SPARSE_STATUS_SUCCESS);
    std::vector<uint8_t> sentinel(bytes + 3 * 128, 0xA5);
    auto workspace = DeviceBuffer::copyFrom(sentinel.data(), sentinel.size());
    uintptr_t address = reinterpret_cast<uintptr_t>(workspace.get());
    uintptr_t aligned = ((address + 127) & ~static_cast<uintptr_t>(127)) + 128;
    const size_t begin = aligned - address;
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::vector<float> actual;
        ASSERT_NO_FATAL_FAILURE(RunGtsv2StorageSolve(handle, stream, device, m, n, ldb,
            reinterpret_cast<void *>(aligned), input, actual));
        ASSERT_NO_FATAL_FAILURE(CheckJacobiRegressionValues(actual, input, golden, m, ldb));
        ASSERT_NO_FATAL_FAILURE(CheckJacobiWorkspaceGuards(workspace, sentinel.size(), begin, bytes));
        if (n == 5) {
            for (int row = 0; row < m; ++row) { EXPECT_FALSE(std::signbit(actual[4 * ldb + row])); }
        }
    }
    CheckDeviceDiagonals(device, lower, diagonal, upper);
}

#endif  // GTSV2_ARCH22_RANGE_FIXTURE_H_
