/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <complex>
#include <vector>
#include <type_traits>
#include "test_common.h"
#include "spsv_npu_wrapper.h"

namespace sparse_test {
namespace {
using CacheComplex = std::complex<float>;

static CsrMatrix MakeScheduleReuseMatrix(int32_t m, bool dependent)
{
    CsrMatrix matrix;
    matrix.rows = matrix.cols = m;
    matrix.rowOffsets = {0};
    for (int32_t row = 0; row < m; ++row) {
        if (dependent && row == m - 1) {
            matrix.colIndices.insert(matrix.colIndices.end(), {0, 1});
            matrix.values.insert(matrix.values.end(), {0x1.8p71f, -1.0f});
        }
        matrix.colIndices.push_back(row);
        matrix.values.push_back(row == 0 ? 3.0f : 1.0f);
        matrix.rowOffsets.push_back(static_cast<int32_t>(matrix.values.size()));
    }
    matrix.nnz = static_cast<int64_t>(matrix.values.size());
    return matrix;
}

template <typename T>
static void VerifyScheduleReuse(aclsparseHandle_t handle, aclrtStream stream,
    SpSVDescrManager &descr, int32_t m, bool dependent, bool waitForAnalysis)
{
    auto matrix = MakeScheduleReuseMatrix(m, dependent);
    std::vector<CacheComplex> values;
    for (float value : matrix.values) { values.emplace_back(value, 0.0f); }
    std::vector<T> input(m, T(1.0f));
    if (dependent) { input.back() = T(0x1p70f); }
    SpSVNpuConfig cfg;
    const aclDataType dtype = std::is_same<T, CacheComplex>::value ? ACL_COMPLEX64 : ACL_FLOAT;
    cfg.valueType = dtype;
    if constexpr (std::is_same<T, CacheComplex>::value) { cfg.complexValues = &values; }
    auto setup = SetupMatrixDescriptor(matrix, cfg);
    auto dx = DeviceBuffer::copyFrom(input.data(), m * sizeof(T));
    auto dy = DeviceBuffer::alloc(m * sizeof(T));
    auto vx = DnVecManager::createConst(m, dx.get(), dtype);
    auto vy = DnVecManager::create(m, dy.get(), dtype);
    const T alpha(1.0f);
    auto workspace = RunAnalysisT(handle, cfg, alpha, setup.matA, vx, vy, descr, dtype);
    if (waitForAnalysis) { ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS); }
    for (int32_t repeat = 0; repeat < 3; ++repeat) {
        std::vector<T> output(m);
        RunSolveT(handle, stream, cfg, alpha, setup.matA, vx, vy, descr, dy, output, dtype);
        if constexpr (std::is_same<T, CacheComplex>::value) {
            EXPECT_FLOAT_EQ(output[0].real(), 1.0f / 3.0f);
            EXPECT_EQ(output[0].imag(), 0.0f);
        } else {
            EXPECT_FLOAT_EQ(output[0], 1.0f / 3.0f);
        }
        for (int32_t row = 1; row < m; ++row) { EXPECT_EQ(output[row], T(1.0f)); }
    }
}
template <typename T>
static void CheckScheduleReuse()
{
    AclEnvScope env;
    HandleManager handle;
    handle.setStream(env.stream());
    for (int32_t m : {3, 257}) {
        for (bool waitForAnalysis : {false, true}) {
            SpSVDescrManager descr;
            for (bool dependent : {false, true, false, true}) {
                SCOPED_TRACE(::testing::Message() << m << ":" << waitForAnalysis << ":" << dependent);
                VerifyScheduleReuse<T>(handle.get(), env.stream(), descr, m, dependent, waitForAnalysis);
            }
        }
    }
}
// The pending-analysis path must retain recovery, and a completed single
// level must never suppress recovery after dependent reanalysis.
} // namespace

TEST(SpsvScheduleCacheTest, ReviewComplexReanalysisInvalidatesSingleLevel)
{
    CheckScheduleReuse<CacheComplex>();
}

TEST(SpsvScheduleCacheTest, ReviewFp32ReanalysisInvalidatesSingleLevel)
{
    CheckScheduleReuse<float>();
}
} // namespace sparse_test
