/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "sparse_test.h"
#include "csv_loader.h"
#include "verify.h"
#include "descriptor_manager.h"
#include "cube_spmm_1d_param.h"
#include "cube_spmm_1d_golden.h"
#include "cube_spmm_1d_npu_wrapper.h"

using namespace sparse_test;
class CubeSpmm1dTest : public ::testing::TestWithParam<CubeSpmm1dParam> {
protected:
    static void SetUpTestSuite()
    {
        env = std::make_unique<AclEnvScope>();
        handle = std::make_unique<HandleManager>();
        handle->setStream(env->stream());
    }
    static void TearDownTestSuite()
    {
        handle.reset();
        env.reset();
    }
    inline static std::unique_ptr<AclEnvScope> env;
    inline static std::unique_ptr<HandleManager> handle;
};

namespace {
void CubeCheckProduct(const CubeInput& a, const CubeSpmm1dParam& p, const aclsparseCubeSpmm1dPlanInfo& info,
                       const std::vector<uint16_t>& b, const std::vector<float>& c)
{
    std::vector<float> logical(size_t(p.m) * p.n);
    for (int col = 0; col < info.paddedN; ++col) {
        for (int row = 0; row < info.paddedM; ++row) {
            const float x = c[size_t(col) * info.paddedM + row];
            if (row < p.m && col < p.n) {
                logical[size_t(row) * p.n + col] = x;
            } else {
                ASSERT_EQ(x, 0) << "padding";
            }
        }
    }
    const auto reference = CubeGolden(a, p.m, p.n, p.k, b, info.paddedN);
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MIXED_TOLERANCE_ANCHOR);
    cfg.SetMixedTol(0.001, 0.001).SetMixedRequiredRatio(1.0).SetMixedMaxAbsErrLim(0.001);
    EXPECT_TRUE(Verifier::verifyVector(logical, reference, cfg, p.id));
}
} // namespace

namespace {
void CubeCheckStressPath(const CubeSpmm1dParam& p, const aclsparseCubeSpmm1dPlanInfo& info)
{
    if (p.mode == 6) {
        ASSERT_GT(info.vectors, 2048LL * (p.m / 16)) << "must exceed the complete-A L0B capacity";
    }
    if (p.mode == 7) {
        EXPECT_GE(uint64_t(p.k) * p.n * 2, 192ULL * 1024 * 1024);
        int64_t cores = 0;
        ASSERT_EQ(aclGetDeviceCapability(info.deviceId, ACL_DEVICE_INFO_AI_CORE_NUM, &cores), ACL_SUCCESS);
        if (std::string(aclrtGetSocName()) == "Ascend910B3" && cores == 20) {
            ASSERT_EQ(info.tilingKey, 57) << "must exercise adaptive N-major dispatch, not explicit key override";
        }
    }
}
} // namespace

TEST_P(CubeSpmm1dTest, ProductAndPadding)
{
    const auto p = GetParam();
    const auto a = CubeMakeInput(p);
    auto plan = CubePrepare(handle->get(), p, a);
    aclsparseCubeSpmm1dPlanInfo info{};
    ASSERT_EQ(aclsparseCubeSpmm1dGetPlanInfo(plan.get(), &info), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_GE(info.paddedM, p.m);
    ASSERT_GE(info.paddedN, p.n);
    CubeCheckStressPath(p, info);
    auto bdev = DeviceBuffer::alloc(info.bBytes);
    auto cdev = DeviceBuffer::alloc(info.cBytes + 128);
    auto workspace = DeviceBuffer::alloc(info.workspaceBytes + 128);
    std::vector<uint16_t> b(info.bBytes / 2, 0);
    std::vector<float> c(info.cBytes / 4 + 32, 123.0f);
    for (int iteration = 0; iteration < 2; ++iteration) {
        for (int row = 0; row < p.k; ++row) {
            for (int col = 0; col < p.n; ++col) {
                b[size_t(row) * info.paddedN + col] = CubeHalf(((row * 7 + col + iteration * 3) % 13 - 6) * 0.125);
            }
        }
        ASSERT_EQ(aclrtMemcpy(bdev.get(), info.bBytes, b.data(), info.bBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(cdev.get(), cdev.size(), c.data(), cdev.size(), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(workspace.get(), workspace.size(), 0x5a, workspace.size()), ACL_SUCCESS);
        ASSERT_EQ(aclsparseCubeSpmm1dExecute(handle->get(), plan.get(), bdev.get(), info.bBytes, cdev.get(),
                                             info.cBytes, workspace.get(), info.workspaceBytes),
                  ACL_SPARSE_STATUS_SUCCESS);
        // The public API is asynchronous: the test, not the operator, waits.
        ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
        cdev.copyToHost(c.data(), cdev.size());
        CubeCheckProduct(a, p, info, b, c);
        for (size_t i = info.cBytes / 4; i < c.size(); ++i) {
            ASSERT_EQ(c[i], 123.0f);
        }
        std::vector<uint8_t> guard(workspace.size());
        workspace.copyToHost(guard.data(), guard.size());
        for (size_t i = info.workspaceBytes; i < guard.size(); ++i) {
            ASSERT_EQ(guard[i], 0x5a);
        }
        std::fill(c.begin(), c.end(), 123.0f);
    }
}

INSTANTIATE_TEST_SUITE_P(Cases, CubeSpmm1dTest, ::testing::ValuesIn([] {
                             std::vector<CubeSpmm1dParam> cases;
                             for (const auto& row : ReadMap(CUBE_SPMM_1D_CSV)) {
                                 CubeSpmm1dParam p;
                                 p.fill(row);
                                 cases.push_back(p);
                             }
                             return cases;
                         }()),
                         [](const ::testing::TestParamInfo<CubeSpmm1dParam>& info) { return info.param.caseId(); });

class CubeSpmm1dContract : public ::testing::Test {
protected:
    void SetUp() override
    {
        env = std::make_unique<AclEnvScope>();
        handle = std::make_unique<HandleManager>();
        handle->setStream(env->stream());
    }
    std::unique_ptr<AclEnvScope> env;
    std::unique_ptr<HandleManager> handle;
};

TEST_F(CubeSpmm1dContract, InvalidPreparation)
{
    int64_t rows[] = {0, 1};
    int64_t cols[] = {0};
    uint16_t values[] = {0x3c00};
    aclsparseCubeSpmm1dPlan_t plan = nullptr;
    auto create = [&](int m, int n, int k, const aclsparseCubeSpmm1dOptions* o) {
        return aclsparseCubeSpmm1dCreateCsrPlan(handle->get(), m, n, k, rows, 2, cols, values, 1, o, &plan);
    };
    EXPECT_EQ(aclsparseCubeSpmm1dCreateCsrPlan(nullptr, 1, 1, 1, rows, 2, cols, values, 1, nullptr, &plan),
              ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(create(0, 1, 1, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(create(1, 0, 1, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(create(1, 1, 0, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    rows[0] = 1;
    EXPECT_EQ(create(1, 1, 1, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    rows[0] = 0;
    rows[1] = 2;
    EXPECT_EQ(create(1, 1, 1, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    rows[1] = 1;
    cols[0] = -1;
    EXPECT_EQ(create(1, 1, 1, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    cols[0] = 0;
    values[0] = 0x7c00;
    EXPECT_EQ(create(1, 1, 1, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    values[0] = 0x3c00;
    aclsparseCubeSpmm1dOptions options{sizeof(options), 63, 0, 0};
    EXPECT_EQ(create(1, 1, 1, &options), ACL_SPARSE_STATUS_INVALID_VALUE);
    options.tilingKey = -1;
    options.reserved = 1;
    EXPECT_EQ(create(1, 1, 1, &options), ACL_SPARSE_STATUS_INVALID_VALUE);
    options.reserved = 0;
    options.workspaceLimitBytes = 1;
    EXPECT_EQ(create(1, 1, 1, &options), ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
    EXPECT_EQ(plan, nullptr);
    EXPECT_EQ(aclsparseCubeSpmm1dGetPlanInfo(nullptr, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCubeSpmm1dDestroyPlan(nullptr), ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(CubeSpmm1dContract, BufferGuardsAndImmutableA)
{
    CubeSpmm1dParam p;
    p.m = 17;
    p.n = 33;
    p.k = 29;
    p.mode = 3;
    p.csr = true;
    auto a = CubeMakeInput(p);
    auto plan = CubePrepare(handle->get(), p, a);
    aclsparseCubeSpmm1dPlanInfo q{};
    ASSERT_EQ(aclsparseCubeSpmm1dGetPlanInfo(plan.get(), &q), ACL_SPARSE_STATUS_SUCCESS);
    std::vector<uint16_t> b(q.bBytes / 2, 0);
    for (int row = 0; row < p.k; ++row) {
        for (int col = 0; col < p.n; ++col) {
            b[size_t(row) * q.paddedN + col] = CubeHalf((1 + (row * 7 + col) % 11) * 0.125);
        }
    }
    const auto reference = CubeGolden(a, p.m, p.n, p.k, b, q.paddedN);
    ASSERT_TRUE(std::any_of(reference.begin(), reference.end(), [](double x) { return x != 0; }));
    auto bd = DeviceBuffer::copyFrom(b.data(), q.bBytes);
    auto cd = DeviceBuffer::alloc(q.cBytes);
    auto wd = DeviceBuffer::alloc(q.workspaceBytes);
    EXPECT_EQ(aclsparseCubeSpmm1dExecute(nullptr, plan.get(), bd.get(), q.bBytes, cd.get(), q.cBytes, wd.get(),
                                         q.workspaceBytes),
              ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseCubeSpmm1dExecute(handle->get(), plan.get(), bd.get(), q.bBytes - 1, cd.get(), q.cBytes,
                                         wd.get(), q.workspaceBytes),
              ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
    EXPECT_EQ(aclsparseCubeSpmm1dExecute(handle->get(), plan.get(), bd.get(), q.bBytes, cd.get(), q.cBytes, wd.get(),
                                         q.workspaceBytes - 1),
              ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
    EXPECT_EQ(aclsparseCubeSpmm1dExecute(handle->get(), plan.get(), cd.get(), q.bBytes, cd.get(), q.cBytes, wd.get(),
                                         q.workspaceBytes),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    std::fill(a.values.begin(), a.values.end(), 0); // Plan owns its prepared A, not these host arrays.
    const auto changed = CubeGolden(a, p.m, p.n, p.k, b, q.paddedN);
    ASSERT_NE(reference, changed) << "host A mutation must change the expected product";
    ASSERT_EQ(aclsparseCubeSpmm1dExecute(handle->get(), plan.get(), bd.get(), q.bBytes, cd.get(), q.cBytes, wd.get(),
                                         q.workspaceBytes),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    std::vector<float> c(q.cBytes / 4, 1);
    cd.copyToHost(c.data(), q.cBytes);
    for (int col = 0; col < q.paddedN; ++col) {
        for (int row = 0; row < q.paddedM; ++row) {
            const double expected = row < p.m && col < p.n ? reference[size_t(row) * p.n + col] : 0;
            ASSERT_NEAR(c[size_t(col) * q.paddedM + row], expected, 0.001);
        }
    }
}

TEST_F(CubeSpmm1dContract, TwoStreamsSameImmutablePlan)
{
    CubeSpmm1dParam p;
    p.m = 33;
    p.n = 127;
    p.k = 73;
    p.mode = 3;
    p.csr = true;
    const auto a = CubeMakeInput(p);
    auto plan = CubePrepare(handle->get(), p, a);
    aclsparseCubeSpmm1dPlanInfo q{};
    ASSERT_EQ(aclsparseCubeSpmm1dGetPlanInfo(plan.get(), &q), ACL_SPARSE_STATUS_SUCCESS);
    aclrtStream second = nullptr;
    ASSERT_EQ(aclrtCreateStream(&second), ACL_SUCCESS);
    struct StreamOwner {
        aclrtStream value;
        ~StreamOwner() { (void)aclrtDestroyStream(value); }
    } owner{second};
    std::vector<uint16_t> b(q.bBytes / 2, 0);
    for (int row = 0; row < p.k; ++row) {
        for (int col = 0; col < p.n; ++col) {
            b[size_t(row) * q.paddedN + col] = CubeHalf(0.5);
        }
    }
    auto bd = DeviceBuffer::copyFrom(b.data(), q.bBytes);
    auto c1 = DeviceBuffer::alloc(q.cBytes), c2 = DeviceBuffer::alloc(q.cBytes);
    auto w1 = DeviceBuffer::alloc(q.workspaceBytes), w2 = DeviceBuffer::alloc(q.workspaceBytes);
    ASSERT_EQ(aclsparseCubeSpmm1dExecute(handle->get(), plan.get(), bd.get(), q.bBytes, c1.get(), q.cBytes, w1.get(),
                                         q.workspaceBytes),
              ACL_SPARSE_STATUS_SUCCESS);
    handle->setStream(second);
    ASSERT_EQ(aclsparseCubeSpmm1dExecute(handle->get(), plan.get(), bd.get(), q.bBytes, c2.get(), q.cBytes, w2.get(),
                                         q.workspaceBytes),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(second), ACL_SUCCESS);
    handle->setStream(env->stream());
    std::vector<float> first(q.cBytes / 4), other(q.cBytes / 4);
    c1.copyToHost(first.data(), q.cBytes);
    c2.copyToHost(other.data(), q.cBytes);
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::EXACT);
    EXPECT_TRUE(Verifier::verifyVector(first, other, cfg, "two_streams"));
}

TEST_F(CubeSpmm1dContract, NativeValidationAndDuplicateOverflow)
{
    int32_t offsets[] = {0, 2}, columns[] = {0, 0};
    uint16_t values[32] = {};
    aclsparseCubeSpmm1dPlan_t plan = nullptr;
    EXPECT_EQ(aclsparseCubeSpmm1dCreatePlan(handle->get(), 16, 16, 2, offsets, 2, columns, values, 2, nullptr, &plan),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    columns[1] = 1;
    values[31] = 0x7e00;
    EXPECT_EQ(aclsparseCubeSpmm1dCreatePlan(handle->get(), 16, 16, 2, offsets, 2, columns, values, 2, nullptr, &plan),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    int64_t rows[] = {0, 2}, duplicateColumns[] = {0, 0};
    uint16_t overflow[] = {0x7bff, 0x7bff};
    EXPECT_EQ(aclsparseCubeSpmm1dCreateCsrPlan(handle->get(), 1, 16, 1, rows, 2, duplicateColumns, overflow, 2, nullptr,
                                               &plan),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(plan, nullptr);
}
