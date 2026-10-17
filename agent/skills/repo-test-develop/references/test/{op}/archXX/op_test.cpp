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

// TEMPLATE: GTest 测试入口（芯片相关）
// 包含两部分：
//   1. TEST_F — null handle / 异常路径测试（不下 CSV）
//   2. TEST_P — CSV 驱动的功能测试（5 步流程：生成数据 → _npu 执行 → 错误码比对 → _cpu golden → Verifier 比对）
// 禁止定义 main 函数（由 test/frame/test_main.cpp 提供）

#include <gtest/gtest.h>
#include "sparse_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"
#include "descriptor_manager.h"
#include "../{{op}}_param.h"
#include "../{{op}}_golden.h"
#include "{{op}}_npu_wrapper.h"

#include "acl/acl.h"
#include "cann_ops_sparse.h"

using namespace sparse_test;

// ===========================================================================
// Part 1: GTest 测试夹具
// ===========================================================================
class {{Op}}Test : public ::testing::TestWithParam<{{Op}}Param> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
        spHandle_ = std::make_unique<HandleManager>();
        spHandle_->setStream(env_->stream());
    }

    static void TearDownTestSuite() {
        spHandle_.reset();
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    inline static std::unique_ptr<HandleManager> spHandle_;
};

// ===========================================================================
// Part 2: TEST_P — CSV 驱动的功能测试
// ===========================================================================
TEST_P({{Op}}Test, GenericSuccess) {
    auto p = GetParam();
    PrintCaseInfoString(p);

    // 期望成功的用例
    ASSERT_EQ(p.expect_result, "ACL_SPARSE_STATUS_SUCCESS");

    // 1. 生成 CSR 数据（使用框架 fill.h）
    auto csr = makeSparseCsr(p.m, p.n, p.sparsity, p.seed);
    const auto xLen = p.transpose ? p.m : p.n;
    const auto yLen = p.transpose ? p.n : p.m;
    auto xVec = makeDenseFloat(xLen, -5.0, 10.0, p.seed + 1);
    auto yInit = makeDenseFloat(yLen, -5.0, 10.0, p.seed + 2);

    // 2. Eigen golden 作为唯一比对基准
    auto yGolden = {{Op}}Golden(csr, xVec, yInit, p.alpha, p.beta, p.transpose);

    // 3. NPU 调用（使用 npu_wrapper.h 封装）
    auto yNpu = {{Op}}NpuWrapper(*spHandle_, env_->stream(),
                                 csr, xVec, yInit,
                                 p.alpha, p.beta, p.transpose,
                                 p.computeType);

    // 4. 精度比对（使用框架 verify.h，阈值从 CSV 读取）
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MERE_MARE)
       .SetMERE(p.mereThreshold)
       .SetMARE(p.mareMultiplier * p.mereThreshold);
    EXPECT_TRUE(Verifier::verifyVector(yNpu, yGolden, cfg, p.caseId()));
}

// 3. 参数化实例化（从 CSV 加载用例）
INSTANTIATE_TEST_SUITE_P(
    {{Op}},
    {{Op}}Test,
    ::testing::ValuesIn(GetCasesFromCsv<{{Op}}Param>("{{op}}_test.csv")),
    [](const ::testing::TestParamInfo<{{Op}}Param>& info) {
        return info.param.caseId();
    }
);

// ===========================================================================
// Part 3: TEST_F — Null Handle / 异常路径测试（不下 CSV）
// ===========================================================================
class {{Op}}ExceptionTest : public ::testing::Test {
protected:
    void SetUp() override {
        env_ = std::make_unique<AclEnvScope>();
        spHandle_ = std::make_unique<HandleManager>();
        spHandle_->setStream(env_->stream());
    }
    void TearDown() override {
        spHandle_.reset();
        env_.reset();
    }
    std::unique_ptr<AclEnvScope> env_;
    std::unique_ptr<HandleManager> spHandle_;
};

// TEMPLATE: 按算子的异常路径添加 TEST_F 用例
TEST_F({{Op}}ExceptionTest, NullHandle) {
    // 传入 nullptr handle，期望返回 ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
    float alpha = 1.0f, beta = 0.0f;
    auto ret = aclsparse{{Op}}(nullptr, ACL_SPARSE_OP_NON_TRANSPOSE,
                               &alpha, nullptr, nullptr,
                               &beta, nullptr, ACL_FLOAT,
                               ACL_SPARSE_{{OP}}_ALG_DEFAULT, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F({{Op}}ExceptionTest, InvalidSpMatDescr) {
    // 传入 nullptr matA，期望返回 ACL_SPARSE_STATUS_INVALID_VALUE
    float alpha = 1.0f, beta = 0.0f;
    auto ret = aclsparse{{Op}}(spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE,
                               &alpha, nullptr, nullptr,
                               &beta, nullptr, ACL_FLOAT,
                               ACL_SPARSE_{{OP}}_ALG_DEFAULT, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// TEMPLATE: 按算子需求添加更多异常路径测试（NullDnVec、InvalidFormat 等）
