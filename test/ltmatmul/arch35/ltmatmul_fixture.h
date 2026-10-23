/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_LTMATMUL_LTMATMUL_FIXTURE_H_
#define TEST_LTMATMUL_LTMATMUL_FIXTURE_H_

// =============================================================================
// ltmatmul 测试 fixture 共享层。
//
// MatmulEnvMixin / MatmulTestFixture / MatmulParamFixture /
// MatmulExceptionTest 原定义于 ltmatmul_test.cpp 匿名 namespace；
// WB_L2 负向系拆分至 ltmatmul_test_neg.cpp 后由两个 TU 共用（具名
// namespace 保证跨 TU 同一类型，无 ODR 歧义；两个 TU 均
// using namespace sparse_test）。
// =============================================================================

#include <memory>

#include <gtest/gtest.h>

#include "sparse_test.h"  // AclEnvScope（test/frame 共享层）

namespace sparse_test {

// ============================================================================
// Test fixture
// ============================================================================
class MatmulEnvMixin {
protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    aclrtStream stream_ = nullptr;
    static void InitEnv() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void FiniEnv() { env_.reset(); }
    void SetUpStream() { stream_ = env_->stream(); }
};

class MatmulTestFixture : public testing::Test, public MatmulEnvMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

template <typename P>
class MatmulParamFixture : public testing::TestWithParam<P>, public MatmulEnvMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

// L2 exception / WB_L2 负向系共用 fixture（原 ltmatmul_test.cpp 本地定义，
// WB_L2 拆分至 ltmatmul_test_neg.cpp 后入共享头保证跨 TU 单一定义）。
class MatmulExceptionTest : public MatmulTestFixture {
};

}  // namespace sparse_test

#endif  // TEST_LTMATMUL_LTMATMUL_FIXTURE_H_
