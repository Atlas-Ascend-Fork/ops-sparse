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

/**
 * @file matmulSearch_test_xcases.cpp
 * @brief TEST_F X20~X23（dtype 超集组合的描述符层接受性）——自
 *        本文件独立组织（同 target 多源链接）。
 *
 *        X20~X23：四主型 A/B × 超集 C/D 组合 → MatmulDescriptorInit
 *        SUCCESS。matmulDescriptor 保持库上原样：描述符层不做 A/B × C/D
 *        dtype 合法性矩阵校验（超集组合放行，与 FP8/FP4 枚举接受的原始
 *        行为一致）；此类组合在执行层另行处理，不在本组用例范围。
 */

#include "test_common.h"

#include "matmulSearch_golden_verify.h"  // 共享层（含 test_helpers / wrapper / golden 链）

#include <gtest/gtest.h>

using namespace sparse_test;

// ============================================================================
// H 组（X20-X23）：dtype 超集组合 → MatmulDescriptorInit SUCCESS
// （描述符层接受性回归锚点：锁定 matmulDescriptor 对超集组合的放行行为）
// ============================================================================

// ---- X20: FP16 A/B × C/D=INT32 → SUCCESS（描述符层接受）----
TEST_F(MatmulSearchExceptionTest, X20Fp16ABWithInt32CDAcceptedAtInit)
{
    sparse_test::SparseLtHandleGuard handle;
    aclsparseLtMatDescriptor_t dA = nullptr, dB = nullptr, dC = nullptr, dD = nullptr;
    ASSERT_EQ(aclsparseLtStructuredDescriptorInit(handle.get(), &dA, 128, 128, 128, 16,
        ACL_FLOAT16, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT),
        ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dB, 128, 128, 128, 16,
        ACL_FLOAT16, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dC, 128, 128, 128, 16,
        ACL_INT32, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dD, 128, 128, 128, 16,
        ACL_INT32, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulDescriptor_t mm = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(
        handle.get(), &mm, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &dA, &dB, &dC, &dD, ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    if (mm) { aclsparseLtMatmulDescriptorDestroy(&mm); }
}

// ---- X21: FP32 A/B × C/D=FP16 → SUCCESS（描述符层接受）----
TEST_F(MatmulSearchExceptionTest, X21Fp32ABWithFp16CDAcceptedAtInit)
{
    sparse_test::SparseLtHandleGuard handle;
    aclsparseLtMatDescriptor_t dA = nullptr, dB = nullptr, dC = nullptr, dD = nullptr;
    ASSERT_EQ(aclsparseLtStructuredDescriptorInit(handle.get(), &dA, 128, 128, 128, 16,
        ACL_FLOAT, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT),
        ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dB, 128, 128, 128, 16,
        ACL_FLOAT, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dC, 128, 128, 128, 16,
        ACL_FLOAT16, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dD, 128, 128, 128, 16,
        ACL_FLOAT16, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulDescriptor_t mm = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(
        handle.get(), &mm, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &dA, &dB, &dC, &dD, ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    if (mm) { aclsparseLtMatmulDescriptorDestroy(&mm); }
}

// ---- X22: INT8 A/B × C/D=FP32 → SUCCESS（描述符层接受）----
TEST_F(MatmulSearchExceptionTest, X22Int8ABWithFp32CDAcceptedAtInit)
{
    sparse_test::SparseLtHandleGuard handle;
    aclsparseLtMatDescriptor_t dA = nullptr, dB = nullptr, dC = nullptr, dD = nullptr;
    ASSERT_EQ(aclsparseLtStructuredDescriptorInit(handle.get(), &dA, 128, 128, 128, 16,
        ACL_INT8, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT),
        ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dB, 128, 128, 128, 16,
        ACL_INT8, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dC, 128, 128, 128, 16,
        ACL_FLOAT, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dD, 128, 128, 128, 16,
        ACL_FLOAT, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulDescriptor_t mm = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(
        handle.get(), &mm, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        &dA, &dB, &dC, &dD, ACL_SPARSE_COMPUTE_32I);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    if (mm) { aclsparseLtMatmulDescriptorDestroy(&mm); }
}

// ---- X23: BF16 A/B × C/D=INT8 → SUCCESS（描述符层接受）----
TEST_F(MatmulSearchExceptionTest, X23Bf16ABWithInt8CDAcceptedAtInit)
{
    sparse_test::SparseLtHandleGuard handle;
    aclsparseLtMatDescriptor_t dA = nullptr, dB = nullptr, dC = nullptr, dD = nullptr;
    ASSERT_EQ(aclsparseLtStructuredDescriptorInit(handle.get(), &dA, 128, 128, 128, 16,
        ACL_BF16, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT),
        ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dB, 128, 128, 128, 16,
        ACL_BF16, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dC, 128, 128, 128, 16,
        ACL_INT8, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtDenseDescriptorInit(handle.get(), &dD, 128, 128, 128, 16,
        ACL_INT8, ACL_SPARSE_ORDER_ROW), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulDescriptor_t mm = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(
        handle.get(), &mm, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &dA, &dB, &dC, &dD, ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    if (mm) { aclsparseLtMatmulDescriptorDestroy(&mm); }
}
