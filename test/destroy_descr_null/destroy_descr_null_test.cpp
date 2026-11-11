/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file destroy_descr_null_test.cpp
 * @brief GTest for descriptor Destroy null semantics (#377, 纯 Host 接口).
 */

// ============================================================================
// 描述符 Destroy 系列 null 语义专测（#377）。
//
// 契约：DestroySpMat / DestroySpVec / DestroyDnVec / DestroyDnMat 对 nullptr
// 一律返回 SUCCESS（null 容忍，对齐 cuSPARSE cusparseDestroy* 与仓内
// SpGEMMDestroyDescr；先例锚点 test/destroy_spmat DS-E01 的注释与断言）。
// 错型句柄（跨描述符类型伪装）一律 INVALID_VALUE 且不 delete，原对象仍可经
// 正确类型 Destroy 释放——错误路径无泄漏。
// ============================================================================

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "cann_ops_sparse.h"

namespace {

constexpr int64_t kSize = 8;
constexpr int64_t kNnz = 4;
constexpr int64_t kRows = 4;
constexpr int64_t kCols = 4;
constexpr int64_t kLd = 4;

// RAII 守卫：Destroy 成功路径置空指针，异常路径兜底释放
template <typename DescrT, typename DestroyFn>
struct DescrGuard {
    DescrT descr = nullptr;
    DestroyFn destroyFn = nullptr;
    ~DescrGuard()
    {
        if (descr != nullptr && destroyFn != nullptr) {
            (void)destroyFn(descr);
        }
    }
};

}  // namespace

// DN-E01: 四类描述符 Destroy 的 null 容忍（#377 核心断言）
TEST(DestroyDescrNullTest, NullReturnsSuccess)
{
    EXPECT_EQ(aclsparseDestroySpMat(nullptr), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseDestroySpVec(nullptr), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseDestroyDnVec(nullptr), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseDestroyDnMat(nullptr), ACL_SPARSE_STATUS_SUCCESS);
}

// DN-E02: 正常创建 -> Destroy 成功（确保 null 分支改动不影响非 null 路径）
TEST(DestroyDescrNullTest, CreateThenDestroySucceeds)
{
    std::vector<float> values(kSize, 1.0F);
    std::vector<int32_t> indices(kNnz, 0);

    aclsparseDnVecDescr_t dnVec = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dnVec, kSize, values.data(), ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseDestroyDnVec(dnVec), ACL_SPARSE_STATUS_SUCCESS);

    aclsparseSpVecDescr_t spVec = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&spVec, kSize, kNnz, indices.data(), values.data(),
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseDestroySpVec(spVec), ACL_SPARSE_STATUS_SUCCESS);

    aclsparseDnMatDescr_t dnMat = nullptr;
    ASSERT_EQ(aclsparseCreateDnMat(&dnMat, kRows, kCols, kLd, values.data(),
                                   ACL_FLOAT, ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseDestroyDnMat(dnMat), ACL_SPARSE_STATUS_SUCCESS);

    aclsparseSpMatDescr_t spMat = nullptr;
    std::vector<int32_t> offsets(kRows + 1, 0);
    ASSERT_EQ(aclsparseCreateCsr(&spMat, kRows, kCols, kNnz, offsets.data(), indices.data(),
                                 values.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseDestroySpMat(spMat), ACL_SPARSE_STATUS_SUCCESS);
}

// DN-E03: SpMat 句柄伪装成 SpVec/DnVec/DnMat -> INVALID_VALUE 且不 delete；
// 原对象随后经 DestroySpMat 正常释放（错误路径无泄漏）
TEST(DestroyDescrNullTest, WrongTypeRejectedWithoutLeak)
{
    std::vector<float> values(kSize, 1.0F);
    std::vector<int32_t> offsets(kRows + 1, 0);
    std::vector<int32_t> indices(kNnz, 0);

    aclsparseSpMatDescr_t spMat = nullptr;
    ASSERT_EQ(aclsparseCreateCsr(&spMat, kRows, kCols, kNnz, offsets.data(), indices.data(),
                                 values.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);

    EXPECT_EQ(aclsparseDestroySpVec(
        reinterpret_cast<aclsparseConstSpVecDescr_t>(spMat)), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseDestroyDnVec(
        reinterpret_cast<aclsparseConstDnVecDescr_t>(spMat)), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseDestroyDnMat(
        reinterpret_cast<aclsparseConstDnMatDescr_t>(spMat)), ACL_SPARSE_STATUS_INVALID_VALUE);

    // 三次错误路径均不应 delete，正确类型仍可正常销毁
    EXPECT_EQ(aclsparseDestroySpMat(spMat), ACL_SPARSE_STATUS_SUCCESS);
}
