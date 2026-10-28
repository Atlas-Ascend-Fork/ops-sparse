/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file spmat_get_strided_batch_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseSpMatGetStridedBatch (arch35).
 *
 * SpMatGetStridedBatch 对标 cuSPARSE cusparseSpMatGetStridedBatch：从任意格式
 * 稀疏矩阵描述符读回 batch 数。纯 Host 侧描述符操作，不启动 NPU kernel。
 *
 *   - TEST_P (SpMatGetStridedBatchTest) : CSV 驱动用例
 *     （coo/csr/csc 三格式默认值=1；coo 经配套算子 Set 后读回写入值；
 *      FP16/FP32, idx0/idx1, 方阵/宽/高）
 *   - TEST_F (SpMatGetStridedBatchExceptionTest): null descr / null 出参 /
 *     SELL / BLOCKED_ELL 格式默认值 / const 描述符
 */

#include "test_common.h"

#include "../spmat_get_strided_batch_param.h"
#include "spmat_get_strided_batch_npu_wrapper.h"
#include "aclsparse_descr_internal.h" // 白盒：签名损坏用例需访问描述符内部字段

using namespace sparse_test;

// ============================================================================
// GTest parameterized fixture: SpMatGetStridedBatchTest
// ============================================================================
class SpMatGetStridedBatchTest : public testing::TestWithParam<SpMatGetStridedBatchParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    SpMatGetStridedBatchParam param_;

    void SetUp() override { param_ = GetParam(); }
};

// ============================================================================
// Test body: success-path parameterized test
// ============================================================================
TEST_P(SpMatGetStridedBatchTest, GetBatchCount)
{
    const auto& p = param_;
    std::cout << "==== " << p.case_name << " ==== format=" << p.format << " rows=" << p.rows << " cols=" << p.cols
              << " nnz=" << p.nnz << " pre_set=" << p.pre_set << " batchCount=" << p.batch_count
              << " batchStride=" << p.batch_stride << " dtype=" << p.dtype << " idx_base=" << p.idx_base << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    // 1. NPU 执行：Create(format) -> (可选 Set) -> Get
    auto npu = SpMatGetStridedBatchNpu(env_->stream(), p.format, p.rows, p.cols, p.nnz, p.pre_set, p.batch_count,
                                       p.batch_stride, p.idx_base, p.isFp16(), p.seed);

    if (p.pre_set == 1) {
        ASSERT_EQ(npu.setRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCooSetStridedBatch failed";
    }
    ASSERT_EQ(npu.getRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseSpMatGetStridedBatch failed";

    // 2. 期望值：pre_set=1 为 Set 传入值，否则为默认值 1（未 batch 化）
    int expected = (p.pre_set == 1) ? p.batch_count : 1;
    EXPECT_EQ(npu.outBatchCount, expected);

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    SpMatGetStridedBatchCases, SpMatGetStridedBatchTest,
    testing::ValuesIn(GetCasesFromCsv<SpMatGetStridedBatchParam>("spmat_get_strided_batch_test.csv")),
    [](const testing::TestParamInfo<SpMatGetStridedBatchParam>& info) { return info.param.case_name; });

// ============================================================================
// Exception / format-coverage test fixture
// ============================================================================
class SpMatGetStridedBatchExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    std::vector<int32_t> rowInd_{0, 1};
    std::vector<int32_t> colInd_{0, 1};
    std::vector<float> values_{1.0f, 2.0f};

    void SetUp() override {}
};

// E1: Get with nullptr descriptor -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpMatGetStridedBatchExceptionTest, GetNullDescr)
{
    int bc = 0;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(nullptr, &bc), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E2: Get with nullptr out pointer -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpMatGetStridedBatchExceptionTest, GetNullOutPointer)
{
    SpMatManager mat = SpMatManager::createCoo(2, 2, 2, rowInd_.data(), colInd_.data(), values_.data(),
                                               ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E3: SELL 格式默认值 = 1（格式无关性）
TEST_F(SpMatGetStridedBatchExceptionTest, GetOnSlicedEllDefaultOne)
{
    SpMatManager mat = SpMatManager::createSlicedEll(4, 4, 4, rowInd_.data(), colInd_.data(), values_.data(), 2, 2,
                                                     ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    int bc = 0;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 1);
}

// E4: BLOCKED_ELL 格式默认值 = 1（格式无关性）
TEST_F(SpMatGetStridedBatchExceptionTest, GetOnBlockedEllDefaultOne)
{
    // 框架暂无 BlockedEll 的 RAII 工厂，用 unique_ptr + Destroy 删除器保证 RAII 语义
    std::vector<int32_t> colInd(8, 0);
    std::vector<float> ellValue(8, 1.0f);
    aclsparseSpMatDescr_t raw = nullptr;
    ASSERT_EQ(aclsparseCreateBlockedEll(&raw, 4, 4, 2, 2, colInd.data(), ellValue.data(), ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    std::unique_ptr<aclsparseSpMatDescr, aclsparseStatus_t (*)(aclsparseConstSpMatDescr_t)> mat(raw,
                                                                                                aclsparseDestroySpMat);
    int bc = 0;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 1);
}

// E5: const 描述符变体（aclsparseConstSpMatDescr_t 原生支持）
TEST_F(SpMatGetStridedBatchExceptionTest, GetOnConstCooDescr)
{
    SpMatManager mat = SpMatManager::createConstCoo(2, 2, 2, rowInd_.data(), colInd_.data(), values_.data(),
                                                    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    int bc = 0;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.cget(), &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 1);
}

// E6: 错型句柄（DnMat 描述符伪装成 SpMat）-> 签名校验拒绝，ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpMatGetStridedBatchExceptionTest, GetOnWrongTypeHandle)
{
    DnMatManager dnMat = DnMatManager::create(2, 2, 2, values_.data(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto* wrongType = reinterpret_cast<aclsparseSpMatDescr_t>(dnMat.get());
    int bc = 0;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(wrongType, &bc), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E7: 签名损坏的句柄 -> ACL_SPARSE_STATUS_INVALID_VALUE（白盒构造，校验后恢复以保证 RAII 释放）
TEST_F(SpMatGetStridedBatchExceptionTest, GetOnCorruptedSignature)
{
    SpMatManager mat = SpMatManager::createCoo(2, 2, 2, rowInd_.data(), colInd_.data(), values_.data(),
                                               ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    const uint32_t savedSignature = mat.get()->signature;
    mat.get()->signature = 0xDEADBEEFU;
    int bc = 0;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &bc), ACL_SPARSE_STATUS_INVALID_VALUE);
    mat.get()->signature = savedSignature; // 恢复签名，使 RAII 析构可正常释放
}

// E8: BSR Set 路径写入的共享 batchCount 经本接口统一读回（跨格式共享字段语义）
TEST_F(SpMatGetStridedBatchExceptionTest, GetAfterBsrSetReturnsSameBatchCount)
{
    // 构造 2x2 块网格、块大小 2x2、2 个非零块的 BSR 描述符（块值数 = 2*2*2 = 8）
    std::vector<int32_t> rowOffsets{0, 1, 2}; // blockRows + 1
    std::vector<int32_t> colInd{0, 1};        // blockNnz
    std::vector<float> bsrValues(8, 1.0f);    // blockNnz * rowBlockSize * colBlockSize
    aclsparseSpMatDescr_t raw = nullptr;
    ASSERT_EQ(aclsparseCreateBsr(&raw, 2, 2, 2, 2, 2, rowOffsets.data(), colInd.data(), bsrValues.data(),
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    std::unique_ptr<aclsparseSpMatDescr, aclsparseStatus_t (*)(aclsparseConstSpMatDescr_t)> mat(raw,
                                                                                                aclsparseDestroySpMat);
    // BSR Set：offsets/columns 广播（stride=0），values 按块覆盖（stride=8）
    ASSERT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 3, 0, 0, 8), ACL_SPARSE_STATUS_SUCCESS);
    int bc = 0;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 3);
}
