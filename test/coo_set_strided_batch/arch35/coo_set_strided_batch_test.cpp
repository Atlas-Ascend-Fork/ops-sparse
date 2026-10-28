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
 * @file coo_set_strided_batch_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseCooSetStridedBatch /
 *        aclsparseSpMatGetStridedBatch (arch35).
 *
 * CooSetStridedBatch 对标 cuSPARSE cusparseCooSetStridedBatch：将 batchCount /
 * batchStride 写入 COO 描述符；SpMatGetStridedBatch 对标
 * cusparseSpMatGetStridedBatch：读回 batchCount。两者均为纯 Host 侧描述符操作，
 * 不启动 NPU kernel；测试侧建设备缓冲存入描述符后断言字段写入/读回一致。
 *
 *   - TEST_P (CooSetStridedBatchTest) : CSV 驱动的成功路径用例
 *     （stride=0 广播（含 nnz=0+stride=0 空矩阵广播）/ stride==nnz 紧凑 /
 *      stride>nnz 带间隔 / batchCount=1/65535 边界，FP16/FP32, idx0/idx1；
 *      经配套算子 aclsparseSpMatGetStridedBatch 读回验证）
 *   - TEST_F (CooSetStridedBatchExceptionTest): null descr / 非 COO 格式 /
 *     batchCount<=0 或超上限 / batchStride 非法（负值、(0,nnz)）/ 覆盖写
 *     （Get 接口专属用例见 test/spmat_get_strided_batch/）
 */

#include "test_common.h"

#include "../coo_set_strided_batch_param.h"
#include "coo_set_strided_batch_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// GTest parameterized fixture: CooSetStridedBatchTest
// ============================================================================
class CooSetStridedBatchTest : public testing::TestWithParam<CooSetStridedBatchParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    CooSetStridedBatchParam param_;

    void SetUp() override { param_ = GetParam(); }
};

// ============================================================================
// Test body: success-path parameterized test
// ============================================================================
TEST_P(CooSetStridedBatchTest, SetGetRoundTrip)
{
    const auto& p = param_;
    std::cout << "==== " << p.case_name << " ==== rows=" << p.rows << " cols=" << p.cols << " nnz=" << p.nnz
              << " batchCount=" << p.batch_count << " batchStride=" << p.batch_stride << " dtype=" << p.dtype
              << " idx_base=" << p.idx_base << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    // 1. NPU 执行：CreateCoo -> 默认 Get -> SetStridedBatch -> Get 闭环验证
    auto npu = CooSetStridedBatchNpu(env_->stream(), p.rows, p.cols, p.nnz, p.batch_count, p.batch_stride, p.idx_base,
                                     p.isFp16(), p.seed);

    ASSERT_EQ(npu.defaultGetRet, ACL_SPARSE_STATUS_SUCCESS) << "default Get failed";
    ASSERT_EQ(npu.setRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCooSetStridedBatch failed";
    ASSERT_EQ(npu.getRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseSpMatGetStridedBatch failed";

    // 2. 新建描述符未 batch 化：默认 batchCount == 1
    EXPECT_EQ(npu.defaultBatchCount, 1);

    // 3. Set 后读回 batchCount == 传入值
    EXPECT_EQ(npu.outBatchCount, p.batch_count);

    // 4. 白盒断言：描述符内部 batchStride / batchCount 与传入一致
    EXPECT_EQ(npu.innerBatchStride, p.batch_stride);
    EXPECT_EQ(npu.innerBatchCount, p.batch_count);

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(CooSetStridedBatchCases, CooSetStridedBatchTest,
                         testing::ValuesIn(GetCasesFromCsv<CooSetStridedBatchParam>("coo_set_strided_batch_test.csv")),
                         [](const testing::TestParamInfo<CooSetStridedBatchParam>& info) {
                             return info.param.case_name;
                         });

// ============================================================================
// Exception test fixture: CooSetStridedBatchExceptionTest
// ============================================================================
class CooSetStridedBatchExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    std::vector<int32_t> rowInd_{0, 1};
    std::vector<int32_t> colInd_{0, 1};
    std::vector<float> values_{1.0f, 2.0f};

    void SetUp() override {}

    SpMatManager CreateCoo2x2()
    {
        return SpMatManager::createCoo(2, 2, 2, rowInd_.data(), colInd_.data(), values_.data(), ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    }

    SpMatManager CreateCsr2x2()
    {
        std::vector<int32_t> rowOff{0, 1, 2};
        return SpMatManager::createCsr(2, 2, 2, rowOff.data(), colInd_.data(), values_.data(), ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    }
};

// E1: Set with nullptr descriptor -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CooSetStridedBatchExceptionTest, SetNullDescr)
{
    EXPECT_EQ(aclsparseCooSetStridedBatch(nullptr, 2, 2), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E2: Set on a non-COO (CSR) descriptor -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(CooSetStridedBatchExceptionTest, SetOnNonCooDescr)
{
    SpMatManager csrMat = CreateCsr2x2();
    EXPECT_EQ(aclsparseCooSetStridedBatch(csrMat.get(), 2, 2), ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}

// E3: batchCount == 0 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CooSetStridedBatchExceptionTest, SetZeroBatchCount)
{
    SpMatManager mat = CreateCoo2x2();
    EXPECT_EQ(aclsparseCooSetStridedBatch(mat.get(), 0, 2), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E4: batchCount < 0 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CooSetStridedBatchExceptionTest, SetNegativeBatchCount)
{
    SpMatManager mat = CreateCoo2x2();
    EXPECT_EQ(aclsparseCooSetStridedBatch(mat.get(), -1, 2), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E5: batchStride 为负值 -> ACL_SPARSE_STATUS_INVALID_VALUE（广播哨兵为 0，负值一律非法）
TEST_F(CooSetStridedBatchExceptionTest, SetNegativeBatchStride)
{
    SpMatManager mat = CreateCoo2x2();
    EXPECT_EQ(aclsparseCooSetStridedBatch(mat.get(), 2, -1), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E6: 0 < batchStride < nnz -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CooSetStridedBatchExceptionTest, SetOverlapBatchStride)
{
    SpMatManager mat = CreateCoo2x2();
    EXPECT_EQ(aclsparseCooSetStridedBatch(mat.get(), 2, 1), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E7: 覆盖写：后一次 Set 覆盖前一次的 batchCount / batchStride
TEST_F(CooSetStridedBatchExceptionTest, SetOverwritePreviousValue)
{
    SpMatManager mat = CreateCoo2x2();
    ASSERT_EQ(aclsparseCooSetStridedBatch(mat.get(), 3, 2), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseCooSetStridedBatch(mat.get(), 5, 0), ACL_SPARSE_STATUS_SUCCESS);
    int bc = 0;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 5);
    EXPECT_EQ(mat.get()->batchStride, 0); // 白盒：stride 被覆盖为 0（广播）
}

// E8: batchCount 超上限 65535 -> ACL_SPARSE_STATUS_INVALID_VALUE（全仓共享不变量）
TEST_F(CooSetStridedBatchExceptionTest, SetBatchCountAboveLimit)
{
    SpMatManager mat = CreateCoo2x2();
    EXPECT_EQ(aclsparseCooSetStridedBatch(mat.get(), 65536, 2), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E9: 地址范围溢出：batchCount*batchStride 乘加超 int64 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CooSetStridedBatchExceptionTest, SetExtentOverflow)
{
    SpMatManager mat = CreateCoo2x2();
    const int64_t hugeStride = static_cast<int64_t>(1) << 50; // (65535-1)*2^50 超出 int64
    EXPECT_EQ(aclsparseCooSetStridedBatch(mat.get(), 65535, hugeStride), ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E10: 缓存失效语义（白盒）：batch 元数据实际变化时 structureGeneration 递增，值未变化不递增
TEST_F(CooSetStridedBatchExceptionTest, SetInvalidatesPreprocessCache)
{
    SpMatManager mat = CreateCoo2x2();
    const uint64_t genBefore = mat.get()->structureGeneration;
    ASSERT_EQ(aclsparseCooSetStridedBatch(mat.get(), 2, 2), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(mat.get()->structureGeneration, genBefore + 1); // 实际变化 -> 失效预处理缓存
    ASSERT_EQ(aclsparseCooSetStridedBatch(mat.get(), 2, 2), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(mat.get()->structureGeneration, genBefore + 1); // 值未变化 -> 不重复失效
}
