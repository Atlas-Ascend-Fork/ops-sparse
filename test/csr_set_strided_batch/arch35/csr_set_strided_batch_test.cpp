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
 * @file csr_set_strided_batch_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseCsrSetStridedBatch
 *        （arch35，getter aclsparseSpMatGetStridedBatch 由
 *        sparse/spmat_get_strided_batch 提供）。
 *
 * CsrSetStridedBatch 对标 cuSPARSE cusparseCsrSetStridedBatch：CreateCsr ->
 * SetStridedBatch -> GetStridedBatch 闭环。Setter 为纯 Host 侧描述符操作，
 * 不启动 NPU kernel；测试侧建设备缓冲存入描述符，Set 后通过公共 getter 回读
 * batchCount，并通过内部结构白盒核对 batch stride 字段（公共 getter 与 cuSPARSE
 * 一致，只回读 batchCount；columnsValuesBatchStride 写入 columnsBatchStride 与
 * valuesBatchStride 两个共享拆分字段同值）。
 *
 *   - TEST_P (CsrSetStridedBatchTest)        : CSV 驱动的成功路径用例
 *                                             (FP16/FP32, idx0/idx1, batchCount
 *                                             1~65535, 步长 0 广播/常规/int64
 *                                             大值/nnz=0 空矩阵)
 *   - TEST_F (CsrSetStridedBatchExceptionTest): null / 签名（含 DnMat 错型句柄）/
 *                                             非 CSR 格式 / 非法 batchCount(0/负/
 *                                             65536) / 负 stride / 重叠 stride /
 *                                             溢出组合（受检乘加 + rows=INT64_MAX
 *                                             跨度构造）/ 覆盖写 / 缓存失效白盒 /
 *                                             const 变体
 */

#include "test_common.h"

#include <limits>

#include "../csr_set_strided_batch_param.h"
#include "csr_set_strided_batch_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// GTest parameterized fixture: CsrSetStridedBatchTest
// ============================================================================
class CsrSetStridedBatchTest : public testing::TestWithParam<CsrSetStridedBatchParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    CsrSetStridedBatchParam param_;

    void SetUp() override { param_ = GetParam(); }
};

// ============================================================================
// Test body: success-path parameterized test
// ============================================================================
TEST_P(CsrSetStridedBatchTest, SetGetRoundTrip) {
    const auto &p = param_;
    std::cout << "==== " << p.case_name << " ==== rows=" << p.rows << " cols=" << p.cols
              << " nnz=" << p.nnz << " dtype=" << p.dtype << " idx_base=" << p.idx_base
              << " batchCount=" << p.batch_count
              << " offsetsStride=" << p.offsets_batch_stride
              << " columnsValuesStride=" << p.columns_values_batch_stride << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    // 1. NPU 侧执行：CreateCsr -> Get(默认) -> SetStridedBatch -> GetStridedBatch
    auto npu = CsrSetStridedBatchNpu(env_->stream(), p.rows, p.cols, p.nnz, p.idx_base,
        p.isFp16(), p.batch_count, p.offsets_batch_stride, p.columns_values_batch_stride);

    ASSERT_EQ(npu.createRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCreateCsr failed";
    ASSERT_EQ(npu.getBeforeRet, ACL_SPARSE_STATUS_SUCCESS) << "GetStridedBatch before set failed";

    // 2. 新建描述符未设置批量语义时，默认 batchCount = 1
    EXPECT_EQ(npu.defaultBatchCount, 1);

    ASSERT_EQ(npu.setRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCsrSetStridedBatch failed";
    ASSERT_EQ(npu.getRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseSpMatGetStridedBatch failed";

    // 3. 公共 getter 回读 batchCount 与 Set 传入一致
    EXPECT_EQ(npu.outBatchCount, p.batch_count);

    // 4. 白盒核对 batch stride 与 Set 传入逐字一致（防 int64 截断）；
    //    columnsValuesBatchStride 同时写入 columnsBatchStride / valuesBatchStride 同值
    EXPECT_EQ(npu.whiteBoxOffsetsBatchStride, p.offsets_batch_stride);
    EXPECT_EQ(npu.whiteBoxColumnsBatchStride, p.columns_values_batch_stride);
    EXPECT_EQ(npu.whiteBoxValuesBatchStride, p.columns_values_batch_stride);

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    CsrSetStridedBatchCases,
    CsrSetStridedBatchTest,
    testing::ValuesIn(GetCasesFromCsv<CsrSetStridedBatchParam>("csr_set_strided_batch_test.csv")),
    [](const testing::TestParamInfo<CsrSetStridedBatchParam> &info) {
        return info.param.case_name;
    });

// ============================================================================
// Exception test fixture: CsrSetStridedBatchExceptionTest
// ============================================================================
class CsrSetStridedBatchExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    std::vector<int32_t> rowOff_{0, 1, 2};
    std::vector<int32_t> colInd_{0, 1};
    std::vector<float> values_{1.0f, 2.0f};

    aclsparseSpMatDescr_t CreateCsr2x2()
    {
        aclsparseSpMatDescr_t descr = nullptr;
        aclsparseStatus_t st = aclsparseCreateCsr(&descr, 2, 2, 2, rowOff_.data(),
            colInd_.data(), values_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        EXPECT_EQ(st, ACL_SPARSE_STATUS_SUCCESS);
        return descr;
    }

    void SetUp() override {}
};

// E1: nullptr descriptor handle -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrSetStridedBatchExceptionTest, NullSpMatDescrHandle) {
    EXPECT_EQ(aclsparseCsrSetStridedBatch(nullptr, 2, 3, 4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E2: wrong-type handle (DnMat punned as SpMat) -> signature check rejects with
//     ACL_SPARSE_STATUS_INVALID_VALUE（错型句柄防御，与 getter / 仓内访问器约定一致）
TEST_F(CsrSetStridedBatchExceptionTest, WrongTypeDnMatHandle) {
    aclsparseDnMatDescr_t dnMat = nullptr;
    ASSERT_EQ(aclsparseCreateDnMat(&dnMat, 2, 2, 2, values_.data(), ACL_FLOAT,
                                   ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    auto *punned = reinterpret_cast<aclsparseSpMatDescr_t>(dnMat);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(punned, 2, 3, 4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(
                  reinterpret_cast<aclsparseConstSpMatDescr_t>(dnMat), &bc),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyDnMat(dnMat);
}

// E3: non-CSR descriptor -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(CsrSetStridedBatchExceptionTest, SetOnNonCsrDescr) {
    aclsparseSpMatDescr_t cooDescr = nullptr;
    ASSERT_EQ(aclsparseCreateCoo(&cooDescr, 2, 2, 2, colInd_.data(), colInd_.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(cooDescr, 2, 3, 4),
              ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    aclsparseDestroySpMat(cooDescr);
}

// E4: batchCount == 0 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrSetStridedBatchExceptionTest, ZeroBatchCount) {
    aclsparseSpMatDescr_t descr = CreateCsr2x2();
    ASSERT_NE(descr, nullptr);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, 0, 3, 4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    // 失败路径不得改写描述符状态
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(descr, &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 1);
    aclsparseDestroySpMat(descr);
}

// E5: batchCount < 0 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrSetStridedBatchExceptionTest, NegativeBatchCount) {
    aclsparseSpMatDescr_t descr = CreateCsr2x2();
    ASSERT_NE(descr, nullptr);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, -2, 3, 4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroySpMat(descr);
}

// E6: batchCount > 65535 -> ACL_SPARSE_STATUS_INVALID_VALUE
//     （batchCount 为全格式共享字段，[1, 65535] 为全仓不变量，与 BSR/COO/DnMat setter 一致）
TEST_F(CsrSetStridedBatchExceptionTest, BatchCountOverLimit) {
    aclsparseSpMatDescr_t descr = CreateCsr2x2();
    ASSERT_NE(descr, nullptr);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, 65536, 3, 4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(descr, &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 1);
    aclsparseDestroySpMat(descr);
}

// E7: negative strides -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrSetStridedBatchExceptionTest, NegativeBatchStride) {
    aclsparseSpMatDescr_t descr = CreateCsr2x2();
    ASSERT_NE(descr, nullptr);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, 2, -1, 4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, 2, 3, -4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroySpMat(descr);
}

// E8: overlapping strides (0 < stride < span) -> ACL_SPARSE_STATUS_INVALID_VALUE
//     （stride 非 0 时必须 >= 对应数组长度：offsets >= rows+1=3，columnsValues >= nnz=2；
//      广播语义仅由 stride=0 承载）
TEST_F(CsrSetStridedBatchExceptionTest, OverlappingBatchStride) {
    aclsparseSpMatDescr_t descr = CreateCsr2x2();
    ASSERT_NE(descr, nullptr);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, 2, 1, 4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, 2, 3, 1),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroySpMat(descr);
}

// E9: strided extent overflow -> ACL_SPARSE_STATUS_INVALID_VALUE
//     batchCount 取合法值（不命中上限分支），确保断言真正锁定溢出防护分支
//     （CheckedStridedExtent）：offsets 侧 (65535-1)*2^50+3 溢出 int64；
//     columnsValues 侧 (2-1)*INT64_MAX+2 受检加法溢出
TEST_F(CsrSetStridedBatchExceptionTest, StridedExtentOverflow) {
    aclsparseSpMatDescr_t descr = CreateCsr2x2();
    ASSERT_NE(descr, nullptr);
    const int64_t overflowStride = static_cast<int64_t>(1) << 50;
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, 65535, overflowStride, 4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, 2, 3, INT64_MAX),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroySpMat(descr);
}

// E10: getter nullptr descriptor / nullptr batchCount out -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrSetStridedBatchExceptionTest, GetNullArgs) {
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(nullptr, &bc),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseSpMatDescr_t descr = CreateCsr2x2();
    ASSERT_NE(descr, nullptr);
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(descr, nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroySpMat(descr);
}

// E11: second Set overwrites the first one (latest wins)
TEST_F(CsrSetStridedBatchExceptionTest, SetOverwriteLatestWins) {
    aclsparseSpMatDescr_t descr = CreateCsr2x2();
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(aclsparseCsrSetStridedBatch(descr, 3, 33, 44), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseCsrSetStridedBatch(descr, 8, 88, 99), ACL_SPARSE_STATUS_SUCCESS);
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(descr, &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 8);
    // 白盒：stride 应为第二次 Set 的值（columnsValues 同值写入两个拆分字段）
    EXPECT_EQ(descr->offsetsBatchStride, 88);
    EXPECT_EQ(descr->columnsBatchStride, 99);
    EXPECT_EQ(descr->valuesBatchStride, 99);
    aclsparseDestroySpMat(descr);
}

// E12: batchCount == 1 with zero strides is a valid "disable batching" call
TEST_F(CsrSetStridedBatchExceptionTest, SingleBatchZeroStrideValid) {
    aclsparseSpMatDescr_t descr = CreateCsr2x2();
    ASSERT_NE(descr, nullptr);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, 1, 0, 0), ACL_SPARSE_STATUS_SUCCESS);
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(descr, &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 1);
    aclsparseDestroySpMat(descr);
}

// E13: batchCount > 1 with zero strides is a valid broadcast call
//      （cuSPARSE 官方样例：所有 batch 共享同一组 CSR 数组）
TEST_F(CsrSetStridedBatchExceptionTest, BroadcastZeroStrideValid) {
    aclsparseSpMatDescr_t descr = CreateCsr2x2();
    ASSERT_NE(descr, nullptr);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, 8, 0, 0), ACL_SPARSE_STATUS_SUCCESS);
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(descr, &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 8);
    aclsparseDestroySpMat(descr);
}

// E14: batchCount == 65535 boundary is valid（与 cuSPARSE SpMM/SDDMM 上限一致）
TEST_F(CsrSetStridedBatchExceptionTest, MaxBatchCountBoundary) {
    aclsparseSpMatDescr_t descr = CreateCsr2x2();
    ASSERT_NE(descr, nullptr);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(descr, 65535, 3, 2), ACL_SPARSE_STATUS_SUCCESS);
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(descr, &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 65535);
    aclsparseDestroySpMat(descr);
}

// E15: const descriptor variant works with the generic getter
TEST_F(CsrSetStridedBatchExceptionTest, ConstDescrGetStridedBatch) {
    aclsparseConstSpMatDescr_t descr = nullptr;
    ASSERT_EQ(aclsparseCreateConstCsr(&descr, 2, 2, 2, rowOff_.data(), colInd_.data(),
                                      values_.data(), ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ONE,
                                      ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(descr, &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 1);
    aclsparseDestroySpMat(descr);
}

// E16: getter accepts non-CSR descriptors (generic accessor, same as cuSPARSE)
TEST_F(CsrSetStridedBatchExceptionTest, GetOnNonCsrDescr) {
    aclsparseSpMatDescr_t cooDescr = nullptr;
    ASSERT_EQ(aclsparseCreateCoo(&cooDescr, 2, 2, 2, colInd_.data(), colInd_.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(cooDescr, &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 1);
    aclsparseDestroySpMat(cooDescr);
}

// E17: 缓存失效语义（白盒）：batch 元数据实际变化时 structureGeneration 递增，
//      值未变化时不重复失效（对齐同族 coo E10 / upstream BSR 终版条件化失效惯例）
TEST_F(CsrSetStridedBatchExceptionTest, SetInvalidatesPreprocessCache) {
    SpMatManager mat = SpMatManager::createCsr(2, 2, 2, rowOff_.data(), colInd_.data(),
                                              values_.data());
    const uint64_t genBefore = mat.get()->structureGeneration;
    ASSERT_EQ(aclsparseCsrSetStridedBatch(mat.get(), 2, 3, 2), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(mat.get()->structureGeneration, genBefore + 1);  // 实际变化 -> 失效预处理缓存
    ASSERT_EQ(aclsparseCsrSetStridedBatch(mat.get(), 2, 3, 2), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(mat.get()->structureGeneration, genBefore + 1);  // 值未变化 -> 不重复失效
}

// E18: Host 边界：rows = INT64_MAX 时 offsetsSpan = rows + 1 有符号溢出，须受检加法
//      拒绝（延迟绑定下无需分配巨大数组即可创建 rows=INT64_MAX 描述符；两个步长为 0
//      也不进入 ValidateBatchStride 的受检乘加，溢出须在跨度构造处拦截），且失败
//      不改写描述符既有状态（检视意见：nino888）
TEST_F(CsrSetStridedBatchExceptionTest, RowsInt64MaxOverflowsOffsetsSpan) {
    // rows=INT64_MAX, cols=1, nnz=0，空指针延迟绑定 + I64 索引即可成功创建
    SpMatManager mat = SpMatManager::createCsr(std::numeric_limits<int64_t>::max(), 1, 0,
                                              nullptr, nullptr, nullptr, ACL_SPARSE_INDEX_64I,
                                              ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_BASE_ZERO,
                                              ACL_FLOAT);
    ASSERT_NE(mat.get(), nullptr);
    EXPECT_EQ(aclsparseCsrSetStridedBatch(mat.get(), 1, 0, 0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    // 失败不改写既有状态：batchCount 保持默认 1
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 1);
}
