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
 * @file create_sliced_ell_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseCreateSlicedEll /
 *        aclsparseCreateConstSlicedEll（arch35，纯 Host 接口）.
 *
 * 对标 cuSPARSE cusparseCreateSlicedEll：Create(参数) -> 通用 Get 访问器回读，
 * 期望元数据 = Create 调用方传入的参数本身（roundtrip oracle），判定口径为
 * 枚举 / int64 的位精确相等（EXACT，零容差）。CreateSlicedEll 为纯 Host 侧描述
 * 符创建接口，不启动 NPU kernel；仓内无 SlicedEllGet 访问器，sliceNnz /
 * numSlices 经内部结构白盒核对（先例 csr_set_strided_batch 的 stride 字段）。
 *
 *   - TEST_P (CreateSlicedEllTest)          : create_sliced_ell_test.csv 驱动的
 *                                             12 个成功用例（基础/tall/wide/large/
 *                                             0 维/nnz=0+null 指针/单 slice/
 *                                             sliceNnz>nnz/I64/fp16/const 变体），
 *                                             GetFormat/GetSize/GetIndexBase/
 *                                             GetStridedBatch + 白盒字段断言
 *   - TEST_F (CreateSlicedEllSemanticsTest) : nnz>0 且三指针全 null 的延迟绑定
 *                                             语义（SE-S01）
 *   - TEST_F (CreateSlicedEllExceptionTest) : OUT 句柄 null（mutable+const）/
 *                                             rows/cols/nnz/sliceNnz/numSlices 负值/
 *                                             非法 sellIdxType（SE-E01~E07），
 *                                             失败后断言 OUT 保持 nullptr 不被污染
 */

#include "test_common.h"
#include "spmat_csv_common.h"      // GetCasesFromCsvChecked：带构成校验的 CSV 装载（描述符测试私有共享）

#include "create_sliced_ell_param.h"
#include "create_sliced_ell_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// GTest parameterized fixture: CreateSlicedEllTest（CSV 驱动）
// ============================================================================
class CreateSlicedEllTest : public testing::TestWithParam<CreateSlicedEllParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// ============================================================================
// Test body: success-path parameterized test（每条 CSV 行一个用例）
// ============================================================================
TEST_P(CreateSlicedEllTest, CreateSlicedEllRoundTrip)
{
    const auto &p = GetParam();
    std::cout << "==== " << p.case_name << " ==== entry=" << p.entry << " rows=" << p.rows
              << " cols=" << p.cols << " nnz=" << p.nnz << " slice_nnz=" << p.slice_nnz
              << " num_slices=" << p.num_slices << " idx_type=" << p.idx_type
              << " idx_base=" << p.idx_base << " dtype=" << p.dtype
              << " null_ptrs=" << p.null_ptrs << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");
    ASSERT_TRUE(p.entry == "sliced_ell" || p.entry == "const_sliced_ell")
        << "unknown entry in CSV: " << p.entry;

    // 1. NPU 侧执行：Create -> GetFormat/GetSize/GetIndexBase/GetStridedBatch + 白盒
    auto r = CreateSlicedEllNpu(env_->stream(), p.entry, p.rows, p.cols, p.nnz, p.slice_nnz,
                                p.num_slices, p.idx_type, p.idx_base, p.isFp16(),
                                p.null_ptrs != 0);

    // 2. Create 成功 + format == SLICED_ELL
    ASSERT_EQ(r.createRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCreateSlicedEll failed";
    ASSERT_EQ(r.getFormatRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.format, ACL_SPARSE_FORMAT_SLICED_ELL) << "format mismatch";

    // 3. GetSize 回读 rows/cols/nnz 与传入一致（EXACT 零容差）
    ASSERT_EQ(r.getSizeRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.rows, p.rows) << "rows roundtrip mismatch";
    EXPECT_EQ(r.cols, p.cols) << "cols roundtrip mismatch";
    EXPECT_EQ(r.nnz, p.nnz) << "nnz roundtrip mismatch";

    // 4. GetIndexBase 回读
    ASSERT_EQ(r.getIndexBaseRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.idxBase, (p.idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE
                                           : ACL_SPARSE_INDEX_BASE_ZERO)
        << "index base roundtrip mismatch";

    // 5. 新建描述符默认 batchCount == 1
    ASSERT_EQ(r.getBatchRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.batchCount, 1) << "default batchCount should be 1";

    // 6. 白盒核对：sliceNnz / numSlices / ptrType / IdxType / valueType
    //（SELL 无公共 Get 访问器，防 int64 截断只能经内部结构核对）
    EXPECT_EQ(r.whiteBoxSliceNnz, p.slice_nnz) << "sliceNnz white-box mismatch";
    EXPECT_EQ(r.whiteBoxNumSlices, p.num_slices) << "numSlices white-box mismatch";
    const aclsparseIndexType_t expectIdxType =
        (p.idx_type == 1) ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    EXPECT_EQ(r.whiteBoxPtrType, expectIdxType) << "ptrType white-box mismatch";
    EXPECT_EQ(r.whiteBoxIdxType, expectIdxType) << "IdxType white-box mismatch";
    EXPECT_EQ(r.whiteBoxValueType, p.isFp16() ? ACL_FLOAT16 : ACL_FLOAT)
        << "valueType white-box mismatch";

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    CreateSlicedEllCases,
    CreateSlicedEllTest,
    testing::ValuesIn(GetCasesFromCsvChecked<CreateSlicedEllParam>(
        "create_sliced_ell_test.csv", 12)),
    [](const testing::TestParamInfo<CreateSlicedEllParam> &info) { return info.param.case_name; });

// ============================================================================
// Semantics test fixture: CreateSlicedEllSemanticsTest
// ============================================================================
class CreateSlicedEllSemanticsTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    void SetUp() override {}
};

// SE-S01: nnz=4 且三个数组指针全 null 创建 -> SUCCESS（实现无指针校验，延迟绑定
// 语义）；GetFormat/GetSize 正常回读（unique_ptr + Destroy 删除器 RAII，形态对齐
// create_bsr_test.cpp 的 BSR-S01）
TEST_F(CreateSlicedEllSemanticsTest, NullPointersDelayedBinding)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateSlicedEll(&descr, 4, 4, 4, 2, 2, nullptr, nullptr, nullptr,
                                       ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                       ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_NE(descr, nullptr);
    std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)> mat(
        descr, &aclsparseDestroySpMat);

    aclsparseFormat_t format = static_cast<aclsparseFormat_t>(-1);
    EXPECT_EQ(aclsparseSpMatGetFormat(mat.get(), &format), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(format, ACL_SPARSE_FORMAT_SLICED_ELL);
    int64_t rows = -1;
    int64_t cols = -1;
    int64_t nnz = -1;
    EXPECT_EQ(aclsparseSpMatGetSize(mat.get(), &rows, &cols, &nnz), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(rows, 4);
    EXPECT_EQ(cols, 4);
    EXPECT_EQ(nnz, 4);
}

// ============================================================================
// Exception test fixture: CreateSlicedEllExceptionTest
// 失败路径直调 API（不经 wrapper）；每条锁定一个校验分支，断言错误码返回且
// OUT 保持 nullptr 不被污染。
// ============================================================================
class CreateSlicedEllExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    // 基线合法参数：4×4、nnz=4、sliceNnz=2、numSlices=2、I32、base0、float；
    // 指针实现不校验，host 占位即可（Create 全程不解引用）
    std::vector<int32_t> hostIdx_{0, 1, 2, 3};
    std::vector<float> hostVal_{1.0F, 2.0F, 3.0F, 4.0F};

    void SetUp() override {}
};

// SE-E01: OUT 句柄 null（mutable + const）-> INVALID_VALUE
TEST_F(CreateSlicedEllExceptionTest, NullOutHandle)
{
    EXPECT_EQ(aclsparseCreateSlicedEll(nullptr, 4, 4, 4, 2, 2, hostIdx_.data(),
                                       hostIdx_.data(), hostVal_.data(), ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCreateConstSlicedEll(nullptr, 4, 4, 4, 2, 2, hostIdx_.data(),
                                            hostIdx_.data(), hostVal_.data(),
                                            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                            ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// SE-E02: rows=-1（其余合法）-> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateSlicedEllExceptionTest, NegativeRows)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateSlicedEll(&descr, -1, 4, 4, 2, 2, hostIdx_.data(),
                                       hostIdx_.data(), hostVal_.data(), ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// SE-E03: cols=-1 -> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateSlicedEllExceptionTest, NegativeCols)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateSlicedEll(&descr, 4, -1, 4, 2, 2, hostIdx_.data(),
                                       hostIdx_.data(), hostVal_.data(), ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// SE-E04: nnz=-1 -> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateSlicedEllExceptionTest, NegativeNnz)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateSlicedEll(&descr, 4, 4, -1, 2, 2, hostIdx_.data(),
                                       hostIdx_.data(), hostVal_.data(), ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// SE-E05: sellIdxType=99 -> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateSlicedEllExceptionTest, InvalidIdxType)
{
    aclsparseSpMatDescr_t descr = nullptr;
    const auto badIdxType = static_cast<aclsparseIndexType_t>(99);
    EXPECT_EQ(aclsparseCreateSlicedEll(&descr, 4, 4, 4, 2, 2, hostIdx_.data(),
                                       hostIdx_.data(), hostVal_.data(), badIdxType,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// SE-E06: sliceNnz=-1 -> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateSlicedEllExceptionTest, NegativeSliceNnz)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateSlicedEll(&descr, 4, 4, 4, -1, 2, hostIdx_.data(),
                                       hostIdx_.data(), hostVal_.data(), ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// SE-E07: numSlices=-1 -> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateSlicedEllExceptionTest, NegativeNumSlices)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateSlicedEll(&descr, 4, 4, 4, 2, -1, hostIdx_.data(),
                                       hostIdx_.data(), hostVal_.data(), ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}
