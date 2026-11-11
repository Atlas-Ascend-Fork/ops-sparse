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
 * @file create_bsr_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseCreateBsr / aclsparseCreateConstBsr
 *       （arch35，纯 Host 接口）.
 *
 * 对标 cuSPARSE cusparseCreateBsr：Create(参数) -> 通用 Get 访问器回读，期望
 * 元数据 = Create 调用方传入的参数本身（roundtrip oracle，GetSize 期望值按块
 * 展开公式计算：rows=blockRows×bs、cols=blockCols×bs、nnz=blockNnz），判定口径
 * 为枚举 / int64 的位精确相等（EXACT，零容差）。CreateBsr 为纯 Host 侧描述符
 * 创建接口，不启动 NPU kernel；仓内无 BsrGet 访问器，块几何字段经内部结构白盒
 * 核对（先例 csr_set_strided_batch 的 stride 字段）。
 *
 *   - TEST_P (CreateBsrTest)          : create_bsr_test.csv 驱动的 14 个成功用例
 *                                      （四 dtype 全覆盖 / ROW+COL / bs 2~128 /
 *                                       0 维 / nnzb=0 / tall / wide / 1×1 / rect /
 *                                       const 变体），GetFormat/GetSize/GetIndexBase/
 *                                       GetStridedBatch + 白盒块几何断言
 *   - TEST_F (CreateBsrSemanticsTest) : INT32 几何边界内极大值（BSR-S01）/
 *                                       batch roundtrip 非广播最小步长（BSR-S02）/
 *                                       覆盖写 latest-wins（BSR-S03）/ 单 batch
 *                                       关闭（BSR-S04）
 *   - TEST_F (CreateBsrExceptionTest) : OUT null / 负块维 / 非正与非方块大小 /
 *                                       不支持的块大小 / 64I 索引 / 非法 idxBase /
 *                                       不支持的 valueType / 非法 order / INT32
 *                                       几何越界 / 零维非零块 / null 存储指针 /
 *                                       地址范围回绕（BSR-E01~E13），失败后断言
 *                                       OUT 保持 nullptr 不被污染
 */

#include "test_common.h"
#include "sparse_dtype_utils.h"    // ParseValueType：CSV dtype 列名 -> aclDataType（BSR 族共享）
#include "spmat_csv_common.h"      // GetCasesFromCsvChecked：带构成校验的 CSV 装载（描述符测试私有共享）
#include "spmat_test_helpers.h"    // MakeValidBsr：语义用例基线 BSR 描述符（与 bsr_set_strided_batch 共享）

#include <limits>

#include "create_bsr_param.h"
#include "create_bsr_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// GTest parameterized fixture: CreateBsrTest（CSV 驱动）
// ============================================================================
class CreateBsrTest : public testing::TestWithParam<CreateBsrParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// ============================================================================
// Test body: success-path parameterized test（每条 CSV 行一个用例）
// ============================================================================
TEST_P(CreateBsrTest, CreateBsrRoundTrip)
{
    const auto &p = GetParam();
    std::cout << "==== " << p.case_name << " ==== entry=" << p.entry
              << " block_rows=" << p.block_rows << " block_cols=" << p.block_cols
              << " block_nnz=" << p.block_nnz << " block_size=" << p.block_size
              << " idx_base=" << p.idx_base << " dtype=" << p.dtype << " order=" << p.order
              << " null_ptrs=" << p.null_ptrs << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");
    ASSERT_TRUE(p.entry == "bsr" || p.entry == "const_bsr")
        << "unknown entry in CSV: " << p.entry;

    // 1. NPU 侧执行：Create -> GetFormat/GetSize/GetIndexBase/GetStridedBatch + 白盒
    auto r = CreateBsrNpu(env_->stream(), p.entry, p.block_rows, p.block_cols, p.block_nnz,
                          p.block_size, p.idx_base, ParseValueType(p.dtype),
                          ParseBsrOrder(p.order), p.null_ptrs != 0);

    // 2. Create 成功 + format == BSR
    ASSERT_EQ(r.createRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCreateBsr failed";
    ASSERT_EQ(r.getFormatRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.format, ACL_SPARSE_FORMAT_BSR) << "format mismatch";

    // 3. GetSize 块展开回读：rows=blockRows×bs、cols=blockCols×bs、nnz=blockNnz
    //（golden = 块展开公式，EXACT 零容差）
    ASSERT_EQ(r.getSizeRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.rows, p.block_rows * p.block_size) << "logical rows (blockRows*bs) mismatch";
    EXPECT_EQ(r.cols, p.block_cols * p.block_size) << "logical cols (blockCols*bs) mismatch";
    EXPECT_EQ(r.nnz, p.block_nnz) << "stored blocks (blockNnz) mismatch";

    // 4. GetIndexBase 回读
    ASSERT_EQ(r.getIndexBaseRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.idxBase, (p.idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE
                                           : ACL_SPARSE_INDEX_BASE_ZERO)
        << "index base roundtrip mismatch";

    // 5. 新建描述符默认 batchCount == 1
    ASSERT_EQ(r.getBatchRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.batchCount, 1) << "default batchCount should be 1";

    // 6. 白盒核对：块几何字段与传入一致（BSR 无公共 Get 访问器，防 int64 截断）
    EXPECT_EQ(r.whiteBoxBlockRows, p.block_rows) << "blockRows white-box mismatch";
    EXPECT_EQ(r.whiteBoxBlockCols, p.block_cols) << "blockCols white-box mismatch";
    EXPECT_EQ(r.whiteBoxBlockNnz, p.block_nnz) << "blockNnz white-box mismatch";
    EXPECT_EQ(r.whiteBoxRowBlockSize, p.block_size) << "rowBlockSize white-box mismatch";
    EXPECT_EQ(r.whiteBoxColBlockSize, p.block_size) << "colBlockSize white-box mismatch";
    EXPECT_EQ(r.whiteBoxBsrValuesCount, p.block_nnz * p.block_size * p.block_size)
        << "bsrValuesCount (nnzb*bs^2) white-box mismatch";
    EXPECT_EQ(r.whiteBoxBsrOrder, ParseBsrOrder(p.order)) << "bsrOrder white-box mismatch";

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    CreateBsrCases,
    CreateBsrTest,
    testing::ValuesIn(GetCasesFromCsvChecked<CreateBsrParam>(
        "create_bsr_test.csv", 14)),
    [](const testing::TestParamInfo<CreateBsrParam> &info) { return info.param.case_name; });

// ============================================================================
// Semantics test fixture: CreateBsrSemanticsTest
// 语义 / batch roundtrip 用例：INT32 几何边界 + BsrSetStridedBatch 载体验证。
// 基线描述符经共享 MakeValidBsr() 构造（与 bsr_set_strided_batch 语义用例
// 共用同一基线：2×2 块网格、bs=2、nnzb=2、host 占位数据，Create 不解引用）。
// ============================================================================
class CreateBsrSemanticsTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    void SetUp() override {}
};

// BSR-S01: INT32 几何边界内极大值 (blockRows=INT32_MAX-1, blockCols=INT32_MAX,
// nnzb=0, bs=2) 创建成功（仅元数据校验，指针不解引用）；GetSize rows 为块展开
// (INT32_MAX-1)×2
TEST_F(CreateBsrSemanticsTest, GeometryInt32BoundaryInRange)
{
    // 占位设备指针：Create 仅做地址算术（offsetsByteExtent=(INT32_MAX-1+1)×4≈2^33，
    // 与有效地址相加不回绕），全程不解引用，无需按 INT32_MAX 规模分配
    auto dOffsets = DeviceBuffer::alloc(sizeof(int32_t));
    aclsparseSpMatDescr_t descr = nullptr;
    const int64_t maxBlockRows = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) - 1;
    const int64_t maxBlockCols = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    EXPECT_EQ(aclsparseCreateBsr(&descr, maxBlockRows, maxBlockCols, 0, 2, 2, dOffsets.get(),
                                 nullptr, nullptr, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT, ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_NE(descr, nullptr);
    std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)> mat(
        descr, &aclsparseDestroySpMat);

    int64_t rows = -1;
    int64_t cols = -1;
    int64_t nnz = -1;
    EXPECT_EQ(aclsparseSpMatGetSize(mat.get(), &rows, &cols, &nnz), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(rows, maxBlockRows * 2) << "rows should be (INT32_MAX-1)*2";
    EXPECT_EQ(cols, maxBlockCols * 2) << "cols should be INT32_MAX*2";
    EXPECT_EQ(nnz, 0);
}

// BSR-S02: batch roundtrip 非广播最小步长：BsrSetStridedBatch(4, 3, 2, 8) ->
// SUCCESS；GetStridedBatch==4；白盒三 stride=={3,2,8}
//（offsetsSpan=blockRows+1=3、columnsSpan=blockNnz=2、valuesSpan=bsrValuesCount=8；
// 与 spmat_get_strided_batch E8 的广播 (0,0,8) 场景区分）
TEST_F(CreateBsrSemanticsTest, BatchRoundTripNonBroadcast)
{
    auto mat = MakeValidBsr();
    ASSERT_NE(mat.get(), nullptr);  // 白盒解引用前判空（Create 为非 fatal EXPECT 校验）

    // Set 之前默认 batchCount == 1
    int before = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &before), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(before, 1);

    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 4, 3, 2, 8), ACL_SPARSE_STATUS_SUCCESS);

    int after = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &after), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(after, 4) << "batchCount roundtrip mismatch";

    // 白盒核对三个 batch stride（防 int64 截断）
    EXPECT_EQ(mat.get()->offsetsBatchStride, 3);
    EXPECT_EQ(mat.get()->columnsBatchStride, 2);
    EXPECT_EQ(mat.get()->valuesBatchStride, 8);
}

// BSR-S03: batch roundtrip 覆盖写 latest-wins：Set(4,3,2,8) -> Set(2,3,2,8) ->
// GetStridedBatch==2
TEST_F(CreateBsrSemanticsTest, BatchRoundTripOverwriteLatestWins)
{
    auto mat = MakeValidBsr();
    ASSERT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 4, 3, 2, 8), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, 8), ACL_SPARSE_STATUS_SUCCESS);

    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 2) << "latest Set should win";
    EXPECT_EQ(mat.get()->offsetsBatchStride, 3);
    EXPECT_EQ(mat.get()->columnsBatchStride, 2);
    EXPECT_EQ(mat.get()->valuesBatchStride, 8);
}

// BSR-S04: 单 batch 关闭批量语义：Set(1, 0, 0, 0) -> SUCCESS；GetStridedBatch==1
TEST_F(CreateBsrSemanticsTest, BatchDisableSingleBatch)
{
    auto mat = MakeValidBsr();
    ASSERT_NE(mat.get(), nullptr);  // 白盒解引用前判空（Create 为非 fatal EXPECT 校验）
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 1, 0, 0, 0), ACL_SPARSE_STATUS_SUCCESS);

    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 1) << "single batch should disable batching";
    EXPECT_EQ(mat.get()->offsetsBatchStride, 0);
    EXPECT_EQ(mat.get()->columnsBatchStride, 0);
    EXPECT_EQ(mat.get()->valuesBatchStride, 0);
}

// ============================================================================
// Exception test fixture: CreateBsrExceptionTest
// 失败路径直调 API（不经 wrapper）；每条锁定一个校验分支，断言错误码返回且
// OUT 保持 nullptr 不被污染。
// ============================================================================
class CreateBsrExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    // 基线合法参数：2×2 块、bs=2、nnzb=2、I32/I32、base0、fp32、ROW
    std::vector<int32_t> hostOffsets_{0, 1, 2};
    std::vector<int32_t> hostColInd_{0, 1};
    std::vector<float> hostValues_{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};

    void SetUp() override {}
};

// BSR-E01: OUT 句柄 null（mutable + const）-> INVALID_VALUE
TEST_F(CreateBsrExceptionTest, NullOutHandle)
{
    EXPECT_EQ(aclsparseCreateBsr(nullptr, 2, 2, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCreateConstBsr(nullptr, 2, 2, 2, 2, 2, hostOffsets_.data(),
                                      hostColInd_.data(), hostValues_.data(),
                                      ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                      ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// BSR-E02: blockRows / blockCols / blockNnz = -1 -> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateBsrExceptionTest, NegativeBlockDims)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateBsr(&descr, -1, 2, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, -1, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, -1, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// BSR-E03: rowBlockSize=0 / colBlockSize=-2 -> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateBsrExceptionTest, NonPositiveBlockSize)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 0, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, -2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// BSR-E04: rowBlockSize=2, colBlockSize=4（非方）-> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateBsrExceptionTest, NonSquareBlockSize)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 4, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// BSR-E05: blockSize ∈ {1, 3, 256}（不支持的块大小）-> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateBsrExceptionTest, UnsupportedBlockSize)
{
    for (const int64_t badBs : {1, 3, 256}) {
        aclsparseSpMatDescr_t descr = nullptr;
        EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, badBs, badBs, hostOffsets_.data(),
                                     hostColInd_.data(), hostValues_.data(),
                                     ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                     ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                     ACL_SPARSE_ORDER_ROW),
                  ACL_SPARSE_STATUS_INVALID_VALUE)
            << "blockSize=" << badBs;
        EXPECT_EQ(descr, nullptr) << "blockSize=" << badBs << " failed Create touched OUT";
    }
}

// BSR-E06: offsets/colInd 索引类型 = 64I -> NOT_SUPPORTED，OUT 保持 nullptr
TEST_F(CreateBsrExceptionTest, UnsupportedIdxType)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_64I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// BSR-E07: idxBase=cast(2) -> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateBsrExceptionTest, InvalidIdxBase)
{
    aclsparseSpMatDescr_t descr = nullptr;
    const auto badBase = static_cast<aclsparseIndexBase_t>(2);
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, badBase, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// BSR-E08: valueType = ACL_DOUBLE / ACL_INT32 -> NOT_SUPPORTED，OUT 保持 nullptr
TEST_F(CreateBsrExceptionTest, UnsupportedValueType)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                 ACL_DOUBLE, ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_INT32,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// BSR-E09: order=cast(2) -> INVALID_VALUE，OUT 保持 nullptr
TEST_F(CreateBsrExceptionTest, InvalidOrder)
{
    aclsparseSpMatDescr_t descr = nullptr;
    const auto badOrder = static_cast<aclsparseOrder_t>(2);
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 badOrder),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// BSR-E10: 几何越界（INT32 上界）-> INVALID_VALUE，OUT 保持 nullptr。
// 存储指针传基线合法指针（几何与 null 校验同返 INVALID_VALUE，指针有效时
// 失败可唯一归因于几何越界分支）：
//   blockRows=INT32_MAX；blockCols=INT32_MAX+1；
//   blockNnz=INT32_MAX+1（base0 越界）；blockNnz=INT32_MAX（base1 越界）
TEST_F(CreateBsrExceptionTest, GeometryOutOfRange)
{
    constexpr int64_t kInt32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    aclsparseSpMatDescr_t descr = nullptr;

    // blockRows = INT32_MAX（上界为 INT32_MAX-1）
    EXPECT_EQ(aclsparseCreateBsr(&descr, kInt32Max, 2, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    // blockCols = INT32_MAX+1（上界为 INT32_MAX）
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, kInt32Max + 1, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    // blockNnz = INT32_MAX+1（base0 上界为 INT32_MAX）
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, kInt32Max + 1, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    // blockNnz = INT32_MAX（base1 上界为 INT32_MAX-1）
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, kInt32Max, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ONE, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// BSR-E11: 零维几何（blockRows=0 或 blockCols=0）且 blockNnz=1 -> INVALID_VALUE，
// OUT 保持 nullptr
TEST_F(CreateBsrExceptionTest, ZeroDimWithNonZeroBlocks)
{
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateBsr(&descr, 0, 2, 1, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 0, 1, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// BSR-E12: null 存储指针 -> INVALID_VALUE，OUT 保持 nullptr：
//   offsets=null（含 nnzb=0，offsets 恒非空）；nnzb=1 且 colInd=null；nnzb=1 且 values=null
TEST_F(CreateBsrExceptionTest, NullStoragePointers)
{
    aclsparseSpMatDescr_t descr = nullptr;

    // offsets=null（nnzb=0 时也必须非空）
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 0, 2, 2, nullptr, nullptr, nullptr,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT, ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    // nnzb=1 且 colInd=null
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 1, 2, 2, hostOffsets_.data(), nullptr,
                                 hostValues_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT, ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    // nnzb=1 且 values=null
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 1, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), nullptr, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}

// BSR-E13: 地址范围回绕（合法 metadata：2×2 块、bs=2、nnzb=2、fp32）：
//   ① bsrRowOffsets=高位指针（offsetsByteExtent=12>0）；
//   ② 有效 offsets + bsrColInd=高位指针（columnsByteExtent=8>0）；
//   ③ 有效 offsets/colInd + bsrValues=高位指针（valuesByteExtent=32>0）
// 均 INVALID_VALUE（lastByteOffset ≥ 3 > UINTPTR_MAX - address = 0，
// CheckedAddressExtent 确定性拒绝；Create 仅地址算术不解引用，无 UB），
// OUT 保持 nullptr。先例 csr/coo_set_strided_batch E9 的同类受检溢出注入
//（大步长 2^50 命中 CheckedStridedExtent；本用例改以 UINTPTR_MAX 高位指针
// 命中 CheckedAddressExtent，思想同为极大值注入触发受检算术防御）。
TEST_F(CreateBsrExceptionTest, AddressExtentOverflow)
{
    void *const highAddr = reinterpret_cast<void *>(UINTPTR_MAX);
    aclsparseSpMatDescr_t descr = nullptr;

    // ① offsets 高位指针
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 2, highAddr, hostColInd_.data(),
                                 hostValues_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT, ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    // ② 有效 offsets + colInd 高位指针
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 2, hostOffsets_.data(), highAddr,
                                 hostValues_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT, ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";

    // ③ 有效 offsets/colInd + values 高位指针
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), highAddr, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                 ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr) << "failed Create should not touch OUT";
}
