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
 * @file bsr_set_strided_batch_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseBsrSetStridedBatch（arch35，
 *        纯 Host 接口）.
 *
 * BsrSetStridedBatch 对标 cuSPARSE cusparseBsrSetStridedBatch：CreateBsr ->
 * SetStridedBatch -> GetStridedBatch 闭环。Setter 为纯 Host 侧描述符操作，
 * 不启动 NPU kernel；三 stride 按元素个数计（offsetsSpan=blockRows+1、
 * columnsSpan=blockNnz、valuesSpan=blockNnz×bs²），0 表示 batch 间广播结构，
 * batchCount>1 且 valuesSpan>0 时 values 不允许广播。公共 getter 只回读
 * batchCount，三个 stride 经内部结构白盒核对（先例 csr_set_strided_batch）。
 *
 *   - TEST_P (BsrSetStridedBatchTest)          : bsr_set_strided_batch_test.csv
 *                                                 驱动 10 个成功用例（bc=1 非零
 *                                                 步长 / 最小非广播 / 间隔排布 /
 *                                                 offsets+columns 广播 / nnzb=0
 *                                                 全广播 / 空值矩阵部分广播 /
 *                                                 fp16 大 bc / bc=65535 上边界 /
 *                                                 bf16 大 valuesSpan / bs=128），
 *                                                 Get 默认值 + Set/Get 回读 + 白盒
 *                                                 batch 四字段断言
 *   - TEST_F (BsrSetStridedBatchSemanticsTest) : 覆盖写 latest-wins 白盒三 stride
 *                                                 逐字段（BSRB-S01）/ changed 条件化
 *                                                 缓存失效白盒（BSRB-S02：
 *                                                 changed=true 递增 structureGeneration
 *                                                 + 清空 activeBuffer/activeSpmvBuffer
 *                                                 + sddmmPrepared.prepared 置 false；
 *                                                 changed=false 全部保持）
 *   - TEST_F (BsrSetStridedBatchExceptionTest) : null / DnMat 错型句柄 / 非 BSR
 *                                                 格式（COO/CSR）/ batchCount 0·负·
 *                                                 越界 / 负 stride / 重叠步长 / 空 values
 *                                                 广播禁令 / blockRows=INT64_MAX 白盒
 *                                                 跨度溢出 / 受检跨度溢出三分量 /
 *                                                 byte extent 溢出三调用点 / 高位
 *                                                 指针注入（BSRB-E01~E11），失败后
 *                                                 断言描述符状态不被污染；白盒构造
 *                                                 用例（E08/E11）save->corrupt->
 *                                                 assert->restore（先例
 *                                                 spmat_get_strided_batch E7）
 */

#include "test_common.h"
#include "sparse_dtype_utils.h"    // ParseValueType：CSV dtype 列名 -> aclDataType（BSR 族共享）
#include "spmat_csv_common.h"      // GetCasesFromCsvChecked：带构成校验的 CSV 装载（描述符测试私有共享）
#include "spmat_test_helpers.h"    // MakeValidBsr：语义用例基线 BSR 描述符（与 create_bsr 共享）

#include <limits>

#include "bsr_set_strided_batch_param.h"
#include "bsr_set_strided_batch_npu_wrapper.h"

using namespace sparse_test;

namespace {
// BSRB-E09 受检跨度溢出注入步长（CG-4.2 具名化，先例 blockedell_get kSentinelI64）：
// 2^50 使 (bc-1)×stride 的受检乘法溢出 int64
constexpr int64_t kMulOverflowStride = static_cast<int64_t>(1) << 50;
// BSRB-E10 byte extent 溢出注入步长：2^61 使 (stride+span)×4B 的受检乘法溢出 int64
constexpr int64_t kByteOverflowStride = static_cast<int64_t>(1) << 61;
}  // namespace

// ============================================================================
// GTest parameterized fixture: BsrSetStridedBatchTest（CSV 驱动）
// ============================================================================
class BsrSetStridedBatchTest : public testing::TestWithParam<BsrSetStridedBatchParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// ============================================================================
// Test body: success-path parameterized test（每条 CSV 行一个用例）
// ============================================================================
TEST_P(BsrSetStridedBatchTest, SetGetRoundTrip)
{
    const auto &p = GetParam();
    std::cout << "==== " << p.case_name << " ==== block_rows=" << p.block_rows
              << " block_cols=" << p.block_cols << " block_nnz=" << p.block_nnz
              << " block_size=" << p.block_size << " dtype=" << p.dtype
              << " batchCount=" << p.batch_count
              << " offsetsStride=" << p.offsets_stride
              << " columnsStride=" << p.columns_stride
              << " valuesStride=" << p.values_stride << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    // 1. NPU 侧执行：CreateBsr(几何) -> Get(默认) -> Set(bc, 三 stride) -> Get 回读
    auto npu = BsrSetStridedBatchNpu(env_->stream(), p.block_rows, p.block_cols,
                                     p.block_nnz, p.block_size, ParseValueType(p.dtype),
                                     p.batch_count, p.offsets_stride, p.columns_stride,
                                     p.values_stride);

    ASSERT_EQ(npu.createRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCreateBsr failed";

    // 2. 新建描述符未设置批量语义时，默认 batchCount = 1
    ASSERT_EQ(npu.getBeforeRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(npu.defaultBatchCount, 1);

    // 3. Set 成功 + 公共 getter 回读 batchCount 与 Set 传入一致
    ASSERT_EQ(npu.setRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseBsrSetStridedBatch failed";
    ASSERT_EQ(npu.getRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseSpMatGetStridedBatch failed";
    EXPECT_EQ(npu.outBatchCount, p.batch_count);

    // 4. 白盒核对 batch 四字段与 Set 传入逐字一致（防 int64 截断，先例 csr wrapper）
    EXPECT_EQ(npu.whiteBoxBatchCount, p.batch_count);
    EXPECT_EQ(npu.whiteBoxOffsetsBatchStride, p.offsets_stride);
    EXPECT_EQ(npu.whiteBoxColumnsBatchStride, p.columns_stride);
    EXPECT_EQ(npu.whiteBoxValuesBatchStride, p.values_stride);

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    BsrSetStridedBatchCases,
    BsrSetStridedBatchTest,
    testing::ValuesIn(GetCasesFromCsvChecked<BsrSetStridedBatchParam>(
        "bsr_set_strided_batch_test.csv", 10)),
    [](const testing::TestParamInfo<BsrSetStridedBatchParam> &info) { return info.param.case_name; });

// ============================================================================
// Semantics test fixture: BsrSetStridedBatchSemanticsTest
// 语义用例：覆盖写 latest-wins 白盒 / changed 条件化缓存失效白盒。
// 基础几何 A 描述符经共享 MakeValidBsr() 构造（与 create_bsr 语义用例共用
// 同一基线：2×2 块网格、bs=2、nnzb=2、host 占位数据，Set 仅地址算术不解引用）。
// ============================================================================
class BsrSetStridedBatchSemanticsTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    void SetUp() override {}
};

// BSRB-S01: 覆盖写 latest-wins：Set(4,7,5,16) -> Set(2,3,2,8) ->
// GetStridedBatch==2 + 白盒三 stride 逐字段==第二次值 {3,2,8}
//（升级 create_bsr BSR-S03 的仅 Get==2 断言）
TEST_F(BsrSetStridedBatchSemanticsTest, OverwriteLatestWinsWhiteBox)
{
    auto mat = MakeValidBsr();
    ASSERT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 4, 7, 5, 16),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, 8),
              ACL_SPARSE_STATUS_SUCCESS);

    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 2) << "latest Set should win";

    // 白盒三 stride 逐字段为第二次 Set 的值
    EXPECT_EQ(mat.get()->batchCount, 2);
    EXPECT_EQ(mat.get()->offsetsBatchStride, 3);
    EXPECT_EQ(mat.get()->columnsBatchStride, 2);
    EXPECT_EQ(mat.get()->valuesBatchStride, 8);
}

// BSRB-S02: 缓存失效白盒（changed 语义）：
// 白盒哨兵构造（activeBuffer/activeSpmvBuffer 置非空、sddmmPrepared.prepared 置
// true，记 structureGeneration）后：Set(2,3,2,8)（changed=true）-> 重复 Set 相同值
//（changed=false）。
//   changed=true：structureGeneration+1、activeBuffer/activeSpmvBuffer 被清空、
//                 sddmmPrepared.prepared==false；
//   changed=false：哨兵与 prepared 全部保持、structureGeneration 不再递增。
// 断言后恢复白盒字段（先例 spmat_get_strided_batch E7 白盒恢复 / csr E17
// generation 断言；闭环 QA 观察项 F5）
TEST_F(BsrSetStridedBatchSemanticsTest, CacheInvalidationWhiteBox)
{
    auto mat = MakeValidBsr();
    ASSERT_NE(mat.get(), nullptr);  // 白盒解引用前判空（Create 为非 fatal EXPECT 校验）
    // 非空哨兵指针：仅作位比较，全程不被解引用
    void *const sentinelBuffer = reinterpret_cast<void *>(0x1234);

    // 白盒哨兵构造
    mat.get()->activeBuffer = sentinelBuffer;
    mat.get()->activeSpmvBuffer = sentinelBuffer;
    mat.get()->sddmmPrepared.prepared = true;
    const uint64_t genBefore = mat.get()->structureGeneration;

    // changed=true：任一字段实际变化 -> 失效预处理缓存
    ASSERT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, 8),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(mat.get()->structureGeneration, genBefore + 1);
    EXPECT_EQ(mat.get()->activeBuffer, nullptr) << "activeBuffer should be invalidated";
    EXPECT_EQ(mat.get()->activeSpmvBuffer, nullptr) << "activeSpmvBuffer should be invalidated";
    EXPECT_FALSE(mat.get()->sddmmPrepared.prepared) << "prepared should be invalidated";

    // 重新置哨兵后重复 Set 相同值（changed=false）：全部保持
    mat.get()->activeBuffer = sentinelBuffer;
    mat.get()->activeSpmvBuffer = sentinelBuffer;
    mat.get()->sddmmPrepared.prepared = true;
    const uint64_t genAfterFirst = mat.get()->structureGeneration;
    ASSERT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, 8),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(mat.get()->structureGeneration, genAfterFirst) << "no invalidation on unchanged Set";
    EXPECT_EQ(mat.get()->activeBuffer, sentinelBuffer) << "unchanged Set should keep sentinel";
    EXPECT_EQ(mat.get()->activeSpmvBuffer, sentinelBuffer) << "unchanged Set should keep sentinel";
    EXPECT_TRUE(mat.get()->sddmmPrepared.prepared) << "unchanged Set should keep prepared";

    // 恢复白盒字段（保证描述符语义干净、可正常 Destroy）
    mat.get()->activeBuffer = nullptr;
    mat.get()->activeSpmvBuffer = nullptr;
    mat.get()->sddmmPrepared.prepared = false;
}

// ============================================================================
// Exception test fixture: BsrSetStridedBatchExceptionTest
// 失败路径直调 API（不经 wrapper）；每条锁定实现源码的一个校验分支。
// 基础几何 A：2×2 块、bs=2、nnzb=2、fp32，span 3/2/8。
// ============================================================================
class BsrSetStridedBatchExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    // 基础几何 A 的占位数据（Set 仅地址算术不解引用，内容任意）
    std::vector<int32_t> hostOffsets_{0, 1, 2};
    std::vector<int32_t> hostColInd_{0, 1};
    std::vector<float> hostValues_{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    // 设备指针占位（BSRB-E11 逐项注入时，其余指针须保持有效设备指针，方案 §5.3）
    DeviceBuffer dOffsets_;
    DeviceBuffer dColInd_;
    DeviceBuffer dValues_;

    void SetUp() override
    {
        dOffsets_ = DeviceBuffer::copyFrom(hostOffsets_.data(),
                                           hostOffsets_.size() * sizeof(int32_t));
        dColInd_ = DeviceBuffer::copyFrom(hostColInd_.data(),
                                          hostColInd_.size() * sizeof(int32_t));
        dValues_ = DeviceBuffer::copyFrom(hostValues_.data(),
                                         hostValues_.size() * sizeof(float));
    }

    // 基础几何 A 描述符（unique_ptr + aclsparseDestroySpMat 删除器 RAII）
    std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)> CreateBsrGeometryA()
    {
        aclsparseSpMatDescr_t descr = nullptr;
        EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 2, dOffsets_.get(),
                                     dColInd_.get(), dValues_.get(),
                                     ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                     ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                     ACL_SPARSE_ORDER_ROW),
                  ACL_SPARSE_STATUS_SUCCESS);
        return std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)>(
            descr, &aclsparseDestroySpMat);
    }

    // 失败后状态不被污染：GetStridedBatch 回读仍为既有值（新建描述符默认 1）
    void ExpectStateUnchanged(aclsparseConstSpMatDescr_t descr, int expectedBatchCount)
    {
        int bc = -1;
        EXPECT_EQ(aclsparseSpMatGetStridedBatch(descr, &bc), ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(bc, expectedBatchCount) << "failed Set should not touch batch fields";
    }
};

// BSRB-E01: nullptr descriptor handle -> ACL_SPARSE_STATUS_INVALID_VALUE（descr.cpp:870）
TEST_F(BsrSetStridedBatchExceptionTest, NullDescr)
{
    EXPECT_EQ(aclsparseBsrSetStridedBatch(nullptr, 2, 3, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// BSRB-E02: wrong-type handle（DnMat 伪装 SpMat）-> INVALID_VALUE
//（签名不符，descr.cpp:870-873；与 getter / 仓内访问器错型防御约定一致）
TEST_F(BsrSetStridedBatchExceptionTest, WrongTypeHandle)
{
    aclsparseDnMatDescr_t dnMat = nullptr;
    ASSERT_EQ(aclsparseCreateDnMat(&dnMat, 2, 2, 2, hostValues_.data(), ACL_FLOAT,
                                   ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    auto *punned = reinterpret_cast<aclsparseSpMatDescr_t>(dnMat);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(punned, 2, 3, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyDnMat(dnMat);
}

// BSRB-E03: 非 BSR 格式描述符（COO / CSR）-> 均 INVALID_VALUE（format != BSR）
TEST_F(BsrSetStridedBatchExceptionTest, NonBsrFormat)
{
    aclsparseSpMatDescr_t cooDescr = nullptr;
    ASSERT_EQ(aclsparseCreateCoo(&cooDescr, 2, 2, 2, hostColInd_.data(),
                                 hostColInd_.data(), hostValues_.data(),
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(cooDescr, 2, 3, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroySpMat(cooDescr);

    aclsparseSpMatDescr_t csrDescr = nullptr;
    ASSERT_EQ(aclsparseCreateCsr(&csrDescr, 2, 2, 2, hostOffsets_.data(),
                                 hostColInd_.data(), hostValues_.data(),
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(csrDescr, 2, 3, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroySpMat(csrDescr);
}

// BSRB-E04: batchCount = 0 / -2 / 65536 -> 均 INVALID_VALUE（descr.cpp:362；
// [1, kMaxStridedBatchCount=65535] 为全格式共享不变量），失败后状态不被污染
TEST_F(BsrSetStridedBatchExceptionTest, InvalidBatchCount)
{
    auto mat = CreateBsrGeometryA();
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 0, 3, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), -2, 3, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 65536, 3, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStateUnchanged(mat.get(), 1);
}

// BSRB-E05: 任一 stride < 0（offsets=-1 / columns=-4 / values=-8）-> 均
// INVALID_VALUE（descr.cpp:363），失败后状态不被污染
TEST_F(BsrSetStridedBatchExceptionTest, NegativeStride)
{
    auto mat = CreateBsrGeometryA();
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, -1, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, -4, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, -8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStateUnchanged(mat.get(), 1);
}

// BSRB-E06: 重叠步长（0<s<span）-> 均 INVALID_VALUE：
//   Set(2, 1, 2, 8)（offsets 1<span=3，descr.cpp:370-371）；
//   Set(2, 3, 1, 8)（columns 1<span=2，:371）；
//   Set(2, 3, 2, 4)（values 4<span=8，:374-376）
//（stride 非 0 时必须 >= 对应数组长度；广播语义仅由 stride=0 承载）
TEST_F(BsrSetStridedBatchExceptionTest, OverlappingStrides)
{
    auto mat = CreateBsrGeometryA();
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 1, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 1, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, 4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStateUnchanged(mat.get(), 1);
}

// BSRB-E07: valuesStride=0 且 valuesSpan>0 且 bc>1 -> INVALID_VALUE
//（空值广播非法，descr.cpp:374-376；valuesSpan=0 时全 0 广播合法由 BSRB-C05 覆盖）
TEST_F(BsrSetStridedBatchExceptionTest, ValuesBroadcastNonEmpty)
{
    auto mat = CreateBsrGeometryA();
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, 0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStateUnchanged(mat.get(), 1);
}

// BSRB-E08: blockRows=INT64_MAX（白盒构造+恢复，CreateBsr 的 INT32 守卫不允许
// 直达，descr.cpp:367-369）-> Set(1, 0, 0, 0) 时 offsetsSpan=blockRows+1 受检
// 加法溢出 -> INVALID_VALUE；恢复后 descr 可正常 Set/Destroy。
//（batchCount 取 1、步长全 0 也不进入受检乘加，溢出须在跨度构造处拦截）
TEST_F(BsrSetStridedBatchExceptionTest, OffsetsSpanOverflow)
{
    auto mat = CreateBsrGeometryA();
    ASSERT_NE(mat.get(), nullptr);  // 白盒解引用前判空（Create 为非 fatal EXPECT 校验）
    const int64_t savedBlockRows = mat.get()->blockRows;
    mat.get()->blockRows = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 1, 0, 0, 0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    mat.get()->blockRows = savedBlockRows;  // 恢复（先例 E7 save->corrupt->assert->restore）

    // 恢复后描述符可正常 Set / Destroy
    ASSERT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, 8),
              ACL_SPARSE_STATUS_SUCCESS);
    int bc = -1;
    EXPECT_EQ(aclsparseSpMatGetStridedBatch(mat.get(), &bc), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bc, 2);
}

// BSRB-E09: 受检跨度溢出三分量（CheckedStridedExtent，descr.cpp:333-337，先例
// csr_set_strided_batch E9 的 2^50 形态）：
//   Set(65535, 2^50, 2, 8)：offsets 侧 (bc-1)×stride=65534×2^50 乘法溢出（:333）；
//   Set(2, 3, INT64_MAX, 8)：columns 侧 (bc-1)×stride 乘法通过、+columnsSpan
//       =INT64_MAX+2 受检加法溢出（:334）；
//   Set(2, 3, 2, INT64_MAX)：values 侧 INT64_MAX+8 受检加法溢出（:335）
// 均 INVALID_VALUE（batchCount 取合法值，确保断言真正锁定溢出防护分支），失败后
// 状态不变
TEST_F(BsrSetStridedBatchExceptionTest, StridedExtentOverflow)
{
    auto mat = CreateBsrGeometryA();
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 65535, kMulOverflowStride, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3,
                                          std::numeric_limits<int64_t>::max(), 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2,
                                          std::numeric_limits<int64_t>::max()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStateUnchanged(mat.get(), 1);
}

// BSRB-E10: byte extent 溢出三分量（fp32 元素 4B；CheckedStridedExtent 通过、
// 元素数×4B 受检乘法溢出，descr.cpp:341-345 三 byte 调用点各有注入）：
//   Set(2, 2^61, 2, 8)：offsets extent=(2-1)×2^61+3，(2^61+3)×4B=2^63+12 溢出
//       （:341-342 CheckedNonnegativeMul 调用点）；
//   Set(2, 3, 2^61, 8)：columns extent=(2-1)×2^61+2，(2^61+2)×4B=2^63+8 溢出
//       （:343-344 CheckedNonnegativeMul 调用点）；
//   Set(2, 3, 2, 2^61)：values extent=(2-1)×2^61+8，(2^61+8)×4B=2^63+32 溢出
//       （:345 CheckedElementExtentBytes 调用点）
// 均 INVALID_VALUE，失败后状态不变
TEST_F(BsrSetStridedBatchExceptionTest, ByteExtentOverflow)
{
    auto mat = CreateBsrGeometryA();
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, kByteOverflowStride, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, kByteOverflowStride, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, kByteOverflowStride),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStateUnchanged(mat.get(), 1);
}

// BSRB-E11: 地址回绕（CheckedAddressExtent，descr.cpp:346-348）：白盒逐项改
// ptrs / idxs / values = (void*)UINTPTR_MAX（其余指针保持有效设备指针）->
// 合法参数 Set(2, 3, 2, 8) 均 INVALID_VALUE：
//   ① ptrs 高位注入：offsetsByteExtent=((2-1)×3+3)×4B=24B>0（方案 §5.3 括注 24B）；
//   ② idxs 高位注入：columnsByteExtent=((2-1)×2+2)×4B=16B>0；
//   ③ values 高位注入：valuesByteExtent=((2-1)×8+8)×4B=64B>0
//（lastByteOffset ≥ 3 > UINTPTR_MAX - address = 0，确定性拒绝；Set 仅地址算术
// 不解引用指针，无 UB；同 create_bsr BSR-E13 注入形态）
// 白盒构造+恢复，descr 可正常 Set/Destroy（先例 E7 save->corrupt->assert->restore）
TEST_F(BsrSetStridedBatchExceptionTest, AddressExtentOverflow)
{
    auto mat = CreateBsrGeometryA();
    ASSERT_NE(mat.get(), nullptr);  // 白盒解引用前判空（Create 为非 fatal EXPECT 校验）
    void *const highAddr = reinterpret_cast<void *>(UINTPTR_MAX);

    // ① ptrs 高位注入（其余保持有效设备指针）
    void *savedPtrs = mat.get()->ptrs;
    mat.get()->ptrs = highAddr;
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStateUnchanged(mat.get(), 1);
    mat.get()->ptrs = savedPtrs;  // 恢复

    // ② idxs 高位注入
    void *savedIdxs = mat.get()->idxs;
    mat.get()->idxs = highAddr;
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStateUnchanged(mat.get(), 1);
    mat.get()->idxs = savedIdxs;  // 恢复

    // ③ values 高位注入
    void *savedValues = mat.get()->values;
    mat.get()->values = highAddr;
    EXPECT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStateUnchanged(mat.get(), 1);
    mat.get()->values = savedValues;  // 恢复

    // 恢复后描述符可正常 Set / Destroy
    ASSERT_EQ(aclsparseBsrSetStridedBatch(mat.get(), 2, 3, 2, 8),
              ACL_SPARSE_STATUS_SUCCESS);
}
