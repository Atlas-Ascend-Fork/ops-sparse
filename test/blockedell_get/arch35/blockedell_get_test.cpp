/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file blockedell_get_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseCreateBlockedEll / aclsparseBlockedEllGet (arch35).
 *
 * BlockedEllGet 对标 cuSPARSE cusparseBlockedEllGet：CreateBlockedEll -> BlockedEllGet 闭环，
 * 读回字段应与 Create 传入一致。BlockedEllGet 是纯 Host 侧描述符访问器，不启动 NPU kernel；
 * 测试侧建设备缓冲存入描述符，BlockedEllGet 读回字段后断言与 Create 传入一致。
 *
 *   - TEST_P (BlockedEllGetTest)        : 10 个 CSV 驱动的成功路径用例（FP16/FP32,
 *                                         idx0/idx1, 多种块大小, 方阵/非方阵/行尾块/列尾块）
 *   - TEST_F (BlockedEllGetSuccessTest) : 非 CSV 成功路径用例（64I 索引 / const 变体
 *                                         round-trip / 部分空 OUT / 零容量 nullptr 指针）
 *   - TEST_F (BlockedEllGetExceptionTest): 错误路径（null descr / 非 BELL 格式（CSR/COO/CSC）/
 *                                         签名损坏句柄 / 非法 Create 参数），
 *                                         错误返回后断言 OUT 哨兵值不被污染
 */

#include "test_common.h"

#include "aclsparse_descr_internal.h" // 白盒：签名损坏用例需访问描述符内部字段

#include "../blockedell_get_param.h"
#include "blockedell_get_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// 错误路径 OUT 哨兵：预置明确非法值，断言错误返回后 OUT 保持哨兵不变（未被污染）。
// int64 字段与设备指针哨兵取 0xDEADBEEF；枚举哨兵取与被测描述符字段可区分的有效枚举值。
// ============================================================================
namespace {
constexpr int64_t kSentinelI64 = static_cast<int64_t>(0xDEADBEEF);

inline void *SentinelDevicePtr()
{
    return reinterpret_cast<void *>(0xDEADBEEFULL);
}

// BlockedEllGet 的 10 个 OUT 字段集合：默认构造即预置哨兵，可整组传入 Get 调用。
struct BlockedEllGetOut {
    int64_t r{kSentinelI64};
    int64_t c{kSentinelI64};
    int64_t bs{kSentinelI64};
    int64_t ec{kSentinelI64};
    void *ci{SentinelDevicePtr()};
    void *v{SentinelDevicePtr()};
    aclsparseIndexType_t it{ACL_SPARSE_INDEX_64I};
    aclsparseIndexBase_t ib{ACL_SPARSE_INDEX_BASE_ONE};
    aclDataType vt{ACL_BF16};
};

// 断言错误路径不污染 OUT：全部字段保持哨兵值不变。
inline void ExpectOutUnchanged(const BlockedEllGetOut &out)
{
    EXPECT_EQ(out.r, kSentinelI64);
    EXPECT_EQ(out.c, kSentinelI64);
    EXPECT_EQ(out.bs, kSentinelI64);
    EXPECT_EQ(out.ec, kSentinelI64);
    EXPECT_EQ(out.ci, SentinelDevicePtr());
    EXPECT_EQ(out.v, SentinelDevicePtr());
    EXPECT_EQ(out.it, ACL_SPARSE_INDEX_64I);
    EXPECT_EQ(out.ib, ACL_SPARSE_INDEX_BASE_ONE);
    EXPECT_EQ(out.vt, ACL_BF16);
}
}  // namespace

// ============================================================================
// GTest parameterized fixture: BlockedEllGetTest
// ============================================================================
class BlockedEllGetTest : public testing::TestWithParam<BlockedEllGetParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    BlockedEllGetParam param_;

    void SetUp() override { param_ = GetParam(); }
};

// ============================================================================
// Test body: success-path parameterized test
// ============================================================================
TEST_P(BlockedEllGetTest, BlockedEllGetRoundTrip) {
    const auto &p = param_;
    std::cout << "==== " << p.case_name << " ==== rows=" << p.rows << " cols=" << p.cols
              << " bs=" << p.ell_block_size << " ellCols=" << p.ell_cols
              << " dtype=" << p.dtype << " idx_base=" << p.idx_base << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    // 1. 生成 Blocked-ELL 输入数据（host 侧仅生成输入，不在 CPU 做算子计算）
    std::vector<int32_t> hostColInd;
    GenerateBlockedEllData(p.rows, p.cols, p.ell_block_size, p.ell_cols, p.idx_base, p.seed,
                           hostColInd);

    // 2. NPU 执行：CreateBlockedEll -> BlockedEllGet accessor 验证
    auto npu = BlockedEllGetNpu(env_->stream(), hostColInd, p.rows, p.cols, p.ell_block_size,
                                p.ell_cols, p.idx_base, p.isFp16());

    ASSERT_EQ(npu.createRet, ACL_SPARSE_STATUS_SUCCESS)
        << "aclsparseCreateBlockedEll failed";
    ASSERT_EQ(npu.getRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseBlockedEllGet failed";

    // 3. 断言 Get 读回的标量元数据 == Create 传入
    EXPECT_EQ(npu.rows, p.rows);
    EXPECT_EQ(npu.cols, p.cols);
    EXPECT_EQ(npu.ellBlockSize, p.ell_block_size);
    EXPECT_EQ(npu.ellCols, p.ell_cols);
    EXPECT_EQ(npu.idxType, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(npu.idxBase, (p.idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE
                                             : ACL_SPARSE_INDEX_BASE_ZERO);
    EXPECT_EQ(npu.valueType, p.isFp16() ? ACL_FLOAT16 : ACL_FLOAT);

    // 4. 断言 Get 读回的设备指针 == Create 传入的设备指针（accessor 语义）
    EXPECT_TRUE(npu.devPtrMatch) << "BlockedEllGet device pointer mismatch";

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    BlockedEllGetCases,
    BlockedEllGetTest,
    testing::ValuesIn(GetCasesFromCsv<BlockedEllGetParam>("blockedell_get_test.csv")),
    [](const testing::TestParamInfo<BlockedEllGetParam> &info) {
        return info.param.case_name;
    });

// ============================================================================
// Success-path test fixture: BlockedEllGetSuccessTest
// 非 CSV 参数化的成功路径用例：全部断言 Create/Get 返回 SUCCESS 且字段回读一致。
// ============================================================================
class BlockedEllGetSuccessTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    // 2x2 矩阵按 bs=2 / ellCols=2 存为单块 BELL（accessor 只存指针，不读数据）
    std::vector<int32_t> colInd_{0};
    std::vector<float> values_{1.0f, 2.0f, 3.0f, 4.0f};

    void SetUp() override {}
};

// E3: 64-bit index type is supported (same as cuSPARSE)
TEST_F(BlockedEllGetSuccessTest, CreateBlockedEllIndex64Supported) {
    // 绑定真实的 int64 列块索引缓冲（indexCount = ceil(2/2)*ceil(2/2) = 1）
    std::vector<int64_t> colInd64{0};
    aclsparseSpMatDescr_t descr = nullptr;
    ASSERT_EQ(aclsparseCreateBlockedEll(&descr, 2, 2, 2, 2, colInd64.data(), values_.data(),
                                        ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_NE(descr, nullptr);
    int64_t r = -1, c = -1, bs = -1, ec = -1;
    void *ci = nullptr;
    void *v = nullptr;
    aclsparseIndexType_t it = ACL_SPARSE_INDEX_32I;
    aclsparseIndexBase_t ib;
    aclDataType vt;
    EXPECT_EQ(aclsparseBlockedEllGet(descr, &r, &c, &bs, &ec, &ci, &v, &it, &ib, &vt),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r, 2);
    EXPECT_EQ(c, 2);
    EXPECT_EQ(bs, 2);
    EXPECT_EQ(ec, 2);
    EXPECT_EQ(it, ACL_SPARSE_INDEX_64I);
    EXPECT_EQ(ci, colInd64.data());
    EXPECT_EQ(v, values_.data());
    aclsparseDestroySpMat(descr);
}

// E6: full const variant success (aclsparseConstBlockedEllGet round-trip)
TEST_F(BlockedEllGetSuccessTest, ConstBlockedEllGetRoundTrip) {
    aclsparseSpMatDescr_t descr = nullptr;
    ASSERT_EQ(aclsparseCreateBlockedEll(&descr, 2, 2, 2, 2, colInd_.data(), values_.data(),
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ONE,
                                        ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    int64_t r = -1, c = -1, bs = -1, ec = -1;
    const void *ci = nullptr;
    const void *v = nullptr;
    aclsparseIndexType_t it;
    aclsparseIndexBase_t ib;
    aclDataType vt;
    EXPECT_EQ(aclsparseConstBlockedEllGet(descr, &r, &c, &bs, &ec, &ci, &v, &it, &ib, &vt),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r, 2);
    EXPECT_EQ(c, 2);
    EXPECT_EQ(bs, 2);
    EXPECT_EQ(ec, 2);
    EXPECT_EQ(ib, ACL_SPARSE_INDEX_BASE_ONE);
    EXPECT_EQ(vt, ACL_FLOAT);
    EXPECT_EQ(ci, colInd_.data());
    EXPECT_EQ(v, values_.data());
    EXPECT_EQ(it, ACL_SPARSE_INDEX_32I);
    aclsparseDestroySpMat(descr);
}

// E7: partial nullptr OUT params -> SUCCESS, only requested fields are written
TEST_F(BlockedEllGetSuccessTest, PartialNullOutParams) {
    // rows=4, cols=8, bs=2, ellCols=4：indexCount = 2*2 = 4，valueCount = 4*4 = 16 个
    // FP16 标量——缓冲按描述符所需容量分配，而不是复用 fixture 的小数组
    std::vector<int32_t> colInd(4, 0);
    std::vector<uint16_t> values(16, 0);
    aclsparseSpMatDescr_t descr = nullptr;
    ASSERT_EQ(aclsparseCreateBlockedEll(&descr, 4, 8, 2, 4, colInd.data(), values.data(),
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT16),
              ACL_SPARSE_STATUS_SUCCESS);
    int64_t r = -1;
    int64_t ec = -1;
    void *v = nullptr;
    aclDataType vt = ACL_FLOAT;
    EXPECT_EQ(aclsparseBlockedEllGet(descr, &r, nullptr, nullptr, &ec, nullptr, &v, nullptr,
                                     nullptr, &vt),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r, 4);
    EXPECT_EQ(ec, 4);
    EXPECT_EQ(v, values.data());
    EXPECT_EQ(vt, ACL_FLOAT16);
    aclsparseDestroySpMat(descr);
}

// N2: 零容量/退化维度合法路径：rows=0 且 ellCols=0 时补齐后块网格容量为 0，
// 数据指针允许为 nullptr，Create/Get 均应成功并回读退化维度字段。
TEST_F(BlockedEllGetSuccessTest, ZeroCapacityNullDataPointers) {
    aclsparseSpMatDescr_t descr = nullptr;
    ASSERT_EQ(aclsparseCreateBlockedEll(&descr, 0, 0, 2, 0, nullptr, nullptr,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_NE(descr, nullptr);
    int64_t r = -1, c = -1, bs = -1, ec = -1;
    void *ci = SentinelDevicePtr();  // 零容量时应回读为 nullptr
    void *v = SentinelDevicePtr();
    aclsparseIndexType_t it = ACL_SPARSE_INDEX_64I;
    aclsparseIndexBase_t ib = ACL_SPARSE_INDEX_BASE_ONE;
    aclDataType vt = ACL_BF16;
    EXPECT_EQ(aclsparseBlockedEllGet(descr, &r, &c, &bs, &ec, &ci, &v, &it, &ib, &vt),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r, 0);
    EXPECT_EQ(c, 0);
    EXPECT_EQ(bs, 2);
    EXPECT_EQ(ec, 0);
    EXPECT_EQ(ci, nullptr);
    EXPECT_EQ(v, nullptr);
    EXPECT_EQ(it, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(ib, ACL_SPARSE_INDEX_BASE_ZERO);
    EXPECT_EQ(vt, ACL_FLOAT);
    aclsparseDestroySpMat(descr);
}

// ============================================================================
// Exception test fixture: BlockedEllGetExceptionTest
// 错误路径用例：断言错误码返回，且 OUT 哨兵值保持不变（未被污染）。
// ============================================================================
class BlockedEllGetExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    // 2x2 矩阵按 bs=2 / ellCols=2 存为单块 BELL（accessor 只存指针，不读数据）
    std::vector<int32_t> colInd_{0};
    std::vector<float> values_{1.0f, 2.0f, 3.0f, 4.0f};

    void SetUp() override {}
};

// E1: nullptr descriptor handle -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(BlockedEllGetExceptionTest, NullSpMatDescrHandle) {
    BlockedEllGetOut out;
    EXPECT_EQ(aclsparseBlockedEllGet(nullptr, &out.r, &out.c, &out.bs, &out.ec, &out.ci,
                                     &out.v, &out.it, &out.ib, &out.vt),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseConstBlockedEllGet(nullptr, &out.r, &out.c, &out.bs, &out.ec,
        const_cast<const void **>(&out.ci), const_cast<const void **>(&out.v),
        &out.it, &out.ib, &out.vt),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    // 错误路径不应污染 OUT：全部字段保持哨兵值不变
    ExpectOutUnchanged(out);
}

// E2: nullptr out handle for CreateBlockedEll -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(BlockedEllGetExceptionTest, CreateBlockedEllNullOutHandle) {
    EXPECT_EQ(aclsparseCreateBlockedEll(nullptr, 2, 2, 2, 2, colInd_.data(), values_.data(),
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E4: invalid create params (negative dims / ellBlockSize <= 0 / bad index type)
//      -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(BlockedEllGetExceptionTest, CreateBlockedEllInvalidParams) {
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateBlockedEll(&descr, -1, 2, 2, 2, colInd_.data(), values_.data(),
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCreateBlockedEll(&descr, 2, -2, 2, 2, colInd_.data(), values_.data(),
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseCreateBlockedEll(&descr, 2, 2, 0, 2, colInd_.data(), values_.data(),
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    // ellCols 的块槽数超过 cols 的补齐块列数 -> 容量校验失败
    EXPECT_EQ(aclsparseCreateBlockedEll(&descr, 4, 4, 2, 6, colInd_.data(), values_.data(),
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    if (descr != nullptr) {
        aclsparseDestroySpMat(descr);
    }
}

// E5: BlockedEllGet on a non-BLOCKED_ELL descriptor (CSR)
//      -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(BlockedEllGetExceptionTest, BlockedEllGetOnNonBellDescr) {
    // 先构造一个 CSR 描述符
    std::vector<int32_t> rowOff{0, 1, 2};
    std::vector<int32_t> colInd{0, 1};
    aclsparseSpMatDescr_t csrDescr = nullptr;
    ASSERT_EQ(aclsparseCreateCsr(&csrDescr, 2, 2, 2, rowOff.data(), colInd.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    BlockedEllGetOut out;  // 哨兵枚举取与 CSR 字段可区分的枚举值
    EXPECT_EQ(aclsparseBlockedEllGet(csrDescr, &out.r, &out.c, &out.bs, &out.ec, &out.ci,
                                      &out.v, &out.it, &out.ib, &out.vt),
              ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    // 错误路径不应污染 OUT：全部字段保持哨兵值不变
    ExpectOutUnchanged(out);
    aclsparseDestroySpMat(csrDescr);
}

// E5b: BlockedEllGet on a non-BLOCKED_ELL descriptor (COO)
//      -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(BlockedEllGetExceptionTest, BlockedEllGetOnCooDescr) {
    std::vector<int32_t> rowInd{0, 1};
    std::vector<int32_t> colInd{0, 1};
    aclsparseSpMatDescr_t cooDescr = nullptr;
    ASSERT_EQ(aclsparseCreateCoo(&cooDescr, 2, 2, 2, rowInd.data(), colInd.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    BlockedEllGetOut out;
    EXPECT_EQ(aclsparseBlockedEllGet(cooDescr, &out.r, &out.c, &out.bs, &out.ec, &out.ci,
                                      &out.v, &out.it, &out.ib, &out.vt),
              ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    // 错误路径不应污染 OUT：全部字段保持哨兵值不变
    ExpectOutUnchanged(out);
    aclsparseDestroySpMat(cooDescr);
}

// E5c: BlockedEllGet on a non-BLOCKED_ELL descriptor (CSC)
//      -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(BlockedEllGetExceptionTest, BlockedEllGetOnCscDescr) {
    std::vector<int32_t> colOff{0, 1, 2};
    std::vector<int32_t> rowInd{0, 1};
    aclsparseSpMatDescr_t cscDescr = nullptr;
    ASSERT_EQ(aclsparseCreateCsc(&cscDescr, 2, 2, 2, colOff.data(), rowInd.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    BlockedEllGetOut out;
    EXPECT_EQ(aclsparseBlockedEllGet(cscDescr, &out.r, &out.c, &out.bs, &out.ec, &out.ci,
                                      &out.v, &out.it, &out.ib, &out.vt),
              ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    // 错误路径不应污染 OUT：全部字段保持哨兵值不变
    ExpectOutUnchanged(out);
    aclsparseDestroySpMat(cscDescr);
}

// E8: 签名损坏的句柄 -> ACL_SPARSE_STATUS_INVALID_VALUE
//（白盒构造：模拟误传非 SpMat 句柄 / 悬垂句柄，校验后恢复以保证正常释放）
TEST_F(BlockedEllGetExceptionTest, GetOnCorruptedSignature) {
    aclsparseSpMatDescr_t descr = nullptr;
    ASSERT_EQ(aclsparseCreateBlockedEll(&descr, 2, 2, 2, 2, colInd_.data(), values_.data(),
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    const uint32_t savedSignature = descr->signature;
    descr->signature = 0xDEADBEEFU;
    BlockedEllGetOut out;
    EXPECT_EQ(aclsparseBlockedEllGet(descr, &out.r, &out.c, &out.bs, &out.ec, &out.ci,
                                      &out.v, &out.it, &out.ib, &out.vt),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseConstBlockedEllGet(descr, &out.r, &out.c, &out.bs, &out.ec,
        const_cast<const void **>(&out.ci), const_cast<const void **>(&out.v),
        &out.it, &out.ib, &out.vt),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    // 错误路径不应污染 OUT：全部字段保持哨兵值不变
    ExpectOutUnchanged(out);
    descr->signature = savedSignature; // 恢复签名，使 Destroy 可正常释放
    aclsparseDestroySpMat(descr);
}
