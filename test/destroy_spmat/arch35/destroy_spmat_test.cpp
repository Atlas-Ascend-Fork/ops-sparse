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
 * @file destroy_spmat_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseDestroySpMat（arch35，纯 Host 接口）.
 *
 * DestroySpMat 对标 cuSPARSE cusparseDestroySpMat：销毁稀疏矩阵描述符。
 * 分支语义：null -> SUCCESS（null 容忍）；签名不符（错型句柄）->
 * INVALID_VALUE 且不 delete；合法 -> 置零签名 + delete -> SUCCESS。
 * aclsparseConstSpMatDescr_t 同签名可传。判定口径为返回码位精确相等。
 *
 *   - TEST_P (DestroySpMatTest)          : destroy_spmat_test.csv 驱动的 11 个
 *                                          成功用例（11 个创建入口：6 mutable +
 *                                          5 const 变体），Create -> 手动
 *                                          Destroy（guard release 语义）->
 *                                          SUCCESS
 *   - TEST_F (DestroySpMatExceptionTest) : null 容忍（DS-E01）/ SpVec + DnMat
 *                                          错型句柄拒绝且随后各自正常销毁
 *                                          （DS-E02，错误路径无泄漏的可观测证明）/
 *                                          白盒损坏签名 -> INVALID_VALUE ->
 *                                          恢复 -> 正常 Destroy SUCCESS
 *                                          （DS-E03，失败路径不 delete 的可观测
 *                                          证明；save->corrupt->assert->restore，
 *                                          先例 spmat_get_strided_batch E7）
 *
 * 覆盖说明：双 destroy（合法销毁后句柄悬垂）不设用例——Destroy 置零签名后即
 * delete，二次调用属 use-after-free UB，不可注入；其签名拒绝分支已由 DS-E02/
 * E03 等价覆盖（对齐 spmat_get_indexbase 覆盖说明惯例）。
 */

#include "test_common.h"
#include "spmat_csv_common.h"      // GetCasesFromCsvChecked：带构成校验的 CSV 装载（描述符测试私有共享）

#include "aclsparse_descr_internal.h"  // 白盒：DS-E03 损坏签名用例需访问 signature 字段

#include "destroy_spmat_param.h"
#include "destroy_spmat_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// 入口注册表（对齐先例 test/spmat_get_indexbase 的 kEntries 位置：编译单元
// 私有数据，不在 wrapper 头文件中定义非 inline 数组）。
// 11 个创建入口：6 mutable + 5 const（blocked_ell 无 const 变体）。
// entry 名查找经共享 FindEntry 模板（test/spmat_descr_common/spmat_test_helpers.h，与
// spmat_get_attribute_test.cpp 共用查找逻辑）。
// ============================================================================
namespace {
const FormatEntry kEntries[] = {
    {"coo", false},            {"const_csr", true},
    {"csr", false},            {"const_coo", true},
    {"csc", false},            {"const_csc", true},
    {"bsr", false},            {"const_bsr", true},
    {"blocked_ell", false},    {"sliced_ell", false},
    {"const_sliced_ell", true},
};

// DS-E03 白盒损坏签名注入值（CG-4.2 具名化）：与全部合法签名可区分
constexpr uint32_t kCorruptedSignature = 0xDEADBEEFU;
}  // namespace

// ============================================================================
// GTest parameterized fixture: DestroySpMatTest（CSV 驱动）
// ============================================================================
class DestroySpMatTest : public testing::TestWithParam<DestroySpMatParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// ============================================================================
// Test body: success-path parameterized test（每条 CSV 行一个用例）
// ============================================================================
TEST_P(DestroySpMatTest, CreateThenDestroyRoundTrip)
{
    const auto &p = GetParam();
    std::cout << "==== " << p.case_name << " ==== entry=" << p.entry << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    const FormatEntry *entry = FindEntry(kEntries, p.entry);
    ASSERT_NE(entry, nullptr) << "unknown entry in CSV: " << p.entry;

    // 占位设备缓冲（Create/Destroy 均不解引用设备数据，内容任意，仅要求指针有效）
    const std::vector<int32_t> hostIdx(16, 0);
    const std::vector<float> hostVal(32, 1.0F);
    auto ptrs = DeviceBuffer::copyFrom(hostIdx.data(), hostIdx.size() * sizeof(int32_t));
    auto idxs = DeviceBuffer::copyFrom(hostIdx.data(), hostIdx.size() * sizeof(int32_t));
    auto vals = DeviceBuffer::copyFrom(hostVal.data(), hostVal.size() * sizeof(float));

    // 1. NPU 侧执行：Create(entry) -> 手动 Destroy（guard release 语义）
    auto r = DestroySpMatNpu(env_->stream(), *entry, ptrs.get(), idxs.get(), vals.get());

    // 2. Create 与 Destroy 均 SUCCESS（返回码位精确相等）
    ASSERT_EQ(r.createRet, ACL_SPARSE_STATUS_SUCCESS) << "create " << entry->name << " failed";
    EXPECT_EQ(r.destroyRet, ACL_SPARSE_STATUS_SUCCESS)
        << "aclsparseDestroySpMat(" << entry->name << ") failed";

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    DestroySpMatCases,
    DestroySpMatTest,
    testing::ValuesIn(GetCasesFromCsvChecked<DestroySpMatParam>(
        "destroy_spmat_test.csv", 11)),
    [](const testing::TestParamInfo<DestroySpMatParam> &info) { return info.param.case_name; });

// ============================================================================
// Exception test fixture: DestroySpMatExceptionTest
// ============================================================================
class DestroySpMatExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    std::vector<int32_t> hostIdx_{0, 1, 2, 3, 4};
    std::vector<float> hostVal_{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};

    void SetUp() override {}
};

// DS-E01: DestroySpMat(nullptr) -> SUCCESS（null 容忍，对齐 cuSPARSE
// cusparseDestroySpMat；descr.cpp:1093-1095；仅返回码断言，无后续解引用）
TEST_F(DestroySpMatExceptionTest, NullReturnsSuccess)
{
    EXPECT_EQ(aclsparseDestroySpMat(nullptr), ACL_SPARSE_STATUS_SUCCESS);
}

// DS-E02: SpVec / DnMat 错型句柄分别伪装 aclsparseConstSpMatDescr_t ->
// 均 INVALID_VALUE（签名不符 :1097-1099 拒绝且不 delete）；两对象随后各自经
// aclsparseDestroySpVec / aclsparseDestroyDnMat 正常销毁——错误路径无泄漏的
// 可观测证明
TEST_F(DestroySpMatExceptionTest, WrongTypeHandle)
{
    // SpVec 句柄伪装（签名 kSpVecSignature != kSpMatSignature）
    aclsparseSpVecDescr_t spVec = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&spVec, 5, 5, hostIdx_.data(), hostVal_.data(),
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                   ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseDestroySpMat(reinterpret_cast<aclsparseConstSpMatDescr_t>(spVec)),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseDestroySpVec(spVec), ACL_SPARSE_STATUS_SUCCESS)
        << "rejected Destroy must not delete; object should still be destroyable";

    // DnMat 句柄伪装（签名 kDnMatSignature != kSpMatSignature）
    aclsparseDnMatDescr_t dnMat = nullptr;
    ASSERT_EQ(aclsparseCreateDnMat(&dnMat, 2, 2, 2, hostVal_.data(), ACL_FLOAT,
                                   ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseDestroySpMat(reinterpret_cast<aclsparseConstSpMatDescr_t>(dnMat)),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseDestroyDnMat(dnMat), ACL_SPARSE_STATUS_SUCCESS)
        << "rejected Destroy must not delete; object should still be destroyable";
}

// DS-E03: 白盒损坏签名（0xDEADBEEF）-> Destroy -> INVALID_VALUE -> 恢复签名 ->
// 再次 Destroy -> SUCCESS：失败路径不 delete 的可观测证明（恢复后可正常销毁）；
// save->corrupt->assert->restore 形态（先例 spmat_get_strided_batch E7 签名恢复）
TEST_F(DestroySpMatExceptionTest, CorruptedSignatureNotDeleted)
{
    aclsparseSpMatDescr_t descr = nullptr;
    ASSERT_EQ(aclsparseCreateCsr(&descr, 4, 4, 5, hostIdx_.data(), hostIdx_.data(),
                                 hostVal_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                 ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_NE(descr, nullptr);

    // 白盒损坏签名 -> Destroy 拒绝
    const uint32_t savedSignature = descr->signature;
    descr->signature = kCorruptedSignature;
    EXPECT_EQ(aclsparseDestroySpMat(descr), ACL_SPARSE_STATUS_INVALID_VALUE);

    // 恢复签名 -> 正常 Destroy SUCCESS（证明失败路径未 delete，无泄漏）
    descr->signature = savedSignature;
    EXPECT_EQ(aclsparseDestroySpMat(descr), ACL_SPARSE_STATUS_SUCCESS);
}
