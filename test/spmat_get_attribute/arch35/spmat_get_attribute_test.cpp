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
 * @file spmat_get_attribute_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseSpMatGetAttribute /
 *        aclsparseSpMatSetAttribute（arch35，纯 Host 接口）.
 *
 * 对标 cuSPARSE cusparseSpMatGetAttribute / cusparseSpMatSetAttribute：
 * Set(fill, diag) -> Get 回读的元数据 roundtrip，判定口径为枚举位精确相等
 * （EXACT，零容差）。Get/SetAttribute 均为纯 Host 侧描述符属性操作，不启动
 * NPU kernel；测试侧建设备缓冲存入描述符，Get 读回后断言与 Set 传入一致。
 *
 *   - TEST_P (SpMatGetAttributeTest)          : spmat_get_attribute_test.csv 驱动，
 *                                               csr / bsr / sliced_ell 入口 ×
 *                                               fill{LOWER,UPPER} × diag{NON_UNIT,UNIT}
 *                                               = 12 用例的 Set->Get 往返（含默认值
 *                                               断言与二次 Get 幂等）
 *   - TEST_F (SpMatGetAttributeSemanticsTest) : 默认值（GA-S01）/ const 签名路径
 *                                               （GA-S02）/ 属性隔离与覆盖写
 *                                               （GA-S03）/ 失败 Set 状态保持（GA-S04）
 *   - TEST_F (SpMatGetAttributeExceptionTest) : null 描述符 / null data / 非法
 *                                               dataSize / 错位指针 / 非法 attribute
 *                                               枚举 / Set 侧共享守卫抽样（GA-E01~E06），
 *                                               错误返回后断言 OUT 哨兵值不被污染
 */

#include <algorithm>

#include "test_common.h"
#include "spmat_csv_common.h"      // GetCasesFromCsvChecked：带构成校验的 CSV 装载（描述符测试私有共享）

#include "spmat_get_attribute_param.h"
#include "spmat_get_attribute_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// 错误路径 OUT 哨兵（int 承载，按 int 比较）：预置明确非法值 99，与全部合法
// 枚举值（0/1）可区分。Get/SetAttribute 的 data 形参为 void*，int 局部变量
// 可直接承载（CS-1.3：越界值不再经 static_cast<枚举> 构造，规避 ISO C++ UB）。
// ============================================================================
namespace {
constexpr int kSentinelEnum = 99;

// 枚举越界注入值的合法构造：等宽位重解释（CS-1.3 方案二，规避
// static_cast<枚举>(越界 int) 的 UB）。以 char 逐字节循环实现——codecheck
// 按 C 内存拷贝类内建函数报告 memcpy 须规避、C++17 无 std::bit_cast；char
// 可安全别名任意对象表示，编译器可将其完全优化为单次 load/store（先例
// test/prune_check/prune_check_golden.h 的 BitCastBits）；枚举与 int 同宽由
// static_assert 锁定。
template <typename EnumT>
inline EnumT EnumFromRawInt(int rawValue)
{
    static_assert(sizeof(EnumT) == sizeof(int), "enum must have int-sized object representation");
    EnumT e{};
    char *dst = reinterpret_cast<char *>(&e);
    const char *src = reinterpret_cast<const char *>(&rawValue);
    for (size_t i = 0; i < sizeof(e); ++i) {
        dst[i] = src[i];
    }
    return e;
}

// ============================================================================
// 入口注册表（对齐先例 test/spmat_get_indexbase 的 kEntries 位置：编译单元
// 私有数据，不在 wrapper 头文件中定义非 inline 数组）。
// 属性与格式无关，仅取有代表性的 csr / bsr / sliced_ell 三种 mutable 入口
// （CSV fill × diag 全组合）+ const_csr / const_bsr 两个 const 入口（语义用例
// GA-S02 的 const 签名 Get 路径）。
// entry 名查找经共享 FindEntry 模板（test/spmat_descr_common/spmat_test_helpers.h，与
// destroy_spmat_test.cpp 共用查找逻辑）。
// ============================================================================
const FormatEntry kEntries[] = {
    {"csr", false},          {"bsr", false},
    {"sliced_ell", false},   {"const_csr", true},
    {"const_bsr", true},
};
}  // namespace

// ============================================================================
// GTest parameterized fixture: SpMatGetAttributeTest（CSV 驱动）
// ============================================================================
class SpMatGetAttributeTest : public testing::TestWithParam<SpMatGetAttributeParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// ============================================================================
// Test body: success-path parameterized test（每条 CSV 行一个用例）
// ============================================================================
TEST_P(SpMatGetAttributeTest, AttributeRoundTrip)
{
    const auto &p = GetParam();
    std::cout << "==== " << p.case_name << " ==== entry=" << p.entry
              << " fill_mode=" << p.fill_mode << " diag_type=" << p.diag_type
              << " idx_base=" << p.idx_base << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    const FormatEntry *entry = FindEntry(kEntries, p.entry);
    ASSERT_NE(entry, nullptr) << "unknown entry in CSV: " << p.entry;
    const aclsparseFillMode_t fill =
        (p.fill_mode == 1) ? ACL_SPARSE_FILL_MODE_UPPER : ACL_SPARSE_FILL_MODE_LOWER;
    const aclsparseDiagType_t diag =
        (p.diag_type == 1) ? ACL_SPARSE_DIAG_TYPE_UNIT : ACL_SPARSE_DIAG_TYPE_NON_UNIT;

    // 1. NPU 侧执行：Create -> Get 默认值 -> Set(fill/diag) -> Get -> 二次 Get
    auto r = SpMatGetAttributeNpu(env_->stream(), *entry, p.idx_base, fill, diag);

    ASSERT_EQ(r.createRet, ACL_SPARSE_STATUS_SUCCESS) << "create " << p.entry << " failed";

    // 2. 新建描述符（未 Set）默认值：fill=LOWER、diag=NON_UNIT
    ASSERT_EQ(r.getDefaultFillRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.defaultFill, ACL_SPARSE_FILL_MODE_LOWER) << "default fillMode should be LOWER";
    ASSERT_EQ(r.getDefaultDiagRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.defaultDiag, ACL_SPARSE_DIAG_TYPE_NON_UNIT) << "default diagType should be NON_UNIT";

    // 3. Set roundtrip 回读与传入一致（EXACT 零容差）
    ASSERT_EQ(r.setFillRet, ACL_SPARSE_STATUS_SUCCESS) << "SetAttribute(FILL_MODE) failed";
    ASSERT_EQ(r.setDiagRet, ACL_SPARSE_STATUS_SUCCESS) << "SetAttribute(DIAG_TYPE) failed";
    ASSERT_EQ(r.getFillRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.outFill, fill) << "fillMode roundtrip mismatch";
    ASSERT_EQ(r.getDiagRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.outDiag, diag) << "diagType roundtrip mismatch";

    // 4. 二次 Get 幂等：同一描述符重复读取结果一致
    ASSERT_EQ(r.getFillAgainRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.outFillAgain, fill) << "repeated get fillMode not idempotent";
    ASSERT_EQ(r.getDiagAgainRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.outDiagAgain, diag) << "repeated get diagType not idempotent";

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    SpMatGetAttributeCases,
    SpMatGetAttributeTest,
    testing::ValuesIn(GetCasesFromCsvChecked<SpMatGetAttributeParam>(
        "spmat_get_attribute_test.csv", 12)),
    [](const testing::TestParamInfo<SpMatGetAttributeParam> &info) { return info.param.case_name; });

// ============================================================================
// Semantics test fixture: SpMatGetAttributeSemanticsTest
// 语义/活性用例：默认值分支、const 签名路径、属性隔离与 latest-wins、失败 Set
// 状态保持。
// ============================================================================
class SpMatGetAttributeSemanticsTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    std::vector<int32_t> hostIdx_{0, 1, 2, 3, 4};
    std::vector<float> hostVal_{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};

    void SetUp() override {}
};

// GA-S01: 新建 csr 描述符（未 Set）Get 默认值：fill==LOWER、diag==NON_UNIT
TEST_F(SpMatGetAttributeSemanticsTest, DefaultAttributesOnFreshDescr)
{
    auto ptrs = DeviceBuffer::copyFrom(hostIdx_.data(), hostIdx_.size() * sizeof(int32_t));
    auto idxs = DeviceBuffer::copyFrom(hostIdx_.data(), hostIdx_.size() * sizeof(int32_t));
    auto vals = DeviceBuffer::copyFrom(hostVal_.data(), hostVal_.size() * sizeof(float));
    SpMatManager mat = SpMatManager::createCsr(4, 4, 5, ptrs.get(), idxs.get(), vals.get());

    int fm = kSentinelEnum;
    int dt = kSentinelEnum;
    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_FILL_MODE, &fm, sizeof(fm)),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(fm, static_cast<int>(ACL_SPARSE_FILL_MODE_LOWER));
    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_DIAG_TYPE, &dt, sizeof(dt)),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(dt, static_cast<int>(ACL_SPARSE_DIAG_TYPE_NON_UNIT));
}

// GA-S02: const_csr / const_bsr 描述符 GetAttribute：返回 SUCCESS 且默认值正确
//（const 签名路径；Set 步骤被 wrapper 跳过，Set 哨兵保持 NOT_INITIALIZED）
TEST_F(SpMatGetAttributeSemanticsTest, ConstDescrGetDefaults)
{
    for (const char *entryName : {"const_csr", "const_bsr"}) {
        const FormatEntry *entry = FindEntry(kEntries, entryName);
        ASSERT_NE(entry, nullptr) << "unknown entry: " << entryName;
        auto r = SpMatGetAttributeNpu(env_->stream(), *entry, 0,
                                      ACL_SPARSE_FILL_MODE_LOWER, ACL_SPARSE_DIAG_TYPE_NON_UNIT);
        std::cout << "---- " << entryName << " ----\n";
        ASSERT_EQ(r.createRet, ACL_SPARSE_STATUS_SUCCESS) << "create " << entryName << " failed";
        ASSERT_EQ(r.getDefaultFillRet, ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(r.defaultFill, ACL_SPARSE_FILL_MODE_LOWER) << entryName << " default fill";
        ASSERT_EQ(r.getDefaultDiagRet, ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(r.defaultDiag, ACL_SPARSE_DIAG_TYPE_NON_UNIT) << entryName << " default diag";
        // const 入口不执行 Set（SetAttribute 需要 mutable 句柄）
        EXPECT_EQ(r.setFillRet, ACL_SPARSE_STATUS_NOT_INITIALIZED);
        EXPECT_EQ(r.setDiagRet, ACL_SPARSE_STATUS_NOT_INITIALIZED);
        // Get 路径不受影响：回读仍为默认值且幂等
        ASSERT_EQ(r.getFillRet, ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(r.outFill, ACL_SPARSE_FILL_MODE_LOWER);
        ASSERT_EQ(r.getDiagRet, ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(r.outDiag, ACL_SPARSE_DIAG_TYPE_NON_UNIT);
        ASSERT_EQ(r.getFillAgainRet, ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(r.outFillAgain, ACL_SPARSE_FILL_MODE_LOWER);
        ASSERT_EQ(r.getDiagAgainRet, ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(r.outDiagAgain, ACL_SPARSE_DIAG_TYPE_NON_UNIT);
    }
}

// GA-S03: 属性隔离（Set fill 不影响 diag）与覆盖写 latest-wins
TEST_F(SpMatGetAttributeSemanticsTest, AttributeIsolationAndOverwrite)
{
    auto ptrs = DeviceBuffer::copyFrom(hostIdx_.data(), hostIdx_.size() * sizeof(int32_t));
    auto idxs = DeviceBuffer::copyFrom(hostIdx_.data(), hostIdx_.size() * sizeof(int32_t));
    auto vals = DeviceBuffer::copyFrom(hostVal_.data(), hostVal_.size() * sizeof(float));
    SpMatManager mat = SpMatManager::createCsr(4, 4, 5, ptrs.get(), idxs.get(), vals.get());

    // 隔离：Set(fill=UPPER) 后 Get(diag) 应保持默认 NON_UNIT 不受影响
    const aclsparseFillMode_t upper = ACL_SPARSE_FILL_MODE_UPPER;
    ASSERT_EQ(aclsparseSpMatSetAttribute(mat.get(), ACL_SPARSE_SPMAT_FILL_MODE, &upper,
                                         sizeof(upper)),
              ACL_SPARSE_STATUS_SUCCESS);
    int dt = kSentinelEnum;
    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_DIAG_TYPE, &dt, sizeof(dt)),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(dt, static_cast<int>(ACL_SPARSE_DIAG_TYPE_NON_UNIT)) << "Set(fill) should not touch diagType";

    // 覆盖写 latest-wins：Set(UPPER) -> Set(LOWER) -> Get == LOWER
    const aclsparseFillMode_t lower = ACL_SPARSE_FILL_MODE_LOWER;
    ASSERT_EQ(aclsparseSpMatSetAttribute(mat.get(), ACL_SPARSE_SPMAT_FILL_MODE, &lower,
                                          sizeof(lower)),
              ACL_SPARSE_STATUS_SUCCESS);
    int fm = kSentinelEnum;
    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_FILL_MODE, &fm, sizeof(fm)),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(fm, static_cast<int>(ACL_SPARSE_FILL_MODE_LOWER)) << "latest Set should win";
}

// GA-S04: Set(fill=99) 值域拒绝后状态保持：返回 INVALID_VALUE，Get 回读旧值不变
//（值域拒绝的返回码断言由 test/spsm 已覆盖，此处复用该调用作前置条件，
// 新增"失败不改状态"断言——不同断言目标）
TEST_F(SpMatGetAttributeSemanticsTest, FailedSetKeepsOldValue)
{
    auto ptrs = DeviceBuffer::copyFrom(hostIdx_.data(), hostIdx_.size() * sizeof(int32_t));
    auto idxs = DeviceBuffer::copyFrom(hostIdx_.data(), hostIdx_.size() * sizeof(int32_t));
    auto vals = DeviceBuffer::copyFrom(hostVal_.data(), hostVal_.size() * sizeof(float));
    SpMatManager mat = SpMatManager::createCsr(4, 4, 5, ptrs.get(), idxs.get(), vals.get());

    // 前置：写入一个非默认值，使"旧值"可区分于默认值
    const aclsparseFillMode_t upper = ACL_SPARSE_FILL_MODE_UPPER;
    ASSERT_EQ(aclsparseSpMatSetAttribute(mat.get(), ACL_SPARSE_SPMAT_FILL_MODE, &upper,
                                         sizeof(upper)),
              ACL_SPARSE_STATUS_SUCCESS);

    // 非法值 99 -> INVALID_VALUE（spsm 已覆盖的返回码语义，复用作前置；
    // 越界枚举注入值经 char 逐字节位重解释构造，规避 static_cast UB——CS-1.3）
    const aclsparseFillMode_t badFill = EnumFromRawInt<aclsparseFillMode_t>(kSentinelEnum);
    EXPECT_EQ(aclsparseSpMatSetAttribute(mat.get(), ACL_SPARSE_SPMAT_FILL_MODE, &badFill,
                                         sizeof(badFill)),
              ACL_SPARSE_STATUS_INVALID_VALUE);

    // 失败不改状态：Get 仍回读旧值 UPPER（int 承载按 int 比较）
    int fm = kSentinelEnum;
    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_FILL_MODE, &fm, sizeof(fm)),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(fm, static_cast<int>(ACL_SPARSE_FILL_MODE_UPPER)) << "failed Set should keep the old value";
}

// ============================================================================
// Exception test fixture: SpMatGetAttributeExceptionTest
// 错误路径用例：断言错误码返回，且 OUT 哨兵值保持不变（未被污染）。
// ============================================================================
class SpMatGetAttributeExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    // Get/SetAttribute 均不解引用设备数据，描述符只借用指针：host 占位缓冲即可
    //（同 blockedell_get 异常夹具做法）；static 保证指针在描述符生命周期内有效。
    SpMatManager CreateCsr4x4()
    {
        static std::vector<int32_t> rowOff(16, 0);
        static std::vector<int32_t> colInd(16, 0);
        static std::vector<float> values(32, 1.0F);
        return SpMatManager::createCsr(4, 4, 5, rowOff.data(), colInd.data(), values.data());
    }

    void SetUp() override {}
};

// GA-E01: Get(nullptr descr, ...) -> INVALID_VALUE，出参不被污染
TEST_F(SpMatGetAttributeExceptionTest, NullSpMatDescr)
{
    int fm = kSentinelEnum;
    EXPECT_EQ(aclsparseSpMatGetAttribute(nullptr, ACL_SPARSE_SPMAT_FILL_MODE, &fm, sizeof(fm)),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(fm, kSentinelEnum) << "error path polluted fillMode";

    int dt = kSentinelEnum;
    EXPECT_EQ(aclsparseSpMatGetAttribute(nullptr, ACL_SPARSE_SPMAT_DIAG_TYPE, &dt, sizeof(dt)),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(dt, kSentinelEnum) << "error path polluted diagType";
}

// GA-E02: Get(valid, DIAG_TYPE, data=null, 4) -> INVALID_VALUE
TEST_F(SpMatGetAttributeExceptionTest, NullDataPointer)
{
    SpMatManager mat = CreateCsr4x4();
    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_DIAG_TYPE, nullptr,
                                         sizeof(aclsparseDiagType_t)),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_FILL_MODE, nullptr,
                                         sizeof(aclsparseFillMode_t)),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// GA-E03: dataSize=8（FILL_MODE）/ 2（DIAG_TYPE）-> INVALID_VALUE，出参哨兵不变
TEST_F(SpMatGetAttributeExceptionTest, WrongDataSize)
{
    SpMatManager mat = CreateCsr4x4();
    int fm = kSentinelEnum;
    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_FILL_MODE, &fm,
                                         sizeof(aclsparseFillMode_t) * 2),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(fm, kSentinelEnum) << "error path polluted fillMode";

    int dt = kSentinelEnum;
    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_DIAG_TYPE, &dt, 2),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(dt, kSentinelEnum) << "error path polluted diagType";
}

// GA-E04: data 错位 1 字节（alignas(8) 缓冲错位）-> INVALID_VALUE，传入的出参
// 缓冲哨兵字节不变：buf 预置 0xA5，两次错位指针调用后 std::equal 全 16 字节
// 逐位比对（哨兵断言校验真正传入 API 的 misaligned 所在缓冲，可发现误写出参；
// 哨兵预置与比对用 std::fill / std::equal 安全等价实现，规避 memset/memcmp）
TEST_F(SpMatGetAttributeExceptionTest, MisalignedDataPointer)
{
    SpMatManager mat = CreateCsr4x4();
    alignas(8) unsigned char buf[16];
    unsigned char expect[16];
    std::fill(buf, buf + sizeof(buf), 0xA5);        // 传入 API 的出参缓冲预置哨兵字节
    std::fill(expect, expect + sizeof(expect), 0xA5);  // 哨兵基线
    void *misaligned = buf + 1;  // 地址 % 4 == 1，不满足枚举对齐要求

    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_FILL_MODE, misaligned,
                                         sizeof(aclsparseFillMode_t)),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_DIAG_TYPE, misaligned,
                                         sizeof(aclsparseDiagType_t)),
              ACL_SPARSE_STATUS_INVALID_VALUE);

    // 出参哨兵不变：misaligned（buf+1）指向 buf 内部，整 16 字节未被改写即证明
    // 错误路径未向传入的出参缓冲写入任何字节
    EXPECT_TRUE(std::equal(buf, buf + sizeof(buf), expect)) << "misaligned OUT buffer polluted";
}

// GA-E05: attribute=2/99/-1（越界枚举经 char 逐字节位重解释构造）-> INVALID_VALUE，出参哨兵不变
TEST_F(SpMatGetAttributeExceptionTest, InvalidAttributeEnum)
{
    SpMatManager mat = CreateCsr4x4();
    for (int raw : {2, 99, -1}) {
        const auto badAttr = EnumFromRawInt<aclsparseSpMatAttribute_t>(raw);
        int v = kSentinelEnum;
        EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), badAttr, &v, sizeof(v)),
                  ACL_SPARSE_STATUS_INVALID_VALUE)
            << "attribute=" << raw;
        EXPECT_EQ(v, kSentinelEnum) << "attribute=" << raw << " polluted OUT";
    }
}

// GA-E06: Set 侧共享守卫抽样（ValidateAttributeAccess 的 Set 侧对称性）：
// Set(null descr) / Set(null data) / Set(wrong size) / Set(非法 attribute) 均 INVALID_VALUE
TEST_F(SpMatGetAttributeExceptionTest, SetSymmetricGuards)
{
    const aclsparseFillMode_t fm = ACL_SPARSE_FILL_MODE_LOWER;
    const aclsparseDiagType_t dt = ACL_SPARSE_DIAG_TYPE_NON_UNIT;

    // Set(null descr, ...)
    EXPECT_EQ(aclsparseSpMatSetAttribute(nullptr, ACL_SPARSE_SPMAT_FILL_MODE, &fm, sizeof(fm)),
              ACL_SPARSE_STATUS_INVALID_VALUE);

    SpMatManager mat = CreateCsr4x4();

    // Set(valid, FILL_MODE, data=null, 4)
    EXPECT_EQ(aclsparseSpMatSetAttribute(mat.get(), ACL_SPARSE_SPMAT_FILL_MODE, nullptr,
                                         sizeof(fm)),
              ACL_SPARSE_STATUS_INVALID_VALUE);

    // Set(valid, DIAG_TYPE, &dt, 8)（错误 dataSize）
    EXPECT_EQ(aclsparseSpMatSetAttribute(mat.get(), ACL_SPARSE_SPMAT_DIAG_TYPE, &dt,
                                         sizeof(dt) * 2),
              ACL_SPARSE_STATUS_INVALID_VALUE);

    // Set(valid, 2, &fm, 4)（非法 attribute 枚举；越界值经 char 逐字节位重解释构造——CS-1.3）
    const auto badAttr = EnumFromRawInt<aclsparseSpMatAttribute_t>(2);
    EXPECT_EQ(aclsparseSpMatSetAttribute(mat.get(), badAttr, &fm, sizeof(fm)),
              ACL_SPARSE_STATUS_INVALID_VALUE);

    // 守卫拒绝后描述符状态不被污染：默认值仍可正常回读（int 承载按 int 比较）
    int outFm = kSentinelEnum;
    EXPECT_EQ(aclsparseSpMatGetAttribute(mat.cget(), ACL_SPARSE_SPMAT_FILL_MODE, &outFm,
                                         sizeof(outFm)),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(outFm, static_cast<int>(ACL_SPARSE_FILL_MODE_LOWER)) << "rejected Set should keep the old value";
}
