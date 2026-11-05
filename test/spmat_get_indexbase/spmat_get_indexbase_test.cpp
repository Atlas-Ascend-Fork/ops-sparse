/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file spmat_get_indexbase_test.cpp
 * @brief GTest tests for aclsparseSpMatGetIndexBase（arch 无关，纯 Host 接口 UT）.
 *
 * 对标 cuSPARSE cusparseSpMatGetIndexBase：读取稀疏矩阵描述符创建时传入的
 * idxBase。GetIndexBase 是纯 Host 侧描述符访问器，不启动 NPU kernel；测试侧
 * 建设备缓冲存入描述符，Get 读回后断言与 Create 传入一致。
 *
 *   - TEST_P (SpMatGetIndexBaseTest)          : spmat_get_indexbase_test.csv 驱动，
 *                                               11 个创建入口（6 mutable + 5 const
 *                                               变体，blocked_ell 无 const 变体）×
 *                                               base{ZERO,ONE} = 22 用例的
 *                                               Create-Get 往返 + 二次 Get 幂等断言
 *   - TEST_F (SpMatGetIndexBaseLivenessTest)  : CsrSetPointers/SpMatSetValues 后
 *                                               idxBase 保持不变（Set 系不触碰）
 *   - TEST_F (SpMatGetIndexBaseExceptionTest) : null descr / null out（容忍口径与
 *                                               aclsparseSpMatGetFormat、aclsparseSpVecGetIndexBase
 *                                               一致）/ 签名不符（SpVec 描述符误传）
 */

#include <cstring>

#include "test_common.h"

#include "spmat_get_indexbase_param.h"

using namespace sparse_test;

namespace {

class MutableSpMatGuard {
public:
    ~MutableSpMatGuard()
    {
        if (p_ != nullptr) {
            (void)aclsparseDestroySpMat(p_);
        }
    }
    aclsparseSpMatDescr_t *put()
    {
        if (p_ != nullptr) {
            (void)aclsparseDestroySpMat(p_);
            p_ = nullptr;
        }
        return &p_;
    }
    aclsparseSpMatDescr_t get() const { return p_; }
    aclsparseConstSpMatDescr_t cget() const { return p_; }

private:
    aclsparseSpMatDescr_t p_ = nullptr;
};

class ConstSpMatGuard {
public:
    ~ConstSpMatGuard()
    {
        if (p_ != nullptr) {
            (void)aclsparseDestroySpMat(p_);
        }
    }
    aclsparseConstSpMatDescr_t *put()
    {
        if (p_ != nullptr) {
            (void)aclsparseDestroySpMat(p_);
            p_ = nullptr;
        }
        return &p_;
    }
    aclsparseConstSpMatDescr_t get() const { return p_; }

private:
    aclsparseConstSpMatDescr_t p_ = nullptr;
};

struct FormatEntry {
    const char *name;
    bool isConst;
};

// 11 个带 idxBase 入参的 SpMat 创建入口注册表（CSV entry 列引用此处的 name，
// isConst 决定 mutable/const 创建路径；blocked_ell 无 const 变体）
const FormatEntry kEntries[] = {
    {"csr", false},           {"const_csr", true},
    {"csc", false},           {"const_csc", true},
    {"coo", false},           {"const_coo", true},
    {"bsr", false},           {"const_bsr", true},
    {"blocked_ell", false},   {"sliced_ell", false},
    {"const_sliced_ell", true},
};

// CSV entry 名 -> 注册表查找；未注册名返回 nullptr（测试侧 ASSERT 拦截）
const FormatEntry *FindEntry(const std::string &name)
{
    for (const auto &e : kEntries) {
        if (name == e.name) {
            return &e;
        }
    }
    return nullptr;
}

// Create/GetIndexBase 均不解引用设备数据，缓冲内容任意，仅要求指针有效
constexpr int64_t kRows = 4;
constexpr int64_t kCols = 4;
constexpr int64_t kNnz = 5;

// 统一 mutable 创建模板：Create 成功/失败都把句柄挂到 guard 上
template <typename CreateFn, typename... Args>
aclsparseStatus_t CreateMutableSpMat(ConstSpMatGuard *guard, CreateFn create, Args... args)
{
    aclsparseSpMatDescr_t descr = nullptr;
    const aclsparseStatus_t status = create(&descr, args...);
    *guard->put() = descr;
    return status;
}

aclsparseStatus_t CreateMutableEntry(const FormatEntry &entry, aclsparseIndexBase_t base,
                                     void *ptrs, void *idxs, void *vals, ConstSpMatGuard *guard)
{
    if (strcmp(entry.name, "csr") == 0) {
        return CreateMutableSpMat(guard, aclsparseCreateCsr, kRows, kCols, kNnz, ptrs, idxs,
            vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    if (strcmp(entry.name, "csc") == 0) {
        return CreateMutableSpMat(guard, aclsparseCreateCsc, kRows, kCols, kNnz, ptrs, idxs,
            vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    if (strcmp(entry.name, "coo") == 0) {
        return CreateMutableSpMat(guard, aclsparseCreateCoo, kRows, kCols, kNnz, ptrs, idxs,
            vals, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    if (strcmp(entry.name, "bsr") == 0) {
        return CreateMutableSpMat(guard, aclsparseCreateBsr, 2, 2, 2, 2, 2, ptrs, idxs, vals,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    }
    if (strcmp(entry.name, "blocked_ell") == 0) {
        return CreateMutableSpMat(guard, aclsparseCreateBlockedEll, kRows, kCols, 2, 2, idxs,
            vals, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    if (strcmp(entry.name, "sliced_ell") == 0) {
        return CreateMutableSpMat(guard, aclsparseCreateSlicedEll, kRows, kCols, 4, 2, 2, ptrs,
            idxs, vals, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    return ACL_SPARSE_STATUS_INVALID_VALUE;
}

aclsparseStatus_t CreateConstEntry(const FormatEntry &entry, aclsparseIndexBase_t base,
                                   const void *ptrs, const void *idxs, const void *vals,
                                   ConstSpMatGuard *guard)
{
    if (strcmp(entry.name, "const_csr") == 0) {
        return aclsparseCreateConstCsr(guard->put(), kRows, kCols, kNnz, ptrs, idxs, vals,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    if (strcmp(entry.name, "const_csc") == 0) {
        return aclsparseCreateConstCsc(guard->put(), kRows, kCols, kNnz, ptrs, idxs, vals,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    if (strcmp(entry.name, "const_coo") == 0) {
        return aclsparseCreateConstCoo(guard->put(), kRows, kCols, kNnz, ptrs, idxs, vals,
            ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    if (strcmp(entry.name, "const_bsr") == 0) {
        return aclsparseCreateConstBsr(guard->put(), 2, 2, 2, 2, 2, ptrs, idxs, vals,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    }
    if (strcmp(entry.name, "const_sliced_ell") == 0) {
        return aclsparseCreateConstSlicedEll(guard->put(), kRows, kCols, 4, 2, 2, ptrs, idxs,
            vals, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    return ACL_SPARSE_STATUS_INVALID_VALUE;
}

} // namespace

// ============================================================================
// GTest parameterized fixture: SpMatGetIndexBaseTest（CSV 驱动）
// ============================================================================
class SpMatGetIndexBaseTest : public testing::TestWithParam<SpMatGetIndexBaseParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// ============================================================================
// Test body: success-path parameterized test（每条 CSV 行一个用例）
// ============================================================================
TEST_P(SpMatGetIndexBaseTest, IndexBaseRoundTrip)
{
    const auto &p = GetParam();
    std::cout << "==== " << p.case_name << " ==== entry=" << p.entry
              << " idx_base=" << p.idx_base << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    const FormatEntry *entry = FindEntry(p.entry);
    ASSERT_NE(entry, nullptr) << "unknown entry in CSV: " << p.entry;
    const aclsparseIndexBase_t base =
        (p.idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    // Create/GetIndexBase 均不解引用设备数据，缓冲内容任意，仅要求指针有效
    const std::vector<int32_t> hostIdx(16, 0);
    const std::vector<float> hostVal(32, 1.0F);
    auto ptrs = DeviceBuffer::copyFrom(hostIdx.data(), hostIdx.size() * sizeof(int32_t));
    auto idxs = DeviceBuffer::copyFrom(hostIdx.data(), hostIdx.size() * sizeof(int32_t));
    auto vals = DeviceBuffer::copyFrom(hostVal.data(), hostVal.size() * sizeof(float));

    ConstSpMatGuard descr;
    const aclsparseStatus_t createRet =
        entry->isConst ? CreateConstEntry(*entry, base, ptrs.get(), idxs.get(), vals.get(), &descr)
                       : CreateMutableEntry(*entry, base, ptrs.get(), idxs.get(), vals.get(), &descr);
    ASSERT_EQ(createRet, ACL_SPARSE_STATUS_SUCCESS)
        << "create " << entry->name << " failed";

    aclsparseIndexBase_t got = static_cast<aclsparseIndexBase_t>(-1);
    EXPECT_EQ(aclsparseSpMatGetIndexBase(descr.get(), &got), ACL_SPARSE_STATUS_SUCCESS)
        << "aclsparseSpMatGetIndexBase(" << entry->name << ") failed";
    EXPECT_EQ(got, base) << entry->name << " index base mismatch";

    // 二次 Get 幂等：同一描述符重复读取结果一致（G1）
    aclsparseIndexBase_t again = static_cast<aclsparseIndexBase_t>(-1);
    EXPECT_EQ(aclsparseSpMatGetIndexBase(descr.get(), &again), ACL_SPARSE_STATUS_SUCCESS)
        << "second aclsparseSpMatGetIndexBase(" << entry->name << ") failed";
    EXPECT_EQ(again, base) << entry->name << " repeated get not idempotent";
}

INSTANTIATE_TEST_SUITE_P(
    SpMatGetIndexBaseCases,
    SpMatGetIndexBaseTest,
    testing::ValuesIn(GetCasesFromCsv<SpMatGetIndexBaseParam>("spmat_get_indexbase_test.csv")),
    [](const testing::TestParamInfo<SpMatGetIndexBaseParam> &info) { return info.param.case_name; });

// ============================================================================
// Liveness test fixture: Set 系列不改变 idxBase（G1 黑盒活性）
// ============================================================================
class SpMatGetIndexBaseLivenessTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// L1: CreateCsr(base=ONE) -> Get == ONE；CsrSetPointers 换指针后 Get 仍 == ONE；
//     SpMatSetValues 换 values 后 Get 仍 == ONE（黑盒钉住 idxBase/baseType 仅由
//     Create 写入，Set 系列与缓存失效逻辑不触碰该字段）
TEST_F(SpMatGetIndexBaseLivenessTest, SetPointersAndValuesKeepIndexBase)
{
    const std::vector<int32_t> hostIdx(16, 0);
    const std::vector<float> hostVal(32, 1.0F);
    auto ptrs = DeviceBuffer::copyFrom(hostIdx.data(), hostIdx.size() * sizeof(int32_t));
    auto idxs = DeviceBuffer::copyFrom(hostIdx.data(), hostIdx.size() * sizeof(int32_t));
    auto vals = DeviceBuffer::copyFrom(hostVal.data(), hostVal.size() * sizeof(float));

    MutableSpMatGuard descr;
    ASSERT_EQ(aclsparseCreateCsr(descr.put(), kRows, kCols, kNnz, ptrs.get(), idxs.get(),
                                 vals.get(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ONE, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseIndexBase_t base = ACL_SPARSE_INDEX_BASE_ZERO;
    ASSERT_EQ(aclsparseSpMatGetIndexBase(descr.cget(), &base), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(base, ACL_SPARSE_INDEX_BASE_ONE);

    auto ptrs2 = DeviceBuffer::copyFrom(hostIdx.data(), hostIdx.size() * sizeof(int32_t));
    auto idxs2 = DeviceBuffer::copyFrom(hostIdx.data(), hostIdx.size() * sizeof(int32_t));
    auto vals2 = DeviceBuffer::copyFrom(hostVal.data(), hostVal.size() * sizeof(float));
    ASSERT_EQ(aclsparseCsrSetPointers(descr.get(), ptrs2.get(), idxs2.get(), vals2.get()),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseSpMatGetIndexBase(descr.cget(), &base), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(base, ACL_SPARSE_INDEX_BASE_ONE) << "CsrSetPointers changed idxBase";

    ASSERT_EQ(aclsparseSpMatSetValues(descr.get(), vals2.get()), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseSpMatGetIndexBase(descr.cget(), &base), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(base, ACL_SPARSE_INDEX_BASE_ONE) << "SpMatSetValues changed idxBase";
}

// ============================================================================
// Exception test fixture: SpMatGetIndexBaseExceptionTest
// ============================================================================
class SpMatGetIndexBaseExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    const std::vector<int32_t> hostIdx_{0, 1, 2, 3, 4};
    const std::vector<float> hostVal_{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
};

// E1: nullptr 描述符 -> ACL_SPARSE_STATUS_INVALID_VALUE（同 cusparse 语义），
//     且错误路径不回写 OUT 出参
TEST_F(SpMatGetIndexBaseExceptionTest, NullSpMatDescr)
{
    aclsparseIndexBase_t base = ACL_SPARSE_INDEX_BASE_ZERO;
    EXPECT_EQ(aclsparseSpMatGetIndexBase(nullptr, &base), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(base, ACL_SPARSE_INDEX_BASE_ZERO) << "error path polluted idxBase";
}

// E2: 合法描述符 + nullptr 输出指针 -> SUCCESS 且不回写
//     （容忍口径与 aclsparseSpMatGetFormat / aclsparseSpVecGetIndexBase 一致）
TEST_F(SpMatGetIndexBaseExceptionTest, NullOutPointer)
{
    auto ptrs = DeviceBuffer::copyFrom(hostIdx_.data(), hostIdx_.size() * sizeof(int32_t));
    auto idxs = DeviceBuffer::copyFrom(hostIdx_.data(), hostIdx_.size() * sizeof(int32_t));
    auto vals = DeviceBuffer::copyFrom(hostVal_.data(), hostVal_.size() * sizeof(float));
    MutableSpMatGuard descr;
    ASSERT_EQ(aclsparseCreateCsr(descr.put(), kRows, kCols, kNnz, ptrs.get(), idxs.get(), vals.get(),
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ONE, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseSpMatGetIndexBase(descr.cget(), nullptr), ACL_SPARSE_STATUS_SUCCESS);
}

// E3: 非 SpMat 描述符（SpVec 签名不符）-> ACL_SPARSE_STATUS_INVALID_VALUE
//     注：误传句柄后 API 仅读取 offset-0 的 signature 字段即短路返回，不再访问
//     其他字段；本用例仅以 offset-0 布局为前提，禁止扩展为读取更多字段。
TEST_F(SpMatGetIndexBaseExceptionTest, WrongDescriptorType)
{
    auto idxs = DeviceBuffer::copyFrom(hostIdx_.data(), hostIdx_.size() * sizeof(int32_t));
    auto vals = DeviceBuffer::copyFrom(hostVal_.data(), hostVal_.size() * sizeof(float));
    auto spVec = SpVecManager::create(5, 5, idxs.get(), vals.get(), ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    aclsparseIndexBase_t base = ACL_SPARSE_INDEX_BASE_ZERO;
    EXPECT_EQ(aclsparseSpMatGetIndexBase(
                  reinterpret_cast<aclsparseConstSpMatDescr_t>(spVec.get()), &base),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(base, ACL_SPARSE_INDEX_BASE_ZERO) << "error path polluted idxBase";
}

// 覆盖说明：Destroy 后的悬垂句柄不再单独设用例——Destroy 置零签名后底层内存已
// 释放，访问属 use-after-free UB；其防御分支（签名不符 -> INVALID_VALUE）已由
// E1（null 句柄）与 E3（SpVec 句柄）等价覆盖。
