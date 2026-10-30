/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * @file matmulDescriptorInit_test.cpp
 * @brief GTest 入口：TEST_P CSV 驱动 + TEST_F nullptr 异常 / Destroy 边界。
 *
 * 依据：分级用例设计、TEST_F/TEST_P 分工、字段断言策略。
 *
 * 本接口为 Host 侧描述符初始化，无数值计算、无 golden。验证目标：
 *   ① 合法入参返回 SUCCESS 且描述符字段被正确填充（白盒访问 internal.h）
 *   ② 非法入参返回需求 2.9 规定的错误码
 *
 * TEST_P：承载所有可用 CSV 行表达的用例（正常/约束违规/数据类型覆盖）。
 * TEST_F：承载纯指针异常（handle/output nullptr、*非空覆盖、mat nullptr）+
 * Destroy 边界。
 *
 * 入口由 test/frame/test_main.cpp 提供，本文件禁止定义 main()。
 */

#include "cann_ops_sparseLt.h"

#include "csv_loader.h" // sparse_test::GetCasesFromCsv
#include "matmulDescriptorInit_npu_wrapper.h"
#include "matmulDescriptorInit_param.h"
#include "sparse_test.h" // sparse_test::AclEnvScope

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>

// ============================================================================
// 全局 ACL 环境单例：TEST_P 与 TEST_F 共享，避免 aclInit 多次调用失败
// ============================================================================
class AclEnvManager
{
  public:
    static sparse_test::AclEnvScope& Get()
    {
        std::call_once(once_, []() { instance_ = std::make_unique<sparse_test::AclEnvScope>(); });
        return *instance_;
    }

  private:
    static std::once_flag once_;
    static std::unique_ptr<sparse_test::AclEnvScope> instance_;
};
std::once_flag AclEnvManager::once_;
std::unique_ptr<sparse_test::AclEnvScope> AclEnvManager::instance_;

// ============================================================================
// 标准 matmul 输入 helper：构造合法的 handle + matA(structured) +
// matB/matC/matD(dense)
// + 可选的 matmul 描述符 Init。消除 TEST_F 中的重复 setup 代码。
// 所有矩阵均为 16x16 FP32 COL，sparsity=50_PERCENT。
// ============================================================================
struct StdMatmulInput
{
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::MatDescrGuard matA, matB, matC, matD;
    sparse_test::MatmulDescrGuard mm;
};

// 构造标准合法输入。initMatmul=true 时同时调用 MatmulDescriptorInit。
// 返回 false 表示前置条件失败（已通过 EXPECT_EQ 记录），调用方应中止测试。
static bool InitStdMatmulInput(StdMatmulInput& in, bool initMatmul = true)
{
    EXPECT_EQ(in.handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    if (in.handle.status() != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    EXPECT_EQ(in.matA.initStructured(in.handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL,
                                     ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    if (in.matA.get() == nullptr) { return false; }
    EXPECT_EQ(in.matB.initDense(in.handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL),
              ACL_SPARSE_STATUS_SUCCESS);
    if (in.matB.get() == nullptr) { return false; }
    EXPECT_EQ(in.matC.initDense(in.handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL),
              ACL_SPARSE_STATUS_SUCCESS);
    if (in.matC.get() == nullptr) { return false; }
    EXPECT_EQ(in.matD.initDense(in.handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL),
              ACL_SPARSE_STATUS_SUCCESS);
    if (in.matD.get() == nullptr) { return false; }
    if (initMatmul)
    {
        EXPECT_EQ(in.mm.init(in.handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, in.matA.ptr(),
                             in.matB.ptr(), in.matC.ptr(), in.matD.ptr(), ACL_SPARSE_COMPUTE_32F),
                  ACL_SPARSE_STATUS_SUCCESS);
        if (in.mm.get() == nullptr) { return false; }
    }
    return true;
}

// ============================================================================
// TEST_P：CSV 驱动参数化用例（L0 正向 / L1 边界约束 / L2 dtype 覆盖）
// ============================================================================
class MatmulDescriptorInitTest : public testing::TestWithParam<sparse_test::MatmulDescriptorInitParam>
{
  public:
    static void SetUpTestSuite() { (void)AclEnvManager::Get(); }
    static void TearDownTestSuite() {}

  protected:
    sparse_test::MatmulDescriptorInitParam param_;
    void SetUp() override { param_ = GetParam(); }
};

// ============================================================================
// CsvDriven 分支辅助函数：将 dense/structured/matmul 三类用例的执行逻辑提取为
// 独立函数，控制 TEST_P 的单函数行数。
// ============================================================================

// CsvDriven 用例的 dense 分支
static void RunDenseCase(const sparse_test::SparseLtHandleGuard& handle,
                         const sparse_test::MatmulDescriptorInitParam& p, aclsparseStatus_t expect)
{
    sparse_test::MatDescrGuard mat;
    auto ret = mat.initDense(handle.ptr(), p.a.rows, p.a.cols, p.a.ld, p.a.align, p.a.dtype, p.a.order);
    EXPECT_EQ(ret, expect) << p.case_name;
    if (p.verifyFields && ret == ACL_SPARSE_STATUS_SUCCESS)
    {
        if (!sparse_test::VerifyMatFields(mat.get(), p.a)) { return; }
    }
}

// CsvDriven 用例的 structured 分支
static void RunStructuredCase(const sparse_test::SparseLtHandleGuard& handle,
                              const sparse_test::MatmulDescriptorInitParam& p, aclsparseStatus_t expect)
{
    sparse_test::MatDescrGuard mat;
    auto ret = mat.initStructured(handle.ptr(), p.a.rows, p.a.cols, p.a.ld, p.a.align, p.a.dtype, p.a.order,
                                  p.a.sparsity);
    EXPECT_EQ(ret, expect) << p.case_name;
    if (p.verifyFields && ret == ACL_SPARSE_STATUS_SUCCESS)
    {
        if (!sparse_test::VerifyMatFields(mat.get(), p.a)) { return; }
    }
}

// CsvDriven 用例的 matmul 分支
// 返回 false 表示前置 InitMatFromParam 失败（已通过 EXPECT_EQ 记录），调用方应中止测试。
static bool RunMatmulCase(const sparse_test::SparseLtHandleGuard& handle,
                          const sparse_test::MatmulDescriptorInitParam& p, aclsparseStatus_t expect)
{
    sparse_test::MatDescrGuard matA, matB, matC, matD;
    EXPECT_EQ(sparse_test::InitMatFromParam(handle.ptr(), matA, p.a), ACL_SPARSE_STATUS_SUCCESS)
        << p.case_name << ": matA init failed";
    if (matA.get() == nullptr) { return false; }
    EXPECT_EQ(sparse_test::InitMatFromParam(handle.ptr(), matB, p.b), ACL_SPARSE_STATUS_SUCCESS)
        << p.case_name << ": matB init failed";
    if (matB.get() == nullptr) { return false; }
    EXPECT_EQ(sparse_test::InitMatFromParam(handle.ptr(), matC, p.c), ACL_SPARSE_STATUS_SUCCESS)
        << p.case_name << ": matC init failed";
    if (matC.get() == nullptr) { return false; }
    EXPECT_EQ(sparse_test::InitMatFromParam(handle.ptr(), matD, p.d), ACL_SPARSE_STATUS_SUCCESS)
        << p.case_name << ": matD init failed";
    if (matD.get() == nullptr) { return false; }

    sparse_test::MatmulDescrGuard mm;
    auto ret = mm.init(handle.ptr(), p.opA, p.opB, matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), p.computeType);
    EXPECT_EQ(ret, expect) << p.case_name;
    if (p.verifyFields && ret == ACL_SPARSE_STATUS_SUCCESS)
    {
        if (!sparse_test::VerifyMatmulFields(mm.get(), p.opA, p.opB, matA.get(), matB.get(), matC.get(), matD.get(),
                                             p.computeType)) { return false; }
    }
    return true;
}

TEST_P(MatmulDescriptorInitTest, CsvDriven)
{
    const auto& p = param_;

    // 创建库句柄
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS) << p.case_name << ": aclsparseLtInit failed";

    aclsparseStatus_t expect = sparse_test::ExpectedStatus(p.expectStatus);

    if (p.target == "dense")
    {
        RunDenseCase(handle, p, expect);
    }
    else if (p.target == "structured")
    {
        RunStructuredCase(handle, p, expect);
    }
    else if (p.target == "matmul")
    {
        if (!RunMatmulCase(handle, p, expect)) { return; }
    }
    else
    {
        FAIL() << p.case_name << ": unknown target '" << p.target << "'";
    }
}

INSTANTIATE_TEST_SUITE_P(MatmulDescriptorInitCases, MatmulDescriptorInitTest,
                         testing::ValuesIn(sparse_test::GetCasesFromCsv<sparse_test::MatmulDescriptorInitParam>(
                             "matmulDescriptorInit_test.csv")),
                         [](const testing::TestParamInfo<sparse_test::MatmulDescriptorInitParam>& info)
                         {
                             // GTest 参数名仅允许字母/数字/下划线，case_name 含 '-' 需替换
                             std::string name = info.param.case_name;
                             std::replace(name.begin(), name.end(), '-', '_');
                             return name;
                         });

// ============================================================================
// TEST_F：纯指针异常 + Destroy 边界 + ALLOC_FAILED 注入（可选）
// CSV 难以表达的纯指针异常走 TEST_F 硬编码
// ============================================================================

class MatmulDescriptorInitFixture : public testing::Test
{
  protected:
    void SetUp() override { (void)AclEnvManager::Get(); }
};

// ---- handle = nullptr 异常 ----
TEST_F(MatmulDescriptorInitFixture, DenseHandleNullptr)
{
    aclsparseLtMatDescriptor_t descr = nullptr;
    auto ret = aclsparseLtDenseDescriptorInit(nullptr, &descr, 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(MatmulDescriptorInitFixture, StructuredHandleNullptr)
{
    aclsparseLtMatDescriptor_t descr = nullptr;
    auto ret = aclsparseLtStructuredDescriptorInit(nullptr, &descr, 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL,
                                                   ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(MatmulDescriptorInitFixture, MatmulHandleNullptr)
{
    aclsparseLtMatmulDescriptor_t descr = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(nullptr, &descr, ACL_SPARSE_OP_NON_TRANSPOSE,
                                               ACL_SPARSE_OP_NON_TRANSPOSE, nullptr, nullptr, nullptr, nullptr,
                                               ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// ---- Dense 输出指针 nullptr / *非空覆盖 ----
TEST_F(MatmulDescriptorInitFixture, DenseMatDescrNullptr)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    auto ret = aclsparseLtDenseDescriptorInit(handle.ptr(), nullptr, 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(MatmulDescriptorInitFixture, DenseMatDescrNonEmpty)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard mat;
    ASSERT_EQ(mat.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    // *mat.ptr() 非空 → 覆盖已有 → INVALID_VALUE
    auto ret = aclsparseLtDenseDescriptorInit(handle.ptr(), mat.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- Matmul 输出指针 nullptr / *非空覆盖 ----
TEST_F(MatmulDescriptorInitFixture, MatmulDescrNullptr)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    auto ret = aclsparseLtMatmulDescriptorInit(handle.ptr(), nullptr, ACL_SPARSE_OP_NON_TRANSPOSE,
                                               ACL_SPARSE_OP_NON_TRANSPOSE, nullptr, nullptr, nullptr, nullptr,
                                               ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(MatmulDescriptorInitFixture, MatmulDescrNonEmpty)
{
    StdMatmulInput in;
    ASSERT_TRUE(InitStdMatmulInput(in, true));
    // *mm.ptr() 非空 → 覆盖 → INVALID_VALUE
    auto ret = aclsparseLtMatmulDescriptorInit(in.handle.ptr(), in.mm.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE,
                                               ACL_SPARSE_OP_NON_TRANSPOSE, in.matA.ptr(), in.matB.ptr(), in.matC.ptr(),
                                               in.matD.ptr(), ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- matA/matB/matC/matD = nullptr（Matmul 接口）----
TEST_F(MatmulDescriptorInitFixture, MatmulMatANullptr)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matB, matC, matD;
    ASSERT_EQ(matB.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matC.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matD.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulDescriptor_t descr = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(handle.ptr(), &descr, ACL_SPARSE_OP_NON_TRANSPOSE,
                                               ACL_SPARSE_OP_NON_TRANSPOSE, nullptr, matB.ptr(), matC.ptr(), matD.ptr(),
                                               ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(MatmulDescriptorInitFixture, MatmulMatBNullptr)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matA, matC, matD;
    ASSERT_EQ(matA.initStructured(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matC.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matD.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulDescriptor_t descr = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(handle.ptr(), &descr, ACL_SPARSE_OP_NON_TRANSPOSE,
                                               ACL_SPARSE_OP_NON_TRANSPOSE, matA.ptr(), nullptr, matC.ptr(), matD.ptr(),
                                               ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(MatmulDescriptorInitFixture, MatmulMatCNullptr)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matA, matB, matD;
    ASSERT_EQ(matA.initStructured(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matB.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matD.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulDescriptor_t descr = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(handle.ptr(), &descr, ACL_SPARSE_OP_NON_TRANSPOSE,
                                               ACL_SPARSE_OP_NON_TRANSPOSE, matA.ptr(), matB.ptr(), nullptr, matD.ptr(),
                                               ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(MatmulDescriptorInitFixture, MatmulMatDNullptr)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matA, matB, matC;
    ASSERT_EQ(matA.initStructured(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matB.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matC.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulDescriptor_t descr = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(handle.ptr(), &descr, ACL_SPARSE_OP_NON_TRANSPOSE,
                                               ACL_SPARSE_OP_NON_TRANSPOSE, matA.ptr(), matB.ptr(), matC.ptr(), nullptr,
                                               ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- Destroy 边界 - 指针 nullptr ----
TEST_F(MatmulDescriptorInitFixture, DestroyPtrNullptr)
{
    EXPECT_EQ(aclsparseLtMatDescriptorDestroy(nullptr), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseLtMatmulDescriptorDestroy(nullptr), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// ---- Destroy 边界 - *descr 为 nullptr（空操作）----
TEST_F(MatmulDescriptorInitFixture, DestroyDescrNullptr)
{
    aclsparseLtMatDescriptor_t matDescr = nullptr;
    EXPECT_EQ(aclsparseLtMatDescriptorDestroy(&matDescr), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulDescriptor_t mmDescr = nullptr;
    EXPECT_EQ(aclsparseLtMatmulDescriptorDestroy(&mmDescr), ACL_SPARSE_STATUS_SUCCESS);
}

// ---- Destroy 语义 - DenseDescriptorInit → Destroy，*ptr 置 nullptr ----
TEST_F(MatmulDescriptorInitFixture, DenseDestroySemantic)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard mat;
    ASSERT_EQ(mat.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseLtMatDescriptorDestroy(mat.ptr()), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(mat.get(), nullptr); // Destroy 后 *ptr 置 nullptr
    mat.release();                 // 阻止 Guard 析构再次 Destroy
}

// ---- Destroy 语义 - MatmulDescriptorInit → Destroy，*ptr 置 nullptr ----
TEST_F(MatmulDescriptorInitFixture, MatmulDestroySemantic)
{
    StdMatmulInput in;
    ASSERT_TRUE(InitStdMatmulInput(in, true));
    EXPECT_EQ(aclsparseLtMatmulDescriptorDestroy(in.mm.ptr()), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(in.mm.get(), nullptr);
    in.mm.release();
}

// ============================================================================
// 补充用例：覆盖 Structured 独立函数的输出指针校验分支 +
// Matmul IsMatDescrValid 的 *mat != nullptr 条件（当前 nullptr 用例仅覆盖指针
// nullptr）
// ============================================================================

// ---- Structured 输出指针 nullptr（Structured 独立函数分支覆盖）----
TEST_F(MatmulDescriptorInitFixture, StructMatDescrNullptr)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    auto ret = aclsparseLtStructuredDescriptorInit(handle.ptr(), nullptr, 16, 16, 16, 16, ACL_FLOAT,
                                                   ACL_SPARSE_ORDER_COL, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- Structured *matDescr 非空（覆盖已有，Structured 独立函数分支）----
TEST_F(MatmulDescriptorInitFixture, StructMatDescrNonEmpty)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard mat;
    ASSERT_EQ(mat.initStructured(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL,
                                 ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    // *mat.ptr() 非空 → 覆盖已有 → INVALID_VALUE
    auto ret = aclsparseLtStructuredDescriptorInit(handle.ptr(), mat.ptr(), 16, 16, 16, 16, ACL_FLOAT,
                                                   ACL_SPARSE_ORDER_COL, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- Matmul 传入指向 nullptr 的 mat 指针（IsMatDescrValid 的 *mat !=
// nullptr 条件）---- 上方 nullptr 用例传 nullptr 覆盖了 mat != nullptr 的 false；本用例传
// &matANull（指针非空但 *matANull==nullptr）， 覆盖 *mat != nullptr 的 false
// 分支，二者共同完成 IsMatDescrValid 的条件覆盖。
TEST_F(MatmulDescriptorInitFixture, MatmulMatStarNull)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matB, matC, matD;
    ASSERT_EQ(matB.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matC.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matD.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatDescriptor_t matANull = nullptr;
    aclsparseLtMatmulDescriptor_t descr = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(handle.ptr(), &descr, ACL_SPARSE_OP_NON_TRANSPOSE,
                                               ACL_SPARSE_OP_NON_TRANSPOSE, &matANull, matB.ptr(), matC.ptr(),
                                               matD.ptr(), ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- matC 为 structured 应被拒绝（ValidateMatmulDescriptorParams 的 C/D
// dense 校验）----
TEST_F(MatmulDescriptorInitFixture, MatmulMatCStructured)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matA, matB, matC, matD;
    ASSERT_EQ(matA.initStructured(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matB.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    // matC 为 structured → 违反 C/D 须为 dense 约束
    ASSERT_EQ(matC.initStructured(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matD.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatmulDescrGuard mm;
    auto ret = mm.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, matA.ptr(), matB.ptr(),
                       matC.ptr(), matD.ptr(), ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- matD 为 structured 应被拒绝（ValidateMatmulDescriptorParams 的 C/D
// dense 校验）----
TEST_F(MatmulDescriptorInitFixture, MatmulMatDStructured)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matA, matB, matC, matD;
    ASSERT_EQ(matA.initStructured(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matB.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(matC.initDense(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL), ACL_SPARSE_STATUS_SUCCESS);
    // matD 为 structured → 违反 C/D 须为 dense 约束
    ASSERT_EQ(matD.initStructured(handle.ptr(), 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatmulDescrGuard mm;
    auto ret = mm.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, matA.ptr(), matB.ptr(),
                       matC.ptr(), matD.ptr(), ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ============================================================================
// SPARSE_MAT_POINTER 属性（D4）：存储不参与执行（matA/matB 参数优先）。
// 对齐 cuSPARSELt CUSPARSELT_MATMUL_SPARSE_MAT_POINTER——官方两份样例均在
// DescriptorInit 后 Set 该属性，本仓补齐枚举与 Set/Get 分支使移植调用序列
// 不再返回 NOT_SUPPORTED；执行语义不变（不接 matA=null 回退链）。
// ============================================================================

// ---- SPARSE_MAT_POINTER：默认 nullptr / Set-Get 往返 / 描述符间默认值隔离 ----
TEST_F(MatmulDescriptorInitFixture, SparseMatPointerRoundTrip)
{
    StdMatmulInput in;
    ASSERT_TRUE(InitStdMatmulInput(in, true));

    // 未 Set 时 Get 得默认值 nullptr。
    void* gotBefore = reinterpret_cast<void*>(0xdeadbeef);
    aclsparseLtConstMatmulDescriptor_t mmConst = in.mm.get();
    EXPECT_EQ(aclsparseLtMatmulDescGetAttribute(in.handle.ptr(), &mmConst,
                ACLSPARSELT_MATMUL_SPARSE_MAT_POINTER, &gotBefore, sizeof(void*)),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(gotBefore, nullptr);

    // Set(dA) 成功后 Get 得同一指针（往返一致）。
    int dummySparseMat = 0;
    void* sparsePtr = &dummySparseMat;  // 存储语义测试，任意非空指针即可
    EXPECT_EQ(aclsparseLtMatmulDescSetAttribute(in.handle.ptr(), in.mm.ptr(),
                ACLSPARSELT_MATMUL_SPARSE_MAT_POINTER, &sparsePtr, sizeof(void*)),
              ACL_SPARSE_STATUS_SUCCESS);
    void* gotAfter = nullptr;
    EXPECT_EQ(aclsparseLtMatmulDescGetAttribute(in.handle.ptr(), &mmConst,
                ACLSPARSELT_MATMUL_SPARSE_MAT_POINTER, &gotAfter, sizeof(void*)),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(gotAfter, sparsePtr);

    // 新描述符不受已 Set 描述符影响（默认值隔离）。
    StdMatmulInput other;
    ASSERT_TRUE(InitStdMatmulInput(other, true));
    void* gotOther = reinterpret_cast<void*>(0xdeadbeef);
    aclsparseLtConstMatmulDescriptor_t mmOtherConst = other.mm.get();
    EXPECT_EQ(aclsparseLtMatmulDescGetAttribute(other.handle.ptr(), &mmOtherConst,
                ACLSPARSELT_MATMUL_SPARSE_MAT_POINTER, &gotOther, sizeof(void*)),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(gotOther, nullptr);
}

// ---- SPARSE_MAT_POINTER：dataSize 错配 → INVALID_VALUE（仿 BIAS_POINTER 模式）----
TEST_F(MatmulDescriptorInitFixture, SparseMatPointerSizeMismatch)
{
    StdMatmulInput in;
    ASSERT_TRUE(InitStdMatmulInput(in, true));
    void* sparsePtr = nullptr;
    EXPECT_EQ(aclsparseLtMatmulDescSetAttribute(in.handle.ptr(), in.mm.ptr(),
                ACLSPARSELT_MATMUL_SPARSE_MAT_POINTER, &sparsePtr, sizeof(int)),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    // Get 侧同款 dataSize 校验。
    aclsparseLtConstMatmulDescriptor_t mmConst = in.mm.get();
    void* got = nullptr;
    EXPECT_EQ(aclsparseLtMatmulDescGetAttribute(in.handle.ptr(), &mmConst,
                ACLSPARSELT_MATMUL_SPARSE_MAT_POINTER, &got, sizeof(int)),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ============================================================================
// ALLOC_FAILED 注入（可选）
// 通过 -DSPARSELT_TEST_INJECT_ALLOC_FAIL 启用；默认不编译以避免全局 operator
// new 重载副作用。
// ============================================================================

#ifdef SPARSELT_TEST_INJECT_ALLOC_FAIL
#include <cstdlib>
#include <new>

static bool g_injectAllocFail = false;

void* operator new(std::size_t n, const std::nothrow_t& tag) noexcept
{
    (void)tag;
    if (g_injectAllocFail)
        return nullptr;
    return std::malloc(n);
}
void operator delete(void* p, const std::nothrow_t&) noexcept
{
    std::free(p);
}

TEST_F(MatmulDescriptorInitFixture, AllocFailedInjection)
{
    sparse_test::SparseLtHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatDescriptor_t descr = nullptr;
    g_injectAllocFail = true;
    auto ret = aclsparseLtDenseDescriptorInit(handle.ptr(), &descr, 16, 16, 16, 16, ACL_FLOAT, ACL_SPARSE_ORDER_COL);
    g_injectAllocFail = false;
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_ALLOC_FAILED);
    EXPECT_EQ(descr, nullptr);
}
#endif // SPARSELT_TEST_INJECT_ALLOC_FAIL
