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
 * @file matmulPlanInit_test.cpp
 * @brief GTest 入口：TEST_P CSV 驱动 + TEST_F nullptr 异常 / Destroy 边界。
 *
 * 隔离设计：不 include matmulDescriptorInit 的测试头文件。
 * 验证目标：
 *   ① 合法入参返回 SUCCESS 且 Plan/AlgSelection 字段被正确填充（白盒访问 internal.h）
 *   ② 非法入参返回规定的错误码
 */

#include "cann_ops_sparseLt.h"

#include "csv_loader.h"
#include "matmulPlanInit_npu_wrapper.h"
#include "matmulPlanInit_param.h"
#include "sparse_test.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>

// ============================================================================
// 全局 ACL 环境单例
// ============================================================================
class PlanAclEnvManager
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
std::once_flag PlanAclEnvManager::once_;
std::unique_ptr<sparse_test::AclEnvScope> PlanAclEnvManager::instance_;

// ============================================================================
// 标准 Plan 输入 helper：构造 handle + matA/B/C/D + matmulDescr + algSelection
// ============================================================================

struct StdPlanInput
{
    sparse_test::PlanHandleGuard handle;
    sparse_test::PlanMatDescrGuard matA, matB, matC, matD;
    sparse_test::PlanMatmulDescrGuard mm;
    sparse_test::AlgSelectionGuard algSel;
};

static bool InitStdPlanInput(StdPlanInput& in, bool initAlgSel = true)
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
    EXPECT_EQ(in.mm.init(in.handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
                         in.matA.ptr(), in.matB.ptr(), in.matC.ptr(), in.matD.ptr(), ACL_SPARSE_COMPUTE_32F),
              ACL_SPARSE_STATUS_SUCCESS);
    if (in.mm.get() == nullptr) { return false; }
    if (initAlgSel)
    {
        EXPECT_EQ(in.algSel.init(in.handle.ptr(), in.mm.ptr(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT),
                  ACL_SPARSE_STATUS_SUCCESS);
        if (in.algSel.get() == nullptr) { return false; }
    }
    return true;
}

// ============================================================================
// TEST_P：CSV 驱动参数化用例
// ============================================================================
class MatmulPlanInitTest : public testing::TestWithParam<sparse_test::MatmulPlanInitParam>
{
  public:
    static void SetUpTestSuite() { (void)PlanAclEnvManager::Get(); }
    static void TearDownTestSuite() {}

  protected:
    sparse_test::MatmulPlanInitParam param_;
    void SetUp() override { param_ = GetParam(); }
};

// CsvDriven plan 分支辅助函数
static bool RunPlanCase(const sparse_test::PlanHandleGuard& handle,
                        const sparse_test::MatmulPlanInitParam& p, aclsparseStatus_t expect)
{
    sparse_test::PlanMatDescrGuard matA, matB, matC, matD;
    EXPECT_EQ(sparse_test::InitMatFromPlanParam(handle.ptr(), matA, p.a), ACL_SPARSE_STATUS_SUCCESS)
        << p.case_name << ": matA init failed";
    if (matA.get() == nullptr) { return false; }
    EXPECT_EQ(sparse_test::InitMatFromPlanParam(handle.ptr(), matB, p.b), ACL_SPARSE_STATUS_SUCCESS)
        << p.case_name << ": matB init failed";
    if (matB.get() == nullptr) { return false; }
    EXPECT_EQ(sparse_test::InitMatFromPlanParam(handle.ptr(), matC, p.c), ACL_SPARSE_STATUS_SUCCESS)
        << p.case_name << ": matC init failed";
    if (matC.get() == nullptr) { return false; }
    EXPECT_EQ(sparse_test::InitMatFromPlanParam(handle.ptr(), matD, p.d), ACL_SPARSE_STATUS_SUCCESS)
        << p.case_name << ": matD init failed";
    if (matD.get() == nullptr) { return false; }

    sparse_test::PlanMatmulDescrGuard mm;
    EXPECT_EQ(mm.init(handle.ptr(), p.opA, p.opB, matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), p.computeType),
              ACL_SPARSE_STATUS_SUCCESS)
        << p.case_name << ": matmulDescr init failed";
    if (mm.get() == nullptr) { return false; }

    sparse_test::AlgSelectionGuard algSel;
    EXPECT_EQ(algSel.init(handle.ptr(), mm.ptr(), p.alg), expect) << p.case_name;
    if (expect != ACL_SPARSE_STATUS_SUCCESS) { return true; }
    if (algSel.get() == nullptr) { return false; }

    if (p.verifyFields)
    {
        if (!sparse_test::VerifyAlgSelectionFields(algSel.get(), mm.get(), p.alg)) { return false; }
    }

    sparse_test::PlanGuard plan;
    EXPECT_EQ(plan.init(handle.ptr(), mm.ptr(), algSel.ptr()), ACL_SPARSE_STATUS_SUCCESS) << p.case_name;
    if (plan.get() == nullptr) { return false; }

    if (p.verifyFields)
    {
        if (!sparse_test::VerifyPlanFields(plan.get(), mm.get(), algSel.get())) { return false; }
    }
    return true;
}

TEST_P(MatmulPlanInitTest, CsvDriven)
{
    const auto& p = param_;

    if (p.expectStatus == sparse_test::PLAN_STATUS_SKIP_SENTINEL)
    {
        GTEST_SKIP() << "Skip: " << p.case_name;
    }

    sparse_test::PlanHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS) << p.case_name << ": aclsparseLtInit failed";

    aclsparseStatus_t expect = sparse_test::PlanExpectedStatus(p.expectStatus);

    if (p.target == "plan")
    {
        if (!RunPlanCase(handle, p, expect)) { return; }
    }
    else if (p.target == "algSelection")
    {
        // AlgSelection 独立测试（alg 非法等）
        sparse_test::PlanMatDescrGuard matA, matB, matC, matD;
        EXPECT_EQ(sparse_test::InitMatFromPlanParam(handle.ptr(), matA, p.a), ACL_SPARSE_STATUS_SUCCESS)
            << p.case_name << ": matA init failed";
        EXPECT_EQ(sparse_test::InitMatFromPlanParam(handle.ptr(), matB, p.b), ACL_SPARSE_STATUS_SUCCESS)
            << p.case_name << ": matB init failed";
        EXPECT_EQ(sparse_test::InitMatFromPlanParam(handle.ptr(), matC, p.c), ACL_SPARSE_STATUS_SUCCESS)
            << p.case_name << ": matC init failed";
        EXPECT_EQ(sparse_test::InitMatFromPlanParam(handle.ptr(), matD, p.d), ACL_SPARSE_STATUS_SUCCESS)
            << p.case_name << ": matD init failed";
        if (matA.get() == nullptr || matB.get() == nullptr || matC.get() == nullptr || matD.get() == nullptr)
        {
            return;
        }
        sparse_test::PlanMatmulDescrGuard mm;
        ASSERT_EQ(mm.init(handle.ptr(), p.opA, p.opB, matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(),
                          p.computeType),
                  ACL_SPARSE_STATUS_SUCCESS);
        sparse_test::AlgSelectionGuard algSel;
        auto ret = algSel.init(handle.ptr(), mm.ptr(), p.alg);
        EXPECT_EQ(ret, expect) << p.case_name;
    }
    else
    {
        FAIL() << p.case_name << ": unknown target '" << p.target << "'";
    }
}

INSTANTIATE_TEST_SUITE_P(MatmulPlanInitCases, MatmulPlanInitTest,
                         testing::ValuesIn(sparse_test::GetCasesFromCsv<sparse_test::MatmulPlanInitParam>(
                             "matmulPlanInit_test.csv")),
                         [](const testing::TestParamInfo<sparse_test::MatmulPlanInitParam>& info)
                         {
                             std::string name = info.param.case_name;
                             std::replace(name.begin(), name.end(), '-', '_');
                             return name;
                         });

// ============================================================================
// TEST_F：纯指针异常 + Destroy 边界
// ============================================================================
class MatmulPlanInitFixture : public testing::Test
{
  protected:
    void SetUp() override { (void)PlanAclEnvManager::Get(); }
};

// ---- AlgSelection handle nullptr ----
TEST_F(MatmulPlanInitFixture, AlgSelectionHandleNullptr)
{
    aclsparseLtMatmulAlgSelection_t sel = nullptr;
    aclsparseLtMatmulDescriptor_t mm = nullptr;
    auto ret = aclsparseLtMatmulAlgSelectionInit(nullptr, &sel, &mm, ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// ---- Plan handle nullptr ----
TEST_F(MatmulPlanInitFixture, PlanHandleNullptr)
{
    aclsparseLtMatmulPlan_t plan = nullptr;
    aclsparseLtMatmulDescriptor_t mm = nullptr;
    aclsparseLtMatmulAlgSelection_t sel = nullptr;
    auto ret = aclsparseLtMatmulPlanInit(nullptr, &plan, &mm, &sel);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// ---- handle points to an uninitialized handle value ----
TEST_F(MatmulPlanInitFixture, AlgSelectionHandleValueNullptr)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, false));
    aclsparseLtHandle_t handle = nullptr;
    EXPECT_EQ(in.algSel.init(&handle, in.mm.ptr(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT),
              ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(in.algSel.get(), nullptr);
}

TEST_F(MatmulPlanInitFixture, PlanHandleValueNullptr)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, true));
    aclsparseLtHandle_t handle = nullptr;
    sparse_test::PlanGuard plan;
    EXPECT_EQ(plan.init(&handle, in.mm.ptr(), in.algSel.ptr()), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(plan.get(), nullptr);
}

// ---- AlgSelection output nullptr / *非空 ----
TEST_F(MatmulPlanInitFixture, AlgSelectionOutputNullptr)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, false));
    auto ret = aclsparseLtMatmulAlgSelectionInit(in.handle.ptr(), nullptr, in.mm.ptr(),
                                                 ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(MatmulPlanInitFixture, AlgSelectionNonEmpty)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, true));
    auto ret = aclsparseLtMatmulAlgSelectionInit(in.handle.ptr(), in.algSel.ptr(), in.mm.ptr(),
                                                 ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- Plan output nullptr / *非空 ----
TEST_F(MatmulPlanInitFixture, PlanOutputNullptr)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, true));
    auto ret = aclsparseLtMatmulPlanInit(in.handle.ptr(), nullptr, in.mm.ptr(), in.algSel.ptr());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(MatmulPlanInitFixture, PlanNonEmpty)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, true));
    sparse_test::PlanGuard plan;
    ASSERT_EQ(plan.init(in.handle.ptr(), in.mm.ptr(), in.algSel.ptr()), ACL_SPARSE_STATUS_SUCCESS);
    auto ret = aclsparseLtMatmulPlanInit(in.handle.ptr(), plan.ptr(), in.mm.ptr(), in.algSel.ptr());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- AlgSelection matmulDescr nullptr / *nullptr ----
TEST_F(MatmulPlanInitFixture, AlgSelectionMatmulDescrNullptr)
{
    sparse_test::PlanHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulAlgSelection_t sel = nullptr;
    auto ret = aclsparseLtMatmulAlgSelectionInit(handle.ptr(), &sel, nullptr, ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(MatmulPlanInitFixture, AlgSelectionMatmulDescrStarNull)
{
    sparse_test::PlanHandleGuard handle;
    ASSERT_EQ(handle.status(), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulDescriptor_t mmNull = nullptr;
    aclsparseLtMatmulAlgSelection_t sel = nullptr;
    auto ret = aclsparseLtMatmulAlgSelectionInit(handle.ptr(), &sel, &mmNull, ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- Plan matmulDescr nullptr / *nullptr ----
TEST_F(MatmulPlanInitFixture, PlanMatmulDescrNullptr)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, true));
    aclsparseLtMatmulPlan_t plan = nullptr;
    auto ret = aclsparseLtMatmulPlanInit(in.handle.ptr(), &plan, nullptr, in.algSel.ptr());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(MatmulPlanInitFixture, PlanMatmulDescrStarNull)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, true));
    aclsparseLtMatmulDescriptor_t mmNull = nullptr;
    aclsparseLtMatmulPlan_t plan = nullptr;
    auto ret = aclsparseLtMatmulPlanInit(in.handle.ptr(), &plan, &mmNull, in.algSel.ptr());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- Plan algSelection nullptr / *nullptr ----
TEST_F(MatmulPlanInitFixture, PlanAlgSelectionNullptr)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, true));
    aclsparseLtMatmulPlan_t plan = nullptr;
    auto ret = aclsparseLtMatmulPlanInit(in.handle.ptr(), &plan, in.mm.ptr(), nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(MatmulPlanInitFixture, PlanAlgSelectionStarNull)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, true));
    aclsparseLtMatmulAlgSelection_t selNull = nullptr;
    aclsparseLtMatmulPlan_t plan = nullptr;
    auto ret = aclsparseLtMatmulPlanInit(in.handle.ptr(), &plan, in.mm.ptr(), &selNull);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ---- Destroy 边界 ----
TEST_F(MatmulPlanInitFixture, DestroyPtrNullptr)
{
    EXPECT_EQ(aclsparseLtMatmulAlgSelectionDestroy(nullptr), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseLtMatmulPlanDestroy(nullptr), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(MatmulPlanInitFixture, DestroyDescrNullptr)
{
    aclsparseLtMatmulAlgSelection_t sel = nullptr;
    EXPECT_EQ(aclsparseLtMatmulAlgSelectionDestroy(&sel), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseLtMatmulPlan_t plan = nullptr;
    EXPECT_EQ(aclsparseLtMatmulPlanDestroy(&plan), ACL_SPARSE_STATUS_SUCCESS);
}

// ---- Destroy 语义：*ptr 置 nullptr ----
TEST_F(MatmulPlanInitFixture, AlgSelectionDestroySemantic)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, true));
    EXPECT_EQ(aclsparseLtMatmulAlgSelectionDestroy(in.algSel.ptr()), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(in.algSel.get(), nullptr);
    in.algSel.release();
}

TEST_F(MatmulPlanInitFixture, PlanDestroySemantic)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, true));
    sparse_test::PlanGuard plan;
    ASSERT_EQ(plan.init(in.handle.ptr(), in.mm.ptr(), in.algSel.ptr()), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseLtMatmulPlanDestroy(plan.ptr()), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(plan.get(), nullptr);
    plan.release();
}

// ---- 完整生命周期：Init → PlanInit → Destroy → 再 Init 复用 ----
TEST_F(MatmulPlanInitFixture, FullLifecycle)
{
    StdPlanInput in;
    ASSERT_TRUE(InitStdPlanInput(in, true));

    sparse_test::PlanGuard plan;
    EXPECT_EQ(plan.init(in.handle.ptr(), in.mm.ptr(), in.algSel.ptr()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_NE(plan.get(), nullptr);
    EXPECT_TRUE(sparse_test::VerifyPlanFields(plan.get(), in.mm.get(), in.algSel.get()));

    EXPECT_EQ(aclsparseLtMatmulPlanDestroy(plan.ptr()), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(plan.get(), nullptr);
    plan.release();

    // 销毁后可再次 Init（plan 变量已置 nullptr）
    EXPECT_EQ(plan.init(in.handle.ptr(), in.mm.ptr(), in.algSel.ptr()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_NE(plan.get(), nullptr);
}
