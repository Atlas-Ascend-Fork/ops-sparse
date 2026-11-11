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
 * @file coo_set_pointers_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseCooSetPointers（arch35，纯 Host 接口）.
 *
 * CooSetPointers 对标 cuSPARSE cusparseCooSetPointers：重绑 COO 描述符的
 * row/col/values 三指针。期望值即调用方传入指针本身（roundtrip oracle，EXACT
 * 零容差）：Set(P1) -> aclsparseCooGet 回读九字段（指针==P1）+ 白盒
 * ptrs==rowInds / idxs==colInds 复用同步断言（descr.cpp:1083-1087——CooGet 读回
 * 的正是复用字段 ptrs/idxs）与 structureGeneration 不变锁定（COO 路径不调用
 * MarkSpMatStructureChanged，与 CSR 行为差异）。
 *
 *   - TEST_P (CooSetPointersTest)           : coo_set_pointers_test.csv 驱动的
 *                                             9 个成功用例（基础/idx1/I64/fp16/
 *                                             nnz=0 全 null/nnz=0 全有效/tall/
 *                                             wide/large），含 nnz=0 时 null 合法
 *                                             补集
 *   - TEST_F (CooSetPointersSemanticsTest)  : 指针变化触发 ResetSpmvActivePlan
 *                                             （三哨兵清空，COP-S01）/ 全同指针
 *                                             不触发（哨兵保持，COP-S02）；
 *                                             双向断言 generation 不变
 *   - TEST_F (CooSetPointersExceptionTest)  : null descr / DnMat 错型句柄 /
 *                                             非 COO 格式 / nnz>0 任一指针 null
 *                                             （COP-E01~E04），失败后断言
 *                                             CooGet 回读仍为旧指针组 P0
 */

#include "test_common.h"
#include "spmat_csv_common.h"      // GetCasesFromCsvChecked：带构成校验的 CSV 装载（描述符测试私有共享）
#include "spmat_test_helpers.h"    // SpMatGetOutParams/WrapSpMatHandle：CooGet 回读哨兵组与句柄 RAII 包装

#include "coo_set_pointers_param.h"
#include "coo_set_pointers_npu_wrapper.h"

using namespace sparse_test;

namespace {
// 白盒哨兵指针：仅作位比较，全程不被解引用
void *const kSentinelBuffer = reinterpret_cast<void *>(0x1234);

// CooGet 九字段回读（哨兵出参组防实现漏写任一字段）：三指针回读值经引用
// 出参返回，几何/类型字段由调用方按需断言（COP-S01/S02 语义用例与
// COP-E04 失败后回读共用本文件内唯一回读实现）
void ReadBackCooGetPointers(const aclsparseSpMatDescr_t descr, void *&rowInd, void *&colInd,
                            void *&values)
{
    SpMatGetOutParams v;
    ASSERT_EQ(aclsparseCooGet(descr, &v.rows, &v.cols, &v.nnz, &v.ptrA, &v.ptrB, &v.ptrC,
                              &v.idxType, &v.idxBase, &v.valueType),
              ACL_SPARSE_STATUS_SUCCESS);
    rowInd = v.ptrA;
    colInd = v.ptrB;
    values = v.ptrC;
}
}  // namespace

// ============================================================================
// GTest parameterized fixture: CooSetPointersTest（CSV 驱动）
// ============================================================================
class CooSetPointersTest : public testing::TestWithParam<CooSetPointersParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// ============================================================================
// Test body: success-path parameterized test（每条 CSV 行一个用例）
// ============================================================================
TEST_P(CooSetPointersTest, SetPointersRoundTrip)
{
    const auto &p = GetParam();
    std::cout << "==== " << p.case_name << " ==== rows=" << p.rows << " cols=" << p.cols
              << " nnz=" << p.nnz << " idx_type=" << p.idx_type << " idx_base=" << p.idx_base
              << " dtype=" << p.dtype << " null_ptrs=" << p.null_ptrs << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    // 1. NPU 侧执行：CreateCoo(P0) -> SetPointers(P1) -> CooGet roundtrip + 白盒
    auto r = CooSetPointersNpu(env_->stream(), p.rows, p.cols, p.nnz, p.idx_type,
                               p.idx_base, p.isFp16(), p.null_ptrs != 0);

    ASSERT_EQ(r.createRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCreateCoo failed";
    ASSERT_EQ(r.setRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCooSetPointers failed";
    ASSERT_EQ(r.getRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCooGet failed";

    // 2. CooGet 九字段 roundtrip：几何/类型与传入一致（EXACT 零容差）
    EXPECT_EQ(r.rows, p.rows) << "rows roundtrip mismatch";
    EXPECT_EQ(r.cols, p.cols) << "cols roundtrip mismatch";
    EXPECT_EQ(r.nnz, p.nnz) << "nnz roundtrip mismatch";
    const aclsparseIndexType_t expectIdxType =
        (p.idx_type == 1) ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    EXPECT_EQ(r.idxType, expectIdxType) << "idxType roundtrip mismatch";
    EXPECT_EQ(r.idxBase, (p.idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE
                                           : ACL_SPARSE_INDEX_BASE_ZERO)
        << "idxBase roundtrip mismatch";
    EXPECT_EQ(r.valueType, p.isFp16() ? ACL_FLOAT16 : ACL_FLOAT)
        << "valueType roundtrip mismatch";

    // 3. 指针 roundtrip：CooGet 回读指针 == Set 传入的 P1
    EXPECT_TRUE(r.outRowIndsMatchP1) << "cooRowInd should match P1";
    EXPECT_TRUE(r.outColIndsMatchP1) << "cooColInd should match P1";
    EXPECT_TRUE(r.outValuesMatchP1) << "cooValues should match P1";

    // 4. 白盒：复用字段同步（ptrs==rowInds、idxs==colInds）+ generation 不变
    EXPECT_TRUE(r.whiteBoxPtrsMatchesRowInds) << "ptrs should mirror rowInds";
    EXPECT_TRUE(r.whiteBoxIdxsMatchesColInds) << "idxs should mirror colInds";
    EXPECT_EQ(r.whiteBoxGenerationAfter, r.whiteBoxGenerationBefore)
        << "COO SetPointers must not touch structureGeneration";

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    CooSetPointersCases,
    CooSetPointersTest,
    testing::ValuesIn(GetCasesFromCsvChecked<CooSetPointersParam>(
        "coo_set_pointers_test.csv", 9)),
    [](const testing::TestParamInfo<CooSetPointersParam> &info) { return info.param.case_name; });

// ============================================================================
// Semantics test fixture: CooSetPointersSemanticsTest
// 白盒语义：Reset 双态（变 / 全同）+ generation 不变锁定。
// 基础几何 4×4×5 fp32；Set/CooGet 均不解引用内容，host 占位缓冲即可。
// ============================================================================
class CooSetPointersSemanticsTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    // 初始指针组 P0（4×4×5）
    std::vector<int32_t> hostRow_{0, 1, 0, 2, 3};
    std::vector<int32_t> hostCol_{0, 1, 2, 3, 1};
    std::vector<float> hostVal_{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    // P1 新指针组
    std::vector<int32_t> newRow_{1, 0, 1, 0, 2};
    std::vector<int32_t> newCol_{1, 2, 0, 3, 0};
    std::vector<float> newVal_{5.0F, 4.0F, 3.0F, 2.0F, 1.0F};

    SpMatManager CreateCooWithP0()
    {
        return SpMatManager::createCoo(4, 4, 5, hostRow_.data(), hostCol_.data(),
                                      hostVal_.data(), ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    }

    // 白盒哨兵：ResetSpmvActivePlan 会清空的三个 SpMV 活性字段
    void SetActivePlanSentinels(const aclsparseSpMatDescr_t descr)
    {
        descr->activeBuffer = kSentinelBuffer;
        descr->activeSpmvBuffer = kSentinelBuffer;
        descr->activeRowOffsets = kSentinelBuffer;
    }

    void SetUp() override {}
};

// COP-S01: 指针变化触发 ResetSpmvActivePlan——白盒哨兵（activeBuffer/
// activeSpmvBuffer/activeRowOffsets 置非空）后 Set(P1)，三哨兵全部清空
//（descr.cpp:1079-1082）+ CooGet 回读 P1；structureGeneration 不变
//（COO 路径无 Mark 调用，与 CSR 行为差异锁定）
TEST_F(CooSetPointersSemanticsTest, PointersChangedResetsActivePlan)
{
    SpMatManager mat = CreateCooWithP0();
    SetActivePlanSentinels(mat.get());
    const uint64_t genBefore = mat.get()->structureGeneration;

    ASSERT_EQ(aclsparseCooSetPointers(mat.get(), newRow_.data(), newCol_.data(),
                                      newVal_.data()),
              ACL_SPARSE_STATUS_SUCCESS);

    EXPECT_EQ(mat.get()->activeBuffer, nullptr) << "ResetSpmvActivePlan should clear";
    EXPECT_EQ(mat.get()->activeSpmvBuffer, nullptr) << "ResetSpmvActivePlan should clear";
    EXPECT_EQ(mat.get()->activeRowOffsets, nullptr) << "ResetSpmvActivePlan should clear";
    EXPECT_EQ(mat.get()->structureGeneration, genBefore)
        << "COO SetPointers must not touch structureGeneration";

    // CooGet 回读 P1
    void *rowInd = nullptr;
    void *colInd = nullptr;
    void *values = nullptr;
    ReadBackCooGetPointers(mat.get(), rowInd, colInd, values);
    EXPECT_EQ(rowInd, newRow_.data());
    EXPECT_EQ(colInd, newCol_.data());
    EXPECT_EQ(values, newVal_.data());
}

// COP-S02: 同参二次调用（Set(P0) 后再 Set(P0)）——三指针全同，
// Reset 条件不满足（:1079），哨兵全部保持 + generation 不变 + CooGet 回读 P0
TEST_F(CooSetPointersSemanticsTest, SamePointersKeepPlan)
{
    SpMatManager mat = CreateCooWithP0();
    SetActivePlanSentinels(mat.get());
    const uint64_t genBefore = mat.get()->structureGeneration;

    ASSERT_EQ(aclsparseCooSetPointers(mat.get(), hostRow_.data(), hostCol_.data(),
                                      hostVal_.data()),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseCooSetPointers(mat.get(), hostRow_.data(), hostCol_.data(),
                                      hostVal_.data()),
              ACL_SPARSE_STATUS_SUCCESS);

    EXPECT_EQ(mat.get()->activeBuffer, kSentinelBuffer) << "same pointers should keep plan";
    EXPECT_EQ(mat.get()->activeSpmvBuffer, kSentinelBuffer) << "same pointers should keep plan";
    EXPECT_EQ(mat.get()->activeRowOffsets, kSentinelBuffer) << "same pointers should keep plan";
    EXPECT_EQ(mat.get()->structureGeneration, genBefore);

    // CooGet 回读 P0
    void *rowInd = nullptr;
    void *colInd = nullptr;
    void *values = nullptr;
    ReadBackCooGetPointers(mat.get(), rowInd, colInd, values);
    EXPECT_EQ(rowInd, hostRow_.data());
    EXPECT_EQ(colInd, hostCol_.data());
    EXPECT_EQ(values, hostVal_.data());
}

// ============================================================================
// Exception test fixture: CooSetPointersExceptionTest
// 失败路径直调 API（不经 wrapper）；基础几何 4×4×5 fp32，初始指针组 P0 有效。
// ============================================================================
class CooSetPointersExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    std::vector<int32_t> hostRow_{0, 1, 0, 2, 3};
    std::vector<int32_t> hostCol_{0, 1, 2, 3, 1};
    std::vector<float> hostVal_{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};

    // 基础几何 4×4×5 COO 描述符（unique_ptr + aclsparseDestroySpMat 删除器 RAII）
    std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)> CreateCoo4x4()
    {
        aclsparseSpMatDescr_t descr = nullptr;
        EXPECT_EQ(aclsparseCreateCoo(&descr, 4, 4, 5, hostRow_.data(), hostCol_.data(),
                                      hostVal_.data(), ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
                  ACL_SPARSE_STATUS_SUCCESS);
        return WrapSpMatHandle(descr);
    }

    // 失败后状态不被污染：CooGet 回读仍为旧指针组 P0（失败不改状态）
    void ExpectGetStillReadsP0(const aclsparseSpMatDescr_t descr)
    {
        void *rowInd = nullptr;
        void *colInd = nullptr;
        void *values = nullptr;
        ReadBackCooGetPointers(descr, rowInd, colInd, values);
        EXPECT_EQ(rowInd, hostRow_.data()) << "failed Set should keep old rowInds";
        EXPECT_EQ(colInd, hostCol_.data()) << "failed Set should keep old colInds";
        EXPECT_EQ(values, hostVal_.data()) << "failed Set should keep old values";
    }

    void SetUp() override {}
};

// COP-E01: nullptr descriptor -> INVALID_VALUE（descr.cpp:1073）
TEST_F(CooSetPointersExceptionTest, NullDescr)
{
    EXPECT_EQ(aclsparseCooSetPointers(nullptr, hostRow_.data(), hostCol_.data(),
                                       hostVal_.data()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// COP-E02: DnMat 错型句柄伪装 SpMat -> INVALID_VALUE（签名不符 :1073-1074）；
// DnMat 随后正常 Destroy
TEST_F(CooSetPointersExceptionTest, WrongTypeHandle)
{
    aclsparseDnMatDescr_t dnMat = nullptr;
    ASSERT_EQ(aclsparseCreateDnMat(&dnMat, 2, 2, 2, hostVal_.data(), ACL_FLOAT,
                                   ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    auto *punned = reinterpret_cast<aclsparseSpMatDescr_t>(dnMat);
    EXPECT_EQ(aclsparseCooSetPointers(punned, hostRow_.data(), hostCol_.data(),
                                       hostVal_.data()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyDnMat(dnMat);
}

// COP-E03: 非 COO 格式描述符（CSR）-> INVALID_VALUE（format != COO :1074）
TEST_F(CooSetPointersExceptionTest, NonCooFormat)
{
    aclsparseSpMatDescr_t csr = nullptr;
    ASSERT_EQ(aclsparseCreateCsr(&csr, 4, 4, 5, hostRow_.data(), hostCol_.data(),
                                  hostVal_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                  ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)> guard(
        csr, &aclsparseDestroySpMat);
    EXPECT_EQ(aclsparseCooSetPointers(csr, hostRow_.data(), hostCol_.data(),
                                       hostVal_.data()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// COP-E04: nnz>0 且任一指针 null（rowIndices/colIndices/values 三断言）->
// 均 INVALID_VALUE（:1075-1076）；每次失败后 CooGet 回读仍为 P0。
// 合法补集：nnz=0 时全 null（COP-C05）
TEST_F(CooSetPointersExceptionTest, NullPointersNonZeroNnz)
{
    auto mat = CreateCoo4x4();

    EXPECT_EQ(aclsparseCooSetPointers(mat.get(), nullptr, hostCol_.data(), hostVal_.data()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectGetStillReadsP0(mat.get());

    EXPECT_EQ(aclsparseCooSetPointers(mat.get(), hostRow_.data(), nullptr, hostVal_.data()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectGetStillReadsP0(mat.get());

    EXPECT_EQ(aclsparseCooSetPointers(mat.get(), hostRow_.data(), hostCol_.data(), nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectGetStillReadsP0(mat.get());
}
