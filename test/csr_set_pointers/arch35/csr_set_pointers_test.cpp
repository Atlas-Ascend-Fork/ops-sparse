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
 * @file csr_set_pointers_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseCsrSetPointers（arch35，纯 Host 接口）.
 *
 * CsrSetPointers 对标 cuSPARSE cusparseCsrSetPointers：重绑 CSR 描述符的
 * rowOffsets/colIndices/values 三指针（经共享 SetCompressedPointers，与
 * CscSetPointers 共用代码路径但入口/格式守卫不同）。期望值即调用方传入指针本身
 * （roundtrip oracle，EXACT 零容差）：Set(P1) -> aclsparseCsrGet 回读十字段
 * （指针==P1）+ 白盒 structureGeneration+1（CSV 行 P1 均相对 P0 发生 ptrs/idxs
 * 变化，structureChanged 分支 descr.cpp:1006-1008）。
 *
 *   - TEST_P (CsrSetPointersTest)           : csr_set_pointers_test.csv 驱动的
 *                                             10 个成功用例（基础/idx1/I64/fp16/
 *                                             nnz=0 idx+val null/rows=0·cols=0
 *                                             offsets null/tall/wide/large）
 *   - TEST_F (CsrSetPointersSemanticsTest)  : 变化五值分态白盒——offsets-only
 *                                             （DTS 失效 + generation+1，ptrs 轴
 *                                             隔离，CSP-S01）/ idxs-only（DTS
 *                                             保持 + generation+1，CSP-S02）/
 *                                             values-only（generation 不变 +
 *                                             Reset 走，CSP-S03）/ 全同（generation
 *                                             不变 + Reset 不走 + 尾部四效应
 *                                             无条件全断言，CSP-S04）
 *   - TEST_F (CsrSetPointersExceptionTest)  : null descr / DnMat 错型句柄 /
 *                                             非 CSR 格式 / 非零维 offsets null /
 *                                             nnz>0 idx·val null（CSP-E01~E05），
 *                                             失败后断言 CsrGet 回读仍为旧指针组 P0
 */

#include "test_common.h"
#include "spmat_csv_common.h"      // GetCasesFromCsvChecked：带构成校验的 CSV 装载（描述符测试私有共享）
#include "spmat_test_helpers.h"    // SpMatGetOutParams/WrapSpMatHandle：CsrGet 回读哨兵组与句柄 RAII 包装

#include "csr_set_pointers_param.h"
#include "csr_set_pointers_npu_wrapper.h"

using namespace sparse_test;

namespace {
// 白盒哨兵指针：仅作位比较，全程不被解引用
void *const kSentinelBuffer = reinterpret_cast<void *>(0x1234);

// CsrGet 十字段回读（哨兵出参组防实现漏写任一字段）：三指针回读值经引用
// 出参返回，几何/类型字段由调用方按需断言（CSP-S01~S04 分态前提验证与
// CSP-E04 失败后回读共用本文件内唯一回读实现）
void ReadBackCsrGetPointers(const aclsparseSpMatDescr_t descr, void *&offsets, void *&colInd,
                            void *&values)
{
    SpMatGetOutParams v;
    ASSERT_EQ(aclsparseCsrGet(descr, &v.rows, &v.cols, &v.nnz, &v.ptrA, &v.ptrB, &v.ptrC,
                              &v.ptrType, &v.idxType, &v.idxBase, &v.valueType),
              ACL_SPARSE_STATUS_SUCCESS);
    offsets = v.ptrA;
    colInd = v.ptrB;
    values = v.ptrC;
}
}  // namespace

// ============================================================================
// GTest parameterized fixture: CsrSetPointersTest（CSV 驱动）
// ============================================================================
class CsrSetPointersTest : public testing::TestWithParam<CsrSetPointersParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// ============================================================================
// Test body: success-path parameterized test（每条 CSV 行一个用例）
// ============================================================================
TEST_P(CsrSetPointersTest, SetPointersRoundTrip)
{
    const auto &p = GetParam();
    std::cout << "==== " << p.case_name << " ==== rows=" << p.rows << " cols=" << p.cols
              << " nnz=" << p.nnz << " idx_type=" << p.idx_type << " idx_base=" << p.idx_base
              << " dtype=" << p.dtype << " null_ptrs=" << p.null_ptrs << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");
    ASSERT_TRUE(p.null_ptrs >= 0 && p.null_ptrs <= 2) << "invalid null_ptrs in CSV";

    // 1. NPU 侧执行：CreateCsr(P0) -> SetPointers(P1 按行构造) -> CsrGet 回读
    auto r = CsrSetPointersNpu(env_->stream(), p.rows, p.cols, p.nnz, p.idx_type,
                               p.idx_base, p.isFp16(), p.null_ptrs);

    ASSERT_EQ(r.createRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCreateCsr failed";
    ASSERT_EQ(r.setRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCsrSetPointers failed";
    ASSERT_EQ(r.getRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCsrGet failed";

    // 2. CsrGet 十字段 roundtrip：几何/类型与传入一致（EXACT 零容差）
    EXPECT_EQ(r.rows, p.rows) << "rows roundtrip mismatch";
    EXPECT_EQ(r.cols, p.cols) << "cols roundtrip mismatch";
    EXPECT_EQ(r.nnz, p.nnz) << "nnz roundtrip mismatch";
    const aclsparseIndexType_t expectIdxType =
        (p.idx_type == 1) ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    EXPECT_EQ(r.ptrType, expectIdxType) << "rowOffsetsType roundtrip mismatch";
    EXPECT_EQ(r.idxType, expectIdxType) << "colIndType roundtrip mismatch";
    EXPECT_EQ(r.idxBase, (p.idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE
                                           : ACL_SPARSE_INDEX_BASE_ZERO)
        << "idxBase roundtrip mismatch";
    EXPECT_EQ(r.valueType, p.isFp16() ? ACL_FLOAT16 : ACL_FLOAT)
        << "valueType roundtrip mismatch";

    // 3. 指针 roundtrip：CsrGet 回读指针 == Set 传入的 P1（null_ptrs 行的
    //    null 分量同步回读为 null）
    EXPECT_TRUE(r.outOffsetsMatchP1) << "csrRowOffsets should match P1";
    EXPECT_TRUE(r.outColIndMatchP1) << "csrColInd should match P1";
    EXPECT_TRUE(r.outValuesMatchP1) << "csrValues should match P1";

    // 4. 白盒：CSV 行 P1 均相对 P0 发生 ptrs/idxs 变化 -> structureChanged
    //    分支应使 generation+1
    EXPECT_EQ(r.whiteBoxGenerationAfter, r.whiteBoxGenerationBefore + 1)
        << "structure rebinding should bump structureGeneration";

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    CsrSetPointersCases,
    CsrSetPointersTest,
    testing::ValuesIn(GetCasesFromCsvChecked<CsrSetPointersParam>(
        "csr_set_pointers_test.csv", 10)),
    [](const testing::TestParamInfo<CsrSetPointersParam> &info) { return info.param.case_name; });

// ============================================================================
// Semantics test fixture: CsrSetPointersSemanticsTest
// 变化五值分态白盒（全变已由 CSV 覆盖，本夹具补齐 offsets-only / idxs-only /
// values-only / 全同 四态）。
// 基础几何 4×4×5 fp32；Set/CsrGet 均不解引用内容，host 占位缓冲即可。
// ============================================================================
class CsrSetPointersSemanticsTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    // 初始指针组 P0（4×4×5：offsets 6 端点 / colInd 5 / values 5）
    std::vector<int32_t> hostOffsets_{0, 1, 2, 3, 4, 5};
    std::vector<int32_t> hostColInd_{0, 1, 2, 3, 4};
    std::vector<float> hostVal_{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    // 分态用新指针组
    std::vector<int32_t> newOffsets_{1, 2, 3, 4, 5, 6};
    std::vector<int32_t> newColInd_{4, 3, 2, 1, 0};
    std::vector<float> newVal_{5.0F, 4.0F, 3.0F, 2.0F, 1.0F};

    SpMatManager CreateCsrWithP0()
    {
        return SpMatManager::createCsr(4, 4, 5, hostOffsets_.data(), hostColInd_.data(),
                                      hostVal_.data(), ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                      ACL_FLOAT);
    }

    // 失败无关的字段回读辅助：断言三指针与传入一致（供分态前提验证）
    void ExpectGetPointers(const aclsparseSpMatDescr_t descr, const void *offsets,
                           const void *colInd, const void *values)
    {
        void *outOffsets = nullptr;
        void *outColInd = nullptr;
        void *outValues = nullptr;
        ReadBackCsrGetPointers(descr, outOffsets, outColInd, outValues);
        EXPECT_EQ(outOffsets, offsets);
        EXPECT_EQ(outColInd, colInd);
        EXPECT_EQ(outValues, values);
    }

    void SetUp() override {}
};

// CSP-S01: offsets-only——Set(新 offsets + P0 idxs + P0 values)：generation+1
//（structureChanged :1006-1008）；activeDenseToSparseBuffer 清空 +
// denseToSparseAnalysisReady==false（:999-1002，ptrs 轴隔离——仅 offsets 变
// 即触发 DenseToSparse 分析失效）；activeBuffer/activeRowOffsets 哨兵清空
//（Reset :995 + 尾部 :1010）
TEST_F(CsrSetPointersSemanticsTest, OffsetsOnlyChangedInvalidatesAnalysis)
{
    SpMatManager mat = CreateCsrWithP0();
    // 白盒哨兵：DTS 分析就绪 + 活性指针
    mat.get()->activeDenseToSparseBuffer = kSentinelBuffer;
    mat.get()->denseToSparseAnalysisReady = true;
    mat.get()->activeBuffer = kSentinelBuffer;
    mat.get()->activeRowOffsets = kSentinelBuffer;
    const uint64_t genBefore = mat.get()->structureGeneration;

    ASSERT_EQ(aclsparseCsrSetPointers(mat.get(), newOffsets_.data(), hostColInd_.data(),
                                      hostVal_.data()),
              ACL_SPARSE_STATUS_SUCCESS);

    EXPECT_EQ(mat.get()->structureGeneration, genBefore + 1)
        << "offsets rebinding is a structure change";
    EXPECT_EQ(mat.get()->activeDenseToSparseBuffer, nullptr)
        << "new offsets require re-analysis";
    EXPECT_EQ(mat.get()->denseToSparseAnalysisReady, false)
        << "new offsets require re-analysis";
    EXPECT_EQ(mat.get()->activeBuffer, nullptr) << "Reset/tail should clear activeBuffer";
    EXPECT_EQ(mat.get()->activeRowOffsets, nullptr) << "Reset should clear activeRowOffsets";

    // offsets-only 前提验证：idxs/values 仍为 P0
    ExpectGetPointers(mat.get(), newOffsets_.data(), hostColInd_.data(), hostVal_.data());
}

// CSP-S02: idxs-only——offsets/values 复用 P0、仅换新 idxs：generation+1
//（idxs 轴 structureChanged）；DenseToSparse 哨兵保持（:999 不触发，idxs 轴
// 隔离验证——新分配的输出 idxs 不需重新分析）；activeRowOffsets 哨兵清空（:995 走）
TEST_F(CsrSetPointersSemanticsTest, IndicesOnlyChanged)
{
    SpMatManager mat = CreateCsrWithP0();
    mat.get()->activeDenseToSparseBuffer = kSentinelBuffer;
    mat.get()->denseToSparseAnalysisReady = true;
    mat.get()->activeRowOffsets = kSentinelBuffer;
    const uint64_t genBefore = mat.get()->structureGeneration;

    ASSERT_EQ(aclsparseCsrSetPointers(mat.get(), hostOffsets_.data(), newColInd_.data(),
                                      hostVal_.data()),
              ACL_SPARSE_STATUS_SUCCESS);

    EXPECT_EQ(mat.get()->structureGeneration, genBefore + 1)
        << "indices rebinding is a structure change";
    EXPECT_EQ(mat.get()->activeDenseToSparseBuffer, kSentinelBuffer)
        << "fresh indices do not require re-analysis";
    EXPECT_EQ(mat.get()->denseToSparseAnalysisReady, true)
        << "fresh indices do not require re-analysis";
    EXPECT_EQ(mat.get()->activeRowOffsets, nullptr) << "Reset should clear activeRowOffsets";

    ExpectGetPointers(mat.get(), hostOffsets_.data(), newColInd_.data(), hostVal_.data());
}

// CSP-S03: values-only——ptrs/idxs 复用 P0、仅换新 values：generation 不变
//（structureChanged=false，:1006 条件不满足）；DenseToSparse 保持；
// activeRowOffsets 哨兵清空（:994-996 任一变化即 Reset）
TEST_F(CsrSetPointersSemanticsTest, ValuesOnlyChanged)
{
    SpMatManager mat = CreateCsrWithP0();
    mat.get()->activeDenseToSparseBuffer = kSentinelBuffer;
    mat.get()->denseToSparseAnalysisReady = true;
    mat.get()->activeRowOffsets = kSentinelBuffer;
    const uint64_t genBefore = mat.get()->structureGeneration;

    ASSERT_EQ(aclsparseCsrSetPointers(mat.get(), hostOffsets_.data(), hostColInd_.data(),
                                      newVal_.data()),
              ACL_SPARSE_STATUS_SUCCESS);

    EXPECT_EQ(mat.get()->structureGeneration, genBefore)
        << "values rebinding is not a structure change";
    EXPECT_EQ(mat.get()->activeDenseToSparseBuffer, kSentinelBuffer)
        << "fresh values do not require re-analysis";
    EXPECT_EQ(mat.get()->denseToSparseAnalysisReady, true);
    EXPECT_EQ(mat.get()->activeRowOffsets, nullptr)
        << "any pointer change should reset the active plan";

    ExpectGetPointers(mat.get(), hostOffsets_.data(), hostColInd_.data(), newVal_.data());
}

// CSP-S04: 全相同指针二次 Set——generation 不变、activeRowOffsets 哨兵保持
//（:994-996 条件不满足，Reset 不走）；activeBuffer/activeSpmvBuffer 仍被清空
//（尾部 :1010-1011 无条件执行）；spmmWorkspaceOwnerId 滚动为新值（:1009，
// ≠ 记录旧值）；spmvCubeMaxRowLengthValid==false（:1012，哨兵被无条件清空）
// ——尾部四效应全断言
TEST_F(CsrSetPointersSemanticsTest, SamePointersTailAlwaysClears)
{
    SpMatManager mat = CreateCsrWithP0();
    // 白盒预置：maxRowLength 哨兵 + 记录 ownerId 旧值 + 活性指针哨兵
    mat.get()->spmvCubeMaxRowLengthValid = true;
    mat.get()->activeBuffer = kSentinelBuffer;
    mat.get()->activeSpmvBuffer = kSentinelBuffer;
    mat.get()->activeRowOffsets = kSentinelBuffer;
    const uint64_t savedOwnerId = mat.get()->spmmWorkspaceOwnerId;
    const uint64_t genBefore = mat.get()->structureGeneration;

    // 同参二次调用（全同 P0）
    ASSERT_EQ(aclsparseCsrSetPointers(mat.get(), hostOffsets_.data(), hostColInd_.data(),
                                      hostVal_.data()),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseCsrSetPointers(mat.get(), hostOffsets_.data(), hostColInd_.data(),
                                      hostVal_.data()),
              ACL_SPARSE_STATUS_SUCCESS);

    EXPECT_EQ(mat.get()->structureGeneration, genBefore)
        << "same pointers must not bump structureGeneration";
    EXPECT_EQ(mat.get()->activeRowOffsets, kSentinelBuffer)
        << "same pointers should keep the active plan";
    EXPECT_EQ(mat.get()->activeBuffer, nullptr)
        << "tail unconditionally clears activeBuffer";
    EXPECT_EQ(mat.get()->activeSpmvBuffer, nullptr)
        << "tail unconditionally clears activeSpmvBuffer";
    EXPECT_NE(mat.get()->spmmWorkspaceOwnerId, savedOwnerId)
        << "tail unconditionally rolls spmmWorkspaceOwnerId";
    EXPECT_FALSE(mat.get()->spmvCubeMaxRowLengthValid)
        << "tail unconditionally invalidates maxRowLength cache";

    ExpectGetPointers(mat.get(), hostOffsets_.data(), hostColInd_.data(), hostVal_.data());
}

// ============================================================================
// Exception test fixture: CsrSetPointersExceptionTest
// 失败路径直调 API（不经 wrapper）；基础几何 4×4×5 fp32，初始指针组 P0 有效。
// ============================================================================
class CsrSetPointersExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    std::vector<int32_t> hostOffsets_{0, 1, 2, 3, 4, 5};
    std::vector<int32_t> hostColInd_{0, 1, 2, 3, 4};
    std::vector<float> hostVal_{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};

    // 基础几何 4×4×5 CSR 描述符（unique_ptr + aclsparseDestroySpMat 删除器 RAII）
    std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)> CreateCsr4x4()
    {
        aclsparseSpMatDescr_t descr = nullptr;
        EXPECT_EQ(aclsparseCreateCsr(&descr, 4, 4, 5, hostOffsets_.data(),
                                     hostColInd_.data(), hostVal_.data(),
                                     ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                     ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
                  ACL_SPARSE_STATUS_SUCCESS);
        return WrapSpMatHandle(descr);
    }

    // 失败后状态不被污染：CsrGet 回读仍为旧指针组 P0（失败不改状态）
    void ExpectGetStillReadsP0(const aclsparseSpMatDescr_t descr)
    {
        void *outOffsets = nullptr;
        void *outColInd = nullptr;
        void *outValues = nullptr;
        ReadBackCsrGetPointers(descr, outOffsets, outColInd, outValues);
        EXPECT_EQ(outOffsets, hostOffsets_.data()) << "failed Set should keep old offsets";
        EXPECT_EQ(outColInd, hostColInd_.data()) << "failed Set should keep old colInd";
        EXPECT_EQ(outValues, hostVal_.data()) << "failed Set should keep old values";
    }

    void SetUp() override {}
};

// CSP-E01: nullptr descriptor -> INVALID_VALUE（descr.cpp:988）
TEST_F(CsrSetPointersExceptionTest, NullDescr)
{
    EXPECT_EQ(aclsparseCsrSetPointers(nullptr, hostOffsets_.data(), hostColInd_.data(),
                                       hostVal_.data()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// CSP-E02: DnMat 错型句柄伪装 SpMat -> INVALID_VALUE（签名不符 :988）；
// DnMat 随后正常 Destroy
TEST_F(CsrSetPointersExceptionTest, WrongTypeHandle)
{
    aclsparseDnMatDescr_t dnMat = nullptr;
    ASSERT_EQ(aclsparseCreateDnMat(&dnMat, 2, 2, 2, hostVal_.data(), ACL_FLOAT,
                                   ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    auto *punned = reinterpret_cast<aclsparseSpMatDescr_t>(dnMat);
    EXPECT_EQ(aclsparseCsrSetPointers(punned, hostOffsets_.data(), hostColInd_.data(),
                                       hostVal_.data()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyDnMat(dnMat);
}

// CSP-E03: 非 CSR 格式描述符（COO）-> INVALID_VALUE（format != CSR :988）
TEST_F(CsrSetPointersExceptionTest, NonCsrFormat)
{
    aclsparseSpMatDescr_t coo = nullptr;
    ASSERT_EQ(aclsparseCreateCoo(&coo, 4, 4, 5, hostOffsets_.data(), hostColInd_.data(),
                                  hostVal_.data(), ACL_SPARSE_INDEX_32I,
                                  ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)> guard(
        coo, &aclsparseDestroySpMat);
    EXPECT_EQ(aclsparseCsrSetPointers(coo, hostOffsets_.data(), hostColInd_.data(),
                                      hostVal_.data()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// CSP-E04: rows>0 且 cols>0 且 offsets=null -> INVALID_VALUE（:989）；
// CsrGet 回读仍为 P0。合法补集：CSP-C06/C07（零维 offsets=null）
TEST_F(CsrSetPointersExceptionTest, NullOffsetsNonZeroDims)
{
    auto mat = CreateCsr4x4();
    EXPECT_EQ(aclsparseCsrSetPointers(mat.get(), nullptr, hostColInd_.data(),
                                      hostVal_.data()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectGetStillReadsP0(mat.get());
}

// CSP-E05: nnz>0 且 colIndices/values null（两断言）-> 均 INVALID_VALUE（:990）；
// CsrGet 回读仍为 P0。合法补集：CSP-C05（nnz=0）
TEST_F(CsrSetPointersExceptionTest, NullIdxOrValNonZeroNnz)
{
    auto mat = CreateCsr4x4();
    EXPECT_EQ(aclsparseCsrSetPointers(mat.get(), hostOffsets_.data(), nullptr,
                                      hostVal_.data()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectGetStillReadsP0(mat.get());

    EXPECT_EQ(aclsparseCsrSetPointers(mat.get(), hostOffsets_.data(), hostColInd_.data(),
                                      nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectGetStillReadsP0(mat.get());
}
