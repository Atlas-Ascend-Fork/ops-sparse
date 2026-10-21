/*
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#include "test_common.h"
#include "spmv.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace sparse_test;

namespace {

struct SpmvParam : SparseTestParamBase {
    std::string case_name;
    std::string op;
    std::string index_base;
    std::string alg;
    std::string pointer_mode;
    float alpha = 1.0f;
    float beta = 0.0f;

    void fillCustom(const csv_map& row) override {
        case_name = parseString(row, "case_name");
        op = parseString(row, "op");
        index_base = parseString(row, "index_base");
        alg = parseString(row, "alg");
        pointer_mode = parseString(row, "pointer_mode");
        alpha = parseFloat(row, "alpha");
        beta = parseFloat(row, "beta");
    }

    std::string caseId() const override { return case_name; }
};

aclsparseOperation_t ParseOperation(const std::string& op) {
    return op == "TRANSPOSE" ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
}

aclsparseIndexBase_t ParseBase(const std::string& base) {
    return base == "ONE" ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
}

aclsparseSpMVAlg_t ParseAlgorithm(const std::string& alg) {
    return alg == "CSR_ALG2" ? ACL_SPARSE_SPMV_CSR_ALG2
         : alg == "CSR_ALG1" ? ACL_SPARSE_SPMV_CSR_ALG1
                              : ACL_SPARSE_SPMV_ALG_DEFAULT;
}

struct CsrFixture {
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_ind;
    std::vector<float> values;
};

CsrFixture MakeFixture(aclsparseIndexBase_t base) {
    const int32_t b = base == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    return {{b, b + 2, b + 3}, {b, b + 2, b + 1}, {2.0f, 3.0f, 4.0f}};
}

std::vector<float> Golden(const SpmvParam& p, aclsparseOperation_t op) {
    if (op == ACL_SPARSE_OP_NON_TRANSPOSE) {
        const std::vector<float> x{1.0f, 2.0f, 3.0f};
        const std::vector<float> y{5.0f, 6.0f};
        return {p.alpha * (2.0f * x[0] + 3.0f * x[2]) + p.beta * y[0],
                p.alpha * (4.0f * x[1]) + p.beta * y[1]};
    }
    const std::vector<float> x{10.0f, 20.0f};
    const std::vector<float> y{7.0f, 8.0f, 9.0f};
    return {p.alpha * (2.0f * x[0]) + p.beta * y[0],
            p.alpha * (4.0f * x[1]) + p.beta * y[1],
            p.alpha * (3.0f * x[0]) + p.beta * y[2]};
}

class SpmvArch35Test : public testing::TestWithParam<SpmvParam> {
protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }
};

testing::AssertionResult RunSpmvWithWorkspace(
    HandleManager& handle, aclsparseOperation_t op, const void* alpha,
    const void* beta, const SpMatManager& mat, const DnVecManager& x,
    DnVecManager& y, aclDataType computeType, aclsparseSpMVAlg_t alg,
    void* workspace, aclrtStream stream) {
    auto status = aclsparseSpMVPreprocess(handle.get(), op, alpha, mat.cget(), x.cget(),
        beta, y.get(), computeType, alg, workspace);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return testing::AssertionFailure() << "aclsparseSpMVPreprocess status=" << status;
    }
    status = aclsparseSpMV(handle.get(), op, alpha, mat.cget(), x.cget(),
        beta, y.get(), computeType, alg, workspace);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return testing::AssertionFailure() << "aclsparseSpMV status=" << status;
    }
    const auto syncStatus = aclrtSynchronizeStream(stream);
    if (syncStatus != ACL_SUCCESS) {
        return testing::AssertionFailure() << "aclrtSynchronizeStream status=" << syncStatus;
    }
    return testing::AssertionSuccess();
}

DeviceBuffer ExecuteThreeStage(HandleManager& handle, aclsparseOperation_t op,
                               const void* alpha, const void* beta,
                               const SpMatManager& mat, const DnVecManager& x,
                               DnVecManager& y, aclsparseSpMVAlg_t alg,
                               aclrtStream stream) {
    size_t bufferSize = 0;
    EXPECT_EQ(aclsparseSpMVGetBufferSize(handle.get(), op, alpha, mat.cget(), x.cget(),
        beta, y.get(), ACL_FLOAT, alg, &bufferSize), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_GE(bufferSize, 64U);
    auto workspace = DeviceBuffer::alloc(bufferSize);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(workspace.raw()) % 64U, 0U);
    EXPECT_TRUE(RunSpmvWithWorkspace(
        handle, op, alpha, beta, mat, x, y, ACL_FLOAT, alg, workspace.raw(), stream));
    return workspace;
}

void ExpectSpmvResult(const std::vector<float>& actual,
                      const std::vector<float>& expected,
                      const std::string& caseName, const char* phase) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f)
            << phase << " index=" << i << " case=" << caseName;
    }
}

std::vector<float> SecondGolden(const SpmvParam& p, aclsparseOperation_t op,
                                const std::vector<float>& first) {
    const std::vector<float> product = op == ACL_SPARSE_OP_NON_TRANSPOSE
        ? std::vector<float>{11.0f, 8.0f}
        : std::vector<float>{20.0f, 80.0f, 30.0f};
    std::vector<float> expected(first.size());
    for (size_t i = 0; i < first.size(); ++i) {
        expected[i] = p.alpha * product[i] + p.beta * first[i];
    }
    return expected;
}

void PrepareScalarPointers(const SpmvParam& p, DeviceBuffer* dAlpha,
                           DeviceBuffer* dBeta, const void** alpha,
                           const void** beta) {
    *alpha = &p.alpha;
    *beta = &p.beta;
    if (p.pointer_mode == "DEVICE") {
        *dAlpha = DeviceBuffer::copyFrom(&p.alpha, sizeof(float));
        *dBeta = DeviceBuffer::copyFrom(&p.beta, sizeof(float));
        *alpha = dAlpha->raw();
        *beta = dBeta->raw();
    }
}

TEST_P(SpmvArch35Test, ThreeStageAndExecute) {
    const SpmvParam& p = GetParam();
    const auto op = ParseOperation(p.op);
    const auto base = ParseBase(p.index_base);
    const auto alg = ParseAlgorithm(p.alg);
    const auto csr = MakeFixture(base);

    const std::vector<float> xHost = op == ACL_SPARSE_OP_NON_TRANSPOSE
        ? std::vector<float>{1.0f, 2.0f, 3.0f}
        : std::vector<float>{10.0f, 20.0f};
    const std::vector<float> yInit = op == ACL_SPARSE_OP_NON_TRANSPOSE
        ? std::vector<float>{5.0f, 6.0f}
        : std::vector<float>{7.0f, 8.0f, 9.0f};

    HandleManager handle;
    handle.setStream(env_->stream());
    ASSERT_EQ(aclsparseSetPointerMode(
        handle.get(), p.pointer_mode == "DEVICE"
            ? ACL_SPARSE_POINTER_MODE_DEVICE : ACL_SPARSE_POINTER_MODE_HOST),
        ACL_SPARSE_STATUS_SUCCESS);

    auto dRow = DeviceBuffer::copyFrom(csr.row_ptr.data(), csr.row_ptr.size() * sizeof(int32_t));
    auto dCol = DeviceBuffer::copyFrom(csr.col_ind.data(), csr.col_ind.size() * sizeof(int32_t));
    auto dVal = DeviceBuffer::copyFrom(csr.values.data(), csr.values.size() * sizeof(float));
    auto dX = DeviceBuffer::copyFrom(xHost.data(), xHost.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yInit.data(), yInit.size() * sizeof(float));
    auto mat = SpMatManager::createCsr(2, 3, 3, dRow.raw(), dCol.raw(), dVal.raw(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    auto x = DnVecManager::create(static_cast<int64_t>(xHost.size()), dX.raw(), ACL_FLOAT);
    auto y = DnVecManager::create(static_cast<int64_t>(yInit.size()), dY.raw(), ACL_FLOAT);

    DeviceBuffer dAlpha;
    DeviceBuffer dBeta;
    const void* alphaPtr = nullptr;
    const void* betaPtr = nullptr;
    PrepareScalarPointers(p, &dAlpha, &dBeta, &alphaPtr, &betaPtr);

    auto workspace = ExecuteThreeStage(handle, op, alphaPtr, betaPtr, mat, x, y,
                                       alg, env_->stream());

    std::vector<float> result(yInit.size());
    dY.copyToHost(result.data(), result.size() * sizeof(float));
    const auto expected = Golden(p, op);
    ExpectSpmvResult(result, expected, p.case_name, "first execution");

    // Reusing the same pattern/workspace must remain valid for a second call.
    ASSERT_EQ(aclsparseSpMV(handle.get(), op, alphaPtr, mat.cget(), x.cget(), betaPtr,
        y.get(), ACL_FLOAT, alg, workspace.raw()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
    dY.copyToHost(result.data(), result.size() * sizeof(float));
    const auto expectedSecond = SecondGolden(p, op, expected);
    ExpectSpmvResult(result, expectedSecond, p.case_name, "second execution");
}

INSTANTIATE_TEST_SUITE_P(
    SpmvA5Cases, SpmvArch35Test,
    testing::ValuesIn(GetCasesFromCsv<SpmvParam>("spmv_test.csv")),
    [](const testing::TestParamInfo<SpmvParam>& info) { return info.param.caseId(); });

class SpmvArch35ExceptionTest : public testing::Test {
protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }
};

TEST_F(SpmvArch35ExceptionTest, RejectUnsupportedFormatAndAlgorithm) {
    HandleManager handle;
    handle.setStream(env_->stream());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    std::vector<int32_t> rows{0, 1, 1};
    std::vector<int32_t> cols{0};
    std::vector<float> vals{1.0f};
    auto dRows = DeviceBuffer::copyFrom(rows.data(), rows.size() * sizeof(int32_t));
    auto dCols = DeviceBuffer::copyFrom(cols.data(), cols.size() * sizeof(int32_t));
    auto dVals = DeviceBuffer::copyFrom(vals.data(), vals.size() * sizeof(float));
    const std::vector<float> xHost{1.0f, 2.0f, 3.0f};
    const std::vector<float> yHost{0.0f, 0.0f};
    auto dX = DeviceBuffer::copyFrom(xHost.data(), xHost.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(float));
    auto mat = SpMatManager::createCoo(2, 3, 1, dRows.raw(), dCols.raw(), dVals.raw(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    auto x = DnVecManager::create(3, dX.raw(), ACL_FLOAT);
    auto y = DnVecManager::create(2, dY.raw(), ACL_FLOAT);
    size_t size = 0;
    EXPECT_EQ(aclsparseSpMVGetBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT, &size), ACL_SPARSE_STATUS_NOT_SUPPORTED);

    auto csr = MakeFixture(ACL_SPARSE_INDEX_BASE_ZERO);
    auto cRow = DeviceBuffer::copyFrom(csr.row_ptr.data(), csr.row_ptr.size() * sizeof(int32_t));
    auto cCol = DeviceBuffer::copyFrom(csr.col_ind.data(), csr.col_ind.size() * sizeof(int32_t));
    auto cVal = DeviceBuffer::copyFrom(csr.values.data(), csr.values.size() * sizeof(float));
    auto cMat = SpMatManager::createCsr(2, 3, 3, cRow.raw(), cCol.raw(), cVal.raw(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    EXPECT_EQ(aclsparseSpMVGetBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, cMat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_COO_ALG1, &size), ACL_SPARSE_STATUS_NOT_SUPPORTED);
    ASSERT_EQ(aclsparseCsrSetStrides(cMat.get(), 2, 1, 1), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseSpMVGetBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, cMat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT, &size), ACL_SPARSE_STATUS_SUCCESS);
}

template <typename T>
std::vector<T> WithStride(const std::vector<T>& logical, size_t stride, T sentinel) {
    std::vector<T> physical(logical.size() * stride, sentinel);
    for (size_t i = 0; i < logical.size(); ++i) {
        physical[i * stride] = logical[i];
    }
    return physical;
}

CsrFixture MakeStridedFixture(uint32_t rows, bool sorted) {
    CsrFixture csr{{1}, {}, {}};
    for (uint32_t row = 0; row < rows; ++row) {
        if (row % 2U == 0U) {
            csr.col_ind.insert(csr.col_ind.end(), sorted ? std::initializer_list<int32_t>{1, 3}
                                                        : std::initializer_list<int32_t>{3, 1});
            csr.values.insert(csr.values.end(), sorted ? std::initializer_list<float>{2, 3}
                                                      : std::initializer_list<float>{3, 2});
        } else {
            csr.col_ind.push_back(2);
            csr.values.push_back(4.0f);
        }
        csr.row_ptr.push_back(static_cast<int32_t>(csr.values.size()) + 1);
    }
    return csr;
}

void ExpectStridedSpmvResult(const DeviceBuffer& output, uint32_t rows,
                            bool transpose, float sentinel) {
    std::vector<float> result(output.size() / sizeof(float));
    output.copyToHost(result.data(), output.size());
    const float evenRows = static_cast<float>((rows + 1U) / 2U);
    const float oddRows = static_cast<float>(rows / 2U);
    const std::vector<float> transExpected{
        4.0f * evenRows + 15.0f, 8.0f * oddRows + 15.0f, 6.0f * evenRows + 15.0f};
    for (size_t i = 0; i < result.size(); ++i) {
        const size_t logical = i / 6U;
        const float expected = i % 6U != 0U ? sentinel
            : transpose ? transExpected[logical] : logical % 2U == 0U ? 37.0f : 31.0f;
        ASSERT_FLOAT_EQ(result[i], expected) << "physical index=" << i;
    }
}

void CheckStridedSpmv(aclrtStream stream, uint32_t rows, aclsparseOperation_t op,
                     aclsparseSpMVAlg_t alg, bool sorted) {
    const bool transpose = op != ACL_SPARSE_OP_NON_TRANSPOSE;
    const auto csr = MakeStridedFixture(rows, sorted);
    const std::vector<float> xLogical = transpose
        ? std::vector<float>(rows, 1.0f) : std::vector<float>{1, 2, 3};
    const size_t outputLength = transpose ? 3U : rows;
    constexpr float sentinel = -1234.0f;
    const auto rp = WithStride(csr.row_ptr, 2, -999);
    const auto ci = WithStride(csr.col_ind, 3, -999);
    const auto av = WithStride(csr.values, 4, sentinel);
    const auto xv = WithStride(xLogical, 5, sentinel);
    const auto yv = WithStride(std::vector<float>(outputLength, 5.0f), 6, sentinel);
    auto dRow = DeviceBuffer::copyFrom(rp.data(), rp.size() * sizeof(int32_t));
    auto dCol = DeviceBuffer::copyFrom(ci.data(), ci.size() * sizeof(int32_t));
    auto dVal = DeviceBuffer::copyFrom(av.data(), av.size() * sizeof(float));
    auto dX = DeviceBuffer::copyFrom(xv.data(), xv.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yv.data(), yv.size() * sizeof(float));
    auto mat = SpMatManager::createCsr(rows, 3, csr.values.size(), dRow.raw(), dCol.raw(),
        dVal.raw(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ONE, ACL_FLOAT);
    auto x = DnVecManager::create(xLogical.size(), dX.raw(), ACL_FLOAT);
    auto y = DnVecManager::create(outputLength, dY.raw(), ACL_FLOAT);
    ASSERT_EQ(aclsparseCsrSetStrides(mat.get(), 2, 3, 4), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseDnVecSetStride(x.get(), 5), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseDnVecSetStride(y.get(), 6), ACL_SPARSE_STATUS_SUCCESS);
    HandleManager handle;
    handle.setStream(stream);
    const float alpha = 2.0f;
    const float beta = 3.0f;
    size_t size = 0;
    ASSERT_EQ(aclsparseSpMVGetBufferSize(handle.get(), op, &alpha, mat.cget(),
        x.cget(), &beta, y.get(), ACL_FLOAT, alg, &size), ACL_SPARSE_STATUS_SUCCESS);
    auto workspace = DeviceBuffer::alloc(size);
    for (const bool preprocess : {false, true}) {
        SCOPED_TRACE(preprocess ? "normalized rows" : "original strided rows");
        ASSERT_EQ(aclrtMemcpy(dY.raw(), yv.size() * sizeof(float), yv.data(),
            yv.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        if (preprocess) {
            ASSERT_EQ(aclsparseSpMVPreprocess(handle.get(), op, &alpha, mat.cget(),
                x.cget(), &beta, y.get(), ACL_FLOAT, alg, workspace.raw()),
                ACL_SPARSE_STATUS_SUCCESS);
        }
        ASSERT_EQ(aclsparseSpMV(handle.get(), op, &alpha, mat.cget(), x.cget(),
            &beta, y.get(), ACL_FLOAT, alg, workspace.raw()), ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        ASSERT_NO_FATAL_FAILURE(ExpectStridedSpmvResult(dY, rows, transpose, sentinel));
    }
}

TEST_F(SpmvArch35ExceptionTest, StridedStoragePreservesPadding) {
    // Cover both N kernels, sorted/unsorted gather, deterministic and atomic T.
    CheckStridedSpmv(env_->stream(), 2, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_SPMV_ALG_DEFAULT, true);
    CheckStridedSpmv(env_->stream(), 65, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_SPMV_ALG_DEFAULT, true);
    CheckStridedSpmv(env_->stream(), 2, ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_SPMV_ALG_DEFAULT, true);
    CheckStridedSpmv(env_->stream(), 2, ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_SPMV_ALG_DEFAULT, false);
    CheckStridedSpmv(env_->stream(), 2, ACL_SPARSE_OP_CONJUGATE_TRANSPOSE, ACL_SPARSE_SPMV_CSR_ALG2, true);
    CheckStridedSpmv(env_->stream(), 4096, ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_SPMV_ALG_DEFAULT, true);
}

TEST_F(SpmvArch35ExceptionTest, SeparateWorkspaceCacheAndStrideIdentity) {
    aclsparseSpMatDescr mat;
    aclsparseDnVecDescr x;
    aclsparseDnVecDescr y;
    alignas(64) uint8_t workspace[256]{};
    alignas(64) uint8_t otherWorkspace[256]{};
    mat.activeBuffer = otherWorkspace;
    SpmvArch35RecordActivePlan(&mat, &x, &y, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, workspace, env_->stream(), 0);
    EXPECT_EQ(mat.activeBuffer, otherWorkspace);
    EXPECT_EQ(mat.activeSpmvBuffer, workspace);
    const auto matches = [&]() {
        return SpmvArch35ActivePlanMatches(&mat, &x, &y, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, workspace, env_->stream(), 0);
    };
    EXPECT_TRUE(matches());
    x.stride = 2;
    EXPECT_FALSE(matches());
    x.stride = 1;
    mat.ptrStride = 2;
    EXPECT_FALSE(matches());
    mat.ptrStride = 1;
    // A different operator overwrote the same allocation.
    mat.activeBuffer = workspace;
    EXPECT_FALSE(matches());
}

#ifdef SPMV_TEST_WITH_SPMM
void ExecuteSpmmWithWorkspace(HandleManager& handle, const SpMatManager& mat,
                              const DnMatManager& b, DnMatManager& c,
                              void* workspace, bool preprocess, aclrtStream stream) {
    const auto op = ACL_SPARSE_OP_NON_TRANSPOSE;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (preprocess) {
        ASSERT_EQ(aclsparseSpMMPreprocess(handle.get(), op, op, &alpha, mat.cget(),
            b.cget(), &beta, c.get(), ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, workspace),
            ACL_SPARSE_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclsparseSpMM(handle.get(), op, op, &alpha, mat.cget(), b.cget(),
        &beta, c.get(), ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, workspace),
        ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
}

TEST_F(SpmvArch35ExceptionTest, ReusesWorkspaceAcrossSpmvAndSpmm) {
    HandleManager handle;
    handle.setStream(env_->stream());
    const auto csr = MakeFixture(ACL_SPARSE_INDEX_BASE_ZERO);
    const std::vector<float> input{1, 2, 3};
    const std::vector<float> output{0, 0};
    auto dr = DeviceBuffer::copyFrom(csr.row_ptr.data(), csr.row_ptr.size() * sizeof(int32_t));
    auto dc = DeviceBuffer::copyFrom(csr.col_ind.data(), csr.col_ind.size() * sizeof(int32_t));
    auto dv = DeviceBuffer::copyFrom(csr.values.data(), csr.values.size() * sizeof(float));
    auto dx = DeviceBuffer::copyFrom(input.data(), input.size() * sizeof(float));
    auto dy = DeviceBuffer::copyFrom(output.data(), output.size() * sizeof(float));
    auto mat = SpMatManager::createCsr(2, 3, 3, dr.raw(), dc.raw(), dv.raw(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    auto x = DnVecManager::create(3, dx.raw(), ACL_FLOAT);
    auto y = DnVecManager::create(2, dy.raw(), ACL_FLOAT);
    auto b = DnMatManager::create(3, 1, 1, dx.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto c = DnMatManager::create(2, 1, 1, dy.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    const auto op = ACL_SPARSE_OP_NON_TRANSPOSE;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    size_t mvBytes = 0;
    size_t mmBytes = 0;
    ASSERT_EQ(aclsparseSpMVGetBufferSize(handle.get(), op, &alpha, mat.cget(),
        x.cget(), &beta, y.get(), ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, &mvBytes),
        ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseSpMMGetBufferSize(handle.get(), op, op, &alpha, mat.cget(),
        b.cget(), &beta, c.get(), ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, &mmBytes),
        ACL_SPARSE_STATUS_SUCCESS);
    auto workspace = DeviceBuffer::alloc(mvBytes > mmBytes ? mvBytes : mmBytes);
    for (const bool mmPreprocess : {false, true}) {
        ASSERT_EQ(aclsparseSpMVPreprocess(handle.get(), op, &alpha, mat.cget(),
            x.cget(), &beta, y.get(), ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.raw()),
            ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
        EXPECT_EQ(SpmvArch35ToMat(mat.cget())->activeBuffer, nullptr);
        ASSERT_NO_FATAL_FAILURE(ExecuteSpmmWithWorkspace(
            handle, mat, b, c, workspace.raw(), mmPreprocess, env_->stream()));
        std::vector<float> result(2);
        dy.copyToHost(result.data(), result.size() * sizeof(float));
        ExpectSpmvResult(result, {11, 8}, "shared workspace", "SpMM after SpMV");
        EXPECT_EQ(SpmvArch35ToMat(mat.cget())->activeSpmvBuffer, nullptr);
        // SpMV must now read the original CSR, not SpMM workspace metadata.
        ASSERT_EQ(aclsparseSpMV(handle.get(), op, &alpha, mat.cget(), x.cget(),
            &beta, y.get(), ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.raw()),
            ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
        dy.copyToHost(result.data(), result.size() * sizeof(float));
        ExpectSpmvResult(result, {11, 8}, "shared workspace", "SpMV after SpMM");
        EXPECT_EQ(SpmvArch35ToMat(mat.cget())->activeBuffer, nullptr);
    }
}
#endif

TEST_F(SpmvArch35ExceptionTest, ZeroNnzAndMisalignedWorkspace) {
    HandleManager handle;
    handle.setStream(env_->stream());
    const float alpha = 1.0f;
    const float beta = 2.0f;
    std::vector<int32_t> rows{0, 0, 0};
    const std::vector<float> xHost{1.0f, 2.0f, 3.0f};
    const std::vector<float> yHost{4.0f, 5.0f};
    auto dRows = DeviceBuffer::copyFrom(rows.data(), rows.size() * sizeof(int32_t));
    auto dX = DeviceBuffer::copyFrom(xHost.data(), xHost.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(float));
    auto mat = SpMatManager::createCsr(2, 3, 0, dRows.raw(), nullptr, nullptr,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    auto x = DnVecManager::create(3, dX.raw(), ACL_FLOAT);
    auto y = DnVecManager::create(2, dY.raw(), ACL_FLOAT);
    size_t size = 0;
    ASSERT_EQ(aclsparseSpMVGetBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT, &size), ACL_SPARSE_STATUS_SUCCESS);
    auto workspace = DeviceBuffer::alloc(size);
    ASSERT_TRUE(RunSpmvWithWorkspace(
        handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, &beta, mat, x, y,
        ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.raw(), env_->stream()));
    std::vector<float> result(2);
    dY.copyToHost(result.data(), result.size() * sizeof(float));
    EXPECT_FLOAT_EQ(result[0], 8.0f);
    EXPECT_FLOAT_EQ(result[1], 10.0f);

    auto* misaligned = static_cast<uint8_t*>(workspace.raw()) + 1U;
    EXPECT_EQ(aclsparseSpMV(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT, misaligned), ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SpmvArch35ExceptionTest, AllowsRepeatedCsrCoordinates) {
    HandleManager handle;
    handle.setStream(env_->stream());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // A 1x1 CSR can contain repeated entries.  This deliberately has nnz >
    // rows*cols and verifies that both entries participate in the sum.
    const std::vector<int32_t> rows{0, 2};
    const std::vector<int32_t> cols{0, 0};
    const std::vector<float> vals{2.0f, 3.0f};
    const std::vector<float> xHost{4.0f};
    const std::vector<float> yHost{9.0f};
    auto dRows = DeviceBuffer::copyFrom(rows.data(), rows.size() * sizeof(int32_t));
    auto dCols = DeviceBuffer::copyFrom(cols.data(), cols.size() * sizeof(int32_t));
    auto dVals = DeviceBuffer::copyFrom(vals.data(), vals.size() * sizeof(float));
    auto dX = DeviceBuffer::copyFrom(xHost.data(), xHost.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(float));
    auto mat = SpMatManager::createCsr(1, 1, 2, dRows.raw(), dCols.raw(), dVals.raw(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    auto x = DnVecManager::create(1, dX.raw(), ACL_FLOAT);
    auto y = DnVecManager::create(1, dY.raw(), ACL_FLOAT);

    size_t workspaceSize = 0;
    ASSERT_EQ(aclsparseSpMVGetBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_CSR_ALG2, &workspaceSize), ACL_SPARSE_STATUS_SUCCESS);
    auto workspace = DeviceBuffer::alloc(workspaceSize);
    ASSERT_TRUE(RunSpmvWithWorkspace(
        handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, &beta, mat, x, y,
        ACL_FLOAT, ACL_SPARSE_SPMV_CSR_ALG2, workspace.raw(), env_->stream()));
    float result = 0.0f;
    dY.copyToHost(&result, sizeof(result));
    EXPECT_FLOAT_EQ(result, 20.0f);
}

TEST_F(SpmvArch35ExceptionTest, RejectsNonzeroNnzWithZeroDimensions) {
    HandleManager handle;
    handle.setStream(env_->stream());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    alignas(64) uint8_t workspace[256]{};
    for (const bool zeroRows : {false, true}) {
        auto mat = SpMatManager::createCsr(zeroRows ? 0 : 2, zeroRows ? 2 : 0,
            1, nullptr, nullptr, nullptr);
        for (const auto op : {ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE}) {
            const bool transpose = op == ACL_SPARSE_OP_TRANSPOSE;
            auto x = DnVecManager::create(zeroRows == transpose ? 0 : 2, nullptr, ACL_FLOAT);
            auto y = DnVecManager::create(zeroRows == transpose ? 2 : 0, nullptr, ACL_FLOAT);
            size_t bytes = 0;
            EXPECT_EQ(aclsparseSpMVGetBufferSize(handle.get(), op, &alpha, mat.cget(), x.cget(),
                &beta, y.get(), ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, &bytes),
                ACL_SPARSE_STATUS_INVALID_VALUE);
            EXPECT_EQ(aclsparseSpMVPreprocess(handle.get(), op, &alpha, mat.cget(), x.cget(),
                &beta, y.get(), ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, workspace),
                ACL_SPARSE_STATUS_INVALID_VALUE);
            EXPECT_EQ(aclsparseSpMV(handle.get(), op, &alpha, mat.cget(), x.cget(),
                &beta, y.get(), ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, workspace),
                ACL_SPARSE_STATUS_INVALID_VALUE);
        }
    }
}

TEST_F(SpmvArch35ExceptionTest, QueriesWorkspaceBeforeBindingStorage) {
    HandleManager handle;
    handle.setStream(env_->stream());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto mat = SpMatManager::createCsr(2, 3, 3, nullptr, nullptr, nullptr);
    auto x = DnVecManager::create(3, nullptr, ACL_FLOAT);
    auto y = DnVecManager::create(2, nullptr, ACL_FLOAT);
    size_t workspaceSize = 0U;
    ASSERT_EQ(aclsparseSpMVGetBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT, &workspaceSize), ACL_SPARSE_STATUS_SUCCESS);
    auto workspace = DeviceBuffer::alloc(workspaceSize);
    EXPECT_EQ(aclsparseSpMVPreprocess(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.raw()), ACL_SPARSE_STATUS_INVALID_VALUE);

    const auto csr = MakeFixture(ACL_SPARSE_INDEX_BASE_ZERO);
    const std::vector<float> xHost{1.0f, 2.0f, 3.0f};
    const std::vector<float> yHost{0.0f, 0.0f};
    auto dRows = DeviceBuffer::copyFrom(csr.row_ptr.data(), csr.row_ptr.size() * sizeof(int32_t));
    auto dCols = DeviceBuffer::copyFrom(csr.col_ind.data(), csr.col_ind.size() * sizeof(int32_t));
    auto dVals = DeviceBuffer::copyFrom(csr.values.data(), csr.values.size() * sizeof(float));
    auto dX = DeviceBuffer::copyFrom(xHost.data(), xHost.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(float));
    mat = SpMatManager::createCsr(2, 3, 3, dRows.raw(), dCols.raw(), nullptr);
    ASSERT_EQ(aclsparseSpMVPreprocess(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.raw()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
    EXPECT_EQ(aclsparseSpMV(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.raw()), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(aclsparseSpMatSetValues(mat.get(), dVals.raw()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseDnVecSetValues(x.get(), dX.raw()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseDnVecSetValues(y.get(), dY.raw()), ACL_SPARSE_STATUS_SUCCESS);

    ASSERT_EQ(aclsparseSpMV(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.raw()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
    std::vector<float> result(2);
    dY.copyToHost(result.data(), result.size() * sizeof(float));
    EXPECT_FLOAT_EQ(result[0], 11.0f);
    EXPECT_FLOAT_EQ(result[1], 8.0f);
}

void CheckInt32Saturation(aclrtStream stream, int32_t rowCount,
                        aclsparseOperation_t operation, aclsparseSpMVAlg_t algorithm) {
    HandleManager handle;
    handle.setStream(stream);
    const int32_t alpha = std::numeric_limits<int32_t>::max();
    const int32_t beta = 0;
    std::vector<int32_t> rows(rowCount + 1, 2);
    rows[0] = 0;
    rows[1] = 1;
    const std::vector<int32_t> cols{0, 1};
    const std::vector<int8_t> values{127, -127};
    const bool transpose = operation != ACL_SPARSE_OP_NON_TRANSPOSE;
    const std::vector<int8_t> xHost(transpose ? rowCount : 2, 127);
    std::vector<int32_t> yHost(transpose ? 2 : rowCount, 0);
    auto dRows = DeviceBuffer::copyFrom(rows.data(), rows.size() * sizeof(int32_t));
    auto dCols = DeviceBuffer::copyFrom(cols.data(), cols.size() * sizeof(int32_t));
    auto dVals = DeviceBuffer::copyFrom(values.data(), values.size());
    auto dX = DeviceBuffer::copyFrom(xHost.data(), xHost.size());
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(int32_t));
    auto mat = SpMatManager::createCsr(rowCount, 2, 2, dRows.raw(), dCols.raw(), dVals.raw(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_INT8);
    auto x = DnVecManager::create(xHost.size(), dX.raw(), ACL_INT8);
    auto y = DnVecManager::create(yHost.size(), dY.raw(), ACL_INT32);
    size_t bytes = 0;
    ASSERT_EQ(aclsparseSpMVGetBufferSize(handle.get(), operation, &alpha, mat.cget(),
        x.cget(), &beta, y.get(), ACL_INT32, algorithm, &bytes), ACL_SPARSE_STATUS_SUCCESS);
    auto workspace = DeviceBuffer::alloc(bytes);
    ASSERT_TRUE(RunSpmvWithWorkspace(handle, operation, &alpha, &beta, mat, x, y,
        ACL_INT32, algorithm, workspace.raw(), stream));
    dY.copyToHost(yHost.data(), yHost.size() * sizeof(int32_t));
    EXPECT_EQ(yHost[0], std::numeric_limits<int32_t>::max());
    EXPECT_EQ(yHost[1], std::numeric_limits<int32_t>::min());
    for (size_t i = 2; i < yHost.size(); ++i) {
        EXPECT_EQ(yHost[i], 0);
    }
}

TEST_F(SpmvArch35ExceptionTest, SaturatesPositiveAndNegativeInt32Overflow) {
    for (const auto operation : {ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
                                 ACL_SPARSE_OP_CONJUGATE_TRANSPOSE}) {
        for (const auto algorithm : {ACL_SPARSE_SPMV_ALG_DEFAULT, ACL_SPARSE_SPMV_CSR_ALG2}) {
            for (const int32_t rows : {2, 4096}) {
                CheckInt32Saturation(env_->stream(), rows, operation, algorithm);
            }
        }
    }
}

TEST_F(SpmvArch35ExceptionTest, RoundsHalfwayResultsToEven) {
    HandleManager handle;
    handle.setStream(env_->stream());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const std::vector<int32_t> rows{0, 2, 4};
    const std::vector<int32_t> cols{0, 1, 0, 1};
    const std::vector<uint16_t> values{0x3C00U, 0x1000U, 0xBC00U, 0x9000U};
    const std::vector<uint16_t> xHost{0x3C00U, 0x3C00U};
    const std::vector<uint16_t> yHost{0U, 0U};
    auto dRows = DeviceBuffer::copyFrom(rows.data(), rows.size() * sizeof(int32_t));
    auto dCols = DeviceBuffer::copyFrom(cols.data(), cols.size() * sizeof(int32_t));
    auto dVals = DeviceBuffer::copyFrom(values.data(), values.size() * sizeof(uint16_t));
    auto dX = DeviceBuffer::copyFrom(xHost.data(), xHost.size() * sizeof(uint16_t));
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(uint16_t));
    auto mat = SpMatManager::createCsr(2, 2, 4, dRows.raw(), dCols.raw(), dVals.raw(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT16);
    auto x = DnVecManager::create(2, dX.raw(), ACL_FLOAT16);
    auto y = DnVecManager::create(2, dY.raw(), ACL_FLOAT16);
    size_t workspaceSize = 0U;
    ASSERT_EQ(aclsparseSpMVGetBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mat.cget(), x.cget(), &beta, y.get(), ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT, &workspaceSize), ACL_SPARSE_STATUS_SUCCESS);
    auto workspace = DeviceBuffer::alloc(workspaceSize);
    ASSERT_TRUE(RunSpmvWithWorkspace(
        handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, &beta, mat, x, y,
        ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.raw(), env_->stream()));
    std::vector<uint16_t> result(2);
    dY.copyToHost(result.data(), result.size() * sizeof(uint16_t));
    EXPECT_EQ(result[0], 0x3C00U);
    EXPECT_EQ(result[1], 0xBC00U);
}

TEST_F(SpmvArch35ExceptionTest, ComplexConjugateTranspose) {
    HandleManager handle;
    handle.setStream(env_->stream());
    const aclsparseComplex alpha{1.0f, 0.0f};
    const aclsparseComplex beta{0.0f, 0.0f};
    const std::vector<int32_t> rows{0, 2, 3};
    const std::vector<int32_t> cols{0, 1, 1};
    const std::vector<aclsparseComplex> vals{{1.0f, 2.0f}, {3.0f, -1.0f},
                                              {2.0f, 4.0f}};
    const std::vector<aclsparseComplex> xHost{{1.0f, -1.0f}, {2.0f, 3.0f}};
    const std::vector<aclsparseComplex> yHost{{9.0f, 8.0f}, {7.0f, 6.0f}};
    auto dRows = DeviceBuffer::copyFrom(rows.data(), rows.size() * sizeof(int32_t));
    auto dCols = DeviceBuffer::copyFrom(cols.data(), cols.size() * sizeof(int32_t));
    auto dVals = DeviceBuffer::copyFrom(vals.data(), vals.size() * sizeof(aclsparseComplex));
    auto dX = DeviceBuffer::copyFrom(xHost.data(), xHost.size() * sizeof(aclsparseComplex));
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(aclsparseComplex));
    auto mat = SpMatManager::createCsr(2, 2, 3, dRows.raw(), dCols.raw(), dVals.raw(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
        ACL_COMPLEX64);
    auto x = DnVecManager::create(2, dX.raw(), ACL_COMPLEX64);
    auto y = DnVecManager::create(2, dY.raw(), ACL_COMPLEX64);
    size_t workspaceSize = 0;
    ASSERT_EQ(aclsparseSpMVGetBufferSize(handle.get(), ACL_SPARSE_OP_CONJUGATE_TRANSPOSE,
        &alpha, mat.cget(), x.cget(), &beta, y.get(), ACL_COMPLEX64,
        ACL_SPARSE_SPMV_CSR_ALG2, &workspaceSize), ACL_SPARSE_STATUS_SUCCESS);
    auto workspace = DeviceBuffer::alloc(workspaceSize);
    ASSERT_TRUE(RunSpmvWithWorkspace(
        handle, ACL_SPARSE_OP_CONJUGATE_TRANSPOSE, &alpha, &beta, mat, x, y,
        ACL_COMPLEX64, ACL_SPARSE_SPMV_CSR_ALG2, workspace.raw(), env_->stream()));
    std::vector<aclsparseComplex> result(2);
    dY.copyToHost(result.data(), result.size() * sizeof(aclsparseComplex));
    EXPECT_FLOAT_EQ(result[0].x, -1.0f);
    EXPECT_FLOAT_EQ(result[0].y, -3.0f);
    EXPECT_FLOAT_EQ(result[1].x, 20.0f);
    EXPECT_FLOAT_EQ(result[1].y, -4.0f);
}

}  // namespace
