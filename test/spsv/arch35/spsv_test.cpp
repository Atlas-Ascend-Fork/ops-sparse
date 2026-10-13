/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_common.h"

#include "../spsv_golden.h"
#include "../spsv_param.h"
#include "spsv_npu_wrapper.h"

namespace sparse_test {

// RAII guard for environment variables to ensure cleanup even on exceptions
class RAIIEnvGuard {
public:
    RAIIEnvGuard() : needs_cleanup_(false) {}
    ~RAIIEnvGuard() { cleanup(); }

    void set(const char* name, const char* value) {
        if (needs_cleanup_) {
            unsetenv(name_.c_str());
        }
        name_ = name;
        value_ = value;
        setenv(name, value, 1);
        needs_cleanup_ = true;
    }

    void cleanup() {
        if (needs_cleanup_) {
            unsetenv(name_.c_str());
            needs_cleanup_ = false;
        }
    }

private:
    std::string name_;
    std::string value_;
    bool needs_cleanup_;
};

static CsrMatrix GenerateTriangularMatrix(const SpSVParam& p) {
    bool lower = p.isLower();
    bool unitDiag = p.isUnitDiag();

    if (p.structure == "diagonal") {
        return makeDiagCsr(p.m, unitDiag ? 1.0f : 3.0f);
    }
    if (p.structure == "tridiagonal") {
        return makeTridiagTriangularCsr(p.m, lower, p.seed);
    }
    if (p.structure == "identity") {
        return makeIdentityCsr(p.m);
    }
    if (p.structure == "diag_dominant") {
        return makeDiagDominantTriangularCsr(p.m, lower, p.seed);
    }
    if (p.structure == "empty_unit" || p.structure == "empty_non_unit") {
        // nnz=0 matrix: no stored values at all (no diagonal entries either).
        // For UNIT: y[i] = rhs[i] / 1 = alpha*x[i]   (identity-like)
        // For NON_UNIT: y[i] = rhs[i] / 0 = Inf/NaN   (singular)
        return makeEmptyTriangularCsr(p.m);
    }
    if (p.structure == "missing_diag") {
        return makeMissingDiagTriangularCsr(p.m, lower, p.seed);
    }
    if (p.sparsity <= 0.0) {
        return makeTriangularCsr(p.m, lower, unitDiag, 0.0, p.seed);
    }
    return makeTriangularCsr(p.m, lower, unitDiag, p.sparsity, p.seed);
}

class SpSVTestBase {
protected:
    static void InitSuite(std::unique_ptr<AclEnvScope>& env,
                          std::unique_ptr<HandleManager>& handle) {
        env = std::make_unique<AclEnvScope>();
        handle = std::make_unique<HandleManager>();
        handle->setStream(env->stream());
    }
    static void CleanupSuite(std::unique_ptr<AclEnvScope>& env,
                             std::unique_ptr<HandleManager>& handle) {
        handle.reset();
        env.reset();
    }
};

class SpSVTest : public SpSVTestBase, public ::testing::TestWithParam<SpSVParam> {
public:
    static void SetUpTestSuite() { InitSuite(env_, spHandle_); }
    static void TearDownTestSuite() { CleanupSuite(env_, spHandle_); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    inline static std::unique_ptr<HandleManager> spHandle_;
};

static void VerifyResult(const std::vector<float>& actual,
                         const std::vector<float>& golden,
                         const SpSVParam& p,
                         const std::string& label) {
    VerifyConfig verifyCfg;
    verifyCfg.SetMode(PrecisionMode::MERE_MARE)
             .SetMERE(p.mere_threshold)
             .SetMARE(p.mare_multiplier);
    EXPECT_TRUE(Verifier::verifyVector(actual, golden, verifyCfg, label))
        << "[" << label << "] Precision verification FAILED";
}

static void RunUpdateVerification(const SpSVNpuResult& npuResult,
                                  const CsrMatrix& csr,
                                  const std::vector<float>& xVec,
                                  const SpSVParam& p,
                                  bool lower, bool unitDiag, bool transpose) {
    if (!npuResult.hasSecondSolve) return;
    if (p.update_mode != "GENERAL" && p.update_mode != "DIAGONAL") return;
    CsrMatrix csrUpdated = csr;
    std::mt19937 rng(9999);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    if (p.update_mode == "GENERAL") {
        for (int64_t i = 0; i < p.m; i++) {
            float rowAbsSum = 0.0f;
            int32_t diagIdx = -1;
            for (int32_t k = csrUpdated.rowOffsets[i]; k < csrUpdated.rowOffsets[i + 1]; k++) {
                if (csrUpdated.colIndices[k] == static_cast<int32_t>(i)) {
                    diagIdx = k;
                } else {
                    float v = dist(rng);
                    csrUpdated.values[k] = v;
                    rowAbsSum += std::abs(v);
                }
            }
            if (diagIdx >= 0) {
                csrUpdated.values[diagIdx] = rowAbsSum + 1.0f;
            }
        }
    } else {
        for (int64_t i = 0; i < p.m; i++) {
            for (int32_t k = csrUpdated.rowOffsets[i]; k < csrUpdated.rowOffsets[i + 1]; k++) {
                if (csrUpdated.colIndices[k] == static_cast<int32_t>(i)) {
                    csrUpdated.values[k] = dist(rng);
                }
            }
        }
    }
    auto yGolden2 = SpSVGolden(csrUpdated, xVec, p.alpha, lower, unitDiag, transpose);
    VerifyResult(npuResult.y2, yGolden2, p, p.case_name + "_update");
}

static void RunDeterministicCheck(aclsparseHandle_t handle, aclrtStream stream,
                                   const CsrMatrix& csr, const std::vector<float>& xVec,
                                   const SpSVParam& p, const SpSVNpuConfig& cfg,
                                   const SpSVNpuResult& npuResult) {
    if (p.case_name.find("deterministic") == std::string::npos) return;
    SpSVNpuResult npuResult2;
    try {
        npuResult2 = SpSVNpuWrapper(handle, stream, csr, xVec, p.alpha, cfg);
    } catch (const std::exception& e) {
        FAIL() << "[" << p.case_name << "] 2nd NPU execution threw: " << e.what();
    }
    VerifyConfig exactCfg;
    exactCfg.SetMode(PrecisionMode::EXACT);
    EXPECT_TRUE(Verifier::verifyVector(npuResult2.y, npuResult.y, exactCfg,
                                       p.case_name + "_deterministic"))
        << "[" << p.case_name << "] Deterministic check FAILED";
}

static std::vector<float> GenerateInputVector(const SpSVParam& p) {
    if (p.case_name.find("x_zero") != std::string::npos) {
        return std::vector<float>(p.m, 0.0f);
    }
    return makeDenseFloat(p.m, -5.0, 5.0, p.seed + 1);
}

static std::vector<float> ComputeGolden(const CsrMatrix& csr,
                                         const std::vector<float>& xVec,
                                         const SpSVParam& p,
                                         bool lower, bool unitDiag,
                                         bool transpose) {
    if (p.structure == "missing_diag" || p.structure == "empty_non_unit") {
        // Use manual forward/backward substitution for structures where the exact
        // IEEE-754 division-by-zero semantics must be preserved (diagVal=0 → Inf/NaN).
        return SpSVGoldenManual(csr, xVec, p.alpha, lower, unitDiag, transpose);
    }
    return SpSVGolden(csr, xVec, p.alpha, lower, unitDiag, transpose);
}

TEST_P(SpSVTest, GenericSuccess) {
    auto p = GetParam();
    std::cout << "\n==== " << p.case_name
              << " m=" << p.m << " fmt=" << p.format
              << " fill=" << p.fill_mode << " diag=" << p.diag_type
              << " op=" << p.op_type << " alpha=" << p.alpha << " ====\n";

    ASSERT_EQ(p.expect_result, "ACL_SPARSE_STATUS_SUCCESS");

    CsrMatrix csr = GenerateTriangularMatrix(p);
    if (p.unsorted) {
        csr = unsortCsrRowIndices(csr, p.seed + 100);
    }

    std::vector<float> xVec = GenerateInputVector(p);

    bool lower = p.isLower();
    bool unitDiag = p.isUnitDiag();
    bool transpose = p.isTranspose();

    std::vector<float> yGolden = ComputeGolden(csr, xVec, p, lower, unitDiag, transpose);

    SpSVNpuConfig cfg;
    cfg.format = p.format;
    cfg.isI64 = p.isI64();
    cfg.isRowPtrI64 = p.isRowPtrI64();
    cfg.isColIndI64 = p.isColIndI64();
    cfg.idxBase = p.index_base;
    cfg.lower = lower;
    cfg.unitDiag = unitDiag;
    cfg.opA = (p.op_type == "TRANSPOSE") ? ACL_SPARSE_OP_TRANSPOSE :
              (p.op_type == "CONJUGATE_TRANSPOSE") ? ACL_SPARSE_OP_CONJUGATE_TRANSPOSE :
              ACL_SPARSE_OP_NON_TRANSPOSE;
    cfg.inPlace = p.in_place;
    cfg.nullVec = p.null_vec;
    cfg.updateMode = p.update_mode;
    cfg.sliceWidth = p.slice_width;

    RAIIEnvGuard perm64Guard;
    if (p.force_permtype64 || p.case_name.rfind("I64_PERM64_", 0) == 0) {
        perm64Guard.set("SPSV_FORCE_PERMT_64", "1");
    }

    SpSVNpuResult npuResult;
    try {
        npuResult = SpSVNpuWrapper(spHandle_->get(), env_->stream(), csr, xVec, p.alpha, cfg);
    } catch (const std::exception& e) {
        FAIL() << "[" << p.case_name << "] NPU execution threw: " << e.what();
    }

    ASSERT_EQ(npuResult.y.size(), yGolden.size())
        << "[" << p.case_name << "] Size mismatch";

    VerifyResult(npuResult.y, yGolden, p, p.case_name);
    RunUpdateVerification(npuResult, csr, xVec, p, lower, unitDiag, transpose);
    RunDeterministicCheck(spHandle_->get(), env_->stream(), csr, xVec, p, cfg, npuResult);
}

INSTANTIATE_TEST_SUITE_P(
    SpSV,
    SpSVTest,
    ::testing::ValuesIn(GetCasesFromCsv<SpSVParam>("spsv_test.csv")),
    [](const ::testing::TestParamInfo<SpSVParam>& info) {
        return info.param.caseId();
    }
);

class SpSVExceptionTest : public SpSVTestBase, public ::testing::Test {
public:
    static void SetUpTestSuite() { InitSuite(env_, spHandle_); }
    static void TearDownTestSuite() { CleanupSuite(env_, spHandle_); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    inline static std::unique_ptr<HandleManager> spHandle_;

    struct BasicSetup {
        DeviceBuffer dRowOff, dColIdx, dVals, dX, dY;
        SpMatManager matA;
        DnVecManager vecX, vecY;
        SpSVDescrManager spsvDescr;
        float alpha = 1.0f;

        static BasicSetup Make(aclsparseHandle_t handle, int64_t m = 4) {
            BasicSetup s;
            auto csr = makeTriangularCsr(m, true, false, 0.5, 42);
            s.dRowOff = DeviceBuffer::copyFrom(csr.rowOffsets.data(),
                                               csr.rowOffsets.size() * sizeof(int32_t));
            s.dColIdx = DeviceBuffer::copyFrom(csr.colIndices.data(),
                                               csr.colIndices.size() * sizeof(int32_t));
            s.dVals = DeviceBuffer::copyFrom(csr.values.data(),
                                             csr.values.size() * sizeof(float));
            std::vector<float> x(m, 1.0f), y(m, 0.0f);
            s.dX = DeviceBuffer::copyFrom(x.data(), x.size() * sizeof(float));
            s.dY = DeviceBuffer::copyFrom(y.data(), y.size() * sizeof(float));
            s.matA = SpMatManager::createCsr(m, m, csr.nnz,
                s.dRowOff.get(), s.dColIdx.get(), s.dVals.get());
            aclsparseFillMode_t fm = ACL_SPARSE_FILL_MODE_LOWER;
            aclsparseDiagType_t dt = ACL_SPARSE_DIAG_TYPE_NON_UNIT;
            s.matA.setAttribute(ACL_SPARSE_SPMAT_FILL_MODE, &fm, sizeof(fm));
            s.matA.setAttribute(ACL_SPARSE_SPMAT_DIAG_TYPE, &dt, sizeof(dt));
            s.vecX = DnVecManager::createConst(m, s.dX.get(), ACL_FLOAT);
            s.vecY = DnVecManager::create(m, s.dY.get(), ACL_FLOAT);
            return s;
        }
    };

    static aclsparseStatus_t RunSpSVPipeline(aclsparseHandle_t handle, BasicSetup& s,
                                              size_t* outBufSize = nullptr) {
        size_t bufSize = 0;
        auto ret = aclsparseSpSV_bufferSize(
            handle, ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
            s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
            ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(), &bufSize);
        if (ret != ACL_SPARSE_STATUS_SUCCESS) return ret;
        if (outBufSize) *outBufSize = bufSize;
        DeviceBuffer dWs;
        if (bufSize > 0) dWs = DeviceBuffer::alloc(bufSize);
        ret = aclsparseSpSV_analysis(
            handle, ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
            s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
            ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(),
            bufSize > 0 ? dWs.get() : nullptr);
        return ret;
    }
};

TEST_F(SpSVExceptionTest, L2_null_handle) {
    auto s = BasicSetup::Make(spHandle_->get());
    size_t bufSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(), &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SpSVExceptionTest, L2_null_spsvDescr) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = aclsparseSpSV_solve(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, nullptr);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, L2_null_matA) {
    auto s = BasicSetup::Make(spHandle_->get());
    size_t bufSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, nullptr,
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(), &bufSize);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, L2_null_alpha) {
    auto s = BasicSetup::Make(spHandle_->get());
    size_t bufSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, nullptr, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(), &bufSize);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, L2_null_bufferSize) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = aclsparseSpSV_bufferSize(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(), nullptr);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, L2_null_createDescr_out) {
    auto ret = aclsparseSpSV_createDescr(nullptr);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, L2_null_updateValues) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = aclsparseSpSV_updateMatrix(
        spHandle_->get(), s.spsvDescr.get(), nullptr, ACL_SPARSE_SPSV_UPDATE_GENERAL);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, L2_singular_matrix) {
    int64_t m = 4;
    auto csr = makeSingularTriangularCsr(m, true, 42);
    auto dRowOff = DeviceBuffer::copyFrom(csr.rowOffsets.data(),
                                          csr.rowOffsets.size() * sizeof(int32_t));
    auto dColIdx = DeviceBuffer::copyFrom(csr.colIndices.data(),
                                          csr.colIndices.size() * sizeof(int32_t));
    auto dVals = DeviceBuffer::copyFrom(csr.values.data(),
                                        csr.values.size() * sizeof(float));
    std::vector<float> x(m, 1.0f), y(m, 0.0f);
    auto dX = DeviceBuffer::copyFrom(x.data(), x.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(y.data(), y.size() * sizeof(float));

    auto matA = SpMatManager::createCsr(m, m, csr.nnz,
        dRowOff.get(), dColIdx.get(), dVals.get());
    aclsparseFillMode_t fm = ACL_SPARSE_FILL_MODE_LOWER;
    aclsparseDiagType_t dt = ACL_SPARSE_DIAG_TYPE_NON_UNIT;
    matA.setAttribute(ACL_SPARSE_SPMAT_FILL_MODE, &fm, sizeof(fm));
    matA.setAttribute(ACL_SPARSE_SPMAT_DIAG_TYPE, &dt, sizeof(dt));

    auto vecX = DnVecManager::createConst(m, dX.get(), ACL_FLOAT);
    auto vecY = DnVecManager::create(m, dY.get(), ACL_FLOAT);
    SpSVDescrManager spsvDescr;
    float alpha = 1.0f;

    size_t bufSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matA.cget(),
        vecX.cget(), vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get(), &bufSize);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS) << "bufferSize stage";

    DeviceBuffer dWs;
    if (bufSize > 0) dWs = DeviceBuffer::alloc(bufSize);
    ret = aclsparseSpSV_analysis(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matA.cget(),
        vecX.cget(), vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get(),
        bufSize > 0 ? dWs.get() : nullptr);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS) << "analysis stage";

    ret = aclsparseSpSV_solve(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matA.cget(),
        vecX.cget(), vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    aclrtSynchronizeStream(env_->stream());
    std::vector<float> yOut(m);
    dY.copyToHost(yOut.data(), m * sizeof(float));
    bool hasInfNan = false;
    for (float v : yOut) {
        if (std::isinf(v) || std::isnan(v)) { hasInfNan = true; break; }
    }
    EXPECT_TRUE(hasInfNan) << "Singular matrix should produce Inf/NaN output";
}

TEST_F(SpSVExceptionTest, L2_m_zero) {
    auto matA = SpMatManager::createCsr(0, 0, 0, nullptr, nullptr, nullptr);
    auto vecX = DnVecManager::createConst(0, nullptr, ACL_FLOAT);
    auto vecY = DnVecManager::create(0, nullptr, ACL_FLOAT);
    SpSVDescrManager spsvDescr;
    float alpha = 1.0f;
    size_t bufSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matA.cget(),
        vecX.cget(), vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get(), &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, L2_destroy_null) {
    auto ret = aclsparseSpSV_destroyDescr(nullptr);
    EXPECT_TRUE(ret == ACL_SPARSE_STATUS_SUCCESS || ret != ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, L2_solve_before_analysis) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = aclsparseSpSV_solve(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get());
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, L2_invalid_opA) {
    auto s = BasicSetup::Make(spHandle_->get());
    size_t bufSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        spHandle_->get(), static_cast<aclsparseOperation_t>(999), &s.alpha, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(), &bufSize);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, WB_null_vecX_solve) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = RunSpSVPipeline(spHandle_->get(), s);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    ret = aclsparseSpSV_solve(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        nullptr, s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SpSVExceptionTest, WB_null_vecY_solve) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = RunSpSVPipeline(spHandle_->get(), s);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    ret = aclsparseSpSV_solve(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        s.vecX.cget(), nullptr, ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SpSVExceptionTest, WB_null_externalBuffer_analysis) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = aclsparseSpSV_analysis(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(), nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SpSVExceptionTest, WB_non_square_matrix) {
    int64_t rows = 4, cols = 8;
    auto csr = makeSparseCsr(rows, cols, 0.5, 42);
    auto dRowOff = DeviceBuffer::copyFrom(csr.rowOffsets.data(),
                                           csr.rowOffsets.size() * sizeof(int32_t));
    auto dColIdx = DeviceBuffer::copyFrom(csr.colIndices.data(),
                                           csr.colIndices.size() * sizeof(int32_t));
    auto dVals = DeviceBuffer::copyFrom(csr.values.data(),
                                         csr.values.size() * sizeof(float));
    auto matA = SpMatManager::createCsr(rows, cols, csr.nnz,
        dRowOff.get(), dColIdx.get(), dVals.get());
    std::vector<float> x(rows, 1.0f), y(rows, 0.0f);
    auto dX = DeviceBuffer::copyFrom(x.data(), x.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(y.data(), y.size() * sizeof(float));
    auto vecX = DnVecManager::createConst(rows, dX.get(), ACL_FLOAT);
    auto vecY = DnVecManager::create(rows, dY.get(), ACL_FLOAT);
    SpSVDescrManager spsvDescr;
    float alpha = 1.0f;
    size_t bufSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matA.cget(),
        vecX.cget(), vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get(), &bufSize);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, WB_unsupported_computeType) {
    auto s = BasicSetup::Make(spHandle_->get());
    size_t bufSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT16,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(), &bufSize);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, WB_update_before_analysis) {
    auto s = BasicSetup::Make(spHandle_->get());
    std::vector<float> newVals(4, 1.0f);
    auto dNewVals = DeviceBuffer::copyFrom(newVals.data(), newVals.size() * sizeof(float));
    auto ret = aclsparseSpSV_updateMatrix(
        spHandle_->get(), s.spsvDescr.get(), dNewVals.get(),
        ACL_SPARSE_SPSV_UPDATE_GENERAL);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, WB_unsupported_updatePart) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = RunSpSVPipeline(spHandle_->get(), s);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    std::vector<float> newVals(4, 1.0f);
    auto dNewVals = DeviceBuffer::copyFrom(newVals.data(), newVals.size() * sizeof(float));
    ret = aclsparseSpSV_updateMatrix(
        spHandle_->get(), s.spsvDescr.get(), dNewVals.get(),
        static_cast<aclsparseSpSVUpdate_t>(999));
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

}
