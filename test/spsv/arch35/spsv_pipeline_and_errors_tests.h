/**
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

#ifndef SPSV_PIPELINE_AND_ERRORS_TESTS_H_
#define SPSV_PIPELINE_AND_ERRORS_TESTS_H_

static void VerifyRealInPlaceCases(
    aclsparseHandle_t handle, aclrtStream stream, const CsrMatrix &upload,
    const std::vector<float> &xVec, const std::vector<float> &yGolden,
    const SpSVParam &p, const SpSVNpuConfig &cfg, const SpSVNpuResult &npuResult)
{
    if (p.structure == "rhscancelf" && p.in_place) {
        SpSVNpuConfig separateCfg = cfg;
        separateCfg.inPlace = false;
        auto separate = SpSVNpuWrapper(handle, stream, upload,
                                       xVec, p.alpha, separateCfg);
        ASSERT_EQ(npuResult.y.size(), separate.y.size());
        for (size_t i = 0; i < npuResult.y.size(); ++i) {
            EXPECT_EQ(npuResult.y[i], separate.y[i])
                << "[" << p.case_name << "] in-place row " << i;
        }
    }
    if (p.structure == "multilevel_inplace" && p.in_place) {
        SpSVNpuConfig separateCfg = cfg;
        separateCfg.inPlace = false;
        for (int repeat = 0; repeat < 3; ++repeat) {
            auto rerun = SpSVNpuWrapper(handle, stream, upload,
                                        xVec, p.alpha, cfg);
            auto separate = SpSVNpuWrapper(handle, stream, upload,
                                           xVec, p.alpha, separateCfg);
            ASSERT_EQ(rerun.y.size(), separate.y.size());
            for (size_t i = 0; i < rerun.y.size(); ++i) {
                EXPECT_EQ(rerun.y[i], separate.y[i])
                    << "[" << p.case_name << "] repeat " << repeat << " row " << i;
                EXPECT_EQ(rerun.y[i], yGolden[i])
                    << "[" << p.case_name << "] repeat " << repeat << " golden row " << i;
            }
        }
    }
}

static void VerifyRealSpecialResults(const SpSVParam &p, const SpSVNpuResult &npuResult)
{
    if (p.structure == "empty_non_unit" && p.alpha == 0.0f) {
        for (size_t i = 0; i < npuResult.y.size(); ++i) {
            EXPECT_TRUE(std::isnan(npuResult.y[i]))
                << "[" << p.case_name << "] row " << i
                << " should preserve 0/0 IEEE-754 behavior";
        }
    }
    if (p.structure == "rhscancel" || p.structure == "rhscancel_normal") {
        ASSERT_FALSE(npuResult.y.empty());
        EXPECT_EQ(npuResult.y.back(), -1.0f);
    }
}

TEST_P(SpSVTest, GenericSuccess) {
    auto p = GetParam();
    std::cout << "\n==== " << p.case_name
              << " m=" << p.m << " fmt=" << p.format
              << " fill=" << p.fill_mode << " diag=" << p.diag_type
              << " op=" << p.op_type << " alpha=" << p.alpha << " ====\n";

    CsrMatrix csr = GenerateTriangularMatrix(p);
    if (p.expect_result == "ACL_SPARSE_STATUS_NOT_SUPPORTED") {
        ASSERT_TRUE(p.isComplex());
        VerifyUnsupportedComplex(p, csr, spHandle_->get());
        return;
    }
    ASSERT_EQ(p.expect_result, "ACL_SPARSE_STATUS_SUCCESS");
    if (p.unsorted) {
        csr = unsortCsrRowIndices(csr, p.seed + 100);
    }

    if (p.isComplex()) {
        RunComplexCase(p, csr, spHandle_->get(), env_->stream());
        return;
    }

    std::vector<float> xVec = GenerateInputVector(p);

    bool lower = p.isLower();
    bool unitDiag = p.isUnitDiag();
    bool transpose = p.isTranspose();

    std::vector<float> yGolden = ComputeGolden(csr, xVec, p, lower, unitDiag, transpose);

    SpSVNpuConfig cfg = MakeConfig(p);
    cfg.nullVec = p.null_vec;

    RAIIEnvGuard perm64Guard;
    if (p.force_permtype64 || p.case_name.rfind("I64_PERM64_", 0) == 0) {
        perm64Guard.set("SPSV_FORCE_PERMT_64", "1");
    }

    // yGolden above is the clean matrix.  The illegal-column cases hand the
    // device one extra COO entry that both builders must skip, so the answer
    // has to be the clean one either way; see AppendOutOfRangeColumn.
    const bool badCol = p.case_name.rfind("REG_badcol", 0) == 0;
    CsrMatrix upload = badCol ? AppendOutOfRangeColumn(csr, lower) : csr;

    SpSVNpuResult npuResult;
    try {
        npuResult = SpSVNpuWrapper(spHandle_->get(), env_->stream(), upload, xVec, p.alpha, cfg);
    } catch (const std::exception& e) {
        FAIL() << "[" << p.case_name << "] NPU execution threw: " << e.what();
    }

    ASSERT_EQ(npuResult.y.size(), yGolden.size())
        << "[" << p.case_name << "] Size mismatch";

    VerifyResult(npuResult.y, yGolden, p, p.case_name);
    ASSERT_NO_FATAL_FAILURE(VerifyRealInPlaceCases(spHandle_->get(), env_->stream(),
        upload, xVec, yGolden, p, cfg, npuResult));
    ASSERT_NO_FATAL_FAILURE(VerifyRealSpecialResults(p, npuResult));
    RunUpdateVerification(npuResult, csr, xVec, p, lower, unitDiag, transpose);
    RunDeterministicCheck(spHandle_->get(), env_->stream(), upload, xVec, p, cfg, npuResult);
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

// Closed-form expectations avoid repeating cancellation in the reference.
static void AddRealReviewEntry(CsrMatrix &a, int col, float value)
{
    a.colIndices.push_back(col);
    a.values.push_back(value);
}

static void AddSmallCancellationRow(
    CsrMatrix &a, std::vector<float> &x, std::vector<float> &expected,
    int kind, int row, int m)
{
    // Cancellation before division must be certified from the large
    // operands, not only from the small rounded numerator.
    if (kind == 6 && row == 0) {
        x[row] = 1.0f + 0x1p-12f;
        expected[row] = x[row];
    }
    if (kind == 6 && row == 1) {
        AddRealReviewEntry(a, 0, 1074003968.0f);
        x[row] = 1074298880.0f;
        expected[row] = 32704.0f;
    }
    if (kind == 0 && row == 4) {
        AddRealReviewEntry(a, 0, 0x1p25f); AddRealReviewEntry(a, 1, 1.0f); AddRealReviewEntry(a, 2, -0x1p25f);
        x[row] = 4.0f; expected[row] = 3.0f;
    }
    if (kind == 0 && row == 5) {
        AddRealReviewEntry(a, 4, -1.0f); x[row] = 0.0f; expected[row] = 3.0f;
    }
    if (kind == 1 && row == 5) {
        AddRealReviewEntry(a, 0, 0x1p100f); AddRealReviewEntry(a, 1, 0x1p50f); AddRealReviewEntry(a, 2, 1.0f);
        AddRealReviewEntry(a, 3, -0x1p100f); AddRealReviewEntry(a, 4, -0x1p50f);
        x[row] = 0.0f; expected[row] = -1.0f;
    }
    if (kind == 1 && row == 6) {
        AddRealReviewEntry(a, 5, -1.0f); x[row] = 0.0f; expected[row] = -1.0f;
    }
}

static void AddRangeAndAliasRow(
    CsrMatrix &a, std::vector<float> &x, std::vector<float> &expected,
    int kind, int row, int m)
{
    if (kind == 2) {
        x[row] = row < 2 ? 2.0f : 1.0f;
        expected[row] = x[row];
        if (row == 2) {
            AddRealReviewEntry(a, 0, 0x1p127f); AddRealReviewEntry(a, 1, -0x1p127f);
        }
    }
    if (kind == 3) {
        expected[row] = row % 2 ? 5.0f : -5.0f;
        x[row] = expected[row];
        if (row >= 64) {
            AddRealReviewEntry(a, row - 64, 1.0f); x[row] += expected[row];
        }
    }
}

static void AddWideCancellationRow(
    CsrMatrix &a, std::vector<float> &x, std::vector<float> &expected,
    int kind, int row, int m)
{
    // Exercise the same three-tier loss in warp and wide-row paths.
    if (kind >= 4 && kind != 6 && row == m - 2) {
        for (int col = 0; col < row; ++col) {
            float v = col == 0 ? 0x1p100f : col == 1 ? 0x1p50f :
                col == 2 ? 1.0f : col == 3 ? -0x1p100f :
                col == 4 ? -0x1p50f : 0.0f;
            AddRealReviewEntry(a, col, v);
        }
        x[row] = 0.0f; expected[row] = -1.0f;
    }
    if (kind >= 4 && kind != 6 && row == m - 1) {
        AddRealReviewEntry(a, row - 1, -1.0f); x[row] = 0.0f; expected[row] = -1.0f;
    }
}

TEST_F(SpSVExceptionTest, ReviewCancellationAndAlias) {
    for (int kind = 0; kind < 7; ++kind) {
        int m = kind == 0 ? 6 : kind == 1 ? 7 : kind == 2 ? 3 : kind == 3 ? 256 :
            kind == 4 ? 40 : kind == 5 ? 300 : 2;
        CsrMatrix a;
        a.rows = a.cols = m;
        a.rowOffsets = {0};
        std::vector<float> x(m, 1.0f), expected(m, 1.0f);
        for (int row = 0; row < m; ++row) {
            AddSmallCancellationRow(a, x, expected, kind, row, m);
            AddRangeAndAliasRow(a, x, expected, kind, row, m);
            AddWideCancellationRow(a, x, expected, kind, row, m);
            AddRealReviewEntry(a, row, 1.0f);
            a.rowOffsets.push_back(static_cast<int32_t>(a.values.size()));
        }
        a.nnz = static_cast<int64_t>(a.values.size());
        std::vector<std::complex<float>> values, xc;
        bool complexTerms = kind >= 4 && kind != 6;
        for (float value : a.values) { values.emplace_back(value, complexTerms ? value : 0.0f); }
        for (float value : x) { xc.emplace_back(value, complexTerms ? value : 0.0f); }
        for (bool inplace : {false, true}) {
            SpSVNpuConfig cfg;
            cfg.inPlace = inplace;
            auto real = SpSVNpuWrapper(spHandle_->get(), env_->stream(), a, x, 1.0f, cfg);
            auto complex = SpSVNpuWrapperComplex(spHandle_->get(), env_->stream(),
                a, values, xc, {1.0f, 0.0f}, cfg);
            for (int row = 0; row < m; ++row) {
                EXPECT_EQ(real.y[row], expected[row]) << kind << ":" << inplace << ":" << row;
                EXPECT_EQ(complex.y[row].real(), expected[row]) << kind << ":" << inplace << ":" << row;
                EXPECT_EQ(complex.y[row].imag(), 0.0f) << kind << ":" << inplace << ":" << row;
            }
        }
    }
}

static void VerifyRealProductRounding(
    aclsparseHandle_t handle, aclrtStream stream, const CsrMatrix &a,
    const std::vector<float> &x, const std::vector<float> &expected, int label)
{
    for (bool inplace : {false, true}) {
        SpSVNpuConfig cfg;
        cfg.inPlace = inplace;
        auto result = SpSVNpuWrapper(handle, stream, a, x, 1.0f, cfg);
        for (int row = 0; row < a.rows; ++row) {
            EXPECT_EQ(result.y[row], expected[row]) << label << ":" << inplace << ":" << row;
        }
    }
}


#endif // SPSV_PIPELINE_AND_ERRORS_TESTS_H_
