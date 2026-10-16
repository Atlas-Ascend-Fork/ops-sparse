/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * @file sddmm_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseSDDMM (Generic API).
 *
 * Operator: C_out = (alpha * X * op(Y) + beta * C) ∘ spy(C)
 *   X: m×k dense (row-major), Y: k×n or n×k dense (row-major, per opY), C: m×n CSR (values in place)
 *
 * Test structure:
 *   - TEST_P (SddmmTest)           : parameterized success-path tests from CSV
 *   - TEST_F (SddmmExceptionTest)  : null-pointer / invalid-param error tests (E1-E9)
 *
 * Precision: framework MIXED_TOLERANCE mode (test/frame/verify.h), dtype-driven
 *   atol/rtol + per-element max(abs_err) limit = max(fixedValue, 32*ULP).
 *
 * Entry point is shared via test/frame/test_main.cpp.
 */

#include "test_common.h"
#include "sddmm_golden.h"
#include "sddmm_npu_wrapper.h"
#include "sddmm_param.h"
#include "../sddmm_test_layout.h"

#include <random>

using namespace sparse_test;

// Shared fixture, input conversion and layout helpers serve both architectures.
using SddmmTest = SddmmFunctionalTestFixture;

// ============================================================================
// Test-body helpers (file-scope). Each helper covers one logical step of the
// TEST_P body so the body stays below the R7 NBNC limit.
//
// Note: ASSERT_* must remain in the TEST_P body — gtest ASSERT macros return
// from the current function, so they cannot abort the test if placed in a
// helper. EXPECT_* (non-aborting) is safe inside helpers.
// ============================================================================

// Step 7: precision verification (MIXED_TOLERANCE, dtype-driven).
// Caller must guarantee npuResult.valuesOut.size() == nnz before invoking.
// FP16 kernel saturates intermediate FP32 result to [-65504, 65504] before
// Cast<float->half> (sddmm_kernel.cpp L293-301). Mirror this saturation in the
// golden so whitebox saturation-path cases (WB_08) align with NPU output. FP32
// path is unaffected (no saturation).
static void VerifySddmmPrecision(const std::string& caseName,
                                  const SddmmNpuResult& npuResult,
                                  const std::vector<double>& expectFp64,
                                  aclDataType dtype, int64_t nnz) {
    if (nnz <= 0) {
        std::cout << "[" << caseName << "] nnz=0 (empty tensor), skip value verify\n";
        return;
    }
    const bool pass = VerifySddmmPrecisionCore(caseName, npuResult, expectFp64, dtype, nnz);
    EXPECT_TRUE(pass) << "MIXED_TOLERANCE verification failed for " << caseName;
}

// Test body: success-path parameterized test
TEST_P(SddmmTest, SddmmFunctional) {
    const auto& p = param_;

    std::cout << "==== " << p.case_name << " ====" << p.description
              << " m=" << p.m << " k=" << p.k << " n=" << p.n
              << " ratio=" << p.sparsity_ratio
              << " alpha=" << p.alpha << " beta=" << p.beta
              << " dtype=" << p.dtype
              << " order_x=" << p.order_x << " order_y=" << p.order_y
              << " value_range=[" << p.value_lo << "," << p.value_hi << "]"
              << " seed=" << p.random_seed << "\n";

    // --- Step 1+2: deterministic sparsity pattern + X/Y (FP64) ---------------
    SddmmInputs inputs = GenerateSddmmInputs(p);
    int64_t nnz = inputs.nnz;
    std::cout << "  nnz=" << nnz << "\n";

    // --- Step 3: parse dtype / op / order enums + golden reference -----------
    aclDataType dtype = ParseDtype(p.dtype);
    aclsparseOperation_t opX = ParseOperation(p.op_x);
    aclsparseOperation_t opY = ParseOperation(p.op_y);
    aclsparseOrder_t orderX = ParseOrder(p.order_x);
    aclsparseOrder_t orderY = ParseOrder(p.order_y);
    std::vector<double> expectFp64 = ComputeGoldenExpect(p, inputs.csrC, inputs.Xf64,
                                                          inputs.Yf64, dtype, opX, opY);

    // --- Step 4: parse remaining ACL enums -----------------------------------
    aclDataType computeType = ParseDtype(p.compute_type);
    aclsparseSDDMMAlg_t alg = ParseSddmmAlg(p.alg);

    // --- Step 5: NPU three-stage workflow ------------------------------------
    SddmmNpuResult npuResult = RunNpuSddmm(p, stream_, inputs.csrC, inputs.Xf64,
                                            inputs.Yf64, dtype, computeType, alg,
                                            opX, opY, orderX, orderY, nnz);

    // --- Step 6: check return codes ------------------------------------------
    ASSERT_EQ(npuResult.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "BufferSize stage failed";
    ASSERT_EQ(npuResult.preprocessRet, ACL_SPARSE_STATUS_SUCCESS)
        << "Preprocess stage failed";
    ASSERT_EQ(npuResult.executeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "Execute stage failed";

    // --- Step 7: precision verification (MIXED_TOLERANCE, dtype-driven) ------
    if (nnz > 0) {
        ASSERT_EQ(npuResult.valuesOut.size(), static_cast<size_t>(nnz));
    }
    VerifySddmmPrecision(p.case_name, npuResult, expectFp64, dtype, nnz);

    std::cout << "[" << p.case_name << "] PASSED (nnz=" << nnz << ")\n";
}

// Parameterized test instantiation from CSV (loaded via csv_loader.h)
INSTANTIATE_TEST_SUITE_P(
    SddmmCases,
    SddmmTest,
    testing::ValuesIn(GetCasesFromCsv<SddmmTestParam>("sddmm_test.csv")),
    [](const testing::TestParamInfo<SddmmTestParam>& info) {
        return info.param.case_name;
    }
);

// ============================================================================
// Exception test fixture: SddmmExceptionTest (E1-E9, parameter validation)
// Tests are TEST_F (not CSV-driven); expect non-SUCCESS error codes.
// ============================================================================

class SddmmExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;
    SddmmCsr csrC_;
    int nnz_ = 0;

    // RAII device buffers and descriptors (auto-released on fixture destruction)
    DeviceBuffer dRowOff_;
    DeviceBuffer dColInd_;
    DeviceBuffer dVals_;
    DeviceBuffer dX_;
    DeviceBuffer dY_;
    DeviceBuffer dBuffer_;
    DnMatManager matX_;
    DnMatManager matY_;
    SpMatManager matC_;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    size_t bufSize_ = 0;

    void SetUp() override {
        stream_ = env_->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        // Small 4×4 CSR for valid-parameter baseline
        csrC_ = MakeSddmmSparsity(4, 4, 0.5, -1.0, 1.0, 42);
        nnz_ = static_cast<int>(csrC_.nnz);

        std::vector<float> xFp32(16, 1.0f);
        std::vector<float> yFp32(16, 1.0f);
        std::vector<float> cInitFp32(nnz_);
        for (int i = 0; i < nnz_; i++) {
            cInitFp32[static_cast<size_t>(i)] = static_cast<float>(csrC_.values[static_cast<size_t>(i)]);
        }

        dRowOff_ = DeviceBuffer::copyFrom(csrC_.rowOffsets.data(), 5 * sizeof(int32_t));
        if (nnz_ > 0) {
            dColInd_ = DeviceBuffer::copyFrom(csrC_.colIndices.data(),
                                              static_cast<size_t>(nnz_) * sizeof(int32_t));
            dVals_ = DeviceBuffer::copyFrom(cInitFp32.data(),
                                            static_cast<size_t>(nnz_) * sizeof(float));
        }
        dX_ = DeviceBuffer::copyFrom(xFp32.data(), 16 * sizeof(float));
        dY_ = DeviceBuffer::copyFrom(yFp32.data(), 16 * sizeof(float));

        // X/Y: 4×4 row-major const dense; C: 4×4 CSR (FP32)
        matX_ = DnMatManager::createConst(4, 4, 4, dX_.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        matY_ = DnMatManager::createConst(4, 4, 4, dY_.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        matC_ = SpMatManager::createCsr(4, 4, nnz_, dRowOff_.get(), dColInd_.get(),
                                        dVals_.get(), ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT);

        // Query buffer size for the valid baseline (may fail if operator not
        // yet implemented; defensive fallback keeps E7 testable).
        bufSize_ = 0;
        auto ret = aclsparseSDDMMBufferSize(
            handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
            ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize_);
        if (ret != ACL_SPARSE_STATUS_SUCCESS || bufSize_ == 0) {
            bufSize_ = 16;
        }
        dBuffer_ = DeviceBuffer::alloc(bufSize_);
    }
};

// E1: null handle -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(SddmmExceptionTest, NullHandle) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// E2: null matX -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SddmmExceptionTest, NullMatX) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, nullptr, matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E3: null matY -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SddmmExceptionTest, NullMatY) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), nullptr, &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E4: null matC -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SddmmExceptionTest, NullMatC) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, nullptr,
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E5: null alpha -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SddmmExceptionTest, NullAlpha) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        nullptr, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E6: null beta -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SddmmExceptionTest, NullBeta) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), nullptr, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E7: null externalBuffer (bufferSize > 0) -> ACL_SPARSE_STATUS_INVALID_VALUE
//     Tested at Execute stage (BufferSize only queries size).
TEST_F(SddmmExceptionTest, NullExternalBuffer) {
    auto ret = aclsparseSDDMM(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E8: dimension mismatch (X.k != Y.k) -> MATRIX_TYPE_NOT_SUPPORTED or INVALID_VALUE
//     Exact code is implementation-defined (test plan §4.3 / Q4).
TEST_F(SddmmExceptionTest, DimMismatch) {
    // Y with k=2 (mismatches X k=4): n=4, k=2 row-major
    std::vector<float> ySmall(8, 1.0f);
    auto dYSmall = DeviceBuffer::copyFrom(ySmall.data(), 8 * sizeof(float));
    auto matYSmall = DnMatManager::createConst(4, 2, 2, dYSmall.raw(),
                                               ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matYSmall.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_TRUE(ret == ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED ||
                ret == ACL_SPARSE_STATUS_INVALID_VALUE)
        << "expected MATRIX_TYPE_NOT_SUPPORTED or INVALID_VALUE, got " << ret;
}

// E9: unsupported computeType (ACL_DOUBLE) -> ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(SddmmExceptionTest, UnsupportedComputeType) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_DOUBLE, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// E10: null CSR ptrs (rowOff) -> ACL_SPARSE_STATUS_INVALID_VALUE
//      Kernel reads ptrs[m] for nnz unconditionally, so ptrs must never be null.
TEST_F(SddmmExceptionTest, NullCsrPtrs) {
    auto matCNullPtrs = SpMatManager::createCsr(4, 4, 0, nullptr, dColInd_.get(),
                                                 dVals_.get(), ACL_SPARSE_INDEX_32I,
                                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                                 ACL_FLOAT);
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matCNullPtrs.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E11: m=0 (X 0×4, Y 4×4, C 0×4 CSR nnz=0) -> ACL_SPARSE_STATUS_INVALID_VALUE
//      0 维由 ValidateSddmmDtypeAndDims 守卫拒绝（维度匹配对 0==0 放行）。
//      dtype combo (全 FP32) / op / CSR 格式 / 索引类型 / ptrs 非空均合法，
//      INVALID_VALUE 只能来自 0 维守卫，而非更早的检查。
TEST_F(SddmmExceptionTest, MZero) {
    // X: 0×4 ROW 序 (ld=4)，0 元素也需非空 values（≥1 元素 buffer）
    std::vector<float> xDummy(4, 1.0f);
    std::vector<float> yVals(16, 1.0f);
    auto dX = DeviceBuffer::copyFrom(xDummy.data(), xDummy.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yVals.data(), yVals.size() * sizeof(float));
    // C: 0×4 CSR，nnz=0，rowOffsets=[0] 非空（kernel 读 ptrs[m]）
    std::vector<int32_t> rowOff = {0};
    auto dRowOff = DeviceBuffer::copyFrom(rowOff.data(), rowOff.size() * sizeof(int32_t));

    auto matX = DnMatManager::createConst(0, 4, 4, dX.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(4, 4, 4, dY.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(0, 4, 0, dRowOff.get(), nullptr, nullptr,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX.cget(), matY.cget(), &beta_, matC.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E12: n=0 (X 4×4, Y 4×0, C 4×0 CSR nnz=0) -> ACL_SPARSE_STATUS_INVALID_VALUE
//      0 维由 ValidateSddmmDtypeAndDims 守卫拒绝（维度匹配对 0==0 放行）。
TEST_F(SddmmExceptionTest, NZero) {
    std::vector<float> xVals(16, 1.0f);
    // Y: 4×0 ROW 序 (ld=1)，0 元素也需非空 values
    std::vector<float> yDummy(1, 1.0f);
    auto dX = DeviceBuffer::copyFrom(xVals.data(), xVals.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yDummy.data(), yDummy.size() * sizeof(float));
    // C: 4×0 CSR，nnz=0，rowOffsets 5 个 0
    std::vector<int32_t> rowOff(5, 0);
    auto dRowOff = DeviceBuffer::copyFrom(rowOff.data(), rowOff.size() * sizeof(int32_t));

    auto matX = DnMatManager::createConst(4, 4, 4, dX.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(4, 0, 1, dY.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(4, 0, 0, dRowOff.get(), nullptr, nullptr,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX.cget(), matY.cget(), &beta_, matC.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E13: k=0 (X 4×0, Y 0×4, C 4×4 CSR nnz=0) -> ACL_SPARSE_STATUS_INVALID_VALUE
//      0 维由 ValidateSddmmDtypeAndDims 守卫拒绝（维度匹配对 0==0 放行）。
//      k=0 时 C 的 nnz 必为 0。
TEST_F(SddmmExceptionTest, KZero) {
    // X: 4×0 ROW 序 (ld=1)，0 元素也需非空 values
    std::vector<float> xDummy(1, 1.0f);
    // Y: 0×4 ROW 序 (ld=4)，0 元素也需非空 values
    std::vector<float> yDummy(4, 1.0f);
    auto dX = DeviceBuffer::copyFrom(xDummy.data(), xDummy.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yDummy.data(), yDummy.size() * sizeof(float));
    // C: 4×4 CSR，nnz=0，rowOffsets 5 个 0
    std::vector<int32_t> rowOff(5, 0);
    auto dRowOff = DeviceBuffer::copyFrom(rowOff.data(), rowOff.size() * sizeof(int32_t));

    auto matX = DnMatManager::createConst(4, 0, 1, dX.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(0, 4, 4, dY.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(4, 4, 0, dRowOff.get(), nullptr, nullptr,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX.cget(), matY.cget(), &beta_, matC.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Tiling stores dense leading dimensions as int32_t.  A public DnMat can
// carry a larger int64_t ld, so reject it before serializing the tiling data.
TEST_F(SddmmExceptionTest, RejectsUnrepresentableLeadingDimension) {
    auto matXTooWide = DnMatManager::createConst(
        4, 4, static_cast<int64_t>(INT32_MAX) + 1, dX_.raw(), ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW);
    size_t bs = 0;
    const auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matXTooWide.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}
