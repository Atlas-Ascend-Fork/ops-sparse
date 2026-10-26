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

#include "sddmm_test_common.h"
#include "../../frame/presmoke.h"

#include <algorithm>
#include <cstdlib>
#include <limits>

using namespace sparse_test;

TEST(VerifierRegressionTest, RejectsNonFiniteMismatchInAbsAndRelModes)
{
    const std::vector<float> finite = {1.0f};
    const std::vector<float> nan = {std::numeric_limits<float>::quiet_NaN()};
    const std::vector<float> posInf = {std::numeric_limits<float>::infinity()};
    const std::vector<float> negInf = {-std::numeric_limits<float>::infinity()};

    for (PrecisionMode mode : {PrecisionMode::ABS, PrecisionMode::REL}) {
        VerifyConfig cfg;
        cfg.SetMode(mode).SetAbsTol(1.0).SetRelTol(1.0);
        EXPECT_FALSE(Verifier::verifyVector(nan, finite, cfg, "nan-vs-finite"));
        EXPECT_FALSE(Verifier::verifyVector(finite, nan, cfg, "finite-vs-nan"));
        EXPECT_FALSE(Verifier::verifyVector(posInf, negInf, cfg, "inf-sign-mismatch"));
    }
}

TEST(VerifierRegressionTest, ComparesIntegralVectorsWithoutFloatNarrowing)
{
    const std::vector<int32_t> output = {16777216};
    const std::vector<int32_t> golden = {16777217};
    VerifyConfig cfg;
    EXPECT_FALSE(Verifier::verifyVector(output, golden, cfg, "int32-low-bit-mismatch"));
}

TEST(SddmmGoldenTest, Fp16DecodeAndRoundTrip)
{
    EXPECT_EQ(Fp16BitsToFp32(0x0001), std::ldexp(1.0f, -24));
    EXPECT_EQ(Fp16BitsToFp32(0x03ff), std::ldexp(1023.0f, -24));
    EXPECT_EQ(Fp16BitsToFp32(0x0400), std::ldexp(1.0f, -14));
    EXPECT_EQ(Fp16BitsToFp32(0x7bff), 65504.0f);
    EXPECT_TRUE(std::signbit(Fp16BitsToFp32(0x8000)));
    EXPECT_TRUE(std::isinf(Fp16BitsToFp32(0xfc00)));
    EXPECT_TRUE(std::isnan(Fp16BitsToFp32(0x7e01)));
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        const uint16_t half = static_cast<uint16_t>(bits);
        const bool nan = (half & 0x7c00u) == 0x7c00u && (half & 0x03ffu) != 0;
        const uint16_t expected = nan ? static_cast<uint16_t>((half & 0x8000u) | 0x7e00u) : half;
        EXPECT_EQ(Fp32ToFp16Bits(Fp16BitsToFp32(half)), expected) << "half bits=" << bits;
    }
}

using SddmmArch22Test = SddmmParameterizedTest;

namespace {

std::vector<SddmmTestParam> GetSddmmArch22Cases()
{
    auto cases = GetCasesFromCsv<SddmmTestParam>("sddmm_test.csv");
    if (SparseTestFullEnabled()) {
        return cases;
    }

    for (auto& p : cases) {
        // These rows target K/cache and row-nnz thresholds. Their dimensions
        // are part of the regression contract and must not be normalized.
        const int caseNumber = std::atoi(p.case_name.c_str() + 4);
        const bool isBoundaryCase = p.case_name == "A22_10" || p.case_name == "A22_11" || p.case_name == "A22_15" ||
                                    p.case_name == "A22_35" || p.case_name == "A22_48" || p.case_name == "A22_49" ||
                                    caseNumber >= 54;
        if (!isBoundaryCase) {
            // Preserve CSV rows and their dtype/op/layout/scalar semantics,
            // while avoiding throughput-scale CPU golden calculations.
            p.m = std::min<int64_t>(p.m, 128);
            p.n = std::min<int64_t>(p.n, 128);
            p.k = std::min<int64_t>(p.k, 64);
        }
    }
    return cases;
}

} // namespace

TEST_P(SddmmArch22Test, SddmmFunctional)
{
    const auto& p = param_;

    aclDataType dtype = ParseDtype(p.dtype);
    aclDataType computeType = ParseDtype(p.compute_type);
    aclsparseOperation_t opX = ParseOperation(p.op_x);
    aclsparseOperation_t opY = ParseOperation(p.op_y);
    aclsparseOrder_t orderX = ParseOrder(p.order_x);
    aclsparseOrder_t orderY = ParseOrder(p.order_y);
    aclsparseSDDMMAlg_t alg = ParseSddmmAlg(p.alg);

    SddmmInputs inputs = GenerateSddmmInputs(p);
    int64_t nnz = inputs.nnz;

    std::vector<double> expectFp64 = ComputeGoldenExpect(p, inputs.csrC, inputs.Xf64, inputs.Yf64, dtype, opX, opY);

    SddmmNpuResult npuResult = RunNpuSddmm(p, stream_, inputs.csrC, inputs.Xf64, inputs.Yf64, dtype, computeType, alg,
                                           opX, opY, orderX, orderY, nnz);

    ASSERT_EQ(npuResult.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(npuResult.preprocessRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(npuResult.executeRet, ACL_SPARSE_STATUS_SUCCESS);

    if (nnz > 0) {
        ASSERT_EQ(npuResult.valuesOut.size(), static_cast<size_t>(nnz));
        VerifySddmmPrecision(p.case_name, npuResult, expectFp64, dtype, nnz);
    }
}

INSTANTIATE_TEST_SUITE_P(SddmmArch22Cases, SddmmArch22Test, testing::ValuesIn(GetSddmmArch22Cases()),
                         [](const testing::TestParamInfo<SddmmTestParam>& info) { return info.param.case_name; });

class SddmmArch22ExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

TEST_F(SddmmArch22ExceptionTest, E1_NullHandle)
{
    size_t bufSize = 0;
    auto ret = aclsparseSDDMMBufferSize(nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SddmmArch22ExceptionTest, E2_UnsupportedComputeType)
{
    HandleManager handle;
    handle.setStream(env_->stream());
    float alpha = 1.0f, beta = 0.0f;
    int64_t m = 4, n = 3, k = 2, nnz = 5;
    std::vector<int32_t> rowOff = {0, 2, 3, 4, 5};
    std::vector<int32_t> colInd = {0, 2, 1, 0, 2};
    std::vector<float> vals(5, 0.0f);
    std::vector<float> hX(m * k, 1.0f);
    std::vector<float> hY(n * k, 1.0f);

    DeviceBuffer dRowOff = DeviceBuffer::copyFrom(rowOff.data(), (m + 1) * sizeof(int32_t));
    DeviceBuffer dColInd = DeviceBuffer::copyFrom(colInd.data(), nnz * sizeof(int32_t));
    DeviceBuffer dVals = DeviceBuffer::copyFrom(vals.data(), nnz * sizeof(float));
    DeviceBuffer dX = DeviceBuffer::copyFrom(hX.data(), m * k * sizeof(float));
    DeviceBuffer dY = DeviceBuffer::copyFrom(hY.data(), n * k * sizeof(float));

    auto matX = DnMatManager::createConst(m, k, k, dX.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(n, k, k, dY.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(m, n, nnz, dRowOff.get(), dColInd.get(), dVals.get(), ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bufSize = 0;
    auto ret = aclsparseSDDMMBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE, &alpha,
                                        matX.cget(), matY.cget(), &beta, matC.get(), ACL_INT32,
                                        ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

TEST_F(SddmmArch22ExceptionTest, E3_DimensionMismatch)
{
    HandleManager handle;
    handle.setStream(env_->stream());
    float alpha = 1.0f, beta = 0.0f;
    int64_t m = 4, n = 3, k = 2, nnz = 5;
    std::vector<int32_t> rowOff = {0, 2, 3, 4, 5};
    std::vector<int32_t> colInd = {0, 2, 1, 0, 2};
    std::vector<float> vals(5, 0.0f);
    std::vector<float> hX(m * k, 1.0f);
    std::vector<float> hY(n * k, 1.0f);

    DeviceBuffer dRowOff = DeviceBuffer::copyFrom(rowOff.data(), (m + 1) * sizeof(int32_t));
    DeviceBuffer dColInd = DeviceBuffer::copyFrom(colInd.data(), nnz * sizeof(int32_t));
    DeviceBuffer dVals = DeviceBuffer::copyFrom(vals.data(), nnz * sizeof(float));
    DeviceBuffer dX = DeviceBuffer::copyFrom(hX.data(), m * k * sizeof(float));
    DeviceBuffer dY = DeviceBuffer::copyFrom(hY.data(), n * k * sizeof(float));

    auto matX = DnMatManager::createConst(m, k + 1, k + 1, dX.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(n, k, k, dY.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(m, n, nnz, dRowOff.get(), dColInd.get(), dVals.get(), ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bufSize = 0;
    auto ret = aclsparseSDDMMBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE, &alpha,
                                        matX.cget(), matY.cget(), &beta, matC.get(), ACL_FLOAT,
                                        ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Empty CSR output is valid in the public A2/A3 three-stage contract.
TEST_F(SddmmArch22ExceptionTest, EmptyRowsBufferSize)
{
    HandleManager handle;
    handle.setStream(env_->stream());
    float alpha = 1.0f, beta = 0.0f;
    const int64_t m = 0, n = 3, k = 2;
    // X has zero rows; its data is not accessed.
    std::vector<float> xDummy(1, 1.0f);
    std::vector<float> yDummy(1, 1.0f);
    DeviceBuffer dX = DeviceBuffer::copyFrom(xDummy.data(), xDummy.size() * sizeof(float));
    DeviceBuffer dY = DeviceBuffer::copyFrom(yDummy.data(), yDummy.size() * sizeof(float));
    // C: 0×3 CSR，nnz=0，rowOffsets=[0] 非空（kernel 读 ptrs[m]）
    std::vector<int32_t> rowOff = {0};
    DeviceBuffer dRowOff = DeviceBuffer::copyFrom(rowOff.data(), rowOff.size() * sizeof(int32_t));

    auto matX = DnMatManager::createConst(m, k, k, dX.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(n, k, k, dY.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(m, n, 0, dRowOff.get(), nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bufSize = 0;
    auto ret = aclsparseSDDMMBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE, &alpha,
                                        matX.cget(), matY.cget(), &beta, matC.get(), ACL_FLOAT,
                                        ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SddmmArch22ExceptionTest, EmptyColumnsBufferSize)
{
    HandleManager handle;
    handle.setStream(env_->stream());
    float alpha = 1.0f, beta = 0.0f;
    const int64_t m = 4, n = 0, k = 2;
    std::vector<float> hX(static_cast<size_t>(m) * static_cast<size_t>(k), 1.0f);
    // Y has zero rows; its data is not accessed.
    std::vector<float> yDummy(1, 1.0f);
    DeviceBuffer dX = DeviceBuffer::copyFrom(hX.data(), hX.size() * sizeof(float));
    DeviceBuffer dY = DeviceBuffer::copyFrom(yDummy.data(), yDummy.size() * sizeof(float));
    // C: 4×0 CSR，nnz=0，rowOffsets 5 个 0
    std::vector<int32_t> rowOff(5, 0);
    DeviceBuffer dRowOff = DeviceBuffer::copyFrom(rowOff.data(), rowOff.size() * sizeof(int32_t));

    auto matX = DnMatManager::createConst(m, k, k, dX.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(n, k, k, dY.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(m, n, 0, dRowOff.get(), nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bufSize = 0;
    auto ret = aclsparseSDDMMBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE, &alpha,
                                        matX.cget(), matY.cget(), &beta, matC.get(), ACL_FLOAT,
                                        ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SddmmArch22ExceptionTest, ZeroReductionBufferSize)
{
    HandleManager handle;
    handle.setStream(env_->stream());
    float alpha = 1.0f, beta = 0.0f;
    const int64_t m = 4, n = 3, k = 0;
    // X and Y have zero reduction columns and valid leading dimensions.
    std::vector<float> xDummy(1, 1.0f);
    std::vector<float> yDummy(1, 1.0f);
    DeviceBuffer dX = DeviceBuffer::copyFrom(xDummy.data(), xDummy.size() * sizeof(float));
    DeviceBuffer dY = DeviceBuffer::copyFrom(yDummy.data(), yDummy.size() * sizeof(float));
    // This case uses an empty pattern; SddmmContract also covers nonempty C.
    std::vector<int32_t> rowOff(5, 0);
    DeviceBuffer dRowOff = DeviceBuffer::copyFrom(rowOff.data(), rowOff.size() * sizeof(int32_t));

    auto matX = DnMatManager::createConst(m, k, 1, dX.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(n, k, 1, dY.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(m, n, 0, dRowOff.get(), nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bufSize = 0;
    auto ret = aclsparseSDDMMBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE, &alpha,
                                        matX.cget(), matY.cget(), &beta, matC.get(), ACL_FLOAT,
                                        ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SddmmArch22ExceptionTest, E4_UnrepresentableLeadingDimension)
{
    HandleManager handle;
    handle.setStream(env_->stream());
    float alpha = 1.0f, beta = 0.0f;
    constexpr int64_t m = 1, n = 1, k = 1, nnz = 1;
    const std::vector<int32_t> rowOff = {0, 1};
    const std::vector<int32_t> colInd = {0};
    const std::vector<float> values = {0.0f};
    const std::vector<float> dense = {1.0f};
    DeviceBuffer dRowOff = DeviceBuffer::copyFrom(rowOff.data(), 2 * sizeof(int32_t));
    DeviceBuffer dColInd = DeviceBuffer::copyFrom(colInd.data(), sizeof(int32_t));
    DeviceBuffer dValues = DeviceBuffer::copyFrom(values.data(), sizeof(float));
    DeviceBuffer dDense = DeviceBuffer::copyFrom(dense.data(), sizeof(float));
    auto matX = DnMatManager::createConst(m, k, static_cast<int64_t>(UINT32_MAX) + 1, dDense.raw(), ACL_FLOAT,
                                          ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(n, k, k, dDense.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(m, n, nnz, dRowOff.get(), dColInd.get(), dValues.get(), ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bufSize = 0;
    const auto ret = aclsparseSDDMMBufferSize(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
                                              &alpha, matX.cget(), matY.cget(), &beta, matC.get(), ACL_FLOAT,
                                              ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}
