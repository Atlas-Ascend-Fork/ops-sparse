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
 * @file gtsv2_strided_batch_test.cpp
 * @brief GTest + CSV-driven L0/L1 test cases + exception tests (E1-E27)
 *        for aclsparseSgtsv2StridedBatch.
 *
 * Tests the batched tridiagonal solver: A^{(i)} y^{(i)} = x^{(i)}, where A is
 * m x m tridiagonal defined by (dl, d, du). The solution y overwrites x in
 * place (IN/OUT semantics).
 *
 * Test organization:
 *   1. Gtsv2StridedBatchTest (TEST_P, CSV-driven):
 *      - L0 (11 cases): basic functional verification
 *      - L1 (55 cases): full coverage + boundary + large shape + outer GM
 *        blocked-CR path (m up to 2^30) + stress + singular
 *
 *   2. Gtsv2StridedBatchExceptionTest (TEST_F, no CSV):
 *      - E1-E12 : main interface parameter validation
 *      - E13-E18: bufferSizeExt parameter validation
 *      - E19-E20: bufferSizeExt pure-UB-path return-value semantics (bufferSize=0)
 *      - E21    : singular matrix behaviour (SUCCESS + Inf/NaN)
 *      - E22-E23: GM path boundary (m > 2^30 NOT_SUPPORTED; m=3072
 *                 pBuffer=nullptr INVALID_VALUE)
 *      - E24-E26: bufferSizeExt GM-path return-value semantics (m=3072/4096
 *                 bufferSize > 0; E26 multi-batch workspace scaling)
 *      - E27    : bufferSizeExt accepts m=4097 (crosses the old 4096 limit)
 *                 with bufferSize > 0
 *
 * Entry point is shared via test/frame/test_main.cpp.
 */

#include "sparse_test.h"
#include "verify.h"
#include "descriptor_manager.h"
#include "gtsv2_strided_batch_param.h"
#include "gtsv2_strided_batch_golden.h"
#include "gtsv2_strided_batch_npu_wrapper.h"

#include "acl/acl.h"
#include "cann_ops_sparse.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace sparse_test;

TEST(Gtsv2StridedBatchGoldenTest, WellCondIsStrictlyDiagonallyDominant)
{
    constexpr int m = 16;
    constexpr int batchCount = 3;
    constexpr int batchStride = 24;
    const auto tri = makeSparseTridiagStrided(m, batchCount, batchStride, "well_cond", 300);

    for (int b = 0; b < batchCount; ++b) {
        const auto base = static_cast<size_t>(b) * batchStride;
        for (int i = 0; i < m; ++i) {
            const auto k = base + i;
            const float offDiagonalSum = std::abs(tri.dl[k]) + std::abs(tri.du[k]);
            EXPECT_GT(std::abs(tri.d[k]), offDiagonalSum) << "batch=" << b << " row=" << i;
        }
        for (int i = m; i < batchStride; ++i) {
            const auto k = base + i;
            EXPECT_FLOAT_EQ(0.0f, tri.dl[k]) << "dl padding, batch=" << b << " offset=" << i;
            EXPECT_FLOAT_EQ(0.0f, tri.d[k]) << "d padding, batch=" << b << " offset=" << i;
            EXPECT_FLOAT_EQ(0.0f, tri.du[k]) << "du padding, batch=" << b << " offset=" << i;
            EXPECT_FLOAT_EQ(0.0f, tri.x[k]) << "x padding, batch=" << b << " offset=" << i;
        }
    }
}

// ============================================================================
// Shared ACL environment (one instance per test-suite)
// ============================================================================

using SharedAclEnvScope = AclEnvScope;

// ============================================================================
// GTest parameterized fixture — CSV-driven L0 + L1 success-path tests
// ============================================================================

class Gtsv2StridedBatchTest : public testing::TestWithParam<Gtsv2StridedBatchTestParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    Gtsv2StridedBatchTestParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override {
        param_  = GetParam();
        stream_ = env_->stream();
    }
};

// ============================================================================
// Helpers: split out from TEST_P to keep method body under 50 nbnc lines
// ============================================================================

static void AssertReturnCodes(const Sgtsv2StridedBatchNpuResult& r,
                              const Gtsv2StridedBatchTestParam& p)
{
    EXPECT_EQ(r.bufferSizeExtRet, ACL_SPARSE_STATUS_SUCCESS)
        << "[" << p.case_name << "] bufferSizeExt should return SUCCESS";
    if (p.m <= 2048) {
        EXPECT_EQ(r.bufferSize, 0u)
            << "[" << p.case_name << "] bufferSize should be 0 for m <= 2048";
    }
    EXPECT_EQ(r.computeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "[" << p.case_name << "] gtsv2StridedBatch should return SUCCESS";
}

static bool CheckSingularInfOrNan(const Sgtsv2StridedBatchNpuResult& r,
                                  const Gtsv2StridedBatchTestParam& p)
{
    for (int b = 0; b < p.batchCount; b++) {
        int64_t base = static_cast<int64_t>(b) * p.batchStride;
        for (int i = 0; i < p.m; i++) {
            float v = r.x[base + i];
            if (std::isinf(v) || std::isnan(v)) {
                return true;
            }
        }
    }
    return false;
}

static bool VerifyPrecisionPerBatch(const Sgtsv2StridedBatchNpuResult& r,
                                    const std::vector<float>& xGolden,
                                    const Gtsv2StridedBatchTestParam& p)
{
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MERE_MARE)
       .SetMERE(p.mere_threshold)
       .SetMARE(p.mare_multiplier);

    bool allPass = true;
    for (int b = 0; b < p.batchCount; b++) {
        int64_t base = static_cast<int64_t>(b) * p.batchStride;
        std::vector<float> xNpuBatch(r.x.begin() + base, r.x.begin() + base + p.m);
        std::vector<float> xGoldenBatch(xGolden.begin() + base, xGolden.begin() + base + p.m);
        std::string batchCaseId = p.case_name + "_b" + std::to_string(b);
        if (!Verifier::verifyVector(xNpuBatch, xGoldenBatch, cfg, batchCaseId)) {
            allPass = false;
        }
    }
    return allPass;
}

// ============================================================================
// Test body: parameterized success-path test (L0 + L1)
// ============================================================================

TEST_P(Gtsv2StridedBatchTest, Sgtsv2StridedBatchCsv) {
    const auto& p = param_;

    std::cout << "==== " << p.case_name
              << " ==== m=" << p.m
              << " batchCount=" << p.batchCount
              << " batchStride=" << p.batchStride
              << " pattern=" << p.pattern
              << " seed=" << p.seed
              << " expect=" << p.expect_result << "\n";

    auto tri = makeSparseTridiagStrided(p.m, p.batchCount, p.batchStride, p.pattern, p.seed);
    auto xGolden = Verifier::toFloat(Sgtsv2StridedBatchGoldenFromTridiag(tri));

    HandleManager handle;
    auto npuResult = Sgtsv2StridedBatchNpu(
        handle, stream_, tri.dl, tri.d, tri.du, tri.x,
        p.m, p.batchCount, p.batchStride);

    AssertReturnCodes(npuResult, p);

    // Fatal-stage guard: if bufferSizeExt/compute failed, npuResult.x is empty
    // and per-batch slicing below would be out-of-bounds. EXPECT_* in
    // AssertReturnCodes is non-fatal, so return explicitly to report a clean
    // test failure instead of crashing the whole binary (relevant while the
    // m > 4096 outer-GM-path operator support is still being integrated).
    if (npuResult.bufferSizeExtRet != ACL_SPARSE_STATUS_SUCCESS ||
        npuResult.computeRet != ACL_SPARSE_STATUS_SUCCESS) {
        return;
    }

    if (p.expect_result == "SUCCESS_NO_OUTPUT") {
        SUCCEED() << "[" << p.case_name << "] SUCCESS_NO_OUTPUT (batchCount=0)";
        return;
    }
    if (p.expect_result == "SINGULAR") {
        EXPECT_TRUE(CheckSingularInfOrNan(npuResult, p))
            << "[" << p.case_name << "] Singular matrix should produce Inf/NaN";
        return;
    }

    ASSERT_EQ(p.expect_result, "SUCCESS")
        << "[" << p.case_name << "] unknown expect_result: " << p.expect_result;
    bool allPass = VerifyPrecisionPerBatch(npuResult, xGolden, p);
    EXPECT_TRUE(allPass)
        << "[" << p.case_name << "] precision verification failed for at least one batch";
    if (allPass) {
        std::cout << "[" << p.case_name << "] PASSED (batchCount=" << p.batchCount << ")\n";
    }
}

// ============================================================================
// Parameterized test instantiation from CSV (L0 + L1)
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    Gtsv2StridedBatchCsvCases,
    Gtsv2StridedBatchTest,
    testing::ValuesIn(GetCasesFromCsv<Gtsv2StridedBatchTestParam>("gtsv2_strided_batch_test.csv")),
    [](const testing::TestParamInfo<Gtsv2StridedBatchTestParam>& info) {
        return info.param.caseId();
    });

// ============================================================================
// Exception test fixture (TEST_F, no CSV)
// ============================================================================

class Gtsv2StridedBatchExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    static constexpr int kDefaultM           = 8;
    static constexpr int kDefaultBatchCount  = 1;
    static constexpr int kDefaultBatchStride = 8;  // m=8 -> m_pad=8 -> alignedM=8

    void SetUp() override {
        stream_ = env_->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        prepareBuffers(kDefaultM, kDefaultBatchCount, kDefaultBatchStride,
                       "well_cond", 999);

        dBuffer_ = DeviceBuffer::alloc(256);
        pBuffer_ = dBuffer_.get();
    }

    void prepareBuffers(int m, int batchCount, int batchStride,
                        const std::string& pattern, uint32_t seed) {
        auto tri = makeSparseTridiagStrided(m, batchCount, batchStride, pattern, seed);
        size_t totalBytes = static_cast<size_t>(batchCount) * batchStride * sizeof(float);
        if (totalBytes == 0) {
            totalBytes = sizeof(float);
            std::vector<float> dummy(1, 0.0f);
            dDl_ = DeviceBuffer::copyFrom(dummy.data(), totalBytes);
            dD_  = DeviceBuffer::copyFrom(dummy.data(), totalBytes);
            dDu_ = DeviceBuffer::copyFrom(dummy.data(), totalBytes);
            dX_  = DeviceBuffer::copyFrom(dummy.data(), totalBytes);
        } else {
            dDl_ = DeviceBuffer::copyFrom(tri.dl.data(), totalBytes);
            dD_  = DeviceBuffer::copyFrom(tri.d.data(),  totalBytes);
            dDu_ = DeviceBuffer::copyFrom(tri.du.data(), totalBytes);
            dX_  = DeviceBuffer::copyFrom(tri.x.data(),  totalBytes);
        }
        pDl_ = reinterpret_cast<const float*>(dDl_.raw());
        pD_  = reinterpret_cast<const float*>(dD_.raw());
        pDu_ = reinterpret_cast<const float*>(dDu_.raw());
        pX_  = reinterpret_cast<float*>(dX_.get());
    }

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;
    DeviceBuffer dDl_, dD_, dDu_, dX_, dBuffer_;
    const float* pDl_ = nullptr;
    const float* pD_  = nullptr;
    const float* pDu_ = nullptr;
    float*       pX_  = nullptr;
    void*        pBuffer_ = nullptr;
};

// ============================================================================
// Main interface exception tests (E1 - E12)
// ============================================================================

TEST_F(Gtsv2StridedBatchExceptionTest, E1_NullHandle) {
    auto status = aclsparseSgtsv2StridedBatch(
        nullptr, kDefaultM, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, pBuffer_);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E2_MTooSmall) {
    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), 2, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, pBuffer_);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E3_MTooLarge) {
    // m upper limit is kMaxMPractical = 2^30 (int32 batchStride contract).
    // m-limit validation runs before the batchStride check, so the small
    // default batchStride does not mask the expected NOT_SUPPORTED.
    constexpr int kMTooLarge = (1 << 30) + 1;
    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kMTooLarge, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, pBuffer_);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E4_NullDl) {
    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kDefaultM, nullptr, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, pBuffer_);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E5_NullD) {
    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kDefaultM, pDl_, nullptr, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, pBuffer_);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E6_NullDu) {
    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kDefaultM, pDl_, pD_, nullptr, pX_,
        kDefaultBatchCount, kDefaultBatchStride, pBuffer_);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E7_NullX) {
    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kDefaultM, pDl_, pD_, pDu_, nullptr,
        kDefaultBatchCount, kDefaultBatchStride, pBuffer_);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E8_BatchCountNegative) {
    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kDefaultM, pDl_, pD_, pDu_, pX_,
        -1, kDefaultBatchStride, pBuffer_);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E9_BatchStrideSmall) {
    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kDefaultM, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride - 1, pBuffer_);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E10_NullBuffer) {
    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kDefaultM, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, nullptr);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_SUCCESS);
    auto aclRet = aclrtSynchronizeStream(stream_);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E11_UbPathAcceptsMisalignedPBuffer) {
    void* rawPtr = nullptr;
    auto aclRet = aclrtMalloc(&rawPtr, 256, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(aclRet, ACL_SUCCESS) << "aclrtMalloc failed for misalignment test";
    ASSERT_NE(rawPtr, nullptr);

    void* misalignedPtr = static_cast<uint8_t*>(rawPtr) + 1;
    EXPECT_NE(reinterpret_cast<uintptr_t>(misalignedPtr) % 128, 0u)
        << "Test setup error: misaligned pointer is actually aligned";

    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kDefaultM, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, misalignedPtr);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_SUCCESS);
    auto syncRet = aclrtSynchronizeStream(stream_);
    EXPECT_EQ(syncRet, ACL_SUCCESS);

    aclrtFree(rawPtr);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E11b_GmPathBufferNotAligned) {
    constexpr int kM = 3072;
    constexpr int kBatchStride = 4096;
    prepareBuffers(kM, 1, kBatchStride, "well_cond", 3072);

    void* rawPtr = nullptr;
    auto aclRet = aclrtMalloc(&rawPtr, 256, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(aclRet, ACL_SUCCESS) << "aclrtMalloc failed for misalignment test";
    ASSERT_NE(rawPtr, nullptr);

    void* misalignedPtr = static_cast<uint8_t*>(rawPtr) + 1;
    EXPECT_NE(reinterpret_cast<uintptr_t>(misalignedPtr) % 128, 0u)
        << "Test setup error: misaligned pointer is actually aligned";

    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        1, kBatchStride, misalignedPtr);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE)
        << "m=3072 (GM path) with misaligned pBuffer should return INVALID_VALUE";

    aclrtFree(rawPtr);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E12_BatchCountZero) {
    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kDefaultM, pDl_, pD_, pDu_, pX_,
        0, kDefaultBatchStride, pBuffer_);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_SUCCESS);
}

// ============================================================================
// bufferSizeExt interface exception tests (E13 - E18)
// ============================================================================

TEST_F(Gtsv2StridedBatchExceptionTest, E13_BufferSizeExt_NullHandle) {
    size_t bufSize = 0;
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        nullptr, kDefaultM, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, &bufSize);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E14_BufferSizeExt_MTooSmall) {
    size_t bufSize = 0;
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), 2, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, &bufSize);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E15_BufferSizeExt_NullInput) {
    size_t bufSize = 0;
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kDefaultM, nullptr, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, &bufSize);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E16_BufferSizeExt_BatchCountNeg) {
    size_t bufSize = 0;
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kDefaultM, pDl_, pD_, pDu_, pX_,
        -1, kDefaultBatchStride, &bufSize);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E17_BufferSizeExt_BatchStrideSmall) {
    size_t bufSize = 0;
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kDefaultM, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride - 1, &bufSize);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E18_BufferSizeExt_NullSizePtr) {
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kDefaultM, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, nullptr);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ============================================================================
// bufferSizeExt normal return-value tests (E19 - E20)
// ============================================================================

TEST_F(Gtsv2StridedBatchExceptionTest, E19_BufferSizeExt_M1024) {
    constexpr int kM = 1024;
    constexpr int kBatchStride = 1024;  // m_pad=1024 -> alignedM=1024
    prepareBuffers(kM, 1, kBatchStride, "well_cond", 1919);

    size_t bufSize = 999;
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        1, kBatchStride, &bufSize);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bufSize, 0u)
        << "m=1024 pure UB path should return bufferSize=0";
}

TEST_F(Gtsv2StridedBatchExceptionTest, E20_BufferSizeExt_M2048) {
    constexpr int kM = 2048;
    constexpr int kBatchStride = 2048;
    prepareBuffers(kM, 1, kBatchStride, "well_cond", 2020);

    size_t bufSize = 999;
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        1, kBatchStride, &bufSize);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bufSize, 0u)
        << "m=2048 pure UB path should return bufferSize=0";
}

// ============================================================================
// Singular matrix behaviour test (E21)
// ============================================================================

TEST_F(Gtsv2StridedBatchExceptionTest, E21_SingularMatrix) {
    constexpr int kM = 16;
    constexpr int kBatchCount = 3;
    constexpr int kBatchStride = 16;  // m_pad=16 -> alignedM=16
    prepareBuffers(kM, kBatchCount, kBatchStride, "singular", 80);

    DeviceBuffer dBuf = DeviceBuffer::alloc(128);

    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        kBatchCount, kBatchStride, dBuf.get());
    EXPECT_EQ(status, ACL_SPARSE_STATUS_SUCCESS)
        << "Singular matrix should still return SUCCESS (host does not validate d values)";

    auto aclRet = aclrtSynchronizeStream(stream_);
    ASSERT_EQ(aclRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed";

    size_t totalBytes = static_cast<size_t>(kBatchCount) * kBatchStride * sizeof(float);
    std::vector<float> xOut(kBatchCount * kBatchStride, 0.0f);
    dX_.copyToHost(xOut.data(), totalBytes);

    bool hasInfOrNan = false;
    for (int b = 0; b < kBatchCount && !hasInfOrNan; b++) {
        int64_t base = static_cast<int64_t>(b) * kBatchStride;
        for (int i = 0; i < kM; i++) {
            float v = xOut[base + i];
            if (std::isinf(v) || std::isnan(v)) {
                hasInfOrNan = true;
                break;
            }
        }
    }
    EXPECT_TRUE(hasInfOrNan)
        << "Singular matrix should produce Inf/NaN in output (cuSPARSE-aligned "
        << "no-pivot behaviour); if not, the operator may have added pivoting";
}

// ============================================================================
// GM path boundary tests (E22 - E23)
//
// The m upper limit is kMaxMPractical = 2^30 (int32 batchStride contract,
// design doc 1.3.A section 8.1). m = 2^30 + 1 must be rejected with
// NOT_SUPPORTED on both the main interface and bufferSizeExt.
// ============================================================================

TEST_F(Gtsv2StridedBatchExceptionTest, E22_MExceedsMaxSupported) {
    constexpr int kM = (1 << 30) + 1;  // > kMaxMPractical (2^30)
    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, pBuffer_);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

TEST_F(Gtsv2StridedBatchExceptionTest, E22b_BufferSizeExt_MExceedsMaxSupported) {
    constexpr int kM = (1 << 30) + 1;  // > kMaxMPractical (2^30)
    size_t bufSize = 999u;
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        kDefaultBatchCount, kDefaultBatchStride, &bufSize);
    ASSERT_EQ(status, ACL_SPARSE_STATUS_NOT_SUPPORTED);
    ASSERT_EQ(bufSize, 0u) << "bufferSizeExt must set bufferSize=0 on NOT_SUPPORTED";
}

TEST_F(Gtsv2StridedBatchExceptionTest, E23_GmPathNullPBuffer) {
    constexpr int kM = 3072;
    constexpr int kBatchStride = 4096;
    prepareBuffers(kM, 1, kBatchStride, "well_cond", 2001);

    auto status = aclsparseSgtsv2StridedBatch(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        1, kBatchStride, nullptr);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ============================================================================
// bufferSizeExt GM path return-value tests (E24 - E26)
//
// Validates the bufferSizeExt interface semantics for the outer GM blocked-CR
// path (m > 2048). These tests complement E19/E20 (pure UB path, m <= 2048,
// bufferSize == 0) by verifying:
//   - E24: GM path mid-point (m=3072) returns SUCCESS with bufferSize > 0
//   - E25: GM path m=4096 (m == mPad, old upper bound) returns SUCCESS with
//          bufferSize > 0
//   - E26: GM path multi-batch workspace scaling.
//          bufferSize must scale with batchCount (not batchPerCore), otherwise
//          multi-batch GM path under-allocates workspace and corrupts results.
//   - E27: m=4097 (crosses the old 4096 limit) is accepted with bufferSize > 0
//          (m > 2^30 NOT_SUPPORTED is covered by E22/E22b).
// ============================================================================

TEST_F(Gtsv2StridedBatchExceptionTest, E24_BufferSizeExt_GMPath_M3072) {
    constexpr int kM = 3072;
    constexpr int kBatchStride = 4096;  // m_pad=4096 -> alignedM=4096
    prepareBuffers(kM, 1, kBatchStride, "well_cond", 2024);

    size_t bufSize = 999;
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        1, kBatchStride, &bufSize);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_SUCCESS)
        << "bufferSizeExt should accept m=3072 (GM path)";
    EXPECT_GT(bufSize, 0u)
        << "GM path (m=3072 > 2048) should return non-zero bufferSize "
        << "for the outer blocked-CR workspace";
}

TEST_F(Gtsv2StridedBatchExceptionTest, E25_BufferSizeExt_GMPath_M4096) {
    constexpr int kM = 4096;
    constexpr int kBatchStride = 4096;  // m_pad=4096 -> alignedM=4096
    prepareBuffers(kM, 1, kBatchStride, "well_cond", 2025);

    size_t bufSize = 999;
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        1, kBatchStride, &bufSize);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_SUCCESS)
        << "bufferSizeExt should accept m=4096 (m == mPad, old upper bound)";
    EXPECT_GT(bufSize, 0u)
        << "GM path (m=4096) should return non-zero bufferSize";
}

TEST_F(Gtsv2StridedBatchExceptionTest, E26_BufferSizeExt_GMPath_MultiBatchScaling) {
    // bufferSizeExt must return a workspace that scales
    // with batchCount (kernel uses global batchIdx to index workspace). The
    // pre-fix code computed workspace = workspacePerBatch * batchPerCore, which
    // under-allocated by a factor of (batchCount / batchPerCore) on multi-core
    // systems. Here we query bufferSize for batchCount=1 and batchCount=10 with
    // the same m (GM path) and assert the 10-batch size is ~10x larger.
    constexpr int kM = 3072;
    constexpr int kBatchStride = 4096;
    prepareBuffers(kM, 10, kBatchStride, "well_cond", 2026);

    size_t bufSize1 = 0;
    auto status1 = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        1, kBatchStride, &bufSize1);
    ASSERT_EQ(status1, ACL_SPARSE_STATUS_SUCCESS)
        << "bufferSizeExt should accept m=3072 batchCount=1 (GM path)";
    ASSERT_GT(bufSize1, 0u)
        << "GM workspace path (m=3072 > 2048) should return non-zero bufferSize";

    size_t bufSize10 = 0;
    auto status10 = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        10, kBatchStride, &bufSize10);
    ASSERT_EQ(status10, ACL_SPARSE_STATUS_SUCCESS)
        << "bufferSizeExt should accept m=3072 batchCount=10 (GM path)";
    ASSERT_GT(bufSize10, 0u)
        << "GM workspace path (m=3072 > 2048) should return non-zero bufferSize";

    // workspace must scale with batchCount. bufSize10 should be at least
    // 9x bufSize1 (10 batches vs 1 batch, allowing for 128B alignment rounding).
    EXPECT_GE(bufSize10, bufSize1 * 9)
        << "bufferSize for batchCount=10 (" << bufSize10
        << ") should be ~10x batchCount=1 (" << bufSize1
        << "); if ~1x, bufferSizeExt still uses batchPerCore (under-allocate)";
}

// ============================================================================
// E27: bufferSizeExt accepts m=4097 (crosses the old 4096 limit)
//
// PR #57 reviewer fix: the old kMaxSupportedM=4096 ceiling is removed and m
// is supported up to 2^30 (outer GM blocked-CR path). m=4097 (mPad=8192,
// K=2) must now return SUCCESS with a non-zero workspace size.
// ============================================================================

TEST_F(Gtsv2StridedBatchExceptionTest, E27_BufferSizeExt_GMPath_M4097) {
    constexpr int kM = 4097;
    constexpr int kBatchStride = 8192;  // mPad=8192 -> alignedM=8192
    prepareBuffers(kM, 1, kBatchStride, "well_cond", 4097);

    size_t bufSize = 0;
    auto status = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle_->get(), kM, pDl_, pD_, pDu_, pX_,
        1, kBatchStride, &bufSize);
    EXPECT_EQ(status, ACL_SPARSE_STATUS_SUCCESS)
        << "bufferSizeExt should accept m=4097 (crosses the old 4096 limit)";
    EXPECT_GT(bufSize, 0u)
        << "GM path (m=4097 > 2048) should return non-zero bufferSize";
}
