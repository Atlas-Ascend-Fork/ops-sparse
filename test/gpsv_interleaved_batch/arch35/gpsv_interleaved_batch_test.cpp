/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// GTest + CSV parameterized test for aclsparseSgpsvInterleavedBatch (L0 + L1, iteration 2)
//
// API: aclsparseSgpsvInterleavedBatch(handle, algo, m,
//          float* ds, float* dl, float* d, float* du, float* dw, float* x,
//          int batchCount, void* pBuffer)
//
// Data layout: row-major interleaved, data[row * batchCount + batch]
// Boundary conditions: ds[0..1][*]=0, dl[0][*]=0, du[m-1][*]=0, dw[m-2..m-1][*]=0

#include "test_common.h"

#include "../gpsv_interleaved_batch_golden.h"
#include "../gpsv_zero_rotation_cases.h"
#include "../gpsv_interleaved_batch_param.h"
#include "gpsv_interleaved_batch_npu_wrapper.h"

namespace sparse_test {

// ----------------------------------------------------------------------------
// Helper for exception-path tests: builds 6 device buffers of the same size
// and (optionally) an aligned 256-byte workspace buffer for the API call.
// Each TEST_F sets its own m/batchCount/algo parameters (some intentionally
// invalid) and calls aclsparseSgpsvInterleavedBatch directly, checking the
// returned status code.
// ----------------------------------------------------------------------------
struct GpsvBufSet {
    DeviceBuffer dDs, dDl, dD, dDu, dDw, dX;
    DeviceBuffer dBuf;  // default-constructed = nullptr-backed via raw()

    // Build 6 device buffers, each holding |totalElems| floats initialized to val.
    // totalElems of 0 is promoted to 1 to avoid empty device allocations.
    static GpsvBufSet Make(int totalElems, float val = 1.0f) {
        GpsvBufSet bs;
        const int n = (totalElems > 0) ? totalElems : 1;
        std::vector<float> dummy(n, val);
        bs.dDs = DeviceBuffer::copyFrom(dummy.data(), n * sizeof(float));
        bs.dDl = DeviceBuffer::copyFrom(dummy.data(), n * sizeof(float));
        bs.dD  = DeviceBuffer::copyFrom(dummy.data(), n * sizeof(float));
        bs.dDu = DeviceBuffer::copyFrom(dummy.data(), n * sizeof(float));
        bs.dDw = DeviceBuffer::copyFrom(dummy.data(), n * sizeof(float));
        bs.dX  = DeviceBuffer::copyFrom(dummy.data(), n * sizeof(float));
        return bs;
    }

    void allocAlignedWorkspace() { dBuf = DeviceBuffer::alloc(256); }

    aclsparseStatus_t call(aclsparseHandle_t h, int algo, int m, int batchCount,
                            void* pBuffer) {
        return aclsparseSgpsvInterleavedBatch(
            h, algo, m,
            static_cast<float*>(dDs.get()),
            static_cast<float*>(dDl.get()),
            static_cast<float*>(dD.get()),
            static_cast<float*>(dDu.get()),
            static_cast<float*>(dDw.get()),
            static_cast<float*>(dX.get()),
            batchCount, pBuffer);
    }

    aclsparseStatus_t call(aclsparseHandle_t h, int algo, int m, int batchCount) {
        return call(h, algo, m, batchCount, dBuf.raw());
    }
};

// ============================================================================
// GTest test fixture
// ============================================================================
class GpsvInterleavedBatchTest : public ::testing::TestWithParam<GpsvInterleavedBatchParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
        spHandle_ = std::make_unique<HandleManager>();
        spHandle_->setStream(env_->stream());
        goldenSelfTestOk_ = RunGoldenSelfTests();
        if (!goldenSelfTestOk_) {
            std::cerr << "[FATAL] QR pentadiagonal golden self-test FAILED!\n";
        }
    }

    static void TearDownTestSuite() {
        spHandle_.reset();
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    inline static std::unique_ptr<HandleManager> spHandle_;
    inline static bool goldenSelfTestOk_ = false;

    static void CheckKnownSolution(const GpsvKnownSolutionCase& test)
    {
        const auto& a = test.matrix;
        const auto actual = GpsvInterleavedBatchNpu(spHandle_->get(), env_->stream(), 0, a.m, a.ds, a.dl, a.d, a.du,
                                                    a.dw, test.rhs, a.batchCount);
        ASSERT_EQ(actual.size(), test.expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            ASSERT_TRUE(std::isfinite(actual[i])) << "index=" << i;
            EXPECT_FLOAT_EQ(actual[i], test.expected[i]) << "index=" << i;
        }
    }

    // Verifies QR pentadiagonal FP64 golden matches known solution within fp64 precision
    static bool RunGoldenSelfTests() {
        bool ok = true;
        // Self-test 1: m=3, batch=2, seed=42 (smallest meaningful pentadiag)
        if (!QrPentadiagGoldenSelfTest(3, 2, 42)) {
            std::cerr << "[Golden self-test] FAIL: m=3, batch=2\n";
            ok = false;
        }
        // Self-test 2: m=5, batch=1, seed=43
        if (!QrPentadiagGoldenSelfTest(5, 1, 43)) {
            std::cerr << "[Golden self-test] FAIL: m=5, batch=1\n";
            ok = false;
        }
        // Self-test 3: m=10, batch=4, seed=44
        if (!QrPentadiagGoldenSelfTest(10, 4, 44)) {
            std::cerr << "[Golden self-test] FAIL: m=10, batch=4\n";
            ok = false;
        }
        // Self-test 4: m=16, batch=4, seed=45 (medium size)
        if (!QrPentadiagGoldenSelfTest(16, 4, 45)) {
            std::cerr << "[Golden self-test] FAIL: m=16, batch=4\n";
            ok = false;
        }
        // Self-test 5: m=1, batch=1 (trivial case)
        if (!QrPentadiagGoldenSelfTest(1, 1, 100)) {
            std::cerr << "[Golden self-test] FAIL: m=1, batch=1\n";
            ok = false;
        }
        // Self-test 6: m=2, batch=1 (single Givens)
        if (!QrPentadiagGoldenSelfTest(2, 1, 101)) {
            std::cerr << "[Golden self-test] FAIL: m=2, batch=1\n";
            ok = false;
        }
        return ok;
    }

    // Generate pentadiagonal matrix data based on CSV parameters
    static PentadiagMatrix GeneratePentadiagMatrix(const GpsvInterleavedBatchParam& p) {
        if (p.case_name.find("identity") != std::string::npos) {
            return makeIdentityPentadiag(p.m, p.batch_count);
        }
        if (p.is_diag_dominant) {
            return makeDiagDominantPentadiag(p.m, p.batch_count, p.seed, p.value_lo, p.value_hi);
        }
        // Non-diagonal-dominant: check for constant diagonal
        double loRange = p.value_hi - p.value_lo;
        if (loRange < 1e-10) {
            float diagVal = static_cast<float>(p.value_lo);
            return makeConstantDiagPentadiag(p.m, p.batch_count, diagVal);
        }
        return makeRandomPentadiag(p.m, p.batch_count, p.value_lo, p.value_hi, p.seed);
    }
};

TEST_F(GpsvInterleavedBatchTest, ZeroRotationsInBandedPermutations)
{
    for (int m = 1; m <= 6; ++m) {
        for (float sign : {1.0F, -1.0F}) {
            SCOPED_TRACE(::testing::Message() << "m=" << m << " sign=" << sign);
            const auto test = MakeGpsvPermutationCase(m, sign);
            ASSERT_NO_FATAL_FAILURE(CheckKnownSolution(test));
        }
    }
}

TEST_F(GpsvInterleavedBatchTest, ScaledWellConditionedSystems)
{
    for (const auto& test : MakeGpsvRangeCases()) {
        SCOPED_TRACE(::testing::Message() << "m=" << test.matrix.m << " rhs0=" << test.rhs.front());
        ASSERT_NO_FATAL_FAILURE(CheckKnownSolution(test));
    }
}

// ============================================================================
// Parameterized test case: GenericSuccess
// ============================================================================
TEST_P(GpsvInterleavedBatchTest, GenericSuccess) {
    auto p = GetParam();
    std::cout << "\n==== " << p.case_name
              << " m=" << p.m << " batch=" << p.batch_count
              << " algo=" << p.algo
              << " diag_dom=" << p.is_diag_dominant << " ====\n";

    // Pre-check golden self-test ran OK
    ASSERT_TRUE(goldenSelfTestOk_)
        << "QR pentadiagonal golden reference self-test failed! Aborting.";

    // Expect success for all CSV cases
    ASSERT_EQ(p.expect_result, "ACL_SPARSE_STATUS_SUCCESS");

    const int m = p.m;
    const int batchCount = p.batch_count;
    const int total = m * batchCount;

    // 1. Generate pentadiagonal matrix
    PentadiagMatrix pent = GeneratePentadiagMatrix(p);

    // 2. Generate right-hand side b
    // For known_sol: compute b = A*x_true (strong verification)
    // For other cases: random b (verify NPU matches golden)
    std::vector<float> b;
    bool isKnownSol = (p.case_name.find("known_sol") != std::string::npos);

    if (isKnownSol) {
        std::mt19937 rng(p.seed + 2000);
        std::uniform_real_distribution<float> xDist(-5.0f, 5.0f);
        std::vector<float> xTrue(total, 0.0f);
        for (int i = 0; i < total; i++) xTrue[i] = xDist(rng);
        b = makeKnownSolutionPentadiag(pent, xTrue);
    } else if (p.is_diag_dominant) {
        b = makePentadiagRHS(m, batchCount, -5.0, 5.0, p.seed + 1000);
    } else {
        // Use seed-derived b for reproducibility (scaled to value range)
        double bRange = std::max(1.0, (p.value_hi - p.value_lo));
        b = makePentadiagRHS(m, batchCount, -bRange, bRange, p.seed + 2000);
    }

    // 3. CPU golden: QR/Givens algorithm FP64
    std::vector<float> xGolden = QrPentadiagSolveGolden(
        pent.ds, pent.dl, pent.d, pent.du, pent.dw, b, m, batchCount);

    // 4. NPU execution
    std::vector<float> xNpu;
    try {
        xNpu = GpsvInterleavedBatchNpu(
            spHandle_->get(), env_->stream(),
            p.algo, m,
            pent.ds, pent.dl, pent.d, pent.du, pent.dw, b,
            batchCount);
    } catch (const std::exception& e) {
        FAIL() << "[" << p.case_name << "] NPU execution threw: " << e.what();
    }

    // 5. Precision verification: MERE/MARE mode
    ASSERT_EQ(xNpu.size(), xGolden.size())
        << "[" << p.case_name << "] Size mismatch: NPU="
        << xNpu.size() << " Golden=" << xGolden.size();

    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MERE_MARE)
       .SetMERE(p.mere_threshold)       // 1e-4
       .SetMARE(p.mare_multiplier);     // 3.0 (MARE < 3.0 * 1e-4 = 3e-4)

    EXPECT_TRUE(Verifier::verifyVector(xNpu, xGolden, cfg, p.case_name))
        << "[" << p.case_name << "] Precision verification FAILED";
}

// ============================================================================
// INSTANTIATE_TEST_SUITE_P from CSV (L0 + L1 cases)
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    GpsvInterleavedBatch,
    GpsvInterleavedBatchTest,
    ::testing::ValuesIn(GetCasesFromCsv<GpsvInterleavedBatchParam>(
        "gpsv_interleaved_batch_test.csv")),
    [](const ::testing::TestParamInfo<GpsvInterleavedBatchParam>& info) {
        return info.param.caseId();
    }
);

// ============================================================================
// Exception path tests (TEST_F, L1 iteration 2)
// These test invalid parameter validation directly via API calls,
// without golden comparison. ACL env is initialized once per suite.
// ============================================================================
class GpsvInterleavedBatchExceptionTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
        spHandle_ = std::make_unique<HandleManager>();
        spHandle_->setStream(env_->stream());
    }
    static void TearDownTestSuite() {
        spHandle_.reset();
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    inline static std::unique_ptr<HandleManager> spHandle_;
};

// m=0 should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, InvalidM0) {
    auto bs = GpsvBufSet::Make(4);
    EXPECT_EQ(bs.call(spHandle_->get(), 0, /*m=*/0, /*bc=*/4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// batchCount=0 should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, InvalidBatchCount0) {
    auto bs = GpsvBufSet::Make(10);
    EXPECT_EQ(bs.call(spHandle_->get(), 0, /*m=*/10, /*bc=*/0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// algo=1 (valid range but not supported) should return ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(GpsvInterleavedBatchExceptionTest, UnsupportedAlgo1) {
    auto bs = GpsvBufSet::Make(10 * 4);
    bs.allocAlignedWorkspace();
    EXPECT_EQ(bs.call(spHandle_->get(), /*algo=*/1, /*m=*/10, /*bc=*/4),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// algo=2 (valid range but not supported) should return ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(GpsvInterleavedBatchExceptionTest, UnsupportedAlgo2) {
    auto bs = GpsvBufSet::Make(10 * 4);
    bs.allocAlignedWorkspace();
    EXPECT_EQ(bs.call(spHandle_->get(), /*algo=*/2, /*m=*/10, /*bc=*/4),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// algo=-1 (out of range) should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, InvalidAlgoNegative) {
    auto bs = GpsvBufSet::Make(10 * 4);
    bs.allocAlignedWorkspace();
    EXPECT_EQ(bs.call(spHandle_->get(), /*algo=*/-1, /*m=*/10, /*bc=*/4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ds=nullptr should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, NullPointerDs) {
    auto bs = GpsvBufSet::Make(4 * 1);
    bs.allocAlignedWorkspace();
    auto ret = aclsparseSgpsvInterleavedBatch(
        spHandle_->get(), 0, /*m=*/4,
        nullptr,  // ds = nullptr
        static_cast<float*>(bs.dDl.get()),
        static_cast<float*>(bs.dD.get()),
        static_cast<float*>(bs.dDu.get()),
        static_cast<float*>(bs.dDw.get()),
        static_cast<float*>(bs.dX.get()),
        /*bc=*/1, bs.dBuf.raw());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// dl=nullptr should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, NullPointerDl) {
    auto bs = GpsvBufSet::Make(4 * 1);
    bs.allocAlignedWorkspace();
    auto ret = aclsparseSgpsvInterleavedBatch(
        spHandle_->get(), 0, /*m=*/4,
        static_cast<float*>(bs.dDs.get()),
        nullptr,  // dl = nullptr
        static_cast<float*>(bs.dD.get()),
        static_cast<float*>(bs.dDu.get()),
        static_cast<float*>(bs.dDw.get()),
        static_cast<float*>(bs.dX.get()),
        /*bc=*/1, bs.dBuf.raw());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// d=nullptr should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, NullPointerD) {
    auto bs = GpsvBufSet::Make(4 * 1);
    bs.allocAlignedWorkspace();
    auto ret = aclsparseSgpsvInterleavedBatch(
        spHandle_->get(), 0, /*m=*/4,
        static_cast<float*>(bs.dDs.get()),
        static_cast<float*>(bs.dDl.get()),
        nullptr,  // d = nullptr
        static_cast<float*>(bs.dDu.get()),
        static_cast<float*>(bs.dDw.get()),
        static_cast<float*>(bs.dX.get()),
        /*bc=*/1, bs.dBuf.raw());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// du=nullptr should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, NullPointerDu) {
    auto bs = GpsvBufSet::Make(4 * 1);
    bs.allocAlignedWorkspace();
    auto ret = aclsparseSgpsvInterleavedBatch(
        spHandle_->get(), 0, /*m=*/4,
        static_cast<float*>(bs.dDs.get()),
        static_cast<float*>(bs.dDl.get()),
        static_cast<float*>(bs.dD.get()),
        nullptr,  // du = nullptr
        static_cast<float*>(bs.dDw.get()),
        static_cast<float*>(bs.dX.get()),
        /*bc=*/1, bs.dBuf.raw());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// dw=nullptr should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, NullPointerDw) {
    auto bs = GpsvBufSet::Make(4 * 1);
    bs.allocAlignedWorkspace();
    auto ret = aclsparseSgpsvInterleavedBatch(
        spHandle_->get(), 0, /*m=*/4,
        static_cast<float*>(bs.dDs.get()),
        static_cast<float*>(bs.dDl.get()),
        static_cast<float*>(bs.dD.get()),
        static_cast<float*>(bs.dDu.get()),
        nullptr,  // dw = nullptr
        static_cast<float*>(bs.dX.get()),
        /*bc=*/1, bs.dBuf.raw());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// x=nullptr should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, NullPointerX) {
    auto bs = GpsvBufSet::Make(4 * 1);
    bs.allocAlignedWorkspace();
    auto ret = aclsparseSgpsvInterleavedBatch(
        spHandle_->get(), 0, /*m=*/4,
        static_cast<float*>(bs.dDs.get()),
        static_cast<float*>(bs.dDl.get()),
        static_cast<float*>(bs.dD.get()),
        static_cast<float*>(bs.dDu.get()),
        static_cast<float*>(bs.dDw.get()),
        nullptr,  // x = nullptr
        /*bc=*/1, bs.dBuf.raw());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// handle=nullptr should return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(GpsvInterleavedBatchExceptionTest, NullHandle) {
    auto bs = GpsvBufSet::Make(4 * 1);
    bs.allocAlignedWorkspace();
    auto ret = aclsparseSgpsvInterleavedBatch(
        nullptr,  // handle = nullptr
        0, /*m=*/4,
        static_cast<float*>(bs.dDs.get()),
        static_cast<float*>(bs.dDl.get()),
        static_cast<float*>(bs.dD.get()),
        static_cast<float*>(bs.dDu.get()),
        static_cast<float*>(bs.dDw.get()),
        static_cast<float*>(bs.dX.get()),
        /*bc=*/1, bs.dBuf.raw());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// pBuffer not 128-byte aligned should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, UnalignedBuffer) {
    auto bs = GpsvBufSet::Make(4 * 1);
    bs.allocAlignedWorkspace();
    void* misalignedPtr = reinterpret_cast<char*>(bs.dBuf.raw()) + 1;
    EXPECT_EQ(bs.call(spHandle_->get(), 0, /*m=*/4, /*bc=*/1, misalignedPtr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// pBuffer=nullptr with m > 1 (workspace required) should return INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, NullBufferMgt1) {
    auto bs = GpsvBufSet::Make(4 * 2);
    EXPECT_EQ(bs.call(spHandle_->get(), 0, /*m=*/4, /*bc=*/2, nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// batchCount=-1 should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, InvalidBatchCountNegative) {
    auto bs = GpsvBufSet::Make(10);
    EXPECT_EQ(bs.call(spHandle_->get(), 0, /*m=*/10, /*bc=*/-1),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// bufferSizeExt with handle=nullptr should return HANDLE_IS_NULLPTR
TEST_F(GpsvInterleavedBatchExceptionTest, BufferSizeExtNullHandle) {
    size_t bufSize = 0;
    EXPECT_EQ(
        aclsparseSgpsvInterleavedBatch_bufferSizeExt(
            nullptr, 0, /*m=*/4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            /*bc=*/1, &bufSize),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// bufferSizeExt with pBufferSizeInBytes=nullptr should return INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, BufferSizeExtNullSizePtr) {
    EXPECT_EQ(
        aclsparseSgpsvInterleavedBatch_bufferSizeExt(
            spHandle_->get(), 0, /*m=*/4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            /*bc=*/1, nullptr),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// bufferSizeExt with algo=-1 should return INVALID_VALUE
TEST_F(GpsvInterleavedBatchExceptionTest, BufferSizeExtInvalidAlgo) {
    size_t bufSize = 0;
    EXPECT_EQ(
        aclsparseSgpsvInterleavedBatch_bufferSizeExt(
            spHandle_->get(), /*algo=*/-1, /*m=*/4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            /*bc=*/1, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// bufferSizeExt with unsupported algo=1 should return NOT_SUPPORTED
TEST_F(GpsvInterleavedBatchExceptionTest, BufferSizeExtUnsupportedAlgo) {
    size_t bufSize = 0;
    EXPECT_EQ(
        aclsparseSgpsvInterleavedBatch_bufferSizeExt(
            spHandle_->get(), /*algo=*/1, /*m=*/4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            /*bc=*/1, &bufSize),
        ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

}  // namespace sparse_test
