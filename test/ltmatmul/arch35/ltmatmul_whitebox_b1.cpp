/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * @file ltmatmul_whitebox_b1.cpp
 * @brief B1 (FastPath gains betaVectorScaling) white-box forged regression case (D5 decision).
 *
 * Defect background: when betaVec is enabled, the host forces the td.beta=0.0f
 * sentinel (beta*C is carried by the vector branch), making FastPath's
 * beta==0.0f always true — in the (alphaVec=0, betaVec=1) mixed state the old
 * FastPath did Cast(acc)->D directly and lost beta*C entirely. This mixed
 * state cannot be constructed via the public setters (disabling ALPHA
 * implicitly disables BETA; enabling BETA implicitly enables ALPHA), so it
 * must be forged: enable ALPHA+BETA through the public setters, then clear
 * the alpha encoding bit of matD->alignment directly (0x80000000, the alpha
 * bit of encode_scaling_flags) — test-infrastructure precedent:
 * prune_check_test.cpp modifies descriptor fields directly.
 *
 * Post-fix behavior: SpltFastPathCond explicitly excludes the mixed state
 * via betaVectorScaling==0 -> correctly takes GeneralPath (the beta vector
 * branch carries beta*C).
 *
 * Scenario 1: FP16 fused (splitK=1), scalar alpha=1 (alpha passed as
 *            nullptr), nonzero beta vector, no bias/act -> assert D
 *            contains beta*C;
 * Scenario 2: INT8->INT32 two-stage (splitK=2), same construction -> same
 *            assertion (INT8 layout-combination constraint
 *            (COL,ROW)->(T,T): A structured (k,m) COL, B dense (n,k) ROW;
 *            with all-1.0 data the physical layout does not affect content).
 * golden: the beta vector branch of MatmulAlphaBetaFp32Vec (alphaVec always 1).
 */

#include "test_common.h"

#include "ltmatmul_npu_wrapper.h"
#include "ltmatmul_golden.h"
#include "ltmatmul_fixture.h"

#include <gtest/gtest.h>

#include <vector>

// White box: encode_scaling_flags encoding bits + md field access.
#include "aclsparselt_internal.h"

using namespace sparse_test;


// Forged result carrier (outside the anonymous namespace for TEST use).
struct B1ForgedResult {
    std::vector<float> hGoldenD;   // FP32-domain golden (beta vector semantics)
    std::vector<float> hNpuD;      // NPU output (converted to FP32)
    aclsparseStatus_t matmulRet = ACL_SPARSE_STATUS_SUCCESS;
};

namespace {

// Storage-domain conversion helper (host float -> T storage bits).
template <typename T>
inline T ToStorage(float v)
{
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return Fp32ToFp16Bits(v);
    } else if constexpr (std::is_same_v<T, bf16_bits_t>) {
        return bf16_bits_t(Fp32ToBf16Bits(v));
    } else {
        return static_cast<T>(v);
    }
}

// ============================================================================
// Forged execution segment (shared skeleton for both scenarios; the
// dimension/descriptor combination is injected by each scenario).
// Data: A/B all 1.0 (matA/matB passed explicitly = already-pruned semantics;
// 2:4 legality is not this case's responsibility), C constant, beta vector
// = 2.0 (per row) -> golden D = k + 2·C.
// The golden beta vector branch reuses MatmulAlphaBetaFp32Vec (alphaVec
// always 1).
// ============================================================================

// ForgeB1MixedState — [codecheck oversized-function split] the descriptor
// segment forging the (alphaVec=0, betaVec=1) mixed state: enable ALPHA+BETA
// through the public setters first (under the linked semantics both turn on
// together), then clear the alpha encoding bit of matD->alignment directly
// (0x80000000, the alpha bit of encode_scaling_flags) — this mixed state
// cannot be constructed via the public setters (disabling ALPHA implicitly
// disables BETA), hence the forge (test-infrastructure precedent:
// prune_check_test.cpp modifies descriptor fields directly). Statements and
// assertions are bit-identical to the pre-split code.
void ForgeB1MixedState(aclsparseLtConstHandle_t handle, SparseLtMatmulDescGuard& md)
{
    int enable = 1;
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(handle, md.ptr(),
        ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &enable, sizeof(int)),
        ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(handle, md.ptr(),
        ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING, &enable, sizeof(int)),
        ACL_SPARSE_STATUS_SUCCESS);

    // Forged: clear the alpha encoding bit -> (alphaVec=0, betaVec=1) mixed state.
    auto* internal = to_matmul_internal(md.get());
    internal->matD->alignment &= ~0x80000000u;
    ASSERT_EQ(decode_alpha_scaling(internal), false);
    ASSERT_EQ(decode_beta_scaling(internal), true);
}

// FillB1GoldenAndReadback — [codecheck oversized-function split] the golden
// computation + NPU output readback segment. golden: alpha bit already
// cleared -> scalar alpha=1 (vector always 1) + beta vector 2.0; A_pruned
// (explicitly passed, all 1.0) -> acc[i,j] = k (all-1.0 multiply-add). The
// readback converts to the FP32 domain. Statements and evaluation order are
// bit-identical to the pre-split code.
template <typename OutT>
void FillB1GoldenAndReadback(int32_t m, int32_t k, int32_t n,
    float cValue, size_t cdEltSize, const std::vector<float>& betaVec,
    const DeviceBuffer& dD, B1ForgedResult& r)
{
    const size_t mk = static_cast<size_t>(m) * k;
    const size_t kn = static_cast<size_t>(k) * n;
    const size_t mn = static_cast<size_t>(m) * n;
    std::vector<float> alphaOnes(static_cast<size_t>(m), 1.0f);
    std::vector<float> Af(mk, 1.0f), Bf(kn, 1.0f), Cf(mn, cValue);
    MatmulAlphaBetaFp32Vec(Af, Bf, Cf, r.hGoldenD, m, k, n, alphaOnes, betaVec);

    // NPU output readback (converted to the FP32 domain).
    std::vector<OutT> hD(mn);
    dD.copyToHost(hD.data(), mn * cdEltSize);
    r.hNpuD = MatmulToFloat(hD);
}

template <typename InT, typename OutT>
void RunB1ForgedCase(int32_t m, int32_t k, int32_t n,
    const SparseLtMatDescGuard& matA, const SparseLtDnMatDescGuard& matB,
    const SparseLtDnMatDescGuard& matC, const SparseLtDnMatDescGuard& matD,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclsparseComputeType_t computeType, float cValue,
    int32_t splitK, int32_t splitKMode, size_t cdEltSize,
    B1ForgedResult& r)
{
    const size_t mk = static_cast<size_t>(m) * k;
    const size_t kn = static_cast<size_t>(k) * n;
    const size_t mn = static_cast<size_t>(m) * n;

    std::vector<InT> hA(mk, ToStorage<InT>(1.0f));
    std::vector<InT> hB(kn, ToStorage<InT>(1.0f));
    std::vector<OutT> hC(mn, ToStorage<OutT>(cValue));
    const std::vector<float> betaVec(static_cast<size_t>(m), 2.0f);

    SparseLtHandleGuard handle;
    SparseLtMatmulDescGuard md(handle.get(), opA, opB,
                               matA.get(), matB.get(), matC.get(), matD.get(),
                               computeType);

    // Enable ALPHA+BETA vector scaling via the public setters, then forge
    // the (alphaVec=0, betaVec=1) mixed state (see ForgeB1MixedState).
    ForgeB1MixedState(handle.get(), md);

    // algSel (splitK/splitKMode injection) + plan + ws.
    SparseLtAlgSelectionGuard algSel(handle.get(), md.get(),
                                     ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    aclsparseStatus_t cfgRet, splitRet;
    SetAlgAttributes(handle.get(), algSel.get(), 0, splitK, cfgRet, splitRet);
    ASSERT_EQ(cfgRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(splitRet, ACL_SPARSE_STATUS_SUCCESS);
    // splitKMode injection (SPLIT_K_MODE attribute).
    int32_t mode = splitKMode;
    ASSERT_EQ(aclsparseLtMatmulAlgSetAttribute(handle.get(), &algSel.get(),
        ACLSPARSELT_MATMUL_SPLIT_K_MODE, &mode, sizeof(int32_t)),
        ACL_SPARSE_STATUS_SUCCESS);
    SparseLtPlanGuard plan(handle.get(), md.get(), algSel.get());
    size_t workspaceSize = 0;
    ASSERT_EQ(aclsparseLtMatmulGetWorkspace(handle.get(), plan.cptr(), &workspaceSize),
              ACL_SPARSE_STATUS_SUCCESS);
    DeviceBuffer dWorkspace = DeviceBuffer::alloc(workspaceSize);

    // Beta vector device buffer (with betaVec enabled, the beta parameter is
    // the vector Device pointer; alpha passed as nullptr = scalar 1.0).
    DeviceBuffer dBetaVec = DeviceBuffer::copyFrom(betaVec.data(),
                                                   betaVec.size() * sizeof(float));

    aclrtStream stream = 0;
    DeviceBuffer dA = DeviceBuffer::copyFrom(hA.data(), mk * sizeof(InT));
    DeviceBuffer dB = DeviceBuffer::copyFrom(hB.data(), kn * sizeof(InT));
    DeviceBuffer dC = DeviceBuffer::copyFrom(hC.data(), mn * cdEltSize);
    DeviceBuffer dD = DeviceBuffer::alloc(mn * cdEltSize);
    r.matmulRet = aclsparseLtMatmul(handle.get(), plan.cptr(), nullptr,
                                    dA.get(), dB.get(), dBetaVec.get(),
                                    dC.get(), dD.get(), dWorkspace.get(), &stream, 1);
    if (r.matmulRet != ACL_SPARSE_STATUS_SUCCESS) { return; }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    // golden: alpha bit cleared -> scalar alpha=1 (vector always 1) + beta
    // vector 2.0; A_pruned (explicitly passed, all 1.0) -> acc[i,j] = k.
    // Then the NPU output readback (see FillB1GoldenAndReadback).
    FillB1GoldenAndReadback<OutT>(m, k, n, cValue, cdEltSize, betaVec, dD, r);
}

}  // namespace

// ============================================================================
// Scenario 1: FP16 fused (splitK=1) — under the forged mixed state, D must
// contain beta*C.
// Under the old code (FastPath without the betaVectorScaling condition),
// beta*C was lost entirely: D = acc = k = 64 != golden = 64 + 2*0.5 = 65 ->
// deterministic failure.
// ============================================================================
// ACL environment fixture (needs aclInit/setDevice before aclsparseLtInit —
// reuses the MatmulEnvMixin paradigm from ltmatmul_fixture.h).
class LtmatmulWhiteboxB1Fixture : public testing::Test, public MatmulEnvMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

TEST_F(LtmatmulWhiteboxB1Fixture, ForgedBetaVecMixedStateFp16Fused)
{
    constexpr int32_t m = 32, k = 64, n = 32;
    SparseLtHandleGuard handle;
    SparseLtMatDescGuard matA(handle.get(), m, k, k, 16, ACL_FLOAT16,
                              ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    SparseLtDnMatDescGuard matB(handle.get(), k, n, n, 16, ACL_FLOAT16,
                                ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matC(handle.get(), m, n, n, 16, ACL_FLOAT16,
                                ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matD(handle.get(), m, n, n, 16, ACL_FLOAT16,
                                ACL_SPARSE_ORDER_ROW);

    B1ForgedResult r;
    RunB1ForgedCase<uint16_t, uint16_t>(m, k, n, matA, matB, matC, matD,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_COMPUTE_16F, /*cValue=*/0.5f,
        /*splitK=*/1, ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL, sizeof(uint16_t), r);
    ASSERT_EQ(r.matmulRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(r.hNpuD.size(), r.hGoldenD.size());
    for (size_t i = 0; i < r.hGoldenD.size(); ++i) {
        EXPECT_FLOAT_EQ(r.hNpuD[i], r.hGoldenD[i])
            << "idx=" << i << " npu=" << r.hNpuD[i] << " golden=" << r.hGoldenD[i]
            << " (beta*C lost means the FastPath mixed-state defect)";
    }
}

// ============================================================================
// Scenario 2: INT8->INT32 two-stage (splitK=2) — same assertion under the
// forged mixed state.
// INT8 layout-combination constraint (COL,ROW)->(T,T): A structured (k,m)
// COL ld=k, B dense (n,k) ROW ld=k (all-1.0 data, so the physical layout does
// not affect content).
// Under the old code, the epilogue INT32 FastPath likewise loses beta*C ->
// D = 64 != golden 66.
// ============================================================================
TEST_F(LtmatmulWhiteboxB1Fixture, ForgedBetaVecMixedStateInt8TwoStage)
{
    constexpr int32_t m = 32, k = 64, n = 32;
    SparseLtHandleGuard handle;
    SparseLtMatDescGuard matA(handle.get(), k, m, k, 16, ACL_INT8,
                              ACL_SPARSE_ORDER_COL, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    SparseLtDnMatDescGuard matB(handle.get(), n, k, k, 16, ACL_INT8,
                                ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matC(handle.get(), m, n, n, 16, ACL_INT32,
                                ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matD(handle.get(), m, n, n, 16, ACL_INT32,
                                ACL_SPARSE_ORDER_ROW);

    B1ForgedResult r;
    RunB1ForgedCase<int8_t, int32_t>(m, k, n, matA, matB, matC, matD,
        ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        ACL_SPARSE_COMPUTE_32I, /*cValue=*/1.0f,
        /*splitK=*/2, ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS, sizeof(int32_t), r);
    ASSERT_EQ(r.matmulRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(r.hNpuD.size(), r.hGoldenD.size());
    for (size_t i = 0; i < r.hGoldenD.size(); ++i) {
        EXPECT_EQ(static_cast<int32_t>(r.hNpuD[i]), static_cast<int32_t>(r.hGoldenD[i]))
            << "idx=" << i << " npu=" << r.hNpuD[i] << " golden=" << r.hGoldenD[i]
            << " (beta*C lost means the FastPath mixed-state defect)";
    }
}
