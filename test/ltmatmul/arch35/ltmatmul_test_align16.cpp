/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// =============================================================================
// 16B (not 32B) aligned GM base regression anchors (D5 round: the three
// DataCopyPad-source host checks were relaxed 32B -> 16B, aligned with the
// cuSPARSELt "All pointers must be aligned to 16 bytes" convention).
//
// Anchors (all buffers 32B-aligned by aclrtMalloc, then offset by exactly
// 16 bytes: FP32 +4 elements, FP16/BF16 +8, INT8 +16):
//   Bias16BFp32/FP16/BF16/Int8Fp32Bias  bias pointer, 4 dtypes, fullload
//   ChunkPathVecBias16B                 chunk (per-tile lazy load) path:
//                                       FP16 + alpha/beta vec + bias, m=16384
//                                       (three-vector fullload budget exceeds
//                                       UB -> biasChunkMode degrades)
//   PerBatchBiasStride4x2Batches16B     B3 per-batch bias source: FP32,
//                                       numBatches=2, biasStride=4 (16B step
//                                       -> batch-1 source lands on a
//                                       16B-not-32B base)
//   VecScalingFp16Fullload16B           alpha/beta vector scaling pointers
//
// These anchors pin the measured fact that DataCopyPad tolerates 16B-aligned
// GM sources (the old 32B wording was a historical over-tightening; the
// experiment evidence lives in .cannbot/sparseLt-fix/tmp/d5-16b-experiment.md
// and in the production comments of matmul_host.cpp / prune_check_host.cpp).
// =============================================================================

#include "test_common.h"

#include "ltmatmul_golden.h"
#include "ltmatmul_npu_wrapper.h"
#include "ltmatmul_exception_helper.h"
#include "ltmatmul_fixture.h"

#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace sparse_test;

namespace {

[[noreturn]] inline void Bail(const std::string& what)
{
    throw std::runtime_error(what);
}

// 16B (not 32B) aligned pointer into a 32B-aligned device allocation.
// The trailing pad keeps the misaligned view inside the allocation.
struct Misaligned16B {
    DeviceBuffer buf;
    void* ptr = nullptr;
    explicit Misaligned16B(size_t bytes)
    {
        buf = DeviceBuffer::alloc(bytes + 64);
        uintptr_t base = reinterpret_cast<uintptr_t>(buf.get());
        base = (base + 31) / 32 * 32;      // snap to a 32B boundary
        ptr = reinterpret_cast<void*>(base + 16);  // 16B-aligned, NOT 32B
    }
};

struct ProbeCfg {
    int32_t m, k, n;
    bool transB = false;  // INT8 (ROW,ROW) op-pair union requires (N,T)
    int32_t numBatches = 1;
    int64_t biasStride = 0;   // elements between per-batch bias sources
    bool withBias = false;
    bool withAlphaVec = false;
    bool withBetaVec = false;
};

// bias element count on GM: broadcast (stride 0) -> m; per-batch ->
// (nb-1)*stride + m (GM layout the kernel reads biasVecGM[b*stride .. +m)).
inline size_t BiasGmElems(const ProbeCfg& cfg)
{
    return (cfg.biasStride == 0)
        ? static_cast<size_t>(cfg.m)
        : static_cast<size_t>(cfg.numBatches - 1) * static_cast<size_t>(cfg.biasStride)
              + static_cast<size_t>(cfg.m);
}

// Host-side logical bias value for batch b, row i (GM layout).
inline float BiasAt(const std::vector<float>& biasGm, const ProbeCfg& cfg,
                    int32_t b, int32_t i)
{
    const size_t off = (cfg.biasStride == 0)
        ? 0 : static_cast<size_t>(b) * static_cast<size_t>(cfg.biasStride);
    return biasGm[off + static_cast<size_t>(i)];
}

// Storage-domain -> float (bit-pattern aware for FP16/BF16).
template <typename T>
inline float ToF32(T v)
{
    if constexpr (std::is_same_v<T, uint16_t>) { return Fp16BitsToFp32(v); }
    else if constexpr (std::is_same_v<T, bf16_bits_t>) { return Bf16BitsToFp32(v.value); }
    else { return static_cast<float>(v); }
}

// CPU golden in FP32 domain (per batch): alpha/beta per-row vectors, C add,
// bias add; INT8 accumulation stays exact in int32 before the FP32 epilogue.
// B/C are read per batch straight from the (nb-copy) host buffers.
template <typename T>
std::vector<float> ProbeGoldenFp32(const ProbeCfg& cfg,
                                   const std::vector<T>& hA, const std::vector<T>& hB,
                                   const std::vector<T>& hC,
                                   const std::vector<float>* alphaVec,
                                   const std::vector<float>* betaVec,
                                   const std::vector<float>& biasGm)
{
    const int32_t m = cfg.m, k = cfg.k, n = cfg.n;
    const size_t kn = static_cast<size_t>(k) * n;
    const size_t mn = static_cast<size_t>(m) * n;
    std::vector<float> Af, Bf0, Cf0;
    std::vector<T> dummyC(mn);
    PromoteInputsToFp32<T>(hA, hB, dummyC, Af, Bf0, Cf0, m, k, n);
    std::vector<float> APruned;
    MatmulPruneInputAWithAlg<T>(Af, APruned, m, k, /*isColOrder=*/false, "STRIP");

    std::vector<float> golden;
    for (int32_t b = 0; b < cfg.numBatches; ++b) {
        const size_t bOff = static_cast<size_t>(b);
        for (int32_t i = 0; i < m; ++i) {
            for (int32_t j = 0; j < n; ++j) {
                // int8 keeps the int32 accumulation exact; others accumulate
                // in float (the kernel's L0C/mmad domain caliber).
                float acc = 0.0f;
                auto bElem = [&](int32_t p) -> float {
                    // transB: physical (n,k) row-major -> logical B[p][j] at
                    // phys[j*k + p]; else compact (k,n): B[p][j] at p*n + j.
                    const size_t off = cfg.transB
                        ? static_cast<size_t>(j) * k + p
                        : static_cast<size_t>(p) * n + j;
                    return ToF32<T>(hB[bOff * kn + off]);
                };
                if constexpr (std::is_same_v<T, int8_t>) {
                    int32_t accI = 0;
                    for (int32_t p = 0; p < k; ++p) {
                        accI += static_cast<int8_t>(APruned[static_cast<size_t>(i) * k + p]) *
                                static_cast<int8_t>(bElem(p));
                    }
                    acc = static_cast<float>(accI);
                } else {
                    for (int32_t p = 0; p < k; ++p) {
                        acc += APruned[static_cast<size_t>(i) * k + p] * bElem(p);
                    }
                }
                float d = acc;
                if (alphaVec != nullptr) { d = (*alphaVec)[static_cast<size_t>(i)] * acc; }
                if (betaVec != nullptr) {
                    d += (*betaVec)[static_cast<size_t>(i)] *
                         ToF32<T>(hC[bOff * mn + static_cast<size_t>(i) * n + j]);
                }
                if (cfg.withBias) { d += BiasAt(biasGm, cfg, b, i); }
                golden.push_back(d);
            }
        }
    }
    return golden;
}

// FP16/BF16 storage quantization of a float vector (roundtrip).
template <typename T>
inline std::vector<float> QuantizeVec(const std::vector<float>& v)
{
    std::vector<float> out = v;
    if constexpr (std::is_same_v<T, uint16_t>) {
        for (auto& x : out) { x = Fp16BitsToFp32(Fp32ToFp16Bits(x)); }
    } else if constexpr (std::is_same_v<T, bf16_bits_t>) {
        for (auto& x : out) { x = Bf16BitsToFp32(Fp32ToBf16Bits(x)); }
    }
    return out;
}

// Host + device data bundle for one probe run: logical A (per-batch
// identical), physical B / C (numBatches copies), and the device mirrors
// (A / pruned / B / C / D).
template <typename T>
struct Align16IO {
    std::vector<T> hA;
    std::vector<T> hB;
    std::vector<T> hC;
    DeviceBuffer dA;
    DeviceBuffer dB;
    DeviceBuffer dC;
    DeviceBuffer dD;
    DeviceBuffer dPruned;
};

// Batch attributes (numBatches + batchStride) on all four mat descriptors.
inline void SetAlign16BatchAttrs(aclsparseLtConstHandle_t handle, const ProbeCfg& cfg,
                                 SparseLtMatDescGuard& matA, SparseLtDnMatDescGuard& matB,
                                 SparseLtDnMatDescGuard& matC, SparseLtDnMatDescGuard& matD,
                                 const std::string& caseId)
{
    const int64_t spanA = static_cast<int64_t>(cfg.m) * cfg.k;
    const int64_t spanB = static_cast<int64_t>(cfg.k) * cfg.n;
    const int64_t spanC = static_cast<int64_t>(cfg.m) * cfg.n;
    auto setBatch = [&](aclsparseLtMatDescriptor_t* mat, int64_t stride) {
        if (aclsparseLtMatDescSetAttribute(handle, mat, ACLSPARSELT_MAT_NUM_BATCHES,
                &cfg.numBatches, sizeof(int32_t)) != ACL_SPARSE_STATUS_SUCCESS ||
            aclsparseLtMatDescSetAttribute(handle, mat, ACLSPARSELT_MAT_BATCH_STRIDE,
                &stride, sizeof(int64_t)) != ACL_SPARSE_STATUS_SUCCESS) {
            Bail(caseId + ": batch attrs set failed");
        }
    };
    setBatch(&matA.get(), spanA);
    setBatch(&matB.get(), spanB);
    setBatch(&matC.get(), spanC);
    setBatch(&matD.get(), spanC);
}

// Workspace size query for the probe plan.
inline size_t QueryAlign16Workspace(aclsparseLtConstHandle_t handle, SparseLtPlanGuard& plan,
                                    const std::string& caseId)
{
    size_t wsSize = 0;
    if (aclsparseLtMatmulGetWorkspace(handle, plan.cptr(), &wsSize)
            != ACL_SPARSE_STATUS_SUCCESS) {
        Bail(caseId + ": GetWorkspace failed");
    }
    return wsSize;
}

// Host data (numBatches copies; single batch when nb==1) + device mirrors.
template <typename T>
Align16IO<T> MakeAlign16HostData(const ProbeCfg& cfg, uint32_t seed)
{
    constexpr size_t kElt = NpuDtypeTrait<T>::kEltSize;
    const int32_t m = cfg.m, k = cfg.k, n = cfg.n;
    Align16IO<T> io;
    io.hA = MatmulGenTestMatrix<T>(m, k, -1.0f, 1.0f, seed);
    std::vector<T> hABuf;
    for (int32_t b = 0; b < cfg.numBatches; ++b) {
        hABuf.insert(hABuf.end(), io.hA.begin(), io.hA.end());
    }
    for (int32_t b = 0; b < cfg.numBatches; ++b) {
        auto bb = MatmulGenTestMatrix<T>(k, n, -1.0f, 1.0f, seed + 100 + static_cast<uint32_t>(b));
        if (cfg.transB) {
            auto phys = TransposeToRowMajor<T>(bb, k, n, k);
            io.hB.insert(io.hB.end(), phys.begin(), phys.end());
        } else {
            io.hB.insert(io.hB.end(), bb.begin(), bb.end());
        }
        auto bc = MatmulGenTestMatrix<T>(m, n, -1.0f, 1.0f, seed + 200 + static_cast<uint32_t>(b));
        io.hC.insert(io.hC.end(), bc.begin(), bc.end());
    }
    io.dA = DeviceBuffer::copyFrom(hABuf.data(), hABuf.size() * kElt);
    io.dB = DeviceBuffer::copyFrom(io.hB.data(), io.hB.size() * kElt);
    io.dC = DeviceBuffer::copyFrom(io.hC.data(), io.hC.size() * kElt);
    io.dD = DeviceBuffer::alloc(io.hC.size() * kElt);
    io.dPruned = DeviceBuffer::alloc(hABuf.size() * kElt);
    return io;
}

// Bias probe family member: 16B-not-32B device pointer. Bias dtype keyed on
// (A/B, C) per SpltBiasDType: FP32 or INT8 x 32I -> FP32; FP16/BF16 -> C
// dtype. No-op (biasGmF stays empty, holder keeps its placeholder) without
// cfg.withBias.
template <typename T>
void AttachBiasProbe16B(aclsparseLtConstHandle_t handle, SparseLtMatmulDescGuard& mm,
                        const ProbeCfg& cfg, uint32_t seed, const std::string& caseId,
                        std::vector<float>& biasGmF, Misaligned16B& biasBuf)
{
    constexpr size_t kElt = NpuDtypeTrait<T>::kEltSize;
    const bool biasIsFp32 = std::is_same_v<T, float> || std::is_same_v<T, int8_t>;
    if (!cfg.withBias) { return; }
    // Non-degenerate bias values (mt19937 uniform [-2,2]) — GenBiasVector
    // is pattern-keyed on seed%10 and can degenerate to all-zero/const.
    std::mt19937 biasRng(seed + 300);
    std::uniform_real_distribution<float> biasDist(-2.0f, 2.0f);
    biasGmF.resize(BiasGmElems(cfg));
    for (auto& v : biasGmF) { v = biasDist(biasRng); }
    // FP16/BF16 storage quantizes the host values (golden must match).
    biasGmF = QuantizeVec<T>(biasGmF);
    const size_t gmElems = BiasGmElems(cfg);
    if (biasIsFp32) {
        biasBuf = Misaligned16B(gmElems * sizeof(float));
        if (aclrtMemcpy(biasBuf.ptr, gmElems * sizeof(float), biasGmF.data(),
                gmElems * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            Bail(caseId + ": bias H2D failed");
        }
    } else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, bf16_bits_t>) {
        // pack the quantized float values as C-dtype bit patterns
        std::vector<T> biasGmBits(gmElems);
        for (size_t i = 0; i < gmElems; ++i) {
            if constexpr (std::is_same_v<T, uint16_t>) {
                biasGmBits[i] = Fp32ToFp16Bits(biasGmF[i]);
            } else {
                biasGmBits[i] = bf16_bits_t(Fp32ToBf16Bits(biasGmF[i]));
            }
        }
        biasBuf = Misaligned16B(gmElems * kElt);
        if (aclrtMemcpy(biasBuf.ptr, gmElems * kElt, biasGmBits.data(),
                gmElems * kElt, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            Bail(caseId + ": bias H2D failed");
        }
    }
    void* biasPtr = biasBuf.ptr;
    if (aclsparseLtMatmulDescSetAttribute(handle, mm.ptr(),
            ACLSPARSELT_MATMUL_BIAS_POINTER, &biasPtr, sizeof(void*))
            != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseLtMatmulDescSetAttribute(handle, mm.ptr(),
            ACLSPARSELT_MATMUL_BIAS_STRIDE, &cfg.biasStride, sizeof(int64_t))
            != ACL_SPARSE_STATUS_SUCCESS) {
        Bail(caseId + ": bias attrs set failed");
    }
}

// alpha/beta vector scaling probe: both vectors on 16B-not-32B device bases
// (fullload path). No-op without cfg.withAlphaVec / cfg.withBetaVec.
inline void AttachScalingVecProbe16B(aclsparseLtConstHandle_t handle, SparseLtMatmulDescGuard& mm,
                                     const ProbeCfg& cfg, uint32_t seed, const std::string& caseId,
                                     std::vector<float>& alphaVec, std::vector<float>& betaVec,
                                     Misaligned16B& alphaBuf, Misaligned16B& betaBuf)
{
    if (!cfg.withAlphaVec && !cfg.withBetaVec) { return; }
    const int32_t m = cfg.m;
    alphaVec = GenScalingVector(m, seed + 400);
    betaVec = cfg.withBetaVec ? GenScalingVector(m, seed + 420)
                              : std::vector<float>(static_cast<size_t>(m), 0.0f);
    // Per-row scaling lives in FP32 regardless of T (float[M] vectors).
    alphaBuf = Misaligned16B(static_cast<size_t>(m) * sizeof(float));
    betaBuf = Misaligned16B(static_cast<size_t>(m) * sizeof(float));
    if (aclrtMemcpy(alphaBuf.ptr, m * sizeof(float), alphaVec.data(),
            m * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(betaBuf.ptr, m * sizeof(float), betaVec.data(),
            m * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        Bail(caseId + ": vec H2D failed");
    }
    int enable = 1;
    if (cfg.withAlphaVec &&
        aclsparseLtMatmulDescSetAttribute(handle, mm.ptr(),
            ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &enable, sizeof(int))
            != ACL_SPARSE_STATUS_SUCCESS) {
        Bail(caseId + ": alpha vec attr failed");
    }
    if (cfg.withBetaVec &&
        aclsparseLtMatmulDescSetAttribute(handle, mm.ptr(),
            ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING, &enable, sizeof(int))
            != ACL_SPARSE_STATUS_SUCCESS) {
        Bail(caseId + ": beta vec attr failed");
    }
}

// Prune + matmul + sync + readback. hD is filled unconditionally (safe read
// of the allocated D buffer); the matmul/sync return codes are reported via
// out-params so the orchestrator keeps the fatal ASSERT gate.
template <typename T>
void RunProbeMatmulReadback(aclsparseLtConstHandle_t handle, SparseLtMatmulDescGuard& mm,
                            SparseLtPlanGuard& plan, const ProbeCfg& cfg,
                            Align16IO<T>& io, DeviceBuffer& dWorkspace,
                            const Misaligned16B& alphaBuf, const Misaligned16B& betaBuf,
                            const std::string& caseId,
                            aclsparseStatus_t& mmRet, int32_t& syncRet,
                            std::vector<T>& hD)
{
    constexpr size_t kElt = NpuDtypeTrait<T>::kEltSize;
    aclrtStream stream = 0;
    aclsparseLtConstMatmulDescriptor_t mdForPrune = mm.get();
    if (aclsparseLtSpMMAPrune(handle, &mdForPrune, io.dA.get(), io.dPruned.get(),
            ACLSPARSELT_PRUNE_SPMMA_STRIP, stream) != ACL_SPARSE_STATUS_SUCCESS) {
        Bail(caseId + ": prune failed");
    }
    float alpha = 1.0f, beta = 0.0f;  // scalar copies unused in vec mode;
    // beta=0 keeps the non-vec probes off the beta*C path (golden matches).
    const void* alphaPtr = cfg.withAlphaVec ? alphaBuf.ptr
                                            : static_cast<const void*>(&alpha);
    const void* betaPtr = cfg.withBetaVec ? betaBuf.ptr : static_cast<const void*>(&beta);
    aclsparseLtConstMatmulPlan_t planForApi = plan.get();
    mmRet = aclsparseLtMatmul(handle, &planForApi, alphaPtr, io.dPruned.get(),
        io.dB.get(), betaPtr, io.dC.get(), io.dD.get(), dWorkspace.get(), &stream, 1);
    syncRet = aclrtSynchronizeStream(stream);

    hD.assign(io.hC.size(), T{});
    io.dD.copyToHost(hD.data(), io.hC.size() * kElt);
}

// Quantize a per-batch FP32 golden slice through the output dtype
// (clamp/round to T, mirroring the kernel's output cast), returned in the
// float domain for the comparison.
template <typename T>
std::vector<float> QuantizeGoldenBatch(const std::vector<float>& goldenF, int32_t b, size_t mn)
{
    std::vector<float> goldenRaw(goldenF.begin() + static_cast<size_t>(b) * mn,
                                 goldenF.begin() + static_cast<size_t>(b + 1) * mn);
    std::vector<T> goldenT;
    TruncateToOutput<T>(goldenRaw, goldenT, mn);
    std::vector<float> goldenBatch(mn);
    for (size_t i = 0; i < mn; ++i) { goldenBatch[i] = ToF32<T>(goldenT[i]); }
    return goldenBatch;
}

// Mixed-tolerance config for one batch verify. INT8->INT8 output mirrors
// VerifyInt8Output's authority recipe (ACL_FLOAT dtype base + atol=1 +
// maxAbsErrLim=1) — the float->half->int8 two-step cast allows a +-1
// quantization difference.
template <typename T>
void FillAlign16VerifyConfig(VerifyConfig& cfgV, const std::vector<float>& goldenBatch,
                             double atol, double rtol)
{
    if constexpr (std::is_same_v<T, int8_t>) {
        applyMixedTolerance(cfgV, ACL_FLOAT, goldenBatch.data(), goldenBatch.size());
        cfgV.SetMixedTol(1.0, 0.0);
        cfgV.SetMixedMaxAbsErrLim(1.0);
    } else {
        applyMixedTolerance(cfgV, NpuDtypeTrait<T>::kAclDtype,
                            goldenBatch.data(), goldenBatch.size());
        cfgV.SetMixedTol(atol, rtol);
    }
}

// Golden + verify: CPU FP32-domain golden per batch (alpha/beta per-row
// vectors, C add, bias add), quantized through the output dtype and compared
// under the dtype's mixed tolerance.
template <typename T>
void VerifyAlign16Result(const ProbeCfg& cfg, const Align16IO<T>& io, const std::vector<T>& hD,
                         const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
                         const std::vector<float>& biasGmF, const std::string& caseId)
{
    const int32_t m = cfg.m, n = cfg.n;
    const std::vector<float>* alphaPtrG = cfg.withAlphaVec ? &alphaVec : nullptr;
    const std::vector<float>* betaPtrG = cfg.withBetaVec ? &betaVec : nullptr;
    auto goldenF = ProbeGoldenFp32<T>(cfg, io.hA, io.hB, io.hC, alphaPtrG, betaPtrG, biasGmF);
    double atol = 0.0, rtol = 0.0;
    if constexpr (std::is_same_v<T, float>) {
        atol = 1e-5; rtol = 1e-5;
    } else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, bf16_bits_t>) {
        atol = 5e-3; rtol = 5e-3;
    } else {
        atol = 1.5; rtol = 0.0;  // INT8->INT8 with FP32-domain bias round
    }
    const size_t mn = static_cast<size_t>(m) * n;
    for (int32_t b = 0; b < cfg.numBatches; ++b) {
        std::vector<float> npuF;
        for (size_t i = 0; i < mn; ++i) {
            const T v = hD[static_cast<size_t>(b) * mn + i];
            if constexpr (std::is_same_v<T, uint16_t>) { npuF.push_back(Fp16BitsToFp32(v)); }
            else if constexpr (std::is_same_v<T, bf16_bits_t>) { npuF.push_back(Bf16BitsToFp32(v.value)); }
            else { npuF.push_back(static_cast<float>(v)); }
        }
        auto goldenBatch = QuantizeGoldenBatch<T>(goldenF, b, mn);
        VerifyConfig cfgV;
        FillAlign16VerifyConfig<T>(cfgV, goldenBatch, atol, rtol);
        EXPECT_TRUE(Verifier::verifyVector(npuF, goldenBatch, cfgV,
                    caseId + "/batch" + std::to_string(b)))
            << caseId << ": batch " << b << " mismatch (16B base anchor)";
    }
}

// End-to-end probe runner: builds the full chain with 16B-not-32B device
// pointers for the probe family (bias / scaling vectors), runs matmul, and
// verifies D against the CPU golden. Thin orchestrator over the per-phase
// helpers above (statement-for-statement mirror of the former monolith).
template <typename T>
void RunAlign16Case(const ProbeCfg& cfg, const std::string& caseId, uint32_t seed)
{
    constexpr aclDataType kDtype = NpuDtypeTrait<T>::kAclDtype;
    constexpr aclsparseComputeType_t kCt = MatmulComputeTrait<T>::kComputeType;
    const int32_t m = cfg.m, k = cfg.k, n = cfg.n;

    SparseLtHandleGuard handle;
    SparseLtMatDescGuard matA(handle.get(), m, k, k, 16, kDtype,
                              ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    // transB: B descriptor is physical (n,k) with ld=k (existing INT8
    // transB-case layout); else compact (k,n) with ld=n.
    SparseLtDnMatDescGuard matB(handle.get(), cfg.transB ? n : k, cfg.transB ? k : n,
                                cfg.transB ? k : n, 16, kDtype, ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matC(handle.get(), m, n, n, 16, kDtype, ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matD(handle.get(), m, n, n, 16, kDtype, ACL_SPARSE_ORDER_ROW);
    SetAlign16BatchAttrs(handle.get(), cfg, matA, matB, matC, matD, caseId);
    const aclsparseOperation_t opB = cfg.transB
        ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
    SparseLtMatmulDescGuard mm(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        opB, matA.get(), matB.get(), matC.get(), matD.get(), kCt);
    SparseLtAlgSelectionGuard algSel(handle.get(), mm.get());
    SparseLtPlanGuard plan(handle.get(), mm.get(), algSel.get());
    const size_t wsSize = QueryAlign16Workspace(handle.get(), plan, caseId);
    auto dWorkspace = DeviceBuffer::alloc(wsSize);

    // ---- host data (numBatches copies; single batch when nb==1) ----
    Align16IO<T> io = MakeAlign16HostData<T>(cfg, seed);
    // ---- probe family: 16B-not-32B device pointers ----
    std::vector<float> biasGmF;
    Misaligned16B biasBuf(0);
    AttachBiasProbe16B<T>(handle.get(), mm, cfg, seed, caseId, biasGmF, biasBuf);
    Misaligned16B alphaBuf(0), betaBuf(0);
    std::vector<float> alphaVec, betaVec;
    AttachScalingVecProbe16B(handle.get(), mm, cfg, seed, caseId,
                             alphaVec, betaVec, alphaBuf, betaBuf);
    // ---- prune + matmul + sync + readback ----
    std::vector<T> hD;
    aclsparseStatus_t mmRet = ACL_SPARSE_STATUS_SUCCESS;
    int32_t syncRet = ACL_SUCCESS;  // aclError (sync return), not aclsparseStatus_t
    RunProbeMatmulReadback<T>(handle.get(), mm, plan, cfg, io, dWorkspace,
                              alphaBuf, betaBuf, caseId, mmRet, syncRet, hD);
    ASSERT_EQ(mmRet, ACL_SPARSE_STATUS_SUCCESS) << caseId << ": Matmul return code";
    ASSERT_EQ(syncRet, ACL_SUCCESS) << caseId << ": sync";
    // ---- golden + verify ----
    VerifyAlign16Result<T>(cfg, io, hD, alphaVec, betaVec, biasGmF, caseId);
}

}  // namespace

class MatmulAlign16Test : public MatmulTestFixture {
};

// bias pointer, FP32 fullload, 16B-not-32B base anchor.
TEST_F(MatmulAlign16Test, Bias16BFp32Fullload) {
    ProbeCfg cfg{128, 128, 128};
    cfg.withBias = true;
    RunAlign16Case<float>(cfg, "bias16B-fp32-fullload", 6100u);
}

// bias pointer, FP16 fullload (bias dtype = C dtype), 16B-not-32B base anchor.
TEST_F(MatmulAlign16Test, Bias16BFp16Fullload) {
    ProbeCfg cfg{128, 128, 128};
    cfg.withBias = true;
    RunAlign16Case<uint16_t>(cfg, "bias16B-fp16-fullload", 6101u);
}

// bias pointer, BF16 fullload, 16B-not-32B base anchor.
TEST_F(MatmulAlign16Test, Bias16BBf16Fullload) {
    ProbeCfg cfg{128, 128, 128};
    cfg.withBias = true;
    RunAlign16Case<bf16_bits_t>(cfg, "bias16B-bf16-fullload", 6102u);
}

// bias pointer, INT8 A/B x C/D=INT8 (bias dtype FP32, 32I domain; opB=T
// satisfies the INT8 (ROW,ROW) op-pair union), 16B-not-32B base anchor.
TEST_F(MatmulAlign16Test, Bias16BInt8Fp32Bias) {
    ProbeCfg cfg{128, 128, 128};
    cfg.transB = true;  // INT8 (ROW,ROW) union: op pair must be (N,T)
    cfg.withBias = true;
    RunAlign16Case<int8_t>(cfg, "bias16B-int8-fp32bias", 6103u);
}

// chunk (per-tile lazy load) path anchor — FP16 + alpha vec + beta vec +
// bias, m=16384: the three-vector fullload budget (m*14 bytes) exceeds UB
// and biasChunkMode degrades to per-tile loading; all three pointers sit on
// 16B-not-32B bases. k/n stay minimal (16/16) to keep the run short.
TEST_F(MatmulAlign16Test, ChunkPathVecBias16B) {
    ProbeCfg cfg{16384, 16, 16};
    cfg.withBias = true;
    cfg.withAlphaVec = true;
    cfg.withBetaVec = true;
    RunAlign16Case<uint16_t>(cfg, "chunk16B-vec-bias", 6104u);
}

// B3 per-batch bias source anchor — FP32, numBatches=2, biasStride=4
// (16B step): batch 1's GM source lands on a 16B-not-32B base (the max
// hardware stress point of the family). Batch 0's source stays 32B-aligned.
TEST_F(MatmulAlign16Test, PerBatchBiasStride4x2Batches16B) {
    ProbeCfg cfg{128, 128, 128};
    cfg.numBatches = 2;
    cfg.biasStride = 4;
    cfg.withBias = true;
    RunAlign16Case<float>(cfg, "perBatchBias16B-stride4-batch2", 6105u);
}

// alpha/beta vector scaling pointers anchor, FP16 — both vectors on
// 16B-not-32B bases (fullload path).
TEST_F(MatmulAlign16Test, VecScalingFp16Fullload16B) {
    ProbeCfg cfg{128, 128, 128};
    cfg.withAlphaVec = true;
    cfg.withBetaVec = true;
    RunAlign16Case<uint16_t>(cfg, "vec16B-fp16-fullload", 6106u);
}
