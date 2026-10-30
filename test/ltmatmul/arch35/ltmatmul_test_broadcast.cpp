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

// =============================================================================
// D2 BATCH_STRIDE=0 跨批广播手写用例族（BC1-BC8 + INT8 代表组）。
//
// 语义（对齐 cuSPARSELt batch_stride=0 官方语义）：batchStride == 0 表示该
// 侧跨批广播——所有 batch 读写 batch 0（kernel 寻址 base + b*0 恒为基址），
// 读侧（A/B/C）为"每批复用 batch 0"，写侧（D）为批间别名写、终态 = 末批
// 输出（单流顺序执行下确定性，BC4 断言）。各侧 stride 独立设置、独立判定
//（CSV 框架单 stride 列无法表达分侧广播，故为手写 TEST_F 族，per-mat
// 属性注入复用 LtBatchMatDescKit 模式）。
//
// 形状 m=k=n=128（紧凑 ROW，span=m*k=k*n=m*n=16384）；FP16 主族覆盖
// elemSize=2 的 V2 天然失效路径，INT8 代表组覆盖 elemSize=1（INT8 的
// (ROW,ROW) op 对并集为 {(N,T),(T,N)}，故 INT8 组 opB=T、B 物理 (n,k)
// 转置铺设，与既有 617/618 用例同构）；BC7 用 FP32 走 splitK=2 两段式
//（temp 区步长为库内独立值 splitK*m*n，逐批独立区与广播的联合验证）。
// 修复前该族全量被三处 `stride<=0` 校验拒绝（INVALID_VALUE），本族为
// 广播放行行为的正向锚点；负值拒绝边界由 WB-L2-52 / PE15 / 负向用例锁定。
// =============================================================================

#include "test_common.h"

#include "ltmatmul_golden.h"
#include "ltmatmul_npu_wrapper.h"
#include "ltmatmul_exception_helper.h"  // MatmulRawParams / MatmulNpuRaw / FillMatmulRawParams
#include "ltmatmul_fixture.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace sparse_test;

namespace {

constexpr int32_t kBcDim = 128;
constexpr int64_t kBcSpan = 128 * 128;  // 紧凑 span（元素单位，三侧同值）

[[noreturn]] inline void Bail(const std::string& what)
{
    throw std::runtime_error(what);
}

// 广播用例配置：per-side stride（0 = 广播）+ splitK + bias。
struct BroadcastCfg {
    int32_t numBatches;
    int64_t strA, strB, strC, strD;
    int32_t splitK = 1;
    int32_t splitKMode = 0;
    bool withBias = false;
};

// per-mat batch 属性注入（LtBatchMatDescKit 模式，stride 按侧独立——
// BATCH_STRIDE setter 无值校验，0/正值均放行；负值由执行期校验拒绝）。
inline void SetOneMatBatchAttrs(aclsparseLtConstHandle_t handle,
                                aclsparseLtMatDescriptor_t* mat,
                                int32_t nb, int64_t stride)
{
    if (aclsparseLtMatDescSetAttribute(handle, mat, ACLSPARSELT_MAT_NUM_BATCHES,
                                       &nb, sizeof(int32_t)) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseLtMatDescSetAttribute(handle, mat, ACLSPARSELT_MAT_BATCH_STRIDE,
                                       &stride, sizeof(int64_t)) != ACL_SPARSE_STATUS_SUCCESS) {
        Bail("broadcast case: batch attrs set failed");
    }
}

// 侧数据生成：恒生成 numBatches 份（广播侧 kernel/golden 均只消费 batch 0 份，
// 即首份——生成全量可让非广播侧直接铺设；transB 时物理 (n,k) 行主序）。
template <typename T>
std::vector<T> GenSideData(int32_t nb, int64_t stride, int32_t rows, int32_t cols,
                           bool transB, uint32_t seed)
{
    const size_t matSize = static_cast<size_t>(rows) * cols;
    std::vector<T> data;
    for (int32_t b = 0; b < nb; ++b) {
        auto batch = MatmulGenTestMatrix<T>(rows, cols, -1.0f, 1.0f, seed + static_cast<uint32_t>(b));
        if (transB) {
            auto phys = TransposeToRowMajor<T>(batch, rows, cols, rows);
            data.insert(data.end(), phys.begin(), phys.end());
        } else {
            data.insert(data.end(), batch.begin(), batch.end());
        }
        for (int64_t pad = static_cast<int64_t>(matSize); pad < stride; ++pad) { data.push_back(T{}); }
    }
    return data;
}

// 广播链执行：A-sparse、N×opB、全 ROW 紧凑、标量 α=1/β=0。transB=true 时
// B 描述符物理 (n,k)。返回读回的 D 缓冲（全量铺设口径）。
template <typename T>
std::vector<T> RunBroadcastChain(const BroadcastCfg& cfg, bool transB,
                                 aclrtStream stream, uint32_t seed)
{
    constexpr int32_t m = kBcDim, k = kBcDim, n = kBcDim;
    constexpr aclDataType kDtype = NpuDtypeTrait<T>::kAclDtype;
    constexpr aclsparseComputeType_t kCt = MatmulComputeTrait<T>::kComputeType;

    SparseLtHandleGuard handle;
    SparseLtMatDescGuard matA(handle.get(), m, k, k, 16, kDtype,
                              ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    // transB 时 B 描述符物理为 (n,k)（ld=k），否则 (k,n)（ld=n）。
    SparseLtDnMatDescGuard matB(handle.get(), transB ? n : k, transB ? k : n,
                                transB ? k : n, 16, kDtype, ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matC(handle.get(), m, n, n, 16, kDtype, ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matD(handle.get(), m, n, n, 16, kDtype, ACL_SPARSE_ORDER_ROW);
    SetOneMatBatchAttrs(handle.get(), &matA.get(), cfg.numBatches, cfg.strA);
    SetOneMatBatchAttrs(handle.get(), &matB.get(), cfg.numBatches, cfg.strB);
    SetOneMatBatchAttrs(handle.get(), &matC.get(), cfg.numBatches, cfg.strC);
    SetOneMatBatchAttrs(handle.get(), &matD.get(), cfg.numBatches, cfg.strD);

    const aclsparseOperation_t opB =
        transB ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
    SparseLtMatmulDescGuard mm(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, opB,
                               matA.get(), matB.get(), matC.get(), matD.get(), kCt);
    SparseLtAlgSelectionGuard algSel(handle.get(), mm.get());
    int32_t algConfigId = 0;
    if (aclsparseLtMatmulAlgSetAttribute(handle.get(), algSel.ptr(),
            ACLSPARSELT_MATMUL_ALG_CONFIG_ID, &algConfigId, sizeof(int32_t)) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseLtMatmulAlgSetAttribute(handle.get(), algSel.ptr(),
            ACLSPARSELT_MATMUL_SPLIT_K, &cfg.splitK, sizeof(int32_t)) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseLtMatmulAlgSetAttribute(handle.get(), algSel.ptr(),
            ACLSPARSELT_MATMUL_SPLIT_K_MODE, &cfg.splitKMode, sizeof(int32_t)) != ACL_SPARSE_STATUS_SUCCESS) {
        Bail("broadcast case: alg attrs set failed");
    }
    SparseLtPlanGuard plan(handle.get(), mm.get(), algSel.get());

    size_t wsSize = 0;
    if (aclsparseLtMatmulGetWorkspace(handle.get(), plan.cptr(), &wsSize) != ACL_SPARSE_STATUS_SUCCESS) {
        Bail("broadcast case: GetWorkspace failed");
    }
    auto dWorkspace = DeviceBuffer::alloc(wsSize);

    // ---- host 数据（恒 nb 份；广播侧 kernel/golden 只消费首份 = batch 0）----
    auto hA = MatmulGenTestMatrix<T>(m, k, -1.0f, 1.0f, seed);
    auto hB = GenSideData<T>(cfg.numBatches, cfg.strB, k, n, transB, seed + 100);
    auto hC = GenSideData<T>(cfg.numBatches, cfg.strC, m, n, false, seed + 200);

    constexpr size_t kElt = NpuDtypeTrait<T>::kEltSize;
    // A 侧（prune 输入/输出）缓冲：广播（strA==0）单份容量；非广播按
    // numBatches*strA 铺设（逐批复制同一份 hA——A[b]=A0 数值等价，golden
    // 仍单次 prune；prune 逐批 launch 读写 b*strA 偏移，容量不足会越界）。
    std::vector<T> hABuf;
    if (cfg.strA == 0) {
        hABuf = hA;
    } else {
        const size_t mk = static_cast<size_t>(m) * k;
        for (int32_t b = 0; b < cfg.numBatches; ++b) {
            hABuf.insert(hABuf.end(), hA.begin(), hA.end());
            for (int64_t pad = static_cast<int64_t>(mk); pad < cfg.strA; ++pad) { hABuf.push_back(T{}); }
        }
    }
    auto dA = DeviceBuffer::copyFrom(hABuf.data(), hABuf.size() * kElt);
    auto dB = DeviceBuffer::copyFrom(hB.data(), hB.size() * kElt);
    auto dC = DeviceBuffer::copyFrom(hC.data(), hC.size() * kElt);
    auto dD = DeviceBuffer::alloc(hC.size() * kElt);
    auto dPruned = DeviceBuffer::alloc(hABuf.size() * kElt);

    // ---- prune（A-sparse；descriptor 携带广播 batch 属性，全部 launch 落 batch 0）----
    aclsparseLtConstMatmulDescriptor_t mdForPrune = mm.get();
    if (aclsparseLtSpMMAPrune(handle.get(), &mdForPrune, dA.get(), dPruned.get(),
                             ACLSPARSELT_PRUNE_SPMMA_STRIP, stream) != ACL_SPARSE_STATUS_SUCCESS) {
        Bail("broadcast case: SpMMAPrune failed");
    }

    // ---- 可选 bias（biasStride=0 跨批广播，与 A 广播正交叠加；dtype = C dtype）----
    DeviceBuffer dBias;
    if (cfg.withBias) {
        auto biasF = GenBiasVector(m, seed + 300);
        if constexpr (std::is_same_v<T, uint16_t>) {
            // bias dtype = C dtype = FP16：FP32 值经 FP16 roundtrip 量化上传。
            std::vector<uint16_t> hBias16(m);
            for (int32_t i = 0; i < m; ++i) { hBias16[i] = Fp32ToFp16Bits(biasF[i]); }
            dBias = DeviceBuffer::copyFrom(hBias16.data(), m * sizeof(uint16_t));
        } else {
            dBias = DeviceBuffer::copyFrom(biasF.data(), m * sizeof(float));
        }
        void* biasPtr = dBias.get();
        const int64_t biasStride = 0;  // 广播：所有 batch 共用同一 bias
        if (aclsparseLtMatmulDescSetAttribute(handle.get(), mm.ptr(),
                ACLSPARSELT_MATMUL_BIAS_POINTER, &biasPtr, sizeof(void*)) != ACL_SPARSE_STATUS_SUCCESS ||
            aclsparseLtMatmulDescSetAttribute(handle.get(), mm.ptr(),
                ACLSPARSELT_MATMUL_BIAS_STRIDE, &biasStride, sizeof(int64_t)) != ACL_SPARSE_STATUS_SUCCESS) {
            Bail("broadcast case: bias attrs set failed");
        }
    }

    // ---- matmul + sync + 读回 ----
    float alpha = 1.0f, beta = 0.0f;
    MatmulRawParams p;
    FillMatmulRawParams(p, handle.get(), plan.get(), &alpha, dPruned.get(), dB.get(),
        &beta, dC.get(), dD.get(), dWorkspace.get(), &stream, 1);
    if (MatmulNpuRaw(p) != ACL_SPARSE_STATUS_SUCCESS) {
        Bail("broadcast case: Matmul failed");
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        Bail("broadcast case: sync failed");
    }

    std::vector<T> hD(hC.size(), T{});
    dD.copyToHost(hD.data(), hC.size() * kElt);
    return hD;
}

// 广播 golden：CPU FP32 域 prune A（batch 0 数据）+ 逐批 matmul (+bias)，
// 返回 packed 逻辑 golden（nb * m*n）。B 侧取 batch 0 份（广播）或
// b*strB 偏移份（transB 时从物理 (n,k) 转回逻辑 (k,n)）。
template <typename T>
std::vector<T> BroadcastGolden(const BroadcastCfg& cfg, bool transB,
                               const std::vector<T>& hA, const std::vector<T>& hB,
                               const std::vector<float>* biasF)
{
    constexpr int32_t m = kBcDim, k = kBcDim, n = kBcDim;
    const size_t kn = static_cast<size_t>(k) * n;
    std::vector<float> Af, Bf0, Cf0;
    std::vector<T> dummyC(static_cast<size_t>(m) * n);
    PromoteInputsToFp32<T>(hA, hB, dummyC, Af, Bf0, Cf0, m, k, n);
    std::vector<float> APruned;
    MatmulPruneInputAWithAlg<T>(Af, APruned, m, k, /*isColOrder=*/false, "STRIP");
    std::vector<T> goldenPacked;
    for (int32_t b = 0; b < cfg.numBatches; ++b) {
        const size_t base = (cfg.strB == 0) ? 0
            : static_cast<size_t>(b) * static_cast<size_t>(cfg.strB);
        std::vector<T> bPhys(hB.begin() + base, hB.begin() + base + kn);
        std::vector<T> bLogical = transB
            ? TransposeToRowMajor<T>(bPhys, n, k, n)  // 物理 (n,k) -> 逻辑 (k,n)
            : bPhys;
        std::vector<float> Bf(kn);
        if constexpr (std::is_same_v<T, uint16_t>) {
            // FP16 位模式直转（static_cast 会把位模式误读为整数）。
            for (size_t i = 0; i < kn; ++i) { Bf[i] = Fp16BitsToFp32(bLogical[i]); }
        } else {
            for (size_t i = 0; i < kn; ++i) { Bf[i] = static_cast<float>(bLogical[i]); }
        }
        std::vector<float> Df;
        MatmulAccumulateFp32(APruned, Bf, Df, m, k, n);
        if (biasF != nullptr) { ApplyBiasFp32(Df, *biasF, m, n); }
        std::vector<T> Db;
        TruncateToOutput<T>(Df, Db, static_cast<size_t>(m) * n);
        goldenPacked.insert(goldenPacked.end(), Db.begin(), Db.end());
    }
    return goldenPacked;
}

// 广播断言：非 D 广播逐批比对（D 缓冲 batch b 段 == golden[b]）；
// D 广播（strD==0）读回单份 = 末批终态，断言 == golden[nb-1]。
template <typename T>
void VerifyBroadcast(const BroadcastCfg& cfg, const std::vector<T>& hD,
                     const std::vector<T>& goldenPacked, const std::string& caseId)
{
    constexpr size_t mn = static_cast<size_t>(kBcDim) * kBcDim;
    double atol = 0.0, rtol = 0.0;
    if constexpr (std::is_same_v<T, float>) {
        atol = 1e-5; rtol = 1e-5;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        atol = 5e-3; rtol = 5e-3;
    } else {
        atol = 1.0; rtol = 0.0;  // INT8→INT8：MERE 档（α=1/β=0/无 act 实测 EXACT）
    }
    if (cfg.strD == 0) {
        // D 广播：所有批别名写同一缓冲，单流顺序执行终态 = 末批输出。
        std::vector<float> npuF = MatmulToFloat(std::vector<T>(hD.begin(), hD.begin() + mn));
        std::vector<float> goldenF = MatmulToFloat(
            std::vector<T>(goldenPacked.end() - mn, goldenPacked.end()));
        VerifyConfig cfgV;
        applyMixedTolerance(cfgV, NpuDtypeTrait<T>::kAclDtype, goldenF.data(), goldenF.size());
        cfgV.SetMixedTol(atol, rtol);
        EXPECT_TRUE(Verifier::verifyVector(npuF, goldenF, cfgV, caseId))
            << caseId << ": D broadcast final state != last batch golden";
        return;
    }
    for (int32_t b = 0; b < cfg.numBatches; ++b) {
        std::vector<T> npuBatch(hD.begin() + static_cast<size_t>(b) * cfg.strD,
                                hD.begin() + static_cast<size_t>(b) * cfg.strD + mn);
        std::vector<T> goldenBatch(goldenPacked.begin() + static_cast<size_t>(b) * mn,
                                   goldenPacked.begin() + static_cast<size_t>(b + 1) * mn);
        std::vector<float> npuF = MatmulToFloat(npuBatch);
        std::vector<float> goldenF = MatmulToFloat(goldenBatch);
        VerifyConfig cfgV;
        applyMixedTolerance(cfgV, NpuDtypeTrait<T>::kAclDtype, goldenF.data(), goldenF.size());
        cfgV.SetMixedTol(atol, rtol);
        EXPECT_TRUE(Verifier::verifyVector(npuF, goldenF, cfgV,
                    caseId + "/batch" + std::to_string(b)))
            << caseId << ": batch " << b << " mismatch";
    }
}

// 端到端封装：跑链 + golden + 断言（biasF 由 caller 先生成再双处消费）。
template <typename T>
void RunBroadcastCase(const BroadcastCfg& cfg, bool transB, aclrtStream stream,
                      const std::string& caseId, uint32_t seed)
{
    constexpr int32_t m = kBcDim;
    auto hA = MatmulGenTestMatrix<T>(m, m, -1.0f, 1.0f, seed);
    auto hB = GenSideData<T>(cfg.numBatches, cfg.strB, m, m, transB, seed + 100);
    auto hD = RunBroadcastChain<T>(cfg, transB, stream, seed);
    std::vector<float> biasF;
    if (cfg.withBias) {
        biasF = GenBiasVector(m, seed + 300);
        if constexpr (std::is_same_v<T, uint16_t>) {
            for (auto& v : biasF) { v = Fp16BitsToFp32(Fp32ToFp16Bits(v)); }
        }
    }
    auto golden = BroadcastGolden<T>(cfg, transB, hA, hB,
                                     cfg.withBias ? &biasF : nullptr);
    VerifyBroadcast<T>(cfg, hD, golden, caseId);
}

}  // namespace

// =============================================================================
// BC1-BC8：FP16 主族（opB=N，(ROW,ROW)+(N,N)——FP16 无 op×order 约束）。
// =============================================================================

class MatmulBroadcastTest : public MatmulTestFixture {
};

// BC1: A 广播（strA=0，B/C/D 逐批）——D[b] = f(A0, B[b], C[b])。
TEST_F(MatmulBroadcastTest, BC1_ABroadcast) {
    BroadcastCfg cfg{3, 0, kBcSpan, kBcSpan, kBcSpan};
    RunBroadcastCase<uint16_t>(cfg, false, stream_, "BC1-A-broadcast", 5100u);
}

// BC2: B 广播（strB=0）——D[b] = f(A[b], B0, C[b])。
TEST_F(MatmulBroadcastTest, BC2_BBroadcast) {
    BroadcastCfg cfg{3, kBcSpan, 0, kBcSpan, kBcSpan};
    RunBroadcastCase<uint16_t>(cfg, false, stream_, "BC2-B-broadcast", 5101u);
}

// BC3: C 广播（strC=0，β=0 下 C 不参与数值，断言链路成功 + D 逐批正确）。
TEST_F(MatmulBroadcastTest, BC3_CBroadcast) {
    BroadcastCfg cfg{3, kBcSpan, kBcSpan, 0, kBcSpan};
    RunBroadcastCase<uint16_t>(cfg, false, stream_, "BC3-C-broadcast", 5102u);
}

// BC4: D 广播（strD=0）——批间别名写，单流顺序执行终态 = 末批输出
//（确定性断言；D 侧广播末批终态语义的实测锚点）。
TEST_F(MatmulBroadcastTest, BC4_DBroadcastLastBatchFinal) {
    BroadcastCfg cfg{2, kBcSpan, kBcSpan, kBcSpan, 0};
    RunBroadcastCase<uint16_t>(cfg, false, stream_, "BC4-D-broadcast", 5103u);
}

// BC5: A+B 双广播——D[b] = f(A0, B0, C[b])。
TEST_F(MatmulBroadcastTest, BC5_ABDoubleBroadcast) {
    BroadcastCfg cfg{3, 0, 0, kBcSpan, kBcSpan};
    RunBroadcastCase<uint16_t>(cfg, false, stream_, "BC5-AB-broadcast", 5104u);
}

// BC6: 全广播退化（四侧 stride=0）——D == f(A0, B0, C0)（等价单矩阵）。
TEST_F(MatmulBroadcastTest, BC6_AllBroadcastDegenerate) {
    BroadcastCfg cfg{2, 0, 0, 0, 0};
    RunBroadcastCase<uint16_t>(cfg, false, stream_, "BC6-all-broadcast", 5105u);
}

// BC7: A 广播 × two-stage（splitK=2 / TWO_KERNELS / FP32）——temp 区步长
// 为库内独立值（splitK*m*n），逐批独立区与 A 广播的联合验证。
TEST_F(MatmulBroadcastTest, BC7_ABroadcastTwoStage) {
    BroadcastCfg cfg{2, 0, kBcSpan, kBcSpan, kBcSpan,
                    /*splitK=*/2, ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS, false};
    RunBroadcastCase<float>(cfg, false, stream_, "BC7-A-broadcast-twostage", 5106u);
}

// BC8: bias 广播（biasStride=0，既有语义）× A 广播叠加——两路广播正交。
TEST_F(MatmulBroadcastTest, BC8_BiasBroadcastWithABroadcast) {
    BroadcastCfg cfg{3, 0, kBcSpan, kBcSpan, kBcSpan,
                    /*splitK=*/1, ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL, /*withBias=*/true};
    RunBroadcastCase<uint16_t>(cfg, false, stream_, "BC8-bias-broadcast", 5107u);
}

// =============================================================================
// INT8 代表组：elemSize=1 的 V2 天然失效路径。INT8 (ROW,ROW) op 对并集为
// {(N,T),(T,N)}，故 opB=T、B 物理 (n,k) 转置铺设（617/618 同构）。
// =============================================================================

// INT8-A：A 广播 × (N,T) 精度链。
TEST_F(MatmulBroadcastTest, INT8_ABroadcastTransB) {
    BroadcastCfg cfg{3, 0, kBcSpan, kBcSpan, kBcSpan};
    RunBroadcastCase<int8_t>(cfg, true, stream_, "INT8-A-broadcast", 5108u);
}

// INT8-B：全广播退化（等价单矩阵）。
TEST_F(MatmulBroadcastTest, INT8_AllBroadcastTransB) {
    BroadcastCfg cfg{2, 0, 0, 0, 0};
    RunBroadcastCase<int8_t>(cfg, true, stream_, "INT8-all-broadcast", 5109u);
}

// INT8-C：D 广播末批终态（INT8 域的写侧广播语义）。
TEST_F(MatmulBroadcastTest, INT8_DBroadcastLastBatchFinal) {
    BroadcastCfg cfg{2, kBcSpan, kBcSpan, kBcSpan, 0};
    RunBroadcastCase<int8_t>(cfg, true, stream_, "INT8-D-broadcast", 5110u);
}
