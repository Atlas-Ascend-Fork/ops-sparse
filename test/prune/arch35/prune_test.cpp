/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * @file prune_test.cpp
 * @brief GTest + CSV-driven test for aclsparseLtSpMMAPrune (prune-only).
 * Formula: A_pruned = Prune_2:4(A)
 * Test structure:
 *   - TEST_P (PruneTest) : parameterized success-path tests from CSV
 *       1. host data generation (range-bounded random, FP32 -> T)
 *       2. CPU golden reference (prune_golden.h: SpMMAPruneGolden)
 *       3. NPU wrapper (prune-only aclsparseLt chain, stops at SpMMAPrune)
 *       4. D2H gather of A_pruned
 *       5. Element-wise verification (prune is deterministic -> exact match
 *          in FP32 domain via MIXED_TOLERANCE, which for exact-equal inputs
 *          trivially passes)
 * Per-case NPU elapsed time (SpMMAPrune, synchronised) is collected via
 * aclrtEvent and reported for performance profiling.
 * Entry point is shared via test/frame/test_main.cpp; this file MUST NOT
 * define main().
 */

#include "test_common.h"

#include "prune_golden.h"
#include "prune_npu_wrapper.h"
#include "prune_param.h"

#include <chrono>

using namespace sparse_test;

namespace {

class PruneAclEnvScopeMixin {
protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    aclrtStream stream_ = nullptr;
    static void InitEnv() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void FiniEnv() { env_.reset(); }
    void SetUpStream() { stream_ = env_->stream(); }
};

class PruneTestFixture : public testing::Test, public PruneAclEnvScopeMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

template <typename P>
class PruneParamFixture : public testing::TestWithParam<P>, public PruneAclEnvScopeMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

inline void VerifyPruneMixedTolerance(const std::vector<float>& npuF,
                                        const std::vector<float>& goldenF,
                                        aclDataType aclDtype, const std::string& caseId)
{
    ASSERT_EQ(npuF.size(), goldenF.size());
    VerifyConfig cfg;
    applyMixedTolerance(cfg, aclDtype, goldenF.data(), goldenF.size());
    if (aclDtype == ACL_INT8) {
        cfg.mixedRtol = 0.0;
        cfg.mixedAtol = 0.0;
    }
    EXPECT_TRUE(Verifier::verifyVector(npuF, goldenF, cfg, caseId));
}

// Deterministic seed per case for reproducible random inputs.
inline uint32_t SeedForCase(int32_t caseId) { return 2000u + static_cast<uint32_t>(caseId); }

// GenFp32Matrix / ToStorageFp32 / ToStorageFp16 / GenTestMatrix
// 定义在 prune_test_util.h 中，通过 `using namespace sparse_test` 可用。

template <typename T>
inline std::vector<T> PreparePruneTestData(const PruneParam& p)
{
    return MatmulGenTestMatrix<T>(p.m, p.k, p.range_low, p.range_high, SeedForCase(p.case_id));
}

inline void ReportPrunePerf(const PruneParam& p,
                             double msGen, double msGolden, double msNpuWall,
                             const PruneNpuResult& npu)
{
    std::cout << "[" << p.caseId() << "] perf: datagen=" << msGen << "ms"
              << " golden=" << msGolden << "ms"
              << " npu_wall=" << msNpuWall << "ms"
              << " npu_device=" << npu.npuMs << "ms"
              << " ws=" << npu.workspaceSize << "B"
              << " (" << p.m << "x" << p.k
              << " " << p.dtype << " alg" << p.alg_config_id
              << " splitk" << p.split_k
              << " " << (p.isTile() ? "TILE" : "STRIP")
              << " " << (p.isTranspose() ? "T" : "N")
              << (p.isColOrder() ? "COL" : "ROW")
              << " side=" << (p.isSparseA() ? "A" : "B") << ")\n";
}

// Templated end-to-end runner shared by both dtype branches and A/B sparse sides.
// p.m/p.k denote the structured matrix shape (A-sparse: A is (m,k);
// B-sparse: B is (m,k) = (logical k, logical n)). p.op is opA (A-sparse) or
// opB (B-sparse). The prune golden formula is symmetric in (m, k, trans, order),
// so the same SpMMAPruneGolden call covers both sides.
// (b) Comparison (uniform across all batches): the expected physical buffer
// = the golden logical result scattered back to physical per (order, effLd)
// (padding/batch gaps == sentinel), compared bit-by-bit with the NPU's
// fully-read-back physical buffer — data region == golden, padding ==
// sentinel (the behavioral anchor for "kernel does not write padding", and
// under in-place the input padding is preserved as-is), and all batch
// regions consistent batch by batch.

template <typename T>
void RunOneCase(const PruneParam& p, aclrtStream stream) {
    const auto t0 = std::chrono::steady_clock::now();

    // 1. Host data generation for the structured matrix (A-sparse -> A; B-sparse -> B).
    std::vector<T> hInput = PreparePruneTestData<T>(p);
    const auto t1 = std::chrono::steady_clock::now();

    // 2. CPU golden reference (prune-only, dtype + direction + pruneAlg aware).
    std::vector<T> hGoldenPruned;
    SpMMAPruneGolden<T>(hInput, hGoldenPruned, p.m, p.k, p.alg_config_id, p.split_k,
                         p.isTranspose(), !p.isColOrder(), p.pruneAlg);
    const auto t2 = std::chrono::steady_clock::now();

    // 3. NPU execution (prune-only aclsparseLt chain).
    std::vector<T> hNpuPruned;
    const aclsparseOperation_t op = p.isTranspose() ? ACL_SPARSE_OP_TRANSPOSE
                                                    : ACL_SPARSE_OP_NON_TRANSPOSE;
    const aclsparseOrder_t order = p.isColOrder() ? ACL_SPARSE_ORDER_COL
                                                     : ACL_SPARSE_ORDER_ROW;
    const aclsparseLtPruneAlg_t pruneAlg = p.isTile()
        ? ACLSPARSELT_PRUNE_SPMMA_TILE
        : ACLSPARSELT_PRUNE_SPMMA_STRIP;
    auto npu = p.isSparseA()
        ? PruneNpu<T>(stream, hInput, hNpuPruned, p.m, p.k, p.alg_config_id, p.split_k,
                       op, order, pruneAlg, p.num_batches, p.batch_stride, p.ld,
                       p.in_place != 0)
        : PruneNpuBSparse<T>(stream, hInput, hNpuPruned, p.m, p.k, p.alg_config_id, p.split_k,
                             op, order, pruneAlg, p.num_batches, p.batch_stride, p.ld,
                             p.in_place != 0);
    const auto t3 = std::chrono::steady_clock::now();

    // 4. Assert the prune API returned SUCCESS.
    //    (prune is independent of alg_config_id / split_k — no AlgSetAttribute
    //    step in the decoupled prune-only test chain.)
    ASSERT_EQ(npu.pruneRet, ACL_SPARSE_STATUS_SUCCESS)
        << "SpMMAPrune failed for " << p.caseId();

    // 5. Performance report (before verification so timing is always printed).
    double msGen = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double msGolden = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double msNpuWall = std::chrono::duration<double, std::milli>(t3 - t2).count();
    ReportPrunePerf(p, msGen, msGolden, msNpuWall, npu);

    // 6. (b) Verification: the expected physical buffer = the golden logical
    //    result scattered back to physical per (order, effLd) + sentinel
    //    padding, compared bit-by-bit with the NPU's full readback (data-region
    //    correctness + padding preservation + per-segment consistency across
    //    all batches; under in-place the input padding is preserved).
    //    Prune is deterministic: the NPU pruned matrix should match the CPU
    //    golden pruned matrix exactly (same elements zeroed, same values kept).
    const int64_t effLd = PruneEffLdOf(p.m, p.k, p.isTranspose(), order, p.ld);
    const std::vector<T> expectedPhys = ScatterLogicalToPhysicalSentinel<T>(
        hGoldenPruned, p.m, p.k, p.isTranspose(), order, effLd,
        p.num_batches, p.batch_stride);
    ASSERT_EQ(hNpuPruned.size(), expectedPhys.size())
        << p.caseId() << " output buffer size mismatch (npu=" << hNpuPruned.size()
        << ", expected=" << expectedPhys.size() << ")";
    std::vector<float> npuF = PruneToFloat(hNpuPruned);
    std::vector<float> goldenF = PruneToFloat(expectedPhys);
    VerifyPruneMixedTolerance(npuF, goldenF, NpuDtypeTrait<T>::kAclDtype, p.caseId());
}

}  // namespace

// =============================================================================
// Parameterized fixture
// =============================================================================

class PruneTest : public PruneParamFixture<PruneParam> {
};

TEST_P(PruneTest, FunctionalPrecision) {
    const auto& p = GetParam();
    std::cout << "\n==== " << p.caseId() << " ===="
              << " m=" << p.m << " k=" << p.k
              << " dtype=" << p.dtype
              << " alg=" << p.alg_config_id << " split_k=" << p.split_k
              << " pruneAlg=" << (p.isTile() ? "TILE" : "STRIP")
              << " op=" << (p.isTranspose() ? "T" : "N")
              << " order=" << (p.isColOrder() ? "COL" : "ROW")
              << " range=[" << p.range_low << "," << p.range_high << "]"
              << " level=" << p.level << "\n";

    if (p.isFp16()) {
        RunOneCase<uint16_t>(p, stream_);
    } else if (p.isBf16()) {
        RunOneCase<bf16_bits_t>(p, stream_);
    } else if (p.isInt8()) {
        RunOneCase<int8_t>(p, stream_);
    } else {
        RunOneCase<float>(p, stream_);
    }
}

INSTANTIATE_TEST_SUITE_P(
    PruneCases,
    PruneTest,
    testing::ValuesIn(GetCasesFromCsv<PruneParam>("prune_test.csv")),
    [](const testing::TestParamInfo<PruneParam>& info) {
        return info.param.caseId();
    }
);

// =============================================================================
// Exception-path tests for aclsparseLtSpMMAPrune.
// Verifies parameter validation: null handle, null descriptor, null d_in/d_out,
// invalid pruneAlg, and null stream.
// =============================================================================

class PruneExceptionTest : public PruneTestFixture {
};

// Helper to create a valid matmul descriptor for prune tests is unnecessary —
// each test builds its own descriptors via RAII guards for clarity.

// Shared descriptor group for prune exception tests (PE3/PE10/PE11/PE12 and
// PruneTestCtx): handle + matA(structured 128³) + matB/matC/matD(dense 128³).
// Two-phase: batch attributes (NUM_BATCHES/BATCH_STRIDE on matA, consistency
// check requires them set before MatmulDescriptorInit) are injected by the
// caller between construction and InitMatmulDescr(), so ASSERT semantics stay
// in the test body. Extracted from the duplicated creation chains in the
// PE3/PE10/PE11/PE12 test bodies (codecheck duplicate-code cleanup).
struct PruneExcDescrs {
    SparseLtHandleGuard handle;
    MatDescrGuard matA;
    MatDescrGuard matB;
    MatDescrGuard matC;
    MatDescrGuard matD;
    MatmulDescrGuard md;

    PruneExcDescrs()
    {
        matA.initStructured(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                            ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        matB.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                       ACL_SPARSE_ORDER_ROW);
        matC.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                       ACL_SPARSE_ORDER_ROW);
        matD.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                       ACL_SPARSE_ORDER_ROW);
    }

    void InitMatmulDescr()
    {
        md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE,
                ACL_SPARSE_OP_NON_TRANSPOSE,
                matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), ACL_SPARSE_COMPUTE_32F);
    }
};

// Shared descriptor chain for prune exception tests PE4/PE5/PE6.
// PruneExcDescrs + 128³ dA/dAPruned buffers; these three tests all create the
// same handle + dA + dAPruned + matA + matB + matC + matD + matmulDesc,
// differing only in the SpMMAPrune call parameters. Members are initialized
// in declaration order so each guard's constructor parameters (which reference
// earlier members) are valid.
struct PruneTestCtx : PruneExcDescrs {
    sparse_test::DeviceBuffer dA;
    sparse_test::DeviceBuffer dAPruned;

    PruneTestCtx()
        : PruneExcDescrs(),
          dA(sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(float))),
          dAPruned(sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(float)))
    {
        InitMatmulDescr();
    }
};

// PE10/PE11/PE12/PE13 共用 batch 布局违规用例构造（nb/stride 参数化）后建
// matmulDescr 并触发 prune，期望 INVALID_VALUE——batch 布局违规在 prune 校验
// 层（validate_prune_batch V1/V2/V3）拦截；bufBytes 为占位缓冲字节数（溢出
// 用例用最小占位，不实际触达读写）。自四用例的重复注入/建链段提取
// （codecheck duplicate-code cleanup）。
// NUM_BATCHES 须对 A/B/C/D 四 mat 同设：
// ValidateBatchConsistency（MatmulDescriptorInit 层）要求四 mat 的 numBatches
// 一致，此前仅对 matA 单侧注入会在 desc init 即被拒（md=null），
// SpMMAPrune 收到 null 描述符返回 INVALID_VALUE——用例碰巧通过但从未触达
// prune 的 V1/V2/V3 校验路径。BATCH_STRIDE 仅对 structured matA 注入
// （prune 校验对象；setter 对任意 int64 放行，含 2^61）；B/C/D 保持默认
// stride=0（dense 侧布局不在 prune 校验范围，desc init 层对 C/D 的 batch
// 跨度检查在 stride=0 时放行）。
static void RunBatchLayoutViolationCase(aclrtStream stream, int32_t nb,
    int64_t stride, size_t bufBytes)
{
    PruneExcDescrs d;
    for (aclsparseLtMatDescriptor_t* mat :
         {d.matA.ptr(), d.matB.ptr(), d.matC.ptr(), d.matD.ptr()}) {
        ASSERT_EQ(aclsparseLtMatDescSetAttribute(d.handle.ptr(), mat,
            ACLSPARSELT_MAT_NUM_BATCHES, &nb, sizeof(int32_t)),
            ACL_SPARSE_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclsparseLtMatDescSetAttribute(d.handle.ptr(), d.matA.ptr(),
        ACLSPARSELT_MAT_BATCH_STRIDE, &stride, sizeof(int64_t)),
        ACL_SPARSE_STATUS_SUCCESS);
    d.InitMatmulDescr();
    // 构造有效性护栏：描述符链必须建立成功，否则违规被 desc init 层提前
    // 拦截（如 batch 一致性不满足），prune 的 V1/V2/V3 从未被触达，用例
    // 将退化为对 null 描述符碰巧返回 INVALID_VALUE。
    ASSERT_NE(d.md.get(), nullptr);
    sparse_test::DeviceBuffer dA = sparse_test::DeviceBuffer::alloc(bufBytes);
    sparse_test::DeviceBuffer dOut = sparse_test::DeviceBuffer::alloc(bufBytes);
    auto ret = aclsparseLtSpMMAPrune(
        d.handle.ptr(), matmulDescCptr(d.md), dA.get(), dOut.get(),
        ACLSPARSELT_PRUNE_SPMMA_STRIP, stream);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// PE1: null handle -> expected non-SUCCESS.
TEST_F(PruneExceptionTest, NullHandle) {
    auto ret = aclsparseLtSpMMAPrune(
        nullptr, nullptr, nullptr, nullptr, ACLSPARSELT_PRUNE_SPMMA_STRIP, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// PE2: null matmulDescr -> expected non-SUCCESS.
TEST_F(PruneExceptionTest, NullDescriptor) {
    SparseLtHandleGuard handle;
    sparse_test::DeviceBuffer dA = sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(float));
    sparse_test::DeviceBuffer dAPruned = sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(float));
    auto ret = aclsparseLtSpMMAPrune(
        handle.ptr(), nullptr, dA.get(), dAPruned.get(), ACLSPARSELT_PRUNE_SPMMA_STRIP, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// PE10/X19a: numBatches>1 且 batchStride < span（V1 违规）-> INVALID_VALUE。
// span（FP32 ROW 128×128，ld=128）= 127*128+128 = 16384；取 stride=8192 < span。
// 构造（四 mat 同 nb=3，见 RunBatchLayoutViolationCase）穿过 desc init 层
// 一致性校验，违规由 prune V1（batchStride < span）拦截。
TEST_F(PruneExceptionTest, BatchStrideSmallerThanSpan) {
    RunBatchLayoutViolationCase(stream_, 3, 8192,
        static_cast<size_t>(3) * 8192 * sizeof(float));
}

// PE11/X19b: numBatches>1 且 batchStride*elemSize 非 16B 倍数（V2 违规）->
// INVALID_VALUE。FP32 elemSize=4，stride=16386（≥span 但 16386*4 % 16 != 0，
// 等价于修复后无乘法判断 16386 % 4 != 0）。构造穿过 desc init 层，违规由
// prune V2（16B 对齐）拦截。
TEST_F(PruneExceptionTest, BatchStrideNot16ByteAligned) {
    RunBatchLayoutViolationCase(stream_, 3, 16386,
        static_cast<size_t>(3) * 16386 * sizeof(float));
}

// PE12/X19c: numBatches×batchStride×elemSize 溢出 int64（V3 违规）->
// INVALID_VALUE。构造 nb=2^20、stride=2^43：单 stride 字节数
// stride*elemSize=2^45 不溢出（与 X19d 的区分点），但
// (nb-1)*stride+span 后 ×elemSize ≈ 2^65 溢出，触发 V3 checked_mul_add
// 拦截（不实际分配内存——校验层在校验通过后才读取缓冲，违规在校验层
// 拦截；DeviceBuffer 用最小占位，prune 不会执行到 kernel）。构造穿过
// desc init 层（C/D 未设 stride，其 batch 跨度检查放行）。
TEST_F(PruneExceptionTest, BatchLayoutOverflowInt64) {
    RunBatchLayoutViolationCase(stream_, 1 << 20, 1LL << 43, 64);
}

// PE13/X19d: numBatches=2 且稀疏 A batchStride*elemSize 乘法本身溢出 int64
// （V2 乘法溢出顺序）-> INVALID_VALUE。FP32 128×128（elemSize=4）下
// stride=1LL<<61 使修复前的 V2 对齐检查先计算 batchStride*elemSize = 2^63
// > INT64_MAX（有符号乘法溢出，先于 V3 的 checked_mul_add 执行，无法保证
// 按预期返回 INVALID_VALUE）。修复后
// （prune_host.cpp V2 无乘法取模判断）：V1 放行（stride ≥ span=16384）、
// V2 放行（2^61 % 4 == 0）、V3 checked_mul_add 兜底拦截
// （totalElems=2^61+span，×4 溢出）-> 稳定 INVALID_VALUE。与 X19c 的区分：
// X19c（nb=2^20、stride=2^43）的单 stride 字节数 stride*elemSize=2^45 不
// 溢出，仅构造 (nb-1)*stride+span 后 ×elemSize 溢出来触发 V3；本条专覆盖
// 「V2 乘法溢出先于受检查乘法执行」的检查顺序缺口（修复后该输入的预期
// 行为由 V3 兜底保证）。构造为四 mat 同设 numBatches=2（穿过
// ValidateBatchConsistency，此前仅 matA 单侧注入的构造在 desc init 即被
// 拒、从未触达 prune 校验——构造缺陷修正）+ 仅稀疏 A 设
// batchStride=1LL<<61。最小占位缓冲（V3 在 resolve_prune_buffers 之前
// 拦截，不触达 kernel）。
TEST_F(PruneExceptionTest, BatchStrideByteMulOverflowInt64) {
    RunBatchLayoutViolationCase(stream_, 2, 1LL << 61, 64);
}

// PE14（负向补例）：COL+NT × batch × m>k 的 V1 低估拦截。
// kernel AlongCol 路径（pruneAlongRow=0、sparseTrans=0）按行主序 (m,k)
// 行距 ld 寻址（SpltPruneLoadRowBlock），单 batch 实际跨度 (m-1)*ld+k =
// 255*256+64 = 65344；修复前 V1 按列主序跨度 (k-1)*ld+m = 63*256+256 =
// 16384 低估——batchStride=20000（≥16384、%8==0 对齐通过）曾放行，
// V1 batch 跨度低估拒绝（(b) 口径）：(N,COL) 为 ColView（trans 路径），
// 物理按 (k,m) 行主序 ld 寻址，单 batch 访问跨度 = (k-1)*ld+m =
// 63*256+256 = 16384；stride=16128 < 16384 → V1 拒绝 INVALID_VALUE
//（batch 区重叠，kernel 按 batch 推进时读入下一 batch 区域 + 末批越界；
// 不触达 kernel，占位缓冲安全）。
// 构造：matA structured (256,64) COL ld=256 FP16（COL 序 ld ≥ rows=256、
// 16 倍数对齐全过）；四 mat 同设 numBatches=2 穿过一致性校验，仅 matA
// 注入 batchStride=16128（B/C/D 保持 stride=0 放行）。
TEST_F(PruneExceptionTest, ColNTBatchSpanUnderestimateRejected) {
    SparseLtHandleGuard handle;
    MatDescrGuard matA;
    matA.initStructured(handle.ptr(), 256, 64, 256, 16, ACL_FLOAT16,
                        ACL_SPARSE_ORDER_COL, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    // md_k = matA->cols = 64：matB dense (64,128)，matC/matD dense (256,128)。
    MatDescrGuard matB;
    matB.initDense(handle.ptr(), 64, 128, 128, 16, ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
    MatDescrGuard matC;
    matC.initDense(handle.ptr(), 256, 128, 128, 16, ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
    MatDescrGuard matD;
    matD.initDense(handle.ptr(), 256, 128, 128, 16, ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
    const int32_t nb = 2;
    for (aclsparseLtMatDescriptor_t* mat :
         {matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr()}) {
        ASSERT_EQ(aclsparseLtMatDescSetAttribute(handle.ptr(), mat,
            ACLSPARSELT_MAT_NUM_BATCHES, &nb, sizeof(int32_t)),
            ACL_SPARSE_STATUS_SUCCESS);
    }
    const int64_t stride = 16128;  // %8==0、< (b) ColView span(16384)
    ASSERT_EQ(aclsparseLtMatDescSetAttribute(handle.ptr(), matA.ptr(),
        ACLSPARSELT_MAT_BATCH_STRIDE, &stride, sizeof(int64_t)),
        ACL_SPARSE_STATUS_SUCCESS);
    MatmulDescrGuard md;
    md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), ACL_SPARSE_COMPUTE_32F);
    ASSERT_NE(md.get(), nullptr);
    // 占位缓冲（V1 在 resolve_prune_buffers 之前拦截，不触达 kernel）。
    sparse_test::DeviceBuffer dA =
        sparse_test::DeviceBuffer::alloc(2 * 16128 * sizeof(uint16_t));
    sparse_test::DeviceBuffer dOut =
        sparse_test::DeviceBuffer::alloc(2 * 16128 * sizeof(uint16_t));
    auto ret = aclsparseLtSpMMAPrune(
        handle.ptr(), matmulDescCptr(md), dA.get(), dOut.get(),
        ACLSPARSELT_PRUNE_SPMMA_STRIP, stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// PE15（D2 负向锁定）：numBatches>1 且 batchStride<0 → INVALID_VALUE。
// D2 广播放行（stride==0 = 跨批广播，正向锚点见 CSV 229/230）不得误伤
// 负值拒绝边界——负 stride 下 prune 逐批 launch 的 batchOff = b*stride*elemSize
// 为负偏移，必然回走越界，恒非法。构造经 RunBatchLayoutViolationCase（四 mat
// 同设 nb、仅稀疏 A 注入 stride=-1，穿过 desc init 层），违规由
// validate_prune_batch 的 stride<0 分支拦截。
TEST_F(PruneExceptionTest, BatchStrideNegativeRejected) {
    RunBatchLayoutViolationCase(stream_, 3, -1, 64);
}

// PE3: null d_in -> expected non-SUCCESS (descriptors no longer bind values,
// so d_in must be passed explicitly to SpMMAPrune).
TEST_F(PruneExceptionTest, NullDinAndMatAValues) {
    PruneTestCtx ctx;
    auto ret = aclsparseLtSpMMAPrune(
        ctx.handle.ptr(), matmulDescCptr(ctx.md), nullptr, ctx.dAPruned.get(),
        ACLSPARSELT_PRUNE_SPMMA_STRIP, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// PE4: null d_out -> expected non-SUCCESS.
TEST_F(PruneExceptionTest, NullDout) {
    PruneTestCtx ctx;
    auto ret = aclsparseLtSpMMAPrune(
        ctx.handle.ptr(), matmulDescCptr(ctx.md), ctx.dA.get(), nullptr,
        ACLSPARSELT_PRUNE_SPMMA_STRIP, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// PE5: [TILE] TILE mode is now a supported algorithm (no longer NOT_SUPPORTED).
// The original UnsupportedPruneAlgTile test expected NOT_SUPPORTED; with TILE
// implementation complete, this test is removed. TILE normal-path coverage is
// provided by CSV cases 23-43. TILE exception paths (null d_out / null handle)
// are covered by TileNullDout and TileNullHandle below.

// PE6: null stream -> expected SUCCESS (default stream is valid in ACL runtime).
// ACL runtime treats nullptr/0 as the default stream, which is
// valid for all aclrtMemcpy / kernel-launch APIs. The prune implementation
// (validate_prune_params) correctly accepts null stream. This aligns with the
// matmul WB_L2_43_Streams0Null fix (null/0 stream is a valid default stream).
TEST_F(PruneExceptionTest, NullStream) {
    PruneTestCtx ctx;
    auto ret = aclsparseLtSpMMAPrune(
        ctx.handle.ptr(), matmulDescCptr(ctx.md), ctx.dA.get(), ctx.dAPruned.get(),
        ACLSPARSELT_PRUNE_SPMMA_STRIP, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// =============================================================================
// [TILE] Exception-path tests with pruneAlg=TILE.
// Verifies that TILE mode also validates parameters (null d_out / null handle).
// =============================================================================

// TILE + null d_out -> expected non-SUCCESS.
// Confirms TILE path also goes through resolve_prune_buffers validation.
TEST_F(PruneExceptionTest, TileNullDout) {
    PruneTestCtx ctx;
    auto ret = aclsparseLtSpMMAPrune(
        ctx.handle.ptr(), matmulDescCptr(ctx.md), ctx.dA.get(), nullptr,
        ACLSPARSELT_PRUNE_SPMMA_TILE, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// TILE + null handle -> expected non-SUCCESS.
TEST_F(PruneExceptionTest, TileNullHandle) {
    auto ret = aclsparseLtSpMMAPrune(
        nullptr, nullptr, nullptr, nullptr,
        ACLSPARSELT_PRUNE_SPMMA_TILE, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// =============================================================================
// [TILE] Property test: TILE 2D structural sparsity.
// TILE pruning applies a row/col joint constraint: within each TS×TS tile,
// every row keeps exactly KEEP nonzeros AND every column keeps exactly KEEP
// nonzeros. STRIP pruning only enforces the per-row constraint (KEEP nonzeros
// per group of TS within each row); the column distribution is unconstrained.
// This test verifies the core value proposition of TILE mode — the 2D
// structural sparsity that enables hardware-aware structured matmul:
//   1. TILE output: each 4×4 tile has exactly 2 nonzeros per row and per column.
//   2. STRIP output: each row group of 4 has exactly 2 nonzeros (row constraint
//      satisfied), but the per-column count within a 4×4 tile is NOT required
//      to be 2 (column constraint not enforced).
// This is a CPU-only property test (no NPU execution needed).
// =============================================================================

namespace {

inline int32_t CountRowNnzInTile(const std::vector<float>& mat, int32_t k,
                                  int32_t baseRow, int32_t baseCol, int32_t ts)
{
    int32_t nnz = 0;
    for (int32_t c = 0; c < ts; ++c) {
        if (mat[static_cast<size_t>(baseRow) * k + (baseCol + c)] != 0.0f) {
            ++nnz;
        }
    }
    return nnz;
}

inline int32_t CountColNnzInTile(const std::vector<float>& mat, int32_t k,
                                  int32_t baseRow, int32_t baseCol, int32_t ts)
{
    int32_t nnz = 0;
    for (int32_t r = 0; r < ts; ++r) {
        if (mat[static_cast<size_t>(baseRow + r) * k + baseCol] != 0.0f) {
            ++nnz;
        }
    }
    return nnz;
}

inline int32_t CountTileTotalNnz(const std::vector<float>& mat, int32_t k,
                                  int32_t baseRow, int32_t baseCol, int32_t ts)
{
    int32_t nnz = 0;
    for (int32_t r = 0; r < ts; ++r) {
        nnz += CountRowNnzInTile(mat, k, baseRow + r, baseCol, ts);
    }
    return nnz;
}

inline void VerifyTile2DConstraint(const std::vector<float>& tileF,
                                    int32_t m, int32_t k, int32_t ts, int32_t keep)
{
    if (ts <= 0) { return; }
    const int32_t safeTs = (ts > 0) ? ts : 1;
    int32_t tileChecked = 0;
    for (int32_t ti = 0; ti < m / safeTs; ++ti) {
        for (int32_t tj = 0; tj < k / safeTs; ++tj) {
            const int32_t br = ti * safeTs;
            const int32_t bc = tj * safeTs;
            for (int32_t r = 0; r < safeTs; ++r) {
                EXPECT_EQ(CountRowNnzInTile(tileF, k, br + r, bc, safeTs), keep)
                    << "TILE row constraint violated at tile(" << ti << "," << tj
                    << ") row " << r;
            }
            for (int32_t c = 0; c < safeTs; ++c) {
                EXPECT_EQ(CountColNnzInTile(tileF, k, br, bc + c, safeTs), keep)
                    << "TILE column constraint violated at tile(" << ti << "," << tj
                    << ") col " << c;
            }
            EXPECT_EQ(CountTileTotalNnz(tileF, k, br, bc, safeTs), keep * safeTs)
                << "TILE total nnz violated at tile(" << ti << "," << tj << ")";
            ++tileChecked;
        }
    }
    std::cout << "[Property] TILE: checked " << tileChecked << " tiles, all satisfy "
              << "2D constraint (2 per row + 2 per col).\n";
}

inline void VerifyStripRowConstraint(const std::vector<float>& stripF,
                                      int32_t m, int32_t k, int32_t ts, int32_t keep)
{
    if (ts <= 0) { return; }
    const int32_t safeTs = (ts > 0) ? ts : 1;
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t gj = 0; gj < k / safeTs; ++gj) {
            EXPECT_EQ(CountRowNnzInTile(stripF, k, i, gj * safeTs, safeTs), keep)
                << "STRIP row constraint violated at row " << i << " group " << gj;
        }
    }
}

inline int32_t CountStripColViolations(const std::vector<float>& stripF,
                                        int32_t m, int32_t k, int32_t ts, int32_t keep)
{
    if (ts <= 0) { return 0; }
    const int32_t safeTs = (ts > 0) ? ts : 1;
    int32_t violations = 0;
    for (int32_t ti = 0; ti < m / safeTs; ++ti) {
        for (int32_t tj = 0; tj < k / safeTs; ++tj) {
            const int32_t br = ti * safeTs;
            const int32_t bc = tj * safeTs;
            for (int32_t c = 0; c < safeTs; ++c) {
                if (CountColNnzInTile(stripF, k, br, bc + c, safeTs) != keep) {
                    ++violations;
                    break;
                }
            }
        }
    }
    return violations;
}

}  // namespace

class PrunePropertyTest : public PruneTestFixture {
};

TEST_F(PrunePropertyTest, TileStructuralSparsity) {
    const int32_t m = 16;
    const int32_t k = 16;
    const int32_t ts = 4;
    const int32_t keep = 2;
    auto hA = GenTestMatrix<uint16_t>(m, k, -1.0f, 1.0f, 7000u);

    std::vector<uint16_t> hStripPruned;
    std::vector<uint16_t> hTilePruned;
    SpMMAPruneGolden<uint16_t>(hA, hStripPruned, m, k, 0, 1, false, true, "STRIP");
    SpMMAPruneGolden<uint16_t>(hA, hTilePruned, m, k, 0, 1, false, true, "TILE");

    auto stripF = PruneToFloat(hStripPruned);
    auto tileF = PruneToFloat(hTilePruned);

    VerifyTile2DConstraint(tileF, m, k, ts, keep);
    VerifyStripRowConstraint(stripF, m, k, ts, keep);

    int32_t stripColViolations = CountStripColViolations(stripF, m, k, ts, keep);
    const int32_t safeTs = (ts > 0) ? ts : 1;
    std::cout << "[Property] STRIP: row constraint satisfied (2 per group per row).\n"
              << "[Property] STRIP: column constraint violated in " << stripColViolations
              << " of " << (m / safeTs) * (k / safeTs) << " tiles (expected: violations likely"
              << " since STRIP has no column constraint).\n";
}

// =============================================================================
// prune in-place (d_in == d_out) cases ((b) caliber: supported for all
// combinations — the output keeps the input layout, each block does
// read-then-write with overlapping read/write regions, and the host sets no
// in-place rejection anymore):
//   N5-1 STRIP x N x ROW in-place -> SUCCESS with values == the
//        out-of-place golden (the README-declared in-place support path,
//        RowView anchor)
//   N5-2 STRIP x T x ROW in-place -> SUCCESS with correct values ((b)
//        ColView in-place anchor; the former NOT_SUPPORTED rejection branch
//        was removed together with the transpose write-back mechanism)
// (For matrix-style coverage of all combinations x compact/padded x
// STRIP/TILE, see the in_place column cases in the CSV.)
// =============================================================================

// 共用：STRIP/N-T/ROW 描述符链（128³ FP32）+ dA 数据
struct PruneInPlaceCtx : PruneExcDescrs {
    sparse_test::DeviceBuffer dA;
    explicit PruneInPlaceCtx(aclsparseOperation_t opA)
    {
        // opA=TRANSPOSE 时 matA 物理 (k,m)：重建描述符组（基类为 N 布局）
        if (opA == ACL_SPARSE_OP_TRANSPOSE) {
            md.init(handle.ptr(), opA, ACL_SPARSE_OP_NON_TRANSPOSE,
                    matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), ACL_SPARSE_COMPUTE_32F);
        } else {
            InitMatmulDescr();
        }
        const size_t mk = 128 * 128;
        dA = sparse_test::DeviceBuffer::copyFrom(
            std::vector<float>(mk, 1.0f).data(), mk * sizeof(float));
    }
};

// N5-1：STRIP × NON_TRANSPOSE × ROW in-place → SUCCESS + 数值正确
TEST_F(PruneExceptionTest, InPlaceStripRowNumerical) {
    PruneInPlaceCtx ctx(ACL_SPARSE_OP_NON_TRANSPOSE);
    const size_t mk = 128 * 128;
    sparse_test::DeviceBuffer dShared = sparse_test::DeviceBuffer::alloc(mk * sizeof(float));
    // 预置输入数据到共享缓冲（in-place：d_in == d_out）
    std::vector<float> hIn(mk, 1.0f);
    ASSERT_EQ(aclrtMemcpy(dShared.get(), mk * sizeof(float), hIn.data(), mk * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    auto ret = aclsparseLtSpMMAPrune(
        ctx.handle.ptr(), matmulDescCptr(ctx.md), dShared.get(), dShared.get(),
        ACLSPARSELT_PRUNE_SPMMA_STRIP, stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    // golden：out-of-place 参考对同一输入（全 1.0 → 每组 4 取前 2 → 1,1,0,0 模式）
    std::vector<float> hOut(mk);
    ASSERT_EQ(aclrtMemcpy(hOut.data(), mk * sizeof(float), dShared.get(), mk * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    std::vector<float> golden;
    SpMMAPruneGolden<float>(hIn, golden, 128, 128, 0, 1,
                            /*transA=*/false, /*isRowOrder=*/true, "STRIP");
    VerifyPruneMixedTolerance(hOut, golden, ACL_FLOAT, "N5-1-inplace-strip-row");
}

// N5-2: STRIP x TRANSPOSE x ROW in-place -> SUCCESS + correct values ((b)
// ColView in-place: the physical (k,m) row-major is read-then-written block
// by block in place. The input must be the physical layout of the stored
// matrix M = Lᵀ (transposed upload); after readback it is transposed back
// to logical (m,k) for comparison with golden — the physical and logical
// zeroed-position sets align through the transpose mapping.
TEST_F(PruneExceptionTest, InPlaceTransposeRowNumerical) {
    PruneInPlaceCtx ctx(ACL_SPARSE_OP_TRANSPOSE);
    const int32_t m = 128;
    const int32_t k = 128;
    const size_t mk = static_cast<size_t>(m) * k;
    std::vector<float> hLogical(mk, 1.0f);
    // Physical input: M = Lᵀ in (k,m) row-major (compact ld=m).
    std::vector<float> hPhys = TransposeToRowMajor<float>(hLogical, m, k, m);
    sparse_test::DeviceBuffer dShared = sparse_test::DeviceBuffer::copyFrom(
        hPhys.data(), mk * sizeof(float));
    auto ret = aclsparseLtSpMMAPrune(
        ctx.handle.ptr(), matmulDescCptr(ctx.md), dShared.get(), dShared.get(),
        ACLSPARSELT_PRUNE_SPMMA_STRIP, stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    std::vector<float> hOutPhys(mk);
    ASSERT_EQ(aclrtMemcpy(hOutPhys.data(), mk * sizeof(float), dShared.get(), mk * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    // Read back physical M_pruned -> transpose back to logical L_pruned.
    std::vector<float> hOut = TransposeToRowMajor<float>(hOutPhys, k, m, k);
    std::vector<float> golden;
    SpMMAPruneGolden<float>(hLogical, golden, m, k, 0, 1,
                            /*transA=*/true, /*isRowOrder=*/true, "STRIP");
    VerifyPruneMixedTolerance(hOut, golden, ACL_FLOAT, "N5-2-inplace-trans-row");
}
