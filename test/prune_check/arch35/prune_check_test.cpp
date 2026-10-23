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

/**
 * @file prune_check_test.cpp
 * @brief aclsparseLtSpMMAPruneCheck（2:4 校验）的 GTest + CSV 驱动测试。
 *
 * 结构：TEST_P（CSV 驱动，expect_status=SUCCESS 正常路径三向一致断言 /
 * DESC_INVALID 描述符层拦截）+ TEST_F（L2 异常路径 E 系列、首调可见性、
 * 多批量语义、host 防御分支白盒）。白盒分支映射见
 * prune_check_whitebox_coverage.md；错误注入用例置于套件尾部。
 *
 * Entry point is shared via test/frame/test_main.cpp; this file MUST NOT
 * define main().
 */

#include "test_common.h"

#include "prune_check_golden.h"
#include "prune_check_npu_wrapper.h"
#include "prune_check_param.h"

// 白盒访问内部结构体（E4b/E8b/E9b 防御分支伪造 —— 公共 API 无法构造
//「md 存在但 matA/matB 非法」的描述符：MatmulDescriptorInit 在 init 时强制
// matA~matD 非空（IsMatDescrValid）且 A/B 恰一个 structured（ValidateSparse
// Position）；numBatches<1 无公共构造入口（SetAttribute 拒绝 < 1）。与
// matmulDescriptorInit 测试的白盒字段访问同机制）。
#include "aclsparselt_mat_descriptor_internal.h"
#include "aclsparselt_matmul_descriptor_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>

namespace {

// =============================================================================
// Fixtures（对齐 prune 测试范式：AclEnvScope 环境初始化 + 流管理）
// =============================================================================

class PruneCheckAclEnvScopeMixin {
protected:
    inline static std::unique_ptr<sparse_test::AclEnvScope> env_;
    aclrtStream stream_ = nullptr;
    static void InitEnv() {
        env_ = std::make_unique<sparse_test::AclEnvScope>();
    }
    static void FiniEnv() { env_.reset(); }
    void SetUpStream() { stream_ = env_->stream(); }
};

class PruneCheckTestFixture : public testing::Test, public PruneCheckAclEnvScopeMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

template <typename P>
class PruneCheckParamFixture : public testing::TestWithParam<P>, public PruneCheckAclEnvScopeMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

// =============================================================================
// 统计与汇报（性能数据口径：多轮采集中位数 + 离散度）
// =============================================================================

struct PerfStats {
    double median = 0.0;
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
    double cvPct = 0.0;   // 变异系数（stddev / mean × 100，离散度指标）
    int rounds = 0;
};

inline PerfStats ComputeStats(std::vector<double> v) {
    PerfStats s;
    s.rounds = static_cast<int>(v.size());
    if (v.empty()) { return s; }
    std::sort(v.begin(), v.end());
    s.min = v.front();
    s.max = v.back();
    const size_t n = v.size();
    s.median = (n % 2 == 1) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
    double sum = 0.0;
    for (double x : v) { sum += x; }
    s.mean = sum / static_cast<double>(n);
    double var = 0.0;
    for (double x : v) { var += (x - s.mean) * (x - s.mean); }
    var /= static_cast<double>(n);
    s.stddev = std::sqrt(var);
    s.cvPct = (s.mean > 0.0) ? (s.stddev / s.mean * 100.0) : 0.0;
    return s;
}

// 每用例确定性种子，保证随机输入可复现。
inline uint32_t SeedForCase(int32_t caseId) { return 4000u + static_cast<uint32_t>(caseId); }

// perf 行（grep 友好，格式对齐 prune 的 perf 行并扩展多轮统计与带宽核算）。
inline void ReportPruneCheckPerf(const sparse_test::PruneCheckParam& p,
                                  double msGen, double msGolden, double msWall,
                                  const PruneCheckNpuResult& npu)
{
    const PerfStats st = ComputeStats(npu.deviceMs);
    const double effBw = (st.median > 0.0)
        ? static_cast<double>(npu.scannedBytes) / st.median / 1.0e6  // B/ms → GB/s
        : 0.0;
    std::cout << "[" << p.caseId() << "] perf: datagen=" << msGen << "ms"
              << " golden=" << msGolden << "ms"
              << " npu_wall=" << msWall << "ms"
              << " npu_device: median=" << st.median << "ms"
              << " mean=" << st.mean << "ms"
              << " min=" << st.min << "ms"
              << " max=" << st.max << "ms"
              << " stddev=" << st.stddev << "ms"
              << " cv=" << st.cvPct << "%"
              << " rounds=" << st.rounds
              << " scanned=" << npu.scannedBytes << "B"
              << " eff_bw=" << effBw << "GB/s"
              << " (" << p.m << "x" << p.k
              << " " << p.dtype
              << " " << (p.isTranspose() ? "T" : "N") << (p.isColOrder() ? "COL" : "ROW")
              << " side=" << (p.isSparseA() ? "A" : "B")
              << " " << p.data_pattern << ")\n";
}

// 权威精度断言的落盘行（2.1 §2.3 口径表逐输出张量：指标/实测/阈值/结论）。
inline void ReportVerdictDValid(const sparse_test::PruneCheckParam& p, int actual, int golden,
                                 int expectValid, int mismatch)
{
    const bool pass = (mismatch == 0) && (actual == expectValid);
    std::cout << "[" << p.caseId() << "] verdict: tensor=d_valid"
              << " metric=EXACT_bitexact"
              << " actual=" << actual
              << " golden=" << golden
              << " expect_valid=" << expectValid
              << " mismatch=" << mismatch
              << " threshold=mismatch==0"
              << " result=" << (pass ? "PASS" : "FAIL") << "\n";
}

inline void ReportVerdictRet(const sparse_test::PruneCheckParam& p, aclsparseStatus_t actual,
                              aclsparseStatus_t expected)
{
    std::cout << "[" << p.caseId() << "] verdict: tensor=ret"
              << " metric=ENUM_exact"
              << " actual=" << static_cast<int>(actual)
              << " expected=" << static_cast<int>(expected)
              << " result=" << (actual == expected ? "PASS" : "FAIL") << "\n";
}

// 已知 pattern 白名单（未知 pattern 显式失败，防构造器静默返回空向量）。
inline bool IsKnownPattern(const std::string& s) {
    static const std::vector<std::string> kKnown = {
        "VALID_SPARSE", "VALID_MIXED", "ALL_ZERO", "SINGLE_VIOLATION", "MULTI_VIOLATION",
        "ALL_VIOLATING", "RANDOM_DENSE", "PRUNED_SIM_STRIP", "PRUNED_SIM_TILE",
        "VALID_NEGZERO", "VALID_SUBNORMAL", "NAN_VALID", "NAN_VIOLATING", "INF_VIOLATING",
    };
    return std::find(kKnown.begin(), kKnown.end(), s) != kKnown.end();
}

// 描述符 32B/elem 整除倍数（FP32:8 / FP16,BF16:16 / INT8:32 —— 与
// GetAlignMultiple(structured 档) 一致，仅用于日志诊断）。
inline int64_t StructAlignMultiple(const std::string& dtype) {
    if (dtype == "float32") { return 8; }
    if (dtype == "int8") { return 32; }
    return 16;  // float16 / bfloat16
}

// =============================================================================
// 正常路径（expect_status=SUCCESS）：5 步流程 + 三向一致权威断言
// =============================================================================

template <typename T>
void RunOneCase(const sparse_test::PruneCheckParam& p, aclrtStream stream) {
    const auto t0 = std::chrono::steady_clock::now();

    // 1. Host data generation（逻辑 (m, k) 行主序）。
    ASSERT_TRUE(IsKnownPattern(p.data_pattern))
        << "unknown data_pattern: " << p.data_pattern;
    std::vector<T> hInput = sparse_test::GenPruneCheckInput<T>(
        p.data_pattern, p.m, p.k, p.isAlongRow(), p.viol_pos,
        p.range_low, p.range_high, SeedForCase(p.case_id));
    ASSERT_EQ(hInput.size(), static_cast<size_t>(p.m) * static_cast<size_t>(p.k))
        << "data generator produced empty/short input for " << p.caseId();
    const auto t1 = std::chrono::steady_clock::now();

    // 2. CPU golden reference（位级判零 + 方向推导同源）。
    const int goldenValid = sparse_test::PruneCheckGolden<T>(hInput, p.m, p.k,
                                                              p.isTranspose(), !p.isColOrder());
    const auto t2 = std::chrono::steady_clock::now();

    // 三向一致第三方：设计期人工预期（非容差项；与 golden 冲突 = 测试设计
    // 缺陷或算子缺陷，须排查定位根因 —— 2.1 §2.1）。
    EXPECT_EQ(goldenValid, p.expect_valid)
        << p.caseId() << ": golden(" << goldenValid << ") != expect_valid("
        << p.expect_valid << ") — test-design defect or operator defect, investigate";

    // 3. NPU execution（A/B 稀疏侧分派；1 warm-up + R 轮计时）。
    const aclsparseOperation_t op = p.isTranspose() ? ACL_SPARSE_OP_TRANSPOSE
                                                    : ACL_SPARSE_OP_NON_TRANSPOSE;
    const aclsparseOrder_t order = p.isColOrder() ? ACL_SPARSE_ORDER_COL
                                                  : ACL_SPARSE_ORDER_ROW;
    PruneCheckNpuResult npu = p.isSparseA()
        ? PruneCheckNpu<T>(stream, hInput, p.m, p.k, op, order)
        : PruneCheckNpuBSparse<T>(stream, hInput, p.m, p.k, op, order);
    const auto t3 = std::chrono::steady_clock::now();

    // 4. 性能与判定落盘行（先于断言输出，保证失败用例也有完整记录）。
    const double msGen = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double msGolden = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double msWall = std::chrono::duration<double, std::milli>(t3 - t2).count();
    ReportPruneCheckPerf(p, msGen, msGolden, msWall, npu);
    const int mismatch = (npu.hValid != goldenValid) ? 1 : 0;
    ReportVerdictDValid(p, npu.hValid, goldenValid, p.expect_valid, mismatch);
    ReportVerdictRet(p, npu.checkRet, ACL_SPARSE_STATUS_SUCCESS);

    // 5. 权威精度断言（口径表：d_valid EXACT 零容差；返回码枚举精确）。
    ASSERT_EQ(npu.checkRet, ACL_SPARSE_STATUS_SUCCESS)
        << "SpMMAPruneCheck failed for " << p.caseId();
    ASSERT_EQ(npu.hValid, goldenValid)
        << p.caseId() << ": d_valid EXACT mismatch (actual=" << npu.hValid
        << " golden=" << goldenValid << "), tolerance=0";
    EXPECT_EQ(npu.hValid, p.expect_valid)
        << p.caseId() << ": d_valid != expect_valid (three-way consistency)";
}

// =============================================================================
// 描述符层拦截路径（expect_status=DESC_INVALID）：
// 非 32B/elem 整除 shape → StructuredDescriptorInit 返回 INVALID_VALUE，
// PruneCheck 不可达（尾组不可达）。
// =============================================================================

template <typename T>
void RunDescInvalidCase(const sparse_test::PruneCheckParam& p) {
    const aclsparseOrder_t order = p.isColOrder() ? ACL_SPARSE_ORDER_COL
                                                  : ACL_SPARSE_ORDER_ROW;
    const auto spec = sparse_test::SparseSideSpecOf<T>(p.m, p.k, p.isTranspose(), order);
    const int64_t mult = StructAlignMultiple(p.dtype);
    // 防御：mult 为 dtype 对齐常量（FP32:8 / FP16,BF16:16 /
    // INT8:32），恒为正的 2 的幂；显式校验该前提，使下方「位与判整除」与取模严格
    // 等价（x % 2^k != 0 ⟺ (x & (2^k-1)) != 0：补码下 x 与 2^k-1 位与即非负数学
    // 余数，判零与截断取模一致）。不可达路径，不影响任何合法输入。
    if (mult <= 0 || (mult & (mult - 1)) != 0) {
        ADD_FAILURE() << p.caseId() << ": StructAlignMultiple(" << p.dtype
                      << ") returned non-power-of-two value " << mult;
        return;
    }
    const int64_t alignMask = mult - 1;  // mult∈{8,16,32} → mask∈{7,15,31}

    // 诊断：定位首个违反整除约束的维度（rows / cols / ld）；mult 为 2 的幂，
    // 位与判整除与取模等价（依据见上方守卫注释），此处无除法/取模位点。
    std::string why;
    if ((spec.physRows & alignMask) != 0) { why = "rows"; }
    else if ((spec.physCols & alignMask) != 0) { why = "cols"; }
    else if ((spec.ld & alignMask) != 0) { why = "ld"; }
    else { why = "ld-constraint(COL ld>=rows)"; }  // 整除全过但仍被拒的兜底诊断
    std::cout << "[" << p.caseId() << "] desc_intercept: shape=" << p.m << "x" << p.k
              << " dtype=" << p.dtype
              << " phys=" << spec.physRows << "x" << spec.physCols << " ld=" << spec.ld
              << " align_multiple=" << mult << " violated=" << why
              << " (32B/elem divisibility rejected at StructuredDescriptorInit;"
              << " tail group unreachable; PruneCheck not reached)\n";

    sparse_test::SparseLtHandleGuard handle;
    const aclsparseStatus_t initRet = sparse_test::TryInitSparseSideDescriptor<T>(
        handle.ptr(), p.m, p.k, p.isTranspose(), order);
    ReportVerdictRet(p, initRet, ACL_SPARSE_STATUS_INVALID_VALUE);
    std::cout << "[" << p.caseId() << "] verdict: tensor=d_valid"
              << " metric=EXACT_bitexact actual=unreachable expected=N/A"
              << " threshold=N/A result=SKIP(descriptor-layer interception)\n";
    ASSERT_EQ(initRet, ACL_SPARSE_STATUS_INVALID_VALUE)
        << p.caseId() << ": non-32B/elem-divisible shape must be rejected at "
        << "aclsparseLtStructuredDescriptorInit (tail group unreachable)";
}

template <typename T>
void DispatchCase(const sparse_test::PruneCheckParam& p, aclrtStream stream) {
    if (p.isDescInvalid()) {
        RunDescInvalidCase<T>(p);
    } else {
        RunOneCase<T>(p, stream);
    }
}

}  // namespace

// =============================================================================
// Parameterized fixture（CSV 驱动）
// =============================================================================

class PruneCheckTest : public PruneCheckParamFixture<sparse_test::PruneCheckParam> {
};

TEST_P(PruneCheckTest, FunctionalPrecision) {
    const auto& p = GetParam();
    std::cout << "\n==== " << p.caseId() << " ===="
              << " m=" << p.m << " k=" << p.k
              << " dtype=" << p.dtype
              << " op=" << (p.isTranspose() ? "T" : "N")
              << " order=" << (p.isColOrder() ? "COL" : "ROW")
              << " alongRow=" << (p.isAlongRow() ? 1 : 0)
              << " side=" << (p.isSparseA() ? "A" : "B")
              << " pattern=" << p.data_pattern
              << (p.viol_pos.empty() ? "" : "(" + p.viol_pos + ")")
              << " range=[" << p.range_low << "," << p.range_high << "]"
              << " level=" << p.level
              << " expect_valid=" << p.expect_valid
              << " expect_status=" << p.expect_status << "\n";

    if (p.isFp16()) {
        DispatchCase<uint16_t>(p, stream_);
    } else if (p.isBf16()) {
        DispatchCase<sparse_test::bf16_bits_t>(p, stream_);
    } else if (p.isInt8()) {
        DispatchCase<int8_t>(p, stream_);
    } else {
        DispatchCase<float>(p, stream_);
    }
}

INSTANTIATE_TEST_SUITE_P(
    PruneCases,
    PruneCheckTest,
    testing::ValuesIn(sparse_test::GetCasesFromCsv<sparse_test::PruneCheckParam>("prune_check_test.csv")),
    [](const testing::TestParamInfo<sparse_test::PruneCheckParam>& info) {
        return info.param.caseId();
    }
);

// =============================================================================
// L2 异常路径（TEST_F，E1-E12 对齐 2.1 §5.3；开发方案 §7 校验表防御分支
// 补充项 E4b/E8b/E9b/E13/E14）
// =============================================================================

class PruneCheckExceptionTest : public PruneCheckTestFixture {
};

// 共享描述符链（FP16 128×128 A-sparse，N+ROW）：handle + dIn + dValid +
// matA(structured) + matB/matC/matD(dense) + matmulDescr。
struct PruneCheckTestCtx {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::DeviceBuffer dIn;
    sparse_test::DeviceBuffer dValid;
    sparse_test::MatDescrGuard matA;
    sparse_test::MatDescrGuard matB;
    sparse_test::MatDescrGuard matC;
    sparse_test::MatDescrGuard matD;
    sparse_test::MatmulDescrGuard md;

    PruneCheckTestCtx()
        : handle(),
          dIn(sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(uint16_t))),
          dValid(sparse_test::DeviceBuffer::alloc(sizeof(int)))
    {
        matA.initStructured(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                            ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        matB.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                       ACL_SPARSE_ORDER_ROW);
        matC.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                       ACL_SPARSE_ORDER_ROW);
        matD.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                       ACL_SPARSE_ORDER_ROW);
        md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE,
                ACL_SPARSE_OP_NON_TRANSPOSE,
                matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), ACL_SPARSE_COMPUTE_32F);
    }
};

// E1: handle == nullptr -> HANDLE_IS_NULLPTR（校验表 #1）。
TEST_F(PruneCheckExceptionTest, NullHandle) {
    PruneCheckTestCtx ctx;
    auto ret = aclsparseLtSpMMAPruneCheck(
        nullptr, sparse_test::matmulDescCptr(ctx.md), ctx.dIn.get(),
        static_cast<int*>(ctx.dValid.get()), stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// E2: handle 非空但 *handle 为空（悬空/未初始化句柄）-> HANDLE_IS_NULLPTR。
TEST_F(PruneCheckExceptionTest, DanglingHandle) {
    PruneCheckTestCtx ctx;
    aclsparseLtHandle_t dangling = nullptr;
    auto ret = aclsparseLtSpMMAPruneCheck(
        &dangling, sparse_test::matmulDescCptr(ctx.md), ctx.dIn.get(),
        static_cast<int*>(ctx.dValid.get()), stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// E3: matmulDescr == nullptr -> INVALID_VALUE（校验表 #2 指针分支）。
TEST_F(PruneCheckExceptionTest, NullMatmulDescr) {
    PruneCheckTestCtx ctx;
    auto ret = aclsparseLtSpMMAPruneCheck(
        ctx.handle.ptr(), nullptr, ctx.dIn.get(),
        static_cast<int*>(ctx.dValid.get()), stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E4: matmulDescr 非空但 *matmulDescr 为空（悬空描述符）-> INVALID_VALUE。
// [触发路径注明] 本构造（未 init 的
// sparse_test::MatmulDescrGuard.ptr()，*matmulDescr == nullptr）实际命中的是校验表 #2 的
// 「*matmulDescr 为空」分支；「matA/matB 为空」分支（#3）经公共 API 不可达
// （aclsparseLtMatmulDescriptorInit 经 IsMatDescrValid 强制 matA~matD 非空，
// aclsparselt_matmul_descriptor.cpp:177），仅作为防伪造描述符的防御分支存在
// —— 由 E4b 白盒覆盖。
TEST_F(PruneCheckExceptionTest, DanglingMatmulDescr) {
    PruneCheckTestCtx ctx;
    sparse_test::MatmulDescrGuard uninitMd;  // *matmulDescr == nullptr
    auto ret = aclsparseLtSpMMAPruneCheck(
        ctx.handle.ptr(), sparse_test::matmulDescCptr(uninitMd), ctx.dIn.get(),
        static_cast<int*>(ctx.dValid.get()), stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E4b（防御分支，白盒伪造）: md 存在但 md->matA == nullptr -> INVALID_VALUE
//（校验表 #3「matA、matB 为空」—— 公共 API 不可构造，见 E4 注释）。
TEST_F(PruneCheckExceptionTest, ForgedNullMatA) {
    PruneCheckTestCtx ctx;
    auto* mdInt = ToInternal(ctx.md.get());
    auto* saved = mdInt->matA;
    mdInt->matA = nullptr;
    auto ret = aclsparseLtSpMMAPruneCheck(
        ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), ctx.dIn.get(),
        static_cast<int*>(ctx.dValid.get()), stream_);
    mdInt->matA = saved;  // 恢复后再由 RAII 析构
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E5: d_in == nullptr -> INVALID_VALUE（校验表 #7）。
TEST_F(PruneCheckExceptionTest, NullDin) {
    PruneCheckTestCtx ctx;
    auto ret = aclsparseLtSpMMAPruneCheck(
        ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), nullptr,
        static_cast<int*>(ctx.dValid.get()), stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E6: d_valid == nullptr -> INVALID_VALUE（校验表 #7）。
TEST_F(PruneCheckExceptionTest, NullDvalid) {
    PruneCheckTestCtx ctx;
    auto ret = aclsparseLtSpMMAPruneCheck(
        ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), ctx.dIn.get(), nullptr, stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E7: d_in 非 32B 对齐（合法 DeviceBuffer 内偏移 4B，同时非 16B/32B 对齐）
// -> INVALID_VALUE（校验表 #8；d_valid 不强制对齐由全部用例天然覆盖）。
TEST_F(PruneCheckExceptionTest, MisalignedDin) {
    PruneCheckTestCtx ctx;
    // 独立缓冲（偏移后地址 ≡ 4 mod 16，缓冲本身多分配防越界）。
    sparse_test::DeviceBuffer buf =
        sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(uint16_t) + 32);
    void* misaligned = sparse_test::MisalignedOffsetPtr(buf);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(misaligned);
    ASSERT_NE(addr % 32u, 0u) << "misaligned pointer construction failed";
    auto ret = aclsparseLtSpMMAPruneCheck(
        ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), misaligned,
        static_cast<int*>(ctx.dValid.get()), stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E7b: d_in 16B 对齐但非 32B 对齐（合法 DeviceBuffer 内偏移 16B，地址
// ≡ 16 mod 32）-> INVALID_VALUE（校验表 #8：d_in 为 kernel DataCopyPad 的
// GM 源基地址，硬件要求 32B 对齐；16B 非 32B 会进入未定义搬运路径——可能
// 507035 或读错数据，须在 host 侧拦截）。
TEST_F(PruneCheckExceptionTest, HalfAlignedDin) {
    PruneCheckTestCtx ctx;
    // 独立缓冲（偏移后地址 ≡ 16 mod 32，缓冲本身多分配防越界）。
    sparse_test::DeviceBuffer buf =
        sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(uint16_t) + 64);
    void* halfAligned = sparse_test::HalfAlignedOffsetPtr(buf);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(halfAligned);
    ASSERT_EQ(addr % 16u, 0u) << "16B-aligned pointer construction failed";
    ASSERT_NE(addr % 32u, 0u) << "non-32B pointer construction failed";
    auto ret = aclsparseLtSpMMAPruneCheck(
        ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), halfAligned,
        static_cast<int*>(ctx.dValid.get()), stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

namespace {

// E8/E9 共用尾段（重复块收敛）：matC/matD dense 占位（FP16 128×128）+
// md.init 断言被 MatmulDescriptorInit 拒绝（INVALID_VALUE）——「无唯一稀疏侧」
// （双 structured / 双 dense）两种形态同归宿；matA/matB 由调用侧构造。
// void 函数内可用 ASSERT/EXPECT（gtest 约束：致命断言要求返回 void）。
void ExpectMatmulInitRejected(const sparse_test::SparseLtHandleGuard& handle,
                              sparse_test::MatDescrGuard& matA, sparse_test::MatDescrGuard& matB)
{
    sparse_test::MatDescrGuard matC;
    ASSERT_EQ(matC.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matD;
    ASSERT_EQ(matD.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatmulDescrGuard md;
    EXPECT_EQ(md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE,
                      ACL_SPARSE_OP_NON_TRANSPOSE,
                      matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(),
                      ACL_SPARSE_COMPUTE_32F),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

}  // namespace

// E8: 双稀疏（A、B 均 50% 稀疏）。
// [落点分层说明] 公共 API 构造（双 initStructured + md.init）
// 在 aclsparseLtMatmulDescriptorInit 即被 INVALID_VALUE 拒绝（ValidateSparse
// Position 要求 A/B 恰一个 structured；与 matmulDescriptorInit 测试 L1-12
// 同口径），链路到不了 PruneCheck —— (a) 断言描述符层拦截。
// PruneCheck 校验表 #4「无唯一稀疏侧 -> NOT_SUPPORTED」（D-1）经公共 API
// 不可达，由 E8b 白盒伪造覆盖。
TEST_F(PruneCheckExceptionTest, DoubleSparse) {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::MatDescrGuard matA;
    ASSERT_EQ(matA.initStructured(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                                  ACL_SPARSE_ORDER_ROW,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matB;
    ASSERT_EQ(matB.initStructured(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                                  ACL_SPARSE_ORDER_ROW,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    // (a) 描述符层拦截：双 structured 被 MatmulDescriptorInit 拒绝。
    ExpectMatmulInitRejected(handle, matA, matB);
}

// E8b（防御分支，白盒伪造）: 合法 A-sparse md 伪造 matB 为 structured
// -> PruneCheck 校验表 #4 -> NOT_SUPPORTED（D-1）。
TEST_F(PruneCheckExceptionTest, ForgedDoubleSparseNotSupported) {
    PruneCheckTestCtx ctx;
    sparse_test::MatDescrGuard matB2;
    ASSERT_EQ(matB2.initStructured(ctx.handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                                   ACL_SPARSE_ORDER_ROW,
                                   ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    auto* mdInt = ToInternal(ctx.md.get());
    auto* saved = mdInt->matB;
    mdInt->matB = matB2.get();
    auto ret = aclsparseLtSpMMAPruneCheck(
        ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), ctx.dIn.get(),
        static_cast<int*>(ctx.dValid.get()), stream_);
    mdInt->matB = saved;
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// E9: 双稠密（A、B 均 dense）—— 分层同 E8：(a) 描述符层 INVALID_VALUE；
// #4 防御分支由 E9b 白盒覆盖（NOT_SUPPORTED）。
TEST_F(PruneCheckExceptionTest, DoubleDense) {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::MatDescrGuard matA;
    ASSERT_EQ(matA.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matB;
    ASSERT_EQ(matB.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    ExpectMatmulInitRejected(handle, matA, matB);
}

// E9b（防御分支，白盒伪造）: 合法 A-sparse md 伪造 matA 为 dense
// -> A、B 均 dense -> PruneCheck #4 -> NOT_SUPPORTED（D-1）。
TEST_F(PruneCheckExceptionTest, ForgedDoubleDenseNotSupported) {
    PruneCheckTestCtx ctx;
    sparse_test::MatDescrGuard matA2;
    ASSERT_EQ(matA2.initDense(ctx.handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                              ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    auto* mdInt = ToInternal(ctx.md.get());
    auto* saved = mdInt->matA;
    mdInt->matA = matA2.get();
    auto ret = aclsparseLtSpMMAPruneCheck(
        ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), ctx.dIn.get(),
        static_cast<int*>(ctx.dValid.get()), stream_);
    mdInt->matA = saved;
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// E10: 稀疏侧 dtype 不支持（ACL_INT32）—— [收紧动作 3] 结构化描述符层可
// 构造（IsSupportedValueType 含 ACL_INT32，GetAlignMultiple structured 档 8，
// 128 均为 8 倍数），链路可达 PruneCheck，由校验表 #6 dtype 白名单拦截
// -> NOT_SUPPORTED。
TEST_F(PruneCheckExceptionTest, Int32SparseSideNotSupported) {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::MatDescrGuard matA;  // structured INT32（可构造）
    ASSERT_EQ(matA.initStructured(handle.ptr(), 128, 128, 128, 16, ACL_INT32,
                                  ACL_SPARSE_ORDER_ROW,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matB;
    ASSERT_EQ(matB.initDense(handle.ptr(), 128, 128, 128, 16, ACL_INT32,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matC;
    ASSERT_EQ(matC.initDense(handle.ptr(), 128, 128, 128, 16, ACL_INT32,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matD;
    ASSERT_EQ(matD.initDense(handle.ptr(), 128, 128, 128, 16, ACL_INT32,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatmulDescrGuard md;
    ASSERT_EQ(md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE,
                      ACL_SPARSE_OP_NON_TRANSPOSE,
                      matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(),
                      ACL_SPARSE_COMPUTE_32I),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::DeviceBuffer dIn =
        sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(int32_t));
    sparse_test::DeviceBuffer dValid = sparse_test::DeviceBuffer::alloc(sizeof(int));
    auto ret = aclsparseLtSpMMAPruneCheck(
        handle.ptr(), sparse_test::matmulDescCptr(md), dIn.get(),
        static_cast<int*>(dValid.get()), stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// E11 / N8: stream == nullptr（默认流）-> SUCCESS 且 d_valid 与 golden 一致
//（属性验证，对齐 prune PE6 先例）。
// [协议说明] 与 CSV wrapper 相同，每次「新 Device 缓冲 + H2D 拷贝」后的首次
// 调用为 warm-up（结果不校验），校验对象为其后的 steady-state 调用 —— 多轮
// 计时基线惯例。fresh 缓冲首调语义（首调必须读到拷贝后数据）由
// PruneCheckDefectExposureTest 单独承载。
// 读写经 aclrtSynchronizeDevice 保证默认流完成后取回。合法 + 违规两种输入
// 各验证一次。
TEST_F(PruneCheckExceptionTest, NullStreamDefaultStream) {
    PruneCheckTestCtx ctx;  // FP16 128×128 N+ROW A-sparse
    const int32_t m = 128;
    const int32_t k = 128;
    const bool alongRow = true;  // N + ROW

    struct InputSpec {
        const char* pattern;
        const char* violPos;
        int expectValid;
    };
    const InputSpec specs[] = {
        {"VALID_SPARSE", "", 0},
        {"SINGLE_VIOLATION", "MIDDLE", 1},
    };
    for (const auto& spec : specs) {
        auto hInput = sparse_test::GenPruneCheckInput<uint16_t>(
            spec.pattern, m, k, alongRow, spec.violPos, -1.0f, 1.0f,
            SeedForCase(9100 + spec.expectValid));
        const int goldenValid = sparse_test::PruneCheckGolden<uint16_t>(hInput, m, k, false, true);
        ASSERT_EQ(goldenValid, spec.expectValid) << "golden self-check failed";

        auto hPhys = hInput;  // N+ROW：物理即逻辑 (m, k) 行主序
        sparse_test::DeviceBuffer dIn =
            sparse_test::DeviceBuffer::copyFrom(hPhys.data(), hPhys.size() * sizeof(uint16_t));
        sparse_test::DeviceBuffer dValid = sparse_test::DeviceBuffer::alloc(sizeof(int));

        // warm-up（fresh 缓冲首调，结果不校验；协议见用例头注释）。
        auto warmRet = aclsparseLtSpMMAPruneCheck(
            ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), dIn.get(),
            static_cast<int*>(dValid.get()), nullptr);
        ASSERT_EQ(warmRet, ACL_SPARSE_STATUS_SUCCESS);

        auto ret = aclsparseLtSpMMAPruneCheck(
            ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), dIn.get(),
            static_cast<int*>(dValid.get()), nullptr);
        EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS)
            << "null stream (default stream) must be accepted, pattern=" << spec.pattern;
        ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
        int hValid = -1;
        dValid.copyToHost(&hValid, sizeof(int));
        EXPECT_EQ(hValid, goldenValid)
            << "d_valid mismatch on default stream, pattern=" << spec.pattern;
    }
}

// E12: 空 shape（rows=0 / cols=0）—— [收紧动作 6] 描述符层 INVALID_VALUE 拦截
//（ValidateMatDescCommonParams rows/cols <= 0 拒绝，链路到不了 PruneCheck；
// PruneCheck 校验表 #9 的防御复检由开发方案 §7 保留）。
TEST_F(PruneCheckExceptionTest, EmptyShapeDescriptorRejected) {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::MatDescrGuard matRows0;
    EXPECT_EQ(matRows0.initStructured(handle.ptr(), 0, 128, 128, 16, ACL_FLOAT16,
                                      ACL_SPARSE_ORDER_ROW,
                                      ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    sparse_test::MatDescrGuard matCols0;
    EXPECT_EQ(matCols0.initStructured(handle.ptr(), 128, 0, 128, 16, ACL_FLOAT16,
                                      ACL_SPARSE_ORDER_ROW,
                                      ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E14（开发方案 D-2，黑盒可构造）: ld < 物理行长 -> INVALID_VALUE（校验表
// #10）。构造 N+COL 且 m ≤ ld < k：matA structured FP16 (16, 4096, ld=2048,
// COL)（整除与 IsLdValid 均满足，描述符可构造），physRowLen = k = 4096 >
// ld = 2048。prune 对该场景静默 clamp，本算子以错误代替 clamp（防越界读）。
TEST_F(PruneCheckExceptionTest, LdBelowPhysRowLenRejected) {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::MatDescrGuard matA;
    ASSERT_EQ(matA.initStructured(handle.ptr(), 16, 4096, 2048, 16, ACL_FLOAT16,
                                  ACL_SPARSE_ORDER_COL,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matB;
    ASSERT_EQ(matB.initDense(handle.ptr(), 4096, 128, 128, 16, ACL_FLOAT16,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matC;
    ASSERT_EQ(matC.initDense(handle.ptr(), 16, 128, 128, 16, ACL_FLOAT16,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matD;
    ASSERT_EQ(matD.initDense(handle.ptr(), 16, 128, 128, 16, ACL_FLOAT16,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatmulDescrGuard md;
    ASSERT_EQ(md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE,
                      ACL_SPARSE_OP_NON_TRANSPOSE,
                      matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(),
                      ACL_SPARSE_COMPUTE_32F),
              ACL_SPARSE_STATUS_SUCCESS);
    // d_in 覆盖 (rows=16) × ld=2048 的物理缓冲。
    sparse_test::DeviceBuffer dIn =
        sparse_test::DeviceBuffer::alloc(16 * 2048 * sizeof(uint16_t));
    sparse_test::DeviceBuffer dValid = sparse_test::DeviceBuffer::alloc(sizeof(int));
    auto ret = aclsparseLtSpMMAPruneCheck(
        handle.ptr(), sparse_test::matmulDescCptr(md), dIn.get(),
        static_cast<int*>(dValid.get()), stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// =============================================================================
// 首调可见性回归用例（编号独立于既有 E/N 体系）
// =============================================================================
//
// [校验目标] fresh Device 缓冲 H2D 拷贝后的首次 SpMMAPruneCheck 调用必须
//        读到拷贝后的数据并产出正确 d_valid —— 单次调用（无 warm-up）是
//        用户最常见用法，首调语义不得依赖任何前置调用。
//
// [暴露机制] 下方交替 pattern 循环：DeviceBuffer 分配器会复用前一 cycle
//        释放的同尺寸页，若首调读到旧页面内容而非刚拷入的数据，实测值将
//        与该 pattern 的 golden 相反（cycle>=1 的旧内容 = 上一 cycle 的
//        pattern），确定性失败 —— 无需依赖特定旧值即可捕获同类首调可见性
//        回归（首调读旧内容、同缓冲次调恢复正确的形态）。
class PruneCheckDefectExposureTest : public PruneCheckTestFixture {
};

TEST_F(PruneCheckDefectExposureTest, FirstCallOnFreshBufferMustSeeCopiedData) {
    PruneCheckTestCtx ctx;  // FP16 128×128 N+ROW A-sparse
    const int32_t m = 128;
    const int32_t k = 128;

    // 交替 pattern 循环：任一循环中「释放缓冲复用 + 首调读旧内容」都会使
    // 实测值与拷入数据的 golden 相反（cycle>=1 的旧内容 = 上一 cycle 的
    // pattern），从而确定性暴露首调可见性回归。
    const struct { const char* pattern; const char* violPos; int expectValid; } cycles[] = {
        {"SINGLE_VIOLATION", "MIDDLE", 1},
        {"VALID_SPARSE", "", 0},
        {"SINGLE_VIOLATION", "MIDDLE", 1},
        {"VALID_SPARSE", "", 0},
    };
    for (const auto& cyc : cycles) {
        auto hInput = sparse_test::GenPruneCheckInput<uint16_t>(
            cyc.pattern, m, k, true /*alongRow*/, cyc.violPos, -1.0f, 1.0f,
            SeedForCase(9500 + cyc.expectValid));
        const int goldenValid = sparse_test::PruneCheckGolden<uint16_t>(hInput, m, k, false, true);
        ASSERT_EQ(goldenValid, cyc.expectValid) << "golden self-check failed";

        // fresh DeviceBuffer（分配器复用上一 cycle 释放的同尺寸页）+ H2D 拷贝。
        sparse_test::DeviceBuffer dIn = sparse_test::DeviceBuffer::copyFrom(
            hInput.data(), hInput.size() * sizeof(uint16_t));
        sparse_test::DeviceBuffer dValid = sparse_test::DeviceBuffer::alloc(sizeof(int));

        // 单次调用（无 warm-up —— 用户最常见用法）。
        auto ret = aclsparseLtSpMMAPruneCheck(
            ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), dIn.get(),
            static_cast<int*>(dValid.get()), stream_);
        ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        int hValid = -1;
        dValid.copyToHost(&hValid, sizeof(int));
        EXPECT_EQ(hValid, goldenValid)
            << "DEFECT: first call on freshly-copied buffer returned stale-content "
            << "result (pattern=" << cyc.pattern << ", expected d_valid=" << goldenValid
            << ", got " << hValid << ")";
    }
}

// =============================================================================
// 多批量语义用例（E13 重做 + B 系列）：d_valid = OR(全部 batch 的判定)——
// 任一 batch 存在违规分组 → 1，全部 batch 合法 → 0；单 batch 判定链（位视图
// 判零、分组/方向派生、几何）完全不变。公共 API 路径：aclsparseLtMatDescSet
// Attribute 设 NUM_BATCHES（A/B/C/D 一致，一致性校验在 MatmulDescriptorInit）+
// BATCH_STRIDE（稀疏侧——A 侧链 matA / B 侧链 matB，两侧批量用例分别承载，
// 证明两侧装配链无缺陷）。分支承载映射见 prune_check_whitebox_coverage.md。
// =============================================================================

class PruneCheckBatchTest : public PruneCheckTestFixture {
};

namespace {

// 每 batch 独立 pattern/violPos（batch 级构造语义：合法/违规矩阵按 batch 组合）。
struct BatchPatternSpec {
    const char* pattern;
    const char* violPos;
};

// 装配批量物理缓冲：逐 batch 生成逻辑 (m,k) 输入（独立种子，确定性）→ 物理布局
// （T+ROW 转置，其余直传，SparseSideSpecOf 同源）→ 放置到 b×batchStride 基址。
// 缓冲总大小 = (numBatches−1)×batchStride + 单 batch 物理跨度。
template <typename T>
std::vector<T> AssembleBatchPhysical(const std::vector<BatchPatternSpec>& batches,
                                     const sparse_test::SparseSidePhysicalSpec& spec,
                                     int32_t m, int32_t k, bool alongRow,
                                     int64_t batchStride, uint32_t seedBase)
{
    const size_t perBatch = spec.physElems;
    std::vector<T> hPhys((static_cast<size_t>(batches.size()) - 1) *
                             static_cast<size_t>(batchStride) + perBatch);
    for (size_t b = 0; b < batches.size(); ++b) {
        auto logical = sparse_test::GenPruneCheckInput<T>(batches[b].pattern, m, k, alongRow,
                                             batches[b].violPos, -1.0f, 1.0f,
                                             seedBase + static_cast<uint32_t>(b));
        auto phys = sparse_test::PrepareSparseSideData<T>(logical, spec, m, k);
        EXPECT_EQ(phys.size(), perBatch) << "per-batch physical size mismatch";
        std::copy(phys.begin(), phys.end(),
                  hPhys.begin() + static_cast<size_t>(b) * static_cast<size_t>(batchStride));
    }
    return hPhys;
}

// 批量功能用例公共尾段：装配 + 批量 golden（batch 循环 OR）+ NPU 执行（A/B
// 稀疏侧分派：PruneCheckNpuBatch / PruneCheckNpuBatchBSparse）+ 三向一致断言
//（golden == expect_valid == NPU，EXACT 零容差）。batchStride 由调用方显式
// 给出（紧凑 = physElems；padding 场景传对齐倍数）。golden 与装配均按稀疏侧
// 自身 (m, k)/(op, order) 推导，侧别只决定 NPU 描述符链。
template <typename T>
void RunBatchFunctionalCase(aclrtStream stream, int32_t m, int32_t k,
                            aclsparseOperation_t op, aclsparseOrder_t order,
                            bool sparseIsA,
                            const std::vector<BatchPatternSpec>& batches,
                            int64_t batchStride, int expectValid, uint32_t seedBase)
{
    const bool trans = (op == ACL_SPARSE_OP_TRANSPOSE);
    const bool isRowOrder = (order == ACL_SPARSE_ORDER_ROW);
    const bool alongRow = (trans != isRowOrder);
    const auto spec = sparse_test::SparseSideSpecOf<T>(m, k, trans, order);
    const auto hPhys = AssembleBatchPhysical<T>(batches, spec, m, k, alongRow,
                                                batchStride, seedBase);
    const int goldenValid = sparse_test::PruneCheckGoldenBatch<T>(
        hPhys, m, k, trans, isRowOrder, static_cast<int32_t>(batches.size()),
        batchStride, spec.ld);
    // 三向一致第三方：设计期人工预期（每 batch pattern 语义的 OR）。
    EXPECT_EQ(goldenValid, expectValid) << "golden self-check (batch OR semantics)";

    const auto r = sparseIsA
        ? PruneCheckNpuBatch<T>(stream, hPhys, m, k, op, order,
                                static_cast<int32_t>(batches.size()), batchStride)
        : PruneCheckNpuBatchBSparse<T>(stream, hPhys, m, k, op, order,
                                       static_cast<int32_t>(batches.size()),
                                       batchStride);
    ASSERT_EQ(r.initRet, ACL_SPARSE_STATUS_SUCCESS) << "MatmulDescriptorInit failed";
    ASSERT_EQ(r.checkRet, ACL_SPARSE_STATUS_SUCCESS) << "SpMMAPruneCheck failed";
    EXPECT_EQ(r.hValid, goldenValid) << "d_valid EXACT mismatch (batch OR semantics)";
    EXPECT_EQ(r.hValid, expectValid) << "d_valid != expect_valid (three-way consistency)";
}

// 批量参数非法用例公共尾段：合法数据作 d_in 占位 + 非法 batchStride → 校验表
// #6/#11 拦截 INVALID_VALUE（任何设备访问前返回，d_in 内容/容量不参与判定）。
template <typename T>
void RunBatchStrideInvalidCase(aclrtStream stream, int32_t m, int32_t k,
                               bool sparseIsA, int32_t numBatches, int64_t batchStride)
{
    // 单 batch 合法数据作 d_in 占位（宿主侧校验先于任何设备访问）。
    auto hPhys = sparse_test::GenPruneCheckInput<T>("VALID_SPARSE", m, k, true, "", -1.0f, 1.0f,
                                       SeedForCase(9690));
    const auto r = sparseIsA
        ? PruneCheckNpuBatch<T>(stream, hPhys, m, k, ACL_SPARSE_OP_NON_TRANSPOSE,
                                ACL_SPARSE_ORDER_ROW, numBatches, batchStride)
        : PruneCheckNpuBatchBSparse<T>(stream, hPhys, m, k,
                                       ACL_SPARSE_OP_NON_TRANSPOSE,
                                       ACL_SPARSE_ORDER_ROW, numBatches, batchStride);
    // 描述符层不校验 batchStride（SetAttribute/MatmulDescriptorInit 均透传），
    // 拦截归宿唯一指向 PruneCheck 校验表 #6 / #11。
    ASSERT_EQ(r.initRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.checkRet, ACL_SPARSE_STATUS_INVALID_VALUE);
}

}  // namespace

// E13 重做（原 D-3「numBatches>1 → NOT_SUPPORTED」已由批量支持取代）：
// 公共 API 路径构造 numBatches=2 批量描述符（A 稀疏侧），全部 batch 合法 →
// SUCCESS / d_valid=0（memset 预清零 + 无违规 block 写出，跨 batch 保持 0）。
// 以下至 MultiBatchTotalTilesOverflowRejected 均为 A 侧批量用例（B 侧对偶
// 见 MultiBatchBSparse* 系列）。
TEST_F(PruneCheckBatchTest, MultiBatchTwoBatchesAllValid) {
    const std::vector<BatchPatternSpec> batches = {{"VALID_SPARSE", ""},
                                                   {"VALID_SPARSE", ""}};
    RunBatchFunctionalCase<uint16_t>(stream_, 128, 128, ACL_SPARSE_OP_NON_TRANSPOSE,
                                     ACL_SPARSE_ORDER_ROW, true, batches,
                                     128 * 128, 0, 9600);
}

// 仅 batch 0 违规 → d_valid=1（违规位于首个 batch 基址，batch 寻址正确性）。
TEST_F(PruneCheckBatchTest, MultiBatchFirstBatchViolation) {
    const std::vector<BatchPatternSpec> batches = {{"SINGLE_VIOLATION", "MIDDLE"},
                                                   {"VALID_SPARSE", ""}};
    RunBatchFunctionalCase<uint16_t>(stream_, 128, 128, ACL_SPARSE_OP_NON_TRANSPOSE,
                                     ACL_SPARSE_ORDER_ROW, true, batches,
                                     128 * 128, 1, 9601);
}

// 仅末 batch 违规（T+ROW 转置物理布局 × 3 batch）：d_valid=1——末 batch 寻址
// （batchIdx>0 基址偏移）与转置物理装配的组合。
TEST_F(PruneCheckBatchTest, MultiBatchLastBatchViolationTransposed) {
    const std::vector<BatchPatternSpec> batches = {{"VALID_SPARSE", ""},
                                                   {"VALID_SPARSE", ""},
                                                   {"SINGLE_VIOLATION", "LAST"}};
    RunBatchFunctionalCase<uint16_t>(stream_, 128, 128, ACL_SPARSE_OP_TRANSPOSE,
                                     ACL_SPARSE_ORDER_ROW, true, batches,
                                     128 * 128, 1, 9602);
}

// 全部 batch 违规（padding batchStride=20480，非紧凑 batch 布局）→ d_valid=1
//（多 batch 多写者写同值路径）。
TEST_F(PruneCheckBatchTest, MultiBatchAllBatchesViolating) {
    const std::vector<BatchPatternSpec> batches = {{"ALL_VIOLATING", ""},
                                                   {"ALL_VIOLATING", ""}};
    RunBatchFunctionalCase<uint16_t>(stream_, 128, 128, ACL_SPARSE_OP_NON_TRANSPOSE,
                                     ACL_SPARSE_ORDER_ROW, true, batches,
                                     20480, 1, 9603);
}

// FP32（u32 路径）× 仅第二 batch 违规 → d_valid=1（批量维度处理的 dtype 独立性）。
TEST_F(PruneCheckBatchTest, MultiBatchFp32SecondBatchViolation) {
    const std::vector<BatchPatternSpec> batches = {{"VALID_SPARSE", ""},
                                                   {"SINGLE_VIOLATION", "MIDDLE"}};
    RunBatchFunctionalCase<float>(stream_, 64, 64, ACL_SPARSE_OP_NON_TRANSPOSE,
                                  ACL_SPARSE_ORDER_ROW, true, batches,
                                  64 * 64, 1, 9604);
}

// numBatches=2 且 batchStride=0（未设置 / 显式置 0）→ 校验表 #6 INVALID_VALUE。
TEST_F(PruneCheckBatchTest, MultiBatchStrideZeroRejected) {
    RunBatchStrideInvalidCase<uint16_t>(stream_, 128, 128, true, 2, 0);
}

// numBatches=2 且 batchStride<0 → 校验表 #6 INVALID_VALUE（int64 负值在判
// ≤0 分支拦截，不做 uint64 转换后的巨值误判）。
TEST_F(PruneCheckBatchTest, MultiBatchStrideNegativeRejected) {
    RunBatchStrideInvalidCase<uint16_t>(stream_, 128, 128, true, 2, -16384);
}

// numBatches=2 且 batchStride=16392（=16384+8，非 32B/elem=16 元素倍数）→
// 校验表 #6 对齐子分支 INVALID_VALUE（batch 基址偏移破坏 32B 倍数性质）。
TEST_F(PruneCheckBatchTest, MultiBatchStrideUnalignedRejected) {
    RunBatchStrideInvalidCase<uint16_t>(stream_, 128, 128, true, 2, 16384 + 8);
}

// numBatches=2 且 batchStride=INT64_MAX 向下取 16 倍数（>0 且 16 倍数，#6 全
// 过）→ 全量跨度 (numBatches−1)×batchStride + (rows−1)×ld + cols 溢出 int64 →
// 校验表 #11 INVALID_VALUE（CheckedMulAdd 乘加回绕防护）。
TEST_F(PruneCheckBatchTest, MultiBatchSpanOverflowRejected) {
    RunBatchStrideInvalidCase<uint16_t>(stream_, 128, 128, true, 2, INT64_MAX / 16 * 16);
}

// numBatches 一致性由描述符层承载：NUM_BATCHES 仅设 A 侧（其余默认 1）→
// MatmulDescriptorInit 即被 INVALID_VALUE 拒绝（ValidateBatchConsistency），
// 链路到不了 PruneCheck。
TEST_F(PruneCheckBatchTest, MultiBatchNumBatchesConsistencyAtDescriptorLayer) {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::MatDescrGuard matA;
    ASSERT_EQ(matA.initStructured(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                                  ACL_SPARSE_ORDER_ROW,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matB;
    ASSERT_EQ(matB.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matC;
    ASSERT_EQ(matC.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matD;
    ASSERT_EQ(matD.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    const int32_t nb = 2;
    ASSERT_EQ(aclsparseLtMatDescSetAttribute(handle.ptr(), matA.ptr(),
                                             ACLSPARSELT_MAT_NUM_BATCHES, &nb, sizeof(nb)),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatmulDescrGuard md;
    EXPECT_EQ(md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE,
                      ACL_SPARSE_OP_NON_TRANSPOSE, matA.ptr(), matB.ptr(),
                      matC.ptr(), matD.ptr(), ACL_SPARSE_COMPUTE_32F),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// numBatches=16（totalTiles = 8×16 = 128 > 64 核）：grid-stride 跨 batch 多轮
// 领 tile（每 block 2 轮，tile 与 batch 交错），仅末 batch 违规 → d_valid=1
//（blockViolation 跨轮/跨 batch 累积 false→true）。
TEST_F(PruneCheckBatchTest, MultiBatchGridStrideAcrossBatches) {
    std::vector<BatchPatternSpec> batches(16, BatchPatternSpec{"VALID_SPARSE", ""});
    batches[15] = {"SINGLE_VIOLATION", "LAST"};
    RunBatchFunctionalCase<uint16_t>(stream_, 128, 128, ACL_SPARSE_OP_NON_TRANSPOSE,
                                     ACL_SPARSE_ORDER_ROW, true, batches,
                                     128 * 128, 1, 9605);
}

// tiling 阶段 totalTiles 溢出（原「数学不可达」防御分支，批量支持后公共 API
// 可构造）：INT8 极限几何 rows=cols=ld=2^31−32（32B/elem 整除、≤int32Max）+
// numBatches=INT32_MAX + batchStride=32 → #6/#9-#11 全过（totalElems ≈ 4.61e18
// < int64max），但 tilesPerMatrix(≈1.88e13) × numBatches(≈2.15e9) ≈ 4.0e22
// 溢出 int64 → compute_prune_check_tiling 的 CheckedMulAdd 拒绝 →
// INVALID_VALUE（设备访问前返回，d_in 为 32B 占位缓冲；matB/matC/matD 用小
// 占位 shape 规避 C/D 维度上限 2097120，INT8 ROW|ROW 布局约束要求 opB=T）。
TEST_F(PruneCheckBatchTest, MultiBatchTotalTilesOverflowRejected) {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::MatDescrGuard matA;  // structured INT8 极限几何
    ASSERT_EQ(matA.initStructured(handle.ptr(), 2147483616, 2147483616, 2147483616, 16,
                                  ACL_INT8, ACL_SPARSE_ORDER_ROW,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matB;  // dense 占位（INT8 ROW|ROW 布局需 opB=T）
    ASSERT_EQ(matB.initDense(handle.ptr(), 128, 128, 128, 16, ACL_INT8,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matC;
    ASSERT_EQ(matC.initDense(handle.ptr(), 128, 128, 128, 16, ACL_INT8,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matD;
    ASSERT_EQ(matD.initDense(handle.ptr(), 128, 128, 128, 16, ACL_INT8,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    const int32_t nb = 2147483647;  // INT32_MAX
    for (sparse_test::MatDescrGuard* mat : {&matA, &matB, &matC, &matD}) {
        ASSERT_EQ(aclsparseLtMatDescSetAttribute(handle.ptr(), mat->ptr(),
                                                 ACLSPARSELT_MAT_NUM_BATCHES, &nb, sizeof(nb)),
                  ACL_SPARSE_STATUS_SUCCESS);
    }
    const int64_t bs = 32;  // > 0 且 32B/1B 元素倍数
    ASSERT_EQ(aclsparseLtMatDescSetAttribute(handle.ptr(), matA.ptr(),
                                             ACLSPARSELT_MAT_BATCH_STRIDE, &bs, sizeof(bs)),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatmulDescrGuard md;
    ASSERT_EQ(md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE,
                      ACL_SPARSE_OP_TRANSPOSE, matA.ptr(), matB.ptr(),
                      matC.ptr(), matD.ptr(), ACL_SPARSE_COMPUTE_32I),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::DeviceBuffer dIn = sparse_test::DeviceBuffer::alloc(32);
    sparse_test::DeviceBuffer dValid = sparse_test::DeviceBuffer::alloc(sizeof(int));
    const auto ret = aclsparseLtSpMMAPruneCheck(
        handle.ptr(), sparse_test::matmulDescCptr(md), dIn.get(),
        static_cast<int*>(dValid.get()), stream_);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// =============================================================================
// B 侧批量用例（B 侧批量补格）：批量参数经稀疏侧描述符装配的 B 侧链——
// PruneCheckNpuBatchBSparse 的 BATCH_STRIDE 落 matB（host DeriveBatchParams 读
// md->matB 的 numBatches/batchStride），与 A 侧批量用例共同证明两侧装配链无
// 缺陷。golden（PruneCheckGoldenBatch，batch 循环 OR）与装配均按稀疏侧自身
// (m, k)/(op, order) 推导，侧别无关节点仅 NPU 描述符链。
// =============================================================================

// B 侧 numBatches=2 全 batch 合法（FP16 N+COL，ColScan 几何 × batch 维度为
// A 侧批量未覆盖组合）→ SUCCESS / d_valid=0（跨 batch 预清零保持）。
TEST_F(PruneCheckBatchTest, MultiBatchBSparseTwoBatchesAllValid) {
    const std::vector<BatchPatternSpec> batches = {{"VALID_SPARSE", ""},
                                                   {"VALID_SPARSE", ""}};
    RunBatchFunctionalCase<uint16_t>(stream_, 128, 128, ACL_SPARSE_OP_NON_TRANSPOSE,
                                     ACL_SPARSE_ORDER_COL, false, batches,
                                     128 * 128, 0, 9606);
}

// B 侧 T+ROW 转置物理 × 仅末 batch 违规（3 batch，违规分布落末 batch）→
// d_valid=1：B 侧链的 batchIdx>0 基址偏移 + 转置物理装配组合（A 侧同构用例
// 的 B 侧对偶）。
TEST_F(PruneCheckBatchTest, MultiBatchBSparseLastBatchViolationTransposed) {
    const std::vector<BatchPatternSpec> batches = {{"VALID_SPARSE", ""},
                                                   {"VALID_SPARSE", ""},
                                                   {"SINGLE_VIOLATION", "LAST"}};
    RunBatchFunctionalCase<uint16_t>(stream_, 128, 128, ACL_SPARSE_OP_TRANSPOSE,
                                     ACL_SPARSE_ORDER_ROW, false, batches,
                                     128 * 128, 1, 9607);
}

// B 侧 numBatches=2 且 batchStride=16392（=16384+8，非 32B/elem=16 元素倍数，
// batchStride 经 B 侧 matB 装配）→ 校验表 #6 对齐子分支 INVALID_VALUE（host
// 读 md->matB 后在宿主侧即拒，d_in 占位不参与判定）。
TEST_F(PruneCheckBatchTest, MultiBatchBSparseStrideUnalignedRejected) {
    RunBatchStrideInvalidCase<uint16_t>(stream_, 128, 128, false, 2, 16384 + 8);
}

// B 侧 FP32（u32 路径）× N+COL（ColScan）× 仅第二 batch 违规 → d_valid=1：
// u32 kernel × B 侧批量 + ColScan × batchIdx>0 寻址的组合（A 侧批量均为
// RowScan 几何）。
TEST_F(PruneCheckBatchTest, MultiBatchBSparseFp32ColScanSecondBatchViolation) {
    const std::vector<BatchPatternSpec> batches = {{"VALID_SPARSE", ""},
                                                   {"SINGLE_VIOLATION", "MIDDLE"}};
    RunBatchFunctionalCase<float>(stream_, 64, 64, ACL_SPARSE_OP_NON_TRANSPOSE,
                                  ACL_SPARSE_ORDER_COL, false, batches,
                                  64 * 64, 1, 9608);
}

// =============================================================================
// padding ld 用例（ld > 物理行长的正向路径此前零覆盖）：
// kernel CopyIn 的 srcStride=(ld−validCols)×elem（prune_check_kernel.cpp
// CopyIn 数据搬运参数）是 padding 能力的唯一消费点 —— ld > 物理行长时源行
// 步长跨过 padding 区，padding 数据不进入 UB / 不参与判定（README「ld 大于
// 物理行长时 padding 区数据不参与判定」声明的用例佐证）。既有全部用例 ld
// 恒等于物理行长（wrapper SparseSideSpecOf 自动推导），本组经 ldOverride
// 参数构造 padding 布局：设备缓冲按 ld 行距分配，host 侧数据准备在 padding
// 区写入非零垃圾（1.0 位模式）。分支承载映射见 prune_check_whitebox_
// coverage.md（K31 新增 + K11/K23 增量）。
// =============================================================================

class PruneCheckPaddingLdTest : public PruneCheckTestFixture {
};

namespace {

// padding ld 功能用例公共尾段：逻辑数据生成 → 物理装配（padding 非零垃圾）→
// golden 双口径交叉（逻辑 golden 与 PruneCheckGoldenBatch 的物理 ld 提取
// golden —— 后者按 ld 步长提取数据区 [0, 物理行长)，padding 区不进入分组
// 判定，两口径必须一致）→ NPU 执行（ldOverride 透传，wrapper 内部装配与本
// 用例 hPhys 同源）→ 三向一致断言。void 函数内可用 ASSERT/EXPECT（gtest
// 约束：致命断言要求返回 void）。
template <typename T>
void RunPaddingLdCase(aclrtStream stream, int32_t m, int32_t k,
                      aclsparseOperation_t op, aclsparseOrder_t order,
                      int64_t ld, const char* pattern, const char* violPos,
                      int expectValid, uint32_t seed)
{
    const bool trans = (op == ACL_SPARSE_OP_TRANSPOSE);
    const bool isRowOrder = (order == ACL_SPARSE_ORDER_ROW);
    const bool alongRow = (trans != isRowOrder);

    // 布局规格（ldOverride 合法性由 SparseSideSpecOf 内部校验承载，此处
    // 防御性复核 padding 布局前提成立）。
    const auto spec = sparse_test::SparseSideSpecOf<T>(m, k, trans, order, ld);
    ASSERT_EQ(spec.ld, ld) << "ldOverride not adopted";
    const int64_t physRowLen = spec.needTranspose ? static_cast<int64_t>(m)
                                                  : static_cast<int64_t>(k);
    ASSERT_GT(spec.ld, physRowLen) << "padding ld prerequisite violated";

    // 逻辑 (m, k) 数据 + 物理装配（padding 区非零垃圾）。
    auto hLogical = sparse_test::GenPruneCheckInput<T>(pattern, m, k, alongRow, violPos,
                                                       -1.0f, 1.0f, seed);
    ASSERT_EQ(hLogical.size(), static_cast<size_t>(m) * static_cast<size_t>(k));
    const auto hPhys = sparse_test::PrepareSparseSideData<T>(hLogical, spec, m, k);
    ASSERT_EQ(hPhys.size(), spec.physElems);

    // padding 区垃圾守卫：逐行 [physRowLen, ld) 全非零 —— 若 padding 参与判定，
    // 每行 padding 构成 (ld−physRowLen)/g 个全非零组（RowScan）或 padding 列的
    // 垂直全非零组（ColScan）→ 必然违规，用例判别性前提。
    const int64_t physRowCnt = spec.needTranspose ? static_cast<int64_t>(k)
                                                  : static_cast<int64_t>(m);
    for (int64_t r = 0; r < physRowCnt; ++r) {
        for (int64_t c = physRowLen; c < spec.ld; ++c) {
            ASSERT_TRUE(sparse_test::PruneCheckBitsTrait<T>::nonzero(
                hPhys[static_cast<size_t>(r) * static_cast<size_t>(spec.ld) +
                      static_cast<size_t>(c)]))
                << "padding region must be garbage-filled (row=" << r << " col=" << c << ")";
        }
    }

    // golden 双口径交叉：逻辑 golden（紧致数据区）== 物理 ld golden
    //（PruneCheckGoldenBatch 按 ld 步长提取数据区，padding 不进入分组判定）。
    const int goldenLogical = sparse_test::PruneCheckGolden<T>(hLogical, m, k, trans, isRowOrder);
    const int goldenPhys = sparse_test::PruneCheckGoldenBatch<T>(
        hPhys, m, k, trans, isRowOrder, 1, 0, spec.ld);
    ASSERT_EQ(goldenLogical, goldenPhys)
        << "golden ld semantics mismatch: padding must not participate in grouping";
    ASSERT_EQ(goldenPhys, expectValid) << "golden self-check (padding ld)";

    // NPU 执行（A 侧链，ldOverride 透传：描述符 ld + 设备缓冲行距 + padding 垃圾）。
    const auto npu = PruneCheckNpu<T>(stream, hLogical, m, k, op, order, ld);
    ASSERT_EQ(npu.checkRet, ACL_SPARSE_STATUS_SUCCESS) << "SpMMAPruneCheck failed (padding ld)";
    EXPECT_EQ(npu.hValid, goldenPhys)
        << "d_valid EXACT mismatch (padding ld): padding garbage must be ignored";
    EXPECT_EQ(npu.hValid, expectValid) << "d_valid != expect_valid (three-way consistency)";
}

}  // namespace

// P1（RowScan，padding 忽略正向佐证）: FP16 m×k=128×144，ld=160（每行
// padding 16 元素），N+ROW A 侧 —— 数据区 [0,144) 合法稀疏（VALID_SPARSE，
// 每组 nnz ≤ keep=2）、padding 区 [144,160) 全非零垃圾 → SUCCESS / d_valid=0
//（三向一致）。判别性：若 padding 参与判定（如 kernel 误以 ld 为有效列数），
// 每行 padding 16 元素构成 4 个全非零组 → d_valid=1，与本断言冲突即暴露。
TEST_F(PruneCheckPaddingLdTest, RowScanPaddingGarbageIgnored) {
    RunPaddingLdCase<uint16_t>(stream_, 128, 144, ACL_SPARSE_OP_NON_TRANSPOSE,
                               ACL_SPARSE_ORDER_ROW, 160, "VALID_SPARSE", "", 0,
                               SeedForCase(9700));
}

// P2（ColScan，padding 忽略正向佐证）: 同几何 N+COL A 侧 —— ColScan
// R=128 × C=144，物理行 stride=ld=160：数据区合法 + padding 全非零 →
// d_valid=0。判别性：若 padding 列参与判定，padding 列 [144,160) 的垂直组
//（同列 4 行全非零）→ d_valid=1。
TEST_F(PruneCheckPaddingLdTest, ColScanPaddingGarbageIgnored) {
    RunPaddingLdCase<uint16_t>(stream_, 128, 144, ACL_SPARSE_OP_NON_TRANSPOSE,
                               ACL_SPARSE_ORDER_COL, 160, "VALID_SPARSE", "", 0,
                               SeedForCase(9701));
}

// P3（RowScan 加固，两态载体）: 数据区违规（SINGLE_VIOLATION，目标组
// nnz=keep+1）+ padding 全非零 → d_valid=1 —— 判定只看数据区：数据区违规
// 检出不受 padding 干扰（与 P1 成对覆盖 padding 布局下 d_valid 两态）。
TEST_F(PruneCheckPaddingLdTest, RowScanDataViolationWithPaddingGarbage) {
    RunPaddingLdCase<uint16_t>(stream_, 128, 144, ACL_SPARSE_OP_NON_TRANSPOSE,
                               ACL_SPARSE_ORDER_ROW, 160, "SINGLE_VIOLATION", "MIDDLE", 1,
                               SeedForCase(9702));
}

// P4（ColScan 加固，两态载体）: 同几何 N+COL —— 数据区违规 + padding 全非零
// → d_valid=1（与 P2 成对覆盖两态）。
TEST_F(PruneCheckPaddingLdTest, ColScanDataViolationWithPaddingGarbage) {
    RunPaddingLdCase<uint16_t>(stream_, 128, 144, ACL_SPARSE_OP_NON_TRANSPOSE,
                               ACL_SPARSE_ORDER_COL, 160, "SINGLE_VIOLATION", "MIDDLE", 1,
                               SeedForCase(9703));
}

// =============================================================================
// 白盒补全用例（WB）：kernel 侧分支经 CSV level=WB 行（case 101-112）承载，
// 本段补 host 侧校验表防御分支（公共 API 不可构造，须伪造内部结构体 —— 与
// E4b/E8b 同机制）与 #12 错误注入路径。分支映射总表见
// prune_check_whitebox_coverage.md。
//
// 顺序说明：本组置于文件末尾 —— gtest 按注册顺序执行，#12 的两个错误注入
// 用例（PreExistingThreadError / DestroyedStream）涉及线程级错误状态置位与
// 清理，排在既有用例之后可将潜在污染面收敛到本组内部。
// =============================================================================

namespace {

// 在共享描述符链 ctx 上执行一次 SpMMAPruneCheck（仅返回码，不校验 d_valid）。
inline aclsparseStatus_t RunCheckOnCtx(PruneCheckTestCtx& ctx, aclrtStream stream) {
    return aclsparseLtSpMMAPruneCheck(
        ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), ctx.dIn.get(),
        static_cast<int*>(ctx.dValid.get()), stream);
}

// 临时伪造内部结构体字段、调用后恢复（防御分支白盒专用；E4b/E8b/E13 手工
// save/restore 的收敛封装；伪造值均为校验早期返回路径，不触达设备访问）。
template <typename M, typename V, typename F>
inline void WithForgedField(M* obj, V M::*field, V value, F&& fn) {
    const V saved = obj->*field;
    obj->*field = value;
    fn();
    obj->*field = saved;
}

}  // namespace

class PruneCheckWhiteboxTest : public PruneCheckTestFixture {
};

// WB-H1（校验表 #3 第二操作数）: md->matB == nullptr -> INVALID_VALUE。
// E4b 覆盖 || 条件的 matA 操作数；本用例补 matB 操作数（公共 API 不可构造：
// MatmulDescriptorInit 强制 matA~matD 非空）。
TEST_F(PruneCheckWhiteboxTest, ForgedNullMatB) {
    PruneCheckTestCtx ctx;
    WithForgedField(ToInternal(ctx.md.get()), &aclsparseLtMatmulDescriptor::matB,
                    static_cast<aclsparseLtMatDescriptor_t>(nullptr), [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
}

// WB-H1b（校验表 #6 numBatches 防御分支）: 伪造 numBatches=0/-1 →
// INVALID_VALUE。SetAttribute 拒绝 < 1（公共 API 不可构造），仅伪造可达；
// numBatches ≥ 1 的批量功能路径由 PruneCheckBatchTest 承载。
TEST_F(PruneCheckWhiteboxTest, ForgedZeroBatchesRejected) {
    PruneCheckTestCtx ctx;
    auto* matAInt = ToInternal(ctx.matA.get());
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::numBatches, int32_t{0}, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::numBatches, int32_t{-1}, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
}

// WB-H2（校验表 #9 零维防御分支）: rows / cols / ld == 0 -> INVALID_VALUE。
// E12 断言的是描述符层 init 拒绝（链路到不了 PruneCheck）；#9 的 PruneCheck
// 防御复检（防伪造描述符）仅伪造可达，本用例为其显式载体。
TEST_F(PruneCheckWhiteboxTest, ForgedZeroScanDimsRejected) {
    PruneCheckTestCtx ctx;  // FP16 128×128 N+ROW A-sparse：rows=cols=ld=128
    auto* matAInt = ToInternal(ctx.matA.get());
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::rows, int64_t{0}, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::cols, int64_t{0}, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::ld, int64_t{0}, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
}

// WB-H3（校验表 #9 整除防御分支）: rows/cols/ld 非 32B/elem 元素倍数。
// FP16 divisor=32B/2B=16；130=8×16+2 非整除，其余子条件（非零/≤int32Max）
// 均合法 → 单独命中整除子分支（描述符 init 已保证整除，仅伪造可达）。
TEST_F(PruneCheckWhiteboxTest, ForgedNonDivisibleScanDimsRejected) {
    PruneCheckTestCtx ctx;
    auto* matAInt = ToInternal(ctx.matA.get());
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::rows, int64_t{130}, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::cols, int64_t{130}, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::ld, int64_t{130}, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
}

// WB-H4（校验表 #9 int32 上限防御分支）: rows / cols > int32Max。
// md_m/md_k 经 static_cast<int32_t> 取维度（aclsparselt_internal.h）：伪造
// 2^31 使 cast 回绕为负 int32（x86-64 gcc 补码回绕），再转 uint64 得超大值
// → td.rows/td.cols > int32Max 子分支命中。
TEST_F(PruneCheckWhiteboxTest, ForgedInt32OverflowDimsRejected) {
    PruneCheckTestCtx ctx;
    auto* matAInt = ToInternal(ctx.matA.get());
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::rows, int64_t{1} << 31, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::cols, int64_t{1} << 31, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
}

// WB-H4b（校验表 #9 ld 上界防御分支）: #9 对 rows/cols/ld 复检 INT32_MAX
// 上界 —— ld 上界拦截巨 ld + 小 rows 场景（spanBytes 落在 int64 内、
// 绕过 #11 跨度拦截直达 kernel，MTE2 以 srcStride≈2^32 字节越界读）。
// 本用例将上界拦截行为固化为断言（WB-H2 同型伪造；构造性实测预演
// 5/5 PASS）：
//   P1 上界拦截场景: rows=16 + ld=2^31+16（16 对齐、≥cols、非零 → 整除与 #10
//      子分支全过；spanBytes≈6.4e10 远小于 int64max → #11 亦过）→ 仅能由
//      #9 ld 上界子分支拒绝 → INVALID_VALUE（任何设备访问前返回；若无
//      该子分支，此输入会启动 kernel 并越界读，构造即设备故障）；
//   P2 负向硬化: ld=-16（int64 负值经 uint64 转换为巨值）→ 同子分支拒绝；
//   P3 上界含边数据点: ld=INT32_MAX（≤ 上界、非 16 对齐）→ 整除子分支拒绝
//      （「≤INT32_MAX 不被上界子分支误拦」由 WB-H5 承载：FP32 ld=2147483640
//      过 #9、由 #11 拒绝）。
TEST_F(PruneCheckWhiteboxTest, ForgedLdUpperBoundRejected) {
    PruneCheckTestCtx ctx;  // FP16 128×128 N+ROW A-sparse：divisor=16，cols=128
    auto* matAInt = ToInternal(ctx.matA.get());
    // P1: 上界拦截场景（小 rows 使 spanBytes 落在 int64 内、绕过 #11 —— 排除
    // INVALID_VALUE 来自 #11 的可能，命中归宿唯一指向 #9 ld 上界子分支）。
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::rows, int64_t{16}, [&] {
        WithForgedField(matAInt, &aclsparseLtMatDescriptor::ld,
                        (int64_t{1} << 31) + 16, [&] {
            EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
        });
    });
    // P2: 负向硬化（负 ld → uint64 巨值，上界子分支拒绝）。
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::ld, int64_t{-16}, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
    // P3: 上界含边数据点（INT32_MAX 本身非 16 对齐 → 整除子分支拒绝）。
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::ld, int64_t{2147483647}, [&] {
        EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_INVALID_VALUE);
    });
}

// WB-H5（校验表 #11 扫描跨度溢出）: FP32 rows=ld=2^31-8 → spanBytes 超 int64。
// 专用 FP32 链：FP16（dataBytes=2）时 rows·ld·2 上限恰为 2^63，与 int64max
// 临界重合不可靠；FP32（dataBytes=4）余量充分：spanElems=(2^31-9)(2^31-8)+128
// ≈ 4.61e18 < int64max（#11 首段过），spanBytes≈1.84e19 > int64max
// → CheckedMulAdd(spanElems, 4, 0) 拒绝 → #11。rows/ld 均为 8 的倍数、
// ≤int32Max、ld≥cols（#9/#10 全过），校验在任何设备访问前返回。
TEST_F(PruneCheckWhiteboxTest, ForgedSpanOverflowRejected) {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::MatDescrGuard matA;
    ASSERT_EQ(matA.initStructured(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                                  ACL_SPARSE_ORDER_ROW,
                                  ACL_SPARSE_LT_SPARSITY_50_PERCENT),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matB;
    ASSERT_EQ(matB.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matC;
    ASSERT_EQ(matC.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatDescrGuard matD;
    ASSERT_EQ(matD.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                             ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::MatmulDescrGuard md;
    ASSERT_EQ(md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE,
                      ACL_SPARSE_OP_NON_TRANSPOSE, matA.ptr(), matB.ptr(),
                      matC.ptr(), matD.ptr(), ACL_SPARSE_COMPUTE_32F),
              ACL_SPARSE_STATUS_SUCCESS);
    sparse_test::DeviceBuffer dIn =
        sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(float));
    sparse_test::DeviceBuffer dValid = sparse_test::DeviceBuffer::alloc(sizeof(int));
    auto* matAInt = ToInternal(matA.get());
    WithForgedField(matAInt, &aclsparseLtMatDescriptor::rows, int64_t{2147483640}, [&] {
        WithForgedField(matAInt, &aclsparseLtMatDescriptor::ld, int64_t{2147483640}, [&] {
            const auto ret = aclsparseLtSpMMAPruneCheck(
                handle.ptr(), sparse_test::matmulDescCptr(md), dIn.get(),
                static_cast<int*>(dValid.get()), stream_);
            EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
        });
    });
}

// WB-H6（校验表有序性）: 组合双违规，断言顺序即优先级（跨返回码可观测）。
// 同返回码的相邻项（如 #8 vs #9 均为 INVALID_VALUE）不可观测，不作断言。
TEST_F(PruneCheckWhiteboxTest, ValidationOrderPriority) {
    PruneCheckTestCtx ctx;
    // #1 > #3：handle null 与伪造 matB=nullptr 并存 → HANDLE_IS_NULLPTR 先决。
    WithForgedField(ToInternal(ctx.md.get()), &aclsparseLtMatmulDescriptor::matB,
                    static_cast<aclsparseLtMatDescriptor_t>(nullptr), [&] {
        const auto ret = aclsparseLtSpMMAPruneCheck(
            nullptr, sparse_test::matmulDescCptr(ctx.md), ctx.dIn.get(),
            static_cast<int*>(ctx.dValid.get()), stream_);
        EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    });
    // #4 > #7：双稀疏伪造（D-1）与 d_in=nullptr 并存 → NOT_SUPPORTED 先于缓冲校验。
    {
        sparse_test::MatDescrGuard matB2;
        ASSERT_EQ(matB2.initStructured(ctx.handle.ptr(), 128, 128, 128, 16, ACL_FLOAT16,
                                       ACL_SPARSE_ORDER_ROW,
                                       ACL_SPARSE_LT_SPARSITY_50_PERCENT),
                  ACL_SPARSE_STATUS_SUCCESS);
        WithForgedField(ToInternal(ctx.md.get()), &aclsparseLtMatmulDescriptor::matB,
                        matB2.get(), [&] {
            const auto ret = aclsparseLtSpMMAPruneCheck(
                ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), nullptr,
                static_cast<int*>(ctx.dValid.get()), stream_);
            EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
        });
    }
    // #5 > #6：dtype 白名单外（INT32，NOT_SUPPORTED）与批量参数非法
    //（numBatches=2 且默认 batchStride=0 ≤ 0，#6 → INVALID_VALUE）并存 →
    // NOT_SUPPORTED（dtype 派生先于批量校验）。
    WithForgedField(ToInternal(ctx.matA.get()), &aclsparseLtMatDescriptor::valueType,
                    ACL_INT32, [&] {
        WithForgedField(ToInternal(ctx.matA.get()), &aclsparseLtMatDescriptor::numBatches,
                        int32_t{2}, [&] {
            const auto ret = aclsparseLtSpMMAPruneCheck(
                ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), ctx.dIn.get(),
                static_cast<int*>(ctx.dValid.get()), stream_);
            EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
        });
    });
    // #5 > #8：dtype 白名单外（INT32，E10 已证可伪造）与 d_in 非 32B 对齐并存
    // → NOT_SUPPORTED（dtype 派生）先于对齐校验。
    {
        sparse_test::DeviceBuffer buf =
            sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(uint16_t) + 32);
        void* misaligned = sparse_test::MisalignedOffsetPtr(buf);
        WithForgedField(ToInternal(ctx.matA.get()), &aclsparseLtMatDescriptor::valueType,
                        ACL_INT32, [&] {
            const auto ret = aclsparseLtSpMMAPruneCheck(
                ctx.handle.ptr(), sparse_test::matmulDescCptr(ctx.md), misaligned,
                static_cast<int*>(ctx.dValid.get()), stream_);
            EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
        });
    }
}

// WB-H7（校验表 #12 前置线程错误分支）: 已有线程级运行时错误 → 不下发任务，
// 直接 EXECUTION_FAILED。触发器：已销毁 stream 上的 memset —— 运行时句柄
// 查表失败（同步拒绝、无设备下发）并置线程级 last error（构造性实测：
// 参数校验类失败如 dst=null 返回 100000 但 peek=0 不置位；句柄查表类
// 失败返回 107017 且 peek 同步置位）。断言后以 aclrtGetLastError（读取
// 并清除，行为经构造性实测验证）
// 清理，并证明清理后同链路恢复正常 —— 防止错误状态污染后续用例。d_valid
// 值不校验（fresh 缓冲首调语义由 PruneCheckDefectExposureTest 单独承载）。
TEST_F(PruneCheckWhiteboxTest, PreExistingThreadErrorNotEnqueued) {
    PruneCheckTestCtx ctx;
    aclrtStream stale = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stale), ACL_SUCCESS);
    ASSERT_EQ(aclrtDestroyStream(stale), ACL_SUCCESS);
    const aclError trig =
        aclrtMemsetAsync(ctx.dValid.get(), sizeof(int), 0, sizeof(int), stale);
    ASSERT_NE(trig, ACL_SUCCESS) << "trigger memset on destroyed stream unexpectedly succeeded";
    ASSERT_NE(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL), ACL_SUCCESS)
        << "trigger failed but did not set thread-level last error";
    // 合法 stream 上调用 → #12 前置探测拒绝，EXECUTION_FAILED（不下发任务）。
    EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    // 清理线程错误（GetLastError 读取并清除）+ 无残留证明。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    EXPECT_EQ(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL), ACL_SUCCESS)
        << "thread error not cleared — would poison subsequent launches";
    // 恢复证明：同链路正常调用返回 SUCCESS。
    EXPECT_EQ(RunCheckOnCtx(ctx, stream_), ACL_SPARSE_STATUS_SUCCESS);
}

// WB-H8（校验表 #12 memset 失败分支）: 已销毁 stream 句柄 → aclrtMemsetAsync
// 句柄查表失败（同步拒绝，无设备下发）→ EXECUTION_FAILED。置于整个套件最后
// 执行：错误注入的潜在影响收敛到本用例内部；调用后清理残余线程错误。
TEST_F(PruneCheckWhiteboxTest, DestroyedStreamMemsetRejected) {
    PruneCheckTestCtx ctx;
    aclrtStream stale = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stale), ACL_SUCCESS);
    ASSERT_EQ(aclrtDestroyStream(stale), ACL_SUCCESS);
    EXPECT_EQ(RunCheckOnCtx(ctx, stale), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);  // 清理残余，防污染
}
