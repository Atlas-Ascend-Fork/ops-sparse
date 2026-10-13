/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and is subject to the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You can obtain a copy of the License at https://www.hiascend.com/
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*
 * aclsparseDenseToSparse arch22 (A2/A3) 全覆盖 gtest UT（NPU 真机）。
 *
 * 覆盖面：
 *   - 基础门面：5 dtype x {CSR/CSC/COO} x base{0,1} x order{ROW,COL}
 *     （含 ld padding、全零矩阵、输入只读校验）；
 *   - 值分布（任务书 §3.5）：普通values按 70% uniform[-1,1] / 20% N(0,1) /
 *     10% 特殊值池（±0/±Inf/NaN/边界/最大有限/最小正规/最小次正规）生成，
 *     complex64 实/虚部独立；
 *   - guard 区检查（任务书 §3.5）：全部 device 缓冲 512B 金丝雀边界，
 *     三阶段执行后逐字节校验，捕获越界写；
 *   - BELL：门面矩阵 + 32/64 块尺寸 + 非整除尾块（官方 reference 语义：
 *     内容发现宽度、正零 padding、-1 空槽）；
 *   - COO+COL 密集行 carve 直达路径（负载不均治理）：fp32/fp16 多密度
 *     多 base、混合密度（前半全满 + 后半稀疏）、复杂交错调度；
 *   - 大形状 stride/面板路径（512x1536 全格式全 dtype）；
 *   - 异常路径（任务书 §3.5 清单）。
 *
 * 判定口径：CPU golden 独立实现（bit 级非零判定 + 固定扫描序装配），
 * 全 dtype bit-wise exact（rtol=0/atol=0，NaN 含 payload，±0 不入结构）。
 *
 * 机器证据：case_id 级 JSON（环境变量 D2S_UT_JSON 指定路径）+
 * gtest 原生 JSON（--gtest_output=json:...）。
 *
 */

#include <gtest/gtest.h>
#include <acl/acl.h>
#include "cann_ops_sparse.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "csv_loader.h"
#include "densetosparse_npu_wrapper.h"

// ===================== 全局状态 =====================


struct CaseRecord {
    std::string caseId;
    std::string gtest;
    std::string status;   // PASS / FAIL
    std::string check;
    std::string dtype;
    std::string shape;
    int64_t nnz = 0;
    int base = 0;
    uint32_t seed = 0;
    std::string detail;
};

static std::vector<CaseRecord> g_records;

static std::string CurrentGtestName()
{
    const ::testing::TestInfo *info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    if (info == nullptr) {
        return "unknown";
    }
    return std::string(info->test_suite_name()) + "." + info->name();
}


static std::string ShapeTag(const CaseCfg &cfg)
{
    return std::to_string(cfg.rows) + "x" + std::to_string(cfg.cols) +
           " ldPad=" + std::to_string(cfg.ldPad) +
           (cfg.order == ACL_SPARSE_ORDER_ROW ? " ROW" : " COL");
}

static std::string Hex4(unsigned v)
{
    static const char *digits = "0123456789abcdef";
    return std::string("\\u") + digits[(v >> 12) & 0xF] +
           digits[(v >> 8) & 0xF] + digits[(v >> 4) & 0xF] + digits[v & 0xF];
}

static void RecordCase(const CaseRecord &r)
{
    g_records.push_back(r);
    std::printf("  [UT-CASE] %-34s %-5s nnz=%-8lld %s\n",
        r.caseId.c_str(), r.status.c_str(), (long long)r.nnz,
        r.detail.c_str());
    std::fflush(stdout);
}

void ExecCase(const std::string &caseId, const CaseCfg &cfg)
{
    const RunOutcome out = RunConvertCase(cfg);
    CaseRecord r;
    r.caseId = caseId;
    r.gtest = CurrentGtestName();
    r.status = out.ok ? "PASS" : "FAIL";
    r.check = "bit-wise exact vs CPU golden";
    r.dtype = DtName(cfg.dtype);
    r.shape = ShapeTag(cfg);
    r.nnz = out.nnz;
    r.base = (int)cfg.base;
    r.seed = cfg.seed;
    r.detail = out.why.empty()
                   ? std::string(FmtName(cfg.format))
                   : std::string(FmtName(cfg.format)) + ": " + out.why;
    RecordCase(r);
    EXPECT_TRUE(out.ok) << caseId << ": " << out.why;
}

static std::string JsonEscape(const std::string &in)
{
    std::string out;
    for (char c : in) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out += Hex4(c);
            } else {
                out += c;
            }
        }
    }
    return out;
}

static void DumpCaseJson()
{
    // Case records are dumped only when D2S_UT_JSON explicitly names an
    // output file; nothing is written to the working directory by default.
    const char *path = std::getenv("D2S_UT_JSON");
    if (path == nullptr || *path == '\0') {
        return;
    }
    std::ofstream f(path);
    if (!f) {
        return;
    }
    f << "{\"records\":[";
    for (size_t i = 0; i < g_records.size(); ++i) {
        const CaseRecord &r = g_records[i];
        f << (i ? "," : "")
          << "{\"case_id\":\"" << JsonEscape(r.caseId) << "\""
          << ",\"gtest\":\"" << JsonEscape(r.gtest) << "\""
          << ",\"status\":\"" << r.status << "\""
          << ",\"check\":\"" << JsonEscape(r.check) << "\""
          << ",\"dtype\":\"" << JsonEscape(r.dtype) << "\""
          << ",\"shape\":\"" << JsonEscape(r.shape) << "\""
          << ",\"nnz\":" << r.nnz
          << ",\"base\":" << r.base
          << ",\"seed\":" << r.seed
          << ",\"detail\":\"" << JsonEscape(r.detail) << "\"}";
    }
    f << "],\"total\":" << g_records.size() << "}";
}

class DenseToSparseUtEnv : public ::testing::Environment {
public:
    void SetUp() override
    {
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
        ASSERT_EQ(aclrtCreateStream(&dts_stream), ACL_SUCCESS);
        ASSERT_EQ(aclsparseCreate(&dts_handle), ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclsparseSetStream(dts_handle, dts_stream),
                  ACL_SPARSE_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        DumpCaseJson();
        if (dts_handle) {
            aclsparseDestroy(dts_handle);
        }
        if (dts_stream) {
            aclrtDestroyStream(dts_stream);
        }
        aclrtResetDevice(0);
        aclFinalize();
    }
};

// ===================== 基础门面 =====================

TEST(Matrix, AllZeroEmptyStructure)
{
    CaseCfg cfg{};
    cfg.format = ACL_SPARSE_FORMAT_CSR;
    cfg.dtype = ACL_FLOAT;
    cfg.base = ACL_SPARSE_INDEX_BASE_ONE;
    cfg.order = ACL_SPARSE_ORDER_ROW;
    cfg.rows = 5;
    cfg.cols = 9;
    cfg.seed = 7;
    cfg.zeroFill = true;
    ExecCase("GATE_ALL_ZERO_CSR_FP32_B1_ROW", cfg);
}

// ===================== CSV 参数矩阵 =====================
// 参数用例（case_id, CaseCfg）来自 arch22/densetosparse_test.csv（cmake
// 自动拷贝到构建目录）；case_id 与 seed 逐条对应原硬编码矩阵。白盒/
// 异常/guard 用例保留代码形态（csv 列无法表达其推导与金丝雀语义）。

using NamedCase = std::pair<std::string, CaseCfg>;

aclsparseFormat_t ParseCsvFormat(const std::string &name)
{
    if (name == "csr") {
        return ACL_SPARSE_FORMAT_CSR;
    }
    if (name == "csc") {
        return ACL_SPARSE_FORMAT_CSC;
    }
    if (name == "coo") {
        return ACL_SPARSE_FORMAT_COO;
    }
    return ACL_SPARSE_FORMAT_BLOCKED_ELL;
}

aclDataType ParseCsvDtype(const std::string &name)
{
    if (name == "int8") {
        return ACL_INT8;
    }
    if (name == "fp16") {
        return ACL_FLOAT16;
    }
    if (name == "bf16") {
        return ACL_BF16;
    }
    if (name == "c64") {
        return ACL_COMPLEX64;
    }
    return ACL_FLOAT;
}

// 构建目录 CSV 优先（cmake 已拷贝到二进制旁），源码目录兜底（仓根独立构建）。
std::string FindArch22CaseCsv()
{
    const char *kName = "densetosparse_test.csv";
    if (std::ifstream(kName).good()) {
        return kName;
    }
    const std::string file = __FILE__;
    const size_t slash = file.find_last_of('/');
    return slash == std::string::npos
               ? std::string(kName)
               : file.substr(0, slash) + "/" + kName;
}

std::vector<NamedCase> LoadArch22Cases()
{
    std::vector<NamedCase> cases;
    for (const auto &row : sparse_test::ReadMap(FindArch22CaseCsv())) {
        CaseCfg cfg{};
        cfg.format = ParseCsvFormat(sparse_test::parseString(row, "format"));
        cfg.dtype = ParseCsvDtype(sparse_test::parseString(row, "dtype"));
        cfg.base = static_cast<aclsparseIndexBase_t>(
            sparse_test::parseInt(row, "base"));
        cfg.order = sparse_test::parseString(row, "order") == "col"
                        ? ACL_SPARSE_ORDER_COL
                        : ACL_SPARSE_ORDER_ROW;
        cfg.rows = sparse_test::parseInt(row, "rows");
        cfg.cols = sparse_test::parseInt(row, "cols");
        cfg.ldPad = sparse_test::parseInt(row, "ld_pad");
        cfg.bellB = sparse_test::parseInt(row, "bell_b");
        cfg.ellCols = sparse_test::parseInt(row, "ell_cols");
        cfg.nnzPerRow = sparse_test::parseInt(row, "nnz_per_row");
        cfg.seed = static_cast<unsigned>(sparse_test::parseInt(row, "seed"));
        cases.emplace_back(sparse_test::parseString(row, "case_name"), cfg);
    }
    return cases;
}

class DenseToSparseCsvTest : public ::testing::TestWithParam<NamedCase> {};

TEST_P(DenseToSparseCsvTest, ParamMatrixCase)
{
    ExecCase(GetParam().first, GetParam().second);
}

INSTANTIATE_TEST_SUITE_P(
    CsvMatrix, DenseToSparseCsvTest, testing::ValuesIn(LoadArch22Cases()),
    [](const testing::TestParamInfo<NamedCase> &info) {
        return info.param.first;
    });

// ===================== COO+COL 密集行 carve =====================// ===================== COO+COL 密集行 carve =====================// ===================== COO+COL 密集行 carve =====================

TEST(DenseCarve, CooColDenseMixedHeads)
{
    // 前半全满（carve）+ 后半稀疏（面板）：混合调度 + 交错。
    for (aclDataType dt : {ACL_FLOAT, ACL_FLOAT16}) {
        CaseCfg cfg{};
        cfg.format = ACL_SPARSE_FORMAT_COO;
        cfg.dtype = dt;
        cfg.base = ACL_SPARSE_INDEX_BASE_ONE;
        cfg.order = ACL_SPARSE_ORDER_COL;
        cfg.rows = 128;
        cfg.cols = 512;
        cfg.ldPad = 0;
        cfg.denseHeadRows = 64;
        cfg.seed = 900u;
        ExecCase("CARVE_MIXED_HEADS_" + std::string(DtName(dt)), cfg);
    }
}

TEST(DenseCarve, CooColPowerLawLike)
{
    // 幂律风格：全满头行 + 递减尾行，覆盖跨窗口 run / wrap / 对齐补偿。
    CaseCfg cfg{};
    cfg.format = ACL_SPARSE_FORMAT_COO;
    cfg.dtype = ACL_FLOAT;
    cfg.base = ACL_SPARSE_INDEX_BASE_ONE;
    cfg.order = ACL_SPARSE_ORDER_COL;
    cfg.rows = 896;
    cfg.cols = 2688;
    cfg.ldPad = 0;
    cfg.nnzPerRow = -1;
    cfg.denseHeadRows = 27;  // 全满头行；其余行用 coin 稀疏
    cfg.seed = 20262318u;
    ExecCase("CARVE_POWER_LAW_896X2688", cfg);
}

TEST(DenseCarve, CooColWrappedRuns)
{
    // 连续 run 起点在窗口中部（runStart != 0）+ 跨 512 槽窗口边界。
    CaseCfg cfg{};
    cfg.format = ACL_SPARSE_FORMAT_COO;
    cfg.dtype = ACL_FLOAT;
    cfg.base = ACL_SPARSE_INDEX_BASE_ONE;
    cfg.order = ACL_SPARSE_ORDER_COL;
    cfg.rows = 64;
    cfg.cols = 2688;
    cfg.ldPad = 0;
    cfg.nnzPerRow = 2000;
    cfg.seed = 3u;
    ExecCase("CARVE_WRAPPED_RUNS_64X2688_N2000", cfg);
}

// ===================== 大形状 stride/面板 =====================

// 面板组前缀 span 超出 pfxSpan 容量（chunks=48 -> spanElems=1602>1536）：
// 必须回退 per-unit 路径而不是用被截断的 span（越界读写）。
TEST(Strided, PanelSpanOverflowFallsBackPerUnit)
{
    CaseCfg cfg{};
    cfg.format = ACL_SPARSE_FORMAT_CSC;
    cfg.dtype = ACL_FLOAT;
    cfg.base = ACL_SPARSE_INDEX_BASE_ZERO;
    cfg.order = ACL_SPARSE_ORDER_ROW;
    cfg.rows = 196608;
    cfg.cols = 33;
    cfg.ldPad = 0;
    cfg.nnzPerRow = 3;
    cfg.seed = 20260921u;
    ExecCase("PANEL_SPAN_OVERFLOW_CSC_FP32_196608X33", cfg);
}

// ===================== 异常路径 =====================

namespace {
struct ErrItem {
    const char *name;
    aclsparseStatus_t got;
    aclsparseStatus_t want;
};

void ExpectErr(std::vector<ErrItem> &err, const char *name,
    aclsparseStatus_t got, aclsparseStatus_t want)
{
    err.push_back({name, got, want});
}
} // namespace

// Index-type / index-base rejections.
static void CheckIndexBaseRejections(aclsparseHandle_t h,
    aclsparseConstDnMatDescr_t A, size_t *bs, void *dOff,
    std::vector<ErrItem> &err)
{
    {
        aclsparseSpMatDescr_t B3 = nullptr;
        aclsparseCreateCsr(&B3, 4, 5, 0, dOff, nullptr, nullptr,
                           ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_64I,
                           ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        ExpectErr(err, "64I index",
               aclsparseDenseToSparseGetBufferSize(
                   h, A, B3, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                   bs),
               ACL_SPARSE_STATUS_NOT_SUPPORTED);
        aclsparseDestroySpMat(B3);
    }
    {
        aclsparseSpMatDescr_t B4 = nullptr;
        aclsparseCreateCsr(&B4, 4, 5, 0, dOff, nullptr, nullptr,
                           ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                           (aclsparseIndexBase_t)5, ACL_FLOAT);
        ExpectErr(err, "invalid base(5)",
               aclsparseDenseToSparseGetBufferSize(
                   h, A, B4, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                   bs),
               ACL_SPARSE_STATUS_INVALID_VALUE);
        aclsparseDestroySpMat(B4);
    }
}

// BELL-geometry rejections.
static void CheckBellGeometryRejections(aclsparseHandle_t h,
    aclsparseConstDnMatDescr_t A, size_t *bs, void *dOff, void *dV,
    std::vector<ErrItem> &err)
{
    {
        aclsparseSpMatDescr_t B5 = nullptr;
        aclsparseCreateBlockedEll(&B5, 4, 5, 0, 4, dOff, dV,
                                  ACL_SPARSE_INDEX_32I,
                                  ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        ExpectErr(err, "BELL blockSize=0",
               aclsparseDenseToSparseGetBufferSize(
                   h, A, B5, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                   bs),
               ACL_SPARSE_STATUS_INVALID_VALUE);
        aclsparseDestroySpMat(B5);
    }
    {
        aclsparseSpMatDescr_t B6 = nullptr;
        aclsparseCreateBlockedEll(&B6, 4, 5, 2, 8, dOff, dV,
                                  ACL_SPARSE_INDEX_32I,
                                  ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        ExpectErr(err, "BELL ellCols>cols",
               aclsparseDenseToSparseGetBufferSize(
                   h, A, B6, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                   bs),
               ACL_SPARSE_STATUS_INVALID_VALUE);
        aclsparseDestroySpMat(B6);
    }
    {
        // b=1 且 65536x65536 -> 任务数 2^32，超出 kernel u32 划分。
        aclsparseDnMatDescr_t Abig = nullptr;
        aclsparseSpMatDescr_t Bbig = nullptr;
        aclsparseCreateDnMat(&Abig, 65536, 65536, 65536, dV, ACL_FLOAT,
                             ACL_SPARSE_ORDER_ROW);
        aclsparseCreateBlockedEll(&Bbig, 65536, 65536, 1, 65536, dOff, dV,
                                  ACL_SPARSE_INDEX_32I,
                                  ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        ExpectErr(err, "BELL task grid > u32",
               aclsparseDenseToSparseGetBufferSize(
                   h, Abig, Bbig, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                   bs),
               ACL_SPARSE_STATUS_NOT_SUPPORTED);
        if (Abig) aclsparseDestroyDnMat(Abig);
        if (Bbig) aclsparseDestroySpMat(Bbig);
    }
}

// Shape / layout / dtype-consistency rejections.
static void CheckShapeDtypeRejections(aclsparseHandle_t h,
    aclsparseConstDnMatDescr_t A, aclsparseSpMatDescr_t B, size_t *bs,
    void *dOff, void *dV, std::vector<ErrItem> &err)
{
    {
        aclsparseDnMatDescr_t A3 = nullptr;
        aclsparseCreateDnMat(&A3, 4, 5, 3, dV, ACL_FLOAT,
                             ACL_SPARSE_ORDER_ROW);
        ExpectErr(err, "ROW ld<cols",
               aclsparseDenseToSparseGetBufferSize(
                   h, A3, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                   bs),
               ACL_SPARSE_STATUS_INVALID_VALUE);
        aclsparseDestroyDnMat(A3);
    }
    {
        aclsparseDnMatDescr_t A2 = nullptr;
        aclsparseCreateDnMat(&A2, 3, 5, 5, dV, ACL_FLOAT,
                             ACL_SPARSE_ORDER_ROW);
        ExpectErr(err, "shape mismatch",
               aclsparseDenseToSparseGetBufferSize(
                   h, A2, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                   bs),
               ACL_SPARSE_STATUS_INVALID_VALUE);
        aclsparseDestroyDnMat(A2);
    }
    {
        aclsparseSpMatDescr_t B2 = nullptr;
        aclsparseCreateCsr(&B2, 4, 5, 0, dOff, nullptr, nullptr,
                           ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                           ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT16);
        ExpectErr(err, "dtype mismatch",
               aclsparseDenseToSparseGetBufferSize(
                   h, A, B2, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                   bs),
               ACL_SPARSE_STATUS_INVALID_VALUE);
        aclsparseDestroySpMat(B2);
    }
}

static void CheckDescriptorRejections(aclsparseHandle_t h,
    aclsparseConstDnMatDescr_t A, aclsparseSpMatDescr_t B, size_t *bs,
    void *dOff, void *dV, std::vector<ErrItem> &err)
{
    CheckIndexBaseRejections(h, A, bs, dOff, err);
    CheckBellGeometryRejections(h, A, bs, dOff, dV, err);
    CheckShapeDtypeRejections(h, A, B, bs, dOff, dV, err);
}

// 尺寸上界：rows > INT32_MAX 在查询入口拒绝（描述符创建本身
// 不设限）。
static void CheckUpperDimLimit(aclsparseHandle_t h, void *dOff,
    std::vector<ErrItem> &err)
{
    void *dA = nullptr;
    if (aclrtMalloc(&dA, 4096, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return;
    }
    aclsparseDnMatDescr_t Abig = nullptr;
    aclsparseSpMatDescr_t Bbig = nullptr;
    const aclsparseStatus_t c1 = aclsparseCreateDnMat(
        &Abig, (int64_t)1 << 31, 1, 1, dA, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW);
    if (c1 == ACL_SPARSE_STATUS_SUCCESS) {
        aclsparseCreateCsr(&Bbig, (int64_t)1 << 31, 1, 0, dOff, nullptr,
                           nullptr, ACL_SPARSE_INDEX_32I,
                           ACL_SPARSE_INDEX_32I,
                           ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        size_t big = 0;
        ExpectErr(err, "rows>INT32_MAX query rejected",
                  aclsparseDenseToSparseGetBufferSize(
                      h, Abig, Bbig, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                      &big),
                  ACL_SPARSE_STATUS_INVALID_VALUE);
    } else {
        ExpectErr(err, "rows>INT32_MAX create rejected", c1,
                  ACL_SPARSE_STATUS_INVALID_VALUE);
    }
    if (Abig) aclsparseDestroyDnMat(Abig);
    if (Bbig) aclsparseDestroySpMat(Bbig);
    aclrtFree(dA);
}

// I32 上界（2147483647）查询不溢出即可。
static void CheckMaxLegalDims(aclsparseHandle_t h, void *dOff,
    std::vector<ErrItem> &err)
{
    void *dA = nullptr;
    if (aclrtMalloc(&dA, 4096, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return;
    }
    aclsparseDnMatDescr_t Abig = nullptr;
    aclsparseSpMatDescr_t Bbig = nullptr;
    const aclsparseStatus_t c2 = aclsparseCreateDnMat(
        &Abig, 2147483647LL, 1, 1, dA, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW);
    if (c2 == ACL_SPARSE_STATUS_SUCCESS) {
        aclsparseCreateCsr(&Bbig, 2147483647LL, 1, 0, dOff, nullptr,
                           nullptr, ACL_SPARSE_INDEX_32I,
                           ACL_SPARSE_INDEX_32I,
                           ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        size_t big = 0;
        const aclsparseStatus_t q = aclsparseDenseToSparseGetBufferSize(
            h, Abig, Bbig, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &big);
        ExpectErr(err, "max-legal-dims query", q,
                  ACL_SPARSE_STATUS_SUCCESS);
    }
    if (Abig) aclsparseDestroyDnMat(Abig);
    if (Bbig) aclsparseDestroySpMat(Bbig);
    aclrtFree(dA);
}

static void CheckDimensionLimits(aclsparseHandle_t h, void *dOff,
    std::vector<ErrItem> &err)
{
    CheckUpperDimLimit(h, dOff, err);
    CheckMaxLegalDims(h, dOff, err);
}



// Basic null/alg/stage-machine rejections on one prepared CSR pair.
static void CheckBasicRejections(aclsparseHandle_t h,
    aclsparseConstDnMatDescr_t A, aclsparseSpMatDescr_t B, void *ws,
    size_t *bs, void *dOff, void *dIdx, void *dV,
    std::vector<ErrItem> &err)
{
    // 三阶段状态
    ExpectErr(err, "convert-before-analysis",
           aclsparseDenseToSparseConvert(
               h, A, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, ws),
           ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(aclsparseDenseToSparseAnalysis(
                  h, A, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, ws),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(dts_stream), ACL_SUCCESS);
    aclsparseCsrSetPointers(B, dOff, dIdx, dV);
    // 空 handle / 描述符 / 指针
    ExpectErr(err, "null handle",
           aclsparseDenseToSparseGetBufferSize(
               nullptr, A, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, bs),
           ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectErr(err, "null matA",
           aclsparseDenseToSparseGetBufferSize(
               h, nullptr, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, bs),
           ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectErr(err, "null matB",
           aclsparseDenseToSparseGetBufferSize(
               h, A, nullptr, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, bs),
           ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectErr(err, "null bufferSize ptr",
           aclsparseDenseToSparseGetBufferSize(
               h, A, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
               nullptr),
           ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectErr(err, "null buffer when required>0",
           aclsparseDenseToSparseAnalysis(
               h, A, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
               nullptr),
           ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectErr(err, "invalid alg",
           aclsparseDenseToSparseGetBufferSize(
               h, A, B,
               (aclsparseDenseToSparseAlg_t)7, bs),
           ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST(ErrorPaths, InvalidArguments)
{
    std::vector<ErrItem> items;

    void *dA = nullptr, *dOff = nullptr, *dIdx = nullptr, *dV = nullptr;
    ASSERT_EQ(aclrtMalloc(&dA, 4096, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dOff, 8 * 4, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dIdx, 4096, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dV, 16384, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_SUCCESS);
    aclsparseDnMatDescr_t A = nullptr;
    aclsparseSpMatDescr_t B = nullptr;
    ASSERT_EQ(aclsparseCreateDnMat(&A, 4, 5, 5, dA, ACL_FLOAT,
                                   ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseCreateCsr(&B, 4, 5, 0, dOff, nullptr, nullptr,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    size_t bs = 0;
    void *ws = nullptr;
    aclsparseDenseToSparseGetBufferSize(
        dts_handle, A, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &bs);
    if (bs) {
        aclrtMalloc(&ws, bs, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    CheckBasicRejections(dts_handle, A, B, ws, &bs, dOff, dIdx, dV, items);

    CheckDescriptorRejections(dts_handle, A, B, &bs, dOff, dV, items);
    CheckDimensionLimits(dts_handle, dOff, items);
    for (const ErrItem &it : items) {
        CaseRecord r;
        r.caseId = std::string("ERR_") + it.name;
        r.gtest = CurrentGtestName();
        r.status = it.got == it.want ? "PASS" : "FAIL";
        r.check = "expected status";
        r.dtype = "-";
        r.shape = "-";
        r.detail = "got=" + std::to_string((int)it.got) +
                   " want=" + std::to_string((int)it.want);
        RecordCase(r);
        EXPECT_EQ(it.got, it.want) << it.name;
    }
}

// Real C-API stage-machine coverage.  In particular, Analysis writes the
// compressed offsets but not the later-bound index/value payload.  Rebinding
// that payload must retain the Analysis result; changing the offsets (or CSR
// strides) changes the analysis contract and requires another Analysis.
namespace {
struct DenseToSparseStageMem {
    void *dense = nullptr;
    void *offsets = nullptr;
    void *workspace = nullptr;
    void *otherWorkspace = nullptr;
    void *indices = nullptr;
    void *values = nullptr;
    void *otherOffsets = nullptr;
    void *rows = nullptr;
    void *cols = nullptr;
    aclsparseDnMatDescr_t matA = nullptr;
    aclsparseSpMatDescr_t matB = nullptr;

    ~DenseToSparseStageMem()
    {
        if (matB) aclsparseDestroySpMat(matB);
        if (matA) aclsparseDestroyDnMat(matA);
        for (void *p : {cols, rows, otherOffsets, values, indices,
                        otherWorkspace, workspace, offsets, dense}) {
            if (p) aclrtFree(p);
        }
    }
};

bool PrepareStagePair(const CaseCfg &cfg, const Dense64 &dense,
    const Golden &golden, DenseToSparseStageMem *mem, size_t *bufferSize)
{
    if (aclrtMalloc(&mem->dense, dense.bytes.size(),
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemcpy(mem->dense, dense.bytes.size(), dense.bytes.data(),
                    dense.bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE) !=
            ACL_SUCCESS ||
        aclsparseCreateDnMat(&mem->matA, cfg.rows, cfg.cols, dense.ld,
                             mem->dense, cfg.dtype, cfg.order) !=
            ACL_SPARSE_STATUS_SUCCESS ||
        aclrtMalloc(&mem->offsets,
                    static_cast<size_t>(golden.majorDim + 1) * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return false;
    }
    const aclsparseStatus_t create = cfg.format == ACL_SPARSE_FORMAT_CSR
        ? aclsparseCreateCsr(&mem->matB, cfg.rows, cfg.cols, 0, mem->offsets,
                             nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                             ACL_SPARSE_INDEX_32I, cfg.base, cfg.dtype)
        : aclsparseCreateCsc(&mem->matB, cfg.rows, cfg.cols, 0, mem->offsets,
                             nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                             ACL_SPARSE_INDEX_32I, cfg.base, cfg.dtype);
    if (create != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseDenseToSparseGetBufferSize(
            dts_handle, mem->matA, mem->matB,
            ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, bufferSize) !=
            ACL_SPARSE_STATUS_SUCCESS ||
        *bufferSize == 0 ||
        aclrtMalloc(&mem->workspace, *bufferSize,
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&mem->otherWorkspace, *bufferSize,
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return false;
    }
    return true;
}

bool AllocateCompressedPayload(DenseToSparseStageMem *mem, int64_t nnz,
    int elemBytes)
{
    return nnz > 0 &&
           aclrtMalloc(&mem->indices, static_cast<size_t>(nnz) * sizeof(int32_t),
                       ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
           aclrtMalloc(&mem->values, static_cast<size_t>(nnz) * elemBytes,
                       ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
}

aclsparseStatus_t SetCompressedPayload(aclsparseFormat_t format,
    aclsparseSpMatDescr_t mat, void *offsets, void *indices, void *values)
{
    return format == ACL_SPARSE_FORMAT_CSR
        ? aclsparseCsrSetPointers(mat, offsets, indices, values)
        : aclsparseCscSetPointers(mat, offsets, indices, values);
}

bool CheckStageCompressedResult(const CaseCfg &cfg, const Dense64 &dense,
    const Golden &golden, const DenseToSparseStageMem &mem)
{
    return CheckCompressedResult(cfg, dense, golden, golden.nnz, mem.offsets,
                                 mem.indices, nullptr, nullptr, mem.values);
}
CaseCfg MakeStageConfig(aclsparseFormat_t format, uint32_t seed)
{
    CaseCfg cfg{};
    cfg.format = format;
    cfg.dtype = ACL_FLOAT;
    cfg.base = ACL_SPARSE_INDEX_BASE_ZERO;
    cfg.order = ACL_SPARSE_ORDER_ROW;
    cfg.rows = 4;
    cfg.cols = 5;
    cfg.seed = seed;
    cfg.nnzPerRow = 3;
    return cfg;
}

aclsparseStatus_t ConvertStage(const DenseToSparseStageMem &mem, void *workspace)
{
    return aclsparseDenseToSparseConvert(dts_handle, mem.matA, mem.matB,
        ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, workspace);
}

void AnalyzeStage(const DenseToSparseStageMem &mem)
{
    ASSERT_EQ(aclsparseDenseToSparseAnalysis(dts_handle, mem.matA, mem.matB,
        ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, mem.workspace), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(dts_stream), ACL_SUCCESS);
}

void ConvertStageAndSynchronize(const DenseToSparseStageMem &mem)
{
    ASSERT_EQ(ConvertStage(mem, mem.workspace), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(dts_stream), ACL_SUCCESS);
}

void RecoverInvalidatedStage(const DenseToSparseStageMem &mem)
{
    EXPECT_EQ(ConvertStage(mem, mem.workspace), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_NO_FATAL_FAILURE(AnalyzeStage(mem));
    ASSERT_NO_FATAL_FAILURE(ConvertStageAndSynchronize(mem));
}

void CheckPayloadRebind(const CaseCfg &cfg, const Dense64 &dense,
    const Golden &golden, DenseToSparseStageMem *mem)
{
    // Convert cannot consume an Analysis that has not succeeded.
    EXPECT_EQ(ConvertStage(*mem, mem->workspace), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_NO_FATAL_FAILURE(AnalyzeStage(*mem));
    ASSERT_TRUE(AllocateCompressedPayload(mem, golden.nnz, dense.elemBytes));
    // Rebinding payload retains the offsets and successful Analysis state.
    ASSERT_EQ(SetCompressedPayload(cfg.format, mem->matB, mem->offsets,
        mem->indices, mem->values), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(ConvertStage(*mem, mem->otherWorkspace), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_NO_FATAL_FAILURE(ConvertStageAndSynchronize(*mem));
    EXPECT_TRUE(CheckStageCompressedResult(cfg, dense, golden, *mem));
}

void CheckCompressedInvalidation(const CaseCfg &cfg, const Dense64 &dense,
    const Golden &golden, DenseToSparseStageMem *mem)
{
    const size_t offsetsBytes = static_cast<size_t>(golden.majorDim + 1) * sizeof(int32_t);
    ASSERT_EQ(aclrtMalloc(&mem->otherOffsets, offsetsBytes,
        ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(mem->otherOffsets, offsetsBytes, mem->offsets,
        offsetsBytes, ACL_MEMCPY_DEVICE_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(SetCompressedPayload(cfg.format, mem->matB, mem->otherOffsets,
        mem->indices, mem->values), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_NO_FATAL_FAILURE(RecoverInvalidatedStage(*mem));
    // Re-analysis writes the replacement offsets, not mem->offsets.
    EXPECT_TRUE(CheckCompressedResult(cfg, dense, golden, golden.nnz,
        mem->otherOffsets, mem->indices, nullptr, nullptr, mem->values));
    if (cfg.format == ACL_SPARSE_FORMAT_CSR) {
        ASSERT_EQ(aclsparseCsrSetStrides(mem->matB, 1, 1, 1), ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_NO_FATAL_FAILURE(RecoverInvalidatedStage(*mem));
        EXPECT_TRUE(CheckCompressedResult(cfg, dense, golden, golden.nnz,
            mem->otherOffsets, mem->indices, nullptr, nullptr, mem->values));
    }
}

bool PrepareCooStagePair(const CaseCfg &cfg, const Dense64 &dense,
    DenseToSparseStageMem *mem)
{
    if (aclrtMalloc(&mem->dense, dense.bytes.size(), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemcpy(mem->dense, dense.bytes.size(), dense.bytes.data(),
            dense.bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclsparseCreateDnMat(&mem->matA, cfg.rows, cfg.cols, dense.ld,
            mem->dense, cfg.dtype, cfg.order) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseCreateCoo(&mem->matB, cfg.rows, cfg.cols, 0, nullptr, nullptr,
            nullptr, ACL_SPARSE_INDEX_32I, cfg.base, cfg.dtype) != ACL_SPARSE_STATUS_SUCCESS) {
        return false;
    }
    size_t bufferSize = 0;
    return aclsparseDenseToSparseGetBufferSize(dts_handle, mem->matA, mem->matB,
               ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize) == ACL_SPARSE_STATUS_SUCCESS &&
           bufferSize > 0 &&
           aclrtMalloc(&mem->workspace, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
}

void BindCooPayload(const Dense64 &dense, const Golden &golden, DenseToSparseStageMem *mem)
{
    const size_t indexBytes = static_cast<size_t>(golden.nnz) * sizeof(int32_t);
    ASSERT_EQ(aclrtMalloc(&mem->rows, indexBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&mem->cols, indexBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&mem->values, static_cast<size_t>(golden.nnz) * dense.elemBytes,
        ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclsparseCooSetPointers(mem->matB, mem->rows, mem->cols,
        mem->values), ACL_SPARSE_STATUS_SUCCESS);
}
} // namespace

TEST(DenseToSparseStage, CompressedPayloadRebindAndInvalidation)
{
    const aclsparseFormat_t formats[] = {ACL_SPARSE_FORMAT_CSR, ACL_SPARSE_FORMAT_CSC};
    for (const aclsparseFormat_t format : formats) {
        SCOPED_TRACE(FmtName(format));
        const CaseCfg cfg = MakeStageConfig(format, 20261012u);
        std::mt19937 rng(cfg.seed);
        const Dense64 dense = MakeDense(cfg, rng);
        const Golden golden = MakeGolden(cfg, dense);
        ASSERT_GT(golden.nnz, 0);
        DenseToSparseStageMem mem;
        size_t bufferSize = 0;
        ASSERT_TRUE(PrepareStagePair(cfg, dense, golden, &mem, &bufferSize));
        ASSERT_NO_FATAL_FAILURE(CheckPayloadRebind(cfg, dense, golden, &mem));
        ASSERT_NO_FATAL_FAILURE(CheckCompressedInvalidation(cfg, dense, golden, &mem));
    }
}

TEST(DenseToSparseStage, CooPayloadRebindKeepsAnalysis)
{
    const CaseCfg cfg = MakeStageConfig(ACL_SPARSE_FORMAT_COO, 20261013u);
    std::mt19937 rng(cfg.seed);
    const Dense64 dense = MakeDense(cfg, rng);
    const Golden golden = MakeGolden(cfg, dense);
    ASSERT_GT(golden.nnz, 0);
    DenseToSparseStageMem mem;
    ASSERT_TRUE(PrepareCooStagePair(cfg, dense, &mem));
    ASSERT_NO_FATAL_FAILURE(AnalyzeStage(mem));
    ASSERT_NO_FATAL_FAILURE(BindCooPayload(dense, golden, &mem));
    ASSERT_NO_FATAL_FAILURE(ConvertStageAndSynchronize(mem));
    EXPECT_TRUE(CheckCompressedResult(cfg, dense, golden, golden.nnz,
        nullptr, nullptr, mem.rows, mem.cols, mem.values));
}

// ===================== guard 区（金丝雀边界）检查（任务书 §3.5） =====================
// 全部 device 缓冲（输入 dense、workspace、CSR/CSC offsets、indices、values、
// BELL pattern/values）前后各 512B 金丝雀（0xA5）填充，完整三阶段执行后逐字节
// 校验金丝雀未被触碰——捕获 kernel 对任意 payload 的越界写。写入范围按
// golden 的精确 extent 分配（offsets=majorDim+1、indices/values=nnz、
// BELL=整除网格），任何超出 extent 的写入都会命中金丝雀。
namespace {
constexpr size_t kGuardZone = 512;
constexpr uint8_t kGuardCanary = 0xA5;

struct Guarded {
    void *base = nullptr;
    uint8_t *user = nullptr;
    size_t total = 0, usable = 0;
    std::vector<uint8_t> host;
};

Guarded AllocGuarded(size_t usable)
{
    Guarded gb;
    gb.usable = usable;
    gb.total = usable + 2 * kGuardZone;
    gb.host.assign(gb.total, kGuardCanary);
    if (aclrtMalloc(&gb.base, gb.total, ACL_MEM_MALLOC_HUGE_FIRST) ==
        ACL_SUCCESS) {
        gb.user = static_cast<uint8_t *>(gb.base) + kGuardZone;
        // 金丝雀区必须真实存在于 device：先整块上传，否则设备侧
        // guard 为未初始化内存，校验会误报。
        if (aclrtMemcpy(gb.base, gb.total, gb.host.data(), gb.total,
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            aclrtFree(gb.base);
            gb.base = nullptr;
            gb.user = nullptr;
        }
    }
    return gb;
}

bool UploadGuarded(Guarded &gb, const void *data, size_t n)
{
    CheckedMemcpy(gb.host.data() + kGuardZone, data, n);
    return aclrtMemcpy(gb.base, gb.total, gb.host.data(), gb.total,
                       ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool CanaryIntact(const Guarded &gb, const char *what)
{
    std::vector<uint8_t> back(gb.total);
    if (aclrtMemcpy(back.data(), gb.total, gb.base, gb.total,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    for (size_t i = 0; i < kGuardZone; ++i) {
        if (back[i] != kGuardCanary ||
            back[gb.total - 1 - i] != kGuardCanary) {
            std::printf("  [guard] %s corrupted at offset %zu / %zu\n",
                        what, i, gb.total - 1 - i);
            return false;
        }
    }
    return true;
}

// Guarded dense upload + DnMat descriptor.
bool GuardCreateDense(const Dense64 &d, const CaseCfg &cfg, Guarded &gd,
    aclsparseDnMatDescr_t *A)
{
    if (gd.user == nullptr || !UploadGuarded(gd, d.bytes.data(),
                                             d.bytes.size())) {
        return false;
    }
    return aclsparseCreateDnMat(A, cfg.rows, cfg.cols, d.ld, gd.user,
                                cfg.dtype, cfg.order) ==
           ACL_SPARSE_STATUS_SUCCESS;
}

// Guarded CSR/CSC payload buffers + descriptor.
bool GuardCreateCsrLike(const CaseCfg &cfg, const Dense64 &d, size_t nnz,
    int majorDim, Guarded *gOff, Guarded *gIdx, Guarded *gVals,
    aclsparseSpMatDescr_t *B)
{
    *gOff = AllocGuarded(static_cast<size_t>(majorDim + 1) * 4);
    *gIdx = AllocGuarded(nnz * 4);
    *gVals = AllocGuarded(nnz * d.elemBytes);
    if (!(gOff->user && gIdx->user && gVals->user)) {
        return false;
    }
    const auto create = cfg.format == ACL_SPARSE_FORMAT_CSC
                            ? aclsparseCreateCsc
                            : aclsparseCreateCsr;
    return create(B, cfg.rows, cfg.cols, static_cast<int64_t>(nnz),
                  gOff->user, gIdx->user, gVals->user, ACL_SPARSE_INDEX_32I,
                  ACL_SPARSE_INDEX_32I, cfg.base,
                  cfg.dtype) == ACL_SPARSE_STATUS_SUCCESS;
}

// Guarded COO payload buffers + descriptor.
bool GuardCreateCoo(const CaseCfg &cfg, const Dense64 &d, size_t nnz,
    Guarded *gRows, Guarded *gCols, Guarded *gVals,
    aclsparseSpMatDescr_t *B)
{
    *gRows = AllocGuarded(nnz * 4);
    *gCols = AllocGuarded(nnz * 4);
    *gVals = AllocGuarded(nnz * d.elemBytes);
    if (!(gRows->user && gCols->user && gVals->user)) {
        return false;
    }
    return aclsparseCreateCoo(B, cfg.rows, cfg.cols,
                              static_cast<int64_t>(nnz), gRows->user,
                              gCols->user, gVals->user,
                              ACL_SPARSE_INDEX_32I, cfg.base,
                              cfg.dtype) == ACL_SPARSE_STATUS_SUCCESS;
}

// Guarded BELL pattern/values buffers + descriptor.
bool GuardCreateBell(const CaseCfg &cfg, const Dense64 &d, Guarded *gPat,
    Guarded *gBell, aclsparseSpMatDescr_t *B)
{
    const int blockRows = cfg.rows / cfg.bellB;
    const int slots = cfg.ellCols / cfg.bellB;
    std::vector<int32_t> pat(static_cast<size_t>(blockRows) * slots);
    for (int i = 0; i < blockRows; ++i) {
        for (int s = 0; s < slots; ++s) {
            pat[static_cast<size_t>(i) * slots + s] = s;
        }
    }
    *gPat = AllocGuarded(pat.size() * 4);
    *gBell = AllocGuarded(static_cast<size_t>(cfg.rows) * cfg.ellCols *
                          d.elemBytes);
    if (!(gPat->user && gBell->user)) {
        return false;
    }
    if (!UploadGuarded(*gPat, pat.data(), pat.size() * 4)) {
        return false;
    }
    return aclsparseCreateBlockedEll(B, cfg.rows, cfg.cols, cfg.bellB,
                                     cfg.ellCols, gPat->user, gBell->user,
                                     ACL_SPARSE_INDEX_32I, cfg.base,
                                     cfg.dtype) == ACL_SPARSE_STATUS_SUCCESS;
}

// 三阶段执行（guard 区描述符与 workspace）。
bool GuardRunStages(const CaseCfg &cfg, aclsparseDnMatDescr_t A,
    aclsparseSpMatDescr_t B, bool isBell, Guarded &gws)
{
    if (!isBell) {
        size_t bs = 0;
        if (aclsparseDenseToSparseGetBufferSize(
                dts_handle, A, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                &bs) != ACL_SPARSE_STATUS_SUCCESS) {
            return false;
        }
        if (bs > 0) {
            gws = AllocGuarded(bs);
            if (gws.user == nullptr) {
                return false;
            }
        }
        if (aclsparseDenseToSparseAnalysis(
                dts_handle, A, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                gws.user ? gws.user : nullptr) != ACL_SPARSE_STATUS_SUCCESS) {
            return false;
        }
    }
    const bool ok = aclsparseDenseToSparseConvert(
                        dts_handle, A, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                        gws.user ? gws.user : nullptr) ==
                    ACL_SPARSE_STATUS_SUCCESS;
    aclrtSynchronizeStream(dts_stream);
    return ok;
}

// 校验全部金丝雀区。
bool GuardCanariesIntact(const Guarded &gd, const Guarded &gws,
    const Guarded &gOff, const Guarded &gIdx, const Guarded &gRows,
    const Guarded &gCols, const Guarded &gVals, const Guarded &gPat,
    const Guarded &gBell)
{
    if (!CanaryIntact(gd, "dense")) {
        return false;
    }
    if (gws.base && !CanaryIntact(gws, "workspace")) {
        return false;
    }
    if (gOff.base && !CanaryIntact(gOff, "offsets")) {
        return false;
    }
    if (gIdx.base && !CanaryIntact(gIdx, "indices")) {
        return false;
    }
    if (gRows.base && !CanaryIntact(gRows, "coo-rows")) {
        return false;
    }
    if (gCols.base && !CanaryIntact(gCols, "coo-cols")) {
        return false;
    }
    if (gVals.base && !CanaryIntact(gVals, "values")) {
        return false;
    }
    if (gPat.base && !CanaryIntact(gPat, "bell-pattern")) {
        return false;
    }
    if (gBell.base && !CanaryIntact(gBell, "bell-values")) {
        return false;
    }
    return true;
}

// One canary round-trip's device state.
struct GuardState {
    Guarded gd, gws, gOff, gIdx, gRows, gCols, gVals, gPat, gBell;
    aclsparseDnMatDescr_t A = nullptr;
    aclsparseSpMatDescr_t B = nullptr;
    bool ok = false;
};

// Descriptors + guarded buffers for the format under test.
bool GuardCreateAll(aclsparseFormat_t fmt, const CaseCfg &cfg,
    const Dense64 &d, const Golden &g, GuardState &st)
{
    const bool isCoo = fmt == ACL_SPARSE_FORMAT_COO;
    const bool isBell = fmt == ACL_SPARSE_FORMAT_BLOCKED_ELL;
    const int majorDim = isBell ? cfg.rows
                                : (fmt == ACL_SPARSE_FORMAT_CSC ? cfg.cols
                                                                : cfg.rows);
    const size_t nnz = isBell ? 0 : static_cast<size_t>(g.nnz);
    st.gd = AllocGuarded(d.bytes.size());
    st.ok = GuardCreateDense(d, cfg, st.gd, &st.A);
    if (st.ok && !isBell) {
        if (!isCoo) {
            st.ok = GuardCreateCsrLike(cfg, d, nnz, majorDim, &st.gOff,
                                       &st.gIdx, &st.gVals, &st.B);
        } else {
            st.ok = GuardCreateCoo(cfg, d, nnz, &st.gRows, &st.gCols,
                                   &st.gVals, &st.B);
        }
    } else if (st.ok) {
        st.ok = GuardCreateBell(cfg, d, &st.gPat, &st.gBell, &st.B);
    }
    return st.ok;
}

// Release the descriptors and guarded buffers of one round trip.
void GuardTearDown(GuardState &st)
{
    if (st.A) aclsparseDestroyDnMat(st.A);
    if (st.B) aclsparseDestroySpMat(st.B);
    for (Guarded *gb : {&st.gd, &st.gws, &st.gOff, &st.gIdx, &st.gRows,
                        &st.gCols, &st.gVals, &st.gPat, &st.gBell}) {
        if (gb->base) aclrtFree(gb->base);
    }
}

// Runs one canary round-trip for the given format; returns pass/fail.
bool RunGuardCase(aclsparseFormat_t fmt, const char *name)
{
    CaseCfg cfg{};
    cfg.format = fmt;
    cfg.dtype = ACL_FLOAT;
    cfg.base = ACL_SPARSE_INDEX_BASE_ZERO;
    cfg.order = ACL_SPARSE_ORDER_ROW;
    cfg.rows = 64;
    cfg.cols = 64;
    cfg.seed = 20260915;
    cfg.bellB = 16;
    cfg.ellCols = 32;
    std::mt19937 rng(cfg.seed);
    Dense64 d = MakeDense(cfg, rng);
    Golden g = MakeGolden(cfg, d);
    const bool isBell = fmt == ACL_SPARSE_FORMAT_BLOCKED_ELL;
    const size_t nnz = isBell ? 0 : static_cast<size_t>(g.nnz);

    GuardState st;
    if (GuardCreateAll(fmt, cfg, d, g, st)) {
        st.ok = GuardRunStages(cfg, st.A, st.B, isBell, st.gws);
    }
    if (st.ok) {
        st.ok = GuardCanariesIntact(st.gd, st.gws, st.gOff, st.gIdx,
                                    st.gRows, st.gCols, st.gVals, st.gPat,
                                    st.gBell);
    }
    const bool ok = st.ok;
    GuardTearDown(st);

    CaseRecord r;
    r.caseId = std::string("GUARD_") + name + "_FP32_64x64";
    r.gtest = CurrentGtestName();
    r.status = ok ? "PASS" : "FAIL";
    r.check = "512B canary guards intact around all device buffers";
    r.dtype = "fp32";
    r.shape = "64x64 ROW";
    r.nnz = static_cast<int64_t>(nnz);
    r.base = 0;
    r.seed = cfg.seed;
    r.detail = name;
    RecordCase(r);
    return ok;
}
} // namespace

TEST(GuardZones, CanaryAroundAllDeviceBuffers)
{
    struct FmtCase {
        const char *name;
        aclsparseFormat_t fmt;
    };
    const FmtCase cases[] = {{"csr", ACL_SPARSE_FORMAT_CSR},
                             {"csc", ACL_SPARSE_FORMAT_CSC},
                             {"coo", ACL_SPARSE_FORMAT_COO},
                             {"bell", ACL_SPARSE_FORMAT_BLOCKED_ELL}};
    for (const FmtCase &fc : cases) {
        EXPECT_TRUE(RunGuardCase(fc.fmt, fc.name))
            << "guard-zone violation in format " << fc.name;
    }
}

// ===================== main =====================

// cmake 接入路径共享 test/frame/test_main.cpp（约定：测试文件不得定义 main），
// 全局环境（acl 初始化/句柄）改由静态初始化器注册——AddGlobalTestEnvironment
// 只需在 RUN_ALL_TESTS 之前完成即可。
// -DD2S_UT_STANDALONE_MAIN 编译时保留自带 main。
namespace {
const int kEnvRegistered = [] {
    ::testing::AddGlobalTestEnvironment(new DenseToSparseUtEnv());
    return 0;
}();
} // namespace

#ifdef D2S_UT_STANDALONE_MAIN
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
