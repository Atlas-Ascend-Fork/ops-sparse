// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE for the full text of the License.
// ============================================================================
// sparse2dense_case_loader.h — sparse2dense CSV 用例装载器：sparse2dense_test.csv ->
// CaseRow 参数表
//（逐列解析 + L2 变异交叉断言 + L1/L2 清单校验；装载失败直接终止）。
// ============================================================================

#ifndef SPARSE2DENSE_CASE_LOADER_H_
#define SPARSE2DENSE_CASE_LOADER_H_

#include "sparse2dense_case_types.h"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
constexpr const char *kCaseCsvName = "sparse2dense_test.csv";

// 装载失败终止（早于 gtest 用例执行，进程级失败语义）
[[noreturn]] inline void LoaderFatal(const std::string &what, const std::string &detail)
{
    std::fprintf(stderr, "[sparse2dense_csv] FATAL %s: %s\n", what.c_str(), detail.c_str());
    std::abort();
}

inline std::string TrimSpace(const std::string &s)
{
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

// 单行 CSV 解析（带引号字段与引号转义；本 CSV 无引号内换行）
inline std::vector<std::string> SplitCsvLine(const std::string &line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    field.push_back('"');
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                field.push_back(c);
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == ',') {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    fields.push_back(field);
    return fields;
}

// ";" 分隔 k=v 对（value_spec / shape 列编码）
inline std::map<std::string, std::string> ParseKv(const std::string &spec)
{
    std::map<std::string, std::string> out;
    size_t pos = 0;
    while (pos <= spec.size()) {
        const size_t sep = spec.find(';', pos);
        const size_t stop = (sep == std::string::npos) ? spec.size() : sep;
        const std::string part = TrimSpace(spec.substr(pos, stop - pos));
        pos = stop + 1;
        if (part.empty()) {
            continue;
        }
        const size_t eq = part.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        out[TrimSpace(part.substr(0, eq))] = TrimSpace(part.substr(eq + 1));
    }
    return out;
}

inline std::string KvGet(const std::map<std::string, std::string> &kv, const char *key, const char *dflt)
{
    const auto it = kv.find(key);
    return (it == kv.end()) ? std::string(dflt) : it->second;
}

// value_spec values= 标签 → VKind + uniform 域（special 标签域不参与，固定 [-1,1]）
struct ValuesSpec {
    VKind kind = VKind::UNIFORM;
    double lo = -1.0;
    double hi = 1.0;
};

inline ValuesSpec ParseValues(const std::string &label)
{
    if (label == "empty") {
        return {VKind::EMPTY, -1.0, 1.0};
    }
    if (label.rfind("uniform70:[", 0) == 0 && label.size() > 11 && label.back() == ']') {
        const std::string inner = label.substr(11, label.size() - 12);
        const size_t colon = inner.find(':');
        if (colon == std::string::npos) {
            LoaderFatal("uniform70 label missing ':'", label);
        }
        return {VKind::UNIFORM, std::strtod(inner.substr(0, colon).c_str(), nullptr),
            std::strtod(inner.substr(colon + 1).c_str(), nullptr)};
    }
    if (label.find("NaN") != std::string::npos) {
        return {VKind::SP_NAN, -1.0, 1.0};
    }
    if (label.find("+inf") != std::string::npos) {
        return {VKind::SP_PINF, -1.0, 1.0};
    }
    if (label.find("-inf") != std::string::npos) {
        return {VKind::SP_NINF, -1.0, 1.0};
    }
    if (label.find("all -0.0") != std::string::npos) {
        return {VKind::ALL_NEG_ZERO, -1.0, 1.0};
    }
    if (label.find("all +0.0") != std::string::npos) {
        return {VKind::ALL_POS_ZERO, -1.0, 1.0};
    }
    if (label.find("real/imag") != std::string::npos) {
        return {VKind::C64_INDEP, -1.0, 1.0};
    }
    LoaderFatal("unknown values label", label);
}

// ---- CSV 标签 → 枚举（与 CSV 列值域一一对应；未知标签即装载失败）----
inline DType ParseDType(const std::string &k)
{
    if (k == "INT8") { return DType::INT8; }
    if (k == "FLOAT16") { return DType::FP16; }
    if (k == "BFLOAT16") { return DType::BF16; }
    if (k == "FLOAT32") { return DType::FP32; }
    if (k == "COMPLEX64") { return DType::COMPLEX64; }
    LoaderFatal("unknown dtype", k);
}

inline Fmt ParseFmt(const std::string &k)
{
    if (k == "csr") { return Fmt::CSR; }
    if (k == "csc") { return Fmt::CSC; }
    if (k == "coo" || k == "blocked_ell") { return Fmt::COO; } // blocked_ell 归一到 COO 承接
    LoaderFatal("unknown format", k);
}

inline Layout ParseLayout(const std::string &k)
{
    if (k == "ROW" || k == "INVALID") { return Layout::ROW; } // 非法 layout 由 runner 注入 order=2
    if (k == "COL") { return Layout::COL; }
    LoaderFatal("unknown layout", k);
}

inline RowPattern ParsePattern(const std::string &k)
{
    if (k == "uniform") { return RowPattern::UNIFORM; }
    if (k == "highly_imbalanced") { return RowPattern::HIGHLY_IMBALANCED; }
    if (k == "one_long_row") { return RowPattern::ONE_LONG_ROW; }
    if (k == "many_empty_rows") { return RowPattern::MANY_EMPTY_ROWS; }
    if (k == "skewed") { return RowPattern::SKEWED; }
    if (k == "diagonal") { return RowPattern::DIAGONAL; }
    if (k == "banded") { return RowPattern::BANDED; }
    if (k == "random") { return RowPattern::RANDOM; }
    if (k == "block_like") { return RowPattern::BLOCK_LIKE; }
    if (k == "power_law") { return RowPattern::POWER_LAW; }
    LoaderFatal("unknown row_pattern", k);
}

inline CoordMode ParseCoords(const std::string &k)
{
    if (k == "shuffled_unique") { return CoordMode::SHUFFLED_UNIQUE; }
    if (k == "reversed_sorted_unique") { return CoordMode::REVERSED_UNIQUE; }
    if (k == "duplicate") { return CoordMode::DUPLICATE; } // 重复坐标（成功语义，任意值判定）
    // sorted_unique / empty 及 L2 非法坐标标签（out_of_bounds_*/*_len_mismatch）：
    // 坐标由负例 runner 按 NegKind 构造，行参数仅作默认 SORTED 承接
    if (k == "sorted_unique" || k == "empty" || k == "out_of_bounds_row" || k == "out_of_bounds_col" ||
        k == "out_of_bounds_neg" || k == "primary_len_mismatch" ||
        k == "values_len_mismatch") {
        return CoordMode::SORTED_UNIQUE;
    }
    LoaderFatal("unknown coords", k);
}

// expected_value 判定语义标签（L1 行专用；L2 负例行的状态码字符串不在此域）
inline EV ParseEv(const std::string &k)
{
    if (k.empty()) { return EV::NONE; }
    if (k == "all_positive_zero") { return EV::ALL_POS_ZERO; }
    if (k == "ordered_input_equivalent") { return EV::ORDERED_EQ; }
    if (k == "base0_equivalent") { return EV::BASE0_EQ; }
    if (k == "neg_zero_bitwise_kept") { return EV::NEG_ZERO; }
    if (k == "equiv_zero_nnz_output") { return EV::EQ_ZERO_NNZ; }
    if (k == "repeat10_bitwise_identical") { return EV::REPEAT10; }
    if (k == "inputs_bitwise_unchanged") { return EV::READONLY; }
    if (k == "buffersize_zero_and_idempotent") { return EV::PROTOCOL; }
    if (k == "col_output_eq_row_transposed") { return EV::TRANSPOSE; }
    if (k == "buffersize_zero_and_buffer_nullable_exec_ok") { return EV::SUCCESS; }
    LoaderFatal("unknown expected_value", k);
}

// 等价断言参数化展开（notes 级语义）
inline Expand ParseExpand(const std::string &caseId)
{
    if (caseId == "L1_061") { return Expand::DTYPE5; }
    if (caseId == "L1_072" || caseId == "L1_073") { return Expand::DTYPE4_FLOAT; }
    if (caseId == "L1_082") { return Expand::CSC_TWIN; }
    return Expand::NONE;
}

// ---- L2 变异映射（case_id → NegKind / 期望拒绝阶段 / CSV expected_value 交叉断言 /
// Real 适用性）。stage：1=CheckMeta（元数据/属性/len/dtype）2=ValidateF3（坐标内容）
// 3=FillTiling（溢出防护）----
struct MutationSpec {
    const char *caseId;
    NegKind neg;
    int stage;
    const char *expect;
    bool skipF3;
    bool realSkip;
    const char *reason;
};

const MutationSpec kL2Mutations[] = {
    {"L2_001", NegKind::NEG_NULL_HANDLE, 1, "ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR", false, true,
     "handle 概念属 aclsparse API 层；Real 直调协议（host_impl CheckMeta 入口）无 handle 形参，"
     "结构性不适用——Mock 描述符链承接"},
    {"L2_002", NegKind::NEG_NULL_MATA, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_003", NegKind::NEG_NULL_MATB, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_004", NegKind::NEG_NULL_BUFPTR, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_005", NegKind::NEG_COO_ROW_OOB, 2, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_006", NegKind::NEG_COO_COL_OOB, 2, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_007", NegKind::NEG_COO_NEG, 2, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_008", NegKind::NEG_CSR_COL_OOB, 2, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_010", NegKind::NEG_CSR_NONMONO, 2, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_011", NegKind::NEG_CSC_NONMONO, 2, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_012", NegKind::NEG_LEN_PRIMARY, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_013", NegKind::NEG_VALUES_LEN, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_014", NegKind::NEG_DTYPE_MISMATCH, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_015", NegKind::NEG_DTYPE_UNSUPPORTED, 1, "ACL_SPARSE_STATUS_NOT_SUPPORTED", false, false, ""},
    {"L2_016", NegKind::NEG_IDX_64I, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, true,
     "索引 dtype 由 S2DSpMatView 的 int32_t 指针类型结构性固定为 I32（Real 直调协议无索引"
     " dtype 形参）——Mock 描述符链承接（接口文档级枚举断言）"},
    {"L2_017", NegKind::NEG_BASE2, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_018", NegKind::NEG_LAYOUT, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_019", NegKind::NEG_LD_ROW, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_020", NegKind::NEG_LD_COL, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_021", NegKind::NEG_FORMAT, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_022", NegKind::NEG_ALG, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_023", NegKind::NEG_ROWS_ZERO, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_025", NegKind::NEG_ALIAS, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_026", NegKind::NEG_OVERFLOW_COO_NNZ, 3, "ACL_SPARSE_STATUS_INVALID_VALUE", true, false, ""},
    {"L2_027", NegKind::NEG_OVERFLOW_ZEROBYTES, 3, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_028", NegKind::NEG_ROWS_MAX, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    // 第十轮评审整改：公开 API 双入口负例（bufferSize + execute 断言，拒绝先于任何 kernel；
    // 形状一致性在描述符层、alias 区间重叠在 CheckMeta 段三——stage=1 元数据阶段语义）
    {"L2_029", NegKind::NEG_SHAPE_ROWS, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_030", NegKind::NEG_SHAPE_COLS, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_031", NegKind::NEG_SHAPE_ROWS, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_032", NegKind::NEG_SHAPE_COLS, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_033", NegKind::NEG_SHAPE_ROWS, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_034", NegKind::NEG_SHAPE_COLS, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_035", NegKind::NEG_ALIAS_INDEX, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_036", NegKind::NEG_ALIAS_PARTIAL_VALUES, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
    {"L2_037", NegKind::NEG_ALIAS_PARTIAL_PRIMARY, 1, "ACL_SPARSE_STATUS_INVALID_VALUE", false, false, ""},
};

const MutationSpec *FindMutation(const std::string &caseId)
{
    for (const MutationSpec &m : kL2Mutations) {
        if (caseId == m.caseId) {
            return &m;
        }
    }
    return nullptr;
}

// 逗号分隔 int32 列表（row_offsets/col_offsets；含 base，以端点为权威 nnz）
inline std::vector<int32_t> ParseOffsets(const std::string &text)
{
    std::vector<int32_t> out;
    size_t pos = 0;
    while (pos <= text.size()) {
        const size_t sep = text.find(',', pos);
        const size_t stop = (sep == std::string::npos) ? text.size() : sep;
        out.push_back(static_cast<int32_t>(
            std::strtol(TrimSpace(text.substr(pos, stop - pos)).c_str(), nullptr, 10)));
        pos = stop + 1;
    }
    return out;
}

// 读表（本文件同目录优先；工作目录兜底承接 __FILE__ 为相对路径的编译方式）
inline std::vector<std::vector<std::string>> ReadCaseCsvTable()
{
    std::vector<std::vector<std::string>> table;
    const std::string file = __FILE__;
    const size_t slash = file.find_last_of("/\\");
    const std::string nearPath = (slash == std::string::npos)
        ? std::string(kCaseCsvName)
        : file.substr(0, slash + 1) + kCaseCsvName;
    bool opened = false;
    for (const std::string &p : {nearPath, std::string(kCaseCsvName)}) {
        std::ifstream in(p);
        if (!in.is_open()) {
            continue;
        }
        opened = true;
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty()) {
                table.push_back(SplitCsvLine(line));
            }
        }
        break;
    }
    if (!opened) {
        LoaderFatal("cannot open case csv", nearPath);
    }
    return table;
}

// 行列取值（列号越界/该行缺列返回空串）
inline std::string CsvCell(const std::vector<std::vector<std::string>> &table, size_t r, int c)
{
    const std::vector<std::string> &f = table[r];
    return (c >= 0 && static_cast<size_t>(c) < f.size()) ? f[static_cast<size_t>(c)]
                                                         : std::string();
}

// 列索引（表头名 → 列号）
struct CsvColumnIndex {
    int id;    // case_id
    int level; // level
    int dtype; // dtype
    int shape; // shape
    int count; // cal_count
    int spec;  // value_spec
    int ev;    // expected_value
};

inline CsvColumnIndex FindCsvColumns(const std::vector<std::string> &header)
{
    auto colOf = [&header](const char *name) -> int {
        for (size_t i = 0; i < header.size(); ++i) {
            if (TrimSpace(header[i]) == name) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };
    CsvColumnIndex cols;
    cols.id = colOf("case_id");
    cols.level = colOf("level");
    cols.dtype = colOf("dtype");
    cols.shape = colOf("shape");
    cols.count = colOf("cal_count");
    cols.spec = colOf("value_spec");
    cols.ev = colOf("expected_value");
    if (cols.id < 0 || cols.level < 0 || cols.dtype < 0 || cols.shape < 0 || cols.count < 0 ||
        cols.spec < 0 || cols.ev < 0) {
        LoaderFatal("case csv header missing required columns", kCaseCsvName);
    }
    return cols;
}

// L2 行级变异落位（成功语义直取或负例变异映射；与 CSV expected_value 逐条交叉断言）
inline void ApplyL2Mutation(CaseRow &row, const std::string &caseId, const std::string &expected)
{
    if (expected == "buffersize_zero_and_buffer_nullable_exec_ok") {
        // 恒 0 语义下 buffer=nullptr 执行成功（黑盒成功语义）
        row.ev = EV::SUCCESS;
        return;
    }
    if (expected == "duplicate_coords_any_value") {
        // 重复坐标成功语义（cuSPARSE 任意值）：非重复位置 bit-wise、重复位置值 ∈ 集合
        row.ev = EV::DUP_ANY;
        return;
    }
    const MutationSpec *m = FindMutation(caseId);
    if (m == nullptr) {
        LoaderFatal("L2 case missing mutation mapping", caseId);
    }
    if (expected != m->expect) { // 与 CSV expected_value 逐条交叉断言（防漂移）
        LoaderFatal("expected_value mismatch vs mutation table",
            caseId + ": csv=" + expected + " table=" + m->expect);
    }
    row.neg = m->neg;
    row.ev = EV::NONE;
    row.expectCode = m->expect;
    row.stage = m->stage;
    row.skipF3 = m->skipF3;
    row.realSkip = m->realSkip;
    row.realSkipReason = m->reason;
}

// row_offsets/col_offsets 显式覆写装载（row_offsets 优先；含 base，以端点为权威 nnz）
inline std::vector<int32_t> ParseRowOffsets(const std::map<std::string, std::string> &kv)
{
    std::string offsets = KvGet(kv, "row_offsets", "");
    if (offsets.empty()) {
        offsets = KvGet(kv, "col_offsets", "");
    }
    if (offsets.empty()) {
        return {};
    }
    return ParseOffsets(offsets);
}

// 单行 → CaseRow（value_spec kv 解析/dtype/fmt/layout/shape/values/coords/pattern/
// seed/offsets/ev/expand；L2 行含变异映射，L0 行由调用方跳过）
inline CaseRow ParseCaseRow(const std::vector<std::vector<std::string>> &table, const CsvColumnIndex &cols,
    size_t r, const std::string &level, std::deque<std::string> *idStorage)
{
    const std::string caseId = CsvCell(table, r, cols.id);
    const std::map<std::string, std::string> kv = ParseKv(CsvCell(table, r, cols.spec));
    const ValuesSpec vs = ParseValues(KvGet(kv, "values", "empty"));
    const std::map<std::string, std::string> shape = ParseKv(CsvCell(table, r, cols.shape));
    if (shape.count("rows") == 0 || shape.count("cols") == 0 || shape.count("ld") == 0) {
        LoaderFatal("shape column missing rows/cols/ld", caseId);
    }

    CaseRow row;
    idStorage->push_back(caseId);
    row.spec.caseId = idStorage->back().c_str(); // deque 驻留：指针在后续增长下稳定
    row.spec.dtype = ParseDType(CsvCell(table, r, cols.dtype));
    row.spec.fmt = ParseFmt(KvGet(kv, "format", "coo"));
    row.spec.layout = ParseLayout(KvGet(kv, "layout", "ROW"));
    row.spec.base = static_cast<int>(std::strtoll(KvGet(kv, "base", "0").c_str(), nullptr, 10));
    row.spec.rows = std::strtoll(shape.at("rows").c_str(), nullptr, 10);
    row.spec.cols = std::strtoll(shape.at("cols").c_str(), nullptr, 10);
    row.spec.ld = std::strtoll(shape.at("ld").c_str(), nullptr, 10);
    row.spec.nnz = std::strtoll(TrimSpace(CsvCell(table, r, cols.count)).c_str(), nullptr, 10);
    row.spec.pattern = ParsePattern(KvGet(kv, "row_pattern", "uniform"));
    row.spec.coords = ParseCoords(KvGet(kv, "coords", "empty"));
    const std::string seedRaw = KvGet(kv, "seed", "0");
    row.spec.seed = (seedRaw == "na" || seedRaw.empty())
        ? 0ULL
        : std::strtoull(seedRaw.c_str(), nullptr, 10);
    row.spec.vkind = vs.kind;
    row.spec.vLo = vs.lo;
    row.spec.vHi = vs.hi;
    row.offsets = ParseRowOffsets(kv);

    // 行级专有字段（L1 缺省即负例域空置；L2_024 成功语义同缺省 + EV::SUCCESS）
    row.ev = EV::NONE;
    row.expand = ParseExpand(caseId);
    row.neg = NegKind::NEG_NONE;
    row.expectCode = "nullptr";
    row.stage = 0;
    row.skipF3 = false;
    row.realSkip = false;
    row.realSkipReason = "";
    const std::string expected = TrimSpace(CsvCell(table, r, cols.ev));
    if (level == "L2") {
        ApplyL2Mutation(row, caseId, expected);
    } else if (level == "L1") {
        row.ev = ParseEv(expected);
    } else {
        LoaderFatal("unknown level", level);
    }
    return row;
}

// 清单校验（条数/编号连续性，防漂移）
inline void ValidateManifest(const std::vector<CaseRow> &l1, const std::vector<CaseRow> &l2)
{
    if (l1.size() != 85) {
        LoaderFatal("L1 count != 85", std::to_string(l1.size()));
    }
    if (l2.size() != 37) {
        LoaderFatal("L2 count != 37", std::to_string(l2.size()));
    }
    for (size_t i = 0; i < l1.size(); ++i) {
        std::ostringstream os;
        os << "L1_" << std::setw(3) << std::setfill('0') << (i + 1);
        if (std::string(l1[i].spec.caseId) != os.str()) {
            LoaderFatal("L1 case_id sequence mismatch (expect L1_001..L1_085)", l1[i].spec.caseId);
        }
    }
    for (size_t i = 0; i < l2.size(); ++i) {
        std::ostringstream os;
        os << "L2_" << std::setw(3) << std::setfill('0') << (i + 1);
        if (std::string(l2[i].spec.caseId) != os.str()) {
            LoaderFatal("L2 case_id sequence mismatch (expect L2_001..L2_037)", l2[i].spec.caseId);
        }
    }
}

// 装载编排：读表 → 列索引 → 逐行解析按 level 分组 → 清单校验 → 汇总落位
inline std::vector<CaseRow> BuildCsvRows(std::deque<std::string> *idStorage)
{
    const std::vector<std::vector<std::string>> table = ReadCaseCsvTable();
    const CsvColumnIndex cols = FindCsvColumns(table[0]);

    // ---- 逐行装载（L0 为手写 TEST_F 参数参照，跳过）----
    std::vector<CaseRow> l1;
    std::vector<CaseRow> l2;
    for (size_t r = 1; r < table.size(); ++r) {
        const std::string level = CsvCell(table, r, cols.level);
        if (level == "L0") {
            continue;
        }
        CaseRow row = ParseCaseRow(table, cols, r, level, idStorage);
        if (level == "L2") {
            l2.push_back(std::move(row));
        } else {
            l1.push_back(std::move(row));
        }
    }
    ValidateManifest(l1, l2);

    // ---- 汇总 + 显式 offsets 指针落位（move 保持 offsets 堆缓冲；行存储进程级存活，
    // CaseSpec 的 caseId/explicitPrimary 指针语义由本存储承载）----
    std::vector<CaseRow> rows;
    rows.reserve(l1.size() + l2.size());
    for (CaseRow &r : l1) {
        rows.push_back(std::move(r));
    }
    for (CaseRow &r : l2) {
        rows.push_back(std::move(r));
    }
    for (CaseRow &r : rows) {
        if (!r.offsets.empty()) {
            r.spec.explicitPrimary = r.offsets.data();
            r.spec.explicitPrimaryLen = static_cast<int>(r.offsets.size());
        }
    }
    return rows;
}


// L1/L2 用例参数表（进程级单次装载）
const std::vector<CaseRow> &LoadCsvRows()
{
    static std::deque<std::string> idStorage;
    static const std::vector<CaseRow> rows = BuildCsvRows(&idStorage);
    return rows;
}

#endif  // SPARSE2DENSE_CASE_LOADER_H_


