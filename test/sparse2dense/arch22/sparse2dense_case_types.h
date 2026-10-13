// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE for the full text of the License.
// ============================================================================
// sparse2dense_case_types.h — sparse2dense CSV 用例类型定义（dtype/format/layout/值分布/
// 等价断言语义与 L2 负例变异枚举、CaseSpec/CaseRow 行结构）。仅被 sparse2dense_case_loader.h
// 与测试源使用。
// ============================================================================

#ifndef SPARSE2DENSE_CASE_TYPES_H_
#define SPARSE2DENSE_CASE_TYPES_H_

#include <cstdint>
#include <string>
#include <vector>

enum class DType { INT8, FP16, BF16, FP32, COMPLEX64 };

// ============================================================================
// 用例参数（CSV 行的工程化编码；用例唯一来源 sparse2dense_test.csv）
// ============================================================================
enum class Fmt { CSR, CSC, COO };
enum class Layout { ROW, COL };
// 官方 common/sparse_patterns.py ROW_PATTERNS 十模式（L1 矩阵 id%10 逐一对应）
enum class RowPattern {
    UNIFORM,            // id%10=0
    HIGHLY_IMBALANCED,  // id%10=1（w[0]=1000）
    ONE_LONG_ROW,       // id%10=2（w[0]=1e12）
    MANY_EMPTY_ROWS,    // id%10=3（前 ceil(nnz/cols) 行活跃）
    SKEWED,             // id%10=4（w[r]=2^min(r,15)）
    DIAGONAL,           // id%10=5（每行 1 元素列=行号）
    BANDED,             // id%10=6
    RANDOM,             // id%10=7（hash 权重 + gcd 步进）
    BLOCK_LIKE,         // id%10=8
    POWER_LAW,          // id%10=9
};
// 坐标存储模式（coords 列）：SORTED=sorted_unique / SHUFFLED=shuffled_unique（Fisher-Yates）/
// REVERSED=reversed_sorted_unique（同坐标集合逆序，L1_065 等价断言用）/
// DUPLICATE=duplicate（末元素坐标覆写为首元素——重复坐标成功语义，任意值判定）
enum class CoordMode { SORTED_UNIQUE, SHUFFLED_UNIQUE, REVERSED_UNIQUE, DUPLICATE };
// values 分布标签（value_spec values= 字段，用例级单分布口径）
enum class VKind {
    UNIFORM,      // uniform70:[lo:hi]（浮点均匀；int8 为整数域四舍五入夹取）
    EMPTY,        // nnz=0
    SP_NAN,       // special:[NaN x1 + uniform xN]（元素 0 = quiet NaN，含 payload 语义）
    SP_PINF,      // special:[+inf x1 + uniform xN]
    SP_NINF,      // special:[-inf x1 + uniform xN]
    ALL_NEG_ZERO, // special:[all -0.0]（-0 bit 与 +0 填充可区分）
    ALL_POS_ZERO, // special:[all +0.0]（输出 ≡ 零 nnz 输出）
    C64_INDEP,    // complex64 实/虚独立注入 +0/-0/NaN/±inf（L1_074）
};

struct CaseSpec {
    const char *caseId;
    DType dtype;
    Fmt fmt;
    Layout layout;
    int base;         // 0 / 1
    int64_t rows;
    int64_t cols;
    int64_t ld;       // ROW ld>=cols / COL ld>=rows
    int64_t nnz;
    RowPattern pattern;
    CoordMode coords; // COO 坐标模式（CSV coords 字段；CSR/CSC 恒 sorted_unique）
    uint64_t seed;
    VKind vkind = VKind::UNIFORM; // values 分布标签（L0 默认 uniform70:[-1:1]）
    double vLo = -1.0;            // uniform 域下界（int8 为整数域夹取下界）
    double vHi = 1.0;             // uniform 域上界
    // L0_009/L0_010 与 L1_063/L1_064 显式 offsets 覆写（含 base；CSV 偏差口径：
    // 以 offsets 端点为权威 nnz，PrepareCase 就地修正）
    const int32_t *explicitPrimary = nullptr;
    int explicitPrimaryLen = 0;
    bool protocolIdempotent = false; // L1_077：bufferSize 双查询幂等断言
};

// L1/L2 CSV 行类型与枚举（详见文件头 CSV 装载说明）
enum class Expand { NONE, DTYPE5, DTYPE4_FLOAT, CSC_TWIN };

// L2 负例变异类别（CSV notes 执行层语义的工程化编码）
enum class NegKind {
    NEG_NONE = 0,
    NEG_NULL_HANDLE,        // L2_001 handle=nullptr
    NEG_NULL_MATA,          // L2_002 matA=nullptr
    NEG_NULL_MATB,          // L2_003 matB=nullptr
    NEG_NULL_BUFPTR,        // L2_004 bufferSize 输出指针空
    NEG_COO_ROW_OOB,        // L2_005 COO 行坐标越界
    NEG_COO_COL_OOB,        // L2_006 COO 列坐标越界
    NEG_COO_NEG,            // L2_007 base1 stored=0 → 逻辑 -1
    NEG_CSR_COL_OOB,        // L2_008 CSR colInd 越界
    NEG_CSR_NONMONO,        // L2_010 CSR rowOffsets 非单调
    NEG_CSC_NONMONO,        // L2_011 CSC colOffsets 非单调
    NEG_LEN_PRIMARY,        // L2_012 len(primary) != rows+1
    NEG_VALUES_LEN,         // L2_013 COO lenPrimary != nnz
    NEG_DTYPE_MISMATCH,     // L2_014 matA/matB dtype 不一致
    NEG_DTYPE_UNSUPPORTED,  // L2_015 values dtype 不在 5 dtype
    NEG_IDX_64I,            // L2_016 索引 64I
    NEG_BASE2,              // L2_017 base=2
    NEG_LAYOUT,             // L2_018 非法 layout
    NEG_LD_ROW,             // L2_019 ROW ld<cols
    NEG_LD_COL,             // L2_020 COL ld<rows
    NEG_FORMAT,             // L2_021 非法 format
    NEG_ALG,                // L2_022 非法 alg
    NEG_ROWS_ZERO,          // L2_023 rows=0
    NEG_ALIAS,              // L2_025 matB 与 matA values alias
    NEG_OVERFLOW_COO_NNZ,   // L2_026 COO nnz 超 UINT32_MAX×tileElems
    NEG_OVERFLOW_ZEROBYTES, // L2_027 zeroTotalBytes 溢出 int64
    NEG_ROWS_MAX,           // L2_028 rows 超 machine_constraint 上界
    // 第十轮评审整改新增（公开 API 双入口负例：描述符级形状一致性 + alias 区间重叠）
    NEG_SHAPE_ROWS,             // L2_029/031/033 matB.rows != matA.rows（三格式行轴）
    NEG_SHAPE_COLS,             // L2_030/032/034 matB.cols != matA.cols（三格式列轴）
    NEG_ALIAS_INDEX,            // L2_035 索引数组与 dense 完全同址（区间重叠）
    NEG_ALIAS_PARTIAL_VALUES,   // L2_036 values 与 dense 部分重叠
    NEG_ALIAS_PARTIAL_PRIMARY,  // L2_037 primary 与 dense 部分重叠
};

// expected_value 判定语义（CSV expected_value 列）
enum class EV {
    NONE = 0,          // matches_oracle（标准 bit-wise 对拍）
    ALL_POS_ZERO,      // all_positive_zero（零 nnz/全 +0；notes 级参数化展开）
    ORDERED_EQ,        // ordered_input_equivalent（逆序 vs 有序输出 bit 一致）
    BASE0_EQ,          // base0_equivalent（base1 坐标+1 输出 == base0 输出）
    NEG_ZERO,          // neg_zero_bitwise_kept（全 -0 bit 保持；浮点 4 dtype 展开）
    EQ_ZERO_NNZ,       // equiv_zero_nnz_output（全 +0 输出 ≡ 零 nnz 输出）
    REPEAT10,          // repeat10_bitwise_identical（同输入重复执行 10 次 bit 一致）
    READONLY,          // inputs_bitwise_unchanged（输入只读显式专项）
    PROTOCOL,          // buffersize_zero_and_idempotent（双阶段协议 + 幂等）
    TRANSPOSE,         // col_output_eq_row_transposed（ROW/COL 输出互为转置）
    SUCCESS,           // buffersize_zero_and_buffer_nullable_exec_ok（L2_024 成功语义）
    DUP_ANY,           // duplicate_coords_any_value（L2_009 重复坐标成功语义：值 ∈ 集合）
};

// L1/L2 CSV 行（运行时装载的参数行）：公共执行参数（dtype/format/shape/
// pattern/coords/seed/values 分布/显式 offsets）复用 CaseSpec 具名段（caseId 即
// spec.caseId，经 SpecOf 展开）；行级专有字段为等价断言语义（ev/expand）与 L2 负例
// 变异/期望（neg/expectCode/stage/skipF3/realSkip）。
struct CaseRow {
    CaseSpec spec;
    EV ev;
    Expand expand;
    NegKind neg;            // L2 变异类别（L1 恒 NEG_NONE）
    const char *expectCode; // CSV expected_value（报错状态码/成功语义）
    int stage;              // 期望拒绝阶段：1=CheckMeta 2=ValidateF3 3=FillTiling
    bool skipF3;            // L2_026 合成参数：数据校验链 D2H 规模不可行，直接 FillTiling 判溢出
    bool realSkip;          // Real 直调路径结构性不适用（mock-only：L2_001/016）
    const char *realSkipReason;
    // 运行时装载的所有权承载：显式 offsets（spec.explicitPrimary 指向本存储）
    std::vector<int32_t> offsets;
};

// ============================================================================
// CSV 运行时装载：sparse2dense_test.csv（本文件同目录）→ CaseRow 参数表。
//   * L1 85 + L2 37 条装载为参数化用例来源；L0 12 条为手写 TEST_F（定制断言），CSV 行
//     仅作参数来源参照，不装载；
//   * dtype/format/layout/shape/nnz/value_spec/expected_value 逐列解析；L2 变异类别/
//     期望拒绝阶段/Real 路径适用性在静态表显式声明，装载时与 CSV expected_value 逐条
//     交叉断言，防漂移；
//   * 装载失败直接终止（CSV 与映射表/清单结构漂移属结构性错误，不以部分用例静默降级）。

#endif // SPARSE2DENSE_CASE_TYPES_H_
