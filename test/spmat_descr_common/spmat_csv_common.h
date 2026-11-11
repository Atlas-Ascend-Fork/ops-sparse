/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPMAT_DESCR_COMMON_SPMAT_CSV_COMMON_H_
#define TEST_SPMAT_DESCR_COMMON_SPMAT_CSV_COMMON_H_

// ============================================================================
// 稀疏描述符 API 测试（spmat_get_attribute / create_sliced_ell / create_bsr /
// bsr_set_strided_batch / coo_set_pointers / csr_set_pointers / destroy_spmat）
// 的 CSV 私有共享件：公共参数基类 + 带构成校验的装载。
//
// 本目录（test/spmat_descr_common/）为上述 7 个测试目录的私有公共区，
// 无 CMakeLists.txt、不会被构建系统发现为测试目标；include 路径由各测试
// 目录的 CMakeLists.txt 以 target_include_directories PRIVATE 显式引入。
// 公共区（test/frame/）零改动：本头文件仅使用 upstream 未动的
// csv_loader.h（SparseTestParamBase / GetCasesFromCsv / 解析助手）。
// ============================================================================

#include <cstddef>
#include <stdexcept>

#include "csv_loader.h"

namespace sparse_test {

// 稀疏矩阵描述符 CSV 测试参数公共字段基类：case_name + 几何（rows/cols/nnz）
// + 索引类型/基 + dtype + null 指针模式 + expect_result，由 create_sliced_ell /
// coo_set_pointers / csr_set_pointers 的 CSV 公共列集共享；子类保留差异字段与
// 各自 doc 注释（null_ptrs 语义随入口不同），fillCustom 先解析差异列再调用
// fillMatCommon（无差异列的子类直接继承基类实现）。
struct SparseMatCommonParamBase : public SparseTestParamBase {
    std::string case_name;

    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    int idx_type = 0;         // 0=ACL_SPARSE_INDEX_32I, 1=ACL_SPARSE_INDEX_64I
    int idx_base = 0;         // 0=ZERO, 1=ONE（入口构造参数，经对应 Get 回读验证）
    std::string dtype;        // float32 / float16
    int null_ptrs = 0;        // null 指针语义见各子类注释
    std::string expect_result;

    bool isFp16() const
    {
        if (dtype == "float16") {
            return true;
        }
        if (dtype == "float32") {
            return false;
        }
        throw std::runtime_error("SparseMatCommonParamBase::isFp16: unsupported CSV dtype column value '" +
                                 dtype + "' (expected float32/float16)");
    }

    void fillCustom(const csv_map& row) override { fillMatCommon(row); }

    std::string caseId() const override { return case_name; }

protected:
    // 公共列解析（子类 fillCustom 解析完自身差异列后调用）
    void fillMatCommon(const csv_map& row) {
        case_name     = parseString(row, "case_name");
        rows          = parseInt64(row, "rows");
        cols          = parseInt64(row, "cols");
        nnz           = parseInt64(row, "nnz");
        idx_type      = parseInt(row, "idx_type");
        idx_base      = parseInt(row, "idx_base");
        dtype         = parseString(row, "dtype");
        // 装载期 dtype 合法性检查（CG-1.1）：未识别列值在 CSV 装载阶段即抛
        // runtime_error（对齐 GetCasesFromCsvChecked 口径，含用例名上下文），
        // 防拼写错误静默降级；isFp16() 复用同一取值域
        if (dtype != "float16" && dtype != "float32") {
            throw std::runtime_error("case '" + case_name + "': unsupported CSV dtype column value '" +
                                     dtype + "' (expected float32/float16)");
        }
        null_ptrs     = parseInt(row, "null_ptrs");
        expect_result = parseString(row, "expect_result");
    }
};

// 带构成校验的装载：装载后断言 CSV 数据行数严格等于 expectedCount（空表
// 也判失败），不符抛 std::runtime_error，使构成漂移（拷贝失败/CRLF 行尾
// 异常/review 误删行等导致的截断或丢行）在装载期即失败、套件不静默缩水
// 变绿（对齐 ops-blas #735 CsvCaseCount / #866 VerifyCsvComposition、
// test/symv/csymv LoadCsymvCases 的装载期 runtime_error 口径）。
// INSTANTIATE_TEST_SUITE_P 在静态初始化期求值，未捕获异常使测试进程启动
// 即终止并打印 what()，构成错误在 CI 中确定性暴露。
// 既有 GetCasesFromCsv 签名不变，其它测试不受影响。
template <typename ParamType>
std::vector<ParamType> GetCasesFromCsvChecked(const std::string& csvPath,
                                              size_t expectedCount) {
    auto rawRows = ReadMap(csvPath);
    if (rawRows.size() != expectedCount) {
        throw std::runtime_error("GetCasesFromCsvChecked: " + csvPath + " expects " +
                                 std::to_string(expectedCount) + " data rows but got " +
                                 std::to_string(rawRows.size()) +
                                 " (CSV truncated/lost lines/composition drifted?)");
    }
    std::vector<ParamType> cases;
    cases.reserve(rawRows.size());
    for (const auto& row : rawRows) {
        ParamType p;
        p.fillCustom(row);
        cases.push_back(std::move(p));
    }
    return cases;
}

}  // namespace sparse_test

#endif  // TEST_SPMAT_DESCR_COMMON_SPMAT_CSV_COMMON_H_
