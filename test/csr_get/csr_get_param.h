/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSR_GET_CSR_GET_PARAM_H_
#define TEST_CSR_GET_CSR_GET_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

/// CsrGet 测试参数（由 csr_get_test.csv 加载）。
/// 字段对齐黑盒测试设计 5.1/5.2 用例表：rows/cols/nnz/row_off_type/col_ind_type/
/// dtype/idx_base/seed 等；CSR 双索引类型独立取值（32I/64I 四组合）。
/// accessor 不读设备数据，无值域生成需求，不设 value_lo/value_hi 字段。
struct CsrGetParam : public SparseTestParamBase {
    std::string case_name;

    int rows = 0;
    int cols = 0;
    int nnz = 0;
    std::string row_off_type;  // "32I" / "64I"（csrRowOffsetsType）
    std::string col_ind_type;  // "32I" / "64I"（csrColIndType，与 row_off_type 独立）
    int idx_base = 0;          // 0=ZERO, 1=ONE
    std::string dtype;         // "float32" / "float16"
    uint32_t seed = 0;

    std::string expect_result;

    bool isFp16() const { return dtype == "float16" || dtype == "fp16"; }

    void fillCustom(const csv_map &row) override {
        case_name       = parseString(row, "case_name");
        rows            = parseInt(row, "rows");
        cols            = parseInt(row, "cols");
        nnz             = parseInt(row, "nnz");
        row_off_type    = parseString(row, "row_off_type");
        col_ind_type    = parseString(row, "col_ind_type");
        idx_base        = parseInt(row, "idx_base");
        dtype           = parseString(row, "dtype");
        seed            = static_cast<uint32_t>(parseInt(row, "seed"));
        expect_result   = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_CSR_GET_CSR_GET_PARAM_H_
