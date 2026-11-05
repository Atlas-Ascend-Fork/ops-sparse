/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_BLOCKEDELL_GET_BLOCKEDELL_GET_PARAM_H_
#define TEST_BLOCKEDELL_GET_BLOCKEDELL_GET_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

/// BlockedEllGet 测试参数（由 blockedell_get_test.csv 加载）。
/// rows/cols 为矩阵逻辑维度，ell_block_size/ell_cols 为 Blocked-ELL 布局参数；
/// 容量按补齐后的块网格 ceil(rows/bs) * ceil(ellCols/bs) * bs^2 计算。
struct BlockedEllGetParam : public SparseTestParamBase {
    std::string case_name;

    int rows = 0;
    int cols = 0;
    int ell_block_size = 1;
    int ell_cols = 0;
    int idx_base = 0;        // 0=ZERO, 1=ONE
    std::string dtype;       // "float32" / "float16"
    uint32_t seed = 0;

    std::string expect_result;

    bool isFp16() const { return dtype == "float16" || dtype == "fp16"; }

    void fillCustom(const csv_map &row) override {
        case_name       = parseString(row, "case_name");
        rows            = parseInt(row, "rows");
        cols            = parseInt(row, "cols");
        ell_block_size  = parseInt(row, "ell_block_size");
        ell_cols        = parseInt(row, "ell_cols");
        idx_base        = parseInt(row, "idx_base");
        dtype           = parseString(row, "dtype");
        seed            = static_cast<uint32_t>(parseInt(row, "seed"));
        expect_result   = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_BLOCKEDELL_GET_BLOCKEDELL_GET_PARAM_H_
