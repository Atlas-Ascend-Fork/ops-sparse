/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CREATE_BSR_CREATE_BSR_PARAM_H_
#define TEST_CREATE_BSR_CREATE_BSR_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

/// CreateBsr 测试参数（由 create_bsr_test.csv 加载）。
/// entry ∈ {bsr, const_bsr}；block_size 同时作 row/col 块大小（实现要求方形块）；
/// null_ptrs=1 时 colInd/values 传 null（仅用于 block_nnz=0 行；offsets 恒传有效指针）。
struct CreateBsrParam : public SparseTestParamBase {
    std::string case_name;

    std::string entry;          // bsr / const_bsr
    int64_t block_rows = 0;
    int64_t block_cols = 0;
    int64_t block_nnz = 0;
    int64_t block_size = 2;     // rowBlockSize == colBlockSize
    int idx_base = 0;           // 0=ZERO, 1=ONE
    std::string dtype;          // float32 / float16 / bfloat16 / complex64
    std::string order;          // ROW / COL
    int null_ptrs = 0;          // 1 -> bsrColInd/bsrValues 传 null（仅 block_nnz=0 行）

    std::string expect_result;

    void fillCustom(const csv_map &row) override {
        case_name   = parseString(row, "case_name");
        entry       = parseString(row, "entry");
        block_rows  = parseInt64(row, "block_rows");
        block_cols  = parseInt64(row, "block_cols");
        block_nnz   = parseInt64(row, "block_nnz");
        block_size  = parseInt64(row, "block_size");
        idx_base    = parseInt(row, "idx_base");
        dtype       = parseString(row, "dtype");
        order       = parseString(row, "order");
        null_ptrs   = parseInt(row, "null_ptrs");
        expect_result = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_CREATE_BSR_CREATE_BSR_PARAM_H_
