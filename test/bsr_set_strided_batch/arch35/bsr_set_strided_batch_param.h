/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_BSR_SET_STRIDED_BATCH_BSR_SET_STRIDED_BATCH_PARAM_H_
#define TEST_BSR_SET_STRIDED_BATCH_BSR_SET_STRIDED_BATCH_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

/// BsrSetStridedBatch 测试参数（由 bsr_set_strided_batch_test.csv 加载）。
/// 三 stride 单位均为元素个数；span 语义：offsetsSpan=block_rows+1、
/// columnsSpan=block_nnz、valuesSpan=block_nnz×block_size²。
/// 几何因子：A=2×2/bs2/nnzb2（span 3/2/8）、B=3×4/bs4/nnzb5（span 4/5/80）、
/// C=2×2/bs2/nnzb0（span 3/0/0）、D=8×8/bs8/nnzb8（span 9/8/512）、
/// E=2×2/bs128/nnzb2（span 3/2/32768），由 CSV 行内几何列直接给定。
struct BsrSetStridedBatchParam : public SparseTestParamBase {
    std::string case_name;

    int64_t block_rows = 0;      // CreateBsr 几何因子
    int64_t block_cols = 0;
    int64_t block_nnz = 0;
    int64_t block_size = 2;      // rowBlockSize == colBlockSize
    std::string dtype;           // float32 / float16 / bfloat16
    int batch_count = 1;         // [1, 65535]
    int64_t offsets_stride = 0;  // 0=广播；>0 须 >= offsetsSpan
    int64_t columns_stride = 0;  // 0=广播；>0 须 >= columnsSpan
    int64_t values_stride = 0;   // bc>1 且 valuesSpan>0 时禁 0 且须 >= valuesSpan

    std::string expect_result;

    void fillCustom(const csv_map &row) override {
        case_name      = parseString(row, "case_name");
        block_rows     = parseInt64(row, "block_rows");
        block_cols     = parseInt64(row, "block_cols");
        block_nnz      = parseInt64(row, "block_nnz");
        block_size     = parseInt64(row, "block_size");
        dtype          = parseString(row, "dtype");
        batch_count    = parseInt(row, "batch_count");
        offsets_stride = parseInt64(row, "offsets_stride");
        columns_stride = parseInt64(row, "columns_stride");
        values_stride  = parseInt64(row, "values_stride");
        expect_result  = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_BSR_SET_STRIDED_BATCH_BSR_SET_STRIDED_BATCH_PARAM_H_
