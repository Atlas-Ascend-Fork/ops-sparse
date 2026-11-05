/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSC_GET_CSC_GET_PARAM_H_
#define TEST_CSC_GET_CSC_GET_PARAM_H_

#include <string>

#include "cann_ops_sparse.h"
#include "csv_loader.h"

namespace sparse_test {

/// CscGet 测试参数（由 csc_get_test.csv 加载）。
/// 字段对齐 0-黑盒测试设计.md §2.2 CSV 列定义；相对 coo_get 新增 ptr_type/idx_type
/// 两列（CSC 的 cscColOffsetsType 与 cscRowIndType 可独立选择，需覆盖四种组合）。
struct CscGetParam : public SparseTestParamBase {
    std::string case_name;

    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int idx_base = 0;        // 0=ZERO, 1=ONE
    int ptr_type = 32;       // cscColOffsetsType: 32=ACL_SPARSE_INDEX_32I, 64=ACL_SPARSE_INDEX_64I
    int idx_type = 32;       // cscRowIndType:     32=ACL_SPARSE_INDEX_32I, 64=ACL_SPARSE_INDEX_64I
    std::string dtype;       // "float32" / "float16"
    double value_lo = -1.0;
    double value_hi = 1.0;
    uint32_t seed = 0;

    std::string expect_result;

    bool isFp16() const { return dtype == "float16" || dtype == "fp16"; }

    aclsparseIndexType_t ptrIndexType() const {
        return (ptr_type == 64) ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    }

    aclsparseIndexType_t idxIndexType() const {
        return (idx_type == 64) ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    }

    void fillCustom(const csv_map &row) override {
        case_name       = parseString(row, "case_name");
        rows            = parseInt(row, "rows");
        cols            = parseInt(row, "cols");
        nnz             = parseInt(row, "nnz");
        idx_base        = parseInt(row, "idx_base");
        ptr_type        = parseInt(row, "ptr_type");
        idx_type        = parseInt(row, "idx_type");
        dtype           = parseString(row, "dtype");
        value_lo        = parseDouble(row, "value_lo");
        value_hi        = parseDouble(row, "value_hi");
        seed            = static_cast<uint32_t>(parseInt(row, "seed"));
        expect_result   = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_CSC_GET_CSC_GET_PARAM_H_
