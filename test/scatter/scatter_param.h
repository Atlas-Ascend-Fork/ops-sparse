/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SCATTER_SCATTER_PARAM_H_
#define TEST_SCATTER_SCATTER_PARAM_H_

#include "csv_loader.h"
#include <cstdint>
#include <string>

namespace sparse_test {

// ============================================================================
// ScatterParam: parameter struct for aclsparseScatter, loaded from CSV.
//
// CSV columns (matching test plan §5):
//   case_name, vec_size, vec_nnz, dn_size, val_type, idx_type,
//   idx_base, idx_sorted, value_pattern, seed, expect_result
// ============================================================================
struct ScatterParam : public SparseTestParamBase {
    std::string case_name;

    int64_t vec_size  = 0;   // vecX.size  (sparse vector logical size)
    int64_t vec_nnz   = 0;   // vecX.nnz   (nonzero count)
    int64_t dn_size   = 0;   // vecY.nums  (dense vector size)

    std::string val_type;    // "INT8" / "FP32" / "FP16" / "BF16" / "COMPLEX64"
    std::string idx_type;    // "I32" / "I64"
    int idx_base    = 0;     // 0 = ZERO, 1 = ONE
    bool idx_sorted = true;  // true = sorted, false = unsorted
    std::string value_pattern;  // "normal" / "special" / "extreme"

    uint32_t seed = 0;

    std::string expect_result;  // "SUCCESS" / "INVALID_VALUE" / "NOT_SUPPORTED"

    void fillCustom(const csv_map& row) override {
        case_name      = parseString(row, "case_name");
        vec_size       = static_cast<int64_t>(parseInt(row, "vec_size"));
        vec_nnz        = static_cast<int64_t>(parseInt(row, "vec_nnz"));
        dn_size        = static_cast<int64_t>(parseInt(row, "dn_size"));
        val_type       = parseString(row, "val_type");
        idx_type       = parseString(row, "idx_type");
        idx_base       = parseInt(row, "idx_base");
        idx_sorted     = parseBool(row, "idx_sorted");
        value_pattern  = parseString(row, "value_pattern");
        seed           = static_cast<uint32_t>(parseInt(row, "seed"));
        expect_result  = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

inline void PrintTo(const ScatterParam& p, std::ostream* os) {
    *os << p.case_name;
}

}  // namespace sparse_test

#endif  // TEST_SCATTER_SCATTER_PARAM_H_
