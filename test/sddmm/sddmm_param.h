/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SDDMM_SDDMM_PARAM_H_
#define TEST_SDDMM_SDDMM_PARAM_H_

#include "csv_loader.h"

#include <cstdint>
#include <string>

namespace sparse_test {

// SDDMM test parameter loaded from CSV.
//
// CSV columns (per test plan §8):
//   case_name,description,m,k,n,sparsity_ratio,alpha,beta,dtype,compute_type,
//   op_x,op_y,order_x,order_y,value_lo,value_hi,alg,mere_threshold,mare_multiplier,
//   expect_result,random_seed
//
// dtype / compute_type / op_x / op_y / order_x / order_y / alg are stored as raw
// enum-name strings and converted to ACL enum values in the test body / NPU
// wrapper (keeps this header free of cann_ops_sparse.h SDDMM declarations that
// may not yet exist).
struct SddmmTestParam : public SparseTestParamBase {
    std::string case_name;
    std::string description;

    int64_t m = 0;  // rows of X / C
    int64_t k = 0;  // reduction dim (cols of X and Y)
    int64_t n = 0;  // rows of Y / cols of C

    double sparsity_ratio = 0.0;
    double alpha = 0.0;
    double beta = 0.0;

    std::string dtype;         // "ACL_FLOAT" / "ACL_FLOAT16"
    std::string compute_type;  // "ACL_FLOAT" / "ACL_FLOAT16"
    std::string op_x;          // "ACL_SPARSE_OP_NON_TRANSPOSE"
    std::string op_y;          // "ACL_SPARSE_OP_NON_TRANSPOSE"

    // Dense matrix memory layout for X / Y.
    // "ACL_SPARSE_ORDER_ROW" (row-major, default) / "ACL_SPARSE_ORDER_COL" (col-major).
    // order_pair = orderX*2 + orderY maps to SDDMM_ORDER_RR/RC/CR/CC in the kernel,
    // selecting the contiguous vs strided GM->UB load path.
    std::string order_x;       // "ACL_SPARSE_ORDER_ROW" / "ACL_SPARSE_ORDER_COL"
    std::string order_y;       // "ACL_SPARSE_ORDER_ROW" / "ACL_SPARSE_ORDER_COL"

    double value_lo = -1.0;
    double value_hi = 1.0;

    std::string alg;           // "ACL_SPARSE_SDDMM_ALG_DEFAULT"

    double mere_threshold = 0.0;
    double mare_multiplier = 0.0;

    std::string expect_result;  // "ACL_SPARSE_STATUS_SUCCESS" for functional cases
    uint32_t random_seed = 0;

    void fillCustom(const csv_map& row) override {
        case_name        = parseString(row, "case_name");
        description      = parseString(row, "description");
        m                = static_cast<int64_t>(parseInt(row, "m"));
        k                = static_cast<int64_t>(parseInt(row, "k"));
        n                = static_cast<int64_t>(parseInt(row, "n"));
        sparsity_ratio   = parseDouble(row, "sparsity_ratio");
        alpha            = parseDouble(row, "alpha");
        beta             = parseDouble(row, "beta");
        dtype            = parseString(row, "dtype");
        compute_type     = parseString(row, "compute_type");
        op_x             = parseString(row, "op_x");
        op_y             = parseString(row, "op_y");
        // order_x / order_y default to ROW when the column is absent or empty,
        // so existing CSV rows (without these columns) keep the historical
        // row-major behaviour (order_pair = SDDMM_ORDER_RR).
        order_x          = parseString(row, "order_x");
        if (order_x.empty()) order_x = "ACL_SPARSE_ORDER_ROW";
        order_y          = parseString(row, "order_y");
        if (order_y.empty()) order_y = "ACL_SPARSE_ORDER_ROW";
        value_lo         = parseDouble(row, "value_lo");
        value_hi         = parseDouble(row, "value_hi");
        alg              = parseString(row, "alg");
        mere_threshold   = parseDouble(row, "mere_threshold");
        mare_multiplier  = parseDouble(row, "mare_multiplier");
        expect_result    = parseString(row, "expect_result");
        random_seed      = static_cast<uint32_t>(parseInt(row, "random_seed"));
    }

    std::string caseId() const override { return case_name; }
};

inline void PrintTo(const SddmmTestParam& p, std::ostream* os) {
    *os << p.case_name;
}

}  // namespace sparse_test

#endif  // TEST_SDDMM_SDDMM_PARAM_H_
