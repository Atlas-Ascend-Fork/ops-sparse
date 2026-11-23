/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPSV_PARAM_H_
#define TEST_SPSV_PARAM_H_

#include <cstdint>
#include <sstream>
#include <string>
#include "csv_loader.h"

namespace sparse_test {

struct SpSVParam : public SparseTestParamBase {
    std::string case_name;
    int64_t m = 0;
    int32_t slice_width = 1;
    std::string format = "CSR";
    std::string index_type = "i32";
    int32_t index_base = 0;  // 0=0-based, 1=1-based
    std::string rowptr_type;  // empty = use index_type; supports mixed index widths (e.g. i64 rowptr + i32 colind)
    std::string colind_type;  // empty = use index_type
    bool force_permtype64 = false;  // set SPSV_FORCE_PERMT_64=1 env var before execution
    std::string fill_mode = "LOWER";
    std::string diag_type = "NON_UNIT";
    std::string op_type = "NON_TRANSPOSE";
    // FLOAT or COMPLEX64.  Absent in a row means FLOAT, so the rows written
    // before this column existed keep their meaning untouched.
    std::string value_type = "FLOAT";
    float alpha = 1.0f;
    // Imaginary part of alpha; only read when value_type is COMPLEX64.
    float alpha_imag = 0.0f;
    double sparsity = 0.5;
    std::string structure = "random_triangular";
    bool in_place = false;
    bool unsorted = false;
    std::string update_mode = "NONE";
    bool null_vec = false;
    double mere_threshold = 1.22e-4;
    double mare_multiplier = 10.0;
    std::string description;
    std::string expect_result = "ACL_SPARSE_STATUS_SUCCESS";
    uint32_t seed = 42;

    void fillCustom(const csv_map& row) override {
        case_name       = parseString(row, "case_name");
        m               = static_cast<int64_t>(parseInt(row, "m"));
        slice_width     = parseInt(row, "slice_width");
        format          = parseString(row, "format");
        index_type      = parseString(row, "index_type");
        index_base      = static_cast<int32_t>(parseInt(row, "index_base"));
        rowptr_type     = parseString(row, "rowptr_type");
        colind_type     = parseString(row, "colind_type");
        force_permtype64 = parseBool(row, "force_permtype64");
        fill_mode       = parseString(row, "fill_mode");
        diag_type       = parseString(row, "diag_type");
        op_type         = parseString(row, "op_type");
        value_type      = parseString(row, "value_type");
        alpha           = parseFloat(row, "alpha");
        alpha_imag      = parseFloat(row, "alpha_imag");
        sparsity        = parseDouble(row, "sparsity");
        structure       = parseString(row, "structure");
        in_place        = parseBool(row, "in_place");
        unsorted        = parseBool(row, "unsorted");
        update_mode     = parseString(row, "update_mode");
        null_vec        = parseBool(row, "null_vec");
        mere_threshold  = parseDouble(row, "mere_threshold");
        mare_multiplier = parseDouble(row, "mare_multiplier");
        description     = parseString(row, "description");
        expect_result   = parseString(row, "expect_result");
        seed            = static_cast<uint32_t>(parseInt(row, "random_seed"));
        if (structure.empty()) structure = "random_triangular";
        if (format.empty()) format = "CSR";
        if (index_type.empty()) index_type = "i32";
        if (fill_mode.empty()) fill_mode = "LOWER";
        if (diag_type.empty()) diag_type = "NON_UNIT";
        if (op_type.empty()) op_type = "NON_TRANSPOSE";
        if (value_type.empty()) value_type = "FLOAT";
        if (update_mode.empty()) update_mode = "NONE";
        if (expect_result.empty()) expect_result = "ACL_SPARSE_STATUS_SUCCESS";
        if (slice_width <= 0) slice_width = 1;
    }

    std::string caseId() const override { return case_name; }

    bool isLower() const { return fill_mode == "LOWER"; }
    bool isUnitDiag() const { return diag_type == "UNIT"; }
    bool isComplex() const { return value_type == "COMPLEX64"; }
    // 0 = N, 1 = T, 2 = H.  The real path only needs the transpose flag, but
    // T and H differ once the values are complex.
    int opCode() const {
        if (op_type == "TRANSPOSE") return 1;
        if (op_type == "CONJUGATE_TRANSPOSE") return 2;
        return 0;
    }
    bool isTranspose() const {
        return op_type == "TRANSPOSE" || op_type == "CONJUGATE_TRANSPOSE";
    }
    bool isI64() const { return index_type == "i64"; }
    // Effective rowPtr/column-offset index type: use rowptr_type if set, else index_type
    bool isRowPtrI64() const {
        return rowptr_type.empty() ? isI64() : (rowptr_type == "i64");
    }
    // Effective colInd/rowInd index type: use colind_type if set, else index_type
    bool isColIndI64() const {
        return colind_type.empty() ? isI64() : (colind_type == "i64");
    }
};

inline void PrintTo(const SpSVParam& p, std::ostream* os) {
    *os << p.case_name;
}

}

#endif
