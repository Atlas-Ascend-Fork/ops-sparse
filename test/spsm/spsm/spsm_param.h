/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPSM_SPSM_PARAM_H_
#define TEST_SPSM_SPSM_PARAM_H_

#include <string>

#include "csv_loader.h"

namespace sparse_test {

// CSV column definition (v2, 26 columns, aligned with spsm_test.csv header):
//   case_name,m,n,ldb,dtype,uplo,trans,diag,order,alpha,value_lo,value_hi,density,seed,
//   expect_result,b_zero,a_empty,inplace,unsorted,
//   matrix_struct,bw,blk,singular,index_base,one_based_data,opB
//
// v2 new fields (relative to v1):
//   diag         : "UNIT" / "NON_UNIT"  (v2 §3.7, NON_UNIT support)
//   order        : "COL" / "ROW"        (v2 §7.3, matB/matC memory layout)
//   inplace      : 0 / 1                (matB/matC share buffer)
//   unsorted     : 0 / 1                (row-internal colInd shuffle)
//   matrix_struct: "RANDOM" / "BANDED" / "BLOCK_DIAG"
//   bw           : bandwidth (BANDED only, 0 otherwise)
//   blk          : block size (BLOCK_DIAG only, 0 otherwise)
//   singular     : 0 / 1                (NON_UNIT zero diagonal, singular matrix)
//   index_base   : "ZERO" / "ONE"       (CSR index base)
//   one_based_data : 0 / 1              (L3 whitebox: 1 -> real 1-based colInd data
//                                        transmitted to NPU descriptor; golden stays 0-based)
//   opB          : "N" / "T"            (op(B); both modes are supported)
struct SpsmParam : public SparseTestParamBase {
    std::string case_name;

    int m = 0;          // A: m x m, B/X: m x n
    int n = 0;          // number of RHS columns
    int ldb = 0;        // leading dimension of B / X (>= m for COL, >= n for ROW)

    std::string dtype;  // "float32" (FP32 only; FP16 not supported)
    std::string uplo;   // "LOWER" / "UPPER"
    std::string trans;  // "N" / "T"
    std::string diag;   // "UNIT" / "NON_UNIT" (v2 new)
    std::string order;  // "COL" / "ROW" (v2 new)

    float alpha = 1.0f;
    double value_lo = -1.0;
    double value_hi = 1.0;
    double density = 0.2;
    uint32_t seed = 0;

    std::string expect_result;  // "SUCCESS" / "NOT_SUPPORTED" / ...

    int b_zero = 0;     // 1 -> B = 0
    int a_empty = 0;    // 1 -> A off-diagonal nnz = 0 (diag still stored for NON_UNIT)
    int inplace = 0;    // 1 -> matB/matC share buffer (v2 new)
    int unsorted = 0;   // 1 -> row-internal colInd shuffle (v2 new)

    std::string matrix_struct;  // "RANDOM" / "BANDED" / "BLOCK_DIAG" (v2 new)
    int bw = 0;                 // bandwidth (BANDED, v2 new)
    int blk = 0;                // block size (BLOCK_DIAG, v2 new)
    int singular = 0;           // 1 -> NON_UNIT zero diagonal (v2 new)

    std::string index_base;     // "ZERO" / "ONE" (v2 new)

    int one_based_data = 0;     // 1 -> real 1-based colInd to NPU (L3 whitebox new)

    std::string opB;            // "N" / "T" (op(B))

    // Convenience accessors
    bool isLower() const { return uplo == "LOWER"; }
    bool isTranspose() const { return trans == "T"; }
    bool isOpBTranspose() const { return opB == "T"; }
    bool isUnitDiag() const { return diag == "UNIT" || diag.empty(); }
    bool isRowOrder() const { return order == "ROW"; }
    bool isInplace() const { return inplace != 0; }
    bool isUnsorted() const { return unsorted != 0; }
    bool isSingular() const { return singular != 0; }
    bool isBanded() const { return matrix_struct == "BANDED"; }
    bool isBlockDiag() const { return matrix_struct == "BLOCK_DIAG"; }
    bool isIndexBaseOne() const { return index_base == "ONE"; }
    bool isOneBasedData() const { return one_based_data != 0; }

    void fillCustom(const csv_map& row) override {
        case_name       = parseString(row, "case_name");
        m               = parseInt(row, "m");
        n               = parseInt(row, "n");
        ldb             = parseInt(row, "ldb");
        dtype           = parseString(row, "dtype");
        uplo            = parseString(row, "uplo");
        trans           = parseString(row, "trans");
        diag            = parseString(row, "diag");
        order           = parseString(row, "order");
        alpha           = parseFloat(row, "alpha");
        value_lo        = parseDouble(row, "value_lo");
        value_hi        = parseDouble(row, "value_hi");
        density         = parseDouble(row, "density");
        seed            = static_cast<uint32_t>(parseInt(row, "seed"));
        expect_result   = parseString(row, "expect_result");
        b_zero          = parseInt(row, "b_zero");
        a_empty         = parseInt(row, "a_empty");
        inplace         = parseInt(row, "inplace");
        unsorted        = parseInt(row, "unsorted");
        matrix_struct   = parseString(row, "matrix_struct");
        bw              = parseInt(row, "bw");
        blk             = parseInt(row, "blk");
        singular        = parseInt(row, "singular");
        index_base      = parseString(row, "index_base");
        one_based_data  = parseInt(row, "one_based_data");
        opB             = parseString(row, "opB");

        // Defaults for backward compatibility
        if (diag.empty()) diag = "UNIT";
        if (order.empty()) order = "COL";
        if (matrix_struct.empty()) matrix_struct = "RANDOM";
        if (index_base.empty()) index_base = "ZERO";
        if (opB.empty()) opB = "N";
        if (ldb == 0) ldb = isRowOrder() ? n : m;
        if (density == 0.0) density = 0.2;
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_SPSM_SPSM_PARAM_H_
