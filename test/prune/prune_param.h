/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_PRUNE_PARAM_H_
#define TEST_PRUNE_PARAM_H_

#include <sstream>
#include <string>

#include "csv_loader.h"

namespace sparse_test {

// =============================================================================
// CSV-driven test parameter for aclsparseLtSpMMAPrune (prune-only).
//
// CSV columns (header must match exactly):
//   case_id,m,k,dtype,alg_config_id,split_k,pruneAlg,op,order,sparse_side,range_low,range_high,level
//
// The prune stage operates only on matrix A (m, k). alg_config_id and split_k
// are carried through the API chain (AlgSelection -> AlgSetAttribute -> Plan)
// but do NOT affect the prune numerical result; they are included to exercise
// the full API link and catch integration regressions.
//
// pruneAlg selects the pruning algorithm:
//   "STRIP" -> 1D top-N per group (existing, default for backward compat)
//   "TILE"  -> 2D tile-level pruning with row/col joint constraint (new)
// =============================================================================

struct PruneParam : public SparseTestParamBase {
    int32_t case_id = 0;
    int32_t m = 0;
    int32_t k = 0;
    std::string dtype;          // "float16" | "float32" | "bfloat16" | "int8"

    int32_t alg_config_id = 0;
    int32_t split_k = 1;

    std::string pruneAlg;       // "STRIP" | "TILE"  (default STRIP for backward compat)

    float range_low = -1.0f;
    float range_high = 1.0f;

    std::string op;             // "NON_TRANSPOSE" | "TRANSPOSE"
    std::string order;          // "ROW" | "COL"

    std::string sparse_side;    // "A" | "B"  (default "A" for backward compat)
    std::string level;          // L0 | L1 | L2
    int32_t num_batches = 1;    // batch 用例（CSV 扩列；缺省 0 → 1 向后兼容）
    int64_t batch_stride = 0;   // 元素单位；缺省 0（nb=1 时无效）
    int64_t ld = 0;             // structured 侧 ld（0 = 紧凑；>min = padded 输入）

    bool isFp16() const { return dtype == "float16"; }
    bool isBf16() const { return dtype == "bfloat16"; }
    bool isInt8() const { return dtype == "int8"; }
    bool isTranspose() const { return op == "TRANSPOSE"; }
    bool isColOrder() const { return order == "COL"; }
    bool isTile() const { return pruneAlg == "TILE"; }
    bool isSparseA() const { return sparse_side != "B"; }   // default A when empty

    void fillCustom(const csv_map& row) override {
        case_id         = parseInt(row, "case_id");
        m               = parseInt(row, "m");
        k               = parseInt(row, "k");
        dtype           = parseString(row, "dtype");
        alg_config_id   = parseInt(row, "alg_config_id");
        split_k         = parseInt(row, "split_k");
        pruneAlg        = parseString(row, "pruneAlg");
        if (pruneAlg.empty()) { pruneAlg = "STRIP"; }  // backward compat
        range_low       = parseFloat(row, "range_low");
        range_high      = parseFloat(row, "range_high");
        op              = parseString(row, "op");
        order           = parseString(row, "order");
        sparse_side     = parseString(row, "sparse_side");
        if (sparse_side.empty()) { sparse_side = "A"; }  // backward compat
        level           = parseString(row, "level");
        num_batches     = parseInt(row, "num_batches");
        if (num_batches <= 0) { num_batches = 1; }       // 缺列/0 → 1（向后兼容）
        batch_stride    = static_cast<int64_t>(parseInt(row, "batch_stride"));
        ld              = static_cast<int64_t>(parseInt(row, "ld"));
    }

    bool hasBatch() const { return num_batches > 1; }

    std::string caseId() const override {
        std::ostringstream oss;
        oss << "case" << case_id << "_" << level;
        return oss.str();
    }
};

inline void PrintTo(const PruneParam& p, std::ostream* os) {
    *os << "case" << p.case_id << "(" << p.dtype << " " << p.m << "x" << p.k
        << " alg" << p.alg_config_id << " splitk" << p.split_k
        << " " << (p.isTile() ? "TILE" : "STRIP")
        << " op=" << (p.isTranspose() ? "T" : "N")
        << " order=" << (p.isColOrder() ? "COL" : "ROW")
        << " side=" << (p.isSparseA() ? "A" : "B") << ")";
}

}  // namespace sparse_test

#endif  // TEST_PRUNE_PARAM_H_
