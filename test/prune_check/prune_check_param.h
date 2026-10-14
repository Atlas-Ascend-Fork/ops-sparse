/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_PRUNE_CHECK_PARAM_H_
#define TEST_PRUNE_CHECK_PARAM_H_

#include <cstdint>
#include <sstream>
#include <string>

#include "csv_loader.h"

namespace sparse_test {

// =============================================================================
// CSV-driven test parameter for aclsparseLtSpMMAPruneCheck.
//
// CSV 列定义（表头须与下述列名逐字一致，依据 2.1 测试方案 §5）：
//   case_id,m,k,dtype,op,order,sparse_side,data_pattern,viol_pos,
//   range_low,range_high,level,expect_valid,expect_status
//
// - 本算子签名不含 alg_config_id / split_k / pruneAlg（PruneCheck 不经过
//   AlgSelection / Plan 链路，对齐需求 §2.2 schema），故无这些列。
// - expect_valid  : 设计期人工推导的独立预期（三向一致交叉断言的第三方，
//                   非容差项；与 golden 冲突时以权威口径 golden 为准并排查）。
// - expect_status : SUCCESS（正常路径：ret=SUCCESS + d_valid 三向断言）|
//                   DESC_INVALID（描述符层 INVALID_VALUE 拦截：非 32B/elem
//                   整除 shape 经 aclsparseLtStructuredDescriptorInit 拒绝，
//                   PruneCheck 不可达；空时默认 SUCCESS）。
// =============================================================================

struct PruneCheckParam : public SparseTestParamBase {
    int32_t case_id = 0;
    int32_t m = 0;
    int32_t k = 0;
    std::string dtype;          // "float16" | "float32" | "bfloat16" | "int8"

    std::string op;             // "NON_TRANSPOSE" | "TRANSPOSE"
    std::string order;          // "ROW" | "COL"
    std::string sparse_side;    // "A" | "B"（默认 A）

    std::string data_pattern;   // 见 prune_check_golden.h 构造器表
    std::string viol_pos;       // "FIRST" | "MIDDLE" | "LAST"（违规类模式；空 = 不适用）

    float range_low = -1.0f;
    float range_high = 1.0f;

    std::string level;          // L0 | L1 | L2

    int32_t expect_valid = 0;   // 人工预期 d_valid（0/1；三向一致第三方）
    std::string expect_status;  // "SUCCESS" | "DESC_INVALID"（空 = SUCCESS）

    bool isFp16() const { return dtype == "float16"; }
    bool isBf16() const { return dtype == "bfloat16"; }
    bool isInt8() const { return dtype == "int8"; }
    bool isTranspose() const { return op == "TRANSPOSE"; }
    bool isColOrder() const { return order == "COL"; }
    bool isSparseA() const { return sparse_side != "B"; }   // default A when empty
    // 剪枝方向（派生因子，与 prune golden / prune_check host 完全同源）：
    // alongRow = (trans != isRowOrder)，isRowOrder = (order == ROW)，即
    // trans == isColOrder。沿行 = N+ROW / T+COL；沿列 = N+COL / T+ROW。
    bool isAlongRow() const { return isTranspose() == isColOrder(); }
    bool isDescInvalid() const { return expect_status == "DESC_INVALID"; }

    void fillCustom(const csv_map& row) override {
        case_id         = parseInt(row, "case_id");
        m               = parseInt(row, "m");
        k               = parseInt(row, "k");
        dtype           = parseString(row, "dtype");
        op              = parseString(row, "op");
        order           = parseString(row, "order");
        sparse_side     = parseString(row, "sparse_side");
        if (sparse_side.empty()) { sparse_side = "A"; }  // 兼容旧版 CSV：留空默认 A 侧
        data_pattern    = parseString(row, "data_pattern");
        viol_pos        = parseString(row, "viol_pos");
        range_low       = parseFloat(row, "range_low");
        range_high      = parseFloat(row, "range_high");
        level           = parseString(row, "level");
        expect_valid    = parseInt(row, "expect_valid");
        expect_status   = parseString(row, "expect_status");
        if (expect_status.empty()) { expect_status = "SUCCESS"; }
    }

    std::string caseId() const override {
        std::ostringstream oss;
        oss << "case" << case_id << "_" << level;
        return oss.str();
    }
};

inline void PrintTo(const PruneCheckParam& p, std::ostream* os) {
    *os << "case" << p.case_id << "(" << p.dtype << " " << p.m << "x" << p.k
        << " op=" << (p.isTranspose() ? "T" : "N")
        << " order=" << (p.isColOrder() ? "COL" : "ROW")
        << " side=" << (p.isSparseA() ? "A" : "B")
        << " " << p.data_pattern
        << (p.viol_pos.empty() ? "" : "(" + p.viol_pos + ")")
        << " expect=" << p.expect_valid
        << (p.isDescInvalid() ? " [DESC_INVALID]" : "") << ")";
}

}  // namespace sparse_test

#endif  // TEST_PRUNE_CHECK_PARAM_H_
