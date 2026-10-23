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

#ifndef TEST_MATMUL_PARAM_H_
#define TEST_MATMUL_PARAM_H_

#include <cfloat>
#include <sstream>
#include <string>

#include "csv_loader.h"

namespace sparse_test {

// =============================================================================
// CSV-driven test parameter for aclsparseLtMatmul (4 dtype × 2 path).
//
// CSV columns (16 columns, v2):
//   case_id, m, k, n, dtype, output_dtype, matrix_type,
//   alpha, beta, transA, transB, alg_config_id, split_k, split_k_mode, prune_alg,
//   alpha_vector_scaling, beta_vector_scaling
//
// dtype:         FP32 / FP16 / BF16 / INT8
// output_dtype:  INT8 / INT32 / -   (INT8场景区分输出类型; 非INT8填-)
// matrix_type:   sparse×dense / dense×dense
// prune_alg:     STRIP / TILE / -   (dense×dense填-)
// alpha_vector_scaling: 0=scalar, 1=per-row device pointer
// beta_vector_scaling:  0=scalar, 1=per-row device pointer
// =============================================================================

struct MatmulParam : public SparseTestParamBase {
    int32_t case_id = 0;
    int32_t m = 0;
    int32_t k = 0;
    int32_t n = 0;
    std::string dtype;          // FP32 / FP16 / BF16 / INT8
    std::string output_dtype;   // INT8 / INT32 / - (INT8场景)
    std::string matrix_type;    // sparse×dense / dense×dense

    float alpha = 1.0f;
    float beta = 0.0f;
    std::string transA;         // "true" / "false"
    std::string transB;         // "true" / "false"
    int32_t algConfigId = 0;
    int32_t splitK = 1;
    int32_t splitKMode = 0;   // 0=ONE_KERNEL, 1=TWO_KERNELS
    std::string prune_alg;      // STRIP / TILE / -
    int32_t alpha_vector_scaling = 0;
    int32_t beta_vector_scaling = 0;

    float range_low = -1.0f;
    float range_high = 1.0f;
    std::string level;
    std::string note;
    std::string order;
    std::string sparse_side;    // A / B (default A)

    // ----- Epilogue fields (backward-compatible, defaults preserve v1 behavior) -----
    int32_t biasEnabled = 0;               // 0=off, 1=on (per-row broadcast bias)
    int64_t biasStride = 0;                // batch bias stride (elems), 0=shared across batches
    int32_t activationType = 0;            // 0=none, 1=ReLU, 2=GeLU, 3=GELU_SCALING only (implies GeLU, F1)
    float reluUpperBound = FLT_MAX;        // FLT_MAX
    float reluThreshold = 0.0f;
    float geluScaling = 1.0f;
    int32_t numBatches = 1;                // >=1; 1=single batch
    int64_t batchStride = 0;               // matrix batch stride (elems), 0=single batch

    // ----- ld/order 布局列（默认 0/row = 既有紧凑行为）-----
    int64_t ldSparse = 0;                  // structured 侧 ld（A-sparse 的 A / B-sparse 的 B）
    int64_t ldDense = 0;                   // dense 侧 ld（A-sparse 的 B / B-sparse 的 A）
    std::string denseOrder;                // dense 侧 order: row/col（空 = row）
    std::string cdOrder;                   // C/D order: row/col（空 = row）
    int64_t ldCd = 0;                      // C/D ld（ROW ≥ n / COL ≥ m；0 = 紧凑）

    // Convenience accessors
    bool isFp32() const { return dtype == "FP32"; }
    bool isFp16() const { return dtype == "FP16"; }
    bool isBf16() const { return dtype == "BF16"; }
    bool isInt8() const { return dtype == "INT8"; }
    bool isInt32Output() const { return output_dtype == "INT32"; }
    bool isInt8Output() const { return output_dtype == "INT8"; }
    bool isDensePath() const { return matrix_type == "dense×dense"; }
    bool isSparsePath() const { return matrix_type == "sparse×dense" || matrix_type.empty(); }
    bool isTransA() const { return transA == "true"; }
    bool isTransB() const { return transB == "true"; }
    bool isColOrder() const { return order == "col"; }
    bool isSparseA() const { return sparse_side != "B"; }
    bool isTilePrune() const { return prune_alg == "TILE"; }
    bool isStripPrune() const { return prune_alg == "STRIP" || prune_alg.empty(); }
    bool isTwoKernels() const { return splitKMode == 1; }
    // Epilogue accessors
    bool hasBias() const { return biasEnabled == 1; }
    bool hasActivation() const { return activationType != 0; }
    bool isRelu() const { return activationType == 1; }
    bool isGelu() const { return activationType == 2; }
    bool hasEpilogue() const { return hasBias() || hasActivation(); }
    bool hasBatch() const { return numBatches > 1; }
    bool biasStrideShared() const { return biasStride == 0; }
    // 布局便捷访问（ltmatmul_npu_chain.h MatmulLayoutOpts 由 test.cpp 组装）
    bool isDenseColOrder() const { return denseOrder == "col"; }
    bool isCdColOrder() const { return cdOrder == "col"; }
    bool hasCdLayout() const { return ldCd > 0 || isCdColOrder(); }
    bool hasLayoutOverride() const
    {
        return ldSparse > 0 || ldDense > 0 || ldCd > 0 ||
               isDenseColOrder() || isCdColOrder();
    }

    // fillEpilogueFields: parse epilogue CSV columns.
    void fillEpilogueFields(const csv_map& row) {
        biasEnabled        = parseInt(row, "bias_enabled");
        biasStride         = static_cast<int64_t>(parseInt(row, "bias_stride"));
        activationType     = parseInt(row, "activation_type");
        reluUpperBound     = parseFloat(row, "relu_upper_bound");
        reluThreshold      = parseFloat(row, "relu_threshold");
        geluScaling        = parseFloat(row, "gelu_scaling");
        numBatches         = parseInt(row, "num_batches");
        batchStride        = static_cast<int64_t>(parseInt(row, "batch_stride"));
    }

    // applyFieldDefaults: set defaults for absent/empty fields.
    void applyFieldDefaults() {
        if (matrix_type.empty()) { matrix_type = "sparse×dense"; }
        if (transA.empty()) { transA = "false"; }
        if (transB.empty()) { transB = "false"; }
        if (order.empty()) { order = "row"; }
        if (sparse_side.empty()) { sparse_side = "A"; }
        if (prune_alg.empty() && isSparsePath()) { prune_alg = "STRIP"; }
        if (prune_alg.empty()) { prune_alg = "-"; }
        if (output_dtype.empty()) { output_dtype = "-"; }
        if (range_low == 0.0f && range_high == 0.0f) {
            range_low = -1.0f;
            range_high = 1.0f;
        }
        if (numBatches == 0) { numBatches = 1; }
        if (reluUpperBound == 0.0f && activationType != 1) {
            reluUpperBound = FLT_MAX;
        }
        if (geluScaling == 0.0f && activationType != 2 && activationType != 3) {
            geluScaling = 1.0f;
        }
    }

    void fillCustom(const csv_map& row) override {
        case_id         = parseInt(row, "case_id");
        m               = parseInt(row, "m");
        k               = parseInt(row, "k");
        n               = parseInt(row, "n");
        dtype           = parseString(row, "dtype");
        output_dtype    = parseString(row, "output_dtype");
        matrix_type     = parseString(row, "matrix_type");
        alpha           = parseFloat(row, "alpha");
        beta            = parseFloat(row, "beta");
        transA          = parseString(row, "transA");
        transB          = parseString(row, "transB");
        algConfigId     = parseInt(row, "alg_config_id");
        splitK           = parseInt(row, "split_k");
        splitKMode       = parseInt(row, "split_k_mode");
        prune_alg       = parseString(row, "prune_alg");
        alpha_vector_scaling = parseInt(row, "alpha_vector_scaling");
        beta_vector_scaling  = parseInt(row, "beta_vector_scaling");
        range_low       = parseFloat(row, "range_low");
        range_high      = parseFloat(row, "range_high");
        level           = parseString(row, "level");
        note            = parseString(row, "note");
        order           = parseString(row, "order");
        sparse_side     = parseString(row, "sparse_side");
        fillEpilogueFields(row);
        // 布局列（缺列/0/空 = 既有紧凑默认，向后兼容）
        ldSparse        = static_cast<int64_t>(parseInt(row, "ld_sparse"));
        ldDense         = static_cast<int64_t>(parseInt(row, "ld_dense"));
        denseOrder      = parseString(row, "dense_order");
        cdOrder         = parseString(row, "cd_order");
        ldCd            = static_cast<int64_t>(parseInt(row, "ld_cd"));
        applyFieldDefaults();
    }

    std::string caseId() const override {
        std::ostringstream oss;
        oss << "case" << case_id << "_" << level;
        return oss.str();
    }
};

inline void PrintTo(const MatmulParam& p, std::ostream* os) {
    *os << "case" << p.case_id << "(" << p.dtype << " " << p.m << "x" << p.k
        << "x" << p.n << " " << p.matrix_type
        << " out=" << p.output_dtype
        << " a=" << p.alpha << " b=" << p.beta
        << " splitk" << p.splitK << " mode" << p.splitKMode
        << " " << (p.isTilePrune() ? "TILE" : "STRIP")
        << " tA=" << (p.isTransA() ? "T" : "N")
        << " tB=" << (p.isTransB() ? "T" : "N")
        << " side=" << (p.isSparseA() ? "A" : "B")
        << " alphaVec=" << p.alpha_vector_scaling
        << " betaVec=" << p.beta_vector_scaling
        << " bias=" << p.biasEnabled
        << " act=" << p.activationType
        << " batch=" << p.numBatches
        << ")";
}

}  // namespace sparse_test

#endif  // TEST_MATMUL_PARAM_H_

