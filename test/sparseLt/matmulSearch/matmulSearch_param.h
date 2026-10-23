/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SPARSELT_MATMULSEARCH_PARAM_H_
#define TEST_SPARSELT_MATMULSEARCH_PARAM_H_

// =============================================================================
// CSV-driven test parameter for aclsparseLtMatmulSearch (v1: 39 columns).
//
// 继承 ltmatmul 的 MatmulParam（31 列），新增 8 个搜索列：
//   search_iterations, check_timing, inplace, mat_a_null, stream_null,
//   alpha_null, beta_null, expect_status
//
// ★ 语义变更声明：
//   alg_config_id / split_k / split_k_mode 三列在【本套件】中语义为
//   「初始 algSelection 预置属性（PlanInit 前经 AlgSetAttribute 设置）」，
//   非固定执行配置（ltmatmul 中为固定配置）。搜索可在候选集内改写该配置，
//   搜索后经 AlgGetAttribute 回读选中值并做 Tier-1/Tier-2 合法性断言。
//
// ★ numStreams 说明：CSV v1 schema（39 列）未含 num_streams 列。
//   B5（case_id=134，numStreams=2 双流数组）按 case 约定，
//   由 fillCustom 中 case_id==134 特判置 num_streams=2，其余用例恒为 1。
//   numStreams<=0 的异常值由 L2 TEST_F（X06/X07）覆盖，不经 CSV。
// =============================================================================

#include "../ltmatmul/ltmatmul_param.h"

namespace sparse_test {

struct MatmulSearchParam : public MatmulParam {
    // ----- Search extension columns (v1: +8) -----
    int32_t search_iterations = 5;   // algSelection SEARCH_ITERATIONS 预置（PlanInit 前 Set）
    int32_t check_timing = 0;        // 1=计时观测用例（逐次交错+顺序平衡测量，ratio 为信息性观测）
    int32_t inplace = 0;             // 1=matC 与 matD 传同一 Device 缓冲（B2 声明性验证）
    int32_t mat_a_null = 0;          // 1=matA=nullptr（剪枝结果预写 ws+aPrunedOffset，B3 回退路径）
    int32_t stream_null = 0;         // 1=streams[0]=nullptr（默认流，B4）
    int32_t alpha_null = 0;          // 1=alpha 指针传 nullptr（语义 1.0，B7）
    int32_t beta_null = 0;           // 1=beta  指针传 nullptr（语义 0.0，B6）
    std::string expect_status = "SUCCESS";  // 预期返回码（v1 全 SUCCESS）

    // B5（case 134）numStreams=2 双流数组（CSV v1 无 num_streams 列，见文件头说明）
    int32_t num_streams = 1;

    // 预置为默认配置（cfg=0, splitK=1, mode=0，即 fused、无 temp 区）——
    // workspace 裁剪推论断言（Tier-2 推论：默认预置 fused-only 窗口）的触发条件。
    bool isDefaultPreset() const { return algConfigId == 0 && splitK == 1 && splitKMode == 0; }

    bool expectSuccess() const { return expect_status.empty() || expect_status == "SUCCESS"; }

    void fillCustom(const csv_map& row) override {
        MatmulParam::fillCustom(row);  // 基类 31 列（含 epilogue/batch 默认值处理）
        search_iterations = parseInt(row, "search_iterations");
        if (search_iterations <= 0) { search_iterations = 5; }  // 缺列/0 → 默认 5
        check_timing  = parseInt(row, "check_timing");
        inplace       = parseInt(row, "inplace");
        mat_a_null    = parseInt(row, "mat_a_null");
        stream_null   = parseInt(row, "stream_null");
        alpha_null    = parseInt(row, "alpha_null");
        beta_null     = parseInt(row, "beta_null");
        expect_status = parseString(row, "expect_status");
        if (expect_status.empty()) { expect_status = "SUCCESS"; }
        // B5（numStreams=2 仅用 streams[0]）：CSV v1 39 列无 num_streams
        // 列，按 case 约定特判（X06/X07 的 0/-1 异常值在 TEST_F 中覆盖）。
        num_streams = (case_id == 134) ? 2 : 1;
    }

    std::string caseId() const override {
        std::ostringstream oss;
        oss << "case" << case_id << "_" << level;
        return oss.str();
    }
};

inline void PrintTo(const MatmulSearchParam& p, std::ostream* os) {
    *os << "case" << p.case_id << "(" << p.dtype << " " << p.m << "x" << p.k
        << "x" << p.n << " " << p.matrix_type
        << " out=" << p.output_dtype
        << " a=" << p.alpha << " b=" << p.beta
        << " preset(cfg" << p.algConfigId << ",k" << p.splitK
        << ",m" << p.splitKMode << ")"
        << " iters=" << p.search_iterations
        << " timing=" << p.check_timing
        << " inplace=" << p.inplace
        << " matANull=" << p.mat_a_null
        << " streamNull=" << p.stream_null
        << " aNull=" << p.alpha_null << " bNull=" << p.beta_null
        << " numStreams=" << p.num_streams
        << ")";
}

}  // namespace sparse_test

#endif  // TEST_SPARSELT_MATMULSEARCH_PARAM_H_
