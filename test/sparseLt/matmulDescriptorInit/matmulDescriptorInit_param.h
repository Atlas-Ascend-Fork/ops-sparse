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

/**
 * @file matmulDescriptorInit_param.h
 * @brief CSV 行 -> 参数结构体，含 dtype/order/op/sparsity/computeType/status 字符串枚举映射。
 *
 * 依据：CSV 列结构、类型定义与校验逻辑。
 * 本接口为 Host 侧描述符初始化，无 golden、无精度比对，CSV 承载正常/约束违规/数据类型覆盖用例。
 */

#ifndef TEST_SPARSELT_MATMULDESCRIPTORINIT_PARAM_H_
#define TEST_SPARSELT_MATMULDESCRIPTORINIT_PARAM_H_

#include "cann_ops_sparseLt.h"

#include "csv_loader.h"

#include <cstdint>
#include <string>

namespace sparse_test
{

// ============================================================================
// 枚举字符串映射（依据类型定义与 dtype 核实）
// ============================================================================

// 非法枚举哨兵值：触发接口返回 NOT_SUPPORTED / INVALID_VALUE
constexpr int32_t INVALID_ENUM_SENTINEL = 999;

// valueType: aclDataType 核实结论
inline aclDataType ParseDtype(const std::string& s)
{
    if (s == "FP32")
        return ACL_FLOAT;
    if (s == "FP16")
        return ACL_FLOAT16;
    if (s == "BF16")
        return ACL_BF16;
    if (s == "INT8")
        return ACL_INT8;
    if (s == "FP8_E4M3")
        return ACL_FLOAT8_E4M3FN;
    if (s == "FP8_E5M2")
        return ACL_FLOAT8_E5M2; // 无 FN 后缀
    if (s == "FP4_E2M1")
        return ACL_FLOAT4_E2M1;
    if (s == "HIFLOAT8")
        return ACL_HIFLOAT8;
    if (s == "FP8_E8M0")
        return ACL_FLOAT8_E8M0;
    if (s == "FP4_E1M2")
        return ACL_FLOAT4_E1M2;
    if (s.empty())
        return ACL_DT_UNDEFINED;
    return static_cast<aclDataType>(INVALID_ENUM_SENTINEL);
}

inline aclsparseOrder_t ParseOrder(const std::string& s)
{
    if (s == "COL")
        return ACL_SPARSE_ORDER_COL;
    if (s == "ROW")
        return ACL_SPARSE_ORDER_ROW;
    if (s.empty())
        return ACL_SPARSE_ORDER_ROW;
    return static_cast<aclsparseOrder_t>(INVALID_ENUM_SENTINEL);
}

inline aclsparseOperation_t ParseOp(const std::string& s)
{
    if (s == "N")
        return ACL_SPARSE_OP_NON_TRANSPOSE;
    if (s == "T")
        return ACL_SPARSE_OP_TRANSPOSE;
    if (s == "CONJ")
        return ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    if (s.empty())
        return ACL_SPARSE_OP_NON_TRANSPOSE;
    return static_cast<aclsparseOperation_t>(INVALID_ENUM_SENTINEL);
}

inline aclsparseLtSparsity_t ParseSparsity(const std::string& s)
{
    if (s == "50_PERCENT")
        return ACL_SPARSE_LT_SPARSITY_50_PERCENT;
    if (s.empty())
        return ACL_SPARSE_LT_SPARSITY_50_PERCENT;
    return static_cast<aclsparseLtSparsity_t>(INVALID_ENUM_SENTINEL);
}

inline aclsparseComputeType_t ParseComputeType(const std::string& s)
{
    if (s == "16F")
        return ACL_SPARSE_COMPUTE_16F;
    if (s == "32F")
        return ACL_SPARSE_COMPUTE_32F;
    if (s == "32I")
        return ACL_SPARSE_COMPUTE_32I;
    if (s.empty())
        return ACL_SPARSE_COMPUTE_16F;
    return static_cast<aclsparseComputeType_t>(INVALID_ENUM_SENTINEL);
}

inline int64_t ParseStatus(const std::string& s)
{
    if (s == "SUCCESS")
        return static_cast<int64_t>(ACL_SPARSE_STATUS_SUCCESS);
    if (s == "INVALID_VALUE")
        return static_cast<int64_t>(ACL_SPARSE_STATUS_INVALID_VALUE);
    if (s == "NOT_SUPPORTED")
        return static_cast<int64_t>(ACL_SPARSE_STATUS_NOT_SUPPORTED);
    if (s == "HANDLE_IS_NULLPTR")
        return static_cast<int64_t>(ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    if (s == "ALLOC_FAILED")
        return static_cast<int64_t>(ACL_SPARSE_STATUS_ALLOC_FAILED);
    return static_cast<int64_t>(ACL_SPARSE_STATUS_INTERNAL_ERROR);
}

inline aclsparseStatus_t ExpectedStatus(int64_t v)
{
    return static_cast<aclsparseStatus_t>(v);
}

// ============================================================================
// 单个矩阵描述符参数（a/b/c/d 共用）
// ============================================================================

struct MatParam
{
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t ld = 0;
    uint32_t align = 0;
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclsparseOrder_t order = ACL_SPARSE_ORDER_ROW;
    aclsparseLtSparsity_t sparsity = ACL_SPARSE_LT_SPARSITY_50_PERCENT;
    bool isStructured = false; // 由 sparse 列是否有值决定
    bool valid = false;        // 该 mat 是否参与构造（rows/cols 是否非零且有 dtype）

    // 从 CSV 行按前缀（a_/b_/c_/d_）解析；rows/cols/ld 用严格版（区分
    // 「未指定=0」与「非法输入=INT64_MIN」，支撑越界/非法负向用例）
    void fill(const csv_map& row, const std::string& prefix)
    {
        rows = parseStrictInt64(row, prefix + "rows");
        cols = parseStrictInt64(row, prefix + "cols");
        ld = parseStrictInt64(row, prefix + "ld");
        align = static_cast<uint32_t>(parseInt(row, prefix + "align"));
        dtype = ParseDtype(parseString(row, prefix + "dtype"));
        order = ParseOrder(parseString(row, prefix + "order"));
        std::string sp = parseString(row, prefix + "sparse");
        if (!sp.empty())
        {
            isStructured = true;
            sparsity = ParseSparsity(sp);
        }
        // mat 参数有效当且仅当 dtype 列非空（dense/structured/matmul 用例均如此）
        valid = !parseString(row, prefix + "dtype").empty();
    }
};

// ============================================================================
// 测试参数结构体（一行 CSV -> 一个用例）
// ============================================================================

struct MatmulDescriptorInitParam : public SparseTestParamBase
{
    std::string case_name;
    std::string level;        // L0/L1/L2
    std::string target;       // dense / structured / matmul
    int64_t expectStatus = 0; // ParseStatus 的返回值
    int verifyFields = 0;     // 0/1

    MatParam a;
    MatParam b;
    MatParam c;
    MatParam d;

    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE;
    aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE;
    aclsparseComputeType_t computeType = ACL_SPARSE_COMPUTE_16F;

    void fillCustom(const csv_map& row) override
    {
        case_name = parseString(row, "case_name");
        level = parseString(row, "level");
        target = parseString(row, "target");
        expectStatus = ParseStatus(parseString(row, "expect_status"));
        verifyFields = parseInt(row, "verify_fields");

        a.fill(row, "a_");
        b.fill(row, "b_");
        c.fill(row, "c_");
        d.fill(row, "d_");

        opA = ParseOp(parseString(row, "op_a"));
        opB = ParseOp(parseString(row, "op_b"));
        computeType = ParseComputeType(parseString(row, "compute_type"));
    }

    std::string caseId() const override { return case_name; }
};

inline void PrintTo(const MatmulDescriptorInitParam& p, std::ostream* os)
{
    *os << p.case_name;
}

} // namespace sparse_test

#endif // TEST_SPARSELT_MATMULDESCRIPTORINIT_PARAM_H_
