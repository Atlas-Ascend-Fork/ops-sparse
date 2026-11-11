/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPMAT_GET_ATTRIBUTE_SPMAT_GET_ATTRIBUTE_PARAM_H_
#define TEST_SPMAT_GET_ATTRIBUTE_SPMAT_GET_ATTRIBUTE_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

/// SpMatGetAttribute 测试参数（由 spmat_get_attribute_test.csv 加载）。
/// entry 引用 wrapper 内的创建入口注册表（csr / bsr / sliced_ell；
/// const_csr / const_bsr 仅出现在语义用例的注册表，CSV 成功用例均为 mutable
/// 入口——SetAttribute 需要 mutable 句柄）。
/// idx_base 仅为入口构造参数，其回读验证归 test/spmat_get_indexbase（去重，不断言）。
struct SpMatGetAttributeParam : public SparseTestParamBase {
    std::string case_name;

    std::string entry;        // 创建入口名，见 wrapper kEntries 注册表
    int fill_mode = 0;        // 0=ACL_SPARSE_FILL_MODE_LOWER, 1=ACL_SPARSE_FILL_MODE_UPPER
    int diag_type = 0;        // 0=ACL_SPARSE_DIAG_TYPE_NON_UNIT, 1=ACL_SPARSE_DIAG_TYPE_UNIT
    int idx_base = 0;         // 入口构造参数（不断言）

    std::string expect_result;

    void fillCustom(const csv_map &row) override {
        case_name     = parseString(row, "case_name");
        entry         = parseString(row, "entry");
        fill_mode     = parseInt(row, "fill_mode");
        diag_type     = parseInt(row, "diag_type");
        idx_base      = parseInt(row, "idx_base");
        expect_result = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_SPMAT_GET_ATTRIBUTE_SPMAT_GET_ATTRIBUTE_PARAM_H_
