/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPMAT_GET_INDEXBASE_SPMAT_GET_INDEXBASE_PARAM_H_
#define TEST_SPMAT_GET_INDEXBASE_SPMAT_GET_INDEXBASE_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

/// SpMatGetIndexBase 测试参数（由 spmat_get_indexbase_test.csv 加载）。
/// entry 引用测试 cpp 内的创建入口注册表（csr / const_csr / ... / const_sliced_ell）。
struct SpMatGetIndexBaseParam : public SparseTestParamBase {
    std::string case_name;

    std::string entry;        // 创建入口名，见 kEntries 注册表
    int idx_base = 0;         // 0=ZERO, 1=ONE

    std::string expect_result;

    void fillCustom(const csv_map &row) override {
        case_name     = parseString(row, "case_name");
        entry         = parseString(row, "entry");
        idx_base      = parseInt(row, "idx_base");
        expect_result = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_SPMAT_GET_INDEXBASE_SPMAT_GET_INDEXBASE_PARAM_H_
