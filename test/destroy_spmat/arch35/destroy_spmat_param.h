/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_DESTROY_SPMAT_DESTROY_SPMAT_PARAM_H_
#define TEST_DESTROY_SPMAT_DESTROY_SPMAT_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

/// DestroySpMat 测试参数（由 destroy_spmat_test.csv 加载）。
/// entry 引用 test cpp 内的创建入口注册表（11 个入口：coo / csr / csc / bsr /
/// blocked_ell / sliced_ell 六个 mutable + const_csr / const_coo / const_csc /
/// const_bsr / const_sliced_ell 五个 const 变体，blocked_ell 无 const 变体），
/// 构造参数沿用 spmat_get_indexbase kEntries 缺省。
struct DestroySpMatParam : public SparseTestParamBase {
    std::string case_name;

    std::string entry;        // 创建入口名，见 kEntries 注册表
    std::string expect_result;

    void fillCustom(const csv_map &row) override {
        case_name     = parseString(row, "case_name");
        entry         = parseString(row, "entry");
        expect_result = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_DESTROY_SPMAT_DESTROY_SPMAT_PARAM_H_
