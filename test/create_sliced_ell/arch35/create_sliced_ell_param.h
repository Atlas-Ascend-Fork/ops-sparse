/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CREATE_SLICED_ELL_CREATE_SLICED_ELL_PARAM_H_
#define TEST_CREATE_SLICED_ELL_CREATE_SLICED_ELL_PARAM_H_

#include "csv_loader.h"
#include "spmat_csv_common.h"    // SparseMatCommonParamBase：公共参数基类（描述符测试私有共享）
#include <string>

namespace sparse_test {

/// CreateSlicedEll 测试参数（由 create_sliced_ell_test.csv 加载）。
/// 公共列（case_name/几何 rows·cols·nnz/idx_type/idx_base/dtype/null_ptrs/
/// expect_result）继承自 SparseMatCommonParamBase；差异列：entry 与
/// slice_nnz/num_slices。entry ∈ {sliced_ell, const_sliced_ell}；null_ptrs=1 时
/// 三个数组指针全部传 nullptr（实现无指针校验，延迟绑定语义），仅用于
/// nnz=0 / 0 维行。
struct CreateSlicedEllParam : public SparseMatCommonParamBase {
    std::string entry;          // sliced_ell / const_sliced_ell
    int64_t slice_nnz = 0;
    int64_t num_slices = 0;

    void fillCustom(const csv_map &row) override {
        entry       = parseString(row, "entry");
        slice_nnz   = parseInt64(row, "slice_nnz");
        num_slices  = parseInt64(row, "num_slices");
        fillMatCommon(row);
    }
};

}  // namespace sparse_test

#endif  // TEST_CREATE_SLICED_ELL_CREATE_SLICED_ELL_PARAM_H_
