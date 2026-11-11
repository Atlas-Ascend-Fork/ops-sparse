/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSR_SET_POINTERS_CSR_SET_POINTERS_PARAM_H_
#define TEST_CSR_SET_POINTERS_CSR_SET_POINTERS_PARAM_H_

#include "csv_loader.h"
#include "spmat_csv_common.h"    // SparseMatCommonParamBase：公共参数基类（描述符测试私有共享）
#include <string>

namespace sparse_test {

/// CsrSetPointers 测试参数（由 csr_set_pointers_test.csv 加载）。
/// 公共列（case_name/几何 rows·cols·nnz/idx_type/idx_base/dtype/null_ptrs/
/// expect_result）继承自 SparseMatCommonParamBase，本结构无差异列。
/// null_ptrs：0=SetPointers 三指针全有效；1=indices+values 传 null（仅 nnz=0 行
/// 合法，offsets 恒有效）；2=offsets 传 null（仅零维行 rows=0 或 cols=0 合法，
/// indices/values 有效）。拒绝侧与合法补集成对覆盖（CSP-E04/E05 vs C05~C07）。
struct CsrSetPointersParam : public SparseMatCommonParamBase {};

}  // namespace sparse_test

#endif  // TEST_CSR_SET_POINTERS_CSR_SET_POINTERS_PARAM_H_
