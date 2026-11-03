/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSR_SET_STRIDED_BATCH_CSR_SET_STRIDED_BATCH_PARAM_H_
#define TEST_CSR_SET_STRIDED_BATCH_CSR_SET_STRIDED_BATCH_PARAM_H_

#include "strided_batch_param.h"

namespace sparse_test {

/// CsrSetStridedBatch 测试参数（由 csr_set_strided_batch_test.csv 加载）。
/// 两个步长取 0 均表示广播（所有 batch 共享对应 CSR 数组）；基类的
/// batch_stride 字段为 COO 单步长语义，在 CSR 语境下闲置（恒 0）。
struct CsrSetStridedBatchParam : public StridedBatchParamBase {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int64_t offsets_batch_stride = 0;
    int64_t columns_values_batch_stride = 0;

    void fillCustom(const csv_map &row) override
    {
        rows = parseInt(row, "rows");
        cols = parseInt(row, "cols");
        nnz = parseInt(row, "nnz");
        offsets_batch_stride = parseInt64(row, "offsets_batch_stride");
        columns_values_batch_stride = parseInt64(row, "columns_values_batch_stride");
        fillStridedBatchCommon(row);
    }
};

}  // namespace sparse_test

#endif  // TEST_CSR_SET_STRIDED_BATCH_CSR_SET_STRIDED_BATCH_PARAM_H_
