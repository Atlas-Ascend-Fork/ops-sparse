/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_COO_SET_STRIDED_BATCH_COO_SET_STRIDED_BATCH_PARAM_H_
#define TEST_COO_SET_STRIDED_BATCH_COO_SET_STRIDED_BATCH_PARAM_H_

#include "strided_batch_param.h"

namespace sparse_test {

/// CooSetStridedBatch 测试参数（由 coo_set_strided_batch_test.csv 加载）。
/// batch_stride 为 0 表示广播（所有 batch 共享同一组 row/col/values 数组）。
struct CooSetStridedBatchParam : public StridedBatchParamBase {
    int rows = 0;
    int cols = 0;
    int nnz = 0;

    void fillCustom(const csv_map& row) override
    {
        rows = parseInt(row, "rows");
        cols = parseInt(row, "cols");
        nnz = parseInt(row, "nnz");
        fillStridedBatchCommon(row);
    }
};

} // namespace sparse_test

#endif // TEST_COO_SET_STRIDED_BATCH_COO_SET_STRIDED_BATCH_PARAM_H_
