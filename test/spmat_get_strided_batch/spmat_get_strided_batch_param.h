/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SP_MAT_GET_STRIDED_BATCH_SP_MAT_GET_STRIDED_BATCH_PARAM_H_
#define TEST_SP_MAT_GET_STRIDED_BATCH_SP_MAT_GET_STRIDED_BATCH_PARAM_H_

#include "strided_batch_param.h"
#include <string>

namespace sparse_test {

/// SpMatGetStridedBatch 测试参数（由 spmat_get_strided_batch_test.csv 加载）。
/// format 为 coo / csr / csc；pre_set=1 时先经 aclsparseCooSetStridedBatch
/// 写入 batchCount / batchStride（仅 coo 格式支持；batch_stride=0 为广播），再 Get 读回。
struct SpMatGetStridedBatchParam : public StridedBatchParamBase {
    std::string format; // "coo" / "csr" / "csc"
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int pre_set = 0; // 0=仅读默认值；1=先 Set 再 Get

    void fillCustom(const csv_map& row) override
    {
        format = parseString(row, "format");
        rows = parseInt(row, "rows");
        cols = parseInt(row, "cols");
        nnz = parseInt(row, "nnz");
        pre_set = parseInt(row, "pre_set");
        fillStridedBatchCommon(row);
    }
};

} // namespace sparse_test

#endif // TEST_SP_MAT_GET_STRIDED_BATCH_SP_MAT_GET_STRIDED_BATCH_PARAM_H_
