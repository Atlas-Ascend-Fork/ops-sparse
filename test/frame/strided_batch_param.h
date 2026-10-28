/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_STRIDED_BATCH_PARAM_H_
#define TEST_FRAME_STRIDED_BATCH_PARAM_H_

#include "csv_loader.h"
#include <cstdint>
#include <string>

namespace sparse_test {

/// StridedBatch 系列算子测试参数公共基类（coo_set_strided_batch /
/// spmat_get_strided_batch 及后续 csr/bsr 变体共用），由 CSV 用例表加载。
/// 派生类在 fillCustom 中解析自身专有列后调用 fillStridedBatchCommon
/// 解析本基类的公共列。
struct StridedBatchParamBase : public SparseTestParamBase {
    std::string case_name;
    int batch_count = 1;
    int64_t batch_stride = 0;
    int idx_base = 0;  // 0=ZERO, 1=ONE
    std::string dtype; // "float32" / "float16"
    uint32_t seed = 0;

    std::string expect_result;

    bool isFp16() const { return dtype == "float16" || dtype == "fp16"; }

    void fillStridedBatchCommon(const csv_map& row)
    {
        case_name = parseString(row, "case_name");
        batch_count = parseInt(row, "batch_count");
        batch_stride = parseInt64(row, "batch_stride");
        idx_base = parseInt(row, "idx_base");
        dtype = parseString(row, "dtype");
        seed = static_cast<uint32_t>(parseInt(row, "seed"));
        expect_result = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

} // namespace sparse_test

#endif // TEST_FRAME_STRIDED_BATCH_PARAM_H_
