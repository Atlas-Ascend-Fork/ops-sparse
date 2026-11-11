/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_SPARSE_DTYPE_UTILS_H_
#define TEST_FRAME_SPARSE_DTYPE_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "acl/acl.h"

namespace sparse_test {

// ============================================================================
// CSV dtype 列名 -> aclDataType 枚举映射与单元素宽度计算，供 BSR 族测试
// wrapper（create_bsr / bsr_set_strided_batch 等）共享；未识别列值抛
// std::runtime_error（CG-1.1 外部数据合法性检查，防拼写错误静默降级）。
// ============================================================================

/// CSV dtype 列名 -> aclDataType（float32/float16/bfloat16/complex64）。
/// 未识别列值抛 std::runtime_error（含列值原文），对齐 GetCasesFromCsvChecked
/// 的 runtime_error 口径；首个消费用例装载/执行期即失败，不再静默回退。
inline aclDataType ParseValueType(const std::string &dtype)
{
    if (dtype == "float32") {
        return ACL_FLOAT;
    }
    if (dtype == "float16") {
        return ACL_FLOAT16;
    }
    if (dtype == "bfloat16") {
        return ACL_BF16;
    }
    if (dtype == "complex64") {
        return ACL_COMPLEX64;
    }
    throw std::runtime_error("ParseValueType: unsupported CSV dtype column value '" + dtype +
                             "' (expected float32/float16/bfloat16/complex64)");
}

/// aclDataType 单元素宽度（字节），默认 float 按 4 字节回退
inline size_t ElementBytes(aclDataType valueType)
{
    switch (valueType) {
        case ACL_FLOAT16:
        case ACL_BF16:
            return sizeof(uint16_t);
        case ACL_COMPLEX64:
            return sizeof(float) * 2u;
        default:
            return sizeof(float);
    }
}

}  // namespace sparse_test

#endif  // TEST_FRAME_SPARSE_DTYPE_UTILS_H_
