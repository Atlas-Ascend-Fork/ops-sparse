/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef SPMM_HOST_H
#define SPMM_HOST_H

#include <stdint.h>
#include <acl/acl_base_rt.h>

#include "../spmm_config.h"
#include "spmm_host_tiling.h"

static inline int32_t SpmmDataTypeFromAcl(aclDataType valueType) {
    if (valueType == ACL_FLOAT)       return SPMM_DTYPE_FP32;
    if (valueType == ACL_FLOAT16)     return SPMM_DTYPE_FP16;
    if (valueType == ACL_INT8)        return SPMM_DTYPE_INT8;
    return SPMM_DTYPE_INVALID;
}

static inline int32_t SpmmDataTypeComboFromAcl(
    aclDataType valueType, aclDataType outputType, aclDataType computeType)
{
    struct SpmmDataTypeCombo {
        aclDataType value;
        aclDataType output;
        aclDataType compute;
        int32_t encoded;
    };
    static const SpmmDataTypeCombo kSupportedCombos[] = {
        {ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, SPMM_DTYPE_FP32},
        {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, SPMM_DTYPE_FP16},
        {ACL_INT8, ACL_INT32, ACL_INT32, SPMM_DTYPE_INT8},
        {ACL_BF16, ACL_BF16, ACL_FLOAT, SPMM_DTYPE_BF16},
        {ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, SPMM_DTYPE_COMPLEX64},
        {ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, SPMM_DTYPE_FP16_FP32},
        {ACL_BF16, ACL_FLOAT, ACL_FLOAT, SPMM_DTYPE_BF16_FP32},
    };
    for (const auto &combo : kSupportedCombos) {
        if (valueType == combo.value && outputType == combo.output &&
            computeType == combo.compute) {
            return combo.encoded;
        }
    }
    return SPMM_DTYPE_INVALID;
}

template<int64_t Alignment>
static inline int64_t spmm_align_up(int64_t value)
{
    static_assert(Alignment > 0, "workspace alignment must be positive");
    return ((value + Alignment - 1) / Alignment) * Alignment;
}

#endif // SPMM_HOST_H
