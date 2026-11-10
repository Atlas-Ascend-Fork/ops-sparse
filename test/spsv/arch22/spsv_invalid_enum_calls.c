/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "spsv_invalid_enum_calls.h"

// Exercise invalid integers at the C ABI without an out-of-range C++ enum cast.
aclsparseStatus_t SpsvTestBufferSizeEnums(aclsparseHandle_t handle, int operation, const void* alpha,
                                          aclsparseSpMatDescr_t matrix, int valueType, int algorithm,
                                          aclsparseSpSVDescr_t state, size_t* bytes)
{
    return aclsparseSpSV_bufferSize(handle, (aclsparseOperation_t)operation, alpha, matrix, NULL, NULL,
                                    (aclDataType)valueType, (aclsparseSpSVAlg_t)algorithm, state, bytes);
}

aclsparseStatus_t SpsvTestUpdateEnum(aclsparseHandle_t handle, aclsparseSpSVDescr_t state, void* values, int updatePart)
{
    return aclsparseSpSV_updateMatrix(handle, state, values, (aclsparseSpSVUpdate_t)updatePart);
}
