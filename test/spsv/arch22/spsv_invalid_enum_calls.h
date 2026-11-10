/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_INVALID_ENUM_CALLS_H_
#define SPSV_INVALID_ENUM_CALLS_H_

#include "cann_ops_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

aclsparseStatus_t SpsvTestBufferSizeEnums(aclsparseHandle_t handle, int operation, const void* alpha,
                                          aclsparseSpMatDescr_t matrix, int valueType, int algorithm,
                                          aclsparseSpSVDescr_t state, size_t* bytes);
aclsparseStatus_t SpsvTestUpdateEnum(aclsparseHandle_t handle, aclsparseSpSVDescr_t state, void* values,
                                     int updatePart);

#ifdef __cplusplus
}
#endif
#endif
