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

#ifndef CGTSV2_KERNEL_H_
#define CGTSV2_KERNEL_H_
#include <cstdint>
#include "cann_ops_sparse.h"
#include "cgtsv2_workspace.h"

struct Cgtsv2TilingData {
    int32_t m;
    int32_t n;
    int32_t ldb;
    uint32_t blocks;
    uint32_t slots;
    uint32_t cores;
};

aclsparseStatus_t cgtsv2_kernel_do(uint8_t* dl, uint8_t* d, uint8_t* du, uint8_t* b,
    uint8_t* workspace, const Cgtsv2TilingData& tiling, void* stream);
#endif
