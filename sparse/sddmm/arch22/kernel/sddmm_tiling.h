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
#ifndef SDDMM_KERNEL_TILING_H_
#define SDDMM_KERNEL_TILING_H_

#include "../sddmm_tiling_fields.h"

struct SddmmKernelTilingData : SddmmTilingFields {};

struct SddmmBsrKernelTiling {
    uint32_t m, n, k, bd, blocks, first, count, yPitch, group;
    int32_t base;
    bool colMajor;
    float ar, ai, br, bi;
};

#endif
