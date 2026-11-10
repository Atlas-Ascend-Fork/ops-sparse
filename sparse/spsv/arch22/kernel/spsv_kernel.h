/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_A2_KERNEL_H_
#define SPSV_A2_KERNEL_H_
#include "spsv_tiling_data.h"

void SpsvA2Analysis(void *ptr, void *idx, void *values, void *workspace,
    const SpsvA2Tiling &tiling, void *stream);
void SpsvA2Solve(void *x, void *y, void *workspace,
    const SpsvA2Tiling &tiling, void *stream);
void SpsvA2Update(void *values, void *workspace, int32_t diagonal,
    const SpsvA2Tiling &tiling, void *stream);
#endif
