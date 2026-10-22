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

// 统一 host 分发器 —— 根据 valType 派发到各自的 __global__ kernel（各自定义在
// spmm_kernel_float.cpp / spmm_kernel_half.cpp / spmm_kernel_bf16.cpp /
// spmm_kernel_complex64.cpp 中）。模式与 sparse/spmv/arch22/kernels/spmv_kernel.cpp
// 的 spmv_kernel_do 一致。

#include "spmm_kernel_entries.h"
#include "../spmm.h"

void spmm_arch22_kernel_launch(
    void *rowOff, void *colIdx, void *values,
    void *matB, void *matC, void *workspace, void *tiling,
    uint32_t blockDim, void *stream, int32_t valType)
{
#define SPMM_LAUNCH(ValT)                                                 \
    spmm_arch22_##ValT<<<blockDim, nullptr, stream>>>(                    \
        (GM_ADDR)rowOff, (GM_ADDR)colIdx, (GM_ADDR)values,                \
        (GM_ADDR)matB, (GM_ADDR)matC,                                     \
        (GM_ADDR)workspace, (GM_ADDR)tiling)

    if (valType == SPMM_VAL_HALF) {
        SPMM_LAUNCH(half);
    } else if (valType == SPMM_VAL_BF16) {
        SPMM_LAUNCH(bfloat16_t);
    } else if (valType == SPMM_VAL_COMPLEX64) {
        SPMM_LAUNCH(complex64);
    } else {
        SPMM_LAUNCH(float);
    }
#undef SPMM_LAUNCH
}

void spmm_arch22_kernel_launch_out(
    void *rowOff, void *colIdx, void *values,
    void *matB, void *matC, void *matY, void *workspace, void *tiling,
    uint32_t blockDim, void *stream, int32_t valType)
{
#define SPMM_LAUNCH_OUT(ValT)                                             \
    spmm_arch22_##ValT##_out<<<blockDim, nullptr, stream>>>(              \
        (GM_ADDR)rowOff, (GM_ADDR)colIdx, (GM_ADDR)values,                \
        (GM_ADDR)matB, (GM_ADDR)matC, (GM_ADDR)matY,                      \
        (GM_ADDR)workspace, (GM_ADDR)tiling)

    if (valType == SPMM_VAL_HALF) {
        SPMM_LAUNCH_OUT(half);
    } else if (valType == SPMM_VAL_BF16) {
        SPMM_LAUNCH_OUT(bfloat16_t);
    } else if (valType == SPMM_VAL_COMPLEX64) {
        SPMM_LAUNCH_OUT(complex64);
    } else {
        SPMM_LAUNCH_OUT(float);
    }
#undef SPMM_LAUNCH_OUT
}
