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
#ifndef SPMV_STRIDE_KERNEL_H
#define SPMV_STRIDE_KERNEL_H

#include <cstdint>
#include "spmv_stride_plan.h"

// Host launchers use ordinary pointers; device entry points retain AscendC GM_ADDR qualifiers.
extern "C" void spmv_stride_pack_do(uint8_t *rows, uint8_t *columns, uint8_t *values,
    uint8_t *x, uint8_t *y, uint8_t *workspace, SpmvStridePlan plan, bool readY,
    uint32_t blocks, void *stream);
extern "C" void spmv_stride_scatter_do(uint8_t *workspace, uint8_t *y,
    SpmvStrideCopy copy, uint32_t blocks, void *stream);

#endif
