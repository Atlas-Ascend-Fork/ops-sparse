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

#ifndef CGTSV2_ARCH22_KERNEL_H_
#define CGTSV2_ARCH22_KERNEL_H_

#include <cstdint>

#include "cgtsv2_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void cgtsv2_factor_kernel_do(GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B, GM_ADDR workspace,
                             const Cgtsv2TilingData& tiling, void* stream);

void cgtsv2_solve_kernel_do(GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B, GM_ADDR workspace,
                            const Cgtsv2TilingData& tiling, void* stream);

void cgtsv2_single_rhs_kernel_do(GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B, GM_ADDR workspace,
                                 const Cgtsv2TilingData& tiling, void* stream);

#endif // CGTSV2_ARCH22_KERNEL_H_
