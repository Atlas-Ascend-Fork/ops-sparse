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

/*!
 * \file scatter_kernel.h
 * \brief aclsparseScatter kernel_do 签名声明（Host / Kernel 共用）。
 */

#ifndef SCATTER_KERNEL_H_
#define SCATTER_KERNEL_H_

#include <cstdint>
#include "scatter_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

extern "C" {

void scatter_kernel_do(
    GM_ADDR indices,
    GM_ADDR values,
    GM_ADDR yVec,
    const ScatterTilingData &tiling,
    void *stream);

}  // extern "C"

#endif  // SCATTER_KERNEL_H_
