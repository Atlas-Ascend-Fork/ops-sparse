/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file scatter_kernel.h
 * \brief scatter kernel_do 启动器声明（Host / Kernel 共用）。
 */

#ifndef SCATTER_KERNEL_H
#define SCATTER_KERNEL_H

#include <stdint.h>
#include "scatter.h"

/* CXX(host) TU 下 GM_ADDR 未定义，此处补齐，使 launch 声明可统一使用 GM_ADDR。
 * ASC(kernel) TU 下编译器已定义 GM_ADDR，#ifndef 跳过重定义。
 * 这样 <<<>>> 调用处不再需要 (GM_ADDR) 强转，消除
 * "cce_global attribute ignored when parsing type" 告警。 */
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

// scatter kernel_do launcher declaration (Host / Kernel shared).
// Defined in scatter_kernel.cpp; linked with C++ linkage (no extern "C"),
// matching the definition. Host code includes this header instead of writing
// an extern forward declaration.
//
// valueType 由 Host 校验链保证 ∈ {ACL_INT8, ACL_FLOAT16, ACL_BF16,
// ACL_FLOAT, ACL_COMPLEX64}，kernel_do 按其分发 5 个模板实例化。
// Tiling 随 kernel 启动参数一起下发（const 引用传递），Host 侧不单独分配
// device 内存存 tiling。
void scatter_kernel_do(
    GM_ADDR valDev, GM_ADDR idxDev, GM_ADDR yDev, aclDataType valueType,
    const ScatterTilingData &tiling, uint32_t blockNum, void *stream);

#endif // SCATTER_KERNEL_H
