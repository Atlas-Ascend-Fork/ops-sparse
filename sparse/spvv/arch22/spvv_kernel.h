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

#ifndef SPVV_KERNEL_H_
#define SPVV_KERNEL_H_

#include <cstdint>
#include <acl/acl_base.h>
#include "spvv_tiling_data.h"

/* CXX(host) TU 下 GM_ADDR 未定义，此处补齐，使 launch 声明可统一使用 GM_ADDR。
 * ASC(kernel) TU 下编译器已定义 GM_ADDR，#ifndef 跳过重定义。
 * 这样 <<<>>> 调用处不再需要 (GM_ADDR) 强转，消除
 * "cce_global attribute ignored when parsing type" 告警。 */
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

// Host-side launcher for the spvv kernel. Defined in spvv_kernel.cpp, called
// from spvv_host.cpp. Tiling is passed by const reference (no host-side copy);
// the <<<>>> launch copies it into kernel parameter space by value.
void spvv_kernel_do(GM_ADDR xIndices, GM_ADDR xValues, GM_ADDR y,
    GM_ADDR output, const SpvvTilingData &tiling, uint32_t blockNum,
    aclDataType valueType, void* stream);

#endif // SPVV_KERNEL_H_
