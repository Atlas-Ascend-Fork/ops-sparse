/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file prune_check_kernel.h
 * \brief PruneCheck kernel launcher 声明（ASC 侧定义，CXX 侧 include 调用）。
 *
 * host.cpp (CXX) include 本头文件并以普通 C 函数方式调用 launcher；
 * kernel.cpp (ASC) 定义 launcher 并包装 <<<>>> 调用。host.cpp 禁止直接写 <<<>>>。
 */

#ifndef SPARSELT_PRUNE_CHECK_KERNEL_H_
#define SPARSELT_PRUNE_CHECK_KERNEL_H_

#include <cstdint>

#include "prune_check_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

// 校验 2:4 结构化稀疏条件：扫描 input，任一分组非零数超限时向 dValid 写 1。
// tiling 以 const 引用接收、by-value 转发进 <<<>>> args block
// （ACL runtime 托管 args block 生命周期，无需 aclrtMalloc tilingGm，也无需
// aclrtSynchronizeStream —— 对齐 cuSPARSELt PruneCheck 的
// "requires no extra storage" / "supports asynchronous execution" 属性）。
extern "C" void splt_prune_check_kernel_do(GM_ADDR input, GM_ADDR dValid,
    uint32_t blockDim, const PruneCheckTilingData& tiling, void* stream);

#endif // SPARSELT_PRUNE_CHECK_KERNEL_H_
