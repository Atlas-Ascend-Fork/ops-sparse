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
#pragma once

#include "kernel_operator.h"

// 四个 dtype 的原地与 C/Y 分离 __global__ kernel 入口。定义分别在 spmm_kernel_float.cpp /
// spmm_kernel_half.cpp / spmm_kernel_bf16.cpp（前三者由 spmm_kernel.h 的
// DEFINE_SPMM_KERNEL 宏展开）和 spmm_kernel_complex64.cpp 中，每个 TU 一个。
//
// 声明集中放在本头文件、由分发器 spmm_kernel_dispatch.cpp include，而不是在
// 分发器里就地写 extern 声明去引用外部函数（G.EXP.05-CPP）。函数声明本身即
// 外部链接，无需 extern 关键字。参数列表与 DEFINE_SPMM_KERNEL 展开出的定义
// 一致。原入口保持 7 参数；后缀为 _out 的适配层专用入口在 matC 后增加 matY。
__global__ __aicore__ void spmm_arch22_float(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);

__global__ __aicore__ void spmm_arch22_half(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);

__global__ __aicore__ void spmm_arch22_bfloat16_t(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);

__global__ __aicore__ void spmm_arch22_complex64(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);

__global__ __aicore__ void spmm_arch22_float_out(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);

__global__ __aicore__ void spmm_arch22_half_out(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);

__global__ __aicore__ void spmm_arch22_bfloat16_t_out(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);

__global__ __aicore__ void spmm_arch22_complex64_out(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
