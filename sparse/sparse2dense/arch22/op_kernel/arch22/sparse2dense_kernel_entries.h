/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */


/*!
 * \file sparse2dense_kernel_entries.h
 * \brief aclsparse_sparse2dense 内核入口声明（Host / Kernel 共用契约头）。
 *
 * 清零与 scatter kernel 入口：定义在 op_kernel/arch22/sparse2dense_kernel.cpp 的
 * GENERATE_S2D_*_KERNEL 宏展开点；指纹内核入口：定义在
 * op_kernel/arch22/sparse2dense_kernel.cpp（非模板 __global__）。
 * Host 侧（sparse2dense_host.cpp）经 <<<>>>
 * 直调（头文件声明模式同 spmv_kernel.h 的 SPMV_EXTERN），kernel_operator.h
 * 提供 GM_ADDR 的 __gm__ 限定；tiling 按值传参（arch35 sparse2dense_kernel 同款）。
 * 定义 TU 同样包含本头，声明与定义单点对齐。
 */

#ifndef SPARSE2DENSE_KERNEL_ENTRIES_H_
#define SPARSE2DENSE_KERNEL_ENTRIES_H_

// tiling_data.h（含 template_argument.h）须先于 kernel_operator.h（AscendC::printf 冲突规避，见 kernel.cpp 注）
#include "sparse2dense_tiling_data.h"
#include "kernel_operator.h"

// ---- 清零内核（TilingKey TK0~TK4 五通道；matB 全 buffer 含 ld padding 写对应 dtype 正零）----
extern __global__ __aicore__ void sparse2dense_zero_kernel_int8(GM_ADDR denseGm, const Sparse2DenseTilingData tiling);
extern __global__ __aicore__ void sparse2dense_zero_kernel_fp16(GM_ADDR denseGm, const Sparse2DenseTilingData tiling);
extern __global__ __aicore__ void sparse2dense_zero_kernel_bf16(GM_ADDR denseGm, const Sparse2DenseTilingData tiling);
extern __global__ __aicore__ void sparse2dense_zero_kernel_fp32(GM_ADDR denseGm, const Sparse2DenseTilingData tiling);
extern __global__ __aicore__ void sparse2dense_zero_kernel_c64(GM_ADDR denseGm, const Sparse2DenseTilingData tiling);

// ---- scatter 内核（TilingKey TK0~TK4 五通道；坐标 scatter values（重复坐标合法，取任一写入值），纯 bit 搬运）----
extern __global__ __aicore__ void sparse2dense_scatter_kernel_int8(GM_ADDR valuesGm, GM_ADDR primaryGm,
    GM_ADDR secondaryGm, GM_ADDR denseGm, const Sparse2DenseTilingData tiling);
extern __global__ __aicore__ void sparse2dense_scatter_kernel_fp16(GM_ADDR valuesGm, GM_ADDR primaryGm,
    GM_ADDR secondaryGm, GM_ADDR denseGm, const Sparse2DenseTilingData tiling);
extern __global__ __aicore__ void sparse2dense_scatter_kernel_bf16(GM_ADDR valuesGm, GM_ADDR primaryGm,
    GM_ADDR secondaryGm, GM_ADDR denseGm, const Sparse2DenseTilingData tiling);
extern __global__ __aicore__ void sparse2dense_scatter_kernel_fp32(GM_ADDR valuesGm, GM_ADDR primaryGm,
    GM_ADDR secondaryGm, GM_ADDR denseGm, const Sparse2DenseTilingData tiling);
extern __global__ __aicore__ void sparse2dense_scatter_kernel_c64(GM_ADDR valuesGm, GM_ADDR primaryGm,
    GM_ADDR secondaryGm, GM_ADDR denseGm, const Sparse2DenseTilingData tiling);

// ---- 指纹内核（坐标校验记忆化内容守卫；params 按值随 <<<>>> 下发；发射状态非 0 由 host 侧回退全量坐标重校承接）----
extern __global__ __aicore__ void sparse2dense_f3_hash_kernel(GM_ADDR primaryGmAddr, GM_ADDR secondaryGmAddr,
    const S2DF3HashParams params, GM_ADDR outGmAddr);

#endif  // SPARSE2DENSE_KERNEL_ENTRIES_H_
