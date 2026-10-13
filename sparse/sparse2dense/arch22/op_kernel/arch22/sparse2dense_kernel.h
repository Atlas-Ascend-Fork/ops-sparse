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
 * \file sparse2dense_kernel.h
 * \brief aclsparse_sparse2dense Kernel 入口契约（Host / Kernel 共用，纯 C++ 无设备依赖）。
 *
 * 双 kernel 协议（与 arch35「清零 + scatter」协议对齐）：
 *   清零 kernel sparse2dense_zero_kernel<V_KIND>       —— matB 全 buffer（含 ld padding）写对应 dtype 正零
 *   scatter kernel sparse2dense_scatter_kernel<V_KIND> —— 坐标 scatter values（重复坐标合法，取任一写入值）
 *
 * TilingKey（V_KIND，dtype 轴，TK0~TK4 与支持的 dtype 组合
 * 一一对应）：入口为模板函数，编译期 if constexpr 选择搬运通道实例；Host 侧以
 * ASCENDC_TPL_ARGS_DECL / GET_TPL_TILING_KEY（ASCENDC_TPL_SEL_PARAM 底层编码）计算
 * TilingKey 后分发对应实例（禁用废弃宏 TILING_KEY_IS / BEGIN_TILING_DATA_DEF）。
 *
 * scope：TK0~TK4 全 5 dtype 通道实现（int8 uint16 造零/SetValue 打包、fp16/bf16 2B
 * 通道、complex64 2×uint32）。
 */

#ifndef SPARSE2DENSE_KERNEL_H_
#define SPARSE2DENSE_KERNEL_H_

#include <cstdint>
#include "sparse2dense_tiling_data.h"

namespace s2d {

/// TilingKey（V_KIND）→ 元素字节宽度（1/2/4/8B；complex64 = 2×float32 相邻）
constexpr inline uint64_t ElemSizeOf(uint32_t valueKind)
{
    switch (valueKind) {
        case SPARSE2DENSE_VK_INT8:
            return 1UL;
        case SPARSE2DENSE_VK_FLOAT16:
        case SPARSE2DENSE_VK_BFLOAT16:
            return 2UL;
        case SPARSE2DENSE_VK_FLOAT32:
            return 4UL;
        case SPARSE2DENSE_VK_COMPLEX64:
            return 8UL;
        default:
            return 0UL;
    }
}

/// 清零 kernel 通道单元字节数（Duplicate 通道 dtype 宽度）：int8 走 uint16（一次覆盖 2×int8，
/// Duplicate 不含 int8）；float16/bfloat16 → half/bfloat16_t（2B）；float32 → float（4B）；
/// complex64 → uint32（4B，实/虚两分量均为 +0f bit 模式全零）
constexpr inline uint64_t ZeroChannelBytes(uint32_t valueKind)
{
    return (valueKind == SPARSE2DENSE_VK_INT8)       ? 2UL
         : (valueKind == SPARSE2DENSE_VK_COMPLEX64)  ? 4UL
         : ElemSizeOf(valueKind);
}

}  // namespace s2d

#endif  // SPARSE2DENSE_KERNEL_H_
