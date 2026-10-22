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

// 实数版（spmm_kernel.h）与 complex64 版（spmm_kernel_complex64.cpp）是两个独立实例化
// 的 device kernel 类，二者的入口样板 Init()/Process() 与 InitGlobalBuffers 里绑定转置
// 工作区的分支完全一致。把这几段收敛成宏、两处各留一行调用，避免逐字重复（重复代码），
// 原 7 参数 Init 的行为保持不变；新增重载允许适配层把 bias 输入与输出分开绑定。
//
// 唯一差异：complex64 在非 generic 分支于 InitLocalBuffers() 之后还要建 J 算子换位/符号
// 表，通过 POST_INIT_LOCAL 形参注入（实数版传空）。

// Init() + Process() 入口。POST_INIT_LOCAL 为非 generic 分支里 InitLocalBuffers() 之后
// 追加的语句（实数版为空，complex64 传 InitJOperatorTables();）。
#define SPMM_ARCH22_DEFINE_KERNEL_ENTRY(POST_INIT_LOCAL)                                            \
public:                                                                                             \
    __aicore__ inline void Init(GM_ADDR rowOff, GM_ADDR colIdx, GM_ADDR values, GM_ADDR matB,       \
        GM_ADDR matC, GM_ADDR workspace, GM_ADDR tilingGm)                                          \
    {                                                                                               \
        Init(rowOff, colIdx, values, matB, matC, matC, workspace, tilingGm);                        \
    }                                                                                               \
    __aicore__ inline void Init(GM_ADDR rowOff, GM_ADDR colIdx, GM_ADDR values, GM_ADDR matB,       \
        GM_ADDR matC, GM_ADDR matY, GM_ADDR workspace, GM_ADDR tilingGm)                            \
    {                                                                                               \
        __gm__ SpmmArch22TilingData *td = reinterpret_cast<__gm__ SpmmArch22TilingData *>(tilingGm);\
        InitTiling(td);                                                                             \
        InitRowRange(td, workspace);                                                                \
        InitGlobalBuffers(td, rowOff, colIdx, values, matB, matC, matY, workspace);                 \
        if (genericPath) {                                                                          \
            InitGenericLocalBuffers();                                                              \
        } else {                                                                                    \
            InitLocalBuffers();                                                                     \
            POST_INIT_LOCAL                                                                         \
        }                                                                                           \
    }                                                                                               \
    __aicore__ inline void Process()                                                                \
    {                                                                                               \
        if (genericPath) {                                                                          \
            ProcessGeneric();                                                                       \
        } else {                                                                                    \
            ProcessChunks(this);                                                                    \
        }                                                                                           \
    }

// opA 为转置/共轭转置时，把预处理阶段写入 workspace 的转置行偏移/源行/置换表绑成
// GlobalTensor。real/complex64 两版逐字一致，收敛成宏在各自 InitGlobalBuffers 末尾调用。
#define SPMM_ARCH22_BIND_TRANSPOSE_GM(td, workspace)                                                \
    do {                                                                                            \
        if (opA != 0) {                                                                             \
            transposeRowOffsetsGm.SetGlobalBuffer(                                                  \
                reinterpret_cast<__gm__ int32_t *>((workspace) + (td)->transposeRowOffset), M + 1); \
            transposeSourceRowsGm.SetGlobalBuffer(                                                  \
                reinterpret_cast<__gm__ int32_t *>((workspace) + (td)->transposeSourceRowOffset),   \
                nnz);                                                                               \
            transposePermutationGm.SetGlobalBuffer(                                                 \
                reinterpret_cast<__gm__ int32_t *>((workspace) + (td)->transposePermutationOffset), \
                nnz);                                                                               \
        }                                                                                           \
    } while (0)
