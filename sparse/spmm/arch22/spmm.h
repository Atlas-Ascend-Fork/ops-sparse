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

#ifndef SPMM_ARCH22_H
#define SPMM_ARCH22_H

#include <stdint.h>

#ifndef __gm__
#define __gm__
#endif

/* CXX(host) TU 下 GM_ADDR 未定义，此处补齐，使 launch 声明可统一使用 GM_ADDR。
 * ASC(kernel) TU 下编译器已定义 GM_ADDR，#ifndef 跳过重定义。
 * 这样 <<<>>> 调用处不再需要 (GM_ADDR) 强转，消除
 * "cce_global attribute ignored when parsing type" 告警。 */
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

#define SPMM_ARCH22_MAX_BLOCK_DIM  24
#define SPMM_ARCH22_BUFFER_NUM     2
#define SPMM_ARCH22_UB_SIZE        (192 * 1024)
#define SPMM_ARCH22_UB_OVERHEAD    (8 * 1024)
#define SPMM_ARCH22_ALIGN          8

#define ROUNDUP(a, b) (((a) + (b) - 1) / (b) * (b))

/* tiling 数据结构，host 端填充后拷贝到 device 供 kernel 读取 */
typedef struct {
     uint32_t M;
     uint32_t K;
     uint32_t N;
     uint32_t nnz;
     uint32_t blockDim;
     uint32_t nChunks;
     float    alphaHost;
     float    betaHost;
     int64_t  reorderOffset;
     int64_t  binEdgeOffset;
} SpmmArch22TilingData;

static inline uint64_t SpmmArch22WorkspaceSize(uint32_t M, uint32_t blockDim)
{
    uint64_t headerSize = 64;
    uint64_t tilingSize = ROUNDUP(sizeof(SpmmArch22TilingData), 64);
    uint64_t reorderSize = ROUNDUP((uint64_t)M * sizeof(int32_t), 64);
    uint64_t binEdgeSize = ROUNDUP(((uint64_t)blockDim + 1) * sizeof(int32_t), 64);
    return headerSize + tilingSize + reorderSize + binEdgeSize;
}

static inline uint32_t SpmmArch22ComputeNChunks(uint32_t N)
{
    uint64_t availUB = SPMM_ARCH22_UB_SIZE - SPMM_ARCH22_UB_OVERHEAD;
    uint64_t buffersPerChunk = 4;
    uint64_t nMax = availUB / (buffersPerChunk * sizeof(float));
    nMax = (nMax / SPMM_ARCH22_ALIGN) * SPMM_ARCH22_ALIGN;
    if (nMax == 0) {
        nMax = SPMM_ARCH22_ALIGN;
    }
    return (uint32_t)((N + nMax - 1) / nMax);
}

/* host 侧 kernel launch 入口，定义于 spmm_kernel.cpp，供 spmm_host.cpp 调用 */
void spmm_arch22_kernel_launch(
    GM_ADDR rowOff, GM_ADDR colIdx, GM_ADDR values,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspace, GM_ADDR tiling,
    uint32_t blockDim, void *stream);

#endif
