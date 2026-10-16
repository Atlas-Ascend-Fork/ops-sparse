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

// sddmm_simd_kernel.h pulls in kernel_operator.h, which defines GM_ADDR as
// `__gm__ uint8_t *`. sddmm_kernel.h must be included after it so the
// kernel_launch declaration and its definition below agree on GM_ADDR.
#include "sddmm_simd_kernel.h"
#include "sddmm_kernel.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2201)
#error "SDDMM arch22: this TU is only for dav-2201 / Ascend 910B (__NPU_ARCH__==2201)."
#endif

extern "C" __global__ __aicore__ void sddmm_kernel_f32(
    GM_ADDR matA, GM_ADDR matB,
    GM_ADDR csrRowOffsets, GM_ADDR csrColIndices, GM_ADDR csrValues,
    SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelSparseSddmm<false> op;
    op.Init(matA, matB, csrRowOffsets, csrColIndices, csrValues, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void sddmm_kernel_f32_strided(
    GM_ADDR matA, GM_ADDR matB,
    GM_ADDR csrRowOffsets, GM_ADDR csrColIndices, GM_ADDR csrValues,
    SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelSparseSddmm<true> op;
    op.Init(matA, matB, csrRowOffsets, csrColIndices, csrValues, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void sddmm_kernel_f16(
    GM_ADDR matA, GM_ADDR matB,
    GM_ADDR csrRowOffsets, GM_ADDR csrColIndices, GM_ADDR csrValues,
    SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelSparseSddmmFp16 op;
    op.Init(matA, matB, csrRowOffsets, csrColIndices, csrValues, tiling, &pipe);
    op.Process();
}

extern "C" aclsparseStatus_t sddmm_kernel_launch(
    GM_ADDR matA, GM_ADDR matB,
    GM_ADDR csrRowOffsets, GM_ADDR csrColIndices, GM_ADDR csrValues,
    const SddmmTilingData &tiling,
    uint32_t numBlocks, void *stream)
{
    if (tiling.dataType == 1u) {
        sddmm_kernel_f16<<<numBlocks, nullptr, stream>>>(
            matA, matB, csrRowOffsets, csrColIndices, csrValues, tiling);
    } else {
        const bool bContig = (tiling.orderY == 0u && tiling.opY == 1u) ||
                             (tiling.orderY == 1u && tiling.opY == 0u);
        if (bContig) {
            sddmm_kernel_f32<<<numBlocks, nullptr, stream>>>(
                matA, matB, csrRowOffsets, csrColIndices, csrValues, tiling);
        } else {
            sddmm_kernel_f32_strided<<<numBlocks, nullptr, stream>>>(
                matA, matB, csrRowOffsets, csrColIndices, csrValues, tiling);
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
