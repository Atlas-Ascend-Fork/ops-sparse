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
 * \file sddmm_csr_cube_kernel.cpp
 * \brief AIC/AIV launch entries for CSR SDDMM dense-band production and gather.
 */

#if defined(SDDMM_ENABLE_CSR_CUBE)

#include "sddmm_csr_cube_kernel.h"

namespace {

template <typename T>
__aicore__ inline void SddmmCsrCubeRun(
    GM_ADDR matX,
    GM_ADDR matY,
    GM_ADDR denseWorkspace,
    const SddmmTilingData &tiling,
    const sddmm_arch35::SddmmCsrCubePlan &plan)
{
    sddmm_arch35::SddmmCsrCubeBandKernel<T> kernel;
    kernel.Init(matX, matY, denseWorkspace, &tiling, &plan);
    kernel.Process();
}

template <typename T>
__aicore__ inline void SddmmCsrCubeGatherRun(
    GM_ADDR sparseValues,
    GM_ADDR denseWorkspace,
    const SddmmTilingData &tiling,
    const sddmm_arch35::SddmmCsrCubePlan &plan)
{
    AscendC::TPipe pipe;
    sddmm_arch35::SddmmCsrCubeGatherKernel<T> kernel;
    kernel.Init(sparseValues, denseWorkspace, &tiling, &plan, &pipe);
    kernel.Process();
}

} // namespace

extern "C" __global__ __aicore__ void sddmm_csr_cube_kernel_fp16(
    GM_ADDR matX,
    GM_ADDR matY,
    GM_ADDR denseWorkspace,
    const SddmmTilingData tiling,
    const sddmm_arch35::SddmmCsrCubePlan plan)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SddmmCsrCubeRun<half>(matX, matY, denseWorkspace, tiling, plan);
}

extern "C" __global__ __aicore__ void sddmm_csr_cube_kernel_bf16(
    GM_ADDR matX,
    GM_ADDR matY,
    GM_ADDR denseWorkspace,
    const SddmmTilingData tiling,
    const sddmm_arch35::SddmmCsrCubePlan plan)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SddmmCsrCubeRun<bfloat16_t>(matX, matY, denseWorkspace, tiling, plan);
}

extern "C" __global__ __aicore__ void sddmm_csr_cube_kernel_fp32(
    GM_ADDR matX,
    GM_ADDR matY,
    GM_ADDR denseWorkspace,
    const SddmmTilingData tiling,
    const sddmm_arch35::SddmmCsrCubePlan plan)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SddmmCsrCubeRun<float>(matX, matY, denseWorkspace, tiling, plan);
}

extern "C" __global__ __aicore__ void sddmm_csr_cube_gather_kernel_fp16(
    GM_ADDR sparseValues,
    GM_ADDR denseWorkspace,
    const SddmmTilingData tiling,
    const sddmm_arch35::SddmmCsrCubePlan plan)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SddmmCsrCubeGatherRun<half>(
        sparseValues, denseWorkspace, tiling, plan);
}

extern "C" __global__ __aicore__ void sddmm_csr_cube_gather_kernel_bf16(
    GM_ADDR sparseValues,
    GM_ADDR denseWorkspace,
    const SddmmTilingData tiling,
    const sddmm_arch35::SddmmCsrCubePlan plan)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SddmmCsrCubeGatherRun<bfloat16_t>(
        sparseValues, denseWorkspace, tiling, plan);
}

extern "C" __global__ __aicore__ void sddmm_csr_cube_gather_kernel_fp32(
    GM_ADDR sparseValues,
    GM_ADDR denseWorkspace,
    const SddmmTilingData tiling,
    const sddmm_arch35::SddmmCsrCubePlan plan)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SddmmCsrCubeGatherRun<float>(
        sparseValues, denseWorkspace, tiling, plan);
}

extern "C" void sddmm_csr_cube_kernel_do(
    GM_ADDR matX,
    GM_ADDR matY,
    GM_ADDR denseWorkspace,
    const SddmmTilingData &tiling,
    const sddmm_arch35::SddmmCsrCubePlan &plan,
    uint32_t numBlocks,
    void *stream)
{
    switch (tiling.inputType) {
        case SDDMM_VALUE_FP16:
            sddmm_csr_cube_kernel_fp16<<<numBlocks, nullptr, stream>>>(
                matX, matY, denseWorkspace, tiling, plan);
            break;
        case SDDMM_VALUE_BF16:
            sddmm_csr_cube_kernel_bf16<<<numBlocks, nullptr, stream>>>(
                matX, matY, denseWorkspace, tiling, plan);
            break;
        case SDDMM_VALUE_FP32:
            sddmm_csr_cube_kernel_fp32<<<numBlocks, nullptr, stream>>>(
                matX, matY, denseWorkspace, tiling, plan);
            break;
        default:
            break;
    }
}

extern "C" void sddmm_csr_cube_gather_kernel_do(
    GM_ADDR sparseValues,
    GM_ADDR denseWorkspace,
    const SddmmTilingData &tiling,
    const sddmm_arch35::SddmmCsrCubePlan &plan,
    uint32_t numBlocks,
    void *stream)
{
    switch (tiling.outputType) {
        case SDDMM_VALUE_FP16:
            sddmm_csr_cube_gather_kernel_fp16<<<numBlocks, nullptr, stream>>>(
                sparseValues, denseWorkspace, tiling, plan);
            break;
        case SDDMM_VALUE_BF16:
            sddmm_csr_cube_gather_kernel_bf16<<<numBlocks, nullptr, stream>>>(
                sparseValues, denseWorkspace, tiling, plan);
            break;
        case SDDMM_VALUE_FP32:
            sddmm_csr_cube_gather_kernel_fp32<<<numBlocks, nullptr, stream>>>(
                sparseValues, denseWorkspace, tiling, plan);
            break;
        default:
            break;
    }
}

#endif // SDDMM_ENABLE_CSR_CUBE
