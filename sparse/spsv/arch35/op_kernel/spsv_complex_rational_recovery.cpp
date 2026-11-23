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

#include "recovery/spsv_complex_recovery_common.h"



// Keep the rare dependency reconstruction out of the scheduled solve's
// resource footprint. Stream ordering publishes every producer block before
// this single-block consumer starts, including on repeated workspace reuse.
extern "C" __global__ __aicore__ void spsv_complex_rational_recovery_kernel(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX, GM_ADDR vecY, GM_ADDR yWide, GM_ADDR diagPtr,
    GM_ADDR workspace, int64_t m, int32_t forward, int32_t nonUnit, float alphaReal, float alphaImag,
    uint64_t alphaDevicePtr, uint32_t riskBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (SpsvComplexPrimaryCompleted(workspace, m)) {
        return;
    }
    if (((__gm__ const SpsvTilingData*)workspace)->numLevels <= 1) {
        return;
    }
    SpsvComplex64 alpha = SpsvRecoveryAlpha(alphaReal, alphaImag, alphaDevicePtr);
    asc_vf_call<SpsvComplexRationalRecoveryBody>(dim3{32, 1, 1}, (__gm__ const int32_t*)rowPtr,
                                             (__gm__ const int32_t*)colInd, (__gm__ const SpsvComplex64*)values,
                                             (__gm__ const SpsvComplex64*)vecX, (__gm__ SpsvComplex64*)vecY,
                                             (__gm__ float*)yWide, (__gm__ const int32_t*)diagPtr, alpha, m,
                                             forward != 0, nonUnit != 0, static_cast<int32_t>(riskBlocks));
}

void spsv_complex_rational_recovery_kernel_do(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX,
                                              GM_ADDR vecY, GM_ADDR workspace, const SpsvTilingData& tiling,
                                              void* stream)
{
    spsv_complex_rational_recovery_kernel<<<1, nullptr, stream>>>(
        rowPtr, colInd, values, vecX, vecY, workspace + tiling.yWideOffset, workspace + tiling.diagPtrOffset, workspace,
        tiling.m, static_cast<int32_t>((tiling.fillMode == 0) == (tiling.opA == 0)),
        static_cast<int32_t>(tiling.diagType == 0), tiling.alpha, tiling.alphaImag, tiling.alphaDevicePtr,
        tiling.numBlocks);
}
