/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "spsv_kernel.h"
#include "spsv_analysis.h"
#include "spsv_solve.h"
#include "spsv_update.h"

using namespace AscendC;
using namespace spsv_a2;

extern "C" __global__ __aicore__ void spsv_a2_analysis_kernel(GM_ADDR ptr,
    GM_ADDR idx, GM_ADDR values, GM_ADDR workspace, SpsvA2Tiling tiling)
{
    if (GetBlockIdx() != 0) return;
    TPipe pipe;
    Analysis kernel(pipe, ptr, idx, values, workspace, tiling);
    kernel.Run();
}
extern "C" __global__ __aicore__ void spsv_a2_solve_kernel(GM_ADDR x,
    GM_ADDR y, GM_ADDR workspace, SpsvA2Tiling tiling)
{
    TPipe pipe;
    if (tiling.components == 1) RunSolve<1>(pipe, x, y, workspace, tiling);
    else RunSolve<2>(pipe, x, y, workspace, tiling);
}
extern "C" __global__ __aicore__ void spsv_a2_update_kernel(GM_ADDR source,
    GM_ADDR workspace, int32_t diagonal, SpsvA2Tiling tiling)
{
    if (GetBlockIdx() != 0) return;
    TPipe pipe;
    if (!diagonal && tiling.format == 0 && tiling.nnz == tiling.sourceNnz &&
        !tiling.conjugate && (!tiling.transpose || tiling.diagonalMatrix)) {
        CopyGeneralValues(pipe, source, workspace, tiling);
        return;
    }
    Update kernel(pipe, source, workspace, tiling);
    kernel.Run(diagonal != 0);
}

void SpsvA2Analysis(void *ptr, void *idx, void *values, void *workspace,
    const SpsvA2Tiling &tiling, void *stream)
{
    spsv_a2_analysis_kernel<<<1, nullptr, stream>>>((GM_ADDR)ptr,
        (GM_ADDR)idx, (GM_ADDR)values, (GM_ADDR)workspace, tiling);
}
void SpsvA2Solve(void *x, void *y, void *workspace,
    const SpsvA2Tiling &tiling, void *stream)
{
    spsv_a2_solve_kernel<<<tiling.blocks, nullptr, stream>>>((GM_ADDR)x,
        (GM_ADDR)y, (GM_ADDR)workspace, tiling);
}
void SpsvA2Update(void *values, void *workspace, int32_t diagonal,
    const SpsvA2Tiling &tiling, void *stream)
{
    SpsvA2Tiling parameters = tiling;
    parameters.sourceNnz = diagonal ? tiling.m : tiling.sourceNnz;
    spsv_a2_update_kernel<<<1, nullptr, stream>>>((GM_ADDR)values,
        (GM_ADDR)workspace, diagonal, parameters);
}
