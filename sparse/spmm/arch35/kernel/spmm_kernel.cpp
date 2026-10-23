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

#include <cstdio>
#include "../spmm_config.h"
#include "spmm_kernel.h"
#include "spmm_kernel_entry_points.h"

using namespace AscendC;

// ============================================================================
// Host-side launch dispatcher — dataType/blockDim passed from spmm_host.cpp.
// ============================================================================
struct SpmmKernelLaunchArgs {
    const void *csrRowOffsets;
    const void *csrColInd;
    const void *csrValues;
    const void *matB;
    const void *initialC;
    void *matC;
    void *workspaceGM;
    void *tilingGM;
    uint32_t blockDim;
    uint32_t aivBlockDim;
    void *stream;
};

static void LaunchComplex64Regbase(const SpmmKernelLaunchArgs &args)
{
    spmm_custom_complex64_regbase_refresh<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrValues, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_complex64_regbase_compute<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.matB, (GM_ADDR)args.initialC, (GM_ADDR)args.matC,
        (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
}

static void LaunchStructuredFallback(const SpmmKernelLaunchArgs &args,
                                     int32_t structuredFallbackCount)
{
    if (structuredFallbackCount <= 0) return;
    spmm_custom_fp32_regbase_fallback<<<1, nullptr, args.stream>>>(
        (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
        (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
        (GM_ADDR)args.initialC, (GM_ADDR)args.matC,
        (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
}

static void LaunchFp32Degree2(const SpmmKernelLaunchArgs &args,
                              int32_t structuredFallbackCount)
{
    spmm_custom_fp32_degree2_refresh<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrValues, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_fp32_degree2_compute<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.matB, (GM_ADDR)args.initialC, (GM_ADDR)args.matC,
        (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    LaunchStructuredFallback(args, structuredFallbackCount);
}

static void LaunchFp32Degree4(const SpmmKernelLaunchArgs &args,
                              int32_t structuredFallbackCount)
{
    spmm_custom_fp32_degree4_refresh<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrValues, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_fp32_degree4_compute<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.matB, (GM_ADDR)args.initialC, (GM_ADDR)args.matC,
        (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    LaunchStructuredFallback(args, structuredFallbackCount);
}

static void LaunchFp32Regbase(const SpmmKernelLaunchArgs &args)
{
    spmm_custom_fp32_regbase_refresh<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrValues, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_fp32_regbase_compute<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.matB, (GM_ADDR)args.matC,
        (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_fp32_regbase_fallback<<<1, nullptr, args.stream>>>(
        (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
        (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
        (GM_ADDR)args.initialC, (GM_ADDR)args.matC,
        (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
}

static void LaunchFp32Cube(const SpmmKernelLaunchArgs &args)
{
    spmm_custom_fp32_cube_refresh<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrValues, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_fp32_cube<<<args.blockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
        (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
        (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_fp32_cube_scatter<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
}

static void LaunchFp16Cube(const SpmmKernelLaunchArgs &args)
{
    spmm_custom_fp16_cube_refresh<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrValues, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_fp16_cube<<<args.blockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
        (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
        (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_fp16_cube_scatter<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
        (GM_ADDR)args.csrValues, (GM_ADDR)args.matB, (GM_ADDR)args.initialC,
        (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
}

static void LaunchBf16Cube(const SpmmKernelLaunchArgs &args)
{
    spmm_custom_bf16_cube_refresh<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrValues, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_bf16_cube<<<args.blockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
        (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
        (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_bf16_cube_scatter<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
        (GM_ADDR)args.csrValues, (GM_ADDR)args.matB, (GM_ADDR)args.initialC,
        (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
}

static void LaunchComplex64Cube(const SpmmKernelLaunchArgs &args,
                                int32_t structuredFallbackCount)
{
    spmm_custom_complex64_refresh_a<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrValues, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_complex64_split_vector<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.matB, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_complex64_split_tail<<<1, nullptr, args.stream>>>(
        (GM_ADDR)args.matB, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_complex64_cube<<<args.blockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
        (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
        (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    spmm_custom_complex64_conditional_fallback<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
        (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
        (GM_ADDR)args.initialC, (GM_ADDR)args.matC,
        (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    if (structuredFallbackCount > 0) {
        spmm_custom_complex64_cube_fallback<<<args.aivBlockDim, nullptr, args.stream>>>(
            (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
            (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
            (GM_ADDR)args.initialC, (GM_ADDR)args.matC,
            (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
    }
    spmm_custom_complex64_cube_scatter<<<args.aivBlockDim, nullptr, args.stream>>>(
        (GM_ADDR)args.initialC, (GM_ADDR)args.matC,
        (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
}

static bool IsCubeDataType(int32_t dataType)
{
    return dataType == SPMM_DTYPE_FP32 || dataType == SPMM_DTYPE_FP16 ||
        dataType == SPMM_DTYPE_BF16 || dataType == SPMM_DTYPE_COMPLEX64;
}

static bool IsComplexRegbaseRows(int32_t cubeRows)
{
    return cubeRows == SPMM_COMPLEX64_DEGREE2_ROWS ||
        cubeRows == SPMM_COMPLEX64_DEGREE8_ROWS ||
        cubeRows == SPMM_COMPLEX64_DEGREE64_ROWS;
}

static bool IsFp32StructuredRows(int32_t cubeRows)
{
    return cubeRows == SPMM_FP32_DEGREE4_ROWS ||
        cubeRows == SPMM_FP32_DEGREE32_ROWS ||
        cubeRows == SPMM_FP32_DEGREE64_MODE;
}

static bool LaunchSimtWithInitialC(int32_t dataType,
                                   const SpmmKernelLaunchArgs &args)
{
    if (dataType == SPMM_DTYPE_FP32) {
        spmm_custom_fp32<<<args.blockDim, nullptr, args.stream>>>(
            (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
            (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
            (GM_ADDR)args.initialC, (GM_ADDR)args.matC,
            (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
        return true;
    }
    if (dataType == SPMM_DTYPE_COMPLEX64) {
        spmm_custom_complex64<<<args.blockDim, nullptr, args.stream>>>(
            (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
            (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
            (GM_ADDR)args.initialC, (GM_ADDR)args.matC,
            (GM_ADDR)args.workspaceGM, (GM_ADDR)args.tilingGM);
        return true;
    }
    return false;
}

static bool LaunchSimtWithoutInitialC(int32_t dataType,
                                      const SpmmKernelLaunchArgs &args)
{
    switch (dataType) {
        case SPMM_DTYPE_FP16:
            spmm_custom_fp16<<<args.blockDim, nullptr, args.stream>>>(
                (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
                (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
                (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM,
                (GM_ADDR)args.tilingGM);
            return true;
        case SPMM_DTYPE_INT8:
            spmm_custom_int8<<<args.blockDim, nullptr, args.stream>>>(
                (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
                (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
                (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM,
                (GM_ADDR)args.tilingGM);
            return true;
        case SPMM_DTYPE_BF16:
            spmm_custom_bf16<<<args.blockDim, nullptr, args.stream>>>(
                (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
                (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
                (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM,
                (GM_ADDR)args.tilingGM);
            return true;
        case SPMM_DTYPE_FP16_FP32:
            spmm_custom_fp16_fp32<<<args.blockDim, nullptr, args.stream>>>(
                (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
                (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
                (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM,
                (GM_ADDR)args.tilingGM);
            return true;
        case SPMM_DTYPE_BF16_FP32:
            spmm_custom_bf16_fp32<<<args.blockDim, nullptr, args.stream>>>(
                (GM_ADDR)args.csrRowOffsets, (GM_ADDR)args.csrColInd,
                (GM_ADDR)args.csrValues, (GM_ADDR)args.matB,
                (GM_ADDR)args.matC, (GM_ADDR)args.workspaceGM,
                (GM_ADDR)args.tilingGM);
            return true;
        default:
            return false;
    }
}

static bool LaunchSimt(int32_t dataType, const SpmmKernelLaunchArgs &args)
{
    if (dataType == SPMM_DTYPE_FP32 || dataType == SPMM_DTYPE_COMPLEX64) {
        return LaunchSimtWithInitialC(dataType, args);
    }
    return LaunchSimtWithoutInitialC(dataType, args);
}

extern "C" void spmm_kernel_launch(
    const void *csrRowOffsets,
    const void *csrColInd,
    const void *csrValues,
    const void *matB,
    const void *initialC,
    void *matC,
    void *workspaceGM,
    void *tilingGM,
    int32_t dataType,
    int32_t cubeRows,
    int32_t structuredFallbackCount,
    uint32_t blockDim,
    uint32_t aivBlockDim,
    void *stream)
{
    const SpmmKernelLaunchArgs args{csrRowOffsets, csrColInd, csrValues, matB,
        initialC, matC, workspaceGM, tilingGM, blockDim, aivBlockDim, stream};
    if (IsComplexRegbaseRows(cubeRows) &&
        dataType == SPMM_DTYPE_COMPLEX64) {
        LaunchComplex64Regbase(args);
    } else if (cubeRows == SPMM_FP32_DEGREE2_ROWS &&
               dataType == SPMM_DTYPE_FP32) {
        LaunchFp32Degree2(args, structuredFallbackCount);
    } else if (IsFp32StructuredRows(cubeRows) &&
               dataType == SPMM_DTYPE_FP32) {
        LaunchFp32Degree4(args, structuredFallbackCount);
    } else if (cubeRows == SPMM_FP32_REGBASE_ROWS && dataType == SPMM_DTYPE_FP32) {
        LaunchFp32Regbase(args);
    } else if (cubeRows != 0 && IsCubeDataType(dataType)) {
        switch (dataType) {
            case SPMM_DTYPE_FP32:
                LaunchFp32Cube(args);
                break;
            case SPMM_DTYPE_FP16:
                LaunchFp16Cube(args);
                break;
            case SPMM_DTYPE_BF16:
                LaunchBf16Cube(args);
                break;
            case SPMM_DTYPE_COMPLEX64:
                LaunchComplex64Cube(args, structuredFallbackCount);
                break;
            default:
                break;
        }
    } else if (!LaunchSimt(dataType, args)) {
        // 未知 dtype：显式报错并拒绝启动，避免静默按 int8 处理导致错误结果。
        fprintf(stderr, "[ERROR] spmm_kernel_launch: unsupported dataType %d "
                        "(expected a supported SpMM dtype combination), kernel not launched\n",
                dataType);
    }
}
