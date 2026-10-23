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

#ifndef SPMM_KERNEL_ENTRY_POINTS_H
#define SPMM_KERNEL_ENTRY_POINTS_H

#include "kernel_operator.h"
#include <stdint.h>

extern "C" __global__ __aicore__ void spmm_custom_fp32(
    GM_ADDR csrRowOffsets,
    GM_ADDR csrColInd,
    GM_ADDR csrValues,
    GM_ADDR matB,
    GM_ADDR initialC,
    GM_ADDR matC,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp32_regbase_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp32_degree2_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp32_degree2_compute(
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp32_degree4_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp32_degree4_compute(
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_complex64_regbase_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_complex64_regbase_compute(
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp32_regbase_compute(
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp32_regbase_fallback(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp32_cube_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp16_cube_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_bf16_cube_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp32_cube(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp32_cube_scatter(
    GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp16(
    GM_ADDR csrRowOffsets,
    GM_ADDR csrColInd,
    GM_ADDR csrValues,
    GM_ADDR matB,
    GM_ADDR matC,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp16_cube(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp16_cube_scatter(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_bf16_cube_scatter(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp16_cube_fallback(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_bf16_cube_fallback(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_int8(
    GM_ADDR csrRowOffsets,
    GM_ADDR csrColInd,
    GM_ADDR csrValues,
    GM_ADDR matB,
    GM_ADDR matC,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_bf16(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_bf16_cube(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_complex64(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_complex64_refresh_a(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_complex64_split_vector(
    GM_ADDR matB, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_complex64_split_tail(
    GM_ADDR matB, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_complex64_conditional_fallback(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_complex64_cube(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_complex64_cube_fallback(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_complex64_cube_scatter(
    GM_ADDR initialC, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_fp16_fp32(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM);
extern "C" __global__ __aicore__ void spmm_custom_bf16_fp32(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM);

#endif  // SPMM_KERNEL_ENTRY_POINTS_H
