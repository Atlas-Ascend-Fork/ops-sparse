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

#ifndef SPMM_KERNEL_H
#define SPMM_KERNEL_H

#include <stdint.h>

extern "C" void spmm_kernel_launch(
    const void *csrRowOffsets, const void *csrColInd,
    const void *csrValues, const void *matB, const void *initialC, void *matC,
    void *workspaceGM, void *tilingGM,
    int32_t dataType, int32_t cubeRows, int32_t structuredFallbackCount,
    uint32_t blockDim,
    uint32_t aivBlockDim, void *stream);

extern "C" void spmm_transpose_values_launch(
    const void *values, const void *permutation, void *transposedValues,
    uint64_t nnz, int32_t valueBytes, bool conjugate,
    uint32_t blockDim, void *stream);

#endif // SPMM_KERNEL_H
