/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_SPARSE_SPMV_ARCH35_KERNEL_H_
#define OPS_SPARSE_SPMV_ARCH35_KERNEL_H_

#include "spmv_tiling_data.h"

extern "C" void spmv_arch35_kernel_do(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR x, GM_ADDR y,
    GM_ADDR workspace, const SpmvArch35TilingData &tiling,
    uint32_t blockDim, void *stream);

extern "C" void spmv_arch35_preprocess_kernel_do(
    GM_ADDR rowPtr, GM_ADDR workspace, uint32_t rows, int32_t indexBase,
    int64_t rowOffsetsStride, uint32_t blockDim, void *stream);

#endif  // OPS_SPARSE_SPMV_ARCH35_KERNEL_H_
