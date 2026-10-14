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
 * \file scatter_kernel.cpp
 * \brief aclsparseScatter SIMT kernel（arch35）：grid-stride + stride-unroll ILP。
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "scatter_kernel.h"

/// Y[X.indices[i] - idxBase] = X.values[i]
/// Unrolled grid-stride hides GM latency (esp. 8-byte complex64 random writes).
template <typename ValT, typename IdxT>
__simt_vf__ __aicore__ __launch_bounds__(kScatterMaxThreadsPerBlock) inline void
ScatterSimtCompute(
    uint64_t nnz,
    __gm__ const IdxT *indices,
    __gm__ const ValT *values,
    __gm__ ValT *yVec,
    IdxT idxBase)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x)
                         + static_cast<uint64_t>(threadIdx.x);
    const uint64_t gridStride = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);
    const uint64_t bigStride = gridStride * static_cast<uint64_t>(kScatterStrideUnroll);

    for (uint64_t base = tid; base < nnz; base += bigStride) {
#pragma unroll
        for (uint32_t u = 0; u < kScatterStrideUnroll; ++u) {
            const uint64_t i = base + static_cast<uint64_t>(u) * gridStride;
            if (i < nnz) {
                const IdxT idx = indices[i] - idxBase;
                yVec[idx] = values[i];
            }
        }
    }
}

template <typename IdxT>
__aicore__ inline void DispatchScatterByValType(
    const ScatterTilingData &tiling,
    GM_ADDR gmIndices, GM_ADDR gmValues, GM_ADDR gmYVec)
{
    const dim3 blockDim{tiling.threadsPerBlock ? tiling.threadsPerBlock : kScatterMaxThreadsPerBlock};
    if (tiling.valType == SCATTER_VAL_FP32) {
        asc_vf_call<ScatterSimtCompute<float, IdxT>>(
            blockDim,
            tiling.nnz,
            (__gm__ const IdxT *)gmIndices,
            (__gm__ const float *)gmValues,
            (__gm__ float *)gmYVec,
            IdxT(tiling.idxBase));
    } else if (tiling.valType == SCATTER_VAL_FP16) {
        asc_vf_call<ScatterSimtCompute<half, IdxT>>(
            blockDim,
            tiling.nnz,
            (__gm__ const IdxT *)gmIndices,
            (__gm__ const half *)gmValues,
            (__gm__ half *)gmYVec,
            IdxT(tiling.idxBase));
    } else if (tiling.valType == SCATTER_VAL_BF16) {
        asc_vf_call<ScatterSimtCompute<bfloat16_t, IdxT>>(
            blockDim,
            tiling.nnz,
            (__gm__ const IdxT *)gmIndices,
            (__gm__ const bfloat16_t *)gmValues,
            (__gm__ bfloat16_t *)gmYVec,
            IdxT(tiling.idxBase));
    } else if (tiling.valType == SCATTER_VAL_INT8) {
        asc_vf_call<ScatterSimtCompute<int8_t, IdxT>>(
            blockDim,
            tiling.nnz,
            (__gm__ const IdxT *)gmIndices,
            (__gm__ const int8_t *)gmValues,
            (__gm__ int8_t *)gmYVec,
            IdxT(tiling.idxBase));
    } else {  // SCATTER_VAL_COMPLEX64 — 8-byte lane copy
        asc_vf_call<ScatterSimtCompute<uint64_t, IdxT>>(
            blockDim,
            tiling.nnz,
            (__gm__ const IdxT *)gmIndices,
            (__gm__ const uint64_t *)gmValues,
            (__gm__ uint64_t *)gmYVec,
            IdxT(tiling.idxBase));
    }
}

extern "C" __global__ __aicore__ void scatter_kernel(
    GM_ADDR gmIndices, GM_ADDR gmValues, GM_ADDR gmYVec,
    const ScatterTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.idxType == SCATTER_IDX_I32) {
        DispatchScatterByValType<int32_t>(tiling, gmIndices, gmValues, gmYVec);
    } else {
        DispatchScatterByValType<int64_t>(tiling, gmIndices, gmValues, gmYVec);
    }
}

extern "C" void scatter_kernel_do(
    GM_ADDR indices, GM_ADDR values, GM_ADDR yVec,
    const ScatterTilingData &tiling, void *stream)
{
    scatter_kernel<<<tiling.numBlocks, nullptr, stream>>>(indices, values, yVec, tiling);
}
