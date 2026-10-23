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

#ifndef SPMM_KERNEL_REGBASE_COMMON_H
#define SPMM_KERNEL_REGBASE_COMMON_H

#include "spmm_kernel_device_common.h"

namespace {

struct SpmmRegBaseTensorState {
    GlobalTensor<float> matB_;
    GlobalTensor<float> initialC_;
    GlobalTensor<float> matC_;
    GlobalTensor<float> packedValues_;
    GlobalTensor<int32_t> rowIds_;
    GlobalTensor<int32_t> rowMeta_;
    __gm__ const SpmmFp32Degree2BatchMeta *batches_{nullptr};
    SpmmKernelTilingData td_{};
    int32_t tileCount_{0};
};

__aicore__ inline SpmmFp32Degree2BatchMeta LoadSpmmRegBaseBatch(
    __gm__ const SpmmFp32Degree2BatchMeta *batches, int32_t index)
{
    SpmmFp32Degree2BatchMeta batch{};
    batch.window_begin = batches[index].window_begin;
    batch.row_count = batches[index].row_count;
    batch.row_meta_begin = batches[index].row_meta_begin;
    batch.window_rows = batches[index].window_rows;
    return batch;
}

__aicore__ inline void InitializeSpmmRegBaseTensors(
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC, GM_ADDR workspaceGM,
    GM_ADDR tilingGM, SpmmKernelTilingData *tiling, GlobalTensor<float> &matBBuffer,
    GlobalTensor<float> &initialCBuffer, GlobalTensor<float> &matCBuffer,
    __gm__ const SpmmFp32Degree2BatchMeta **batches,
    GlobalTensor<int32_t> &rowIds, GlobalTensor<int32_t> &rowMeta,
    GlobalTensor<float> &packedValues)
{
    *tiling = LoadSpmmKernelTilingData(tilingGM);
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
    matBBuffer.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matB));
    initialCBuffer.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initialC));
    matCBuffer.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matC));
    *batches = reinterpret_cast<__gm__ const SpmmFp32Degree2BatchMeta *>(
        workspace + tiling->cube_meta_offset);
    rowIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        workspace + tiling->cube_row_ids_offset));
    rowMeta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        workspace + tiling->cube_slot1_offset));
    packedValues.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        workspace + tiling->cube_packed_a_offset));
}

template<typename PackedT>
__aicore__ inline SpmmKernelTilingData InitializeSpmmRegBaseRefreshData(
    GM_ADDR workspaceGM, GM_ADDR tilingGM,
    __gm__ const SpmmFp32RegBaseHeader **header,
    __gm__ const int32_t **rowMeta, __gm__ PackedT **packedValues)
{
    const SpmmKernelTilingData tiling = LoadSpmmKernelTilingData(tilingGM);
    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
    *header = reinterpret_cast<__gm__ const SpmmFp32RegBaseHeader *>(
        workspace + tiling.cube_scratch_offset);
    *rowMeta = reinterpret_cast<__gm__ const int32_t *>(
        workspace + tiling.cube_slot1_offset);
    *packedValues = reinterpret_cast<__gm__ PackedT *>(
        workspace + tiling.cube_packed_a_offset);
    return tiling;
}

} // namespace

#endif // SPMM_KERNEL_REGBASE_COMMON_H
