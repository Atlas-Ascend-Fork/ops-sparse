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

#include "spmm_kernel_device_common.h"

using namespace AscendC;

namespace {

__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
RefreshSpmmTransposeValues(__gm__ const uint8_t *values,
    __gm__ const int32_t *permutation, __gm__ uint8_t *output,
    uint64_t first, uint64_t end, int32_t valueBytes, bool conjugate)
{
    for (uint64_t index = first + threadIdx.x; index < end; index += blockDim.x) {
        const uint64_t source = static_cast<uint64_t>(permutation[index]);
        if (conjugate) {
            const auto input = reinterpret_cast<__gm__ const float *>(values) + source * 2;
            auto destination = reinterpret_cast<__gm__ float *>(output) + index * 2;
            destination[0] = input[0];
            destination[1] = -input[1];
        } else if (valueBytes == 8) {
            auto destination = reinterpret_cast<__gm__ uint32_t *>(output) + index * 2;
            const auto input = reinterpret_cast<__gm__ const uint32_t *>(values) + source * 2;
            destination[0] = input[0];
            destination[1] = input[1];
        } else if (valueBytes == 4) {
            reinterpret_cast<__gm__ uint32_t *>(output)[index] =
                reinterpret_cast<__gm__ const uint32_t *>(values)[source];
        } else if (valueBytes == 2) {
            reinterpret_cast<__gm__ uint16_t *>(output)[index] =
                reinterpret_cast<__gm__ const uint16_t *>(values)[source];
        } else {
            output[index] = values[source];
        }
    }
}

} // namespace

extern "C" __global__ __aicore__ void spmm_custom_transpose_values(
    GM_ADDR values, GM_ADDR permutation, GM_ADDR output,
    uint64_t nnz, int32_t valueBytes, bool conjugate)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const SpmmBlockWorkRange range = GetSpmmBlockWorkRange(nnz);
    asc_vf_call<RefreshSpmmTransposeValues>(dim3{kMaxSimtThreadsPerBlock},
        reinterpret_cast<__gm__ const uint8_t *>(values),
        reinterpret_cast<__gm__ const int32_t *>(permutation),
        reinterpret_cast<__gm__ uint8_t *>(output), range.first, range.end, valueBytes, conjugate);
}

extern "C" void spmm_transpose_values_launch(
    const void *values, const void *permutation, void *transposedValues,
    uint64_t nnz, int32_t valueBytes, bool conjugate,
    uint32_t blockDim, void *stream)
{
    if (nnz != 0) {
        spmm_custom_transpose_values<<<blockDim, nullptr, stream>>>(
            (GM_ADDR)values, (GM_ADDR)permutation, (GM_ADDR)transposedValues,
            nnz, valueBytes, conjugate);
    }
}
