/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSPARSE_SPMM_1D_KERNEL_UTILS_H
#define ACLSPARSE_SPMM_1D_KERNEL_UTILS_H

/* Builds the AIC-AIV synchronization configuration. */
__aicore__ inline int64_t GET_FFST_MSG(int64_t mode, int64_t flagId) { return 1 | (mode << 4) | (flagId << 8); }

__aicore__ inline int64_t ROUND(int64_t num, int64_t paddingNum)
{
    if (paddingNum == 0) {
        return 0;
    }
    return (num + paddingNum - 1) / paddingNum * paddingNum;
}

struct SpmmKernelInput {
    int M, N, K, vec_length;
    __gm__ __fp16* values;
    __gm__ int *col_indices, *row_offsets;
    __gm__ __fp16* B;
    __gm__ float* C;
    __gm__ void* ffts_addr;
    __gm__ __fp16* workspace1;
    int64_t b_pack_slot_stride_elems, total_b_pack_elems;
    int64_t max_vectors_per_row, num_n_blocks, ldc;
};

#include "spmm_1d_fp16_utils.h"

#endif
