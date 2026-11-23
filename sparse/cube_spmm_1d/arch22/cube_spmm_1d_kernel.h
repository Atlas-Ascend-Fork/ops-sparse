/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUBE_SPMM_1D_KERNEL_H_
#define CUBE_SPMM_1D_KERNEL_H_
#include <cstddef>
#include <type_traits>
#include "internal/spmm_1d_rt.h"
// Argument layout consumed by the CCE kernel entry points.
struct KernelArgs {
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t vectorLength;
    uint16_t* values;
    int32_t* columnIndices;
    int32_t* rowOffsets;
    uint16_t* b;
    float* c;
    void* fftsAddress;
    uint16_t* workspace;
    int64_t bPackSlotStrideElements;
    int64_t totalBPackElements;
    int64_t maxVectorsPerRow;
    int64_t numNBlocks;
    int64_t ldc;
    int64_t tilingKey;
};
// Only binary16 addresses cross the host/device boundary; no host FP16 arithmetic.
static_assert(sizeof(void*) == 8 && std::is_standard_layout<KernelArgs>::value, "64-bit kernel ABI");
static_assert(sizeof(KernelArgs) == 120 && offsetof(KernelArgs, values) == 16 &&
                  offsetof(KernelArgs, bPackSlotStrideElements) == 72, "CCE entry argument layout");
#endif
