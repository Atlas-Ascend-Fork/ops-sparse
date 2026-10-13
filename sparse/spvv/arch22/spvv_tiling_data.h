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

#ifndef SPVV_TILING_DATA_H_
#define SPVV_TILING_DATA_H_

#include <stdint.h>
#include <unistd.h>

#ifndef __gm__
#define __gm__
#endif

// Tile size for vectorized computation loop
#define SPVV_TILE_LENGTH 3072

// Minimum nnz per core: for small nnz, use fewer cores (each doing at least
// this many elements) to amortize per-core launch/sync overhead.
#define SPVV_MIN_NNZ_PER_CORE 512

// Tiling data passed directly as kernel parameter (no device buffer needed)
struct SpvvTilingData {
    uint32_t nnz;
    uint64_t yLen;
    uint32_t nnzPerCore;
};

#endif
