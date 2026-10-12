/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef SCATTER_H
#define SCATTER_H

#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"

#include <stdint.h>

#ifndef __gm__
#define __gm__
#endif

/// 将 const SpVec 描述符转为内部结构体指针（解除 const，仅读用途）。
/// vecX 按 aclsparseConstSpVecDescr_t 声明（const），const 解除统一在此完成，
/// 禁止在业务代码中直接 const_cast。
inline struct aclsparseSpVecDescr *ScatterToSpVecInner(aclsparseConstSpVecDescr_t desc)
{
    return const_cast<struct aclsparseSpVecDescr *>(
        reinterpret_cast<const struct aclsparseSpVecDescr *>(desc));
}

// Maximum number of cores for tiling（coreNnzOffset/coreNnzCount 定长数组容量防护）
constexpr uint32_t SCATTER_MAX_CORE_NUM = 64;

// Maximum nnz elements per tile (UB capacity constraint, power of 2)
constexpr uint32_t SCATTER_TILE_NN_MAX = 4096;

// Number of buffers per queue (2 = DoubleBuffer, overlap MTE2 with VEC+MTE3)
constexpr uint32_t SCATTER_BUFFER_NUM = 2;

// Tiling data structure passed to the kernel（随 kernel 启动参数 const 引用下发，0 workspace）
struct ScatterTilingData {
    uint32_t nnz;                                      // total non-zero elements
    uint32_t blockNum;                                  // actual number of blocks used
    uint32_t tileNn;                                    // elements per tile (always power of 2)
    uint32_t tileNnAligned;                             // tileNn round up to dtype alignment element count
                                                       // (int8:32 / fp16,bf16:16 / fp32:8 / complex64:4)
    int32_t idxBase;                                    // SpVec index base (0/1), kernel 内统一减法
    uint64_t yLen;                                      // Y vector length in bytes (size * elemBytes)
    uint32_t coreNnzOffset[SCATTER_MAX_CORE_NUM];      // per-core start offset
    uint32_t coreNnzCount[SCATTER_MAX_CORE_NUM];       // per-core element count
};

#endif
