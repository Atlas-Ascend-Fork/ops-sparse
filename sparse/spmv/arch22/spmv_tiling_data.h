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

#ifndef SPMV_TILING_DATA_H
#define SPMV_TILING_DATA_H

#include <cstdint>

struct SpmvTilingData {
    uint32_t totalRowsNum; // 矩阵总行数
    uint32_t totalColNum;  // 矩阵总列数
    uint32_t indexBase;    // CSR 索引基值：0 或 1
    uint32_t readY;        // 1 = beta 非零需要读 y；0 = beta 为 0，直接覆写不读 y
    uint32_t totalNnz;     // 矩阵总非零元数
};

#endif // SPMV_TILING_DATA_H
