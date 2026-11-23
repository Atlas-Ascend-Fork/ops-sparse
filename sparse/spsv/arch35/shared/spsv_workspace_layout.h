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

#ifndef SHARED_SPSV_WORKSPACE_LAYOUT_H_
#define SHARED_SPSV_WORKSPACE_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include "spsv_tiling_data.h"

// Single source of truth for workspace layout. Returned struct carries all
// sub-allocation offsets plus the total size, so callers no longer need to
// pass 12 out-parameters.
struct WorkspaceOffsets {
    int64_t diagPtrOffset{-1};
    int64_t levelPtrOffset{-1};
    int64_t levelRowOffset{-1};
    int64_t validCountOffset{-1};
    int64_t csrRowPtrOffset{-1};
    int64_t csrColIndOffset{-1};
    int64_t csrValuesOffset{-1};
    int64_t permOffset{-1};
    int64_t transRowPtrOffset{-1};
    int64_t transColIndOffset{-1};
    int64_t transValuesOffset{-1};
    int64_t transPermOffset{-1};
    int64_t yWideOffset{-1};
    size_t totalSize{0};
};

// Aggregate initialization evaluates these formulas in either host, scalar
// device or SIMT code without a cross-execution-mode helper call. Byte capacity
// retains the half-float tail before host alignment; arenaWords floors it only
// for certificate word-capacity comparisons, exactly as the prior formulas did.
struct SpsvComplexWorkspaceGeometry {
    uint64_t rows;
    uint64_t rationalDigits{(2U * static_cast<uint64_t>(kSpsvRationalBitsPerRow) * rows + 527U) / 16U};
    uint64_t arenaBytes{27U * rows * sizeof(float) +
                        static_cast<uint64_t>(kSpsvRationalTargetBatch + 3) * rationalDigits * sizeof(uint16_t)};
    uint64_t arenaWords{arenaBytes / sizeof(float)};
    uint64_t rowPrepWord{6U * rows};
    uint64_t riskWord{24U * rows};
    uint64_t digitWord{27U * rows};
    uint64_t binary64FactorWord{24U * rows};
    uint64_t binary64FlagWord{34U * rows};
    uint64_t binary64SavedWord{36U * rows};
};

#endif // SHARED_SPSV_WORKSPACE_LAYOUT_H_
