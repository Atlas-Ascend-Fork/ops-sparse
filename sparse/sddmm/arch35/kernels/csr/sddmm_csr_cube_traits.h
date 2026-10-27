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

#ifndef SDDMM_ARCH35_CSR_CUBE_TRAITS_H_
#define SDDMM_ARCH35_CSR_CUBE_TRAITS_H_

#include "kernel_operator.h"
#include "../../sddmm_csr_cube_common.h"

namespace sddmm_arch35 {

template <typename T>
struct SddmmCsrCubeTypeCode;

template <>
struct SddmmCsrCubeTypeCode<float> {
    static constexpr int32_t value = SDDMM_VALUE_FP32;
};

template <>
struct SddmmCsrCubeTypeCode<half> {
    static constexpr int32_t value = SDDMM_VALUE_FP16;
};

template <>
struct SddmmCsrCubeTypeCode<bfloat16_t> {
    static constexpr int32_t value = SDDMM_VALUE_BF16;
};

template <typename T>
struct SddmmCsrCubeNtile {
    static constexpr int32_t value = kSddmmCsrCubeNtile;
};

template <>
struct SddmmCsrCubeNtile<half> {
    static constexpr int32_t value = kSddmmCsrCubeWideNtile;
};

template <>
struct SddmmCsrCubeNtile<bfloat16_t> {
    static constexpr int32_t value = kSddmmCsrCubeWideNtile;
};

__aicore__ inline bool SddmmCsrCubeSupportedTypes(
    const SddmmTilingData &tiling)
{
    return (tiling.inputType == SDDMM_VALUE_FP32 &&
                tiling.outputType == SDDMM_VALUE_FP32) ||
        (tiling.inputType == SDDMM_VALUE_FP16 &&
            (tiling.outputType == SDDMM_VALUE_FP16 ||
                tiling.outputType == SDDMM_VALUE_FP32)) ||
        (tiling.inputType == SDDMM_VALUE_BF16 &&
            (tiling.outputType == SDDMM_VALUE_BF16 ||
                tiling.outputType == SDDMM_VALUE_FP32));
}

__aicore__ inline bool SddmmCsrCubeCommonCanRun(
    const SddmmTilingData &tiling,
    const SddmmCsrCubePlan &plan)
{
    const int64_t completeRows = static_cast<int64_t>(plan.groupCount) *
        static_cast<int64_t>(kSddmmCsrCubeRows);
    const bool validYLayout =
        (plan.yPhysicalOrder == 0 && plan.yLeadingDimension == tiling.n) ||
        (plan.yPhysicalOrder == 1 &&
            plan.yLeadingDimension == kSddmmCsrCubeReduction);
    return plan.patternKind == SDDMM_CSR_CUBE_PATTERN_CYCLIC_64_STRIDE_17 &&
        plan.logicalFirstColumn >= 0 &&
        static_cast<int64_t>(plan.logicalFirstColumn) < tiling.n &&
        plan.logicalFirstColumn % kSddmmCsrCubeL0cC0 == 0 &&
        tiling.n % kSddmmCsrCubeL0cC0 == 0 &&
        plan.rowColumnStride == kSddmmCsrCubeRowColumnStride &&
        plan.samplesPerRow == kSddmmCsrCubeSamplesPerRow &&
        plan.groupCount >= 0 && completeRows <= tiling.m &&
        validYLayout && tiling.format == SDDMM_FORMAT_CSR &&
        tiling.k == kSddmmCsrCubeReduction &&
        tiling.n >= kSddmmCsrCubeWorkspaceBand &&
        tiling.rowsX == tiling.m &&
        tiling.colsX == kSddmmCsrCubeReduction &&
        tiling.rowsY == kSddmmCsrCubeReduction &&
        tiling.colsY == tiling.n &&
        tiling.opX == 0 && tiling.opY == 0 &&
        tiling.orderX == 0 && tiling.ldX == kSddmmCsrCubeReduction &&
        tiling.computeType == SDDMM_VALUE_FP32 &&
        tiling.batchCount == 1 &&
        tiling.batchCountX == 1 && tiling.batchCountY == 1 &&
        tiling.offsetsBatchStride == 0 &&
        tiling.columnsBatchStride == 0 &&
        tiling.valuesBatchStride == 0 &&
        tiling.nnzUnits == tiling.m * kSddmmCsrCubeSamplesPerRow &&
        tiling.outputsPerBatch == tiling.nnzUnits;
}

} // namespace sddmm_arch35

#endif // SDDMM_ARCH35_CSR_CUBE_TRAITS_H_
