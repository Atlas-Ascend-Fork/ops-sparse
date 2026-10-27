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
 * \file sddmm_csr_cube_common.h
 * \brief Host/Kernel shared metadata for the DAV-3510 CSR Cube path.
 */

#ifndef SDDMM_ARCH35_CSR_CUBE_COMMON_H_
#define SDDMM_ARCH35_CSR_CUBE_COMMON_H_

#include <cstdint>

#include "sddmm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

namespace sddmm_arch35 {

constexpr int32_t kSddmmCsrCubeRows = 16;
constexpr int32_t kSddmmCsrCubeReduction = 128;
constexpr int32_t kSddmmCsrCubeSamplesPerRow = 64;
constexpr int32_t kSddmmCsrCubeSamplesPerGroup =
    kSddmmCsrCubeRows * kSddmmCsrCubeSamplesPerRow;
constexpr int32_t kSddmmCsrCubeRowColumnStride =
    kSddmmCsrCyclic64RowStride;
constexpr int32_t kSddmmCsrCubeLogicalBand =
    kSddmmCsrCubeSamplesPerRow +
    (kSddmmCsrCubeRows - 1) * kSddmmCsrCubeRowColumnStride;
constexpr int32_t kSddmmCsrCubeWorkspaceBand = 320;
constexpr int32_t kSddmmCsrCubeNtile = 64;
constexpr int32_t kSddmmCsrCubeWideNtile = 128;
constexpr int32_t kSddmmCsrCubeKtile = 128;
constexpr int32_t kSddmmCsrCubeL0cC0 = 16;
constexpr uint8_t kSddmmCsrCubeNonFinalAccumulation = 2;
constexpr uint8_t kSddmmCsrCubeFinalAccumulation = 3;

enum SddmmCsrCubePatternKind : int32_t {
    SDDMM_CSR_CUBE_PATTERN_NONE = 0,
    SDDMM_CSR_CUBE_PATTERN_CYCLIC_64_STRIDE_17 = 1,
};

struct SddmmCsrCubePlan {
    int32_t patternKind;
    int32_t logicalFirstColumn;
    int32_t rowColumnStride;
    int32_t samplesPerRow;
    int32_t yPhysicalOrder;
    int32_t yLeadingDimension;
    int32_t groupCount;
    int32_t reserved;
};

static_assert(sizeof(SddmmCsrCubePlan) == 32,
    "CSR Cube plan is copied by value and must remain a compact POD");

constexpr uint64_t SddmmCsrCubeDenseWorkspaceBytes(uint64_t rows)
{
    return (rows / static_cast<uint64_t>(kSddmmCsrCubeRows)) *
        static_cast<uint64_t>(kSddmmCsrCubeRows) *
        static_cast<uint64_t>(kSddmmCsrCubeWorkspaceBand) * sizeof(float);
}

constexpr uint64_t SddmmCsrCubeUsefulOutputBytes(uint64_t rows)
{
    return (rows / static_cast<uint64_t>(kSddmmCsrCubeRows)) *
        static_cast<uint64_t>(kSddmmCsrCubeRows) *
        static_cast<uint64_t>(kSddmmCsrCubeLogicalBand) * sizeof(float);
}

constexpr uint64_t SddmmCsrCubeExecutedFlops(uint64_t rows)
{
    return (rows / static_cast<uint64_t>(kSddmmCsrCubeRows)) *
        static_cast<uint64_t>(kSddmmCsrCubeRows) *
        static_cast<uint64_t>(kSddmmCsrCubeWorkspaceBand) *
        static_cast<uint64_t>(kSddmmCsrCubeReduction) * 2ULL;
}

} // namespace sddmm_arch35

extern "C" {

void sddmm_csr_cube_kernel_do(
    GM_ADDR matX, GM_ADDR matY, GM_ADDR denseWorkspace,
    const SddmmTilingData &tiling,
    const sddmm_arch35::SddmmCsrCubePlan &plan,
    uint32_t numBlocks, void *stream);

void sddmm_csr_cube_gather_kernel_do(
    GM_ADDR sparseValues, GM_ADDR denseWorkspace,
    const SddmmTilingData &tiling,
    const sddmm_arch35::SddmmCsrCubePlan &plan,
    uint32_t numBlocks, void *stream);

} // extern "C"

#endif // SDDMM_ARCH35_CSR_CUBE_COMMON_H_
