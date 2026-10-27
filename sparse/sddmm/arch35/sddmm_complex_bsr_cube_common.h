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
 * \file sddmm_complex_bsr_cube_common.h
 * \brief Ascend 950 complex64 BSR Cube fast-path contract and launchers.
 */

#ifndef SDDMM_ARCH35_COMPLEX_BSR_CUBE_COMMON_H_
#define SDDMM_ARCH35_COMPLEX_BSR_CUBE_COMMON_H_

#include <cstdint>
#include <limits>

#include "sddmm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

namespace sddmm_arch35 {

constexpr int32_t kSddmmComplexBsrCubeReduction = 128;
constexpr int32_t kSddmmComplexBsrCubeExpandedReduction = 256;
constexpr int32_t kSddmmComplexBsrCubeSmallReduction = 64;
constexpr uint64_t kSddmmComplexBsrCubeWorkspaceAlignment = 64ULL;
// The pack kernel uses the basic DataCopy overload for GM-to-UB transfers.
// Legal COMPLEX64 descriptors need only 8-byte alignment, so Host must keep
// the Cube path behind the stronger GM alignment required by that overload.
constexpr uint64_t kSddmmComplexBsrCubeInputAlignment = 32ULL;
constexpr uint64_t kSddmmComplexBsrCubeWorkspaceLimit =
    128ULL * 1024ULL * 1024ULL - kSddmmWorkspaceBytes;

// These constants describe the kernel's tiles, not a platform's capacity.
// Host compares their byte requirements with the queried L0A/L0B capacities.
template <int32_t BlockDim, int32_t Reduction>
struct SddmmComplexBsrCubeTileConfig {
    static_assert(BlockDim == 16 || BlockDim == 32 || BlockDim == 64,
        "unsupported complex BSR Cube block size");
    static_assert(Reduction == kSddmmComplexBsrCubeSmallReduction ||
        Reduction == kSddmmComplexBsrCubeReduction,
        "unsupported complex BSR Cube reduction");
    static constexpr int32_t kExpandedRows = BlockDim * 2;
    static constexpr int32_t kExpandedK = Reduction * 2;
    static constexpr int32_t kKtile = BlockDim == 16 ? kExpandedK : 128;
    static constexpr uint64_t kABytes =
        static_cast<uint64_t>(kExpandedRows) * kKtile * sizeof(float);
    static constexpr uint64_t kBBytes =
        static_cast<uint64_t>(kKtile) * BlockDim * sizeof(float);
};

struct SddmmComplexBsrCubeGroupedTileConfig {
    using SingleTile = SddmmComplexBsrCubeTileConfig<16, kSddmmComplexBsrCubeReduction>;
    static constexpr int32_t kBlockColumns = 16;
    static constexpr int32_t kBlockCount = 4;
    static constexpr int32_t kRows = SingleTile::kExpandedRows;
    static constexpr int32_t kColumns = kBlockColumns * kBlockCount;
    static constexpr int32_t kReduction = SingleTile::kExpandedK;
    static constexpr uint64_t kABytes =
        static_cast<uint64_t>(kRows) * kReduction * sizeof(float);
    static constexpr uint64_t kBBytes =
        static_cast<uint64_t>(kReduction) * kColumns * sizeof(float);
};

struct SddmmComplexBsrCubeL0Requirements {
    uint64_t aBytes = 0;
    uint64_t bBytes = 0;
};

template <int32_t BlockDim, int32_t Reduction>
constexpr SddmmComplexBsrCubeL0Requirements
SddmmComplexBsrCubeTileL0Requirements()
{
    using Tile = SddmmComplexBsrCubeTileConfig<BlockDim, Reduction>;
    if constexpr (BlockDim == 16 && Reduction == kSddmmComplexBsrCubeReduction) {
        // Consecutive columns can select the grouped path inside the kernel.
        using GroupedTile = SddmmComplexBsrCubeGroupedTileConfig;
        return {Tile::kABytes > GroupedTile::kABytes ? Tile::kABytes : GroupedTile::kABytes,
            Tile::kBBytes > GroupedTile::kBBytes ? Tile::kBBytes : GroupedTile::kBBytes};
    }
    return {Tile::kABytes, Tile::kBBytes};
}

inline SddmmComplexBsrCubeL0Requirements
SddmmComplexBsrCubeGetL0Requirements(const SddmmTilingData &tiling)
{
    if (tiling.k != kSddmmComplexBsrCubeSmallReduction &&
        tiling.k != kSddmmComplexBsrCubeReduction) {
        return {};
    }
    switch (tiling.blockDim) {
        case 16:
            return tiling.k == kSddmmComplexBsrCubeSmallReduction ?
                SddmmComplexBsrCubeTileL0Requirements<16, 64>() :
                SddmmComplexBsrCubeTileL0Requirements<16, 128>();
        case 32:
            return tiling.k == kSddmmComplexBsrCubeSmallReduction ?
                SddmmComplexBsrCubeTileL0Requirements<32, 64>() :
                SddmmComplexBsrCubeTileL0Requirements<32, 128>();
        case 64:
            return tiling.k == kSddmmComplexBsrCubeSmallReduction ?
                SddmmComplexBsrCubeTileL0Requirements<64, 64>() :
                SddmmComplexBsrCubeTileL0Requirements<64, 128>();
        default:
            return {};
    }
}

inline bool SddmmComplexBsrCubeFitsL0(
    const SddmmTilingData &tiling, uint64_t l0aBytes, uint64_t l0bBytes)
{
    const auto required = SddmmComplexBsrCubeGetL0Requirements(tiling);
    return required.aBytes != 0ULL && required.bBytes != 0ULL &&
        required.aBytes <= l0aBytes && required.bBytes <= l0bBytes;
}

struct SddmmComplexBsrCubeWorkspaceLayout {
    uint64_t packedXOffset = 0;
    uint64_t packedXBytes = 0;
    uint64_t packedYOffset = 0;
    uint64_t packedYBytes = 0;
    uint64_t scratchOffset = 0;
    uint64_t scratchBytes = 0;
    uint64_t totalBytes = 0;
};

inline bool SddmmComplexBsrCubeCheckedMul(
    uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    if (lhs != 0ULL && rhs > kMax / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline bool SddmmComplexBsrCubeCheckedAdd(
    uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    if (rhs > kMax - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

inline bool SddmmComplexBsrCubeAlign(
    uint64_t value, uint64_t &result)
{
    const uint64_t remainder =
        value % kSddmmComplexBsrCubeWorkspaceAlignment;
    if (remainder == 0ULL) {
        result = value;
        return true;
    }
    return SddmmComplexBsrCubeCheckedAdd(value,
        kSddmmComplexBsrCubeWorkspaceAlignment - remainder, result);
}

// Private region layout, relative to the aligned base passed to the combined
// launcher.  Packed X is a block-row-major (2b)x(2k) real matrix per BSR row;
// packed Y is [Yr; Yi] with shape (2k)xn; scratch is two bxb FP32 planes per
// physical BSR block.
inline bool SddmmComplexBsrCubeMakeWorkspaceLayout(
    const SddmmTilingData &tiling,
    SddmmComplexBsrCubeWorkspaceLayout &layout)
{
    layout = {};
    if (tiling.m <= 0 || tiling.n <= 0 || tiling.k <= 0 ||
        tiling.outputsPerBatch <= 0) {
        return false;
    }

    uint64_t packedXElements = 0;
    uint64_t packedYElements = 0;
    uint64_t scratchElements = 0;
    if (!SddmmComplexBsrCubeCheckedMul(
            static_cast<uint64_t>(tiling.m),
            static_cast<uint64_t>(tiling.k), packedXElements) ||
        !SddmmComplexBsrCubeCheckedMul(
            packedXElements, 4ULL, packedXElements) ||
        !SddmmComplexBsrCubeCheckedMul(
            static_cast<uint64_t>(tiling.n),
            static_cast<uint64_t>(tiling.k), packedYElements) ||
        !SddmmComplexBsrCubeCheckedMul(
            packedYElements, 2ULL, packedYElements) ||
        !SddmmComplexBsrCubeCheckedMul(
            static_cast<uint64_t>(tiling.outputsPerBatch),
            2ULL, scratchElements) ||
        !SddmmComplexBsrCubeCheckedMul(
            packedXElements, sizeof(float), layout.packedXBytes) ||
        !SddmmComplexBsrCubeCheckedMul(
            packedYElements, sizeof(float), layout.packedYBytes) ||
        !SddmmComplexBsrCubeCheckedMul(
            scratchElements, sizeof(float), layout.scratchBytes)) {
        layout = {};
        return false;
    }

    uint64_t next = 0;
    if (!SddmmComplexBsrCubeAlign(layout.packedXBytes, next)) {
        layout = {};
        return false;
    }
    layout.packedYOffset = next;
    if (!SddmmComplexBsrCubeCheckedAdd(
            layout.packedYOffset, layout.packedYBytes, next) ||
        !SddmmComplexBsrCubeAlign(next, layout.scratchOffset) ||
        !SddmmComplexBsrCubeCheckedAdd(
            layout.scratchOffset, layout.scratchBytes, next) ||
        !SddmmComplexBsrCubeAlign(next, layout.totalBytes)) {
        layout = {};
        return false;
    }
    return layout.totalBytes != 0ULL;
}

inline uint64_t SddmmComplexBsrCubeWorkspaceBytes(
    const SddmmTilingData &tiling)
{
    SddmmComplexBsrCubeWorkspaceLayout layout{};
    return SddmmComplexBsrCubeMakeWorkspaceLayout(tiling, layout) ?
        layout.totalBytes : 0ULL;
}

inline bool SddmmComplexBsrCubeDenseLayout(const SddmmTilingData &tiling)
{
    return tiling.rowsX == tiling.m && tiling.colsX == tiling.k &&
        tiling.rowsY == tiling.k && tiling.colsY == tiling.n &&
        tiling.opX == 0 && tiling.opY == 0 &&
        tiling.orderX == 0 && tiling.orderY == 0 &&
        tiling.ldX == tiling.k && tiling.ldY == tiling.n &&
        tiling.pointerMode == 0 && tiling.usePackedY == 0;
}

inline bool SddmmComplexBsrCubeSingleBatch(const SddmmTilingData &tiling)
{
    return tiling.batchCount == 1 && tiling.batchCountX == 1 &&
        tiling.batchCountY == 1 && tiling.offsetsBatchStride == 0 &&
        tiling.columnsBatchStride == 0 && tiling.valuesBatchStride == 0;
}

inline bool SddmmComplexBsrCubeSparseShape(const SddmmTilingData &tiling)
{
    const bool supportedBlock = tiling.blockDim == 16 ||
        tiling.blockDim == 32 || tiling.blockDim == 64;
    return tiling.format == SDDMM_FORMAT_BSR && supportedBlock &&
        (tiling.blockOrder == 0 || tiling.blockOrder == 1) &&
        (tiling.indexBase == 0 || tiling.indexBase == 1) &&
        tiling.m > 0 && tiling.n > 0 && tiling.nnzUnits > 0 &&
        tiling.m % tiling.blockDim == 0 && tiling.n % tiling.blockDim == 0 &&
        tiling.blockRows == tiling.m / tiling.blockDim &&
        tiling.blockCols == tiling.n / tiling.blockDim;
}

inline bool SddmmComplexBsrCubeCanRun(const SddmmTilingData &tiling)
{
    const bool supportedReduction =
        tiling.k == kSddmmComplexBsrCubeSmallReduction ||
        tiling.k == kSddmmComplexBsrCubeReduction;
    if (!SddmmComplexBsrCubeSparseShape(tiling) ||
        tiling.inputType != SDDMM_VALUE_COMPLEX64 ||
        tiling.outputType != SDDMM_VALUE_COMPLEX64 ||
        tiling.computeType != SDDMM_VALUE_COMPLEX64 || !supportedReduction ||
        !SddmmComplexBsrCubeDenseLayout(tiling) ||
        !SddmmComplexBsrCubeSingleBatch(tiling)) {
        return false;
    }

    uint64_t blockArea = 0;
    uint64_t expectedOutputs = 0;
    if (!SddmmComplexBsrCubeCheckedMul(
            static_cast<uint64_t>(tiling.blockDim),
            static_cast<uint64_t>(tiling.blockDim), blockArea) ||
        !SddmmComplexBsrCubeCheckedMul(
            static_cast<uint64_t>(tiling.nnzUnits),
            blockArea, expectedOutputs) ||
        expectedOutputs != static_cast<uint64_t>(tiling.outputsPerBatch)) {
        return false;
    }

    const uint64_t workspaceBytes =
        SddmmComplexBsrCubeWorkspaceBytes(tiling);
    return workspaceBytes != 0ULL &&
        workspaceBytes <= kSddmmComplexBsrCubeWorkspaceLimit;
}

// This is a sub-predicate of the validated Cube path. A physical-block upper
// bound avoids paying for strided gathers when Y is already mostly covered.
#if defined(__NPU_ARCH__)
__aicore__
#endif
inline bool SddmmComplexBsrCubeUseSparseYPack(const SddmmTilingData &tiling)
{
    return tiling.blockDim == 64 && tiling.k == 128 &&
        tiling.nnzUnits > 0 && tiling.nnzUnits <= 64 &&
        tiling.nnzUnits * 64 <= tiling.n / 2;
}

} // namespace sddmm_arch35

extern "C" {

void sddmm_complex_bsr_cube_pack_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns, GM_ADDR matX, GM_ADDR matY,
    GM_ADDR packedX, GM_ADDR packedY,
    const SddmmTilingData &tiling, uint32_t vectorBlockDim, void *stream);

void sddmm_complex_bsr_cube_compute_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns,
    GM_ADDR packedX, GM_ADDR packedY, GM_ADDR scratch,
    const SddmmTilingData &tiling, uint32_t cubeBlockDim, void *stream);

void sddmm_complex_bsr_cube_epilogue_kernel_do(
    GM_ADDR sparseValues, GM_ADDR scratch,
    const SddmmTilingData &tiling, uint32_t vectorBlockDim, void *stream);

// workspace points to the beginning of the private region described by
// SddmmComplexBsrCubeMakeWorkspaceLayout.  All three kernels are submitted to
// one stream in pack -> Cube -> epilogue order.
void sddmm_complex_bsr_cube_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns, GM_ADDR sparseValues,
    GM_ADDR matX, GM_ADDR matY, GM_ADDR workspace,
    const SddmmTilingData &tiling,
    uint32_t cubeBlockDim, uint32_t vectorBlockDim, void *stream);

} // extern "C"

#endif // SDDMM_ARCH35_COMPLEX_BSR_CUBE_COMMON_H_
