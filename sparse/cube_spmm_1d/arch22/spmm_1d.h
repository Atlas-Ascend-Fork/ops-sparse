/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSPARSE_SPMM_1D_INTERNAL_H_
#define ACLSPARSE_SPMM_1D_INTERNAL_H_

#include <cstdint>
#include <limits>
#include <optional>

#include "internal/spmm_1d_tiling_keys.h"

struct SpmmTilingConfig {
    int32_t aic_n_batch;
    int32_t sparse_k_tile;
    int32_t b_pack_n_batch;
    int32_t tiling_key;
    const char *variant_name;
    const char *kernel_name;
    const char *binary_name;
};

constexpr int32_t SPMM_1D_VECTOR_LENGTH = 16;
constexpr int32_t SPMM_1D_BPACK_512_N_BATCH = 32;
constexpr int32_t SPMM_1D_BPACK_1024_N_BATCH = 64;
constexpr int32_t SPMM_1D_BPACK_2048_N_BATCH = 128;
constexpr int32_t SPMM_1D_BPACK_PIPE_DEPTH = 2;
constexpr int64_t SPMM_1D_SMALL_ROW_MAX_VECTORS = 160;
constexpr int64_t SPMM_1D_N_PANEL_512_COLUMNS = 512;
constexpr int64_t SPMM_1D_N_PANEL_1024_COLUMNS = 1024;
constexpr int64_t SPMM_1D_HALF_AIC32_TILE_COLUMNS = 256;
constexpr int64_t SPMM_1D_WORKSPACE_ALIGNMENT_BYTES = 512;

constexpr const char *SPMM_1D_WORKSPACE_KERNEL_NAME =
    "spmm_1d_kernel_workspace_unified";
constexpr const char *SPMM_1D_WORKSPACE_BINARY_NAME =
    "spmm_1d_kernel_workspace_unified.o";
constexpr const char *SPMM_1D_DIRECT_KERNEL_NAME =
    "spmm_1d_kernel_direct";
constexpr const char *SPMM_1D_DIRECT_BINARY_NAME =
    "spmm_1d_kernel_direct.o";

inline SpmmTilingConfig MakeWorkspaceTiling(
    int32_t aicNBatch,
    int32_t sparseKTile,
    int32_t bPackNBatch,
    int32_t tilingKey,
    const char *variantName)
{
    return {
        aicNBatch,
        sparseKTile,
        bPackNBatch,
        tilingKey,
        variantName,
        tilingKey == 57 ? "spmm_1d_kernel_nmajor" : SPMM_1D_WORKSPACE_KERNEL_NAME,
        tilingKey == 57 ? "spmm_1d_kernel_nmajor.o" : SPMM_1D_WORKSPACE_BINARY_NAME,
    };
}

inline SpmmTilingConfig MakeDirectTiling(
    int32_t aicNBatch,
    int32_t sparseKTile,
    int32_t bPackNBatch,
    int32_t tilingKey,
    const char *variantName)
{
    const bool pipeline = tilingKey == 38 || tilingKey == 39;
    return {
        aicNBatch,
        sparseKTile,
        bPackNBatch,
        tilingKey,
        variantName,
        pipeline ? "spmm_1d_kernel_direct_pipeline" : SPMM_1D_DIRECT_KERNEL_NAME,
        pipeline ? "spmm_1d_kernel_direct_pipeline.o" : SPMM_1D_DIRECT_BINARY_NAME,
    };
}

inline int SpmmFusionGroup(int key) { return key == 48 ? 2 : key == 49 ? 4 : key == 50 ? 8 : 0; }
inline int SpmmFusionCapacity(int group)
{
    if (group == 0) {
        return 0;
    }
    return (64 * group < 2048 / group) ? 64 * group : 2048 / group;
}
inline SpmmTilingConfig MakeFusionTiling(int32_t nt, int32_t kt, int32_t pack, int32_t key, const char *name)
{
    const char *kernel = key == 48 ? "spmm_1d_kernel_fusion2" : key == 49 ? "spmm_1d_kernel_fusion4" : "spmm_1d_kernel_fusion8";
    const char *binary = key == 48 ? "spmm_1d_kernel_fusion2.o" : key == 49 ? "spmm_1d_kernel_fusion4.o" : "spmm_1d_kernel_fusion8.o";
    return {nt,kt,pack,key,name,kernel,binary};
}

inline int64_t SpmmCeilDivPositive(int64_t value, int64_t divisor)
{
    if (value <= 0 || divisor == 0 || divisor < 0) {
        return 0;
    }
    return value / divisor + ((value % divisor) != 0 ? 1 : 0);
}

inline std::optional<SpmmTilingConfig> SpmmTryResidentPolicy(int64_t maxVectorsPerRow, int64_t nColumns)
{
    // Rows of at most 2048 vectors fit in 64 KiB L0B.
    constexpr int64_t residentARowVectors =
        (64 * 1024) / (SPMM_1D_VECTOR_LENGTH * 2);
    if (maxVectorsPerRow > 256 && nColumns >= 256 && nColumns <= 2048) {
        if (maxVectorsPerRow > residentARowVectors && nColumns > 256) {
            return MakeWorkspaceTiling(32, 64, SPMM_1D_BPACK_512_N_BATCH,
                SPMM_TILING_AIVPP_L0CPP_P32_AIC32_K64_BPACK512, "aic32_ktile64_bpack512_aivpp_l0cpp_p32");
        }
        return MakeWorkspaceTiling(16, 128, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_AIVPP_L0CPP_P32_AIC16_K128_BPACK1024, "aic16_ktile128_bpack1024_aivpp_l0cpp_p32");
    }

    return std::nullopt;
}

inline std::optional<SpmmTilingConfig> SpmmTrySinglePanelPolicy(int64_t maxVectorsPerRow, int64_t nColumns)
{
    const int64_t nPack512Panels = SpmmCeilDivPositive(nColumns, SPMM_1D_N_PANEL_512_COLUMNS);
    const bool one512Panel = (nPack512Panels == 1);
    const bool atLeastHalf512Tile = nColumns > SPMM_1D_HALF_AIC32_TILE_COLUMNS;
    // Tiling policy for one 512-column panel.
    if (one512Panel && atLeastHalf512Tile) {
        if (maxVectorsPerRow <= 16) {
            return MakeWorkspaceTiling(32, 64, SPMM_1D_BPACK_512_N_BATCH,
                SPMM_TILING_AIVPP_L0CPP_P32_AIC32_K64_BPACK512, "aic32_ktile64_bpack512_aivpp_l0cpp_p32");
        }
        if (maxVectorsPerRow <= 32) {
            return MakeWorkspaceTiling(32, 64, SPMM_1D_BPACK_512_N_BATCH,
                SPMM_TILING_AIVPP_L0CPP_AIC32_K64_BPACK512, "aic32_ktile64_bpack512_aivpp_l0cpp");
        }
        if (maxVectorsPerRow <= 64) {
            return MakeWorkspaceTiling(16, 128, SPMM_1D_BPACK_512_N_BATCH,
                SPMM_TILING_AIVPP_AIC16_K128_BPACK512, "aic16_ktile128_bpack512_aivpp");
        }
        if (maxVectorsPerRow <= 128) {
            return MakeWorkspaceTiling(16, 128, SPMM_1D_BPACK_512_N_BATCH,
                SPMM_TILING_AIVPP_L0CPP_P32_AIC16_K128_BPACK512, "aic16_ktile128_bpack512_aivpp_l0cpp_p32");
        }
        if (maxVectorsPerRow <= 256 || maxVectorsPerRow > 1024) {
            return MakeWorkspaceTiling(32, 64, SPMM_1D_BPACK_512_N_BATCH,
                SPMM_TILING_AIVPP_L0CPP_P32_AIC32_K64_BPACK512, "aic32_ktile64_bpack512_aivpp_l0cpp_p32");
        }
    }

    return std::nullopt;
}

inline std::optional<SpmmTilingConfig> SpmmTryMediumPolicy(int64_t maxVectorsPerRow, int64_t nColumns)
{
    const int64_t nPack512Panels = SpmmCeilDivPositive(nColumns, SPMM_1D_N_PANEL_512_COLUMNS);
    const int64_t nPack1024Panels = SpmmCeilDivPositive(nColumns, SPMM_1D_N_PANEL_1024_COLUMNS);
    const bool one1024Panel = (nPack1024Panels == 1);
    const bool two1024Panels = (nPack1024Panels == 2);
    const bool fourOrFewer1024Panels = (nPack1024Panels >= 1 && nPack1024Panels <= 4);
    // Very short rows use direct gather for two 1024-column panels.
    if (two1024Panels && maxVectorsPerRow <= 40) {
        return MakeDirectTiling(64, 32, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_DIRECT_AIC64_K32, "aic64_ktile32_direct");
    }

    // Use direct sparse gather for short rows with up to four panels.
    if (nColumns >= SPMM_1D_N_PANEL_1024_COLUMNS &&
        fourOrFewer1024Panels && maxVectorsPerRow <= 32) {
        return MakeDirectTiling(64, 32, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_DIRECT_AIC64_K32, "aic64_ktile32_direct");
    }

    // Use two 512-column packed panels for this width range.
    if (one1024Panel && nPack512Panels == 2 &&
        maxVectorsPerRow > 32 && maxVectorsPerRow <= 128) {
        return MakeWorkspaceTiling(32, 64, SPMM_1D_BPACK_512_N_BATCH,
                SPMM_TILING_AIVPP_L0CPP_P32_AIC32_K64_BPACK512, "aic32_ktile64_bpack512_aivpp_l0cpp_p32");
    }

    if (two1024Panels &&
        maxVectorsPerRow > 40 && maxVectorsPerRow <= 64) {
        return MakeWorkspaceTiling(32, 64, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_L0CPP_AIC32_K64_BPACK1024, "aic32_ktile64_bpack1024_l0cpp");
    }
    if (two1024Panels &&
        maxVectorsPerRow > 64 && maxVectorsPerRow <= 256) {
        return MakeWorkspaceTiling(16, 128, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_AIVPP_L0CPP_P32_AIC16_K128_BPACK1024, "aic16_ktile128_bpack1024_aivpp_l0cpp_p32");
    }

    return std::nullopt;
}

inline std::optional<SpmmTilingConfig> SpmmTryPipelinePolicy(int64_t maxVectorsPerRow, int64_t nColumns)
{
    const int64_t nPack1024Panels = SpmmCeilDivPositive(nColumns, SPMM_1D_N_PANEL_1024_COLUMNS);
    const bool fourToEight1024Panels = (nPack1024Panels >= 4 && nPack1024Panels <= 8);
    const bool fiveToEight1024Panels = (nPack1024Panels >= 5 && nPack1024Panels <= 8);
    // Fused AIV panel and L0C pipelines for medium and large N.
    if (fourToEight1024Panels &&
        maxVectorsPerRow > 160 && maxVectorsPerRow <= 1024) {
        return MakeWorkspaceTiling(16, 128, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_AIVPP_L0CPP_P32_AIC16_K128_BPACK1024, "aic16_ktile128_bpack1024_aivpp_l0cpp_p32");
    }
    if (fourToEight1024Panels &&
        maxVectorsPerRow > 32 && maxVectorsPerRow <= 160) {
        return MakeWorkspaceTiling(16, 128, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_AIVPP_L0CPP_AIC16_K128_BPACK1024, "aic16_ktile128_bpack1024_aivpp_l0cpp");
    }

    // L0C ping-pong for very short rows.
    if (fiveToEight1024Panels && maxVectorsPerRow <= 32) {
        return MakeWorkspaceTiling(32, 64, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_L0CPP_AIC32_K64_BPACK1024, "aic32_ktile64_bpack1024_l0cpp");
    }

    return std::nullopt;
}

inline std::optional<SpmmTilingConfig> SpmmTryWidePolicy(int64_t maxVectorsPerRow, int64_t nColumns)
{
    const int64_t nPack1024Panels = SpmmCeilDivPositive(nColumns, SPMM_1D_N_PANEL_1024_COLUMNS);
    // Large-N policies use larger K tiles or 2048-column B packs.
    if (nPack1024Panels >= 16 &&
        maxVectorsPerRow > 256 && maxVectorsPerRow <= 512) {
        return MakeWorkspaceTiling(8, 256, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_AIVPP_AIC8_K256_BPACK1024, "aic8_ktile256_bpack1024_aivpp");
    }
    if (nPack1024Panels >= 16 &&
        maxVectorsPerRow > 512 && maxVectorsPerRow <= 2048) {
        return MakeWorkspaceTiling(16, 128, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_AIVPP_AIC16_K128_BPACK1024, "aic16_ktile128_bpack1024_aivpp");
    }
    if (nPack1024Panels >= 16 &&
        maxVectorsPerRow > 32 && maxVectorsPerRow <= 256) {
        return MakeWorkspaceTiling(16, 128, SPMM_1D_BPACK_2048_N_BATCH,
                SPMM_TILING_AIVPP_AIC16_K128_BPACK2048, "aic16_ktile128_bpack2048_aivpp");
    }

    return std::nullopt;
}

inline std::optional<SpmmTilingConfig> SpmmTryMidrangePolicy(int64_t maxVectorsPerRow, int64_t nColumns)
{
    const int64_t nPack1024Panels = SpmmCeilDivPositive(nColumns, SPMM_1D_N_PANEL_1024_COLUMNS);
    // Tiling policies for mid-range large-N shapes.
    if (nPack1024Panels >= 8 &&
        maxVectorsPerRow >= 96 && maxVectorsPerRow <= 160) {
        return MakeWorkspaceTiling(16, 128, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_AIVPP_AIC16_K128_BPACK1024, "aic16_ktile128_bpack1024_aivpp");
    }
    if (nPack1024Panels >= 8 &&
        maxVectorsPerRow > 160 && maxVectorsPerRow <= 256) {
        return MakeWorkspaceTiling(8, 256, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_AIVPP_AIC8_K256_BPACK1024, "aic8_ktile256_bpack1024_aivpp");
    }

    return std::nullopt;
}

inline SpmmTilingConfig SpmmFallbackProductionPolicy(int64_t maxVectorsPerRow, int64_t nColumns)
{
    const int64_t nPack1024Panels = SpmmCeilDivPositive(nColumns, SPMM_1D_N_PANEL_1024_COLUMNS);
    if (maxVectorsPerRow <= SPMM_1D_SMALL_ROW_MAX_VECTORS) {
        if (nPack1024Panels <= 1) {
            return MakeWorkspaceTiling(16, 128, SPMM_1D_BPACK_512_N_BATCH,
                SPMM_TILING_WORKSPACE_AIC16_K128_BPACK512, "aic16_ktile128_bpack512");
        }
        return MakeWorkspaceTiling(16, 128, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_WORKSPACE_AIC16_K128_BPACK1024, "aic16_ktile128_bpack1024");
    }

    if (maxVectorsPerRow <= 256) {
        return MakeWorkspaceTiling(8, 256, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_WORKSPACE_AIC8_K256_BPACK1024, "aic8_ktile256");
    }
    if (maxVectorsPerRow <= 512) {
        return MakeWorkspaceTiling(4, 512, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_WORKSPACE_AIC4_K512_BPACK1024, "aic4_ktile512");
    }
    if (maxVectorsPerRow <= 1024) {
        return MakeWorkspaceTiling(2, 1024, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_WORKSPACE_AIC2_K1024_BPACK1024, "aic2_ktile1024");
    }
    return MakeWorkspaceTiling(1, 2048, SPMM_1D_BPACK_1024_N_BATCH,
                SPMM_TILING_WORKSPACE_AIC1_K2048_BPACK1024, "aic1_ktile2048");
}

inline SpmmTilingConfig SelectSpmmProductionTilingConfig(int64_t maxVectorsPerRow, int64_t nColumns)
{
    for (auto select : {SpmmTryResidentPolicy, SpmmTrySinglePanelPolicy, SpmmTryMediumPolicy,
                        SpmmTryPipelinePolicy, SpmmTryWidePolicy, SpmmTryMidrangePolicy}) {
        const auto result = select(maxVectorsPerRow, nColumns);
        if (result.has_value()) {
            return *result;
        }
    }
    return SpmmFallbackProductionPolicy(maxVectorsPerRow, nColumns);
}

inline int64_t SpmmAlignUp(int64_t value, int64_t alignment)
{
    if (value < 0 || alignment == 0 || alignment < 0 ||
        value > std::numeric_limits<int64_t>::max() - (alignment - 1)) {
        return -1;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

inline int64_t GetSpmmPaddedN(int64_t n, const SpmmTilingConfig &tiling)
{
    return SpmmAlignUp(
        n,
        static_cast<int64_t>(tiling.aic_n_batch) * SPMM_1D_VECTOR_LENGTH);
}

// Resolve padding before returning the configuration. The caller pads using
// this result; selecting again on that width must produce the same key and
// workspace shape (e.g. short rows at N=900 resolve directly to key 9).
inline SpmmTilingConfig SelectSpmmTilingConfig(
    int64_t maxVectorsPerRow,
    int64_t nColumns)
{
    const SpmmTilingConfig initial =
        SelectSpmmProductionTilingConfig(maxVectorsPerRow, nColumns);
    const int64_t paddedN = GetSpmmPaddedN(nColumns, initial);
    if (paddedN <= 0) {
        return initial;
    }
    return SelectSpmmProductionTilingConfig(maxVectorsPerRow, paddedN);
}

#endif  // ACLSPARSE_SPMM_1D_INTERNAL_H_
