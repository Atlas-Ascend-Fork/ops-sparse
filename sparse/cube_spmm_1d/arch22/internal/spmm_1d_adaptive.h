/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSPARSE_SPMM_1D_ADAPTIVE_H_
#define ACLSPARSE_SPMM_1D_ADAPTIVE_H_
#include <algorithm>
#include <vector>
#include "spmm_1d_structural.h"

// Calibration parameters are separate from the capacity-derived group limits.
// Planning reads shape/row pointers only, never matrix identities or timings.
struct SpmmAdaptiveParameters {
    int minGroupsPerCore = 2;
    int minTileSavingNumerator = 1, minTileSavingDenominator = 4;
    int slabBudgetNumerator = 1, slabBudgetDenominator = 1;
    int originalCoreBudgetNumerator = 3, originalCoreBudgetDenominator = 4;
    int coverageNumerator = 9, coverageDenominator = 10;
    int maxLoadNumerator = 5, maxLoadDenominator = 4;
    // Measured by aclrtGetDeviceInfo(ACL_DEV_ATTR_L2_CACHE_SIZE). This is
    // a crossover proxy, not a claim that every byte is cache-resident.
    int64_t denseWorkingSetBytes = 192LL * 1024 * 1024;
};

inline int SpmmFusionKey(int group) { return group == 2 ? 48 : group == 4 ? 49 : 50; }

struct SpmmAdaptiveLoad {
    std::vector<int64_t> vectors, rows, tiles;
    int64_t fusedRows = 0;
    explicit SpmmAdaptiveLoad(int cores) : vectors(cores), rows(cores), tiles(cores) {}
};

inline int64_t SpmmAdaptiveMax(const std::vector<int64_t>& values)
{
    return *std::max_element(values.begin(), values.end());
}

inline SpmmAdaptiveLoad SpmmOriginalLoad(int m, const int32_t* offsets, int cores)
{
    SpmmAdaptiveLoad load(std::max(cores, 0));
    if (cores == 0 || cores < 0) return load;
    for (int row = 0; row < m / 16; ++row) {
        const int64_t length = static_cast<int64_t>(offsets[row + 1]) - offsets[row];
        load.vectors[row % cores] += length;
        load.rows[row % cores] += length > 0;
    }
    return load;
}

inline bool SpmmUseNMajorSlabs(int m, int n, int k, int64_t pmax, int64_t total,
                               int cores, const SpmmAdaptiveLoad& original,
                               const SpmmAdaptiveParameters& parameters)
{
    if (n == 0 || cores == 0 || parameters.slabBudgetDenominator == 0 ||
        parameters.originalCoreBudgetDenominator == 0) return false;
    int64_t originalCoreBytesPerColumn = 0;
    for (int core = 0; core < cores; ++core) {
        originalCoreBytesPerColumn = std::max(originalCoreBytesPerColumn,
            2 * original.vectors[core] + 64 * original.rows[core]);
    }
    // Full slab address domain, per-core rings and A/index state are cache-reuse
    // proxies, not claims of residency or a private L2 partition per core.
    const int64_t budget = parameters.denseWorkingSetBytes /
        parameters.slabBudgetDenominator * parameters.slabBudgetNumerator;
    const int64_t bytesPerColumn = 2LL * k + 4LL * m;
    const int64_t q = (pmax + 15) / 16 * 16;
    const int64_t originalCoreBudget = parameters.denseWorkingSetBytes /
        parameters.originalCoreBudgetDenominator * parameters.originalCoreBudgetNumerator;
    const bool originalCoreLarge = originalCoreBytesPerColumn >=
        originalCoreBudget / n + (originalCoreBudget % n != 0);
    if (originalCoreLarge && n > 1024 && static_cast<int64_t>(k) * 2 >=
            parameters.denseWorkingSetBytes / n + (parameters.denseWorkingSetBytes % n != 0) &&
        bytesPerColumn <= budget / 1024 && total <= budget / 36) {
        const int64_t remaining = budget - bytesPerColumn * 1024 - total * 36;
        return q <= remaining / (4LL * 1024 * cores) && m / 16 >= cores * parameters.minGroupsPerCore;
    }
    return false;
}

inline int64_t SpmmFusionTileCount(int n, int64_t length, int64_t maxWidth)
{
    if (length == 0) return 0;
    const int64_t rounded = (length + 15) / 16 * 16;
    if (rounded == 0) return 0;
    const int64_t tile = std::min<int64_t>(maxWidth, 32768 / rounded / 16 * 16);
    if (tile == 0) return 0;
    return (n + tile - 1) / tile;
}

inline SpmmAdaptiveLoad SpmmFusionLoad(int rows, int n, const int32_t* offsets, int cores, int group)
{
    SpmmAdaptiveLoad load(std::max(cores, 0));
    if (cores == 0 || cores < 0 || group == 0 || group < 0) return load;
    for (int first = 0; first < rows; first += group) {
        const int stop = std::min(first + group, rows);
        const int core = (first / group) % cores;
        const int64_t length = static_cast<int64_t>(offsets[stop]) - offsets[first];
        bool allNonempty = true;
        for (int row = first; row < stop; ++row) {
            const bool populated = offsets[row] != offsets[row + 1];
            allNonempty = allNonempty && populated;
            load.rows[core] += populated;
        }
        load.vectors[core] += length;
        const bool fuse = stop - first == group && allNonempty && length <= SpmmFusionCapacity(group);
        if (fuse) {
            load.fusedRows += group;
            load.tiles[core] += SpmmFusionTileCount(n, length, 512 / group);
        } else {
            for (int row = first; row < stop; ++row) {
                const int64_t len = static_cast<int64_t>(offsets[row + 1]) - offsets[row];
                load.tiles[core] += SpmmFusionTileCount(n, len, 512);
            }
        }
    }
    return load;
}

inline int SpmmLargestFusionGroup(int m, int n, int k, const SpmmAdaptiveParameters& parameters)
{
    if (n == 0) return 0;
    const int64_t bytesPerColumn = static_cast<int64_t>(k) * 2 + static_cast<int64_t>(m) * 4;
    const bool largeWorkingSet = bytesPerColumn >= parameters.denseWorkingSetBytes / n +
        (parameters.denseWorkingSetBytes % n != 0);
    return largeWorkingSet ? 8 : (n <= 512 ? 4 : 2);
}

inline bool SpmmFusionCoverageFits(int64_t nonempty, int n, int baseTile, bool lightDirect,
                                   const SpmmAdaptiveLoad& load, const SpmmAdaptiveLoad& original,
                                   const SpmmAdaptiveParameters& parameters)
{
    if (load.fusedRows * parameters.coverageDenominator >= nonempty * parameters.coverageNumerator) return true;
    if (!lightDirect || baseTile == 0) return false;
    const int64_t oldTiles = SpmmAdaptiveMax(original.rows) * ((static_cast<int64_t>(n) + baseTile - 1) / baseTile);
    const int64_t newTiles = SpmmAdaptiveMax(load.tiles);
    return newTiles * parameters.minTileSavingDenominator <= oldTiles *
        (parameters.minTileSavingDenominator - parameters.minTileSavingNumerator);
}

inline bool SpmmFusionBalanceFits(const SpmmAdaptiveLoad& load, const SpmmAdaptiveLoad& original,
                                  const SpmmAdaptiveParameters& parameters)
{
    return SpmmAdaptiveMax(load.vectors) * parameters.maxLoadDenominator <=
               SpmmAdaptiveMax(original.vectors) * parameters.maxLoadNumerator &&
           SpmmAdaptiveMax(load.rows) * parameters.maxLoadDenominator <=
               SpmmAdaptiveMax(original.rows) * parameters.maxLoadNumerator;
}

inline SpmmStructuralDecision SpmmSelectAdaptive(
    int m, int n, int k, int64_t pmax, int64_t total, int64_t empty,
    const int32_t* offsets, int cores,
    const SpmmAdaptiveParameters& parameters = SpmmAdaptiveParameters())
{
    auto fallback = SpmmSelectStructural(m, n, k, pmax, total, empty);
    if (cores == 0 || cores < 0 || n == 0 || n < 0 || total <= 0) return fallback;
    const auto original = SpmmOriginalLoad(m, offsets, cores);
    if (SpmmUseNMajorSlabs(m, n, k, pmax, total, cores, original, parameters)) {
        fallback = {57, false, 40};
    }
    const int rows = m / 16;
    const int64_t nonempty = rows - empty;
    if (pmax > 256 || nonempty <= 0) return fallback;
    const auto* baseCandidate = SpmmFindCandidate(fallback.key);
    const int baseTile = baseCandidate->tiling.aic_n_batch * 16;
    const bool lightDirect = fallback.key == 39 && pmax <= 16;
    for (int group = SpmmLargestFusionGroup(m, n, k, parameters); group >= 2; group /= 2) {
        if (static_cast<int64_t>(rows) < static_cast<int64_t>(group) * cores * parameters.minGroupsPerCore) continue;
        const auto load = SpmmFusionLoad(rows, n, offsets, cores, group);
        if (!SpmmFusionCoverageFits(nonempty, n, baseTile, lightDirect, load, original, parameters)) continue;
        if (!SpmmFusionBalanceFits(load, original, parameters)) continue;
        return {SpmmFusionKey(group), true, 20 + group};
    }
    return fallback;
}
#endif
