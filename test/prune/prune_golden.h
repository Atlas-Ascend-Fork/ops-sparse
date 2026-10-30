/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_PRUNE_GOLDEN_H_
#define TEST_PRUNE_GOLDEN_H_

// =============================================================================
// CPU Golden reference for aclsparseLtSpMMAPrune (prune-only, no matmul).
//
// Formula: A_pruned = StructuredPrune(A)
//   - A: (m, k) row-major, structured 50% pruned
//   - A_pruned: (m, k) row-major, same shape
//
// Two prune algorithms are supported:
//   STRIP (1D top-N per group): existing, group=4 keep=2 for FP16, group=2 keep=1 for FP32.
//   TILE  (2D row/col joint constraint): new, 4x4 tile 2:2 for FP16 (90 configs),
//        2x2 tile 1:1 for FP32 (2 configs). Edge tiles (m or k not TS-multiple)
//        degenerate to STRIP.
//
// This header is a CPU reference used only for test comparison. It does NOT
// depend on any aclsparse/aclsparseLt API and can be compiled independently.
// FP16<->FP32 conversion reuses the IEEE754 round-to-nearest-even helpers
// validated in test/spmm/arch35/spmm_test.cpp.
// =============================================================================

#include <array>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

// Reuse shared FP16<->FP32 conversion helpers, dtype traits, and structured
// pruning primitives (PruneRowFp32 / PruneColFp32 / PickLargestAbs) from the
// local prune_test_util.h (decoupled from alg_set_attribute / matmul test headers).
#include "prune_test_util.h"

namespace sparse_test {

// -----------------------------------------------------------------------------
// Trait alias: PruneDtypeTrait reuses the shared AlgSetAttrDtypeTrait defined
// in alg_set_attribute_golden.h. Kept as a template alias so existing prune
// golden code (SpMMAPruneGolden / PruneToFloat) does not need to change its
// trait references.
// -----------------------------------------------------------------------------
template <typename T>
using PruneDtypeTrait = AlgSetAttrDtypeTrait<T>;

// =============================================================================
// TILE mode: 2D tile-level structured pruning with row/col joint constraint.
//
// For each TS×TS tile, enumerate all valid configurations (each row keeps
// exactly KEEP elements, each column keeps exactly KEEP elements) and select
// the one maximizing L1-norm (sum of absolute values of kept elements).
// Tie-breaking: strict > (first max wins), consistent with NPU kernel.
//
// Edge handling: incomplete tiles (m or k not TS-multiple) degenerate to STRIP
// (1D top-N), reusing PruneRowFp32 / PruneColFp32.
// =============================================================================

// -----------------------------------------------------------------------------
// Config table generation (runtime, generated once via static local).
//
// Enumerates all valid TS×TS configurations satisfying the row/col joint
// constraint. Each config is stored as std::array<uint8_t, TS> where element r
// is the column bitmask for row r (bit c = 1 means column c is kept).
//
// Generation logic:
//   1. Collect all TS-bit row masks with exactly KEEP bits set (C(TS,KEEP) values,
//      in ascending numeric order).
//   2. Enumerate all (rowMasks)^TS combinations with row 0 as the most
//      significant (slowest-varying) digit and row TS-1 as the least
//      significant (fastest-varying) digit. This matches the kernel's
//      precomputed lookup table enumeration order (kTile4x4Configs /
//      kTile2x2Configs in prune_kernel.cpp).
//   3. Filter: each column must have exactly KEEP bits set across all TS rows.
//
// Result: 90 configs for FP16 (TS=4, KEEP=2), 2 configs for FP32 (TS=2, KEEP=1).
//
// IMPORTANT: the enumeration order and tie-breaking rule (strict >) must match
// the NPU kernel exactly for element-wise comparison to pass. The kernel uses
// a precomputed lookup table generated with the same row 0 = most significant
// convention.
// -----------------------------------------------------------------------------
template <int TS, int KEEP>
inline std::vector<uint8_t> GenerateRowMasks()
{
    std::vector<uint8_t> rowMasks;
    for (int mask = 0; mask < (1 << TS); ++mask) {
        if (__builtin_popcount(static_cast<uint32_t>(mask)) == KEEP) {
            rowMasks.push_back(static_cast<uint8_t>(mask));
        }
    }
    return rowMasks;
}

template <int TS>
inline int CountColBits(const std::array<uint8_t, TS>& cfg, int c)
{
    int colSum = 0;
    for (int r = 0; r < TS; ++r) {
        if (cfg[r] & (1u << c)) { ++colSum; }
    }
    return colSum;
}

template <int TS, int KEEP>
inline bool ConfigSatisfiesCols(const std::array<uint8_t, TS>& cfg)
{
    for (int c = 0; c < TS; ++c) {
        if (CountColBits<TS>(cfg, c) != KEEP) { return false; }
    }
    return true;
}

template <int TS, int KEEP>
inline std::vector<std::array<uint8_t, TS>> EnumerateValidConfigs(
    const std::vector<uint8_t>& rowMasks)
{
    std::vector<std::array<uint8_t, TS>> valid;
    const int nMasks = static_cast<int>(rowMasks.size());
    int total = 1;
    for (int r = 0; r < TS; ++r) { total *= nMasks; }

    std::array<uint8_t, TS> cfg{};
    for (int idx = 0; idx < total; ++idx) {
        int tmp = idx;
        for (int r = TS - 1; r >= 0; --r) {
            cfg[r] = rowMasks[tmp % nMasks];
            tmp /= nMasks;
        }
        if (ConfigSatisfiesCols<TS, KEEP>(cfg)) { valid.push_back(cfg); }
    }
    return valid;
}

template <int TS, int KEEP>
inline const std::vector<std::array<uint8_t, TS>>& GetTileConfigs()
{
    static const auto configs = []() {
        auto rowMasks = GenerateRowMasks<TS, KEEP>();
        return EnumerateValidConfigs<TS, KEEP>(rowMasks);
    }();
    return configs;
}

// -----------------------------------------------------------------------------
// PruneTileBlock: prune a single TS×TS tile using the config table.
//
// Inputs:
//   tileIn  - TS×TS sub-block from Af, row-major with stride rowStride (= k).
//   configs - precomputed valid configuration table.
// Output:
//   tileOut - TS×TS pruned block, same layout. ALL TS*TS elements are written
//             (selected elements keep their value, others are zeroed).
//
// Selection: enumerate all configs, compute L1-norm = sum of |tileIn[r][c]|
// for kept elements. Pick the config with maximum L1-norm.
// Tie-breaking: strict > (first max wins), matching NPU kernel.
// -----------------------------------------------------------------------------
template <int TS, int KEEP>
inline void PruneTileBlock(const float* tileIn, float* tileOut, int32_t rowStride,
                           const std::vector<std::array<uint8_t, TS>>& configs)
{
    float bestSum = -1.0f;
    size_t bestIdx = 0;
    for (size_t ci = 0; ci < configs.size(); ++ci) {
        const auto& cfg = configs[ci];
        float sum = 0.0f;
        for (int r = 0; r < TS; ++r) {
            const float* row = tileIn + static_cast<int64_t>(r) * rowStride;
            for (int c = 0; c < TS; ++c) {
                if (cfg[r] & (1u << c)) {
                    sum += std::fabs(row[c]);
                }
            }
        }
        if (sum > bestSum) {  // strict >: first max wins (tie-break with kernel)
            bestSum = sum;
            bestIdx = ci;
        }
    }

    const auto& best = configs[bestIdx];
    for (int r = 0; r < TS; ++r) {
        float* row = tileOut + static_cast<int64_t>(r) * rowStride;
        for (int c = 0; c < TS; ++c) {
            row[c] = (best[r] & (1u << c)) ? tileIn[static_cast<int64_t>(r) * rowStride + c]
                                           : 0.0f;
        }
    }
}

// -----------------------------------------------------------------------------
// PruneTileLoop: overwrite complete tile regions with TILE results.
//
// Shared by PruneTileRowFp32 and PruneTileColFp32 to avoid code duplication.
// Iterates over all TS×TS tiles in the complete region (i < m/TS*TS, j < k/TS*TS)
// and applies PruneTileBlock using the precomputed config table.
// -----------------------------------------------------------------------------
template <int TS, int KEEP>
inline void PruneTileLoop(const std::vector<float>& Af, std::vector<float>& Af_pruned,
                           int32_t m, int32_t k)
{
    const auto& configs = GetTileConfigs<TS, KEEP>();
    const int32_t tileRows = m / TS;
    const int32_t tileCols = k / TS;
    for (int32_t ti = 0; ti < tileRows; ++ti) {
        for (int32_t tj = 0; tj < tileCols; ++tj) {
            const int32_t baseRow = ti * TS;
            const int32_t baseCol = tj * TS;
            const float* tileIn = Af.data() + static_cast<int64_t>(baseRow) * k + baseCol;
            float* tileOut = Af_pruned.data() + static_cast<int64_t>(baseRow) * k + baseCol;
            PruneTileBlock<TS, KEEP>(tileIn, tileOut, k, configs);
        }
    }
}

// -----------------------------------------------------------------------------
// PruneTileRowFp32: TILE pruning with row-direction edge degradation.
//
// On Af(m, k) row-major:
//   1. Full STRIP pass (PruneRowFp32): handles edge rows (m % TS tail) and
//      edge columns (k % TS tail groups zeroed).
//   2. Overwrite complete tile regions (i < m/TS*TS, j < k/TS*TS) with TILE
//      results (PruneTileBlock).
//
// Note: TILE optimization result is direction-independent (same L1-norm max
// config for a given tile). Direction only affects edge STRIP direction.
// -----------------------------------------------------------------------------
template <int TS, int KEEP>
inline void PruneTileRowFp32(const std::vector<float>& Af, std::vector<float>& Af_pruned,
                               int32_t m, int32_t k)
{
    PruneRowFp32(Af, Af_pruned, m, k, TS, KEEP);
    PruneTileLoop<TS, KEEP>(Af, Af_pruned, m, k);
}

// -----------------------------------------------------------------------------
// PruneTileColFp32: TILE pruning with column-direction edge degradation.
//
// On Af(m, k) row-major:
//   1. Full STRIP pass (PruneColFp32): handles edge rows and edge columns.
//   2. Overwrite complete tile regions with TILE results (PruneTileBlock).
//
// Same TILE result as PruneTileRowFp32 for complete tiles; differs only in
// edge STRIP direction (column-wise vs row-wise).
// -----------------------------------------------------------------------------
template <int TS, int KEEP>
inline void PruneTileColFp32(const std::vector<float>& Af, std::vector<float>& Af_pruned,
                               int32_t m, int32_t k)
{
    PruneColFp32(Af, Af_pruned, m, k, TS, KEEP);
    PruneTileLoop<TS, KEEP>(Af, Af_pruned, m, k);
}

// -----------------------------------------------------------------------------
// GoldenTranspose: (m,k) row-major -> (k,m) row-major transpose (pure CPU
// function). Used for the golden same-view treatment of ColView combinations
// (plan section 2.1.3: TILE tie-breaking must enumerate configs on the same
// physical view as the kernel).
// -----------------------------------------------------------------------------
inline std::vector<float> GoldenTranspose(const std::vector<float>& a,
                                          int32_t rows, int32_t cols)
{
    std::vector<float> t(static_cast<size_t>(cols) * static_cast<size_t>(rows));
    for (int32_t i = 0; i < rows; ++i) {
        for (int32_t j = 0; j < cols; ++j) {
            t[static_cast<size_t>(j) * static_cast<size_t>(rows) + static_cast<size_t>(i)] =
                a[static_cast<size_t>(i) * static_cast<size_t>(cols) + static_cast<size_t>(j)];
        }
    }
    return t;
}

// -----------------------------------------------------------------------------
// Golden entry: A_pruned = StructuredPrune(A)
//
// Template parameter T:
//   float    -> FP32 input/output (1:2 sparsity for STRIP, 2x2 tile 1:1 for TILE)
//   uint16_t -> FP16 bit-pattern input/output (2:4 sparsity for STRIP, 4x4 tile 2:2 for TILE)
//
// pruneAlg:
//   "STRIP" (default) -> 1D top-N per group (existing behavior, backward compatible)
//   "TILE"            -> 2D tile-level pruning with row/col joint constraint;
//                        edge tiles degenerate to STRIP
//
// The prune operation is deterministic and independent of alg_config_id /
// split_k (those only influence NPU matmul algorithm selection); they are
// accepted to align with the operator API signature but unused here.
//
// Direction logic ((b) cuSPARSELt semantics: grouping always runs along
// op-view rows, i.e. the rows of logical (m,k)):
//   RowView (alongRow holds when transA and isRowOrder differ, N+ROW /
//     T+COL): the kernel processes on the (m,k) physical view -> golden
//     uses the Row variant directly (behavior unchanged).
//   ColView (alongRow does not hold, N+COL / T+ROW): the kernel processes
//     on the (k,m) physical view -> golden transposes to (k,m) then uses
//     the Col variant (same-view enumeration guarantees TILE tie-breaking
//     is bit-identical to the kernel), and the result is transposed back
//     to (m,k). Under STRIP this is mathematically equivalent to grouping
//     along logical rows (identical group sets).
// Input/output are both logical (m,k) row-major compact (physical layout
// scattering is carried by the comparison side).
// -----------------------------------------------------------------------------
template <typename T>
void SpMMAPruneGolden(
    const std::vector<T>& A, std::vector<T>& A_pruned,
    int32_t m, int32_t k,
    int32_t alg_config_id, int32_t split_k,
    bool transA, bool isRowOrder,
    const std::string& pruneAlg = "STRIP")
{
    (void)alg_config_id;  // prune is independent of algorithm selection
    (void)split_k;        // prune is independent of split-K

    using Trait = PruneDtypeTrait<T>;

    // Step 1: promote input A to FP32 (row-major).
    const size_t mk = static_cast<size_t>(m) * static_cast<size_t>(k);
    std::vector<float> Af(mk);
    for (size_t i = 0; i < mk; ++i) Af[i] = Trait::toFp32(A[i]);

    // Step 2: structured prune on A (FP32), dtype-aware + direction-aware.
    // FP32 -> TS=2, KEEP=1; FP16/BF16/INT8 -> TS=4, KEEP=2.
    // Matches kernel: groupSize = std::is_same_v<T, float> ? 2 : 4.
    const int32_t ts = std::is_same_v<T, float> ? 2 : 4;
    const int32_t keep = std::is_same_v<T, float> ? 1 : 2;
    const bool alongRow = (transA != isRowOrder);

    std::vector<float> Af_pruned;
    const bool isTile = (pruneAlg == "TILE");
    if (alongRow) {
        // RowView: the kernel processes on the (m,k) view (existing
        // behavior unchanged).
        if (isTile) {
            if constexpr (!std::is_same_v<T, float>) {
                PruneTileRowFp32<4, 2>(Af, Af_pruned, m, k);
            } else {
                PruneTileRowFp32<2, 1>(Af, Af_pruned, m, k);
            }
        } else {
            PruneRowFp32(Af, Af_pruned, m, k, ts, keep);
        }
    } else {
        // ColView: the kernel processes on the (k,m) physical view ->
        // golden same-view treatment (transpose + Col variant + transpose
        // back).
        std::vector<float> At = GoldenTranspose(Af, m, k);          // (k,m)
        std::vector<float> At_pruned;
        if (isTile) {
            if constexpr (!std::is_same_v<T, float>) {
                PruneTileColFp32<4, 2>(At, At_pruned, k, m);
            } else {
                PruneTileColFp32<2, 1>(At, At_pruned, k, m);
            }
        } else {
            PruneColFp32(At, At_pruned, k, m, ts, keep);
        }
        Af_pruned = GoldenTranspose(At_pruned, k, m);               // back to (m,k)
    }

    // Step 3: truncate back to T.
    A_pruned.resize(mk);
    for (size_t i = 0; i < mk; ++i) {
        A_pruned[i] = Trait::fromFp32(Af_pruned[i]);
    }
}

// -----------------------------------------------------------------------------
// Helper: convert a T storage vector to FP32 for verification comparison.
// -----------------------------------------------------------------------------
template <typename T>
inline std::vector<float> PruneToFloat(const std::vector<T>& v)
{
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = PruneDtypeTrait<T>::toFp32(v[i]);
    }
    return out;
}

}  // namespace sparse_test

#endif  // TEST_PRUNE_GOLDEN_H_
