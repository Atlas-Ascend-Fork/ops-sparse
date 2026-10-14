/**
 * Copyright (c) 2026 liutaw.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSM_ARCH22_KERNEL_H
#define SPSM_ARCH22_KERNEL_H
#include <cstdint>
#include "spsm_tiling_data.h"
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif
extern "C" void SpsmA2CompensatedLaunch(GM_ADDR offsets, GM_ADDR indices, GM_ADDR values,
    GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR status, GM_ADDR lowFirst, GM_ADDR lowTail,
    int64_t lowRows, GM_ADDR levelOrder, GM_ADDR levelOffsets,
    const SpsmA2TilingData &tiling, uint32_t blocks, void *stream);
extern "C" void SpsmA2LevelsLaunch(GM_ADDR offsets, GM_ADDR columns, GM_ADDR order,
    GM_ADDR levelOffsets, int32_t rows, int32_t upper, void *stream);
// Rows of a diagonal-only canonical matrix carry no mutual dependency, so this
// variant parallelises over rows as well as RHS tiles. Arithmetic is unchanged.
extern "C" void SpsmA2DiagonalLaunch(GM_ADDR offsets, GM_ADDR values,
    GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR status, const SpsmA2TilingData &tiling,
    uint32_t blocks, void *stream);

// Internal Device preprocessing stage. Original entry order is retained here;
// deterministic sorting and duplicate reduction consume these COO work arrays.
// Caller provides distinct, nnz-sized row/column/value output arrays.

extern "C" void SpsmA2ExpandLaunch(GM_ADDR first, GM_ADDR second, GM_ADDR values,
    GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outValues, GM_ADDR status,
    const SpsmA2FormatTilingData &tiling, void *stream);

// A stable pass over (row,col), retaining original position and value bits.
// run=0 sorts independent 512-entry tiles and generates original positions.
// run>=512 merges adjacent sorted runs. All output arrays are disjoint from
// the inputs; the caller ping-pongs two sets until run >= nnz.

extern "C" void SpsmA2SortPassLaunch(GM_ADDR rows, GM_ADDR cols, GM_ADDR values, GM_ADDR order,
    GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outValues, GM_ADDR outOrder, GM_ADDR status,
    const SpsmA2SortTilingData &tiling, void *stream);

// Input is validated COO, stably sorted by (row,col). Output CSR is base zero.
// Every output and scratch array is distinct from the inputs. Allocate rows+1
// entries for rowStarts/rowPtr, rows counts, and nnz capacity for other outputs.
// groupStarts/groupEnds refer to half-open spans in the sorted input, allowing
// original-position mappings to be retained for subsequent value refreshes.

extern "C" void SpsmA2CanonicalLaunch(GM_ADDR rows, GM_ADDR cols, GM_ADDR values,
    GM_ADDR rowStarts, GM_ADDR rowCounts, GM_ADDR rowPtr, GM_ADDR outCols, GM_ADDR outValues,
    GM_ADDR groupStarts, GM_ADDR groupEnds, GM_ADDR status,
    const SpsmA2CanonicalTilingData &tiling, void *stream);

// Both arrays use original entry order, with op(A) already applied. GENERAL
// copies expanded values to the snapshot. DIAGONAL replaces only row==col,
// retaining current off-diagonals, and writes both snapshot and expanded values.
// snapshot is distinct from expanded values and is initialized by GENERAL.

extern "C" void SpsmA2RefreshLaunch(GM_ADDR rows, GM_ADDR cols, GM_ADDR values,
    GM_ADDR snapshot, GM_ADDR status, const SpsmA2RefreshTilingData &tiling, void *stream);

// Internal copy: source and destination regions must not overlap. The public
// in-place solve path snapshots op(B) into workspace before copying back to C.
// Stride byte gaps must fit the DataCopyPad uint32_t stride fields.
extern "C" void SpsmA2DenseLaunch(GM_ADDR source, GM_ADDR destination, GM_ADDR status,
    const SpsmA2DenseTilingData &tiling, uint32_t blocks, void *stream);

// Rebuilds the canonical values of a duplicate-free matrix from the freshly expanded
// values and the permutation saved during Analysis, replacing the sort and merge
// passes UpdateMatrix would otherwise repeat. Each block owns whole rows, so the
// NON_UNIT diagonal check only inspects values that same block has just written.
extern "C" void SpsmA2GatherLaunch(GM_ADDR perm, GM_ADDR source, GM_ADDR cols, GM_ADDR rowPtr,
    GM_ADDR output, GM_ADDR status, const SpsmA2GatherTilingData &tiling, uint32_t blocks, void *stream);

// The status slot's first word retains the existing preparation error contract.
// +1 marks non-finite input, +2 packed failure, +3 legacy overflow. Guard resets all on
// every Solve; consumers run later on the same stream without a host readback.
extern "C" void SpsmA2PackedGuardLaunch(GM_ADDR offsets, GM_ADDR values, GM_ADDR dense,
    GM_ADDR alpha, GM_ADDR status, const SpsmA2TilingData &tiling, uint32_t blocks, void *stream);
extern "C" void SpsmA2PackedSolveLaunch(GM_ADDR offsets, GM_ADDR indices, GM_ADDR values,
    GM_ADDR dense, GM_ADDR alpha, GM_ADDR status, GM_ADDR first, GM_ADDR tail, int64_t lowRows,
    int32_t begin, int32_t end, const SpsmA2TilingData &tiling, uint32_t blocks, void *stream);
extern "C" void SpsmA2PackedConvertLaunch(GM_ADDR dense, GM_ADDR first, GM_ADDR tail,
    GM_ADDR status, int64_t lowRows, const SpsmA2TilingData &tiling, uint32_t blocks, void *stream);
extern "C" void SpsmA2PackedRetryLaunch(GM_ADDR dense, GM_ADDR first, GM_ADDR tail,
    GM_ADDR status, int64_t lowRows, const SpsmA2TilingData &tiling, uint32_t blocks, void *stream);
#endif
