/**
 * Copyright (c) 2026 liutaw.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "spsm_plan.h"
#include <algorithm>
#include <limits>
#include <vector>
#include "acl/acl.h"
#include "spsm_kernel.h"
namespace spsm_a2 {
namespace {
constexpr size_t kAlignment = 64;
bool Append(size_t count, size_t width, size_t &cursor, size_t &offset)
{
    const size_t limit = std::numeric_limits<size_t>::max();
    if (width != 0 && count > (limit - kAlignment + 1) / width) { return false; }
    size_t bytes = count * width;
    bytes = bytes == 0 ? kAlignment : (bytes + kAlignment - 1) / kAlignment * kAlignment;
    if (cursor > limit - bytes) { return false; }
    offset = cursor; cursor += bytes; return true;
}
bool AppendSet(size_t nnz, size_t parts, size_t &cursor, WorkspaceSet &set)
{
    return Append(nnz, 4, cursor, set.rows) && Append(nnz, 4, cursor, set.cols) &&
        Append(nnz, parts * 4, cursor, set.values) && Append(nnz, 4, cursor, set.order);
}
bool ValidCountsAndEnums(const PlanConfig &c)
{
    if (c.rows < 0 || c.rhs < 0 || c.nnz < 0 || c.blocks < 1 || c.blocks > 1024) { return false; }
    if (c.base < 0 || c.base > 1 || c.format < 0 || c.format > 2 ||
        c.opA < 0 || c.opA > 2 || c.opB < 0 || c.opB > 2) { return false; }
    return true;
}
bool ValidFlags(const PlanConfig &c)
{
    if (c.upper < 0 || c.upper > 1 || c.unit < 0 || c.unit > 1 ||
        c.complex < 0 || c.complex > 1 || c.deviceAlpha < 0 || c.deviceAlpha > 1) { return false; }
    return true;
}
// Requires rows != 0 and rhs != 0; divisors are the positive row count and strides.
bool ValidStrides(const PlanConfig &c)
{
    if (c.bRowStride < 1 || c.bColStride < 1 || c.cRowStride < 1 || c.cColStride < 1) { return false; }
    uint64_t bytes = c.complex ? 8 : 4;
    if (static_cast<uint64_t>(c.bColStride - 1) > UINT32_MAX / bytes ||
        static_cast<uint64_t>(c.cColStride - 1) > UINT32_MAX / bytes) { return false; }
    const uint64_t maxElements = INT64_MAX / bytes;
    if (static_cast<uint64_t>(c.bRowStride) > maxElements / static_cast<uint32_t>(c.rows) ||
        static_cast<uint64_t>(c.cRowStride) > maxElements / static_cast<uint32_t>(c.rows)) { return false; }
    uint64_t bLast = static_cast<uint64_t>(c.rows - 1) * c.bRowStride;
    uint64_t cLast = static_cast<uint64_t>(c.rows - 1) * c.cRowStride;
    if (static_cast<uint64_t>(c.rhs - 1) > (maxElements - bLast) / c.bColStride ||
        static_cast<uint64_t>(c.rhs - 1) > (maxElements - cLast) / c.cColStride) { return false; }
    return true;
}
bool ValidConfig(const PlanConfig &c)
{
    if (!ValidCountsAndEnums(c)) { return false; }
    if (!ValidFlags(c)) { return false; }
    if (c.rows == 0 && c.nnz != 0) { return false; }
    if (c.rows != 0 && c.rhs != 0) { return ValidStrides(c); }
    return true;
}
uint8_t *At(void *workspace, size_t offset) { return static_cast<uint8_t *>(workspace) + offset; }
WorkspaceSet SortedSet(const PlanConfig &config, const WorkspaceLayout &layout)
{
    WorkspaceSet current = layout.b;
    bool isB = true;
    for (int64_t run = 512; run < config.nnz; run *= 2) { isB = !isB; }
    current = isB ? layout.b : layout.a; return current;
}
WorkspaceSet OutputSet(const PlanConfig &config, const WorkspaceLayout &layout)
{
    auto sorted = SortedSet(config, layout);
    return sorted.rows == layout.a.rows ? layout.b : layout.a;
}
void Sort(const PlanConfig &c, const WorkspaceLayout &l, void *workspace, void *stream)
{
    auto src = l.a, dst = l.b;
    SpsmA2SortTilingData t{0, c.nnz, c.complex, c.blocks};
    SpsmA2SortPassLaunch(At(workspace, src.rows), At(workspace, src.cols), At(workspace, src.values),
        At(workspace, src.order), At(workspace, dst.rows), At(workspace, dst.cols), At(workspace, dst.values),
        At(workspace, dst.order), At(workspace, l.status), t, stream);
    std::swap(src, dst);
    for (int64_t run = 512; run < c.nnz; run *= 2) {
        t.run = run;
        SpsmA2SortPassLaunch(At(workspace, src.rows), At(workspace, src.cols), At(workspace, src.values),
            At(workspace, src.order), At(workspace, dst.rows), At(workspace, dst.cols), At(workspace, dst.values),
            At(workspace, dst.order), At(workspace, l.status), t, stream);
        std::swap(src, dst);
    }
}
void LegacySolve(const PlanConfig &c, const WorkspaceLayout &l, WorkspaceSet csr, WorkspaceSet dead,
    void *deviceAlpha, void *workspace, void *stream, SpsmA2TilingData solve)
{
    uint32_t solveTiles = (static_cast<uint64_t>(c.rhs) + kSpsmA2SolveColumns - 1) / kSpsmA2SolveColumns;
    if (c.levelCount) {
        for (int32_t level = 0; level < c.levelCount; ++level) {
            solve.level = level;
            bool parallel = (c.levelParallelMask >> level) & 1;
            solve.independentMode = parallel ? 3 : 4;
            SpsmA2CompensatedLaunch(At(workspace, l.rowPtr), At(workspace, csr.cols), At(workspace, csr.values),
                At(workspace, l.dense), At(workspace, l.dense), static_cast<uint8_t *>(deviceAlpha),
                At(workspace, l.status), At(workspace, dead.rows), At(workspace, l.lowTail), l.lowRows,
                At(workspace, l.levelOrder), At(workspace, l.levelOffsets), solve,
                parallel ? c.blocks : std::min<uint32_t>(c.blocks, solveTiles), stream);
        }
    } else {
        if (c.independentRows) {
            solve.independentMode = 1;
            SpsmA2CompensatedLaunch(At(workspace, l.rowPtr), At(workspace, csr.cols), At(workspace, csr.values),
                At(workspace, l.dense), At(workspace, l.dense), static_cast<uint8_t *>(deviceAlpha),
                At(workspace, l.status), At(workspace, dead.rows), At(workspace, l.lowTail), l.lowRows,
                At(workspace, l.levelOrder), At(workspace, l.levelOffsets), solve,
                c.blocks, stream);
            solve.independentMode = 2;
        }
        SpsmA2CompensatedLaunch(At(workspace, l.rowPtr), At(workspace, csr.cols), At(workspace, csr.values),
            At(workspace, l.dense), At(workspace, l.dense), static_cast<uint8_t *>(deviceAlpha),
            At(workspace, l.status), At(workspace, dead.rows), At(workspace, l.lowTail), l.lowRows,
            At(workspace, l.levelOrder), At(workspace, l.levelOffsets), solve,
            std::min<uint32_t>(c.blocks, solveTiles), stream);
    }
}
void PackedSolve(const PlanConfig &c, const WorkspaceLayout &l, WorkspaceSet csr, WorkspaceSet dead,
    void *b, void *deviceAlpha, void *workspace, void *stream, uint32_t copyBlocks,
    SpsmA2DenseTilingData pack, SpsmA2TilingData &solve)
{
    // Finite non-diagonal complex64: independent/level schedules first try the
    // compensated vector solver, scanning both high/low planes before retrying;
    // other schedules try extended range directly. Non-finite A/B/alpha keeps the
    // original path. Packed encoding failure restores op(B) and falls back too.
    // Every retry precedes the final write to C, retaining original B for B=C.
    // Status word 0 is a preparation error; words 1/2/3 are non-finite input,
    // packed failure and legacy overflow. Guard resets words 1-3 for each Solve.
    // Same-stream kernel ordering provides completion; no cross-block spinning
    // or host status readback is introduced. Structure selects order, not accuracy.
    // Reuse the existing workspace; keep Solve asynchronous on the
    // handle stream. Eligibility and recovery are decided on device.
    SpsmA2PackedGuardLaunch(At(workspace, l.rowPtr), At(workspace, csr.values),
        At(workspace, l.dense), static_cast<uint8_t *>(deviceAlpha), At(workspace, l.status),
        solve, c.blocks, stream);
    // Structure chooses scheduling, never numerical eligibility. The old
    // parallel/batched path is retained for wide dependency schedules.
    // Every solved high AND low component is checked before reusing it.
    bool legacyFirst = c.independentRows || c.levelCount;
    if (legacyFirst) {
        LegacySolve(c, l, csr, dead, deviceAlpha, workspace, stream, solve);
        SpsmA2PackedRetryLaunch(At(workspace, l.dense), At(workspace, dead.rows),
            At(workspace, l.lowTail), At(workspace, l.status), l.lowRows, solve, c.blocks, stream);
        solve.packedFallback = 2;
        pack.packedFallback = 2;
        SpsmA2DenseLaunch(static_cast<uint8_t *>(b), At(workspace, l.dense),
            At(workspace, l.status), pack, copyBlocks, stream);
    }
    constexpr int32_t chunkRows = 1024;
    for (int64_t begin = 0; begin < c.rows; begin += chunkRows) {
        SpsmA2PackedSolveLaunch(At(workspace, l.rowPtr), At(workspace, csr.cols),
            At(workspace, csr.values), At(workspace, l.dense), static_cast<uint8_t *>(deviceAlpha),
            At(workspace, l.status), At(workspace, dead.rows), At(workspace, l.lowTail), l.lowRows,
            static_cast<int32_t>(begin), static_cast<int32_t>(std::min<int64_t>(c.rows, begin + chunkRows)),
            solve, std::min<uint32_t>(c.blocks, c.rhs), stream);
    }
    SpsmA2PackedConvertLaunch(At(workspace, l.dense), At(workspace, dead.rows),
        At(workspace, l.lowTail), At(workspace, l.status), l.lowRows, solve, copyBlocks, stream);
    // A failed/ineligible attempt leaves packed data or raw B in dense.
    // Restore op(B) before the legacy kernels; output/alias is untouched.
    pack.packedFallback = legacyFirst ? 3 : 1;
    SpsmA2DenseLaunch(static_cast<uint8_t *>(b), At(workspace, l.dense), At(workspace, l.status),
        pack, copyBlocks, stream);
    solve.packedFallback = legacyFirst ? 3 : 1;
}
}
bool BuildWorkspace(const PlanConfig &config, WorkspaceLayout &layout)
{
    // Size depends on the current device's block count: query and analyze on the
    // same device. Packed high/low word planes reuse dense, the dead sorting set
    // and lowTail; permutation, canonical columns and current-value snapshot stay
    // live for Update. This layout is unchanged by source directory regrouping.
    if (!ValidConfig(config)) { return false; }
    size_t parts = config.complex ? 2 : 1, cursor = 0;
    WorkspaceLayout candidate{};
    const size_t m = config.rows, rhs = config.rhs, nnz = config.nnz;
    if (rhs != 0 && m > std::numeric_limits<size_t>::max() / rhs) { return false; }
    bool ok = AppendSet(nnz, parts, cursor, candidate.a) && AppendSet(nnz, parts, cursor, candidate.b) &&
        Append(m + 1, 4, cursor, candidate.rowStarts) && Append(m, 4, cursor, candidate.rowCounts) &&
        Append(m + 1, 4, cursor, candidate.rowPtr) && Append(nnz, parts * 4, cursor, candidate.snapshot) &&
        Append(config.blocks, kSpsmA2StatusStride * 4, cursor, candidate.status) &&
        Append(nnz, 4, cursor, candidate.perm) &&
        Append(m * rhs, parts * 4, cursor, candidate.dense);
    if (!ok) { return false; }
    // Canonicalization has consumed the sorted set. Prepare reconstructs it on
    // every update from original indices and the separate current-value snapshot.
    size_t setBytes = candidate.b.rows - candidate.a.rows;
    candidate.lowRows = rhs == 0 ? m : std::min(m, setBytes / (rhs * parts * 4));
    if (!Append((m - candidate.lowRows) * rhs, parts * 4, cursor, candidate.lowTail)) { return false; }
    if (m <= kSpsmA2LevelRows && (!Append(m, 4, cursor, candidate.levelOrder) ||
        !Append(m + 4, 4, cursor, candidate.levelOffsets))) { return false; }
    candidate.bytes = cursor; layout = candidate; return true;
}
void Prepare(const PlanConfig &c, const WorkspaceLayout &l, void *first, void *second,
    const void *values, void *workspace, bool diagonal, void *stream)
{
    // The fast path never sorts, so it expands straight into the sorted set's storage
    // and gathers from there into the output set. That keeps source and destination in
    // different regions for either parity, and leaves the output set's canonical column
    // indices — which Solve reads — untouched.
    const bool fast = c.noDuplicates != 0 && c.nnz != 0;
    auto sorted = SortedSet(c, l), output = OutputSet(c, l);
    auto staging = fast ? sorted : l.a;
    SpsmA2FormatTilingData expand{c.rows, c.nnz, c.base, c.format, c.opA, c.upper, c.complex, c.blocks};
    // The device launcher uses GM_ADDR; expansion only reads the input values.
    SpsmA2ExpandLaunch(static_cast<uint8_t *>(first), static_cast<uint8_t *>(second), static_cast<uint8_t *>(const_cast<void *>(values)),
        At(workspace, staging.rows), At(workspace, staging.cols), At(workspace, staging.values),
        At(workspace, l.status), expand, stream);
    SpsmA2RefreshTilingData refresh{c.nnz, c.complex, diagonal, c.blocks};
    SpsmA2RefreshLaunch(At(workspace, staging.rows), At(workspace, staging.cols), At(workspace, staging.values),
        At(workspace, l.snapshot), At(workspace, l.status), refresh, stream);
    if (fast) {
        // One entry per canonical slot, so this is a permutation with no addition; the
        // merge scan's accumulation order cannot be disturbed here.
        SpsmA2GatherTilingData gather{c.rows, c.nnz, c.complex, c.unit, c.blocks};
        SpsmA2GatherLaunch(At(workspace, l.perm), At(workspace, staging.values), At(workspace, output.cols),
            At(workspace, l.rowPtr), At(workspace, output.values), At(workspace, l.status), gather,
            std::min<uint32_t>(c.blocks, static_cast<uint32_t>(c.rows)), stream);
        return;
    }
    Sort(c, l, workspace, stream);
    auto src = SortedSet(c, l), dst = OutputSet(c, l);
    SpsmA2CanonicalTilingData canonical{c.rows, c.nnz, c.complex, c.unit, c.blocks};
    SpsmA2CanonicalLaunch(At(workspace, src.rows), At(workspace, src.cols), At(workspace, src.values),
        At(workspace, l.rowStarts), At(workspace, l.rowCounts), At(workspace, l.rowPtr), At(workspace, dst.cols),
        At(workspace, dst.values),
        At(workspace, dst.rows), At(workspace, dst.order), At(workspace, l.status), canonical, stream);
}
void Solve(const PlanConfig &c, const WorkspaceLayout &l, void *b, void *output,
    void *deviceAlpha, HostScalar hostAlpha, void *workspace, void *stream)
{
    if (c.rows == 0 || c.rhs == 0) { return; }
    auto csr = OutputSet(c, l);
    uint32_t rhsTiles = (static_cast<uint64_t>(c.rhs) + kSpsmA2RhsTile - 1) / kSpsmA2RhsTile;
    uint32_t copyBlocks = std::min<uint64_t>(c.blocks, static_cast<uint64_t>(c.rows) * rhsTiles);
    SpsmA2DenseTilingData pack{c.bRowStride, c.bColStride, c.rhs, 1, c.rows, c.rhs, c.complex, c.opB == 2, c.blocks};
    SpsmA2DenseLaunch(static_cast<uint8_t *>(b), At(workspace, l.dense), At(workspace, l.status), pack, copyBlocks,
        stream);
    SpsmA2TilingData solve{c.rhs, c.rhs, c.rows, c.rhs, c.nnz, 0, c.upper ^ (c.opA != 0), c.unit,
        c.deviceAlpha, c.blocks, hostAlpha.real, hostAlpha.imag, c.complex};
    if (c.diagonalOnly) {
        // Every row is independent here, so the grid spans rows as well as RHS tiles
        // instead of the general path's single tile.
        SpsmA2DiagonalLaunch(At(workspace, l.rowPtr), At(workspace, csr.values),
            At(workspace, l.dense), At(workspace, l.dense), static_cast<uint8_t *>(deviceAlpha),
            At(workspace, l.status), solve, copyBlocks, stream);
    } else {
        auto dead = SortedSet(c, l);
        if (c.complex) {
            PackedSolve(c, l, csr, dead, b, deviceAlpha, workspace, stream, copyBlocks, pack, solve);
        }
        LegacySolve(c, l, csr, dead, deviceAlpha, workspace, stream, solve);
    }
    SpsmA2DenseTilingData unpack{c.rhs, 1, c.cRowStride, c.cColStride, c.rows, c.rhs, c.complex, 0, c.blocks};
    SpsmA2DenseLaunch(At(workspace, l.dense), static_cast<uint8_t *>(output), At(workspace, l.status), unpack,
        copyBlocks, stream);
}
bool ReadStatus(const PlanConfig &c, const WorkspaceLayout &l, void *workspace, void *stream, bool &failed)
{
    // Analysis/Update alone perform this synchronous readback. Invalid offsets,
    // indices, triangle membership or missing/zero NON_UNIT diagonals invalidate
    // the plan; readback failure maps to INTERNAL_ERROR. Solve never calls this.
    failed = false;
    if (c.blocks == 0) { return true; }
    // The preparation kernels are queued on the handle stream; the slots are only
    // meaningful once they have all retired.
    if (aclrtSynchronizeStream(static_cast<aclrtStream>(stream)) != ACL_SUCCESS) { return false; }
    const size_t slots = static_cast<size_t>(c.blocks) * static_cast<size_t>(kSpsmA2StatusStride);
    const size_t bytes = slots * sizeof(int32_t);
    std::vector<int32_t> host(slots, 0);
    if (aclrtMemcpy(host.data(), bytes, At(workspace, l.status), bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    for (uint32_t block = 0; block < c.blocks; ++block) {
        if (host[static_cast<size_t>(block) * kSpsmA2StatusStride] != 0) { failed = true; break; }
    }
    return true;
}
bool DetectNoDuplicates(const PlanConfig &c, const WorkspaceLayout &l, void *workspace, bool &unique)
{
    unique = false;
    if (c.rows == 0 || c.nnz == 0) { return true; }
    int32_t total = 0;
    if (aclrtMemcpy(&total, sizeof(total), At(workspace, l.rowPtr) + static_cast<size_t>(c.rows) * sizeof(int32_t),
        sizeof(total), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    // Canonicalization emits one entry per distinct coordinate, so an unchanged count
    // means nothing was merged.
    unique = total == c.nnz;
    return true;
}
bool BuildLevels(PlanConfig &c, const WorkspaceLayout &l, void *workspace, void *stream)
{
    // At most 8192 rows retain an extra align64(m*4)+align64((m+4)*4) schedule,
    // separate from lowTail. More than 64 stages use the general solver. Stages
    // can combine narrow graph levels; levelCount is not the graph depth. This
    // preparation phase reads three metadata words; Solve remains asynchronous.
    c.levelCount = 0;
    c.levelParallelMask = 0;
    if (c.diagonalOnly || c.independentRows || c.rows == 0 || c.rhs == 0 || c.rows > kSpsmA2LevelRows) { return true; }
    auto csr = OutputSet(c, l);
    SpsmA2LevelsLaunch(At(workspace, l.rowPtr), At(workspace, csr.cols), At(workspace, l.levelOrder),
        At(workspace, l.levelOffsets), c.rows, c.upper ^ (c.opA != 0), stream);
    if (aclrtSynchronizeStream(static_cast<aclrtStream>(stream)) != ACL_SUCCESS) { return false; }
    uint32_t metadata[3]{};
    if (aclrtMemcpy(metadata, sizeof(metadata), At(workspace, l.levelOffsets) + (c.rows + 1) * sizeof(int32_t),
        sizeof(metadata), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) { return false; }
    if (metadata[0] > 0 && metadata[0] <= kSpsmA2MaxLevels) {
        c.levelCount = metadata[0];
        c.levelParallelMask = uint64_t(metadata[1]) | (uint64_t(metadata[2]) << 32);
    }
    return true;
}
bool SavePermutation(const PlanConfig &c, const WorkspaceLayout &l, void *workspace)
{
    if (c.nnz == 0) { return true; }
    const size_t bytes = static_cast<size_t>(c.nnz) * sizeof(int32_t);
    auto sorted = SortedSet(c, l);
    return aclrtMemcpy(At(workspace, l.perm), bytes, At(workspace, sorted.order), bytes,
        ACL_MEMCPY_DEVICE_TO_DEVICE) == ACL_SUCCESS;
}
bool DetectDiagonalOnly(PlanConfig &c, const WorkspaceLayout &l, void *workspace, bool &diagonal)
{
    diagonal = false;
    c.independentRows = 0;
    if (c.rows == 0 || c.nnz == 0) { return true; }
    const size_t rows = static_cast<size_t>(c.rows);
    // The canonical row offsets are rows + 1 entries; a diagonal-only matrix has one
    // entry per row, so the final offset equals the row count before anything else
    // is worth reading back.
    std::vector<int32_t> offsets(rows + 1, 0);
    const size_t offsetBytes = (rows + 1) * sizeof(int32_t);
    if (aclrtMemcpy(offsets.data(), offsetBytes, At(workspace, l.rowPtr), offsetBytes,
        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    int32_t singleRows = 0;
    int32_t longest = 0;
    for (size_t row = 0; row < rows; ++row) {
        int32_t length = offsets[row + 1] - offsets[row];
        singleRows += length <= 1;
        longest = std::max(longest, length);
    }
    // This is only a scheduling hint. The device checks column identity before
    // treating any row as independent, including UNIT rows with one off-diagonal.
    c.independentRows = c.rows >= 128 && singleRows >= c.rows / 2 && longest >= 32;
    if (offsets[rows] != c.rows) { return true; }
    for (size_t row = 0; row < rows; ++row) {
        if (offsets[row + 1] - offsets[row] != 1) { return true; }
    }
    // One entry per row is not enough on its own: under UNIT diagonal an off-diagonal
    // entry would also satisfy it. Confirm every column index is its own row.
    auto csr = OutputSet(c, l);
    std::vector<int32_t> cols(rows, 0);
    const size_t colBytes = rows * sizeof(int32_t);
    if (aclrtMemcpy(cols.data(), colBytes, At(workspace, csr.cols), colBytes,
        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    // Canonicalization has already rebased the indices: Solve hands the kernel base 0
    // and the kernel subtracts that, so these columns are plain zero-based row ids.
    // Subtracting the descriptor's base again would reject every base-1 matrix.
    for (size_t row = 0; row < rows; ++row) {
        if (cols[row] != static_cast<int32_t>(row)) { return true; }
    }
    diagonal = true;
    return true;
}
}
