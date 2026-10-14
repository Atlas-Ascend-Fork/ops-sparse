/**
 * Copyright (c) 2026 liutaw.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSM_ARCH22_PLAN_H
#define SPSM_ARCH22_PLAN_H
#include <cstddef>
#include <cstdint>
namespace spsm_a2 {
// Host-only configuration, validated before launching the internal pipeline.
// B strides already describe op(B); C strides describe its physical output.
struct PlanConfig {
    int32_t rows{0}, rhs{0}, nnz{0}, base{0}, format{0}, opA{0}, opB{0};
    int32_t upper{0}, unit{0}, complex{0}, blocks{0}, deviceAlpha{0};
    // Set by DetectDiagonalOnly after preparation; selects the parallel solve path.
    int32_t diagonalOnly{0};
    int32_t independentRows{0};
    int32_t levelCount{0};
    uint64_t levelParallelMask{0};
    // Set by DetectNoDuplicates after preparation. Without duplicate coordinates the
    // canonical entries map one to one onto the sorted entries, so UpdateMatrix can
    // refresh values through the saved permutation instead of sorting and merging.
    int32_t noDuplicates{0};
    int64_t bRowStride{0}, bColStride{0}, cRowStride{0}, cColStride{0};
};
struct WorkspaceSet { size_t rows{0}, cols{0}, values{0}, order{0}; };
struct WorkspaceLayout {
    WorkspaceSet a{}, b{};
    // Low solution rows reuse the dead sorted set; excess rows use an appended tail.
    size_t lowRows{0}, lowTail{0};
    size_t rowStarts{0}, rowCounts{0}, rowPtr{0}, snapshot{0}, status{0}, dense{0}, bytes{0};
    // Sorted-order permutation saved during Analysis. It needs storage of its own:
    // Solve borrows the whole sorted set, low word included, for its compensation
    // rows, so the copy the sort itself leaves behind does not survive a solve.
    size_t perm{0};
    size_t levelOrder{0}, levelOffsets{0};
};
struct HostScalar { float real{0}, imag{0}; };
bool BuildWorkspace(const PlanConfig &config, WorkspaceLayout &layout);
void Prepare(const PlanConfig &config, const WorkspaceLayout &layout, void *first,
    void *second, const void *values, void *workspace, bool diagonal, void *stream);
void Solve(const PlanConfig &config, const WorkspaceLayout &layout, void *b, void *c,
    void *deviceAlpha, HostScalar hostAlpha, void *workspace, void *stream);
// Reads back the per-block status slots the preprocessing kernels wrote. Synchronises
// the stream first, so it must only be called from the preparation path (Analysis and
// UpdateMatrix); the solve path stays asynchronous. Returns false when the readback
// itself failed, in which case `failed` is meaningless.
bool ReadStatus(const PlanConfig &config, const WorkspaceLayout &layout, void *workspace,
    void *stream, bool &failed);
// Reports whether every row of the canonical matrix holds exactly its diagonal entry.
// Reads back the canonical row offsets and column indices, so it must only be called
// from the preparation path after ReadStatus has already synchronised the stream.
// Returns false when the readback failed, in which case `diagonal` is meaningless.
bool DetectDiagonalOnly(PlanConfig &config, const WorkspaceLayout &layout, void *workspace,
    bool &diagonal);
// Reports whether canonicalization merged nothing, i.e. the input held no duplicate
// coordinates. Reads back the canonical row offsets, so it shares ReadStatus's
// requirement of being called only from the preparation path.
bool DetectNoDuplicates(const PlanConfig &config, const WorkspaceLayout &layout, void *workspace,
    bool &unique);
// Copies the permutation the sort left in the sorted set into its own region.
bool SavePermutation(const PlanConfig &config, const WorkspaceLayout &layout, void *workspace);
bool BuildLevels(PlanConfig &config, const WorkspaceLayout &layout, void *workspace, void *stream);
}
#endif
