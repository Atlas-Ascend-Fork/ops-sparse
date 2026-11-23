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

#ifndef OP_HOST_SPSV_WORKSPACE_H_
#define OP_HOST_SPSV_WORKSPACE_H_

#include "../shared/spsv_workspace_layout.h"

namespace
{
// CSC stores A as the CSR of A^T, so op is remapped onto the stored matrix:
// N -> T, T -> N, and H -> N on the conjugate of the stored values.  That
// conjugation has no representation in the op enum, so it travels as a flag and
// is applied when analysis copies the values into the workspace CSR.  Real
// values need no flag because conjugation is the identity there.
static bool NeedsCscConjugate(bool isCsc, int32_t op, bool isComplex) { return isCsc && op == 2 && isComplex; }

static void ReserveSolveWorkspace(WorkspaceOffsets& out, int64_t m, bool needsTrans, size_t valueSize,
                                  size_t& offset)
{
    using spsv::AlignUpInternal;

    if (valueSize == sizeof(aclsparseComplex)) {
        // Float-float shadow of Y for the complex solve: 4 floats per row.
        out.yWideOffset = static_cast<int64_t>(offset);
        // 4 floats per row for the wide Y, 2 per row to stash X when the
        // caller solves in place, and 18 for the diagonal reciprocal, prepared
        // RHS, scale, loss status and bounded-path constants.
        const SpsvComplexWorkspaceGeometry geometry{static_cast<uint64_t>(m)};
        size_t solveBytes = AlignUpInternal(static_cast<size_t>(geometry.arenaBytes));
        // Once a transpose exists, solve and both update modes consume only
        // that representation. The normalized CSR values/permutation are
        // analysis scratch and can hold the solve state after analysis ends.
        // Do not overlap the live CSR row/column indices or level metadata.
        if (needsTrans && out.csrValuesOffset >= 0 &&
            static_cast<size_t>(out.diagPtrOffset - out.csrValuesOffset) >= solveBytes) {
            out.yWideOffset = out.csrValuesOffset;
        } else {
            offset += solveBytes;
        }
        return;
    }

    // Original RHS, interleaved (solution low, forward error), marked
    // dependency scales/modular solutions, then one O(m) big-integer set
    // per batched target plus a shared modulus and denominator.
    // 620 bits/row bounds integer determinants for all finite FP32 input
    // exponents, an alpha*x product, and up to INT64_MAX row entries.
    size_t capacity =
        (static_cast<size_t>(kSpsvRationalBitsPerRow) * static_cast<size_t>(m) + 512u + 15u) / 16u;
    out.yWideOffset = static_cast<int64_t>(offset);
    size_t solveBytes = AlignUpInternal(static_cast<size_t>(m) * 5u * sizeof(float) +
                                         static_cast<size_t>(kSpsvRationalTargetBatch + 2) * capacity * sizeof(uint16_t));
    if (m >= 1024) {
        // Width sixteen always bounds the tree selected from widths 1..16.
        // The real path shares its ten-word storage with the complex tree.
        size_t parts = (static_cast<size_t>(m) + 15u) / 16u;
        size_t leaves = 1u;
        while (leaves < parts) {
            leaves *= 2u;
        }
        size_t treeWords = ((leaves + parts) * 16u * 17u + 2u * leaves * 16u + static_cast<size_t>(m)) * 10u;
        size_t scaledBytes = AlignUpInternal((treeWords + 24u * static_cast<size_t>(m) + m) * sizeof(float));
        solveBytes = solveBytes > scaledBytes ? solveBytes : scaledBytes;
    }
    offset += solveBytes;
}

// Compute all workspace sub-allocation offsets.
// - The first offset is 512B-aligned (kAlign) to satisfy GM base alignment.
// - Subsequent sub-allocations are 64B-aligned (kInternalAlign) to reduce
//   workspace bloat for small matrices.
// - When nnz==0 all offsets are -1 (uniform sentinel), totalSize is 0.
// - permType controls the element size for nnz-scaled workspace arrays
//   (perm, transPerm, diagPtr): 0=int32_t, 1=int64_t (nnz > INT32_MAX).
static WorkspaceOffsets ComputeWorkspaceOffsets(int64_t m, int64_t nnz, int32_t format, size_t rowPtrSize,
                                                size_t colIndSize, bool needsTrans, int32_t permType,
                                                int32_t idxBase, size_t valueSize, bool conjValues)
{
    using spsv::AlignUp;
    using spsv::AlignUpInternal;

    WorkspaceOffsets out{};

    if (nnz == 0) {
        return out;
    }

    // permSize: element size for nnz-scaled arrays (perm, transPerm, diagPtr).
    // When permType==1 (nnz > INT32_MAX with I64 index), use 8-byte elements.
    size_t permSize = (permType == 1) ? sizeof(int64_t) : sizeof(int32_t);

    // First sub-allocation uses 512B alignment (GM base).
    size_t offset = AlignUp(sizeof(SpsvTilingData));

    // Build workspace CSR copy for: COO(2), SELL(3), or 1-based CSR(0)/CSC(1).
    // For 1-based inputs, workspace CSR holds 0-based indices for downstream ops.
    // rowPtrSize: element width for row-offset arrays (csrRowPtr, transRowPtr)
    // colIndSize: element width for column-index arrays (csrColInd, transColInd)
    if (format == 2 || format == 3 || idxBase == 1 || conjValues) {
        out.csrRowPtrOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(m + 1) * rowPtrSize);
        out.csrColIndOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * colIndSize);
        out.csrValuesOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * valueSize);
        out.permOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * permSize);
    }

    out.diagPtrOffset = static_cast<int64_t>(offset);
    offset += AlignUpInternal(static_cast<size_t>(m) * permSize);
    out.levelPtrOffset = static_cast<int64_t>(offset);
    offset += AlignUpInternal(static_cast<size_t>(m + 1) * sizeof(int32_t));
    out.levelRowOffset = static_cast<int64_t>(offset);
    offset += AlignUpInternal(static_cast<size_t>(m) * sizeof(int32_t));
    out.validCountOffset = static_cast<int64_t>(offset);
    offset += AlignUpInternal(static_cast<size_t>(m) * sizeof(int32_t));

    if (needsTrans) {
        out.transRowPtrOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(m + 1) * rowPtrSize);
        out.transColIndOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * colIndSize);
        out.transValuesOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * valueSize);
        out.transPermOffset = static_cast<int64_t>(offset);
        offset += AlignUpInternal(static_cast<size_t>(nnz) * permSize);
    }

    ReserveSolveWorkspace(out, m, needsTrans, valueSize, offset);

    out.totalSize = offset;
    return out;
}

// Callers choose current matrix attributes (bufferSize) or cached analysis
// attributes (solve). Only the common CSC/conjugation/layout derivation is shared.
static WorkspaceOffsets ComputeMatrixWorkspaceOffsets(int64_t m, int64_t nnz, int32_t format, int32_t opA,
                                                      size_t rowPtrSize, size_t colIndSize, int32_t permType,
                                                      int32_t idxBase, size_t valueSize)
{
    bool csc = format == 1;
    bool needsTrans = csc ? opA == 0 : opA != 0;
    bool conjValues = NeedsCscConjugate(csc, opA, valueSize == sizeof(aclsparseComplex));
    return ComputeWorkspaceOffsets(m, nnz, csc ? 0 : format, rowPtrSize, colIndSize, needsTrans, permType, idxBase,
                                   valueSize, conjValues);
}

} // namespace

#endif // OP_HOST_SPSV_WORKSPACE_H_
