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

#ifndef OP_HOST_SPSV_TILING_H_
#define OP_HOST_SPSV_TILING_H_

namespace {

static float ReadHostScalar(const void* p)
{
    if (p == nullptr) {
        return 0.0f;
    }
    return *static_cast<const float*>(p);
}

static aclsparseComplex ReadHostComplexScalar(const void* p)
{
    if (p == nullptr) {
        return aclsparseComplex{0.0f, 0.0f};
    }
    return *static_cast<const aclsparseComplex*>(p);
}

// SIMT threads per block for the solve.  Both value types measure best at a
// constant rather than at a function of m, but not at the same constant.
//
// complex64 = 256.  Swept 64 / 128 / 256 over the 104 complex64 shapes of
// the 206-case set, four passes per value in ABBA order, median per case
// (run-to-run spread 0.11% median):
//
//     64    geomean 0.921   worst 0.393   faster on 38 of 104
//     128   geomean 0.964   worst 0.652   faster on 65 of 104
//     256   reference
//
// so 64 costs 7.9% in geometric mean and 2.5x on the shape it hurts most.
// An earlier sweep did choose 64, and this file used to carry its numbers:
// that sweep was run when the complex path was single precision plus a
// refinement pass.  Carrying the solution in float-float instead costs 2.0x
// to 2.6x the arithmetic per nonzero, which is work to spread across
// threads, and it moves the best block width the other way.
//
// float32: unchanged.  The one sweep that covered it sampled two float32
// shapes, one of which swings between 0.44 and 0.80 run to run, so there is
// nothing to move it on.
constexpr uint32_t kSpsvComplexNthreads = 256u;
constexpr uint32_t kSpsvRealNthreadsSmall = 64u;
constexpr uint32_t kSpsvRealNthreadsLarge = 256u;
// Below this many rows the real solve cannot fill wider blocks.
constexpr int64_t kSpsvRealSmallMThreshold = 5000;

// The block width the solve launches with, for both value types in one
// place.  valueType is the descriptor's cached type: 1 is complex64.
static uint32_t SelectNthreads(int32_t valueType, int64_t m)
{
    if (valueType == 1) {
        return kSpsvComplexNthreads;
    }
    return (m <= kSpsvRealSmallMThreshold) ? kSpsvRealNthreadsSmall : kSpsvRealNthreadsLarge;
}

static void ComputeFormatWorkspace(SpsvTilingData& tiling, aclsparseSpSVDescr_t spsvDescr, int64_t diagPtrOff,
                                   int64_t levelPtrOff, int64_t levelRowOff, int64_t validCountOff,
                                   int64_t csrRowPtrOff, int64_t csrColIndOff, int64_t csrValuesOff,
                                   int64_t permOff, int64_t transRowPtrOff, int64_t transColIndOff,
                                   int64_t transValuesOff, int64_t transPermOff, int64_t yWideOff)
{
    tiling.levelPtrOffset = levelPtrOff;
    tiling.levelRowOffset = levelRowOff;
    tiling.diagPtrOffset = diagPtrOff;
    tiling.validCountOffset = validCountOff;
    tiling.numLevels = 0;
    tiling.csrRowPtrOffset = csrRowPtrOff;
    tiling.csrColIndOffset = csrColIndOff;
    tiling.csrValuesOffset = csrValuesOff;
    tiling.permOffset = permOff;
    tiling.transRowPtrOffset = transRowPtrOff;
    tiling.transColIndOffset = transColIndOff;
    tiling.transValuesOffset = transValuesOff;
    tiling.transPermOffset = transPermOff;
    tiling.yWideOffset = yWideOff;

    spsvDescr->diagPtrOffset = diagPtrOff;
    spsvDescr->csrValuesOffset = csrValuesOff;
    spsvDescr->permOffset = permOff;
    spsvDescr->transValuesOffset = transValuesOff;
    spsvDescr->transPermOffset = transPermOff;
}

// The tiling fields that are copies of what analysis cached on the
// descriptor, so BuildTilingData reads as the decisions it actually makes.
static void FillCachedTilingFields(SpsvTilingData& tiling, aclsparseSpSVDescr_t spsvDescr)
{
    tiling.valueType = spsvDescr->cachedValueType;
    tiling.indexType = spsvDescr->cachedIndexType;
    tiling.colIndType = spsvDescr->cachedColIndType;
    tiling.idxBase = spsvDescr->cachedIdxBase;
    tiling.numSlices = spsvDescr->cachedNumSlices;
    tiling.sliceWidth = spsvDescr->cachedSliceWidth;
}

static SpsvTilingData BuildTilingData(aclsparseConstSpMatDescr_t matA, float alpha,
                                      aclsparseSpSVDescr_t spsvDescr)
{
    (void)matA;  // All attributes read from spsvDescr cache (cuSPARSE convention).
    // Use cached attributes from analysis phase (cuSPARSE convention: solve
    // and updateMatrix read from descriptor, not from matA).
    int32_t format = spsvDescr->cachedFormat;
    int32_t fillMode = spsvDescr->cachedFillMode;
    int32_t diagType = spsvDescr->cachedDiagType;
    int32_t op = spsvDescr->cachedOpA;

    bool conjValues = NeedsCscConjugate(format == 1, op, spsvDescr->cachedValueType == 1);
    if (format == 1) {  // CSC: cachedFormat=1 means CSC
        format = 0;
        fillMode = (fillMode == 0) ? 1 : 0;
        op = (op == 0) ? 1 : 0;
    }

    int64_t m = spsvDescr->cachedM;
    int64_t nnz = spsvDescr->cachedNnz;
    size_t rowPtrSize = spsvDescr->cachedIdxSize;
    size_t colIndSize = spsvDescr->cachedColIndSize;

    int32_t permType = spsvDescr->cachedPermType;

    size_t valueSize = (spsvDescr->cachedValueType == 1) ? sizeof(aclsparseComplex) : sizeof(float);
    WorkspaceOffsets offsets = ComputeMatrixWorkspaceOffsets(m, nnz, spsvDescr->cachedFormat, spsvDescr->cachedOpA,
                                                             rowPtrSize, colIndSize, permType, spsvDescr->cachedIdxBase,
                                                             valueSize);

    uint32_t nthreads = SelectNthreads(spsvDescr->cachedValueType, m);

    SpsvTilingData tiling{};
    tiling.m = m;
    tiling.nnz = nnz;
    tiling.alpha = alpha;
    tiling.alphaImag = 0.0f;
    FillCachedTilingFields(tiling, spsvDescr);
    tiling.fillMode = fillMode;
    tiling.diagType = diagType;
    tiling.opA = op;
    tiling.format = format;
    tiling.nthreads = nthreads;
    tiling.permType = permType;
    tiling.conjValues = conjValues ? 1 : 0;
    // numBlocks is computed and overwritten by ComputeNumBlocks in
    // LaunchSpSVAnalysisKernel / LaunchSpSVSolveKernel before launch.
    tiling.numBlocks = 0;

    ComputeFormatWorkspace(tiling, spsvDescr, offsets.diagPtrOffset, offsets.levelPtrOffset,
                           offsets.levelRowOffset, offsets.validCountOffset, offsets.csrRowPtrOffset,
                           offsets.csrColIndOffset, offsets.csrValuesOffset, offsets.permOffset,
                           offsets.transRowPtrOffset, offsets.transColIndOffset, offsets.transValuesOffset,
                           offsets.transPermOffset, offsets.yWideOffset);

    return tiling;
}

static void GetCsrPointers(aclsparseConstSpMatDescr_t matA, uint8_t*& rowPtr, uint8_t*& colInd,
                           uint8_t*& values)
{
    auto* mat = spsv::ToMatInner(matA);
    rowPtr = reinterpret_cast<uint8_t*>(mat->ptrs);
    colInd = reinterpret_cast<uint8_t*>(mat->idxs);
    values = reinterpret_cast<uint8_t*>(mat->values);
}

// Compute the number of AI cores (blocks) to launch.
// Deep level-set heuristic: when estimated average level width < 256 rows,
// the per-level SyncAll overhead in multi-core solve dominates. Fall back to
// single core to avoid paying that overhead for negligible parallelism.
// estimatedNumLevels ≈ nnz/m is a rough lower bound; numLevels is only known
// after analysis, so this estimate may miss some deep cases.
static uint32_t ComputeNumBlocks(int64_t m, int64_t nnz, uint32_t nthreads)
{
    if (nthreads == 0) {
        return 1;
    }
    if (m <= static_cast<int64_t>(nthreads)) {
        return 1;
    }
    // Reserve useful cores before analysis. Solve dispatch uses the actual
    // level width; nnz/m does not estimate the depth of a sparse triangle.
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        return 1;
    }
    uint64_t usefulCores = (static_cast<uint64_t>(m) + nthreads - 1u) / nthreads;
    uint32_t numBlocks = (usefulCores < aivCoreNum) ? static_cast<uint32_t>(usefulCores) : aivCoreNum;
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    return numBlocks;
}

static uint32_t SelectComplexScaledBlocks(const SpsvTilingData& tiling, uint32_t current)
{
    if (tiling.valueType != 1 || tiling.diagType != 0 || tiling.m < 1024 || tiling.m > 8192 ||
        tiling.nnz / tiling.m > 16 || tiling.numLevels != tiling.m) {
        return current;
    }
    // Affine tree nodes are independent within a level. The ordinary
    // ceil(m / nthreads) row budget would leave most cores idle here.
    uint32_t available = GetAivCoreCount();
    return available > current ? available : current;
}

} // namespace

#endif // OP_HOST_SPSV_TILING_H_
