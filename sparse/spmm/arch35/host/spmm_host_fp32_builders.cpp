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

#include <iostream>

#include <algorithm>
#include <limits>
#include <new>
#include <numeric>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "spmm_host_builders.h"
#include "spmm_host.h"

#ifndef __CCE_AICORE__
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "kernel_tiling/kernel_tiling.h"
#endif

namespace spmm_host_builders {

using spmm_host_internal::CsrHostData;
using spmm_host_internal::IsValidCsrRowRange;
using spmm_host_internal::LoadCsrHostData;
using spmm_host_internal::RegBaseWorkspaceFits;
using spmm_host_internal::ResetCubeSelection;
using spmm_host_internal::WsOffsets;

struct Fp32Degree2CandidateRow {
    int32_t start;
    int32_t row;
    int32_t valueOffset;
};

void SortFp32Candidates(std::vector<Fp32Degree2CandidateRow> *candidates)
{
    std::stable_sort(candidates->begin(), candidates->end(),
        [](const Fp32Degree2CandidateRow &lhs,
           const Fp32Degree2CandidateRow &rhs) {
            if (lhs.start != rhs.start) {
                return lhs.start < rhs.start;
            }
            return lhs.row < rhs.row;
        });
}

void BuildFp32Degree2CandidateBatches(
    const std::vector<Fp32Degree2CandidateRow> &candidates,
    std::vector<SpmmFp32Degree2BatchMeta> *batches,
    std::vector<int32_t> *rowIds,
    std::vector<SpmmFp32Degree2RowMeta> *rowMeta)
{
    size_t cursor = 0;
    while (cursor < candidates.size()) {
        const int32_t base = candidates[cursor].start;
        size_t end = cursor;
        int32_t windowEnd = base;
        while (end < candidates.size() &&
               end - cursor < SPMM_FP32_DEGREE2_ROWS) {
            const int32_t rowStart = candidates[end].start;
            if (rowStart + SPMM_FP32_DEGREE2_DEGREE - base >
                SPMM_FP32_DEGREE2_WINDOW_ROWS) {
                break;
            }
            windowEnd = std::max(
                windowEnd, rowStart + SPMM_FP32_DEGREE2_DEGREE);
            ++end;
        }
        const size_t actualRows = end - cursor;
        const size_t storedRows = std::max(
            actualRows, static_cast<size_t>(SPMM_FP32_DEGREE2_MIN_BATCH_ROWS));
        const int32_t rowMetaBegin = static_cast<int32_t>(rowMeta->size());
        batches->push_back(SpmmFp32Degree2BatchMeta{
            base, static_cast<int32_t>(storedRows), rowMetaBegin,
            windowEnd - base});
        for (size_t index = cursor; index < end; ++index) {
            const auto &item = candidates[index];
            rowIds->push_back(item.row);
            rowMeta->push_back(SpmmFp32Degree2RowMeta{
                item.start - base, item.valueOffset});
        }
        for (size_t index = actualRows; index < storedRows; ++index) {
            const auto &item = candidates[cursor];
            rowIds->push_back(item.row);
            rowMeta->push_back(SpmmFp32Degree2RowMeta{
                item.start - base, item.valueOffset});
        }
        cursor = end;
    }
}

void BuildFp32Degree2WrapBatches(
    const std::vector<Fp32Degree2CandidateRow> &wrapCandidates, int64_t k,
    std::vector<SpmmFp32Degree2BatchMeta> *batches,
    std::vector<int32_t> *rowIds,
    std::vector<SpmmFp32Degree2RowMeta> *rowMeta)
{
    size_t cursor = 0;
    while (cursor < wrapCandidates.size()) {
        const size_t end = std::min(
            wrapCandidates.size(),
            cursor + static_cast<size_t>(SPMM_FP32_DEGREE2_ROWS));
        const int32_t rowMetaBegin = static_cast<int32_t>(rowMeta->size());
        batches->push_back(SpmmFp32Degree2BatchMeta{
            -static_cast<int32_t>(k), static_cast<int32_t>(end - cursor),
            rowMetaBegin, SPMM_FP32_DEGREE2_DEGREE});
        for (size_t index = cursor; index < end; ++index) {
            const auto &item = wrapCandidates[index];
            rowIds->push_back(item.row);
            rowMeta->push_back(SpmmFp32Degree2RowMeta{0, item.valueOffset});
        }
        cursor = end;
    }
}

void BuildFp32Degree2Batches(
    const std::vector<Fp32Degree2CandidateRow> &candidates,
    const std::vector<Fp32Degree2CandidateRow> &wrapCandidates, int64_t k,
    std::vector<SpmmFp32Degree2BatchMeta> *batches,
    std::vector<int32_t> *rowIds,
    std::vector<SpmmFp32Degree2RowMeta> *rowMeta)
{
    batches->reserve(candidates.size() / 80 + 2);
    rowIds->reserve(candidates.size() + wrapCandidates.size());
    rowMeta->reserve(candidates.size() + wrapCandidates.size());
    BuildFp32Degree2CandidateBatches(candidates, batches, rowIds, rowMeta);
    BuildFp32Degree2WrapBatches(wrapCandidates, k, batches, rowIds, rowMeta);
}

aclsparseStatus_t ClassifyFp32Degree2Rows(
    const std::vector<int32_t> &rowOff, const std::vector<int32_t> &colInd,
    int64_t m, int64_t k, int32_t indexBase, uint64_t nnz,
    std::vector<Fp32Degree2CandidateRow> *candidates,
    std::vector<Fp32Degree2CandidateRow> *wrapCandidates,
    std::vector<int32_t> *fallbackRows, uint64_t *mappedNnz)
{
    candidates->reserve(static_cast<size_t>(m));
    fallbackRows->reserve(64);
    for (int64_t row = 0; row < m; ++row) {
        const int32_t begin = rowOff[static_cast<size_t>(row)] - indexBase;
        const int32_t end = rowOff[static_cast<size_t>(row) + 1] - indexBase;
        if (!IsValidCsrRowRange(begin, end, nnz)) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        bool rowEligible = end - begin == SPMM_FP32_DEGREE2_DEGREE;
        int32_t start = -1;
        if (rowEligible) {
            start = colInd[static_cast<size_t>(begin)] - indexBase;
            const int32_t next =
                colInd[static_cast<size_t>(begin + 1)] - indexBase;
            rowEligible = start >= 0 &&
                static_cast<int64_t>(start) + 1 < k && next == start + 1;
        }
        if (rowEligible) {
            candidates->push_back(Fp32Degree2CandidateRow{
                start, static_cast<int32_t>(row), begin});
            *mappedNnz += SPMM_FP32_DEGREE2_DEGREE;
        } else if (end - begin == SPMM_FP32_DEGREE2_DEGREE &&
                   colInd[static_cast<size_t>(begin)] - indexBase == 0 &&
                   colInd[static_cast<size_t>(begin + 1)] - indexBase == k - 1 &&
                   k <= std::numeric_limits<int32_t>::max()) {
            wrapCandidates->push_back(Fp32Degree2CandidateRow{
                0, static_cast<int32_t>(row), begin});
            *mappedNnz += SPMM_FP32_DEGREE2_DEGREE;
        } else {
            fallbackRows->push_back(static_cast<int32_t>(row));
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

struct Fp32Degree4Config {
    int32_t rows;
    int32_t tileN;
    int32_t windowRows;
    int32_t minAverageRows;
    int32_t maxTasks;
};

Fp32Degree4Config GetFp32Degree4Config(int32_t degree)
{
    const bool degree64 = degree == SPMM_FP32_DEGREE64_DEGREE;
    const bool degree32 = degree == SPMM_FP32_DEGREE32_DEGREE;
    const int32_t rows = degree64 ? SPMM_FP32_DEGREE64_ROWS :
        (degree32 ? SPMM_FP32_DEGREE32_ROWS : SPMM_FP32_DEGREE4_ROWS);
    const int32_t tileN = degree64 ? SPMM_FP32_DEGREE64_TILE_N :
        (degree32 ? SPMM_FP32_DEGREE32_TILE_N : SPMM_FP32_DEGREE4_TILE_N);
    const int32_t windowRows = degree64 ? SPMM_FP32_DEGREE64_WINDOW_ROWS :
        (degree32 ? SPMM_FP32_DEGREE32_WINDOW_ROWS : SPMM_FP32_DEGREE4_WINDOW_ROWS);
    const int32_t minAverageRows = degree == SPMM_FP32_DEGREE4_DEGREE ?
        SPMM_FP32_DEGREE4_MIN_AVERAGE_ROWS :
        (degree == SPMM_FP32_DEGREE4_MAX_DEGREE ?
            SPMM_FP32_DEGREE8_MIN_AVERAGE_ROWS :
         degree == SPMM_FP32_DEGREE16_DEGREE ?
            SPMM_FP32_DEGREE16_MIN_AVERAGE_ROWS :
         degree64 ? SPMM_FP32_DEGREE64_MIN_AVERAGE_ROWS :
            SPMM_FP32_DEGREE32_MIN_AVERAGE_ROWS);
    const int32_t maxTasks = degree64 ? SPMM_FP32_DEGREE64_MAX_TASKS :
        (degree32 ? SPMM_FP32_DEGREE32_MAX_TASKS : SPMM_FP32_DEGREE4_MAX_TASKS);
    return {rows, tileN, windowRows, minAverageRows, maxTasks};
}

aclsparseStatus_t ClassifyFp32Degree4Row(
    const std::vector<int32_t> &rowOff, const std::vector<int32_t> &colInd,
    int64_t row, int64_t k, int32_t indexBase, uint64_t nnz,
    int32_t structuralDegree, std::vector<Fp32Degree2CandidateRow> *candidates,
    std::vector<Fp32Degree2CandidateRow> *wrapCandidates,
    std::vector<int32_t> *fallbackRows, uint64_t *mappedNnz)
{
    const int32_t begin = rowOff[static_cast<size_t>(row)] - indexBase;
    const int32_t end = rowOff[static_cast<size_t>(row) + 1] - indexBase;
    if (!IsValidCsrRowRange(begin, end, nnz)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    bool rowEligible = end - begin == structuralDegree;
    int32_t start = -1;
    if (rowEligible) {
        start = colInd[static_cast<size_t>(begin)] - indexBase;
        rowEligible = start >= 0 && static_cast<int64_t>(start) + structuralDegree <= k;
        for (int32_t ordinal = 1; rowEligible && ordinal < structuralDegree; ++ordinal) {
            rowEligible = colInd[static_cast<size_t>(begin + ordinal)] - indexBase == start + ordinal;
        }
    }
    if (rowEligible) {
        candidates->push_back({start, static_cast<int32_t>(row), begin});
        *mappedNnz += structuralDegree;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    int32_t lowCount = 0;
    while (end - begin == structuralDegree && lowCount < structuralDegree &&
           colInd[static_cast<size_t>(begin + lowCount)] - indexBase == lowCount) {
        ++lowCount;
    }
    const int32_t highCount = structuralDegree - lowCount;
    const int32_t highStart = static_cast<int32_t>(k) - highCount;
    bool wrapEligible = lowCount > 0 && highCount > 0;
    for (int32_t ordinal = 0; wrapEligible && ordinal < highCount; ++ordinal) {
        wrapEligible = colInd[static_cast<size_t>(begin + lowCount + ordinal)] - indexBase == highStart + ordinal;
    }
    if (wrapEligible) {
        wrapCandidates->push_back({highStart, static_cast<int32_t>(row), begin});
        *mappedNnz += structuralDegree;
    } else {
        fallbackRows->push_back(static_cast<int32_t>(row));
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ClassifyFp32Degree4Rows(
    const std::vector<int32_t> &rowOff, const std::vector<int32_t> &colInd,
    int64_t m, int64_t k, int32_t indexBase, uint64_t nnz,
    int32_t structuralDegree, std::vector<Fp32Degree2CandidateRow> *candidates,
    std::vector<Fp32Degree2CandidateRow> *wrapCandidates,
    std::vector<int32_t> *fallbackRows, uint64_t *mappedNnz)
{
    candidates->reserve(static_cast<size_t>(m));
    fallbackRows->reserve(64);
    for (int64_t row = 0; row < m; ++row) {
        const aclsparseStatus_t status = ClassifyFp32Degree4Row(
            rowOff, colInd, row, k, indexBase, nnz, structuralDegree,
            candidates, wrapCandidates, fallbackRows, mappedNnz);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

void BuildFp32Degree4CandidateBatches(
    const std::vector<Fp32Degree2CandidateRow> &candidates,
    int32_t structuralDegree, int32_t structuralRows,
    int32_t structuralWindowRows,
    std::vector<SpmmFp32Degree2BatchMeta> *batches,
    std::vector<int32_t> *rowIds,
    std::vector<SpmmFp32Degree2RowMeta> *rowMeta)
{
    if (structuralRows <= 0) {
        return;
    }
    size_t cursor = 0;
    while (cursor < candidates.size()) {
        const int32_t base = candidates[cursor].start;
        size_t end = cursor;
        int32_t windowEnd = base;
        while (end < candidates.size() &&
               end - cursor < static_cast<size_t>(structuralRows)) {
            const int32_t rowStart = candidates[end].start;
            if (rowStart + structuralDegree - base > structuralWindowRows) {
                break;
            }
            windowEnd = std::max(windowEnd, rowStart + structuralDegree);
            ++end;
        }
        const int32_t rowMetaBegin = static_cast<int32_t>(rowMeta->size());
        batches->push_back(SpmmFp32Degree2BatchMeta{
            base, static_cast<int32_t>(end - cursor), rowMetaBegin,
            windowEnd - base});
        for (size_t index = cursor; index < end; ++index) {
            const auto &item = candidates[index];
            rowIds->push_back(item.row);
            rowMeta->push_back(SpmmFp32Degree2RowMeta{
                item.start - base, item.valueOffset});
        }
        cursor = end;
    }
}

void BuildFp32Degree4WrapBatches(
    const std::vector<Fp32Degree2CandidateRow> &wrapCandidates, int64_t k,
    int32_t structuralDegree, int32_t structuralRows,
    std::vector<SpmmFp32Degree2BatchMeta> *batches,
    std::vector<int32_t> *rowIds,
    std::vector<SpmmFp32Degree2RowMeta> *rowMeta)
{
    if (structuralRows <= 0) {
        return;
    }
    size_t cursor = 0;
    while (cursor < wrapCandidates.size()) {
        const int32_t highStart = wrapCandidates[cursor].start;
        size_t end = cursor;
        while (end < wrapCandidates.size() &&
               end - cursor < static_cast<size_t>(structuralRows) &&
               wrapCandidates[end].start == highStart) {
            ++end;
        }
        const int32_t highCount = static_cast<int32_t>(k) - highStart;
        const int32_t lowCount = structuralDegree - highCount;
        const int32_t rowMetaBegin = static_cast<int32_t>(rowMeta->size());
        batches->push_back(SpmmFp32Degree2BatchMeta{
            -(highStart + 1), static_cast<int32_t>(end - cursor),
            rowMetaBegin, lowCount});
        for (size_t index = cursor; index < end; ++index) {
            const auto &item = wrapCandidates[index];
            rowIds->push_back(item.row);
            rowMeta->push_back(SpmmFp32Degree2RowMeta{0, item.valueOffset});
        }
        cursor = end;
    }
}

void BuildFp32Degree4Batches(
    const std::vector<Fp32Degree2CandidateRow> &candidates,
    const std::vector<Fp32Degree2CandidateRow> &wrapCandidates, int64_t k,
    int32_t structuralDegree, int32_t structuralRows,
    int32_t structuralWindowRows,
    std::vector<SpmmFp32Degree2BatchMeta> *batches,
    std::vector<int32_t> *rowIds,
    std::vector<SpmmFp32Degree2RowMeta> *rowMeta)
{
    if (structuralRows <= 0) {
        return;
    }
    const size_t rowsPerBatch = static_cast<size_t>(
        std::max(structuralRows, 1));
    batches->reserve(candidates.size() / rowsPerBatch + 2);
    rowIds->reserve(candidates.size());
    rowMeta->reserve(candidates.size());
    BuildFp32Degree4CandidateBatches(
        candidates, structuralDegree, structuralRows, structuralWindowRows,
        batches, rowIds, rowMeta);
    BuildFp32Degree4WrapBatches(wrapCandidates, k, structuralDegree,
                                structuralRows, batches, rowIds, rowMeta);
}

struct Fp32Degree2Layout {
    std::vector<SpmmFp32Degree2BatchMeta> batches;
    std::vector<int32_t> rowIds;
    std::vector<SpmmFp32Degree2RowMeta> rowMeta;
    std::vector<int32_t> fallbackRows;
};

aclsparseStatus_t PrepareFp32Degree2Layout(
    const aclsparseSpMatDescr *matA, Fp32Degree2Layout *layout)
{
    CsrHostData csr;
    const aclsparseStatus_t status = LoadCsrHostData(matA, &csr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    std::vector<Fp32Degree2CandidateRow> candidates;
    std::vector<Fp32Degree2CandidateRow> wrapCandidates;
    uint64_t mappedNnz = 0;
    const aclsparseStatus_t classifyStatus = ClassifyFp32Degree2Rows(
        csr.rowOffsets, csr.columnIndices, static_cast<int64_t>(matA->rows),
        static_cast<int64_t>(matA->cols), csr.indexBase, matA->nnz,
        &candidates, &wrapCandidates, &layout->fallbackRows, &mappedNnz);
    if (classifyStatus != ACL_SPARSE_STATUS_SUCCESS) {
        return classifyStatus;
    }
    const size_t mappedRows = candidates.size() + wrapCandidates.size();
    if (mappedRows < SPMM_FP32_DEGREE2_MIN_ROWS ||
        layout->fallbackRows.size() > 64 ||
        static_cast<__int128>(mappedNnz) * 1000 <
            static_cast<__int128>(matA->nnz) * 999) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    SortFp32Candidates(&candidates);
    BuildFp32Degree2Batches(candidates, wrapCandidates, matA->cols,
                            &layout->batches, &layout->rowIds, &layout->rowMeta);
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool IsUsableFp32Degree2Layout(const Fp32Degree2Layout &layout)
{
    return !layout.batches.empty() && layout.fallbackRows.size() <= 64 &&
        layout.rowIds.size() >= layout.batches.size() *
            SPMM_FP32_DEGREE2_MIN_AVERAGE_ROWS &&
        layout.rowIds.size() <= static_cast<size_t>(INT32_MAX) &&
        layout.batches.size() <= static_cast<size_t>(INT32_MAX);
}

aclsparseStatus_t UploadFp32Degree2Layout(
    uint8_t *workspace, const WsOffsets &off, const Fp32Degree2Layout &layout,
    bool *uploaded)
{
    *uploaded = false;
    if (!IsUsableFp32Degree2Layout(layout)) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const size_t batchesBytes = layout.batches.size() * sizeof(layout.batches[0]);
    const size_t rowIdsBytes = layout.rowIds.size() * sizeof(layout.rowIds[0]);
    const size_t rowMetaBytes = layout.rowMeta.size() * sizeof(layout.rowMeta[0]);
    const size_t packedBytes = layout.rowMeta.size() *
        SPMM_FP32_DEGREE2_DEGREE * sizeof(float);
    const size_t fallbackBytes = layout.fallbackRows.size() * sizeof(int32_t);
    const SpmmFp32RegBaseHeader header{static_cast<int32_t>(layout.batches.size()),
        static_cast<int32_t>(layout.rowMeta.size()),
        static_cast<int32_t>(layout.fallbackRows.size()), 0};
    if (!RegBaseWorkspaceFits(off, batchesBytes, rowIdsBytes, rowMetaBytes,
                              packedBytes, sizeof(header) + fallbackBytes)) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (aclrtMemcpy(workspace + off.cubeMetaOff, batchesBytes, layout.batches.data(), batchesBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeRowIdsOff, rowIdsBytes, layout.rowIds.data(), rowIdsBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeSlot1Off, rowMetaBytes, layout.rowMeta.data(), rowMetaBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeScratchOff, sizeof(header), &header, sizeof(header), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        (!layout.fallbackRows.empty() && aclrtMemcpy(workspace + off.cubeScratchOff + sizeof(header), fallbackBytes, layout.fallbackRows.data(), fallbackBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE)) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    *uploaded = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildFp32Degree2Data(
    aclsparseSpMatDescr *matA, uint8_t *workspace, const WsOffsets &off,
    bool candidate, bool *eligible, int32_t *selectedRows,
    int32_t *selectedGroups, int32_t *selectedFallbackCount)
{
    ResetCubeSelection(eligible, selectedRows, selectedGroups, selectedFallbackCount);
    if (!candidate || matA == nullptr || workspace == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    try {
        Fp32Degree2Layout layout;
        aclsparseStatus_t status = PrepareFp32Degree2Layout(matA, &layout);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        bool uploaded = false;
        status = UploadFp32Degree2Layout(workspace, off, layout, &uploaded);
        if (status != ACL_SPARSE_STATUS_SUCCESS || !uploaded) {
            return status;
        }
        *eligible = true;
        *selectedRows = SPMM_FP32_DEGREE2_ROWS;
        *selectedGroups = static_cast<int32_t>(layout.batches.size());
        *selectedFallbackCount = static_cast<int32_t>(layout.fallbackRows.size());
        return ACL_SPARSE_STATUS_SUCCESS;
    } catch (const std::bad_alloc &) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
}

bool IsFp32Degree4StructuralDegree(int32_t degree)
{
    return degree == SPMM_FP32_DEGREE4_DEGREE ||
        degree == SPMM_FP32_DEGREE4_MAX_DEGREE ||
        degree == SPMM_FP32_DEGREE16_DEGREE ||
        degree == SPMM_FP32_DEGREE32_DEGREE ||
        degree == SPMM_FP32_DEGREE64_DEGREE;
}

void SortFp32Degree4WrapCandidates(
    std::vector<Fp32Degree2CandidateRow> *wrapCandidates)
{
    std::stable_sort(wrapCandidates->begin(), wrapCandidates->end(),
        [](const Fp32Degree2CandidateRow &lhs,
           const Fp32Degree2CandidateRow &rhs) {
            if (lhs.start != rhs.start) {
                return lhs.start < rhs.start;
            }
            return lhs.row < rhs.row;
        });
}

aclsparseStatus_t PrepareFp32Degree4Layout(
    const aclsparseSpMatDescr *matA, const Fp32Degree4Config &config,
    Fp32Degree2Layout *layout)
{
    CsrHostData csr;
    const aclsparseStatus_t status = LoadCsrHostData(matA, &csr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    std::vector<Fp32Degree2CandidateRow> candidates;
    std::vector<Fp32Degree2CandidateRow> wrapCandidates;
    uint64_t mappedNnz = 0;
    const aclsparseStatus_t classifyStatus = ClassifyFp32Degree4Rows(
        csr.rowOffsets, csr.columnIndices, static_cast<int64_t>(matA->rows),
        static_cast<int64_t>(matA->cols), csr.indexBase, matA->nnz,
        matA->spmmUniformDegree, &candidates, &wrapCandidates,
        &layout->fallbackRows, &mappedNnz);
    if (classifyStatus != ACL_SPARSE_STATUS_SUCCESS ||
        layout->fallbackRows.size() > 64 ||
        static_cast<__int128>(mappedNnz) * 1000 <
            static_cast<__int128>(matA->nnz) * 995) {
        return classifyStatus;
    }
    SortFp32Candidates(&candidates);
    SortFp32Degree4WrapCandidates(&wrapCandidates);
    BuildFp32Degree4Batches(candidates, wrapCandidates, matA->cols,
                            matA->spmmUniformDegree, config.rows,
                            config.windowRows, &layout->batches,
                            &layout->rowIds, &layout->rowMeta);
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool IsUsableFp32Degree4Layout(const Fp32Degree2Layout &layout, int32_t n,
                               const Fp32Degree4Config &config)
{
    const uint64_t tileCount =
        (static_cast<uint64_t>(n) + config.tileN - 1) / config.tileN;
    const uint64_t taskCount = layout.batches.size() * tileCount;
    return !layout.batches.empty() &&
        layout.rowIds.size() >= layout.batches.size() * config.minAverageRows &&
        taskCount <= static_cast<uint64_t>(config.maxTasks) &&
        layout.rowIds.size() <= static_cast<size_t>(INT32_MAX) &&
        layout.batches.size() <= static_cast<size_t>(INT32_MAX);
}

aclsparseStatus_t UploadFp32Degree4Layout(
    uint8_t *workspace, const WsOffsets &off, int32_t structuralDegree,
    const Fp32Degree2Layout &layout, bool *uploaded)
{
    *uploaded = false;
    const size_t batchesBytes = layout.batches.size() * sizeof(layout.batches[0]);
    const size_t rowIdsBytes = layout.rowIds.size() * sizeof(layout.rowIds[0]);
    const size_t rowMetaBytes = layout.rowMeta.size() * sizeof(layout.rowMeta[0]);
    const size_t packedBytes = layout.rowMeta.size() * structuralDegree * sizeof(float);
    const size_t fallbackBytes = layout.fallbackRows.size() * sizeof(int32_t);
    const SpmmFp32RegBaseHeader header{static_cast<int32_t>(layout.batches.size()),
        static_cast<int32_t>(layout.rowMeta.size()),
        static_cast<int32_t>(layout.fallbackRows.size()), 0};
    if (!RegBaseWorkspaceFits(off, batchesBytes, rowIdsBytes, rowMetaBytes,
                              packedBytes, sizeof(header) + fallbackBytes)) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (aclrtMemcpy(workspace + off.cubeMetaOff, batchesBytes, layout.batches.data(),
                    batchesBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeRowIdsOff, rowIdsBytes, layout.rowIds.data(),
                    rowIdsBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeSlot1Off, rowMetaBytes, layout.rowMeta.data(),
                    rowMetaBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeScratchOff, sizeof(header), &header,
                    sizeof(header), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        (!layout.fallbackRows.empty() && aclrtMemcpy(
            workspace + off.cubeScratchOff + sizeof(header), fallbackBytes,
            layout.fallbackRows.data(), fallbackBytes,
            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE)) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    *uploaded = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildFp32Degree4Data(
    aclsparseSpMatDescr *matA, int32_t n, uint8_t *workspace,
    const WsOffsets &off, bool candidate, bool *eligible,
    int32_t *selectedRows, int32_t *selectedGroups,
    int32_t *selectedFallbackCount)
{
    ResetCubeSelection(eligible, selectedRows, selectedGroups,
                       selectedFallbackCount);
    if (!candidate || matA == nullptr || workspace == nullptr || n <= 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    try {
        const int32_t structuralDegree = matA->spmmUniformDegree;
        if (!IsFp32Degree4StructuralDegree(structuralDegree)) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        const Fp32Degree4Config config = GetFp32Degree4Config(structuralDegree);
        Fp32Degree2Layout layout;
        aclsparseStatus_t status = PrepareFp32Degree4Layout(matA, config, &layout);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        if (!IsUsableFp32Degree4Layout(layout, n, config)) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        bool uploaded = false;
        status = UploadFp32Degree4Layout(workspace, off, structuralDegree,
                                         layout, &uploaded);
        if (status != ACL_SPARSE_STATUS_SUCCESS || !uploaded) {
            return status;
        }
        *eligible = true;
        *selectedRows =
            structuralDegree == SPMM_FP32_DEGREE64_DEGREE ?
                SPMM_FP32_DEGREE64_MODE : config.rows;
        *selectedGroups = static_cast<int32_t>(layout.batches.size());
        *selectedFallbackCount = static_cast<int32_t>(layout.fallbackRows.size());
        return ACL_SPARSE_STATUS_SUCCESS;
    } catch (const std::bad_alloc &) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
}

struct Fp32RegBaseCandidateRow {
    int32_t start;
    int32_t row;
    int32_t degree;
    int32_t valueBegin;
};

struct Fp32RegBaseLayout {
    std::vector<Fp32RegBaseCandidateRow> candidates;
    std::vector<int32_t> fallbackRows;
    std::vector<SpmmFp32RegBaseBatchMeta> batches;
    std::vector<int32_t> rowIds;
    std::vector<int32_t> valueBegins;
    std::vector<int32_t> candidateDegrees;
    std::vector<int32_t> localStarts;
    std::vector<SpmmFp32RegBaseRefreshRun> refreshRuns;
};

aclsparseStatus_t ClassifyFp32RegBaseRow(
    const CsrHostData &csr, int64_t row, int64_t k, uint64_t nnz,
    Fp32RegBaseCandidateRow *candidate)
{
    const int32_t begin = csr.rowOffsets[static_cast<size_t>(row)] - csr.indexBase;
    const int32_t end = csr.rowOffsets[static_cast<size_t>(row) + 1] - csr.indexBase;
    if (!IsValidCsrRowRange(begin, end, nnz)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const int32_t degree = end - begin;
    int32_t start = degree > 0 ?
        csr.columnIndices[static_cast<size_t>(begin)] - csr.indexBase : -1;
    bool eligible = degree >= 6 && degree <= SPMM_FP32_REGBASE_DEGREE &&
        start >= 0 && start < k &&
        static_cast<int64_t>(start) + SPMM_FP32_REGBASE_DEGREE <= k;
    for (int32_t ordinal = 0; ordinal < degree; ++ordinal) {
        const int32_t col = csr.columnIndices[static_cast<size_t>(begin + ordinal)] -
            csr.indexBase;
        if (col < 0 || col >= k) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        eligible = eligible && col == start + ordinal;
    }
    if (eligible) {
        *candidate = {start, static_cast<int32_t>(row), degree, begin};
    }
    return eligible ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_NOT_SUPPORTED;
}

aclsparseStatus_t ClassifyFp32RegBaseRows(
    const aclsparseSpMatDescr *matA, const CsrHostData &csr,
    Fp32RegBaseLayout *layout, uint64_t *mappedNnz)
{
    const int64_t m = static_cast<int64_t>(matA->rows);
    const int64_t k = static_cast<int64_t>(matA->cols);
    layout->candidates.reserve(static_cast<size_t>(m));
    layout->fallbackRows.reserve(64);
    for (int64_t row = 0; row < m; ++row) {
        Fp32RegBaseCandidateRow item{};
        const aclsparseStatus_t status = ClassifyFp32RegBaseRow(
            csr, row, k, matA->nnz, &item);
        if (status == ACL_SPARSE_STATUS_INVALID_VALUE) {
            return status;
        }
        if (status == ACL_SPARSE_STATUS_SUCCESS) {
            layout->candidates.push_back(item);
            *mappedNnz += static_cast<uint64_t>(item.degree);
        } else {
            layout->fallbackRows.push_back(static_cast<int32_t>(row));
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

void SortFp32RegBaseCandidates(
    std::vector<Fp32RegBaseCandidateRow> *candidates)
{
    std::stable_sort(candidates->begin(), candidates->end(),
        [](const Fp32RegBaseCandidateRow &lhs,
           const Fp32RegBaseCandidateRow &rhs) {
            return lhs.start != rhs.start ? lhs.start < rhs.start : lhs.row < rhs.row;
        });
}

void AppendFp32RegBaseBatch(const std::vector<Fp32RegBaseCandidateRow> &candidates,
    size_t cursor, size_t count, Fp32RegBaseLayout *layout,
    std::vector<int32_t> *candidateSlots)
{
    const int32_t windowBegin = candidates[cursor].start;
    const int32_t windowRows = candidates[cursor + count - 1].start - windowBegin +
        SPMM_FP32_REGBASE_DEGREE;
    layout->batches.push_back({windowBegin, static_cast<int32_t>(count),
        static_cast<int32_t>(layout->rowIds.size()), windowRows});
    for (size_t local = 0; local < count; ++local) {
        const auto &item = candidates[cursor + local];
        (*candidateSlots)[static_cast<size_t>(item.row)] =
            static_cast<int32_t>(layout->rowIds.size());
        layout->rowIds.push_back(item.row);
        layout->valueBegins.push_back(item.valueBegin);
        layout->candidateDegrees.push_back(item.degree);
        layout->localStarts.push_back(item.start - windowBegin);
    }
}

void BuildFp32RegBaseBatches(Fp32RegBaseLayout *layout, int64_t m,
                             std::vector<int32_t> *candidateSlots)
{
    layout->rowIds.reserve(layout->candidates.size());
    layout->valueBegins.reserve(layout->candidates.size());
    layout->candidateDegrees.reserve(layout->candidates.size());
    layout->localStarts.reserve(layout->candidates.size());
    candidateSlots->assign(static_cast<size_t>(m), -1);
    for (size_t cursor = 0; cursor < layout->candidates.size();) {
        const int32_t windowBegin = layout->candidates[cursor].start;
        size_t count = 1;
        while (cursor + count < layout->candidates.size() &&
               count < SPMM_FP32_REGBASE_ROWS &&
               layout->candidates[cursor + count].start - windowBegin +
                   SPMM_FP32_REGBASE_DEGREE <= SPMM_FP32_REGBASE_WINDOW_ROWS) {
            ++count;
        }
        AppendFp32RegBaseBatch(layout->candidates, cursor, count, layout,
                               candidateSlots);
        cursor += count;
    }
}

int32_t GetFp32RegBasePackedStride(const std::vector<int32_t> &candidateSlots,
    const std::vector<int32_t> &rowOffsets, int64_t row, int64_t m,
    int32_t degree)
{
    if (row + 1 >= m) {
        return 1;
    }
    const int32_t firstSlot = candidateSlots[static_cast<size_t>(row)];
    const int32_t nextSlot = candidateSlots[static_cast<size_t>(row) + 1];
    const int32_t nextDegree = rowOffsets[static_cast<size_t>(row) + 2] -
        rowOffsets[static_cast<size_t>(row) + 1];
    const int32_t delta = nextSlot - firstSlot;
    return nextSlot >= 0 && nextDegree == degree && delta > 0 && delta <= 4096 ?
        delta : 1;
}

void BuildFp32RegBaseRefreshRuns(const CsrHostData &csr, int64_t m,
    const std::vector<int32_t> &candidateSlots, Fp32RegBaseLayout *layout)
{
    layout->refreshRuns.reserve(layout->candidates.size() / 512 + 16);
    for (int64_t row = 0; row < m;) {
        const int32_t firstSlot = candidateSlots[static_cast<size_t>(row)];
        if (firstSlot < 0) {
            ++row;
            continue;
        }
        const int32_t degree = csr.rowOffsets[static_cast<size_t>(row) + 1] -
            csr.rowOffsets[static_cast<size_t>(row)];
        const int32_t stride = GetFp32RegBasePackedStride(candidateSlots,
            csr.rowOffsets, row, m, degree);
        int32_t count = 1;
        while (row + count < m && count < SPMM_FP32_REGBASE_MAX_REFRESH_ROWS &&
               candidateSlots[static_cast<size_t>(row + count)] ==
                   firstSlot + count * stride &&
               csr.rowOffsets[static_cast<size_t>(row + count) + 1] -
                   csr.rowOffsets[static_cast<size_t>(row + count)] == degree) {
            ++count;
        }
        layout->refreshRuns.push_back({
            csr.rowOffsets[static_cast<size_t>(row)] - csr.indexBase,
            count, degree, firstSlot, stride});
        row += count;
    }
    if (layout->refreshRuns.size() > layout->candidates.size() / 512 + 16) {
        layout->refreshRuns.clear();
    }
}

bool IsUsableFp32RegBaseLayout(const Fp32RegBaseLayout &layout)
{
    const uint64_t packedElements = static_cast<uint64_t>(layout.candidates.size()) *
        SPMM_FP32_REGBASE_DEGREE;
    return !layout.batches.empty() &&
        layout.candidates.size() >= layout.batches.size() * 120 &&
        layout.batches.size() <= static_cast<size_t>(INT32_MAX) &&
        layout.candidates.size() <= static_cast<size_t>(INT32_MAX) &&
        layout.fallbackRows.size() <= static_cast<size_t>(INT32_MAX) &&
        packedElements <= std::numeric_limits<size_t>::max() / sizeof(float);
}

std::vector<int32_t> BuildFp32RegBaseSlot1(const Fp32RegBaseLayout &layout)
{
    std::vector<int32_t> slot1;
    slot1.reserve(layout.fallbackRows.size() + layout.candidates.size() * 3);
    slot1.insert(slot1.end(), layout.fallbackRows.begin(), layout.fallbackRows.end());
    slot1.insert(slot1.end(), layout.valueBegins.begin(), layout.valueBegins.end());
    slot1.insert(slot1.end(), layout.candidateDegrees.begin(), layout.candidateDegrees.end());
    slot1.insert(slot1.end(), layout.localStarts.begin(), layout.localStarts.end());
    return slot1;
}

aclsparseStatus_t UploadFp32RegBaseLayout(uint8_t *workspace, const WsOffsets &off,
    const Fp32RegBaseLayout &layout, bool *uploaded)
{
    *uploaded = false;
    const std::vector<int32_t> slot1 = BuildFp32RegBaseSlot1(layout);
    const size_t batchesBytes = layout.batches.size() * sizeof(layout.batches[0]);
    const size_t rowIdsBytes = layout.rowIds.size() * sizeof(layout.rowIds[0]);
    const size_t slot1Bytes = slot1.size() * sizeof(slot1[0]);
    const size_t packedBytes = layout.candidates.size() *
        SPMM_FP32_REGBASE_DEGREE * sizeof(float);
    const size_t controlBytes = sizeof(SpmmFp32RegBaseHeader) +
        layout.refreshRuns.size() * sizeof(layout.refreshRuns[0]);
    if (!RegBaseWorkspaceFits(off, batchesBytes, rowIdsBytes, slot1Bytes,
                              packedBytes, controlBytes)) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const SpmmFp32RegBaseHeader header{static_cast<int32_t>(layout.batches.size()),
        static_cast<int32_t>(layout.refreshRuns.size()),
        static_cast<int32_t>(layout.fallbackRows.size()),
        static_cast<int32_t>(layout.candidates.size())};
    if (aclrtMemcpy(workspace + off.cubeMetaOff, batchesBytes, layout.batches.data(),
                    batchesBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeRowIdsOff, rowIdsBytes, layout.rowIds.data(),
                    rowIdsBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        (!slot1.empty() && aclrtMemcpy(workspace + off.cubeSlot1Off, slot1Bytes,
            slot1.data(), slot1Bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) ||
        aclrtMemset(workspace + off.cubePackedAOff, packedBytes, 0, packedBytes) !=
            ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeScratchOff, sizeof(header), &header,
                    sizeof(header), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        (!layout.refreshRuns.empty() && aclrtMemcpy(
            workspace + off.cubeScratchOff + sizeof(header),
            layout.refreshRuns.size() * sizeof(layout.refreshRuns[0]),
            layout.refreshRuns.data(),
            layout.refreshRuns.size() * sizeof(layout.refreshRuns[0]),
            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE)) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    *uploaded = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildFp32RegBaseData(
    aclsparseSpMatDescr *matA, uint8_t *workspace, const WsOffsets &off,
    bool candidate, bool *eligible, int32_t *selectedRows,
    int32_t *selectedGroups)
{
    ResetCubeSelection(eligible, selectedRows, selectedGroups);
    if (!candidate || matA == nullptr || workspace == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    try {
        CsrHostData csr;
        aclsparseStatus_t status = LoadCsrHostData(matA, &csr);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        Fp32RegBaseLayout layout;
        uint64_t mappedNnz = 0;
        status = ClassifyFp32RegBaseRows(matA, csr, &layout, &mappedNnz);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        if (layout.candidates.size() < 65536 || layout.fallbackRows.size() > 64 ||
            static_cast<__int128>(mappedNnz) * 1000 <
                static_cast<__int128>(matA->nnz) * 999 ||
            mappedNnz < static_cast<uint64_t>(layout.candidates.size()) * 6) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        SortFp32RegBaseCandidates(&layout.candidates);
        std::vector<int32_t> candidateSlots;
        BuildFp32RegBaseBatches(&layout, matA->rows, &candidateSlots);
        BuildFp32RegBaseRefreshRuns(csr, matA->rows, candidateSlots, &layout);
        if (!IsUsableFp32RegBaseLayout(layout)) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        bool uploaded = false;
        status = UploadFp32RegBaseLayout(workspace, off, layout, &uploaded);
        if (status != ACL_SPARSE_STATUS_SUCCESS || !uploaded) {
            return status;
        }
        *eligible = true;
        *selectedRows = SPMM_FP32_REGBASE_ROWS;
        *selectedGroups = static_cast<int32_t>(layout.batches.size());
        return ACL_SPARSE_STATUS_SUCCESS;
    } catch (const std::bad_alloc &) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
}

struct Fp32PackedSegment {
    int32_t row;
    int32_t start;
    int32_t valueStart;
    std::vector<float> values;
};

constexpr int32_t kFp32CubeMaxWorkAmplification = 4;

bool BuildFp32CubeTiling(int32_t rows, int32_t packedK, int32_t n,
                         uint8_t *buffer)
{
#ifndef __CCE_AICORE__
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr || buffer == nullptr) {
        return false;
    }
    matmul_tiling::MultiCoreMatmulTiling api(*platform);
    // The outer SpMM scheduler assigns complete independent tiles to AICs.
    // Each pure-Cube Matmul instance therefore consumes a one-core tiling.
    api.SetDim(1);
    api.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT, false);
    api.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT, false);
    api.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT);
    api.SetOrgShape(rows, n, packedK);
    api.SetShape(rows, n, packedK);
    api.SetSingleShape(rows, SPMM_CUBE_MAX_N, -1);
    api.SetBias(false);
    api.SetBufferSpace(-1, -1, -1);
    optiling::TCubeTiling tiling;
    if (api.GetTiling(tiling) == -1 || tiling.GetDataSize() > SPMM_CUBE_TILING_BYTES ||
        tiling.get_singleCoreN() <= 0) {
        return false;
    }
    std::fill(buffer, buffer + SPMM_CUBE_TILING_BYTES, 0);
    tiling.SaveToBuffer(buffer, tiling.GetDataSize());
    return true;
#else
    (void)rows;
    (void)packedK;
    (void)n;
    (void)buffer;
    return false;
#endif
}

struct Fp32PackedHostData {
    std::vector<int32_t> rowOffsets;
    std::vector<int32_t> columnIndices;
    std::vector<float> values;
};

struct Fp32PackedLayout {
    std::vector<Fp32PackedSegment> segments;
    std::vector<int32_t> segmentOrder;
    std::vector<std::vector<int32_t>> groups;
    std::vector<int32_t> groupBase;
    int32_t groupRows{SPMM_FP32_CUBE_FAST_ROWS};
    int32_t packedK{SPMM_FP32_CUBE_FAST_K};
};

struct Fp32PackedBuffers {
    std::vector<SpmmCubeGroupMeta> meta;
    std::vector<float> packedA;
    std::vector<int32_t> firstSlot;
    std::vector<int32_t> secondSlot;
    std::vector<uint32_t> valueMap;
};

aclsparseStatus_t LoadFp32PackedHostData(const aclsparseSpMatDescr *matA,
                                         Fp32PackedHostData *host)
{
    host->rowOffsets.resize(static_cast<size_t>(matA->rows) + 1);
    host->columnIndices.resize(static_cast<size_t>(matA->nnz));
    host->values.resize(static_cast<size_t>(matA->nnz));
    if (aclrtMemcpy(host->rowOffsets.data(),
                    host->rowOffsets.size() * sizeof(int32_t), matA->ptrs,
                    host->rowOffsets.size() * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE ||
        aclrtMemcpy(host->columnIndices.data(),
                    host->columnIndices.size() * sizeof(int32_t), matA->idxs,
                    host->columnIndices.size() * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE ||
        aclrtMemcpy(host->values.data(), host->values.size() * sizeof(float),
                    matA->values, host->values.size() * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildFp32PackedSegments(
    const aclsparseSpMatDescr *matA, const Fp32PackedHostData &host,
    std::vector<Fp32PackedSegment> *segments, bool *ordered)
{
    *ordered = false;
    const int32_t indexBase = matA->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    segments->reserve(static_cast<size_t>(matA->rows) * 2);
    for (int64_t row = 0; row < matA->rows; ++row) {
        const int32_t begin = host.rowOffsets[static_cast<size_t>(row)] - indexBase;
        const int32_t end = host.rowOffsets[static_cast<size_t>(row) + 1] - indexBase;
        if (!IsValidCsrRowRange(begin, end, matA->nnz)) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        int32_t previous = -1;
        Fp32PackedSegment current{};
        for (int32_t p = begin; p < end; ++p) {
            const int32_t column = host.columnIndices[static_cast<size_t>(p)] - indexBase;
            if (column < 0 || static_cast<uint64_t>(column) >= matA->cols) {
                return ACL_SPARSE_STATUS_INVALID_VALUE;
            }
            if (column <= previous) return ACL_SPARSE_STATUS_SUCCESS;
            if (current.values.empty() || column != previous + 1) {
                if (!current.values.empty()) segments->push_back(std::move(current));
                current = {static_cast<int32_t>(row), column, p, {}};
            }
            current.values.push_back(host.values[static_cast<size_t>(p)]);
            previous = column;
        }
        if (!current.values.empty()) segments->push_back(std::move(current));
    }
    *ordered = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

void BuildFp32PackedSegmentOrder(Fp32PackedLayout *layout)
{
    layout->segmentOrder.resize(layout->segments.size());
    std::iota(layout->segmentOrder.begin(), layout->segmentOrder.end(), 0);
    std::stable_sort(layout->segmentOrder.begin(), layout->segmentOrder.end(),
        [&layout](int32_t lhs, int32_t rhs) {
            const auto &left = layout->segments[static_cast<size_t>(lhs)];
            const auto &right = layout->segments[static_cast<size_t>(rhs)];
            return left.start != right.start ? left.start < right.start :
                left.row < right.row;
        });
}

void SelectFp32PackedGroupShape(Fp32PackedLayout *layout)
{
    for (const Fp32PackedSegment &segment : layout->segments) {
        const int32_t alignedStart = (segment.start / 8) * 8;
        const int32_t segmentEnd = segment.start +
            static_cast<int32_t>(segment.values.size());
        if (segmentEnd - alignedStart > SPMM_FP32_CUBE_FAST_K) {
            layout->groupRows = SPMM_CUBE_MIN_ROWS;
            layout->packedK = SPMM_CUBE_MAX_K;
            return;
        }
    }
}

bool BuildFp32PackedGroups(int64_t columns, Fp32PackedLayout *layout)
{
    for (int32_t index : layout->segmentOrder) {
        const Fp32PackedSegment &segment =
            layout->segments[static_cast<size_t>(index)];
        const int32_t segmentEnd = segment.start +
            static_cast<int32_t>(segment.values.size());
        SpmmCubeGroupMeta window{};
        // B rows are aligned by the row-major N stride. At the K tail,
        // shift the complete window inside B without rounding its row index.
        if (!spmm_host_internal::BuildPackedWindow<8>(columns, segment.start, segmentEnd - 1,
                               layout->packedK, &window) &&
            !spmm_host_internal::BuildPackedWindow<1>(columns, segment.start, segmentEnd - 1,
                               layout->packedK, &window)) return false;
        if (layout->groups.empty() ||
            layout->groups.back().size() == static_cast<size_t>(layout->groupRows) ||
            segmentEnd - layout->groupBase.back() > layout->packedK) {
            layout->groups.emplace_back();
            layout->groupBase.push_back(window.col_base);
        }
        layout->groups.back().push_back(index);
    }
    return !layout->groups.empty();
}

bool HasAcceptableFp32PackedAmplification(const aclsparseSpMatDescr *matA,
                                          const Fp32PackedLayout &layout)
{
    const __int128 cubeElements = static_cast<__int128>(layout.groups.size()) *
        layout.groupRows * layout.packedK;
    return cubeElements <= static_cast<__int128>(matA->nnz) *
        kFp32CubeMaxWorkAmplification;
}

bool PlaceFp32PackedSegment(const Fp32PackedSegment &segment, int32_t slot,
    int32_t groupBase, int32_t packedK, Fp32PackedBuffers *buffers)
{
    int32_t *slotRecord = &buffers->firstSlot[static_cast<size_t>(segment.row)];
    if (*slotRecord >= 0) slotRecord = &buffers->secondSlot[static_cast<size_t>(segment.row)];
    if (*slotRecord >= 0) return false;
    const int32_t localStart = segment.start - groupBase;
    if (localStart < 0 || localStart + static_cast<int32_t>(segment.values.size()) >
        packedK) return false;
    *slotRecord = slot;
    std::copy(segment.values.begin(), segment.values.end(),
              buffers->packedA.begin() +
                  static_cast<size_t>(slot) * packedK + localStart);
    std::iota(buffers->valueMap.begin() + segment.valueStart,
              buffers->valueMap.begin() + segment.valueStart + segment.values.size(),
              static_cast<uint32_t>(static_cast<size_t>(slot) * packedK + localStart));
    return true;
}

bool BuildFp32PackedBuffers(const aclsparseSpMatDescr *matA,
    const Fp32PackedLayout &layout, Fp32PackedBuffers *buffers)
{
    buffers->meta.resize(layout.groups.size());
    const size_t paddedRows = layout.groups.size() *
        static_cast<size_t>(layout.groupRows);
    if (paddedRows > UINT32_MAX / static_cast<size_t>(layout.packedK)) return false;
    buffers->packedA.assign(paddedRows * static_cast<size_t>(layout.packedK), 0.0f);
    buffers->valueMap.assign(static_cast<size_t>(matA->nnz), UINT32_MAX);
    buffers->firstSlot.assign(static_cast<size_t>(matA->rows), -1);
    buffers->secondSlot.assign(static_cast<size_t>(matA->rows), -1);
    for (size_t group = 0; group < layout.groups.size(); ++group) {
        if (layout.groupBase[group] < 0 ||
            static_cast<int64_t>(layout.groupBase[group]) + layout.packedK >
                static_cast<int64_t>(matA->cols)) return false;
        buffers->meta[group] = {layout.groupBase[group], layout.packedK};
        for (size_t local = 0; local < layout.groups[group].size(); ++local) {
            const auto &segment = layout.segments[static_cast<size_t>(
                layout.groups[group][local])];
            const int32_t slot = static_cast<int32_t>(
                group * layout.groupRows + local);
            if (!PlaceFp32PackedSegment(segment, slot, layout.groupBase[group],
                                        layout.packedK, buffers)) return false;
        }
    }
    return true;
}

template <typename T>
aclError UploadFp32PackedBuffer(uint8_t *workspace, int64_t offset,
                                 const std::vector<T> &buffer)
{
    const size_t bytes = buffer.size() * sizeof(T);
    return aclrtMemcpy(workspace + offset, bytes, buffer.data(), bytes,
                       ACL_MEMCPY_HOST_TO_DEVICE);
}

aclsparseStatus_t UploadFp32PackedLayout(uint8_t *workspace,
    const WsOffsets &off, int32_t n, const Fp32PackedLayout &layout,
    const Fp32PackedBuffers &buffers, bool *uploaded)
{
    *uploaded = false;
    if (!spmm_host_internal::WorkspaceRegionFits(off.cubeAValueMapOff, off.cubePackedAOff,
                             buffers.valueMap.size() * sizeof(uint32_t))) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    std::vector<uint8_t> cubeTiling(SPMM_CUBE_TILING_BYTES, 0);
    if (!BuildFp32CubeTiling(layout.groupRows, layout.packedK, n,
                             cubeTiling.data())) return ACL_SPARSE_STATUS_SUCCESS;
    if (UploadFp32PackedBuffer(workspace, off.cubeMetaOff, buffers.meta) != ACL_ERROR_NONE ||
        UploadFp32PackedBuffer(workspace, off.cubeRowIdsOff, buffers.firstSlot) != ACL_ERROR_NONE ||
        UploadFp32PackedBuffer(workspace, off.cubeSlot1Off, buffers.secondSlot) != ACL_ERROR_NONE ||
        UploadFp32PackedBuffer(workspace, off.cubeAValueMapOff, buffers.valueMap) != ACL_ERROR_NONE ||
        UploadFp32PackedBuffer(workspace, off.cubePackedAOff, buffers.packedA) != ACL_ERROR_NONE ||
        UploadFp32PackedBuffer(workspace, off.cubeTilingOff, cubeTiling) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    *uploaded = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}


aclsparseStatus_t BuildFp32PackedData(aclsparseSpMatDescr *matA, int32_t n,
                                      uint8_t *workspace,
                                      const WsOffsets &off, bool candidate,
                                      bool *eligible, int32_t *selectedRows,
                                      int32_t *selectedGroups)
{
    ResetCubeSelection(eligible, selectedRows, selectedGroups);
    if (!candidate || matA == nullptr || matA->rows == 0 || matA->nnz == 0 ||
        matA->ptrType != ACL_SPARSE_INDEX_32I || matA->IdxType != ACL_SPARSE_INDEX_32I) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    Fp32PackedHostData host;
    aclsparseStatus_t status = LoadFp32PackedHostData(matA, &host);
    if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
    Fp32PackedLayout layout;
    bool ordered = false;
    status = BuildFp32PackedSegments(matA, host, &layout.segments, &ordered);
    if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
    if (!ordered) return ACL_SPARSE_STATUS_SUCCESS;
    BuildFp32PackedSegmentOrder(&layout);

    SelectFp32PackedGroupShape(&layout);
    if (!BuildFp32PackedGroups(static_cast<int64_t>(matA->cols), &layout)) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    if (!HasAcceptableFp32PackedAmplification(matA, layout)) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    Fp32PackedBuffers buffers;
    if (!BuildFp32PackedBuffers(matA, layout, &buffers)) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    bool uploaded = false;
    status = UploadFp32PackedLayout(workspace, off, n, layout, buffers, &uploaded);
    if (status != ACL_SPARSE_STATUS_SUCCESS || !uploaded) return status;
    *eligible = true;
    *selectedRows = layout.groupRows;
    *selectedGroups = static_cast<int32_t>(layout.groups.size());
    return ACL_SPARSE_STATUS_SUCCESS;
}


} // namespace spmm_host_builders
