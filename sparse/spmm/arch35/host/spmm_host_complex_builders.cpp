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

namespace spmm_host_builders {

using spmm_host_internal::ComplexRegBaseMinAverageRows;
using spmm_host_internal::ComplexRegBaseRows;
using spmm_host_internal::ComplexRegBaseWindowRows;
using spmm_host_internal::CsrHostData;
using spmm_host_internal::IsComplexRegBaseDegree;
using spmm_host_internal::IsOfficialComplexP03Shape;
using spmm_host_internal::IsValidCsrRowRange;
using spmm_host_internal::LoadCsrHostData;
using spmm_host_internal::RegBaseWorkspaceFits;
using spmm_host_internal::ResetCubeSelection;
using spmm_host_internal::WsOffsets;

struct ComplexRegBaseCandidateRow {
    int32_t start;
    int32_t row;
    int32_t valueOffset;
    int32_t lowCount;
    int32_t highCount;
};

struct ComplexRegBaseLayout {
    std::vector<SpmmFp32Degree2BatchMeta> batches;
    std::vector<int32_t> rowIds;
    std::vector<SpmmFp32Degree2RowMeta> rowMeta;
};

aclsparseStatus_t ClassifyComplexRegBaseRow(
    const std::vector<int32_t> &rowOff, const std::vector<int32_t> &colInd,
    int64_t row, int64_t k, int32_t indexBase, uint64_t nnz, int32_t degree,
    std::vector<ComplexRegBaseCandidateRow> *candidates,
    std::vector<ComplexRegBaseCandidateRow> *wrapCandidates, bool *supported)
{
    const int32_t begin = rowOff[static_cast<size_t>(row)] - indexBase;
    const int32_t end = rowOff[static_cast<size_t>(row) + 1] - indexBase;
    if (!IsValidCsrRowRange(begin, end, nnz)) return ACL_SPARSE_STATUS_INVALID_VALUE;
    if (end - begin != degree) { *supported = false; return ACL_SPARSE_STATUS_SUCCESS; }
    int32_t columns[SPMM_COMPLEX64_REGBASE_MAX_DEGREE]{};
    for (int32_t ordinal = 0; ordinal < degree; ++ordinal) {
        columns[ordinal] = colInd[static_cast<size_t>(begin + ordinal)] - indexBase;
        if (columns[ordinal] < 0 || static_cast<int64_t>(columns[ordinal]) >= k ||
            (ordinal > 0 && columns[ordinal] <= columns[ordinal - 1])) {
            *supported = false;
            return ACL_SPARSE_STATUS_SUCCESS;
        }
    }
    int32_t split = 0;
    while (split + 1 < degree && columns[split + 1] == columns[split] + 1) ++split;
    if (split + 1 == degree) {
        candidates->push_back({columns[0], static_cast<int32_t>(row), begin, 0, 0});
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const int32_t lowCount = split + 1;
    const int32_t highCount = degree - lowCount;
    bool wrap = lowCount > 0 && highCount > 0 && columns[0] == 0;
    for (int32_t ordinal = split + 2; ordinal < degree; ++ordinal) {
        wrap = wrap && columns[ordinal] == columns[ordinal - 1] + 1;
    }
    const int32_t highStart = columns[split + 1];
    if (!wrap || static_cast<int64_t>(highStart) + highCount != k) {
        *supported = false;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    wrapCandidates->push_back({highStart, static_cast<int32_t>(row), begin, lowCount, highCount});
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t CollectComplexRegBaseCandidates(
    const aclsparseSpMatDescr *matA, const CsrHostData &csr,
    std::vector<ComplexRegBaseCandidateRow> *candidates,
    std::vector<ComplexRegBaseCandidateRow> *wrapCandidates)
{
    bool supported = true;
    candidates->reserve(static_cast<size_t>(matA->rows));
    for (int64_t row = 0; row < static_cast<int64_t>(matA->rows) && supported; ++row) {
        const aclsparseStatus_t status = ClassifyComplexRegBaseRow(
            csr.rowOffsets, csr.columnIndices, row, static_cast<int64_t>(matA->cols),
            csr.indexBase, matA->nnz, matA->spmmUniformDegree, candidates,
            wrapCandidates, &supported);
        if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
    }
    return supported ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_NOT_SUPPORTED;
}

void SortComplexRegBaseCandidates(
    std::vector<ComplexRegBaseCandidateRow> *candidates,
    std::vector<ComplexRegBaseCandidateRow> *wrapCandidates)
{
    std::stable_sort(candidates->begin(), candidates->end(), [](const auto &lhs, const auto &rhs) {
        return lhs.start != rhs.start ? lhs.start < rhs.start : lhs.row < rhs.row;
    });
    std::stable_sort(wrapCandidates->begin(), wrapCandidates->end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.start != rhs.start) return lhs.start < rhs.start;
        return lhs.highCount != rhs.highCount ? lhs.highCount < rhs.highCount : lhs.row < rhs.row;
    });
}

void BuildComplexRegBaseBatches(
    const std::vector<ComplexRegBaseCandidateRow> &candidates,
    const std::vector<ComplexRegBaseCandidateRow> &wrapCandidates, int32_t degree,
    int32_t rowsPerGroup, int32_t maxWindowRows, ComplexRegBaseLayout *layout)
{
    layout->batches.reserve(candidates.size() / 80 + 2);
    size_t cursor = 0;
    while (cursor < candidates.size()) {
        const int32_t base = candidates[cursor].start;
        size_t end = cursor;
        int32_t windowEnd = base;
        while (end < candidates.size() && end - cursor < static_cast<size_t>(rowsPerGroup) &&
               candidates[end].start + degree - base <= maxWindowRows) {
            windowEnd = std::max(windowEnd, candidates[end].start + degree);
            ++end;
        }
        const int32_t metaBegin = static_cast<int32_t>(layout->rowMeta.size());
        layout->batches.push_back({base, static_cast<int32_t>(end - cursor), metaBegin, windowEnd - base});
        for (size_t index = cursor; index < end; ++index) {
            const auto &item = candidates[index];
            layout->rowIds.push_back(item.row);
            layout->rowMeta.push_back({item.start - base, item.valueOffset});
        }
        cursor = end;
    }
    cursor = 0;
    while (cursor < wrapCandidates.size()) {
        const auto &first = wrapCandidates[cursor];
        size_t end = cursor;
        while (end < wrapCandidates.size() && end - cursor < static_cast<size_t>(rowsPerGroup) &&
               wrapCandidates[end].start == first.start && wrapCandidates[end].highCount == first.highCount) ++end;
        const int32_t metaBegin = static_cast<int32_t>(layout->rowMeta.size());
        layout->batches.push_back({-first.start - 1, static_cast<int32_t>(end - cursor), metaBegin, first.lowCount});
        for (size_t index = cursor; index < end; ++index) {
            layout->rowIds.push_back(wrapCandidates[index].row);
            layout->rowMeta.push_back({0, wrapCandidates[index].valueOffset});
        }
        cursor = end;
    }
}

aclsparseStatus_t UploadComplexRegBaseLayout(
    uint8_t *workspace, const WsOffsets &off, int32_t degree,
    const ComplexRegBaseLayout &layout, bool *uploaded)
{
    *uploaded = false;
    const size_t batchesBytes = layout.batches.size() * sizeof(layout.batches[0]);
    const size_t rowIdsBytes = layout.rowIds.size() * sizeof(layout.rowIds[0]);
    const size_t rowMetaBytes = layout.rowMeta.size() * sizeof(layout.rowMeta[0]);
    const size_t packedBytes = layout.rowMeta.size() * static_cast<size_t>(degree) * sizeof(uint64_t);
    const SpmmFp32RegBaseHeader header{static_cast<int32_t>(layout.batches.size()), static_cast<int32_t>(layout.rowMeta.size()), 0, 0};
    if (!RegBaseWorkspaceFits(off, batchesBytes, rowIdsBytes, rowMetaBytes, packedBytes, sizeof(header))) return ACL_SPARSE_STATUS_SUCCESS;
    if (aclrtMemcpy(workspace + off.cubeMetaOff, batchesBytes, layout.batches.data(), batchesBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeRowIdsOff, rowIdsBytes, layout.rowIds.data(), rowIdsBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeSlot1Off, rowMetaBytes, layout.rowMeta.data(), rowMetaBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeScratchOff, sizeof(header), &header, sizeof(header), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    *uploaded = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildComplexRegBaseData(
    aclsparseSpMatDescr *matA, uint8_t *workspace, const WsOffsets &off,
    bool candidate, bool *eligible, int32_t *selectedRows, int32_t *selectedGroups)
{
    ResetCubeSelection(eligible, selectedRows, selectedGroups);
    if (!candidate || matA == nullptr || workspace == nullptr ||
        !IsComplexRegBaseDegree(matA->spmmUniformDegree)) return ACL_SPARSE_STATUS_SUCCESS;
    try {
        CsrHostData csr;
        aclsparseStatus_t status = LoadCsrHostData(matA, &csr);
        if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
        std::vector<ComplexRegBaseCandidateRow> candidates;
        std::vector<ComplexRegBaseCandidateRow> wraps;
        status = CollectComplexRegBaseCandidates(matA, csr, &candidates, &wraps);
        if (status == ACL_SPARSE_STATUS_NOT_SUPPORTED) return ACL_SPARSE_STATUS_SUCCESS;
        if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
        SortComplexRegBaseCandidates(&candidates, &wraps);
        ComplexRegBaseLayout layout;
        layout.rowIds.reserve(static_cast<size_t>(matA->rows));
        layout.rowMeta.reserve(static_cast<size_t>(matA->rows));
        const int32_t degree = matA->spmmUniformDegree;
        BuildComplexRegBaseBatches(candidates, wraps, degree, ComplexRegBaseRows(degree), ComplexRegBaseWindowRows(degree), &layout);
        if (layout.batches.empty() || layout.rowIds.size() != matA->rows || layout.rowMeta.size() != matA->rows ||
            layout.rowIds.size() < layout.batches.size() * static_cast<size_t>(ComplexRegBaseMinAverageRows(degree)) ||
            layout.batches.size() > static_cast<size_t>(INT32_MAX)) return ACL_SPARSE_STATUS_SUCCESS;
        bool uploaded = false;
        status = UploadComplexRegBaseLayout(workspace, off, degree, layout, &uploaded);
        if (status != ACL_SPARSE_STATUS_SUCCESS || !uploaded) return status;
        *eligible = true;
        *selectedRows = ComplexRegBaseRows(degree);
        *selectedGroups = static_cast<int32_t>(layout.batches.size());
        return ACL_SPARSE_STATUS_SUCCESS;
    } catch (const std::bad_alloc &) { return ACL_SPARSE_STATUS_ALLOC_FAILED; }
}

struct ComplexDegree64Segment {
    int32_t row;
    int32_t begin;
    int32_t end;
    int32_t minCol;
    int32_t maxCol;
};

aclsparseStatus_t ValidateComplexDegree64Row(
    const std::vector<int32_t> &rowOff, const std::vector<int32_t> &colInd,
    int64_t row, int64_t k, int32_t indexBase, uint64_t nnz,
    int32_t *begin, int32_t *end, bool *eligible)
{
    *begin = rowOff[static_cast<size_t>(row)] - indexBase;
    *end = rowOff[static_cast<size_t>(row) + 1] - indexBase;
    if (!IsValidCsrRowRange(*begin, *end, nnz)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *eligible = *end - *begin == SPMM_FP32_DEGREE64_DEGREE;
    int32_t previous = -1;
    for (int32_t p = *begin; p < *end; ++p) {
        const int32_t col = colInd[static_cast<size_t>(p)] - indexBase;
        if (col < 0 || static_cast<int64_t>(col) >= k) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        *eligible = *eligible && col > previous;
        previous = col;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

void AppendComplexDegree64Run(
    const std::vector<int32_t> &colInd, int64_t row, int32_t indexBase,
    int32_t begin, int32_t end, std::vector<ComplexDegree64Segment> *segments)
{
    for (int32_t first = begin; first < end; first += 32) {
        const int32_t last = std::min(first + 32, end);
        segments->push_back({static_cast<int32_t>(row), first, last,
            colInd[static_cast<size_t>(first)] - indexBase,
            colInd[static_cast<size_t>(last - 1)] - indexBase});
    }
}

void BuildComplexDegree64RowSegments(
    const std::vector<int32_t> &colInd, int64_t row, int32_t indexBase,
    int32_t begin, int32_t end, std::vector<ComplexDegree64Segment> *segments)
{
    int32_t runBegin = begin;
    for (int32_t p = begin + 1; p < end; ++p) {
        const int32_t col = colInd[static_cast<size_t>(p)] - indexBase;
        const int32_t prior = colInd[static_cast<size_t>(p - 1)] - indexBase;
        if (col != prior + 1) {
            AppendComplexDegree64Run(colInd, row, indexBase, runBegin, p, segments);
            runBegin = p;
        }
    }
    AppendComplexDegree64Run(colInd, row, indexBase, runBegin, end, segments);
}

aclsparseStatus_t BuildComplexDegree64Segments(
    const std::vector<int32_t> &rowOff, const std::vector<int32_t> &colInd,
    int64_t m, int64_t k, int32_t indexBase, uint64_t nnz,
    std::vector<ComplexDegree64Segment> *segments,
    std::vector<int32_t> *fallbackRows, std::vector<uint8_t> *segmentedRows)
{
    segments->reserve(static_cast<size_t>(m) * 2);
    for (int64_t row = 0; row < m; ++row) {
        int32_t begin = 0;
        int32_t end = 0;
        bool eligible = false;
        const aclsparseStatus_t status = ValidateComplexDegree64Row(
            rowOff, colInd, row, k, indexBase, nnz, &begin, &end, &eligible);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        if (!eligible) {
            fallbackRows->push_back(static_cast<int32_t>(row));
            continue;
        }
        std::vector<ComplexDegree64Segment> rowSegments;
        rowSegments.reserve(SPMM_COMPLEX64_SEGMENTED_MAX_SLOTS);
        BuildComplexDegree64RowSegments(colInd, row, indexBase, begin, end, &rowSegments);
        if (rowSegments.size() < 2 || rowSegments.size() > SPMM_COMPLEX64_SEGMENTED_MAX_SLOTS) {
            fallbackRows->push_back(static_cast<int32_t>(row));
            continue;
        }
        segments->insert(segments->end(), rowSegments.begin(), rowSegments.end());
        (*segmentedRows)[static_cast<size_t>(row)] = 1;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t UploadComplexDegree64Workspace(
    uint8_t *workspace, const WsOffsets &off, int64_t rows,
    const std::vector<SpmmCubeGroupMeta> &meta,
    const std::vector<int32_t> &slotToRow, const std::vector<int32_t> &fallbackRows,
    const std::vector<int32_t> &rowSlots, const std::vector<uint32_t> &valueToPackedSlot,
    const std::vector<uint16_t> &packedA, size_t planeElements)
{
    if (aclrtMemcpy(workspace + off.cubeMetaOff, meta.size() * sizeof(meta[0]), meta.data(), meta.size() * sizeof(meta[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeRowIdsOff, slotToRow.size() * sizeof(slotToRow[0]), slotToRow.data(), slotToRow.size() * sizeof(slotToRow[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeAValueMapOff, valueToPackedSlot.size() * sizeof(valueToPackedSlot[0]), valueToPackedSlot.data(), valueToPackedSlot.size() * sizeof(valueToPackedSlot[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        (!fallbackRows.empty() && aclrtMemcpy(workspace + off.cubeSlot1Off, fallbackRows.size() * sizeof(fallbackRows[0]), fallbackRows.data(), fallbackRows.size() * sizeof(fallbackRows[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) ||
        aclrtMemcpy(workspace + off.cubeSlot1Off + static_cast<size_t>(rows) * sizeof(int32_t), rowSlots.size() * sizeof(rowSlots[0]), rowSlots.data(), rowSlots.size() * sizeof(rowSlots[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    for (size_t plane = 0; plane < 6; ++plane) {
        const uint16_t *source = packedA.data() + plane * planeElements;
        if (aclrtMemcpy(workspace + off.cubePackedAOff + plane * planeElements * sizeof(uint16_t), planeElements * sizeof(uint16_t), source, planeElements * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

struct ComplexDegree64GroupPlan {
    std::vector<std::vector<size_t>> groups;
};

void BuildComplexDegree64GroupPlan(
    std::vector<ComplexDegree64Segment> *segments,
    ComplexDegree64GroupPlan *plan)
{
    std::stable_sort(segments->begin(), segments->end(),
        [](const ComplexDegree64Segment &lhs, const ComplexDegree64Segment &rhs) {
            return lhs.minCol < rhs.minCol;
        });
    int32_t groupMin = 0;
    int32_t groupMax = -1;
    for (size_t index = 0; index < segments->size(); ++index) {
        const ComplexDegree64Segment &segment = (*segments)[index];
        const bool startsGroup = plan->groups.empty() ||
            plan->groups.back().size() == SPMM_CUBE_MIN_ROWS ||
            std::max(groupMax, segment.maxCol) - std::min(groupMin, segment.minCol) >=
                SPMM_CUBE_MAX_K;
        if (startsGroup) {
            plan->groups.emplace_back();
            groupMin = segment.minCol;
            groupMax = segment.maxCol;
        } else {
            groupMin = std::min(groupMin, segment.minCol);
            groupMax = std::max(groupMax, segment.maxCol);
        }
        plan->groups.back().push_back(index);
    }
}

bool IsComplexDegree64MappingConsistent(
    const std::vector<int32_t> &rowOff, int64_t rows, int32_t indexBase,
    const std::vector<uint8_t> &segmentedRows,
    const std::vector<uint32_t> &valueToPackedSlot)
{
    const uint32_t unmapped = std::numeric_limits<uint32_t>::max();
    for (int64_t row = 0; row < rows; ++row) {
        const int32_t begin = rowOff[static_cast<size_t>(row)] - indexBase;
        const int32_t end = rowOff[static_cast<size_t>(row) + 1] - indexBase;
        const bool shouldBeMapped = segmentedRows[static_cast<size_t>(row)] != 0;
        for (int32_t value = begin; value < end; ++value) {
            if ((valueToPackedSlot[static_cast<size_t>(value)] != unmapped) != shouldBeMapped) return false;
        }
    }
    return true;
}

struct ComplexCubeStorage {
    std::vector<SpmmCubeGroupMeta> meta;
    std::vector<int32_t> slotToRow;
    std::vector<uint32_t> valueToPackedSlot;
    std::vector<uint16_t> packedA;
    size_t planeElements = 0;
};

struct ComplexDegree64PackedLayout : ComplexCubeStorage {
    std::vector<int32_t> rowSlots;
};

bool IsComplexDegree64StorageSupported(int64_t groups, int64_t paddedRows,
                                       __int128 *planeElements,
                                       __int128 *packedElements)
{
    *planeElements = static_cast<__int128>(paddedRows) * SPMM_CUBE_MAX_K;
    *packedElements = *planeElements * 6;
    return groups > 0 &&
        *planeElements <= std::numeric_limits<uint32_t>::max() &&
        *packedElements <= std::numeric_limits<size_t>::max();
}

aclsparseStatus_t InitializeComplexCubeStorage(size_t groups, uint64_t nnz,
                                               ComplexCubeStorage *storage)
{
    const int64_t paddedRows = static_cast<int64_t>(groups) * SPMM_CUBE_MIN_ROWS;
    __int128 planeElements = 0;
    __int128 packedElements = 0;
    if (!IsComplexDegree64StorageSupported(static_cast<int64_t>(groups), paddedRows,
                                           &planeElements, &packedElements)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    storage->slotToRow.assign(static_cast<size_t>(paddedRows), -1);
    storage->valueToPackedSlot.assign(static_cast<size_t>(nnz), std::numeric_limits<uint32_t>::max());
    storage->packedA.assign(static_cast<size_t>(packedElements), 0);
    storage->meta.reserve(groups);
    storage->planeElements = static_cast<size_t>(planeElements);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t UploadComplexCubePlanes(uint8_t *workspace, const WsOffsets &off,
                                          const ComplexCubeStorage &storage)
{
    for (size_t plane = 0; plane < 6; ++plane) {
        const uint16_t *source = storage.packedA.data() + plane * storage.planeElements;
        if (aclrtMemcpy(workspace + off.cubePackedAOff + plane * storage.planeElements * sizeof(uint16_t),
                        storage.planeElements * sizeof(uint16_t), source,
                        storage.planeElements * sizeof(uint16_t),
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool MapComplexPackedValue(const std::vector<int32_t> &colInd, int32_t indexBase,
    int32_t colBase, int32_t position, int64_t slot,
    std::vector<uint32_t> *valueToPackedSlot)
{
    const int32_t localK = colInd[static_cast<size_t>(position)] - indexBase - colBase;
    if (localK < 0 || localK >= SPMM_CUBE_MAX_K) return false;
    const size_t element = static_cast<size_t>(slot) * SPMM_CUBE_MAX_K +
        static_cast<size_t>(localK);
    (*valueToPackedSlot)[static_cast<size_t>(position)] =
        static_cast<uint32_t>(element);
    return true;
}

aclsparseStatus_t FillComplexDegree64Group(
    const std::vector<ComplexDegree64Segment> &segments,
    const std::vector<size_t> &groupSegments, const std::vector<int32_t> &colInd,
    int32_t indexBase, int64_t k, int64_t firstSlot,
    ComplexDegree64PackedLayout *layout, bool *packed)
{
    int32_t minCol = static_cast<int32_t>(k);
    int32_t maxCol = -1;
    for (const size_t index : groupSegments) {
        minCol = std::min(minCol, segments[index].minCol);
        maxCol = std::max(maxCol, segments[index].maxCol);
    }
    const int32_t colBase = std::min(minCol, static_cast<int32_t>(k - SPMM_CUBE_MAX_K));
    if (minCol < colBase || maxCol - colBase >= SPMM_CUBE_MAX_K) {
        *packed = false;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const int32_t requiredK = maxCol - colBase + 1;
    layout->meta.emplace_back(SpmmCubeGroupMeta{colBase, requiredK <= 32 ? 32 : SPMM_CUBE_MAX_K});
    for (size_t local = 0; local < groupSegments.size(); ++local) {
        const int64_t slot = firstSlot + static_cast<int64_t>(local);
        const ComplexDegree64Segment &segment = segments[groupSegments[local]];
        layout->slotToRow[static_cast<size_t>(slot)] = segment.row;
        int32_t *slots = layout->rowSlots.data() + static_cast<size_t>(segment.row) * SPMM_COMPLEX64_SEGMENTED_MAX_SLOTS;
        size_t slotIndex = 0;
        while (slotIndex < SPMM_COMPLEX64_SEGMENTED_MAX_SLOTS && slots[slotIndex] >= 0) ++slotIndex;
        if (slotIndex == SPMM_COMPLEX64_SEGMENTED_MAX_SLOTS) {
            *packed = false;
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        slots[slotIndex] = static_cast<int32_t>(slot);
        for (int32_t p = segment.begin; p < segment.end; ++p) {
            if (!MapComplexPackedValue(colInd, indexBase, colBase, p, slot,
                                       &layout->valueToPackedSlot)) {
                *packed = false;
                return ACL_SPARSE_STATUS_SUCCESS;
            }
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildComplexDegree64PackedLayout(
    const std::vector<ComplexDegree64Segment> &segments,
    const std::vector<std::vector<size_t>> &groups, const std::vector<int32_t> &colInd,
    int32_t indexBase, int64_t rows, int64_t k, uint64_t nnz,
    ComplexDegree64PackedLayout *layout, bool *packed)
{
    *packed = false;
    const aclsparseStatus_t storageStatus = InitializeComplexCubeStorage(
        groups.size(), nnz, layout);
    if (storageStatus != ACL_SPARSE_STATUS_SUCCESS) return storageStatus;
    layout->rowSlots.assign(static_cast<size_t>(rows) * SPMM_COMPLEX64_SEGMENTED_MAX_SLOTS, -1);
    for (size_t group = 0; group < groups.size(); ++group) {
        bool groupPacked = true;
        const aclsparseStatus_t status = FillComplexDegree64Group(
            segments, groups[group], colInd, indexBase, k,
            static_cast<int64_t>(group) * SPMM_CUBE_MIN_ROWS, layout, &groupPacked);
        if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
        if (!groupPacked) return ACL_SPARSE_STATUS_SUCCESS;
    }
    *packed = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool IsComplexDegree64RequestSupported(const aclsparseSpMatDescr *matA,
                                       const aclsparseDnMatDescr *matB, int32_t n,
                                       bool candidate)
{
    return candidate && matA != nullptr && matB != nullptr &&
        matA->spmmUniformDegree == SPMM_FP32_DEGREE64_DEGREE &&
        matA->rows != 0 && matA->cols != 0 && matA->nnz != 0 &&
        n > 0 && matA->ptrType == ACL_SPARSE_INDEX_32I &&
        matA->IdxType == ACL_SPARSE_INDEX_32I;
}

bool HasComplexDegree64Capacity(int64_t rows, int64_t columns, int32_t n)
{
    const int64_t maxGroups = (rows + 1) / 2;
    const __int128 maxPlaneElements = static_cast<__int128>(maxGroups) *
        SPMM_CUBE_MIN_ROWS * SPMM_CUBE_MAX_K;
    return maxGroups > 0 && maxGroups <= std::numeric_limits<int32_t>::max() &&
        columns >= SPMM_CUBE_MAX_K && columns <= std::numeric_limits<int32_t>::max() &&
        maxPlaneElements <= std::numeric_limits<uint32_t>::max() &&
        maxPlaneElements * 6 <= std::numeric_limits<size_t>::max() &&
        static_cast<__int128>(columns) * n <= std::numeric_limits<size_t>::max() &&
        static_cast<__int128>(rows) * SPMM_COMPLEX64_SEGMENTED_MAX_SLOTS <=
            std::numeric_limits<size_t>::max();
}

aclsparseStatus_t LoadComplexDegree64Csr(
    const aclsparseSpMatDescr *matA, int64_t rows, std::vector<int32_t> *rowOff,
    std::vector<int32_t> *colInd)
{
    rowOff->resize(static_cast<size_t>(rows) + 1);
    colInd->resize(static_cast<size_t>(matA->nnz));
    if (aclrtMemcpy(rowOff->data(), rowOff->size() * sizeof(int32_t), matA->ptrs,
                    rowOff->size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE ||
        aclrtMemcpy(colInd->data(), colInd->size() * sizeof(int32_t), matA->idxs,
                    colInd->size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool IsComplexDegree64PlanUsable(const std::vector<ComplexDegree64Segment> &segments,
                                 const ComplexDegree64GroupPlan &plan,
                                 const std::vector<int32_t> &fallbackRows,
                                 int64_t rows)
{
    const int64_t maxGroups = (rows + 1) / 2;
    return segments.size() >= SPMM_CUBE_MIN_ROWS && !plan.groups.empty() &&
        plan.groups.size() <= static_cast<size_t>(maxGroups) &&
        plan.groups.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) &&
        fallbackRows.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max());
}

aclsparseStatus_t FinalizeComplexDegree64Cube(
    uint8_t *workspace, const WsOffsets &off, int64_t rows, int32_t indexBase,
    const std::vector<int32_t> &rowOff, const std::vector<uint8_t> &segmentedRows,
    const std::vector<int32_t> &fallbackRows, const ComplexDegree64PackedLayout &layout,
    int32_t groups, bool *eligible, int32_t *selectedRows,
    int32_t *selectedGroups, int32_t *selectedFallbackCount)
{
    if (!IsComplexDegree64MappingConsistent(
            rowOff, rows, indexBase, segmentedRows, layout.valueToPackedSlot)) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const aclsparseStatus_t uploadStatus = UploadComplexDegree64Workspace(
        workspace, off, rows, layout.meta, layout.slotToRow, fallbackRows,
        layout.rowSlots, layout.valueToPackedSlot, layout.packedA, layout.planeElements);
    if (uploadStatus != ACL_SPARSE_STATUS_SUCCESS) return uploadStatus;
    *eligible = true;
    *selectedRows = SPMM_COMPLEX64_SEGMENTED_CUBE_MODE;
    *selectedGroups = groups;
    *selectedFallbackCount = static_cast<int32_t>(fallbackRows.size());
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildComplexDegree64CubePrepackedDataCore(
    aclsparseSpMatDescr *matA, int32_t n, uint8_t *workspace, const WsOffsets &off,
    bool *eligible, int32_t *selectedRows, int32_t *selectedGroups,
    int32_t *selectedFallbackCount)
{
    const int64_t m = static_cast<int64_t>(matA->rows);
    const int64_t k = static_cast<int64_t>(matA->cols);
    const int32_t indexBase =
        matA->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    if (!HasComplexDegree64Capacity(m, k, n)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    try {
        std::vector<int32_t> rowOff;
        std::vector<int32_t> colInd;
        const aclsparseStatus_t copyStatus = LoadComplexDegree64Csr(matA, m, &rowOff, &colInd);
        if (copyStatus != ACL_SPARSE_STATUS_SUCCESS) return copyStatus;

        std::vector<ComplexDegree64Segment> segments;
        std::vector<int32_t> fallbackRows;
        std::vector<uint8_t> segmentedRows(static_cast<size_t>(m), 0);
        const aclsparseStatus_t segmentStatus = BuildComplexDegree64Segments(
            rowOff, colInd, m, k, indexBase, matA->nnz, &segments,
            &fallbackRows, &segmentedRows);
        if (segmentStatus != ACL_SPARSE_STATUS_SUCCESS) {
            return segmentStatus;
        }

        ComplexDegree64GroupPlan plan;
        BuildComplexDegree64GroupPlan(&segments, &plan);
        const std::vector<std::vector<size_t>> &cubeGroups = plan.groups;

        if (!IsComplexDegree64PlanUsable(segments, plan, fallbackRows, m)) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        ComplexDegree64PackedLayout layout;
        bool packed = false;
        const aclsparseStatus_t layoutStatus = BuildComplexDegree64PackedLayout(
            segments, cubeGroups, colInd, indexBase, m, k, matA->nnz, &layout, &packed);
        if (layoutStatus != ACL_SPARSE_STATUS_SUCCESS || !packed) return layoutStatus;

        return FinalizeComplexDegree64Cube(
            workspace, off, m, indexBase, rowOff, segmentedRows, fallbackRows, layout,
            static_cast<int32_t>(cubeGroups.size()), eligible, selectedRows,
            selectedGroups, selectedFallbackCount);
    } catch (const std::bad_alloc &) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
}

aclsparseStatus_t BuildComplexDegree64CubePrepackedData(
    aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB, int32_t n,
    uint8_t *workspace, const WsOffsets &off, bool candidate,
    bool *eligible, int32_t *selectedRows, int32_t *selectedGroups,
    int32_t *selectedFallbackCount)
{
    ResetCubeSelection(eligible, selectedRows, selectedGroups, selectedFallbackCount);
    if (workspace == nullptr || !IsComplexDegree64RequestSupported(matA, matB, n, candidate)) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    return BuildComplexDegree64CubePrepackedDataCore(
        matA, n, workspace, off, eligible, selectedRows, selectedGroups,
        selectedFallbackCount);
}

struct ComplexCubeRowLayout {
    std::vector<int32_t> rows;
    std::vector<int32_t> rowMin;
    std::vector<int32_t> rowMax;
    std::vector<uint8_t> eligible;
};

struct ComplexCubePackedLayout : ComplexCubeStorage {};

aclsparseStatus_t AnalyzeComplexCubeRows(
    const std::vector<int32_t> &rowOff, const std::vector<int32_t> &colInd,
    int64_t rows, int64_t columns, int32_t indexBase, uint64_t nnz,
    ComplexCubeRowLayout *layout)
{
    layout->rows.resize(static_cast<size_t>(rows));
    std::iota(layout->rows.begin(), layout->rows.end(), 0);
    layout->rowMin.assign(static_cast<size_t>(rows), static_cast<int32_t>(columns));
    layout->rowMax.assign(static_cast<size_t>(rows), -1);
    layout->eligible.assign(static_cast<size_t>(rows), 1);
    for (int64_t row = 0; row < rows; ++row) {
        const int32_t begin = rowOff[static_cast<size_t>(row)] - indexBase;
        const int32_t end = rowOff[static_cast<size_t>(row) + 1] - indexBase;
        if (!IsValidCsrRowRange(begin, end, nnz)) return ACL_SPARSE_STATUS_INVALID_VALUE;
        int32_t previous = -1;
        for (int32_t p = begin; p < end; ++p) {
            const int32_t col = colInd[static_cast<size_t>(p)] - indexBase;
            if (col < 0 || static_cast<int64_t>(col) >= columns) return ACL_SPARSE_STATUS_INVALID_VALUE;
            layout->eligible[static_cast<size_t>(row)] &= col > previous;
            previous = col;
            layout->rowMin[static_cast<size_t>(row)] = std::min(layout->rowMin[static_cast<size_t>(row)], col);
            layout->rowMax[static_cast<size_t>(row)] = std::max(layout->rowMax[static_cast<size_t>(row)], col);
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

void SortComplexCubeRows(ComplexCubeRowLayout *layout)
{
    std::stable_sort(layout->rows.begin(), layout->rows.end(), [&](int32_t lhs, int32_t rhs) {
        const int32_t lhsKey = layout->rowMax[static_cast<size_t>(lhs)] < 0 ?
            std::numeric_limits<int32_t>::max() : layout->rowMin[static_cast<size_t>(lhs)];
        const int32_t rhsKey = layout->rowMax[static_cast<size_t>(rhs)] < 0 ?
            std::numeric_limits<int32_t>::max() : layout->rowMin[static_cast<size_t>(rhs)];
        return lhsKey < rhsKey;
    });
}

void BuildComplexCubeGroups(const ComplexCubeRowLayout &layout,
                            std::vector<std::vector<int32_t>> *groups,
                            std::vector<int32_t> *fallbackRows)
{
    int32_t groupMin = 0;
    int32_t groupMax = -1;
    for (const int32_t row : layout.rows) {
        const int32_t minCol = layout.rowMin[static_cast<size_t>(row)];
        const int32_t maxCol = layout.rowMax[static_cast<size_t>(row)];
        if (!layout.eligible[static_cast<size_t>(row)] || maxCol < minCol ||
            maxCol - minCol >= SPMM_CUBE_MAX_K) {
            fallbackRows->push_back(row);
            continue;
        }
        const bool startGroup = groups->empty() ||
            groups->back().size() == SPMM_CUBE_MIN_ROWS ||
            std::max(groupMax, maxCol) - std::min(groupMin, minCol) >= SPMM_CUBE_MAX_K;
        if (startGroup) {
            groups->emplace_back();
            groupMin = minCol;
            groupMax = maxCol;
        } else {
            groupMin = std::min(groupMin, minCol);
            groupMax = std::max(groupMax, maxCol);
        }
        groups->back().push_back(row);
    }
}

aclsparseStatus_t FillComplexCubeGroup(
    const std::vector<int32_t> &groupRows, const ComplexCubeRowLayout &rows,
    const std::vector<int32_t> &rowOff, const std::vector<int32_t> &colInd,
    int32_t indexBase, int64_t columns, int64_t firstSlot,
    ComplexCubePackedLayout *layout, bool *packed)
{
    int32_t minCol = static_cast<int32_t>(columns);
    int32_t maxCol = -1;
    for (size_t local = 0; local < groupRows.size(); ++local) {
        const int32_t row = groupRows[local];
        const int64_t slot = firstSlot + static_cast<int64_t>(local);
        layout->slotToRow[static_cast<size_t>(slot)] = row;
        if (rows.rowMax[static_cast<size_t>(row)] >= 0) {
            minCol = std::min(minCol, rows.rowMin[static_cast<size_t>(row)]);
            maxCol = std::max(maxCol, rows.rowMax[static_cast<size_t>(row)]);
        }
    }
    const int32_t colBase = std::min(minCol, static_cast<int32_t>(columns - SPMM_CUBE_MAX_K));
    if (maxCol < minCol || minCol < colBase || maxCol - colBase >= SPMM_CUBE_MAX_K) {
        *packed = false;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const int32_t requiredK = maxCol - colBase + 1;
    layout->meta.emplace_back(SpmmCubeGroupMeta{colBase, requiredK <= 32 ? 32 : SPMM_CUBE_MAX_K});
    for (size_t local = 0; local < groupRows.size(); ++local) {
        const int32_t row = groupRows[local];
        const int32_t begin = rowOff[static_cast<size_t>(row)] - indexBase;
        const int32_t end = rowOff[static_cast<size_t>(row) + 1] - indexBase;
        const int64_t slot = firstSlot + static_cast<int64_t>(local);
        for (int32_t p = begin; p < end; ++p) {
            if (!MapComplexPackedValue(colInd, indexBase, colBase, p, slot,
                                       &layout->valueToPackedSlot)) {
                *packed = false;
                return ACL_SPARSE_STATUS_SUCCESS;
            }
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildComplexCubePackedLayout(
    const std::vector<std::vector<int32_t>> &groups, const ComplexCubeRowLayout &rows,
    const std::vector<int32_t> &rowOff, const std::vector<int32_t> &colInd,
    int32_t indexBase, int64_t columns, uint64_t nnz,
    ComplexCubePackedLayout *layout, bool *packed)
{
    *packed = false;
    const aclsparseStatus_t storageStatus = InitializeComplexCubeStorage(
        groups.size(), nnz, layout);
    if (storageStatus != ACL_SPARSE_STATUS_SUCCESS) return storageStatus;
    for (size_t group = 0; group < groups.size(); ++group) {
        bool groupPacked = true;
        const aclsparseStatus_t status = FillComplexCubeGroup(
            groups[group], rows, rowOff, colInd, indexBase, columns,
            static_cast<int64_t>(group) * SPMM_CUBE_MIN_ROWS, layout, &groupPacked);
        if (status != ACL_SPARSE_STATUS_SUCCESS || !groupPacked) return status;
    }
    *packed = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t UploadComplexCubeWorkspace(
    uint8_t *workspace, const WsOffsets &off, const ComplexCubePackedLayout &layout,
    const std::vector<int32_t> &fallbackRows)
{
    if (aclrtMemcpy(workspace + off.cubeMetaOff, layout.meta.size() * sizeof(layout.meta[0]), layout.meta.data(), layout.meta.size() * sizeof(layout.meta[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeRowIdsOff, layout.slotToRow.size() * sizeof(layout.slotToRow[0]), layout.slotToRow.data(), layout.slotToRow.size() * sizeof(layout.slotToRow[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(workspace + off.cubeAValueMapOff, layout.valueToPackedSlot.size() * sizeof(layout.valueToPackedSlot[0]), layout.valueToPackedSlot.data(), layout.valueToPackedSlot.size() * sizeof(layout.valueToPackedSlot[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        (!fallbackRows.empty() && aclrtMemcpy(workspace + off.cubeSlot1Off, fallbackRows.size() * sizeof(fallbackRows[0]), fallbackRows.data(), fallbackRows.size() * sizeof(fallbackRows[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE)) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return UploadComplexCubePlanes(workspace, off, layout);
}

bool GetComplexCubeCapacity(int64_t rows, int64_t columns, int32_t n,
                            uint64_t nnz, int64_t *maxGroups)
{
    *maxGroups = IsOfficialComplexP03Shape(rows, columns, n, nnz) ?
        (rows + SPMM_CUBE_MIN_ROWS - 1) / SPMM_CUBE_MIN_ROWS :
        (rows + spmm_host_internal::kComplexCubeReserveRowsPerGroup - 1) /
        spmm_host_internal::kComplexCubeReserveRowsPerGroup;
    const __int128 maxPlaneElements = static_cast<__int128>(*maxGroups) *
        SPMM_CUBE_MIN_ROWS * SPMM_CUBE_MAX_K;
    return *maxGroups > 0 && *maxGroups <= std::numeric_limits<int32_t>::max() &&
        columns >= SPMM_CUBE_MAX_K && columns <= std::numeric_limits<int32_t>::max() &&
        maxPlaneElements <= std::numeric_limits<uint32_t>::max() &&
        maxPlaneElements * 6 <= std::numeric_limits<size_t>::max() &&
        static_cast<__int128>(columns) * n <= std::numeric_limits<size_t>::max();
}

bool IsComplexCubePlanUsable(int64_t rows, int64_t maxGroups,
                             const std::vector<std::vector<int32_t>> &groups,
                             const std::vector<int32_t> &fallbackRows)
{
    const int64_t cubeRows = rows - static_cast<int64_t>(fallbackRows.size());
    return cubeRows >= SPMM_CUBE_MIN_ROWS && !groups.empty() &&
        groups.size() <= static_cast<size_t>(maxGroups) &&
        groups.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) &&
        fallbackRows.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max());
}

// The Complex64 Cube path keeps a decomposed copy of both sparse A and dense
// B in the caller-owned preprocess workspace.  It is intentionally stricter
// than the generic SIMT path: any input that cannot be represented by the
// FP16 high/low model simply leaves `eligible` false and is computed by SIMT.
// Structural CSR errors still return an API error, consistent with the other
// preprocess builders below.
aclsparseStatus_t BuildComplexCubePrepackedDataCore(
    aclsparseSpMatDescr *matA, int32_t n, uint8_t *workspace,
    const WsOffsets &off,
    bool *eligible, int32_t *selectedRows, int32_t *selectedGroups,
    int32_t *selectedFallbackCount)
{
    const int64_t m = static_cast<int64_t>(matA->rows);
    const int64_t k = static_cast<int64_t>(matA->cols);
    const int32_t indexBase = matA->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    int64_t maxGroups = 0;
    if (!GetComplexCubeCapacity(m, k, n, matA->nnz, &maxGroups)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    try {
        std::vector<int32_t> rowOff;
        std::vector<int32_t> colInd;
        const aclsparseStatus_t copyStatus = LoadComplexDegree64Csr(matA, m, &rowOff, &colInd);
        if (copyStatus != ACL_SPARSE_STATUS_SUCCESS) return copyStatus;
        ComplexCubeRowLayout rowLayout;
        const aclsparseStatus_t rowStatus = AnalyzeComplexCubeRows(
            rowOff, colInd, m, k, indexBase, matA->nnz, &rowLayout);
        if (rowStatus != ACL_SPARSE_STATUS_SUCCESS) return rowStatus;
        SortComplexCubeRows(&rowLayout);
        std::vector<std::vector<int32_t>> cubeGroups;
        std::vector<int32_t> fallbackRows;
        BuildComplexCubeGroups(rowLayout, &cubeGroups, &fallbackRows);

        if (!IsComplexCubePlanUsable(m, maxGroups, cubeGroups, fallbackRows)) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        ComplexCubePackedLayout packedLayout;
        bool packed = false;
        const aclsparseStatus_t packedStatus = BuildComplexCubePackedLayout(
            cubeGroups, rowLayout, rowOff, colInd, indexBase, k, matA->nnz,
            &packedLayout, &packed);
        if (packedStatus != ACL_SPARSE_STATUS_SUCCESS || !packed) return packedStatus;
        const aclsparseStatus_t uploadStatus = UploadComplexCubeWorkspace(
            workspace, off, packedLayout, fallbackRows);
        if (uploadStatus != ACL_SPARSE_STATUS_SUCCESS) return uploadStatus;

        *eligible = true;
        *selectedRows = SPMM_CUBE_MIN_ROWS;
        *selectedGroups = static_cast<int32_t>(cubeGroups.size());
        *selectedFallbackCount = static_cast<int32_t>(fallbackRows.size());
        return ACL_SPARSE_STATUS_SUCCESS;
    } catch (const std::bad_alloc &) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
}

aclsparseStatus_t BuildComplexCubePrepackedData(
    aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB, int32_t n,
    uint8_t *workspace, const WsOffsets &off, bool candidate,
    bool *eligible, int32_t *selectedRows, int32_t *selectedGroups,
    int32_t *selectedFallbackCount)
{
    if (matA != nullptr && matA->spmmUniformDegree == SPMM_FP32_DEGREE64_DEGREE) {
        return BuildComplexDegree64CubePrepackedData(
            matA, matB, n, workspace, off, candidate, eligible,
            selectedRows, selectedGroups, selectedFallbackCount);
    }
    ResetCubeSelection(eligible, selectedRows, selectedGroups, selectedFallbackCount);
    if (!candidate || matA == nullptr || matB == nullptr || workspace == nullptr ||
        matA->rows == 0 || matA->cols == 0 || matA->nnz == 0 || n <= 0 ||
        matA->ptrType != ACL_SPARSE_INDEX_32I || matA->IdxType != ACL_SPARSE_INDEX_32I) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    return BuildComplexCubePrepackedDataCore(
        matA, n, workspace, off, eligible, selectedRows, selectedGroups,
        selectedFallbackCount);
}

} // namespace spmm_host_builders
