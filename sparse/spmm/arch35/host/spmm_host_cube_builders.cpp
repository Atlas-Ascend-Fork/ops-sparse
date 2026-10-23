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

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "spmm_host_builders.h"

namespace spmm_host_builders {

using spmm_host_internal::IsValidCsrRowRange;
using spmm_host_internal::ResetCubeSelection;
using spmm_host_internal::WsOffsets;

struct CubePrepackCsr {
    std::vector<int32_t> rowOff;
    std::vector<int32_t> colInd;
    std::vector<uint16_t> values;
};

struct CubePrepackRows {
    std::vector<int32_t> rows;
    std::vector<int32_t> fallbackRows;
    std::vector<int32_t> rowMin;
    std::vector<int32_t> rowMax;
};

struct CubePrepackPlan {
    int32_t rowsPerGroup = 0;
    std::vector<std::vector<int32_t>> groups;
    std::vector<int32_t> fallbackRows;
    uint64_t mappedNnz = 0;
};

aclsparseStatus_t LoadCubePrepackCsr(const aclsparseSpMatDescr *matA,
                                     int64_t rows, CubePrepackCsr *csr)
{
    csr->rowOff.resize(static_cast<size_t>(rows) + 1);
    csr->colInd.resize(static_cast<size_t>(matA->nnz));
    csr->values.resize(static_cast<size_t>(matA->nnz));
    if (aclrtMemcpy(csr->rowOff.data(), csr->rowOff.size() * sizeof(int32_t), matA->ptrs,
                    csr->rowOff.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE ||
        (matA->nnz != 0 && aclrtMemcpy(csr->colInd.data(), csr->colInd.size() * sizeof(int32_t), matA->idxs,
                                       csr->colInd.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) ||
        (matA->nnz != 0 && aclrtMemcpy(csr->values.data(), csr->values.size() * sizeof(uint16_t), matA->values,
                                       csr->values.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE)) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t AnalyzeCubePrepackRows(
    const aclsparseSpMatDescr *matA, const CubePrepackCsr &csr, int32_t indexBase,
    CubePrepackRows *layout)
{
    const int64_t rows = static_cast<int64_t>(matA->rows);
    layout->rows.reserve(static_cast<size_t>(rows));
    layout->fallbackRows.reserve(static_cast<size_t>(rows));
    layout->rowMin.assign(static_cast<size_t>(rows), 0);
    layout->rowMax.assign(static_cast<size_t>(rows), -1);
    for (int64_t row = 0; row < rows; ++row) {
        const int32_t begin = csr.rowOff[static_cast<size_t>(row)] - indexBase;
        const int32_t end = csr.rowOff[static_cast<size_t>(row) + 1] - indexBase;
        if (!IsValidCsrRowRange(begin, end, matA->nnz)) return ACL_SPARSE_STATUS_INVALID_VALUE;
        int32_t previous = -1;
        bool eligible = end > begin;
        for (int32_t p = begin; p < end; ++p) {
            const int32_t col = csr.colInd[static_cast<size_t>(p)] - indexBase;
            if (col < 0 || static_cast<uint64_t>(col) >= matA->cols) return ACL_SPARSE_STATUS_INVALID_VALUE;
            eligible &= col > previous;
            previous = col;
            if (layout->rowMax[static_cast<size_t>(row)] < 0) layout->rowMin[static_cast<size_t>(row)] = col;
            layout->rowMax[static_cast<size_t>(row)] = col;
        }
        eligible &= layout->rowMax[static_cast<size_t>(row)] >= layout->rowMin[static_cast<size_t>(row)] &&
            layout->rowMax[static_cast<size_t>(row)] - layout->rowMin[static_cast<size_t>(row)] < SPMM_CUBE_MAX_K;
        (eligible ? layout->rows : layout->fallbackRows).push_back(static_cast<int32_t>(row));
    }
    std::stable_sort(layout->rows.begin(), layout->rows.end(), [&](int32_t lhs, int32_t rhs) {
        const int32_t lhsKey = layout->rowMax[static_cast<size_t>(lhs)] < 0 ? std::numeric_limits<int32_t>::max() : layout->rowMin[static_cast<size_t>(lhs)];
        const int32_t rhsKey = layout->rowMax[static_cast<size_t>(rhs)] < 0 ? std::numeric_limits<int32_t>::max() : layout->rowMin[static_cast<size_t>(rhs)];
        return lhsKey < rhsKey;
    });
    return ACL_SPARSE_STATUS_SUCCESS;
}

CubePrepackPlan BuildCubePrepackPlan(const CubePrepackRows &rows,
                                     const CubePrepackCsr &csr,
                                     int64_t columns, int32_t rowsPerGroup)
{
    CubePrepackPlan plan;
    plan.rowsPerGroup = rowsPerGroup;
    plan.fallbackRows = rows.fallbackRows;
    size_t cursor = 0;
    while (cursor < rows.rows.size()) {
        const int32_t count = static_cast<int32_t>(std::min(
            rows.rows.size() - cursor, static_cast<size_t>(rowsPerGroup)));
        int32_t minCol = std::numeric_limits<int32_t>::max();
        int32_t maxCol = -1;
        for (int32_t local = 0; local < count; ++local) {
            const int32_t row = rows.rows[cursor + static_cast<size_t>(local)];
            minCol = std::min(minCol, rows.rowMin[static_cast<size_t>(row)]);
            maxCol = std::max(maxCol, rows.rowMax[static_cast<size_t>(row)]);
        }
        SpmmCubeGroupMeta window{};
        if (!spmm_host_internal::BuildCubePrepackWindow(columns, minCol, maxCol, &window)) {
            plan.fallbackRows.push_back(rows.rows[cursor]);
            ++cursor;
            continue;
        }
        plan.groups.emplace_back(rows.rows.begin() + cursor,
                                 rows.rows.begin() + cursor + count);
        for (int32_t local = 0; local < count; ++local) {
            const int32_t row = rows.rows[cursor + static_cast<size_t>(local)];
            plan.mappedNnz += static_cast<uint64_t>(
                csr.rowOff[static_cast<size_t>(row) + 1] - csr.rowOff[static_cast<size_t>(row)]);
        }
        cursor += static_cast<size_t>(count);
    }
    return plan;
}

bool HasCubePrepackCoverage(const CubePrepackPlan &plan, uint64_t nnz)
{
    return !plan.groups.empty() &&
        static_cast<__int128>(plan.mappedNnz) * 100 >= static_cast<__int128>(nnz) * 95;
}

bool SelectCubePrepackPlan(const CubePrepackRows &rows, const CubePrepackCsr &csr,
                           int64_t columns, uint64_t nnz, CubePrepackPlan *plan)
{
    CubePrepackPlan plan32 = BuildCubePrepackPlan(rows, csr, columns, SPMM_CUBE_ROWS);
    if (HasCubePrepackCoverage(plan32, nnz)) {
        *plan = std::move(plan32);
        return true;
    }
    CubePrepackPlan plan16 = BuildCubePrepackPlan(rows, csr, columns, SPMM_CUBE_MIN_ROWS);
    if (!HasCubePrepackCoverage(plan16, nnz)) return false;
    *plan = std::move(plan16);
    return true;
}

struct CubePrepackPackedLayout {
    std::vector<SpmmCubeGroupMeta> meta;
    std::vector<int32_t> rowIds;
    std::vector<uint16_t> packedA;
    std::vector<uint32_t> valueMap;
};

bool FillCubePrepackGroup(const CubePrepackPlan &plan, size_t group,
                          const CubePrepackRows &rows, const CubePrepackCsr &csr,
                          int64_t columns, int32_t indexBase, CubePrepackPackedLayout *layout)
{
    const int64_t firstSlot = static_cast<int64_t>(group) * plan.rowsPerGroup;
    const std::vector<int32_t> &groupRows = plan.groups[group];
    int32_t minCol = std::numeric_limits<int32_t>::max();
    int32_t maxCol = -1;
    for (size_t local = 0; local < groupRows.size(); ++local) {
        const int32_t row = groupRows[local];
        layout->rowIds[static_cast<size_t>(firstSlot) + local] = row;
        minCol = std::min(minCol, rows.rowMin[static_cast<size_t>(row)]);
        maxCol = std::max(maxCol, rows.rowMax[static_cast<size_t>(row)]);
    }
    SpmmCubeGroupMeta window{};
    if (!spmm_host_internal::BuildCubePrepackWindow(columns, minCol, maxCol, &window)) return false;
    layout->meta[group] = window;
    for (size_t local = 0; local < groupRows.size(); ++local) {
        const int32_t row = groupRows[local];
        const int64_t slot = firstSlot + static_cast<int64_t>(local);
        const int32_t begin = csr.rowOff[static_cast<size_t>(row)] - indexBase;
        const int32_t end = csr.rowOff[static_cast<size_t>(row) + 1] - indexBase;
        for (int32_t p = begin; p < end; ++p) {
            const int32_t localK = csr.colInd[static_cast<size_t>(p)] - indexBase - window.col_base;
            layout->packedA[static_cast<size_t>(slot) * SPMM_CUBE_MAX_K + localK] = csr.values[static_cast<size_t>(p)];
            layout->valueMap[static_cast<size_t>(p)] = static_cast<uint32_t>(
                static_cast<size_t>(slot) * SPMM_CUBE_MAX_K + localK);
        }
    }
    return true;
}

bool BuildCubePrepackPackedLayout(const CubePrepackPlan &plan,
                                  const CubePrepackRows &rows,
                                  const CubePrepackCsr &csr, int64_t columns, int32_t indexBase,
                                  CubePrepackPackedLayout *layout)
{
    const int64_t paddedRows = static_cast<int64_t>(plan.groups.size()) * plan.rowsPerGroup;
    if (paddedRows > UINT32_MAX / SPMM_CUBE_MAX_K) return false;
    layout->meta.resize(plan.groups.size());
    layout->rowIds.assign(static_cast<size_t>(paddedRows), -1);
    layout->packedA.assign(static_cast<size_t>(paddedRows) * SPMM_CUBE_MAX_K, 0);
    layout->valueMap.assign(csr.values.size(), UINT32_MAX);
    for (size_t group = 0; group < plan.groups.size(); ++group) {
        if (!FillCubePrepackGroup(plan, group, rows, csr, columns, indexBase, layout)) return false;
    }
    return true;
}

aclsparseStatus_t UploadCubePrepackLayout(uint8_t *workspace, const WsOffsets &off,
                                          const CubePrepackPlan &plan,
                                          const CubePrepackPackedLayout &layout)
{
    if (!spmm_host_internal::WorkspaceRegionFits(off.cubeAValueMapOff, off.cubePackedAOff,
                                                 layout.valueMap.size() * sizeof(uint32_t))) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    if ((!layout.meta.empty() && aclrtMemcpy(workspace + off.cubeMetaOff, layout.meta.size() * sizeof(layout.meta[0]), layout.meta.data(), layout.meta.size() * sizeof(layout.meta[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) ||
        (!layout.rowIds.empty() && aclrtMemcpy(workspace + off.cubeRowIdsOff, layout.rowIds.size() * sizeof(layout.rowIds[0]), layout.rowIds.data(), layout.rowIds.size() * sizeof(layout.rowIds[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) ||
        (!plan.fallbackRows.empty() && aclrtMemcpy(workspace + off.cubeSlot1Off, plan.fallbackRows.size() * sizeof(plan.fallbackRows[0]), plan.fallbackRows.data(), plan.fallbackRows.size() * sizeof(plan.fallbackRows[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) ||
        (!layout.packedA.empty() && aclrtMemcpy(workspace + off.cubePackedAOff, layout.packedA.size() * sizeof(layout.packedA[0]), layout.packedA.data(), layout.packedA.size() * sizeof(layout.packedA[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) ||
        (!layout.valueMap.empty() && aclrtMemcpy(workspace + off.cubeAValueMapOff, layout.valueMap.size() * sizeof(layout.valueMap[0]), layout.valueMap.data(), layout.valueMap.size() * sizeof(layout.valueMap[0]), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE)) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildCubePrepackedData(aclsparseSpMatDescr *matA, uint8_t *workspace,
                                         const WsOffsets &off, bool candidate,
                                         bool *eligible, int32_t *selectedRows,
                                         int32_t *selectedGroups,
                                         int32_t *selectedFallbackCount)
{
    ResetCubeSelection(eligible, selectedRows, selectedGroups,
                       selectedFallbackCount);
    if (!candidate || matA->rows == 0 || matA->ptrType != ACL_SPARSE_INDEX_32I ||
        matA->IdxType != ACL_SPARSE_INDEX_32I) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const int64_t m = static_cast<int64_t>(matA->rows);
    CubePrepackCsr csr;
    const aclsparseStatus_t copyStatus = LoadCubePrepackCsr(matA, m, &csr);
    if (copyStatus != ACL_SPARSE_STATUS_SUCCESS) return copyStatus;
    const int32_t indexBase = matA->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    CubePrepackRows rows;
    const aclsparseStatus_t rowStatus = AnalyzeCubePrepackRows(matA, csr, indexBase, &rows);
    if (rowStatus != ACL_SPARSE_STATUS_SUCCESS) return rowStatus;

    CubePrepackPlan plan;
    const int64_t columns = static_cast<int64_t>(matA->cols);
    if (!SelectCubePrepackPlan(rows, csr, columns, matA->nnz, &plan)) return ACL_SPARSE_STATUS_SUCCESS;
    const int32_t cubeRows = plan.rowsPerGroup;
    const int64_t groups = static_cast<int64_t>(plan.groups.size());
    CubePrepackPackedLayout packed;
    if (!BuildCubePrepackPackedLayout(plan, rows, csr, columns, indexBase, &packed)) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const aclsparseStatus_t uploadStatus = UploadCubePrepackLayout(workspace, off, plan, packed);
    if (uploadStatus != ACL_SPARSE_STATUS_SUCCESS) return uploadStatus;
    *eligible = true;
    *selectedRows = cubeRows;
    *selectedGroups = static_cast<int32_t>(groups);
    *selectedFallbackCount = static_cast<int32_t>(plan.fallbackRows.size());
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace spmm_host_builders
