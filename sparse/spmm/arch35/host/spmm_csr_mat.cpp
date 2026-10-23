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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <functional>
#include <queue>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "spmm_host.h"
#include "spmm_csr_mat.h"
#include "spmm_host_internal.h"
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"

namespace {

void BuildIdentityRowReorder(int64_t m,
    int32_t blockDim,
    std::vector<int32_t> *reorder,
    std::vector<int32_t> *binEdges);

// Greedy bin packing: sort rows by nnz descending, then place each row into the
// bin with the smallest accumulated load. This balances workload across cores.
void GreedyRowBinPack(const std::vector<int32_t> &rowNnz,
                      int32_t binNum,
                      std::vector<int32_t> *reorder,
                      std::vector<int32_t> *binEdges,
                      bool *identityReorder,
                      int32_t *uniformDegree)
{
    const int32_t m = static_cast<int32_t>(rowNnz.size());

    // Equal-length rows are already perfectly balanced. Keep them contiguous
    // instead of sorting and assigning them round-robin across cores. This is
    // the common path for the official performance cases and preserves CSR,
    // dense-output and reorder-table locality.
    const auto [minIt, maxIt] = std::minmax_element(rowNnz.begin(), rowNnz.end());
    *uniformDegree = m == 0 ? 0 :
        ((*minIt == *maxIt) ? *minIt : -1);
    const bool nearUniform = m == 0 || (maxIt != rowNnz.end() &&
                                        *maxIt - *minIt <= 1);
    if (nearUniform) {
        BuildIdentityRowReorder(m, binNum, reorder, binEdges);
        *identityReorder = true;
        return;
    }
    *identityReorder = false;

    std::vector<int32_t> order(m);
    for (int32_t i = 0; i < m; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        if (rowNnz[a] != rowNnz[b]) {
            return rowNnz[a] > rowNnz[b];
        }
        return a < b;
    });

    std::vector<std::vector<int32_t>> bins(binNum);
    using BinLoad = std::pair<int64_t, int32_t>;
    std::priority_queue<BinLoad, std::vector<BinLoad>, std::greater<BinLoad>> leastLoaded;
    for (int32_t b = 0; b < binNum; ++b) {
        leastLoaded.emplace(0, b);
    }
    for (int32_t idx : order) {
        const BinLoad current = leastLoaded.top();
        leastLoaded.pop();
        const int32_t pick = current.second;
        bins[pick].push_back(idx);
        // Every row also initializes and writes N output elements. Giving each
        // row a unit fixed cost prevents zero-nnz rows from accumulating in a
        // single bin while preserving nnz-dominated balancing for long rows.
        leastLoaded.emplace(current.first + static_cast<int64_t>(rowNnz[idx]) + 1, pick);
    }

    reorder->resize(m);
    binEdges->resize(static_cast<size_t>(binNum) + 1);
    int32_t cursor = 0;
    (*binEdges)[0] = 0;
    for (int32_t b = 0; b < binNum; ++b) {
        for (int32_t r : bins[b]) {
            (*reorder)[cursor++] = r;
        }
        (*binEdges)[b + 1] = cursor;
    }
}

void BuildIdentityRowReorder(int64_t m,
    int32_t blockDim,
    std::vector<int32_t> *reorder,
    std::vector<int32_t> *binEdges)
{
    reorder->resize(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) {
        (*reorder)[i] = static_cast<int32_t>(i);
    }
    binEdges->resize(static_cast<size_t>(blockDim) + 1);
    const int64_t rowsPerBin = (m + blockDim - 1) / blockDim;
    for (int32_t b = 0; b <= blockDim; ++b) {
        const int64_t edge = static_cast<int64_t>(b) * rowsPerBin;
        (*binEdges)[b] = static_cast<int32_t>(edge < m ? edge : m);
    }
}

uint32_t BuildGreedyRowReorderFromCsr(const aclsparseSpMatDescr *matDesc,
    int64_t m,
    int32_t blockDim,
    std::vector<int32_t> *reorder,
    std::vector<int32_t> *binEdges,
    bool *identityReorder,
    int32_t *uniformDegree)
{
    if (matDesc->ptrType != ACL_SPARSE_INDEX_32I) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    std::vector<int32_t> rowOff(static_cast<size_t>(m) + 1);
    aclError aclRet = aclrtMemcpy(rowOff.data(),
                                  sizeof(int32_t) * (m + 1),
                                  matDesc->ptrs,
                                  sizeof(int32_t) * (m + 1),
                                  ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    std::vector<int32_t> rowNnz(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) {
        rowNnz[i] = rowOff[i + 1] - rowOff[i];
    }
    GreedyRowBinPack(rowNnz, blockDim, reorder, binEdges, identityReorder,
                     uniformDegree);
    return ACL_SPARSE_STATUS_SUCCESS;
}

uint32_t WriteReorderToWorkspace(uint8_t *dWorkspaceBase,
    int64_t reorderOffset,
    int64_t binEdgeOffset,
    int64_t m,
    int32_t blockDim,
    const std::vector<int32_t> &reorder,
    const std::vector<int32_t> &binEdges)
{
    aclError aclRet = ACL_ERROR_NONE;
    if (m > 0) {
        aclRet = aclrtMemcpy(dWorkspaceBase + reorderOffset,
                             sizeof(int32_t) * m,
                             reorder.data(),
                             sizeof(int32_t) * m,
                             ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    aclRet = aclrtMemcpy(dWorkspaceBase + binEdgeOffset,
                         sizeof(int32_t) * (blockDim + 1),
                         binEdges.data(),
                         sizeof(int32_t) * (blockDim + 1),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

uint32_t SpmmCsrMat::DoPreProcess(uint8_t *dWorkspaceBase,
                                  int64_t reorderOffset,
                                  int64_t binEdgeOffset,
                                  bool identity)
{
    if (dWorkspaceBase == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (m < 0 || blockDim <= 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    std::vector<int32_t> reorder;
    std::vector<int32_t> binEdges;

    if (identity || matDesc == nullptr || matDesc->ptrs == nullptr) {
        BuildIdentityRowReorder(m, blockDim, &reorder, &binEdges);
        identityReorder = true;
        uniformDegree = nnz == 0 ? 0 : -1;
    } else {
        uint32_t st = BuildGreedyRowReorderFromCsr(
            matDesc, m, blockDim, &reorder, &binEdges, &identityReorder,
            &uniformDegree);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }

    return WriteReorderToWorkspace(dWorkspaceBase, reorderOffset, binEdgeOffset,
                                   m, blockDim, reorder, binEdges);
}

namespace {

aclsparseStatus_t LoadTransposeCsr(const aclsparseSpMatDescr *matA,
                                  spmm_host_internal::CsrHostData *data)
{
    data->indexBase = matA->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    data->rowOffsets.resize(static_cast<size_t>(matA->rows) + 1);
    data->columnIndices.resize(static_cast<size_t>(matA->nnz));
    const size_t rowBytes = data->rowOffsets.size() * sizeof(int32_t);
    const size_t columnBytes = data->columnIndices.size() * sizeof(int32_t);
    if (aclrtMemcpy(data->rowOffsets.data(), rowBytes, matA->ptrs, rowBytes,
        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE ||
        (columnBytes != 0 && aclrtMemcpy(data->columnIndices.data(), columnBytes,
            matA->idxs, columnBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE)) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateTransposeCsr(const aclsparseSpMatDescr *matA,
                                      const spmm_host_internal::CsrHostData &data)
{
    if (data.rowOffsets.front() != data.indexBase ||
        static_cast<int64_t>(data.rowOffsets.back()) - data.indexBase !=
            static_cast<int64_t>(matA->nnz)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    for (size_t row = 1; row < data.rowOffsets.size(); ++row) {
        const int64_t begin = static_cast<int64_t>(data.rowOffsets[row - 1]) - data.indexBase;
        const int64_t end = static_cast<int64_t>(data.rowOffsets[row]) - data.indexBase;
        if (begin < 0 || end < begin || static_cast<uint64_t>(end) > matA->nnz) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    for (int32_t column : data.columnIndices) {
        const int64_t index = static_cast<int64_t>(column) - data.indexBase;
        if (index < 0 || static_cast<uint64_t>(index) >= matA->cols) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

void BuildTransposeCsr(const spmm_host_internal::CsrHostData &source,
                       std::vector<int32_t> *offsets,
                       std::vector<int32_t> *columns,
                       std::vector<int32_t> *permutation)
{
    for (int32_t column : source.columnIndices) {
        ++(*offsets)[static_cast<size_t>(column - source.indexBase) + 1];
    }
    for (size_t column = 1; column < offsets->size(); ++column) {
        (*offsets)[column] += (*offsets)[column - 1];
    }
    std::vector<int32_t> cursor(offsets->begin(), offsets->end() - 1);
    for (size_t row = 0; row + 1 < source.rowOffsets.size(); ++row) {
        const int32_t end = source.rowOffsets[row + 1] - source.indexBase;
        for (int32_t index = source.rowOffsets[row] - source.indexBase; index < end; ++index) {
            const int32_t column = source.columnIndices[index] - source.indexBase;
            const int32_t destination = cursor[column]++;
            (*columns)[destination] = static_cast<int32_t>(row);
            (*permutation)[destination] = index;
        }
    }
}

aclsparseStatus_t CopyTransposeVector(uint8_t *workspace, int64_t offset,
                                     const std::vector<int32_t> &values)
{
    if (values.empty()) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const size_t bytes = values.size() * sizeof(int32_t);
    return aclrtMemcpy(workspace + offset, bytes, values.data(), bytes,
        ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE ?
        ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_EXECUTION_FAILED;
}

} // namespace

aclsparseStatus_t SpmmCsrMat::DoTransposePreProcess(
    uint8_t *workspace, const spmm_host_internal::WsOffsets &offsets)
{
    spmm_host_internal::CsrHostData source;
    aclsparseStatus_t status = LoadTransposeCsr(matDesc, &source);
    if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
    status = ValidateTransposeCsr(matDesc, source);
    if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
    std::vector<int32_t> rowOffsets(static_cast<size_t>(k) + 1, 0);
    std::vector<int32_t> columns(static_cast<size_t>(nnz));
    std::vector<int32_t> permutation(static_cast<size_t>(nnz));
    BuildTransposeCsr(source, &rowOffsets, &columns, &permutation);
    status = CopyTransposeVector(workspace, offsets.transposeRowsOff, rowOffsets);
    if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
    status = CopyTransposeVector(workspace, offsets.transposeColumnsOff, columns);
    if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
    status = CopyTransposeVector(workspace, offsets.transposePermutationOff, permutation);
    if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
    std::vector<int32_t> reorder;
    std::vector<int32_t> edges;
    BuildIdentityRowReorder(k, blockDim, &reorder, &edges);
    return static_cast<aclsparseStatus_t>(WriteReorderToWorkspace(workspace,
        offsets.reorderOff, offsets.binEdgeOff, k, blockDim, reorder, edges));
}
