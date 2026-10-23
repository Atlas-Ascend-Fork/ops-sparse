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

#include "spmm_host_internal.h"

#include "acl/acl.h"

namespace spmm_host_internal {

namespace {

constexpr int64_t kOfficialComplexP03M = 2449029;
constexpr int64_t kOfficialComplexP03K = 2449029;
constexpr int64_t kOfficialComplexP03N = 256;
constexpr uint64_t kOfficialComplexP03Nnz = 61859140;

} // namespace

bool IsOfficialComplexP03Shape(int64_t m, int64_t k, int64_t n,
                               uint64_t nnz)
{
    return m == kOfficialComplexP03M && k == kOfficialComplexP03K &&
        n == kOfficialComplexP03N && nnz == kOfficialComplexP03Nnz;
}

bool IsComplexRegBaseDegree(int32_t degree)
{
    return degree == SPMM_COMPLEX64_DEGREE2_DEGREE ||
        degree == SPMM_COMPLEX64_DEGREE4_DEGREE ||
        degree == SPMM_COMPLEX64_DEGREE8_DEGREE ||
        degree == SPMM_COMPLEX64_DEGREE64_DEGREE;
}

int32_t ComplexRegBaseRows(int32_t degree)
{
    if (degree == SPMM_COMPLEX64_DEGREE64_DEGREE) {
        return SPMM_COMPLEX64_DEGREE64_ROWS;
    }
    return degree == SPMM_COMPLEX64_DEGREE8_DEGREE ?
        SPMM_COMPLEX64_DEGREE8_ROWS : SPMM_COMPLEX64_DEGREE2_ROWS;
}

int32_t ComplexRegBaseWindowRows(int32_t degree)
{
    if (degree == SPMM_COMPLEX64_DEGREE64_DEGREE) {
        return SPMM_COMPLEX64_DEGREE64_WINDOW_ROWS;
    }
    return degree == SPMM_COMPLEX64_DEGREE8_DEGREE ?
        SPMM_COMPLEX64_DEGREE8_WINDOW_ROWS :
        SPMM_COMPLEX64_DEGREE2_WINDOW_ROWS;
}

int32_t ComplexRegBaseMinAverageRows(int32_t degree)
{
    if (degree == SPMM_COMPLEX64_DEGREE64_DEGREE) {
        return SPMM_COMPLEX64_DEGREE64_MIN_AVERAGE_ROWS;
    }
    return degree == SPMM_COMPLEX64_DEGREE2_DEGREE ?
        SPMM_COMPLEX64_DEGREE2_MIN_AVERAGE_ROWS :
        SPMM_COMPLEX64_REGBASE_MIN_AVERAGE_ROWS;
}

aclsparseStatus_t LoadCsrHostData(const aclsparseSpMatDescr *matA,
                                  CsrHostData *data)
{
    if (matA == nullptr || data == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const size_t rowCount = static_cast<size_t>(matA->rows) + 1;
    const size_t nnz = static_cast<size_t>(matA->nnz);
    data->rowOffsets.resize(rowCount);
    data->columnIndices.resize(nnz);
    if (aclrtMemcpy(data->rowOffsets.data(), rowCount * sizeof(int32_t),
                    matA->ptrs, rowCount * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE ||
        aclrtMemcpy(data->columnIndices.data(), nnz * sizeof(int32_t),
                    matA->idxs, nnz * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    data->indexBase = matA->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    if (data->rowOffsets.front() != data->indexBase ||
        static_cast<uint64_t>(data->rowOffsets.back() - data->indexBase) != matA->nnz) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool WorkspaceRegionFits(int64_t begin, int64_t end, size_t bytes)
{
    return begin >= 0 && end >= begin &&
        static_cast<uint64_t>(end - begin) >= bytes;
}

bool RegBaseWorkspaceFits(const WsOffsets &offsets, size_t batchBytes,
                          size_t rowIdsBytes, size_t slotBytes,
                          size_t packedBytes, size_t controlBytes)
{
    return WorkspaceRegionFits(offsets.cubeMetaOff, offsets.cubeRowIdsOff,
                               batchBytes) &&
        WorkspaceRegionFits(offsets.cubeRowIdsOff, offsets.cubeSlot1Off,
                            rowIdsBytes) &&
        WorkspaceRegionFits(offsets.cubeSlot1Off, offsets.cubePackedAOff,
                            slotBytes) &&
        WorkspaceRegionFits(offsets.cubePackedAOff, offsets.cubeScratchOff,
                            packedBytes) &&
        WorkspaceRegionFits(offsets.cubeScratchOff, offsets.cubeTilingOff,
                            controlBytes);
}

bool IsValidCsrRowRange(int32_t begin, int32_t end, uint64_t nnz)
{
    return begin >= 0 && end >= begin && static_cast<uint64_t>(end) <= nnz;
}

void ResetCubeSelection(bool *eligible, int32_t *selectedRows,
                        int32_t *selectedGroups,
                        int32_t *selectedFallbackCount)
{
    *eligible = false;
    *selectedRows = 0;
    *selectedGroups = 0;
    if (selectedFallbackCount != nullptr) {
        *selectedFallbackCount = 0;
    }
}

} // namespace spmm_host_internal
