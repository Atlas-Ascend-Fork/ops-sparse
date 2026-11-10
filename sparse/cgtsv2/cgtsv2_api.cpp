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

#include <limits>
#include "log/log.h"
#include "aclsparse_host_utils.h"
#include "cann_ops_sparse.h"
#include "cgtsv2_workspace.h"
#include "cgtsv2_backend.h"

namespace {
using cgtsv2::CheckedMultiply;

aclsparseStatus_t ValidateDataPointers(const aclsparseComplex_t* dl, const aclsparseComplex_t* d,
                                       const aclsparseComplex_t* du, const aclsparseComplex_t* B)
{
    if (dl == nullptr) {
        OP_LOGE("aclsparseCgtsv2", "dl is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (d == nullptr) {
        OP_LOGE("aclsparseCgtsv2", "d is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (du == nullptr) {
        OP_LOGE("aclsparseCgtsv2", "du is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (B == nullptr) {
        OP_LOGE("aclsparseCgtsv2", "B is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool IsAddressSpanRepresentable(const void* address, size_t bytes)
{
    if (bytes == 0 || bytes > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
        return false;
    }
    return reinterpret_cast<uintptr_t>(address) <= std::numeric_limits<uintptr_t>::max() - (bytes - 1);
}

bool AreDataSpansRepresentable(int m, int n, int ldb, const aclsparseComplex_t* dl, const aclsparseComplex_t* d,
                               const aclsparseComplex_t* du, const aclsparseComplex_t* B)
{
    size_t coefficientBytes = 0;
    size_t bElements = 0;
    size_t bBytes = 0;
    if (!CheckedMultiply(static_cast<size_t>(m), sizeof(aclsparseComplex_t), coefficientBytes) ||
        !CheckedMultiply(static_cast<size_t>(ldb), static_cast<size_t>(n), bElements) ||
        !CheckedMultiply(bElements, sizeof(aclsparseComplex_t), bBytes)) {
        return false;
    }
    return IsAddressSpanRepresentable(dl, coefficientBytes) && IsAddressSpanRepresentable(d, coefficientBytes) &&
           IsAddressSpanRepresentable(du, coefficientBytes) && IsAddressSpanRepresentable(B, bBytes);
}

aclsparseStatus_t Validate(aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, const aclsparseComplex_t *b, int ldb, size_t &bytes)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseCgtsv2", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    bytes = 0;
    if (m < 3 || n < 0 || ldb < m) {
        OP_LOGE("aclsparseCgtsv2", "invalid dimensions: m=%d, n=%d, ldb=%d", m, n, ldb);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Preserve the existing handle-signature check, including empty calls.
    // A null stream is valid; the accessor neither creates nor synchronizes it.
    aclrtStream stream = nullptr;
    const auto status = aclsparseGetStream(handle, &stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseCgtsv2", "handle stream validation failed: status=%d", static_cast<int>(status));
        return status;
    }
    if (n == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (!Cgtsv2WorkspaceBytes(m, n, ldb, bytes)) {
        OP_LOGE("aclsparseCgtsv2", "workspace size is not representable");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (ValidateDataPointers(dl, d, du, b) != ACL_SPARSE_STATUS_SUCCESS ||
        !AreDataSpansRepresentable(m, n, ldb, dl, d, du, b)) {
        OP_LOGE("aclsparseCgtsv2", "data pointers or address spans are invalid");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
} // namespace

extern "C" aclsparseStatus_t aclsparseCgtsv2_bufferSizeExt(aclsparseHandle_t handle,
    int m, int n, const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, const aclsparseComplex_t *B, int ldb,
    size_t *pBufferSizeInBytes)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseCgtsv2", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (pBufferSizeInBytes == nullptr) {
        OP_LOGE("aclsparseCgtsv2", "pBufferSizeInBytes is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    size_t bytes = 0;
    const auto status = Validate(handle, m, n, dl, d, du, B, ldb, bytes);
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        *pBufferSizeInBytes = bytes;
    } else {
        OP_LOGE("aclsparseCgtsv2", "workspace query validation failed: status=%d", static_cast<int>(status));
    }
    return status;
}

extern "C" aclsparseStatus_t aclsparseCgtsv2(aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, aclsparseComplex_t *B, int ldb, void *pBuffer)
{
    size_t bytes = 0;
    const auto status = Validate(handle, m, n, dl, d, du, B, ldb, bytes);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseCgtsv2", "solve validation failed: status=%d", static_cast<int>(status));
        return status;
    }
    if (n == 0) { return ACL_SPARSE_STATUS_SUCCESS; }
    if (pBuffer == nullptr || reinterpret_cast<uintptr_t>(pBuffer) % cgtsv2::kWorkspaceAlignment != 0 ||
        !IsAddressSpanRepresentable(pBuffer, bytes)) {
        OP_LOGE("aclsparseCgtsv2", "workspace must have a representable span and 128-byte aligned non-null base");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const auto launchStatus = Cgtsv2Dispatch(handle, m, n, dl, d, du, B, ldb, pBuffer);
    if (launchStatus != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseCgtsv2", "backend dispatch failed: status=%d", static_cast<int>(launchStatus));
    }
    return launchStatus;
}
