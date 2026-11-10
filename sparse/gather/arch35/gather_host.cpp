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

/*!
 * \file gather_host.cpp
 * \brief aclsparseGather Host 侧实现：参数校验 + Kernel launch。
 *
 * Gather:  X.values[i] = Y[X.indices[i] - idxBase]  for i = 0 .. nnz-1.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "acl/acl_rt.h"
#include "acl/acl_base_rt.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "gather_tiling_data.h"
#include "gather_kernel.h"

namespace {

bool SupportedValueType(aclDataType type)
{
    return type == ACL_FLOAT16 || type == ACL_BF16 || type == ACL_FLOAT || type == ACL_COMPLEX64 ||
           type == ACL_DOUBLE;
}

size_t ValueTypeSize(aclDataType type)
{
    switch (type) {
        case ACL_FLOAT16:
        case ACL_BF16:
            return sizeof(uint16_t);
        case ACL_FLOAT:
            return sizeof(uint32_t);
        case ACL_COMPLEX64:
        case ACL_DOUBLE:
            return sizeof(uint64_t);
        default:
            return 0;
    }
}

size_t IndexTypeSize(aclsparseIndexType_t type)
{
    if (type == ACL_SPARSE_INDEX_32I) {
        return sizeof(int32_t);
    }
    if (type == ACL_SPARSE_INDEX_64I) {
        return sizeof(int64_t);
    }
    return 0;
}

bool SafeByteSize(uint64_t count, size_t elementSize, size_t &bytes)
{
    if (elementSize == 0 || count > std::numeric_limits<size_t>::max() / elementSize) {
        return false;
    }
    bytes = static_cast<size_t>(count) * elementSize;
    return true;
}

bool AddressRangeIsValid(const void *ptr, size_t bytes)
{
    if (bytes == 0) {
        return true;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    return ptr != nullptr && begin <= std::numeric_limits<uintptr_t>::max() - bytes;
}

bool AddressRangesOverlap(const void *lhs, size_t lhsBytes, const void *rhs, size_t rhsBytes)
{
    if (lhsBytes == 0 || rhsBytes == 0) {
        return false;
    }
    const uintptr_t lhsBegin = reinterpret_cast<uintptr_t>(lhs);
    const uintptr_t rhsBegin = reinterpret_cast<uintptr_t>(rhs);
    return lhsBegin < rhsBegin + rhsBytes && rhsBegin < lhsBegin + lhsBytes;
}

bool IsPointerOnCurrentDevice(const void *ptr, int32_t currentDevice)
{
    aclrtPtrAttributes attributes{};
    if (aclrtPointerGetAttributes(ptr, &attributes) != ACL_SUCCESS) {
        return false;
    }
    const bool deviceAccessible = attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE ||
                                  attributes.location.type == ACL_MEM_LOCATION_TYPE_MANAGED;
    return deviceAccessible && static_cast<int32_t>(attributes.location.id) == currentDevice;
}

// ===========================================================================
// 参数校验
// ===========================================================================
static aclsparseStatus_t ValidateDescriptorObjects(
    aclsparseHandle_t handle, aclsparseConstDnVecDescr_t vecY, aclsparseSpVecDescr_t vecX)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseGather", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (vecY == nullptr) {
        OP_LOGE("aclsparseGather", "vecY is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecX == nullptr) {
        OP_LOGE("aclsparseGather", "vecX is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecY->signature != kDnVecSignature || vecX->signature != kSpVecSignature) {
        OP_LOGE("aclsparseGather", "invalid DnVec/SpVec descriptor signature");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateTypesAndShape(
    aclsparseConstDnVecDescr_t vecY, aclsparseSpVecDescr_t vecX)
{
    if (vecX->valueType != vecY->valueType) {
        OP_LOGE("aclsparseGather", "value type mismatch: X=%d, Y=%d", vecX->valueType, vecY->valueType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!SupportedValueType(vecY->valueType)) {
        OP_LOGE("aclsparseGather", "unsupported value type: %d", vecY->valueType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (IndexTypeSize(vecX->idxType) == 0) {
        OP_LOGE("aclsparseGather", "unsupported index type: %d", vecX->idxType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (vecX->idxBase != ACL_SPARSE_INDEX_BASE_ZERO && vecX->idxBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE("aclsparseGather", "invalid index base: %d", vecX->idxBase);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecY->nums != vecX->size) {
        OP_LOGE("aclsparseGather", "vector size mismatch: Y=%lu, X=%lu", vecY->nums, vecX->size);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecX->nnz > vecX->size) {
        OP_LOGE("aclsparseGather", "nnz(%lu) exceeds vector size(%lu)", vecX->nnz, vecX->size);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateDataBuffers(
    aclsparseConstDnVecDescr_t vecY, aclsparseSpVecDescr_t vecX)
{
    if (vecX->nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (vecY->nums == 0 || vecY->values == nullptr || vecX->indices == nullptr || vecX->values == nullptr) {
        OP_LOGE("aclsparseGather", "non-empty input has zero size or null data pointer");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    size_t yBytes = 0;
    size_t xBytes = 0;
    size_t indexBytes = 0;
    const size_t valueSize = ValueTypeSize(vecY->valueType);
    if (!SafeByteSize(vecY->nums, valueSize, yBytes) || !SafeByteSize(vecX->nnz, valueSize, xBytes) ||
        !SafeByteSize(vecX->nnz, IndexTypeSize(vecX->idxType), indexBytes) ||
        !AddressRangeIsValid(vecY->values, yBytes) || !AddressRangeIsValid(vecX->values, xBytes) ||
        !AddressRangeIsValid(vecX->indices, indexBytes)) {
        OP_LOGE("aclsparseGather", "buffer byte-size or address range overflow");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (AddressRangesOverlap(vecY->values, yBytes, vecX->values, xBytes) ||
        AddressRangesOverlap(vecX->indices, indexBytes, vecX->values, xBytes)) {
        OP_LOGE("aclsparseGather", "output values overlap an input buffer");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    int32_t currentDevice = -1;
    if (aclrtGetDevice(&currentDevice) != ACL_SUCCESS || !IsPointerOnCurrentDevice(vecY->values, currentDevice) ||
        !IsPointerOnCurrentDevice(vecX->indices, currentDevice) ||
        !IsPointerOnCurrentDevice(vecX->values, currentDevice)) {
        OP_LOGE("aclsparseGather", "data pointer is not on the current NPU device");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateGatherParams(
    aclsparseHandle_t handle, aclsparseConstDnVecDescr_t vecY, aclsparseSpVecDescr_t vecX)
{
    aclsparseStatus_t status = ValidateDescriptorObjects(handle, vecY, vecX);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateTypesAndShape(vecY, vecX);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    return ValidateDataBuffers(vecY, vecX);
}

// ===========================================================================
// Kernel launch
// ===========================================================================
static aclsparseStatus_t LaunchGatherKernel(
    aclsparseHandle_t handle, aclsparseConstDnVecDescr_t vecY, aclsparseSpVecDescr_t vecX)
{
    GatherTilingData tiling{};
    tiling.nnz = vecX->nnz;
    tiling.idxBase = vecX->idxBase;
    tiling.valType = vecY->valueType;
    tiling.idxType = vecX->idxType;

    uint32_t maxCoreNum = GetAivCoreCount();
    CHECK_RET(maxCoreNum > 0, OP_LOGE("aclsparseGather", "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    tiling.numBlocks = std::min<uint64_t>(CeilDiv<uint64_t>(tiling.nnz, kGatherMaxThreadsPerBlock), maxCoreNum);

    OP_LOGD(
        "aclsparseGather", "Tiling: nnz=%ld, idxBase=%d, valType=%d, idxType=%d, numBlocks=%u", tiling.nnz,
        tiling.idxBase, tiling.valType, tiling.idxType, tiling.numBlocks);

    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    gather_kernel_do(
        reinterpret_cast<GM_ADDR>(vecX->indices), reinterpret_cast<GM_ADDR>(const_cast<void*>(vecY->values)),
        reinterpret_cast<GM_ADDR>(vecX->values), tiling, handle->stream);

    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseGather", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

// ============================================================================
// Public API
// ============================================================================
extern "C" {

aclsparseStatus_t aclsparseGather(aclsparseHandle_t handle, aclsparseConstDnVecDescr_t vecY, aclsparseSpVecDescr_t vecX)
{
    aclsparseStatus_t st = ValidateGatherParams(handle, vecY, vecX);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (vecX->nnz == 0) {
        OP_LOGD("aclsparseGather", "nnz=0, nothing to gather");
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    return LaunchGatherKernel(handle, vecY, vecX);
}

} // extern "C"
