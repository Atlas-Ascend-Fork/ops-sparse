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
 * \file aclsparse_descr.cpp
 * \brief 稀疏/稠密向量、稠密矩阵与 CSR 稀疏矩阵描述符的公共构造/销毁接口。
 *
 * 这些 API 与具体算子架构无关，需在 arch22(SpMV) / arch35(SpMM) 等所有 SOC 构建中可用。
 */

#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"

#include <cstdint>
#include <limits>
#include <new>
#include <atomic>

namespace {

uint64_t NextSpmmWorkspaceOwnerId()
{
    static std::atomic<uint64_t> nextOwnerId{1};
    return nextOwnerId.fetch_add(1, std::memory_order_relaxed);
}

static int64_t CeilDiv(int64_t value, int64_t divisor)
{
    if (value < 0) {
        return 0;
    }
    // value + divisor - 1 会在 value 接近 INT64_MAX 时有符号溢出（UB），且这里的
    // CeilDiv 排在 BlockedEllCapacity 的容量/溢出检查之前。对非负 value 与正
    // divisor，value / divisor + (有余数) 恒不溢出，且与向上取整等价；除法与取模
    // 都放进 divisor > 0 的分支里，非正除数直接返回 0。
    return divisor > 0 ? value / divisor + (value % divisor != 0 ? 1 : 0) : 0;
}

static aclsparseStatus_t ValidateAttributeParams(const void* spMatDescr, const void* data)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (data == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static bool IsValidSparseValueType(aclDataType valueType)
{
    switch (valueType) {
        case ACL_FLOAT:
        case ACL_FLOAT16:
        case ACL_DOUBLE:
        case ACL_INT8:
        case ACL_INT16:
        case ACL_INT32:
        case ACL_INT64:
        case ACL_UINT8:
        case ACL_UINT16:
        case ACL_UINT32:
        case ACL_UINT64:
        case ACL_BF16:
        case ACL_FLOAT8_E4M3FN:
        case ACL_FLOAT8_E5M2:
        case ACL_FLOAT4_E2M1:
        case ACL_COMPLEX64:
        case ACL_COMPLEX128:
            return true;
        default:
            return false;
    }
}

static bool ValidBatchExtent(int count, int64_t stride, uint64_t extent, uint64_t elementBytes)
{
    if (elementBytes == 0) {
        return false;
    }
    const uint64_t limit = uint64_t(std::numeric_limits<int64_t>::max()) / elementBytes;
    if (count < 1 || count > 65535 || stride < 0 || extent > limit) {
        return false;
    }
    if (count == 1) {
        return stride == 0 || uint64_t(stride) >= extent;
    }
    return uint64_t(stride) >= extent && uint64_t(stride) <= (limit - extent) / (count - 1);
}

static uint64_t SparseValueBytes(aclDataType type)
{
    switch (type) {
        case ACL_FLOAT16:
        case ACL_BF16:
        case ACL_INT16:
        case ACL_UINT16:
            return 2;
        case ACL_FLOAT:
        case ACL_INT32:
        case ACL_UINT32:
            return 4;
        case ACL_DOUBLE:
        case ACL_COMPLEX64:
        case ACL_INT64:
        case ACL_UINT64:
            return 8;
        case ACL_COMPLEX128:
            return 16;
        default:
            return 1;
    }
}

static bool IsValidElementStride(uint64_t logicalSize, int64_t stride)
{
    if (stride <= 0) {
        return false;
    }
    if (logicalSize <= 1u) {
        return true;
    }
    return static_cast<uint64_t>(stride) <=
           static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / (logicalSize - 1u);
}

static aclsparseStatus_t ValidateAttributeAccess(const void* spMatDescr, const void* data,
                                                 aclsparseSpMatAttribute_t attribute, size_t dataSize)
{
    aclsparseStatus_t st = ValidateAttributeParams(spMatDescr, data);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (attribute == ACL_SPARSE_SPMAT_FILL_MODE) {
        if (dataSize != sizeof(aclsparseFillMode_t) ||
            reinterpret_cast<uintptr_t>(data) % alignof(aclsparseFillMode_t) != 0) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    } else if (attribute == ACL_SPARSE_SPMAT_DIAG_TYPE) {
        if (dataSize != sizeof(aclsparseDiagType_t) ||
            reinterpret_cast<uintptr_t>(data) % alignof(aclsparseDiagType_t) != 0) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    } else {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static void ResetSpmvActivePlan(aclsparseSpMatDescr_t descr)
{
    descr->activeBuffer = nullptr;
    descr->activeSpmvBuffer = nullptr;
    descr->activeRowOffsets = nullptr;
    descr->activeColIndices = nullptr;
    descr->activeValues = nullptr;
    descr->activeRowOffsetsStride = 1;
    descr->activeColIndStride = 1;
    descr->activeValuesStride = 1;
    descr->activeXStride = 1;
    descr->activeYStride = 1;
    descr->activeRows = descr->activeCols = descr->activeNnz = 0;
    descr->activeBase = ACL_SPARSE_INDEX_BASE_ZERO;
    descr->activeValueType = ACL_FLOAT;
    descr->activeXType = ACL_FLOAT;
    descr->activeYType = ACL_FLOAT;
    descr->activeComputeType = ACL_FLOAT;
    descr->activeOp = ACL_SPARSE_OP_NON_TRANSPOSE;
    descr->activeAlg = -1;
    descr->activeStream = nullptr;
    descr->activeStreamEpoch = 0;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseCreateDnVec(aclsparseDnVecDescr_t* dnVecDescr, int64_t size, void* values,
                                       aclDataType valueType)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Null storage is allowed at descriptor creation so callers can query
    // operator workspace before allocating and binding the vector data.
    // Compute APIs remain responsible for rejecting unbound non-empty data.
    if (!IsValidSparseValueType(valueType)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = new (std::nothrow) aclsparseDnVecDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->signature = kDnVecSignature;
    inner->nums = static_cast<uint64_t>(size);
    inner->values = values;
    inner->valueType = valueType;
    *dnVecDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstDnVec(aclsparseConstDnVecDescr_t* dnVecDescr, int64_t size, const void* values,
                                            aclDataType valueType)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseDnVecDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateDnVec(&tmp, size, const_cast<void*>(values), valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *dnVecDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroyDnVec(aclsparseConstDnVecDescr_t dnVecDescr)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = const_cast<aclsparseDnVecDescr*>(dnVecDescr);
    if (inner->signature != kDnVecSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    inner->signature = 0;
    delete inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateSpVec(aclsparseSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, void* indices,
                                       void* values, aclsparseIndexType_t idxType, aclsparseIndexBase_t idxBase,
                                       aclDataType valueType)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size < 0 || nnz < 0 || nnz > size) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && (indices == nullptr || values == nullptr)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (idxType != ACL_SPARSE_INDEX_32I && idxType != ACL_SPARSE_INDEX_64I) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (idxBase != ACL_SPARSE_INDEX_BASE_ZERO && idxBase != ACL_SPARSE_INDEX_BASE_ONE) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsValidSparseValueType(valueType)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = new (std::nothrow) aclsparseSpVecDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->size = static_cast<uint64_t>(size);
    inner->nnz = static_cast<uint64_t>(nnz);
    inner->indices = indices;
    inner->values = values;
    inner->idxType = idxType;
    inner->idxBase = idxBase;
    inner->valueType = valueType;
    inner->signature = kSpVecSignature;
    *spVecDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstSpVec(aclsparseConstSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz,
                                            const void* indices, const void* values, aclsparseIndexType_t idxType,
                                            aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseSpVecDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateSpVec(&tmp, size, nnz, const_cast<void*>(indices), const_cast<void*>(values),
                                                idxType, idxBase, valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *spVecDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroySpVec(aclsparseConstSpVecDescr_t spVecDescr)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = const_cast<aclsparseSpVecDescr*>(spVecDescr);
    if (inner->signature != kSpVecSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    inner->signature = 0;
    delete inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpVecGet(aclsparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices,
                                    void** values, aclsparseIndexType_t* idxType, aclsparseIndexBase_t* idxBase,
                                    aclDataType* valueType)
{
    if (spVecDescr == nullptr || spVecDescr->signature != kSpVecSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size != nullptr) {
        *size = static_cast<int64_t>(spVecDescr->size);
    }
    if (nnz != nullptr) {
        *nnz = static_cast<int64_t>(spVecDescr->nnz);
    }
    if (indices != nullptr) {
        *indices = spVecDescr->indices;
    }
    if (values != nullptr) {
        *values = spVecDescr->values;
    }
    if (idxType != nullptr) {
        *idxType = spVecDescr->idxType;
    }
    if (idxBase != nullptr) {
        *idxBase = spVecDescr->idxBase;
    }
    if (valueType != nullptr) {
        *valueType = spVecDescr->valueType;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstSpVecGet(aclsparseConstSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz,
                                         const void** indices, const void** values, aclsparseIndexType_t* idxType,
                                         aclsparseIndexBase_t* idxBase, aclDataType* valueType)
{
    if (spVecDescr == nullptr || spVecDescr->signature != kSpVecSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size != nullptr) {
        *size = static_cast<int64_t>(spVecDescr->size);
    }
    if (nnz != nullptr) {
        *nnz = static_cast<int64_t>(spVecDescr->nnz);
    }
    if (indices != nullptr) {
        *indices = spVecDescr->indices;
    }
    if (values != nullptr) {
        *values = spVecDescr->values;
    }
    if (idxType != nullptr) {
        *idxType = spVecDescr->idxType;
    }
    if (idxBase != nullptr) {
        *idxBase = spVecDescr->idxBase;
    }
    if (valueType != nullptr) {
        *valueType = spVecDescr->valueType;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpVecGetIndexBase(aclsparseConstSpVecDescr_t spVecDescr, aclsparseIndexBase_t* idxBase)
{
    if (spVecDescr == nullptr || spVecDescr->signature != kSpVecSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (idxBase != nullptr) {
        *idxBase = spVecDescr->idxBase;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpVecGetValues(aclsparseSpVecDescr_t spVecDescr, void** values)
{
    if (spVecDescr == nullptr || spVecDescr->signature != kSpVecSignature || values == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *values = spVecDescr->values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstSpVecGetValues(aclsparseConstSpVecDescr_t spVecDescr, const void** values)
{
    if (spVecDescr == nullptr || spVecDescr->signature != kSpVecSignature || values == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *values = spVecDescr->values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpVecSetValues(aclsparseSpVecDescr_t spVecDescr, void* values)
{
    if (spVecDescr == nullptr || spVecDescr->signature != kSpVecSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    spVecDescr->values = values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnVecGet(aclsparseDnVecDescr_t dnVecDescr, int64_t* size, void** values,
                                    aclDataType* valueType)
{
    if (dnVecDescr == nullptr || dnVecDescr->signature != kDnVecSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size != nullptr) {
        *size = static_cast<int64_t>(dnVecDescr->nums);
    }
    if (values != nullptr) {
        *values = dnVecDescr->values;
    }
    if (valueType != nullptr) {
        *valueType = dnVecDescr->valueType;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstDnVecGet(aclsparseConstDnVecDescr_t dnVecDescr, int64_t* size, const void** values,
                                         aclDataType* valueType)
{
    if (dnVecDescr == nullptr || dnVecDescr->signature != kDnVecSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size != nullptr) {
        *size = static_cast<int64_t>(dnVecDescr->nums);
    }
    if (values != nullptr) {
        *values = dnVecDescr->values;
    }
    if (valueType != nullptr) {
        *valueType = dnVecDescr->valueType;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnVecGetValues(aclsparseDnVecDescr_t dnVecDescr, void** values)
{
    if (dnVecDescr == nullptr || dnVecDescr->signature != kDnVecSignature || values == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *values = dnVecDescr->values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstDnVecGetValues(aclsparseConstDnVecDescr_t dnVecDescr, const void** values)
{
    if (dnVecDescr == nullptr || dnVecDescr->signature != kDnVecSignature || values == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *values = dnVecDescr->values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnVecSetValues(aclsparseDnVecDescr_t dnVecDescr, void* values)
{
    if (dnVecDescr == nullptr || dnVecDescr->signature != kDnVecSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    dnVecDescr->values = values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnVecSetStride(aclsparseDnVecDescr_t dnVecDescr, int64_t stride)
{
    if (dnVecDescr == nullptr || dnVecDescr->signature != kDnVecSignature ||
        !IsValidElementStride(dnVecDescr->nums, stride)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    dnVecDescr->stride = stride;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnVecGetStride(aclsparseConstDnVecDescr_t dnVecDescr, int64_t* stride)
{
    if (dnVecDescr == nullptr || dnVecDescr->signature != kDnVecSignature || stride == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *stride = dnVecDescr->stride;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateCsr(aclsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                     void* csrRowOffsets, void* csrColInd, void* csrValues,
                                     aclsparseIndexType_t csrRowOffsetsType, aclsparseIndexType_t csrColIndType,
                                     aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rows < 0 || cols < 0 || nnz < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Null CSR pointers are allowed at Create for delayed binding (SetPointers /
    // later BufferSize/Execute validate before use). See densetosparse L2_17,
    // sddmm NullCsrPtrs.
    aclsparseStatus_t idxSt = AclsparseValidateSupportedCsrIndexTypesExtended(csrRowOffsetsType, csrColIndType);
    if (idxSt != ACL_SPARSE_STATUS_SUCCESS) {
        return idxSt;
    }
    if (idxBase != ACL_SPARSE_INDEX_BASE_ZERO && idxBase != ACL_SPARSE_INDEX_BASE_ONE) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsValidSparseValueType(valueType)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = new (std::nothrow) aclsparseSpMatDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->format = ACL_SPARSE_FORMAT_CSR;
    inner->spmmWorkspaceOwnerId = NextSpmmWorkspaceOwnerId();
    inner->rows = static_cast<uint64_t>(rows);
    inner->cols = static_cast<uint64_t>(cols);
    inner->nnz = static_cast<uint64_t>(nnz);
    inner->ptrs = csrRowOffsets;
    inner->idxs = csrColInd;
    inner->values = csrValues;
    inner->baseType = idxBase;
    inner->ptrType = csrRowOffsetsType;
    inner->IdxType = csrColIndType;
    inner->valueType = valueType;
    inner->signature = kSpMatSignature;
    *spMatDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

namespace {
bool ValidBsrShape(aclsparseSpMatDescr_t* descr, int64_t blockRows, int64_t blockCols, int64_t blockNnz,
                   int64_t rowBlockSize, int64_t colBlockSize, aclsparseIndexBase_t base, aclsparseOrder_t order)
{
    if (!descr || blockRows < 0 || blockCols < 0 || blockNnz < 0 || rowBlockSize != colBlockSize || rowBlockSize < 2 ||
        rowBlockSize > 128 || (rowBlockSize & (rowBlockSize - 1)) ||
        (base != ACL_SPARSE_INDEX_BASE_ZERO && base != ACL_SPARSE_INDEX_BASE_ONE) ||
        (order != ACL_SPARSE_ORDER_ROW && order != ACL_SPARSE_ORDER_COL)) {
        return false;
    }
    return true;
}
} // namespace

static bool SupportedBsrTypes(aclsparseIndexType_t ptrType, aclsparseIndexType_t idxType, aclDataType type)
{
    if (ptrType != ACL_SPARSE_INDEX_32I || idxType != ACL_SPARSE_INDEX_32I ||
        (type != ACL_FLOAT && type != ACL_FLOAT16 && type != ACL_BF16 && type != ACL_COMPLEX64)) {
        return false;
    }
    return true;
}

aclsparseStatus_t aclsparseCreateBsr(aclsparseSpMatDescr_t* descr, int64_t blockRows, int64_t blockCols,
                                     int64_t blockNnz, int64_t rowBlockSize, int64_t colBlockSize, void* offsets,
                                     void* columns, void* values, aclsparseIndexType_t ptrType,
                                     aclsparseIndexType_t idxType, aclsparseIndexBase_t base, aclDataType type,
                                     aclsparseOrder_t order)
{
    if (!ValidBsrShape(descr, blockRows, blockCols, blockNnz, rowBlockSize, colBlockSize, base, order)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!SupportedBsrTypes(ptrType, idxType, type)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const uint64_t bd = rowBlockSize;
    const uint64_t limit = std::numeric_limits<int64_t>::max();
    if (uint64_t(blockRows) > limit / bd || uint64_t(blockCols) > limit / bd ||
        uint64_t(blockNnz) > limit / (bd * bd * SparseValueBytes(type)) || blockRows > INT32_MAX - 1 ||
        blockCols > INT32_MAX || blockNnz > INT32_MAX - (base == ACL_SPARSE_INDEX_BASE_ONE) || !offsets ||
        (blockNnz && (!columns || !values)) || (blockNnz && (!blockRows || !blockCols))) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = new (std::nothrow) aclsparseSpMatDescr();
    if (!inner) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    inner->format = ACL_SPARSE_FORMAT_BSR;
    inner->rows = uint64_t(blockRows) * bd;
    inner->cols = uint64_t(blockCols) * bd;
    inner->nnz = blockNnz;
    inner->blockSize = bd;
    inner->blockOrder = order;
    inner->ptrs = offsets;
    inner->idxs = columns;
    inner->values = values;
    inner->ptrType = ptrType;
    inner->IdxType = idxType;
    inner->baseType = base;
    inner->valueType = type;
    *descr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstBsr(aclsparseConstSpMatDescr_t* descr, int64_t blockRows, int64_t blockCols,
                                          int64_t blockNnz, int64_t rowBlockSize, int64_t colBlockSize,
                                          const void* offsets, const void* columns, const void* values,
                                          aclsparseIndexType_t ptrType, aclsparseIndexType_t idxType,
                                          aclsparseIndexBase_t base, aclDataType type, aclsparseOrder_t order)
{
    if (!descr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseSpMatDescr_t mutableDescr = nullptr;
    const auto status = aclsparseCreateBsr(&mutableDescr, blockRows, blockCols, blockNnz, rowBlockSize, colBlockSize,
                                           const_cast<void*>(offsets), const_cast<void*>(columns),
                                           const_cast<void*>(values), ptrType, idxType, base, type, order);
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        *descr = mutableDescr;
    }
    return status;
}

aclsparseStatus_t aclsparseBsrSetStridedBatch(aclsparseSpMatDescr_t descr, int count, int64_t offsets, int64_t columns,
                                              int64_t values)
{
    if (!descr || descr->signature != kSpMatSignature || descr->format != ACL_SPARSE_FORMAT_BSR ||
        !ValidBatchExtent(count, offsets, descr->rows / descr->blockSize + 1, 4) ||
        !ValidBatchExtent(count, columns, descr->nnz, 4) ||
        !ValidBatchExtent(count, values, descr->nnz * descr->blockSize * descr->blockSize,
                          SparseValueBytes(descr->valueType))) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    descr->batchCount = count;
    descr->offsetsBatchStride = offsets;
    descr->columnsBatchStride = columns;
    descr->valuesBatchStride = values;
    descr->sddmmState.reset();
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstCsr(aclsparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols,
                                          int64_t nnz, const void* csrRowOffsets, const void* csrColInd,
                                          const void* csrValues, aclsparseIndexType_t csrRowOffsetsType,
                                          aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase,
                                          aclDataType valueType)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseSpMatDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateCsr(&tmp, rows, cols, nnz, const_cast<void*>(csrRowOffsets),
                                              const_cast<void*>(csrColInd), const_cast<void*>(csrValues),
                                              csrRowOffsetsType, csrColIndType, idxBase, valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *spMatDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

/// Blocked-ELL 的容量换算与溢出校验。
///
/// rows/cols/ellCols 都可能留下尾块，容量按补齐后的块网格计算，使每个 slot 恒占
/// ellBlockSize^2 个标量；几何整除时与 rows*ellCols 一致。
static aclsparseStatus_t BlockedEllCapacity(int64_t rows, int64_t cols, int64_t blockSize, int64_t ellCols,
                                            int64_t* indexCount, int64_t* valueCount)
{
    if (blockSize == 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const int64_t blockRows = CeilDiv(rows, blockSize);
    const int64_t slots = CeilDiv(ellCols, blockSize);
    // Slots 以整块列计数，上界是补齐后的块列数而不是 cols；有列尾块时 ellCols
    // 可能超过 cols。
    if (slots > CeilDiv(cols, blockSize)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const int64_t limit = std::numeric_limits<int64_t>::max();
    if ((blockRows != 0 && slots > limit / blockRows) || blockSize > limit / blockSize) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *indexCount = blockRows * slots;
    const int64_t blockScalars = blockSize * blockSize;
    if (*indexCount != 0 && blockScalars > limit / *indexCount) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *valueCount = *indexCount * blockScalars;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateBlockedEll(aclsparseSpMatDescr_t* descr, int64_t rows, int64_t cols,
                                            int64_t ellBlockSize, int64_t ellCols, void* ellColInd, void* ellValue,
                                            aclsparseIndexType_t indexType, aclsparseIndexBase_t indexBase,
                                            aclDataType valueType)
{
    if (descr == nullptr || rows < 0 || cols < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (ellBlockSize <= 0 || ellCols < 0 || (indexType != ACL_SPARSE_INDEX_32I && indexType != ACL_SPARSE_INDEX_64I) ||
        (indexBase != ACL_SPARSE_INDEX_BASE_ZERO && indexBase != ACL_SPARSE_INDEX_BASE_ONE)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 描述符只约束上界与容量：ellCols 可以取到补齐后的列范围（尾块的 slot 仍然
    // 存在并填零）。是否必须为整块列由各架构 host 层自行决定——arch22 在
    // ValidateBell() 中要求整除，arch35 支持列尾块。
    int64_t indexCount = 0;
    int64_t valueCount = 0;
    const aclsparseStatus_t capacity = BlockedEllCapacity(rows, cols, ellBlockSize, ellCols, &indexCount, &valueCount);
    if (capacity != ACL_SPARSE_STATUS_SUCCESS) {
        return capacity;
    }
    if ((indexCount > 0 && ellColInd == nullptr) || (valueCount > 0 && ellValue == nullptr)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = new (std::nothrow) aclsparseSpMatDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    inner->format = ACL_SPARSE_FORMAT_BLOCKED_ELL;
    inner->rows = static_cast<uint64_t>(rows);
    inner->cols = static_cast<uint64_t>(cols);
    inner->nnz = static_cast<uint64_t>(valueCount);
    inner->idxs = ellColInd;
    inner->ellColInd = ellColInd;
    inner->values = ellValue;
    inner->baseType = indexBase;
    inner->ptrType = indexType;
    inner->IdxType = indexType;
    inner->valueType = valueType;
    inner->ellBlockSize = static_cast<uint64_t>(ellBlockSize);
    inner->ellCols = static_cast<uint64_t>(ellCols);
    inner->signature = kSpMatSignature;
    *descr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SetCompressedPointers(aclsparseSpMatDescr_t descr, aclsparseFormat_t format, void* offsets,
                                               void* indices, void* values)
{
    if (descr == nullptr || descr->signature != kSpMatSignature || descr->format != format ||
        (offsets == nullptr && descr->rows > 0 && descr->cols > 0) ||
        (descr->nnz > 0 && (indices == nullptr || values == nullptr))) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (descr->ptrs != offsets || descr->idxs != indices) {
        descr->sddmmState.reset();
    }
    if (descr->ptrs != offsets || descr->idxs != indices || descr->values != values) {
        ResetSpmvActivePlan(descr);
    }
    // Analysis already wrote these offsets. A new offsets allocation requires
    // re-analysis; freshly allocated output indices/values do not.
    if (descr->ptrs != offsets) {
        descr->activeDenseToSparseBuffer = nullptr;
        descr->denseToSparseAnalysisReady = false;
    }
    descr->ptrs = offsets;
    descr->idxs = indices;
    descr->values = values;
    descr->spmmWorkspaceOwnerId = NextSpmmWorkspaceOwnerId();
    descr->activeBuffer = nullptr;
    descr->activeSpmvBuffer = nullptr;
    descr->spmvCubeMaxRowLengthValid = false;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCsrSetPointers(aclsparseSpMatDescr_t descr, void* rowOffsets, void* colIndices, void* values)
{
    return SetCompressedPointers(descr, ACL_SPARSE_FORMAT_CSR, rowOffsets, colIndices, values);
}

aclsparseStatus_t aclsparseCsrSetStrides(aclsparseSpMatDescr_t descr, int64_t rowOffsetsStride, int64_t colIndStride,
                                         int64_t valuesStride)
{
    if (descr == nullptr || descr->format != ACL_SPARSE_FORMAT_CSR ||
        !IsValidElementStride(descr->rows + 1u, rowOffsetsStride) || !IsValidElementStride(descr->nnz, colIndStride) ||
        !IsValidElementStride(descr->nnz, valuesStride)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    descr->ptrStride = rowOffsetsStride;
    descr->idxStride = colIndStride;
    descr->valueStride = valuesStride;
    descr->spmmWorkspaceOwnerId = NextSpmmWorkspaceOwnerId();
    descr->activeDenseToSparseBuffer = nullptr;
    descr->denseToSparseAnalysisReady = false;
    descr->activeBuffer = nullptr;
    descr->activeSpmvBuffer = nullptr;
    descr->spmvCubeMaxRowLengthValid = false;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCsrGetStrides(aclsparseConstSpMatDescr_t descr, int64_t* rowOffsetsStride,
                                         int64_t* colIndStride, int64_t* valuesStride)
{
    if (descr == nullptr || descr->format != ACL_SPARSE_FORMAT_CSR || rowOffsetsStride == nullptr ||
        colIndStride == nullptr || valuesStride == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *rowOffsetsStride = descr->ptrStride;
    *colIndStride = descr->idxStride;
    *valuesStride = descr->valueStride;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCscSetPointers(aclsparseSpMatDescr_t descr, void* colOffsets, void* rowIndices, void* values)
{
    return SetCompressedPointers(descr, ACL_SPARSE_FORMAT_CSC, colOffsets, rowIndices, values);
}

aclsparseStatus_t aclsparseCooSetPointers(aclsparseSpMatDescr_t descr, void* rowIndices, void* colIndices, void* values)
{
    if (descr == nullptr || descr->signature != kSpMatSignature || descr->format != ACL_SPARSE_FORMAT_COO ||
        (descr->nnz > 0 && (rowIndices == nullptr || colIndices == nullptr || values == nullptr))) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (descr->rowInds != rowIndices || descr->colInds != colIndices || descr->values != values) {
        ResetSpmvActivePlan(descr);
    }
    descr->rowInds = rowIndices;
    descr->colInds = colIndices;
    descr->ptrs = rowIndices;
    descr->idxs = colIndices;
    descr->values = values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroySpMat(aclsparseConstSpMatDescr_t spMatDescr)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    auto* inner = const_cast<aclsparseSpMatDescr*>(spMatDescr);
    if (inner->signature != kSpMatSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    inner->signature = 0;
    delete inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMatGetFormat(aclsparseConstSpMatDescr_t spMatDescr, aclsparseFormat_t* format)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (format != nullptr) {
        *format = spMatDescr->format;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMatGetValues(aclsparseSpMatDescr_t spMatDescr, void** values)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = spMatDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstSpMatGetValues(aclsparseConstSpMatDescr_t spMatDescr, const void** values)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = spMatDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMatSetValues(aclsparseSpMatDescr_t spMatDescr, void* values)
{
    if (spMatDescr == nullptr || spMatDescr->signature != kSpMatSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (spMatDescr->values != values) {
        ResetSpmvActivePlan(spMatDescr);
    }
    spMatDescr->values = values;
    spMatDescr->spmmWorkspaceOwnerId = NextSpmmWorkspaceOwnerId();
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMatGetSize(aclsparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols,
                                        int64_t* nnz)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rows != nullptr) {
        *rows = static_cast<int64_t>(spMatDescr->rows);
    }
    if (cols != nullptr) {
        *cols = static_cast<int64_t>(spMatDescr->cols);
    }
    if (nnz != nullptr) {
        *nnz = static_cast<int64_t>(spMatDescr->nnz);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateDnMat(aclsparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld,
                                       void* values, aclDataType valueType, aclsparseOrder_t order)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Allow zero dims (SparseToDense / DenseToSparse empty matrix paths).
    if (rows < 0 || cols < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 稀疏侧 offsets/indices 走 I32：超出 INT32_MAX 的维度在创建期拒绝。
    if (rows > static_cast<int64_t>(INT32_MAX) || cols > static_cast<int64_t>(INT32_MAX)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (ld < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rows > 0 && cols > 0 && values == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsValidSparseValueType(valueType)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (order != ACL_SPARSE_ORDER_ROW && order != ACL_SPARSE_ORDER_COL) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // ld 约束：行主序需 >= cols；列主序需 >= rows（零维时另一侧可为 0）。
    if (order == ACL_SPARSE_ORDER_ROW && ld < cols) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (order == ACL_SPARSE_ORDER_COL && ld < rows) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = new (std::nothrow) aclsparseDnMatDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    inner->signature = kDnMatSignature;
    inner->rows = rows;
    inner->cols = cols;
    inner->ld = ld;
    inner->order = order;
    inner->values = values;
    inner->valueType = valueType;
    *dnMatDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroyDnMat(aclsparseConstDnMatDescr_t dnMatDescr)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = const_cast<aclsparseDnMatDescr*>(dnMatDescr);
    if (inner->signature != kDnMatSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    inner->signature = 0;
    delete inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnMatGet(aclsparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld,
                                    void** values, aclDataType* valueType, aclsparseOrder_t* order)
{
    if (dnMatDescr == nullptr || dnMatDescr->signature != kDnMatSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rows != nullptr) {
        *rows = dnMatDescr->rows;
    }
    if (cols != nullptr) {
        *cols = dnMatDescr->cols;
    }
    if (ld != nullptr) {
        *ld = dnMatDescr->ld;
    }
    if (values != nullptr) {
        *values = dnMatDescr->values;
    }
    if (valueType != nullptr) {
        *valueType = dnMatDescr->valueType;
    }
    if (order != nullptr) {
        *order = dnMatDescr->order;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstDnMatGet(aclsparseConstDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols,
                                         int64_t* ld, const void** values, aclDataType* valueType,
                                         aclsparseOrder_t* order)
{
    if (dnMatDescr == nullptr || dnMatDescr->signature != kDnMatSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rows != nullptr) {
        *rows = dnMatDescr->rows;
    }
    if (cols != nullptr) {
        *cols = dnMatDescr->cols;
    }
    if (ld != nullptr) {
        *ld = dnMatDescr->ld;
    }
    if (values != nullptr) {
        *values = dnMatDescr->values;
    }
    if (valueType != nullptr) {
        *valueType = dnMatDescr->valueType;
    }
    if (order != nullptr) {
        *order = dnMatDescr->order;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnMatGetValues(aclsparseDnMatDescr_t dnMatDescr, void** values)
{
    if (dnMatDescr == nullptr || dnMatDescr->signature != kDnMatSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = dnMatDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstDnMatGetValues(aclsparseConstDnMatDescr_t dnMatDescr, const void** values)
{
    if (dnMatDescr == nullptr || dnMatDescr->signature != kDnMatSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = dnMatDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnMatSetValues(aclsparseDnMatDescr_t dnMatDescr, void* values)
{
    if (dnMatDescr == nullptr || dnMatDescr->signature != kDnMatSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    dnMatDescr->values = values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnMatGetStridedBatch(aclsparseConstDnMatDescr_t descr, int* count, int64_t* stride)
{
    if (!descr || descr->signature != kDnMatSignature || !count || !stride) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *count = descr->batchCount;
    *stride = descr->batchStride;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnMatSetStridedBatch(aclsparseDnMatDescr_t descr, int count, int64_t stride)
{
    if (!descr || descr->signature != kDnMatSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const uint64_t major = descr->order == ACL_SPARSE_ORDER_ROW ? descr->rows : descr->cols;
    const uint64_t minor = descr->order == ACL_SPARSE_ORDER_ROW ? descr->cols : descr->rows;
    const uint64_t limit = std::numeric_limits<int64_t>::max();
    if (minor > limit || (major > 1 && uint64_t(descr->ld) > (limit - minor) / (major - 1))) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const uint64_t extent = major && minor ? (major - 1) * uint64_t(descr->ld) + minor : 0;
    if (!ValidBatchExtent(count, stride, extent, SparseValueBytes(descr->valueType))) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    descr->batchCount = count;
    descr->batchStride = stride;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstDnMat(aclsparseConstDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols,
                                            int64_t ld, const void* values, aclDataType valueType,
                                            aclsparseOrder_t order)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseDnMatDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateDnMat(&tmp, rows, cols, ld, const_cast<void*>(values), valueType, order);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *dnMatDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMatSetAttribute(aclsparseSpMatDescr_t spMatDescr, aclsparseSpMatAttribute_t attribute,
                                             const void* data, size_t dataSize)
{
    aclsparseStatus_t st = ValidateAttributeAccess(spMatDescr, data, attribute, dataSize);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (attribute == ACL_SPARSE_SPMAT_FILL_MODE) {
        aclsparseFillMode_t fm = *static_cast<const aclsparseFillMode_t*>(data);
        if (fm != ACL_SPARSE_FILL_MODE_LOWER && fm != ACL_SPARSE_FILL_MODE_UPPER) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        spMatDescr->fillMode = fm;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (attribute == ACL_SPARSE_SPMAT_DIAG_TYPE) {
        aclsparseDiagType_t dt = *static_cast<const aclsparseDiagType_t*>(data);
        if (dt != ACL_SPARSE_DIAG_TYPE_NON_UNIT && dt != ACL_SPARSE_DIAG_TYPE_UNIT) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        spMatDescr->diagType = dt;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    return ACL_SPARSE_STATUS_INVALID_VALUE;
}

aclsparseStatus_t aclsparseSpMatGetAttribute(aclsparseConstSpMatDescr_t spMatDescr, aclsparseSpMatAttribute_t attribute,
                                             void* data, size_t dataSize)
{
    aclsparseStatus_t st = ValidateAttributeAccess(spMatDescr, data, attribute, dataSize);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (attribute == ACL_SPARSE_SPMAT_FILL_MODE) {
        *static_cast<aclsparseFillMode_t*>(data) = spMatDescr->fillMode;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (attribute == ACL_SPARSE_SPMAT_DIAG_TYPE) {
        *static_cast<aclsparseDiagType_t*>(data) = spMatDescr->diagType;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    return ACL_SPARSE_STATUS_INVALID_VALUE;
}

aclsparseStatus_t aclsparseCreateCoo(aclsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                     void* cooRowInd, void* cooColInd, void* cooValues, aclsparseIndexType_t cooIdxType,
                                     aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    aclsparseStatus_t st = ValidateSpMatCreateParams(spMatDescr, rows, cols, nnz);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (cooIdxType != ACL_SPARSE_INDEX_32I && cooIdxType != ACL_SPARSE_INDEX_64I) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = new (std::nothrow) aclsparseSpMatDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    inner->format = ACL_SPARSE_FORMAT_COO;
    inner->rows = static_cast<uint64_t>(rows);
    inner->cols = static_cast<uint64_t>(cols);
    inner->nnz = static_cast<uint64_t>(nnz);
    // COO reuses the 'ptrs' field to store cooRowInd (there is no row-offset
    // pointer array in COO). 'ptrType' is set to cooIdxType so that the
    // ptrType == IdxType check in SpSV passes naturally for COO.
    inner->ptrs = cooRowInd;
    inner->idxs = cooColInd;
    inner->rowInds = cooRowInd;
    inner->colInds = cooColInd;
    inner->values = cooValues;
    inner->baseType = idxBase;
    inner->ptrType = cooIdxType;
    inner->IdxType = cooIdxType;
    inner->valueType = valueType;
    inner->signature = kSpMatSignature;
    *spMatDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstCoo(aclsparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols,
                                          int64_t nnz, const void* cooRowInd, const void* cooColInd,
                                          const void* cooValues, aclsparseIndexType_t cooIdxType,
                                          aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseSpMatDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateCoo(&tmp, rows, cols, nnz, const_cast<void*>(cooRowInd),
                                              const_cast<void*>(cooColInd), const_cast<void*>(cooValues), cooIdxType,
                                              idxBase, valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *spMatDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateSlicedEll(aclsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                           int64_t sliceNnz, int64_t numSlices, void* sellSlicePtr, void* sellColInd,
                                           void* sellValues, aclsparseIndexType_t sellIdxType,
                                           aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    aclsparseStatus_t st = ValidateSpMatCreateParams(spMatDescr, rows, cols, nnz);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (sellIdxType != ACL_SPARSE_INDEX_32I && sellIdxType != ACL_SPARSE_INDEX_64I) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (sliceNnz < 0 || numSlices < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = new (std::nothrow) aclsparseSpMatDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    inner->format = ACL_SPARSE_FORMAT_SLICED_ELL;
    inner->rows = static_cast<uint64_t>(rows);
    inner->cols = static_cast<uint64_t>(cols);
    inner->nnz = static_cast<uint64_t>(nnz);
    inner->sliceNnz = static_cast<uint64_t>(sliceNnz);
    inner->numSlices = numSlices;
    inner->ptrs = sellSlicePtr;
    inner->idxs = sellColInd;
    inner->values = sellValues;
    inner->baseType = idxBase;
    inner->ptrType = sellIdxType;
    inner->IdxType = sellIdxType;
    inner->valueType = valueType;
    inner->signature = kSpMatSignature;
    *spMatDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstSlicedEll(aclsparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols,
                                                int64_t nnz, int64_t sliceNnz, int64_t numSlices,
                                                const void* sellSlicePtr, const void* sellColInd,
                                                const void* sellValues, aclsparseIndexType_t sellIdxType,
                                                aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseSpMatDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateSlicedEll(&tmp, rows, cols, nnz, sliceNnz, numSlices,
                                                    const_cast<void*>(sellSlicePtr), const_cast<void*>(sellColInd),
                                                    const_cast<void*>(sellValues), sellIdxType, idxBase, valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *spMatDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
