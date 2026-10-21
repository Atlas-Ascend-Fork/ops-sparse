/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_SPARSE_SPMV_ARCH35_H_
#define OPS_SPARSE_SPMV_ARCH35_H_

#include <cstddef>
#include <cstdint>

#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "spmv_tiling_data.h"

inline aclsparseContext *SpmvArch35ToHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<aclsparseContext *>(handle);
}

inline aclsparseSpMatDescr *SpmvArch35ToMat(aclsparseConstSpMatDescr_t desc)
{
    return const_cast<aclsparseSpMatDescr *>(
        reinterpret_cast<const aclsparseSpMatDescr *>(desc));
}

inline aclsparseDnVecDescr *SpmvArch35ToVec(aclsparseConstDnVecDescr_t desc)
{
    return const_cast<aclsparseDnVecDescr *>(
        reinterpret_cast<const aclsparseDnVecDescr *>(desc));
}

inline bool SpmvArch35DescriptorsValid(aclsparseConstSpMatDescr_t matDesc,
                                        aclsparseConstDnVecDescr_t xDesc,
                                        aclsparseDnVecDescr_t yDesc)
{
    const auto *mat = reinterpret_cast<const aclsparseSpMatDescr *>(matDesc);
    const auto *x = reinterpret_cast<const aclsparseDnVecDescr *>(xDesc);
    const auto *y = reinterpret_cast<const aclsparseDnVecDescr *>(yDesc);
    return mat != nullptr && x != nullptr && y != nullptr &&
           mat->signature == kSpMatSignature &&
           x->signature == kDnVecSignature &&
           y->signature == kDnVecSignature;
}

inline bool SpmvArch35ActivePlanMatches(const aclsparseSpMatDescr *mat,
                                        const aclsparseDnVecDescr *x,
                                        const aclsparseDnVecDescr *y,
                                        aclsparseOperation_t opA,
                                        aclDataType computeType,
                                        aclsparseSpMVAlg_t alg,
                                        const void *buffer,
                                        const void *stream,
                                        uint64_t streamEpoch)
{
    return mat != nullptr && x != nullptr && y != nullptr && buffer != nullptr &&
           mat->activeSpmvBuffer == buffer && mat->activeBuffer != buffer &&
           mat->activeRowOffsets == mat->ptrs &&
           mat->activeColIndices == mat->idxs &&
           mat->activeValues == mat->values &&
           mat->activeRowOffsetsStride == mat->ptrStride &&
           mat->activeColIndStride == mat->idxStride &&
           mat->activeValuesStride == mat->valueStride &&
           mat->activeXStride == x->stride && mat->activeYStride == y->stride &&
           mat->activeRows == mat->rows && mat->activeCols == mat->cols &&
           mat->activeNnz == mat->nnz && mat->activeBase == mat->baseType &&
           mat->activeValueType == mat->valueType &&
           mat->activeXType == x->valueType && mat->activeYType == y->valueType &&
           mat->activeComputeType == computeType && mat->activeOp == opA &&
           mat->activeAlg == static_cast<int32_t>(alg) &&
           mat->activeStream == stream &&
           mat->activeStreamEpoch == streamEpoch;
}

inline void SpmvArch35RecordActivePlan(aclsparseSpMatDescr *mat,
                                       const aclsparseDnVecDescr *x,
                                       const aclsparseDnVecDescr *y,
                                       aclsparseOperation_t opA,
                                       aclDataType computeType,
                                       aclsparseSpMVAlg_t alg,
                                       const void *buffer,
                                       const void *stream,
                                       uint64_t streamEpoch)
{
    if (mat == nullptr || x == nullptr || y == nullptr) {
        return;
    }
    mat->activeSpmvBuffer = buffer;
    mat->activeRowOffsets = mat->ptrs;
    mat->activeColIndices = mat->idxs;
    mat->activeValues = mat->values;
    mat->activeRowOffsetsStride = mat->ptrStride;
    mat->activeColIndStride = mat->idxStride;
    mat->activeValuesStride = mat->valueStride;
    mat->activeXStride = x->stride;
    mat->activeYStride = y->stride;
    mat->activeRows = mat->rows;
    mat->activeCols = mat->cols;
    mat->activeNnz = mat->nnz;
    mat->activeBase = mat->baseType;
    mat->activeValueType = mat->valueType;
    mat->activeXType = x->valueType;
    mat->activeYType = y->valueType;
    mat->activeComputeType = computeType;
    mat->activeOp = opA;
    mat->activeAlg = static_cast<int32_t>(alg);
    mat->activeStream = stream;
    mat->activeStreamEpoch = streamEpoch;
}

#endif  // OPS_SPARSE_SPMV_ARCH35_H_
