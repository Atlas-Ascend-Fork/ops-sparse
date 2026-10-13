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
 * \file spgemm_validate.cpp
 * \brief SpGEMM 入参校验与标量读取。
 */

#include "spgemm_common.h"

#include <acl/acl.h>

namespace {

// opA/opB 仅支持 NON_TRANSPOSE。
aclsparseStatus_t ValidateOperations(aclsparseOperation_t opA, aclsparseOperation_t opB)
{
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE || opB != ACL_SPARSE_OP_NON_TRANSPOSE) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 三矩阵均须为 CSR。COO/CSC/BSR 等一律拒绝。
aclsparseStatus_t ValidateFormats(const aclsparseSpMatDescr *matA,
                                 const aclsparseSpMatDescr *matB,
                                 const aclsparseSpMatDescr *matC)
{
    if (matA->format != ACL_SPARSE_FORMAT_CSR ||
        matB->format != ACL_SPARSE_FORMAT_CSR ||
        matC->format != ACL_SPARSE_FORMAT_CSR) {
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 索引：rowOffsets 与 colInd 均须为 ACL_SPARSE_INDEX_32I 且 zero-based。
aclsparseStatus_t ValidateIndices(const aclsparseSpMatDescr *mat)
{
    aclsparseStatus_t st = AclsparseValidateSupportedCsrIndexTypes(mat->ptrType, mat->IdxType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (mat->baseType != ACL_SPARSE_INDEX_BASE_ZERO) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// dtype：A/B/C 的 valueType 与 computeType 四者必须一致，且落在支持集内。
aclsparseStatus_t ValidateDtypes(const aclsparseSpMatDescr *matA,
                                const aclsparseSpMatDescr *matB,
                                const aclsparseSpMatDescr *matC,
                                aclDataType computeType)
{
    uint32_t dtypeId = 0;
    if (!SpgemmDtypeFromAcl(computeType, dtypeId)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matA->valueType != computeType || matB->valueType != computeType ||
        matC->valueType != computeType) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 维度：A[M,K] × B[K,N] → C[M,N]，要求 A.cols == B.rows 且 C 的维度与 M/N 匹配。
// 不支持广播。
aclsparseStatus_t ValidateDimensions(const aclsparseSpMatDescr *matA,
                                    const aclsparseSpMatDescr *matB,
                                    const aclsparseSpMatDescr *matC)
{
    if (matA->cols != matB->rows) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matC->rows != matA->rows || matC->cols != matB->cols) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 索引受 int32 约束：维度与 nnz 均不得超过 INT32_MAX。
    constexpr uint64_t kInt32Max = 2147483647ULL;
    if (matA->rows > kInt32Max || matA->cols > kInt32Max || matB->cols > kInt32Max ||
        matA->nnz > kInt32Max || matB->nnz > kInt32Max || matC->nnz > kInt32Max) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateAlg(aclsparseSpGEMMAlg_t alg)
{
    switch (alg) {
        case ACL_SPARSE_SPGEMM_DEFAULT:
        case ACL_SPARSE_SPGEMM_ALG1:
        case ACL_SPARSE_SPGEMM_ALG2:
        case ACL_SPARSE_SPGEMM_ALG3:
            return ACL_SPARSE_STATUS_SUCCESS;
        default:
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
}

}  // namespace

aclsparseStatus_t ValidateSpgemmInputs(
    const aclsparseSpMatDescr *matA, const aclsparseSpMatDescr *matB,
    const aclsparseSpMatDescr *matC, aclsparseOperation_t opA,
    aclsparseOperation_t opB, aclDataType computeType, aclsparseSpGEMMAlg_t alg)
{
    if (matA == nullptr || matB == nullptr || matC == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = ValidateOperations(opA, opB);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateFormats(matA, matB, matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateIndices(matA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateIndices(matB);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateIndices(matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateDtypes(matA, matB, matC, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateDimensions(matA, matB, matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return ValidateAlg(alg);
}

aclsparseStatus_t ValidateSpgemmStageConsistency(
    const aclsparseSpGEMMDescr *descr, const aclsparseSpMatDescr *matA,
    const aclsparseSpMatDescr *matB, aclDataType computeType,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclsparseSpGEMMAlg_t alg)
{
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (descr->m != static_cast<uint64_t>(matA->rows) ||
        descr->k != static_cast<uint64_t>(matA->cols) ||
        descr->n != static_cast<uint64_t>(matB->cols) ||
        descr->nnzA != static_cast<uint64_t>(matA->nnz) ||
        descr->nnzB != static_cast<uint64_t>(matB->nnz)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (descr->computeType != computeType || descr->alg != alg) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static inline float SpgemmBf16RawToF32(uint16_t raw)
{
    uint32_t bits = static_cast<uint32_t>(raw) << 16;
    float out = 0.0f;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline float SpgemmFp16RawToF32(uint16_t raw)
{
    const uint32_t sign = static_cast<uint32_t>(raw & 0x8000U) << 16;
    const uint32_t exp = (raw >> 10) & 0x1FU;
    const uint32_t mant = raw & 0x3FFU;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant != 0) {
            uint32_t expNorm = 127 - 15 + 1;
            uint32_t mantNorm = mant;
            while ((mantNorm & 0x400U) == 0) {
                mantNorm <<= 1;
                expNorm--;
            }
            mantNorm &= 0x3FFU;
            bits = sign | (expNorm << 23) | (mantNorm << 13);
        } else {
            bits = sign;
        }
    } else if (exp == 0x1FU) {
        bits = sign | 0x7F800000U | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out = 0.0f;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

aclsparseStatus_t SpgemmReadScalar(const void *p, aclsparsePointerMode_t mode,
                                   uint32_t dtypeId, float &re, float &im)
{
    re = 0.0f;
    im = 0.0f;
    if (p == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const bool isComplex = SpgemmDtypeIsComplex(dtypeId);
    const size_t nComp = isComplex ? 2 : 1;

    // fp16/bf16 标量按 computeType 存储，需先按原类型读出再提升到 fp32。
    if (dtypeId == SPGEMM_DTYPE_FP16 || dtypeId == SPGEMM_DTYPE_BF16) {
        uint16_t raw = 0;
        if (mode == ACL_SPARSE_POINTER_MODE_DEVICE) {
            if (aclrtMemcpy(&raw, sizeof(raw), p, sizeof(raw),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
                return ACL_SPARSE_STATUS_EXECUTION_FAILED;
            }
        } else {
            raw = *static_cast<const uint16_t *>(p);
        }
        re = (dtypeId == SPGEMM_DTYPE_BF16) ? SpgemmBf16RawToF32(raw)
                                              : SpgemmFp16RawToF32(raw);
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    float comp[2] = {0.0f, 0.0f};
    if (mode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        const size_t bytes = nComp * sizeof(float);
        if (aclrtMemcpy(comp, bytes, p, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    } else {
        const float *src = static_cast<const float *>(p);
        comp[0] = src[0];
        if (isComplex) {
            comp[1] = src[1];
        }
    }
    re = comp[0];
    im = comp[1];
    return ACL_SPARSE_STATUS_SUCCESS;
}
