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

#ifndef OP_HOST_SPSV_PARAMETER_VALIDATION_H_
#define OP_HOST_SPSV_PARAMETER_VALIDATION_H_

namespace
{
static aclsparseStatus_t ValidateSpSVIndexTypes(const char* tag, const aclsparseSpMatDescr* mat)
{
    if (mat->baseType != ACL_SPARSE_INDEX_BASE_ZERO && mat->baseType != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(tag, "unsupported baseType=%d, only ZERO or ONE supported", mat->baseType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (mat->format != ACL_SPARSE_FORMAT_CSR && mat->format != ACL_SPARSE_FORMAT_CSC &&
        mat->format != ACL_SPARSE_FORMAT_COO && mat->format != ACL_SPARSE_FORMAT_SLICED_ELL) {
        OP_LOGE(tag, "unsupported format=%d", mat->format);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // Validate individual index types (ptrType and IdxType independently).
    if (mat->ptrType != ACL_SPARSE_INDEX_32I && mat->ptrType != ACL_SPARSE_INDEX_64I) {
        OP_LOGE(tag, "unsupported ptrType=%d", mat->ptrType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (mat->IdxType != ACL_SPARSE_INDEX_32I && mat->IdxType != ACL_SPARSE_INDEX_64I) {
        OP_LOGE(tag, "unsupported idxType=%d", mat->IdxType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // Reject (I32, I64): ptrType is I32 cannot address nnz > INT32_MAX;
    // colInd being I64 in this combo is unsupported (no practical use case).
    // Allowed combinations: (I32, I32), (I64, I32), (I64, I64).
    if (mat->ptrType == ACL_SPARSE_INDEX_32I && mat->IdxType == ACL_SPARSE_INDEX_64I) {
        OP_LOGE(tag, "unsupported index combination: ptrType=I32, idxType=I64");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Debug override: when SPSV_FORCE_PERMT_64=1 is set, force int64 workspace
// arrays (perm/transPerm/diagPtr) regardless of nnz. Used for validating
// the <IdxT, int64_t> kernel template instantiation on small matrices.
static bool ShouldForcePermType64()
{
    const char* env = std::getenv("SPSV_FORCE_PERMT_64");
    return env && env[0] == '1' && env[1] == '\0';
}

static aclsparseStatus_t ValidateSpSVCommonParams(const char* tag, aclsparseConstSpMatDescr_t matA,
                                                  aclsparseSpSVDescr_t spsvDescr, aclDataType computeType,
                                                  aclsparseSpSVAlg_t alg)
{
    if (matA == nullptr) {
        OP_LOGE(tag, "matA is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (spsvDescr == nullptr) {
        OP_LOGE(tag, "spsvDescr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != ACL_FLOAT && computeType != ACL_COMPLEX64) {
        OP_LOGE(tag, "unsupported computeType=%d, expected ACL_FLOAT or ACL_COMPLEX64", computeType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (alg != ACL_SPARSE_SPSV_ALG_DEFAULT) {
        OP_LOGE(tag, "unsupported alg=%d", alg);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    auto* mat = spsv::ToMatInner(matA);
    if (mat->valueType != computeType) {
        OP_LOGE(tag, "matrix valueType=%d does not match computeType=%d", mat->valueType, computeType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // The complex kernel instantiates I32 row/column indices and workspace
    // permutations only. Reject the same unsupported input at every stage.
    if (computeType == ACL_COMPLEX64 &&
        (mat->ptrType != ACL_SPARSE_INDEX_32I || mat->IdxType != ACL_SPARSE_INDEX_32I ||
         mat->nnz > static_cast<uint64_t>(INT32_MAX) || ShouldForcePermType64())) {
        OP_LOGE(tag, "complex64 requires I32 row/col indices and 32-bit workspace permutations");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (mat->rows > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(tag, "matrix rows=%" PRIu64 " exceeds INT32_MAX (kernel uses int32_t internally)", mat->rows);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // nnz > INT32_MAX requires I64 index type (ptrType=ACL_SPARSE_INDEX_64I).
    // I32 + nnz > INT32_MAX is explicitly rejected below.
    if (mat->ptrType == ACL_SPARSE_INDEX_32I && mat->nnz > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(tag, "I32 index type cannot represent nnz=%" PRIu64 " > INT32_MAX", mat->nnz);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (mat->rows != mat->cols) {
        OP_LOGE(tag, "matrix must be square, rows=%" PRIu64 " cols=%" PRIu64, mat->rows, mat->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ValidateSpSVIndexTypes(tag, mat);
}

static aclsparseStatus_t ValidateOpA(const char* tag, aclsparseOperation_t opA)
{
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE && opA != ACL_SPARSE_OP_TRANSPOSE &&
        opA != ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) {
        OP_LOGE(tag, "invalid opA=%d", opA);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static bool NeedsTranspose(aclsparseOperation_t opA) { return opA != ACL_SPARSE_OP_NON_TRANSPOSE; }


} // namespace

#endif // OP_HOST_SPSV_PARAMETER_VALIDATION_H_
