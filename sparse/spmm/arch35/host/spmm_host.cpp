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
#include <stdint.h>
#include <stdlib.h>
#include <new>
#include <mutex>
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <utility>
#include <unordered_map>
#include <vector>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "spmm_host.h"
#include "../kernel/spmm_kernel.h"
#include "spmm_csr_mat.h"
#include "spmm_host_builders.h"
#include "spmm_host_internal.h"

#ifndef __CCE_AICORE__
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "kernel_tiling/kernel_tiling.h"
#endif

namespace {

using spmm_host_internal::CsrHostData;
using spmm_host_internal::ComplexRegBaseMinAverageRows;
using spmm_host_internal::ComplexRegBaseRows;
using spmm_host_internal::ComplexRegBaseWindowRows;
using spmm_host_internal::IsComplexRegBaseDegree;
using spmm_host_internal::IsOfficialComplexP03Shape;
using spmm_host_internal::IsValidCsrRowRange;
using spmm_host_internal::LoadCsrHostData;
using spmm_host_internal::RegBaseWorkspaceFits;
using spmm_host_internal::ResetCubeSelection;
using spmm_host_builders::BuildComplexCubePrepackedData;
using spmm_host_builders::BuildComplexRegBaseData;
using spmm_host_builders::BuildCubePrepackedData;
using spmm_host_builders::BuildFp32Degree2Data;
using spmm_host_builders::BuildFp32Degree4Data;
using spmm_host_builders::BuildFp32PackedData;
using spmm_host_builders::BuildFp32RegBaseData;

struct SpmmWorkspaceIdentity {
    uint64_t ownerId = 0;
    uintptr_t matAPtrs = 0;
    uintptr_t matAIdxs = 0;
    uintptr_t matAValues = 0;
    uintptr_t matBValues = 0;
    uintptr_t matCValues = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    int64_t bRows = 0;
    int64_t bCols = 0;
    int64_t bLd = 0;
    int64_t cRows = 0;
    int64_t cCols = 0;
    int64_t cLd = 0;
    int64_t ptrStride = 1;
    int64_t idxStride = 1;
    int64_t valueStride = 1;
    aclsparseOrder_t bOrder = ACL_SPARSE_ORDER_ROW;
    aclsparseOrder_t cOrder = ACL_SPARSE_ORDER_ROW;
    aclDataType matAType = ACL_FLOAT;
    aclDataType matBType = ACL_FLOAT;
    aclDataType matCType = ACL_FLOAT;
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE;
    aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE;
    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseSpMMAlg_t alg = ACL_SPARSE_SPMM_ALG_DEFAULT;
    aclDataType computeType = ACL_FLOAT;
    uintptr_t stream = 0;
    uintptr_t alpha = 0;
    uintptr_t beta = 0;
    uint64_t alphaBits[2] = {0, 0};
    uint64_t betaBits[2] = {0, 0};
};

std::mutex &SpmmWorkspaceRegistryMutex()
{
    static std::mutex mutex;
    return mutex;
}

spmm_host_internal::WorkspaceOwnershipRegistry<SpmmWorkspaceIdentity> &SpmmWorkspaceRegistry()
{
    static spmm_host_internal::WorkspaceOwnershipRegistry<SpmmWorkspaceIdentity> registry;
    return registry;
}

template <typename T>
void CopyScalarBits(const void *value, uint64_t bits[2])
{
    static_assert(sizeof(T) <= sizeof(bits[0]) * 2, "scalar does not fit workspace identity");
    if (value != nullptr) {
        std::copy_n(static_cast<const unsigned char *>(value), sizeof(T),
                    reinterpret_cast<unsigned char *>(bits));
    }
}

SpmmWorkspaceIdentity MakeSpmmWorkspaceIdentity(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, aclsparsePointerMode_t pointerMode,
    aclrtStream stream, const void *buffer,
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE)
{
    SpmmWorkspaceIdentity identity{};
    identity.ownerId = matA->spmmWorkspaceOwnerId;
    identity.matAPtrs = reinterpret_cast<uintptr_t>(matA->ptrs);
    identity.matAIdxs = reinterpret_cast<uintptr_t>(matA->idxs);
    identity.matAValues = reinterpret_cast<uintptr_t>(matA->values);
    identity.matBValues = reinterpret_cast<uintptr_t>(matB->values);
    identity.matCValues = reinterpret_cast<uintptr_t>(matC->values);
    identity.rows = static_cast<int64_t>(matA->rows);
    identity.cols = static_cast<int64_t>(matA->cols);
    identity.nnz = static_cast<int64_t>(matA->nnz);
    identity.bRows = matB->rows;
    identity.bCols = matB->cols;
    identity.bLd = matB->ld;
    identity.cRows = matC->rows;
    identity.cCols = matC->cols;
    identity.cLd = matC->ld;
    identity.ptrStride = matA->ptrStride;
    identity.idxStride = matA->idxStride;
    identity.valueStride = matA->valueStride;
    identity.bOrder = matB->order;
    identity.cOrder = matC->order;
    identity.matAType = matA->valueType;
    identity.matBType = matB->valueType;
    identity.matCType = matC->valueType;
    identity.opA = opA;
    identity.opB = opB;
    identity.pointerMode = pointerMode;
    identity.alg = alg;
    identity.computeType = computeType;
    identity.stream = reinterpret_cast<uintptr_t>(stream);
    if (pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        identity.alpha = reinterpret_cast<uintptr_t>(alpha);
        identity.beta = reinterpret_cast<uintptr_t>(beta);
    } else if (computeType == ACL_COMPLEX64) {
        CopyScalarBits<aclsparseComplex>(alpha, identity.alphaBits);
        CopyScalarBits<aclsparseComplex>(beta, identity.betaBits);
    } else {
        CopyScalarBits<float>(alpha, identity.alphaBits);
        CopyScalarBits<float>(beta, identity.betaBits);
    }
    (void)buffer;
    return identity;
}

bool SameSpmmWorkspaceIdentity(const SpmmWorkspaceIdentity &lhs,
                               const SpmmWorkspaceIdentity &rhs)
{
    return lhs.ownerId == rhs.ownerId && lhs.matAPtrs == rhs.matAPtrs &&
        lhs.matAIdxs == rhs.matAIdxs && lhs.matAValues == rhs.matAValues &&
        lhs.matBValues == rhs.matBValues && lhs.matCValues == rhs.matCValues &&
        lhs.rows == rhs.rows && lhs.cols == rhs.cols && lhs.nnz == rhs.nnz &&
        lhs.bRows == rhs.bRows && lhs.bCols == rhs.bCols && lhs.bLd == rhs.bLd &&
        lhs.cRows == rhs.cRows && lhs.cCols == rhs.cCols && lhs.cLd == rhs.cLd &&
        lhs.ptrStride == rhs.ptrStride && lhs.idxStride == rhs.idxStride &&
        lhs.valueStride == rhs.valueStride && lhs.bOrder == rhs.bOrder &&
        lhs.cOrder == rhs.cOrder && lhs.matAType == rhs.matAType &&
        lhs.matBType == rhs.matBType && lhs.matCType == rhs.matCType &&
        lhs.opA == rhs.opA && lhs.opB == rhs.opB && lhs.pointerMode == rhs.pointerMode &&
        lhs.alg == rhs.alg && lhs.computeType == rhs.computeType &&
        lhs.stream == rhs.stream && lhs.alpha == rhs.alpha && lhs.beta == rhs.beta &&
        std::memcmp(lhs.alphaBits, rhs.alphaBits, sizeof(lhs.alphaBits)) == 0 &&
        std::memcmp(lhs.betaBits, rhs.betaBits, sizeof(lhs.betaBits)) == 0;
}

void InvalidateSpmmWorkspace(const void *buffer)
{
    std::lock_guard<std::mutex> lock(SpmmWorkspaceRegistryMutex());
    SpmmWorkspaceRegistry().Erase(buffer);
}

bool IsOwnedSpmmWorkspace(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, aclsparsePointerMode_t pointerMode,
    aclrtStream stream, const void *buffer,
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE)
{
    if (matA->activeBuffer != buffer || matA->activeSpmvBuffer == buffer) {
        return false;
    }
    const SpmmWorkspaceIdentity expected = MakeSpmmWorkspaceIdentity(
        matA, matB, matC, opB, alpha, beta, computeType, alg, pointerMode, stream, buffer, opA);
    std::lock_guard<std::mutex> lock(SpmmWorkspaceRegistryMutex());
    const auto *found = SpmmWorkspaceRegistry().Find(buffer);
    return found != nullptr && SameSpmmWorkspaceIdentity(*found, expected);
}

void PublishSpmmWorkspace(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, aclsparsePointerMode_t pointerMode,
    aclrtStream stream, const void *buffer,
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE)
{
    const SpmmWorkspaceIdentity identity = MakeSpmmWorkspaceIdentity(
        matA, matB, matC, opB, alpha, beta, computeType, alg, pointerMode, stream, buffer, opA);
    std::lock_guard<std::mutex> lock(SpmmWorkspaceRegistryMutex());
    SpmmWorkspaceRegistry().Publish(buffer, identity);
}

// Outer launch blockDim comes from the CANN platform configuration. Keep the
// original Arch35 fallback for environments where that query is unavailable;
// a platform-provided count remains authoritative.
constexpr uint32_t kSpmmBlockDimFallback = 24u;

uint32_t GetSpmmBlockDim()
{
    static uint32_t cached = 0u;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&]() {
#ifndef __CCE_AICORE__
        auto *plat = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (plat != nullptr) {
            const uint32_t aiv = plat->GetCoreNumAiv();
            if (aiv > 0u) {
                cached = aiv;
                return;
            }
        }
#endif
        cached = kSpmmBlockDimFallback;
    });
    return cached;
}

uint32_t GetSpmmCubeBlockDim()
{
    static uint32_t cached = 0u;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&]() {
#ifndef __CCE_AICORE__
        auto *plat = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (plat != nullptr && plat->GetCoreNumAic() > 0u) {
            cached = plat->GetCoreNumAic();
            return;
        }
#endif
        // Preserve the fallback used by the original guarded Cube path when
        // no platform configuration is available; do not retune it here.
        cached = 28u;
    });
    return cached;
}

uint64_t GetSpmmLibApiWorkspaceSize()
{
    static uint64_t cached = 0u;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&]() {
#ifndef __CCE_AICORE__
        auto *plat = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (plat != nullptr) {
            cached = plat->GetLibApiWorkSpaceSize();
        }
#endif
    });
    return cached;
}

float ScalarReadF32(const void *p) {
    if (p == nullptr) {
        return 0.0f;
    }
    return *static_cast<const float *>(p);
}

int32_t ScalarReadI32(const void *p) {
    if (p == nullptr) {
        return 0;
    }
    return *static_cast<const int32_t *>(p);
}

aclsparseComplex ScalarReadComplex64(const void *p)
{
    if (p == nullptr) {
        return aclsparseComplex{0.0f, 0.0f};
    }
    return *static_cast<const aclsparseComplex *>(p);
}

bool IsHostScalarZero(const void *scalar, aclDataType computeType)
{
    if (computeType == ACL_COMPLEX64) {
        const aclsparseComplex value = ScalarReadComplex64(scalar);
        return value.x == 0.0f && value.y == 0.0f;
    }
    if (computeType == ACL_INT32) {
        return ScalarReadI32(scalar) == 0;
    }
    return ScalarReadF32(scalar) == 0.0f;
}

bool IsHostScalarValue(const void *scalar, aclDataType computeType, float expected)
{
    return computeType == ACL_FLOAT && ScalarReadF32(scalar) == expected;
}

bool IsHostComplexScalarValue(const void *scalar, float expectedReal,
                              float expectedImag)
{
    const aclsparseComplex value = ScalarReadComplex64(scalar);
    return value.x == expectedReal && value.y == expectedImag;
}

void SetTilingAlphaBeta(SpmmHostTilingData *td,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparsePointerMode_t pointerMode)
{
    td->pointer_mode = (pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) ? 1 : 0;
    td->alpha_ptr = 0;
    td->beta_ptr = 0;
    td->alpha_host = 0.0f;
    td->beta_host = 0.0f;
    td->alpha_imag = 0.0f;
    td->beta_imag = 0.0f;
    td->alpha_int = 0;
    td->beta_int = 0;
    if (pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        td->alpha_ptr = reinterpret_cast<uint64_t>(alpha);
        td->beta_ptr = reinterpret_cast<uint64_t>(beta);
    } else if (computeType == ACL_COMPLEX64) {
        const aclsparseComplex alphaValue = ScalarReadComplex64(alpha);
        const aclsparseComplex betaValue = ScalarReadComplex64(beta);
        td->alpha_host = alphaValue.x;
        td->alpha_imag = alphaValue.y;
        td->beta_host = betaValue.x;
        td->beta_imag = betaValue.y;
    } else if (computeType == ACL_INT32) {
        // Keep integer scalars in their native representation. A float staging
        // field cannot represent all int32 values exactly above 2^24.
        td->alpha_int = ScalarReadI32(alpha);
        td->beta_int = ScalarReadI32(beta);
    } else {
        td->alpha_host = ScalarReadF32(alpha);
        td->beta_host = ScalarReadF32(beta);
    }
}

int32_t ComputeSpmmHighPrecision(aclsparseSpMMAlg_t alg, int32_t dataType)
{
    return (alg == ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG &&
            dataType == SPMM_DTYPE_FP32) ?
               1 :
               0;
}

// arch35 支持的 dtype 组合。
static const AclsparseSpmmDtypeCombo kSpmmDtypeCombos[] = {
    {ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT},
    {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT},
    {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT},
    {ACL_BF16, ACL_BF16, ACL_BF16, ACL_FLOAT},
    {ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_FLOAT},
    {ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64},
    {ACL_INT8, ACL_INT8, ACL_INT32, ACL_INT32},
};

static bool IsSupportedSpmmDtypeCombo(const aclsparseSpMatDescr *matA,
                                      const aclsparseDnMatDescr *matB,
                                      const aclsparseDnMatDescr *matC,
                                      aclDataType computeType)
{
    return AclsparseMatchSpmmDtypeCombo(
        matA, matB, matC, computeType, kSpmmDtypeCombos,
        sizeof(kSpmmDtypeCombos) / sizeof(kSpmmDtypeCombos[0]));
}

static bool SpmmDimensionsMatch(const aclsparseSpMatDescr *matA,
                                const aclsparseDnMatDescr *matB,
                                const aclsparseDnMatDescr *matC,
                                aclsparseOperation_t opA, aclsparseOperation_t opB)
{
    const uint64_t m = opA == ACL_SPARSE_OP_NON_TRANSPOSE ? matA->rows : matA->cols;
    const uint64_t k = opA == ACL_SPARSE_OP_NON_TRANSPOSE ? matA->cols : matA->rows;
    const bool outputMatches = m == static_cast<uint64_t>(matC->rows);
    if (!outputMatches) {
        return false;
    }
    if (opB == ACL_SPARSE_OP_NON_TRANSPOSE) {
        return k == static_cast<uint64_t>(matB->rows) &&
               matB->cols == matC->cols;
    }
    return k == static_cast<uint64_t>(matB->cols) &&
           matB->rows == matC->cols;
}

static aclsparseStatus_t ValidateSpmmDataPointers(
    const aclsparseSpMatDescr *matA,
    const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC)
{
    if (matA->rows > 0 && matA->ptrs == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->nnz > 0 && (matA->idxs == nullptr || matA->values == nullptr)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matB->rows > 0 && matB->cols > 0 && matB->values == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matC->rows > 0 && matC->cols > 0 && matC->values == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const int64_t bRequiredLd =
        (matB->order == ACL_SPARSE_ORDER_ROW) ? matB->cols : matB->rows;
    const int64_t cRequiredLd =
        (matC->order == ACL_SPARSE_ORDER_ROW) ? matC->cols : matC->rows;
    if (matB->ld < bRequiredLd || matC->ld < cRequiredLd) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static bool IsSupportedSpmmAlg(aclsparseSpMMAlg_t alg)
{
    return alg == ACL_SPARSE_SPMM_ALG_DEFAULT ||
           alg == ACL_SPARSE_SPMM_CSR_ALG1 ||
           alg == ACL_SPARSE_SPMM_CSR_ALG2 ||
           alg == ACL_SPARSE_SPMM_CSR_ALG3 ||
           alg == ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG;
}

static aclsparseStatus_t ValidateSpmmOperations(aclsparseOperation_t opA,
                                                aclsparseOperation_t opB,
                                                aclsparseSpMMAlg_t alg)
{
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE && opA != ACL_SPARSE_OP_TRANSPOSE &&
        opA != ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (opB != ACL_SPARSE_OP_NON_TRANSPOSE && opB != ACL_SPARSE_OP_TRANSPOSE &&
        opB != ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (alg == ACL_SPARSE_SPMM_CSR_ALG3 &&
        (opA != ACL_SPARSE_OP_NON_TRANSPOSE ||
         opB == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmmDescriptorBasics(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, aclsparseOperation_t opA,
    aclsparseOperation_t opB, aclsparseSpMMAlg_t alg)
{
    if (matA == nullptr || matB == nullptr || matC == nullptr) {
        OP_LOGE("aclsparseSpMM", "matA, matB, or matC is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!IsSupportedSpmmAlg(alg)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    aclsparseStatus_t st = ValidateSpmmOperations(opA, opB, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpMM", "unsupported operation: opA=%d, opB=%d, status=%d",
                static_cast<int>(opA), static_cast<int>(opB), static_cast<int>(st));
        return st;
    }
    if (matA->format != ACL_SPARSE_FORMAT_CSR) {
        OP_LOGE("aclsparseSpMM", "unsupported sparse matrix format %d (CSR required)",
                static_cast<int>(matA->format));
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    aclsparseStatus_t idxSt = AclsparseValidateSupportedCsrIndexTypes(matA->ptrType, matA->IdxType);
    if (idxSt != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpMM", "unsupported CSR index types: ptrType=%d, idxType=%d, status=%d",
                static_cast<int>(matA->ptrType), static_cast<int>(matA->IdxType), static_cast<int>(idxSt));
        return idxSt;
    }
    if (matA->baseType != ACL_SPARSE_INDEX_BASE_ZERO &&
        matA->baseType != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE("aclsparseSpMM", "unsupported index base %d (ZERO or ONE required)",
                static_cast<int>(matA->baseType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matA->nnz > static_cast<uint64_t>(INT32_MAX)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmmInt32Dimensions(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC)
{
    if (matA->rows > static_cast<uint64_t>(INT32_MAX) ||
        matA->cols > static_cast<uint64_t>(INT32_MAX) ||
        matB->rows > static_cast<int64_t>(INT32_MAX) ||
        matB->cols > static_cast<int64_t>(INT32_MAX) ||
        matB->ld > static_cast<int64_t>(INT32_MAX) ||
        matC->rows > static_cast<int64_t>(INT32_MAX) ||
        matC->cols > static_cast<int64_t>(INT32_MAX) ||
        matC->ld > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE("aclsparseSpMM",
                "matrix dimension or leading dimension exceeds INT32_MAX");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmmInputs(const aclsparseSpMatDescr *matA,
                                   const aclsparseDnMatDescr *matB,
                                   const aclsparseDnMatDescr *matC,
                                   aclsparseOperation_t opA,
                                   aclsparseOperation_t opB,
                                   aclDataType computeType,
                                   aclsparseSpMMAlg_t alg)
{
    aclsparseStatus_t status = ValidateSpmmDescriptorBasics(
        matA, matB, matC, opA, opB, alg);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (!IsSupportedSpmmDtypeCombo(matA, matB, matC, computeType)) {
        OP_LOGE("aclsparseSpMM",
                "unsupported data type combination: matA=%d, matB=%d, matC=%d, computeType=%d",
                static_cast<int>(matA->valueType), static_cast<int>(matB->valueType),
                static_cast<int>(matC->valueType), static_cast<int>(computeType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!SpmmDimensionsMatch(matA, matB, matC, opA, opB)) {
        OP_LOGE("aclsparseSpMM",
                "SpMM matrix dimensions do not match: matA=(%llu, %llu), matB=(%lld, %lld), matC=(%lld, %lld)",
                static_cast<unsigned long long>(matA->rows), static_cast<unsigned long long>(matA->cols),
                static_cast<long long>(matB->rows), static_cast<long long>(matB->cols),
                static_cast<long long>(matC->rows), static_cast<long long>(matC->cols));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->rows == 0 || matA->cols == 0 || matC->cols == 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    status = ValidateSpmmInt32Dimensions(matA, matB, matC);
    return status == ACL_SPARSE_STATUS_SUCCESS ?
        ValidateSpmmDataPointers(matA, matB, matC) : status;
}

using spmm_host_internal::WsOffsets;

bool WsOffsetsFitTiling(const WsOffsets &offsets)
{
    const int64_t maxTilingOffset = std::numeric_limits<int32_t>::max();
    return offsets.tilingOff >= 0 && offsets.tilingOff <= maxTilingOffset &&
        offsets.reorderOff >= 0 && offsets.reorderOff <= maxTilingOffset &&
        offsets.binEdgeOff >= 0 && offsets.binEdgeOff <= maxTilingOffset;
}

// The Cube path is deliberately narrow. It is only selected after preprocess
// has proved that every 32-row group fits the packed-column representation;
// all other layouts, dtypes and CSR patterns retain the SIMT implementation.
constexpr bool kEnableArch35Cube = true;
// Keep the high/low Complex64 path closed until the integrated active-buffer
// refresh, accuracy, fallback and performance gates pass on Ascend 950PR.
constexpr bool kEnableArch35ComplexCube = true;
// Column-local grouping can close a group before it reaches 16 rows. Reserve
// bounded headroom for an average of 14 rows; more fragmented inputs retain
// the generic SIMT path during preprocess.
constexpr int32_t kComplexCubeReserveRowsPerGroup =
    spmm_host_internal::kComplexCubeReserveRowsPerGroup;

bool IsCubeInputDtype(aclDataType valueType, aclDataType outputType,
                      aclDataType computeType)
{
    return (valueType == ACL_FLOAT16 || valueType == ACL_BF16) &&
        outputType == valueType && computeType == ACL_FLOAT;
}

bool IsFp32CubeInputDtype(aclDataType valueType, aclDataType outputType,
                          aclDataType computeType)
{
    return valueType == ACL_FLOAT && outputType == ACL_FLOAT &&
        computeType == ACL_FLOAT;
}

bool IsComplexCubeInputDtype(aclDataType valueType, aclDataType outputType,
                             aclDataType computeType)
{
    return valueType == ACL_COMPLEX64 && outputType == ACL_COMPLEX64 &&
        computeType == ACL_COMPLEX64;
}

bool ShouldReserveComplexRegBaseWorkspace(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    aclDataType computeType)
{
    return kEnableArch35Cube &&
        IsComplexCubeInputDtype(
            matA->valueType, matC->valueType, computeType) &&
        opB == ACL_SPARSE_OP_NON_TRANSPOSE &&
        matB->order == ACL_SPARSE_ORDER_ROW &&
        matC->order == ACL_SPARSE_ORDER_ROW &&
        matB->ld == matC->cols && matC->ld == matC->cols &&
        matC->cols >= SPMM_COMPLEX64_REGBASE_MIN_N &&
        matA->rows >= SPMM_COMPLEX64_REGBASE_MIN_ROWS &&
        matA->nnz > 0 && matA->nnz % matA->rows == 0 &&
        (matA->nnz / matA->rows == SPMM_COMPLEX64_DEGREE2_DEGREE ||
         matA->nnz / matA->rows == SPMM_COMPLEX64_DEGREE4_DEGREE ||
         matA->nnz / matA->rows == SPMM_COMPLEX64_DEGREE8_DEGREE ||
         matA->nnz / matA->rows == SPMM_COMPLEX64_DEGREE64_DEGREE);
}

bool ShouldReserveCubeWorkspace(const aclsparseSpMatDescr *matA,
                                const aclsparseDnMatDescr *matB,
                                const aclsparseDnMatDescr *matC,
                                aclsparseOperation_t opB,
                                aclDataType computeType)
{
    if (ShouldReserveComplexRegBaseWorkspace(
            matA, matB, matC, opB, computeType)) {
        return true;
    }
    if (!kEnableArch35Cube || opB != ACL_SPARSE_OP_NON_TRANSPOSE ||
        matB->order != ACL_SPARSE_ORDER_ROW ||
        matC->order != ACL_SPARSE_ORDER_ROW ||
        matB->ld != matC->cols || matC->ld != matC->cols ||
        matC->cols <= 0 || matA->rows == 0 || matA->nnz == 0) {
        return false;
    }
    if (IsFp32CubeInputDtype(
            matA->valueType, matC->valueType, computeType)) {
        return true;
    }
    if (kEnableArch35ComplexCube && IsComplexCubeInputDtype(
            matA->valueType, matC->valueType, computeType) &&
        matA->rows >= SPMM_CUBE_MIN_ROWS &&
        matA->cols >= SPMM_CUBE_MAX_K) {
        return true;
    }
    return IsCubeInputDtype(
               matA->valueType, matC->valueType, computeType) &&
        matC->cols <= SPMM_CUBE_MAX_N && (matC->cols % 16) == 0;
}

bool ShouldReserveFp32CubeWorkspace(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    aclDataType computeType)
{
    return ShouldReserveCubeWorkspace(
               matA, matB, matC, opB, computeType) &&
        IsFp32CubeInputDtype(
            matA->valueType, matC->valueType, computeType) &&
        matC->cols <= SPMM_FP32_CUBE_MAX_N &&
        (matC->cols % 16) == 0;
}

__int128 AlignWorkspace(__int128 value)
{
    return ((value + SPMM_WS_ALIGN - 1) / SPMM_WS_ALIGN) * SPMM_WS_ALIGN;
}

WsOffsets InitializeWsOffsets(int64_t m, int32_t blockDim)
{
    WsOffsets o;
    o.tilingOff = SPMM_WS_HEADER_BYTES;
    o.reorderOff = spmm_align_up<SPMM_WS_ALIGN>(
        o.tilingOff + static_cast<int64_t>(sizeof(SpmmTilingWireV1)));
    o.binEdgeOff = spmm_align_up<SPMM_WS_ALIGN>(
        o.reorderOff + static_cast<int64_t>(sizeof(int32_t)) * m);
    o.cubeMetaOff = spmm_align_up<SPMM_WS_ALIGN>(
        o.binEdgeOff + static_cast<int64_t>(sizeof(int32_t)) * (blockDim + 1));
    o.cubeRowIdsOff = o.cubeMetaOff;
    o.cubeSlot1Off = o.cubeMetaOff;
    o.cubeAValueMapOff = o.cubeMetaOff;
    o.cubePackedAOff = o.cubeMetaOff;
    o.cubeComplexStatusOff = o.cubeMetaOff;
    o.cubeComplexBOff = o.cubeMetaOff;
    o.cubeScratchOff = o.cubeMetaOff;
    o.cubeTilingOff = o.cubeMetaOff;
    o.cubeLibWorkspaceOff = o.cubeMetaOff;
    return o;
}

bool WorkspaceValuesFitInt64(std::initializer_list<__int128> values)
{
    const __int128 maxValue = std::numeric_limits<int64_t>::max();
    for (const __int128 value : values) {
        if (value > maxValue) {
            return false;
        }
    }
    return true;
}

struct ComplexCubeWorkspaceSizing {
    __int128 metaEnd;
    __int128 rowIdsOff;
    __int128 rowIdsBytes;
    __int128 slot1Entries;
    __int128 slot1Off;
    __int128 aMapOff;
    __int128 packedAOff;
    __int128 statusOff;
    __int128 complexBOff;
    __int128 scratchOff;
    __int128 total;
};

ComplexCubeWorkspaceSizing BuildComplexCubeWorkspaceSizing(
    int64_t m, int64_t k, int64_t n, uint64_t nnz, int64_t cubeMetaOff)
{
    const bool degree64Reserve = m > 0 &&
        static_cast<__int128>(m) * SPMM_FP32_DEGREE64_DEGREE ==
            static_cast<__int128>(nnz);
    const bool compactP03Reserve = IsOfficialComplexP03Shape(m, k, n, nnz);
    const int64_t groupCount = degree64Reserve ? (m + 1) / 2 :
        (compactP03Reserve ?
            (m + SPMM_CUBE_MIN_ROWS - 1) / SPMM_CUBE_MIN_ROWS :
            (m + kComplexCubeReserveRowsPerGroup - 1) /
                kComplexCubeReserveRowsPerGroup);
    const int64_t paddedRows = groupCount * SPMM_CUBE_MIN_ROWS;
    ComplexCubeWorkspaceSizing sizing;
    sizing.metaEnd = static_cast<__int128>(cubeMetaOff) +
        static_cast<__int128>(sizeof(SpmmCubeGroupMeta)) * groupCount;
    sizing.rowIdsOff = AlignWorkspace(sizing.metaEnd);
    sizing.rowIdsBytes = static_cast<__int128>(sizeof(int32_t)) * paddedRows;
    sizing.slot1Entries = degree64Reserve ? static_cast<__int128>(m) *
        (SPMM_COMPLEX64_SEGMENTED_MAX_SLOTS + 1) : static_cast<__int128>(m);
    sizing.slot1Off = AlignWorkspace(sizing.rowIdsOff + sizing.rowIdsBytes);
    sizing.aMapOff = AlignWorkspace(sizing.slot1Off +
        static_cast<__int128>(sizeof(int32_t)) * sizing.slot1Entries);
    sizing.packedAOff = AlignWorkspace(sizing.aMapOff +
        static_cast<__int128>(sizeof(uint32_t)) * nnz);
    sizing.statusOff = AlignWorkspace(sizing.packedAOff +
        static_cast<__int128>(sizeof(uint16_t)) * 6 * paddedRows *
            SPMM_CUBE_MAX_K);
    sizing.complexBOff = AlignWorkspace(sizing.statusOff +
        static_cast<__int128>(sizeof(uint32_t)));
    sizing.scratchOff = AlignWorkspace(sizing.complexBOff +
        static_cast<__int128>(sizeof(uint16_t)) * 4 * k * n);
    sizing.total = AlignWorkspace(sizing.scratchOff +
        static_cast<__int128>(sizeof(float)) * 2 * paddedRows * n);
    return sizing;
}

bool ApplyComplexCubeWorkspaceSizing(
    WsOffsets &o, const ComplexCubeWorkspaceSizing &sizing)
{
    if (!WorkspaceValuesFitInt64({sizing.metaEnd, sizing.rowIdsOff,
                                  sizing.slot1Off, sizing.aMapOff,
                                  sizing.packedAOff, sizing.statusOff,
                                  sizing.complexBOff, sizing.scratchOff,
                                  sizing.total})) {
        o.totalBytes = -1;
        return false;
    }
    o.cubeRowIdsOff = static_cast<int64_t>(sizing.rowIdsOff);
    o.cubeSlot1Off = static_cast<int64_t>(sizing.slot1Off);
    o.cubeAValueMapOff = static_cast<int64_t>(sizing.aMapOff);
    o.cubePackedAOff = static_cast<int64_t>(sizing.packedAOff);
    o.cubeComplexStatusOff = static_cast<int64_t>(sizing.statusOff);
    o.cubeComplexBOff = static_cast<int64_t>(sizing.complexBOff);
    o.cubeScratchOff = static_cast<int64_t>(sizing.scratchOff);
    o.cubeTilingOff = o.cubeScratchOff;
    o.cubeLibWorkspaceOff = o.cubeScratchOff;
    o.totalBytes = static_cast<int64_t>(sizing.total);
    return true;
}

struct CubeRowCapacity {
    int64_t maxGroupCount;
    int64_t maxPaddedRows;
};

CubeRowCapacity GetCubeRowCapacity(int64_t m, bool reserveFp32)
{
    const int64_t maxSegmentCount = reserveFp32 ? m * 2 : m;
    const int64_t maxGroupCount =
        (maxSegmentCount + SPMM_CUBE_MIN_ROWS - 1) / SPMM_CUBE_MIN_ROWS;
    const int64_t maxWideGroupCount =
        (maxSegmentCount + SPMM_CUBE_ROWS - 1) / SPMM_CUBE_ROWS;
    return {maxGroupCount, std::max(maxGroupCount * SPMM_CUBE_MIN_ROWS,
                                    maxWideGroupCount * SPMM_CUBE_ROWS)};
}

struct StandardCubeMetaSizing {
    int64_t maxPaddedRows;
    __int128 metaEnd;
    __int128 rowIdsBytes;
    __int128 slot1Entries;
};

StandardCubeMetaSizing BuildStandardCubeMetaSizing(
    int64_t m, bool reserveFp32, bool reserveFp32Degree2,
    bool reserveComplexRegBase, int64_t cubeMetaOff)
{
    const CubeRowCapacity rows = GetCubeRowCapacity(m, reserveFp32);
    const int64_t maxStructuredBatchCount = reserveComplexRegBase ? m :
        (rows.maxPaddedRows + SPMM_STRUCTURED_MIN_AVERAGE_ROWS_RESERVE - 1) /
            SPMM_STRUCTURED_MIN_AVERAGE_ROWS_RESERVE;
    const __int128 cubeMetaBytes = static_cast<__int128>(sizeof(SpmmCubeGroupMeta)) *
        rows.maxGroupCount;
    const bool reserveAnyFp32 = reserveFp32 || reserveFp32Degree2;
    const __int128 structuredMetaBytes = (reserveFp32Degree2 || reserveComplexRegBase) ?
        static_cast<__int128>(sizeof(SpmmFp32Degree2BatchMeta)) *
            maxStructuredBatchCount : 0;
    const bool reserveStructured = reserveAnyFp32 || reserveComplexRegBase;
    const __int128 structuredRows = reserveFp32Degree2 ?
        static_cast<__int128>(rows.maxPaddedRows) : static_cast<__int128>(m);
    const __int128 rowIdsBytes = static_cast<__int128>(sizeof(int32_t)) *
        (reserveStructured ? structuredRows : rows.maxPaddedRows);
    const bool reserveAnyDegree2 = reserveFp32Degree2 || reserveComplexRegBase;
    const __int128 slot1Entries = reserveStructured ?
        structuredRows * (reserveFp32 ? 3 : (reserveAnyDegree2 ? 2 : 1)) :
        static_cast<__int128>(m);
    return {rows.maxPaddedRows, static_cast<__int128>(cubeMetaOff) +
        std::max(cubeMetaBytes, structuredMetaBytes), rowIdsBytes, slot1Entries};
}

struct StandardCubeWorkspaceSizing {
    __int128 metaEnd;
    __int128 rowIdsOff;
    __int128 rowIdsBytes;
    __int128 slot1Entries;
    __int128 slot1Off;
    __int128 aMapOff;
    __int128 packedAOff;
    __int128 scratchOff;
    __int128 tilingOff;
    __int128 libWorkspaceOff;
    __int128 total;
};

StandardCubeWorkspaceSizing BuildStandardCubeWorkspaceSizing(
    int64_t m, int64_t k, int64_t n, uint64_t nnz, bool reserveFp32,
    bool reserveFp32Degree2, bool reserveComplexRegBase, int64_t cubeMetaOff)
{
    const StandardCubeMetaSizing meta = BuildStandardCubeMetaSizing(
        m, reserveFp32, reserveFp32Degree2, reserveComplexRegBase, cubeMetaOff);
    const bool reserveAnyDegree2 = reserveFp32Degree2 || reserveComplexRegBase;
    const __int128 rowIdsOff = AlignWorkspace(meta.metaEnd);
    const __int128 slot1Off = AlignWorkspace(rowIdsOff + meta.rowIdsBytes);
    const __int128 aMapOff = AlignWorkspace(
        slot1Off + meta.slot1Entries * sizeof(int32_t));
    const __int128 packedAOff = AlignWorkspace(aMapOff +
        (reserveComplexRegBase ? 0 : static_cast<__int128>(nnz) * sizeof(uint32_t)));
    const int64_t packedK = reserveFp32 ? SPMM_FP32_CUBE_MAX_K : SPMM_CUBE_MAX_K;
    const int64_t packedElementBytes = reserveComplexRegBase ?
        sizeof(uint64_t) : (reserveAnyDegree2 || reserveFp32 ? sizeof(float) :
                             sizeof(uint16_t));
    const __int128 packedCubeElements = reserveComplexRegBase ? 0 :
        static_cast<__int128>(meta.maxPaddedRows) * packedK;
    const __int128 packedRegBaseElements = reserveFp32 ?
        static_cast<__int128>(std::max(m, k)) * SPMM_FP32_REGBASE_DEGREE : 0;
    const __int128 packedDegree2Elements = reserveFp32Degree2 ?
        (reserveFp32Degree2 ? static_cast<__int128>(meta.maxPaddedRows) : 0) *
            SPMM_FP32_STRUCTURED_MAX_DEGREE : 0;
    const __int128 packedComplexRegBaseElements = reserveComplexRegBase ?
        static_cast<__int128>(m) * SPMM_COMPLEX64_REGBASE_MAX_DEGREE : 0;
    const __int128 packedElements = std::max(
        std::max(packedCubeElements, packedRegBaseElements),
        std::max(packedDegree2Elements, packedComplexRegBaseElements));
    const __int128 scratchOff = AlignWorkspace(packedAOff +
        packedElementBytes * packedElements);
    const __int128 scratchBytes = reserveFp32 ?
        static_cast<__int128>(sizeof(float)) * meta.maxPaddedRows * n :
        (reserveAnyDegree2 ? static_cast<__int128>(sizeof(SpmmFp32RegBaseHeader)) +
            static_cast<__int128>(sizeof(int32_t)) * m :
            static_cast<__int128>(sizeof(float)) * meta.maxPaddedRows * n);
    const bool reserveStructured = reserveFp32 || reserveAnyDegree2;
    const __int128 tilingOff = reserveStructured ?
        AlignWorkspace(scratchOff + scratchBytes) : scratchOff;
    const __int128 libWorkspaceOff = reserveFp32 ?
        AlignWorkspace(tilingOff + SPMM_CUBE_TILING_BYTES) : tilingOff;
    const __int128 total = reserveFp32 ?
        libWorkspaceOff + static_cast<__int128>(GetSpmmLibApiWorkspaceSize()) :
        (reserveAnyDegree2 ? tilingOff : scratchOff + scratchBytes);
    return {meta.metaEnd, rowIdsOff, meta.rowIdsBytes, meta.slot1Entries,
        slot1Off, aMapOff, packedAOff, scratchOff, tilingOff, libWorkspaceOff,
        AlignWorkspace(total)};
}

bool ApplyStandardCubeWorkspaceSizing(
    WsOffsets &o, const StandardCubeWorkspaceSizing &sizing)
{
    if (!WorkspaceValuesFitInt64({sizing.metaEnd, sizing.rowIdsOff,
                                  sizing.slot1Off, sizing.aMapOff, sizing.packedAOff,
                                  sizing.scratchOff, sizing.tilingOff,
                                  sizing.libWorkspaceOff, sizing.total})) {
        o.totalBytes = -1;
        return false;
    }
    o.cubeRowIdsOff = static_cast<int64_t>(sizing.rowIdsOff);
    o.cubeAValueMapOff = static_cast<int64_t>(sizing.aMapOff);
    o.cubeSlot1Off = static_cast<int64_t>(sizing.slot1Off);
    o.cubePackedAOff = static_cast<int64_t>(sizing.packedAOff);
    o.cubeComplexStatusOff = o.cubePackedAOff;
    o.cubeScratchOff = static_cast<int64_t>(sizing.scratchOff);
    o.cubeTilingOff = static_cast<int64_t>(sizing.tilingOff);
    o.cubeLibWorkspaceOff = static_cast<int64_t>(sizing.libWorkspaceOff);
    o.totalBytes = static_cast<int64_t>(sizing.total);
    return true;
}

WsOffsets ComputeWsOffsets(int64_t m, int64_t k, int64_t n, uint64_t nnz,
                           int32_t blockDim,
                           bool reserveCube, bool reserveFp32,
                           bool reserveFp32Degree2,
                           bool reserveComplex,
                           bool reserveComplexRegBase) {
    WsOffsets o = InitializeWsOffsets(m, blockDim);
    o.cubeReserved = reserveCube;
    o.complexReserved = reserveComplex;
    if (!reserveCube) {
        o.totalBytes = o.cubeMetaOff;
        return o;
    }
    if (reserveComplex) {
        ApplyComplexCubeWorkspaceSizing(o, BuildComplexCubeWorkspaceSizing(
            m, k, n, nnz, o.cubeMetaOff));
        return o;
    }
    ApplyStandardCubeWorkspaceSizing(o, BuildStandardCubeWorkspaceSizing(
        m, k, n, nnz, reserveFp32, reserveFp32Degree2,
        reserveComplexRegBase, o.cubeMetaOff));
    return o;
}

WsOffsets ComputeTransposeWsOffsets(const aclsparseSpMatDescr *matA,
                                    const aclsparseDnMatDescr *matC)
{
    WsOffsets offsets = ComputeWsOffsets(matA->cols, matA->rows, matC->cols,
        matA->nnz, GetSpmmBlockDim(), false, false, false, false, false);
    const __int128 rowOff = offsets.totalBytes;
    const __int128 columnOff = AlignWorkspace(rowOff +
        (static_cast<__int128>(matA->cols) + 1) * sizeof(int32_t));
    const __int128 permutationOff = AlignWorkspace(columnOff +
        static_cast<__int128>(matA->nnz) * sizeof(int32_t));
    const __int128 valuesOff = AlignWorkspace(permutationOff +
        static_cast<__int128>(matA->nnz) * sizeof(int32_t));
    const size_t valueBytes = matA->valueType == ACL_COMPLEX64 ? 8 :
        (matA->valueType == ACL_FLOAT ? 4 : (matA->valueType == ACL_INT8 ? 1 : 2));
    const __int128 total = AlignWorkspace(valuesOff +
        static_cast<__int128>(matA->nnz) * valueBytes);
    if (!WorkspaceValuesFitInt64({rowOff, columnOff, permutationOff, valuesOff, total})) {
        offsets.totalBytes = -1;
        return offsets;
    }
    offsets.transposeRowsOff = static_cast<int64_t>(rowOff);
    offsets.transposeColumnsOff = static_cast<int64_t>(columnOff);
    offsets.transposePermutationOff = static_cast<int64_t>(permutationOff);
    offsets.transposeValuesOff = static_cast<int64_t>(valuesOff);
    offsets.totalBytes = static_cast<int64_t>(total);
    return offsets;
}

WsOffsets ComputeDescriptorWsOffsets(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    aclDataType computeType,
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE)
{
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE) {
        return ComputeTransposeWsOffsets(matA, matC);
    }
    const bool reserveCube = ShouldReserveCubeWorkspace(
        matA, matB, matC, opB, computeType);
    const bool reserveFp32 = ShouldReserveFp32CubeWorkspace(
        matA, matB, matC, opB, computeType);
    const bool reserveFp32Degree2 = IsFp32CubeInputDtype(
        matA->valueType, matC->valueType, computeType) && reserveCube;
    const bool reserveComplexRegBase = ShouldReserveComplexRegBaseWorkspace(
        matA, matB, matC, opB, computeType);
    const bool reserveComplex = IsComplexCubeInputDtype(
        matA->valueType, matC->valueType, computeType) &&
        reserveCube && !reserveComplexRegBase;
    return ComputeWsOffsets(
        static_cast<int64_t>(matA->rows), static_cast<int64_t>(matA->cols),
        static_cast<int64_t>(matC->cols), matA->nnz,
        static_cast<int32_t>(GetSpmmBlockDim()), reserveCube, reserveFp32,
        reserveFp32Degree2, reserveComplex, reserveComplexRegBase);
}

bool CanUseComplexCubePath(const aclsparseSpMatDescr *matA,
                           const aclsparseDnMatDescr *matB,
                           const aclsparseDnMatDescr *matC,
                           aclsparseOperation_t opB, const void *alpha,
                           const void *beta, aclDataType computeType,
                           aclsparsePointerMode_t pointerMode)
{
    return kEnableArch35Cube && kEnableArch35ComplexCube &&
        IsComplexCubeInputDtype(matA->valueType, matC->valueType, computeType) &&
        pointerMode == ACL_SPARSE_POINTER_MODE_HOST &&
        opB == ACL_SPARSE_OP_NON_TRANSPOSE &&
        matB->order == ACL_SPARSE_ORDER_ROW && matC->order == ACL_SPARSE_ORDER_ROW &&
        matB->ld == matC->cols && matC->ld == matC->cols &&
        matC->cols > 0 &&
        // The packed Cube path supports complete column tiles.
        matC->cols % SPMM_CUBE_MAX_N == 0 &&
        matA->rows >= SPMM_CUBE_MIN_ROWS &&
        matA->cols >= SPMM_CUBE_MAX_K && matA->nnz > 0 &&
        (matC->cols == SPMM_CUBE_MAX_N ||
         matA->spmmUniformDegree == SPMM_FP32_DEGREE16_DEGREE ||
         matA->spmmUniformDegree == SPMM_FP32_DEGREE32_DEGREE ||
         matA->spmmUniformDegree == SPMM_FP32_DEGREE64_DEGREE) &&
        matA->ptrType == ACL_SPARSE_INDEX_32I &&
        matA->IdxType == ACL_SPARSE_INDEX_32I &&
        (reinterpret_cast<uintptr_t>(matA->values) % alignof(uint64_t)) == 0 &&
        (reinterpret_cast<uintptr_t>(matB->values) % alignof(uint64_t)) == 0 &&
        (reinterpret_cast<uintptr_t>(matC->values) % alignof(uint64_t)) == 0 &&
        IsHostComplexScalarValue(alpha, 1.0f, 0.0f) &&
        (IsHostComplexScalarValue(beta, 0.0f, 0.0f) ||
         IsHostComplexScalarValue(beta, 1.0f, 0.0f));
}

bool CanUseCubePath(const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
                    const aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
                    const void *alpha, const void *beta, aclDataType computeType,
                    aclsparsePointerMode_t pointerMode)
{
    if (CanUseComplexCubePath(matA, matB, matC, opB, alpha, beta,
                              computeType, pointerMode)) {
        return true;
    }
    return kEnableArch35Cube &&
        (IsCubeInputDtype(matA->valueType, matC->valueType, computeType) ||
         IsFp32CubeInputDtype(matA->valueType, matC->valueType, computeType)) &&
        pointerMode == ACL_SPARSE_POINTER_MODE_HOST && opB == ACL_SPARSE_OP_NON_TRANSPOSE &&
        matB->order == ACL_SPARSE_ORDER_ROW && matC->order == ACL_SPARSE_ORDER_ROW &&
        matB->ld == matC->cols && matC->ld == matC->cols && matC->cols > 0 &&
        matC->cols <= (IsFp32CubeInputDtype(matA->valueType, matC->valueType,
                                            computeType) ? SPMM_FP32_CUBE_MAX_N : SPMM_CUBE_MAX_N) &&
        (matC->cols % 16) == 0 &&
        matA->rows > 0 && matA->nnz > 0 &&
        matA->ptrType == ACL_SPARSE_INDEX_32I && matA->IdxType == ACL_SPARSE_INDEX_32I &&
        IsHostScalarValue(alpha, computeType, 1.0f) &&
        (IsHostScalarValue(beta, computeType, 0.0f) ||
         IsHostScalarValue(beta, computeType, 1.0f));
}

bool CanUseFp32RegBasePath(const aclsparseSpMatDescr *matA,
                           const aclsparseDnMatDescr *matB,
                           const aclsparseDnMatDescr *matC,
                           aclsparseOperation_t opB, const void *alpha,
                           const void *beta, aclDataType computeType,
                           aclsparseSpMMAlg_t alg,
                           aclsparsePointerMode_t pointerMode)
{
    // This is the original Arch35 RegBase dispatch boundary. It is kept for
    // compatibility with the established large-row selection and is not a
    // tuning parameter for this change.
    constexpr uint64_t kMinimumRows = 65536;
    return IsFp32CubeInputDtype(matA->valueType, matC->valueType, computeType) &&
        alg != ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG &&
        pointerMode == ACL_SPARSE_POINTER_MODE_HOST &&
        opB == ACL_SPARSE_OP_NON_TRANSPOSE &&
        matB->order == ACL_SPARSE_ORDER_ROW &&
        matC->order == ACL_SPARSE_ORDER_ROW &&
        matB->ld == SPMM_FP32_REGBASE_N &&
        matC->ld == SPMM_FP32_REGBASE_N &&
        matC->cols == SPMM_FP32_REGBASE_N &&
        matA->rows >= kMinimumRows && matA->nnz > 0 &&
        matA->ptrType == ACL_SPARSE_INDEX_32I &&
        matA->IdxType == ACL_SPARSE_INDEX_32I &&
        IsHostScalarValue(alpha, computeType, 1.0f) &&
        IsHostScalarValue(beta, computeType, 1.0f);
}

bool CanUseCommonFp32StructuredPath(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, aclsparsePointerMode_t pointerMode)
{
    return IsFp32CubeInputDtype(
        matA->valueType, matC->valueType, computeType) &&
        alg != ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG &&
        pointerMode == ACL_SPARSE_POINTER_MODE_HOST &&
        opB == ACL_SPARSE_OP_NON_TRANSPOSE &&
        matB->order == ACL_SPARSE_ORDER_ROW &&
        matC->order == ACL_SPARSE_ORDER_ROW &&
        matB->ld == matC->cols && matC->ld == matC->cols &&
        matA->nnz > 0 && matA->ptrType == ACL_SPARSE_INDEX_32I &&
        matA->IdxType == ACL_SPARSE_INDEX_32I &&
        IsHostScalarValue(alpha, computeType, 1.0f) &&
        IsHostScalarValue(beta, computeType, 1.0f);
}

bool CanUseFp32Degree2Path(const aclsparseSpMatDescr *matA,
                           const aclsparseDnMatDescr *matB,
                           const aclsparseDnMatDescr *matC,
                           aclsparseOperation_t opB, const void *alpha,
                           const void *beta, aclDataType computeType,
                           aclsparseSpMMAlg_t alg,
                           aclsparsePointerMode_t pointerMode)
{
    return CanUseCommonFp32StructuredPath(
        matA, matB, matC, opB, alpha, beta, computeType, alg, pointerMode) &&
        matC->cols > 0 &&
        matA->rows >= SPMM_FP32_DEGREE2_MIN_ROWS;
}

bool CanUseFp32Degree4Path(const aclsparseSpMatDescr *matA,
                           const aclsparseDnMatDescr *matB,
                           const aclsparseDnMatDescr *matC,
                           aclsparseOperation_t opB, const void *alpha,
                           const void *beta, aclDataType computeType,
                           aclsparseSpMMAlg_t alg,
                           aclsparsePointerMode_t pointerMode)
{
    if (!CanUseCommonFp32StructuredPath(
            matA, matB, matC, opB, alpha, beta, computeType, alg,
            pointerMode) || matA->rows <= 0) {
        return false;
    }
    const int32_t minColumns = matA->spmmUniformDegree ==
        SPMM_FP32_DEGREE32_DEGREE ? SPMM_FP32_DEGREE32_MIN_N :
        matA->spmmUniformDegree == SPMM_FP32_DEGREE64_DEGREE ?
        SPMM_FP32_DEGREE64_MIN_N : SPMM_FP32_DEGREE4_MIN_N;
    return matC->cols >= minColumns;
}

bool IsComplexRegBaseSelection(const aclsparseSpMatDescr *matA)
{
    return matA != nullptr && IsComplexRegBaseDegree(matA->spmmUniformDegree) &&
        matA->spmmCubeRows == ComplexRegBaseRows(matA->spmmUniformDegree);
}

bool CanUseComplexRegBasePath(const aclsparseSpMatDescr *matA,
                              const aclsparseDnMatDescr *matB,
                              const aclsparseDnMatDescr *matC,
                              aclsparseOperation_t opB, const void *alpha,
                              const void *beta, aclDataType computeType,
                              aclsparsePointerMode_t pointerMode)
{
    return IsComplexCubeInputDtype(
               matA->valueType, matC->valueType, computeType) &&
        pointerMode == ACL_SPARSE_POINTER_MODE_HOST &&
        opB == ACL_SPARSE_OP_NON_TRANSPOSE &&
        matB->order == ACL_SPARSE_ORDER_ROW &&
        matC->order == ACL_SPARSE_ORDER_ROW &&
        matB->ld == matC->cols && matC->ld == matC->cols &&
        matC->cols >= SPMM_COMPLEX64_REGBASE_MIN_N &&
        matA->rows >= SPMM_COMPLEX64_REGBASE_MIN_ROWS &&
        IsComplexRegBaseDegree(matA->spmmUniformDegree) &&
        matA->nnz == matA->rows *
            static_cast<uint64_t>(matA->spmmUniformDegree) &&
        matA->ptrType == ACL_SPARSE_INDEX_32I &&
        matA->IdxType == ACL_SPARSE_INDEX_32I &&
        (reinterpret_cast<uintptr_t>(matA->values) % alignof(uint64_t)) == 0 &&
        (reinterpret_cast<uintptr_t>(matB->values) % alignof(uint64_t)) == 0 &&
        (reinterpret_cast<uintptr_t>(matC->values) % alignof(uint64_t)) == 0 &&
        IsHostComplexScalarValue(alpha, 1.0f, 0.0f) &&
         IsHostComplexScalarValue(beta, 1.0f, 0.0f);
}

aclsparseStatus_t BuildComplexSpmmCubeData(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB,
    aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparsePointerMode_t pointerMode, uint8_t *buffer, const WsOffsets &off,
    bool reserveComplexRegBase, bool *cubeEligible, int32_t *cubeRows,
    int32_t *cubeGroupCount, int32_t *complexFallbackCount)
{
    aclsparseStatus_t status = BuildComplexRegBaseData(
        matA, buffer, off, CanUseComplexRegBasePath(
            matA, matB, matC, opB, alpha, beta, computeType, pointerMode),
        cubeEligible, cubeRows, cubeGroupCount);
    if (status != ACL_SPARSE_STATUS_SUCCESS || *cubeEligible ||
        reserveComplexRegBase) {
        return status;
    }
    return BuildComplexCubePrepackedData(
        matA, matB, static_cast<int32_t>(matC->cols), buffer, off,
        CanUseComplexCubePath(matA, matB, matC, opB, alpha, beta, computeType,
                              pointerMode),
        cubeEligible, cubeRows, cubeGroupCount, complexFallbackCount);
}

aclsparseStatus_t BuildFp32SpmmCubeData(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB,
    aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, aclsparsePointerMode_t pointerMode,
    uint8_t *buffer, const WsOffsets &off, bool *cubeEligible,
    int32_t *cubeRows, int32_t *cubeGroupCount, int32_t *fallbackCount)
{
    aclsparseStatus_t status = BuildFp32RegBaseData(
        matA, buffer, off, CanUseFp32RegBasePath(
            matA, matB, matC, opB, alpha, beta, computeType, alg, pointerMode),
        cubeEligible, cubeRows, cubeGroupCount);
    if (status == ACL_SPARSE_STATUS_SUCCESS && !*cubeEligible) {
        status = BuildFp32Degree2Data(matA, buffer, off, CanUseFp32Degree2Path(
            matA, matB, matC, opB, alpha, beta, computeType, alg, pointerMode),
            cubeEligible, cubeRows, cubeGroupCount, fallbackCount);
    }
    if (status == ACL_SPARSE_STATUS_SUCCESS && !*cubeEligible) {
        status = BuildFp32Degree4Data(matA, static_cast<int32_t>(matC->cols),
            buffer, off, CanUseFp32Degree4Path(matA, matB, matC, opB, alpha,
            beta, computeType, alg, pointerMode), cubeEligible, cubeRows,
            cubeGroupCount, fallbackCount);
    }
    if (status == ACL_SPARSE_STATUS_SUCCESS && !*cubeEligible) {
        status = BuildFp32PackedData(matA, static_cast<int32_t>(matC->cols),
            buffer, off, CanUseCubePath(matA, matB, matC, opB, alpha, beta,
            computeType, pointerMode), cubeEligible, cubeRows, cubeGroupCount);
    }
    return status;
}

aclsparseStatus_t BuildSpmmCubeData(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB,
    aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, aclsparsePointerMode_t pointerMode,
    uint8_t *buffer, const WsOffsets &off, bool *cubeEligible,
    int32_t *cubeRows, int32_t *cubeGroupCount,
    int32_t *degree2FallbackCount, int32_t *complexFallbackCount)
{
    const bool complexInput = IsComplexCubeInputDtype(
        matA->valueType, matC->valueType, computeType);
    const bool reserveComplexRegBase = ShouldReserveComplexRegBaseWorkspace(
        matA, matB, matC, opB, computeType);
    if (complexInput) {
        return BuildComplexSpmmCubeData(
            matA, matB, matC, opB, alpha, beta, computeType, pointerMode,
            buffer, off, reserveComplexRegBase, cubeEligible, cubeRows,
            cubeGroupCount, complexFallbackCount);
    }
    if (IsFp32CubeInputDtype(matA->valueType, matC->valueType, computeType)) {
        return BuildFp32SpmmCubeData(
            matA, matB, matC, opB, alpha, beta, computeType, alg, pointerMode,
            buffer, off, cubeEligible, cubeRows, cubeGroupCount,
            degree2FallbackCount);
    }
    return BuildCubePrepackedData(
        matA, buffer, off, CanUseCubePath(matA, matB, matC, opB, alpha, beta,
        computeType, pointerMode), cubeEligible, cubeRows, cubeGroupCount,
        degree2FallbackCount);
}

bool IsFp32StructuredCubeRows(int32_t rows)
{
    return rows == SPMM_FP32_DEGREE4_ROWS ||
        rows == SPMM_FP32_DEGREE32_ROWS ||
        rows == SPMM_FP32_DEGREE64_MODE;
}

SpmmHostTilingData BuildBasicSpmmHostTilingData(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, const WsOffsets &off,
    aclsparseOperation_t opB, aclsparseSpMMAlg_t alg)
{
    SpmmHostTilingData td{};
    td.m = static_cast<int32_t>(matA->rows);
    td.n = static_cast<int32_t>(matC->cols);
    td.ldb = static_cast<int32_t>(matB->ld);
    td.ldc = static_cast<int32_t>(matC->ld);
    td.reorder_offset = static_cast<int32_t>(off.reorderOff);
    td.bin_edge_offset = static_cast<int32_t>(off.binEdgeOff);
    td.high_precision = ComputeSpmmHighPrecision(
        alg, SpmmDataTypeFromAcl(matA->valueType));
    td.opB = static_cast<int32_t>(opB);
    td.index_base = matA->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    td.order_pair = static_cast<int32_t>(matB->order) * 2 +
        static_cast<int32_t>(matC->order);
    td.sparse_empty = matA->nnz == 0 ? 1 : 0;
    td.row_reorder_identity = matA->spmmIdentityReorder ? 1 : 0;
    td.uniform_degree = matA->spmmUniformDegree;
    return td;
}

void PopulateCubeTilingData(
    SpmmHostTilingData *td, const aclsparseSpMatDescr *matA,
    const aclsparseDnMatDescr *matC, const WsOffsets &off,
    aclDataType computeType, bool cubeEligible, int32_t cubeRows,
    int32_t cubeGroupCount, int32_t fallbackCount)
{
    const bool complexInput = IsComplexCubeInputDtype(
        matA->valueType, matC->valueType, computeType);
    td->cube_mode = cubeEligible ? 1 : 0;
    td->cube_rows = cubeRows;
    td->cube_max_k = IsFp32CubeInputDtype(
        matA->valueType, matC->valueType, computeType) ?
        (cubeRows == SPMM_FP32_CUBE_FAST_ROWS ?
            SPMM_FP32_CUBE_FAST_K : SPMM_FP32_CUBE_MAX_K) : SPMM_CUBE_MAX_K;
    td->cube_meta_offset = static_cast<uint64_t>(off.cubeMetaOff);
    td->cube_row_ids_offset = static_cast<uint64_t>(off.cubeRowIdsOff);
    td->cube_slot1_offset = static_cast<uint64_t>(off.cubeSlot1Off);
    td->cube_packed_a_offset = static_cast<uint64_t>(off.cubePackedAOff);
    td->cube_complex_b_offset = static_cast<uint64_t>(off.cubeComplexBOff);
    td->cube_complex_b_plane_elements = complexInput ?
        static_cast<uint64_t>(matA->cols) * static_cast<uint64_t>(matC->cols) : 0;
    td->cube_complex_status_offset = static_cast<uint64_t>(off.cubeComplexStatusOff);
    td->cube_a_value_map_offset = static_cast<uint64_t>(off.cubeAValueMapOff);
    td->cube_complex_a_plane_elements = complexInput && cubeEligible ?
        static_cast<uint64_t>(cubeGroupCount) * SPMM_CUBE_MIN_ROWS * SPMM_CUBE_MAX_K : 0;
    td->cube_a_value_nnz = cubeEligible ? matA->nnz : 0;
    td->cube_complex_fallback_count = cubeEligible ? fallbackCount : 0;
    td->cube_scratch_offset = static_cast<uint64_t>(off.cubeScratchOff);
    td->cube_tiling_offset = static_cast<uint64_t>(off.cubeTilingOff);
    td->cube_lib_workspace_offset = static_cast<uint64_t>(off.cubeLibWorkspaceOff);
    td->cube_group_count = cubeEligible ? cubeGroupCount : 0;
    td->cube_tile_n = IsFp32CubeInputDtype(
        matA->valueType, matC->valueType, computeType) ?
        SPMM_CUBE_MAX_N : (complexInput ? static_cast<int32_t>(matC->cols) : 0);
}

SpmmHostTilingData BuildSpmmHostTilingData(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, const WsOffsets &off,
    aclsparseOperation_t opB, const void *alpha, const void *beta,
    aclDataType computeType, aclsparseSpMMAlg_t alg,
    aclsparsePointerMode_t pointerMode, bool cubeEligible, int32_t cubeRows,
    int32_t cubeGroupCount, int32_t fallbackCount)
{
    SpmmHostTilingData td = BuildBasicSpmmHostTilingData(
        matA, matB, matC, off, opB, alg);
    PopulateCubeTilingData(&td, matA, matC, off, computeType, cubeEligible,
                           cubeRows, cubeGroupCount, fallbackCount);
    const bool complexInput = IsComplexCubeInputDtype(
        matA->valueType, matC->valueType, computeType);
    td.cube_beta_one = cubeEligible &&
        (complexInput ? IsHostComplexScalarValue(beta, 1.0f, 0.0f) :
                        IsHostScalarValue(beta, computeType, 1.0f)) ? 1 : 0;
    SetTilingAlphaBeta(&td, alpha, beta, computeType, pointerMode);
    return td;
}

bool IsActiveCubePathUsable(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, aclsparsePointerMode_t pointerMode)
{
    if (!matA->spmmCubeEligible) {
        return false;
    }
    const bool complexInput = IsComplexCubeInputDtype(
        matA->valueType, matC->valueType, computeType);
    if (matA->valueType == ACL_FLOAT &&
        matA->spmmCubeRows == SPMM_FP32_REGBASE_ROWS) {
        return CanUseFp32RegBasePath(matA, matB, matC, opB, alpha, beta,
                                     computeType, alg, pointerMode);
    }
    if (matA->valueType == ACL_FLOAT &&
        matA->spmmCubeRows == SPMM_FP32_DEGREE2_ROWS) {
        return CanUseFp32Degree2Path(matA, matB, matC, opB, alpha, beta,
                                     computeType, alg, pointerMode);
    }
    if (matA->valueType == ACL_FLOAT &&
        IsFp32StructuredCubeRows(matA->spmmCubeRows)) {
        return CanUseFp32Degree4Path(matA, matB, matC, opB, alpha, beta,
                                     computeType, alg, pointerMode);
    }
    if (complexInput && IsComplexRegBaseSelection(matA)) {
        return CanUseComplexRegBasePath(matA, matB, matC, opB, alpha, beta,
                                        computeType, pointerMode);
    }
    return CanUseCubePath(matA, matB, matC, opB, alpha, beta,
                          computeType, pointerMode);
}

} // namespace

size_t spmm_host_internal::SpmmWorkspaceRegistrationCount()
{
    std::lock_guard<std::mutex> lock(SpmmWorkspaceRegistryMutex());
    return SpmmWorkspaceRegistry().Size();
}

aclsparseStatus_t aclsparseSpMMGetBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, size_t *size)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpMMGetBufferSize", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (size == nullptr) {
        OP_LOGE("aclsparseSpMMGetBufferSize", "size is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseSpMatDescr *matAInner = (aclsparseSpMatDescr *)matA;
    aclsparseDnMatDescr *matBInner = (aclsparseDnMatDescr *)matB;
    aclsparseDnMatDescr *matCInner = (aclsparseDnMatDescr *)matC;
    aclsparseStatus_t st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                                            opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpMMGetBufferSize",
                "ValidateSpmmInputs failed, status=%d", static_cast<int>(st));
        return st;
    }
    WsOffsets off = ComputeDescriptorWsOffsets(
        matAInner, matBInner, matCInner, opB, computeType, opA);
    if (off.totalBytes < 0 || !WsOffsetsFitTiling(off)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    *size = static_cast<size_t>(off.totalBytes);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t PreprocessSpmmCsr(
    aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matC,
    const WsOffsets &off, const void *alpha, aclDataType computeType,
    aclsparsePointerMode_t pointerMode, void *buffer)
{
    SpmmCsrMat csr(matA, static_cast<int32_t>(GetSpmmBlockDim()));
    const bool skipSparsePreprocess = matA->nnz == 0 || matC->cols == 0 ||
        (pointerMode == ACL_SPARSE_POINTER_MODE_HOST &&
         IsHostScalarZero(alpha, computeType));
    uint32_t pre = csr.DoPreProcess(static_cast<uint8_t *>(buffer),
                                    off.reorderOff,
                                    off.binEdgeOff,
                                    /*identity=*/skipSparsePreprocess);
    if (pre != ACL_SPARSE_STATUS_SUCCESS) {
        return static_cast<aclsparseStatus_t>(pre);
    }
    matA->spmmIdentityReorder = csr.IsIdentityReorder();
    matA->spmmUniformDegree = csr.GetUniformDegree();
    return ACL_SPARSE_STATUS_SUCCESS;
}

void ResetSpmmCubeMetadata(aclsparseSpMatDescr *matA)
{
    matA->spmmCubeEligible = false;
    matA->spmmCubeRows = 0;
    matA->spmmCubeGroupCount = 0;
    matA->spmmDegree2FallbackCount = 0;
    matA->spmmComplexFallbackCount = 0;
}

struct SpmmCubeBuildResult {
    bool eligible = false;
    int32_t rows = 0;
    int32_t groups = 0;
    int32_t degree2FallbackCount = 0;
    int32_t complexFallbackCount = 0;
};

aclsparseStatus_t BuildAndPublishSpmmCubeData(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB,
    aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, aclsparsePointerMode_t pointerMode,
    uint8_t *buffer, const WsOffsets &off, SpmmCubeBuildResult *result)
{
    ResetSpmmCubeMetadata(matA);
    const aclsparseStatus_t status = BuildSpmmCubeData(
        matA, matB, matC, opB, alpha, beta, computeType, alg, pointerMode,
        buffer, off, &result->eligible, &result->rows, &result->groups,
        &result->degree2FallbackCount, &result->complexFallbackCount);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    matA->spmmCubeEligible = result->eligible;
    matA->spmmCubeRows = result->rows;
    matA->spmmCubeGroupCount = result->groups;
    matA->spmmDegree2FallbackCount = result->degree2FallbackCount;
    matA->spmmComplexFallbackCount = result->complexFallbackCount;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t WriteSpmmHostTilingData(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB,
    aclsparseDnMatDescr *matC, const WsOffsets &off,
    aclsparseOperation_t opB, const void *alpha, const void *beta,
    aclDataType computeType, aclsparseSpMMAlg_t alg,
    aclsparsePointerMode_t pointerMode, void *buffer,
    const SpmmCubeBuildResult &cube)
{
    const bool complexInput = IsComplexCubeInputDtype(
        matA->valueType, matC->valueType, computeType);
    const int32_t fallbackCount = complexInput ? cube.complexFallbackCount :
        cube.degree2FallbackCount;
    const SpmmHostTilingData td = BuildSpmmHostTilingData(
        matA, matB, matC, off, opB, alpha, beta, computeType, alg, pointerMode,
        cube.eligible, cube.rows, cube.groups, fallbackCount);
    const SpmmTilingWireV1 wire = SerializeSpmmTiling(td);
    return aclrtMemcpy(static_cast<uint8_t *>(buffer) + off.tilingOff,
        sizeof(SpmmTilingWireV1), &wire, sizeof(SpmmTilingWireV1),
        ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE ?
        ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_EXECUTION_FAILED;
}

void MakeTransposeDescriptor(const aclsparseSpMatDescr *source,
                             const WsOffsets &offsets, void *buffer,
                             aclsparseSpMatDescr *transpose)
{
    auto *base = static_cast<uint8_t *>(buffer);
    transpose->rows = source->cols;
    transpose->cols = source->rows;
    transpose->nnz = source->nnz;
    transpose->ptrs = base + offsets.transposeRowsOff;
    transpose->idxs = base + offsets.transposeColumnsOff;
    transpose->values = base + offsets.transposeValuesOff;
    transpose->baseType = ACL_SPARSE_INDEX_BASE_ZERO;
    transpose->ptrType = ACL_SPARSE_INDEX_32I;
    transpose->IdxType = ACL_SPARSE_INDEX_32I;
    transpose->format = ACL_SPARSE_FORMAT_CSR;
    transpose->valueType = source->valueType;
    transpose->spmmIdentityReorder = true;
    transpose->spmmUniformDegree = source->nnz == 0 ? 0 : -1;
}

aclsparseStatus_t BuildTransposeSpmmTiling(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB,
    aclsparseDnMatDescr *matC, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, aclsparsePointerMode_t pointerMode,
    void *buffer, const WsOffsets &offsets)
{
    InvalidateSpmmWorkspace(buffer);
    matA->activeBuffer = nullptr;
    if (matA->activeSpmvBuffer == buffer) matA->activeSpmvBuffer = nullptr;
    ResetSpmmCubeMetadata(matA);
    SpmmCsrMat csr(matA, static_cast<int32_t>(GetSpmmBlockDim()));
    const aclsparseStatus_t status = csr.DoTransposePreProcess(
        static_cast<uint8_t *>(buffer), offsets);
    if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
    aclsparseSpMatDescr transpose;
    MakeTransposeDescriptor(matA, offsets, buffer, &transpose);
    return WriteSpmmHostTilingData(&transpose, matB, matC, offsets,
        opB, alpha, beta, computeType, alg, pointerMode, buffer, {});
}

aclsparseStatus_t RunTransposeSpmm(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB,
    aclsparseDnMatDescr *matC, const void *initialC,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, aclsparsePointerMode_t pointerMode,
    aclrtStream stream, void *buffer, const WsOffsets &offsets)
{
    // Identity includes opA, dense layouts and scalar bits or device pointers.
    // Device scalar values are loaded by the existing SIMT kernel on each run.
    if (!IsOwnedSpmmWorkspace(matA, matB, matC, opB, alpha, beta,
        computeType, alg, pointerMode, stream, buffer, opA)) {
        const aclsparseStatus_t status = BuildTransposeSpmmTiling(matA,
            matB, matC, opB, alpha, beta, computeType, alg, pointerMode,
            buffer, offsets);
        if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
    }
    auto *base = static_cast<uint8_t *>(buffer);
    const int32_t valueBytes = matA->valueType == ACL_COMPLEX64 ? 8 :
        (matA->valueType == ACL_FLOAT ? 4 : (matA->valueType == ACL_INT8 ? 1 : 2));
    spmm_transpose_values_launch(matA->values,
        base + offsets.transposePermutationOff, base + offsets.transposeValuesOff,
        matA->nnz, valueBytes, matA->valueType == ACL_COMPLEX64 &&
        opA == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE, GetSpmmBlockDim(), stream);
    spmm_kernel_launch(base + offsets.transposeRowsOff,
        base + offsets.transposeColumnsOff, base + offsets.transposeValuesOff,
        matB->values, initialC, matC->values, buffer, base + offsets.tilingOff,
        SpmmDataTypeComboFromAcl(matA->valueType, matC->valueType, computeType),
        0, 0, GetSpmmBlockDim(), GetSpmmBlockDim(), stream);
    PublishSpmmWorkspace(matA, matB, matC, opB, alpha, beta,
        computeType, alg, pointerMode, stream, buffer, opA);
    matA->activeBuffer = buffer;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 计算 SpMM 的 reorder/bin/tiling 并写入 device workspace buffer。
// 该函数重建 workspace 及与其绑定的 host-side cube metadata。
static aclsparseStatus_t SpmmBuildTilingToBuffer(
    aclsparseSpMatDescr *matAInner, aclsparseDnMatDescr *matBInner,
    aclsparseDnMatDescr *matCInner, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, aclsparsePointerMode_t pointerMode, void *buffer)
{
    if (matAInner->activeSpmvBuffer == buffer) {
        matAInner->activeSpmvBuffer = nullptr;
    }
    InvalidateSpmmWorkspace(buffer);
    matAInner->activeBuffer = nullptr;
    ResetSpmmCubeMetadata(matAInner);
    matAInner->spmmIdentityReorder = false;
    matAInner->spmmUniformDegree = -1;
    const WsOffsets off = ComputeDescriptorWsOffsets(
        matAInner, matBInner, matCInner, opB, computeType);
    if (off.totalBytes < 0 || !WsOffsetsFitTiling(off)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    aclsparseStatus_t status = PreprocessSpmmCsr(
        matAInner, matCInner, off, alpha, computeType, pointerMode, buffer);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    SpmmCubeBuildResult cube;
    status = BuildAndPublishSpmmCubeData(
        matAInner, matBInner, matCInner, opB, alpha, beta, computeType, alg,
        pointerMode, static_cast<uint8_t *>(buffer), off, &cube);
    return status == ACL_SPARSE_STATUS_SUCCESS ? WriteSpmmHostTilingData(
        matAInner, matBInner, matCInner, off, opB, alpha, beta, computeType,
        alg, pointerMode, buffer, cube) : status;
}

static aclsparseStatus_t SpmmRefreshActiveTiling(
    void *buffer, const WsOffsets &off,
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opB, const void *alpha, const void *beta,
    aclDataType computeType, aclsparseSpMMAlg_t alg,
    aclsparsePointerMode_t pointerMode)
{
    if (matA->activeSpmvBuffer == buffer) {
        matA->activeSpmvBuffer = nullptr;
    }
    // Every tiling field is derivable from descriptors and workspace offsets.
    // Rebuilding it avoids a synchronous Device-to-Host read on every execute.
    const bool complexInput = IsComplexCubeInputDtype(
        matA->valueType, matC->valueType, computeType);
    const bool cubeEligible = IsActiveCubePathUsable(
        matA, matB, matC, opB, alpha, beta, computeType, alg, pointerMode);
    const int32_t fallbackCount = complexInput ?
        matA->spmmComplexFallbackCount : matA->spmmDegree2FallbackCount;
    const SpmmHostTilingData td = BuildSpmmHostTilingData(
        matA, matB, matC, off, opB, alpha, beta, computeType, alg,
        pointerMode, cubeEligible, matA->spmmCubeRows,
        matA->spmmCubeGroupCount, fallbackCount);
    const SpmmTilingWireV1 wire = SerializeSpmmTiling(td);
    aclError ret = aclrtMemcpy(static_cast<uint8_t *>(buffer) + off.tilingOff,
                               sizeof(SpmmTilingWireV1),
                               &wire, sizeof(SpmmTilingWireV1),
                               ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SpmmEnsureTilingReady(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opB, const void *alpha, const void *beta,
    aclDataType computeType, aclsparseSpMMAlg_t alg,
    aclsparsePointerMode_t pointerMode, aclrtStream stream,
    void *buffer, const WsOffsets &off)
{
    if (IsOwnedSpmmWorkspace(matA, matB, matC, opB, alpha, beta,
                             computeType, alg, pointerMode, stream, buffer)) {
        const bool complexRegBaseActive = matA->spmmCubeEligible &&
            matA->valueType == ACL_COMPLEX64 &&
            IsComplexRegBaseSelection(matA) &&
            CanUseComplexRegBasePath(
                matA, matB, matC, opB, alpha, beta, computeType,
                pointerMode);
        const bool fp32RegBaseActive = matA->spmmCubeEligible &&
            matA->valueType == ACL_FLOAT &&
            matA->spmmCubeRows == SPMM_FP32_REGBASE_ROWS &&
            CanUseFp32RegBasePath(
                matA, matB, matC, opB, alpha, beta, computeType, alg,
                pointerMode);
        const bool fp32Degree2Active = matA->spmmCubeEligible &&
            matA->valueType == ACL_FLOAT &&
            matA->spmmCubeRows == SPMM_FP32_DEGREE2_ROWS &&
            CanUseFp32Degree2Path(
                matA, matB, matC, opB, alpha, beta, computeType, alg,
                pointerMode);
        const bool fp32MultiDegreeActive = matA->spmmCubeEligible &&
            matA->valueType == ACL_FLOAT &&
            IsFp32StructuredCubeRows(matA->spmmCubeRows) &&
            CanUseFp32Degree4Path(
                matA, matB, matC, opB, alpha, beta, computeType, alg,
                pointerMode);
        if (complexRegBaseActive || fp32RegBaseActive ||
            fp32Degree2Active || fp32MultiDegreeActive) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        return SpmmRefreshActiveTiling(buffer, off, matA, matB, matC,
                                       opB, alpha, beta, computeType, alg, pointerMode);
    }
    return SpmmBuildTilingToBuffer(matA, matB, matC, opB,
                                   alpha, beta, computeType, alg, pointerMode, buffer);
}

static aclsparseStatus_t SpmmRunKernel(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    const void *initialC, aclDataType computeType, bool cubeMode,
    void *buffer, const WsOffsets &off, aclrtStream stream)
{
    const bool complexRegBaseSelected = cubeMode &&
        matA->valueType == ACL_COMPLEX64 &&
        IsComplexRegBaseSelection(matA);
    if (cubeMode && matA->valueType == ACL_COMPLEX64 &&
        !complexRegBaseSelected) {
        aclError ret = aclrtMemsetAsync(
            static_cast<uint8_t *>(buffer) + off.cubeComplexStatusOff,
            sizeof(int32_t), 0, sizeof(int32_t), stream);
        if (ret != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    const int32_t cubeRows = cubeMode ? matA->spmmCubeRows : 0;
    const bool regBaseSelected =
        (cubeRows == SPMM_FP32_REGBASE_ROWS ||
         cubeRows == SPMM_FP32_DEGREE2_ROWS ||
         cubeRows == SPMM_FP32_DEGREE4_ROWS ||
         cubeRows == SPMM_FP32_DEGREE32_ROWS ||
         cubeRows == SPMM_FP32_DEGREE64_MODE) &&
        matA->valueType == ACL_FLOAT;
    spmm_kernel_launch(matA->ptrs, matA->idxs, matA->values,
                       matB->values, initialC, matC->values, buffer,
                       static_cast<uint8_t *>(buffer) + off.tilingOff,
                       SpmmDataTypeComboFromAcl(matA->valueType, matC->valueType,
                                                computeType),
                       cubeRows,
                       matA->valueType == ACL_COMPLEX64 &&
                               !complexRegBaseSelected ?
                           matA->spmmComplexFallbackCount :
                           matA->spmmDegree2FallbackCount,
                       (regBaseSelected || complexRegBaseSelected) ?
                           GetSpmmBlockDim() :
                           ((cubeMode && matA->valueType == ACL_FLOAT) ? GetSpmmCubeBlockDim() :
                            (cubeMode ? GetSpmmCubeBlockDim() : GetSpmmBlockDim())),
                       GetSpmmBlockDim(), stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool IsValidInitialC(const aclsparseDnMatDescr *initialC,
                     const aclsparseDnMatDescr *matC)
{
    return initialC != nullptr && initialC->values != nullptr &&
        initialC->rows == matC->rows && initialC->cols == matC->cols &&
        initialC->ld == matC->ld && initialC->order == matC->order &&
        initialC->valueType == matC->valueType;
}

bool SupportsSeparateInitialC(const aclsparseSpMatDescr *matA,
                              const aclsparseDnMatDescr *matB,
                              const aclsparseDnMatDescr *matC,
                              aclDataType computeType)
{
    if (matC->order != ACL_SPARSE_ORDER_ROW) {
        return false;
    }
    const bool complex = computeType == ACL_COMPLEX64 &&
        matA->valueType == ACL_COMPLEX64 && matB->valueType == ACL_COMPLEX64 &&
        matC->valueType == ACL_COMPLEX64;
    const bool fp32 = computeType == ACL_FLOAT &&
        matA->valueType == ACL_FLOAT && matB->valueType == ACL_FLOAT &&
        matC->valueType == ACL_FLOAT;
    return complex || fp32;
}

bool SupportsCubeSeparateInitialC(const aclsparseSpMatDescr *matA,
                                  bool cubeMode)
{
    if (!cubeMode) {
        return true;
    }
    if (matA->valueType == ACL_COMPLEX64) {
        return true;
    }
    return matA->spmmCubeRows == SPMM_FP32_DEGREE2_ROWS ||
        IsFp32StructuredCubeRows(matA->spmmCubeRows);
}

aclsparseStatus_t GetSpmmExecutionContext(
    aclsparseHandle_t handle, aclsparseSpMatDescr *matA,
    aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclDataType computeType,
    aclsparsePointerMode_t *pointerMode, aclrtStream *stream, WsOffsets *off)
{
    aclsparseStatus_t status = aclsparseGetPointerMode(handle, pointerMode);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = aclsparseGetStream(handle, stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    *off = ComputeDescriptorWsOffsets(matA, matB, matC, opB, computeType, opA);
    return off->totalBytes < 0 || !WsOffsetsFitTiling(*off) ?
        ACL_SPARSE_STATUS_NOT_SUPPORTED : ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildRequestedSpmmTiling(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB,
    aclsparseDnMatDescr *matC, aclsparseOperation_t opA,
    aclsparseOperation_t opB, const void *alpha, const void *beta,
    aclDataType computeType, aclsparseSpMMAlg_t alg,
    aclsparsePointerMode_t pointerMode, void *buffer)
{
    if (opA == ACL_SPARSE_OP_NON_TRANSPOSE) {
        return SpmmBuildTilingToBuffer(matA, matB, matC, opB,
            alpha, beta, computeType, alg, pointerMode, buffer);
    }
    const WsOffsets offsets = ComputeDescriptorWsOffsets(
        matA, matB, matC, opB, computeType, opA);
    if (offsets.totalBytes < 0 || !WsOffsetsFitTiling(offsets)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return BuildTransposeSpmmTiling(matA, matB, matC, opB,
        alpha, beta, computeType, alg, pointerMode, buffer, offsets);
}

aclsparseStatus_t aclsparseSpMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpMMPreprocess", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (alpha == nullptr || beta == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseStatus_t modeSt = aclsparseGetPointerMode(handle, &pointerMode);
    if (modeSt != ACL_SPARSE_STATUS_SUCCESS) {
        return modeSt;
    }
    aclrtStream stream = nullptr;
    modeSt = aclsparseGetStream(handle, &stream);
    if (modeSt != ACL_SPARSE_STATUS_SUCCESS) {
        return modeSt;
    }
    aclsparseSpMatDescr *matAInner = (aclsparseSpMatDescr *)matA;
    aclsparseDnMatDescr *matBInner = (aclsparseDnMatDescr *)matB;
    aclsparseDnMatDescr *matCInner = (aclsparseDnMatDescr *)matC;
    aclsparseStatus_t st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                                            opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpMMPreprocess",
                "ValidateSpmmInputs failed, status=%d", static_cast<int>(st));
        return st;
    }
    if (buffer == nullptr) {
        OP_LOGE("aclsparseSpMMPreprocess", "buffer is nullptr");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    st = BuildRequestedSpmmTiling(matAInner, matBInner, matCInner, opA,
        opB, alpha, beta, computeType, alg, pointerMode, buffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpMMPreprocess",
                "SpmmBuildTilingToBuffer failed, status=%d", static_cast<int>(st));
        return st;
    }

    PublishSpmmWorkspace(matAInner, matBInner, matCInner, opB, alpha, beta,
                         computeType, alg, pointerMode, stream, buffer, opA);
    matAInner->activeBuffer = buffer;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpmmExecutionInputs(
    aclsparseHandle_t handle, const void *alpha, const void *beta,
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *initialC, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclDataType computeType,
    aclsparseSpMMAlg_t alg, bool *separateInitialC)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (alpha == nullptr || beta == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const aclsparseStatus_t status = ValidateSpmmInputs(
        matA, matB, matC, opA, opB, computeType, alg);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (!IsValidInitialC(initialC, matC)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *separateInitialC = initialC->values != matC->values;
    return *separateInitialC && !SupportsSeparateInitialC(
        matA, matB, matC, computeType) ? ACL_SPARSE_STATUS_NOT_SUPPORTED :
        ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SpmmExecute(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseConstDnMatDescr_t initialC,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    aclsparseSpMatDescr *matAInner = (aclsparseSpMatDescr *)matA;
    aclsparseDnMatDescr *matBInner = (aclsparseDnMatDescr *)matB;
    const aclsparseDnMatDescr *initialCInner =
        (const aclsparseDnMatDescr *)initialC;
    aclsparseDnMatDescr *matCInner = (aclsparseDnMatDescr *)matC;
    bool separateInitialC = false;
    aclsparseStatus_t st = ValidateSpmmExecutionInputs(
        handle, alpha, beta, matAInner, matBInner, initialCInner, matCInner,
        opA, opB, computeType, alg, &separateInitialC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    if (buffer == nullptr) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    aclrtStream stream = nullptr;
    WsOffsets off{};
    st = GetSpmmExecutionContext(handle, matAInner, matBInner, matCInner,
                                 opA, opB, computeType, &pointerMode, &stream, &off);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE) {
        return RunTransposeSpmm(matAInner, matBInner, matCInner,
            initialCInner->values, opA, opB, alpha, beta, computeType, alg,
            pointerMode, stream, buffer, off);
    }
    st = SpmmEnsureTilingReady(matAInner, matBInner, matCInner, opB,
                               alpha, beta, computeType, alg, pointerMode, stream, buffer, off);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    const bool cubeMode = IsActiveCubePathUsable(
        matAInner, matBInner, matCInner, opB, alpha, beta, computeType, alg,
        pointerMode);
    // These paths read initial C independently and can write a fresh output.
    if (separateInitialC && !SupportsCubeSeparateInitialC(matAInner, cubeMode)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    st = SpmmRunKernel(matAInner, matBInner, matCInner, initialCInner->values,
                       computeType, cubeMode, buffer, off, stream);
    if (st == ACL_SPARSE_STATUS_SUCCESS) {
        // A direct SpMM call may build the workspace lazily. Mark it active so
        // subsequent calls with the same descriptor and buffer can refresh only
        // scalar/layout fields, matching the optional-preprocess CUDA flow.
        PublishSpmmWorkspace(matAInner, matBInner, matCInner, opB, alpha, beta,
                             computeType, alg, pointerMode, stream, buffer);
        matAInner->activeBuffer = buffer;
    }
    return st;
}

aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    return SpmmExecute(handle, opA, opB, alpha, matA, matB, beta, matC, matC,
                       computeType, alg, buffer);
}

extern "C" aclsparseStatus_t aclsparseSpMMPrivateOutOfPlace(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseConstDnMatDescr_t initialC,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    return SpmmExecute(handle, opA, opB, alpha, matA, matB, beta, initialC, matC,
                       computeType, alg, buffer);
}
