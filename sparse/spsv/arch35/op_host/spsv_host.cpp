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

#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <new>
#include "log/log.h"
#include "cann_ops_sparse.h"
#include "spsv_host_utils.h"
#include "../op_kernel/spsv_kernel.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_spsv_descr.h"
#include "aclsparse_host_utils.h"
#include "spsv_schedule_cache.h"

#include "spsv_parameter_validation.h"
#include "spsv_workspace.h"
#include "spsv_tiling.h"
#include "spsv_launch.h"

extern "C" {

aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t* spsvDescr)
{
    if (spsvDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* inner = new (std::nothrow) aclsparseSpSVDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    *spsvDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr)
{
    if (spsvDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t status = SpsvReleaseScheduleCache(spsvDescr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    delete spsvDescr;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static size_t SpSVWorkspaceSizeForMatrix(aclsparseConstSpMatDescr_t matA, aclsparseOperation_t opA)
{
    auto* mat = spsv::ToMatInner(matA);
    int64_t m = static_cast<int64_t>(mat->rows);
    int64_t nnz = static_cast<int64_t>(mat->nnz);
    int32_t format = spsv::FormatToInt(mat->format);
    // rowPtrSize corresponds to ptrType; colIndSize corresponds to IdxType.
    size_t rowPtrSize = (mat->ptrType == ACL_SPARSE_INDEX_64I) ? 8u : 4u;
    size_t colIndSize = (mat->IdxType == ACL_SPARSE_INDEX_64I) ? 8u : 4u;
    int32_t permType = (nnz > static_cast<int64_t>(INT32_MAX)) ? 1 : 0;
    if (permType == 0 && ShouldForcePermType64()) {
        permType = 1;
    }
    int32_t idxBase = (mat->baseType == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    size_t valueSize = (mat->valueType == ACL_COMPLEX64) ? sizeof(aclsparseComplex) : sizeof(float);
    return ComputeMatrixWorkspaceOffsets(m, nnz, format, static_cast<int32_t>(opA), rowPtrSize, colIndSize, permType,
                                         idxBase, valueSize)
        .totalSize;
}

aclsparseStatus_t aclsparseSpSV_bufferSize(aclsparseHandle_t handle, aclsparseOperation_t opA,
                                           const void* alpha, aclsparseConstSpMatDescr_t matA,
                                           aclsparseConstDnVecDescr_t vecX, aclsparseDnVecDescr_t vecY,
                                           aclDataType computeType, aclsparseSpSVAlg_t alg,
                                           aclsparseSpSVDescr_t spsvDescr, size_t* bufferSize)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpSV_bufferSize", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (bufferSize == nullptr) {
        OP_LOGE("aclsparseSpSV_bufferSize", "bufferSize is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclsparseSpSV_bufferSize", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st = ValidateOpA("aclsparseSpSV_bufferSize", opA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    st = ValidateSpSVCommonParams("aclsparseSpSV_bufferSize", matA, spsvDescr, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateSpSVDenseVecTypes("aclsparseSpSV_bufferSize", matA, vecX, vecY, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    *bufferSize = SpSVWorkspaceSizeForMatrix(matA, opA);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpSV_analysis(aclsparseHandle_t handle, aclsparseOperation_t opA,
                                         const void* alpha, aclsparseConstSpMatDescr_t matA,
                                         aclsparseConstDnVecDescr_t vecX, aclsparseDnVecDescr_t vecY,
                                         aclDataType computeType, aclsparseSpSVAlg_t alg,
                                         aclsparseSpSVDescr_t spsvDescr, void* externalBuffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpSV_analysis", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclsparseSpSV_analysis", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st = ValidateOpA("aclsparseSpSV_analysis", opA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    st = ValidateSpSVCommonParams("aclsparseSpSV_analysis", matA, spsvDescr, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateSpSVDenseVecTypes("aclsparseSpSV_analysis", matA, vecX, vecY, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    auto* mat = spsv::ToMatInner(matA);
    CacheMatrixAttributes(matA, opA, spsvDescr);
    if (mat->nnz == 0) {
        spsvDescr->analysisLaunched = true;
        spsvDescr->workspaceBuffer = nullptr;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    if (externalBuffer == nullptr) {
        OP_LOGE("aclsparseSpSV_analysis", "externalBuffer is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return LaunchSpSVAnalysisKernel(handle, opA, alpha, matA, spsvDescr, externalBuffer);
}

aclsparseStatus_t aclsparseSpSV_solve(aclsparseHandle_t handle, aclsparseOperation_t opA, const void* alpha,
                                      aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
                                      aclsparseDnVecDescr_t vecY, aclDataType computeType,
                                      aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st = ValidateOpA("aclsparseSpSV_solve", opA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    st = ValidateSpSVCommonParams("aclsparseSpSV_solve", matA, spsvDescr, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (vecX == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "vecX is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecY == nullptr) {
        OP_LOGE("aclsparseSpSV_solve", "vecY is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    st = ValidateSpSVDenseVecTypes("aclsparseSpSV_solve", matA, vecX, vecY, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (!spsvDescr->analysisLaunched) {
        OP_LOGE("aclsparseSpSV_solve", "analysis not launched, call aclsparseSpSV_analysis first");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    st = ValidateSpSVSolveConsistency("aclsparseSpSV_solve", opA, matA, spsvDescr);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (spsvDescr->cachedNnz == 0) {
        return HandleSolveZeroNnz(handle, alpha, vecX, vecY, spsvDescr);
    }

    return LaunchSpSVSolveKernel(handle, opA, alpha, matA, vecX, vecY, spsvDescr);
}

aclsparseStatus_t aclsparseSpSV_updateMatrix(aclsparseHandle_t handle, aclsparseSpSVDescr_t spsvDescr,
                                             void* newValues, aclsparseSpSVUpdate_t updatePart)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpSV_updateMatrix", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (spsvDescr == nullptr) {
        OP_LOGE("aclsparseSpSV_updateMatrix", "spsvDescr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (newValues == nullptr) {
        OP_LOGE("aclsparseSpSV_updateMatrix", "newValues is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!spsvDescr->analysisLaunched) {
        OP_LOGE("aclsparseSpSV_updateMatrix", "analysis not launched");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (spsvDescr->cachedNnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (spsvDescr->workspaceBuffer == nullptr) {
        OP_LOGE("aclsparseSpSV_updateMatrix", "workspaceBuffer is nullptr after analysis");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto* h = spsv::ToInternalHandle(handle);
    aclrtStream useStream = h->stream;

    SpsvTilingData tiling{};
    InitUpdateTilingData(spsvDescr, tiling);

    uint8_t* ws = reinterpret_cast<uint8_t*>(spsvDescr->workspaceBuffer);
    uint8_t* newVals = reinterpret_cast<uint8_t*>(newValues);

    // clear-then-launch-then-peek（评审响应）：Peek 只读不清除，陈旧线程级错误
    // 会被归因本次下发且一次失败后永久粘滞；下发前 aclrtGetLastError 清除
    // （!424 spsm 分片已实证直接 Peek 误伤 10 例）。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    if (updatePart == ACL_SPARSE_SPSV_UPDATE_GENERAL) {
        UpdateMatrixGeneral(spsvDescr, newValues, ws, newVals, tiling, useStream);
    } else if (updatePart == ACL_SPARSE_SPSV_UPDATE_DIAGONAL) {
        UpdateMatrixDiagonal(spsvDescr, ws, newVals, tiling, useStream);
    } else {
        OP_LOGE("aclsparseSpSV_updateMatrix", "unsupported updatePart=%d", updatePart);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // UpdateMatrixGeneral/Diagonal 为 void helper，含 5 个内部下发点；
    // 入口单点检查覆盖全部（issue #357）。
    // 检查必须先于 updateMatrixCalled 置位（评审响应）：失败时若标志已置，
    // ValidateSpSVSolveConsistency 的 !updateMatrixCalled 放行条件被假标志
    // 绕过，后续 solve 用陈旧 workspace 副本静默算错。与 analysis 路径
    // （analysisLaunched 置位前检查）口径一致。
    const aclError updateEnqueueStatus = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (updateEnqueueStatus != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpSV_updateMatrix", "kernel enqueue failed: %d",
                static_cast<int>(updateEnqueueStatus));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    spsvDescr->updateMatrixCalled = true;

    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // extern "C"
