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

#include "cann_ops_sparse.h"
#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <new>
#include <string.h>
#include <vector>
#include "unistd.h"
#include "spvv_tiling_data.h"
#include "spvv_kernel.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"
#include "tiling/platform/platform_ascendc.h"

// ---------------------------------------------------------------------------
// Validate aclsparseSpvv parameters. Indices ordering/uniqueness/range and
// nnz <= yLen are NOT checked (would require a D2H copy); caller must guarantee.
// Returns ACL_SPARSE_STATUS_SUCCESS on success, else an error status.
// ---------------------------------------------------------------------------
static aclsparseStatus_t ValidateSpvvParams(aclsparseHandle_t handle, aclsparseOperation_t op,
    aclsparseSpVecDescr *xInner, aclsparseDnVecDescr *yInner, void *result,
    aclDataType computeType)
{
    if (handle == nullptr) {
        fprintf(stderr, "aclsparseSpvv: handle is nullptr\n");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (xInner == nullptr || yInner == nullptr || result == nullptr) {
        fprintf(stderr, "aclsparseSpvv: x/y/result must be non-null\n");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (op != ACL_SPARSE_OP_NON_TRANSPOSE) {
        fprintf(stderr, "aclsparseSpvv: op %d not supported (only NON_TRANSPOSE)\n",
            static_cast<int>(op));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (computeType != ACL_FLOAT) {
        fprintf(stderr, "aclsparseSpvv: computeType %d not supported (only FP32)\n",
            static_cast<int>(computeType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (xInner->idxType != ACL_SPARSE_INDEX_32I) {
        fprintf(stderr, "aclsparseSpvv: idxType not supported (only INDEX_32I)\n");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (xInner->idxBase != ACL_SPARSE_INDEX_BASE_ZERO) {
        fprintf(stderr, "aclsparseSpvv: idxBase not supported (only BASE_ZERO)\n");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (xInner->valueType != ACL_FLOAT && xInner->valueType != ACL_FLOAT16) {
        fprintf(stderr, "aclsparseSpvv: x valueType %d not supported (FP32/FP16)\n",
            static_cast<int>(xInner->valueType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (yInner->valueType != xInner->valueType) {
        fprintf(stderr, "aclsparseSpvv: y valueType %d != x valueType %d\n",
            static_cast<int>(yInner->valueType), static_cast<int>(xInner->valueType));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// aclsparseSpvv — execute sparse vector - dense vector dot product
//
// Computes: result = sum(x_values[i] * y[x_indices[i]]) for i in [0, nnz)
//
// Currently supported:
//   - op:              ACL_SPARSE_OP_NON_TRANSPOSE only
//   - valueType (x/y): ACL_FLOAT or ACL_FLOAT16 (x and y must match)
//   - computeType:     ACL_FLOAT (accumulation and output are always FP32)
//   - idxType:         ACL_SPARSE_INDEX_32I
//   - idxBase:         ACL_SPARSE_INDEX_BASE_ZERO
//   - indices:         sorted in strictly ascending order (unique), and every
//                      index in [0, yLen-1] — NOT validated (would require a
//                      D2H copy); caller must guarantee. Out-of-range indices
//                      cause out-of-bounds y reads (undefined result/crash).
//   - result:          DEVICE pointer to one float (FP32, regardless of valueType)
//
// Runtime validation covers null pointers, op/computeType/idxType/idxBase/valueType
// compatibility. Indices ordering/uniqueness/range and nnz <= yLen are assumed.
//
// The kernel uses AtomicAdd for cross-core reduction directly to the output.
// No device buffer is needed — tiling data is passed as a kernel parameter.
// ---------------------------------------------------------------------------

aclsparseStatus_t aclsparseSpvv(aclsparseHandle_t handle, aclsparseOperation_t op,
    aclsparseConstSpVecDescr_t x, aclsparseConstDnVecDescr_t y,
    void *result, aclDataType computeType)
{
    aclsparseSpVecDescr *xInner = const_cast<aclsparseSpVecDescr *>(x);
    aclsparseDnVecDescr *yInner = const_cast<aclsparseDnVecDescr *>(y);

    aclsparseStatus_t st = ValidateSpvvParams(handle, op, xInner, yInner, result, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    uint32_t nnz = static_cast<uint32_t>(xInner->nnz);
    uint64_t yLen = yInner->nums;

    aclrtStream stream = nullptr;
    st = aclsparseGetStream(handle, &stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "aclsparseSpvv: aclsparseGetStream failed, st=%d\n",
            static_cast<int>(st));
        return st;
    }

    // Edge case: no non-zero elements — zero the result
    if (nnz == 0) {
        CHECK_ACL(aclrtMemsetAsync(result, sizeof(float), 0, sizeof(float), stream));
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    // ---- Compute tiling ----
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum == 0) {
        fprintf(stderr, "aclsparseSpvv: GetCoreNumAiv returned 0\n");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    // Keep at least SPVV_MIN_NNZ_PER_CORE per core; blockNum is the active core count.
    uint32_t nnzPerCore = (nnz + coreNum - 1) / coreNum;
    if (nnzPerCore < SPVV_MIN_NNZ_PER_CORE) {
        nnzPerCore = SPVV_MIN_NNZ_PER_CORE;
    }
    uint32_t blockNum = (nnz + nnzPerCore - 1) / nnzPerCore;

    SpvvTilingData tiling{nnz, yLen, nnzPerCore};

    // Zero result on `stream` before launch: every core AtomicAdds into a known-zero output.
    CHECK_ACL(aclrtMemsetAsync(result, sizeof(float), 0, sizeof(float), stream));

    spvv_kernel_do(reinterpret_cast<GM_ADDR>(xInner->indices),
        reinterpret_cast<GM_ADDR>(xInner->values),
        reinterpret_cast<GM_ADDR>(yInner->values),
        reinterpret_cast<GM_ADDR>(result),
        tiling, blockNum, xInner->valueType, stream);

    return ACL_SPARSE_STATUS_SUCCESS;
}
