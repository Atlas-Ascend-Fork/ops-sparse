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
 * \file scatter_host.cpp
 * \brief aclsparseScatter Host 侧实现：参数校验 + Kernel launch。
 *
 * 结构：aclsparseScatter 内部拆分为 ValidateScatterParams + LaunchScatterKernel。
 * 不校验 indices 值合法性（host 侧无法读 device 内存，与 cuSPARSE 一致，越界索引行为未定义）。
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "scatter.h"
#include "scatter_tiling_data.h"
#include "scatter_kernel.h"

static constexpr const char *kTag = "aclsparseScatter";

namespace {

static aclsparseStatus_t ValidateSpVecDescr(aclsparseConstSpVecDescr_t vecX,
                                            const aclsparseSpVecDescr *&xInner)
{
    if (vecX == nullptr) {
        OP_LOGE(kTag, "vecX is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    xInner = ScatterToSpVecInner(vecX);
    if (xInner->valueType != ACL_FLOAT &&
        xInner->valueType != ACL_FLOAT16 &&
        xInner->valueType != ACL_BF16 &&
        xInner->valueType != ACL_INT8 &&
        xInner->valueType != ACL_COMPLEX64) {
        OP_LOGE(kTag, "unsupported vecX.valueType %d (FP32/FP16/BF16/INT8/COMPLEX64 only)",
                static_cast<int>(xInner->valueType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (xInner->idxType != ACL_SPARSE_INDEX_32I &&
        xInner->idxType != ACL_SPARSE_INDEX_64I) {
        OP_LOGE(kTag, "unsupported vecX.idxType %d (I32/I64 only)",
                static_cast<int>(xInner->idxType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (xInner->idxBase != ACL_SPARSE_INDEX_BASE_ZERO &&
        xInner->idxBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(kTag, "unsupported vecX.idxBase %d",
                static_cast<int>(xInner->idxBase));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (xInner->nnz > xInner->size) {
        OP_LOGE(kTag, "vecX.nnz=%llu > vecX.size=%llu",
                static_cast<unsigned long long>(xInner->nnz),
                static_cast<unsigned long long>(xInner->size));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (xInner->nnz > 0) {
        if (xInner->indices == nullptr) {
            OP_LOGE(kTag, "vecX.indices is nullptr (nnz=%llu > 0)",
                    static_cast<unsigned long long>(xInner->nnz));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        if (xInner->values == nullptr) {
            OP_LOGE(kTag, "vecX.values is nullptr (nnz=%llu > 0)",
                    static_cast<unsigned long long>(xInner->nnz));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateDnVecAndCompatibility(
    aclsparseDnVecDescr_t vecY,
    const aclsparseSpVecDescr *xInner)
{
    if (vecY == nullptr) {
        OP_LOGE(kTag, "vecY is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const aclsparseDnVecDescr *yInner = ScatterToDnVecInner(vecY);
    if (yInner->valueType != xInner->valueType) {
        OP_LOGE(kTag, "valueType mismatch: vecX=%d, vecY=%d",
                static_cast<int>(xInner->valueType),
                static_cast<int>(yInner->valueType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (xInner->size > yInner->nums) {
        OP_LOGE(kTag, "vecX.size=%llu > vecY.nums=%llu",
                static_cast<unsigned long long>(xInner->size),
                static_cast<unsigned long long>(yInner->nums));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (xInner->nnz > 0 && yInner->values == nullptr) {
        OP_LOGE(kTag, "vecY.values is nullptr (nnz=%llu > 0)",
                static_cast<unsigned long long>(xInner->nnz));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 任务书 §2.4：未声明的输入输出 alias 返回参数错误
    if (xInner->nnz > 0 && yInner->values != nullptr) {
        if (xInner->values == yInner->values || xInner->indices == yInner->values) {
            OP_LOGE(kTag, "undeclared Device pointer alias between vecX and vecY.values");
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateScatterParams(
    aclsparseConstSpVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY)
{
    const aclsparseSpVecDescr *xInner = nullptr;
    aclsparseStatus_t st = ValidateSpVecDescr(vecX, xInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return ValidateDnVecAndCompatibility(vecY, xInner);
}

static int32_t MapValueType(aclDataType valueType)
{
    switch (valueType) {
        case ACL_FLOAT:
            return SCATTER_VAL_FP32;
        case ACL_FLOAT16:
            return SCATTER_VAL_FP16;
        case ACL_BF16:
            return SCATTER_VAL_BF16;
        case ACL_INT8:
            return SCATTER_VAL_INT8;
        case ACL_COMPLEX64:
            return SCATTER_VAL_COMPLEX64;
        default:
            break;
    }
    return SCATTER_VAL_FP32;
}

static aclsparseStatus_t BuildScatterTiling(
    const aclsparseSpVecDescr *xInner, ScatterTilingData &tiling)
{
    constexpr uint64_t kScatterNnzUpperLimit =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    if (xInner->nnz > kScatterNnzUpperLimit) {
        OP_LOGE(kTag, "nnz=%llu exceeds UINT32_MAX, not supported",
                static_cast<unsigned long long>(xInner->nnz));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE(kTag, "GetAivCoreCount returned 0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    const uint32_t nnzU32 = static_cast<uint32_t>(xInner->nnz);
    const uint32_t threadsPerBlock = (xInner->nnz < kScatterSmallNnzThreshold)
                                         ? kScatterSmallThreadsPerBlock
                                         : kScatterMaxThreadsPerBlock;
    uint32_t useNumBlocks = std::min(aivCoreNum,
        CeilDiv<uint32_t>(nnzU32, threadsPerBlock));
    if (useNumBlocks == 0) {
        useNumBlocks = 1;
    }

    tiling.nnz = xInner->nnz;
    tiling.idxBase = (xInner->idxBase == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    tiling.idxType = (xInner->idxType == ACL_SPARSE_INDEX_64I)
                     ? SCATTER_IDX_I64 : SCATTER_IDX_I32;
    tiling.valType = MapValueType(xInner->valueType);
    tiling.threadsPerBlock = threadsPerBlock;
    tiling.numBlocks = useNumBlocks;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchScatterKernel(
    aclsparseHandle_t handle,
    aclsparseConstSpVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY)
{
    auto *h = ScatterToInternalHandle(handle);
    aclrtStream stream = h->stream;
    if (stream == nullptr) {
        OP_LOGE(kTag, "stream is nullptr, please call aclsparseSetStream first");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    const aclsparseSpVecDescr *xInner = ScatterToSpVecInner(vecX);
    const aclsparseDnVecDescr *yInner = ScatterToDnVecInner(vecY);
    if (xInner->nnz == 0) {
        OP_LOGD(kTag, "nnz=0, skip kernel launch");
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    ScatterTilingData tiling{};
    aclsparseStatus_t st = BuildScatterTiling(xInner, tiling);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    OP_LOGI(kTag,
            "launching scatter kernel: nnz=%llu, numBlocks=%u, threads=%u, "
            "idxBase=%d, idxType=%d, valType=%d",
            static_cast<unsigned long long>(tiling.nnz), tiling.numBlocks,
            tiling.threadsPerBlock, tiling.idxBase, tiling.idxType, tiling.valType);

    scatter_kernel_do(
        reinterpret_cast<GM_ADDR>(xInner->indices),
        reinterpret_cast<GM_ADDR>(xInner->values),
        reinterpret_cast<GM_ADDR>(yInner->values),
        tiling, stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

extern "C" aclsparseStatus_t aclsparseScatter(
    aclsparseHandle_t handle,
    aclsparseConstSpVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY)
{
    if (handle == nullptr) {
        OP_LOGE(kTag, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = ValidateScatterParams(vecX, vecY);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return LaunchScatterKernel(handle, vecX, vecY);
}
