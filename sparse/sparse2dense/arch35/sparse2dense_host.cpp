/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file sparse2dense_host.cpp
 * \brief SparseToDense Host 侧实现：2 个 Generic API 入口。
 *
 * API 清单：
 *   F1. aclsparseSparseToDense_bufferSize — 查询 workspace 大小
 *   F2. aclsparseSparseToDense            — 执行 Sparse→Dense 转换
 *
 * 支持 CSR / CSC / COO；值类型 FP32/FP16/BF16/INT32/INT8/COMPLEX64。
 * arch35 使用 SIMT（对标 gather/scatter/densetosparse），对标 cuSPARSE SparseToDense。
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "sparse2dense.h"
#include "sparse2dense_tiling_data.h"
#include "sparse2dense_kernel.h"

static constexpr const char *kTag = "aclsparseSparseToDense";

namespace {

// ===========================================================================
// 公共参数校验
// ===========================================================================

static aclsparseStatus_t ValidateSparseDescriptor(const aclsparseSpMatDescr *matInner)
{
    if (matInner->format != ACL_SPARSE_FORMAT_CSR &&
        matInner->format != ACL_SPARSE_FORMAT_CSC &&
        matInner->format != ACL_SPARSE_FORMAT_COO) {
        OP_LOGE(kTag, "unsupported format %d (CSR/CSC/COO required)",
                static_cast<int>(matInner->format));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    aclsparseStatus_t idxSt =
        AclsparseValidateSupportedCsrIndexTypes(matInner->ptrType, matInner->IdxType);
    if (idxSt != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kTag, "unsupported index type ptr=%d idx=%d (only ACL_SPARSE_INDEX_32I)",
                matInner->ptrType, matInner->IdxType);
        return idxSt;
    }
    if (matInner->baseType != ACL_SPARSE_INDEX_BASE_ZERO &&
        matInner->baseType != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(kTag, "unsupported indexBase %d", static_cast<int>(matInner->baseType));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclDataType valType = matInner->valueType;
    if (valType != ACL_FLOAT && valType != ACL_FLOAT16 && valType != ACL_BF16 &&
        valType != ACL_INT32 && valType != ACL_INT8 && valType != ACL_COMPLEX64) {
        OP_LOGE(kTag, "unsupported valueType %d", static_cast<int>(valType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matInner->rows > static_cast<uint64_t>(INT32_MAX) ||
        matInner->cols > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(kTag, "m or n exceeds INT32_MAX, not supported");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Shape/ld/dtype only — used by bufferSize (values may still be nullptr).
static aclsparseStatus_t ValidateDenseDescriptorMeta(
    const aclsparseDnMatDescr *dnInner, aclDataType valType,
    uint64_t sparseRows, uint64_t sparseCols)
{
    if (dnInner->valueType != valType) {
        OP_LOGE(kTag, "matB valueType %d != matA valueType %d",
                static_cast<int>(dnInner->valueType), static_cast<int>(valType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (dnInner->rows != static_cast<int64_t>(sparseRows) ||
        dnInner->cols != static_cast<int64_t>(sparseCols)) {
        OP_LOGE(kTag, "dimension mismatch: matA(%llu x %llu) vs matB(%ld x %ld)",
                static_cast<unsigned long long>(sparseRows),
                static_cast<unsigned long long>(sparseCols),
                dnInner->rows, dnInner->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (dnInner->ld < 0) {
        OP_LOGE(kTag, "invalid ld=%ld", dnInner->ld);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (dnInner->ld > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE(kTag, "ld=%ld exceeds INT32_MAX", dnInner->ld);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    bool isColMajor = (dnInner->order == ACL_SPARSE_ORDER_COL);
    int64_t minLd = isColMajor ? dnInner->rows : dnInner->cols;
    if (dnInner->ld < minLd) {
        OP_LOGE(kTag, "ld=%ld < %s minimum %ld",
                dnInner->ld, isColMajor ? "rows" : "cols", minLd);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateDenseDescriptorForExecute(
    const aclsparseDnMatDescr *dnInner, aclDataType valType,
    uint64_t sparseRows, uint64_t sparseCols)
{
    aclsparseStatus_t st =
        ValidateDenseDescriptorMeta(dnInner, valType, sparseRows, sparseCols);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (dnInner->values == nullptr) {
        OP_LOGE(kTag, "matB.values is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSparseToDenseCommon(
    aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB, bool requireDenseValues)
{
    if (handle == nullptr) {
        OP_LOGE(kTag, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matA == nullptr || matB == nullptr) {
        OP_LOGE(kTag, "matA or matB is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *matInner = Sparse2DenseToConstMatInner(matA);
    aclsparseStatus_t st = ValidateSparseDescriptor(matInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    auto *dnInner = Sparse2DenseToDnMatInner(matB);
    if (requireDenseValues) {
        return ValidateDenseDescriptorForExecute(
            dnInner, matInner->valueType, matInner->rows, matInner->cols);
    }
    return ValidateDenseDescriptorMeta(
        dnInner, matInner->valueType, matInner->rows, matInner->cols);
}

// ===========================================================================
// SIMT tiling + launch
// ===========================================================================

// Physical plane: ROW → m*ld, COL → n*ld (majorDim * ld * esize).
static inline size_t DenseStorageBytes(int32_t m, int32_t n, int32_t ld,
                                       bool isColMajor, size_t elemSize)
{
    const int64_t majorDim = isColMajor ? static_cast<int64_t>(n)
                                        : static_cast<int64_t>(m);
    return static_cast<size_t>(majorDim * static_cast<int64_t>(ld)) * elemSize;
}

static aclsparseStatus_t ComputeSimtTiling(
    const aclsparseSpMatDescr *matInner, int32_t m, int32_t n,
    Sparse2DenseTilingData &tiling, uint32_t &useBlocks)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE(kTag, "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    // CSR/CSC/COO scatter is nnz-parallel; size blocks by nnz (and dense zero words).
    if (matInner->nnz > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(kTag, "nnz=%llu exceeds INT32_MAX",
                static_cast<unsigned long long>(matInner->nnz));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    uint64_t workItems = matInner->nnz;
    (void)m;
    (void)n;
    // Dense zero dominates P scenes; size blocks by max(nnz-axis, dense uint64 words).
    const uint64_t denseWords = tiling.denseBytes >> 3;
    if (denseWords > workItems) {
        workItems = denseWords;
    }
    if (workItems == 0 && tiling.denseBytes == 0) {
        useBlocks = 0;
        tiling.numBlocks = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (workItems == 0) {
        workItems = 1;
    }
    useBlocks = static_cast<uint32_t>(std::min<uint64_t>(
        CeilDiv<uint64_t>(workItems, kSparse2DenseMaxThreadsPerBlock), aivCoreNum));
    if (useBlocks == 0) {
        useBlocks = 1;
    }
    tiling.numBlocks = useBlocks;
    OP_LOGD(kTag, "SIMT tiling: fmt=%d workItems=%llu denseBytes=%llu blocks=%u threads=%u",
            tiling.format, static_cast<unsigned long long>(workItems),
            static_cast<unsigned long long>(tiling.denseBytes), useBlocks,
            kSparse2DenseMaxThreadsPerBlock);
    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

aclsparseStatus_t aclsparseSparseToDense_bufferSize(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    size_t *bufferSize)
{
    // Workspace size does not depend on dense->values; allow values==nullptr.
    aclsparseStatus_t st = ValidateSparseToDenseCommon(handle, matA, matB, false);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    if (bufferSize == nullptr) {
        OP_LOGE(kTag, "bufferSize is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (alg != ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT) {
        OP_LOGE(kTag, "unsupported alg %d", static_cast<int>(alg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    *bufferSize = 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ZeroDenseOutput(const aclsparseDnMatDescr *dnInner,
                                          size_t dnMatBytes, aclrtStream stream)
{
    aclError ret = aclrtMemsetAsync(dnInner->values, dnMatBytes, 0, dnMatBytes, stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(kTag, "aclrtMemsetAsync failed, ret=%d", ret);
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateAndZeroOutput(const aclsparseSpMatDescr *matInner,
                                                const aclsparseDnMatDescr *dnInner,
                                                int32_t m, int32_t n,
                                                size_t elemSize, size_t dnMatBytes,
                                                aclrtStream stream)
{
    (void)elemSize;
    if (m == 0 || n == 0) {
        if (dnMatBytes > 0) {
            aclsparseStatus_t zst = ZeroDenseOutput(dnInner, dnMatBytes, stream);
            if (zst != ACL_SPARSE_STATUS_SUCCESS) { return zst; }
        }
        OP_LOGD(kTag, "empty matrix, output zeroed, skip kernel launch");
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (matInner->nnz == 0) {
        aclsparseStatus_t zst = ZeroDenseOutput(dnInner, dnMatBytes, stream);
        if (zst != ACL_SPARSE_STATUS_SUCCESS) { return zst; }
        OP_LOGD(kTag, "nnz=0, output zeroed, skip kernel launch");
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (matInner->ptrs == nullptr) {
        OP_LOGE(kTag, "sparse ptrs is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matInner->idxs == nullptr || matInner->values == nullptr) {
        OP_LOGE(kTag, "idxs or values is null (nnz=%llu > 0)",
                static_cast<unsigned long long>(matInner->nnz));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Hot path: kernel SIMT uint64 zero replaces host aclrtMemsetAsync.
    (void)dnInner;
    (void)stream;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchSparseToDenseKernel(
    const aclsparseSpMatDescr *matInner, const aclsparseDnMatDescr *dnInner,
    int32_t m, int32_t n, bool isColMajor, size_t elemSize, size_t dnMatBytes,
    aclrtStream stream)
{
    Sparse2DenseTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.indexBase = static_cast<int32_t>(matInner->baseType);
    tiling.valueType = Sparse2DenseValTypeFromAcl(matInner->valueType);
    tiling.isColMajor = isColMajor ? 1 : 0;
    tiling.ld = static_cast<int32_t>(dnInner->ld);
    tiling.format = Sparse2DenseFormatFromAcl(matInner->format);
    tiling.nnz = matInner->nnz;
    tiling.denseBytes = static_cast<uint64_t>(dnMatBytes);
    (void)elemSize;

    uint32_t useBlocks = 0;
    aclsparseStatus_t st = ComputeSimtTiling(matInner, m, n, tiling, useBlocks);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (useBlocks == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    sparse2dense_kernel_do(
        reinterpret_cast<GM_ADDR>(matInner->ptrs),
        reinterpret_cast<GM_ADDR>(matInner->idxs),
        reinterpret_cast<GM_ADDR>(matInner->values),
        reinterpret_cast<GM_ADDR>(dnInner->values),
        tiling, useBlocks, stream);

    OP_LOGI(kTag,
            "SIMT launched: m=%d n=%d fmt=%d blocks=%u nnz=%llu valType=%d order=%s",
            m, n, tiling.format, useBlocks,
            static_cast<unsigned long long>(tiling.nnz), tiling.valueType,
            tiling.isColMajor ? "col-major" : "row-major");
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSparseToDense(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    void *buffer)
{
    // Async on handle stream: no SynchronizeStream here; caller syncs.
    // Invalid device coords / COO duplicates are undefined (cuSPARSE-like):
    // scatter is last-write-wins; no sync-back status.
    aclsparseStatus_t st = ValidateSparseToDenseCommon(handle, matA, matB, true);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    if (alg != ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT) {
        OP_LOGE(kTag, "unsupported alg %d", static_cast<int>(alg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    (void)buffer;

    auto *h = Sparse2DenseToInternalHandle(handle);
    auto *matInner = Sparse2DenseToConstMatInner(matA);
    auto *dnInner = Sparse2DenseToDnMatInner(matB);

    aclrtStream stream = h->stream;
    if (stream == nullptr) {
        OP_LOGE(kTag, "stream is nullptr, please call aclsparseSetStream first");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    int32_t m = static_cast<int32_t>(matInner->rows);
    int32_t n = static_cast<int32_t>(matInner->cols);
    size_t elemSize = AclDataTypeSize(matInner->valueType);
    const bool isColMajor = (dnInner->order == ACL_SPARSE_ORDER_COL);
    size_t dnMatBytes = DenseStorageBytes(m, n, static_cast<int32_t>(dnInner->ld),
                                          isColMajor, elemSize);

    aclsparseStatus_t zst = ValidateAndZeroOutput(matInner, dnInner, m, n,
                                                   elemSize, dnMatBytes, stream);
    if (zst != ACL_SPARSE_STATUS_SUCCESS) { return zst; }
    if (m == 0 || n == 0 || matInner->nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    return LaunchSparseToDenseKernel(matInner, dnInner, m, n, isColMajor, elemSize,
                                     dnMatBytes, stream);
}

}  // extern "C"
