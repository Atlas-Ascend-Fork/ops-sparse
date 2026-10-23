/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*
 * aclsparseDenseToSparse arch22 host protocol.
 *
 * Three-stage flow (cuSPARSE DenseToSparse semantics): GetBufferSize is a
 * pure query; Analysis counts the nonzero structure, publishes nnz and the
 * workspace prefix, and registers the workspace buffer on the descriptor;
 * Convert requires the very same buffer and writes the payload trusting
 * the published prefix (structure is never recounted). Device indices are
 * I32-only (64I rejected); complex64 values are modeled as 8-byte elements.
 */

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "cann_ops_sparse.h"
#include "densetosparse_kernel.h"
#include "densetosparse_tiling_data.h"
#include "log/log.h"

namespace {

constexpr const char *kApi = "aclsparseDenseToSparse";
constexpr size_t kAlignment = 32;

bool IsIndexTypeValid(aclsparseIndexType_t type)
{
    return type == ACL_SPARSE_INDEX_32I;
}

bool IsIndexBaseValid(aclsparseIndexBase_t base)
{
    return base == ACL_SPARSE_INDEX_BASE_ZERO ||
           base == ACL_SPARSE_INDEX_BASE_ONE;
}

uint32_t GetElementBytes(aclDataType type)
{
    if (type == ACL_INT8) {
        return 1;
    }
    if (type == ACL_FLOAT16 || type == ACL_BF16) {
        return 2;
    }
    if (type == ACL_FLOAT) {
        return 4;
    }
    if (type == ACL_COMPLEX64) {
        return 8;
    }
    return 0;
}

bool IsSupportedFormat(aclsparseFormat_t format)
{
    return format == ACL_SPARSE_FORMAT_CSR || format == ACL_SPARSE_FORMAT_CSC ||
           format == ACL_SPARSE_FORMAT_COO ||
           format == ACL_SPARSE_FORMAT_BLOCKED_ELL;
}

bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t *result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}
// Ceil-padded block grid task count of a BELL descriptor (single source
// for the launch sizing and the uint32-partition bound check).
bool BellTaskCount(const aclsparseSpMatDescr *sparse, uint64_t *tasks)
{
    return CheckedMultiply(
        (sparse->rows + sparse->ellBlockSize - 1) / sparse->ellBlockSize,
        (sparse->ellCols + sparse->ellBlockSize - 1) / sparse->ellBlockSize,
        tasks);
}


aclsparseStatus_t ValidateDenseSpan(const aclsparseDnMatDescr *dense,
                                    uint32_t elementBytes)
{
    if (dense->rows < 0 || dense->cols < 0 || dense->ld < 0 ||
        (dense->order != ACL_SPARSE_ORDER_ROW &&
         dense->order != ACL_SPARSE_ORDER_COL)) {
        OP_LOGE(kApi,
                "invalid dense descriptor: rows=%" PRId64 ", cols=%" PRId64
                ", ld=%" PRId64 ", order=%d",
                dense->rows, dense->cols, dense->ld,
                static_cast<int>(dense->order));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (dense->rows > 0 && dense->cols > 0 &&
        ((dense->order == ACL_SPARSE_ORDER_ROW && dense->ld < dense->cols) ||
         (dense->order == ACL_SPARSE_ORDER_COL && dense->ld < dense->rows))) {
        const int64_t minLd =
            dense->order == ACL_SPARSE_ORDER_ROW ? dense->cols : dense->rows;
        OP_LOGE(kApi,
                "invalid dense leading dimension: ld=%" PRId64
                ", expected >= %" PRId64 " for order=%d",
                dense->ld, minLd, static_cast<int>(dense->order));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (dense->rows == 0 || dense->cols == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const uint64_t major = dense->order == ACL_SPARSE_ORDER_ROW
                               ? static_cast<uint64_t>(dense->rows - 1)
                               : static_cast<uint64_t>(dense->cols - 1);
    const uint64_t minor = dense->order == ACL_SPARSE_ORDER_ROW
                               ? static_cast<uint64_t>(dense->cols - 1)
                               : static_cast<uint64_t>(dense->rows - 1);
    uint64_t address = 0;
    if (!CheckedMultiply(major, static_cast<uint64_t>(dense->ld), &address) ||
        address > std::numeric_limits<uint64_t>::max() - minor) {
        OP_LOGE(kApi,
                "dense element offset overflows: major=%" PRIu64
                ", minor=%" PRIu64 ", ld=%" PRId64,
                major, minor, dense->ld);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    address += minor;
    uint64_t bytes = 0;
    if (!CheckedMultiply(address + 1, elementBytes, &bytes) ||
        bytes > std::numeric_limits<size_t>::max()) {
        OP_LOGE(kApi,
                "dense byte span overflows: lastOffset=%" PRIu64
                ", elementBytes=%u",
                address, elementBytes);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// The values payload spans the ceil-padded block grid: tail block
// rows/cols hold dtype-positive zeros; nnz must match and the payload
// pointers must be present for non-empty grids.
aclsparseStatus_t ValidateBellPayload(const aclsparseSpMatDescr *sparse)
{
    const uint64_t paddedRows =
        (sparse->rows + sparse->ellBlockSize - 1) / sparse->ellBlockSize *
        sparse->ellBlockSize;
    uint64_t valueCount = 0;
    if (!CheckedMultiply(paddedRows, sparse->ellCols, &valueCount)) {
        OP_LOGE(kApi,
                "BELL value count overflows: rows=%" PRIu64
                ", ellCols=%" PRIu64,
                sparse->rows, sparse->ellCols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (valueCount != sparse->nnz) {
        OP_LOGE(kApi,
                "invalid BELL nnz=%" PRIu64
                ", expected rows * ellCols=%" PRIu64,
                sparse->nnz, valueCount);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (valueCount > 0 &&
        (sparse->values == nullptr || sparse->ellColInd == nullptr)) {
        OP_LOGE(kApi,
                "BELL payload pointer is nullptr while valueCount=%" PRIu64,
                valueCount);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// The convert kernel partitions the task grid in uint32; reject grids
// that would overflow the partition (b=1 with a >2^32 block grid). The
// margin below 2^32 keeps the per-core (core * ceil(tasks / cores))
// range math inside uint32 as well.
aclsparseStatus_t ValidateBellTaskGrid(const aclsparseSpMatDescr *sparse)
{
    constexpr uint64_t kBellMaxTasks = 0xFFFFFFF0ULL;
    uint64_t tasks = 0;
    if (!BellTaskCount(sparse, &tasks) || tasks > kBellMaxTasks) {
        OP_LOGE(kApi,
                "blocked-ell task grid exceeds the uint32 kernel "
                "partition: rows=%" PRIu64 ", ellCols=%" PRIu64
                ", blockSize=%" PRIu64,
                sparse->rows, sparse->ellCols, sparse->ellBlockSize);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateBell(const aclsparseSpMatDescr *sparse)
{
    aclsparseStatus_t status = ACL_SPARSE_STATUS_SUCCESS;
    if (sparse->ellBlockSize == 0) {
        OP_LOGE(kApi, "invalid BELL ellBlockSize=0");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (sparse->ellBlockSize >
        static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE(kApi, "BELL ellBlockSize=%" PRIu64 " exceeds INT32_MAX",
                sparse->ellBlockSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // ellCols is bounded by the ceil-padded column extent (enforced at
    // descriptor creation); trailing block columns hold positive zeros.
    if (sparse->ellCols % sparse->ellBlockSize != 0) {
        OP_LOGE(kApi,
                "invalid BELL ellCols=%" PRIu64
                " not divisible by ellBlockSize=%" PRIu64,
                sparse->ellCols, sparse->ellBlockSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    status = ValidateBellTaskGrid(sparse);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    return ValidateBellPayload(sparse);
}

aclsparseStatus_t ValidateDescriptorConfiguration(
    const aclsparseDnMatDescr *dense, const aclsparseSpMatDescr *sparse,
    aclsparseDenseToSparseAlg_t alg, uint32_t *elementBytes)
{
    if (alg != ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT) {
        OP_LOGE(kApi, "invalid algorithm=%d, only default is supported",
                static_cast<int>(alg));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsSupportedFormat(sparse->format)) {
        OP_LOGE(kApi, "invalid sparse format=%d",
                static_cast<int>(sparse->format));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsIndexTypeValid(sparse->ptrType)) {
        OP_LOGE(kApi, "unsupported sparse offset type=%d (I32 only)",
                static_cast<int>(sparse->ptrType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!IsIndexTypeValid(sparse->IdxType)) {
        OP_LOGE(kApi, "unsupported sparse index type=%d (I32 only)",
                static_cast<int>(sparse->IdxType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!IsIndexBaseValid(sparse->baseType)) {
        OP_LOGE(kApi, "invalid sparse index base=%d",
                static_cast<int>(sparse->baseType));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *elementBytes = GetElementBytes(dense->valueType);
    if (*elementBytes == 0) {
        OP_LOGE(kApi, "unsupported dense value type=%d",
                static_cast<int>(dense->valueType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (GetElementBytes(sparse->valueType) != *elementBytes ||
        dense->valueType != sparse->valueType) {
        OP_LOGE(kApi, "value type mismatch: dense=%d, sparse=%d",
                static_cast<int>(dense->valueType),
                static_cast<int>(sparse->valueType));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateDescriptorShape(const aclsparseDnMatDescr *dense,
                                          const aclsparseSpMatDescr *sparse)
{
    if (dense->rows < 0 || dense->cols < 0) {
        OP_LOGE(kApi, "invalid dense shape: rows=%" PRId64 ", cols=%" PRId64,
                dense->rows, dense->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (sparse->rows != static_cast<uint64_t>(dense->rows) ||
        sparse->cols != static_cast<uint64_t>(dense->cols)) {
        OP_LOGE(kApi,
                "shape mismatch: dense=(%" PRId64 ", %" PRId64
                "), sparse=(%" PRIu64 ", %" PRIu64 ")",
                dense->rows, dense->cols, sparse->rows, sparse->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (sparse->rows >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        sparse->cols >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE(kApi,
                "sparse shape exceeds INT32_MAX: rows=%" PRIu64
                ", cols=%" PRIu64,
                sparse->rows, sparse->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (sparse->rows > 0 && sparse->cols > 0 && dense->values == nullptr) {
        OP_LOGE(kApi, "dense values is nullptr for non-empty matrix");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSparseStorage(const aclsparseSpMatDescr *sparse)
{
    if (sparse->format == ACL_SPARSE_FORMAT_BLOCKED_ELL) {
        return ValidateBell(sparse);
    }
    if ((sparse->format == ACL_SPARSE_FORMAT_CSR ||
         sparse->format == ACL_SPARSE_FORMAT_CSC) &&
        sparse->ptrs == nullptr) {
        OP_LOGE(kApi, "sparse offsets is nullptr for format=%d",
                static_cast<int>(sparse->format));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateStaticParams(aclsparseHandle_t handle,
                                       aclsparseConstDnMatDescr_t matA,
                                       aclsparseSpMatDescr_t matB,
                                       aclsparseDenseToSparseAlg_t alg)
{
    if (handle == nullptr) {
        OP_LOGE(kApi, "handle is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA == nullptr) {
        OP_LOGE(kApi, "matA is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matB == nullptr) {
        OP_LOGE(kApi, "matB is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const auto *dense = matA;
    auto *sparse = matB;
    uint32_t elementBytes = 0;
    aclsparseStatus_t status =
        ValidateDescriptorConfiguration(dense, sparse, alg, &elementBytes);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateDescriptorShape(dense, sparse);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateDenseSpan(dense, elementBytes);
    return status == ACL_SPARSE_STATUS_SUCCESS
               ? ValidateSparseStorage(sparse)
               : status;
}

bool CheckedAddSize(size_t lhs, size_t rhs, size_t *result)
{
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool AlignSize(size_t value, size_t *result)
{
    if (value > std::numeric_limits<size_t>::max() - (kAlignment - 1)) {
        return false;
    }
    *result = (value + kAlignment - 1) & ~(kAlignment - 1);
    return true;
}

aclsparseStatus_t ComputeUnitCount(const aclsparseSpMatDescr *sparse,
                                   aclsparseOrder_t order, uint64_t *unitCount)
{
    uint64_t logical = 0;
    if (!CheckedMultiply(sparse->rows, sparse->cols, &logical) ||
        logical > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE(kApi,
                "logical element count is unsupported: rows=%" PRIu64
                ", cols=%" PRIu64,
                sparse->rows, sparse->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (logical == 0 || sparse->format == ACL_SPARSE_FORMAT_BLOCKED_ELL) {
        *unitCount = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (sparse->format == ACL_SPARSE_FORMAT_COO) {
        if (order == ACL_SPARSE_ORDER_COL) {
            // Column-major storage: adopt the CSR row x chunk units so the
            // column-panel path applies (emission stays row-major).
            const uint64_t chunks =
                (sparse->cols + kD2sMinorChunk - 1) / kD2sMinorChunk;
            if (!CheckedMultiply(sparse->rows, chunks, unitCount)) {
                OP_LOGE(kApi, "workspace unit count overflows");
                return ACL_SPARSE_STATUS_INVALID_VALUE;
            }
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        *unitCount = (logical + kD2sCooTile - 1) / kD2sCooTile;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const uint64_t major =
        sparse->format == ACL_SPARSE_FORMAT_CSC ? sparse->cols : sparse->rows;
    const uint64_t minor =
        sparse->format == ACL_SPARSE_FORMAT_CSC ? sparse->rows : sparse->cols;
    const uint64_t chunks = (minor + kD2sMinorChunk - 1) / kD2sMinorChunk;
    if (!CheckedMultiply(major, chunks, unitCount)) {
        OP_LOGE(kApi,
                "workspace unit count overflows: major=%" PRIu64
                ", chunks=%" PRIu64,
                major, chunks);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Accumulate one scan level's 32B-aligned bytes; false on overflow.
bool AddWorkspaceLevel(uint64_t count, size_t *bytes)
{
    if (count > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
        OP_LOGE(kApi, "workspace level is too large: count=%" PRIu64, count);
        return false;
    }
    size_t levelBytes = 0;
    if (!AlignSize(static_cast<size_t>(count) * sizeof(uint64_t), &levelBytes) ||
        !CheckedAddSize(*bytes, levelBytes, bytes)) {
        OP_LOGE(kApi, "workspace size overflows: count=%" PRIu64, count);
        return false;
    }
    return true;
}

// Workspace = header + one uint64 level per scan depth (each 32B aligned).
aclsparseStatus_t ComputeWorkspace(const aclsparseSpMatDescr *sparse,
                                   aclsparseOrder_t order, size_t *bufferSize)
{
    uint64_t count = 0;
    aclsparseStatus_t status = ComputeUnitCount(sparse, order, &count);
    if (status != ACL_SPARSE_STATUS_SUCCESS || count == 0) {
        *bufferSize = 0;
        return status;
    }
    size_t bytes = kD2sWorkspaceHeaderBytes;
    // One iteration per intermediate scan level (count strictly decreases
    // while above one batch); the final level fits a single batch and is
    // accumulated after the loop.
    while (count > kD2sScanBatch) {
        if (!AddWorkspaceLevel(count, &bytes)) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        count = (count + kD2sScanBatch - 1) / kD2sScanBatch;
    }
    if (!AddWorkspaceLevel(count, &bytes)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *bufferSize = bytes;
    return ACL_SPARSE_STATUS_SUCCESS;
}

uint32_t JudgeTileLen(const aclsparseSpMatDescr *sparse,
                     aclsparseOrder_t order, uint32_t elementBytes)
{
    if (sparse->format == ACL_SPARSE_FORMAT_COO) {
        if (order == ACL_SPARSE_ORDER_COL) {
            // Row x chunk units walk up to a 4096-element minor span in
            // one gather; the 1024 flat-tile budget would shrink the
            // slot buffer and expose sub-32B-gap gather overruns.
            return elementBytes == 8 ? kD2sTileElemsComplex
                                     : kD2sTileElems;
        }
        return static_cast<uint32_t>(kD2sCooTile);
    }
    // complex64 doubles the fp32 operand stream; halve the tile.
    return elementBytes == 8 ? kD2sTileElemsComplex : kD2sTileElems;
}

aclsparseStatus_t BuildTiling(const aclsparseDnMatDescr *dense,
                              const aclsparseSpMatDescr *sparse,
                              uint32_t numBlocks,
                              DenseToSparseTilingData *tiling)
{
    if (numBlocks == 0) {
        OP_LOGE(kApi, "invalid numBlocks=0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    tiling->rows = sparse->rows;
    tiling->cols = sparse->cols;
    tiling->ld = static_cast<uint64_t>(dense->ld);
    tiling->nnz = sparse->nnz;
    tiling->ellBlockSize = sparse->ellBlockSize;
    tiling->ellCols = sparse->ellCols;
    aclsparseStatus_t status =
        ComputeUnitCount(sparse, dense->order, &tiling->unitCount);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    tiling->statusOffset = kD2sStatusOffset;
    tiling->nnzOffset = kD2sNnzOffset;
    tiling->level0Offset = kD2sLevel0Offset;
    tiling->format = static_cast<uint32_t>(sparse->format);
    tiling->order = static_cast<uint32_t>(dense->order);
    tiling->base = static_cast<uint32_t>(sparse->baseType);
    tiling->elementBytes = GetElementBytes(dense->valueType);
    tiling->numBlocks = numBlocks;
    tiling->tileLen = JudgeTileLen(sparse,
        static_cast<aclsparseOrder_t>(dense->order), tiling->elementBytes);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Core count by scanned element volume (a purely task-count-based split
// left the large scenes on one or two cores).
uint32_t GetLaunchBlocks(uint64_t work)
{
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        return 0;
    }
    const uint64_t needed =
        std::max<uint64_t>(1, (work + kD2sWorkPerCore - 1) / kD2sWorkPerCore);
    return static_cast<uint32_t>(std::min<uint64_t>(cores, needed));
}

aclsparseStatus_t ValidateConvertPointers(const aclsparseSpMatDescr *sparse)
{
    if (sparse->format == ACL_SPARSE_FORMAT_COO) {
        if (sparse->nnz > 0 && sparse->rowInds == nullptr) {
            OP_LOGE(kApi, "COO rowInds is nullptr while nnz=%" PRIu64,
                    sparse->nnz);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        if (sparse->nnz > 0 && sparse->colInds == nullptr) {
            OP_LOGE(kApi, "COO colInds is nullptr while nnz=%" PRIu64,
                    sparse->nnz);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        if (sparse->nnz > 0 && sparse->values == nullptr) {
            OP_LOGE(kApi, "COO values is nullptr while nnz=%" PRIu64,
                    sparse->nnz);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (sparse->format == ACL_SPARSE_FORMAT_CSR ||
        sparse->format == ACL_SPARSE_FORMAT_CSC) {
        if (sparse->nnz > 0 && sparse->idxs == nullptr) {
            OP_LOGE(kApi,
                    "sparse indices is nullptr for format=%d, nnz=%" PRIu64,
                    static_cast<int>(sparse->format), sparse->nnz);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        if (sparse->nnz > 0 && sparse->values == nullptr) {
            OP_LOGE(kApi,
                    "sparse values is nullptr for format=%d, nnz=%" PRIu64,
                    static_cast<int>(sparse->format), sparse->nnz);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ReadAnalysisResult(aclsparseContext *context,
                                     uint8_t *workspace,
                                     const DenseToSparseTilingData &tiling,
                                     uint64_t logical, uint64_t *actualNnz)
{
    int32_t deviceStatus = kD2sDeviceStatusSuccess;
    aclError ret = aclrtMemcpyAsync(
        &deviceStatus, sizeof(deviceStatus), workspace + tiling.statusOffset,
        sizeof(deviceStatus), ACL_MEMCPY_DEVICE_TO_HOST, context->stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpyAsync(actualNnz, sizeof(*actualNnz),
                               workspace + tiling.nnzOffset,
                               sizeof(*actualNnz), ACL_MEMCPY_DEVICE_TO_HOST,
                               context->stream);
    }
    if (ret != ACL_SUCCESS ||
        aclrtSynchronizeStream(context->stream) != ACL_SUCCESS) {
        OP_LOGE(kApi, "failed to copy or synchronize analysis result");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    if (deviceStatus != kD2sDeviceStatusSuccess ||
        *actualNnz >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        *actualNnz > logical) {
        OP_LOGE(kApi,
                "invalid analysis result: deviceStatus=%d, nnz=%" PRIu64
                ", logical=%" PRIu64,
                deviceStatus, *actualNnz, logical);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Builds the tiling and launches the analysis kernel sequence (shared
// by the empty and non-empty matrix paths).
aclsparseStatus_t DispatchAnalysis(const aclsparseDnMatDescr *dense,
    aclsparseSpMatDescr *sparse, void *buffer, uint32_t blocks,
    aclsparseContext *context, DenseToSparseTilingData *tilingOut)
{
    DenseToSparseTilingData tiling{};
    aclsparseStatus_t status = BuildTiling(dense, sparse, blocks, &tiling);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    densetosparse_analysis_kernel_do(
        reinterpret_cast<GM_ADDR>(dense->values),
        reinterpret_cast<GM_ADDR>(sparse->ptrs),
        reinterpret_cast<GM_ADDR>(buffer), blocks, tiling,
        context->stream);
    *tilingOut = tiling;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Empty-matrix analysis: CSR/CSC offsets are filled by the empty-offsets
// kernel; COO publishes nnz=0 directly.
aclsparseStatus_t AnalysisEmptyMatrix(const aclsparseDnMatDescr *dense,
    aclsparseSpMatDescr *sparse, void *buffer, aclsparseHandle_t handle)
{
    sparse->nnz = 0;
    if (sparse->format != ACL_SPARSE_FORMAT_COO) {
        const uint32_t blocks = GetLaunchBlocks(1);
        if (blocks == 0) {
            OP_LOGE(kApi, "GetAivCoreCount returned 0 during analysis");
            return ACL_SPARSE_STATUS_INTERNAL_ERROR;
        }
        DenseToSparseTilingData unused{};
        return DispatchAnalysis(dense, sparse, buffer, blocks, handle,
                                &unused);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// BELL staging is per-task in UB: launch one core per task whenever the
// core count allows it (a purely element-based split leaves small task
// sets on a single core and overflows the per-core staging).
uint64_t ConvertWorkAmount(const aclsparseSpMatDescr *sparse,
    uint64_t logical)
{
    uint64_t tasks = 0;
    if (sparse->format == ACL_SPARSE_FORMAT_BLOCKED_ELL) {
        (void)BellTaskCount(sparse, &tasks);
        return tasks * kD2sWorkPerCore;
    }
    return logical;
}

aclsparseStatus_t LaunchAnalysis(aclsparseHandle_t handle,
                                 aclsparseConstDnMatDescr_t matA,
                                 aclsparseSpMatDescr_t matB, void *buffer)
{
    const auto *dense = matA;
    auto *sparse = matB;
    if (sparse->format == ACL_SPARSE_FORMAT_BLOCKED_ELL) {
        // BELL performs no structure discovery.
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    uint64_t logical = 0;
    if (!CheckedMultiply(sparse->rows, sparse->cols, &logical)) {
        OP_LOGE(kApi,
                "logical element count overflows: rows=%" PRIu64
                ", cols=%" PRIu64,
                sparse->rows, sparse->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (logical == 0) {
        return AnalysisEmptyMatrix(dense, sparse, buffer, handle);
    }
    const uint32_t blocks = GetLaunchBlocks(logical);
    if (blocks == 0) {
        OP_LOGE(kApi, "GetAivCoreCount returned 0 during analysis");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    DenseToSparseTilingData tiling{};
    aclsparseStatus_t status =
        DispatchAnalysis(dense, sparse, buffer, blocks, handle, &tiling);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    uint64_t actualNnz = 0;
    status = ReadAnalysisResult(handle, static_cast<uint8_t *>(buffer),
                                tiling, logical, &actualNnz);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    sparse->nnz = actualNnz;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t LaunchConvert(aclsparseHandle_t handle,
                                aclsparseConstDnMatDescr_t matA,
                                aclsparseSpMatDescr_t matB, void *buffer)
{
    const auto *dense = matA;
    auto *sparse = matB;
    if (sparse->format != ACL_SPARSE_FORMAT_BLOCKED_ELL &&
        (!sparse->denseToSparseAnalysisReady || sparse->activeDenseToSparseBuffer != buffer)) {
        OP_LOGE(kApi,
                "workspace buffer mismatch: Convert requires the same buffer "
                "registered by a successful Analysis (active=%p, buffer=%p)",
                sparse->activeDenseToSparseBuffer, buffer);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    uint64_t logical = 0;
    if (!CheckedMultiply(sparse->rows, sparse->cols, &logical)) {
        OP_LOGE(kApi,
                "logical element count overflows: rows=%" PRIu64
                ", cols=%" PRIu64,
                sparse->rows, sparse->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const uint64_t work = ConvertWorkAmount(sparse, logical);
    if (work == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const uint32_t blocks = GetLaunchBlocks(work);
    if (blocks == 0) {
        OP_LOGE(kApi, "GetAivCoreCount returned 0 during conversion");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    DenseToSparseTilingData tiling{};
    aclsparseStatus_t status = BuildTiling(dense, sparse, blocks, &tiling);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    densetosparse_convert_kernel_do(
        reinterpret_cast<GM_ADDR>(dense->values),
        reinterpret_cast<GM_ADDR>(buffer),
        reinterpret_cast<GM_ADDR>(sparse->ptrs),
        reinterpret_cast<GM_ADDR>(sparse->idxs),
        reinterpret_cast<GM_ADDR>(sparse->rowInds),
        reinterpret_cast<GM_ADDR>(sparse->colInds),
        reinterpret_cast<GM_ADDR>(sparse->values),
        reinterpret_cast<GM_ADDR>(sparse->ellColInd), blocks, tiling,
        handle->stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseDenseToSparseGetBufferSize(
    aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matA,
    aclsparseSpMatDescr_t matB, aclsparseDenseToSparseAlg_t alg,
    size_t *bufferSize)
{
    if (bufferSize == nullptr) {
        OP_LOGE(kApi, "bufferSize is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseStatus_t status = ValidateStaticParams(handle, matA, matB, alg);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    // Pure host query: no descriptor mutation, no stage state.
    return ComputeWorkspace(matB, matA->order, bufferSize);
}

aclsparseStatus_t aclsparseDenseToSparseAnalysis(
    aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matA,
    aclsparseSpMatDescr_t matB, aclsparseDenseToSparseAlg_t alg, void *buffer)
{
    aclsparseStatus_t status = ValidateStaticParams(handle, matA, matB, alg);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    size_t required = 0;
    status = ComputeWorkspace(matB, matA->order, &required);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (required > 0 && buffer == nullptr) {
        OP_LOGE(kApi, "buffer is nullptr while required size=%zu", required);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Analysis rewrites the sparse structure; other preprocess metadata can
    // no longer be reused even when their workspace address happens to match.
    matB->activeBuffer = nullptr;
    matB->activeSpmvBuffer = nullptr;
    matB->activeDenseToSparseBuffer = nullptr;
    matB->denseToSparseAnalysisReady = false;
    matB->spmvCubeMaxRowLengthValid = false;
    status = LaunchAnalysis(handle, matA, matB, buffer);
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        matB->activeDenseToSparseBuffer = buffer;
        matB->denseToSparseAnalysisReady = true;
    }
    return status;
}

aclsparseStatus_t aclsparseDenseToSparseConvert(
    aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matA,
    aclsparseSpMatDescr_t matB, aclsparseDenseToSparseAlg_t alg, void *buffer)
{
    aclsparseStatus_t status = ValidateStaticParams(handle, matA, matB, alg);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    auto *sparse = matB;
    size_t required = 0;
    status = ComputeWorkspace(sparse, matA->order, &required);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (required > 0 && buffer == nullptr) {
        OP_LOGE(kApi, "buffer is nullptr while required size=%zu", required);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    status = ValidateConvertPointers(sparse);
    return status == ACL_SPARSE_STATUS_SUCCESS
               ? LaunchConvert(handle, matA, matB, buffer)
               : status;
}

} // extern "C"
