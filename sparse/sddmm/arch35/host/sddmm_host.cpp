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
 * \file sddmm_host.cpp
 * \brief Ascend 950 SDDMM three-stage Host implementation.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>

#include "securec.h"
#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "../sddmm_bsr8_common.h"
#include "../sddmm_bsr_cube_common.h"
#include "../sddmm_complex_bsr_cube_common.h"
#include "../sddmm_complex_row_common.h"
#include "../sddmm_csr_cube_common.h"
#include "../sddmm_kernel.h"
#include "../sddmm_tiling_data.h"
#include "sddmm_pattern_host.h"

namespace {

constexpr const char *kSddmmApi = "aclsparseSDDMM";
constexpr uint32_t kSddmmDirectPath = 1U;
constexpr int32_t kMaxBatchCount = 65535;
constexpr std::array<int64_t, 7> kSupportedBsrBlockSizes = {2, 4, 8, 16, 32, 64, 128};
// Packing thresholds amortize the dense Y transpose across sampled outputs.
constexpr int64_t kPackedYMinReduction = 32;
constexpr int64_t kPackedYMinColumns = 64;
constexpr int64_t kPackedYMinOutputsPerColumn = 2;

struct SddmmCallContext {
    aclsparseContext *handle = nullptr;
    const aclsparseDnMatDescr *matX = nullptr;
    const aclsparseDnMatDescr *matY = nullptr;
    aclsparseSpMatDescr *matC = nullptr;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    int64_t outputsPerBatch = 0;
    int64_t totalOutputs = 0;
    size_t workspaceBytes = 0;
    size_t genericWorkspaceBytes = 0;
    int64_t packedYBytes = 0;
    int64_t packedYOffset = 0;
    uint64_t bsrCubeScratchBytes = 0;
    uint64_t bsrCubeScratchOffset = 0;
    uint64_t complexBsrCubeWorkspaceBytes = 0;
    uint64_t complexBsrCubeWorkspaceOffset = 0;
    uint64_t csrCubeScratchBytes = 0;
    uint64_t csrCubeScratchOffset = 0;
    bool usePackedY = false;
    bool useBsrCube = false;
    bool useComplexBsrCube = false;
    bool useCsrCube = false;
    sddmm_arch35::SddmmCsrCubePlan csrCubePlan{};
    SddmmTilingData tiling{};
};

bool CheckedAdd(int64_t lhs, int64_t rhs, int64_t &result)
{
    if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool CheckedMul(int64_t lhs, int64_t rhs, int64_t &result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAlign(uint64_t value, uint64_t alignment, uint64_t &result)
{
    if (alignment == 0) {
        return false;
    }
    const uint64_t remainder = value % alignment;
    if (remainder == 0) {
        result = value;
        return true;
    }
    const uint64_t padding = alignment - remainder;
    if (value > std::numeric_limits<uint64_t>::max() - padding) {
        return false;
    }
    result = value + padding;
    return true;
}

bool CheckedExtent(int32_t batchCount, int64_t stride, int64_t span, int64_t &extent)
{
    if (batchCount < 1 || stride < 0 || span < 0) {
        return false;
    }
    if (span == 0) {
        extent = 0;
        return true;
    }
    int64_t lastBatchOffset = 0;
    return CheckedMul(static_cast<int64_t>(batchCount - 1), stride, lastBatchOffset) &&
        CheckedAdd(lastBatchOffset, span, extent);
}

int64_t SddmmDataTypeBytes(aclDataType dataType)
{
    switch (dataType) {
        case ACL_FLOAT16:
        case ACL_BF16:
            return 2;
        case ACL_FLOAT:
            return 4;
        case ACL_COMPLEX64:
            return 8;
        default:
            return 0;
    }
}

int32_t SddmmValueTypeCode(aclDataType dataType)
{
    switch (dataType) {
        case ACL_FLOAT:
            return SDDMM_VALUE_FP32;
        case ACL_FLOAT16:
            return SDDMM_VALUE_FP16;
        case ACL_BF16:
            return SDDMM_VALUE_BF16;
        case ACL_COMPLEX64:
            return SDDMM_VALUE_COMPLEX64;
        default:
            return SDDMM_VALUE_INVALID;
    }
}

bool IsAlignedPointer(const void *pointer, uint64_t alignment)
{
    return pointer == nullptr || alignment == 0 ||
        (reinterpret_cast<uintptr_t>(pointer) % alignment) == 0;
}

bool CheckedPointerExtent(const void *pointer, int64_t elements, int64_t elementBytes)
{
    if (elements < 0 || elementBytes <= 0) {
        return false;
    }
    if (elements == 0) {
        return true;
    }
    if (pointer == nullptr) {
        return false;
    }
    int64_t bytes = 0;
    if (!CheckedMul(elements, elementBytes, bytes)) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(pointer);
    return static_cast<uint64_t>(bytes - 1) <=
        static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max() - base);
}

bool IsSupportedBlockSize(int64_t blockSize)
{
    return std::find(kSupportedBsrBlockSizes.begin(), kSupportedBsrBlockSizes.end(), blockSize) !=
        kSupportedBsrBlockSizes.end();
}

aclsparseStatus_t ValidateDenseShape(const aclsparseDnMatDescr *matrix, int64_t &physicalSpan)
{
    if (matrix == nullptr || matrix->signature != kDnMatSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matrix->rows < 0 || matrix->cols < 0 || matrix->ld <= 0 ||
        (matrix->order != ACL_SPARSE_ORDER_ROW && matrix->order != ACL_SPARSE_ORDER_COL)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const int64_t minimumLd = matrix->order == ACL_SPARSE_ORDER_ROW ? matrix->cols : matrix->rows;
    if (matrix->ld < minimumLd || matrix->batchCount < 1 || matrix->batchCount > kMaxBatchCount ||
        matrix->batchStride < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const int64_t major = matrix->order == ACL_SPARSE_ORDER_ROW ? matrix->rows : matrix->cols;
    if (!CheckedMul(major, matrix->ld, physicalSpan)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matrix->batchCount > 1 && matrix->batchStride < physicalSpan) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateDenseDescriptor(const aclsparseDnMatDescr *matrix, int64_t &physicalSpan)
{
    const auto status = ValidateDenseShape(matrix, physicalSpan);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    int64_t extent = 0;
    const int64_t dataTypeBytes = SddmmDataTypeBytes(matrix->valueType);
    if (dataTypeBytes == 0) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!CheckedExtent(matrix->batchCount, matrix->batchStride, physicalSpan, extent) ||
        !CheckedPointerExtent(matrix->values, matrix->rows > 0 && matrix->cols > 0 ? extent : 0,
                              dataTypeBytes) ||
        !IsAlignedPointer(matrix->values, static_cast<uint64_t>(dataTypeBytes))) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSparseIndexMetadata(const aclsparseSpMatDescr *matrix)
{
    if (matrix->ptrType != ACL_SPARSE_INDEX_32I || matrix->IdxType != ACL_SPARSE_INDEX_32I) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matrix->baseType != ACL_SPARSE_INDEX_BASE_ZERO &&
        matrix->baseType != ACL_SPARSE_INDEX_BASE_ONE) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const uint64_t base = static_cast<uint64_t>(matrix->baseType);
    if (matrix->nnz > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) - base) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSparsePointers(
    const aclsparseSpMatDescr *matrix, int64_t majorCount, int64_t outputsPerBatch)
{
    int64_t offsetsSpan = 0;
    if (!CheckedAdd(majorCount, 1, offsetsSpan) || matrix->ptrs == nullptr ||
        !IsAlignedPointer(matrix->ptrs, alignof(int32_t)) ||
        !IsAlignedPointer(matrix->idxs, alignof(int32_t))) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matrix->nnz > 0 && matrix->idxs == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const int64_t valueBytes = SddmmDataTypeBytes(matrix->valueType);
    if (valueBytes == 0) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (outputsPerBatch > 0 && (matrix->values == nullptr ||
        !IsAlignedPointer(matrix->values, static_cast<uint64_t>(valueBytes)))) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    int64_t offsetsExtent = 0;
    int64_t columnsExtent = 0;
    int64_t valuesExtent = 0;
    if (!CheckedExtent(matrix->batchCount, matrix->offsetsBatchStride, offsetsSpan, offsetsExtent) ||
        !CheckedExtent(matrix->batchCount, matrix->columnsBatchStride,
                       static_cast<int64_t>(matrix->nnz), columnsExtent) ||
        !CheckedExtent(matrix->batchCount, matrix->valuesBatchStride,
                       outputsPerBatch, valuesExtent) ||
        !CheckedPointerExtent(matrix->ptrs, offsetsExtent, sizeof(int32_t)) ||
        !CheckedPointerExtent(matrix->idxs, columnsExtent, sizeof(int32_t)) ||
        !CheckedPointerExtent(matrix->values, valuesExtent, valueBytes)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateCsrDescriptor(aclsparseSpMatDescr *matrix, int64_t &outputsPerBatch)
{
    if (matrix->ptrStride != 1 || matrix->idxStride != 1 || matrix->valueStride != 1) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matrix->batchCount != 1 || matrix->offsetsBatchStride != 0 ||
        matrix->columnsBatchStride != 0 || matrix->valuesBatchStride != 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    outputsPerBatch = static_cast<int64_t>(matrix->nnz);
    return ValidateSparsePointers(matrix, static_cast<int64_t>(matrix->rows), outputsPerBatch);
}

aclsparseStatus_t ValidateBsrMetadata(const aclsparseSpMatDescr *matrix)
{
    if (matrix->blockRows < 0 || matrix->blockCols < 0 || matrix->blockNnz < 0 ||
        matrix->rowBlockSize != matrix->colBlockSize ||
        !IsSupportedBlockSize(matrix->rowBlockSize) ||
        (matrix->bsrOrder != ACL_SPARSE_ORDER_ROW && matrix->bsrOrder != ACL_SPARSE_ORDER_COL) ||
        matrix->batchCount < 1 || matrix->batchCount > kMaxBatchCount ||
        matrix->offsetsBatchStride < 0 || matrix->columnsBatchStride < 0 ||
        matrix->valuesBatchStride < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateBsrDescriptor(aclsparseSpMatDescr *matrix, int64_t &outputsPerBatch)
{
    const auto status = ValidateBsrMetadata(matrix);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    int64_t logicalRows = 0;
    int64_t logicalCols = 0;
    int64_t blockArea = 0;
    if (!CheckedMul(matrix->blockRows, matrix->rowBlockSize, logicalRows) ||
        !CheckedMul(matrix->blockCols, matrix->colBlockSize, logicalCols) ||
        !CheckedMul(matrix->rowBlockSize, matrix->colBlockSize, blockArea) ||
        !CheckedMul(matrix->blockNnz, blockArea, outputsPerBatch) ||
        static_cast<uint64_t>(logicalRows) != matrix->rows ||
        static_cast<uint64_t>(logicalCols) != matrix->cols ||
        static_cast<uint64_t>(matrix->blockNnz) != matrix->nnz ||
        matrix->bsrValuesCount != outputsPerBatch) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int64_t offsetsSpan = 0;
    if (!CheckedAdd(matrix->blockRows, 1, offsetsSpan) ||
        (matrix->offsetsBatchStride != 0 && matrix->offsetsBatchStride < offsetsSpan) ||
        (matrix->columnsBatchStride != 0 && matrix->columnsBatchStride < matrix->blockNnz) ||
        (matrix->batchCount > 1 && outputsPerBatch > 0 &&
         matrix->valuesBatchStride < outputsPerBatch)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ValidateSparsePointers(matrix, matrix->blockRows, outputsPerBatch);
}

aclsparseStatus_t ValidateSparseDescriptor(aclsparseSpMatDescr *matrix, int64_t &outputsPerBatch)
{
    if (matrix == nullptr || matrix->signature != kSpMatSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matrix->format != ACL_SPARSE_FORMAT_CSR && matrix->format != ACL_SPARSE_FORMAT_BSR) {
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (matrix->rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        matrix->cols > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    aclsparseStatus_t status = ValidateSparseIndexMetadata(matrix);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    return matrix->format == ACL_SPARSE_FORMAT_CSR ?
        ValidateCsrDescriptor(matrix, outputsPerBatch) :
        ValidateBsrDescriptor(matrix, outputsPerBatch);
}

aclsparseStatus_t ValidateOperationsAndAlgorithm(
    aclsparseOperation_t opX, aclsparseOperation_t opY, aclsparseSDDMMAlg_t algorithm)
{
    if ((opX != ACL_SPARSE_OP_NON_TRANSPOSE && opX != ACL_SPARSE_OP_TRANSPOSE) ||
        (opY != ACL_SPARSE_OP_NON_TRANSPOSE && opY != ACL_SPARSE_OP_TRANSPOSE) ||
        algorithm != ACL_SPARSE_SDDMM_ALG_DEFAULT) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateDataTypes(
    const aclsparseDnMatDescr *matX, const aclsparseDnMatDescr *matY,
    const aclsparseSpMatDescr *matC, aclDataType computeType)
{
    if (matX->valueType != matY->valueType) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const aclDataType inputType = matX->valueType;
    const aclDataType outputType = matC->valueType;
    if (inputType == ACL_FLOAT && outputType == ACL_FLOAT && computeType == ACL_FLOAT) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (inputType == ACL_COMPLEX64 && outputType == ACL_COMPLEX64 && computeType == ACL_COMPLEX64) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (inputType == ACL_FLOAT16 &&
        (outputType == ACL_FLOAT16 || outputType == ACL_FLOAT) && computeType == ACL_FLOAT) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (inputType == ACL_BF16 &&
        (outputType == ACL_FLOAT ||
            (matC->format == ACL_SPARSE_FORMAT_BSR && outputType == ACL_BF16)) &&
        computeType == ACL_FLOAT) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
}

aclsparseStatus_t ResolveDimensions(
    const aclsparseDnMatDescr *matX, const aclsparseDnMatDescr *matY,
    const aclsparseSpMatDescr *matC,
    aclsparseOperation_t opX, aclsparseOperation_t opY,
    int64_t &m, int64_t &n, int64_t &k)
{
    m = static_cast<int64_t>(matC->rows);
    n = static_cast<int64_t>(matC->cols);
    if (opX == ACL_SPARSE_OP_NON_TRANSPOSE) {
        if (matX->rows != m) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        k = matX->cols;
    } else {
        if (matX->cols != m) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        k = matX->rows;
    }
    const bool yShapeMatches = opY == ACL_SPARSE_OP_NON_TRANSPOSE ?
        (matY->rows == k && matY->cols == n) :
        (matY->rows == n && matY->cols == k);
    return yShapeMatches ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_INVALID_VALUE;
}

aclsparseStatus_t ValidateBatchCompatibility(
    const aclsparseDnMatDescr *matX, const aclsparseDnMatDescr *matY,
    const aclsparseSpMatDescr *matC)
{
    const int32_t outputBatch = matC->format == ACL_SPARSE_FORMAT_BSR ? matC->batchCount : 1;
    const bool xMatches = matX->batchCount == 1 || matX->batchCount == outputBatch;
    const bool yMatches = matY->batchCount == 1 || matY->batchCount == outputBatch;
    return xMatches && yMatches ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_INVALID_VALUE;
}

bool HasPackedYLayout(const SddmmCallContext &context,
    aclsparseOperation_t opX, aclsparseOperation_t opY)
{
    return opX == ACL_SPARSE_OP_NON_TRANSPOSE && opY == ACL_SPARSE_OP_NON_TRANSPOSE &&
        context.matX->order == ACL_SPARSE_ORDER_ROW &&
        context.matY->order == ACL_SPARSE_ORDER_ROW &&
        context.matX->ld == context.k && context.matY->ld == context.n &&
        (context.matX->valueType == ACL_FLOAT || context.matX->valueType == ACL_FLOAT16) &&
        context.matY->batchCount == 1 && context.k >= kPackedYMinReduction &&
        context.n >= kPackedYMinColumns;
}

bool CanPackY(const SddmmCallContext &context,
    aclsparseOperation_t opX, aclsparseOperation_t opY)
{
    const bool bsrVectorLayout = context.matC->format == ACL_SPARSE_FORMAT_BSR &&
        context.matC->rowBlockSize == sddmm_arch35::kSddmmBsr8Side &&
        context.k == sddmm_arch35::kSddmmBsr8K &&
        context.matC->batchCount == 1 && context.matX->batchCount == 1;
    return context.workspaceBytes != 0 &&
        (context.matC->format == ACL_SPARSE_FORMAT_CSR || bsrVectorLayout) &&
        HasPackedYLayout(context, opX, opY);
}

void ConfigurePackedYWorkspace(
    SddmmCallContext &context,
    aclsparseOperation_t opX, aclsparseOperation_t opY)
{
    context.usePackedY = false;
    context.packedYBytes = 0;
    context.packedYOffset = 0;
    context.workspaceBytes = (context.m == 0 || context.n == 0) ? 0 :
        static_cast<size_t>(kSddmmWorkspaceBytes);
    if (!CanPackY(context, opX, opY)) {
        return;
    }

    // Packing is profitable only when the sampled workload reuses Y enough to
    // amortize one dense k-by-n transpose in the Execute stage.
    int64_t reuseThreshold = 0;
    if (!CheckedMul(context.n, kPackedYMinOutputsPerColumn, reuseThreshold) ||
        context.totalOutputs < reuseThreshold) {
        return;
    }
    int64_t packedElements = 0;
    int64_t packedBytes = 0;
    const int64_t elementBytes = SddmmDataTypeBytes(context.matY->valueType);
    if (!CheckedMul(context.k, context.n, packedElements) ||
        !CheckedMul(packedElements, elementBytes, packedBytes) ||
        packedBytes <= 0 ||
        static_cast<uint64_t>(packedBytes) > kSddmmPackedYWorkspaceLimit) {
        return;
    }
    uint64_t privateOffset = 0;
    if (!CheckedAlign(kSddmmWorkspaceBytes, kSddmmWorkspaceAlignment,
            privateOffset) ||
        privateOffset > std::numeric_limits<uint64_t>::max() -
            static_cast<uint64_t>(packedBytes)) {
        return;
    }
    uint64_t totalBytes = 0;
    if (!CheckedAlign(privateOffset + static_cast<uint64_t>(packedBytes),
            kSddmmWorkspaceAlignment, totalBytes) ||
        totalBytes > kSddmmPackedYWorkspaceLimit ||
        privateOffset > static_cast<uint64_t>(
            std::numeric_limits<int64_t>::max()) ||
        totalBytes >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return;
    }
    context.usePackedY = true;
    context.packedYBytes = packedBytes;
    context.packedYOffset = static_cast<int64_t>(privateOffset);
    context.workspaceBytes = static_cast<size_t>(totalBytes);
}

void ConfigureBsrCubeWorkspace(SddmmCallContext &context)
{
    context.useBsrCube = false;
    context.bsrCubeScratchBytes = 0;
    context.bsrCubeScratchOffset = 0;
    if (!sddmm_arch35::SddmmBsrCubeCanRun(context.tiling)) {
        return;
    }
    const uint64_t scratchBytes =
        sddmm_arch35::SddmmBsrCubeScratchBytes(context.tiling);
    if (scratchBytes == 0 ||
        scratchBytes > sddmm_arch35::kSddmmBsrCubeScratchLimit ||
        scratchBytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) -
            kSddmmWorkspaceBytes) {
        return;
    }
    uint64_t totalBytes = kSddmmWorkspaceBytes + scratchBytes;
    const uint64_t remainder = totalBytes % kSddmmWorkspaceAlignment;
    if (remainder != 0) {
        const uint64_t padding = kSddmmWorkspaceAlignment - remainder;
        if (totalBytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - padding) {
            return;
        }
        totalBytes += padding;
    }
    context.useBsrCube = true;
    context.bsrCubeScratchBytes = scratchBytes;
    context.bsrCubeScratchOffset = kSddmmWorkspaceBytes;
    context.workspaceBytes = static_cast<size_t>(totalBytes);
}

bool HasComplexBsrCubeL0Capacity(const SddmmTilingData &tiling)
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return false;
    }
    uint64_t l0aBytes = 0;
    uint64_t l0bBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aBytes);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bBytes);
    return sddmm_arch35::SddmmComplexBsrCubeFitsL0(tiling, l0aBytes, l0bBytes);
}

void ConfigureComplexBsrCubeWorkspace(SddmmCallContext &context)
{
    context.useComplexBsrCube = false;
    context.complexBsrCubeWorkspaceBytes = 0;
    context.complexBsrCubeWorkspaceOffset = 0;
    if (!sddmm_arch35::SddmmComplexBsrCubeCanRun(context.tiling) ||
        !IsAlignedPointer(context.matX->values,
            sddmm_arch35::kSddmmComplexBsrCubeInputAlignment) ||
        !IsAlignedPointer(context.matY->values,
            sddmm_arch35::kSddmmComplexBsrCubeInputAlignment)) {
        return;
    }
    // BufferSize, Preprocess and Execute share this admission check. Keep
    // unsupported platforms on the generic path before reserving Cube scratch.
    if (!HasComplexBsrCubeL0Capacity(context.tiling)) {
        return;
    }
    const uint64_t privateBytes =
        sddmm_arch35::SddmmComplexBsrCubeWorkspaceBytes(context.tiling);
    if (privateBytes == 0ULL ||
        privateBytes > sddmm_arch35::kSddmmComplexBsrCubeWorkspaceLimit) {
        return;
    }
    uint64_t privateOffset = static_cast<uint64_t>(context.workspaceBytes);
    const uint64_t remainder =
        privateOffset % sddmm_arch35::kSddmmComplexBsrCubeWorkspaceAlignment;
    if (remainder != 0ULL) {
        const uint64_t padding =
            sddmm_arch35::kSddmmComplexBsrCubeWorkspaceAlignment - remainder;
        if (privateOffset > std::numeric_limits<uint64_t>::max() - padding) {
            return;
        }
        privateOffset += padding;
    }
    if (privateOffset >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max()) -
            privateBytes || privateOffset + privateBytes >
                sddmm_arch35::kSddmmComplexBsrCubeWorkspaceLimit +
                sddmm_arch35::kSddmmComplexBsrCubeWorkspaceAlignment) {
        return;
    }
    context.useComplexBsrCube = true;
    context.complexBsrCubeWorkspaceBytes = privateBytes;
    context.complexBsrCubeWorkspaceOffset = privateOffset;
    context.workspaceBytes = static_cast<size_t>(privateOffset + privateBytes);
}

bool HasCsrCubeDataLayout(const SddmmTilingData &tiling)
{
    const bool supportedType =
        (tiling.inputType == SDDMM_VALUE_FP32 && tiling.outputType == SDDMM_VALUE_FP32) ||
        (tiling.inputType == SDDMM_VALUE_FP16 &&
            (tiling.outputType == SDDMM_VALUE_FP16 || tiling.outputType == SDDMM_VALUE_FP32)) ||
        (tiling.inputType == SDDMM_VALUE_BF16 &&
            (tiling.outputType == SDDMM_VALUE_BF16 || tiling.outputType == SDDMM_VALUE_FP32));
    const bool validYLayout = (tiling.orderY == 0 && tiling.ldY == tiling.n) ||
        (tiling.orderY == 1 && tiling.ldY == tiling.k);
    return supportedType && validYLayout && tiling.opX == 0 && tiling.opY == 0 &&
        tiling.orderX == 0 && tiling.ldX == tiling.k;
}

bool CanUseCsrCube(const SddmmTilingData &tiling)
{
    return tiling.format == SDDMM_FORMAT_CSR &&
        tiling.computeType == SDDMM_VALUE_FP32 && tiling.pointerMode == 0 &&
        tiling.k == sddmm_arch35::kSddmmCsrCubeReduction && tiling.m > 0 &&
        tiling.m % sddmm_arch35::kSddmmCsrCubeRows == 0 &&
        tiling.n >= sddmm_arch35::kSddmmCsrCubeWorkspaceBand &&
        tiling.n % sddmm_arch35::kSddmmCsrCubeL0cC0 == 0 &&
        tiling.batchCount == 1 && tiling.batchCountX == 1 && tiling.batchCountY == 1 &&
        tiling.offsetsBatchStride == 0 && tiling.columnsBatchStride == 0 &&
        tiling.valuesBatchStride == 0 &&
        tiling.nnzUnits == tiling.m * sddmm_arch35::kSddmmCsrCubeSamplesPerRow &&
        tiling.outputsPerBatch == tiling.nnzUnits && HasCsrCubeDataLayout(tiling);
}

void ConfigureCsrCubePlan(SddmmCallContext &context)
{
    const auto &tiling = context.tiling;
    context.csrCubePlan.patternKind =
        sddmm_arch35::SDDMM_CSR_CUBE_PATTERN_CYCLIC_64_STRIDE_17;
    context.csrCubePlan.logicalFirstColumn = 0;
    context.csrCubePlan.rowColumnStride =
        sddmm_arch35::kSddmmCsrCubeRowColumnStride;
    context.csrCubePlan.samplesPerRow =
        sddmm_arch35::kSddmmCsrCubeSamplesPerRow;
    context.csrCubePlan.yPhysicalOrder = tiling.orderY;
    context.csrCubePlan.yLeadingDimension =
        static_cast<int32_t>(tiling.ldY);
    context.csrCubePlan.groupCount = static_cast<int32_t>(
        tiling.m / sddmm_arch35::kSddmmCsrCubeRows);
}

void ConfigureCsrCubeWorkspace(SddmmCallContext &context)
{
    context.useCsrCube = false;
    context.csrCubeScratchBytes = 0;
    context.csrCubeScratchOffset = 0;
    context.csrCubePlan = {};
    const auto &tiling = context.tiling;
    if (!CanUseCsrCube(tiling)) {
        return;
    }

    const uint64_t scratchBytes =
        sddmm_arch35::SddmmCsrCubeDenseWorkspaceBytes(
            static_cast<uint64_t>(tiling.m));
    if (scratchBytes == 0 || scratchBytes > kSddmmPackedYWorkspaceLimit) {
        return;
    }
    uint64_t privateOffset = 0;
    if (!CheckedAlign(kSddmmWorkspaceBytes, kSddmmWorkspaceAlignment,
            privateOffset)) {
        return;
    }
    const uint64_t packedYBytes = context.usePackedY ?
        static_cast<uint64_t>(context.packedYBytes) : 0ULL;
    // Execute either returns from the Cube path or reaches the generic
    // packed-Y path.  The two private regions therefore share one offset.
    const uint64_t privateBytes = packedYBytes > scratchBytes ?
        packedYBytes : scratchBytes;
    if (privateOffset > std::numeric_limits<uint64_t>::max() -
            privateBytes) {
        return;
    }
    uint64_t totalBytes = 0;
    if (!CheckedAlign(privateOffset + privateBytes,
            kSddmmWorkspaceAlignment, totalBytes) ||
        totalBytes > static_cast<uint64_t>(
            std::numeric_limits<size_t>::max())) {
        return;
    }

    ConfigureCsrCubePlan(context);
    context.useCsrCube = true;
    context.csrCubeScratchBytes = scratchBytes;
    context.csrCubeScratchOffset = privateOffset;
    context.workspaceBytes = static_cast<size_t>(totalBytes);
}

void FillTiling(
    const SddmmCallContext &context,
    aclsparseOperation_t opX, aclsparseOperation_t opY,
    aclDataType computeType,
    SddmmTilingData &tiling)
{
    const auto *matX = context.matX;
    const auto *matY = context.matY;
    const auto *matC = context.matC;
    tiling.m = context.m;
    tiling.n = context.n;
    tiling.k = context.k;
    tiling.rowsX = matX->rows;
    tiling.colsX = matX->cols;
    tiling.rowsY = matY->rows;
    tiling.colsY = matY->cols;
    tiling.ldX = matX->ld;
    tiling.ldY = matY->ld;
    tiling.blockRows = matC->format == ACL_SPARSE_FORMAT_BSR ? matC->blockRows : context.m;
    tiling.blockCols = matC->format == ACL_SPARSE_FORMAT_BSR ? matC->blockCols : context.n;
    tiling.nnzUnits = static_cast<int64_t>(matC->nnz);
    tiling.outputsPerBatch = context.outputsPerBatch;
    tiling.batchCount = matC->format == ACL_SPARSE_FORMAT_BSR ? matC->batchCount : 1;
    tiling.batchCountX = matX->batchCount;
    tiling.batchCountY = matY->batchCount;
    tiling.xBatchStride = matX->batchStride;
    tiling.yBatchStride = matY->batchStride;
    tiling.offsetsBatchStride = matC->offsetsBatchStride;
    tiling.columnsBatchStride = matC->columnsBatchStride;
    tiling.valuesBatchStride = matC->valuesBatchStride;
    tiling.packedYBatchStride = context.k * context.n;
    tiling.packedYOffset = context.packedYOffset;
    tiling.format = matC->format == ACL_SPARSE_FORMAT_BSR ? SDDMM_FORMAT_BSR : SDDMM_FORMAT_CSR;
    tiling.indexBase = static_cast<int32_t>(matC->baseType);
    tiling.blockDim = matC->format == ACL_SPARSE_FORMAT_BSR ? static_cast<int32_t>(matC->rowBlockSize) : 1;
    tiling.blockOrder = matC->format == ACL_SPARSE_FORMAT_BSR ? static_cast<int32_t>(matC->bsrOrder) : 0;
    tiling.opX = opX == ACL_SPARSE_OP_TRANSPOSE ? 1 : 0;
    tiling.opY = opY == ACL_SPARSE_OP_TRANSPOSE ? 1 : 0;
    tiling.orderX = static_cast<int32_t>(matX->order);
    tiling.orderY = static_cast<int32_t>(matY->order);
    tiling.inputType = SddmmValueTypeCode(matX->valueType);
    tiling.outputType = SddmmValueTypeCode(matC->valueType);
    tiling.computeType = SddmmValueTypeCode(computeType);
    tiling.pointerMode = static_cast<int32_t>(context.handle->pointerMode);
    tiling.usePackedY = context.usePackedY ? 1 : 0;
    // Pattern properties are produced by Preprocess and are deliberately not
    // inferred from descriptor dimensions on Host.
    tiling.patternFlags = SDDMM_PATTERN_FLAG_NONE;
    tiling.patternLogicalFirstColumn = -1;
}

aclsparseStatus_t ValidateCallDescriptors(SddmmCallContext &context,
    aclsparseOperation_t opX, aclsparseOperation_t opY)
{
    int64_t xSpan = 0;
    int64_t ySpan = 0;
    auto status = ValidateDenseDescriptor(context.matX, xSpan);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateDenseDescriptor(context.matY, ySpan);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateSparseDescriptor(context.matC, context.outputsPerBatch);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ResolveDimensions(context.matX, context.matY, context.matC,
                               opX, opY, context.m, context.n, context.k);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (context.k > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if ((context.m == 0 || context.n == 0) && context.outputsPerBatch != 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateDeviceScalars(const SddmmCallContext &context,
    const void *alpha, const void *beta, aclDataType computeType)
{
    if (context.handle->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        const int64_t scalarBytes = SddmmDataTypeBytes(computeType);
        if (scalarBytes <= 0 || !IsAlignedPointer(alpha, alignof(float)) ||
            !IsAlignedPointer(beta, alignof(float)) ||
            !CheckedPointerExtent(alpha, 1, scalarBytes) ||
            !CheckedPointerExtent(beta, 1, scalarBytes)) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ConfigureSddmmWorkspace(SddmmCallContext &context,
    aclsparseOperation_t opX, aclsparseOperation_t opY, aclDataType computeType)
{
    if (!context.matC->sddmmQueryScratch.Release()) {
        OP_LOGE(kSddmmApi, "previous workspace query has not completed or released its scratch");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    ConfigurePackedYWorkspace(context, opX, opY);
    context.genericWorkspaceBytes = context.workspaceBytes;
    FillTiling(context, opX, opY, computeType, context.tiling);
    ConfigureComplexBsrCubeWorkspace(context);
    ConfigureBsrCubeWorkspace(context);
    ConfigureCsrCubeWorkspace(context);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSddmmCall(
    aclsparseHandle_t handle,
    aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t algorithm, SddmmCallContext &context)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (alpha == nullptr || beta == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    context.handle = reinterpret_cast<aclsparseContext *>(handle);
    context.matX = matX;
    context.matY = matY;
    context.matC = matC;
    if (context.handle->pointerMode != ACL_SPARSE_POINTER_MODE_HOST &&
        context.handle->pointerMode != ACL_SPARSE_POINTER_MODE_DEVICE) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseStatus_t status = ValidateOperationsAndAlgorithm(opX, opY, algorithm);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateCallDescriptors(context, opX, opY);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateDataTypes(context.matX, context.matY, context.matC, computeType);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateDeviceScalars(context, alpha, beta, computeType);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateBatchCompatibility(context.matX, context.matY, context.matC);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    const int64_t outputBatch = context.matC->format == ACL_SPARSE_FORMAT_BSR ?
        context.matC->batchCount : 1;
    if (!CheckedMul(context.outputsPerBatch, outputBatch, context.totalOutputs)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ConfigureSddmmWorkspace(context, opX, opY, computeType);
}

SddmmPreparedKey BuildPreparedKey(
    const SddmmCallContext &context,
    aclsparseOperation_t opX, aclsparseOperation_t opY,
    aclDataType computeType, aclsparseSDDMMAlg_t algorithm,
    const void *workspace)
{
    const auto *matX = context.matX;
    const auto *matY = context.matY;
    const auto *matC = context.matC;
    SddmmPreparedKey key{};
    key.structureGeneration = matC->structureGeneration;
    key.format = matC->format;
    key.rows = matC->rows;
    key.cols = matC->cols;
    key.nnz = matC->nnz;
    key.offsets = matC->ptrs;
    key.columns = matC->idxs;
    key.indexBase = matC->baseType;
    key.offsetsType = matC->ptrType;
    key.columnsType = matC->IdxType;
    key.valueTypeC = matC->valueType;
    key.blockRows = matC->blockRows;
    key.blockCols = matC->blockCols;
    key.blockNnz = matC->blockNnz;
    key.rowBlockSize = matC->rowBlockSize;
    key.colBlockSize = matC->colBlockSize;
    key.blockOrder = matC->bsrOrder;
    key.batchCountC = matC->batchCount;
    key.offsetsBatchStride = matC->offsetsBatchStride;
    key.columnsBatchStride = matC->columnsBatchStride;
    key.valuesBatchStride = matC->valuesBatchStride;
    key.rowsX = matX->rows;
    key.colsX = matX->cols;
    key.ldX = matX->ld;
    key.orderX = matX->order;
    key.valueTypeX = matX->valueType;
    key.batchCountX = matX->batchCount;
    key.batchStrideX = matX->batchStride;
    key.opX = opX;
    key.rowsY = matY->rows;
    key.colsY = matY->cols;
    key.ldY = matY->ld;
    key.orderY = matY->order;
    key.valueTypeY = matY->valueType;
    key.batchCountY = matY->batchCount;
    key.batchStrideY = matY->batchStride;
    key.opY = opY;
    key.computeType = computeType;
    key.algorithm = algorithm;
    key.workspace = context.workspaceBytes == 0 ? nullptr : workspace;
    key.workspaceBytes = context.workspaceBytes;
    return key;
}

bool PreparedKeysEqual(const SddmmPreparedKey &lhs, const SddmmPreparedKey &rhs)
{
    return lhs.structureGeneration == rhs.structureGeneration && lhs.format == rhs.format &&
        lhs.rows == rhs.rows && lhs.cols == rhs.cols && lhs.nnz == rhs.nnz &&
        lhs.offsets == rhs.offsets && lhs.columns == rhs.columns && lhs.indexBase == rhs.indexBase &&
        lhs.offsetsType == rhs.offsetsType && lhs.columnsType == rhs.columnsType &&
        lhs.valueTypeC == rhs.valueTypeC && lhs.blockRows == rhs.blockRows &&
        lhs.blockCols == rhs.blockCols && lhs.blockNnz == rhs.blockNnz &&
        lhs.rowBlockSize == rhs.rowBlockSize && lhs.colBlockSize == rhs.colBlockSize &&
        lhs.blockOrder == rhs.blockOrder && lhs.batchCountC == rhs.batchCountC &&
        lhs.offsetsBatchStride == rhs.offsetsBatchStride &&
        lhs.columnsBatchStride == rhs.columnsBatchStride &&
        lhs.valuesBatchStride == rhs.valuesBatchStride && lhs.rowsX == rhs.rowsX &&
        lhs.colsX == rhs.colsX && lhs.ldX == rhs.ldX && lhs.orderX == rhs.orderX &&
        lhs.valueTypeX == rhs.valueTypeX && lhs.batchCountX == rhs.batchCountX &&
        lhs.batchStrideX == rhs.batchStrideX && lhs.opX == rhs.opX &&
        lhs.rowsY == rhs.rowsY && lhs.colsY == rhs.colsY && lhs.ldY == rhs.ldY &&
        lhs.orderY == rhs.orderY && lhs.valueTypeY == rhs.valueTypeY &&
        lhs.batchCountY == rhs.batchCountY && lhs.batchStrideY == rhs.batchStrideY &&
        lhs.opY == rhs.opY && lhs.computeType == rhs.computeType &&
        lhs.algorithm == rhs.algorithm && lhs.workspace == rhs.workspace &&
        lhs.workspaceBytes == rhs.workspaceBytes;
}

SddmmPreparedKey BuildWorkspacePlanKey(
    const SddmmCallContext &context,
    aclsparseOperation_t opX, aclsparseOperation_t opY,
    aclDataType computeType, aclsparseSDDMMAlg_t algorithm)
{
    auto key = BuildPreparedKey(context, opX, opY, computeType, algorithm, nullptr);
    // Replacing or editing indices changes preprocessing, not the capacity
    // already allocated by the caller for this metadata.
    key.structureGeneration = 0;
    key.offsets = nullptr;
    key.columns = nullptr;
    key.workspace = nullptr;
    key.workspaceBytes = 0;
    return key;
}

bool FitsWorkspace(uint64_t offset, uint64_t bytes, size_t capacity)
{
    return offset <= capacity && bytes <= capacity - offset;
}

aclsparseStatus_t ApplyWorkspacePlan(
    SddmmCallContext &context,
    aclsparseOperation_t opX, aclsparseOperation_t opY,
    aclDataType computeType, aclsparseSDDMMAlg_t algorithm)
{
    const auto &plan = context.matC->sddmmWorkspacePlan;
    if (!plan.queried) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const auto key = BuildWorkspacePlanKey(context, opX, opY, computeType, algorithm);
    if (!PreparedKeysEqual(plan.key, key)) {
        OP_LOGE(kSddmmApi, "matrix metadata changed; query workspace size again");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const size_t capacity = plan.workspaceBytes;
    context.useCsrCube = context.useCsrCube && FitsWorkspace(
        context.csrCubeScratchOffset, context.csrCubeScratchBytes, capacity);
    context.useBsrCube = context.useBsrCube && FitsWorkspace(
        context.bsrCubeScratchOffset, context.bsrCubeScratchBytes, capacity);
    context.useComplexBsrCube = context.useComplexBsrCube && FitsWorkspace(
        context.complexBsrCubeWorkspaceOffset, context.complexBsrCubeWorkspaceBytes, capacity);
    context.usePackedY = context.usePackedY && FitsWorkspace(
        static_cast<uint64_t>(context.packedYOffset),
        static_cast<uint64_t>(context.packedYBytes), capacity);
    context.tiling.usePackedY = context.usePackedY ? 1 : 0;
    context.workspaceBytes = capacity;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateWorkspace(
    const SddmmCallContext &context, const void *workspace, bool preprocess)
{
    if (context.workspaceBytes == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (workspace == nullptr) {
        return preprocess ? ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES :
                            ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return IsAlignedPointer(workspace, kSddmmWorkspaceAlignment) ?
        ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_INVALID_VALUE;
}

void SavePreparedState(
    aclsparseSpMatDescr *matC, const SddmmPreparedKey &key,
    const SddmmValidationResult &validation)
{
    auto &state = matC->sddmmPrepared;
    state.key = key;
    state.workspace = key.workspace;
    state.workspaceBytes = key.workspaceBytes;
    state.patternGenerationSnapshot = matC->structureGeneration;
    ++state.preprocessEpoch;
    if (state.preprocessEpoch == 0) {
        state.preprocessEpoch = 1;
    }
    state.path = kSddmmDirectPath;
    state.patternFlags = validation.patternFlags;
    state.patternLogicalFirstColumn = validation.logicalFirstColumn;
    state.prepared = true;
}

aclsparseStatus_t AllocatePatternPartials(
    const SddmmCallContext &context, uint32_t numBlocks,
    std::unique_ptr<SddmmQueryPartial[]> &partials, size_t &scratchBytes)
{
    if (numBlocks > std::numeric_limits<size_t>::max() / sizeof(SddmmQueryPartial)) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    scratchBytes = static_cast<size_t>(numBlocks) * sizeof(SddmmQueryPartial);
    partials.reset(new (std::nothrow) SddmmQueryPartial[numBlocks]);
    if (partials == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    auto &scratch = context.matC->sddmmQueryScratch;
    if (aclrtMalloc(&scratch.buffer, scratchBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        OP_LOGE(kSddmmApi, "failed to allocate pattern metadata");
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    scratch.stream = context.handle->stream;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ReadPatternValidation(
    const SddmmCallContext &context, void *workspace,
    SddmmValidationResult &validation, bool &pending)
{
    pending = false;
    const uint32_t coreCount = GetAivCoreCount();
    if (coreCount == 0U) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    uint32_t numBlocks = 0U;
    if (!sddmm_arch35::PlanPatternValidation(context.tiling, coreCount, numBlocks) || numBlocks == 0U) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    size_t scratchBytes = 0;
    std::unique_ptr<SddmmQueryPartial[]> partials;
    const auto allocationStatus = AllocatePatternPartials(context, numBlocks, partials, scratchBytes);
    if (allocationStatus != ACL_SPARSE_STATUS_SUCCESS) {
        return allocationStatus;
    }
    auto &scratch = context.matC->sddmmQueryScratch;
    scratch.pending = true;
    pending = true;
    sddmm_pattern_validate_kernel_do(
        reinterpret_cast<GM_ADDR>(context.matC->ptrs),
        reinterpret_cast<GM_ADDR>(context.matC->idxs),
        reinterpret_cast<GM_ADDR>(scratch.buffer), context.tiling, numBlocks, context.handle->stream);
    if (aclrtSynchronizeStream(context.handle->stream) != ACL_SUCCESS) {
        OP_LOGE(kSddmmApi, "failed to synchronize pattern validation");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    scratch.pending = false;
    pending = false;
    // No Host copy starts until the kernel completed. A failed wait keeps the
    // descriptor-owned scratch alive for ConfigureSddmmWorkspace to drain.
    const aclError copyStatus = aclrtMemcpy(
        partials.get(), scratchBytes, scratch.buffer, scratchBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    const bool released = scratch.Release();
    if (copyStatus != ACL_SUCCESS || !released) {
        OP_LOGE(kSddmmApi, "failed to read or release pattern validation metadata");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    if (!sddmm_arch35::ReducePatternValidation(partials.get(), numBlocks, context.tiling, validation)) {
        OP_LOGE(kSddmmApi, "invalid partial pattern validation result");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    // Keep the caller's queried capacity and stable validation prefix. Only
    // the existing result object is copied; temporary per-core slots stay private.
    if (aclrtMemcpy(workspace, sizeof(validation), &validation, sizeof(validation),
            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        OP_LOGE(kSddmmApi, "failed to write pattern validation result");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool HasCsrCubePattern(uint32_t flags, int32_t firstColumn)
{
    return (flags & SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17) != 0U &&
        firstColumn >= 0 && firstColumn % sddmm_arch35::kSddmmCsrCubeL0cC0 == 0;
}

aclsparseStatus_t ReadPatternQuery(
    const SddmmCallContext &context, void *workspace,
    SddmmQueryPartial *partials, uint32_t numBlocks, bool &pending)
{
    pending = true;
    sddmm_pattern_query_kernel_do(
        reinterpret_cast<GM_ADDR>(context.matC->ptrs),
        reinterpret_cast<GM_ADDR>(context.matC->idxs),
        reinterpret_cast<GM_ADDR>(workspace), context.tiling, numBlocks, context.handle->stream);
    if (aclrtSynchronizeStream(context.handle->stream) != ACL_SUCCESS) {
        OP_LOGE(kSddmmApi, "failed to synchronize pattern query");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    pending = false;
    const size_t bytes = static_cast<size_t>(numBlocks) * sizeof(SddmmQueryPartial);
    if (aclrtMemcpy(partials, bytes, workspace, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        OP_LOGE(kSddmmApi, "failed to read pattern query results");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t QueryCsrCubeWorkspace(SddmmCallContext &context)
{
    if (!context.useCsrCube || context.workspaceBytes <= context.genericWorkspaceBytes) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const uint32_t coreCount = GetAivCoreCount();
    if (coreCount == 0U) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    const uint32_t numBlocks = static_cast<uint32_t>(std::min<uint64_t>(
        coreCount, static_cast<uint64_t>(context.tiling.m)));
    // Only one result slot per active core is temporary. Borrowing the handle's
    // workspace could overwrite another operator's persistent preprocessing.
    size_t scratchBytes = 0;
    std::unique_ptr<SddmmQueryPartial[]> partials;
    const auto allocationStatus = AllocatePatternPartials(context, numBlocks, partials, scratchBytes);
    if (allocationStatus != ACL_SPARSE_STATUS_SUCCESS) {
        return allocationStatus;
    }
    auto &scratch = context.matC->sddmmQueryScratch;
    SddmmValidationResult validation{};
    const auto status = ReadPatternQuery(context, scratch.buffer, partials.get(), numBlocks, scratch.pending);
    if (scratch.pending) {
        return status;
    }
    const bool released = scratch.Release();
    if (status != ACL_SPARSE_STATUS_SUCCESS || !released) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    if (!sddmm_arch35::ReducePatternQuery(partials.get(), numBlocks, context.tiling.n, validation)) {
        OP_LOGE(kSddmmApi, "invalid partial pattern query result");
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    // Invalid patterns are still rejected by Preprocess/Execute. A size
    // query must not reserve Cube scratch for them in the meantime.
    if (validation.magic != kSddmmValidationMagic ||
        validation.status != SDDMM_PATTERN_VALID ||
        !HasCsrCubePattern(validation.patternFlags, validation.logicalFirstColumn)) {
        context.useCsrCube = false;
        context.workspaceBytes = context.genericWorkspaceBytes;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

void InvalidateAliasedWorkspaceOwners(
    const SddmmCallContext &context, const void *workspace)
{
    auto *matC = context.matC;
    // activeBuffer is shared by older sparse operators.  Reusing the exact
    // allocation for SDDMM overwrites their private workspace metadata, so the
    // previous owner must rebuild instead of treating the buffer as active.
    if (context.workspaceBytes != 0 && matC->activeBuffer == workspace) {
        matC->activeBuffer = nullptr;
    }
    if (context.workspaceBytes != 0 && matC->activeSpmvBuffer == workspace) {
        matC->activeSpmvBuffer = nullptr;
    }
}

aclsparseStatus_t PreparePattern(
    SddmmCallContext &context,
    aclsparseOperation_t opX, aclsparseOperation_t opY,
    aclDataType computeType, aclsparseSDDMMAlg_t algorithm,
    void *workspace)
{
    auto *matC = context.matC;
    InvalidateAliasedWorkspaceOwners(context, workspace);
    AclsparseInvalidateSddmmPreparedState(matC);
    const SddmmPreparedKey key = BuildPreparedKey(context, opX, opY, computeType, algorithm, workspace);
    SddmmValidationResult validation{};
    validation.magic = kSddmmValidationMagic;
    validation.logicalFirstColumn = -1;
    if (context.workspaceBytes == 0) {
        SavePreparedState(matC, key, validation);
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    bool pending = false;
    const auto status = ReadPatternValidation(context, workspace, validation, pending);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (validation.magic != kSddmmValidationMagic || validation.status != SDDMM_PATTERN_VALID) {
        OP_LOGE(kSddmmApi, "invalid sparse pattern, device status=%d", validation.status);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    SavePreparedState(matC, key, validation);
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool IsPrepared(
    const SddmmCallContext &context, const SddmmPreparedKey &key)
{
    const auto &state = context.matC->sddmmPrepared;
    return state.prepared && state.path == kSddmmDirectPath &&
        state.patternGenerationSnapshot == context.matC->structureGeneration &&
        state.workspace == key.workspace && state.workspaceBytes == key.workspaceBytes &&
        PreparedKeysEqual(state.key, key);
}

aclsparseStatus_t PopulateScalarTiling(
    const SddmmCallContext &context, const void *alpha, const void *beta,
    aclDataType computeType, SddmmTilingData &tiling)
{
    tiling.alphaPtr = 0;
    tiling.betaPtr = 0;
    tiling.alphaReal = 0.0F;
    tiling.alphaImag = 0.0F;
    tiling.betaReal = 0.0F;
    tiling.betaImag = 0.0F;
    if (context.handle->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        tiling.alphaPtr = reinterpret_cast<uintptr_t>(alpha);
        tiling.betaPtr = reinterpret_cast<uintptr_t>(beta);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (computeType == ACL_COMPLEX64) {
        aclsparseComplex alphaValue{};
        aclsparseComplex betaValue{};
        if (memcpy_s(&alphaValue, sizeof(alphaValue), alpha, sizeof(alphaValue)) != EOK ||
            memcpy_s(&betaValue, sizeof(betaValue), beta, sizeof(betaValue)) != EOK) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        tiling.alphaReal = alphaValue.x;
        tiling.alphaImag = alphaValue.y;
        tiling.betaReal = betaValue.x;
        tiling.betaImag = betaValue.y;
    } else {
        if (memcpy_s(&tiling.alphaReal, sizeof(float), alpha, sizeof(float)) != EOK ||
            memcpy_s(&tiling.betaReal, sizeof(float), beta, sizeof(float)) != EOK) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

bool IsHostNoOp(const SddmmCallContext &context, const SddmmTilingData &tiling)
{
    if (context.handle->pointerMode != ACL_SPARSE_POINTER_MODE_HOST) {
        return false;
    }
    if (tiling.computeType == SDDMM_VALUE_COMPLEX64) {
        return tiling.alphaReal == 0.0F && tiling.alphaImag == 0.0F &&
            tiling.betaReal == 1.0F && tiling.betaImag == 0.0F;
    }
    return tiling.alphaReal == 0.0F && tiling.betaReal == 1.0F;
}

bool IsHostZeroAlpha(
    const SddmmCallContext &context, const SddmmTilingData &tiling)
{
    return context.handle->pointerMode == ACL_SPARSE_POINTER_MODE_HOST &&
        tiling.alphaReal == 0.0F &&
        (tiling.computeType != SDDMM_VALUE_COMPLEX64 || tiling.alphaImag == 0.0F);
}

uint64_t EstimateCsrCubeGatherMaxWork(
    const SddmmTilingData &tiling,
    const sddmm_arch35::SddmmCsrCubePlan &plan,
    uint32_t blockCount)
{
    // A gatherable group uses one bulk operation; a wrapping group executes
    // all sixteen rows separately.  The 64 MiB Cube-scratch limit bounds this
    // scan to 3276 groups.
    uint64_t maxWork = 0;
    for (uint32_t block = 0; block < blockCount; ++block) {
        uint64_t blockWork = 0;
        for (int64_t group = block; group < plan.groupCount;
             group += static_cast<int64_t>(blockCount)) {
            const int64_t rowStart =
                group * sddmm_arch35::kSddmmCsrCubeRows;
            const int64_t logicalStart =
                (plan.logicalFirstColumn +
                    rowStart * plan.rowColumnStride) % tiling.n;
            const bool gatherable = logicalStart <=
                tiling.n - sddmm_arch35::kSddmmCsrCubeLogicalBand;
            blockWork += gatherable ? 1U :
                static_cast<uint64_t>(sddmm_arch35::kSddmmCsrCubeRows);
        }
        maxWork = blockWork > maxWork ? blockWork : maxWork;
    }
    return maxWork;
}

uint32_t ChooseCsrCubeGatherBlocks(
    const SddmmTilingData &tiling,
    const sddmm_arch35::SddmmCsrCubePlan &plan,
    uint32_t defaultBlocks)
{
    if (defaultBlocks <= 1U || plan.groupCount <= 0 || tiling.n <= 0 ||
        static_cast<uint32_t>(plan.groupCount) <= defaultBlocks ||
        plan.logicalFirstColumn < 0 ||
        static_cast<int64_t>(plan.logicalFirstColumn) >= tiling.n ||
        plan.rowColumnStride <= 0 ||
        tiling.n < sddmm_arch35::kSddmmCsrCubeLogicalBand ||
        (tiling.outputType != SDDMM_VALUE_FP16 &&
            tiling.outputType != SDDMM_VALUE_BF16) ||
        tiling.alphaReal != 1.0F || tiling.betaReal != 0.0F) {
        return defaultBlocks;
    }

    // Identity gather handles a wrapping group row by row.  When the grid
    // stride divides the cyclic group period, those expensive groups remain
    // phase-locked to a small subset of AIVs on every period.  One fewer block
    // can break that phase lock while retaining essentially all parallelism.
    const int64_t groupColumnStep =
        static_cast<int64_t>(sddmm_arch35::kSddmmCsrCubeRows) *
        plan.rowColumnStride;
    const int64_t patternPeriod =
        tiling.n / std::gcd(tiling.n, groupColumnStep);
    if (static_cast<int64_t>(plan.groupCount) < patternPeriod ||
        patternPeriod % static_cast<int64_t>(defaultBlocks) != 0) {
        return defaultBlocks;
    }

    const uint32_t dephasedBlocks = defaultBlocks - 1U;
    return EstimateCsrCubeGatherMaxWork(tiling, plan, dephasedBlocks) <
        EstimateCsrCubeGatherMaxWork(tiling, plan, defaultBlocks) ?
        dephasedBlocks : defaultBlocks;
}

aclsparseStatus_t LaunchCsrCube(const SddmmCallContext &context,
    const SddmmTilingData &tiling, void *workspace, uint32_t numBlocks)
{
    const uint32_t aicCoreCount = GetAicCoreCount();
    if (aicCoreCount == 0U) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    auto plan = context.csrCubePlan;
    plan.logicalFirstColumn = context.matC->sddmmPrepared.patternLogicalFirstColumn;
    const uint32_t cubeBlocks =
        static_cast<uint32_t>(plan.groupCount) < aicCoreCount ?
        static_cast<uint32_t>(plan.groupCount) : aicCoreCount;
    auto *scratch = static_cast<uint8_t *>(workspace) +
        context.csrCubeScratchOffset;
    sddmm_csr_cube_kernel_do(
        reinterpret_cast<GM_ADDR>(context.matX->values),
        reinterpret_cast<GM_ADDR>(context.matY->values),
        reinterpret_cast<GM_ADDR>(scratch), tiling, plan,
        cubeBlocks, context.handle->stream);
    sddmm_csr_cube_gather_kernel_do(
        reinterpret_cast<GM_ADDR>(context.matC->values),
        reinterpret_cast<GM_ADDR>(scratch), tiling, plan,
        ChooseCsrCubeGatherBlocks(tiling, plan, numBlocks),
        context.handle->stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t LaunchComplexBsrCube(const SddmmCallContext &context,
    const SddmmTilingData &tiling, void *workspace, uint32_t numBlocks)
{
    const uint32_t aicCoreCount = GetAicCoreCount();
    if (aicCoreCount == 0U) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    const uint64_t cubeWorkItems =
        static_cast<uint64_t>(tiling.nnzUnits);
    const uint32_t cubeBlocks = cubeWorkItems < aicCoreCount ?
        static_cast<uint32_t>(cubeWorkItems) : aicCoreCount;
    auto *privateWorkspace = static_cast<uint8_t *>(workspace) +
        context.complexBsrCubeWorkspaceOffset;
    sddmm_complex_bsr_cube_kernel_do(
        reinterpret_cast<GM_ADDR>(context.matC->ptrs),
        reinterpret_cast<GM_ADDR>(context.matC->idxs),
        reinterpret_cast<GM_ADDR>(context.matC->values),
        reinterpret_cast<GM_ADDR>(context.matX->values),
        reinterpret_cast<GM_ADDR>(context.matY->values),
        reinterpret_cast<GM_ADDR>(privateWorkspace), tiling,
        cubeBlocks, numBlocks, context.handle->stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t LaunchBsrCube(const SddmmCallContext &context,
    const SddmmTilingData &tiling, void *workspace, uint32_t numBlocks)
{
    const uint32_t aicCoreCount = GetAicCoreCount();
    if (aicCoreCount == 0U) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    const uint64_t cubeWorkItems = static_cast<uint64_t>(tiling.batchCount) *
        static_cast<uint64_t>(tiling.nnzUnits);
    const uint32_t cubeBlocks = cubeWorkItems < aicCoreCount ?
        static_cast<uint32_t>(cubeWorkItems) : aicCoreCount;
    auto *scratch = static_cast<uint8_t *>(workspace) +
        context.bsrCubeScratchOffset;
    sddmm_bsr_cube_kernel_do(
        reinterpret_cast<GM_ADDR>(context.matC->ptrs),
        reinterpret_cast<GM_ADDR>(context.matC->idxs),
        reinterpret_cast<GM_ADDR>(context.matC->values),
        reinterpret_cast<GM_ADDR>(context.matX->values),
        reinterpret_cast<GM_ADDR>(context.matY->values),
        reinterpret_cast<GM_ADDR>(scratch), tiling,
        cubeBlocks, numBlocks, context.handle->stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t LaunchGenericSddmm(const SddmmCallContext &context,
    SddmmTilingData &tiling, void *workspace, uint32_t numBlocks, uint32_t coreCount)
{
    const bool hostZeroAlpha = IsHostZeroAlpha(context, tiling);
    GM_ADDR effectiveY = reinterpret_cast<GM_ADDR>(context.matY->values);
    // A zero alpha needs no dense operand. Device scalars are resolved by the
    // generic kernel before any dense load, so do not prepack them on Host.
    if (hostZeroAlpha || tiling.pointerMode != 0) {
        tiling.usePackedY = 0;
    }
    // Independent CSR outputs access adjacent dense columns together. A full
    // transpose makes those SIMT loads strided for an irregular pattern.
    if (tiling.format == SDDMM_FORMAT_CSR && tiling.inputType == SDDMM_VALUE_FP32 &&
        (tiling.patternFlags & SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64) == 0U) {
        tiling.usePackedY = 0;
    }
    if (tiling.format == SDDMM_FORMAT_BSR &&
        (tiling.pointerMode != 0 || tiling.alphaReal != 1.0F || tiling.betaReal != 0.0F)) {
        tiling.usePackedY = 0;
    }
    if (tiling.usePackedY != 0) {
        auto *packedY = static_cast<uint8_t *>(workspace) + tiling.packedYOffset;
        sddmm_pack_y_kernel_do(
            reinterpret_cast<GM_ADDR>(context.matY->values),
            reinterpret_cast<GM_ADDR>(packedY), tiling, coreCount, context.handle->stream);
        effectiveY = reinterpret_cast<GM_ADDR>(packedY);
    }
    const uint32_t computeBlocks = tiling.format == SDDMM_FORMAT_BSR && tiling.usePackedY != 0 ?
        static_cast<uint32_t>(std::min<uint64_t>(tiling.nnzUnits, coreCount)) : numBlocks;
    tiling.numBlocks = static_cast<int32_t>(computeBlocks);
    sddmm_kernel_do(
        reinterpret_cast<GM_ADDR>(context.matC->ptrs),
        reinterpret_cast<GM_ADDR>(context.matC->idxs),
        reinterpret_cast<GM_ADDR>(context.matC->values),
        reinterpret_cast<GM_ADDR>(context.matX->values),
        effectiveY,
        tiling, computeBlocks, context.handle->stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t LaunchPreparedSddmm(const SddmmCallContext &context,
    SddmmTilingData &tiling, void *workspace, uint32_t numBlocks, uint32_t coreCount)
{
    const auto &preparedState = context.matC->sddmmPrepared;
    const bool hostZeroAlpha = IsHostZeroAlpha(context, tiling);
    if (context.useCsrCube && !hostZeroAlpha &&
        HasCsrCubePattern(preparedState.patternFlags, preparedState.patternLogicalFirstColumn)) {
        return LaunchCsrCube(context, tiling, workspace, numBlocks);
    }
    if (tiling.format == SDDMM_FORMAT_CSR && !hostZeroAlpha &&
        sddmm_arch35::SddmmComplexRowCanRun(tiling)) {
        sddmm_complex_row_kernel_do(
            reinterpret_cast<GM_ADDR>(context.matC->ptrs),
            reinterpret_cast<GM_ADDR>(context.matC->idxs),
            reinterpret_cast<GM_ADDR>(context.matC->values),
            reinterpret_cast<GM_ADDR>(context.matX->values),
            reinterpret_cast<GM_ADDR>(context.matY->values),
            tiling, numBlocks, context.handle->stream);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (context.useComplexBsrCube && !hostZeroAlpha) {
        return LaunchComplexBsrCube(context, tiling, workspace, numBlocks);
    }
    if (context.useBsrCube && !hostZeroAlpha) {
        return LaunchBsrCube(context, tiling, workspace, numBlocks);
    }
    return LaunchGenericSddmm(context, tiling, workspace, numBlocks, coreCount);
}

aclsparseStatus_t LaunchSddmm(
    SddmmCallContext &context, const void *alpha, const void *beta,
    aclDataType computeType, void *workspace)
{
    if (context.m == 0 || context.n == 0 || context.totalOutputs == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    SddmmTilingData tiling = context.tiling;
    // IsPrepared/PreparePattern has bound these flags to the exact descriptor
    // key, workspace, and structure generation used by this launch.
    tiling.patternFlags = context.matC->sddmmPrepared.patternFlags;
    tiling.patternLogicalFirstColumn =
        context.matC->sddmmPrepared.patternLogicalFirstColumn;
    const auto scalarStatus = PopulateScalarTiling(context, alpha, beta, computeType, tiling);
    if (scalarStatus != ACL_SPARSE_STATUS_SUCCESS) {
        return scalarStatus;
    }
    if (IsHostNoOp(context, tiling)) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const uint32_t coreCount = GetAivCoreCount();
    if (coreCount == 0U) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    const uint64_t blocksNeeded =
        (static_cast<uint64_t>(context.totalOutputs) + kSddmmThreadsPerBlock - 1U) /
        kSddmmThreadsPerBlock;
    const uint32_t numBlocks = blocksNeeded < coreCount ? static_cast<uint32_t>(blocksNeeded) : coreCount;
    tiling.numBlocks = static_cast<int32_t>(numBlocks);
    // A warm launch can overwrite another operator's cached tiling with
    // packed Y or Cube scratch even though SDDMM remains prepared.
    InvalidateAliasedWorkspaceOwners(context, workspace);
    return LaunchPreparedSddmm(context, tiling, workspace, numBlocks, coreCount);
}

} // namespace

aclsparseStatus_t aclsparseSDDMMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t algorithm, size_t *size)
{
    if (size == nullptr) {
        return handle == nullptr ? ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR :
                                   ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    SddmmCallContext context{};
    aclsparseStatus_t status = ValidateSddmmCall(
        handle, opX, opY, alpha, matX, matY, beta, matC, computeType, algorithm, context);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = QueryCsrCubeWorkspace(context);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    auto &plan = matC->sddmmWorkspacePlan;
    plan.key = BuildWorkspacePlanKey(context, opX, opY, computeType, algorithm);
    plan.workspaceBytes = context.workspaceBytes;
    plan.queried = true;
    AclsparseInvalidateSddmmPreparedState(matC);
    *size = context.workspaceBytes;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSDDMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t algorithm, void *buffer)
{
    SddmmCallContext context{};
    aclsparseStatus_t status = ValidateSddmmCall(
        handle, opX, opY, alpha, matX, matY, beta, matC, computeType, algorithm, context);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ApplyWorkspacePlan(context, opX, opY, computeType, algorithm);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateWorkspace(context, buffer, true);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    return PreparePattern(context, opX, opY, computeType, algorithm, buffer);
}

aclsparseStatus_t aclsparseSDDMM(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t algorithm, void *buffer)
{
    SddmmCallContext context{};
    aclsparseStatus_t status = ValidateSddmmCall(
        handle, opX, opY, alpha, matX, matY, beta, matC, computeType, algorithm, context);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ApplyWorkspacePlan(context, opX, opY, computeType, algorithm);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateWorkspace(context, buffer, false);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    const SddmmPreparedKey key = BuildPreparedKey(
        context, opX, opY, computeType, algorithm, buffer);
    if (!IsPrepared(context, key)) {
        status = PreparePattern(context, opX, opY, computeType, algorithm, buffer);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
    }
    return LaunchSddmm(context, alpha, beta, computeType, buffer);
}
