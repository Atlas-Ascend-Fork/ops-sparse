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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>

#include "log/log.h"
#include "securec.h"
#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif
#include "spgemm_kernel.h"

namespace {

constexpr const char *kSpGemmTag = "aclsparseSpGEMM";
constexpr size_t kSpGemmAlignment = 128U;

struct SpGemmWorkLayout {
    size_t counts = 0;
    size_t offsets = 0;
    size_t blockSums = 0;
    size_t blockOffsets = 0;
    size_t total = 0;
    size_t regular = 0;
    size_t error = 0;
    size_t bytes = 0;
};

struct SpGemmComputeLayout {
    size_t cursors = 0;
    size_t candidateCols = 0;
    size_t candidateVals = 0;
    size_t uniqueCounts = 0;
    size_t blockSums = 0;
    size_t blockOffsets = 0;
    size_t total = 0;
    size_t error = 0;
    size_t bytes = 0;
};

struct SpGemmLegacyLayout {
    SpGemmWorkLayout work;
    size_t computeOffset = 0;
    size_t computeCapacity = 0;
    size_t zeroScalarOffset = 0;
    size_t bytes = 0;
};

static size_t AlignUp(size_t value)
{
    if (value > std::numeric_limits<size_t>::max() - (kSpGemmAlignment - 1U)) {
        return std::numeric_limits<size_t>::max();
    }
    return (value + kSpGemmAlignment - 1U) & ~(kSpGemmAlignment - 1U);
}

static bool AppendRegion(size_t count, size_t elementSize, size_t &cursor, size_t &offset)
{
    cursor = AlignUp(cursor);
    if (cursor == std::numeric_limits<size_t>::max() ||
        (elementSize != 0U && count > (std::numeric_limits<size_t>::max() - cursor) / elementSize)) {
        return false;
    }
    offset = cursor;
    cursor += count * elementSize;
    return true;
}

static uint32_t ScanChunkSize(uint64_t rows)
{
    // Short/medium matrices benefit from many shallow independent scans.  The
    // 1024-entry chunk remains preferable for very large task-table cases,
    // where it bounds the serial scan of chunk totals.
    return rows <= 32768U ? 32U : kSpGemmScanChunk;
}

static uint64_t NumScanChunks(uint64_t rows)
{
    uint32_t chunkSize = ScanChunkSize(rows);
    if (chunkSize == 0U) {
        OP_LOGE(kSpGemmTag, "scan chunk size must be nonzero");
        return 0U;
    }
    return rows == 0 ? 1U : (rows + chunkSize - 1U) / chunkSize;
}

static bool BuildWorkLayout(uint64_t rows, SpGemmWorkLayout &layout)
{
    size_t cursor = 0;
    uint64_t chunks = NumScanChunks(rows);
    if (chunks == 0U || rows > std::numeric_limits<size_t>::max() ||
        chunks > std::numeric_limits<size_t>::max()) {
        return false;
    }
    bool ok = AppendRegion(static_cast<size_t>(rows), sizeof(int64_t), cursor, layout.counts) &&
        AppendRegion(static_cast<size_t>(rows + 1U), sizeof(int64_t), cursor, layout.offsets) &&
        AppendRegion(static_cast<size_t>(chunks), sizeof(int64_t), cursor, layout.blockSums) &&
        AppendRegion(static_cast<size_t>(chunks), sizeof(int64_t), cursor, layout.blockOffsets) &&
        AppendRegion(1U, sizeof(int64_t), cursor, layout.total) &&
        AppendRegion(1U, sizeof(int32_t), cursor, layout.regular) &&
        AppendRegion(1U, sizeof(int32_t), cursor, layout.error);
    layout.bytes = ok ? AlignUp(cursor) : std::numeric_limits<size_t>::max();
    return ok && layout.bytes != std::numeric_limits<size_t>::max();
}

static size_t ValueTypeSize(aclDataType type)
{
    if (type == ACL_FLOAT16 || type == ACL_BF16) {
        return 2U;
    }
    if (type == ACL_FLOAT) {
        return 4U;
    }
    if (type == ACL_COMPLEX64) {
        return 8U;
    }
    return 0U;
}

static bool BuildComputeLayout(
    uint64_t rows, uint64_t nnzA, uint64_t products,
    aclDataType type, SpGemmComputeLayout &layout)
{
    size_t cursor = 0;
    uint64_t chunks = NumScanChunks(rows);
    size_t valueSize = ValueTypeSize(type);
    if (valueSize == 0U || rows > std::numeric_limits<size_t>::max() ||
        nnzA > std::numeric_limits<size_t>::max() || products > std::numeric_limits<size_t>::max() ||
        chunks > std::numeric_limits<size_t>::max()) {
        return false;
    }
    bool ok = AppendRegion(static_cast<size_t>(nnzA), sizeof(int32_t), cursor, layout.cursors) &&
        AppendRegion(static_cast<size_t>(products), sizeof(int32_t), cursor, layout.candidateCols) &&
        AppendRegion(static_cast<size_t>(products), valueSize, cursor, layout.candidateVals) &&
        AppendRegion(static_cast<size_t>(rows), sizeof(int32_t), cursor, layout.uniqueCounts) &&
        AppendRegion(static_cast<size_t>(chunks), sizeof(int64_t), cursor, layout.blockSums) &&
        AppendRegion(static_cast<size_t>(chunks), sizeof(int64_t), cursor, layout.blockOffsets) &&
        AppendRegion(1U, sizeof(int64_t), cursor, layout.total) &&
        AppendRegion(1U, sizeof(int32_t), cursor, layout.error);
    layout.bytes = ok ? AlignUp(cursor) : std::numeric_limits<size_t>::max();
    return ok && layout.bytes != std::numeric_limits<size_t>::max();
}

static bool MultiplyWithoutOverflow(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (rhs != 0U && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool BuildLegacyLayout(
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    aclDataType type, SpGemmLegacyLayout &layout)
{
    if (!BuildWorkLayout(matA->rows, layout.work)) {
        return false;
    }
    uint64_t productsByA = 0;
    uint64_t productsByRows = 0;
    if (!MultiplyWithoutOverflow(matA->nnz, std::min(matB->cols, matB->nnz), productsByA) ||
        !MultiplyWithoutOverflow(matA->rows, matB->nnz, productsByRows)) {
        return false;
    }
    uint64_t maxProducts = std::min(productsByA, productsByRows);
    maxProducts = std::min<uint64_t>(
        maxProducts, static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
    SpGemmComputeLayout compute{};
    if (!BuildComputeLayout(matA->rows, matA->nnz, maxProducts, type, compute)) {
        return false;
    }
    layout.computeOffset = AlignUp(layout.work.bytes);
    if (layout.computeOffset == std::numeric_limits<size_t>::max() ||
        compute.bytes > std::numeric_limits<size_t>::max() - layout.computeOffset) {
        return false;
    }
    layout.computeCapacity = compute.bytes;
    layout.zeroScalarOffset = AlignUp(layout.computeOffset + compute.bytes);
    constexpr size_t kScalarBytes = sizeof(aclsparseComplex);
    if (layout.zeroScalarOffset == std::numeric_limits<size_t>::max() ||
        kScalarBytes > std::numeric_limits<size_t>::max() - layout.zeroScalarOffset) {
        return false;
    }
    layout.bytes = AlignUp(layout.zeroScalarOffset + kScalarBytes);
    return layout.bytes != std::numeric_limits<size_t>::max();
}

static bool IsSupportedType(aclDataType type)
{
    return type == ACL_FLOAT16 || type == ACL_BF16 ||
        type == ACL_FLOAT || type == ACL_COMPLEX64;
}

static bool IsSupportedAlg(aclsparseSpGEMMAlg_t alg)
{
    return alg == ACL_SPARSE_SPGEMM_DEFAULT || alg == ACL_SPARSE_SPGEMM_ALG1 ||
        alg == ACL_SPARSE_SPGEMM_ALG2 || alg == ACL_SPARSE_SPGEMM_ALG3;
}

static aclsparseStatus_t ValidateDescriptor(const aclsparseSpMatDescr *mat, const char *name)
{
    if (mat == nullptr) {
        OP_LOGE(kSpGemmTag, "%s is nullptr", name);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (mat->format != ACL_SPARSE_FORMAT_CSR) {
        OP_LOGE(kSpGemmTag, "%s must use CSR format", name);
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (mat->ptrType != ACL_SPARSE_INDEX_32I || mat->IdxType != ACL_SPARSE_INDEX_32I) {
        OP_LOGE(kSpGemmTag, "%s requires int32 row offsets and column indices", name);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (mat->baseType != ACL_SPARSE_INDEX_BASE_ZERO) {
        OP_LOGE(kSpGemmTag, "%s requires zero-based indices", name);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    constexpr uint64_t kI32Max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    if (mat->rows > kI32Max || mat->cols > kI32Max || mat->nnz > kI32Max) {
        OP_LOGE(kSpGemmTag, "%s shape/nnz exceeds int32 CSR limits", name);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateInputPointers(const aclsparseSpMatDescr *mat, const char *name)
{
    if (mat->rows > 0 && mat->ptrs == nullptr) {
        OP_LOGE(kSpGemmTag, "%s rowOffsets is nullptr", name);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (mat->nnz > 0 && (mat->idxs == nullptr || mat->values == nullptr)) {
        OP_LOGE(kSpGemmTag, "%s column indices or values is nullptr", name);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateBasicArguments(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t descr)
{
    if (handle == nullptr) {
        OP_LOGE(kSpGemmTag, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (descr == nullptr || descr->signature != kSpGemmSignature || alpha == nullptr || beta == nullptr) {
        OP_LOGE(kSpGemmTag, "descriptor or scalar pointer is invalid");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE || opB != ACL_SPARSE_OP_NON_TRANSPOSE) {
        OP_LOGE(kSpGemmTag, "only NON_TRANSPOSE is supported, opA=%d, opB=%d",
            static_cast<int>(opA), static_cast<int>(opB));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!IsSupportedAlg(alg) || !IsSupportedType(computeType)) {
        OP_LOGE(kSpGemmTag, "unsupported algorithm or compute type, alg=%d, computeType=%d",
            static_cast<int>(alg), static_cast<int>(computeType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateMatrices(
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    aclsparseSpMatDescr_t matC, aclDataType computeType)
{
    aclsparseStatus_t status = ValidateDescriptor(matA, "matA");
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateDescriptor(matB, "matB");
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateDescriptor(matC, "matC");
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateInputPointers(matA, "matA");
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateInputPointers(matB, "matB");
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (matA->cols != matB->rows || matC->rows != matA->rows || matC->cols != matB->cols) {
        OP_LOGE(kSpGemmTag, "matrix dimensions are incompatible");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->valueType != computeType || matB->valueType != computeType || matC->valueType != computeType) {
        OP_LOGE(kSpGemmTag, "matrix value types must match computeType=%d", static_cast<int>(computeType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateCommon(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t descr)
{
    aclsparseStatus_t status = ValidateBasicArguments(
        handle, opA, opB, alpha, beta, computeType, alg, descr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    return ValidateMatrices(matA, matB, matC, computeType);
}

static void RecordProblem(
    aclsparseSpGEMMDescr *descr, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, aclsparseSpMatDescr_t matC,
    aclDataType computeType, aclsparseSpGEMMAlg_t alg)
{
    descr->matA = matA;
    descr->matB = matB;
    descr->matC = matC;
    descr->m = matA->rows;
    descr->k = matA->cols;
    descr->n = matB->cols;
    descr->nnzA = matA->nnz;
    descr->nnzB = matB->nnz;
    descr->computeType = computeType;
    descr->alg = alg;
}

static bool ProblemMatches(
    const aclsparseSpGEMMDescr *descr, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, aclsparseSpMatDescr_t matC,
    aclDataType computeType, aclsparseSpGEMMAlg_t alg)
{
    return descr->matA == matA && descr->matB == matB && descr->matC == matC &&
        descr->m == matA->rows && descr->k == matA->cols && descr->n == matB->cols &&
        descr->nnzA == matA->nnz && descr->nnzB == matB->nnz &&
        descr->computeType == computeType && descr->alg == alg;
}

static uint32_t LaunchBlocks(uint64_t rows)
{
    uint32_t cores = GetAivCoreCount();
    if (cores == 0U) {
        return 0U;
    }
    uint64_t needed = (rows + kSpGemmThreads - 1U) / kSpGemmThreads;
    if (needed == 0U) {
        needed = 1U;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(cores, needed));
}

static uint32_t RowsPerBlock(uint64_t rows, uint32_t blocks)
{
    return blocks == 0U ? 0U : static_cast<uint32_t>((rows + blocks - 1U) / blocks);
}

static uint32_t ScanBlocks(uint64_t rows)
{
    uint32_t cores = GetAivCoreCount();
    if (cores == 0U) {
        return 0U;
    }
    // One SIMT thread scans one kSpGemmScanChunk chunk. Launching every AIV
    // core for a short row array leaves almost all blocks idle.
    uint64_t chunks = NumScanChunks(rows);
    uint64_t needed = (chunks + kSpGemmThreads - 1U) / kSpGemmThreads;
    return static_cast<uint32_t>(std::min<uint64_t>(cores, std::max<uint64_t>(1U, needed)));
}

static int32_t MapValueType(aclDataType type)
{
    if (type == ACL_FLOAT16) {
        return SPGEMM_VAL_FP16;
    }
    if (type == ACL_BF16) {
        return SPGEMM_VAL_BF16;
    }
    if (type == ACL_FLOAT) {
        return SPGEMM_VAL_FP32;
    }
    return SPGEMM_VAL_COMPLEX64;
}

static bool CopyHostValue(void *destination, size_t destinationSize,
    const void *source, size_t sourceSize)
{
    if (memcpy_s(destination, destinationSize, source, sourceSize) != EOK) {
        OP_LOGE(kSpGemmTag, "failed to copy a host scalar value");
        return false;
    }
    return true;
}

static bool HalfToFloat(uint16_t bits, float &value)
{
    uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
    uint32_t exponent = (bits >> 10U) & 0x1FU;
    uint32_t mantissa = bits & 0x03FFU;
    uint32_t result = 0;
    if (exponent == 0U && mantissa == 0U) {
        result = sign;
    } else if (exponent == 0U) {
        uint32_t shift = static_cast<uint32_t>(__builtin_clz(mantissa)) - 21U;
        mantissa <<= shift;
        result = sign | ((127U - 14U - shift) << 23U) | ((mantissa << 13U) & 0x7FFFFFU);
    } else if (exponent == 31U) {
        result = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        result = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    value = 0.0F;
    return CopyHostValue(&value, sizeof(value), &result, sizeof(result));
}

static bool BFloat16ToFloat(uint16_t bits, float &value)
{
    uint32_t result = static_cast<uint32_t>(bits) << 16U;
    value = 0.0F;
    return CopyHostValue(&value, sizeof(value), &result, sizeof(result));
}

static bool ReadHostScalar(const void *ptr, aclDataType type, float &real, float &imag)
{
    real = 0.0F;
    imag = 0.0F;
    if (type == ACL_FLOAT16 || type == ACL_BF16) {
        uint16_t bits = 0;
        if (!CopyHostValue(&bits, sizeof(bits), ptr, sizeof(bits))) {
            return false;
        }
        return type == ACL_FLOAT16 ? HalfToFloat(bits, real) : BFloat16ToFloat(bits, real);
    } else if (type == ACL_FLOAT) {
        return CopyHostValue(&real, sizeof(real), ptr, sizeof(real));
    } else {
        aclsparseComplex value{};
        if (!CopyHostValue(&value, sizeof(value), ptr, sizeof(value))) {
            return false;
        }
        real = value.x;
        imag = value.y;
        return true;
    }
}

static aclsparseStatus_t ClearDevice(void *ptr, size_t bytes, aclrtStream stream)
{
    aclError ret = aclrtMemsetAsync(ptr, bytes, 0, bytes, stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(kSpGemmTag, "aclrtMemsetAsync failed, bytes=%zu, ret=%d", bytes, static_cast<int>(ret));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ReadDeviceResult(
    const void *totalDevice, const void *regularDevice, const void *errorDevice,
    aclrtStream stream, int64_t &total, int32_t &regular, int32_t &error)
{
    aclError ret = aclrtMemcpyAsync(&total, sizeof(total), totalDevice, sizeof(total),
        ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret == ACL_SUCCESS && regularDevice != nullptr) {
        ret = aclrtMemcpyAsync(&regular, sizeof(regular), regularDevice, sizeof(regular),
            ACL_MEMCPY_DEVICE_TO_HOST, stream);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpyAsync(&error, sizeof(error), errorDevice, sizeof(error),
            ACL_MEMCPY_DEVICE_TO_HOST, stream);
    }
    if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        OP_LOGE(kSpGemmTag, "failed to read or synchronize a device stage result, ret=%d",
            static_cast<int>(ret));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static bool IsRegularCandidate(
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB)
{
    return matA->rows > 0 && matA->rows == matA->cols &&
        matA->rows == matB->rows && matA->rows == matB->cols && matA->nnz == matB->nnz &&
        matA->nnz % matA->rows == 0 && matA->nnz / matA->rows > 0 &&
        matA->nnz / matA->rows <= 8 &&
        (matA->nnz / matA->rows) * (matA->nnz / matA->rows) <= matA->rows;
}

static aclsparseStatus_t InitializeWorkEstimation(
    uint8_t *base, const SpGemmWorkLayout &layout,
    bool regularCandidate, aclrtStream stream)
{
    aclsparseStatus_t status = ClearDevice(base + layout.error, sizeof(int32_t), stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    aclError ret = aclrtMemsetAsync(base + layout.regular, sizeof(int32_t),
        regularCandidate ? 0xFF : 0, sizeof(int32_t), stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(kSpGemmTag, "failed to initialize regular-path state, ret=%d", static_cast<int>(ret));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static SpGemmValidateTilingData MakeValidateTiling(
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    bool regularCandidate, uint64_t validationRows, uint32_t validateBlocks)
{
    SpGemmValidateTilingData validate{};
    validate.m = static_cast<int32_t>(matA->rows);
    validate.k = static_cast<int32_t>(matA->cols);
    validate.n = static_cast<int32_t>(matB->cols);
    validate.nnzA = static_cast<int32_t>(matA->nnz);
    validate.nnzB = static_cast<int32_t>(matB->nnz);
    validate.regularDegree = regularCandidate ?
        static_cast<int32_t>(matA->nnz / matA->rows) : 0;
    validate.rowsPerBlock = RowsPerBlock(validationRows, validateBlocks);
    return validate;
}

static aclsparseStatus_t LaunchValidation(
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    uint8_t *base, const SpGemmWorkLayout &layout,
    const SpGemmValidateTilingData &validate, uint32_t validateBlocks,
    aclrtStream stream)
{
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    spgemm_validate_kernel_do(
        reinterpret_cast<GM_ADDR>(matA->ptrs), reinterpret_cast<GM_ADDR>(matA->idxs),
        reinterpret_cast<GM_ADDR>(matB->ptrs), reinterpret_cast<GM_ADDR>(matB->idxs),
        reinterpret_cast<GM_ADDR>(base + layout.offsets),
        reinterpret_cast<GM_ADDR>(base + layout.total),
        reinterpret_cast<GM_ADDR>(base + layout.regular),
        reinterpret_cast<GM_ADDR>(base + layout.error), validate, validateBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpGEMM", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchProductCount(
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    uint8_t *base, const SpGemmWorkLayout &layout,
    uint32_t workBlocks, aclrtStream stream)
{
    if (matA->rows == 0) {
        aclsparseStatus_t status = ClearDevice(base + layout.offsets, sizeof(int64_t), stream);
        if (status == ACL_SPARSE_STATUS_SUCCESS) {
            status = ClearDevice(base + layout.total, sizeof(int64_t), stream);
        }
        return status;
    }
    SpGemmWorkTilingData work{};
    work.m = static_cast<int32_t>(matA->rows);
    work.k = static_cast<int32_t>(matA->cols);
    work.nnzA = static_cast<int32_t>(matA->nnz);
    work.nnzB = static_cast<int32_t>(matB->nnz);
    work.rowsPerBlock = RowsPerBlock(matA->rows, workBlocks);
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    spgemm_work_kernel_do(
        reinterpret_cast<GM_ADDR>(matA->ptrs), reinterpret_cast<GM_ADDR>(matA->idxs),
        reinterpret_cast<GM_ADDR>(matB->ptrs), reinterpret_cast<GM_ADDR>(base + layout.counts),
        reinterpret_cast<GM_ADDR>(base + layout.regular),
        reinterpret_cast<GM_ADDR>(base + layout.error), work, workBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpGEMM", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }


    SpGemmScanTilingData scan{};
    scan.count = static_cast<int32_t>(matA->rows);
    scan.numChunks = static_cast<int32_t>(NumScanChunks(matA->rows));
    scan.chunkSize = static_cast<int32_t>(ScanChunkSize(matA->rows));
    scan.outerBlocks = ScanBlocks(matA->rows);
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    spgemm_scan_i64_kernel_do(
        reinterpret_cast<GM_ADDR>(base + layout.counts),
        reinterpret_cast<GM_ADDR>(base + layout.offsets),
        reinterpret_cast<GM_ADDR>(base + layout.blockSums),
        reinterpret_cast<GM_ADDR>(base + layout.blockOffsets),
        reinterpret_cast<GM_ADDR>(base + layout.total),
        reinterpret_cast<GM_ADDR>(base + layout.regular),
        reinterpret_cast<GM_ADDR>(base + layout.error), scan, scan.outerBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpGEMM", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }


    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchWorkEstimation(
    aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, aclsparseSpGEMMDescr_t descr,
    const SpGemmWorkLayout &layout, void *buffer)
{
    auto *context = reinterpret_cast<aclsparseContext *>(handle);
    if (context->stream == nullptr) {
        OP_LOGE(kSpGemmTag, "handle stream is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *base = static_cast<uint8_t *>(buffer);
    bool regularCandidate = IsRegularCandidate(matA, matB);
    aclsparseStatus_t status = InitializeWorkEstimation(
        base, layout, regularCandidate, context->stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    uint64_t validationRows = std::max(matA->rows, matB->rows);
    uint32_t validateBlocks = LaunchBlocks(validationRows);
    uint32_t workBlocks = LaunchBlocks(matA->rows);
    if (validateBlocks == 0U || workBlocks == 0U) {
        OP_LOGE(kSpGemmTag, "failed to obtain a valid AIV block count");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    SpGemmValidateTilingData validate = MakeValidateTiling(
        matA, matB, regularCandidate, validationRows, validateBlocks);
    LaunchValidation(matA, matB, base, layout, validate, validateBlocks, context->stream);
    status = LaunchProductCount(matA, matB, base, layout, workBlocks, context->stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    int64_t products = 0;
    int32_t regular = 0;
    int32_t error = 0;
    status = ReadDeviceResult(base + layout.total, base + layout.regular,
        base + layout.error, context->stream, products, regular, error);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (error != 0 || products < 0 || products > std::numeric_limits<int32_t>::max()) {
        OP_LOGE(kSpGemmTag, "invalid CSR input or product count, error=%d, products=%lld",
            error, static_cast<long long>(products));
        return error != 0 ? ACL_SPARSE_STATUS_INVALID_VALUE : ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    descr->numProducts = products;
    descr->regularDegree = regular != 0 ? validate.regularDegree : 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t BuildComputeTiling(
    aclsparseHandle_t handle, const void *alpha, const void *beta,
    aclDataType type, uint64_t rows, uint32_t blocks,
    SpGemmComputeTilingData &tiling)
{
    auto *context = reinterpret_cast<aclsparseContext *>(handle);
    tiling.valType = MapValueType(type);
    tiling.rowsPerBlock = RowsPerBlock(rows, blocks);
    if (context->pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        if (!ReadHostScalar(alpha, type, tiling.alphaReal, tiling.alphaImag) ||
            !ReadHostScalar(beta, type, tiling.betaReal, tiling.betaImag)) {
            return ACL_SPARSE_STATUS_INTERNAL_ERROR;
        }
    } else {
        tiling.alphaPtr = reinterpret_cast<uint64_t>(alpha);
        tiling.betaPtr = reinterpret_cast<uint64_t>(beta);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateComputeOutput(
    aclsparseHandle_t handle, aclsparseSpMatDescr_t matC,
    const void *beta, aclDataType type)
{
    if (matC->ptrs == nullptr) {
        OP_LOGE(kSpGemmTag, "matC rowOffsets is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *context = reinterpret_cast<aclsparseContext *>(handle);
    if (context->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    float real = 0.0F;
    float imag = 0.0F;
    if (!ReadHostScalar(beta, type, real, imag)) {
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    bool betaNonzero = real != 0.0F || imag != 0.0F;
    if (betaNonzero && matC->nnz > 0 && (matC->idxs == nullptr || matC->values == nullptr)) {
        OP_LOGE(kSpGemmTag, "beta is nonzero but matC column indices or values is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchComputeKernel(
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    aclsparseSpMatDescr_t matC, uint8_t *workBase, uint8_t *computeBase,
    const SpGemmWorkLayout &workLayout, const SpGemmComputeLayout &computeLayout,
    const SpGemmComputeTilingData &compute, uint32_t blocks, aclrtStream stream)
{
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    spgemm_compute_kernel_do(
        reinterpret_cast<GM_ADDR>(matA->ptrs), reinterpret_cast<GM_ADDR>(matA->idxs),
        reinterpret_cast<GM_ADDR>(matA->values), reinterpret_cast<GM_ADDR>(matB->ptrs),
        reinterpret_cast<GM_ADDR>(matB->idxs), reinterpret_cast<GM_ADDR>(matB->values),
        reinterpret_cast<GM_ADDR>(matC->ptrs), reinterpret_cast<GM_ADDR>(matC->idxs),
        reinterpret_cast<GM_ADDR>(matC->values), reinterpret_cast<GM_ADDR>(workBase + workLayout.offsets),
        reinterpret_cast<GM_ADDR>(computeBase + computeLayout.cursors),
        reinterpret_cast<GM_ADDR>(computeBase + computeLayout.candidateCols),
        reinterpret_cast<GM_ADDR>(computeBase + computeLayout.candidateVals),
        reinterpret_cast<GM_ADDR>(computeBase + computeLayout.uniqueCounts),
        reinterpret_cast<GM_ADDR>(computeBase + computeLayout.error), compute, blocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpGEMM", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchNnzScan(
    aclsparseConstSpMatDescr_t matA, aclsparseSpMatDescr_t matC,
    uint8_t *base, const SpGemmComputeLayout &layout, aclrtStream stream)
{
    SpGemmScanTilingData scan{};
    scan.count = static_cast<int32_t>(matA->rows);
    scan.numChunks = static_cast<int32_t>(NumScanChunks(matA->rows));
    scan.chunkSize = static_cast<int32_t>(ScanChunkSize(matA->rows));
    scan.outerBlocks = ScanBlocks(matA->rows);
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    spgemm_scan_i32_kernel_do(
        reinterpret_cast<GM_ADDR>(base + layout.uniqueCounts),
        reinterpret_cast<GM_ADDR>(matC->ptrs),
        reinterpret_cast<GM_ADDR>(base + layout.blockSums),
        reinterpret_cast<GM_ADDR>(base + layout.blockOffsets),
        reinterpret_cast<GM_ADDR>(base + layout.total),
        reinterpret_cast<GM_ADDR>(base + layout.error), scan, scan.outerBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpGEMM", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchNonEmptyCompute(
    aclsparseHandle_t handle, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    const void *beta, aclsparseSpMatDescr_t matC,
    aclsparseSpGEMMDescr_t descr, const SpGemmComputeLayout &layout,
    uint8_t *base, uint32_t blocks, bool &complete)
{
    auto *context = reinterpret_cast<aclsparseContext *>(handle);
    SpGemmComputeTilingData compute{};
    compute.m = static_cast<int32_t>(matA->rows);
    compute.k = static_cast<int32_t>(matA->cols);
    compute.n = static_cast<int32_t>(matB->cols);
    compute.nnzA = static_cast<int32_t>(matA->nnz);
    compute.nnzB = static_cast<int32_t>(matB->nnz);
    aclsparseStatus_t status = BuildComputeTiling(
        handle, alpha, beta, descr->computeType, matA->rows, blocks, compute);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    // DEVICE 模式不在 Host 侧读取标量，避免为判断 beta 引入 D2H 同步；
    // 由通用 Kernel 直接读取 device beta，可同时覆盖 beta 为零和非零的情况。
    bool hostZeroBeta = !descr->forceGenericPath &&
        context->pointerMode == ACL_SPARSE_POINTER_MODE_HOST &&
        compute.betaReal == 0.0F && compute.betaImag == 0.0F;
    compute.regularDegree = hostZeroBeta ? descr->regularDegree : 0;
    compute.directOutput = compute.regularDegree > 0 && matC->idxs != nullptr &&
        matC->values != nullptr && matC->nnz == static_cast<uint64_t>(descr->numProducts);
    SpGemmWorkLayout workLayout{};
    if (!BuildWorkLayout(matA->rows, workLayout)) {
        OP_LOGE(kSpGemmTag, "failed to build the work workspace layout");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *workBase = static_cast<uint8_t *>(descr->externalBuffer1);
    status = LaunchComputeKernel(matA, matB, matC, workBase, base,
        workLayout, layout, compute, blocks, context->stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    complete = compute.regularDegree > 0;
    if (complete) {
        matC->nnz = static_cast<uint64_t>(descr->numProducts);
        descr->nnzC = descr->numProducts;
        descr->copyRequired = compute.directOutput == 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    status = LaunchNnzScan(matA, matC, base, layout, context->stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t FinalizeCompute(
    uint8_t *base, const SpGemmComputeLayout &layout, aclrtStream stream,
    aclsparseSpMatDescr_t matC, aclsparseSpGEMMDescr_t descr)
{
    int64_t nnzC = 0;
    int32_t unused = 0;
    int32_t error = 0;
    aclsparseStatus_t status = ReadDeviceResult(
        base + layout.total, nullptr, base + layout.error, stream, nnzC, unused, error);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (error != 0 || nnzC < 0 || nnzC > std::numeric_limits<int32_t>::max()) {
        OP_LOGE(kSpGemmTag, "invalid compute result, error=%d, nnzC=%lld",
            error, static_cast<long long>(nnzC));
        return error != 0 ? ACL_SPARSE_STATUS_INVALID_VALUE : ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    matC->nnz = static_cast<uint64_t>(nnzC);
    descr->nnzC = nnzC;
    descr->copyRequired = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchCompute(
    aclsparseHandle_t handle, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    const void *beta, aclsparseSpMatDescr_t matC,
    aclsparseSpGEMMDescr_t descr, const SpGemmComputeLayout &layout,
    void *buffer)
{
    auto *context = reinterpret_cast<aclsparseContext *>(handle);
    if (context->stream == nullptr) {
        OP_LOGE(kSpGemmTag, "handle stream is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseStatus_t status = ValidateComputeOutput(handle, matC, beta, descr->computeType);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    auto *base = static_cast<uint8_t *>(buffer);
    status = ClearDevice(base + layout.error, sizeof(int32_t), context->stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    uint32_t blocks = LaunchBlocks(matA->rows);
    if (blocks == 0U) {
        OP_LOGE(kSpGemmTag, "failed to obtain a valid AIV block count");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    bool complete = false;
    if (matA->rows > 0) {
        status = LaunchNonEmptyCompute(
            handle, alpha, matA, matB, beta, matC, descr, layout, base, blocks, complete);
        if (status != ACL_SPARSE_STATUS_SUCCESS || complete) {
            return status;
        }
    } else {
        status = ClearDevice(matC->ptrs, sizeof(int32_t), context->stream);
        if (status == ACL_SPARSE_STATUS_SUCCESS) {
            status = ClearDevice(base + layout.total, sizeof(int64_t), context->stream);
        }
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
    }
    return FinalizeCompute(base, layout, context->stream, matC, descr);
}

static aclsparseStatus_t ValidateStageProblem(
    const aclsparseSpGEMMDescr *descr, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, aclsparseSpMatDescr_t matC,
    aclDataType type, aclsparseSpGEMMAlg_t alg)
{
    return ProblemMatches(descr, matA, matB, matC, type, alg) ?
        ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_INVALID_VALUE;
}

static aclsparseStatus_t ValidateCopyStage(
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t descr)
{
    if (descr->state != AclsparseSpGemmState::COMPUTED ||
        ValidateStageProblem(descr, matA, matB, matC, computeType, alg) != ACL_SPARSE_STATUS_SUCCESS ||
        descr->externalBuffer1 == nullptr || descr->externalBuffer2 == nullptr) {
        OP_LOGE(kSpGemmTag, "invalid Copy state, stage problem, or workspace lifetime");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matC->ptrs == nullptr || (matC->nnz > 0 && (matC->idxs == nullptr || matC->values == nullptr))) {
        OP_LOGE(kSpGemmTag, "matC output pointers are invalid for nnz=%llu",
            static_cast<unsigned long long>(matC->nnz));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchCopy(
    aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMDescr_t descr)
{
    auto *context = reinterpret_cast<aclsparseContext *>(handle);
    if (context->stream == nullptr) {
        OP_LOGE(kSpGemmTag, "handle stream is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    SpGemmWorkLayout workLayout{};
    SpGemmComputeLayout computeLayout{};
    if (!BuildWorkLayout(matA->rows, workLayout) ||
        !BuildComputeLayout(matA->rows, matA->nnz,
            static_cast<uint64_t>(descr->numProducts), computeType, computeLayout)) {
        OP_LOGE(kSpGemmTag, "failed to rebuild a workspace layout for Copy");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    uint32_t blocks = LaunchBlocks(matA->rows);
    if (blocks == 0U) {
        OP_LOGE(kSpGemmTag, "failed to obtain a valid AIV block count");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    SpGemmCopyTilingData copy{};
    copy.m = static_cast<int32_t>(matA->rows);
    copy.valType = MapValueType(computeType);
    copy.numProducts = static_cast<int32_t>(descr->numProducts);
    copy.nnzC = static_cast<int32_t>(descr->nnzC);
    copy.rowsPerBlock = RowsPerBlock(matA->rows, blocks);
    copy.numBlocks = blocks;
    auto *workBase = static_cast<uint8_t *>(descr->externalBuffer1);
    auto *computeBase = static_cast<uint8_t *>(descr->externalBuffer2);
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    spgemm_copy_kernel_do(
        reinterpret_cast<GM_ADDR>(workBase + workLayout.offsets),
        reinterpret_cast<GM_ADDR>(computeBase + computeLayout.uniqueCounts),
        reinterpret_cast<GM_ADDR>(computeBase + computeLayout.candidateCols),
        reinterpret_cast<GM_ADDR>(computeBase + computeLayout.candidateVals),
        reinterpret_cast<GM_ADDR>(matC->ptrs), reinterpret_cast<GM_ADDR>(matC->idxs),
        reinterpret_cast<GM_ADDR>(matC->values), copy, blocks, context->stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpGEMM", "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }


    return ACL_SPARSE_STATUS_SUCCESS;
}

static void ReleaseLegacyDescriptor(aclsparseSpMatDescr_t matC)
{
    if (matC != nullptr) {
        matC->legacySpGemmDescr.reset();
        matC->legacySpGemmBuffer = nullptr;
    }
}

static aclsparseStatus_t PrepareLegacyStages(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t descr,
    const SpGemmLegacyLayout &layout, void *buffer, size_t &size2)
{
    size_t size1 = 0;
    aclsparseStatus_t status = aclsparseSpGEMMWorkEstimation(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
        descr, &size1, nullptr);
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        status = aclsparseSpGEMMWorkEstimation(
            handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
            descr, &size1, buffer);
    }
    size_t size3 = 0;
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        status = aclsparseSpGEMMEstimateMemory(
            handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
            descr, 1.0F, &size3, nullptr, &size2);
    }
    if (status == ACL_SPARSE_STATUS_SUCCESS && size2 > layout.computeCapacity) {
        OP_LOGE(kSpGemmTag, "legacy compute workspace exceeds queried capacity");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    return status;
}

static aclsparseStatus_t LaunchLegacyCompute(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, aclsparseSpMatDescr_t matC,
    aclDataType computeType, aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t descr, const SpGemmLegacyLayout &layout,
    void *buffer, size_t &size2)
{
    uint64_t hostZero = 0U;
    const void *zeroBeta = &hostZero;
    auto *context = reinterpret_cast<aclsparseContext *>(handle);
    auto *base = static_cast<uint8_t *>(buffer);
    if (context->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        zeroBeta = base + layout.zeroScalarOffset;
        aclsparseStatus_t status = ClearDevice(
            const_cast<void *>(zeroBeta), sizeof(aclsparseComplex), context->stream);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
    }
    descr->forceGenericPath = true;
    aclsparseStatus_t status = aclsparseSpGEMMCompute(
        handle, opA, opB, alpha, matA, matB, zeroBeta, matC, computeType, alg,
        descr, &size2, base + layout.computeOffset);
    descr->forceGenericPath = false;
    return status;
}

}  // namespace

extern "C" aclsparseStatus_t aclsparseSpGEMMCreateDescr(aclsparseSpGEMMDescr_t *descr)
{
    if (descr == nullptr || *descr != nullptr) {
        OP_LOGE(kSpGemmTag, "descriptor output is nullptr or already initialized");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *inner = new (std::nothrow) aclsparseSpGEMMDescr();
    if (inner == nullptr) {
        OP_LOGE(kSpGemmTag, "failed to allocate the SpGEMM descriptor");
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    inner->signature = kSpGemmSignature;
    *descr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseSpGEMMDestroyDescr(aclsparseSpGEMMDescr_t descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (descr->signature != kSpGemmSignature) {
        OP_LOGE(kSpGemmTag, "descriptor signature is invalid");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    delete descr;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseSpGEMMWorkEstimation(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t descr,
    size_t *bufferSize1, void *externalBuffer1)
{
    aclsparseStatus_t status = ValidateCommon(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, descr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (bufferSize1 == nullptr) {
        OP_LOGE(kSpGemmTag, "bufferSize1 is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    SpGemmWorkLayout layout{};
    if (!BuildWorkLayout(matA->rows, layout)) {
        OP_LOGE(kSpGemmTag, "failed to build the work workspace layout");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (externalBuffer1 == nullptr) {
        RecordProblem(descr, matA, matB, matC, computeType, alg);
        descr->requiredBuffer1 = layout.bytes;
        descr->state = AclsparseSpGemmState::WORK_SIZE_QUERIED;
        *bufferSize1 = layout.bytes;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (!ProblemMatches(descr, matA, matB, matC, computeType, alg) || *bufferSize1 < layout.bytes) {
        OP_LOGE(kSpGemmTag, "problem changed or buffer1 is too small, provided=%zu, required=%zu",
            *bufferSize1, layout.bytes);
        return *bufferSize1 < layout.bytes ? ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES :
            ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    status = LaunchWorkEstimation(handle, matA, matB, descr, layout, externalBuffer1);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    descr->externalBuffer1 = externalBuffer1;
    descr->state = AclsparseSpGemmState::WORK_ESTIMATED;
    *bufferSize1 = layout.bytes;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseSpGEMMGetNumProducts(
    aclsparseSpGEMMDescr_t descr, int64_t *numProds)
{
    if (descr == nullptr || descr->signature != kSpGemmSignature || numProds == nullptr) {
        OP_LOGE(kSpGemmTag, "descriptor or numProds pointer is invalid");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (descr->state < AclsparseSpGemmState::WORK_ESTIMATED) {
        OP_LOGE(kSpGemmTag, "WorkEstimation must execute before GetNumProducts");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *numProds = descr->numProducts;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseSpGEMMEstimateMemory(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t descr,
    float chunkFraction, size_t *bufferSize3, void *externalBuffer3,
    size_t *bufferSize2)
{
    (void)externalBuffer3;
    aclsparseStatus_t status = ValidateCommon(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, descr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (bufferSize3 == nullptr || bufferSize2 == nullptr ||
        descr->state < AclsparseSpGemmState::WORK_ESTIMATED ||
        ValidateStageProblem(descr, matA, matB, matC, computeType, alg) != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kSpGemmTag, "invalid memory-estimation output pointer, state, or stage problem");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if ((alg == ACL_SPARSE_SPGEMM_ALG2 || alg == ACL_SPARSE_SPGEMM_ALG3) &&
        (!(chunkFraction > 0.0F) || chunkFraction > 1.0F)) {
        OP_LOGE(kSpGemmTag, "chunkFraction must be in (0, 1] for ALG2/ALG3, got %.8f", chunkFraction);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    SpGemmComputeLayout layout{};
    if (!BuildComputeLayout(matA->rows, matA->nnz,
        static_cast<uint64_t>(descr->numProducts), computeType, layout)) {
        OP_LOGE(kSpGemmTag, "failed to build the compute workspace layout");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *bufferSize3 = 0U;
    *bufferSize2 = layout.bytes;
    descr->requiredBuffer2 = layout.bytes;
    descr->chunkFraction = chunkFraction;
    descr->state = AclsparseSpGemmState::MEMORY_ESTIMATED;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseSpGEMMCompute(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t descr,
    size_t *bufferSize2, void *externalBuffer2)
{
    aclsparseStatus_t status = ValidateCommon(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, descr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (bufferSize2 == nullptr || descr->state < AclsparseSpGemmState::WORK_ESTIMATED ||
        ValidateStageProblem(descr, matA, matB, matC, computeType, alg) != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kSpGemmTag, "invalid bufferSize2, state, or stage problem");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    SpGemmComputeLayout layout{};
    if (!BuildComputeLayout(matA->rows, matA->nnz,
        static_cast<uint64_t>(descr->numProducts), computeType, layout)) {
        OP_LOGE(kSpGemmTag, "failed to build the compute workspace layout");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (externalBuffer2 == nullptr) {
        *bufferSize2 = layout.bytes;
        descr->requiredBuffer2 = layout.bytes;
        descr->state = AclsparseSpGemmState::COMPUTE_SIZE_QUERIED;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (*bufferSize2 < layout.bytes || descr->externalBuffer1 == nullptr) {
        OP_LOGE(kSpGemmTag, "buffer2 is too small or WorkEstimation has no workspace, provided=%zu, required=%zu",
            *bufferSize2, layout.bytes);
        return *bufferSize2 < layout.bytes ? ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES :
            ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    status = LaunchCompute(handle, alpha, matA, matB, beta, matC, descr, layout, externalBuffer2);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    descr->externalBuffer2 = externalBuffer2;
    descr->state = AclsparseSpGemmState::COMPUTED;
    *bufferSize2 = layout.bytes;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseSpGEMMCopy(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t descr)
{
    aclsparseStatus_t status = ValidateCommon(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, descr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateCopyStage(matA, matB, matC, computeType, alg, descr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (!descr->copyRequired) {
        descr->state = AclsparseSpGemmState::COPIED;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (matC->nnz == 0) {
        descr->state = AclsparseSpGemmState::COPIED;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    status = LaunchCopy(handle, matA, matC, computeType, descr);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    descr->state = AclsparseSpGemmState::COPIED;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseSpGEMMSetCInValid(
    aclsparseSpGEMMDescr_t descr, int32_t cInDataValid)
{
    if (descr == nullptr || descr->signature != kSpGemmSignature ||
        descr->state < AclsparseSpGemmState::WORK_ESTIMATED) {
        OP_LOGE(kSpGemmTag, "WorkEstimation must execute before SetCInValid");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    descr->cInDataValid = cInDataValid != 0 ? 1 : 0;
    if (descr->matC != nullptr) {
        descr->matC->cInDataValid = descr->cInDataValid;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseSpGEMMGetBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, size_t *size)
{
    if (size == nullptr) {
        OP_LOGE(kSpGemmTag, "legacy workspace size output is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseSpGEMMDescr validator{};
    validator.signature = kSpGemmSignature;
    aclsparseStatus_t status = ValidateCommon(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, &validator);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    SpGemmLegacyLayout layout{};
    if (!BuildLegacyLayout(matA, matB, computeType, layout)) {
        OP_LOGE(kSpGemmTag, "failed to build the legacy workspace layout");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    *size = layout.bytes;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseSpGEMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, void *buffer)
{
    if (buffer == nullptr) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    size_t legacyBytes = 0;
    aclsparseStatus_t status = aclsparseSpGEMMGetBufferSize(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, &legacyBytes);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    SpGemmLegacyLayout layout{};
    if (!BuildLegacyLayout(matA, matB, computeType, layout) || layout.bytes != legacyBytes) {
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    ReleaseLegacyDescriptor(matC);
    auto *descr = new (std::nothrow) aclsparseSpGEMMDescr();
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    descr->signature = kSpGemmSignature;
    size_t size2 = 0;
    status = PrepareLegacyStages(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
        descr, layout, buffer, size2);
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        status = LaunchLegacyCompute(
            handle, opA, opB, alpha, matA, matB, matC, computeType, alg,
            descr, layout, buffer, size2);
    }
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        delete descr;
        return status;
    }
    matC->legacySpGemmDescr.reset(descr);
    matC->legacySpGemmBuffer = buffer;
    matC->activeBuffer = buffer;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseSpGEMM(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg, void *buffer)
{
    if (buffer == nullptr) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    size_t legacyBytes = 0;
    aclsparseStatus_t status = aclsparseSpGEMMGetBufferSize(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, &legacyBytes);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (matC->legacySpGemmDescr == nullptr || matC->legacySpGemmBuffer != buffer ||
        !ProblemMatches(matC->legacySpGemmDescr.get(), matA, matB, matC, computeType, alg)) {
        status = aclsparseSpGEMMPreprocess(
            handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, buffer);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
    }
    SpGemmLegacyLayout layout{};
    if (!BuildLegacyLayout(matA, matB, computeType, layout) || layout.bytes != legacyBytes) {
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    aclsparseSpGEMMDescr_t descr = matC->legacySpGemmDescr.get();
    size_t size2 = descr->requiredBuffer2;
    if (size2 > layout.computeCapacity) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    descr->forceGenericPath = false;
    status = aclsparseSpGEMMCompute(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
        descr, &size2, static_cast<uint8_t *>(buffer) + layout.computeOffset);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = aclsparseSpGEMMCopy(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, descr);
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        matC->activeBuffer = buffer;
    }
    return status;
}
