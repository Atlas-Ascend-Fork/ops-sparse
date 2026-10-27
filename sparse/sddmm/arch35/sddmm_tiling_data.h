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
 * \file sddmm_tiling_data.h
 * \brief Host/Kernel shared POD data for the Ascend 950 SDDMM implementation.
 */

#ifndef SDDMM_TILING_DATA_H_
#define SDDMM_TILING_DATA_H_

#include <cstddef>
#include <cstdint>

constexpr uint32_t kSddmmThreadsPerBlock = 256U;
constexpr uint32_t kSddmmWarpSize = 32U;
constexpr uint64_t kSddmmWorkspaceAlignment = 64ULL;
constexpr uint64_t kSddmmWorkspaceBytes = 64ULL;
constexpr uint64_t kSddmmPackedYWorkspaceLimit = 64ULL * 1024ULL * 1024ULL;
constexpr uint64_t kSddmmValidationMagic = 0x5344444D4D563031ULL; // "SDDMMV01"
constexpr int32_t kSddmmCsrCyclic64RowStride = 17;

enum SddmmFormatCode : int32_t {
    SDDMM_FORMAT_CSR = 0,
    SDDMM_FORMAT_BSR = 1,
};

enum SddmmValueTypeCode : int32_t {
    SDDMM_VALUE_FP32 = 0,
    SDDMM_VALUE_FP16 = 1,
    SDDMM_VALUE_BF16 = 2,
    SDDMM_VALUE_COMPLEX64 = 3,
    SDDMM_VALUE_INVALID = -1,
};

enum SddmmValidationStatus : int32_t {
    SDDMM_PATTERN_VALID = 0,
    SDDMM_PATTERN_BAD_ENDPOINT = 1,
    SDDMM_PATTERN_NOT_MONOTONIC = 2,
    SDDMM_PATTERN_BAD_COLUMN = 3,
};

enum SddmmPatternFlags : uint32_t {
    SDDMM_PATTERN_FLAG_NONE = 0U,
    // Every CSR row contains exactly 64 strictly sorted columns whose set is
    // one consecutive run modulo n. Only the device pattern validator may
    // establish this flag; Execute must not infer it from row length alone.
    SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64 = 1U << 0,
    // In addition to CYCLIC_64, row r starts at
    // (logicalFirstColumn + 17 * r) modulo n.  Cube dispatch requires this
    // stronger device-validated property.
    SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17 = 1U << 1,
};

// Fixed-size device result in the workspace prefix or BufferSize's temporary
// query allocation. Only this object is copied back after device validation.
struct SddmmValidationResult {
    uint64_t magic;
    int32_t status;
    uint32_t patternFlags;
    int32_t logicalFirstColumn;
    int32_t reserved;
};

static_assert(sizeof(SddmmValidationResult) <= kSddmmWorkspaceBytes,
    "SDDMM validation metadata must fit in the fixed workspace prefix");

// Every core owns one cache line in a temporary allocation. The extra batch
// index preserves the serial validator's first-error ordering across batches;
// it is never written to the caller's fixed workspace prefix.
struct alignas(kSddmmWorkspaceAlignment) SddmmQueryPartial {
    SddmmValidationResult validation;
    int64_t firstInvalidBatch;
};

static_assert(sizeof(SddmmValidationResult) == 24,
    "SDDMM validation result must keep its Host/Kernel ABI size");
static_assert(sizeof(SddmmQueryPartial) == kSddmmWorkspaceBytes,
    "SDDMM partial results must occupy separate cache lines");
static_assert(alignof(SddmmQueryPartial) == kSddmmWorkspaceAlignment,
    "SDDMM partial results must be cache-line aligned");
static_assert(offsetof(SddmmQueryPartial, firstInvalidBatch) == 24,
    "SDDMM partial batch index must follow the validation result");

// Dynamic launch data is passed by value.  Workspace contains only stable
// preprocess output and never alpha/beta or a GM copy of this structure.
struct SddmmTilingData {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t rowsX;
    int64_t colsX;
    int64_t rowsY;
    int64_t colsY;
    int64_t ldX;
    int64_t ldY;
    int64_t blockRows;
    int64_t blockCols;
    int64_t nnzUnits;       // CSR: nnz; BSR: blockNnz.
    int64_t outputsPerBatch;
    int64_t batchCount;
    int64_t batchCountX;
    int64_t batchCountY;
    int64_t xBatchStride;
    int64_t yBatchStride;
    int64_t offsetsBatchStride;
    int64_t columnsBatchStride;
    int64_t valuesBatchStride;
    int64_t packedYBatchStride;
    int64_t packedYOffset;
    uint64_t alphaPtr;
    uint64_t betaPtr;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    int32_t format;
    int32_t indexBase;
    int32_t blockDim;
    int32_t blockOrder;
    int32_t opX;
    int32_t opY;
    int32_t orderX;
    int32_t orderY;
    int32_t inputType;
    int32_t outputType;
    int32_t computeType;
    int32_t pointerMode;
    int32_t numBlocks;
    int32_t usePackedY;
    uint32_t patternFlags;
    int32_t patternLogicalFirstColumn;
};

static_assert(sizeof(SddmmTilingData) == 280,
    "SDDMM tiling launch data must keep its Host/Kernel ABI size");
static_assert(alignof(SddmmTilingData) == 8,
    "SDDMM tiling launch data must keep its Host/Kernel ABI alignment");
static_assert(offsetof(SddmmTilingData, patternFlags) == 272,
    "SDDMM pattern flags must keep their Host/Kernel ABI offset");
static_assert(offsetof(SddmmTilingData, patternLogicalFirstColumn) == 276,
    "SDDMM logical first column must keep its Host/Kernel ABI offset");

#endif // SDDMM_TILING_DATA_H_
