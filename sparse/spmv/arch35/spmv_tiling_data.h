/*
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
#ifndef OPS_SPARSE_SPMV_ARCH35_TILING_DATA_H_
#define OPS_SPARSE_SPMV_ARCH35_TILING_DATA_H_

#include <cstddef>
#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

// Runtime type IDs are deliberately independent of aclDataType enum values.
enum SpmvArch35Type : int32_t {
    SPMV35_COMPUTE_I32 = 0,
    SPMV35_COMPUTE_F32 = 1,
    SPMV35_COMPUTE_C64 = 2,
    SPMV35_VALUE_I8 = 0,
    SPMV35_VALUE_F16 = 1,
    SPMV35_VALUE_BF16 = 2,
    SPMV35_VALUE_F32 = 3,
    SPMV35_VALUE_C64 = 4,
    SPMV35_OUTPUT_I32 = 0,
    SPMV35_OUTPUT_F16 = 1,
    SPMV35_OUTPUT_BF16 = 2,
    SPMV35_OUTPUT_F32 = 3,
    SPMV35_OUTPUT_C64 = 4,
};

// This structure is published to the first 192 bytes of the device workspace
// by a stream-ordered A5 initialization kernel. Keep it POD and use
// fixed-width fields so Host and device layouts agree.
struct SpmvArch35TilingData {
    uint32_t rows;
    uint32_t cols;
    uint32_t nnz;
    uint32_t indexBase;
    uint32_t rowPtrBase;
    uint32_t rowsPerBlock;
    uint32_t valueType;
    uint32_t outputType;
    uint32_t computeType;
    uint32_t opA;
    uint32_t alg;
    uint32_t useNormalizedRowPtr;
    uint32_t reserved;
    int32_t alphaInt;
    int32_t betaInt;
    float alphaReal;
    float betaReal;
    float alphaImag;
    float betaImag;
    uint64_t alphaPtr;
    uint64_t betaPtr;
    uint64_t accumulatorOffset;
    int64_t rowOffsetsStride;
    int64_t colIndStride;
    int64_t valuesStride;
    int64_t xStride;
    int64_t yStride;
};

// Inclusive row limits shared by host launch selection and device dispatch.
constexpr uint32_t kSpmvFloatSmallRowsMax = 64U;
constexpr uint32_t kSpmvTransposeGatherRowsMax = 4095U;

constexpr size_t kSpmvArch35WorkspaceAlignment = 64U;
constexpr uint64_t kSpmvArch35MaxI32Value = 0x7FFFFFFFULL;
constexpr size_t kSpmvArch35HeaderBytes = 192U;
constexpr size_t kSpmvArch35SortedFlagOffset = kSpmvArch35HeaderBytes - sizeof(int32_t);
static_assert(sizeof(SpmvArch35TilingData) <= kSpmvArch35HeaderBytes,
              "A5 SpMV tiling payload must fit in the workspace header");
static_assert(sizeof(SpmvArch35TilingData) <= kSpmvArch35SortedFlagOffset,
              "CSR sorted flag must not overlap the A5 SpMV tiling payload");

inline bool SpmvArch35AddSize(size_t a, size_t b, size_t *out)
{
    if (out == nullptr || b > static_cast<size_t>(-1) - a) {
        return false;
    }
    *out = a + b;
    return true;
}

inline bool SpmvArch35AlignSize(size_t value, size_t *out)
{
    constexpr size_t mask = kSpmvArch35WorkspaceAlignment - 1U;
    if (out == nullptr || value > static_cast<size_t>(-1) - mask) {
        return false;
    }
    *out = (value + mask) & ~mask;
    return true;
}

inline size_t SpmvArch35AccumulatorBytes(uint32_t computeType)
{
    return computeType == SPMV35_COMPUTE_I32 ? sizeof(int64_t)
         : computeType == SPMV35_COMPUTE_C64 ? sizeof(float) * 2U
                                             : sizeof(float);
}

inline bool SpmvArch35WorkspaceBytes(uint64_t rows, uint64_t cols,
                                     uint32_t computeType, bool transpose,
                                     size_t *bytes)
{
    if (bytes == nullptr || rows > kSpmvArch35MaxI32Value ||
        cols > kSpmvArch35MaxI32Value ||
        (computeType != SPMV35_COMPUTE_I32 &&
         computeType != SPMV35_COMPUTE_F32 &&
         computeType != SPMV35_COMPUTE_C64)) {
        return false;
    }
    size_t end = kSpmvArch35HeaderBytes;
    if (rows + 1U > static_cast<uint64_t>(static_cast<size_t>(-1) / sizeof(int32_t))) {
        return false;
    }
    const size_t rowBytes = static_cast<size_t>(rows + 1U) * sizeof(int32_t);
    if (!SpmvArch35AddSize(end, rowBytes, &end) || !SpmvArch35AlignSize(end, &end)) {
        return false;
    }
    if (transpose) {
        const size_t accBytes = SpmvArch35AccumulatorBytes(computeType);
        if (accBytes == 0U ||
            cols > static_cast<uint64_t>(static_cast<size_t>(-1) / accBytes) ||
            !SpmvArch35AddSize(end, static_cast<size_t>(cols) * accBytes, &end) ||
            !SpmvArch35AlignSize(end, &end)) {
            return false;
        }
    }
    *bytes = end;
    return true;
}

inline bool SpmvArch35AccumulatorOffset(uint64_t rows, size_t *offset)
{
    if (offset == nullptr || rows > kSpmvArch35MaxI32Value ||
        rows + 1U > static_cast<uint64_t>(static_cast<size_t>(-1) / sizeof(int32_t))) {
        return false;
    }
    size_t end = 0;
    const size_t rowBytes = static_cast<size_t>(rows + 1U) * sizeof(int32_t);
    if (!SpmvArch35AddSize(kSpmvArch35HeaderBytes, rowBytes, &end) ||
        !SpmvArch35AlignSize(end, offset)) {
        return false;
    }
    return true;
}

#endif  // OPS_SPARSE_SPMV_ARCH35_TILING_DATA_H_
