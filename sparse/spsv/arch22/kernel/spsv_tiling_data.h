/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_A2_TILING_DATA_H_
#define SPSV_A2_TILING_DATA_H_
#include <cstdint>

namespace spsv_a2_ub {
// InitBuffer rounds each allocation up to a 32-byte UB block.
constexpr uint32_t BufferBytes(uint32_t elements, uint32_t elementBytes)
{
    constexpr uint32_t alignment = 32;
    return (elements * elementBytes + alignment - 1) / alignment * alignment;
}

// Each scalar stores a count and independently scaled binary64 components.
constexpr uint32_t kWideCompactTerms = 20;
constexpr uint32_t kWideRegularTerms = 4;
constexpr uint32_t SolutionWords(uint32_t terms) { return 3 * terms + 1; }
constexpr uint32_t kSolutionWords = SolutionWords(kWideCompactTerms);
constexpr uint32_t kUnitPrecisionLogWords = 9;
constexpr uint32_t kUnitPrecisionWords = 1u << kUnitPrecisionLogWords;
constexpr uint32_t kUnitDigitBits = 12;
constexpr uint32_t kUnitMaximumDigits = (kUnitPrecisionWords * 32 + kUnitDigitBits - 1) / kUnitDigitBits;
constexpr uint32_t kUnitPrefixDigits = 32;
constexpr uint32_t kUnitSpanBits = kUnitPrecisionLogWords + 2;
constexpr uint32_t kUnitVectorCapacity = BufferBytes(kUnitMaximumDigits + 7, sizeof(int32_t)) / sizeof(int32_t);
constexpr uint32_t kUnitShortCapacity = BufferBytes(kUnitPrefixDigits + kUnitMaximumDigits + 2, sizeof(int16_t)) /
                                        sizeof(int16_t);
constexpr uint32_t kUnitCarryBytes = BufferBytes(72, sizeof(int32_t));
constexpr uint32_t kUnitMathBytes = 5 * kUnitVectorCapacity * sizeof(int32_t) + kUnitShortCapacity * sizeof(int16_t) +
                                    kUnitCarryBytes;
constexpr uint32_t kUnitPrecisionStride = (kUnitPrefixDigits + kUnitMaximumDigits + 1) / 2 + 1;
static_assert(kUnitMaximumDigits < (1u << kUnitSpanBits), "Digit span must fit metadata");
constexpr uint32_t kRowCapacity = 2048;
constexpr uint32_t kColumnCapacity = 4096;
constexpr uint32_t kValueCapacity = 8192;
constexpr uint32_t kInputCapacity = 2048;
constexpr uint32_t kAlphaCapacity = 8;
// Capacities count scalar FP32 components, not complex elements or rows.
constexpr uint32_t kResidentCapacity = 24576;
constexpr uint32_t kCompactCapacity = 512;
constexpr uint32_t kLowBytes = BufferBytes(kCompactCapacity, sizeof(float));
// Two ping-pong iterates, each retaining high and low components.
constexpr uint32_t kIterateBytes = BufferBytes(2 * 2 * kCompactCapacity, sizeof(float));
constexpr uint32_t kCommonBytes = BufferBytes(kRowCapacity, sizeof(int32_t)) +
    BufferBytes(kColumnCapacity, sizeof(int32_t)) + BufferBytes(kValueCapacity, sizeof(float)) +
    BufferBytes(kInputCapacity, sizeof(float)) + BufferBytes(kAlphaCapacity, sizeof(float));
constexpr uint32_t kSolveBytes = kCommonBytes + BufferBytes(kResidentCapacity, sizeof(float));
constexpr uint32_t kAccurateBytes = kCommonBytes + BufferBytes(kCompactCapacity, sizeof(float)) + kLowBytes;
constexpr uint32_t kIterativeBytes = kCommonBytes + BufferBytes(kCompactCapacity, sizeof(float)) + kIterateBytes;
constexpr uint32_t kRequiredBytes = kSolveBytes + kUnitMathBytes;
static_assert(kRequiredBytes <= 192 * 1024, "Solve and integer vector scratch must fit A2/A3 UB");

static_assert(kCompactCapacity <= kResidentCapacity, "Compact solution must fit the resident cache");
// The failed iterative path and its direct fallback share a TPipe. Its UB
// allocations remain live until that pipe is destroyed.
static_assert(kIterativeBytes + kAccurateBytes <= kRequiredBytes, "Compact fallback exceeds the UB budget");
} // namespace spsv_a2_ub

struct SpsvA2Tiling {
    int64_t m = 0;
    int64_t nnz = 0;
    int64_t sourceNnz = 0;
    uint64_t rowOffset = 0;
    uint64_t colOffset = 0;
    uint64_t valueOffset = 0;
    uint64_t permOffset = 0;
    uint64_t scratchOffset = 0;
    uint64_t solutionOffset = 0;
    uint64_t inputOffset = 0;
    uint64_t alphaDevice = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    int32_t components = 1;
    int32_t format = 0;
    int32_t base = 0;
    int32_t transpose = 0;
    int32_t conjugate = 0;
    int32_t lower = 1;
    int32_t unit = 0;
    int32_t slices = 0;
    int32_t sliceWidth = 0;
    int32_t diagonalMatrix = 0;
    uint32_t blocks = 1;
    uint64_t precisionOffset = 0;
    int32_t wideTerms = spsv_a2_ub::kWideCompactTerms;
};
#endif
