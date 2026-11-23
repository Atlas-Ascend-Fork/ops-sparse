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

#ifndef SPSV_TILING_DATA_H_
#define SPSV_TILING_DATA_H_

#include <cstdint>

// Maximum number of SIMT threads per block, shared by kernel and host code.
// Canonical definition; spsv.h re-exports into namespace spsv.
constexpr uint32_t kSimtMaxThreads = 2048u;

// Exact-retry determinant bound, shared by host allocation and device layout.
// 554 input-product exponent bits + 63 row-count bits, with spare headroom.
constexpr int64_t kSpsvRationalBitsPerRow = 620;
constexpr int64_t kSpsvRationalTargetBatch = 8;

#if defined(__CCE__)
__host__ __aicore__
#endif
    inline constexpr int64_t
    SpsvComplexSingletonSuffixBegin(int32_t firstWidth, int32_t suffix, int32_t prefix)
{
    return static_cast<int64_t>(firstWidth) + suffix - prefix;
}

// TilingData is ~160 bytes, passed by value as kernel parameter.
// This is within the kernel parameter size limit for 351x.
struct SpsvTilingData {
    int64_t m;
    int64_t nnz;

    float alpha;
    float alphaImag;
    int32_t valueType; // 0=ACL_FLOAT, 1=ACL_COMPLEX64
    int32_t fillMode;
    int32_t diagType;
    int32_t opA;
    int32_t format;
    int32_t indexType;  // 0=I32, 1=I64 for rowPtr (ptrType)
    int32_t colIndType; // 0=I32, 1=I64 for colInd (idxType); may differ from indexType
    int32_t idxBase;    // 0=0-based, 1=1-based (Fortran indexing)
    // Conjugate the values when copying them into the workspace CSR.  Set for
    // CSC with CONJUGATE_TRANSPOSE, which the host rewrites to a plain solve
    // on the conjugate of the stored matrix.
    int32_t conjValues;

    int64_t levelPtrOffset;
    int64_t levelRowOffset;
    int64_t diagPtrOffset;
    int64_t validCountOffset;
    // numLevels: unknown (0) or a completed host snapshot in solve params. The analysis
    // kernel writes the computed numLevels into workspace (as SpsvTilingData
    // at workspace offset 0). The solve kernel reads numLevels from workspace,
    // NOT from this tiling field. The host snapshot only prunes empty recovery launches.
    int32_t numLevels;
    int32_t primaryStatus; // overwritten by each eligible primary launch

    // permType: 0 = int32_t workspace arrays (perm, transPerm, diagPtr);
    //           1 = int64_t workspace arrays (used when nnz > INT32_MAX with I64 index).
    int32_t permType;

    int64_t csrRowPtrOffset;
    int64_t csrColIndOffset;
    int64_t csrValuesOffset;
    int64_t permOffset;

    int64_t transRowPtrOffset;
    int64_t transColIndOffset;
    int64_t transValuesOffset;
    int64_t transPermOffset;

    // Float-float (hi, lo) shadow of Y used by the complex64 solve:
    // 4 floats per row (real hi/lo, imag hi/lo).
    int64_t yWideOffset;

    int32_t numSlices;
    int32_t sliceWidth;

    uint32_t nthreads;
    uint32_t numBlocks;

    // Device pointer to alpha value (float). 0 means "invalid / not set";
    // when non-zero, the kernel reads alpha from this device address instead
    // of the 'alpha' field above. The caller must ensure the pointer is at
    // least 4-byte aligned (float alignment requirement).
    uint64_t alphaDevicePtr;
};


#if defined(__CCE__)
__host__ __aicore__
#endif
    inline bool SpsvUseRealScaledPrimary(const SpsvTilingData& t)
{
    return t.valueType == 0 && t.diagType == 0 && t.m >= 1024 && t.nnz / t.m >= 7 && t.nnz / t.m <= 16 &&
           t.numLevels == t.m && t.indexType == 0 && t.colIndType == 0 && t.permType == 0;
}

#if defined(__CCE__)
__host__ __aicore__
#endif
inline bool SpsvUseRealBinary64Primary(const SpsvTilingData& t)
{
    return t.valueType == 0 && t.diagType != 0 && t.m >= 1024 && t.m <= 8192 &&
           t.nnz / t.m >= 7 && t.nnz / t.m <= 16 && t.numLevels == t.m &&
           t.indexType == 0 && t.colIndType == 0 && t.permType == 0;
}

#if defined(__CCE__)
__host__ __aicore__
#endif
inline bool SpsvUseComplexBinary64Primary(const SpsvTilingData& t)
{
    // Amortize binary64 replay preparation on deep sparse dependency schedules.
    // Device admission independently validates every row and all operands.
    return t.valueType == 1 && t.m >= 1024 && t.nnz / t.m <= 16 &&
           (t.m >= 8192 || t.numLevels == 0 || t.numLevels == t.m || (t.diagType != 0 && t.numLevels >= t.m / 2)) &&
           t.indexType == 0 && t.colIndType == 0 && t.permType == 0;
}

#if defined(__CCE__)
__host__ __aicore__
#endif
inline bool SpsvUseEarlyComplexBinary64Primary(const SpsvTilingData& t)
{
    return SpsvUseComplexBinary64Primary(t) && t.diagType != 0 && t.m <= 12288 &&
           (t.numLevels >= t.m / 2 || (t.m >= 8192 && t.numLevels >= t.m / 4));
}

#endif // SPSV_TILING_DATA_H_
