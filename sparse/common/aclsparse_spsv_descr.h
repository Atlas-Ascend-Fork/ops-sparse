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

#ifndef ACLSPARSE_SPSV_DESCR_H_
#define ACLSPARSE_SPSV_DESCR_H_

#include <cstdint>
#include "acl/acl.h"

struct aclsparseSpSVDescr {
    // True once the analysis kernel has been launched (submitted to stream).
    // The kernel may still be executing asynchronously; correctness relies on
    // same-stream ordering (solve kernel waits for analysis to finish).
    bool analysisLaunched = false;
    // True once aclsparseSpSV_updateMatrix has been called. After updateMatrix,
    // the user's mat->values pointer may legitimately differ from the pointer
    // used during analysis, so the pointer-consistency check in solve is relaxed.
    bool updateMatrixCalled = false;
    void *workspaceBuffer = nullptr;

    // A same-stream asynchronous snapshot of the structural analysis result.
    // A pending/stale snapshot must never suppress dependency recovery.
    int32_t cachedNumLevels = 0;
    int32_t *analysisNumLevels = nullptr;
    aclrtEvent analysisEvent = nullptr;
    aclrtStream analysisCopyStream = nullptr;
    bool analysisCopyPending = false;
    bool analysisEventRecorded = false;
    bool analysisSnapshotCurrent = false;

    int64_t cachedM = 0;
    int64_t cachedNnz = 0;
    int32_t cachedFormat = 0;
    int32_t cachedDiagType = 0;
    int32_t cachedIdxBase = 0;        // 0=0-based, 1=1-based (Fortran indexing)
    int32_t cachedOpA = -1;           // operation type (0=NON_TRANSPOSE, 1=TRANSPOSE, 2=CONJUGATE_TRANSPOSE)
    int32_t cachedFillMode = 0;       // LOWER(0) or UPPER(1)
    int32_t cachedNumSlices = 0;      // for SELL format
    int32_t cachedSliceWidth = 0;     // for SELL format (stores mat->sliceNnz)
    size_t cachedIdxSize = 4;         // rowPtr element size: 4 bytes (I32) or 8 bytes (I64)
    size_t cachedColIndSize = 4;     // colInd element size: 4 bytes (I32) or 8 bytes (I64)
    int32_t cachedPermType = 0;      // 0=int32 workspace arrays, 1=int64 (nnz > INT32_MAX)
    int32_t cachedIndexType = 0;      // 0=I32, 1=I64 for rowPtr (ptrType)
    int32_t cachedColIndType = 0;    // 0=I32, 1=I64 for colInd (idxType)
    int32_t cachedValueType = 0;     // 0=ACL_FLOAT, 1=ACL_COMPLEX64

    // Weak reference to the user's matrix values pointer at analysis time.
    // The user must keep this memory valid until solve completes. If the user
    // frees or reallocates matrix values after analysis, this becomes dangling.
    void *currentValues = nullptr;

    int64_t diagPtrOffset = 0;
    int64_t csrValuesOffset = -1;
    int64_t permOffset = -1;
    int64_t transValuesOffset = -1;
    int64_t transPermOffset = -1;
};

#endif // ACLSPARSE_SPSV_DESCR_H_
