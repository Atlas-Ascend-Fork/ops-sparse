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

#include "cube_spmm_preprocess.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include "acl/acl.h"
#include "aclsparse_fp16_utils.h"
#include "aclsparse_descr_internal.h"

namespace {

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

// 校验 M / K 是否在 cube_spmm 内核支持的范围内。
// 这里只预处理稀疏矩阵 A，不涉及 N，因此只校验 M 和 K。
inline bool AreDimensionsSupported(int64_t M, int64_t K)
{
    constexpr int64_t kMaxInt32 = static_cast<int64_t>(INT32_MAX);
    constexpr int64_t kMaxInt64 = std::numeric_limits<int64_t>::max();
    if (M <= 0 || M > kMaxInt32 || K <= 0 || K > kMaxInt32) {
        return false;
    }
    if (K > 0 && M > kMaxInt64 / K) {
        return false;
    }
    return true;
}

using aclsparse::Float16BitsToFloat32;
using aclsparse::Float32ToFloat16Bits;

struct CooElem {
    int32_t row;
    int32_t col;
    uint16_t val;
};

bool AllocDeviceAndCopy(void **out, const void *hostData, size_t bytes)
{
    if (out == nullptr) {
        return false;
    }
    const size_t allocationBytes = std::max<size_t>(bytes, 32);
    aclError ret = aclrtMalloc(out, allocationBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        *out = nullptr;
        return false;
    }
    if (hostData != nullptr && bytes > 0) {
        ret = aclrtMemcpy(*out, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            aclrtFree(*out);
            *out = nullptr;
            return false;
        }
    }
    return true;
}

void FreeAll(void *rw, void *col, void *vals, void *core)
{
    if (rw != nullptr) { aclrtFree(rw); }
    if (col != nullptr) { aclrtFree(col); }
    if (vals != nullptr) { aclrtFree(vals); }
    if (core != nullptr) { aclrtFree(core); }
}

// RAII holder for device buffers allocated during preprocess. If preprocess
// fails part-way through, the destructor frees any buffers that were not
// successfully committed to the descriptor.
struct DeviceBuffersGuard {
    void *rwPtr = nullptr;
    void *colRef = nullptr;
    void *vals = nullptr;
    void *coreInfo = nullptr;

    ~DeviceBuffersGuard()
    {
        FreeAll(rwPtr, colRef, vals, coreInfo);
    }

    void Release()
    {
        rwPtr = nullptr;
        colRef = nullptr;
        vals = nullptr;
        coreInfo = nullptr;
    }

    void CommitTo(aclsparseCubeSpmmMatDescr_t matA, bool needsOutputPreclear)
    {
        FreeAll(matA->rwPtr, matA->colRef, matA->vals, matA->coreInfo);
        matA->rwPtr = rwPtr;
        matA->colRef = colRef;
        matA->vals = vals;
        matA->coreInfo = coreInfo;
        matA->needsSafeOutputInit = needsOutputPreclear;
        Release();
    }
};

}  // namespace

static aclsparseStatus_t CopyCooFromDeviceToHost(
    int64_t nnz,
    const void *cooRowsDev, const void *cooColsDev, const void *cooValsDev,
    std::vector<int32_t> &cooRows, std::vector<int32_t> &cooCols, std::vector<uint16_t> &cooVals)
{
    if (nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclError ret = aclrtMemcpy(cooRows.data(), cooRows.size() * sizeof(int32_t),
                               cooRowsDev, cooRows.size() * sizeof(int32_t),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    ret = aclrtMemcpy(cooCols.data(), cooCols.size() * sizeof(int32_t),
                      cooColsDev, cooCols.size() * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    ret = aclrtMemcpy(cooVals.data(), cooVals.size() * sizeof(uint16_t),
                      cooValsDev, cooVals.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static void BuildCsrFromCoo(
    int64_t M, int64_t K,
    const std::vector<int32_t> &cooRows,
    const std::vector<int32_t> &cooCols,
    const std::vector<uint16_t> &cooVals,
    std::vector<int64_t> &csrRowPtr,
    std::vector<int32_t> &csrColIdx,
    std::vector<uint16_t> &csrVals)
{
    std::vector<CooElem> elems;
    elems.reserve(cooRows.size());
    for (size_t i = 0; i < cooRows.size(); ++i) {
        int32_t r = cooRows[i];
        int32_t c = cooCols[i];
        if (r < 0 || r >= M || c < 0 || c >= K) {
            continue;
        }
        elems.push_back({r, c, cooVals[i]});
    }

    std::sort(elems.begin(), elems.end(), [](const CooElem &a, const CooElem &b) {
        if (a.row != b.row) { return a.row < b.row; }
        if (a.col != b.col) { return a.col < b.col; }
        return false;
    });

    csrRowPtr.assign(static_cast<size_t>(M) + 1, 0);
    csrColIdx.clear();
    csrVals.clear();
    csrColIdx.reserve(elems.size());
    csrVals.reserve(elems.size());

    for (size_t i = 0; i < elems.size();) {
        int32_t r = elems[i].row;
        int32_t c = elems[i].col;
        float sum = Float16BitsToFloat32(elems[i].val);
        size_t j = i + 1;
        while (j < elems.size() && elems[j].row == r && elems[j].col == c) {
            sum += Float16BitsToFloat32(elems[j].val);
            ++j;
        }
        csrColIdx.push_back(c);
        csrVals.push_back(Float32ToFloat16Bits(sum));
        csrRowPtr[static_cast<size_t>(r) + 1] += 1;
        i = j;
    }
    for (size_t i = 1; i <= static_cast<size_t>(M); ++i) {
        csrRowPtr[i] += csrRowPtr[i - 1];
    }
}

static aclsparseStatus_t BuildBcsrFromCsr(
    int64_t M, int64_t safeBlockM, int64_t safeBlockK,
    const std::vector<int64_t> &csrRowPtr,
    const std::vector<int32_t> &csrColIdx,
    const std::vector<uint16_t> &csrVals,
    std::vector<int64_t> &rwPartition,
    std::vector<int32_t> &tcColRef,
    std::vector<uint16_t> &blockVals)
{
    if (safeBlockM <= 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (safeBlockK <= 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int64_t blockRows = CeilDiv(M, safeBlockM);
    int64_t blockSize = safeBlockM * safeBlockK;

    rwPartition.assign(static_cast<size_t>(blockRows) + 1, 0);
    tcColRef.clear();
    std::map<std::pair<int64_t, int32_t>, int32_t> uniqueCol;

    for (int64_t rw = 0; rw < blockRows; ++rw) {
        std::set<int32_t> colsInRw;
        int64_t rowBegin = rw * safeBlockM;
        int64_t rowEnd = std::min(rowBegin + safeBlockM, M);
        for (int64_t r = rowBegin; r < rowEnd; ++r) {
            for (int64_t idx = csrRowPtr[r]; idx < csrRowPtr[r + 1]; ++idx) {
                colsInRw.insert(csrColIdx[idx]);
            }
        }
        if (!colsInRw.empty()) {
            int32_t localIdx = 0;
            for (int32_t origCol : colsInRw) {
                uniqueCol[{rw, origCol}] = localIdx;
                tcColRef.push_back(origCol);
                ++localIdx;
            }
            int64_t padded = CeilDiv(localIdx, safeBlockK) * safeBlockK;
            for (int64_t k = localIdx; k < padded; ++k) {
                tcColRef.push_back(tcColRef.back());
            }
            rwPartition[rw + 1] = rwPartition[rw] + padded / safeBlockK;
        } else {
            rwPartition[rw + 1] = rwPartition[rw];
        }
    }

    int64_t numBlocks = rwPartition[blockRows];
    int64_t colLen = numBlocks * safeBlockK;
    int64_t valLen = numBlocks * blockSize;

    blockVals.assign(static_cast<size_t>(valLen), 0);
    for (int64_t r = 0; r < M; ++r) {
        int64_t rw = r / safeBlockM;
        for (int64_t idx = csrRowPtr[r]; idx < csrRowPtr[r + 1]; ++idx) {
            int32_t origCol = csrColIdx[idx];
            auto it = uniqueCol.find({rw, origCol});
            if (it == uniqueCol.end()) {
                return ACL_SPARSE_STATUS_INTERNAL_ERROR;
            }
            int32_t localCol = it->second;
            int64_t tcId = rwPartition[rw] + localCol / safeBlockK;
            int64_t offset = tcId * blockSize + (r % safeBlockM) * safeBlockK + (localCol % safeBlockK);
            float v = Float16BitsToFloat32(blockVals[offset]) + Float16BitsToFloat32(csrVals[idx]);
            blockVals[offset] = Float32ToFloat16Bits(v);
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static void BuildCoreInfoPartition(
    int64_t blockRows, int64_t numBlocks, int32_t safeNumCores,
    const std::vector<int64_t> &rwPartition,
    std::vector<int32_t> &coreInfo)
{
    if (safeNumCores <= 0) {
        coreInfo.clear();
        return;
    }
    coreInfo.assign(static_cast<size_t>(safeNumCores) * 4, 0);
    if (numBlocks <= 0) {
        for (int32_t c = 0; c < safeNumCores; ++c) {
            coreInfo[c * 4 + 0] = blockRows;
            coreInfo[c * 4 + 1] = blockRows;
            coreInfo[c * 4 + 2] = 0;
            coreInfo[c * 4 + 3] = 0;
        }
        return;
    }

    int64_t base = numBlocks / safeNumCores;
    int64_t rem = numBlocks % safeNumCores;
    int64_t rw = 0;
    int64_t blkOffset = 0;
    for (int32_t c = 0; c < safeNumCores; ++c) {
        int64_t quota = base + (c < rem ? 1 : 0);
        if (rw >= blockRows || quota == 0) {
            coreInfo[c * 4 + 0] = blockRows;
            coreInfo[c * 4 + 1] = blockRows;
            coreInfo[c * 4 + 2] = 0;
            coreInfo[c * 4 + 3] = 0;
            continue;
        }
        int64_t rwStart = rw;
        int64_t blkStart = blkOffset;
        int64_t remaining = quota;
        while (remaining > 0 && rw < blockRows) {
            int64_t wBlocks = rwPartition[rw + 1] - rwPartition[rw];
            int64_t avail = wBlocks - blkOffset;
            if (avail <= remaining) {
                remaining -= avail;
                ++rw;
                blkOffset = 0;
            } else {
                blkOffset += remaining;
                remaining = 0;
            }
        }
        int64_t rwEnd;
        int64_t blkEnd;
        if (rw > rwStart) {
            blkEnd = (blkOffset == 0) ? (rwPartition[rw] - rwPartition[rw - 1])
                                      : blkOffset;
            rwEnd = (blkOffset == 0) ? rw : rw + 1;
        } else {
            rwEnd = rw + 1;
            blkEnd = blkOffset;
        }
        coreInfo[c * 4 + 0] = rwStart;
        coreInfo[c * 4 + 1] = rwEnd;
        coreInfo[c * 4 + 2] = blkStart;
        coreInfo[c * 4 + 3] = blkEnd;
    }
}

static aclsparseStatus_t AllocateAndCopyBcsrOutputs(
    const std::vector<int64_t> &rwPartition,
    const std::vector<int32_t> &tcColRef,
    const std::vector<uint16_t> &blockVals,
    const std::vector<int32_t> &coreInfo,
    void **rwPtrOut, void **colRefOut, void **valsOut, void **coreInfoOut,
    int64_t *rwPtrElems, int64_t *colRefElems, int64_t *valsElems, int64_t *coreInfoElems)
{
    *rwPtrElems = static_cast<int64_t>(rwPartition.size());
    *colRefElems = static_cast<int64_t>(tcColRef.size());
    *valsElems = static_cast<int64_t>(blockVals.size());
    *coreInfoElems = static_cast<int64_t>(coreInfo.size());

    *rwPtrOut = nullptr;
    *colRefOut = nullptr;
    *valsOut = nullptr;
    *coreInfoOut = nullptr;

    if (!AllocDeviceAndCopy(rwPtrOut, rwPartition.data(),
                            static_cast<size_t>(*rwPtrElems) * sizeof(int64_t))) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    if (!AllocDeviceAndCopy(colRefOut, tcColRef.data(),
                            static_cast<size_t>(*colRefElems) * sizeof(int32_t))) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    if (!AllocDeviceAndCopy(valsOut, blockVals.data(),
                            static_cast<size_t>(*valsElems) * sizeof(uint16_t))) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    if (!AllocDeviceAndCopy(coreInfoOut, coreInfo.data(),
                            static_cast<size_t>(*coreInfoElems) * sizeof(int32_t))) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static bool NeedsOutputPreclear(
    const std::vector<int64_t> &rwPartition, const std::vector<int32_t> &coreInfo)
{
    // The original partitioner covers leading/internal empty windows, but not
    // an empty tail after the last nonempty window (including an entirely empty A).
    if (rwPartition.back() == rwPartition[rwPartition.size() - 2]) {
        return true;
    }
    for (size_t core = 0; core < coreInfo.size(); core += 4) {
        const int32_t begin = coreInfo[core];
        const int32_t end = coreInfo[core + 1];
        if (begin >= end) {
            continue;
        }
        const int64_t lastBlocks = rwPartition[end] - rwPartition[end - 1];
        if (coreInfo[core + 2] > 0 || coreInfo[core + 3] < lastBlocks) {
            return true;
        }
    }
    return false;
}

static aclsparseStatus_t ValidateBcsrInputs(
    int64_t M, int64_t K, int64_t nnz,
    const void *cooRowsDev, const void *cooColsDev, const void *cooValsDev,
    int64_t blockM, int64_t blockK, int32_t numCores)
{
    if (M <= 0 || K <= 0 || nnz < 0 || numCores <= 0 ||
        blockM <= 0 || blockK <= 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // M/K are stored as int64_t in the kernel and used as int64_t indices. Keep
    // M * K <= INT64_MAX to avoid overflow in BCSR block-index products.
    if (!AreDimensionsSupported(M, K)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (nnz > M * K) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (cooRowsDev == nullptr || cooColsDev == nullptr || cooValsDev == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t CubeSpmmBuildBcsrOnDevice(
    int64_t M, int64_t K, int64_t nnz,
    const void *cooRowsDev, const void *cooColsDev, const void *cooValsDev,
    int64_t blockM, int64_t blockK, int32_t numCores,
    void **rwPtrOut, void **colRefOut, void **valsOut, void **coreInfoOut,
    int64_t *rwPtrElems, int64_t *colRefElems, int64_t *valsElems, int64_t *coreInfoElems,
    bool *needsOutputPreclearOut)
{
    aclsparseStatus_t st = ValidateBcsrInputs(
        M, K, nnz, cooRowsDev, cooColsDev, cooValsDev, blockM, blockK, numCores);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (rwPtrOut == nullptr || colRefOut == nullptr || valsOut == nullptr || coreInfoOut == nullptr ||
        rwPtrElems == nullptr || colRefElems == nullptr || valsElems == nullptr || coreInfoElems == nullptr ||
        needsOutputPreclearOut == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // Re-assign to local const variables to help static analyzers see that
    // these divisors have already been validated as positive.
    const int64_t safeBlockM = blockM;
    const int64_t safeBlockK = blockK;
    const int32_t safeNumCores = numCores;

    std::vector<int32_t> cooRows(static_cast<size_t>(nnz));
    std::vector<int32_t> cooCols(static_cast<size_t>(nnz));
    std::vector<uint16_t> cooVals(static_cast<size_t>(nnz));
    st = CopyCooFromDeviceToHost(
        nnz, cooRowsDev, cooColsDev, cooValsDev, cooRows, cooCols, cooVals);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    std::vector<int64_t> csrRowPtr;
    std::vector<int32_t> csrColIdx;
    std::vector<uint16_t> csrVals;
    BuildCsrFromCoo(M, K, cooRows, cooCols, cooVals, csrRowPtr, csrColIdx, csrVals);

    std::vector<int64_t> rwPartition;
    std::vector<int32_t> tcColRef;
    std::vector<uint16_t> blockVals;
    st = BuildBcsrFromCsr(
        M, safeBlockM, safeBlockK,
        csrRowPtr, csrColIdx, csrVals,
        rwPartition, tcColRef, blockVals);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    int64_t blockRows = CeilDiv(M, safeBlockM);
    int64_t numBlocks = rwPartition[blockRows];
    std::vector<int32_t> coreInfo;
    BuildCoreInfoPartition(
        blockRows, numBlocks, safeNumCores, rwPartition, coreInfo);
    *needsOutputPreclearOut = NeedsOutputPreclear(rwPartition, coreInfo);

    return AllocateAndCopyBcsrOutputs(
        rwPartition, tcColRef, blockVals, coreInfo,
        rwPtrOut, colRefOut, valsOut, coreInfoOut,
        rwPtrElems, colRefElems, valsElems, coreInfoElems);
}

// ============================================================================
// Public Cube SpMM preprocess entry point.
// ============================================================================

aclsparseStatus_t aclsparseCubeSpmmPreprocess(
    aclsparseHandle_t handle,
    const void *alpha, aclsparseCubeSpmmMatDescr_t matA,
    const int32_t *cooRows, const int32_t *cooCols, const uint16_t *cooVals,
    aclDataType computeType)
{
    if (cooRows == nullptr || cooCols == nullptr || cooVals == nullptr) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (matA == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matA->rows <= 0 || matA->cols <= 0 || matA->nnz < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != ACL_FLOAT) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matA->numCores <= 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // Build Cube-BCSR on device into local temporaries. Only after the whole
    // preprocess succeeds do we free the descriptor's old buffers and commit the
    // new ones, avoiding leaks on partial failure or repeated preprocess calls.
    int64_t rwPtrElems = 0;
    int64_t colRefElems = 0;
    int64_t valsElems = 0;
    int64_t coreInfoElems = 0;
    DeviceBuffersGuard newBuffers;
    bool needsOutputPreclear = false;
    aclsparseStatus_t st = CubeSpmmBuildBcsrOnDevice(
        matA->rows, matA->cols, matA->nnz,
        cooRows, cooCols, cooVals,
        matA->blockM, matA->blockK, matA->numCores,
        &newBuffers.rwPtr, &newBuffers.colRef, &newBuffers.vals, &newBuffers.coreInfo,
        &rwPtrElems, &colRefElems, &valsElems, &coreInfoElems, &needsOutputPreclear);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    newBuffers.CommitTo(matA, needsOutputPreclear);

    return ACL_SPARSE_STATUS_SUCCESS;
}
