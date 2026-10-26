/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#include <stdint.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"
#include "sddmm.h"
#include "sddmm_kernel.h"

namespace {

// Outer launch blockDim = AIV core count from CANN platform config.
constexpr uint32_t kSddmmBlockDimFallback = 24u;

uint32_t GetSddmmBlockDim()
{
    static uint32_t cached = 0u;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&]() {
#ifndef __CCE_AICORE__
        auto* plat = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (plat != nullptr) {
            const uint32_t aiv = plat->GetCoreNumAiv();
            if (aiv > 0u) {
                cached = aiv;
                return;
            }
        }
#endif
        cached = kSddmmBlockDimFallback;
    });
    return cached;
}

// S3 pattern cache: records the CSR pattern signature (rows, nnz) associated
// with each workspace buffer when tiling was last fully built. Keyed by the
// same buffer pointer used by aclsparseSpMatDescr::activeBuffer. Kept sddmm-
// local so the common descriptor header stays unmodified. Single-stream
// serialization is assumed (equivalent to the prior descriptor-field scheme);
// no locking. A reused buffer pointer simply overwrites its entry, so growth
// is bounded by the number of distinct buffers ever seen.
//
// 注意：非线程安全，调用者须保证单 stream 串行调用。多 stream 并发访问
// GetSddmmPatternCache() 返回的映射表需要外部加锁。
struct SddmmPatternSig {
    uint64_t rows;
    uint64_t nnz;
    uint64_t k;
};

static std::unordered_map<const void*, SddmmPatternSig>& GetSddmmPatternCache()
{
    static std::unordered_map<const void*, SddmmPatternSig> cache;
    return cache;
}

// Returns true only when buffer is cached AND its recorded pattern matches the
// current rows/nnz. A cache miss is treated as a pattern change so the caller
// performs a full rebuild instead of refreshing stale tiling data.
static bool SddmmPatternMatch(const void* buffer, uint64_t rows, uint64_t nnz, uint64_t k)
{
    auto& cache = GetSddmmPatternCache();
    auto it = cache.find(buffer);
    if (it == cache.end()) {
        return false;
    }
    return it->second.rows == rows && it->second.nnz == nnz && it->second.k == k;
}

static void SddmmPatternRecord(const void* buffer, uint64_t rows, uint64_t nnz, uint64_t k)
{
    GetSddmmPatternCache()[buffer] = SddmmPatternSig{rows, nnz, k};
}

float ScalarReadF32(const void* p)
{
    if (p == nullptr) {
        return 0.0f;
    }
    return *static_cast<const float*>(p);
}

// Convert IEEE-754 half-precision bit pattern (uint16_t) to float.
// Used to read alpha/beta when computeType=ACL_FLOAT16 per ACL generic API
// convention: alpha/beta pointer type is determined by computeType.
//
// 注意：test/sddmm/sddmm_golden.h 中有同功能实现 Fp16BitsToFp32，
// 两者算法一致，host.cpp 侧独立保留以便 sparse 库不依赖测试头文件。
static float SddmmFp16BitsToFp32(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0u) {
        if (mant == 0u) {
            f = sign;
        } else {
            uint32_t shift = static_cast<uint32_t>(__builtin_clz(mant)) - 21u;
            mant <<= shift;
            exp = 1u;
            f = sign | ((exp + 127u - 15u - shift) << 23) | ((mant << 13) & 0x7FFFFFu);
        }
    } else if (exp == 31u) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float v;
    __builtin_memcpy(&v, &f, sizeof(float));
    return v;
}

// Read alpha/beta scalar as float according to ACL computeType convention:
//   ACL_FLOAT   -> pointer points to float (4 bytes)
//   ACL_FLOAT16 -> pointer points to half (2 bytes), convert to float
static float SddmmReadScalarToF32(const void* p, aclDataType computeType)
{
    if (p == nullptr) {
        return 0.0f;
    }
    if (computeType == ACL_FLOAT16) {
        uint16_t bits;
        __builtin_memcpy(&bits, p, sizeof(uint16_t));
        return SddmmFp16BitsToFp32(bits);
    }
    return ScalarReadF32(p);
}

struct SddmmSupportedDtypeCombo {
    aclDataType matX;
    aclDataType matY;
    aclDataType matC;
    aclDataType computeType;
};

static bool IsSupportedSddmmDtypeCombo(const aclsparseDnMatDescr* matX, const aclsparseDnMatDescr* matY,
                                       const aclsparseSpMatDescr* matC, aclDataType computeType)
{
    static const SddmmSupportedDtypeCombo kCombos[] = {
        {ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT},
        {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16},
    };
    for (const auto& combo : kCombos) {
        if (matX->valueType == combo.matX && matY->valueType == combo.matY && matC->valueType == combo.matC &&
            computeType == combo.computeType) {
            return true;
        }
    }
    return false;
}

static bool SddmmDimensionsMatch(const aclsparseDnMatDescr* matX, const aclsparseDnMatDescr* matY,
                                 const aclsparseSpMatDescr* matC, aclsparseOperation_t opX, aclsparseOperation_t opY)
{
    const int64_t m = static_cast<int64_t>(matC->rows);
    const int64_t n = static_cast<int64_t>(matC->cols);
    int64_t k;
    if (opX == ACL_SPARSE_OP_NON_TRANSPOSE) {
        if (matX->rows != m) {
            return false;
        }
        k = matX->cols;
    } else {
        if (matX->cols != m) {
            return false;
        }
        k = matX->rows;
    }
    if (opY == ACL_SPARSE_OP_NON_TRANSPOSE) {
        return (matY->rows == k && matY->cols == n);
    }
    return (matY->rows == n && matY->cols == k);
}

static bool IsSupportedSddmmAlg(aclsparseSDDMMAlg_t alg) { return alg == ACL_SPARSE_SDDMM_ALG_DEFAULT; }

static aclsparseStatus_t ValidateSddmmOperations(aclsparseOperation_t opX, aclsparseOperation_t opY)
{
    if (opX != ACL_SPARSE_OP_NON_TRANSPOSE && opX != ACL_SPARSE_OP_TRANSPOSE) {
        OP_LOGE("aclsparseSDDMM", "opX must be NON_TRANSPOSE or TRANSPOSE, got %d", static_cast<int>(opX));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (opY != ACL_SPARSE_OP_NON_TRANSPOSE && opY != ACL_SPARSE_OP_TRANSPOSE) {
        OP_LOGE("aclsparseSDDMM", "opY must be NON_TRANSPOSE or TRANSPOSE, got %d", static_cast<int>(opY));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSddmmDescriptors(const aclsparseDnMatDescr* matX, const aclsparseDnMatDescr* matY,
                                                  const aclsparseSpMatDescr* matC)
{
    if (matX == nullptr || matY == nullptr || matC == nullptr) {
        OP_LOGE("aclsparseSDDMM", "matX/matY/matC is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matC->format != ACL_SPARSE_FORMAT_CSR) {
        OP_LOGE("aclsparseSDDMM", "matC format must be CSR, got %d", static_cast<int>(matC->format));
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    aclsparseStatus_t idxSt = AclsparseValidateSupportedCsrIndexTypes(matC->ptrType, matC->IdxType);
    if (idxSt != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSDDMM", "CSR index type not supported (ptr=%d, idx=%d)", static_cast<int>(matC->ptrType),
                static_cast<int>(matC->IdxType));
        return idxSt;
    }
    if (matC->baseType != ACL_SPARSE_INDEX_BASE_ZERO) {
        OP_LOGE("aclsparseSDDMM", "baseType must be ZERO, got %d", static_cast<int>(matC->baseType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSddmmDtypeAndDims(const aclsparseDnMatDescr* matX, const aclsparseDnMatDescr* matY,
                                                   const aclsparseSpMatDescr* matC, aclsparseOperation_t opX,
                                                   aclsparseOperation_t opY, aclDataType computeType,
                                                   aclsparseSDDMMAlg_t alg)
{
    if (!IsSupportedSddmmDtypeCombo(matX, matY, matC, computeType)) {
        OP_LOGE("aclsparseSDDMM", "unsupported dtype combo (X=%d, Y=%d, C=%d, compute=%d)",
                static_cast<int>(matX->valueType), static_cast<int>(matY->valueType), static_cast<int>(matC->valueType),
                static_cast<int>(computeType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!SddmmDimensionsMatch(matX, matY, matC, opX, opY)) {
        OP_LOGE("aclsparseSDDMM", "dimension mismatch: X(%ldx%ld) Y(%ldx%ld) C(%lux%lu)", static_cast<long>(matX->rows),
                static_cast<long>(matX->cols), static_cast<long>(matY->rows), static_cast<long>(matY->cols),
                static_cast<unsigned long>(matC->rows), static_cast<unsigned long>(matC->cols));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matX->rows == 0 || matX->cols == 0 || matY->rows == 0 || matY->cols == 0 || matC->rows == 0 ||
        matC->cols == 0) {
        OP_LOGE("aclsparseSDDMM",
                "zero dimension not supported "
                "(X=%ldx%ld, Y=%ldx%ld, C=%lux%lu)",
                static_cast<long>(matX->rows), static_cast<long>(matX->cols), static_cast<long>(matY->rows),
                static_cast<long>(matY->cols), static_cast<unsigned long>(matC->rows),
                static_cast<unsigned long>(matC->cols));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsSupportedSddmmAlg(alg)) {
        OP_LOGE("aclsparseSDDMM", "alg not supported: %d", static_cast<int>(alg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matC->rows > static_cast<uint64_t>(INT32_MAX) || matC->cols > static_cast<uint64_t>(INT32_MAX) ||
        matX->rows > static_cast<int64_t>(INT32_MAX) || matX->cols > static_cast<int64_t>(INT32_MAX) ||
        matY->rows > static_cast<int64_t>(INT32_MAX) || matY->cols > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE("aclsparseSDDMM", "dimension exceeds INT32_MAX");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // ldx/ldy are serialized as int32_t in SddmmTilingData.  Reject values
    // that cannot round-trip instead of silently truncating the GM stride.
    if (matX->ld > static_cast<int64_t>(INT32_MAX) || matY->ld > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE("aclsparseSDDMM", "dense leading dimension exceeds INT32_MAX (X=%ld, Y=%ld)",
                static_cast<long>(matX->ld), static_cast<long>(matY->ld));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSddmmCsrData(const aclsparseSpMatDescr* matC)
{
    // S2: nnz must fit int32_t (kernel reads nnz as int32_t).
    if (matC->nnz > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE("aclsparseSDDMM", "matC->nnz (%lu) exceeds INT32_MAX", static_cast<unsigned long>(matC->nnz));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // O1: CSR row pointers (ptrs) must never be null — the kernel always
    // dereferences ptrs[m] to read nnz, even when nnz == 0.
    if (matC->ptrs == nullptr) {
        OP_LOGE("aclsparseSDDMM", "matC->ptrs is null (kernel reads ptrs[m] for nnz)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // When nnz > 0, column indices and values must also be non-null.
    if (matC->nnz > 0) {
        if (matC->idxs == nullptr || matC->values == nullptr) {
            OP_LOGE("aclsparseSDDMM", "matC has nnz=%lu but idxs/values is null",
                    static_cast<unsigned long>(matC->nnz));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSddmmInputs(const aclsparseDnMatDescr* matX, const aclsparseDnMatDescr* matY,
                                      const aclsparseSpMatDescr* matC, aclsparseOperation_t opX,
                                      aclsparseOperation_t opY, aclDataType computeType, aclsparseSDDMMAlg_t alg)
{
    aclsparseStatus_t st = ValidateSddmmDescriptors(matX, matY, matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateSddmmOperations(opX, opY);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateSddmmDtypeAndDims(matX, matY, matC, opX, opY, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return ValidateSddmmCsrData(matC);
}

// Returns workspace start byte offsets for the TilingData / reorder / bin-edge
// regions, computed once and shared by GetBufferSize / Preprocess / Sddmm.
struct WsOffsets {
    int64_t tilingOff;
    int64_t reorderOff;
    int64_t binEdgeOff;
    int64_t totalBytes;
};

WsOffsets ComputeWsOffsets(int64_t m, int32_t blockDim)
{
    WsOffsets o;
    o.tilingOff = SDDMM_WS_HEADER_BYTES;
    o.reorderOff = sddmm_align_up(o.tilingOff + static_cast<int64_t>(sizeof(SddmmTilingData)), SDDMM_WS_ALIGN);
    o.binEdgeOff = sddmm_align_up(o.reorderOff + static_cast<int64_t>(sizeof(int32_t)) * m, SDDMM_WS_ALIGN);
    o.totalBytes = sddmm_align_up(o.binEdgeOff + static_cast<int64_t>(sizeof(int32_t)) * (blockDim + 1),
                                  SDDMM_WS_ALIGN);
    return o;
}

// Greedy bin packing: sort rows by nnz descending, then place each row into the
// bin with the smallest accumulated load. This balances workload across cores.
void GreedyRowBinPack(const std::vector<int32_t>& rowNnz, int32_t binNum, std::vector<int32_t>* reorder,
                      std::vector<int32_t>* binEdges)
{
    const int32_t m = static_cast<int32_t>(rowNnz.size());

    std::vector<int32_t> order(m);
    for (int32_t i = 0; i < m; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return rowNnz[a] > rowNnz[b]; });

    std::vector<std::vector<int32_t>> bins(binNum);
    std::vector<int64_t> binLoad(binNum, 0);
    for (int32_t idx : order) {
        int32_t pick = 0;
        for (int32_t b = 1; b < binNum; ++b) {
            if (binLoad[b] < binLoad[pick]) {
                pick = b;
            }
        }
        bins[pick].push_back(idx);
        binLoad[pick] += rowNnz[idx];
    }

    reorder->resize(m);
    binEdges->resize(static_cast<size_t>(binNum) + 1);
    int32_t cursor = 0;
    (*binEdges)[0] = 0;
    for (int32_t b = 0; b < binNum; ++b) {
        for (int32_t r : bins[b]) {
            (*reorder)[cursor++] = r;
        }
        (*binEdges)[b + 1] = cursor;
    }
}

aclsparseStatus_t BuildGreedyRowReorderFromCsr(const aclsparseSpMatDescr* matDesc, int64_t m, int32_t blockDim,
                                               std::vector<int32_t>* reorder, std::vector<int32_t>* binEdges)
{
    if (matDesc->ptrType != ACL_SPARSE_INDEX_32I) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    std::vector<int32_t> rowOff(static_cast<size_t>(m) + 1);
    aclError aclRet = aclrtMemcpy(rowOff.data(), sizeof(int32_t) * (m + 1), matDesc->ptrs, sizeof(int32_t) * (m + 1),
                                  ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSDDMM", "aclrtMemcpy D2H rowOff failed: %d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    std::vector<int32_t> rowNnz(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) {
        rowNnz[i] = rowOff[i + 1] - rowOff[i];
    }
    GreedyRowBinPack(rowNnz, blockDim, reorder, binEdges);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t WriteReorderToWorkspace(uint8_t* dWorkspaceBase, int64_t reorderOffset, int64_t binEdgeOffset,
                                          int64_t m, int32_t blockDim, const std::vector<int32_t>& reorder,
                                          const std::vector<int32_t>& binEdges)
{
    aclError aclRet = aclrtMemcpy(dWorkspaceBase + reorderOffset, sizeof(int32_t) * m, reorder.data(),
                                  sizeof(int32_t) * m, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSDDMM", "aclrtMemcpy H2D reorder failed: %d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    aclRet = aclrtMemcpy(dWorkspaceBase + binEdgeOffset, sizeof(int32_t) * (blockDim + 1), binEdges.data(),
                         sizeof(int32_t) * (blockDim + 1), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSDDMM", "aclrtMemcpy H2D binEdges failed: %d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// UB buffer accounting constants (must match kernel sddmm_kernel.cpp):
//   reduceTmp = 32KB, dotTmp = 32B, stridedBuf = 32KB, offsetBuf = 4KB,
//   plus alignment slack.
constexpr int32_t kSddmmUbFixedOverhead = SddmmBudget::kUbFixedOverhead; // 32KB reduce + 32B dotTmp + 32KB strided +
                                                                         // 4KB offset + slack
constexpr int32_t kSddmmKTileCap = 4095;                                 // DataCopyPad blockCount limit (strided)
constexpr int32_t kSddmmNTileCap = 64;                                   // batch scaling upper bound

// Compute kTile / nTile based on UB capacity (design §2.3 / §3.3).
//   FP32: per-K = 20 bytes (xQue 2×4 + yQue 2×4 + mulBuf 4, double-buffered)
//         per-N = 16 bytes (accBuf+cValsBuf+resultBuf+colIndBuf, all float/int32)
//   FP16: per-K = 20 bytes (xQue 2×2 + yQue 2×2 + mulBuf4 + xFp324 + yFp324)
//         per-N = 20 bytes (accBuf+resultBuf+colIndBuf float/int32 + cValsBuf+outBuf half + cValsFp32Buf float)
void ComputeSddmmTilingParams(int32_t k, int32_t dataType, int32_t& kTile, int32_t& nTile)
{
    const uint64_t ubSize = GetUbSize();
    const uint64_t ubAvailable = (ubSize > 0) ? ubSize : 253952ULL; // fallback 248KB

    const int32_t nTileFixed = kSddmmNTileCap;
    uint64_t perKElem = 20;
    uint64_t perNElem;
    if (dataType == SDDMM_DTYPE_FP16) {
        perNElem = 20;
    } else {
        perNElem = 16;
    }

    const uint64_t fixedOverhead = static_cast<uint64_t>(kSddmmUbFixedOverhead) +
                                   static_cast<uint64_t>(nTileFixed) * perNElem;
    const uint64_t ubForK = (ubAvailable > fixedOverhead) ? (ubAvailable - fixedOverhead) : 0;
    uint64_t maxKTile = (perKElem > 0) ? (ubForK / perKElem) : 1;
    if (maxKTile > static_cast<uint64_t>(kSddmmKTileCap)) {
        maxKTile = static_cast<uint64_t>(kSddmmKTileCap);
    }
    if (maxKTile < 1) {
        maxKTile = 1;
    }

    const uint64_t kU = static_cast<uint64_t>(k > 0 ? k : 1);
    kTile = static_cast<int32_t>(maxKTile < kU ? maxKTile : kU);
    if (kTile < 1) {
        kTile = 1;
    }
    nTile = nTileFixed;
}

} // namespace

// ============================================================================
// Stage 1: BufferSize
// ============================================================================
aclsparseStatus_t aclsparseSDDMMBufferSize(aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
                                           const void* alpha, aclsparseConstDnMatDescr_t matX,
                                           aclsparseConstDnMatDescr_t matY, const void* beta,
                                           aclsparseSpMatDescr_t matC, aclDataType computeType, aclsparseSDDMMAlg_t alg,
                                           size_t* size)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSDDMM", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (size == nullptr) {
        OP_LOGE("aclsparseSDDMM", "size is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE("aclsparseSDDMM", "alpha/beta is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseDnMatDescr* matXInner = (aclsparseDnMatDescr*)matX;
    aclsparseDnMatDescr* matYInner = (aclsparseDnMatDescr*)matY;
    aclsparseSpMatDescr* matCInner = (aclsparseSpMatDescr*)matC;
    aclsparseStatus_t st = ValidateSddmmInputs(matXInner, matYInner, matCInner, opX, opY, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    WsOffsets off = ComputeWsOffsets(static_cast<int64_t>(matCInner->rows), static_cast<int32_t>(GetSddmmBlockDim()));
    *size = static_cast<size_t>(off.totalBytes);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static SddmmTilingData SddmmHostTiling(const aclsparseDnMatDescr* matXInner, const aclsparseDnMatDescr* matYInner,
                                       const aclsparseSpMatDescr* matCInner, aclsparseOperation_t opX,
                                       aclsparseOperation_t opY, const WsOffsets& off)
{
    SddmmTilingData td{};
    td.m = static_cast<int32_t>(matCInner->rows);
    td.n = static_cast<int32_t>(matCInner->cols);
    td.k = (opX == ACL_SPARSE_OP_TRANSPOSE) ? static_cast<int32_t>(matXInner->rows) :
                                              static_cast<int32_t>(matXInner->cols);
    td.ldx = static_cast<int32_t>(matXInner->ld);
    td.ldy = static_cast<int32_t>(matYInner->ld);
    td.reorderOffset = static_cast<int32_t>(off.reorderOff);
    td.binEdgeOffset = static_cast<int32_t>(off.binEdgeOff);
    td.opX = (opX == ACL_SPARSE_OP_TRANSPOSE) ? 1 : 0;
    td.opY = (opY == ACL_SPARSE_OP_TRANSPOSE) ? 1 : 0;
    td.orderPair = static_cast<int32_t>(matXInner->order) * 2 + static_cast<int32_t>(matYInner->order);
    td.dataType = SddmmDataTypeFromAcl(matCInner->valueType);
    ComputeSddmmTilingParams(td.k, td.dataType, td.kTile, td.nTile);

    return td;
}

// Build reorder/bin/tiling and write into device workspace buffer.
static aclsparseStatus_t SddmmBuildTilingToBuffer(aclsparseDnMatDescr* matXInner, aclsparseDnMatDescr* matYInner,
                                                  aclsparseSpMatDescr* matCInner, aclsparseOperation_t opX,
                                                  aclsparseOperation_t opY, const void* alpha, const void* beta,
                                                  aclDataType computeType, aclsparseSDDMMAlg_t alg, void* buffer)
{
    const int64_t m = static_cast<int64_t>(matCInner->rows);
    const int32_t blockDim = static_cast<int32_t>(GetSddmmBlockDim());
    WsOffsets off = ComputeWsOffsets(m, blockDim);

    // 1) Compute reorder table + bin edges, copy into device workspace.
    // ptrs is guaranteed non-null by ValidateSddmmInputs.
    std::vector<int32_t> reorder;
    std::vector<int32_t> binEdges;
    {
        aclsparseStatus_t rst = BuildGreedyRowReorderFromCsr(matCInner, m, blockDim, &reorder, &binEdges);
        if (rst != ACL_SPARSE_STATUS_SUCCESS) {
            return rst;
        }
    }
    aclsparseStatus_t wst = WriteReorderToWorkspace(static_cast<uint8_t*>(buffer), off.reorderOff, off.binEdgeOff, m,
                                                    blockDim, reorder, binEdges);
    if (wst != ACL_SPARSE_STATUS_SUCCESS) {
        return wst;
    }

    // 2) Build TilingData on host, write to device workspace.
    SddmmTilingData td = SddmmHostTiling(matXInner, matYInner, matCInner, opX, opY, off);

    aclError aclRet = aclrtMemcpy(static_cast<uint8_t*>(buffer) + off.tilingOff, sizeof(SddmmTilingData), &td,
                                  sizeof(SddmmTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSDDMM", "aclrtMemcpy H2D tiling failed: %d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    // Record active buffer + pattern signature so that subsequent calls can
    // detect pattern changes and trigger a full rebuild instead of a refresh.
    // The pattern signature is kept in the sddmm-local side-table (not the
    // public descriptor) to avoid modifying the common header.
    matCInner->activeBuffer = buffer;
    const int64_t kForSig = (opX == ACL_SPARSE_OP_TRANSPOSE) ? matXInner->rows : matXInner->cols;
    SddmmPatternRecord(buffer, matCInner->rows, matCInner->nnz, kForSig);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SddmmEnsureTilingReady(aclsparseDnMatDescr* matX, aclsparseDnMatDescr* matY,
                                                aclsparseSpMatDescr* matC, aclsparseOperation_t opX,
                                                aclsparseOperation_t opY, const void* alpha, const void* beta,
                                                aclDataType computeType, aclsparseSDDMMAlg_t alg, void* buffer,
                                                const WsOffsets& off)
{
    // Fast path: same buffer AND same CSR pattern (rows + nnz). Only then can
    // we safely reuse reorder/binEdges. Mutable geometry/scalars are launch arguments.
    // If the pattern changed (e.g. descriptor reused with different CSR data),
    // fall through to a full rebuild to avoid stale reorder/binEdges. Pattern
    // is verified via the sddmm-local side-table (buffer key); a cache miss
    // also forces a rebuild.
    if (matC->activeBuffer == buffer) {
        const int64_t kForSig = (opX == ACL_SPARSE_OP_TRANSPOSE) ? matX->rows : matX->cols;
        if (SddmmPatternMatch(buffer, matC->rows, matC->nnz, kForSig)) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
    }
    return SddmmBuildTilingToBuffer(matX, matY, matC, opX, opY, alpha, beta, computeType, alg, buffer);
}

static aclsparseStatus_t SddmmBindScalars(aclsparseHandle_t handle, const void* alpha, const void* beta,
                                          aclDataType computeType, SddmmTilingData& td)
{
    aclsparsePointerMode_t pointerMode;
    const auto status = aclsparseGetPointerMode(handle, &pointerMode);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSDDMM", "pointer mode query failed: %d", static_cast<int>(status));
        return status;
    }
    td.scalarType = SddmmDataTypeFromAcl(computeType);
    if (pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        td.alphaDevice = reinterpret_cast<uintptr_t>(alpha);
        td.betaDevice = reinterpret_cast<uintptr_t>(beta);
    } else if (pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        td.alphaHost = SddmmReadScalarToF32(alpha, computeType);
        td.betaHost = SddmmReadScalarToF32(beta, computeType);
    } else {
        OP_LOGE("aclsparseSDDMM", "invalid scalar pointer mode");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SddmmRunKernel(aclsparseSpMatDescr* matC, aclsparseDnMatDescr* matX, aclsparseDnMatDescr* matY,
                                        void* buffer, const SddmmTilingData& td, aclrtStream stream)
{
    return sddmm_kernel_launch(reinterpret_cast<GM_ADDR>(matC->ptrs), reinterpret_cast<GM_ADDR>(matC->idxs),
                               reinterpret_cast<GM_ADDR>(matC->values), reinterpret_cast<GM_ADDR>(matX->values),
                               reinterpret_cast<GM_ADDR>(matY->values), reinterpret_cast<GM_ADDR>(buffer), td,
                               SddmmDataTypeFromAcl(matC->valueType), GetSddmmBlockDim(), stream);
}

static aclsparseStatus_t SddmmValidateAndCast(aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matX,
                                              aclsparseConstDnMatDescr_t matY, aclsparseSpMatDescr_t matC,
                                              const void* alpha, const void* beta, void* buffer, bool isPreprocess,
                                              aclsparseOperation_t opX, aclsparseOperation_t opY,
                                              aclDataType computeType, aclsparseSDDMMAlg_t alg,
                                              aclsparseDnMatDescr** matXInner, aclsparseDnMatDescr** matYInner,
                                              aclsparseSpMatDescr** matCInner)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSDDMM", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    *matXInner = (aclsparseDnMatDescr*)matX;
    *matYInner = (aclsparseDnMatDescr*)matY;
    *matCInner = (aclsparseSpMatDescr*)matC;
    aclsparseStatus_t st = ValidateSddmmInputs(*matXInner, *matYInner, *matCInner, opX, opY, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE("aclsparseSDDMM", "alpha/beta is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (buffer == nullptr) {
        if (isPreprocess) {
            OP_LOGE("aclsparseSDDMM", "buffer is nullptr in Preprocess");
            return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
        }
        OP_LOGE("aclsparseSDDMM", "buffer is nullptr in Execute");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Stage 2: Preprocess
// ============================================================================
aclsparseStatus_t aclsparseSDDMMPreprocess(aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
                                           const void* alpha, aclsparseConstDnMatDescr_t matX,
                                           aclsparseConstDnMatDescr_t matY, const void* beta,
                                           aclsparseSpMatDescr_t matC, aclDataType computeType, aclsparseSDDMMAlg_t alg,
                                           void* buffer)
{
    aclsparseDnMatDescr* matXInner = nullptr;
    aclsparseDnMatDescr* matYInner = nullptr;
    aclsparseSpMatDescr* matCInner = nullptr;
    aclsparseStatus_t st = SddmmValidateAndCast(handle, matX, matY, matC, alpha, beta, buffer, true, opX, opY,
                                                computeType, alg, &matXInner, &matYInner, &matCInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    st = SddmmBuildTilingToBuffer(matXInner, matYInner, matCInner, opX, opY, alpha, beta, computeType, alg, buffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // activeBuffer / pattern signature are now set inside SddmmBuildTilingToBuffer.
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Stage 3: Execute
// ============================================================================
aclsparseStatus_t aclsparseSDDMM(aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
                                 const void* alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
                                 const void* beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
                                 aclsparseSDDMMAlg_t alg, void* buffer)
{
    aclsparseDnMatDescr* matXInner = nullptr;
    aclsparseDnMatDescr* matYInner = nullptr;
    aclsparseSpMatDescr* matCInner = nullptr;
    aclsparseStatus_t st = SddmmValidateAndCast(handle, matX, matY, matC, alpha, beta, buffer, false, opX, opY,
                                                computeType, alg, &matXInner, &matYInner, &matCInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    aclrtStream stream = nullptr;
    aclsparseGetStream(handle, &stream);
    WsOffsets off = ComputeWsOffsets(static_cast<int64_t>(matCInner->rows), static_cast<int32_t>(GetSddmmBlockDim()));
    st = SddmmEnsureTilingReady(matXInner, matYInner, matCInner, opX, opY, alpha, beta, computeType, alg, buffer, off);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    SddmmTilingData td = SddmmHostTiling(matXInner, matYInner, matCInner, opX, opY, off);
    st = SddmmBindScalars(handle, alpha, beta, computeType, td);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return SddmmRunKernel(matCInner, matXInner, matYInner, buffer, td, stream);
}
