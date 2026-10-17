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
#include <vector>
#include <iostream>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "tiling/platform/platform_ascendc.h"
#include "cube_spmm.h"

extern "C" void cube_spmm_kernel_launch(
    GM_ADDR rw_ptr, GM_ADDR col_ref, GM_ADDR vals,
    GM_ADDR b, GM_ADDR core_info, GM_ADDR c,
    GM_ADDR workspaceGM, const cube_spmm::CubeSpmmTilingData &tiling,
    uint32_t usedCoreNum, void *stream);

namespace cube_spmm {

inline aclsparseCubeSpmmMatDescr *ToCubeSpmmMatInner(aclsparseConstCubeSpmmMatDescr_t desc)
{
    return const_cast<aclsparseCubeSpmmMatDescr *>(
        reinterpret_cast<const aclsparseCubeSpmmMatDescr *>(desc));
}

static bool IsSupportedCubeSpmmDtypeCombo(
    const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC,
    aclDataType computeType)
{
    return (matB->valueType == ACL_FLOAT16) &&
           (matC->valueType == ACL_FLOAT) &&
           (computeType == ACL_FLOAT);
}

static aclsparseStatus_t ValidateDenseByteSizes(
    int64_t M, int64_t K, int64_t bLd, int64_t cLd)
{
    constexpr uint64_t kMaxSizeT = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    if (static_cast<uint64_t>(M) * static_cast<uint64_t>(cLd) * sizeof(float) > kMaxSizeT ||
        static_cast<uint64_t>(K) * static_cast<uint64_t>(bLd) * sizeof(uint16_t) > kMaxSizeT) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateDescriptorsNotNull(
    aclsparseConstCubeSpmmMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseConstDnMatDescr_t matC)
{
    if (matA == nullptr || matB == nullptr || matC == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateDescriptorShapes(
    aclsparseConstCubeSpmmMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseConstDnMatDescr_t matC)
{
    if (matA->rows <= 0 || matA->cols <= 0 || matA->nnz < 0 ||
        matB->rows <= 0 || matB->cols <= 0 ||
        matC->rows <= 0 || matC->cols <= 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->cols != matB->rows) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->rows != matC->rows || matB->cols != matC->cols) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matB->cols % kTileN != 0) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateDescriptorLayouts(
    aclsparseConstDnMatDescr_t matB,
    aclsparseConstDnMatDescr_t matC)
{
    if (matB->ld < matB->cols || matC->ld < matC->cols) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matB->values == nullptr || matC->values == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    constexpr int64_t kMaxInt32 = static_cast<int64_t>(INT32_MAX);
    if (matB->ld > kMaxInt32 || matC->ld > kMaxInt32) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matB->order != ACL_SPARSE_ORDER_ROW || matC->order != ACL_SPARSE_ORDER_ROW) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateCubeSpmmCommon(
    aclsparseConstCubeSpmmMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseConstDnMatDescr_t matC,
    aclDataType computeType)
{
    aclsparseStatus_t st = ValidateDescriptorsNotNull(matA, matB, matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateDescriptorShapes(matA, matB, matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateDescriptorLayouts(matB, matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (!IsSupportedCubeSpmmDtypeCombo(matB, matC, computeType)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matA->numCores <= 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!AreDimensionsSupported(matA->rows, matA->cols, matB->cols)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ValidateDenseByteSizes(matA->rows, matA->cols, matB->ld, matC->ld);
}

static aclsparseStatus_t BuildCubeSpmmTilingData(
    const aclsparseCubeSpmmMatDescr *matA,
    const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC,
    CubeSpmmTilingData &td)
{
    // Dimension arithmetic is performed in int64_t; TilingData stores M/K/N and
    // the leading dimensions as int32_t because they are bounded by INT32_MAX,
    // and only the BCSR block-count product M * K is allowed to reach INT64_MAX.
    int64_t M = matA->rows;
    int64_t K = matA->cols;
    int64_t N = matB->cols;

    td.M = static_cast<int32_t>(M);
    td.N = static_cast<int32_t>(N);
    td.K = static_cast<int32_t>(K);
    td.usedCoreNum = static_cast<uint32_t>(matA->numCores);

    // Tile sizes are still hardcoded to 16 in the kernel; reject anything else.
    td.tileM = cube_spmm::kTileM;
    td.tileN = cube_spmm::kTileN;
    if (td.tileM != 16 || td.tileN != 16) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    td.tailM = (M % td.tileM == 0) ? td.tileM : static_cast<int32_t>(M % td.tileM);

    uint32_t alignNum = 32 / sizeof(uint16_t);  // 16
    uint32_t lastKLength = static_cast<uint32_t>(K % alignNum);
    if (lastKLength == 0) {
        lastKLength = alignNum;
    }
    td.lastKLength = lastKLength;

    td.bLd = static_cast<int32_t>(matB->ld);
    td.cLd = static_cast<int32_t>(matC->ld);
    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace cube_spmm

// ============================================================================
// Dense B padding helper (separated from sparse A preprocessing so that A can
// be reused while B is padded per call).
// ============================================================================

namespace {

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

inline int64_t ComputePaddedN(int64_t N, int64_t blockK)
{
    if (blockK <= 0) {
        return -1;
    }
    // L0B alignment: blockK * N_pad * sizeof(float16) must be 512B aligned.
    // CopyInB stride also requires blockK * sizeof(float16) to be 32B aligned.
    int64_t bytesPerKBlock = blockK * static_cast<int64_t>(sizeof(uint16_t));
    if (bytesPerKBlock <= 0 || bytesPerKBlock % 32 != 0) {
        return -1;
    }
    int64_t alignN = 512 / bytesPerKBlock;
    if (alignN <= 0) {
        alignN = 1;
    }
    return CeilDiv(N, alignN) * alignN;
}

static aclsparseStatus_t PadDenseMatrixB(
    int64_t K, int64_t N, int64_t ld, const void *bHost,
    int64_t blockK,
    int64_t *nPadOut, void **bPadOut)
{
    if (nPadOut == nullptr || bPadOut == nullptr || bHost == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (blockK * static_cast<int64_t>(sizeof(uint16_t)) % 32 != 0) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    int64_t nPad = ComputePaddedN(N, blockK);
    if (nPad < 0) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    size_t rowBytes = static_cast<size_t>(N) * sizeof(uint16_t);
    size_t paddedRowBytes = static_cast<size_t>(nPad) * sizeof(uint16_t);
    if (paddedRowBytes == 0 || static_cast<size_t>(K) > std::numeric_limits<size_t>::max() / paddedRowBytes) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    size_t totalBytes = static_cast<size_t>(K) * paddedRowBytes;

    // Build padded B on host, then transfer to device in one H2D copy.
    std::vector<uint8_t> bHostPad(totalBytes, 0);
    const uint8_t *src = static_cast<const uint8_t *>(bHost);
    size_t srcStride = static_cast<size_t>(ld) * sizeof(uint16_t);
    for (int64_t r = 0; r < K; ++r) {
        std::copy_n(src + r * srcStride, rowBytes,
                    bHostPad.data() + r * paddedRowBytes);
    }

    void *bPad = nullptr;
    aclError ret = aclrtMalloc(&bPad, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMemcpy(bPad, totalBytes, bHostPad.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(bPad);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    *nPadOut = nPad;
    *bPadOut = bPad;
    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

aclsparseStatus_t aclsparseCubeSpmmPadDenseMatrixB(
    aclsparseHandle_t handle,
    int64_t bRows, int64_t bCols, int64_t bLd, const void *bValues,
    aclDataType bType, aclsparseOrder_t bOrder,
    int64_t *nPadOut, void **bPadOut)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (bValues == nullptr || nPadOut == nullptr || bPadOut == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (bRows <= 0 || bCols <= 0 || bLd < bCols) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (bType != ACL_FLOAT16 || bOrder != ACL_SPARSE_ORDER_ROW) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return PadDenseMatrixB(bRows, bCols, bLd, bValues,
                           cube_spmm::kTileK, nPadOut, bPadOut);
}

// ============================================================================
// Descriptor create / destroy
// ============================================================================

aclsparseStatus_t aclsparseCreateCubeSpmmMat(
    aclsparseCubeSpmmMatDescr_t *descr,
    int64_t rows, int64_t cols, int64_t nnz,
    int64_t blockM, int64_t blockK, int32_t numCores)
{
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (rows <= 0 || cols <= 0 || nnz < 0 ||
        blockM <= 0 || blockK <= 0 || numCores <= 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Cube SpMM kernel currently hardcodes 16x16 BCSR blocks.
    if (blockM != cube_spmm::kTileM || blockK != cube_spmm::kTileK) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // rows/cols are stored as int32_t in TilingData and bounded by INT32_MAX.
    // Keep rows * cols within INT64_MAX to avoid overflow in BCSR block-index
    // products. nnz must not exceed the dense element count.
    constexpr int64_t kMaxInt32 = static_cast<int64_t>(INT32_MAX);
    constexpr int64_t kMaxInt64 = std::numeric_limits<int64_t>::max();
    if (rows > kMaxInt32 || cols > kMaxInt32 || rows > kMaxInt64 / cols ||
        nnz > rows * cols) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *inner = new (std::nothrow) aclsparseCubeSpmmMatDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    inner->rows = rows;
    inner->cols = cols;
    inner->nnz = nnz;
    inner->blockM = blockM;
    inner->blockK = blockK;
    inner->numCores = numCores;
    // Cube SpMM 当前固定使用 0-based int32 索引 + float16 值。
    inner->baseType = ACL_SPARSE_INDEX_BASE_ZERO;
    inner->ptrType = ACL_SPARSE_INDEX_32I;
    inner->IdxType = ACL_SPARSE_INDEX_32I;
    inner->valueType = ACL_FLOAT16;
    *descr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroyCubeSpmmMat(
    aclsparseConstCubeSpmmMatDescr_t descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    auto *inner = cube_spmm::ToCubeSpmmMatInner(descr);
    if (inner->rwPtr != nullptr) { aclrtFree(inner->rwPtr); }
    if (inner->colRef != nullptr) { aclrtFree(inner->colRef); }
    if (inner->vals != nullptr) { aclrtFree(inner->vals); }
    if (inner->coreInfo != nullptr) { aclrtFree(inner->coreInfo); }
    delete inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Cube SpMM GetBufferSize / Preprocess / SpMM
// ============================================================================

aclsparseStatus_t aclsparseCubeSpmmGetBufferSize(
    aclsparseHandle_t handle,
    const void *alpha, aclsparseConstCubeSpmmMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    size_t *size)
{
    if (size == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseStatus_t st = cube_spmm::ValidateCubeSpmmCommon(matA, matB, matC, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // Tiling travels with the kernel launch as a by-value argument, so no
    // device workspace is required.
    *size = 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCubeSpmm(
    aclsparseHandle_t handle,
    const void *alpha, aclsparseConstCubeSpmmMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    void *buffer)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = cube_spmm::ValidateCubeSpmmCommon(matA, matB, matC, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    auto *matAInner = cube_spmm::ToCubeSpmmMatInner(matA);
    if (matAInner->rwPtr == nullptr || matAInner->colRef == nullptr ||
        matAInner->vals == nullptr || matAInner->coreInfo == nullptr) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    // 每次 SpMM 都按当前 B/C 维度重建 tiling；tiling 作为 kernel 启动参数
    // 按值随 <<<>>> 一起下发，无需写入 workspace。buffer 作为 workspace 形参
    // 透传给 kernel（与官方 spmm 形态一致，当前 kernel 内未实际使用）。
    cube_spmm::CubeSpmmTilingData td{};
    st = cube_spmm::BuildCubeSpmmTilingData(matAInner, matB, matC, td);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    aclrtStream stream = nullptr;
    aclsparseStatus_t getStreamSt = aclsparseGetStream(handle, &stream);
    if (getStreamSt != ACL_SPARSE_STATUS_SUCCESS) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    cube_spmm_kernel_launch(
        reinterpret_cast<GM_ADDR>(matAInner->rwPtr),
        reinterpret_cast<GM_ADDR>(matAInner->colRef),
        reinterpret_cast<GM_ADDR>(matAInner->vals),
        reinterpret_cast<GM_ADDR>(matB->values),
        reinterpret_cast<GM_ADDR>(matAInner->coreInfo),
        reinterpret_cast<GM_ADDR>(matC->values),
        reinterpret_cast<GM_ADDR>(buffer), td,
        static_cast<uint32_t>(matAInner->numCores), stream);

    aclError ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
