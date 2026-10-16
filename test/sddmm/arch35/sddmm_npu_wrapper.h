/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SDDMM_SDDMM_ARCH35_SDDMM_NPU_WRAPPER_H_
#define TEST_SDDMM_SDDMM_ARCH35_SDDMM_NPU_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "sddmm_golden.h"  // shares MakeSddmmSparsity + FP16 bit helpers with golden

namespace sparse_test {

// ============================================================================
// NPU workflow result for SDDMM.
// valuesOut holds the updated C values (CSR values array) converted to FP64.
// ============================================================================

struct SddmmNpuResult {
    std::vector<double> valuesOut;  // size = nnz, FP64
    size_t bufferSize = 0;

    // Individual API return codes for debugging / exception-path assertions.
    aclsparseStatus_t bufferSizeRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t preprocessRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t executeRet = ACL_SPARSE_STATUS_SUCCESS;
};

// ============================================================================
// String -> ACL enum helpers.
// Defined here (not in param.h) because aclsparseSDDMMAlg_t may not yet be
// declared in cann_ops_sparse.h; the wrapper is the natural place that already
// pulls in the full ACL header set.
// ============================================================================

inline aclDataType ParseDtype(const std::string& s) {
    if (s == "ACL_FLOAT16") return ACL_FLOAT16;
    if (s == "ACL_FLOAT") return ACL_FLOAT;
    return ACL_FLOAT;
}

inline aclsparseOperation_t ParseOperation(const std::string& s) {
    if (s == "ACL_SPARSE_OP_TRANSPOSE") return ACL_SPARSE_OP_TRANSPOSE;
    if (s == "ACL_SPARSE_OP_CONJUGATE_TRANSPOSE") return ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    return ACL_SPARSE_OP_NON_TRANSPOSE;
}

// Parse dense matrix memory layout. Defaults to row-major for any unrecognised
// string (keeps existing ROW-only CSV rows unchanged).
inline aclsparseOrder_t ParseOrder(const std::string& s) {
    if (s == "ACL_SPARSE_ORDER_COL") return ACL_SPARSE_ORDER_COL;
    return ACL_SPARSE_ORDER_ROW;
}

// aclsparseSDDMMAlg_t / ACL_SPARSE_SDDMM_ALG_DEFAULT are declared by the
// operator implementation in cann_ops_sparse.h (requirement §2.2).
inline aclsparseSDDMMAlg_t ParseSddmmAlg(const std::string& s) {
    if (s == "ACL_SPARSE_SDDMM_ALG_DEFAULT") return ACL_SPARSE_SDDMM_ALG_DEFAULT;
    return ACL_SPARSE_SDDMM_ALG_DEFAULT;
}

// ============================================================================
// SDDMM NPU three-stage workflow (RAII-managed).
//
// Template parameter T:
//   float    -> FP32 path (dtype = ACL_FLOAT)
//   uint16_t -> FP16 path (dtype = ACL_FLOAT16, host stores IEEE-754 bit patterns)
//
// Stages (per requirement §2.2 / test plan §2.3):
//   1. aclsparseSDDMMBufferSize  -> query workspace size
//   2. aclsparseSDDMMPreprocess  -> optional preprocessing
//   3. aclsparseSDDMM            -> execute, C.values updated in place
//
// All descriptors and device buffers are RAII-managed (HandleManager /
// DnMatManager / SpMatManager / DeviceBuffer). No bare aclsparseDestroy* calls.
// ============================================================================

// alpha / beta are host scalars passed by pointer (const void*).
// Per ACL convention, the pointer type matches computeType:
//   ACL_FLOAT   -> float (4 bytes)
//   ACL_FLOAT16 -> half  (2 bytes)
inline void PrepareSddmmScalars(float alpha, float beta, aclDataType computeType,
                                 float& alphaHost, float& betaHost,
                                 uint16_t& alphaFp16, uint16_t& betaFp16,
                                 const void*& alphaPtr, const void*& betaPtr) {
    alphaHost = alpha;
    betaHost = beta;
    alphaPtr = &alphaHost;
    betaPtr = &betaHost;
    if (computeType == ACL_FLOAT16) {
        alphaFp16 = Fp32ToFp16Bits(alpha);
        betaFp16 = Fp32ToFp16Bits(beta);
        alphaPtr = &alphaFp16;
        betaPtr = &betaFp16;
    }
}

// Create the three SDDMM descriptors (X / Y dense, C sparse) from the device
// buffers already allocated by the caller.
//
// X descriptor shape depends on opX:
//   NON_TRANSPOSE: X is m×k (rows=m, cols=k)  -> computes X·Y
//   TRANSPOSE:     X is k×m (rows=k, cols=m)  -> computes X^T·Y
// Leading dimension follows the memory layout:
//   row-major (order=ROW): ld = cols
//   col-major (order=COL): ld = rows
// Y descriptor shape depends on opY:
//   NON_TRANSPOSE: Y is k×n (rows=k, cols=n)  -> computes X·Y
//   TRANSPOSE:     Y is n×k (rows=n, cols=k)  -> computes X·Y^T
template <typename T>
inline void CreateSddmmDescriptors(
    int64_t m, int64_t n, int64_t k,
    aclsparseOperation_t opX, aclsparseOperation_t opY,
    aclsparseOrder_t orderX, aclsparseOrder_t orderY,
    aclDataType dtype, int64_t nnz,
    void* dXRaw, void* dYRaw,
    void* dRowOffRaw, void* dColIndRaw, void* dValsRaw,
    DnMatManager& matX, DnMatManager& matY, SpMatManager& matC) {
    const bool xTransposed = (opX == ACL_SPARSE_OP_TRANSPOSE);
    const int64_t xRows = xTransposed ? k : m;
    const int64_t xCols = xTransposed ? m : k;
    const int64_t xLd = (orderX == ACL_SPARSE_ORDER_ROW) ? xCols : xRows;
    matX = DnMatManager::createConst(xRows, xCols, xLd, dXRaw, dtype, orderX);
    const bool yTransposed = (opY == ACL_SPARSE_OP_TRANSPOSE);
    const int64_t yRows = yTransposed ? n : k;
    const int64_t yCols = yTransposed ? k : n;
    const int64_t yLd = (orderY == ACL_SPARSE_ORDER_ROW) ? yCols : yRows;
    matY = DnMatManager::createConst(yRows, yCols, yLd, dYRaw, dtype, orderY);
    // C: m×n CSR, values updated in place (non-const).
    matC = SpMatManager::createCsr(m, n, nnz, dRowOffRaw, dColIndRaw, dValsRaw,
                                    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                    ACL_SPARSE_INDEX_BASE_ZERO, dtype);
}

// Read back updated C values (CSR values array, in place) and convert to FP64.
template <typename T>
inline void ReadBackSddmmValues(DeviceBuffer& dVals, int64_t nnz, SddmmNpuResult& result) {
    if (nnz <= 0) {
        return;
    }
    std::vector<T> hValsOut(static_cast<size_t>(nnz));
    dVals.copyToHost(hValsOut.data(), static_cast<size_t>(nnz) * sizeof(T));
    result.valuesOut.resize(static_cast<size_t>(nnz));
    for (int64_t i = 0; i < nnz; i++) {
        if constexpr (std::is_same_v<T, float>) {
            result.valuesOut[static_cast<size_t>(i)] =
                static_cast<double>(hValsOut[static_cast<size_t>(i)]);
        } else {
            result.valuesOut[static_cast<size_t>(i)] =
                static_cast<double>(Fp16BitsToFp32(hValsOut[static_cast<size_t>(i)]));
        }
    }
}

// Run the three SDDMM stages (BufferSize → Preprocess → Execute) and wait for
// kernel completion. Returns true on success; on failure records the failing
// stage's return code in `result` and returns false. The workspace buffer is
// allocated internally and freed when this function returns (the kernel has
// been synced before return, so the workspace is no longer needed).
inline bool RunSddmmStages(HandleManager& handle,
                            aclsparseOperation_t opX, aclsparseOperation_t opY,
                            const void* alphaPtr, const void* betaPtr,
                            DnMatManager& matX, DnMatManager& matY,
                            SpMatManager& matC,
                            aclDataType computeType, aclsparseSDDMMAlg_t alg,
                            aclrtStream stream, SddmmNpuResult& result) {
    // --- Stage 1: BufferSize -------------------------------------------------
    result.bufferSizeRet = aclsparseSDDMMBufferSize(
        handle.get(), opX, opY, alphaPtr,
        matX.cget(), matY.cget(), betaPtr, matC.get(),
        computeType, alg, &result.bufferSize);
    if (result.bufferSizeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[SDDMM NPU] aclsparseSDDMMBufferSize failed: "
                  << result.bufferSizeRet << std::endl;
        return false;
    }

    // --- Stage 2: Preprocess (optional but called per requirement §2.2) ------
    DeviceBuffer dBuffer;
    if (result.bufferSize > 0) {
        dBuffer = DeviceBuffer::alloc(result.bufferSize);
    }
    result.preprocessRet = aclsparseSDDMMPreprocess(
        handle.get(), opX, opY, alphaPtr,
        matX.cget(), matY.cget(), betaPtr, matC.get(),
        computeType, alg, dBuffer.get());
    if (result.preprocessRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[SDDMM NPU] aclsparseSDDMMPreprocess failed: "
                  << result.preprocessRet << std::endl;
        return false;
    }

    // --- Stage 3: Execute ----------------------------------------------------
    result.executeRet = aclsparseSDDMM(
        handle.get(), opX, opY, alphaPtr,
        matX.cget(), matY.cget(), betaPtr, matC.get(),
        computeType, alg, dBuffer.get());
    if (result.executeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[SDDMM NPU] aclsparseSDDMM failed: "
                  << result.executeRet << std::endl;
        return false;
    }

    // Wait for kernel completion before reading back C values.
    auto aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "[SDDMM NPU] aclrtSynchronizeStream failed: " << aclRet << std::endl;
        return false;
    }
    return true;
}

template <typename T>
inline SddmmNpuResult SddmmNpu(
    HandleManager& handle, aclrtStream stream,
    int64_t m, int64_t n, int64_t k,
    aclsparseOperation_t opX, aclsparseOperation_t opY,
    float alpha, float beta,
    aclDataType dtype, aclDataType computeType,
    aclsparseSDDMMAlg_t alg,
    aclsparseOrder_t orderX, aclsparseOrder_t orderY,
    const std::vector<int32_t>& rowOffsets,
    const std::vector<int32_t>& colIndices,
    const std::vector<T>& hCInitValues,
    const std::vector<T>& hX,
    const std::vector<T>& hY,
    int64_t nnz) {
    SddmmNpuResult result;
    handle.setStream(stream);

    // --- Device buffers ------------------------------------------------------
    // rowOffsets always present (m+1 ints). colInd / values empty when nnz == 0.
    DeviceBuffer dRowOff = DeviceBuffer::copyFrom(
        rowOffsets.data(), static_cast<size_t>(m + 1) * sizeof(int32_t));
    DeviceBuffer dColInd;
    DeviceBuffer dVals;
    if (nnz > 0) {
        dColInd = DeviceBuffer::copyFrom(
            colIndices.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
        dVals = DeviceBuffer::copyFrom(
            hCInitValues.data(), static_cast<size_t>(nnz) * sizeof(T));
    }
    // Y element count is n*k regardless of opY (n*k == k*n). The flat buffer
    // is the same; only the descriptor shape/ld and the kernel access pattern
    // differ between opY=NON_TRANSPOSE (Y is k×n, ld=n) and opY=TRANSPOSE
    // (Y is n×k, ld=k).
    DeviceBuffer dX = DeviceBuffer::copyFrom(
        hX.data(), static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(T));
    DeviceBuffer dY = DeviceBuffer::copyFrom(
        hY.data(), static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(T));

    // --- Descriptors + scalars -----------------------------------------------
    DnMatManager matX;
    DnMatManager matY;
    SpMatManager matC;
    CreateSddmmDescriptors<T>(m, n, k, opX, opY, orderX, orderY, dtype, nnz,
                              dX.raw(), dY.raw(), dRowOff.get(), dColInd.get(),
                              dVals.get(), matX, matY, matC);
    float alphaHost = 0.0f;
    float betaHost = 0.0f;
    uint16_t alphaFp16 = 0;
    uint16_t betaFp16 = 0;
    const void* alphaPtr = nullptr;
    const void* betaPtr = nullptr;
    PrepareSddmmScalars(alpha, beta, computeType, alphaHost, betaHost,
                        alphaFp16, betaFp16, alphaPtr, betaPtr);

    // --- Three-stage workflow + sync -----------------------------------------
    if (!RunSddmmStages(handle, opX, opY, alphaPtr, betaPtr,
                        matX, matY, matC, computeType, alg, stream, result)) {
        return result;
    }

    // --- Read back updated C values (CSR values array, in place) -------------
    ReadBackSddmmValues<T>(dVals, nnz, result);
    return result;
}

}  // namespace sparse_test

#endif  // TEST_SDDMM_SDDMM_ARCH35_SDDMM_NPU_WRAPPER_H_
