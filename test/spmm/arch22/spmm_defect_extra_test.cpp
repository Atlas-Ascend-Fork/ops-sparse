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

// Defect regression tests D1-D6 (host-side validation, buffer/tiling-reuse,
// DEVICE pointer mode and the SpMM pattern cache). Kept in a separate
// translation unit so spmm_test.cpp stays below the codecheck source-file
// size limit.

#include "spmm_defect_common.h"
#include "spmm_csr_mat.h"
#include <math.h>
#include <vector>

// ============================================================================
// Boundary/defect regression tests (D1/D2/D3 fixes, see PR history).
// These target host-side validation and buffer/tiling-reuse logic directly
// rather than end-to-end numerics.
// ============================================================================

// D3 regression: matA->rows is narrowed to *signed* int32_t in
// spmm_csr_mat.cpp's row-order preprocessing (GreedyRowBinPack/
// BuildIdentityReorder/DoPreProcess), so it must be rejected above
// INT32_MAX, not just above UINT32_MAX like cols/nnz/matB->cols. This only
// exercises host-side validation (GetBufferSize does not allocate memory
// proportional to rows), so it is safe to run with huge declared row counts.
static int test_d3_row_bound_boundary(aclrtStream stream)
{
    printf("--- Test: D3 regression - matA->rows bound must be INT32_MAX, not UINT32_MAX ---\n");
    aclsparseHandle_t handle = NULL;
    CHECK_ACL_SPARSE(aclsparseCreate(&handle))
    CHECK_ACL_SPARSE(aclsparseSetStream(handle, stream))
    CHECK_ACL_SPARSE(aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_HOST))

    float dummyB = 0.0f, dummyC = 0.0f;
    float alpha = 1.0f, beta = 0.0f;
    const int64_t k = 4, n = 4;
    aclsparseDnMatDescr_t matB = NULL, matC = NULL;
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(&matB, k, n, n, &dummyB, ACL_FLOAT, ACL_SPARSE_ORDER_ROW))

    struct RowCase { int64_t rows; bool expectSupported; const char *label; };
    const RowCase cases[] = {
        {static_cast<int64_t>(INT32_MAX), true, "rows == INT32_MAX (boundary, must be accepted)"},
        {static_cast<int64_t>(INT32_MAX) + 1, false, "rows == INT32_MAX+1 (must now be rejected)"},
        {static_cast<int64_t>(UINT32_MAX), false, "rows == UINT32_MAX (old bound, must still be rejected)"},
    };
    int fail = 0;
    for (const auto &c : cases) {
        aclsparseSpMatDescr_t matA = NULL;
        CHECK_ACL_SPARSE(aclsparseCreateCsr(&matA, c.rows, k, 1, nullptr, nullptr, nullptr,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT))
        CHECK_ACL_SPARSE(aclsparseCreateDnMat(&matC, c.rows, n, n, &dummyC, ACL_FLOAT, ACL_SPARSE_ORDER_ROW))
        size_t size = 0;
        aclsparseStatus_t st = aclsparseSpMMGetBufferSize(handle,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, &size);
        bool supported = (st == ACL_SPARSE_STATUS_SUCCESS);
        if (supported != c.expectSupported) {
            printf("  [Fail] %s: status=%d supported=%d expected=%d\n",
                c.label, static_cast<int>(st), supported, c.expectSupported);
            fail = 1;
        } else {
            printf("  [OK] %s\n", c.label);
        }
        CHECK_ACL_SPARSE(aclsparseDestroySpMat(matA))
        CHECK_ACL_SPARSE(aclsparseDestroyDnMat(matC))
    }
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(matB))
    CHECK_ACL_SPARSE(aclsparseDestroy(handle))
    printf("  [%s] D3 row bound regression.\r\n\n", fail ? "Failed" : "Success");
    return fail;
}

// D1 regression: aclsparseSpMMGetBufferSize() must succeed with null
// matA/matB/matC data buffers (it only needs dimensions and never
// dereferences them), while aclsparseSpMMPreprocess()/aclsparseSpMM() must
// cleanly reject null buffers instead of crashing (no abort()/segfault).
static int test_d1_null_buffer_handling(aclrtStream stream)
{
    printf("--- Test: D1 regression - null buffer handling differs by entry point ---\n");
    aclsparseHandle_t handle = NULL;
    CHECK_ACL_SPARSE(aclsparseCreate(&handle))
    CHECK_ACL_SPARSE(aclsparseSetStream(handle, stream))
    CHECK_ACL_SPARSE(aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_HOST))

    const int64_t m = 4, k = 4, n = 4;
    float alpha = 1.0f, beta = 0.0f;
    aclsparseSpMatDescr_t matA = NULL;
    CHECK_ACL_SPARSE(aclsparseCreateCsr(&matA, m, k, 1, nullptr, nullptr, nullptr,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT))
    aclsparseDnMatDescr_t matB = NULL, matC = NULL;
    float placeholder = 0.0f;
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(&matB, k, n, n, &placeholder, ACL_FLOAT, ACL_SPARSE_ORDER_ROW))
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(&matC, m, n, n, &placeholder, ACL_FLOAT, ACL_SPARSE_ORDER_ROW))
    CHECK_ACL_SPARSE(aclsparseDnMatSetValues(matB, nullptr))

    int fail = 0;
    size_t size = 0;
    aclsparseStatus_t st = aclsparseSpMMGetBufferSize(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, &size);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        printf("  [Fail] GetBufferSize should succeed with null matA/matB buffers (dims-only), got status=%d\n",
            static_cast<int>(st));
        fail = 1;
    } else {
        printf("  [OK] GetBufferSize succeeds with null buffers (size=%zu)\n", size);
    }

    void *dBuffer = NULL;
    CHECK_ACL(aclrtMalloc(&dBuffer, (size > 0 ? size : 4096), ACL_MEM_MALLOC_HUGE_FIRST))
    st = aclsparseSpMMPreprocess(handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, dBuffer);
    if (st == ACL_SPARSE_STATUS_SUCCESS) {
        printf("  [Fail] Preprocess should reject null matB->values, but returned SUCCESS\n");
        fail = 1;
    } else {
        printf("  [OK] Preprocess cleanly rejects null matB->values (status=%d, no crash)\n",
            static_cast<int>(st));
    }

    CHECK_ACL(aclrtFree(dBuffer))
    CHECK_ACL_SPARSE(aclsparseDestroySpMat(matA))
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(matB))
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(matC))
    CHECK_ACL_SPARSE(aclsparseDestroy(handle))
    printf("  [%s] D1 null buffer regression.\r\n\n", fail ? "Failed" : "Success");
    return fail;
}

// GetBufferSize does not otherwise dereference the handle, so pin the public
// API contract explicitly instead of accidentally accepting nullptr.
static int test_d4_get_buffer_size_null_handle()
{
    printf("--- Test: D4 regression - GetBufferSize rejects null handle ---\n");
    float alpha = 1.0f, beta = 0.0f;
    size_t size = 0;
    const aclsparseStatus_t st = aclsparseSpMMGetBufferSize(nullptr,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, nullptr, nullptr, &beta, nullptr, ACL_FLOAT,
        ACL_SPARSE_SPMM_ALG_DEFAULT, &size);
    const bool pass = st == ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    printf("  [%s] null handle status=%d (expected=%d)\r\n\n", pass ? "Success" : "Failed",
        static_cast<int>(st), static_cast<int>(ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR));
    return pass ? 0 : 1;
}

// matA1 is 4x3 with 5 nonzeros, matA2 is 64x3 with 64. Both multiply the same
// 3x2 dense B, so the sparse operand's shape is the only thing that differs.
// Builds matA1 (the small sparse operand), the shared dense B, and C1.
// Builds matA2, a much larger and unrelated sparse operand, plus its output C2.
static int d2_build_large_operands(D2Operands &ops)
{
    const int64_t nnz2 = kD2M2;
    std::vector<int> hRowPtrA2(kD2M2 + 1);
    std::vector<int> hColIndA2(nnz2);
    std::vector<float> hValA2(nnz2);
    for (int64_t i = 0; i < kD2M2; i++) {
        hRowPtrA2[i] = static_cast<int>(i);
        hColIndA2[i] = static_cast<int>(i % kD2K);
        hValA2[i] = static_cast<float>(i + 1);
    }
    hRowPtrA2[kD2M2] = static_cast<int>(nnz2);
    const size_t rowBytes = hRowPtrA2.size() * sizeof(int);
    const size_t colBytes = hColIndA2.size() * sizeof(int);
    const size_t valBytes = hValA2.size() * sizeof(float);
    CHECK_ACL(aclrtMalloc((void **)&ops.dRowPtrA2, rowBytes, ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&ops.dColIndA2, colBytes, ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&ops.dValA2, valBytes, ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&ops.dC2, static_cast<size_t>(kD2M2) * kD2N * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMemcpy(ops.dRowPtrA2, rowBytes, hRowPtrA2.data(), rowBytes,
        ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(ops.dColIndA2, colBytes, hColIndA2.data(), colBytes,
        ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(ops.dValA2, valBytes, hValA2.data(), valBytes,
        ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL_SPARSE(aclsparseCreateCsr(&ops.matA2, kD2M2, kD2K, nnz2,
        ops.dRowPtrA2, ops.dColIndA2, ops.dValA2,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT))
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(&ops.matC2, kD2M2, kD2N, kD2N, ops.dC2, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW))
    return 0;
}

// Runs the probe: matA1 into the shared buffer, then matA2 (which overwrites
// the buffer's cached preprocessing data), then matA1 again without another
// Preprocess. Execute must reject that stale workspace instead of rebuilding
// CSR data on the host. A fresh Preprocess(matA1) must restore normal execution.
static int d2_run_probe(aclsparseHandle_t handle, aclrtStream stream, const D2Operands &ops,
    void *sharedBuffer, aclsparseStatus_t *probeStatus, aclsparseStatus_t *recoveryStatus)
{
    float alpha = 1.0f, beta = 0.0f;
    CHECK_ACL_SPARSE(aclsparseSpMMPreprocess(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, ops.matA1, ops.matB, &beta, ops.matC1,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, sharedBuffer))
    CHECK_ACL_SPARSE(aclsparseSpMM(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, ops.matA1, ops.matB, &beta, ops.matC1,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, sharedBuffer))
    CHECK_ACL_SPARSE(aclsparseSpMMPreprocess(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, ops.matA2, ops.matB, &beta, ops.matC2,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, sharedBuffer))
    CHECK_ACL_SPARSE(aclsparseSpMM(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, ops.matA2, ops.matB, &beta, ops.matC2,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, sharedBuffer))
    std::vector<float> zeros(static_cast<size_t>(kD2M1) * kD2N, 0.0f);
    const size_t c1Bytes = zeros.size() * sizeof(float);
    CHECK_ACL(aclrtMemcpy(ops.dC1, c1Bytes, zeros.data(), c1Bytes, ACL_MEMCPY_HOST_TO_DEVICE))
    *probeStatus = aclsparseSpMM(handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA1, ops.matB, &beta, ops.matC1, ACL_FLOAT,
        ACL_SPARSE_SPMM_ALG_DEFAULT, sharedBuffer);
    *recoveryStatus = aclsparseSpMMPreprocess(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, ops.matA1, ops.matB, &beta, ops.matC1,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, sharedBuffer);
    if (*recoveryStatus == ACL_SPARSE_STATUS_SUCCESS) {
        *recoveryStatus = aclsparseSpMM(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, ops.matA1, ops.matB, &beta, ops.matC1,
            ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, sharedBuffer);
    }
    float nextAlpha = 3.0f, nextBeta = 1.0f;
    if (*recoveryStatus == ACL_SPARSE_STATUS_SUCCESS) {
        *recoveryStatus = aclsparseSpMM(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &nextAlpha, ops.matA1, ops.matB, &nextBeta, ops.matC1,
            ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, sharedBuffer);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream))
    return 0;
}

// Reads C1 back and compares it against what matA1 must still produce.
static int d2_check_result(const D2Operands &ops)
{
    std::vector<float> hC1(static_cast<size_t>(kD2M1) * kD2N);
    const size_t bytes = hC1.size() * sizeof(float);
    CHECK_ACL(aclrtMemcpy(hC1.data(), bytes, ops.dC1, bytes, ACL_MEMCPY_DEVICE_TO_HOST))
    int fail = 0;
    for (size_t i = 0; i < hC1.size(); i++) {
        const float expected = 4.0f * kD2Expected1[i];
        if (hC1[i] != expected) {
            printf("  [Fail] C[%zu]=%.4f expected %.4f (stale workspace or stream order?)\n",
                i, hC1[i], expected);
            fail = 1;
        }
    }
    if (fail == 0) {
        printf("  [OK] re-Preprocess recovery and ordered scalar refreshes are correct\n");
    }
    return fail;
}

// Releases everything the two d2_build_*_operands helpers allocated.
static int d2_release_operands(D2Operands &ops)
{
    CHECK_ACL_SPARSE(aclsparseDestroySpMat(ops.matA1))
    CHECK_ACL_SPARSE(aclsparseDestroySpMat(ops.matA2))
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(ops.matB))
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(ops.matC1))
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(ops.matC2))
    CHECK_ACL(aclrtFree(ops.dRowPtrA1))
    CHECK_ACL(aclrtFree(ops.dColIndA1))
    CHECK_ACL(aclrtFree(ops.dValA1))
    CHECK_ACL(aclrtFree(ops.dB))
    CHECK_ACL(aclrtFree(ops.dC1))
    CHECK_ACL(aclrtFree(ops.dRowPtrA2))
    CHECK_ACL(aclrtFree(ops.dColIndA2))
    CHECK_ACL(aclrtFree(ops.dValA2))
    CHECK_ACL(aclrtFree(ops.dC2))
    return 0;
}

// D2 regression: two independent matA descriptors sharing one workspace
// buffer (a legitimate memory-saving pattern: Preprocess(matA1, buf); ...;
// Preprocess(matA2, buf); ...) must not let aclsparseSpMM(matA1, buf) reuse
// matA2's cached M/K/N tiling just because matA1->activeBuffer == buf.
static int test_d2_shared_buffer_pattern_mismatch(aclrtStream stream)
{
    printf("--- Test: D2 regression - shared-buffer tiling must track pattern, not just buffer identity ---\n");
    aclsparseHandle_t handle = NULL;
    CHECK_ACL_SPARSE(aclsparseCreate(&handle))
    CHECK_ACL_SPARSE(aclsparseSetStream(handle, stream))
    CHECK_ACL_SPARSE(aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_HOST))

    D2Operands ops;
    CHECK_TEST(d2_build_small_operands(ops) == 0, "failed to build matA1/B/C1");
    CHECK_TEST(d2_build_large_operands(ops) == 0, "failed to build matA2/C2");

    // Size the shared buffer for the larger matrix (matA2).
    float alpha = 1.0f, beta = 0.0f;
    size_t bufSize2 = 0;
    CHECK_ACL_SPARSE(aclsparseSpMMGetBufferSize(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, ops.matA2, ops.matB, &beta, ops.matC2,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, &bufSize2))
    void *sharedBuffer = NULL;
    CHECK_ACL(aclrtMalloc(&sharedBuffer, bufSize2, ACL_MEM_MALLOC_HUGE_FIRST))

    aclsparseStatus_t probeStatus = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t recoveryStatus = ACL_SPARSE_STATUS_SUCCESS;
    CHECK_TEST(d2_run_probe(handle, stream, ops, sharedBuffer, &probeStatus, &recoveryStatus) == 0,
        "D2 probe sequence failed before reaching the check");

    int fail = 0;
    if (probeStatus != ACL_SPARSE_STATUS_INVALID_VALUE) {
        printf("  [Fail] stale SpMM(matA1) returned status=%d, expected INVALID_VALUE\n",
            static_cast<int>(probeStatus));
        fail = 1;
    }
    if (recoveryStatus != ACL_SPARSE_STATUS_SUCCESS) {
        printf("  [Fail] SpMM(matA1) after re-Preprocess returned status=%d\n",
            static_cast<int>(recoveryStatus));
        fail = 1;
    } else if (fail == 0) {
        fail = d2_check_result(ops);
    }

    CHECK_ACL(aclrtFree(sharedBuffer))
    CHECK_TEST(d2_release_operands(ops) == 0, "failed to release D2 operands");
    CHECK_ACL_SPARSE(aclsparseDestroy(handle))
    printf("  [%s] D2 shared-buffer regression.\r\n\n", fail ? "Failed" : "Success");
    return fail;
}

static void d5_release(aclsparseHandle_t handle, D2Operands &ops,
    void *buffer, float *dAlpha, float *dBeta)
{
    CHECK_ACL(aclrtFree(buffer))
    CHECK_ACL(aclrtFree(dAlpha))
    CHECK_ACL(aclrtFree(dBeta))
    CHECK_ACL_SPARSE(aclsparseDestroySpMat(ops.matA1))
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(ops.matB))
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(ops.matC1))
    CHECK_ACL(aclrtFree(ops.dRowPtrA1))
    CHECK_ACL(aclrtFree(ops.dColIndA1))
    CHECK_ACL(aclrtFree(ops.dValA1))
    CHECK_ACL(aclrtFree(ops.dB))
    CHECK_ACL(aclrtFree(ops.dC1))
    CHECK_ACL_SPARSE(aclsparseDestroy(handle))
}

// D5 regression: DEVICE pointer mode must read alpha/beta from device memory
// with a stream-ordered device-to-device copy, not a synchronous D2H read.
static int test_d5_device_pointer_mode(aclrtStream stream)
{
    printf("--- Test: D5 regression - DEVICE pointer mode reads alpha/beta on-stream ---\n");
    aclsparseHandle_t handle = NULL;
    CHECK_ACL_SPARSE(aclsparseCreate(&handle))
    CHECK_ACL_SPARSE(aclsparseSetStream(handle, stream))
    CHECK_ACL_SPARSE(aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_DEVICE))

    D2Operands ops;
    CHECK_TEST(d2_build_small_operands(ops) == 0, "failed to build D5 operands");

    std::vector<float> hC0(static_cast<size_t>(kD2M1) * kD2N, 1.0f);
    const size_t cBytes = hC0.size() * sizeof(float);
    CHECK_ACL(aclrtMemcpy(ops.dC1, cBytes, hC0.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE))

    float *dAlpha = NULL, *dBeta = NULL;
    CHECK_ACL(aclrtMalloc((void **)&dAlpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&dBeta, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST))
    const float alpha = 2.0f, beta = 0.5f;
    CHECK_ACL(aclrtMemcpy(dAlpha, sizeof(float), &alpha, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(dBeta, sizeof(float), &beta, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE))

    size_t bufSize = 0;
    CHECK_ACL_SPARSE(aclsparseSpMMGetBufferSize(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, dAlpha, ops.matA1, ops.matB, dBeta, ops.matC1,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, &bufSize))
    void *buffer = NULL;
    CHECK_ACL(aclrtMalloc(&buffer, bufSize, ACL_MEM_MALLOC_HUGE_FIRST))

    CHECK_ACL_SPARSE(aclsparseSpMMPreprocess(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, dAlpha, ops.matA1, ops.matB, dBeta, ops.matC1,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, buffer))
    CHECK_ACL_SPARSE(aclsparseSpMM(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, dAlpha, ops.matA1, ops.matB, dBeta, ops.matC1,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, buffer))
    CHECK_ACL(aclrtSynchronizeStream(stream))

    std::vector<float> hC(static_cast<size_t>(kD2M1) * kD2N);
    CHECK_ACL(aclrtMemcpy(hC.data(), cBytes, ops.dC1, cBytes, ACL_MEMCPY_DEVICE_TO_HOST))
    int fail = 0;
    for (size_t i = 0; i < hC.size(); i++) {
        const float expected = alpha * kD2Expected1[i] + beta * hC0[i];
        if (fabsf(hC[i] - expected) > 1e-5f) {
            printf("  [Fail] C[%zu]=%.4f expected %.4f\n", i, hC[i], expected);
            fail = 1;
        }
    }
    if (fail == 0) {
        printf("  [OK] DEVICE pointer mode alpha/beta read correctly\n");
    }

    d5_release(handle, ops, buffer, dAlpha, dBeta);
    printf("  [%s] D5 DEVICE pointer mode.\r\n\n", fail ? "Failed" : "Success");
    return fail;
}

constexpr int kD6Workspaces = 20;

struct D6Ops {
    aclsparseDnMatDescr_t matB = NULL;
    aclsparseDnMatDescr_t matC = NULL;
    aclsparseSpMatDescr_t matA[kD6Workspaces] = {};
    void *buffers[kD6Workspaces] = {};
    int *dRowPtr = NULL;
    int *dColInd = NULL;
    float *dVal = NULL;
    float *dB = NULL;
    float *dC = NULL;
};

static int d6_check(aclsparseHandle_t handle, const D6Ops &ops, aclrtStream stream)
{
    const float alpha = 1.0f, beta = 0.0f;
    const aclsparseStatus_t st = aclsparseSpMM(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, ops.matA[0], ops.matB, &beta, ops.matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, ops.buffers[0]);
    CHECK_ACL(aclrtSynchronizeStream(stream))

    int fail = 0;
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        printf("  [Fail] re-execute of descriptor[0] returned status=%d, expected SUCCESS\n",
            static_cast<int>(st));
        fail = 1;
    }
    std::vector<float> hC(static_cast<size_t>(kD2M1) * kD2N);
    const size_t cBytes = hC.size() * sizeof(float);
    CHECK_ACL(aclrtMemcpy(hC.data(), cBytes, ops.dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST))
    for (size_t i = 0; i < hC.size(); i++) {
        if (fabsf(hC[i] - kD2Expected1[i]) > 1e-5f) {
            printf("  [Fail] C[%zu]=%.4f expected %.4f\n", i, hC[i], kD2Expected1[i]);
            fail = 1;
        }
    }
    if (fail == 0) {
        printf("  [OK] 20-workspace re-execute is correct (no eviction)\n");
    }
    return fail;
}

static void d6_release(aclsparseHandle_t handle, D6Ops &ops)
{
    for (int i = 0; i < kD6Workspaces; i++) {
        CHECK_ACL(aclrtFree(ops.buffers[i]))
        CHECK_ACL_SPARSE(aclsparseDestroySpMat(ops.matA[i]))
    }
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(ops.matB))
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(ops.matC))
    CHECK_ACL(aclrtFree(ops.dRowPtr))
    CHECK_ACL(aclrtFree(ops.dColInd))
    CHECK_ACL(aclrtFree(ops.dVal))
    CHECK_ACL(aclrtFree(ops.dB))
    CHECK_ACL(aclrtFree(ops.dC))
    CHECK_ACL_SPARSE(aclsparseDestroy(handle))
}

static void d6_alloc_and_fill(D6Ops &ops)
{
    CHECK_ACL(aclrtMalloc((void **)&ops.dRowPtr, sizeof(kD2RowPtrA1), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&ops.dColInd, sizeof(kD2ColIndA1), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&ops.dVal, sizeof(kD2ValA1), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&ops.dB, sizeof(kD2B), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&ops.dC, static_cast<size_t>(kD2M1) * kD2N * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMemcpy(ops.dRowPtr, sizeof(kD2RowPtrA1), kD2RowPtrA1, sizeof(kD2RowPtrA1),
        ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(ops.dColInd, sizeof(kD2ColIndA1), kD2ColIndA1, sizeof(kD2ColIndA1),
        ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(ops.dVal, sizeof(kD2ValA1), kD2ValA1, sizeof(kD2ValA1),
        ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(ops.dB, sizeof(kD2B), kD2B, sizeof(kD2B), ACL_MEMCPY_HOST_TO_DEVICE))
}

// D6 regression: the SpMM pattern cache must not evict a live entry once more
// than 16 workspaces have been preprocessed.
static int test_d6_twenty_workspaces_no_eviction(aclrtStream stream)
{
    printf("--- Test: D6 regression - 20 workspaces must not evict a live entry ---\n");
    aclsparseHandle_t handle = NULL;
    CHECK_ACL_SPARSE(aclsparseCreate(&handle))
    CHECK_ACL_SPARSE(aclsparseSetStream(handle, stream))
    CHECK_ACL_SPARSE(aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_HOST))

    const int64_t nnz = 5;

    D6Ops ops;
    d6_alloc_and_fill(ops);
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(&ops.matB, kD2K, kD2N, kD2N, ops.dB, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW))
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(&ops.matC, kD2M1, kD2N, kD2N, ops.dC, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW))

    const float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < kD6Workspaces; i++) {
        CHECK_ACL_SPARSE(aclsparseCreateCsr(&ops.matA[i], kD2M1, kD2K, nnz,
            ops.dRowPtr, ops.dColInd, ops.dVal,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT))
        size_t bufSize = 0;
        CHECK_ACL_SPARSE(aclsparseSpMMGetBufferSize(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, ops.matA[i], ops.matB, &beta, ops.matC,
            ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, &bufSize))
        CHECK_ACL(aclrtMalloc(&ops.buffers[i], bufSize, ACL_MEM_MALLOC_HUGE_FIRST))
        CHECK_ACL_SPARSE(aclsparseSpMMPreprocess(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, ops.matA[i], ops.matB, &beta, ops.matC,
            ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, ops.buffers[i]))
        CHECK_ACL_SPARSE(aclsparseSpMM(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, ops.matA[i], ops.matB, &beta, ops.matC,
            ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, ops.buffers[i]))
    }

    int fail = d6_check(handle, ops, stream);
    d6_release(handle, ops);
    printf("  [%s] D6 20-workspace pattern cache.\r\n\n", fail ? "Failed" : "Success");
    return fail;
}

// #1 边界算术回归：直接调用生产代码共用的端点截断 SpmmArch22ClampBinEnd（被
// GreedyBlockBinPack / BuildIdentityReorder 调用），验证 m 接近 INT32_MAX 时端点
// 不回绕。无需分配 m 大小的数组；m 取 2147483584=2^31-64（64 对齐的最后整块边界）
// 到 INT32_MAX 之间，覆盖 lo+blk 越过 int32 上限的区间。旧 int32 实现下这些取值
// 会回绕成负值，使断言失败，故可作为回归护栏。
static int test_binpack_endpoint_boundary()
{
    int fail = 0;
    const int64_t mValues[] = {
        2147483584LL, 2147483585LL, 2147483646LL, 2147483647LL,  // 2147483647 == INT32_MAX
    };
    const int32_t blockValues[] = {1, 2, 16, 40, 64, 128};

    for (int64_t m : mValues) {
        for (int32_t block : blockValues) {
            // BuildIdentityReorder：binEdges[b+1] = ClampBinEnd(b*rowsPerBin, rowsPerBin, m)。
            const int64_t rowsPerBin = (m + block - 1) / block;
            int64_t prev = 0;
            for (int32_t b = 0; b < block; b++) {
                const int32_t end = SpmmArch22ClampBinEnd(
                    static_cast<int64_t>(b) * rowsPerBin, rowsPerBin, m);
                if (end < prev || end < 0 || end > m) { fail++; }
                prev = end;
            }
            if (prev != m) { fail++; }

            // GreedyBlockBinPack：hi = ClampBinEnd(blkId*blk, blk, m)，块应恰好划分 [0, m)。
            const int64_t nBlocks = (m + block - 1) / block;
            int64_t covered = 0;
            int64_t expectedLo = 0;
            for (int64_t blkId = 0; blkId < nBlocks; blkId++) {
                const int64_t lo = blkId * block;
                const int32_t hi = SpmmArch22ClampBinEnd(lo, block, m);
                if (lo != expectedLo || hi <= lo || hi > m) { fail++; }
                covered += hi - lo;
                expectedLo = hi;
            }
            if (covered != m || expectedLo != m) { fail++; }
        }
    }
    printf("  [%s] binpack endpoint int64 boundary arithmetic.\r\n\n", fail ? "Failed" : "Success");
    return fail;
}

int run_defect_regression_tests_extra(aclrtStream stream)
{
    int fail = 0;
    fail += test_binpack_endpoint_boundary();
    fail += test_d3_row_bound_boundary(stream);
    fail += test_d1_null_buffer_handling(stream);
    fail += test_d4_get_buffer_size_null_handle();
    fail += test_d2_shared_buffer_pattern_mismatch(stream);
    fail += test_d5_device_pointer_mode(stream);
    fail += test_d6_twenty_workspaces_no_eviction(stream);
    return fail;
}
