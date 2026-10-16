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

#include <acl/acl.h>           // aclrtMalloc, aclrtMemcpy, etc.
#include <cann_ops_sparse.h>   // aclsparseSpMM
#include <stdio.h>             // printf
#include <stdlib.h>            // EXIT_FAILURE
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <random>

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

#define CHECK_ACL_SPARSE(x)                                                                       \
    do {                                                                                          \
        aclsparseStatus_t __ret = x;                                                                \
        if (__ret != ACL_SPARSE_STATUS_SUCCESS) {                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclSparseError:" << __ret << std::endl; \
        }                                                                                         \
    } while (0);

// Thread-safe random column index generation (replaces non-reentrant rand())
static inline int random_col(uint64_t cols)
{
    thread_local std::mt19937 rng(42);
    if (cols == 0) {
        return 0;
    }
    std::uniform_int_distribution<int> dist(0, static_cast<int>(cols) - 1);
    return dist(rng);
}

// Fill one CSR row with non-repeating random column indices.
// Returns the updated global non-zero cursor (cur + row_nnz).
static inline uint64_t fill_row_random_cols(
    uint64_t row_nnz, uint64_t cols,
    int *indices, float *values, uint64_t cur)
{
    std::unordered_set<int> used;
    used.reserve(row_nnz * 2);
    for (uint64_t j = 0; j < row_nnz; j++) {
        int col;
        do {
            col = random_col(cols);
        } while (used.count(col));
        used.insert(col);
        indices[cur] = col;
        values[cur] = 0.1f + 0.0001f * cur;
        cur++;
    }
    return cur;
}

void generate_random_csr(uint64_t rows, uint64_t cols, uint64_t nnz,
    int *offsets, int *indices, float *values)
{
    offsets[0] = 0;
    if (rows == 0 || cols == 0) {
        return;
    }
    uint64_t cur = 0;
    for (uint64_t i = 0; i < rows; i++) {
        uint64_t row_nnz = (rows != 0) ? (nnz / rows) : 0;
        if (rows != 0 && i < nnz % rows) {
            row_nnz++;
        }
        cur = fill_row_random_cols(row_nnz, cols, indices, values, cur);
        offsets[i + 1] = (int)cur;
    }
}

// Generate CSR with power-law row distribution (Zipf-like):
//   A small fraction of rows contain most of the non-zeros,
//   mimicking real-world graph data (social networks, GNN datasets).
void generate_powerlaw_csr(uint64_t rows, uint64_t cols, uint64_t nnz,
    int *offsets, int *indices, float *values)
{
    offsets[0] = 0;
    if (rows == 0 || cols == 0) {
        return;
    }
    // Zipf weights: weight[i] = 1 / (i + 1)
    std::vector<double> weight(rows);
    double total = 0.0;
    for (uint64_t i = 0; i < rows; i++) {
        weight[i] = 1.0 / (i + 1);
        total += weight[i];
    }
    if (total == 0.0) {
        return;
    }
    // Assign nnz per row proportional to weight
    std::vector<uint64_t> row_nnz(rows, 0);
    uint64_t assigned = 0;
    for (uint64_t i = 0; i < rows; i++) {
        row_nnz[i] = (uint64_t)(nnz * weight[i] / total);
        assigned += row_nnz[i];
    }
    // Distribute remaining nnz to top rows
    for (uint64_t i = 0; assigned < nnz && i < rows; i++) {
        row_nnz[i]++;
        assigned++;
    }

    uint64_t cur = 0;
    for (uint64_t i = 0; i < rows; i++) {
        uint64_t r_nnz = row_nnz[i];
        if (r_nnz > cols) r_nnz = cols;
        cur = fill_row_random_cols(r_nnz, cols, indices, values, cur);
        offsets[i + 1] = (int)cur;
    }
}

void spmm_csr_cpu(uint64_t M, uint64_t K, uint64_t N,
    const int *offsets, const int *indices, const float *values,
    const float *B, const float *C, float *Y,
    float alpha, float beta)
{
    for (uint64_t i = 0; i < M; i++) {
        for (uint64_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int p = offsets[i]; p < offsets[i + 1]; p++) {
                acc += values[p] * B[(uint64_t)indices[p] * N + j];
            }
            Y[i * N + j] = alpha * acc + beta * C[i * N + j];
        }
    }
}

int verify_result(const float *actual, const float *expected,
    uint64_t count, float rel_tol, float abs_tol)
{
    uint64_t error_count = 0;
    float max_diff = 0.0f;
    for (uint64_t i = 0; i < count; i++) {
        float diff = fabs(actual[i] - expected[i]);
        float denom = fmax(fabs(expected[i]), 1e-6f);
        if (diff > max_diff) {
            max_diff = diff;
        }
        float rel_diff = (denom != 0.0f) ? (diff / denom) : 0.0f;
        if (diff > abs_tol && rel_diff > rel_tol) {
            if (error_count < 10) {
                printf("  [%lu] expected=%f, actual=%f\r\n", i, expected[i], actual[i]);
            }
            error_count++;
        }
    }
    printf("  MaxDiff=%.6e, ErrorCount=%lu/%lu\n", max_diff, error_count, count);
    return (int)error_count;
}

// Run NPU SpMM pipeline (device alloc → warmup → timed runs → verify).
// All device resources are freed before returning.
// Returns npu_ms; *out_error_count is set by verify_result.
static void alloc_and_copy_spmm_device(
    uint64_t M, uint64_t K, uint64_t N, uint64_t nnz,
    const int *hA_offsets, const int *hA_indices, const float *hA_values,
    const float *hB, const float *hC,
    int **dA_offsets, int **dA_indices, float **dA_values, float **dB, float **dC)
{
    CHECK_ACL(aclrtMalloc((void **)dA_offsets, (M + 1) * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)dA_indices, nnz * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)dA_values, nnz * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)dB, K * N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)dC, M * N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMemcpy(*dA_offsets, (M + 1) * sizeof(int),
        hA_offsets, (M + 1) * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(*dA_indices, nnz * sizeof(int),
        hA_indices, nnz * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(*dA_values, nnz * sizeof(float),
        hA_values, nnz * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(*dB, K * N * sizeof(float),
        hB, K * N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(*dC, M * N * sizeof(float),
        hC, M * N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE))
}

// Set up ACL descriptors, preprocess and warmup on pre-allocated device buffers.
static void setup_acl_and_warmup(
    uint64_t M, uint64_t K, uint64_t N, uint64_t nnz,
    float alpha, float beta,
    const float *hC, aclrtStream stream,
    int *dA_offsets, int *dA_indices, float *dA_values, float *dB, float *dC,
    aclsparseHandle_t *handle, aclsparseSpMatDescr_t *matA,
    aclsparseDnMatDescr_t *matB, aclsparseDnMatDescr_t *matC,
    void **dBuffer)
{
    const int warmup = 3;
    CHECK_ACL_SPARSE(aclsparseCreate(handle))
    CHECK_ACL_SPARSE(aclsparseSetStream(*handle, stream))
    CHECK_ACL_SPARSE(aclsparseCreateCsr(matA, M, K, nnz,
        dA_offsets, dA_indices, dA_values,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT))
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(matB, K, N, N, dB, ACL_FLOAT, ACL_SPARSE_ORDER_ROW))
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(matC, M, N, N, dC, ACL_FLOAT, ACL_SPARSE_ORDER_ROW))
    size_t bufferSize = 0;
    CHECK_ACL_SPARSE(aclsparseSpMMGetBufferSize(*handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, *matA, *matB, &beta, *matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, &bufferSize))
    CHECK_ACL(aclrtMalloc(dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL_SPARSE(aclsparseSpMMPreprocess(*handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, *matA, *matB, &beta, *matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, *dBuffer))
    for (int w = 0; w < warmup; w++) {
        CHECK_ACL(aclrtMemcpy(dC, M * N * sizeof(float), hC, M * N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE))
        CHECK_ACL_SPARSE(aclsparseSpMM(*handle,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha, *matA, *matB, &beta, *matC,
            ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, *dBuffer))
    }
    CHECK_ACL(aclrtSynchronizeStream(stream))
}

static void cleanup_spmm_device(
    aclrtEvent evStart, aclrtEvent evEnd,
    aclsparseSpMatDescr_t matA, aclsparseDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclsparseHandle_t handle, void *dBuffer,
    int *dA_offsets, int *dA_indices, float *dA_values, float *dB, float *dC)
{
    CHECK_ACL(aclrtDestroyEvent(evStart))
    CHECK_ACL(aclrtDestroyEvent(evEnd))
    CHECK_ACL_SPARSE(aclsparseDestroySpMat(matA))
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(matB))
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(matC))
    CHECK_ACL_SPARSE(aclsparseDestroy(handle))
    CHECK_ACL(aclrtFree(dBuffer))
    CHECK_ACL(aclrtFree(dA_offsets))
    CHECK_ACL(aclrtFree(dA_indices))
    CHECK_ACL(aclrtFree(dA_values))
    CHECK_ACL(aclrtFree(dB))
    CHECK_ACL(aclrtFree(dC))
}

static double run_npu_spmm_benchmark(
    uint64_t M, uint64_t K, uint64_t N, uint64_t nnz,
    float alpha, float beta,
    const int *hA_offsets, const int *hA_indices, const float *hA_values,
    const float *hB, const float *hC, float *hY, const float *hYBase,
    aclrtStream stream, int *out_error_count)
{
    const int repeat = 10;
    int *dA_offsets, *dA_indices;
    float *dA_values, *dB, *dC;
    alloc_and_copy_spmm_device(M, K, N, nnz, hA_offsets, hA_indices, hA_values,
        hB, hC, &dA_offsets, &dA_indices, &dA_values, &dB, &dC);

    aclsparseHandle_t handle = NULL;
    aclsparseSpMatDescr_t matA = NULL;
    aclsparseDnMatDescr_t matB, matC;
    void *dBuffer;
    setup_acl_and_warmup(M, K, N, nnz, alpha, beta, hC, stream,
        dA_offsets, dA_indices, dA_values, dB, dC,
        &handle, &matA, &matB, &matC, &dBuffer);

    aclrtEvent evStart, evEnd;
    CHECK_ACL(aclrtCreateEvent(&evStart))
    CHECK_ACL(aclrtCreateEvent(&evEnd))
    float total_npu_ms = 0.0f;
    for (int r = 0; r < repeat; r++) {
        CHECK_ACL(aclrtMemcpy(dC, M * N * sizeof(float), hC, M * N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE))
        CHECK_ACL(aclrtRecordEvent(evStart, stream))
        CHECK_ACL_SPARSE(aclsparseSpMM(handle,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, dBuffer))
        CHECK_ACL(aclrtRecordEvent(evEnd, stream))
        CHECK_ACL(aclrtSynchronizeStream(stream))
        float elapsed_ms = 0.0f;
        CHECK_ACL(aclrtEventElapsedTime(&elapsed_ms, evStart, evEnd))
        total_npu_ms += elapsed_ms;
    }
    double npu_ms = total_npu_ms / repeat;

    CHECK_ACL(aclrtMemcpy(hY, M * N * sizeof(float), dC, M * N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST))
    *out_error_count = verify_result(hY, hYBase, M * N, 1e-3f, 1e-3f);

    cleanup_spmm_device(evStart, evEnd, matA, matB, matC, handle, dBuffer,
        dA_offsets, dA_indices, dA_values, dB, dC);
    return npu_ms;
}

// Distribution type for CSR generation
enum CsrDist { DIST_UNIFORM, DIST_POWERLAW };

// Allocate all host buffers with overflow validation (G.RES.02-CPP).
// Returns true if all allocations succeeded, false if any overflowed or failed.
static bool allocate_test_buffers(
    uint64_t M, uint64_t K, uint64_t N, uint64_t nnz,
    int **hA_offsets, int **hA_indices, float **hA_values,
    float **hB, float **hC, float **hY, float **hYBase)
{
    *hA_offsets = NULL;
    *hA_indices = NULL;
    *hA_values  = NULL;
    *hB = NULL;
    *hC = NULL;
    *hY = NULL;
    *hYBase = NULL;

    if (N == 0 || M == 0 || nnz == 0) {
        return false;
    }

    // N must be validated before use in allocation size multiplication
    if (N > SIZE_MAX / sizeof(float)) {
        return false;
    }

    size_t sz_m1  = (size_t)M + 1;
    size_t sz_nnz = (size_t)nnz;
    size_t sz_kxn = (size_t)K * (size_t)N;
    size_t sz_mxn = (size_t)M * (size_t)N;

    if (sz_m1 <= SIZE_MAX / sizeof(int)) {
        *hA_offsets = (int *)malloc(sz_m1 * sizeof(int));
    }
    if (sz_nnz <= SIZE_MAX / sizeof(int)) {
        *hA_indices = (int *)malloc(sz_nnz * sizeof(int));
    }
    if (sz_nnz <= SIZE_MAX / sizeof(float)) {
        *hA_values  = (float *)malloc(sz_nnz * sizeof(float));
    }
    if (sz_kxn <= SIZE_MAX / sizeof(float)) {
        *hB = (float *)malloc(sz_kxn * sizeof(float));
    }
    if (sz_mxn <= SIZE_MAX / sizeof(float)) {
        *hC = (float *)malloc(sz_mxn * sizeof(float));
    }
    if (sz_mxn <= SIZE_MAX / sizeof(float)) {
        *hY = (float *)calloc(sz_mxn, sizeof(float));
    }
    if (sz_mxn <= SIZE_MAX / sizeof(float)) {
        *hYBase = (float *)calloc(sz_mxn, sizeof(float));
    }
    return (*hA_offsets && *hA_indices && *hA_values &&
            *hB && *hC && *hY && *hYBase);
}

static double prepare_host_data_and_cpu_baseline(
    uint64_t M, uint64_t K, uint64_t N, uint64_t nnz, CsrDist dist,
    float alpha, float beta,
    int **hA_offsets, int **hA_indices, float **hA_values,
    float **hB, float **hC, float **hY, float **hYBase, int cpu_repeat)
{
    if (cpu_repeat <= 0) {
        cpu_repeat = 1;
    }
    if (!allocate_test_buffers(M, K, N, nnz,
            hA_offsets, hA_indices, hA_values, hB, hC, hY, hYBase)) {
        return 0.0;
    }

    if (dist == DIST_POWERLAW) {
        generate_powerlaw_csr(M, K, nnz, *hA_offsets, *hA_indices, *hA_values);
    } else {
        generate_random_csr(M, K, nnz, *hA_offsets, *hA_indices, *hA_values);
    }
    for (uint64_t i = 0; i < K * N; i++) {
        (*hB)[i] = 0.5f + 0.0001f * (i % 1000);
    }
    for (uint64_t i = 0; i < M * N; i++) {
        (*hC)[i] = 0.01f * (i % 100);
    }

    // A single host calculation is sufficient to produce the golden output.
    // Repeating it only makes the displayed host benchmark less noisy; it does
    // not increase accuracy coverage and made the default presmoke exceed its
    // time budget on wide matrices.  Keep the old ten-run benchmark available
    // explicitly for performance investigation.
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int r = 0; r < cpu_repeat; r++) {
        spmm_csr_cpu(M, K, N, *hA_offsets, *hA_indices, *hA_values,
                     *hB, *hC, *hYBase, alpha, beta);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    const double elapsed_ms = (ts1.tv_sec - ts0.tv_sec) * 1e3 +
                              (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
    if (cpu_repeat > 0) {
        return elapsed_ms / cpu_repeat;
    }
    return elapsed_ms;
}

int run_spmm_test(uint64_t M, uint64_t K, uint64_t N, uint64_t nnz,
    float alpha, float beta, CsrDist dist, aclrtStream stream, int cpu_repeat)
{
    if (cpu_repeat <= 0) {
        cpu_repeat = 1;
    }
    const char *dist_name = (dist == DIST_POWERLAW) ? "powerlaw" : "uniform";
    printf("--- Test: M=%lu K=%lu N=%lu nnz=%lu alpha=%.1f beta=%.1f dist=%s ---\n",
           M, K, N, nnz, alpha, beta, dist_name);

    if (M == 0 || K == 0 || N == 0 || nnz == 0) {
        printf("  [Skip] invalid matrix size (M=%lu K=%lu N=%lu nnz=%lu)\n", M, K, N, nnz);
        return 0;
    }

    int *hA_offsets, *hA_indices;
    float *hA_values, *hB, *hC, *hY, *hYBase;
    double cpu_ms = prepare_host_data_and_cpu_baseline(M, K, N, nnz, dist,
        alpha, beta, &hA_offsets, &hA_indices, &hA_values,
        &hB, &hC, &hY, &hYBase, cpu_repeat);

    int error_count = 0;
    double npu_ms = run_npu_spmm_benchmark(M, K, N, nnz, alpha, beta,
        hA_offsets, hA_indices, hA_values, hB, hC, hY, hYBase,
        stream, &error_count);

    double relative_time = (npu_ms > 0.0) ? (cpu_ms / npu_ms) : 0.0;
    printf("  CPU reference (%d run%s): %.3f ms  |  NPU benchmark (10 runs): %.3f ms"
           "  |  Relative time: %.2fx\n",
           cpu_repeat, cpu_repeat == 1 ? "" : "s", cpu_ms, npu_ms, relative_time);
    if (error_count > 0) {
        printf("  [Failed] test case accuracy is verification failed.\r\n\n");
    } else {
        printf("  [Success] test case accuracy is verification passed.\r\n\n");
    }

    free(hA_offsets);
    free(hA_indices);
    free(hA_values);
    free(hB);
    free(hC);
    free(hY);
    free(hYBase);

    return (error_count > 0) ? 1 : 0;
}

// One test case parameter set (M,K,N,nnz,alpha,beta,distribution)
struct TestCaseParam {
    uint64_t M, K, N, nnz;
    float alpha, beta;
    CsrDist dist;
};

static const TestCaseParam kTestCases[] = {
    {16, 16, 16, 128, 1.0f, 1.0f, DIST_UNIFORM},
    {128, 256, 32, 3277, 1.0f, 0.0f, DIST_UNIFORM},
    {256, 256, 64, 6553, 1.0f, 0.0f, DIST_UNIFORM},
    {512, 512, 128, 26214, 2.0f, 0.5f, DIST_UNIFORM},
    {1024, 1024, 256, 104858, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 256, 1048576, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 256, 1048576, 1.0f, 0.0f, DIST_POWERLAW},
    {16384, 16384, 128, 4194304, 1.0f, 0.0f, DIST_UNIFORM},
    {16384, 16384, 128, 4194304, 1.0f, 0.0f, DIST_POWERLAW},
    {65536, 65536, 64, 16777216, 1.0f, 0.0f, DIST_UNIFORM},
    {65536, 65536, 64, 16777216, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 4096, 4096, 1048576, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 4096, 1048576, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 4096, 128, 167772, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 128, 4194304, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 256, 1048576, 0.0f, 1.0f, DIST_UNIFORM},
    {4096, 4096, 256, 1048576, 2.5f, 1.5f, DIST_UNIFORM},
    {32, 32, 32, 51, 1.0f, 0.0f, DIST_UNIFORM},
    {32, 32, 32, 51, 1.0f, 0.0f, DIST_POWERLAW},
    {64, 64, 64, 204, 1.0f, 0.0f, DIST_UNIFORM},
    {64, 64, 64, 204, 1.0f, 0.0f, DIST_POWERLAW},
    {128, 128, 128, 819, 1.0f, 0.0f, DIST_UNIFORM},
    {128, 128, 128, 819, 1.0f, 0.0f, DIST_POWERLAW},
    {256, 256, 64, 3276, 1.0f, 0.0f, DIST_UNIFORM},
    {256, 256, 64, 3276, 1.0f, 0.0f, DIST_POWERLAW},
    {256, 128, 128, 1638, 1.0f, 0.0f, DIST_UNIFORM},
    {256, 128, 128, 1638, 1.0f, 0.0f, DIST_POWERLAW},
    {512, 256, 128, 6553, 1.0f, 0.0f, DIST_UNIFORM},
    {512, 256, 128, 6553, 1.0f, 0.0f, DIST_POWERLAW},
    {512, 512, 256, 13107, 1.0f, 0.0f, DIST_UNIFORM},
    {512, 512, 256, 13107, 1.0f, 0.0f, DIST_POWERLAW},
    {1024, 512, 256, 26214, 1.0f, 0.0f, DIST_UNIFORM},
    {1024, 512, 256, 26214, 1.0f, 0.0f, DIST_POWERLAW},
    {1024, 1024, 128, 52428, 1.0f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 1.0f, 0.0f, DIST_POWERLAW},
    {2048, 1024, 128, 104857, 1.0f, 0.0f, DIST_UNIFORM},
    {2048, 1024, 128, 104857, 1.0f, 0.0f, DIST_POWERLAW},
    {2048, 2048, 256, 209715, 1.0f, 0.0f, DIST_UNIFORM},
    {2048, 2048, 256, 209715, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 2048, 256, 419430, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 2048, 256, 419430, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 4096, 128, 838860, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 128, 838860, 1.0f, 0.0f, DIST_POWERLAW},
    {8192, 4096, 64, 1677721, 1.0f, 0.0f, DIST_UNIFORM},
    {8192, 4096, 64, 1677721, 1.0f, 0.0f, DIST_POWERLAW},
    {8192, 8192, 64, 3355443, 1.0f, 0.0f, DIST_UNIFORM},
    {8192, 8192, 64, 3355443, 1.0f, 0.0f, DIST_POWERLAW},
    {16384, 4096, 32, 3355443, 1.0f, 0.0f, DIST_UNIFORM},
    {16384, 4096, 32, 3355443, 1.0f, 0.0f, DIST_POWERLAW},
    {16384, 8192, 32, 6710886, 1.0f, 0.0f, DIST_UNIFORM},
    {16384, 8192, 32, 6710886, 1.0f, 0.0f, DIST_POWERLAW},
    {32768, 4096, 16, 6710886, 1.0f, 0.0f, DIST_UNIFORM},
    {32768, 4096, 16, 6710886, 1.0f, 0.0f, DIST_POWERLAW},
    {32768, 8192, 16, 13421772, 1.0f, 0.0f, DIST_UNIFORM},
    {32768, 8192, 16, 13421772, 1.0f, 0.0f, DIST_POWERLAW},
    {31, 31, 31, 48, 1.0f, 0.0f, DIST_UNIFORM},
    {31, 31, 31, 48, 1.0f, 0.0f, DIST_POWERLAW},
    {63, 63, 63, 198, 1.0f, 0.0f, DIST_UNIFORM},
    {63, 63, 63, 198, 1.0f, 0.0f, DIST_POWERLAW},
    {127, 127, 127, 806, 1.0f, 0.0f, DIST_UNIFORM},
    {127, 127, 127, 806, 1.0f, 0.0f, DIST_POWERLAW},
    {255, 127, 63, 1619, 1.0f, 0.0f, DIST_UNIFORM},
    {255, 127, 63, 1619, 1.0f, 0.0f, DIST_POWERLAW},
    {255, 255, 127, 3251, 1.0f, 0.0f, DIST_UNIFORM},
    {255, 255, 127, 3251, 1.0f, 0.0f, DIST_POWERLAW},
    {511, 255, 127, 6515, 1.0f, 0.0f, DIST_UNIFORM},
    {511, 255, 127, 6515, 1.0f, 0.0f, DIST_POWERLAW},
    {511, 511, 255, 13056, 1.0f, 0.0f, DIST_UNIFORM},
    {511, 511, 255, 13056, 1.0f, 0.0f, DIST_POWERLAW},
    {1023, 511, 255, 26137, 1.0f, 0.0f, DIST_UNIFORM},
    {1023, 511, 255, 26137, 1.0f, 0.0f, DIST_POWERLAW},
    {1023, 1023, 127, 52326, 1.0f, 0.0f, DIST_UNIFORM},
    {1023, 1023, 127, 52326, 1.0f, 0.0f, DIST_POWERLAW},
    {2047, 1023, 255, 104704, 1.0f, 0.0f, DIST_UNIFORM},
    {2047, 1023, 255, 104704, 1.0f, 0.0f, DIST_POWERLAW},
    {4095, 2047, 255, 419123, 1.0f, 0.0f, DIST_UNIFORM},
    {4095, 2047, 255, 419123, 1.0f, 0.0f, DIST_POWERLAW},
    {4095, 4095, 127, 838451, 1.0f, 0.0f, DIST_UNIFORM},
    {4095, 4095, 127, 838451, 1.0f, 0.0f, DIST_POWERLAW},
    {8191, 4095, 63, 1677107, 1.0f, 0.0f, DIST_UNIFORM},
    {8191, 4095, 63, 1677107, 1.0f, 0.0f, DIST_POWERLAW},
    {16383, 4095, 63, 3354419, 1.0f, 0.0f, DIST_UNIFORM},
    {16383, 4095, 63, 3354419, 1.0f, 0.0f, DIST_POWERLAW},
    {32767, 8191, 31, 13419724, 1.0f, 0.0f, DIST_UNIFORM},
    {32767, 8191, 31, 13419724, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 4096, 256, 16777, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 256, 16777, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 4096, 256, 167772, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 256, 167772, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 4096, 256, 335544, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 256, 335544, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 4096, 256, 838860, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 256, 838860, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 4096, 256, 1677721, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 256, 1677721, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 4096, 256, 4194304, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 256, 4194304, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 4096, 256, 8388608, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 256, 8388608, 1.0f, 0.0f, DIST_POWERLAW},
    {1024, 1024, 128, 52428, 0.0f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 0.0f, 1.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 1.0f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 1.0f, 1.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, -1.0f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, -1.0f, 1.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 1.0f, -1.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 0.5f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 0.5f, 1.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 2.0f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 2.0f, 0.5f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 2.5f, 1.5f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, -0.5f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, -2.0f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 0.0f, -1.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 0.0f, 2.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 1.5f, 0.5f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 3.0f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, -3.0f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 128, 52428, 0.0f, -2.0f, DIST_UNIFORM},
    {2048, 2048, 512, 83886, 1.0f, 0.0f, DIST_UNIFORM},
    {2048, 2048, 512, 83886, 1.0f, 0.0f, DIST_POWERLAW},
    {1, 1, 1, 1, 1.0f, 0.0f, DIST_UNIFORM},
    {1, 16, 1, 8, 1.0f, 0.0f, DIST_UNIFORM},
    {1, 64, 64, 32, 1.0f, 0.0f, DIST_UNIFORM},
    {64, 1, 64, 32, 1.0f, 0.0f, DIST_UNIFORM},
    {64, 64, 1, 32, 1.0f, 0.0f, DIST_UNIFORM},
    {2, 64, 64, 32, 1.0f, 0.0f, DIST_UNIFORM},
    {65536, 16, 16, 524288, 1.0f, 0.0f, DIST_UNIFORM},
    {256, 256, 64, 65536, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 4096, 128, 1, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 128, 10, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 4096, 64, 8388608, 1.0f, 0.0f, DIST_UNIFORM},
    {2048, 2048, 32, 3774873, 1.0f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 64, 32768, -1.0f, -1.0f, DIST_UNIFORM},
    {1024, 1024, 64, 32768, -2.5f, 0.5f, DIST_UNIFORM},
    {256, 512, 32, 1310, 1.0f, 0.0f, DIST_UNIFORM},
    {512, 1024, 64, 10485, 0.0f, 1.0f, DIST_UNIFORM},
    {1024, 2048, 128, 20971, 0.0f, 0.0f, DIST_POWERLAW},
    {2048, 4096, 256, 83886, 0.5f, 1.0f, DIST_UNIFORM},
    {512, 2048, 32, 10485, 2.0f, 0.0f, DIST_UNIFORM},
    {1024, 4096, 64, 419430, 0.0f, 0.0f, DIST_POWERLAW},
    {2048, 8192, 32, 1677721, 1.0f, 0.0f, DIST_POWERLAW},
    {128, 1024, 16, 2621, 0.0f, 0.0f, DIST_POWERLAW},
    {256, 2048, 16, 52428, 1.0f, 0.5f, DIST_UNIFORM},
    {512, 4096, 16, 104857, 2.0f, 1.0f, DIST_UNIFORM},
    {256, 256, 64, 655, 0.0f, 0.0f, DIST_UNIFORM},
    {1024, 1024, 128, 10485, 0.0f, 2.0f, DIST_UNIFORM},
    {4096, 4096, 256, 167772, 0.0f, 0.0f, DIST_UNIFORM},
    {128, 512, 32, 655, 0.0f, 1.0f, DIST_UNIFORM},
    {2048, 4096, 64, 83886, 0.0f, 1.0f, DIST_UNIFORM},
    {1023, 1024, 32, 1047, 3.2f, 2.9f, DIST_UNIFORM},
    {2048, 128, 1024, 5242, 2.1f, -0.8f, DIST_POWERLAW},
    {32768, 32768, 2, 268435456, 2.6f, 1.1f, DIST_POWERLAW},
    {64, 2047, 64, 13100, -1.7f, -0.7f, DIST_UNIFORM},
    {4095, 16383, 32, 16772096, 3.7f, 1.7f, DIST_POWERLAW},
    {2048, 32, 512, 3276, -2.5f, 2.5f, DIST_POWERLAW},
    {1023, 16383, 128, 167598, 0.2f, 2.5f, DIST_POWERLAW},
    {63, 256, 256, 161, 3.0f, 0.2f, DIST_UNIFORM},
    {32767, 1023, 64, 3352064, -0.1f, 3.0f, DIST_POWERLAW},
    {256, 255, 2048, 65, 3.9f, -2.1f, DIST_UNIFORM},
    {512, 8191, 512, 419379, 0.1f, 0.6f, DIST_UNIFORM},
    {63, 255, 256, 1606, 2.4f, -2.3f, DIST_UNIFORM},
    {511, 4096, 512, 2093, 0.5f, -0.3f, DIST_POWERLAW},
    {32767, 32767, 2, 107367628, 1.1f, -2.4f, DIST_UNIFORM},
    {4095, 8192, 64, 335462, 3.1f, 0.2f, DIST_POWERLAW},
    {255, 16, 512, 204, -2.1f, -0.8f, DIST_UNIFORM},
    {8192, 2048, 16, 167772, -2.3f, -0.1f, DIST_UNIFORM},
    {64, 511, 64, 327, 4.6f, -2.0f, DIST_POWERLAW},
    {255, 2047, 1024, 5219, 2.7f, -0.6f, DIST_UNIFORM},
    {8191, 4095, 64, 1677107, -2.0f, -1.7f, DIST_POWERLAW},
    {16384, 16, 128, 26214, -2.9f, 1.2f, DIST_UNIFORM},
    {32, 2048, 32, 65, -2.4f, -1.6f, DIST_POWERLAW},
    {8191, 127, 128, 104025, 2.8f, 2.3f, DIST_UNIFORM},
    {1023, 127, 128, 64960, 3.5f, -1.9f, DIST_POWERLAW},
    {128, 8191, 1024, 20968, 0.3f, 2.2f, DIST_POWERLAW},
    {32, 8191, 32, 262, 2.8f, 1.8f, DIST_POWERLAW},
    {128, 2048, 128, 2621, -1.9f, -1.9f, DIST_POWERLAW},
    {63, 2048, 32, 6451, -2.6f, 3.0f, DIST_UNIFORM},
    {16, 64, 128, 10, 0.9f, -1.7f, DIST_POWERLAW},
    {15, 32, 64, 24, 4.9f, -1.4f, DIST_UNIFORM},
    {63, 8192, 1024, 129024, -1.8f, -1.2f, DIST_POWERLAW},
    {32, 1023, 16, 8184, -2.5f, 0.5f, DIST_POWERLAW},
    {255, 255, 64, 65, 3.8f, -2.6f, DIST_UNIFORM},
    {64, 8191, 128, 26211, 4.5f, 0.4f, DIST_UNIFORM},
    {1023, 2047, 16, 209408, 0.4f, 0.5f, DIST_UNIFORM},
    {255, 16384, 256, 41779, -1.1f, -0.6f, DIST_POWERLAW},
    {8191, 4095, 64, 1677107, 4.4f, 2.6f, DIST_POWERLAW},
    {16, 63, 32, 1, 1.0f, -2.2f, DIST_UNIFORM},
    {32768, 64, 128, 41943, -1.7f, 2.0f, DIST_POWERLAW},
    {16383, 8192, 16, 33552384, 4.5f, -2.4f, DIST_POWERLAW},
    {256, 4096, 32, 1048, -0.8f, 0.6f, DIST_UNIFORM},
    {16383, 16384, 8, 67104768, 1.0f, -1.5f, DIST_POWERLAW},
    {32, 64, 1024, 1024, -2.6f, -1.0f, DIST_POWERLAW},
    {256, 4095, 256, 10483, 1.4f, -0.4f, DIST_POWERLAW},
    {16, 128, 32, 512, 1.4f, 2.0f, DIST_UNIFORM},
    {1023, 511, 64, 26137, -2.7f, -0.8f, DIST_UNIFORM},
    {32, 32768, 128, 262144, 2.3f, -0.9f, DIST_UNIFORM},
    {511, 31, 64, 158, 4.8f, 1.9f, DIST_UNIFORM},
    {31, 16, 64, 124, 3.3f, -0.5f, DIST_POWERLAW},
    {8191, 32767, 8, 5367889, 3.3f, -2.4f, DIST_UNIFORM},
};

// The presmoke suite uses small functional shapes so its CPU golden calculation
// is suitable for presmoke. It still covers both CSR distributions, scalar
// coefficients, non-aligned dimensions, empty-row inputs and a dense-output
// pressure shape. Ordinary build.sh runs execute kTestCases at full scale;
// only SPARSE_TEST_PRESMOKE=1 selects this reduced suite.
static const TestCaseParam kPresmokeCases[] = {
    // Small and scalar boundaries.
    {16, 16, 16, 128, 1.0f, 1.0f, DIST_UNIFORM},
    {1, 1, 1, 1, 1.0f, 0.0f, DIST_UNIFORM},
    {1, 16, 1, 8, 1.0f, 0.0f, DIST_UNIFORM},
    {64, 64, 1, 32, 1.0f, 0.0f, DIST_UNIFORM},

    // Coefficients and non-square / non-aligned shapes.
    {256, 256, 64, 6553, 2.0f, 0.5f, DIST_UNIFORM},
    {256, 256, 64, 6553, 0.0f, 1.0f, DIST_UNIFORM},
    {256, 256, 64, 6553, -1.0f, 1.0f, DIST_UNIFORM},
    {256, 256, 64, 6553, 2.5f, 1.5f, DIST_UNIFORM},
    {256, 128, 128, 1638, 1.0f, 0.0f, DIST_UNIFORM},
    {256, 128, 128, 1638, 1.0f, 0.0f, DIST_POWERLAW},
    {255, 127, 63, 1619, 1.0f, 0.0f, DIST_UNIFORM},
    {255, 127, 63, 1619, 1.0f, 0.0f, DIST_POWERLAW},
    {255, 128, 255, 655, 2.1f, -0.8f, DIST_POWERLAW},
    {511, 512, 32, 523, 3.2f, 2.9f, DIST_UNIFORM},
    {512, 512, 128, 1, 1.0f, 0.0f, DIST_UNIFORM},
    {512, 512, 128, 10, 1.0f, 0.0f, DIST_UNIFORM},

    // Representative small pressure cases: both distributions, wide output,
    // high row count and dense-per-row work without the former CPU stress.
    {512, 512, 128, 26214, 1.0f, 0.0f, DIST_UNIFORM},
    {512, 512, 128, 26214, 1.0f, 0.0f, DIST_POWERLAW},
    {4096, 256, 64, 52428, 1.0f, 0.0f, DIST_UNIFORM},
    {4096, 256, 64, 52428, 1.0f, 0.0f, DIST_POWERLAW},
    {512, 512, 512, 13107, 1.0f, 0.0f, DIST_UNIFORM},
    {512, 512, 512, 13107, 1.0f, 0.0f, DIST_POWERLAW},
    {512, 512, 128, 131072, 1.0f, 0.0f, DIST_UNIFORM},
    {512, 512, 128, 131072, 1.0f, 0.0f, DIST_POWERLAW},
};

static int run_test_cases(const TestCaseParam *test_cases, size_t case_count,
    const char *suite_name, aclrtStream stream, int cpu_repeat)
{
    printf("========== SpMM %s suite: %zu cases ==========\n", suite_name, case_count);
    int fail = 0;
    for (size_t i = 0; i < case_count; i++) {
        const TestCaseParam &tc = test_cases[i];
        fail += run_spmm_test(tc.M, tc.K, tc.N, tc.nnz,
            tc.alpha, tc.beta, tc.dist, stream, cpu_repeat);
    }
    return fail;
}

// Zero-dim guard verification: when any of M/K/N is 0, ValidateSpmmInputs must
// return INVALID_VALUE (dimension matching passes 0==0, the zero-dim guard on
// matA->rows/cols and matC->cols rejects). Only calls GetBufferSize, never
// executes the kernel. Dimension relations: A is M×K CSR, B is K×n DnMat (ROW),
// C is m×N DnMat (ROW); the dimension matching in AclsparseValidateSpmmCommon
// (sparse/common/aclsparse_descr_internal.h) requires A.cols==B.rows(K),
// A.rows==C.rows(M), B.cols==C.cols(N).
// Descriptor-layer constraints: aclsparseCreateDnMat accepts 0-dim (ld>0,
// ROW order ld>=cols, non-null values >=1 element); aclsparseCreateCsr accepts
// nnz=0 (colInd/values may be null, rowOffsets stays non-null: 1 element [0]
// when M=0, else M+1 zeros).
static int run_spmm_zero_dim_case(const char *name, uint64_t M, uint64_t K, uint64_t N,
    aclrtStream stream)
{
    printf("--- Zero-dim guard case: %s (M=%lu K=%lu N=%lu) ---\n", name, M, K, N);

    int *dA_offsets = NULL;
    float *dB = NULL;
    float *dC = NULL;
    aclsparseHandle_t handle = NULL;
    aclsparseSpMatDescr_t matA = NULL;
    aclsparseDnMatDescr_t matB = NULL;
    aclsparseDnMatDescr_t matC = NULL;

    // CSR A: M×K, nnz=0, rowOffsets non-null
    std::vector<int> hOffsets((size_t)M + 1, 0);
    CHECK_ACL(aclrtMalloc((void **)&dA_offsets, hOffsets.size() * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMemcpy(dA_offsets, hOffsets.size() * sizeof(int),
        hOffsets.data(), hOffsets.size() * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE))

    // B: K×N ROW order, C: M×N ROW order; ld = (N>0 ? N : 1);
    // values must be non-null (>=1 element) even for 0-dim matrices
    const uint64_t ld = (N > 0) ? N : 1;
    const size_t bElems = (size_t)(K > 0 ? K : 1) * (size_t)ld;
    const size_t cElems = (size_t)(M > 0 ? M : 1) * (size_t)ld;
    CHECK_ACL(aclrtMalloc((void **)&dB, bElems * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&dC, cElems * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST))

    CHECK_ACL_SPARSE(aclsparseCreate(&handle))
    CHECK_ACL_SPARSE(aclsparseSetStream(handle, stream))
    CHECK_ACL_SPARSE(aclsparseCreateCsr(&matA, (int64_t)M, (int64_t)K, 0,
        dA_offsets, NULL, NULL,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT))
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(&matB, (int64_t)K, (int64_t)N, (int64_t)ld,
        dB, ACL_FLOAT, ACL_SPARSE_ORDER_ROW))
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(&matC, (int64_t)M, (int64_t)N, (int64_t)ld,
        dC, ACL_FLOAT, ACL_SPARSE_ORDER_ROW))

    // dtype combo (all FP32) and dimension matching (0==0 passes) are valid,
    // so INVALID_VALUE can only come from the zero-dim guard.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    size_t bufferSize = 0;
    aclsparseStatus_t ret = aclsparseSpMMGetBufferSize(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    const int pass = (ret == ACL_SPARSE_STATUS_INVALID_VALUE);
    printf("  [%s] GetBufferSize ret=%d (expect %d INVALID_VALUE): %s\n",
        name, (int)ret, (int)ACL_SPARSE_STATUS_INVALID_VALUE,
        pass ? "PASS" : "FAIL");

    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(matB))
    CHECK_ACL_SPARSE(aclsparseDestroyDnMat(matC))
    CHECK_ACL_SPARSE(aclsparseDestroySpMat(matA))
    CHECK_ACL_SPARSE(aclsparseDestroy(handle))
    CHECK_ACL(aclrtFree(dA_offsets))
    CHECK_ACL(aclrtFree(dB))
    CHECK_ACL(aclrtFree(dC))
    return pass ? 0 : 1;
}

// Read the process startup environment before ACL can create worker threads.
// Pass the resulting values down explicitly; no runtime getenv calls are needed.
static bool startup_option_enabled(char *const envp[], const char *name)
{
    if (envp == nullptr) {
        return false;
    }
    const size_t name_length = strlen(name);
    for (size_t i = 0; envp[i] != nullptr; ++i) {
        if (strncmp(envp[i], name, name_length) == 0 && envp[i][name_length] == '=') {
            return strcmp(envp[i] + name_length + 1, "1") == 0;
        }
    }
    return false;
}

int main(int, char **, char **envp)
{
    const bool full_suite = !startup_option_enabled(envp, "SPARSE_TEST_PRESMOKE");
    const int cpu_repeat = full_suite && startup_option_enabled(envp, "SPMM_TEST_CPU_BENCHMARK") ? 10 : 1;
    int32_t deviceId = 0;
    aclrtStream stream = 0;
    CHECK_ACL(aclInit(NULL))
    CHECK_ACL(aclrtSetDevice(deviceId))
    CHECK_ACL(aclrtCreateStream(&stream))

    int fail = full_suite
        ? run_test_cases(kTestCases, sizeof(kTestCases) / sizeof(kTestCases[0]), "full", stream, cpu_repeat)
        : run_test_cases(kPresmokeCases, sizeof(kPresmokeCases) / sizeof(kPresmokeCases[0]),
            "presmoke", stream, cpu_repeat);
    fail += run_spmm_zero_dim_case("zero_dim_m0", 0, 16, 8, stream);
    fail += run_spmm_zero_dim_case("zero_dim_k0", 8, 0, 8, stream);
    fail += run_spmm_zero_dim_case("zero_dim_n0", 8, 16, 0, stream);

    printf("========================================\n");
    printf("Total: %d failed\n", fail);
    printf("========================================\n");

    CHECK_ACL(aclrtDestroyStream(stream))
    CHECK_ACL(aclrtResetDevice(deviceId))
    CHECK_ACL(aclFinalize())
    return fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
