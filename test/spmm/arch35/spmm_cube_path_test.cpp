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

#include <vector>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include <acl/acl.h>
#include <cann_ops_sparse.h>

#include "spmm_config.h"
#include "spmm_tiling_wire.h"
#include "spmm_test_utils.h"

namespace {

using spmm_test::CheckAcl;
using spmm_test::CheckSparse;
using spmm_test::FloatToHalf;
using spmm_test::HalfToFloat;

struct CaseConfig {
    const char *name;
    int32_t m;
    bool betaOne;
    bool usePreprocess;
    int32_t repeats;
    bool forceWideSpan;
    bool duplicateColumn;
    bool descendingColumns;
    bool emptyTailGroup;
    int32_t columnDelta;
    bool contiguousSeven;
    bool expectCube;
    aclsparseSpMMAlg_t alg;
    bool splitWideGroups = false;
    bool expectFallback = false;
    int32_t expectedCubeRows = 0;
    int32_t expectedFallbackCount = -1;
};

struct PathHostData {
    int32_t k = 0;
    int32_t n = 128;
    std::vector<int32_t> rows;
    std::vector<int32_t> cols;
    std::vector<float> aFloat;
    std::vector<uint16_t> a;
    std::vector<uint16_t> b;
    std::vector<uint16_t> c;
    std::vector<float> expected;
};

struct PathDeviceData {
    void *dRows = nullptr;
    void *dCols = nullptr;
    void *dA = nullptr;
    void *dB = nullptr;
    void *dC = nullptr;
    void *workspace = nullptr;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
};

void BuildPathPattern(const CaseConfig &cfg, PathHostData *host)
{
    host->k = cfg.splitWideGroups ? 128 : (cfg.contiguousSeven ? cfg.m + 7 : 64);
    host->rows.assign(static_cast<size_t>(cfg.m) + 1, 0);
    for (int32_t row = 0; row < cfg.m; ++row) {
        if (!cfg.emptyTailGroup || row == 0) {
            const int32_t columnVariation = cfg.columnDelta == 1 ? 15 : 16;
            int32_t col0 = 8 + row % columnVariation;
            int32_t col1 = col0 + cfg.columnDelta;
            if (cfg.splitWideGroups) {
                // Keep the two row bands disjoint so the packed span exceeds K=48.
                col0 = row < cfg.m / 2 ? row % 8 : 64 + (row % 8);
                col1 = col0 + 1;
            } else if (cfg.forceWideSpan && row == 0) {
                col0 = 0;
                col1 = 48;
            } else if (cfg.duplicateColumn && row == 0) {
                col0 = 7;
                col1 = 7;
            } else if (cfg.descendingColumns && row == 0) {
                col0 = 9;
                col1 = 5;
            }
            if (cfg.contiguousSeven) {
                for (int32_t offset = 0; offset < 7; ++offset) {
                    host->cols.push_back(row + offset);
                    host->aFloat.push_back(1.0f);
                }
            } else {
                host->cols.push_back(col0);
                host->cols.push_back(col1);
                host->aFloat.push_back(0.0625f * static_cast<float>((row % 5) + 1));
                host->aFloat.push_back(-0.03125f * static_cast<float>((row % 7) + 1));
            }
        }
        host->rows[static_cast<size_t>(row) + 1] =
            static_cast<int32_t>(host->cols.size());
    }
}

void QuantizePathInputs(const CaseConfig &cfg, PathHostData *host,
                        std::vector<float> *bFloat)
{
    host->a.resize(host->aFloat.size());
    for (size_t i = 0; i < host->a.size(); ++i) {
        host->a[i] = FloatToHalf(host->aFloat[i]);
        host->aFloat[i] = HalfToFloat(host->a[i]);
    }
    bFloat->assign(static_cast<size_t>(host->k) * host->n, 0.0f);
    host->b.resize(bFloat->size());
    for (size_t i = 0; i < host->b.size(); ++i) {
        (*bFloat)[i] = 0.015625f * static_cast<float>(static_cast<int32_t>(i % 17) - 8);
        host->b[i] = FloatToHalf((*bFloat)[i]);
        (*bFloat)[i] = HalfToFloat(host->b[i]);
    }
    host->c.assign(static_cast<size_t>(cfg.m) * host->n,
        FloatToHalf(cfg.betaOne ? 0.25f : 0.0f));
}

void BuildPathExpected(const CaseConfig &cfg, const PathHostData &host,
                       const std::vector<float> &bFloat, std::vector<float> *expected)
{
    for (int32_t repeat = 0; repeat < cfg.repeats; ++repeat) {
        for (int32_t row = 0; row < cfg.m; ++row) {
            for (int32_t col = 0; col < host.n; ++col) {
                float sum = 0.0f;
                for (int32_t p = host.rows[row]; p < host.rows[row + 1]; ++p) {
                    sum += host.aFloat[p] * bFloat[static_cast<size_t>(host.cols[p]) * host.n + col];
                }
                const size_t index = static_cast<size_t>(row) * host.n + col;
                const float result = sum + (cfg.betaOne ? (*expected)[index] : 0.0f);
                (*expected)[index] = HalfToFloat(FloatToHalf(result));
            }
        }
    }
}

void BuildPathHostData(const CaseConfig &cfg, PathHostData *host)
{
    BuildPathPattern(cfg, host);
    std::vector<float> bFloat;
    QuantizePathInputs(cfg, host, &bFloat);
    host->expected.assign(host->c.size(), cfg.betaOne ? 0.25f : 0.0f);
    BuildPathExpected(cfg, *host, bFloat, &host->expected);
}

bool AllocatePathResources(aclrtStream stream, const CaseConfig &cfg,
                           const PathHostData &host, PathDeviceData *device)
{
    bool pass = CheckAcl(aclrtMalloc(&device->dRows, host.rows.size() * sizeof(int32_t),
        ACL_MEM_MALLOC_HUGE_FIRST), "malloc rows") &&
        CheckAcl(aclrtMalloc(&device->dCols, host.cols.size() * sizeof(int32_t),
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc cols") &&
        CheckAcl(aclrtMalloc(&device->dA, host.a.size() * sizeof(uint16_t),
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc A") &&
        CheckAcl(aclrtMalloc(&device->dB, host.b.size() * sizeof(uint16_t),
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc B") &&
        CheckAcl(aclrtMalloc(&device->dC, host.c.size() * sizeof(uint16_t),
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc C");
    if (pass) {
        pass = CheckAcl(aclrtMemcpy(device->dRows, host.rows.size() * sizeof(int32_t),
            host.rows.data(), host.rows.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy rows") &&
            CheckAcl(aclrtMemcpy(device->dCols, host.cols.size() * sizeof(int32_t),
                host.cols.data(), host.cols.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy cols") &&
            CheckAcl(aclrtMemcpy(device->dA, host.a.size() * sizeof(uint16_t),
                host.a.data(), host.a.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy A") &&
            CheckAcl(aclrtMemcpy(device->dB, host.b.size() * sizeof(uint16_t),
                host.b.data(), host.b.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy B") &&
            CheckAcl(aclrtMemcpy(device->dC, host.c.size() * sizeof(uint16_t),
                host.c.data(), host.c.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy C");
    }
    if (pass) {
        pass = CheckSparse(aclsparseCreate(&device->handle), "create handle") &&
            CheckSparse(aclsparseSetStream(device->handle, stream), "set stream") &&
            CheckSparse(aclsparseCreateCsr(&device->matA, cfg.m, host.k, host.cols.size(),
                device->dRows, device->dCols, device->dA, ACL_SPARSE_INDEX_32I,
                ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT16), "create CSR") &&
            CheckSparse(aclsparseCreateDnMat(&device->matB, host.k, host.n, host.n,
                device->dB, ACL_FLOAT16, ACL_SPARSE_ORDER_ROW), "create B") &&
            CheckSparse(aclsparseCreateDnMat(&device->matC, cfg.m, host.n, host.n,
                device->dC, ACL_FLOAT16, ACL_SPARSE_ORDER_ROW), "create C");
    }
    return pass;
}

bool ExecutePathCase(aclrtStream stream, const CaseConfig &cfg,
                     const PathHostData &host, PathDeviceData *device,
                     SpmmTilingWireV1 *tiling, size_t *workspaceSize)
{
    const float alpha = 1.0f;
    const float beta = cfg.betaOne ? 1.0f : 0.0f;
    bool pass = CheckSparse(aclsparseSpMMGetBufferSize(device->handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha,
        device->matA, device->matB, &beta, device->matC, ACL_FLOAT, cfg.alg,
        workspaceSize), "get buffer size") &&
        CheckAcl(aclrtMalloc(&device->workspace, *workspaceSize,
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
    if (pass && cfg.usePreprocess) {
        pass = CheckSparse(aclsparseSpMMPreprocess(device->handle,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha,
            device->matA, device->matB, &beta, device->matC, ACL_FLOAT,
            cfg.alg, device->workspace), "preprocess");
    }
    for (int32_t repeat = 0; pass && repeat < cfg.repeats; ++repeat) {
        pass = CheckSparse(aclsparseSpMM(device->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, device->matA, device->matB,
            &beta, device->matC, ACL_FLOAT, cfg.alg, device->workspace), "execute");
    }
    pass = pass && CheckAcl(aclrtSynchronizeStream(stream), "synchronize");
    if (pass) {
        pass = CheckAcl(aclrtMemcpy(tiling, sizeof(*tiling),
            static_cast<uint8_t *>(device->workspace) + SPMM_WS_HEADER_BYTES,
            sizeof(*tiling), ACL_MEMCPY_DEVICE_TO_HOST), "copy tiling");
    }
    return pass;
}

bool ValidatePathTiling(const CaseConfig &cfg, const SpmmTilingWireV1 &tiling)
{
    const bool actualCube = tiling.cube_mode != 0;
    if (actualCube != cfg.expectCube) {
        std::printf("%s dispatch mismatch: expected_cube=%d actual_cube=%d\n",
            cfg.name, cfg.expectCube, actualCube);
        return false;
    }
    if (cfg.expectedCubeRows > 0 &&
        (tiling.cube_mode == 0 || tiling.cube_rows != cfg.expectedCubeRows)) {
        std::printf("%s grouping mismatch: expected_rows=%d actual_rows=%d\n",
            cfg.name, cfg.expectedCubeRows, tiling.cube_rows);
        return false;
    }
    if (cfg.expectFallback && tiling.cube_complex_fallback_count <= 0) {
        std::printf("%s fallback mismatch: expected positive count, actual=%d\n",
            cfg.name, tiling.cube_complex_fallback_count);
        return false;
    }
    if (cfg.expectedFallbackCount >= 0 &&
        tiling.cube_complex_fallback_count != cfg.expectedFallbackCount) {
        std::printf("%s fallback mismatch: expected=%d actual=%d\n",
            cfg.name, cfg.expectedFallbackCount,
            tiling.cube_complex_fallback_count);
        return false;
    }
    return true;
}

bool VerifyPathResult(const CaseConfig &cfg, const PathHostData &host,
                      float *maxError)
{
    int32_t reported = 0;
    std::vector<int32_t> groupErrors((static_cast<size_t>(cfg.m) + SPMM_CUBE_ROWS - 1) /
        SPMM_CUBE_ROWS, 0);
    std::vector<int32_t> columnErrors(host.n, 0);
    for (size_t i = 0; i < host.c.size(); ++i) {
        const float actual = HalfToFloat(host.c[i]);
        const float error = std::fabs(actual - host.expected[i]);
        *maxError = std::max(*maxError, error);
        if (error > 0.001953125f && reported < 8) {
            std::printf("%s mismatch[%zu]: expected=%g actual=%g error=%g\n",
                cfg.name, i, host.expected[i], actual, error);
            ++reported;
        }
        if (error > 0.001953125f) {
            ++groupErrors[(i / host.n) / SPMM_CUBE_ROWS];
            ++columnErrors[i % host.n];
        }
    }
    if (reported != 0) {
        std::printf("%s failing_groups:", cfg.name);
        for (size_t i = 0; i < groupErrors.size(); ++i) {
            if (groupErrors[i] != 0) std::printf(" %zu:%d", i, groupErrors[i]);
        }
        std::printf("\n%s failing_columns:", cfg.name);
        for (size_t i = 0; i < columnErrors.size(); ++i) {
            if (columnErrors[i] != 0) std::printf(" %zu:%d", i, columnErrors[i]);
        }
        std::printf("\n");
    }
    return *maxError <= 0.001953125f;
}

void ReleasePathResources(PathDeviceData *device)
{
    if (device->matB != nullptr) aclsparseDestroyDnMat(device->matB);
    if (device->matC != nullptr) aclsparseDestroyDnMat(device->matC);
    if (device->matA != nullptr) aclsparseDestroySpMat(device->matA);
    if (device->handle != nullptr) aclsparseDestroy(device->handle);
    if (device->workspace != nullptr) aclrtFree(device->workspace);
    if (device->dRows != nullptr) aclrtFree(device->dRows);
    if (device->dCols != nullptr) aclrtFree(device->dCols);
    if (device->dA != nullptr) aclrtFree(device->dA);
    if (device->dB != nullptr) aclrtFree(device->dB);
    if (device->dC != nullptr) aclrtFree(device->dC);
}

bool RunCase(aclrtStream stream, const CaseConfig &cfg)
{
    PathHostData host;
    BuildPathHostData(cfg, &host);
    PathDeviceData device;
    bool pass = AllocatePathResources(stream, cfg, host, &device);
    SpmmTilingWireV1 tiling{};
    size_t workspaceSize = 0;
    if (pass) {
        pass = ExecutePathCase(stream, cfg, host, &device, &tiling, &workspaceSize);
    }
    const bool actualCube = tiling.cube_mode != 0;
    if (pass) pass = ValidatePathTiling(cfg, tiling);
    if (pass) {
        pass = CheckAcl(aclrtMemcpy(host.c.data(), host.c.size() * sizeof(uint16_t),
            device.dC, host.c.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "copy C");
    }
    float maxError = 0.0f;
    if (pass) pass = VerifyPathResult(cfg, host, &maxError);
    std::printf("%s: %s cube=%d max_error=%g workspace=%zu\n", cfg.name,
                pass ? "PASS" : "FAIL", actualCube, maxError, workspaceSize);
    ReleasePathResources(&device);
    return pass;
}

} // namespace

static bool RunCubePathTests(aclrtStream stream)
{
    const CaseConfig cases[] = {
        {"CUBE_K16", 32, false, true, 1, false, false, false, false, 1, false, true, ACL_SPARSE_SPMM_CSR_ALG1},
        {"CUBE_K32", 32, false, true, 1, false, false, false, false, 16, false, true, ACL_SPARSE_SPMM_CSR_ALG1},
        {"FALLBACK_K39_CONTIGUOUS7", 32, false, true, 1, false, false, false, false, 32, true, false, ACL_SPARSE_SPMM_CSR_ALG1},
        {"CUBE_57_GROUPS_TAIL_FALLBACK_CONTIGUOUS7", 1856, false, true, 1, false, false, false, false, 32, true, true, ACL_SPARSE_SPMM_CSR_ALG1, false, true, SPMM_CUBE_ROWS, 32},
        {"CUBE_K48_FULL32_BETA0_ALG1", 32, false, true, 1, false, false, false, false, 32, false, true, ACL_SPARSE_SPMM_CSR_ALG1},
        {"CUBE_K48_TAIL35_BETA1_REPEAT_ALG2", 35, true, true, 2, false, false, false, false, 32, false, true, ACL_SPARSE_SPMM_CSR_ALG2, false, false, SPMM_CUBE_ROWS, 0},
        {"CUBE_K48_LAZY_PREPROCESS_ALG3", 33, false, false, 1, false, false, false, false, 32, false, true, ACL_SPARSE_SPMM_CSR_ALG3, false, false, SPMM_CUBE_ROWS, 0},
        {"CUBE_16_ROW_FALLBACK", 32, false, true, 1, false, false, false, false, 1, false, true, ACL_SPARSE_SPMM_CSR_ALG1, true, false, SPMM_CUBE_MIN_ROWS},
        {"CUBE_PARTIAL_ROW_FALLBACK", 336, true, true, 2, true, false, false, false, 32, false, true, ACL_SPARSE_SPMM_CSR_ALG2, false, true},
        {"FALLBACK_SPAN49", 32, false, true, 1, true, false, false, false, 32, false, true, ACL_SPARSE_SPMM_CSR_ALG1, false, true, 0, 1},
        {"FALLBACK_DUPLICATE_COLUMN", 32, false, true, 1, false, true, false, false, 32, false, true, ACL_SPARSE_SPMM_CSR_ALG1, false, true, 0, 1},
        {"FALLBACK_DESCENDING_COLUMNS", 32, false, true, 1, false, false, true, false, 32, false, true, ACL_SPARSE_SPMM_CSR_ALG1, false, true, 0, 1},
        {"FALLBACK_EMPTY_GROUP", 65, false, true, 1, false, false, false, true, 32, false, true, ACL_SPARSE_SPMM_CSR_ALG1, false, true, 0, 64},
    };
    std::printf("##### A5_SPMM_CUBE_PATH_START #####\n");
    bool pass = true;
    for (const auto &testCase : cases) pass = RunCase(stream, testCase) && pass;
    std::printf("##### A5_SPMM_CUBE_PATH_RESULT #####\nOverall: %s\n", pass ? "PASS" : "FAIL");

    return pass;
}

using spmm_test::SpmmDeviceTest;

TEST_F(SpmmDeviceTest, CubeDispatch)
{
    ASSERT_TRUE(RunCubePathTests(stream_));
}
