/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <cann_ops_sparse.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "spmm_config.h"
#include "spmm_tiling_wire.h"
#include "spmm_test_utils.h"

namespace {

constexpr size_t kWorkspaceGuardBytes = 64;

using spmm_test::CheckAcl;
using spmm_test::CheckSparse;
using spmm_test::Bf16ToFloat;
using spmm_test::FloatToBf16;
using spmm_test::FloatToHalf;
using spmm_test::HalfToFloat;

struct DtypeHostData {
    int32_t m = 0;
    int32_t k = 128;
    int32_t n = 256;
    bool fp16 = false;
    float initial = 0.0f;
    std::vector<int32_t> rowPtr;
    std::vector<int32_t> colInd;
    std::vector<float> aFloat;
    std::vector<uint16_t> a;
    std::vector<uint16_t> b;
    std::vector<uint16_t> c;
};

struct DtypeDeviceData {
    void *dRow = nullptr;
    void *dCol = nullptr;
    void *dA = nullptr;
    void *dB = nullptr;
    void *dC = nullptr;
    void *workspace = nullptr;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
};

uint16_t EncodeDtype(bool fp16, float value)
{
    return fp16 ? FloatToHalf(value) : FloatToBf16(value);
}

float DecodeDtype(bool fp16, uint16_t value)
{
    return fp16 ? HalfToFloat(value) : Bf16ToFloat(value);
}

void BuildDtypeHostData(aclDataType dtype, bool betaOne, bool mixedFallback,
                        DtypeHostData *data)
{
    data->m = mixedFallback ? 336 : 64;
    data->fp16 = dtype == ACL_FLOAT16;
    data->initial = betaOne ? 0.25f : 0.0f;
    data->rowPtr.assign(static_cast<size_t>(data->m) + 1, 0);
    for (int32_t row = 0; row < data->m; ++row) {
        const int32_t groupBase = row < 32 ? 0 : 48;
        const int32_t start = groupBase + (row % 32);
        for (int32_t j = 0; j < 7; ++j) {
            data->colInd.push_back(mixedFallback && row == 0 && j == 6 ?
                95 : start + j);
            data->aFloat.push_back(0.0625f * static_cast<float>((row + j) % 5 + 1));
        }
        data->rowPtr[static_cast<size_t>(row) + 1] =
            static_cast<int32_t>(data->colInd.size());
    }

    data->a.resize(data->colInd.size());
    data->b.resize(static_cast<size_t>(data->k) * data->n);
    data->c.resize(static_cast<size_t>(data->m) * data->n);
    for (size_t i = 0; i < data->a.size(); ++i) {
        data->a[i] = EncodeDtype(data->fp16, data->aFloat[i]);
    }
    for (size_t i = 0; i < data->b.size(); ++i) {
        data->b[i] = EncodeDtype(data->fp16,
            0.015625f * static_cast<float>(static_cast<int32_t>(i % 19) - 9));
    }
    std::fill(data->c.begin(), data->c.end(), EncodeDtype(data->fp16, data->initial));
}

bool AllocateDtypeResources(aclrtStream stream, aclDataType dtype,
                            const DtypeHostData &host, DtypeDeviceData *device)
{
    bool ok = CheckAcl(aclrtMalloc(&device->dRow, host.rowPtr.size() * sizeof(int32_t),
        ACL_MEM_MALLOC_HUGE_FIRST), "malloc row") &&
        CheckAcl(aclrtMalloc(&device->dCol, host.colInd.size() * sizeof(int32_t),
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc col") &&
        CheckAcl(aclrtMalloc(&device->dA, host.a.size() * sizeof(uint16_t),
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc A") &&
        CheckAcl(aclrtMalloc(&device->dB, host.b.size() * sizeof(uint16_t),
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc B") &&
        CheckAcl(aclrtMalloc(&device->dC, host.c.size() * sizeof(uint16_t),
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc C");
    if (ok) {
        ok = CheckAcl(aclrtMemcpy(device->dRow, host.rowPtr.size() * sizeof(int32_t),
            host.rowPtr.data(), host.rowPtr.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy row") &&
            CheckAcl(aclrtMemcpy(device->dCol, host.colInd.size() * sizeof(int32_t),
                host.colInd.data(), host.colInd.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy col") &&
            CheckAcl(aclrtMemcpy(device->dA, host.a.size() * sizeof(uint16_t),
                host.a.data(), host.a.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy A") &&
            CheckAcl(aclrtMemcpy(device->dB, host.b.size() * sizeof(uint16_t),
                host.b.data(), host.b.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy B") &&
            CheckAcl(aclrtMemcpy(device->dC, host.c.size() * sizeof(uint16_t),
                host.c.data(), host.c.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy C");
    }
    if (ok) {
        ok = CheckSparse(aclsparseCreate(&device->handle), "create handle") &&
            CheckSparse(aclsparseSetStream(device->handle, stream), "set stream") &&
            CheckSparse(aclsparseCreateCsr(&device->matA, host.m, host.k,
                static_cast<int64_t>(host.colInd.size()), device->dRow, device->dCol, device->dA,
                ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                ACL_SPARSE_INDEX_BASE_ZERO, dtype), "create CSR") &&
            CheckSparse(aclsparseCreateDnMat(&device->matB, host.k, host.n, host.n,
                device->dB, dtype, ACL_SPARSE_ORDER_ROW), "create B") &&
            CheckSparse(aclsparseCreateDnMat(&device->matC, host.m, host.n, host.n,
                device->dC, dtype, ACL_SPARSE_ORDER_ROW), "create C");
    }
    return ok;
}

bool ExecuteDtypeCase(aclrtStream stream, bool betaOne, DtypeDeviceData *device,
                      SpmmTilingWireV1 *tiling,
                      size_t *workspaceSize, bool preprocess = true)
{
    const float alpha = 1.0f;
    const float beta = betaOne ? 1.0f : 0.0f;
    bool ok = CheckSparse(aclsparseSpMMGetBufferSize(device->handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha,
        device->matA, device->matB, &beta, device->matC, ACL_FLOAT,
        ACL_SPARSE_SPMM_CSR_ALG1, workspaceSize), "get buffer size") &&
        CheckAcl(aclrtMalloc(&device->workspace, *workspaceSize + kWorkspaceGuardBytes,
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
    const std::vector<uint8_t> guard(kWorkspaceGuardBytes, 0xA5);
    if (ok) {
        ok = CheckAcl(aclrtMemcpy(static_cast<uint8_t *>(device->workspace) + *workspaceSize,
            guard.size(), guard.data(), guard.size(), ACL_MEMCPY_HOST_TO_DEVICE), "workspace guard");
    }
    if (ok && preprocess) {
        ok = CheckSparse(aclsparseSpMMPreprocess(device->handle,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha,
            device->matA, device->matB, &beta, device->matC, ACL_FLOAT,
            ACL_SPARSE_SPMM_CSR_ALG1, device->workspace), "preprocess");
    }
    if (ok) {
        ok = CheckSparse(aclsparseSpMM(device->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
                ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, device->matA, device->matB,
                &beta, device->matC, ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1,
                device->workspace), "execute") &&
            CheckAcl(aclrtSynchronizeStream(stream), "synchronize");
    }
    if (ok) {
        ok = CheckAcl(aclrtMemcpy(tiling, sizeof(*tiling),
            static_cast<uint8_t *>(device->workspace) + SPMM_WS_HEADER_BYTES,
            sizeof(*tiling), ACL_MEMCPY_DEVICE_TO_HOST), "copy tiling");
    }
    return ok;
}

bool CheckDtypeOutput(bool betaOne, bool mixedFallback, DtypeHostData *host,
                      DtypeDeviceData *device, const SpmmTilingWireV1 &tiling,
                      float *maxError, bool expectCube = true)
{
    const char *dtypeName = host->fp16 ? "FP16" : "BF16";
    bool ok = true;
    if (tiling.cube_mode != (expectCube ? 1 : 0)) {
        std::printf("FAIL dtype=%s beta=%d cube_mode=%d\n", dtypeName, betaOne, tiling.cube_mode);
        ok = false;
    }
    if (ok && mixedFallback && tiling.cube_complex_fallback_count <= 0) {
        std::printf("FAIL dtype=%s beta=%d fallback_rows=%d\n", dtypeName, betaOne,
            tiling.cube_complex_fallback_count);
        ok = false;
    }
    if (ok) {
        ok = CheckAcl(aclrtMemcpy(host->c.data(), host->c.size() * sizeof(uint16_t),
            device->dC, host->c.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST),
            "copy result");
    }
    if (ok) {
        for (int32_t row = 0; row < host->m; ++row) {
            for (int32_t col = 0; col < host->n; ++col) {
                float sum = 0.0f;
                for (int32_t p = host->rowPtr[row]; p < host->rowPtr[row + 1]; ++p) {
                    sum += DecodeDtype(host->fp16, host->a[static_cast<size_t>(p)]) *
                        DecodeDtype(host->fp16,
                            host->b[static_cast<size_t>(host->colInd[p]) * host->n + col]);
                }
                const float expected = DecodeDtype(host->fp16,
                    EncodeDtype(host->fp16, sum + (betaOne ? host->initial : 0.0f)));
                const float actual = DecodeDtype(host->fp16,
                    host->c[static_cast<size_t>(row) * host->n + col]);
                if (!std::isfinite(actual)) return false;
                *maxError = std::max(*maxError, std::fabs(expected - actual));
            }
        }
        ok = *maxError <= (host->fp16 ? 0.0025f : 0.02f);
    }
    return ok;
}

void ReleaseDtypeResources(DtypeDeviceData *device)
{
    if (device->matB) aclsparseDestroyDnMat(device->matB);
    if (device->matC) aclsparseDestroyDnMat(device->matC);
    if (device->matA) aclsparseDestroySpMat(device->matA);
    if (device->handle) aclsparseDestroy(device->handle);
    if (device->workspace) aclrtFree(device->workspace);
    if (device->dRow) aclrtFree(device->dRow);
    if (device->dCol) aclrtFree(device->dCol);
    if (device->dA) aclrtFree(device->dA);
    if (device->dB) aclrtFree(device->dB);
    if (device->dC) aclrtFree(device->dC);
}

bool RunCase(aclrtStream stream, aclDataType dtype, bool betaOne,
             bool mixedFallback = false)
{
    DtypeHostData host;
    BuildDtypeHostData(dtype, betaOne, mixedFallback, &host);
    DtypeDeviceData device;
    bool ok = AllocateDtypeResources(stream, dtype, host, &device);
    SpmmTilingWireV1 tiling{};
    size_t workspaceSize = 0;
    if (ok) ok = ExecuteDtypeCase(stream, betaOne, &device, &tiling, &workspaceSize);
    float maxError = 0.0f;
    if (ok) ok = CheckDtypeOutput(betaOne, mixedFallback, &host, &device, tiling, &maxError);
    std::printf("CUBE_DTYPE=%s N=%d BETA=%d MODE=%d FALLBACK_ROWS=%d "
                "MAX_ERROR=%.9g RESULT=%s\n",
        host.fp16 ? "FP16" : "BF16", host.n, betaOne ? 1 : 0, tiling.cube_mode,
        tiling.cube_complex_fallback_count, maxError, ok ? "PASS" : "FAIL");
    ReleaseDtypeResources(&device);
    return ok;
}

struct BoundaryConfig {
    const char *name;
    int32_t k;
    int32_t first;
    int32_t last;
    int32_t expectedBase;
    int32_t expectedPackedK;
};

void BuildBoundaryInputs(aclDataType dtype, const BoundaryConfig &cfg, DtypeHostData *host)
{
    host->m = 32;
    host->k = cfg.k;
    host->n = 16;
    host->fp16 = dtype == ACL_FLOAT16;
    host->rowPtr.push_back(0);
    for (int32_t row = 0; row < host->m; ++row) {
        for (int32_t col = cfg.first; col <= cfg.last; ++col) {
            host->colInd.push_back(col);
            host->a.push_back(EncodeDtype(host->fp16, 0.0625f * ((col - cfg.first) % 5 + 1)));
        }
        host->rowPtr.push_back(static_cast<int32_t>(host->colInd.size()));
    }
    host->b.resize(static_cast<size_t>(cfg.k) * host->n + 512,
        EncodeDtype(host->fp16, std::numeric_limits<float>::quiet_NaN()));
    for (size_t i = 0; i < static_cast<size_t>(cfg.k) * host->n; ++i) {
        host->b[i] = EncodeDtype(host->fp16, 0.015625f * (static_cast<int32_t>(i % 19) - 9));
    }
    host->c.assign(static_cast<size_t>(host->m) * host->n + 64, EncodeDtype(host->fp16, 0.25f));
}

bool CheckBoundaryMetadata(const BoundaryConfig &cfg, const DtypeDeviceData &device,
                            const SpmmTilingWireV1 &tiling, size_t workspaceSize)
{
    const bool expectCube = cfg.expectedPackedK != 0;
    if ((tiling.cube_mode != 0) != expectCube) return false;
    if (!expectCube) return true;
    if (tiling.cube_group_count <= 0) return false;
    const size_t bytes = static_cast<size_t>(tiling.cube_group_count) * sizeof(SpmmCubeGroupMeta);
    if (tiling.cube_meta_offset > workspaceSize || bytes > workspaceSize - tiling.cube_meta_offset) return false;
    std::vector<SpmmCubeGroupMeta> meta(tiling.cube_group_count);
    if (!CheckAcl(aclrtMemcpy(meta.data(), bytes,
        static_cast<uint8_t *>(device.workspace) + tiling.cube_meta_offset,
        bytes, ACL_MEMCPY_DEVICE_TO_HOST), "boundary meta")) return false;
    for (const auto &window : meta) {
        // DMA reads every packed row, including rows whose packed A is zero.
        if (window.col_base < 0 || window.packed_k <= 0 ||
            static_cast<int64_t>(window.col_base) + window.packed_k > cfg.k ||
            window.col_base != cfg.expectedBase || window.packed_k != cfg.expectedPackedK) return false;
    }
    return true;
}

bool CheckBoundaryGuards(const DtypeHostData &host, const DtypeDeviceData &device,
                          size_t workspaceSize)
{
    std::vector<uint16_t> denseB(host.b.size());
    std::vector<uint8_t> workspaceGuard(kWorkspaceGuardBytes);
    if (!CheckAcl(aclrtMemcpy(denseB.data(), denseB.size() * sizeof(uint16_t), device.dB,
        denseB.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "B input guard") ||
        !CheckAcl(aclrtMemcpy(workspaceGuard.data(), workspaceGuard.size(),
            static_cast<uint8_t *>(device.workspace) + workspaceSize,
            workspaceGuard.size(), ACL_MEMCPY_DEVICE_TO_HOST), "workspace guard result")) return false;
    const size_t logicalOutput = static_cast<size_t>(host.m) * host.n;
    return denseB == host.b &&
        std::all_of(host.c.begin() + logicalOutput, host.c.end(), [&](uint16_t value) {
            return value == EncodeDtype(host.fp16, 0.25f);
        }) && std::all_of(workspaceGuard.begin(), workspaceGuard.end(), [](uint8_t value) {
            return value == 0xA5;
        });
}

bool RunBoundaryCase(aclrtStream stream, aclDataType dtype, const BoundaryConfig &cfg,
                      bool preprocess)
{
    DtypeHostData host;
    BuildBoundaryInputs(dtype, cfg, &host);
    DtypeDeviceData device;
    SpmmTilingWireV1 tiling{};
    size_t workspaceSize = 0;
    float maxError = 0.0f;
    bool pass = AllocateDtypeResources(stream, dtype, host, &device) &&
        ExecuteDtypeCase(stream, false, &device, &tiling, &workspaceSize, preprocess) &&
        CheckBoundaryMetadata(cfg, device, tiling, workspaceSize) &&
        CheckDtypeOutput(false, false, &host, &device, tiling, &maxError, cfg.expectedPackedK != 0) &&
        CheckBoundaryGuards(host, device, workspaceSize);
    std::printf("BOUNDARY=%s DTYPE=%s PREPROCESS=%d RESULT=%s\n", cfg.name,
        host.fp16 ? "FP16" : "BF16", preprocess, pass ? "PASS" : "FAIL");
    ReleaseDtypeResources(&device);
    return pass;
}

bool RunBoundaryTests(aclrtStream stream)
{
    const BoundaryConfig cases[] = {
        {"REVIEWER_K16_TAIL", 16, 12, 15, 0, 16},
        {"SMALL_K1", 1, 0, 0, 0, 0},
        {"SMALL_K15", 15, 11, 14, 0, 0},
        {"UNALIGNED_K17", 17, 13, 16, 0, 0},
        {"TAIL_K20", 20, 16, 19, 4, 16},
        {"TAIL_K32", 32, 28, 31, 16, 16},
        {"UNALIGNED_K33", 33, 29, 32, 0, 0},
        {"TAIL_K48", 48, 44, 47, 32, 16},
        {"FULL_K48", 48, 0, 47, 0, 48},
        {"UNCHANGED_SAFE_WINDOW", 128, 8, 11, 8, 16},
    };
    bool pass = true;
    for (aclDataType dtype : {ACL_FLOAT16, ACL_BF16}) {
        for (const auto &cfg : cases) {
            pass = RunBoundaryCase(stream, dtype, cfg, false) && pass;
            pass = RunBoundaryCase(stream, dtype, cfg, true) && pass;
        }
    }
    return pass;
}

} // namespace

static bool RunCubeDtypeTests(aclrtStream stream)
{
    bool pass = true;
    for (aclDataType dtype : {ACL_FLOAT16, ACL_BF16}) {
        pass = RunCase(stream, dtype, false) && pass;
        pass = RunCase(stream, dtype, true) && pass;
        pass = RunCase(stream, dtype, true, true) && pass;
    }
    pass = RunBoundaryTests(stream) && pass;
    return pass;
}

using spmm_test::SpmmDeviceTest;

TEST_F(SpmmDeviceTest, CubeDtypesAndBoundaries)
{
    ASSERT_TRUE(RunCubeDtypeTests(stream_));
}
