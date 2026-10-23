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

/*
 * Focused Ascend 950PR regression for the guarded Complex64 RegBase path.
 * The public API lifecycle is reused after changing all mutable inputs so the
 * test also covers execute-time sparse-value refresh for degree 2, 4 and 8.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <acl/acl.h>
#include <cann_ops_sparse.h>

#include "spmm_config.h"
#include "spmm_tiling_wire.h"
#include "spmm_test_utils.h"

extern "C" aclsparseStatus_t aclsparseSpMMPrivateOutOfPlace(
    aclsparseHandle_t handle, aclsparseOperation_t opA,
    aclsparseOperation_t opB, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseConstDnMatDescr_t initialC,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer);

namespace {

using spmm_test::CheckAcl;
using spmm_test::CheckSparse;

constexpr int32_t kM = 1040;
constexpr int32_t kK = 521;
constexpr int32_t kN = SPMM_COMPLEX64_REGBASE_MIN_N;
constexpr double kRtol = 1.0 / 1024.0;
constexpr double kAtol = 1.0 / 65536.0;
constexpr double kHardLimit = 1.0e-2;

struct TestConfig {
    int32_t degree;
    int32_t expectedRows;
    const char *name;
};

struct HostInput {
    std::vector<int32_t> rows;
    std::vector<int32_t> columns;
    std::vector<aclsparseComplex> values;
    std::vector<aclsparseComplex> matB;
    std::vector<aclsparseComplex> matC;
};

using DeviceInput = spmm_test::SpmmDeviceBuffers;

HostInput BuildInput(const TestConfig &config, uint64_t salt)
{
    HostInput input;
    input.rows.resize(static_cast<size_t>(kM) + 1);
    input.columns.resize(static_cast<size_t>(kM) * config.degree);
    input.values.resize(input.columns.size());
    for (int32_t row = 0; row < kM; ++row) {
        input.rows[static_cast<size_t>(row)] = row * config.degree;
        const int32_t start = static_cast<int32_t>(
            (static_cast<int64_t>(row) * 131) % kK);
        int32_t *columns = input.columns.data() +
            static_cast<size_t>(row) * config.degree;
        for (int32_t ordinal = 0; ordinal < config.degree; ++ordinal) {
            columns[ordinal] = (start + ordinal) % kK;
        }
        std::sort(columns, columns + config.degree);
    }
    input.rows.back() = static_cast<int32_t>(input.columns.size());
    for (size_t index = 0; index < input.values.size(); ++index) {
        input.values[index] = aclsparseComplex{
            spmm_test::InputValue(index, salt + 1, 128.0f),
            spmm_test::InputValue(index, salt + 2, 128.0f)};
    }
    input.matB.resize(static_cast<size_t>(kK) * kN);
    for (size_t index = 0; index < input.matB.size(); ++index) {
        input.matB[index] = aclsparseComplex{
            spmm_test::InputValue(index, salt + 3, 128.0f),
            spmm_test::InputValue(index, salt + 4, 128.0f)};
    }
    input.matC.resize(static_cast<size_t>(kM) * kN);
    for (size_t index = 0; index < input.matC.size(); ++index) {
        input.matC[index] = aclsparseComplex{
            spmm_test::InputValue(index, salt + 5, 128.0f),
            spmm_test::InputValue(index, salt + 6, 128.0f)};
    }
    return input;
}

bool CopyMutableInputs(const HostInput &input, const DeviceInput &device)
{
    return CheckAcl(aclrtMemcpy(
                device.values, input.values.size() * sizeof(aclsparseComplex),
                input.values.data(), input.values.size() * sizeof(aclsparseComplex),
                ACL_MEMCPY_HOST_TO_DEVICE), "copy values") &&
        CheckAcl(aclrtMemcpy(
                device.matB, input.matB.size() * sizeof(aclsparseComplex),
                input.matB.data(), input.matB.size() * sizeof(aclsparseComplex),
                ACL_MEMCPY_HOST_TO_DEVICE), "copy B") &&
        CheckAcl(aclrtMemcpy(
                device.matC, input.matC.size() * sizeof(aclsparseComplex),
                input.matC.data(), input.matC.size() * sizeof(aclsparseComplex),
                ACL_MEMCPY_HOST_TO_DEVICE), "copy C");
}

std::complex<double> GoldenElement(const HostInput &input, int32_t row,
                                   int32_t column)
{
    const size_t output = static_cast<size_t>(row) * kN + column;
    std::complex<double> expected(input.matC[output].x, input.matC[output].y);
    for (int32_t position = input.rows[static_cast<size_t>(row)];
         position < input.rows[static_cast<size_t>(row) + 1]; ++position) {
        const aclsparseComplex value = input.values[static_cast<size_t>(position)];
        const aclsparseComplex dense = input.matB[
            static_cast<size_t>(input.columns[static_cast<size_t>(position)]) * kN + column];
        expected += std::complex<double>(value.x, value.y) *
            std::complex<double>(dense.x, dense.y);
    }
    return expected;
}

struct MatchStats {
    uint64_t matchedReal = 0;
    uint64_t matchedImag = 0;
    uint64_t hardFailures = 0;
    double maxError = 0.0;
};

MatchStats CountMatches(const HostInput &input,
                        const std::vector<aclsparseComplex> &actual)
{
    MatchStats stats;
    for (int32_t row = 0; row < kM; ++row) {
        for (int32_t column = 0; column < kN; ++column) {
            const size_t output = static_cast<size_t>(row) * kN + column;
            const std::complex<double> expected = GoldenElement(input, row, column);
            const double realError = std::fabs(
                static_cast<double>(actual[output].x) - expected.real());
            const double imagError = std::fabs(
                static_cast<double>(actual[output].y) - expected.imag());
            stats.maxError = std::max(stats.maxError, std::max(realError, imagError));
            stats.matchedReal += realError <=
                kAtol + kRtol * std::fabs(expected.real());
            stats.matchedImag += imagError <=
                kAtol + kRtol * std::fabs(expected.imag());
            stats.hardFailures += realError > kHardLimit || imagError > kHardLimit ||
                !std::isfinite(actual[output].x) ||
                !std::isfinite(actual[output].y);
        }
    }
    return stats;
}

bool Verify(const HostInput &input, void *deviceOutput, const char *phase)
{
    std::vector<aclsparseComplex> actual(input.matC.size());
    if (!CheckAcl(aclrtMemcpy(
            actual.data(), actual.size() * sizeof(aclsparseComplex),
            deviceOutput, actual.size() * sizeof(aclsparseComplex),
            ACL_MEMCPY_DEVICE_TO_HOST), "copy output")) {
        return false;
    }
    const MatchStats stats = CountMatches(input, actual);
    const double elements = static_cast<double>(actual.size());
    const double realRate = stats.matchedReal / elements;
    const double imagRate = stats.matchedImag / elements;
    const bool pass = realRate >= 0.99 && imagRate >= 0.99 &&
        stats.hardFailures == 0;
    std::printf(
        "PHASE=%s MATCH_RATE_REAL=%.12f MATCH_RATE_IMAG=%.12f "
        "MAX_ERROR=%.12g HARD_FAILURES=%llu RESULT=%s\n",
        phase, realRate, imagRate, stats.maxError,
        static_cast<unsigned long long>(stats.hardFailures),
        pass ? "PASS" : "FAIL");
    return pass;
}

bool VerifyInitialCUnchanged(const HostInput &input, void *deviceInitialC)
{
    std::vector<aclsparseComplex> actual(input.matC.size());
    if (!CheckAcl(aclrtMemcpy(
            actual.data(), actual.size() * sizeof(aclsparseComplex),
            deviceInitialC, actual.size() * sizeof(aclsparseComplex),
            ACL_MEMCPY_DEVICE_TO_HOST), "copy initial C")) {
        return false;
    }
    bool pass = true;
    for (size_t index = 0; index < actual.size(); ++index) {
        pass = pass && actual[index].x == input.matC[index].x &&
            actual[index].y == input.matC[index].y;
    }
    std::printf("SEPARATE_INITIAL_C_UNCHANGED=%s\n", pass ? "PASS" : "FAIL");
    return pass;
}

void Free(DeviceInput *device)
{
    spmm_test::ReleaseSpmmDeviceBuffers(device);
}

struct ComplexRuntime {
    DeviceInput device;
    aclrtStream stream = nullptr;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
    aclsparseDnMatDescr_t initialC = nullptr;
    aclsparseDnMatDescr_t matOut = nullptr;
    size_t workspaceBytes = 0;
    aclsparseComplex alpha{1.0f, 0.0f};
    aclsparseComplex betaOne{1.0f, 0.0f};
    aclsparseComplex betaZero{0.0f, 0.0f};

    void Cleanup()
    {
        if (matB) aclsparseDestroyDnMat(matB);
        if (matOut) aclsparseDestroyDnMat(matOut);
        if (initialC) aclsparseDestroyDnMat(initialC);
        if (matC) aclsparseDestroyDnMat(matC);
        if (matA) aclsparseDestroySpMat(matA);
        if (handle) aclsparseDestroy(handle);
        Free(&device);
        if (stream) aclrtDestroyStream(stream);
    }
};

bool AllocateComplexRuntime(const HostInput &input, ComplexRuntime *runtime)
{
    return CheckAcl(aclrtCreateStream(&runtime->stream), "create stream") &&
        CheckAcl(aclrtMalloc(&runtime->device.rows, input.rows.size() * sizeof(int32_t),
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc rows") &&
        CheckAcl(aclrtMalloc(&runtime->device.columns, input.columns.size() * sizeof(int32_t),
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc columns") &&
        CheckAcl(aclrtMalloc(&runtime->device.values,
            input.values.size() * sizeof(aclsparseComplex), ACL_MEM_MALLOC_HUGE_FIRST), "malloc values") &&
        CheckAcl(aclrtMalloc(&runtime->device.matB,
            input.matB.size() * sizeof(aclsparseComplex), ACL_MEM_MALLOC_HUGE_FIRST), "malloc B") &&
        CheckAcl(aclrtMalloc(&runtime->device.matC,
            input.matC.size() * sizeof(aclsparseComplex), ACL_MEM_MALLOC_HUGE_FIRST), "malloc C") &&
        CheckAcl(aclrtMalloc(&runtime->device.matOut,
            input.matC.size() * sizeof(aclsparseComplex), ACL_MEM_MALLOC_HUGE_FIRST), "malloc output");
}

bool UploadComplexInput(const HostInput &input, const ComplexRuntime &runtime)
{
    return CheckAcl(aclrtMemcpy(runtime.device.rows, input.rows.size() * sizeof(int32_t),
        input.rows.data(), input.rows.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy rows") &&
        CheckAcl(aclrtMemcpy(runtime.device.columns, input.columns.size() * sizeof(int32_t),
            input.columns.data(), input.columns.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy columns") &&
        CopyMutableInputs(input, runtime.device);
}

bool PrepareComplexSpmm(const HostInput &input, ComplexRuntime *runtime)
{
    return CheckSparse(aclsparseCreate(&runtime->handle), "create handle") &&
        CheckSparse(aclsparseSetStream(runtime->handle, runtime->stream), "set stream") &&
        CheckSparse(aclsparseCreateCsr(&runtime->matA, kM, kK, input.values.size(), runtime->device.rows,
            runtime->device.columns, runtime->device.values, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_COMPLEX64), "create CSR") &&
        CheckSparse(aclsparseCreateDnMat(&runtime->matB, kK, kN, kN, runtime->device.matB,
            ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW), "create B") &&
        CheckSparse(aclsparseCreateDnMat(&runtime->matC, kM, kN, kN, runtime->device.matC,
            ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW), "create C") &&
        CheckSparse(aclsparseSpMMGetBufferSize(runtime->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &runtime->alpha, runtime->matA, runtime->matB,
            &runtime->betaOne, runtime->matC, ACL_COMPLEX64, ACL_SPARSE_SPMM_CSR_ALG2,
            &runtime->workspaceBytes), "buffer size") &&
        CheckAcl(aclrtMalloc(&runtime->device.workspace, runtime->workspaceBytes,
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace") &&
        CheckSparse(aclsparseSpMMPreprocess(runtime->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &runtime->alpha, runtime->matA, runtime->matB,
            &runtime->betaOne, runtime->matC, ACL_COMPLEX64, ACL_SPARSE_SPMM_CSR_ALG2,
            runtime->device.workspace), "preprocess");
}

bool CheckComplexDispatch(const TestConfig &config, const ComplexRuntime &runtime)
{
    SpmmTilingWireV1 tiling{};
    SpmmFp32RegBaseHeader header{};
    const bool copied = CheckAcl(aclrtMemcpy(&tiling, sizeof(tiling),
        static_cast<uint8_t *>(runtime.device.workspace) + SPMM_WS_HEADER_BYTES,
        sizeof(tiling), ACL_MEMCPY_DEVICE_TO_HOST), "copy tiling") &&
        CheckAcl(aclrtMemcpy(&header, sizeof(header),
            static_cast<uint8_t *>(runtime.device.workspace) + tiling.cube_scratch_offset,
            sizeof(header), ACL_MEMCPY_DEVICE_TO_HOST), "copy header");
    const bool dispatchPass = copied && tiling.cube_mode == 1 && tiling.uniform_degree == config.degree &&
        tiling.cube_rows == config.expectedRows && tiling.cube_group_count == header.batch_count &&
        header.batch_count > 0 && header.refresh_run_count == kM && header.fallback_count == 0;
    std::printf("NAME=%s DEGREE=%d WORKSPACE_BYTES=%zu CUBE_MODE=%d CUBE_ROWS=%d "
                "GROUPS=%d REFRESH_ROWS=%d FALLBACK_ROWS=%d DISPATCH=%s\n",
                config.name, config.degree, runtime.workspaceBytes, tiling.cube_mode,
                tiling.cube_rows, tiling.cube_group_count, header.refresh_run_count,
                header.fallback_count, dispatchPass ? "PASS" : "FAIL");
    return dispatchPass;
}

bool LaunchComplex(const ComplexRuntime &runtime)
{
    return CheckSparse(aclsparseSpMM(runtime.handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &runtime.alpha, runtime.matA, runtime.matB,
        &runtime.betaOne, runtime.matC, ACL_COMPLEX64, ACL_SPARSE_SPMM_CSR_ALG2,
        runtime.device.workspace), "execute");
}

bool RunInPlacePhase(const HostInput &input, const ComplexRuntime &runtime,
                     const char *phase, const char *syncLabel)
{
    return LaunchComplex(runtime) && CheckAcl(aclrtSynchronizeStream(runtime.stream), syncLabel) &&
        Verify(input, runtime.device.matC, phase);
}

bool RunSeparateOutputPhase(const HostInput &input, ComplexRuntime *runtime)
{
    const size_t bytes = input.matC.size() * sizeof(aclsparseComplex);
    return CopyMutableInputs(input, runtime->device) &&
        CheckAcl(aclrtMemset(runtime->device.matOut, bytes, 0xa5, bytes), "poison separate output") &&
        CheckSparse(aclsparseCreateDnMat(&runtime->initialC, kM, kN, kN, runtime->device.matC,
            ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW), "create initial C") &&
        CheckSparse(aclsparseCreateDnMat(&runtime->matOut, kM, kN, kN, runtime->device.matOut,
            ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW), "create separate output") &&
        CheckSparse(aclsparseSpMMPreprocess(runtime->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &runtime->alpha, runtime->matA, runtime->matB,
            &runtime->betaOne, runtime->matOut, ACL_COMPLEX64, ACL_SPARSE_SPMM_CSR_ALG2,
            runtime->device.workspace), "separate preprocess") &&
        CheckSparse(aclsparseSpMMPrivateOutOfPlace(runtime->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &runtime->alpha, runtime->matA, runtime->matB,
            &runtime->betaOne, runtime->initialC, runtime->matOut, ACL_COMPLEX64,
            ACL_SPARSE_SPMM_CSR_ALG2, runtime->device.workspace), "separate execute") &&
        CheckAcl(aclrtSynchronizeStream(runtime->stream), "separate sync") &&
        Verify(input, runtime->device.matOut, "SEPARATE_OUTPUT") &&
        VerifyInitialCUnchanged(input, runtime->device.matC);
}

bool CheckBetaZeroGuard(ComplexRuntime *runtime)
{
    SpmmTilingWireV1 tiling{};
    const bool prepared = CheckSparse(aclsparseSpMMPreprocess(runtime->handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, &runtime->alpha,
        runtime->matA, runtime->matB, &runtime->betaZero, runtime->matC, ACL_COMPLEX64,
        ACL_SPARSE_SPMM_CSR_ALG2, runtime->device.workspace), "beta-zero preprocess") &&
        CheckAcl(aclrtMemcpy(&tiling, sizeof(tiling),
            static_cast<uint8_t *>(runtime->device.workspace) + SPMM_WS_HEADER_BYTES,
            sizeof(tiling), ACL_MEMCPY_DEVICE_TO_HOST), "copy rejected tiling");
    if (!prepared) return false;
    const bool rejectionPass = tiling.cube_mode == 0 && tiling.cube_rows == 0;
    std::printf("BETA_ZERO_GUARD=%s\n", rejectionPass ? "PASS" : "FAIL");
    return rejectionPass;
}

bool RunComplexRegbaseCase(const TestConfig &config)
{
    std::printf("##### COMPLEX64_REGBASE_CASE_START #####\n"
                "NAME=%s DEGREE=%d EXPECTED_ROWS=%d\n",
                config.name, config.degree, config.expectedRows);
    const HostInput initial = BuildInput(config, 11);
    const HostInput updated = BuildInput(config, 37);
    ComplexRuntime runtime;
    bool pass = AllocateComplexRuntime(initial, &runtime) &&
        UploadComplexInput(initial, runtime) &&
        PrepareComplexSpmm(initial, &runtime) &&
        CheckComplexDispatch(config, runtime) &&
        RunInPlacePhase(initial, runtime, "INITIAL", "initial sync");
    if (pass) {
        pass = CopyMutableInputs(updated, runtime.device) &&
            RunInPlacePhase(updated, runtime, "ACTIVE_UPDATE", "updated sync");
    }
    if (pass) pass = RunSeparateOutputPhase(updated, &runtime);
    if (pass) pass = CheckBetaZeroGuard(&runtime);
    runtime.Cleanup();
    std::printf("##### COMPLEX64_REGBASE_CASE_RESULT #####\n"
                "NAME=%s DEGREE=%d RESULT=%s\n",
                config.name, config.degree, pass ? "PASS" : "FAIL");
    return pass;
}


enum class PackingCase { SmallAReal, SmallAImag, SmallBReal, SmallBImag, Components,
                         CancellationA, CancellationB, BTail, OverflowA, OverflowB, Safe };

struct PackingConfig {
    PackingCase mode;
    int32_t n;
    bool betaOne = false;
    int32_t k = 48;
};

struct CubeProbe {
    spmm_test::SpmmDeviceBuffers device;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
    size_t workspaceBytes = 0;

    ~CubeProbe()
    {
        if (matC != nullptr) aclsparseDestroyDnMat(matC);
        if (matB != nullptr) aclsparseDestroyDnMat(matB);
        if (matA != nullptr) aclsparseDestroySpMat(matA);
        if (handle != nullptr) aclsparseDestroy(handle);
        for (void *buffer : {device.rows, device.columns, device.values,
                             device.matB, device.matC, device.workspace}) {
            if (buffer != nullptr) aclrtFree(buffer);
        }
    }
};

aclsparseComplex PackingA(PackingCase mode, int32_t position)
{
    constexpr float tiny = 0x1p-27f;
    constexpr float large = 32768.0f;
    if (mode == PackingCase::SmallAReal) return {tiny, 0.0f};
    if (mode == PackingCase::SmallAImag) return {0.0f, tiny};
    if (mode == PackingCase::Components) return {tiny, -tiny};
    if (mode == PackingCase::SmallBReal || mode == PackingCase::CancellationB) return {large, 0.0f};
    if (mode == PackingCase::SmallBImag) return {0.0f, large};
    if (mode == PackingCase::CancellationA) {
        return {position < 30 ? (position % 2 == 0 ? 1.0f : -1.0f) :
                (position == 30 ? tiny : 0.0f), 0.0f};
    }
    if (mode == PackingCase::OverflowA) return {65536.0f, 0.0f};
    if (mode == PackingCase::OverflowB) return {0x1p-16f, 0.0f};
    return {0.5f, 0.0f};
}

aclsparseComplex PackingB(PackingCase mode, int32_t row, bool tail)
{
    constexpr float tiny = 0x1p-27f;
    if (mode == PackingCase::SmallBReal) return {tiny, 0.0f};
    if (mode == PackingCase::SmallBImag) return {0.0f, tiny};
    if (mode == PackingCase::SmallAReal || mode == PackingCase::SmallAImag ||
        mode == PackingCase::CancellationA) return {32768.0f, 0.0f};
    if (mode == PackingCase::Components) return {32768.0f, 32768.0f};
    if (mode == PackingCase::CancellationB) {
        return {row < 30 ? (row % 2 == 0 ? 1.0f : -1.0f) :
                (row == 30 ? tiny : 0.0f), 0.0f};
    }
    if (mode == PackingCase::BTail && tail) return {tiny, 0.0f};
    if (mode == PackingCase::OverflowA) return {0x1p-16f, 0.0f};
    if (mode == PackingCase::OverflowB) return {65536.0f, 0.0f};
    return {0.25f, 0.0f};
}

bool AllocateCubeProbe(aclrtStream stream, const PackingConfig &cfg, CubeProbe *probe)
{
    auto &device = probe->device;
    const int32_t n = cfg.n;
    return CheckAcl(aclrtMalloc(&device.rows, 17 * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "rows") &&
        CheckAcl(aclrtMalloc(&device.columns, 512 * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "columns") &&
        CheckAcl(aclrtMalloc(&device.values, 512 * sizeof(aclsparseComplex), ACL_MEM_MALLOC_HUGE_FIRST), "values") &&
        CheckAcl(aclrtMalloc(&device.matB, cfg.k * n * sizeof(aclsparseComplex) + 64, ACL_MEM_MALLOC_HUGE_FIRST), "B") &&
        CheckAcl(aclrtMalloc(&device.matC, 16 * n * sizeof(aclsparseComplex) + 64, ACL_MEM_MALLOC_HUGE_FIRST), "C") &&
        CheckAcl(aclrtMemset(device.matB, cfg.k * n * sizeof(aclsparseComplex) + 64, 0xa5,
                            cfg.k * n * sizeof(aclsparseComplex) + 64), "guard B") &&
        CheckAcl(aclrtMemset(device.matC, 16 * n * sizeof(aclsparseComplex) + 64, 0xa5,
                            16 * n * sizeof(aclsparseComplex) + 64), "guard C") &&
        CheckSparse(aclsparseCreate(&probe->handle), "create") &&
        CheckSparse(aclsparseSetStream(probe->handle, stream), "stream") &&
        CheckSparse(aclsparseCreateCsr(&probe->matA, 16, cfg.k, 512, device.rows, device.columns,
            device.values, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, ACL_COMPLEX64), "A") &&
        CheckSparse(aclsparseCreateDnMat(&probe->matB, cfg.k, n, n, device.matB,
            ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW), "B") &&
        CheckSparse(aclsparseCreateDnMat(&probe->matC, 16, n, n, device.matC,
            ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW), "C");
}

bool PrepareCubeProbe(const PackingConfig &cfg, aclrtStream stream, CubeProbe *probe)
{
    std::vector<int32_t> rows(17), columns(512);
    for (int32_t row = 0; row <= 16; ++row) rows[static_cast<size_t>(row)] = row * 32;
    for (int32_t p = 0; p < 512; ++p) columns[static_cast<size_t>(p)] = p % 32;
    const aclsparseComplex alpha{1.0f, 0.0f}, beta{cfg.betaOne ? 1.0f : 0.0f, 0.0f};
    return AllocateCubeProbe(stream, cfg, probe) &&
        CheckAcl(aclrtMemcpy(probe->device.rows, rows.size() * sizeof(int32_t), rows.data(),
            rows.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "upload rows") &&
        CheckAcl(aclrtMemcpy(probe->device.columns, columns.size() * sizeof(int32_t), columns.data(),
            columns.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "upload columns") &&
        CheckSparse(aclsparseSpMMGetBufferSize(probe->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, probe->matA, probe->matB, &beta, probe->matC,
            ACL_COMPLEX64, ACL_SPARSE_SPMM_CSR_ALG2, &probe->workspaceBytes), "size") &&
        CheckAcl(aclrtMalloc(&probe->device.workspace, probe->workspaceBytes + 64,
            ACL_MEM_MALLOC_HUGE_FIRST), "workspace") &&
        CheckAcl(aclrtMemset(probe->device.workspace, probe->workspaceBytes + 64,
            0xa5, probe->workspaceBytes + 64), "guard workspace");
}

bool RunCubeProbePhase(const PackingConfig &cfg, aclrtStream stream, CubeProbe &probe)
{
    std::vector<aclsparseComplex> values(512), dense(static_cast<size_t>(cfg.k) * cfg.n);
    for (size_t p = 0; p < values.size(); ++p) values[p] = PackingA(cfg.mode, static_cast<int32_t>(p % 32));
    const size_t tailStart = dense.size() / 32 * 32;
    for (size_t p = 0; p < dense.size(); ++p) {
        dense[p] = PackingB(cfg.mode, static_cast<int32_t>(p / cfg.n), p >= tailStart);
    }
    const aclsparseComplex initial{cfg.betaOne ? 0.5f : 0.0f, cfg.betaOne ? -0.25f : 0.0f};
    const std::vector<aclsparseComplex> inputC(static_cast<size_t>(16) * cfg.n, initial);
    const aclsparseComplex alpha{1.0f, 0.0f}, beta{cfg.betaOne ? 1.0f : 0.0f, 0.0f};
    const bool pass = CheckAcl(aclrtMemcpy(probe.device.values, values.size() * sizeof(values[0]), values.data(),
        values.size() * sizeof(values[0]), ACL_MEMCPY_HOST_TO_DEVICE), "upload values") &&
        CheckAcl(aclrtMemcpy(probe.device.matB, dense.size() * sizeof(dense[0]), dense.data(),
            dense.size() * sizeof(dense[0]), ACL_MEMCPY_HOST_TO_DEVICE), "upload B") &&
        CheckAcl(aclrtMemcpy(probe.device.matC, inputC.size() * sizeof(inputC[0]), inputC.data(),
            inputC.size() * sizeof(inputC[0]), ACL_MEMCPY_HOST_TO_DEVICE), "restore C") &&
        CheckSparse(aclsparseSpMM(probe.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha, probe.matA, probe.matB, &beta, probe.matC, ACL_COMPLEX64,
            ACL_SPARSE_SPMM_CSR_ALG2, probe.device.workspace), "execute") &&
        CheckAcl(aclrtSynchronizeStream(stream), "sync");
    if (!pass) return false;
    std::vector<aclsparseComplex> output(static_cast<size_t>(16) * cfg.n);
    if (!CheckAcl(aclrtMemcpy(output.data(), output.size() * sizeof(output[0]), probe.device.matC,
        output.size() * sizeof(output[0]), ACL_MEMCPY_DEVICE_TO_HOST), "copy output")) return false;
    for (size_t index = 0; index < output.size(); ++index) {
        std::complex<double> expected(initial.x, initial.y);
        for (size_t p = 0; p < 32; ++p) {
            const auto &a = values[index / cfg.n * 32 + p];
            const auto &b = dense[p * cfg.n + index % cfg.n];
            expected += std::complex<double>(a.x, a.y) * std::complex<double>(b.x, b.y);
        }
        const std::complex<double> actual(output[index].x, output[index].y);
        if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
            std::abs(actual - expected) > kAtol + kRtol * std::abs(expected)) {
            std::fprintf(stderr, "PACKING_MODE=%d K=%d N=%d INDEX=%zu ACTUAL=(%g,%g) EXPECTED=(%g,%g)\n",
                static_cast<int>(cfg.mode), cfg.k, cfg.n, index, actual.real(), actual.imag(),
                expected.real(), expected.imag());
            return false;
        }
    }
    return true;
}

bool CheckCubeProbeDispatch(const PackingConfig &cfg, const CubeProbe &probe)
{
    SpmmTilingWireV1 td{};
    int32_t status = 0;
    if (!CheckAcl(aclrtMemcpy(&td, sizeof(td), static_cast<uint8_t *>(probe.device.workspace) +
        SPMM_WS_HEADER_BYTES, sizeof(td), ACL_MEMCPY_DEVICE_TO_HOST), "tiling")) return false;
    if (cfg.n % SPMM_CUBE_MAX_N != 0) return td.cube_mode == 0;
    if (td.cube_mode != 1) return false;
    if (!CheckAcl(aclrtMemcpy(&status, sizeof(status), static_cast<uint8_t *>(probe.device.workspace) +
        td.cube_complex_status_offset, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST), "status")) return false;
    return (status != 0) == (cfg.mode != PackingCase::Safe);
}

bool CheckCubeProbeGuards(const PackingConfig &cfg, const CubeProbe &probe)
{
    std::vector<uint8_t> guard(64);
    const std::pair<const void *, size_t> buffers[] = {
        {probe.device.matB, static_cast<size_t>(cfg.k) * cfg.n * sizeof(aclsparseComplex)},
        {probe.device.matC, static_cast<size_t>(16) * cfg.n * sizeof(aclsparseComplex)},
        {probe.device.workspace, probe.workspaceBytes}};
    for (const auto &buffer : buffers) {
        if (!CheckAcl(aclrtMemcpy(guard.data(), guard.size(), static_cast<const uint8_t *>(buffer.first) +
            buffer.second, guard.size(), ACL_MEMCPY_DEVICE_TO_HOST), "guard") ||
            !std::all_of(guard.begin(), guard.end(), [](uint8_t x) { return x == 0xa5; })) return false;
    }
    return true;
}

}  // namespace

using spmm_test::SpmmDeviceTest;

class ComplexRegbaseTest : public SpmmDeviceTest,
                           public ::testing::WithParamInterface<TestConfig> {};

TEST_P(ComplexRegbaseTest, RefreshesMutableInputs)
{
    ASSERT_TRUE(RunComplexRegbaseCase(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Degrees, ComplexRegbaseTest, ::testing::Values(
    TestConfig{SPMM_COMPLEX64_DEGREE2_DEGREE, SPMM_COMPLEX64_DEGREE2_ROWS, "DEGREE_2"},
    TestConfig{SPMM_COMPLEX64_DEGREE4_DEGREE, SPMM_COMPLEX64_DEGREE4_ROWS, "DEGREE_4"},
    TestConfig{SPMM_COMPLEX64_DEGREE8_DEGREE, SPMM_COMPLEX64_DEGREE8_ROWS, "DEGREE_8"}));


class ComplexPackingTest : public SpmmDeviceTest,
                           public ::testing::WithParamInterface<PackingConfig> {};

TEST_P(ComplexPackingTest, RangeAndReconstruction)
{
    CubeProbe probe;
    const auto &cfg = GetParam();
    ASSERT_TRUE(PrepareCubeProbe(cfg, stream_, &probe));
    ASSERT_TRUE(RunCubeProbePhase(cfg, stream_, probe));
    ASSERT_TRUE(CheckCubeProbeDispatch(cfg, probe));
    const PackingConfig safe{PackingCase::Safe, cfg.n, cfg.betaOne, cfg.k};
    ASSERT_TRUE(RunCubeProbePhase(safe, stream_, probe));
    ASSERT_TRUE(CheckCubeProbeDispatch(safe, probe));
    ASSERT_TRUE(CheckCubeProbeGuards(cfg, probe));
}

INSTANTIATE_TEST_SUITE_P(ComplexCube, ComplexPackingTest, ::testing::Values(
    PackingConfig{PackingCase::SmallAReal, 256}, PackingConfig{PackingCase::SmallAImag, 256},
    PackingConfig{PackingCase::SmallBReal, 256}, PackingConfig{PackingCase::SmallBImag, 272, false, 49},
    PackingConfig{PackingCase::Components, 256}, PackingConfig{PackingCase::CancellationA, 256},
    PackingConfig{PackingCase::CancellationB, 257}, PackingConfig{PackingCase::BTail, 272, false, 49},
    PackingConfig{PackingCase::OverflowA, 256}, PackingConfig{PackingCase::OverflowB, 257},
    PackingConfig{PackingCase::Safe, 256}, PackingConfig{PackingCase::SmallAReal, 256, true},
    PackingConfig{PackingCase::SmallBReal, 257, true},
    PackingConfig{PackingCase::Safe, 16, false, 49},
    PackingConfig{PackingCase::Safe, 512, false, 49},
    PackingConfig{PackingCase::SmallBImag, 256, false, 49}));
