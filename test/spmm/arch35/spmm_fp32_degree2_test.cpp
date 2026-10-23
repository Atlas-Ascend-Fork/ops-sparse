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
 * Focused Ascend 950PR regression for the guarded FP32 degree-2 path.
 * This test intentionally exercises the public aclsparseSpMM lifecycle rather
 * than calling the device kernels directly.
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

#include <acl/acl.h>
#include <cann_ops_sparse.h>
#include "spmm_config.h"
#include "spmm_tiling_wire.h"
#include "spmm_test_utils.h"

namespace {

using spmm_test::CheckAcl;
using spmm_test::CheckSparse;

struct HostInput {
    int32_t m = 0;
    int32_t k = 0;
    int32_t n = 0;
    std::vector<int32_t> rows;
    std::vector<int32_t> columns;
    std::vector<float> values;
    std::vector<float> matB;
    std::vector<float> matC;
};

void FillPayload(HostInput *input, uint64_t valueSalt, uint64_t bSalt,
                 uint64_t cSalt)
{
    input->values.resize(input->columns.size());
    for (size_t index = 0; index < input->values.size(); ++index) {
        input->values[index] = spmm_test::InputValue(index, valueSalt);
    }
    input->matB.resize(static_cast<size_t>(input->k) * input->n);
    for (size_t index = 0; index < input->matB.size(); ++index) {
        input->matB[index] = spmm_test::InputValue(index, bSalt);
    }
    input->matC.resize(static_cast<size_t>(input->m) * input->n);
    for (size_t index = 0; index < input->matC.size(); ++index) {
        input->matC[index] = spmm_test::InputValue(index, cSalt);
    }
}

using DeviceInput = spmm_test::SpmmDeviceBuffers;

HostInput BuildEligible(uint64_t salt, bool mutate, bool wrapFinal)
{
    // Twenty-six groups of eighty rows share a consecutive two-column
    // window. One final row is deliberately left for the generic fallback.
    HostInput input;
    input.m = 2081;
    input.k = 54;
    input.n = 256;
    input.rows.resize(static_cast<size_t>(input.m) + 1, 0);
    input.columns.reserve(static_cast<size_t>(input.m) * 2);
    input.values.reserve(static_cast<size_t>(input.m) * 2);
    for (int32_t row = 0; row < input.m; ++row) {
        if (row == input.m - 1 && wrapFinal) {
            input.columns.push_back(0);
            input.columns.push_back(input.k - 1);
        } else if (row == input.m - 1) {
            input.columns.push_back(0);
        } else {
            const int32_t group = row / 80;
            const int32_t start = group * 2;
            input.columns.push_back(start);
            input.columns.push_back(start + 1);
        }
        input.rows[static_cast<size_t>(row) + 1] =
            static_cast<int32_t>(input.columns.size());
    }
    FillPayload(&input, salt + (mutate ? 17 : 1),
                salt + (mutate ? 19 : 2), salt + (mutate ? 23 : 3));
    return input;
}

HostInput BuildRejected(uint64_t salt)
{
    // The columns are strictly non-consecutive, so the degree-2 guard must
    // reject the path. N is kept at 256 to ensure rejection is structural.
    HostInput input;
    input.m = 1024;
    input.k = 2048;
    input.n = 256;
    input.rows.resize(static_cast<size_t>(input.m) + 1, 0);
    input.columns.reserve(static_cast<size_t>(input.m) * 2);
    for (int32_t row = 0; row < input.m; ++row) {
        const int32_t start = (row * 2) % (input.k - 2);
        input.columns.push_back(start);
        input.columns.push_back(start + 2);
        input.rows[static_cast<size_t>(row) + 1] =
            static_cast<int32_t>(input.columns.size());
    }
    FillPayload(&input, salt + 1, salt + 2, salt + 3);
    return input;
}

std::vector<float> Golden(const HostInput &input)
{
    std::vector<float> expected = input.matC;
    for (int32_t row = 0; row < input.m; ++row) {
        for (int32_t column = 0; column < input.n; ++column) {
            double accumulator = 0.0;
            for (int32_t position = input.rows[static_cast<size_t>(row)];
                 position < input.rows[static_cast<size_t>(row) + 1];
                 ++position) {
                const size_t dense = static_cast<size_t>(
                    input.columns[static_cast<size_t>(position)]) * input.n +
                    column;
                accumulator += static_cast<double>(
                    input.values[static_cast<size_t>(position)]) *
                    static_cast<double>(input.matB[dense]);
            }
            const size_t output = static_cast<size_t>(row) * input.n + column;
            expected[output] = static_cast<float>(
                accumulator + static_cast<double>(input.matC[output]));
        }
    }
    return expected;
}

bool CopyMutableInputs(const HostInput &input, const DeviceInput &device)
{
    return CheckAcl(aclrtMemcpy(device.values,
                                input.values.size() * sizeof(float),
                                input.values.data(),
                                input.values.size() * sizeof(float),
                                ACL_MEMCPY_HOST_TO_DEVICE), "copy values") &&
        CheckAcl(aclrtMemcpy(device.matB,
                             input.matB.size() * sizeof(float),
                             input.matB.data(),
                             input.matB.size() * sizeof(float),
                             ACL_MEMCPY_HOST_TO_DEVICE), "copy B") &&
        CheckAcl(aclrtMemcpy(device.matC,
                             input.matC.size() * sizeof(float),
                             input.matC.data(),
                             input.matC.size() * sizeof(float),
                             ACL_MEMCPY_HOST_TO_DEVICE), "copy C");
}

bool Verify(const HostInput &input, const std::vector<float> &expected,
            void *deviceOutput, const char *phase)
{
    std::vector<float> actual(expected.size());
    if (!CheckAcl(aclrtMemcpy(actual.data(), actual.size() * sizeof(float),
                              deviceOutput, actual.size() * sizeof(float),
                              ACL_MEMCPY_DEVICE_TO_HOST), "copy output")) {
        return false;
    }
    double maxError = 0.0;
    size_t failures = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double error = std::fabs(static_cast<double>(actual[index]) -
                                       static_cast<double>(expected[index]));
        maxError = std::max(maxError, error);
        if (!std::isfinite(actual[index]) || error > 2.0e-4) {
            if (failures < 4) {
                std::printf("MISMATCH=%s INDEX=%zu EXPECTED=%.9g ACTUAL=%.9g\n",
                            phase, index, expected[index], actual[index]);
            }
            ++failures;
        }
    }
    std::printf("PHASE=%s MAX_ERROR=%.9g FAILURES=%zu RESULT=%s\n",
                phase, maxError, failures, failures == 0 ? "PASS" : "FAIL");
    (void)input;
    return failures == 0;
}

void Free(DeviceInput *device)
{
    spmm_test::ReleaseSpmmDeviceBuffers(device);
}

struct Degree2Runtime {
    aclrtStream stream = nullptr;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
    DeviceInput device;
};

void CleanupDegree2Runtime(Degree2Runtime *runtime)
{
    if (runtime->matB) aclsparseDestroyDnMat(runtime->matB);
    if (runtime->matC) aclsparseDestroyDnMat(runtime->matC);
    if (runtime->matA) aclsparseDestroySpMat(runtime->matA);
    if (runtime->handle) aclsparseDestroy(runtime->handle);
    Free(&runtime->device);
    if (runtime->stream) aclrtDestroyStream(runtime->stream);
}

bool AllocateDegree2Runtime(const HostInput &input, Degree2Runtime *runtime)
{
    return CheckAcl(aclrtCreateStream(&runtime->stream), "create stream") &&
        CheckAcl(aclrtMalloc(&runtime->device.rows,
            input.rows.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "malloc rows") &&
        CheckAcl(aclrtMalloc(&runtime->device.columns,
            input.columns.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "malloc columns") &&
        CheckAcl(aclrtMalloc(&runtime->device.values,
            input.values.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), "malloc values") &&
        CheckAcl(aclrtMalloc(&runtime->device.matB,
            input.matB.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), "malloc B") &&
        CheckAcl(aclrtMalloc(&runtime->device.matC,
            input.matC.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), "malloc C");
}

bool UploadDegree2Input(const HostInput &input, const Degree2Runtime &runtime)
{
    return CheckAcl(aclrtMemcpy(runtime.device.rows,
        input.rows.size() * sizeof(int32_t), input.rows.data(),
        input.rows.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy rows") &&
        CheckAcl(aclrtMemcpy(runtime.device.columns,
            input.columns.size() * sizeof(int32_t), input.columns.data(),
            input.columns.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy columns") &&
        CopyMutableInputs(input, runtime.device);
}

bool PrepareDegree2Spmm(const HostInput &input, Degree2Runtime *runtime,
                        size_t *workspaceBytes)
{
    constexpr float alpha = 1.0f;
    constexpr float beta = 1.0f;
    return CheckSparse(aclsparseCreate(&runtime->handle), "create handle") &&
        CheckSparse(aclsparseSetStream(runtime->handle, runtime->stream), "set stream") &&
        CheckSparse(aclsparseCreateCsr(&runtime->matA, input.m, input.k, input.values.size(),
            runtime->device.rows, runtime->device.columns, runtime->device.values,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
            ACL_FLOAT), "create CSR") &&
        CheckSparse(aclsparseCreateDnMat(&runtime->matB, input.k, input.n, input.n,
            runtime->device.matB, ACL_FLOAT, ACL_SPARSE_ORDER_ROW), "create B") &&
        CheckSparse(aclsparseCreateDnMat(&runtime->matC, input.m, input.n, input.n,
            runtime->device.matC, ACL_FLOAT, ACL_SPARSE_ORDER_ROW), "create C") &&
        CheckSparse(aclsparseSpMMGetBufferSize(runtime->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, runtime->matA, runtime->matB, &beta,
            runtime->matC, ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG2, workspaceBytes), "buffer size") &&
        CheckAcl(aclrtMalloc(&runtime->device.workspace, *workspaceBytes,
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace") &&
        CheckSparse(aclsparseSpMMPreprocess(runtime->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, runtime->matA, runtime->matB, &beta,
            runtime->matC, ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG2,
            runtime->device.workspace), "preprocess");
}

bool CheckDegree2Dispatch(const Degree2Runtime &runtime, size_t workspaceBytes,
                          bool expectDegree2, int32_t expectedRefreshRows,
                          int32_t expectedFallbackRows)
{
    SpmmTilingWireV1 tiling{};
    SpmmFp32RegBaseHeader header{};
    const bool copied = CheckAcl(aclrtMemcpy(&tiling, sizeof(tiling),
        static_cast<uint8_t *>(runtime.device.workspace) + SPMM_WS_HEADER_BYTES,
        sizeof(tiling), ACL_MEMCPY_DEVICE_TO_HOST), "copy tiling") &&
        CheckAcl(aclrtMemcpy(&header, sizeof(header),
            static_cast<uint8_t *>(runtime.device.workspace) + tiling.cube_scratch_offset,
            sizeof(header), ACL_MEMCPY_DEVICE_TO_HOST), "copy header");
    if (!copied) return false;
    const bool dispatchPass = expectDegree2 ?
        (tiling.cube_mode == 1 && tiling.cube_rows == SPMM_FP32_DEGREE2_ROWS &&
         tiling.cube_group_count == header.batch_count && header.batch_count > 0 &&
         header.refresh_run_count == expectedRefreshRows &&
         header.fallback_count == expectedFallbackRows) :
        (tiling.cube_mode == 0 && tiling.cube_rows == 0);
    std::printf("WORKSPACE_BYTES=%zu CUBE_MODE=%d CUBE_ROWS=%d "
                "GROUPS=%d REFRESH_ROWS=%d FALLBACK_ROWS=%d DISPATCH=%s\n",
                workspaceBytes, tiling.cube_mode, tiling.cube_rows,
                tiling.cube_group_count, header.refresh_run_count,
                header.fallback_count, dispatchPass ? "PASS" : "FAIL");
    return dispatchPass;
}

bool ExecuteDegree2Spmm(const Degree2Runtime &runtime)
{
    constexpr float alpha = 1.0f;
    constexpr float beta = 1.0f;
    return CheckSparse(aclsparseSpMM(runtime.handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, runtime.matA, runtime.matB, &beta,
        runtime.matC, ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG2,
        runtime.device.workspace), "execute");
}

bool VerifyInitialDegree2Case(const HostInput &input, const std::vector<float> &expected,
                              const Degree2Runtime &runtime)
{
    return ExecuteDegree2Spmm(runtime) &&
        CheckAcl(aclrtSynchronizeStream(runtime.stream), "initial sync") &&
        Verify(input, expected, runtime.device.matC, "INITIAL");
}

bool VerifyUpdatedDegree2Case(const HostInput &input,
                              const Degree2Runtime &runtime)
{
    const std::vector<float> expected = Golden(input);
    return CopyMutableInputs(input, runtime.device) && ExecuteDegree2Spmm(runtime) &&
        CheckAcl(aclrtSynchronizeStream(runtime.stream), "updated sync") &&
        Verify(input, expected, runtime.device.matC, "ACTIVE_UPDATE");
}

bool RunCase(const char *name, const HostInput &initial,
             const HostInput *updated, bool expectDegree2,
             int32_t expectedRefreshRows, int32_t expectedFallbackRows)
{
    std::printf("CASE=%s M=%d K=%d N=%d NNZ=%zu\n", name, initial.m,
                initial.k, initial.n, initial.values.size());
    const std::vector<float> expected = Golden(initial);
    Degree2Runtime runtime;
    size_t workspaceBytes = 0;
    bool pass = AllocateDegree2Runtime(initial, &runtime) &&
        UploadDegree2Input(initial, runtime) &&
        PrepareDegree2Spmm(initial, &runtime, &workspaceBytes) &&
        CheckDegree2Dispatch(runtime, workspaceBytes, expectDegree2,
                             expectedRefreshRows, expectedFallbackRows) &&
        VerifyInitialDegree2Case(initial, expected, runtime);
    if (pass && updated != nullptr) pass = VerifyUpdatedDegree2Case(*updated, runtime);
    CleanupDegree2Runtime(&runtime);
    std::printf("CASE_RESULT=%s RESULT=%s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

static bool RunFp32Degree2Tests(aclrtStream)
{
    const HostInput eligible = BuildEligible(11, false, false);
    const HostInput updated = BuildEligible(37, true, false);
    const HostInput wrap = BuildEligible(53, false, true);
    const HostInput rejected = BuildRejected(19);
    bool pass = RunCase("ELIGIBLE_REFRESH", eligible, &updated, true, 2080, 1);
    pass = RunCase("WRAP_FUSION", wrap, nullptr, true, 2081, 0) && pass;
    pass = RunCase("NONCONSECUTIVE_FALLBACK", rejected, nullptr, false, 0, 0) && pass;
    return pass;
}

using spmm_test::SpmmDeviceTest;

TEST_F(SpmmDeviceTest, Fp32Degree2Refresh)
{
    ASSERT_TRUE(RunFp32Degree2Tests(stream_));
}
