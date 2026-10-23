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
#include <algorithm>
#include <cstdint>
#include <cmath>

#include <cann_ops_sparse.h>
#include <acl/acl.h>
#include "spmm_config.h"
#include "spmm_tiling_wire.h"
#include "spmm_test_utils.h"

namespace {

using spmm_test::CheckAcl;
using spmm_test::CheckSparse;

constexpr int32_t kM = 65538;
constexpr int32_t kK = kM + SPMM_FP32_REGBASE_DEGREE;
constexpr int32_t kN = SPMM_FP32_REGBASE_N;
constexpr int32_t kFallbackRow = 37;
constexpr int32_t kDuplicateStartRow = kM - 1;

struct HostInput {
    std::vector<int32_t> rows;
    std::vector<int32_t> columns;
    std::vector<float> values;
    std::vector<float> matB;
    std::vector<float> matC;
};

using DeviceInput = spmm_test::SpmmDeviceBuffers;

HostInput BuildInput(uint64_t salt)
{
    HostInput input;
    input.rows.resize(static_cast<size_t>(kM) + 1, 0);
    input.columns.reserve(static_cast<size_t>(kM) * SPMM_FP32_REGBASE_DEGREE);
    input.values.reserve(static_cast<size_t>(kM) * SPMM_FP32_REGBASE_DEGREE);
    for (int32_t row = 0; row < kM; ++row) {
        const int32_t start = row == kDuplicateStartRow ? 0 :
            static_cast<int32_t>((static_cast<int64_t>(row) * 32771) % kM);
        if (row == kFallbackRow) {
            input.columns.push_back(1);
            input.columns.push_back(3);
        } else {
            const int32_t degree = row < kM / 2 ? 6 : 7;
            for (int32_t ordinal = 0; ordinal < degree; ++ordinal) {
                input.columns.push_back(start + ordinal);
            }
        }
        input.rows[static_cast<size_t>(row) + 1] =
            static_cast<int32_t>(input.columns.size());
    }
    input.values.resize(input.columns.size());
    for (size_t index = 0; index < input.values.size(); ++index) {
        input.values[index] = spmm_test::InputValue(index, salt + 1);
    }
    input.matB.resize(static_cast<size_t>(kK) * kN);
    for (size_t index = 0; index < input.matB.size(); ++index) {
        input.matB[index] = spmm_test::InputValue(index, salt + 2);
    }
    input.matC.resize(static_cast<size_t>(kM) * kN);
    for (size_t index = 0; index < input.matC.size(); ++index) {
        input.matC[index] = spmm_test::InputValue(index, salt + 3);
    }
    return input;
}

std::vector<float> Golden(const HostInput &input, float beta)
{
    std::vector<float> expected = input.matC;
    for (int32_t row = 0; row < kM; ++row) {
        for (int32_t column = 0; column < kN; ++column) {
            float accumulator = 0.0f;
            for (int32_t position = input.rows[static_cast<size_t>(row)];
                 position < input.rows[static_cast<size_t>(row) + 1]; ++position) {
                const size_t dense = static_cast<size_t>(
                    input.columns[static_cast<size_t>(position)]) * kN + column;
                accumulator += input.values[static_cast<size_t>(position)] *
                    input.matB[dense];
            }
            const size_t output = static_cast<size_t>(row) * kN + column;
            expected[output] = accumulator + beta * input.matC[output];
        }
    }
    return expected;
}

bool CopyMutableInputs(const HostInput &input, const DeviceInput &device)
{
    const auto copy = [](void *destination, const std::vector<float> &source,
                         const char *label) {
        const size_t bytes = source.size() * sizeof(float);
        return CheckAcl(aclrtMemcpy(destination, bytes, source.data(), bytes,
                                    ACL_MEMCPY_HOST_TO_DEVICE), label);
    };
    return copy(device.values, input.values, "copy values") &&
        copy(device.matB, input.matB, "copy B") &&
        copy(device.matC, input.matC, "copy C");
}

bool Verify(const std::vector<float> &expected, void *deviceOutput,
            const char *phase)
{
    std::vector<float> actual(expected.size());
    if (!CheckAcl(aclrtMemcpy(actual.data(), actual.size() * sizeof(float),
                              deviceOutput, actual.size() * sizeof(float),
                              ACL_MEMCPY_DEVICE_TO_HOST), "copy output")) {
        return false;
    }
    double maxError = 0.0;
    size_t failures = 0;
    const auto record = [&](size_t index, double error) {
        maxError = std::max(maxError, error);
        if (std::isfinite(actual[index]) && error <= 2.0e-4) return;
        if (failures < 4) {
            std::printf("%s mismatch[%zu]: expected=%.9g actual=%.9g error=%.9g\n",
                        phase, index, expected[index], actual[index], error);
        }
        ++failures;
    };
    for (size_t index = 0; index < actual.size(); ++index) {
        record(index, std::fabs(static_cast<double>(actual[index]) -
                                static_cast<double>(expected[index])));
    }
    const size_t fallbackBegin = static_cast<size_t>(kFallbackRow) * kN;
    double fallbackMaxError = 0.0;
    for (int32_t column = 0; column < kN; ++column) {
        fallbackMaxError = std::max(fallbackMaxError,
            std::fabs(static_cast<double>(actual[fallbackBegin + column]) -
                      static_cast<double>(expected[fallbackBegin + column])));
    }
    std::printf("PHASE=%s MAX_ERROR=%.9g FALLBACK_MAX_ERROR=%.9g FAILURES=%zu RESULT=%s\n",
                phase, maxError, fallbackMaxError, failures,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0;
}

void Free(DeviceInput *device)
{
    spmm_test::ReleaseSpmmDeviceBuffers(device);
}

struct RegbaseRuntime {
    aclrtStream stream = nullptr;
    DeviceInput device;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
    float alpha = 1.0f;
    float betaOne = 1.0f;
    float betaZero = 0.0f;
    size_t workspaceBytes = 0;
};

void CleanupRuntime(RegbaseRuntime *runtime)
{
    if (runtime->matB) aclsparseDestroyDnMat(runtime->matB);
    if (runtime->matC) aclsparseDestroyDnMat(runtime->matC);
    if (runtime->matA) aclsparseDestroySpMat(runtime->matA);
    if (runtime->handle) aclsparseDestroy(runtime->handle);
    Free(&runtime->device);
    if (runtime->stream) aclrtDestroyStream(runtime->stream);
}

bool AllocateRuntime(const HostInput &input, RegbaseRuntime *runtime)
{
    const size_t rowBytes = input.rows.size() * sizeof(int32_t);
    const size_t columnBytes = input.columns.size() * sizeof(int32_t);
    const size_t valueBytes = input.values.size() * sizeof(float);
    const size_t bBytes = input.matB.size() * sizeof(float);
    const size_t cBytes = input.matC.size() * sizeof(float);
    return CheckAcl(aclrtCreateStream(&runtime->stream), "create stream") &&
        CheckAcl(aclrtMalloc(&runtime->device.rows, rowBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc rows") &&
        CheckAcl(aclrtMalloc(&runtime->device.columns, columnBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc columns") &&
        CheckAcl(aclrtMalloc(&runtime->device.values, valueBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc values") &&
        CheckAcl(aclrtMalloc(&runtime->device.matB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc B") &&
        CheckAcl(aclrtMalloc(&runtime->device.matC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc C");
}

bool UploadStaticInput(const HostInput &input, const DeviceInput &device)
{
    const size_t rowBytes = input.rows.size() * sizeof(int32_t);
    const size_t columnBytes = input.columns.size() * sizeof(int32_t);
    return CheckAcl(aclrtMemcpy(device.rows, rowBytes, input.rows.data(), rowBytes,
                                ACL_MEMCPY_HOST_TO_DEVICE), "copy rows") &&
        CheckAcl(aclrtMemcpy(device.columns, columnBytes, input.columns.data(), columnBytes,
                             ACL_MEMCPY_HOST_TO_DEVICE), "copy columns") &&
        CopyMutableInputs(input, device);
}

bool PrepareSpmm(const HostInput &input, RegbaseRuntime *runtime)
{
    return CheckSparse(aclsparseCreate(&runtime->handle), "create handle") &&
        CheckSparse(aclsparseSetStream(runtime->handle, runtime->stream), "set stream") &&
        CheckSparse(aclsparseCreateCsr(&runtime->matA, kM, kK, input.columns.size(),
            runtime->device.rows, runtime->device.columns, runtime->device.values,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
            ACL_FLOAT), "create CSR") &&
        CheckSparse(aclsparseCreateDnMat(&runtime->matB, kK, kN, kN, runtime->device.matB,
            ACL_FLOAT, ACL_SPARSE_ORDER_ROW), "create B") &&
        CheckSparse(aclsparseCreateDnMat(&runtime->matC, kM, kN, kN, runtime->device.matC,
            ACL_FLOAT, ACL_SPARSE_ORDER_ROW), "create C") &&
        CheckSparse(aclsparseSpMMGetBufferSize(runtime->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &runtime->alpha, runtime->matA, runtime->matB,
            &runtime->betaOne, runtime->matC, ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG2,
            &runtime->workspaceBytes), "buffer size") &&
        CheckAcl(aclrtMalloc(&runtime->device.workspace, runtime->workspaceBytes,
            ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace") &&
        CheckSparse(aclsparseSpMMPreprocess(runtime->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &runtime->alpha, runtime->matA, runtime->matB,
            &runtime->betaOne, runtime->matC, ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG2,
            runtime->device.workspace), "preprocess");
}

bool CopyDispatchData(const RegbaseRuntime &runtime, SpmmTilingWireV1 *tiling,
                      SpmmFp32RegBaseHeader *header)
{
    return CheckAcl(aclrtMemcpy(tiling, sizeof(*tiling),
        static_cast<uint8_t *>(runtime.device.workspace) + SPMM_WS_HEADER_BYTES,
        sizeof(*tiling), ACL_MEMCPY_DEVICE_TO_HOST), "copy tiling") &&
        CheckAcl(aclrtMemcpy(header, sizeof(*header),
            static_cast<uint8_t *>(runtime.device.workspace) + tiling->cube_scratch_offset,
            sizeof(*header), ACL_MEMCPY_DEVICE_TO_HOST), "copy header");
}

bool IsWorkspaceLayoutValid(const SpmmTilingWireV1 &tiling,
                            const SpmmFp32RegBaseHeader &header,
                            size_t workspaceBytes)
{
    const uint64_t batchBytes = static_cast<uint64_t>(header.batch_count) *
        sizeof(SpmmFp32RegBaseBatchMeta);
    const uint64_t rowIdBytes = static_cast<uint64_t>(header.candidate_count) * sizeof(int32_t);
    const uint64_t slot1Bytes = static_cast<uint64_t>(header.fallback_count +
        3 * header.candidate_count) * sizeof(int32_t);
    const uint64_t packedBytes = static_cast<uint64_t>(header.candidate_count) *
        SPMM_FP32_REGBASE_DEGREE * sizeof(float);
    const uint64_t controlBytes = sizeof(SpmmFp32RegBaseHeader) +
        static_cast<uint64_t>(header.refresh_run_count) * sizeof(SpmmFp32RegBaseRefreshRun);
    return tiling.cube_meta_offset + batchBytes <= tiling.cube_row_ids_offset &&
        tiling.cube_row_ids_offset + rowIdBytes <= tiling.cube_slot1_offset &&
        tiling.cube_slot1_offset + slot1Bytes <= tiling.cube_packed_a_offset &&
        tiling.cube_packed_a_offset + packedBytes <= tiling.cube_scratch_offset &&
        tiling.cube_scratch_offset + controlBytes <= tiling.cube_tiling_offset &&
        tiling.cube_tiling_offset <= workspaceBytes;
}

bool CopyMetadata(const RegbaseRuntime &runtime, const SpmmTilingWireV1 &tiling,
                  const SpmmFp32RegBaseHeader &header,
                  std::vector<SpmmFp32RegBaseBatchMeta> *batches,
                  std::vector<int32_t> *rowIds, std::vector<int32_t> *slot1)
{
    const uint64_t batchBytes = static_cast<uint64_t>(header.batch_count) *
        sizeof(SpmmFp32RegBaseBatchMeta);
    const uint64_t rowIdBytes = static_cast<uint64_t>(header.candidate_count) * sizeof(int32_t);
    const uint64_t slot1Bytes = static_cast<uint64_t>(header.fallback_count +
        3 * header.candidate_count) * sizeof(int32_t);
    batches->resize(static_cast<size_t>(header.batch_count));
    rowIds->resize(static_cast<size_t>(header.candidate_count));
    slot1->resize(static_cast<size_t>(header.fallback_count + 3 * header.candidate_count));
    return CheckAcl(aclrtMemcpy(batches->data(), batchBytes,
        static_cast<uint8_t *>(runtime.device.workspace) + tiling.cube_meta_offset,
        batchBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy batches") &&
        CheckAcl(aclrtMemcpy(rowIds->data(), rowIdBytes,
            static_cast<uint8_t *>(runtime.device.workspace) + tiling.cube_row_ids_offset,
            rowIdBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy row ids") &&
        CheckAcl(aclrtMemcpy(slot1->data(), slot1Bytes,
            static_cast<uint8_t *>(runtime.device.workspace) + tiling.cube_slot1_offset,
            slot1Bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy candidate metadata");
}

bool IsMetadataValid(const HostInput &input, const SpmmFp32RegBaseHeader &header,
                     const std::vector<SpmmFp32RegBaseBatchMeta> &batches,
                     const std::vector<int32_t> &rowIds, const std::vector<int32_t> &slot1)
{
    const int32_t *valueBegins = slot1.data() + header.fallback_count;
    const int32_t *degrees = valueBegins + header.candidate_count;
    const int32_t *localStarts = degrees + header.candidate_count;
    bool valid = slot1[0] == kFallbackRow &&
        std::find(rowIds.begin(), rowIds.end(), kDuplicateStartRow) != rowIds.end();
    for (const auto &batch : batches) {
        valid = valid && batch.row_count > 0 && batch.row_count <= SPMM_FP32_REGBASE_ROWS &&
            batch.row_id_begin >= 0 && batch.row_id_begin + batch.row_count <= header.candidate_count &&
            batch.window_rows > 0 && batch.window_rows <= SPMM_FP32_REGBASE_WINDOW_ROWS;
        for (int32_t local = 0; valid && local < batch.row_count; ++local) {
            const int32_t position = batch.row_id_begin + local;
            const int32_t row = rowIds[static_cast<size_t>(position)];
            if (row < 0 || row >= kM) return false;
            const int32_t begin = input.rows[static_cast<size_t>(row)];
            const int32_t degree = input.rows[static_cast<size_t>(row) + 1] - begin;
            const int32_t localStart = localStarts[position];
            valid = valueBegins[position] == begin && degrees[position] == degree &&
                localStart >= 0 && localStart + SPMM_FP32_REGBASE_DEGREE <= batch.window_rows &&
                batch.window_begin + localStart == input.columns[static_cast<size_t>(begin)];
        }
    }
    return valid;
}

bool AuditWorkspace(const HostInput &input, const RegbaseRuntime &runtime)
{
    SpmmTilingWireV1 tiling{};
    SpmmFp32RegBaseHeader header{};
    const bool copied = CopyDispatchData(runtime, &tiling, &header);
    const bool dispatchPass = copied && tiling.cube_mode == 1 &&
        tiling.cube_rows == SPMM_FP32_REGBASE_ROWS &&
        tiling.cube_group_count == header.batch_count && header.batch_count > 0 &&
        header.refresh_run_count == 0 && header.candidate_count == kM - 1 &&
        header.fallback_count == 1;
    const bool layoutPass = copied && IsWorkspaceLayoutValid(tiling, header, runtime.workspaceBytes);
    std::vector<SpmmFp32RegBaseBatchMeta> batches;
    std::vector<int32_t> rowIds;
    std::vector<int32_t> slot1;
    const bool metadataCopied = copied && CopyMetadata(runtime, tiling, header, &batches, &rowIds, &slot1);
    const bool metadataPass = metadataCopied && IsMetadataValid(input, header, batches, rowIds, slot1);
    std::printf("WORKSPACE_BYTES=%zu BATCHES=%d CANDIDATES=%d FALLBACK_ROWS=%d "
                "DISPATCH=%s LAYOUT=%s METADATA=%s\n",
                runtime.workspaceBytes, header.batch_count, header.candidate_count,
                header.fallback_count, dispatchPass ? "PASS" : "FAIL",
                layoutPass ? "PASS" : "FAIL", metadataPass ? "PASS" : "FAIL");
    return dispatchPass && layoutPass && metadataPass;
}

bool Execute(const RegbaseRuntime &runtime, const float *beta)
{
    return CheckSparse(aclsparseSpMM(runtime.handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &runtime.alpha, runtime.matA, runtime.matB, beta,
        runtime.matC, ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG2, runtime.device.workspace), "execute") &&
        CheckAcl(aclrtSynchronizeStream(runtime.stream), "synchronize");
}

bool RunStandardPhase(const HostInput &input, const RegbaseRuntime &runtime,
                      const float *beta, const char *phase)
{
    const std::vector<float> expected = Golden(input, *beta);
    return Execute(runtime, beta) && Verify(expected, runtime.device.matC, phase);
}

bool RunBetaZeroPhase(const HostInput &input, const RegbaseRuntime &runtime)
{
    const size_t cBytes = input.matC.size() * sizeof(float);
    const std::vector<float> expected = Golden(input, runtime.betaZero);
    SpmmTilingWireV1 tiling{};
    const bool pass = CheckAcl(aclrtMemcpy(runtime.device.matC, cBytes, input.matC.data(), cBytes,
        ACL_MEMCPY_HOST_TO_DEVICE), "restore C") && Execute(runtime, &runtime.betaZero) &&
        Verify(expected, runtime.device.matC, "BETA_ZERO_FALLBACK") &&
        CheckAcl(aclrtMemcpy(&tiling, sizeof(tiling),
            static_cast<uint8_t *>(runtime.device.workspace) + SPMM_WS_HEADER_BYTES,
            sizeof(tiling), ACL_MEMCPY_DEVICE_TO_HOST), "copy fallback tiling");
    const bool fallbackDispatch = pass && tiling.cube_mode == 0;
    std::printf("BETA_ZERO_CUBE_MODE=%d RESULT=%s\n", tiling.cube_mode,
                fallbackDispatch ? "PASS" : "FAIL");
    return fallbackDispatch;
}

}  // namespace

static bool RunFp32RegbaseTests(aclrtStream)
{
    HostInput input = BuildInput(10);
    RegbaseRuntime runtime;
    bool pass = AllocateRuntime(input, &runtime) &&
        UploadStaticInput(input, runtime.device) && PrepareSpmm(input, &runtime) &&
        AuditWorkspace(input, runtime) && RunStandardPhase(input, runtime, &runtime.betaOne, "INITIAL");
    if (pass) {
        input = BuildInput(100);
        pass = CopyMutableInputs(input, runtime.device) &&
            RunStandardPhase(input, runtime, &runtime.betaOne, "ACTIVE_VALUES_AND_B");
    }
    if (pass) pass = RunBetaZeroPhase(input, runtime);

    CleanupRuntime(&runtime);
    return pass;
}

using spmm_test::SpmmDeviceTest;

TEST_F(SpmmDeviceTest, Fp32RegbaseRefresh)
{
    ASSERT_TRUE(RunFp32RegbaseTests(stream_));
}
