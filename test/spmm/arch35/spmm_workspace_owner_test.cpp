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

#include <acl/acl.h>
#include <cann_ops_sparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "spmm_config.h"
#include "spmm_tiling_wire.h"
#include "spmm_test_utils.h"
#include "host/spmm_host_internal.h"
#include "aclsparse_descr_internal.h"

namespace {

bool CopyH2D(void *dst, const void *src, size_t bytes)
{
    return aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool CopyD2H(void *dst, const void *src, size_t bytes)
{
    return aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

bool CheckOutput(void *deviceC, const std::vector<float> &expected, aclDataType dtype)
{
    std::vector<float> actual(expected.size());
    if (dtype == ACL_FLOAT16 || dtype == ACL_BF16) {
        std::vector<uint16_t> encoded(expected.size());
        if (!CopyD2H(encoded.data(), deviceC, encoded.size() * sizeof(uint16_t))) {
            return false;
        }
        std::transform(encoded.begin(), encoded.end(), actual.begin(), dtype == ACL_FLOAT16 ?
            spmm_test::HalfToFloat : spmm_test::Bf16ToFloat);
    } else if (!CopyD2H(actual.data(), deviceC, actual.size() * sizeof(float))) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!(std::fabs(actual[i] - expected[i]) < 1.0e-5F)) {
            std::printf("workspace output mismatch at %zu: actual=%g expected=%g\n",
                        i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

bool CheckPackedPath(void *workspace)
{
    SpmmTilingWireV1 tiling{};
    if (!CopyD2H(&tiling, static_cast<uint8_t *>(workspace) + SPMM_WS_HEADER_BYTES,
                 sizeof(tiling))) {
        return false;
    }
    if (tiling.cube_mode != 1) {
        std::printf("workspace packed regression requires cube_mode=1, got %d\n", tiling.cube_mode);
        return false;
    }
    return true;
}

bool RunOnce(aclsparseHandle_t handle, aclrtStream stream, aclsparseSpMatDescr_t matA,
             aclsparseDnMatDescr_t matB, aclsparseDnMatDescr_t matC, void *workspace,
             const std::vector<float> &expected, void *deviceC, bool preprocess,
             aclDataType dtype = ACL_FLOAT, bool requirePacked = false,
             bool betaOne = false)
{
    const float alpha = 1.0F;
    const float beta = betaOne ? 1.0F : 0.0F;
    if (preprocess && aclsparseSpMMPreprocess(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, ACL_FLOAT,
            ACL_SPARSE_SPMM_CSR_ALG2, workspace) != ACL_SPARSE_STATUS_SUCCESS) {
        return false;
    }
    if (aclsparseSpMM(handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG2,
            workspace) != ACL_SPARSE_STATUS_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }
    return (!requirePacked || CheckPackedPath(workspace)) && CheckOutput(deviceC, expected, dtype);
}

struct TestResources {
    int64_t m = 4;
    int64_t k = 4;
    int64_t n = 2;
    int64_t nnz = 4;
    aclDataType dtype = ACL_FLOAT;
    aclrtStream stream = nullptr;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA1 = nullptr;
    aclsparseSpMatDescr_t matA2 = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
    void *dRow1 = nullptr;
    void *dRow2 = nullptr;
    void *dCol1 = nullptr;
    void *dCol2 = nullptr;
    void *dVal1 = nullptr;
    void *dVal2 = nullptr;
    void *dB = nullptr;
    void *dC = nullptr;
    void *workspace = nullptr;
};

bool AllocateBuffers(TestResources &resources, size_t indexBytes, size_t columnBytes,
                     size_t valueBytes, size_t denseBytes, size_t outputBytes)
{
    return aclrtMalloc(&resources.dRow1, indexBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(&resources.dRow2, indexBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(&resources.dCol1, columnBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(&resources.dCol2, columnBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(&resources.dVal1, valueBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(&resources.dVal2, valueBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(&resources.dB, denseBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(&resources.dC, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
}

template <typename ValueT>
bool CopyInputs(const TestResources &resources, const std::vector<int32_t> &row1,
                const std::vector<int32_t> &row2, const std::vector<int32_t> &col1,
                const std::vector<int32_t> &col2, const std::vector<ValueT> &val1,
                const std::vector<ValueT> &val2, const std::vector<ValueT> &denseB,
                size_t indexBytes, size_t valueBytes, size_t denseBytes)
{
    return CopyH2D(resources.dRow1, row1.data(), indexBytes) &&
        CopyH2D(resources.dRow2, row2.data(), indexBytes) &&
        CopyH2D(resources.dCol1, col1.data(), col1.size() * sizeof(int32_t)) &&
        CopyH2D(resources.dCol2, col2.data(), col2.size() * sizeof(int32_t)) &&
        CopyH2D(resources.dVal1, val1.data(), valueBytes) &&
        CopyH2D(resources.dVal2, val2.data(), valueBytes) &&
        CopyH2D(resources.dB, denseB.data(), denseBytes);
}

bool CreateDescriptors(TestResources &resources)
{
    return aclsparseCreate(&resources.handle) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseSetStream(resources.handle, resources.stream) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseCreateCsr(&resources.matA1, resources.m, resources.k, resources.nnz,
            resources.dRow1, resources.dCol1, resources.dVal1, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, resources.dtype) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseCreateCsr(&resources.matA2, resources.m, resources.k, resources.nnz,
            resources.dRow2, resources.dCol2, resources.dVal2, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, resources.dtype) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseCreateDnMat(&resources.matB, resources.k, resources.n, resources.n, resources.dB, resources.dtype,
            ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseCreateDnMat(&resources.matC, resources.m, resources.n, resources.n, resources.dC, resources.dtype,
            ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS;
}

bool CheckNewOwnerId(const TestResources &resources, uint64_t previous)
{
    const auto *first = reinterpret_cast<const aclsparseSpMatDescr *>(resources.matA1);
    const auto *second = reinterpret_cast<const aclsparseSpMatDescr *>(resources.matA2);
    if (first->spmmWorkspaceOwnerId == previous ||
        first->spmmWorkspaceOwnerId == second->spmmWorkspaceOwnerId) {
        std::printf("workspace owner generation reused a descriptor identity\n");
        return false;
    }
    return true;
}

bool CheckOwnerUpdates(TestResources &resources)
{
    const auto *first = reinterpret_cast<const aclsparseSpMatDescr *>(resources.matA1);
    uint64_t previous = first->spmmWorkspaceOwnerId;
    if (aclsparseSpMatSetValues(resources.matA1, resources.dVal1) != ACL_SPARSE_STATUS_SUCCESS ||
        !CheckNewOwnerId(resources, previous)) {
        return false;
    }
    previous = first->spmmWorkspaceOwnerId;
    if (aclsparseCsrSetPointers(resources.matA1, resources.dRow1, resources.dCol1,
            resources.dVal1) != ACL_SPARSE_STATUS_SUCCESS || !CheckNewOwnerId(resources, previous)) {
        return false;
    }
    previous = first->spmmWorkspaceOwnerId;
    return aclsparseCsrSetStrides(resources.matA1, 1, 1, 1) == ACL_SPARSE_STATUS_SUCCESS &&
        CheckNewOwnerId(resources, previous);
}

bool PrepareWorkspace(TestResources &resources, size_t &workspaceSize)
{
    const float alpha = 1.0F;
    const float beta = 0.0F;
    if (aclsparseSpMMGetBufferSize(resources.handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, resources.matA1, resources.matB, &beta,
            resources.matC, ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG2, &workspaceSize) !=
            ACL_SPARSE_STATUS_SUCCESS) {
        return false;
    }
    return aclrtMalloc(&resources.workspace, std::max<size_t>(workspaceSize, 1U),
        ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
}

bool RunAlternation(TestResources &resources, const std::vector<float> &expect1,
                    const std::vector<float> &expect2, bool preprocessFirst = true,
                    bool requirePacked = false)
{
    return RunOnce(resources.handle, resources.stream, resources.matA1, resources.matB,
            resources.matC, resources.workspace, expect1, resources.dC, preprocessFirst,
            resources.dtype, requirePacked) &&
        RunOnce(resources.handle, resources.stream, resources.matA2, resources.matB,
            resources.matC, resources.workspace, expect2, resources.dC, false,
            resources.dtype, requirePacked) &&
        RunOnce(resources.handle, resources.stream, resources.matA1, resources.matB,
            resources.matC, resources.workspace, expect1, resources.dC, false,
            resources.dtype, requirePacked);
}

void Cleanup(TestResources &resources)
{
    if (resources.matC != nullptr) (void)aclsparseDestroyDnMat(resources.matC);
    if (resources.matB != nullptr) (void)aclsparseDestroyDnMat(resources.matB);
    if (resources.matA2 != nullptr) (void)aclsparseDestroySpMat(resources.matA2);
    if (resources.matA1 != nullptr) (void)aclsparseDestroySpMat(resources.matA1);
    if (resources.handle != nullptr) (void)aclsparseDestroy(resources.handle);
    if (resources.workspace != nullptr) (void)aclrtFree(resources.workspace);
    if (resources.dC != nullptr) (void)aclrtFree(resources.dC);
    if (resources.dB != nullptr) (void)aclrtFree(resources.dB);
    if (resources.dVal2 != nullptr) (void)aclrtFree(resources.dVal2);
    if (resources.dVal1 != nullptr) (void)aclrtFree(resources.dVal1);
    if (resources.dCol2 != nullptr) (void)aclrtFree(resources.dCol2);
    if (resources.dCol1 != nullptr) (void)aclrtFree(resources.dCol1);
    if (resources.dRow2 != nullptr) (void)aclrtFree(resources.dRow2);
    if (resources.dRow1 != nullptr) (void)aclrtFree(resources.dRow1);
}

bool PrepareSimtCase(TestResources &resources, size_t &workspaceSize)
{
    const std::vector<int32_t> row1{0, 2, 3, 3, 4};
    const std::vector<int32_t> col1{0, 1, 2, 3};
    const std::vector<float> val1{1, 2, 3, 4};
    const std::vector<int32_t> row2{0, 1, 1, 3, 4};
    const std::vector<int32_t> col2{3, 2, 0, 1};
    const std::vector<float> val2{5, 6, 7, 8};
    const std::vector<float> denseB{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> expect1{7, 10, 15, 18, 0, 0, 28, 32};
    const size_t indexBytes = row1.size() * sizeof(int32_t);
    const size_t columnBytes = col1.size() * sizeof(int32_t);
    const size_t valueBytes = val1.size() * sizeof(float);
    const size_t denseBytes = denseB.size() * sizeof(float);
    const size_t outputBytes = expect1.size() * sizeof(float);
    bool ok = AllocateBuffers(resources, indexBytes, columnBytes, valueBytes, denseBytes, outputBytes) &&
        CopyInputs(resources, row1, row2, col1, col2, val1, val2, denseB, indexBytes,
            valueBytes, denseBytes) && CreateDescriptors(resources) && CheckOwnerUpdates(resources);
    return ok && PrepareWorkspace(resources, workspaceSize);
}

bool RunSimtCase(aclrtStream stream)
{
    const std::vector<float> expect1{7, 10, 15, 18, 0, 0, 28, 32};
    const std::vector<float> expect2{35, 40, 0, 0, 37, 50, 24, 32};
    TestResources resources;
    resources.stream = stream;
    size_t workspaceSize = 0;
    bool ok = PrepareSimtCase(resources, workspaceSize) &&
        RunAlternation(resources, expect1, expect2);
    if (ok) {
        // Mirror SpMV ownership invalidation after it overwrites the shared workspace.
        auto *mat = reinterpret_cast<aclsparseSpMatDescr *>(resources.matA1);
        mat->activeBuffer = nullptr;
        mat->activeSpmvBuffer = resources.workspace;
        ok = aclrtMemset(resources.workspace, workspaceSize, 0, workspaceSize) == ACL_SUCCESS &&
            RunOnce(resources.handle, stream, resources.matA1, resources.matB, resources.matC,
                resources.workspace, expect1, resources.dC, false) &&
            mat->activeSpmvBuffer == nullptr;
    }
    std::printf("spmm workspace owner alternation: %s\n", ok ? "PASS" : "FAIL");
    Cleanup(resources);
    return ok;
}

constexpr int32_t kPackedRows = 64;
constexpr int32_t kPackedColumns = 128;
constexpr int32_t kPackedOutputColumns = 256;
constexpr int32_t kPackedDegree = 7;

struct PackedInputs {
    std::vector<int32_t> rows;
    std::vector<int32_t> columns1;
    std::vector<int32_t> columns2;
    std::vector<uint16_t> values1;
    std::vector<uint16_t> values2;
    std::vector<uint16_t> denseB;
    std::vector<float> expected1;
    std::vector<float> expected2;
};

std::vector<float> PackedGolden(const PackedInputs &inputs, bool second)
{
    const auto &columns = second ? inputs.columns2 : inputs.columns1;
    const auto &values = second ? inputs.values2 : inputs.values1;
    std::vector<float> result(kPackedRows * kPackedOutputColumns, 0.0F);
    for (int32_t row = 0; row < kPackedRows; ++row) {
        for (int32_t col = 0; col < kPackedOutputColumns; ++col) {
            for (int32_t p = inputs.rows[row]; p < inputs.rows[row + 1]; ++p) {
                result[row * kPackedOutputColumns + col] += spmm_test::HalfToFloat(values[p]) *
                    spmm_test::HalfToFloat(inputs.denseB[columns[p] * kPackedOutputColumns + col]);
            }
        }
    }
    return result;
}

PackedInputs BuildPackedInputs()
{
    PackedInputs inputs;
    inputs.rows.push_back(0);
    for (int32_t row = 0; row < kPackedRows; ++row) {
        const int32_t start = (row < 32 ? 0 : 48) + row % 32;
        for (int32_t j = 0; j < kPackedDegree; ++j) {
            inputs.columns1.push_back(start + j);
            inputs.columns2.push_back(start + j + 8);
            inputs.values1.push_back(spmm_test::FloatToHalf(0.125F * (j + 1)));
            inputs.values2.push_back(spmm_test::FloatToHalf(0.125F * (j + 2)));
        }
        inputs.rows.push_back(static_cast<int32_t>(inputs.columns1.size()));
    }
    for (int32_t row = 0; row < kPackedColumns; ++row) {
        for (int32_t col = 0; col < kPackedOutputColumns; ++col) {
            inputs.denseB.push_back(spmm_test::FloatToHalf(0.0625F * ((3 * row + col) % 17 - 8)));
        }
    }
    inputs.expected1 = PackedGolden(inputs, false);
    inputs.expected2 = PackedGolden(inputs, true);
    return inputs;
}

bool PreparePackedCase(TestResources &resources, const PackedInputs &inputs)
{
    resources.m = kPackedRows;
    resources.k = kPackedColumns;
    resources.n = kPackedOutputColumns;
    resources.nnz = kPackedRows * kPackedDegree;
    resources.dtype = ACL_FLOAT16;
    const size_t rowBytes = inputs.rows.size() * sizeof(int32_t);
    const size_t columnBytes = inputs.columns1.size() * sizeof(int32_t);
    const size_t valueBytes = inputs.values1.size() * sizeof(uint16_t);
    const size_t denseBytes = inputs.denseB.size() * sizeof(uint16_t);
    const size_t outputBytes = inputs.expected1.size() * sizeof(uint16_t);
    size_t workspaceSize = 0;
    return AllocateBuffers(resources, rowBytes, columnBytes, valueBytes, denseBytes, outputBytes) &&
        CopyInputs(resources, inputs.rows, inputs.rows, inputs.columns1, inputs.columns2,
                   inputs.values1, inputs.values2, inputs.denseB, rowBytes, valueBytes, denseBytes) &&
        CreateDescriptors(resources) && CheckOwnerUpdates(resources) &&
        PrepareWorkspace(resources, workspaceSize);
}

bool RunPackedCase(aclrtStream stream, bool preprocessFirst)
{
    const PackedInputs inputs = BuildPackedInputs();
    if (inputs.expected1 == inputs.expected2) {
        return false;
    }
    TestResources resources;
    resources.stream = stream;
    const bool ok = PreparePackedCase(resources, inputs) &&
        RunAlternation(resources, inputs.expected1, inputs.expected2, preprocessFirst, true);
    std::printf("spmm packed workspace alternation (%s): %s\n",
                preprocessFirst ? "preprocess" : "direct", ok ? "PASS" : "FAIL");
    Cleanup(resources);
    return ok;
}

bool CheckRegistryRules()
{
    spmm_host_internal::WorkspaceOwnershipRegistry<size_t> registry;
    const size_t capacity = spmm_host_internal::kSpmmWorkspaceRegistryCapacity;
    std::vector<uint8_t> keys(capacity + 2);
    for (size_t i = 0; i < keys.size(); ++i) {
        registry.Publish(&keys[i], i);
        if (registry.Size() > capacity) return false;
    }
    if (registry.Find(&keys[0]) != nullptr || registry.Find(&keys[1]) != nullptr) return false;
    registry.Publish(&keys.back(), 99);
    if (registry.Size() != capacity || *registry.Find(&keys.back()) != 99) return false;
    registry.Erase(&keys.back());
    if (registry.Size() != capacity - 1 || registry.Find(&keys.back()) != nullptr) return false;
    for (size_t i = 0; i < capacity * 3; ++i) {
        registry.Publish(&keys.back(), i);
        registry.Erase(&keys.back());
    }
    return registry.Size() == capacity - 1;
}

bool RunLifecycleRegistration(TestResources &resources, void *workspace,
                               const std::vector<float> &expected)
{
    if (aclsparseCreateCsr(&resources.matA2, resources.m, resources.k, resources.nnz,
        resources.dRow1, resources.dCol1, resources.dVal1, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT) != ACL_SPARSE_STATUS_SUCCESS) return false;
    const bool pass = RunOnce(resources.handle, resources.stream, resources.matA2,
        resources.matB, resources.matC, workspace, expected, resources.dC, false);
    const bool destroyed = aclsparseDestroySpMat(resources.matA2) == ACL_SPARSE_STATUS_SUCCESS;
    resources.matA2 = nullptr;
    return pass && destroyed && spmm_host_internal::SpmmWorkspaceRegistrationCount() <=
        spmm_host_internal::kSpmmWorkspaceRegistryCapacity;
}

bool RunDistinctWorkspaceLifecycle(aclrtStream stream)
{
    const std::vector<float> expected{7, 10, 15, 18, 0, 0, 28, 32};
    TestResources resources;
    resources.stream = stream;
    size_t workspaceSize = 0;
    bool pass = PrepareSimtCase(resources, workspaceSize) &&
        RunOnce(resources.handle, stream, resources.matA1, resources.matB, resources.matC,
            resources.workspace, expected, resources.dC, false);
    const size_t count = spmm_host_internal::kSpmmWorkspaceRegistryCapacity + 1;
    const size_t stride = ((workspaceSize + 511) / 512) * 512;
    void *pool = nullptr;
    if (pass) pass = aclrtMalloc(&pool, stride * count, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    if (pass) {
        aclsparseDestroySpMat(resources.matA2);
        resources.matA2 = nullptr;
        // Nonoverlapping regions keep every workspace address distinct even if ACL reuses freed allocations.
        for (size_t i = 0; i < count; ++i) {
            void *workspace = static_cast<uint8_t *>(pool) + i * stride;
            if (!RunLifecycleRegistration(resources, workspace, expected)) {
                pass = false;
                break;
            }
        }
    }
    if (pass) {
        pass = spmm_host_internal::SpmmWorkspaceRegistrationCount() == count - 1 &&
            aclrtMemset(resources.workspace, workspaceSize, 0, workspaceSize) == ACL_SUCCESS &&
            RunOnce(resources.handle, stream, resources.matA1, resources.matB, resources.matC,
                resources.workspace, expected, resources.dC, false);
    }
    if (pool != nullptr) aclrtFree(pool);
    Cleanup(resources);
    std::printf("DISTINCT_WORKSPACE_LIFECYCLE_AND_EVICTED_REBUILD=%s\n", pass ? "PASS" : "FAIL");
    return pass;
}


struct PackedReuseConfig {
    aclDataType dtype;
    int32_t k;
    int32_t firstColumn;
    bool preprocess;
    bool cube;
    bool twoSegments = false;
    bool mixedFallback = false;
    bool betaOne = false;
    int32_t n = 256;
};

template <typename ValueT>
std::vector<ValueT> PackedConstant(aclDataType dtype, size_t count, float value)
{
    const ValueT encoded = dtype == ACL_FLOAT ? static_cast<ValueT>(value) :
        static_cast<ValueT>(dtype == ACL_FLOAT16 ? spmm_test::FloatToHalf(value) :
                                                   spmm_test::FloatToBf16(value));
    return std::vector<ValueT>(count, encoded);
}

bool CreatePackedReuseDescriptors(TestResources &resources, aclDataType dtype)
{
    return aclsparseCreate(&resources.handle) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseSetStream(resources.handle, resources.stream) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseCreateCsr(&resources.matA1, resources.m, resources.k, resources.nnz,
            resources.dRow1, resources.dCol1, resources.dVal1, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, dtype) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseCreateDnMat(&resources.matB, resources.k, resources.n, resources.n,
            static_cast<uint8_t *>(resources.dB) + 64, dtype,
            ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseCreateDnMat(&resources.matC, resources.m, resources.n, resources.n,
            static_cast<uint8_t *>(resources.dC) + 64, dtype,
            ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS;
}

template <typename ValueT>
bool PreparePackedReuse(const PackedReuseConfig &cfg, TestResources &resources,
                         size_t *workspaceBytes)
{
    resources.m = 32;
    resources.k = cfg.k;
    resources.n = cfg.n;
    resources.dtype = cfg.dtype;
    resources.nnz = resources.m * (cfg.twoSegments ? 32 : 16);
    std::vector<int32_t> rows(33, 0), columns;
    for (int32_t row = 0; row < 32; ++row) {
        for (int32_t j = 0; j < (cfg.twoSegments ? 32 : 16); ++j) {
            columns.push_back(cfg.firstColumn + j % 16 + (j >= 16 ? 64 : 0));
        }
        if (cfg.mixedFallback && row == 0) columns.back() = cfg.k - 1;
        rows[static_cast<size_t>(row) + 1] = static_cast<int32_t>(columns.size());
    }
    const auto values = PackedConstant<ValueT>(cfg.dtype, columns.size(), 1.0f);
    std::vector<ValueT> dense(static_cast<size_t>(cfg.k) * resources.n + 128 / sizeof(ValueT),
                             static_cast<ValueT>(0));
    auto ones = PackedConstant<ValueT>(cfg.dtype, static_cast<size_t>(cfg.k) * resources.n, 1.0f);
    if (cfg.n != 256) {
        for (size_t i = 0; i < ones.size(); ++i) {
            ones[i] = static_cast<ValueT>((i % cfg.n % 7 + 1) * 0.125f);
        }
    }
    std::copy(ones.begin(), ones.end(), dense.begin() + 64 / sizeof(ValueT));
    std::vector<ValueT> output(static_cast<size_t>(resources.m) * resources.n + 128 / sizeof(ValueT), 0);
    const float alpha = 1.0f, beta = cfg.betaOne ? 1.0f : 0.0f;
    bool pass = AllocateBuffers(resources, rows.size() * sizeof(int32_t), columns.size() * sizeof(int32_t),
        values.size() * sizeof(ValueT), dense.size() * sizeof(ValueT), output.size() * sizeof(ValueT)) &&
        CopyH2D(resources.dRow1, rows.data(), rows.size() * sizeof(int32_t)) &&
        CopyH2D(resources.dCol1, columns.data(), columns.size() * sizeof(int32_t)) &&
        CopyH2D(resources.dVal1, values.data(), values.size() * sizeof(ValueT)) &&
        CopyH2D(resources.dB, dense.data(), dense.size() * sizeof(ValueT)) &&
        CopyH2D(resources.dC, output.data(), output.size() * sizeof(ValueT)) &&
        CreatePackedReuseDescriptors(resources, cfg.dtype) &&
        aclsparseSpMMGetBufferSize(resources.handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, resources.matA1, resources.matB, &beta,
            resources.matC, ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG2,
            workspaceBytes) == ACL_SPARSE_STATUS_SUCCESS;
    return pass && aclrtMalloc(&resources.workspace, *workspaceBytes + 64,
        ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMemset(resources.workspace, *workspaceBytes + 64, 0xa5, *workspaceBytes + 64) == ACL_SUCCESS;
}

bool CheckPackedReuseMetadata(const PackedReuseConfig &cfg, const TestResources &resources)
{
    SpmmTilingWireV1 tiling{};
    if (!CopyD2H(&tiling, static_cast<uint8_t *>(resources.workspace) + SPMM_WS_HEADER_BYTES,
                 sizeof(tiling)) || (tiling.cube_mode != 0) != cfg.cube) return false;
    if (!cfg.cube) return true;
    std::vector<SpmmCubeGroupMeta> meta(static_cast<size_t>(tiling.cube_group_count));
    if (!CopyD2H(meta.data(), static_cast<uint8_t *>(resources.workspace) + tiling.cube_meta_offset,
                 meta.size() * sizeof(meta[0]))) return false;
    for (const auto &group : meta) {
        const int32_t readK = cfg.dtype == ACL_FLOAT ? tiling.cube_max_k : group.packed_k;
        if (group.col_base < 0 || static_cast<int64_t>(group.col_base) + readK > cfg.k) return false;
    }
    return true;
}

bool CheckPackedReuseGuards(const TestResources &resources, size_t elementBytes,
                            size_t workspaceBytes)
{
    std::vector<uint8_t> bytes(64);
    for (const void *buffer : {resources.dB, resources.dC}) {
        if (!CopyD2H(bytes.data(), buffer, bytes.size()) ||
            !std::all_of(bytes.begin(), bytes.end(), [](uint8_t x) { return x == 0; })) return false;
        const size_t elements = static_cast<size_t>(buffer == resources.dB ? resources.k : resources.m) * resources.n;
        if (!CopyD2H(bytes.data(), static_cast<const uint8_t *>(buffer) + 64 + elements * elementBytes,
                     bytes.size()) ||
            !std::all_of(bytes.begin(), bytes.end(), [](uint8_t x) { return x == 0; })) return false;
    }
    return CopyD2H(bytes.data(), static_cast<uint8_t *>(resources.workspace) + workspaceBytes, bytes.size()) &&
        std::all_of(bytes.begin(), bytes.end(), [](uint8_t x) { return x == 0xa5; });
}

template <typename ValueT>
bool RunPackedReuse(const PackedReuseConfig &cfg, aclrtStream stream)
{
    TestResources resources;
    resources.stream = stream;
    size_t workspaceBytes = 0;
    bool pass = PreparePackedReuse<ValueT>(cfg, resources, &workspaceBytes);
    const float degree = cfg.twoSegments ? 32.0f : 16.0f;
    std::vector<float> expected(static_cast<size_t>(resources.m) * resources.n, degree);
    if (cfg.n != 256) {
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] *= (i % cfg.n % 7 + 1) * 0.125f;
        }
    }
    void *output = pass ? static_cast<uint8_t *>(resources.dC) + 64 : nullptr;
    if (pass) pass = RunOnce(resources.handle, stream, resources.matA1, resources.matB,
        resources.matC, resources.workspace, expected, output, cfg.preprocess, cfg.dtype, false, cfg.betaOne) &&
        CheckPackedReuseMetadata(cfg, resources);
    for (float value : {2.0f, 2.0f, 0.0f}) {
        if (!pass) break;
        const auto updated = PackedConstant<ValueT>(cfg.dtype, static_cast<size_t>(resources.nnz), value);
        for (size_t i = 0; i < expected.size(); ++i) {
            const float factor = cfg.n == 256 ? 1.0f : (i % cfg.n % 7 + 1) * 0.125f;
            expected[i] = (cfg.betaOne ? expected[i] : 0.0f) + degree * value * factor;
        }
        pass = CopyH2D(resources.dVal1, updated.data(), updated.size() * sizeof(ValueT)) &&
            RunOnce(resources.handle, stream, resources.matA1, resources.matB, resources.matC,
                resources.workspace, expected, output, false, cfg.dtype, false, cfg.betaOne) &&
            CheckPackedReuseMetadata(cfg, resources);
    }
    if (pass) pass = CheckPackedReuseGuards(resources, sizeof(ValueT), workspaceBytes);
    Cleanup(resources);
    return pass;
}

} // namespace

static bool RunWorkspaceOwnerTests(aclrtStream stream)
{
    bool pass = CheckRegistryRules();
    pass = RunSimtCase(stream) && pass;
    pass = RunPackedCase(stream, false) && pass;
    pass = RunPackedCase(stream, true) && pass;
    pass = RunDistinctWorkspaceLifecycle(stream) && pass;
    return pass;
}

using spmm_test::SpmmDeviceTest;

TEST_F(SpmmDeviceTest, WorkspaceOwnershipAndLifecycle)
{
    ASSERT_TRUE(RunWorkspaceOwnerTests(stream_));
}


class PackedValuesTest : public SpmmDeviceTest,
                         public ::testing::WithParamInterface<PackedReuseConfig> {};

TEST_P(PackedValuesTest, BoundsAndInPlaceUpdates)
{
    const auto &cfg = GetParam();
    ASSERT_TRUE(cfg.dtype == ACL_FLOAT ? RunPackedReuse<float>(cfg, stream_) :
                                        RunPackedReuse<uint16_t>(cfg, stream_));
}

INSTANTIATE_TEST_SUITE_P(RealCube, PackedValuesTest, ::testing::Values(
    PackedReuseConfig{ACL_FLOAT, 16, 0, false, false},
    PackedReuseConfig{ACL_FLOAT, 16, 0, true, false},
    PackedReuseConfig{ACL_FLOAT, 63, 47, false, false},
    PackedReuseConfig{ACL_FLOAT, 64, 0, false, true},
    PackedReuseConfig{ACL_FLOAT, 64, 0, true, true},
    PackedReuseConfig{ACL_FLOAT, 64, 48, false, true},
    PackedReuseConfig{ACL_FLOAT, 72, 56, true, true},
    PackedReuseConfig{ACL_FLOAT, 73, 57, false, true},
    PackedReuseConfig{ACL_FLOAT, 73, 57, true, true},
    PackedReuseConfig{ACL_FLOAT, 65, 49, false, true},
    PackedReuseConfig{ACL_FLOAT, 71, 55, true, true},
    PackedReuseConfig{ACL_FLOAT, 73, 57, false, true, false, false, false, 272},
    PackedReuseConfig{ACL_FLOAT, 73, 57, true, true, false, false, true, 528},
    PackedReuseConfig{ACL_FLOAT, 128, 0, false, true, true},
    PackedReuseConfig{ACL_FLOAT, 128, 0, false, true, true, false, false, 272},
    PackedReuseConfig{ACL_FLOAT, 128, 0, true, true, true, false, true, 528},
    PackedReuseConfig{ACL_FLOAT16, 16, 0, false, true},
    PackedReuseConfig{ACL_FLOAT16, 16, 0, true, true},
    PackedReuseConfig{ACL_BF16, 16, 0, false, true},
    PackedReuseConfig{ACL_BF16, 16, 0, true, true},
    PackedReuseConfig{ACL_FLOAT16, 128, 0, false, true, false, true},
    PackedReuseConfig{ACL_BF16, 128, 0, true, true, false, true},
    PackedReuseConfig{ACL_FLOAT, 64, 0, true, true, false, false, true},
    PackedReuseConfig{ACL_FLOAT16, 16, 0, false, true, false, false, true},
    PackedReuseConfig{ACL_BF16, 16, 0, true, true, false, false, true}));
