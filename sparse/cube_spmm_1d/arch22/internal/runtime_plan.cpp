/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "plan_internal.h"
#include <acl/acl.h>

#include <dlfcn.h>
#include <sys/stat.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../cube_spmm_1d_kernel.h"
#include "../spmm_1d.h"
#include "spmm_1d_adaptive.h"

#ifndef SPMM_1D_SUPPORTED_SOC
#define SPMM_1D_SUPPORTED_SOC 0
#endif

namespace {

struct RegisteredBinaryStorage {
    std::unique_ptr<char[]> data;
    size_t size = 0;
    void* handle = nullptr;
    bool ready = false;
};

struct SpmmLaunchPlan {
    SpmmTilingConfig tiling{};
    int64_t runtimeCoreCount = 0;
    int64_t groupDim = 0;
    int64_t paddedN = 0;
    int64_t bPackSlotStrideElements = 0;
    int64_t requiredBPackElements = 0;
    size_t requiredWorkspaceBytes = 0;
};

std::unordered_map<std::string, RegisteredBinaryStorage>& RegisteredKernels()
{
    // Successful registrations and their backing data remain live until process exit.
    static std::unordered_map<std::string, RegisteredBinaryStorage> kernels;
    return kernels;
}

std::mutex& RegistrationMutex()
{
    static std::mutex mutex;
    return mutex;
}

bool CheckedMultiplyInt64(int64_t lhs, int64_t rhs, int64_t* result)
{
    if (result == nullptr || lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool CheckedElementsToBytes(int64_t elements, size_t* bytes)
{
    if (bytes == nullptr || elements < 0) {
        return false;
    }
    const uint64_t unsignedElements = static_cast<uint64_t>(elements);
    if (unsignedElements > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
        return false;
    }
    *bytes = static_cast<size_t>(unsignedElements) * sizeof(uint16_t);
    return true;
}

bool AlignWorkspaceBytes(size_t value, size_t* aligned)
{
    if (aligned == nullptr) {
        return false;
    }
    const size_t alignment = static_cast<size_t>(SPMM_1D_WORKSPACE_ALIGNMENT_BYTES);
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return false;
    }
    *aligned = ((value + alignment - 1) / alignment) * alignment;
    if (*aligned == 0) {
        *aligned = alignment;
    }
    return true;
}

std::string ResolveKernelBinary(const char* binaryName)
{
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&CubeSpmm1dInternalCreatePlan), &info) != 0 && info.dli_fname != nullptr) {
        // Keep lookup relative to this library, without requiring GCC 8's filesystem library.
        const std::string libraryPath(info.dli_fname);
        const auto separator = libraryPath.find_last_of('/');
        const std::string libraryDir = separator == std::string::npos ? "" : libraryPath.substr(0, separator + 1);
        const std::string installed = libraryDir + "cube_spmm_1d_kernels/" + binaryName;
        struct stat metadata{};
        if (stat(installed.c_str(), &metadata) == 0 && S_ISREG(metadata.st_mode)) {
            return installed;
        }
    }

    return {};
}

RegisteredBinaryStorage ReadKernelBinary(const char* binaryName)
{
    const std::string binaryPath = ResolveKernelBinary(binaryName);
    if (binaryPath.empty()) {
        return {};
    }

    std::ifstream input(binaryPath, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        return {};
    }

    const std::streampos end = input.tellg();
    if (end <= 0) {
        return {};
    }

    const size_t size = static_cast<size_t>(end);
    auto buffer = std::make_unique<char[]>(size);
    input.seekg(0, std::ios::beg);
    input.read(buffer.get(), static_cast<std::streamsize>(size));
    if (!input || static_cast<size_t>(input.gcount()) != size) {
        return {};
    }
    return {std::move(buffer), size, nullptr, false};
}

aclError RegisterBinaryKernel(const char* kernelName, const char* binaryName)
{
    auto binaryStorage = ReadKernelBinary(binaryName);
    if (!binaryStorage.data) {
        return static_cast<aclError>(1);
    }

    // Allocate the cache node before creating any runtime registration. No allocation
    // is permitted between a successful runtime call and marking this entry ready.
    auto& kernels = RegisteredKernels();
    auto entry = kernels.try_emplace(kernelName).first;
    auto& storage = entry->second;
    storage = std::move(binaryStorage);
    rtDevBinary_t binary{};
    binary.data = storage.data.get();
    binary.length = storage.size;
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;

    aclError ret = rtDevBinaryRegister(&binary, &storage.handle);
    if (ret != ACL_ERROR_NONE) {
        kernels.erase(entry);
        return ret;
    }

    ret = rtFunctionRegister(storage.handle, kernelName, kernelName, kernelName, 0);
    if (ret != ACL_ERROR_NONE) {
        if (rtDevBinaryUnRegister(storage.handle) == ACL_ERROR_NONE) {
            kernels.erase(entry);
        }
        // If rollback failed, keep both the handle and data for the next retry.
        return ret;
    }

    storage.ready = true;
    return ACL_ERROR_NONE;
}

aclsparseStatus_t EnsureKernelRegistered(const SpmmTilingConfig& tiling)
{
    std::lock_guard<std::mutex> lock(RegistrationMutex());
    auto& kernels = RegisteredKernels();
    auto entry = kernels.find(tiling.kernel_name);
    if (entry != kernels.end()) {
        if (entry->second.ready) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        if (rtDevBinaryUnRegister(entry->second.handle) != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        kernels.erase(entry);
    }

    const aclError ret = RegisterBinaryKernel(tiling.kernel_name, tiling.binary_name);
    if (ret != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t QueryRuntimeAiCoreCount(int64_t* coreCount)
{
    if (coreCount == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int32_t deviceId = 0;
    aclError ret = aclrtGetDevice(&deviceId);
    if (ret != ACL_ERROR_NONE || deviceId < 0) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    int64_t queried = 0;
    ret = aclGetDeviceCapability(static_cast<uint32_t>(deviceId), ACL_DEVICE_INFO_AI_CORE_NUM, &queried);
    if (ret != ACL_ERROR_NONE || queried <= 0) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    *coreCount = queried;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t CompleteWorkspacePlan(int64_t maxVectorsPerRow, bool direct, SpmmLaunchPlan* plan)
{
    if (direct) {
        plan->bPackSlotStrideElements = 0;
        plan->requiredBPackElements = 0;
        plan->requiredWorkspaceBytes = 512;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const int fusionGroup = SpmmFusionGroup(plan->tiling.tiling_key);
    if (fusionGroup) {
        if (maxVectorsPerRow > 256) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        maxVectorsPerRow = std::max<int64_t>(maxVectorsPerRow, SpmmFusionCapacity(fusionGroup));
    }
    constexpr int32_t vectorLength = SPMM_1D_VECTOR_LENGTH;
    const int64_t maxRowKRound = SpmmAlignUp(maxVectorsPerRow, 16);
    if (maxRowKRound < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    int64_t bPackBatchColumns = 0;
    if (!CheckedMultiplyInt64(static_cast<int64_t>(plan->tiling.b_pack_n_batch), static_cast<int64_t>(vectorLength),
                              &bPackBatchColumns) ||
        !CheckedMultiplyInt64(maxRowKRound, bPackBatchColumns, &plan->bPackSlotStrideElements)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    int64_t perCoreElements = 0;
    if (!CheckedMultiplyInt64(SPMM_1D_BPACK_PIPE_DEPTH, plan->bPackSlotStrideElements, &perCoreElements) ||
        !CheckedMultiplyInt64(plan->groupDim, perCoreElements, &plan->requiredBPackElements)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    size_t rawBytes = 0;
    if (!CheckedElementsToBytes(plan->requiredBPackElements, &rawBytes) ||
        !AlignWorkspaceBytes(rawBytes, &plan->requiredWorkspaceBytes)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (fusionGroup) {
        int64_t extraElements = 0;
        size_t extraBytes = 0;
        if (!CheckedMultiplyInt64(plan->groupDim, 2 * 32768, &extraElements) ||
            !CheckedElementsToBytes(extraElements, &extraBytes) ||
            plan->requiredWorkspaceBytes > std::numeric_limits<size_t>::max() - extraBytes) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        plan->requiredWorkspaceBytes += extraBytes;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

KernelArgs MakeKernelArgs(int32_t m, int32_t n, int32_t k, int32_t vectorLength, const void* values,
                          int64_t maxVectorsPerRow, const int32_t* columnIndices, const int32_t* rowOffsets,
                          const void* b, void* c, void* bPackWorkspace, uint64_t fftsAddressRaw,
                          const SpmmLaunchPlan& plan)
{
    constexpr int64_t cubeBlock = 16;
    const int64_t numNBlocks = (static_cast<int64_t>(n) + cubeBlock - 1) / cubeBlock;

    KernelArgs args{};
    args.m = m;
    args.n = n;
    args.k = k;
    args.vectorLength = vectorLength;
    args.values = static_cast<uint16_t*>(const_cast<void*>(values));
    args.columnIndices = const_cast<int32_t*>(columnIndices);
    args.rowOffsets = const_cast<int32_t*>(rowOffsets);
    args.b = static_cast<uint16_t*>(const_cast<void*>(b));
    args.c = static_cast<float*>(c);
    args.fftsAddress = reinterpret_cast<void*>(fftsAddressRaw);
    args.workspace = static_cast<uint16_t*>(bPackWorkspace);
    args.bPackSlotStrideElements = plan.bPackSlotStrideElements;
    args.totalBPackElements = plan.requiredBPackElements;
    args.maxVectorsPerRow = maxVectorsPerRow;
    args.numNBlocks = numNBlocks;
    args.ldc = m;
    args.tilingKey = plan.tiling.tiling_key;

    return args;
}

aclsparseStatus_t ValidateLaunch(int32_t m, int32_t n, int32_t k, int32_t vectorLength, const void* values,
                                 int64_t maxVectorsPerRow, const int32_t* columnIndices, const int32_t* rowOffsets,
                                 const void* b, void* c, void* bPackWorkspace, const SpmmLaunchPlan* prepared)
{
    if (m <= 0 || n <= 0 || k <= 0 || vectorLength != SPMM_1D_VECTOR_LENGTH || (m % SPMM_1D_VECTOR_LENGTH) != 0 ||
        maxVectorsPerRow < 0 || (maxVectorsPerRow > 0 && (values == nullptr || columnIndices == nullptr)) ||
        rowOffsets == nullptr || b == nullptr || c == nullptr || bPackWorkspace == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (prepared == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t LaunchSpmm(aclsparseHandle_t handle, int32_t m, int32_t n, int32_t k, int32_t vectorLength,
                             const void* values, int64_t maxVectorsPerRow, const int32_t* columnIndices,
                             const int32_t* rowOffsets, const void* b, void* c, void* bPackWorkspace,
                             size_t bPackWorkspaceSize, const SpmmLaunchPlan* prepared = nullptr)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
#if !SPMM_1D_SUPPORTED_SOC
    return ACL_SPARSE_STATUS_ARCH_MISMATCH;
#endif
    auto validation = ValidateLaunch(m, n, k, vectorLength, values, maxVectorsPerRow, columnIndices, rowOffsets, b, c,
                                     bPackWorkspace, prepared);
    if (validation != ACL_SPARSE_STATUS_SUCCESS) {
        return validation;
    }
    const SpmmLaunchPlan& plan = *prepared;
    aclsparseStatus_t status = ACL_SPARSE_STATUS_SUCCESS;
    if (bPackWorkspaceSize < plan.requiredWorkspaceBytes) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    aclrtStream stream = nullptr;
    status = aclsparseGetStream(handle, &stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    status = EnsureKernelRegistered(plan.tiling);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    uint64_t fftsAddressRaw = 0;
    uint32_t fftsLength = 0;
    aclError ret = rtGetC2cCtrlAddr(&fftsAddressRaw, &fftsLength);
    if (ret != ACL_ERROR_NONE || fftsAddressRaw == 0 || fftsLength == 0) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    KernelArgs args = MakeKernelArgs(m, n, k, vectorLength, values, maxVectorsPerRow, columnIndices, rowOffsets, b, c,
                                     bPackWorkspace, fftsAddressRaw, plan);

    ret = rtKernelLaunch(plan.tiling.kernel_name, static_cast<uint32_t>(plan.groupDim), &args, sizeof(args), nullptr,
                         stream);
    if (ret != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    // Submission is asynchronous; the caller owns the stream and buffer lifetime.
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

struct CubeSpmm1dInternalPlan {
    SpmmLaunchPlan launch{};
    CubeSpmm1dInternalPlanInfo info{};
};

namespace {
aclsparseStatus_t ReadOptions(const CubeSpmm1dInternalPlanOptions* options, CubeSpmm1dInternalPlanOptions& opts)
{
    if (options == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (options->structSize != sizeof(opts) ||
        (options->flags & ~(CUBE_SPMM_1D_INTERNAL_PLAN_TILE_PADDING | CUBE_SPMM_1D_INTERNAL_PLAN_RULE_PADDING |
                            CUBE_SPMM_1D_INTERNAL_PLAN_FINITE_INPUTS)) != 0 ||
        options->policy < CUBE_SPMM_1D_INTERNAL_POLICY_PHASE1 ||
        options->policy > CUBE_SPMM_1D_INTERNAL_POLICY_STRUCTURAL) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    opts = *options;
    return ACL_SPARSE_STATUS_SUCCESS;
}
aclsparseStatus_t AnalyzeRows(const int32_t* offsets, size_t count, int32_t k, int64_t& pmax, int64_t& empty)
{
    for (size_t i = 0; i + 1 < count; ++i) {
        if (offsets[i + 1] < offsets[i]) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        const int64_t p = static_cast<int64_t>(offsets[i + 1]) - offsets[i];
        if (p > k) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        pmax = std::max(pmax, p);
        empty += p == 0;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
struct Selection {
    int32_t device = -1;
    int64_t cores = 0, groups = 0;
    int64_t pmax = 0, empty = 0;
    SpmmStructuralDecision structural{};
};
aclsparseStatus_t SelectStructure(int32_t m, int32_t n, int32_t k, const int32_t* offsets, size_t count,
                                  const CubeSpmm1dInternalPlanOptions& opts, Selection& selection)
{
    auto status = AnalyzeRows(offsets, count, k, selection.pmax, selection.empty);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (aclrtGetDevice(&selection.device) != ACL_ERROR_NONE || selection.device < 0) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    status = QueryRuntimeAiCoreCount(&selection.cores);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (selection.cores > std::numeric_limits<int32_t>::max()) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    selection.groups = std::min<int64_t>(m / 16, selection.cores);
    const char* soc = aclrtGetSocName();
    const bool calibratedSoc = soc != nullptr && std::string(soc) == "Ascend910B3";
    const bool finite = (opts.flags & CUBE_SPMM_1D_INTERNAL_PLAN_FINITE_INPUTS) != 0;
    selection.structural = finite && calibratedSoc && selection.cores == 20 ?
                               SpmmSelectAdaptive(m, n, k, selection.pmax, offsets[count - 1], selection.empty, offsets,
                                                  static_cast<int>(selection.groups)) :
                               SpmmSelectStructural(m, n, k, selection.pmax, offsets[count - 1], selection.empty);
    return ACL_SPARSE_STATUS_SUCCESS;
}
aclsparseStatus_t SelectKey(int32_t n, const Selection& selection, const CubeSpmm1dInternalPlanOptions& opts, int& key)
{
    const int phase1Key = SelectSpmmTilingConfig(selection.pmax, n).tiling_key;
    key = opts.policy == CUBE_SPMM_1D_INTERNAL_POLICY_EXPLICIT ?
              opts.tilingKey :
              (opts.policy == CUBE_SPMM_1D_INTERNAL_POLICY_STRUCTURAL ? selection.structural.key : phase1Key);
    if (opts.policy != CUBE_SPMM_1D_INTERNAL_POLICY_EXPLICIT) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (SpmmFindCandidate(opts.tilingKey) == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const bool finite = (opts.flags & CUBE_SPMM_1D_INTERNAL_PLAN_FINITE_INPUTS) != 0;
    if (SpmmFusionGroup(opts.tilingKey) && (!finite || selection.pmax > 256)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
aclsparseStatus_t CompleteSelection(int32_t n, const Selection& selection, const CubeSpmm1dInternalPlanOptions& opts,
                                    SpmmLaunchPlan& bestPlan)
{
    int key = -1;
    auto status = SelectKey(n, selection, opts, key);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    const auto* candidate = SpmmFindCandidate(key);
    if (candidate == nullptr || !SpmmCandidateIsLegal(*candidate)) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (opts.allowedKeys != 0 && (opts.allowedKeys & (uint64_t(1) << key)) == 0) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    bestPlan.tiling = candidate->tiling;
    bestPlan.runtimeCoreCount = selection.cores;
    bestPlan.groupDim = selection.groups;
    const bool tilePadding = (opts.flags & CUBE_SPMM_1D_INTERNAL_PLAN_RULE_PADDING) != 0 ?
                                 (opts.policy != CUBE_SPMM_1D_INTERNAL_POLICY_STRUCTURAL ||
                                  !selection.structural.minimalPadding) :
                                 (opts.flags & CUBE_SPMM_1D_INTERNAL_PLAN_TILE_PADDING) != 0;
    bestPlan.paddedN = tilePadding ? GetSpmmPaddedN(n, bestPlan.tiling) : SpmmAlignUp(n, 16);
    if (bestPlan.paddedN <= 0 || bestPlan.paddedN > std::numeric_limits<int32_t>::max()) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    status = CompleteWorkspacePlan(selection.pmax, candidate->direct, &bestPlan);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (opts.workspaceLimitBytes > 0 && bestPlan.requiredWorkspaceBytes > opts.workspaceLimitBytes) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
void PopulateInfo(CubeSpmm1dInternalPlan& plan, int32_t m, int32_t n, int32_t k, int64_t total,
                  const Selection& selection, const CubeSpmm1dInternalPlanOptions& opts)
{
    const auto& launch = plan.launch;
    const auto& tiling = launch.tiling;
    plan.info = {m,
                 n,
                 static_cast<int32_t>(launch.paddedN),
                 k,
                 16,
                 selection.device,
                 static_cast<int32_t>(selection.cores),
                 static_cast<int32_t>(selection.groups),
                 tiling.tiling_key,
                 tiling.aic_n_batch,
                 tiling.sparse_k_tile,
                 tiling.b_pack_n_batch * 16,
                 selection.pmax,
                 total,
                 selection.empty,
                 launch.requiredWorkspaceBytes,
                 opts.policy};
}
} // namespace

aclsparseStatus_t CubeSpmm1dInternalCreatePlan(aclsparseHandle_t handle, int32_t m, int32_t n, int32_t k,
                                               const int32_t* offsets, size_t count,
                                               const CubeSpmm1dInternalPlanOptions* options,
                                               CubeSpmm1dInternalPlan_t* output)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (output == nullptr || *output != nullptr || m <= 0 || n <= 0 || k <= 0 || m % 16 != 0 || offsets == nullptr ||
        count != static_cast<size_t>(m / 16 + 1) || offsets[0] != 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    CubeSpmm1dInternalPlanOptions opts{
        sizeof(opts), CUBE_SPMM_1D_INTERNAL_POLICY_STRUCTURAL, -1, CUBE_SPMM_1D_INTERNAL_PLAN_RULE_PADDING, 0, 0};
    auto status = ReadOptions(options, opts);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    Selection selection;
    status = SelectStructure(m, n, k, offsets, count, opts, selection);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    SpmmLaunchPlan launch{};
    status = CompleteSelection(n, selection, opts, launch);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    auto* plan = new (std::nothrow) CubeSpmm1dInternalPlan;
    if (plan == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    plan->launch = launch;
    PopulateInfo(*plan, m, n, k, offsets[count - 1], selection, opts);
    *output = plan;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t CubeSpmm1dInternalGetPlanInfo(CubeSpmm1dInternalPlan_t plan, CubeSpmm1dInternalPlanInfo* info)
{
    if (plan == nullptr || info == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *info = plan->info;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t CubeSpmm1dInternalDestroyPlan(CubeSpmm1dInternalPlan_t plan)
{
    delete plan;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t CubeSpmm1dInternalExecute(aclsparseHandle_t handle, CubeSpmm1dInternalPlan_t plan, const void* values,
                                            const int32_t* columnIndices, const int32_t* rowOffsets, const void* b,
                                            void* c, void* workspace, size_t workspaceBytes)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (plan == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int32_t device = -1;
    if (aclrtGetDevice(&device) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    if (device != plan->info.deviceId) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return LaunchSpmm(handle, plan->info.m, plan->info.paddedN, plan->info.k, 16, values, plan->info.maxVectorsPerRow,
                      columnIndices, rowOffsets, b, c, workspace, workspaceBytes, &plan->launch);
}
