/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "aclsparse_handle_internal.h"
#include "internal/spmm_1d_rt.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {
bool denyAllocations = false;
int allocationsUntilFailure = -1;
bool denyAfterFunction = false;
int binaryCalls = 0, functionCalls = 0, unregisterCalls = 0, launchCalls = 0;
int binaryFailures = 0, functionFailures = 0, unregisterFailures = 0, freeFailures = 0;
void* freeTarget = nullptr;
const void* registeredData = nullptr;
aclrtContext currentContext = reinterpret_cast<aclrtContext>(0x1000);
std::map<void*, int> freeAttempts;
std::set<void*> liveAllocations;
std::string libraryPath;
int libraryLookupResult = 1;
bool nullLibraryName = false;
constexpr const char* kernelFamilies[] = {"workspace_unified", "direct_pipeline", "fusion2", "direct"};

bool FailOnce(int& remaining)
{
    if (remaining == 0) {
        return false;
    }
    --remaining;
    return true;
}
} // namespace

void* operator new(std::size_t size)
{
    if (denyAllocations || allocationsUntilFailure == 0) {
        throw std::bad_alloc();
    }
    if (allocationsUntilFailure > 0) {
        --allocationsUntilFailure;
    }
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

extern "C" aclError __wrap_aclrtMalloc(void** p, size_t size, aclrtMemMallocPolicy)
{
    if (posix_memalign(p, 32, size) != 0) {
        return 1;
    }
    freeAttempts[*p] = 0;
    liveAllocations.insert(*p);
    return ACL_SUCCESS;
}
extern "C" aclError __wrap_aclrtFree(void* p)
{
    ++freeAttempts.at(p);
    if (p == freeTarget && FailOnce(freeFailures)) {
        return 1;
    }
    if (liveAllocations.erase(p) != 1) {
        return 1;
    }
    std::free(p);
    return ACL_SUCCESS;
}
extern "C" aclError __wrap_aclrtMemcpy(void* dst, size_t, const void* src, size_t size, aclrtMemcpyKind)
{
    std::copy_n(static_cast<const unsigned char*>(src), size, static_cast<unsigned char*>(dst));
    return ACL_SUCCESS;
}
extern "C" aclError __wrap_aclrtGetCurrentContext(aclrtContext* context)
{
    *context = currentContext;
    return ACL_SUCCESS;
}
extern "C" aclError __wrap_aclrtGetDevice(int32_t* device)
{
    *device = 1;
    return ACL_SUCCESS;
}
extern "C" aclError __wrap_aclGetDeviceCapability(uint32_t, int32_t, int64_t* cores)
{
    *cores = 20;
    return ACL_SUCCESS;
}
extern "C" const char* __wrap_aclrtGetSocName() { return "Ascend910B3"; }
extern "C" aclError __wrap_aclrtMemsetAsync(void* p, size_t, int32_t value, size_t size, aclrtStream)
{
    std::fill_n(static_cast<unsigned char*>(p), size, static_cast<unsigned char>(value));
    return ACL_SUCCESS;
}
extern "C" int __wrap_dladdr(const void*, Dl_info* info)
{
    info->dli_fname = nullLibraryName ? nullptr : libraryPath.c_str();
    return libraryLookupResult;
}
extern "C" aclError __wrap_rtDevBinaryRegister(const rtDevBinary_t* binary, void** handle)
{
    ++binaryCalls;
    if (FailOnce(binaryFailures)) {
        return 1;
    }
    registeredData = binary->data;
    *handle = reinterpret_cast<void*>(uintptr_t(0x2000 + binaryCalls));
    return ACL_SUCCESS;
}
extern "C" aclError __wrap_rtFunctionRegister(void*, const void*, const char*, const void*, uint32_t)
{
    ++functionCalls;
    if (FailOnce(functionFailures)) {
        return 1;
    }
    denyAllocations = denyAfterFunction;
    return ACL_SUCCESS;
}
extern "C" aclError __wrap_rtDevBinaryUnRegister(void*)
{
    ++unregisterCalls;
    // The registered backing bytes must survive until rollback actually succeeds.
    if (*static_cast<const unsigned char*>(registeredData) != 0x7f) {
        return 1;
    }
    return FailOnce(unregisterFailures) ? 1 : ACL_SUCCESS;
}
extern "C" aclError __wrap_rtGetC2cCtrlAddr(uint64_t* address, uint32_t* length)
{
    *address = 0x4000;
    *length = 8;
    return ACL_SUCCESS;
}
extern "C" aclError __wrap_rtKernelLaunch(const void*, uint32_t, void*, uint32_t, void*, aclrtStream)
{
    ++launchCalls;
    return ACL_SUCCESS;
}
extern "C" aclsparseStatus_t aclsparseGetStream(aclsparseHandle_t handle, aclrtStream* stream)
{
    *stream = handle->stream;
    return ACL_SPARSE_STATUS_SUCCESS;
}

namespace {
struct FreeHostBuffer {
    void operator()(void* p) const { std::free(p); }
};
using HostBuffer = std::unique_ptr<void, FreeHostBuffer>;

HostBuffer AlignedBuffer(size_t bytes)
{
    void* p = nullptr;
    if (posix_memalign(&p, 32, bytes) != 0) {
        throw std::bad_alloc();
    }
    return HostBuffer(p);
}

class CubeSpmm1dFault : public ::testing::Test {
protected:
    void SetUp() override
    {
        binaryCalls = functionCalls = unregisterCalls = launchCalls = 0;
        binaryFailures = functionFailures = unregisterFailures = freeFailures = 0;
        freeTarget = nullptr;
        denyAllocations = denyAfterFunction = false;
        allocationsUntilFailure = -1;
        currentContext = reinterpret_cast<aclrtContext>(0x1000);
        ASSERT_TRUE(liveAllocations.empty());
        freeAttempts.clear();
        libraryLookupResult = 1;
        nullLibraryName = false;
        char temporary[] = "/tmp/cube spmm_1d_fault_XXXXXX";
        ASSERT_NE(mkdtemp(temporary), nullptr);
        directory = temporary;
        ASSERT_EQ(mkdir((directory + "/cube_spmm_1d_kernels").c_str(), 0700), 0);
        libraryPath = directory + "/libops_sparse.so";
        for (const char* family : kernelFamilies) {
            std::ofstream file(directory + "/cube_spmm_1d_kernels/spmm_1d_kernel_" + family + ".o", std::ios::binary);
            file.put(0x7f);
            ASSERT_TRUE(file.good());
        }
    }
    void TearDown() override
    {
        denyAllocations = false;
        allocationsUntilFailure = -1;
        freeFailures = 0;
        if (plan != nullptr) {
            EXPECT_EQ(aclsparseCubeSpmm1dDestroyPlan(plan), ACL_SPARSE_STATUS_SUCCESS);
        }
        EXPECT_TRUE(liveAllocations.empty());
        if (!directory.empty()) {
            for (const char* family : kernelFamilies) {
                EXPECT_EQ(unlink((directory + "/cube_spmm_1d_kernels/spmm_1d_kernel_" + family + ".o").c_str()), 0);
            }
            EXPECT_EQ(rmdir((directory + "/cube_spmm_1d_kernels").c_str()), 0);
            EXPECT_EQ(rmdir(directory.c_str()), 0);
        }
    }
    void Prepare(int key)
    {
        const int32_t rows[] = {0, 1}, cols[] = {0};
        std::array<uint16_t, 16> values;
        values.fill(0x3c00);
        aclsparseCubeSpmm1dOptions options{sizeof(options), key, 0, 0};
        ASSERT_EQ(aclsparseCubeSpmm1dCreatePlan(&handle, 16, 16, 16, rows, 2, cols, values.data(), 1, &options, &plan),
                  ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclsparseCubeSpmm1dGetPlanInfo(plan, &info), ACL_SPARSE_STATUS_SUCCESS);
        b = AlignedBuffer(info.bBytes);
        c = AlignedBuffer(info.cBytes);
        workspace = AlignedBuffer(info.workspaceBytes);
    }
    aclsparseStatus_t Execute()
    {
        return aclsparseCubeSpmm1dExecute(&handle, plan, b.get(), info.bBytes, c.get(), info.cBytes,
                                          workspace.get(), info.workspaceBytes);
    }
    aclsparseContext handle;
    aclsparseCubeSpmm1dPlan_t plan = nullptr;
    aclsparseCubeSpmm1dPlanInfo info{};
    HostBuffer b, c, workspace;
    std::string directory;
};

TEST_F(CubeSpmm1dFault, BinaryLookupFailureAndRetry)
{
    Prepare(9);
    libraryLookupResult = 0;
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    libraryLookupResult = 1;
    nullLibraryName = true;
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    nullLibraryName = false;
    libraryPath = directory + "/missing/libops_sparse.so";
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    libraryPath = directory + "/libops_sparse.so";
    const std::string binary = directory + "/cube_spmm_1d_kernels/spmm_1d_kernel_direct.o";
    ASSERT_EQ(unlink(binary.c_str()), 0);
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    ASSERT_EQ(mkdir(binary.c_str(), 0700), 0);
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    ASSERT_EQ(rmdir(binary.c_str()), 0);
    {
        std::ofstream empty(binary, std::ios::binary);
        ASSERT_TRUE(empty.good());
    }
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    EXPECT_EQ(binaryCalls, 0);
    ASSERT_EQ(unlink(binary.c_str()), 0);
    // stat follows a valid symlink, matching the previous regular-file check.
    ASSERT_EQ(symlink("spmm_1d_kernel_workspace_unified.o", binary.c_str()), 0);
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(binaryCalls, 1);
    EXPECT_EQ(launchCalls, 1);
}

TEST_F(CubeSpmm1dFault, RegistrationFailureRollbackAndRetry)
{
    Prepare(34);
    binaryFailures = 1;
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    EXPECT_EQ(functionCalls, 0);
    functionFailures = 1;
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    EXPECT_EQ(unregisterCalls, 1);
    functionFailures = 1;
    unregisterFailures = 2;
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    const int registered = binaryCalls;
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_EXECUTION_FAILED);
    EXPECT_EQ(binaryCalls, registered) << "do not re-register while rollback is pending";
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(binaryCalls, registered + 1);
    EXPECT_EQ(unregisterCalls, 4);
    EXPECT_EQ(launchCalls, 1);
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(binaryCalls, registered + 1) << "successful registration must be cached";
}

TEST_F(CubeSpmm1dFault, AllocationFailureBeforeRegistrationCanRetry)
{
    Prepare(39);
    bool completed = false;
    for (int budget = 0; budget < 128; ++budget) {
        allocationsUntilFailure = budget;
        const auto status = Execute();
        allocationsUntilFailure = -1;
        if (status == ACL_SPARSE_STATUS_SUCCESS) {
            completed = true;
            break;
        }
        // iostream may translate an internal allocation failure into failbit.
        ASSERT_TRUE(status == ACL_SPARSE_STATUS_ALLOC_FAILED || status == ACL_SPARSE_STATUS_EXECUTION_FAILED)
            << "allocation budget=" << budget << " status=" << status;
        ASSERT_EQ(binaryCalls, 0) << "cache allocation must precede runtime registration";
    }
    EXPECT_TRUE(completed);
    EXPECT_EQ(binaryCalls, 1);
}

TEST_F(CubeSpmm1dFault, NoAllocationAfterFunctionRegistration)
{
    Prepare(48);
    denyAfterFunction = true;
    const auto status = Execute();
    denyAllocations = false;
    denyAfterFunction = false;
    EXPECT_EQ(status, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(functionCalls, 1);
    EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(binaryCalls, 1);
}

TEST_F(CubeSpmm1dFault, DestroyRetainsOnlyFailedBuffersAndCanRetry)
{
    for (int failedBuffer = 0; failedBuffer < 3; ++failedBuffer) {
        freeAttempts.clear();
        Prepare(34);
        ASSERT_EQ(liveAllocations.size(), 3U);
        freeTarget = *std::next(liveAllocations.begin(), failedBuffer);
        freeFailures = 2;
        EXPECT_EQ(aclsparseCubeSpmm1dDestroyPlan(plan), ACL_SPARSE_STATUS_EXECUTION_FAILED);
        EXPECT_EQ(liveAllocations.size(), 1U);
        EXPECT_EQ(aclsparseCubeSpmm1dGetPlanInfo(plan, &info), ACL_SPARSE_STATUS_INVALID_VALUE);
        EXPECT_EQ(Execute(), ACL_SPARSE_STATUS_INVALID_VALUE);
        EXPECT_EQ(aclsparseCubeSpmm1dDestroyPlan(plan), ACL_SPARSE_STATUS_EXECUTION_FAILED);
        EXPECT_EQ(aclsparseCubeSpmm1dDestroyPlan(plan), ACL_SPARSE_STATUS_SUCCESS);
        plan = nullptr;
        EXPECT_EQ(freeAttempts.at(freeTarget), 3);
        for (const auto& entry : freeAttempts) {
            EXPECT_EQ(entry.second, entry.first == freeTarget ? 3 : 1);
        }
    }
}

TEST_F(CubeSpmm1dFault, WrongContextDoesNotStartDestroy)
{
    Prepare(34);
    currentContext = reinterpret_cast<aclrtContext>(0x1001);
    EXPECT_EQ(aclsparseCubeSpmm1dDestroyPlan(plan), ACL_SPARSE_STATUS_INVALID_VALUE);
    currentContext = reinterpret_cast<aclrtContext>(0x1000);
    EXPECT_EQ(aclsparseCubeSpmm1dGetPlanInfo(plan, &info), ACL_SPARSE_STATUS_SUCCESS);
    for (const auto& entry : freeAttempts) {
        EXPECT_EQ(entry.second, 0);
    }
}
} // namespace
