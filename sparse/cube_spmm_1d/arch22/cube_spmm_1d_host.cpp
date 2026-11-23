/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "internal/plan_internal.h"
#include "internal/csr_conversion.h"
#include "log/log.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint64_t kPlanSignature = 0x435542453144504cULL;
struct DeviceArray {
    void* data = nullptr;
    size_t bytes = 0;
    ~DeviceArray()
    {
        if (Release() != ACL_SUCCESS) {
            OP_LOGE("CubeSpmm1d", "Failed to release device A during preparation cleanup");
        }
    }
    aclError Release()
    {
        if (data == nullptr) {
            return ACL_SUCCESS;
        }
        const aclError status = aclrtFree(data);
        if (status == ACL_SUCCESS) {
            data = nullptr;
            bytes = 0;
        }
        return status;
    }
    bool Upload(const void* source, size_t size)
    {
        if (size == 0) {
            return true;
        }
        if (aclrtMalloc(&data, size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
            return false;
        }
        bytes = size;
        return aclrtMemcpy(data, bytes, source, size, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    }
};

aclsparseStatus_t CheckHandle(aclsparseHandle_t handle)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    return handle->signature == kAclSparseHandleSignature ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_INVALID_VALUE;
}

template <class F>
aclsparseStatus_t Protect(F function) noexcept
{
    try {
        const auto status = function();
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            OP_LOGE("CubeSpmm1d", "API rejected input or runtime operation, status=%d", static_cast<int>(status));
        }
        return status;
    } catch (const std::bad_alloc&) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    } catch (const std::overflow_error&) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    } catch (const std::invalid_argument&) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    } catch (...) {
        OP_LOGE("CubeSpmm1d", "Host preparation failed");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
}

bool Overlap(const void* first, size_t firstBytes, const void* second, size_t secondBytes)
{
    const uintptr_t a = reinterpret_cast<uintptr_t>(first);
    const uintptr_t b = reinterpret_cast<uintptr_t>(second);
    if (firstBytes > UINTPTR_MAX - a || secondBytes > UINTPTR_MAX - b) {
        return true;
    }
    return firstBytes != 0 && secondBytes != 0 && a < b + secondBytes && b < a + firstBytes;
}
} // namespace

struct aclsparseCubeSpmm1dPlan {
    uint64_t signature = kPlanSignature;
    CubeSpmm1dInternalPlan_t kernel = nullptr;
    aclsparseCubeSpmm1dPlanInfo info{};
    aclrtContext context = nullptr;
    bool destroying = false;
    DeviceArray offsets, columns, values;
    ~aclsparseCubeSpmm1dPlan() { (void)CubeSpmm1dInternalDestroyPlan(kernel); }
};

namespace {
aclsparseStatus_t CheckPrepare(aclsparseHandle_t handle, int32_t m, int32_t n, int32_t k,
                               const aclsparseCubeSpmm1dOptions* options, aclsparseCubeSpmm1dPlan_t* output)
{
    const auto status = CheckHandle(handle);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (output == nullptr || *output != nullptr || m <= 0 || m > INT32_MAX - 15 || n <= 0 || k <= 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (options != nullptr &&
        (options->structSize != sizeof(*options) || options->reserved != 0 || options->tilingKey < -1)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

CubeSpmm1dInternalPlanOptions InternalOptions(const aclsparseCubeSpmm1dOptions* options)
{
    CubeSpmm1dInternalPlanOptions opts{
        sizeof(opts),
        CUBE_SPMM_1D_INTERNAL_POLICY_STRUCTURAL,
        -1,
        CUBE_SPMM_1D_INTERNAL_PLAN_RULE_PADDING | CUBE_SPMM_1D_INTERNAL_PLAN_FINITE_INPUTS,
        0,
        0};
    if (options != nullptr) {
        opts.tilingKey = options->tilingKey;
        opts.workspaceLimitBytes = options->workspaceLimitBytes;
        if (opts.tilingKey >= 0) {
            opts.policy = CUBE_SPMM_1D_INTERNAL_POLICY_EXPLICIT;
        }
    }
    return opts;
}

aclsparseStatus_t Prepare(aclsparseHandle_t handle, const ascspmm::VectorCsr& a, int32_t n,
                          const aclsparseCubeSpmm1dOptions* options, aclsparseCubeSpmm1dPlan_t* output)
{
    if (a.colidx.size() > static_cast<size_t>(INT32_MAX / 16)) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    std::unique_ptr<aclsparseCubeSpmm1dPlan> plan(new (std::nothrow) aclsparseCubeSpmm1dPlan);
    if (!plan) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    if (aclrtGetCurrentContext(&plan->context) != ACL_SUCCESS || plan->context == nullptr) {
        return ACL_SPARSE_STATUS_NOT_INITIALIZED;
    }
    const auto opts = InternalOptions(options);
    auto status = CubeSpmm1dInternalCreatePlan(handle, a.mpad, n, a.k, a.rowptr.data(), a.rowptr.size(), &opts,
                                               &plan->kernel);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    CubeSpmm1dInternalPlanInfo info{};
    status = CubeSpmm1dInternalGetPlanInfo(plan->kernel, &info);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    plan->info = {a.m,
                  n,
                  a.k,
                  a.mpad,
                  info.paddedN,
                  info.tilingKey,
                  info.blockDim,
                  info.deviceId,
                  static_cast<int64_t>(a.colidx.size()),
                  ascspmm::checked_bytes(uint64_t(a.k) * info.paddedN, 2),
                  ascspmm::checked_bytes(uint64_t(a.mpad) * info.paddedN, 4),
                  info.workspaceBytes,
                  0};
    if (!plan->offsets.Upload(a.rowptr.data(), ascspmm::checked_bytes(a.rowptr.size(), 4)) ||
        !plan->columns.Upload(a.colidx.data(), ascspmm::checked_bytes(a.colidx.size(), 4)) ||
        !plan->values.Upload(a.values.data(), ascspmm::checked_bytes(a.values.size(), 2))) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    plan->info.persistentABytes = plan->offsets.bytes + plan->columns.bytes + plan->values.bytes;
    OP_LOGD("CubeSpmm1d", "Prepared M=%d N=%d K=%d key=%d blocks=%d", a.m, n, a.k, plan->info.tilingKey,
            plan->info.blockDim);
    *output = plan.release();
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateAliases(aclsparseCubeSpmm1dPlan_t plan, const void* b, size_t nb, void* c, size_t nc,
                                  void* workspace, size_t nw)
{
    if (Overlap(b, nb, c, nc) || Overlap(b, nb, workspace, nw) || Overlap(c, nc, workspace, nw)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const DeviceArray* arrays[] = {&plan->offsets, &plan->columns, &plan->values};
    for (const auto* a : arrays) {
        if (Overlap(b, nb, a->data, a->bytes) || Overlap(c, nc, a->data, a->bytes) ||
            Overlap(workspace, nw, a->data, a->bytes)) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateExecute(aclsparseHandle_t handle, aclsparseCubeSpmm1dPlan_t plan, const void* b, size_t nb,
                                  void* c, size_t nc, void* workspace, size_t nw)
{
    const auto status = CheckHandle(handle);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (plan == nullptr || plan->signature != kPlanSignature || plan->destroying ||
        b == nullptr || c == nullptr || workspace == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const auto& q = plan->info;
    if (nb < q.bBytes || nc < q.cBytes || nw < q.workspaceBytes) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (reinterpret_cast<uintptr_t>(b) % 32 != 0 || reinterpret_cast<uintptr_t>(c) % 32 != 0 ||
        reinterpret_cast<uintptr_t>(workspace) % 32 != 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclrtContext context = nullptr;
    if (aclrtGetCurrentContext(&context) != ACL_SUCCESS || context != plan->context) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ValidateAliases(plan, b, nb, c, nc, workspace, nw);
}

} // namespace

extern "C" __attribute__((visibility("default"))) aclsparseStatus_t
aclsparseCubeSpmm1dCreatePlan(aclsparseHandle_t handle, int32_t m, int32_t n, int32_t k, const int32_t* offsets,
                              size_t count, const int32_t* columns, const uint16_t* values, size_t vectors,
                              const aclsparseCubeSpmm1dOptions* options, aclsparseCubeSpmm1dPlan_t* output)
{
    return Protect([&]() {
        auto status = CheckPrepare(handle, m, n, k, options, output);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        if (m % 16 != 0 || offsets == nullptr || count != static_cast<size_t>(m / 16 + 1) ||
            vectors > static_cast<size_t>(INT32_MAX / 16) || offsets[0] != 0 ||
            offsets[count - 1] != static_cast<int64_t>(vectors) ||
            (vectors != 0 && (columns == nullptr || values == nullptr))) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        ascspmm::VectorCsr a;
        a.m = m;
        a.mpad = m;
        a.k = k;
        a.rowptr.assign(offsets, offsets + count);
        for (size_t row = 1; row < count; ++row) {
            if (offsets[row] < offsets[row - 1] || offsets[row] > static_cast<int64_t>(vectors)) {
                return ACL_SPARSE_STATUS_INVALID_VALUE;
            }
            int32_t previous = -1;
            for (int32_t p = offsets[row - 1]; p < offsets[row]; ++p) {
                if (columns[p] <= previous || columns[p] >= k) {
                    return ACL_SPARSE_STATUS_INVALID_VALUE;
                }
                previous = columns[p];
            }
        }
        for (size_t i = 0; i < vectors * 16; ++i) {
            if ((values[i] & 0x7c00) == 0x7c00) {
                return ACL_SPARSE_STATUS_INVALID_VALUE;
            }
        }
        if (vectors != 0) {
            a.colidx.assign(columns, columns + vectors);
            a.values.assign(values, values + vectors * 16);
        }
        return Prepare(handle, a, n, options, output);
    });
}

extern "C" __attribute__((visibility("default"))) aclsparseStatus_t
aclsparseCubeSpmm1dCreateCsrPlan(aclsparseHandle_t handle, int32_t m, int32_t n, int32_t k, const int64_t* offsets,
                                 size_t count, const int64_t* columns, const uint16_t* values, size_t nnz,
                                 const aclsparseCubeSpmm1dOptions* options, aclsparseCubeSpmm1dPlan_t* output)
{
    return Protect([&]() {
        auto status = CheckPrepare(handle, m, n, k, options, output);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        if (offsets == nullptr || count != static_cast<size_t>(m) + 1 || nnz > INT64_MAX ||
            (nnz != 0 && (columns == nullptr || values == nullptr))) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        if (offsets[0] != 0 || offsets[count - 1] != static_cast<int64_t>(nnz)) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        ascspmm::Csr csr;
        csr.m = m;
        csr.k = k;
        csr.rowptr.assign(offsets, offsets + count);
        if (nnz != 0) {
            csr.colidx.assign(columns, columns + nnz);
        }
        csr.values.reserve(nnz);
        for (size_t i = 0; i < nnz; ++i) {
            csr.values.push_back(ascspmm::from_half(values[i]));
        }
        return Prepare(handle, ascspmm::convert(csr), n, options, output);
    });
}

extern "C" __attribute__((visibility("default"))) aclsparseStatus_t
aclsparseCubeSpmm1dGetPlanInfo(aclsparseCubeSpmm1dPlan_t plan, aclsparseCubeSpmm1dPlanInfo* info)
{
    if (plan == nullptr || plan->signature != kPlanSignature || plan->destroying || info == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *info = plan->info;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) aclsparseStatus_t
aclsparseCubeSpmm1dExecute(aclsparseHandle_t handle, aclsparseCubeSpmm1dPlan_t plan, const void* b, size_t nb, void* c,
                           size_t nc, void* workspace, size_t nw)
{
    return Protect([&]() {
        auto status = ValidateExecute(handle, plan, b, nb, c, nc, workspace, nw);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        aclrtStream stream = nullptr;
        status = aclsparseGetStream(handle, &stream);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        if (aclrtMemsetAsync(c, plan->info.cBytes, 0, plan->info.cBytes, stream) != ACL_SUCCESS) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        if (plan->info.vectors == 0) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        return CubeSpmm1dInternalExecute(handle, plan->kernel, plan->values.data,
                                         static_cast<int32_t*>(plan->columns.data),
                                         static_cast<int32_t*>(plan->offsets.data), b, c, workspace, nw);
    });
}

extern "C" __attribute__((visibility("default"))) aclsparseStatus_t
aclsparseCubeSpmm1dDestroyPlan(aclsparseCubeSpmm1dPlan_t plan)
{
    if (plan == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (plan->signature != kPlanSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclrtContext context = nullptr;
    if (aclrtGetCurrentContext(&context) != ACL_SUCCESS || context != plan->context) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    plan->destroying = true;
    // Try every allocation. Successful releases are cleared, failed ones remain
    // owned by this plan; only DestroyPlan may be retried after partial teardown.
    const aclError offsetsStatus = plan->offsets.Release();
    const aclError columnsStatus = plan->columns.Release();
    const aclError valuesStatus = plan->values.Release();
    if (offsetsStatus != ACL_SUCCESS || columnsStatus != ACL_SUCCESS || valuesStatus != ACL_SUCCESS) {
        OP_LOGE("CubeSpmm1d", "Device A release failed: offsets=%d columns=%d values=%d",
                offsetsStatus, columnsStatus, valuesStatus);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    plan->signature = 0;
    delete plan;
    return ACL_SPARSE_STATUS_SUCCESS;
}
