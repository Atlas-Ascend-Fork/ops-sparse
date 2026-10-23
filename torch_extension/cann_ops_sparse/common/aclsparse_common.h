// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

#ifndef CANN_OPS_SPARSE_ACLSPARSE_COMMON_H_
#define CANN_OPS_SPARSE_ACLSPARSE_COMMON_H_

// CANN's ACL base header must precede torch_npu headers.  torch_npu may carry
// an older ACL copy which otherwise defines the same types a second time.
#include <acl/acl_base.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "cann_ops_sparse.h"

namespace cann_ops_sparse {

inline void CheckAclSparseStatus(aclsparseStatus_t status, const char *operation)
{
    TORCH_CHECK(status == ACL_SPARSE_STATUS_SUCCESS, operation,
        " failed with ACLSparse status ", static_cast<int>(status));
}

#define ACLSPARSE_CHECK(expression) \
    ::cann_ops_sparse::CheckAclSparseStatus((expression), #expression)

class AclSparseHandleEntry {
public:
    explicit AclSparseHandleEntry(aclrtStream stream)
    {
        CheckAclSparseStatus(aclsparseCreate(&handle_), "aclsparseCreate");
        CheckAclSparseStatus(aclsparseSetStream(handle_, stream), "aclsparseSetStream");
        CheckAclSparseStatus(aclsparseSetPointerMode(handle_, ACL_SPARSE_POINTER_MODE_HOST),
            "aclsparseSetPointerMode");
    }

    ~AclSparseHandleEntry()
    {
        if (handle_ != nullptr) {
            (void)aclsparseDestroy(handle_);
        }
    }

    AclSparseHandleEntry(const AclSparseHandleEntry &) = delete;
    AclSparseHandleEntry &operator=(const AclSparseHandleEntry &) = delete;

    aclsparseHandle_t handle() const
    {
        return handle_;
    }

    std::mutex mutex;

private:
    aclsparseHandle_t handle_ = nullptr;
};

class AclSparseHandleCache {
public:
    static std::shared_ptr<AclSparseHandleEntry> Acquire(int device_index, aclrtStream stream)
    {
        const Key key {device_index, reinterpret_cast<uintptr_t>(stream)};
        std::lock_guard<std::mutex> lock(CacheMutex());
        const auto found = Cache().find(key);
        if (found != Cache().end()) {
            return found->second;
        }

        auto entry = std::make_shared<AclSparseHandleEntry>(stream);
        Cache().emplace(key, entry);
        return entry;
    }

    // The builder calls this from Python's exit callbacks, before torch_npu
    // destroys its streams and resets devices. Static destruction is too late
    // because aclsparseDestroy synchronizes the borrowed stream and frees ACL
    // memory. Select each handle's device while that runtime is still alive.
    static void Clear()
    {
        std::lock_guard<std::mutex> lock(CacheMutex());
        for (auto &entry : Cache()) {
            c10::DeviceGuard guard(c10::Device(c10::DeviceType::PrivateUse1,
                static_cast<c10::DeviceIndex>(entry.first.device_index)));
            entry.second.reset();
        }
        Cache().clear();
    }

private:
    struct Key {
        int device_index;
        uintptr_t stream;

        bool operator<(const Key &other) const
        {
            return device_index != other.device_index ? device_index < other.device_index : stream < other.stream;
        }
    };

    static std::map<Key, std::shared_ptr<AclSparseHandleEntry>> &Cache()
    {
        static std::map<Key, std::shared_ptr<AclSparseHandleEntry>> cache;
        return cache;
    }

    static std::mutex &CacheMutex()
    {
        static std::mutex mutex;
        return mutex;
    }
};

class AclSparseContext {
public:
    explicit AclSparseContext(const at::Tensor &input)
        : device_index_(input.device().index())
    {
        TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
            "ACLSparse only supports NPU tensors, but got ", input.device());
        TORCH_CHECK(device_index_ >= 0,
            "ACLSparse requires an indexed NPU device, but got ", input.device());
        device_guard_.reset_device(input.device());
    }

    AclSparseContext(const AclSparseContext &) = delete;
    AclSparseContext &operator=(const AclSparseContext &) = delete;

    void Initialize()
    {
        stream_.emplace(c10_npu::getCurrentNPUStream(device_index_));
        handle_entry_ = AclSparseHandleCache::Acquire(device_index_, stream_->stream());
        // ACLSparse does not publish a concurrent-use contract for one handle.
        // Hold this lock until the wrapper finishes all calls through the handle.
        handle_lock_ = std::unique_lock<std::mutex>(handle_entry_->mutex);
    }

    aclsparseHandle_t handle() const
    {
        TORCH_CHECK(handle_entry_ != nullptr, "ACLSparse context is not initialized");
        return handle_entry_->handle();
    }

    void RecordTensor(const at::Tensor &tensor) const
    {
        if (!tensor.defined()) {
            return;
        }
        TORCH_CHECK(stream_.has_value(), "ACLSparse context is not initialized");
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), *stream_);
    }

    template <typename... Tensors>
    void RecordTensors(const Tensors &...tensors) const
    {
        (RecordTensor(tensors), ...);
    }

private:
    int device_index_;
    c10::OptionalDeviceGuard device_guard_;
    std::optional<c10_npu::NPUStream> stream_;
    std::shared_ptr<AclSparseHandleEntry> handle_entry_;
    std::unique_lock<std::mutex> handle_lock_;
};

class AclSparseWorkspace {
public:
    AclSparseWorkspace(AclSparseContext &context, const at::Tensor &reference, size_t size)
        : context_(&context)
    {
        if (size == 0) {
            return;
        }
        workspace_ = at::empty({static_cast<int64_t>(size)}, reference.options().dtype(at::kByte));
    }

    ~AclSparseWorkspace()
    {
        context_->RecordTensor(workspace_);
    }

    void *data() const
    {
        return workspace_.defined() ? workspace_.data_ptr() : nullptr;
    }

private:
    AclSparseContext *context_;
    at::Tensor workspace_;
};

template <typename Callable>
decltype(auto) RunAclSparse(const at::Tensor &input, Callable &&callable)
{
    AclSparseContext context(input);
    context.Initialize();
    return std::forward<Callable>(callable)(context);
}

} // namespace cann_ops_sparse

#endif // CANN_OPS_SPARSE_ACLSPARSE_COMMON_H_
