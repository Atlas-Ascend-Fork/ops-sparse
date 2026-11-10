/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_DEVICE_BUFFER_H_
#define SPSV_ARCH22_DEVICE_BUFFER_H_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "securec.h"
#include "test_common.h"

namespace spsv_test {
template <class To, class From>
To BitCopy(const From& value)
{
    static_assert(sizeof(To) == sizeof(From), "Bit copies require equal sizes");
    static_assert(std::is_trivially_copyable_v<To> && std::is_trivially_copyable_v<From>,
                  "Bit copies require trivially copyable types");
    To result{};
    if (memcpy_s(&result, sizeof(result), &value, sizeof(value)) != EOK) {
        throw std::runtime_error("SpSV bit copy failed");
    }
    return result;
}

inline void CheckAcl(aclError status, const char *what)
{
    if (status != ACL_SUCCESS) throw std::runtime_error(std::string(what) + ": " + std::to_string(status));
}
inline void CheckSparse(aclsparseStatus_t status, const char *what)
{
    if (status != ACL_SPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": " + std::to_string(status));
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes = 0) : bytes_(bytes)
    {
        if (bytes != 0) CheckAcl(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~DeviceBuffer()
    {
        if (data_ != nullptr)
            aclrtFree(data_);
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    void *get() const { return data_; }
    aclError FreeNow()
    {
        if (data_ == nullptr) return ACL_SUCCESS;
        const auto status = aclrtFree(data_);
        if (status == ACL_SUCCESS) {
            data_ = nullptr;
            bytes_ = 0;
        }
        return status;
    }
    void Put(const void *host, size_t bytes)
    {
        if (bytes != 0) CheckAcl(aclrtMemcpy(data_, bytes_, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    }
    template<class T> void Put(const std::vector<T> &host) { Put(host.data(), host.size() * sizeof(T)); }
    template<class T> std::vector<T> Get(size_t count) const
    {
        std::vector<T> host(count);
        if (count != 0) CheckAcl(aclrtMemcpy(host.data(), count * sizeof(T), data_, count * sizeof(T),
            ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        return host;
    }
private:
    void *data_ = nullptr;
    size_t bytes_ = 0;
};
} // namespace spsv_test
#endif
