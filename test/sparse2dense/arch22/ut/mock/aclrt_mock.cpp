// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ============================================================================
// aclrt_mock.cpp — aclrt 运行时桩实现（Host 内存语义，真实运行面与签名同 acl/acl_rt.h）。
// 仅实现 op_host 源码引用的 4 个符号；UT 二进制不链接 libascendcl，符号由本桩闭合。
#include "aclrt_mock.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace s2d_ut_mock {
namespace {
std::mutex g_mu;
std::vector<MemcpyRecord> g_memcpy;
std::vector<HostAllocRecord> g_alloc;
}  // namespace

void ResetLogs()
{
    std::lock_guard<std::mutex> lk(g_mu);
    g_memcpy.clear();
    g_alloc.clear();
}

const std::vector<MemcpyRecord> &MemcpyLog()
{
    return g_memcpy;
}

const std::vector<HostAllocRecord> &HostAllocLog()
{
    return g_alloc;
}

bool InAnyHostAlloc(const void *p, size_t len)
{
    const uintptr_t q = reinterpret_cast<uintptr_t>(p);
    for (const auto &a : g_alloc) {
        const uintptr_t b = reinterpret_cast<uintptr_t>(a.ptr);
        if (q >= b && q + len <= b + a.bytes) {
            return true;
        }
    }
    return false;
}

void RecordMemcpy(void *dst, const void *src, size_t bytes, int kind)
{
    std::lock_guard<std::mutex> lk(g_mu);
    g_memcpy.push_back(MemcpyRecord{dst, src, bytes, kind});
}

void RecordAlloc(void *ptr, size_t bytes)
{
    std::lock_guard<std::mutex> lk(g_mu);
    g_alloc.push_back(HostAllocRecord{ptr, bytes});
}

}  // namespace s2d_ut_mock

extern "C" {

// 签名与 acl_rt.h L2119 一致
aclError aclrtMallocHost(void **hostPtr, size_t size)
{
    if (hostPtr == nullptr || size == 0UL) {
        return ACL_ERROR_INVALID_PARAM;
    }
    void *p = nullptr;
    if (posix_memalign(&p, 64UL, size) != 0 || p == nullptr) {
        return ACL_ERROR_INTERNAL_ERROR;  // 资源不足（与真实接口失败路径同语义）
    }
    *hostPtr = p;
    s2d_ut_mock::RecordAlloc(p, size);
    return ACL_SUCCESS;
}

// 签名与 acl_rt.h L2152 一致
aclError aclrtFreeHost(void *hostPtr)
{
    free(hostPtr);
    return ACL_SUCCESS;
}

// 签名与 acl_rt.h L2249 一致；UT 无设备，D2H/D2D 均按 Host 直拷承接（校验链只用 D2H）。
// 拷贝以 std::copy 承接（长度已校验 count <= destMax、无重叠，位元等价 memcpy）。
aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
    aclrtMemcpyKind kind, aclrtStream stream)
{
    (void)stream;
    if (dst == nullptr || src == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (count > destMax) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (count != 0UL) {
        const char *s = static_cast<const char *>(src);
        std::copy(s, s + count, static_cast<char *>(dst));
    }
    s2d_ut_mock::RecordMemcpy(dst, src, count, static_cast<int>(kind));
    return ACL_SUCCESS;
}

// 签名与 acl_rt.h L2755 一致；UT 同步即返回（拷贝已在 Host 内完成）
aclError aclrtSynchronizeStream(aclrtStream stream)
{
    (void)stream;
    return ACL_SUCCESS;
}

}  // extern "C"
