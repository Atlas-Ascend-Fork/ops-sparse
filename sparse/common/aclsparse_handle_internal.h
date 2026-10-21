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

/*!
 * \file aclsparse_handle_internal.h
 * \brief ops-sparse handle 内部结构定义（不对外暴露）。
 */

#pragma once

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>

// Opaque-handle guard.  The public type is a void pointer, so every backend
// must reject stale/arbitrary pointers before dereferencing the internal
// fields.  Keep this value private to the implementation; it is not an ABI
// field exposed to callers.
constexpr uint32_t kAclSparseHandleSignature = 0xAC1F9501U;

#include "cann_ops_sparse.h"

/* ========== 版本信息 ========== */
#define ACLSPARSE_VERSION_MAJOR 1
#define ACLSPARSE_VERSION_MINOR 0
#define ACLSPARSE_VERSION_PATCH 0

/**
 * @brief 将版本号 (major, minor, patch) 编码为整型。
 *
 * 编码规则：MAJOR * 10000 + MINOR * 100 + PATCH，
 * 例如 1.0.0 -> 10000、1.2.3 -> 10203。
 */
#define ACLSPARSE_MAKE_VERSION(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))

#define ACLSPARSE_VERSION \
    ACLSPARSE_MAKE_VERSION(ACLSPARSE_VERSION_MAJOR, ACLSPARSE_VERSION_MINOR, ACLSPARSE_VERSION_PATCH)

/**
 * @brief Default initial workspace size: 4 MiB.
 */
constexpr size_t ACLSPARSE_DEFAULT_WORKSPACE_SIZE = 4U * 1024U * 1024U;

/**
 * @brief ops-sparse handle 内部结构体
 *
 * 仅包含 kernel 执行所需的最小化字段：
 *   - stream：当前 stream，用于 kernel 异步执行
 *   - pointerMode：指针模式（HOST 或 DEVICE）
 *   - workspace：内部 workspace，供算子临时存储
 *
 * 该结构体定义仅在实现文件中可见，对外部用户完全不可见。
 * 对外接口使用 void* / void** 作为 handle 类型。
 */
struct aclsparseContext {
    uint32_t signature = kAclSparseHandleSignature;
    /* ========== Stream ========== */
    aclrtStream stream = nullptr;     ///< 当前 stream，用于 kernel 执行
    // Monotonically increasing generation.  A descriptor's preprocess result
    // is tied to both the stream and this generation, so switching A->B->A
    // cannot accidentally revive metadata produced on the old stream epoch.
    uint64_t stream_epoch = 0;

    /* ========== Pointer Mode ========== */
    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;

    /* ========== Default workspace (library managed) ========== */
    void* default_workspace = nullptr;
    size_t default_workspace_size = 0;

    /* ========== User workspace (user managed) ========== */
    void* user_workspace = nullptr;
    size_t user_workspace_size = 0;

    bool use_user_workspace = false;
};

/**
 * @brief 获取当前活跃 workspace 指针
 */
inline void* aclsparseGetEffectiveWorkspace(const aclsparseContext* h)
{
    if (h == nullptr) { return nullptr; }
    return h->use_user_workspace ? h->user_workspace : h->default_workspace;
}

/**
 * @brief 获取当前活跃 workspace 大小
 */
inline size_t aclsparseGetEffectiveWorkspaceSize(const aclsparseContext* h)
{
    if (h == nullptr) { return 0; }
    return h->use_user_workspace ? h->user_workspace_size : h->default_workspace_size;
}

/**
 * @brief 保证当前活跃 workspace 至少有 size 字节可用。
 *
 * 使用默认 workspace 时按需扩容（grow-only，旧显存先释放再重新分配）；用户通过
 * aclsparseSetWorkspace 设置了自己的 workspace 时不做任何分配，仅校验容量——
 * 用户显式提供的显存不应被库悄悄替换掉。
 *
 * 调用成功后即可用 aclsparseGetEffectiveWorkspace 取到不小于 size 的缓冲区。
 */
aclsparseStatus_t aclsparseEnsureDefaultWorkspace(aclsparseContext* h, size_t size);

/**
 * @brief 切回默认 workspace（不清除用户 workspace 缓存字段）
 */
inline void aclsparseResetToDefaultWorkspace(aclsparseContext* h)
{
    if (h != nullptr) {
        h->use_user_workspace = false;
    }
}
