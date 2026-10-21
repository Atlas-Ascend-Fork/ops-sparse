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
 * \file aclsparse_auxiliary.cpp
 * \brief ops-sparse handle 生命周期、stream、版本信息等辅助函数实现。
 */

#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"

#include <new>

namespace {

/**
 * @brief 将对外 void* 句柄安全转换为内部 aclsparseContext*。
 */
inline aclsparseContext *ToInternal(aclsparseHandle_t handle)
{
    return reinterpret_cast<aclsparseContext *>(handle);
}

/**
 * @brief 释放默认 workspace 设备内存
 */
void freeDefaultWorkspace(aclsparseContext *h)
{
    if (h == nullptr || h->default_workspace == nullptr) { return; }
    aclrtFree(h->default_workspace);
    h->default_workspace = nullptr;
    h->default_workspace_size = 0;
}

/**
 * @brief 分配默认 workspace
 */
aclsparseStatus_t allocateDefaultWorkspace(aclsparseContext *h, size_t size)
{
    if (h == nullptr || size == 0) { return ACL_SPARSE_STATUS_INVALID_VALUE; }
    void *ptr = nullptr;
    const aclError aclRet = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) { return ACL_SPARSE_STATUS_ALLOC_FAILED; }
    h->default_workspace = ptr;
    h->default_workspace_size = size;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

aclsparseStatus_t aclsparseEnsureDefaultWorkspace(aclsparseContext *h, size_t size)
{
    if (h == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (size == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    // 已经够大（含用户 workspace 的情形）直接复用，避免无谓的 free/malloc。
    if (size <= aclsparseGetEffectiveWorkspaceSize(h) &&
        aclsparseGetEffectiveWorkspace(h) != nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    // 用户自管 workspace 时库不代为扩容，由调用方决定是加大 buffer 还是切回默认。
    if (h->use_user_workspace) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    freeDefaultWorkspace(h);
    return allocateDefaultWorkspace(h, size);
}

extern "C" {

aclsparseStatus_t aclsparseCreate(aclsparseHandle_t *handle)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    // 防止重复创建覆盖已有指针导致的内存泄漏。
    if (*handle != nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto *h = new (std::nothrow) aclsparseContext();
    if (h == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    h->signature = kAclSparseHandleSignature;
    h->stream = nullptr;

    // 分配默认 workspace
    aclsparseStatus_t wsStatus = allocateDefaultWorkspace(h, ACLSPARSE_DEFAULT_WORKSPACE_SIZE);
    if (wsStatus != ACL_SPARSE_STATUS_SUCCESS) {
        delete h;
        return wsStatus;
    }

    *handle = reinterpret_cast<aclsparseHandle_t>(h);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroy(aclsparseHandle_t handle)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    auto *h = ToInternal(handle);

    if (h->signature != kAclSparseHandleSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // 同步 stream，等待设备完成所有操作
    if (h->stream != nullptr) {
        aclrtSynchronizeStream(h->stream);
    }

    // 释放默认 workspace
    freeDefaultWorkspace(h);

    // 清除用户 workspace 引用（不释放用户内存）
    h->user_workspace = nullptr;
    h->user_workspace_size = 0;
    h->use_user_workspace = false;

    // stream 由用户托管，此处仅清理句柄自身字段。
    h->stream = nullptr;
    h->signature = 0;

    delete h;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSetStream(aclsparseHandle_t handle, aclrtStream stream)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    auto *h = ToInternal(handle);
    if (h->signature != kAclSparseHandleSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (h->stream != stream) {
        ++h->stream_epoch;
        // Overflow is not observable in practice, but avoid wrapping to the
        // initial epoch if a long-lived handle is repeatedly retargeted.
        if (h->stream_epoch == 0) {
            h->stream_epoch = 1;
        }
    }
    h->stream = stream;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseGetStream(aclsparseHandle_t handle, aclrtStream *stream)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (stream == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto *h = ToInternal(handle);
    if (h->signature != kAclSparseHandleSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *stream = h->stream;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSetWorkspace(aclsparseHandle_t handle, void *workspace, size_t workspaceSize)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    auto *h = ToInternal(handle);
    if (h->signature != kAclSparseHandleSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // workspace == nullptr: 切回默认 workspace
    if (workspace == nullptr) {
        aclsparseResetToDefaultWorkspace(h);
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    if (workspaceSize == 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // Grow-only 策略：新 size 不大于当前 user workspace size 时保持不变
    if (h->use_user_workspace && workspaceSize <= h->user_workspace_size) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    h->user_workspace = workspace;
    h->user_workspace_size = workspaceSize;
    h->use_user_workspace = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseGetWorkspace(aclsparseHandle_t handle, void **workspace, size_t *workspaceSize)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (workspace == nullptr || workspaceSize == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto *h = ToInternal(handle);
    if (h->signature != kAclSparseHandleSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *workspace = aclsparseGetEffectiveWorkspace(h);
    *workspaceSize = aclsparseGetEffectiveWorkspaceSize(h);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseGetVersion(aclsparseHandle_t handle, int *version)
{
    if (version == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    *version = ACLSPARSE_VERSION;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateCsc(aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
    int64_t nnz, void *cscColOffsets, void *cscRowInd, void *cscValues,
    aclsparseIndexType_t cscColOffsetsType, aclsparseIndexType_t cscRowIndType,
    aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    aclsparseStatus_t st = ValidateSpMatCreateParams(spMatDescr, rows, cols, nnz);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // Null CSC pointers are allowed at Create for delayed binding; validate at use.
    if (cscColOffsetsType != ACL_SPARSE_INDEX_32I && cscColOffsetsType != ACL_SPARSE_INDEX_64I) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (cscRowIndType != ACL_SPARSE_INDEX_32I && cscRowIndType != ACL_SPARSE_INDEX_64I) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *inner = new (std::nothrow) aclsparseSpMatDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    inner->format = ACL_SPARSE_FORMAT_CSC;
    inner->rows = static_cast<uint64_t>(rows);
    inner->cols = static_cast<uint64_t>(cols);
    inner->nnz = static_cast<uint64_t>(nnz);
    inner->ptrs = cscColOffsets;
    inner->idxs = cscRowInd;
    inner->values = cscValues;
    inner->baseType = idxBase;
    inner->ptrType = cscColOffsetsType;
    inner->IdxType = cscRowIndType;
    inner->valueType = valueType;
    inner->signature = kSpMatSignature;
    *spMatDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstCsc(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows,
    int64_t cols, int64_t nnz, const void *cscColOffsets, const void *cscRowInd,
    const void *cscValues, aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType, aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseSpMatDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateCsc(&tmp, rows, cols, nnz,
        const_cast<void *>(cscColOffsets), const_cast<void *>(cscRowInd),
        const_cast<void *>(cscValues), cscColOffsetsType, cscRowIndType, idxBase, valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *spMatDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSetPointerMode(aclsparseHandle_t handle, aclsparsePointerMode_t mode)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (mode != ACL_SPARSE_POINTER_MODE_HOST && mode != ACL_SPARSE_POINTER_MODE_DEVICE) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto *h = ToInternal(handle);
    if (h->signature != kAclSparseHandleSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    h->pointerMode = mode;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseGetPointerMode(aclsparseHandle_t handle, aclsparsePointerMode_t *mode)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (mode == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto *h = ToInternal(handle);
    if (h->signature != kAclSparseHandleSignature) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *mode = h->pointerMode;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
