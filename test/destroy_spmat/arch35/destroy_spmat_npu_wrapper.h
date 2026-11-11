/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_DESTROY_SPMAT_DESTROY_SPMAT_NPU_WRAPPER_H_
#define TEST_DESTROY_SPMAT_DESTROY_SPMAT_NPU_WRAPPER_H_

#include <cstring>
#include <iostream>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "spmat_test_helpers.h"  // SpMatGuard/FormatEntry/kRows 常量/创建模板（与 spmat_get_attribute wrapper 共享）

// 本文件 CreateMutableEntry/CreateConstEntry 分发与 Destroy wrapper 主体经
// using 声明引用 frame 守卫、入口结构体与几何常量（首个使用点为前者）
using sparse_test::FormatEntry;
using sparse_test::SpMatGuard;
using sparse_test::kCols;
using sparse_test::kNnz;
using sparse_test::kRows;

// ============================================================================
// 入口描述（注册表 kEntries 定义在 test.cpp 内，对齐先例 spmat_get_indexbase
// 的 kEntries 位置——编译单元私有数据，避免头文件定义非 inline 数组）。
// 11 个创建入口：6 mutable + 5 const（blocked_ell 无 const 变体），
// 构造参数沿用 spmat_get_indexbase 缺省：csr/csc/coo=4×4×5、
// bsr=2,2,2,2,2+ROW、blocked_ell=4,4,2,2、sliced_ell=4,4,4,2,2。
// ============================================================================

// ============================================================================
// 按注册表入口创建描述符（签名与几何沿用 spmat_get_indexbase 缺省）。
// mutable / const 两组创建签名不同，按 entry.isConst 分组分发（阶段函数），
// 未注册名统一 INVALID_VALUE 回落。
// ============================================================================
inline aclsparseStatus_t CreateMutableEntry(const FormatEntry &entry, void *ptrs, void *idxs,
                                            void *vals, SpMatGuard *guard)
{
    if (strcmp(entry.name, "csr") == 0) {
        return CreateSpMatEntry(guard, aclsparseCreateCsr, kRows, kCols, kNnz, ptrs, idxs,
            vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
            ACL_FLOAT);
    }
    if (strcmp(entry.name, "csc") == 0) {
        return CreateSpMatEntry(guard, aclsparseCreateCsc, kRows, kCols, kNnz, ptrs, idxs,
            vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
            ACL_FLOAT);
    }
    if (strcmp(entry.name, "coo") == 0) {
        return CreateSpMatEntry(guard, aclsparseCreateCoo, kRows, kCols, kNnz, ptrs, idxs,
            vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    }
    if (strcmp(entry.name, "bsr") == 0) {
        return CreateSpMatEntry(guard, aclsparseCreateBsr, 2, 2, 2, 2, 2, ptrs, idxs, vals,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
            ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    }
    if (strcmp(entry.name, "blocked_ell") == 0) {
        return CreateSpMatEntry(guard, aclsparseCreateBlockedEll, kRows, kCols, 2, 2, idxs,
            vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    }
    if (strcmp(entry.name, "sliced_ell") == 0) {
        return CreateSpMatEntry(guard, aclsparseCreateSlicedEll, kRows, kCols, 4, 2, 2,
            ptrs, idxs, vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    }
    return ACL_SPARSE_STATUS_INVALID_VALUE;
}

// const 变体入口分发（const_csr/const_csc/const_coo/const_bsr/const_sliced_ell；
// blocked_ell 无 const 变体）
inline aclsparseStatus_t CreateConstEntry(const FormatEntry &entry, void *ptrs, void *idxs,
                                          void *vals, SpMatGuard *guard)
{
    if (strcmp(entry.name, "const_csr") == 0) {
        return CreateConstSpMatEntry(guard, aclsparseCreateConstCsr, kRows, kCols, kNnz,
            ptrs, idxs, vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    }
    if (strcmp(entry.name, "const_csc") == 0) {
        return CreateConstSpMatEntry(guard, aclsparseCreateConstCsc, kRows, kCols, kNnz,
            ptrs, idxs, vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    }
    if (strcmp(entry.name, "const_coo") == 0) {
        return CreateConstSpMatEntry(guard, aclsparseCreateConstCoo, kRows, kCols, kNnz,
            ptrs, idxs, vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    }
    if (strcmp(entry.name, "const_bsr") == 0) {
        return CreateConstSpMatEntry(guard, aclsparseCreateConstBsr, 2, 2, 2, 2, 2, ptrs,
            idxs, vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    }
    if (strcmp(entry.name, "const_sliced_ell") == 0) {
        return CreateConstSpMatEntry(guard, aclsparseCreateConstSlicedEll, kRows, kCols, 4,
            2, 2, ptrs, idxs, vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
            ACL_FLOAT);
    }
    return ACL_SPARSE_STATUS_INVALID_VALUE;
}

inline aclsparseStatus_t CreateEntry(const FormatEntry &entry, void *ptrs, void *idxs,
                                     void *vals, SpMatGuard *guard)
{
    return entry.isConst ? CreateConstEntry(entry, ptrs, idxs, vals, guard)
                         : CreateMutableEntry(entry, ptrs, idxs, vals, guard);
}

// ============================================================================
// NPU Workflow Result
// ============================================================================
struct DestroySpMatNpuResult {
    aclsparseStatus_t createRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t destroyRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
};

// ============================================================================
// NPU wrapper：Create(entry) -> 手动 Destroy（guard release 语义：RAII 管理器
// 释放所有权后显式调用，避免双重释放）。纯 Host 侧描述符操作，不启动 NPU
// kernel；设备缓冲仅占位（描述符只借用设备指针、不解引用内容）。
// ============================================================================
inline DestroySpMatNpuResult DestroySpMatNpu(aclrtStream stream, const FormatEntry &entry,
                                             void *ptrs, void *idxs, void *vals)
{
    using namespace sparse_test;
    (void)stream;
    DestroySpMatNpuResult r{};

    SpMatGuard guard;
    r.createRet = CreateEntry(entry, ptrs, idxs, vals, &guard);
    if (r.createRet != ACL_SPARSE_STATUS_SUCCESS || guard.get() == nullptr) {
        std::cerr << "[NPU] create " << entry.name << " failed: " << r.createRet << std::endl;
        return r;
    }

    // guard release 语义：释放所有权后显式 Destroy，杜绝双重释放
    aclsparseSpMatDescr_t descr = guard.release();
    r.destroyRet = aclsparseDestroySpMat(descr);
    if (r.destroyRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseDestroySpMat(" << entry.name
                  << ") failed: " << r.destroyRet << std::endl;
        // Destroy 失败则句柄未被 delete，交还 guard 兜底释放
        *guard.put() = descr;
    }
    return r;
}

#endif  // TEST_DESTROY_SPMAT_DESTROY_SPMAT_NPU_WRAPPER_H_
