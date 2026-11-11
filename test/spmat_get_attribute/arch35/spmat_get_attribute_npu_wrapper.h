/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPMAT_GET_ATTRIBUTE_SPMAT_GET_ATTRIBUTE_NPU_WRAPPER_H_
#define TEST_SPMAT_GET_ATTRIBUTE_SPMAT_GET_ATTRIBUTE_NPU_WRAPPER_H_

#include <cstring>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "spmat_test_helpers.h"  // SpMatGuard/FormatEntry/kRows 常量/创建模板（与 destroy_spmat wrapper 共享）

// ============================================================================
// NPU Workflow Result：每步返回码 + 回读值（纯 Host 元数据，无张量计算）。
// 未执行步骤保持 NOT_INITIALIZED 哨兵，不误报 SUCCESS。
// ============================================================================
struct SpMatGetAttributeNpuResult {
    aclsparseStatus_t createRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};

    // Set 之前的默认值回读（期望 LOWER / NON_UNIT）
    aclsparseStatus_t getDefaultFillRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t getDefaultDiagRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseFillMode_t defaultFill{static_cast<aclsparseFillMode_t>(-1)};
    aclsparseDiagType_t defaultDiag{static_cast<aclsparseDiagType_t>(-1)};

    // SetAttribute（仅 mutable 入口执行；const 入口跳过并保持哨兵）
    aclsparseStatus_t setFillRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t setDiagRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};

    // Set 之后的回读 + 二次 Get 幂等
    aclsparseStatus_t getFillRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t getDiagRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseFillMode_t outFill{static_cast<aclsparseFillMode_t>(-1)};
    aclsparseDiagType_t outDiag{static_cast<aclsparseDiagType_t>(-1)};
    aclsparseStatus_t getFillAgainRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t getDiagAgainRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseFillMode_t outFillAgain{static_cast<aclsparseFillMode_t>(-1)};
    aclsparseDiagType_t outDiagAgain{static_cast<aclsparseDiagType_t>(-1)};
};

// 本文件仅 CreateEntry 分发引用 frame 入口原语与几何常量，using 声明紧邻首个使用点
using sparse_test::FormatEntry;
using sparse_test::SpMatGuard;
using sparse_test::kCols;
using sparse_test::kNnz;
using sparse_test::kRows;

// ============================================================================
// 入口描述（注册表 kEntries 定义在 test.cpp 内，对齐先例 spmat_get_indexbase
// 的 kEntries 位置——编译单元私有数据，避免头文件定义非 inline 数组）。
// 属性与格式无关，仅取有代表性的 csr / bsr / sliced_ell 三种 mutable 入口
// （CSV fill × diag 全组合）+ const_csr / const_bsr 两个 const 入口（语义用例
// GA-S02 的 const 签名 Get 路径）。
// ============================================================================

// 按注册表入口创建描述符（签名与几何沿用 spmat_get_indexbase 缺省）
inline aclsparseStatus_t CreateEntry(const FormatEntry &entry, aclsparseIndexBase_t base,
                                     void *ptrs, void *idxs, void *vals, SpMatGuard *guard)
{
    if (strcmp(entry.name, "csr") == 0) {
        return CreateSpMatEntry(guard, aclsparseCreateCsr, kRows, kCols, kNnz, ptrs, idxs,
            vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    if (strcmp(entry.name, "bsr") == 0) {
        return CreateSpMatEntry(guard, aclsparseCreateBsr, 2, 2, 2, 2, 2, ptrs, idxs, vals,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    }
    if (strcmp(entry.name, "sliced_ell") == 0) {
        return CreateSpMatEntry(guard, aclsparseCreateSlicedEll, kRows, kCols, 4, 2, 2, ptrs,
            idxs, vals, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    if (strcmp(entry.name, "const_csr") == 0) {
        return CreateConstSpMatEntry(guard, aclsparseCreateConstCsr, kRows, kCols, kNnz, ptrs,
            idxs, vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT);
    }
    if (strcmp(entry.name, "const_bsr") == 0) {
        return CreateConstSpMatEntry(guard, aclsparseCreateConstBsr, 2, 2, 2, 2, 2, ptrs, idxs,
            vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, ACL_FLOAT,
            ACL_SPARSE_ORDER_ROW);
    }
    return ACL_SPARSE_STATUS_INVALID_VALUE;
}

// ============================================================================
// NPU wrapper：Create(入口) -> Get 默认值 -> Set(fill/diag) -> Get 回读 -> 二次 Get
// 纯 Host 侧描述符属性操作，不启动 NPU kernel；设备缓冲仅占位（描述符只借用
// 设备指针、不解引用内容）。
// 入口由调用方（test.cpp 的 kEntries 注册表）解析后传入；
// const 入口走 Get-only 路径（SetAttribute 需要 mutable 句柄，Set 步骤跳过并
// 保持 NOT_INITIALIZED 哨兵），供语义用例 GA-S02 的 const 签名路径验证复用。
// ============================================================================
inline SpMatGetAttributeNpuResult SpMatGetAttributeNpu(aclrtStream stream,
    const FormatEntry &entry, int idxBase, aclsparseFillMode_t fill, aclsparseDiagType_t diag)
{
    using namespace sparse_test;
    (void)stream;
    SpMatGetAttributeNpuResult r{};

    const aclsparseIndexBase_t base =
        (idxBase == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    // 占位设备缓冲（内容任意，仅要求指针有效）
    const std::vector<int32_t> hostIdx(16, 0);
    const std::vector<float> hostVal(32, 1.0F);
    auto ptrs = DeviceBuffer::copyFrom(hostIdx.data(), hostIdx.size() * sizeof(int32_t));
    auto idxs = DeviceBuffer::copyFrom(hostIdx.data(), hostIdx.size() * sizeof(int32_t));
    auto vals = DeviceBuffer::copyFrom(hostVal.data(), hostVal.size() * sizeof(float));

    SpMatGuard descr;
    r.createRet = CreateEntry(entry, base, ptrs.get(), idxs.get(), vals.get(), &descr);
    if (r.createRet != ACL_SPARSE_STATUS_SUCCESS || descr.get() == nullptr) {
        std::cerr << "[NPU] create " << entry.name << " failed: " << r.createRet << std::endl;
        return r;
    }

    // 1. Set 之前回读默认值（LOWER / NON_UNIT）
    r.getDefaultFillRet = aclsparseSpMatGetAttribute(descr.cget(), ACL_SPARSE_SPMAT_FILL_MODE,
        &r.defaultFill, sizeof(r.defaultFill));
    r.getDefaultDiagRet = aclsparseSpMatGetAttribute(descr.cget(), ACL_SPARSE_SPMAT_DIAG_TYPE,
        &r.defaultDiag, sizeof(r.defaultDiag));

    // 2. SetAttribute roundtrip（仅 mutable 入口）
    if (!entry.isConst) {
        r.setFillRet = aclsparseSpMatSetAttribute(descr.get(), ACL_SPARSE_SPMAT_FILL_MODE,
            &fill, sizeof(fill));
        r.setDiagRet = aclsparseSpMatSetAttribute(descr.get(), ACL_SPARSE_SPMAT_DIAG_TYPE,
            &diag, sizeof(diag));
    }

    // 3. Set 之后回读（const 入口等价于再次读默认值）
    r.getFillRet = aclsparseSpMatGetAttribute(descr.cget(), ACL_SPARSE_SPMAT_FILL_MODE,
        &r.outFill, sizeof(r.outFill));
    r.getDiagRet = aclsparseSpMatGetAttribute(descr.cget(), ACL_SPARSE_SPMAT_DIAG_TYPE,
        &r.outDiag, sizeof(r.outDiag));

    // 4. 二次 Get 幂等
    r.getFillAgainRet = aclsparseSpMatGetAttribute(descr.cget(), ACL_SPARSE_SPMAT_FILL_MODE,
        &r.outFillAgain, sizeof(r.outFillAgain));
    r.getDiagAgainRet = aclsparseSpMatGetAttribute(descr.cget(), ACL_SPARSE_SPMAT_DIAG_TYPE,
        &r.outDiagAgain, sizeof(r.outDiagAgain));
    return r;
}

#endif  // TEST_SPMAT_GET_ATTRIBUTE_SPMAT_GET_ATTRIBUTE_NPU_WRAPPER_H_
