/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CREATE_SLICED_ELL_CREATE_SLICED_ELL_NPU_WRAPPER_H_
#define TEST_CREATE_SLICED_ELL_CREATE_SLICED_ELL_NPU_WRAPPER_H_

#include <cstdint>
#include <iostream>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "spmat_test_helpers.h"    // SpMatCreateGetResult/ReadBackGenericGets：Create 系公共回读段
#include "aclsparse_descr_internal.h"  // 白盒：SELL 无公共 Get 访问器，核对 sliceNnz/numSlices 等字段

// frame 共享助手统一置于 sparse_test 命名空间，本文件的结构体定义经
// using 声明引用（函数体内已有 using namespace sparse_test）
using sparse_test::SpMatCreateGetResult;

// ============================================================================
// NPU Workflow Result：Create + 通用 Get 回读公共段 + 白盒字段（纯 Host 元数据）。
// 未执行步骤（Create 失败提前返回）保持 NOT_INITIALIZED 哨兵，不误报 SUCCESS。
// ============================================================================
struct CreateSlicedEllNpuResult : public SpMatCreateGetResult {
    // 白盒字段（先例 csr_set_strided_batch：SELL 无公共 Get 访问器，
    // sliceNnz / numSlices 只能经内部结构核对，防 int64 截断）
    int64_t whiteBoxSliceNnz{-1};
    int64_t whiteBoxNumSlices{-1};
    aclsparseIndexType_t whiteBoxPtrType{static_cast<aclsparseIndexType_t>(-1)};
    aclsparseIndexType_t whiteBoxIdxType{static_cast<aclsparseIndexType_t>(-1)};
    aclDataType whiteBoxValueType{static_cast<aclDataType>(-1)};
};

// ============================================================================
// 缓冲准备阶段（阶段函数）：占位设备缓冲（内容任意，Create 不解引用）：
// slicePtr 按 numSlices+1 端点、colInd 按 sliceNnz×numSlices 槽位、values 按
// nnz 元素；nullPtrs=1 时三个指针全部留空（传 nullptr，延迟绑定语义）。
// ============================================================================
struct SlicedEllBuffers {
    sparse_test::DeviceBuffer slicePtr;
    sparse_test::DeviceBuffer colInd;
    sparse_test::DeviceBuffer vals;
};

inline SlicedEllBuffers AllocSlicedEllBuffers(int64_t nnz, int64_t sliceNnz, int64_t numSlices,
                                              int idxType, bool isFp16, bool nullPtrs)
{
    using namespace sparse_test;
    SlicedEllBuffers b;
    if (!nullPtrs) {
        const size_t idxBytes = (idxType == 1) ? sizeof(int64_t) : sizeof(int32_t);
        const size_t valBytes = isFp16 ? sizeof(uint16_t) : sizeof(float);
        b.slicePtr = DeviceBuffer::alloc((static_cast<size_t>(numSlices) + 1u) * idxBytes);
        b.colInd = DeviceBuffer::alloc(static_cast<size_t>(sliceNnz) *
                                       static_cast<size_t>(numSlices) * idxBytes);
        b.vals = DeviceBuffer::alloc(static_cast<size_t>(nnz) * valBytes);
    }
    return b;
}

// ============================================================================
// NPU wrapper：CreateSlicedEll / CreateConstSlicedEll -> 通用 Get 访问器验证。
// 纯 Host 侧描述符创建，不启动 NPU kernel。
//
// 注意（wrapper 适用范围声明）：设备缓冲仅按单矩阵规模占位分配，用于承载
// 描述符持有的设备指针；Create/Get 均不访问数组内容（元数据借用 + 延迟绑定）。
// SELL 使用 SpMatManager::createSlicedEll / createConstSlicedEll 工厂（RAII，
// 失败抛 std::runtime_error，由本 wrapper 捕获并置失败返回码）。
// ============================================================================
inline CreateSlicedEllNpuResult CreateSlicedEllNpu(aclrtStream stream,
    const std::string &entry, int64_t rows, int64_t cols, int64_t nnz,
    int64_t sliceNnz, int64_t numSlices, int idxType, int idxBase,
    bool isFp16, bool nullPtrs)
{
    using namespace sparse_test;
    (void)stream;
    CreateSlicedEllNpuResult r{};

    const aclsparseIndexType_t sellIdxType =
        (idxType == 1) ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    const aclsparseIndexBase_t idxBaseEnum =
        (idxBase == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
    const aclDataType valueType = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;

    // 1. 缓冲准备阶段（nullPtrs=1 时三指针全传 nullptr，延迟绑定语义）
    auto buf = AllocSlicedEllBuffers(nnz, sliceNnz, numSlices, idxType, isFp16, nullPtrs);

    // 2. 创建阶段（SpMatManager RAII 工厂；const 变体句柄由工厂转存 mutable 视图）
    SpMatManager mat;
    try {
        if (entry == "const_sliced_ell") {
            mat = SpMatManager::createConstSlicedEll(rows, cols, nnz, buf.slicePtr.get(),
                buf.colInd.get(), buf.vals.get(), sliceNnz, numSlices, sellIdxType, idxBaseEnum,
                valueType);
        } else {
            mat = SpMatManager::createSlicedEll(rows, cols, nnz, buf.slicePtr.get(),
                buf.colInd.get(), buf.vals.get(), sliceNnz, numSlices, sellIdxType, idxBaseEnum,
                valueType);
        }
    } catch (const std::exception &e) {
        std::cerr << "[NPU] aclsparseCreateSlicedEll failed: " << e.what() << std::endl;
        r.createRet = ACL_SPARSE_STATUS_INVALID_VALUE;  // 工厂抛异常 = Create 未成功
        return r;
    }
    r.createRet = ACL_SPARSE_STATUS_SUCCESS;

    // 3. Get 回读阶段（Create 语义载体，成功路径；四个通用访问器共享回读）
    if (!sparse_test::ReadBackGenericGets(mat.cget(), r)) {
        std::cerr << "[NPU] generic Get accessor failed after create" << std::endl;
        return r;
    }

    // 4. 白盒核对阶段：sliceNnz / numSlices / ptrType / IdxType / valueType 与传入一致
    r.whiteBoxSliceNnz = static_cast<int64_t>(mat.get()->sliceNnz);
    r.whiteBoxNumSlices = mat.get()->numSlices;
    r.whiteBoxPtrType = mat.get()->ptrType;
    r.whiteBoxIdxType = mat.get()->IdxType;
    r.whiteBoxValueType = mat.get()->valueType;
    return r;
}

#endif  // TEST_CREATE_SLICED_ELL_CREATE_SLICED_ELL_NPU_WRAPPER_H_
