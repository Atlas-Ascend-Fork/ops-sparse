/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"
#include "descriptor_manager.h"

// ============================================================================
// NPU Workflow Result
// ============================================================================
struct CooSetStridedBatchNpuResult {
    aclsparseStatus_t defaultGetRet{ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR};
    aclsparseStatus_t setRet{ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR};
    aclsparseStatus_t getRet{ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR};

    // Set 前读回的默认 batchCount（未 batch 化描述符应为 1）
    int defaultBatchCount{-1};
    // Set 后 SpMatGetStridedBatch 读回的 batchCount
    int outBatchCount{-1};
    // 白盒读回：描述符内部 batchStride（cuSPARSE 未暴露 stride 的公开读回接口，
    // 借助内部头直接断言字段写入值）
    int64_t innerBatchStride{-2};
    int32_t innerBatchCount{-1};
};

// ============================================================================
// 生成 COO 测试数据（host 侧仅生成输入，不在 CPU 做算子计算；模式同 coo_get）
// ============================================================================
inline void GenerateCooData(int rows, int cols, int nnz, int idxBase, uint32_t seed, std::vector<int32_t>& rowInd,
                            std::vector<int32_t>& colInd)
{
    rowInd.clear();
    colInd.clear();
    if (nnz <= 0 || rows <= 0 || cols <= 0) {
        return;
    }
    rowInd.reserve(nnz);
    colInd.reserve(nnz);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> rowDist(0, static_cast<int32_t>(rows) - 1);
    std::uniform_int_distribution<int32_t> colDist(0, static_cast<int32_t>(cols) - 1);
    int32_t base = (idxBase == 1) ? 1 : 0;
    for (int i = 0; i < nnz; i++) {
        rowInd.push_back(rowDist(rng) + base);
        colInd.push_back(colDist(rng) + base);
    }
}

// ============================================================================
// NPU wrapper: CreateCoo -> (default Get) -> CooSetStridedBatch -> Get 闭环验证
// 全程在 NPU 侧建设备缓冲存指针；Host 仅做元数据生成与断言。
// 描述符生命周期由 SpMatManager RAII 管理；batchStride=0 时为广播（所有
// batch 共享数组），设备缓冲长度为 nnz；否则为 batchStride * batchCount
// （每 batch 数据区间 [b*stride, b*stride+nnz)）。
// ============================================================================
inline CooSetStridedBatchNpuResult CooSetStridedBatchNpu(aclrtStream stream, int rows, int cols, int nnz,
                                                         int batchCount, int64_t batchStride, int idxBase, bool isFp16,
                                                         uint32_t seed)
{
    using namespace sparse_test;
    (void)stream;
    CooSetStridedBatchNpuResult r{};

    // 1. 生成 COO 输入数据，并按文档布局契约补齐到 totalElems：
    //    stride=0（广播）时三数组长度为 nnz；否则为 batchStride * batchCount，
    //    第 b 个 batch 起点为 b*batchStride（[nnz, totalElems) 区间为填充）。
    std::vector<int32_t> hostRowInd;
    std::vector<int32_t> hostColInd;
    GenerateCooData(rows, cols, nnz, idxBase, seed, hostRowInd, hostColInd);

    // 2. 输入张量迁移到 NPU 设备（nnz=0 时传 nullptr，与 cuSPARSE 空矩阵约定一致）
    int64_t totalElems = (batchStride == 0) ? nnz : batchStride * batchCount;
    if (totalElems < nnz) {
        totalElems = nnz; // 防御：非法 stride 由 Set 的校验负责报错
    }
    hostRowInd.resize(static_cast<size_t>(totalElems), 0);
    hostColInd.resize(static_cast<size_t>(totalElems), 0);
    int32_t* rowPtr = nullptr;
    int32_t* colPtr = nullptr;
    void* valPtr = nullptr;
    DeviceBuffer dRow;
    DeviceBuffer dCol;
    DeviceBuffer dVal;
    if (nnz > 0) {
        dRow = DeviceBuffer::copyFrom(hostRowInd.data(), static_cast<size_t>(totalElems) * sizeof(int32_t));
        dCol = DeviceBuffer::copyFrom(hostColInd.data(), static_cast<size_t>(totalElems) * sizeof(int32_t));
        rowPtr = static_cast<int32_t*>(dRow.get());
        colPtr = static_cast<int32_t*>(dCol.get());
        size_t valBytes = isFp16 ? (static_cast<size_t>(totalElems) * sizeof(uint16_t)) :
                                   (static_cast<size_t>(totalElems) * sizeof(float));
        dVal = DeviceBuffer::alloc(valBytes);
        valPtr = dVal.get();
    }

    aclDataType valueType = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;
    aclsparseIndexBase_t idxBaseEnum = (idxBase == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    // 3. aclsparseCreateCoo（失败抛异常，由 GTest 捕获判 FAIL）
    SpMatManager mat = SpMatManager::createCoo(rows, cols, nnz, rowPtr, colPtr, valPtr, ACL_SPARSE_INDEX_32I,
                                               idxBaseEnum, valueType);

    // 4. Set 前读回默认 batchCount（未 batch 化描述符应为 1）
    r.defaultGetRet = aclsparseSpMatGetStridedBatch(mat.get(), &r.defaultBatchCount);
    if (r.defaultGetRet != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("default aclsparseSpMatGetStridedBatch failed");
    }

    // 5. aclsparseCooSetStridedBatch
    r.setRet = aclsparseCooSetStridedBatch(mat.get(), batchCount, batchStride);
    if (r.setRet != ACL_SPARSE_STATUS_SUCCESS) {
        return r;
    }

    // 6. Set 后读回 batchCount
    r.getRet = aclsparseSpMatGetStridedBatch(mat.get(), &r.outBatchCount);
    if (r.getRet != ACL_SPARSE_STATUS_SUCCESS) {
        return r;
    }

    // 7. 白盒读回内部 batchStride / batchCount 字段
    r.innerBatchStride = mat.get()->batchStride;
    r.innerBatchCount = mat.get()->batchCount;
    return r;
}
