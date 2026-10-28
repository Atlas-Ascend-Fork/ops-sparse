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
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

// ============================================================================
// NPU Workflow Result
// ============================================================================
struct SpMatGetStridedBatchNpuResult {
    aclsparseStatus_t setRet{ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR};
    aclsparseStatus_t getRet{ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR};

    // Get 读回的 batchCount：pre_set=0 时应为默认值 1，pre_set=1 时应为 Set 传入值
    int outBatchCount{-1};
};

// ============================================================================
// 生成稀疏索引测试数据（host 侧仅生成输入，不在 CPU 做算子计算；模式同 coo_get）
// ============================================================================
inline void GenerateIndexData(int count, int range, int idxBase, uint32_t seed, std::vector<int32_t>& indices)
{
    indices.clear();
    if (count <= 0 || range <= 0) {
        return;
    }
    indices.reserve(count);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(range) - 1);
    int32_t base = (idxBase == 1) ? 1 : 0;
    for (int i = 0; i < count; i++) {
        indices.push_back(dist(rng) + base);
    }
}

// 生成压缩格式 offsets 数组（长度 range+1，单调非减，末值 = count；内容不被访问）
inline void GenerateOffsets(int range, int count, std::vector<int32_t>& offsets)
{
    offsets.assign(static_cast<size_t>(range) + 1, 0);
    for (int i = 0; i <= range; i++) {
        offsets[static_cast<size_t>(i)] = static_cast<int32_t>(static_cast<int64_t>(i) * count /
                                                               (range > 0 ? range : 1));
    }
}

// ============================================================================
// NPU wrapper: Create(coo/csr/csc) -> (可选 CooSetStridedBatch) -> Get 验证
// 全程在 NPU 侧建设备缓冲存指针；Host 仅做元数据生成与断言。
// 描述符生命周期由 SpMatManager RAII 管理。
// ============================================================================
inline SpMatGetStridedBatchNpuResult SpMatGetStridedBatchNpu(aclrtStream stream, const std::string& format, int rows,
                                                             int cols, int nnz, int preSet, int batchCount,
                                                             int64_t batchStride, int idxBase, bool isFp16,
                                                             uint32_t seed)
{
    using namespace sparse_test;
    (void)stream;
    SpMatGetStridedBatchNpuResult r{};

    // 1. 生成 host 侧输入数据
    std::vector<int32_t> hostInd1;
    std::vector<int32_t> hostInd2;
    if (format == "csr") {
        GenerateOffsets(rows, nnz, hostInd1);                  // rowOffsets: rows+1
        GenerateIndexData(nnz, cols, idxBase, seed, hostInd2); // colInd
    } else if (format == "csc") {
        GenerateOffsets(cols, nnz, hostInd1);                       // colOffsets: cols+1
        GenerateIndexData(nnz, rows, idxBase, seed, hostInd2);      // rowInd
    } else {                                                        // coo
        GenerateIndexData(nnz, rows, idxBase, seed, hostInd1);      // rowInd
        GenerateIndexData(nnz, cols, idxBase, seed + 1U, hostInd2); // colInd
    }

    // 2. 输入张量迁移到 NPU 设备（设备缓冲仅按单矩阵规模占位：本 wrapper 只做
    //    元数据读回测试，不按 batch 布局分配，可承载超大 batch_stride 的用例）
    DeviceBuffer dInd1 = DeviceBuffer::copyFrom(hostInd1.data(), hostInd1.size() * sizeof(int32_t));
    DeviceBuffer dInd2 = DeviceBuffer::copyFrom(hostInd2.data(), hostInd2.size() * sizeof(int32_t));
    size_t valBytes = isFp16 ? (static_cast<size_t>(nnz) * sizeof(uint16_t)) :
                               (static_cast<size_t>(nnz) * sizeof(float));
    DeviceBuffer dVal = DeviceBuffer::alloc(nnz > 0 ? valBytes : 0);

    aclDataType valueType = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;
    aclsparseIndexBase_t idxBaseEnum = (idxBase == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    // 3. 按格式创建描述符（RAII，失败抛异常，由 GTest 捕获判 FAIL）
    SpMatManager mat;
    if (format == "csr") {
        mat = SpMatManager::createCsr(rows, cols, nnz, dInd1.get(), dInd2.get(), dVal.get(), ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_32I, idxBaseEnum, valueType);
    } else if (format == "csc") {
        mat = SpMatManager::createCsc(rows, cols, nnz, dInd1.get(), dInd2.get(), dVal.get(), ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_32I, idxBaseEnum, valueType);
    } else { // coo
        mat = SpMatManager::createCoo(rows, cols, nnz, dInd1.get(), dInd2.get(), dVal.get(), ACL_SPARSE_INDEX_32I,
                                      idxBaseEnum, valueType);
    }

    // 4. 可选：先经配套算子写入 batchCount（仅 COO 支持SetStridedBatch）
    if (preSet == 1) {
        r.setRet = aclsparseCooSetStridedBatch(mat.get(), batchCount, batchStride);
        if (r.setRet != ACL_SPARSE_STATUS_SUCCESS) {
            return r;
        }
    }

    // 5. Get 读回 batchCount
    r.getRet = aclsparseSpMatGetStridedBatch(mat.get(), &r.outBatchCount);
    if (r.getRet != ACL_SPARSE_STATUS_SUCCESS) {
        return r;
    }
    return r;
}
