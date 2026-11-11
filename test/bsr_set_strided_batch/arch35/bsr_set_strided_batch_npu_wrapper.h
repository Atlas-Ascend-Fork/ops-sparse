/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_BSR_SET_STRIDED_BATCH_BSR_SET_STRIDED_BATCH_NPU_WRAPPER_H_
#define TEST_BSR_SET_STRIDED_BATCH_BSR_SET_STRIDED_BATCH_NPU_WRAPPER_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "sparse_dtype_utils.h"    // ParseValueType：CSV dtype 列名 -> aclDataType（BSR 族共享）
#include "aclsparse_descr_internal.h"  // 白盒：公共 getter 只回读 batchCount，三 stride 经内部结构核对

// ============================================================================
// NPU Workflow Result
// ============================================================================
struct BsrSetStridedBatchNpuResult {
    aclsparseStatus_t createRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t getBeforeRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};  // Set 之前 GetStridedBatch（默认 1）
    aclsparseStatus_t setRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};        // aclsparseBsrSetStridedBatch 返回码
    aclsparseStatus_t getRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};        // Set 之后 GetStridedBatch 返回码

    int defaultBatchCount{0};  // Set 之前回读的 batchCount（期望 1）
    int outBatchCount{0};      // Set 之后回读的 batchCount（期望与输入一致）

    // 白盒字段验证：公共 getter 只回读 batchCount（与 cuSPARSE 一致），三个
    // batch stride 通过内部描述符结构直接核对，防止 int64 截断（先例 csr wrapper）。
    int whiteBoxBatchCount{-1};
    int64_t whiteBoxOffsetsBatchStride{-1};
    int64_t whiteBoxColumnsBatchStride{-1};
    int64_t whiteBoxValuesBatchStride{-1};
};

// ============================================================================
// NPU wrapper：CreateBsr(几何) -> Get(默认 1) -> SetStridedBatch -> Get 回读。
// 纯 Host 侧描述符操作，不启动 NPU kernel。BSR 无 SpMatManager 工厂，沿用
// unique_ptr + aclsparseDestroySpMat 删除器 RAII 形态（先例 csr_set_strided_batch
// wrapper / spmat_get_strided_batch E8）。
//
// 注意（wrapper 适用范围声明）：设备缓冲仅按单 batch 大小占位分配，用于承载
// 描述符持有的设备指针；setter/getter 为纯元数据操作、不访问数组内容。本
// wrapper 仅用于元数据测试，不可直接扩展为批量消费算子（SpMM/SDDMM）的
// wrapper——后者需按 (batchCount-1)*stride+span 的 extent 分配设备缓冲。
//
// DeviceBuffer::copyFrom / alloc 失败时抛 std::runtime_error（框架语义），
// gtest 以未捕获异常判用例失败。
// ============================================================================
inline BsrSetStridedBatchNpuResult BsrSetStridedBatchNpu(aclrtStream stream,
    int64_t blockRows, int64_t blockCols, int64_t blockNnz, int64_t blockSize,
    aclDataType valueType, int batchCount,
    int64_t offsetsBatchStride, int64_t columnsBatchStride, int64_t valuesBatchStride)
{
    using namespace sparse_test;
    (void)stream;
    BsrSetStridedBatchNpuResult r{};

    // 输入张量迁移到 NPU 设备（setter 不读数据，设备缓冲仅占位）：
    // offsets 为 blockRows+1 端点、colInd 为 blockNnz 索引、values 为 nnzb×bs² 元素
    //（缓冲准备阶段：元素宽度按 valueType 经共享 ElementBytes 计算）
    const std::vector<int32_t> hostOffsets(static_cast<size_t>(blockRows) + 1u, 0);
    const std::vector<int32_t> hostColInd(static_cast<size_t>(blockNnz), 0);
    auto dOffsets = DeviceBuffer::copyFrom(hostOffsets.data(),
                                           hostOffsets.size() * sizeof(int32_t));
    auto dColInd = DeviceBuffer::copyFrom(hostColInd.data(),
                                          hostColInd.size() * sizeof(int32_t));
    const int64_t valuesCount = blockNnz * blockSize * blockSize;
    auto dValues = DeviceBuffer::alloc(static_cast<size_t>(valuesCount) *
                                       sparse_test::ElementBytes(valueType));

    // aclsparseCreateBsr（unique_ptr + aclsparseDestroySpMat 删除器 RAII）
    aclsparseSpMatDescr_t descr = nullptr;
    r.createRet = aclsparseCreateBsr(&descr, blockRows, blockCols, blockNnz, blockSize,
        blockSize, dOffsets.get(), dColInd.get(), dValues.get(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, valueType, ACL_SPARSE_ORDER_ROW);
    if (r.createRet != ACL_SPARSE_STATUS_SUCCESS || descr == nullptr) {
        std::cerr << "[NPU] aclsparseCreateBsr failed: " << r.createRet << std::endl;
        return r;
    }
    std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)> mat(
        descr, &aclsparseDestroySpMat);

    // Set 之前回读默认 batchCount（新建描述符未设置批量语义时应为 1）
    r.getBeforeRet = aclsparseSpMatGetStridedBatch(mat.get(), &r.defaultBatchCount);
    if (r.getBeforeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseSpMatGetStridedBatch(before set) failed: "
                  << r.getBeforeRet << std::endl;
        return r;
    }

    // aclsparseBsrSetStridedBatch
    r.setRet = aclsparseBsrSetStridedBatch(mat.get(), batchCount, offsetsBatchStride,
                                           columnsBatchStride, valuesBatchStride);
    if (r.setRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseBsrSetStridedBatch failed: " << r.setRet << std::endl;
        return r;
    }

    // Set 之后 GetStridedBatch 回读 + 白盒核对 batch 四字段（防 int64 截断）
    r.getRet = aclsparseSpMatGetStridedBatch(mat.get(), &r.outBatchCount);
    if (r.getRet == ACL_SPARSE_STATUS_SUCCESS) {
        r.whiteBoxBatchCount = mat.get()->batchCount;
        r.whiteBoxOffsetsBatchStride = mat.get()->offsetsBatchStride;
        r.whiteBoxColumnsBatchStride = mat.get()->columnsBatchStride;
        r.whiteBoxValuesBatchStride = mat.get()->valuesBatchStride;
    }
    return r;
}

#endif  // TEST_BSR_SET_STRIDED_BATCH_BSR_SET_STRIDED_BATCH_NPU_WRAPPER_H_
