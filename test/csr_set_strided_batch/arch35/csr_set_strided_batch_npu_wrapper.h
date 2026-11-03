/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSR_SET_STRIDED_BATCH_CSR_SET_STRIDED_BATCH_NPU_WRAPPER_H_
#define TEST_CSR_SET_STRIDED_BATCH_CSR_SET_STRIDED_BATCH_NPU_WRAPPER_H_

#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "aclsparse_descr_internal.h"

// ============================================================================
// NPU Workflow Result
// ============================================================================
struct CsrSetStridedBatchNpuResult {
    aclsparseStatus_t createRet;
    aclsparseStatus_t getBeforeRet;   // Set 之前 GetStridedBatch 的返回码（默认 batchCount=1）
    aclsparseStatus_t setRet;         // aclsparseCsrSetStridedBatch 返回码
    aclsparseStatus_t getRet;         // Set 之后 GetStridedBatch 返回码

    int defaultBatchCount{0};         // Set 之前回读的 batchCount（期望 1）
    int outBatchCount{0};             // Set 之后回读的 batchCount（期望与输入一致）

    // 白盒字段验证：公共 getter 只回读 batchCount（与 cuSPARSE 一致），
    // 两个 batch stride 通过内部描述符结构直接核对，防止 int64 截断；
    // columnsValuesBatchStride 写入共享拆分字段 columnsBatchStride 与
    // valuesBatchStride 同值（cuSPARSE CSR 语义：列索引/值数组共用一个步长）。
    int64_t whiteBoxOffsetsBatchStride{-1};
    int64_t whiteBoxColumnsBatchStride{-1};
    int64_t whiteBoxValuesBatchStride{-1};
};

// ============================================================================
// NPU wrapper: CreateCsr -> (Get 默认值) -> SetStridedBatch -> GetStridedBatch
// 纯 Host 侧描述符操作，不启动 NPU kernel。
//
// 注意（wrapper 适用范围声明）：设备缓冲仅按单 batch 大小占位分配，用于承载
// 描述符持有的设备指针；setter/getter 为纯元数据操作、不访问数组内容。本
// wrapper 仅用于元数据测试，不可直接扩展为批量消费算子（SpMM/SDDMM）的
// wrapper——后者需按 (batchCount-1)*stride+span 的 extent 分配设备缓冲。
//
// DeviceBuffer::copyFrom / alloc 失败时抛 std::runtime_error（框架语义），
// gtest 以未捕获异常判用例失败，不会以空指针静默通过宽松的 CreateCsr。
// ============================================================================
inline CsrSetStridedBatchNpuResult CsrSetStridedBatchNpu(aclrtStream stream,
    int rows, int cols, int nnz, int idx_base, bool isFp16,
    int batchCount, int64_t offsetsBatchStride, int64_t columnsValuesBatchStride)
{
    using namespace sparse_test;
    (void)stream;
    CsrSetStridedBatchNpuResult r{};

    // 输入张量迁移到 NPU 设备（setter 不读数据，设备缓冲仅占位）
    std::vector<int32_t> hostRowOffsets(static_cast<size_t>(rows) + 1u, 0);
    std::vector<int32_t> hostColInd(static_cast<size_t>(nnz), 0);
    auto dRowOff = DeviceBuffer::copyFrom(hostRowOffsets.data(),
                                          (static_cast<size_t>(rows) + 1u) * sizeof(int32_t));
    auto dColInd = DeviceBuffer::copyFrom(hostColInd.data(),
                                          static_cast<size_t>(nnz) * sizeof(int32_t));
    size_t valBytes = isFp16 ? (static_cast<size_t>(nnz) * sizeof(uint16_t))
                             : (static_cast<size_t>(nnz) * sizeof(float));
    auto dValues = DeviceBuffer::alloc(valBytes);

    aclDataType valueType = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;
    aclsparseIndexBase_t idxBase =
        (idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    // aclsparseCreateCsr
    aclsparseSpMatDescr_t descr = nullptr;
    r.createRet = aclsparseCreateCsr(&descr, rows, cols, nnz,
        dRowOff.get(), dColInd.get(), dValues.get(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_32I, idxBase, valueType);
    if (r.createRet != ACL_SPARSE_STATUS_SUCCESS || descr == nullptr) {
        std::cerr << "[NPU] aclsparseCreateCsr failed: " << r.createRet << std::endl;
        return r;
    }

    // Set 之前回读默认 batchCount（新建描述符未设置批量语义时应为 1）
    r.getBeforeRet = aclsparseSpMatGetStridedBatch(descr, &r.defaultBatchCount);
    if (r.getBeforeRet != ACL_SPARSE_STATUS_SUCCESS) {
        aclsparseDestroySpMat(descr);
        std::cerr << "[NPU] aclsparseSpMatGetStridedBatch(before set) failed: "
                  << r.getBeforeRet << std::endl;
        return r;
    }

    // aclsparseCsrSetStridedBatch
    r.setRet = aclsparseCsrSetStridedBatch(descr, batchCount,
                                           offsetsBatchStride, columnsValuesBatchStride);
    if (r.setRet != ACL_SPARSE_STATUS_SUCCESS) {
        aclsparseDestroySpMat(descr);
        std::cerr << "[NPU] aclsparseCsrSetStridedBatch failed: " << r.setRet << std::endl;
        return r;
    }

    // aclsparseSpMatGetStridedBatch 回读 + 白盒核对三个 batch stride 字段
    r.getRet = aclsparseSpMatGetStridedBatch(descr, &r.outBatchCount);
    if (r.getRet == ACL_SPARSE_STATUS_SUCCESS) {
        r.whiteBoxOffsetsBatchStride = descr->offsetsBatchStride;
        r.whiteBoxColumnsBatchStride = descr->columnsBatchStride;
        r.whiteBoxValuesBatchStride = descr->valuesBatchStride;
    }

    aclsparseDestroySpMat(descr);
    return r;
}

#endif  // TEST_CSR_SET_STRIDED_BATCH_CSR_SET_STRIDED_BATCH_NPU_WRAPPER_H_
