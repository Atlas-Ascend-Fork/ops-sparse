/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

// ============================================================================
// NPU Workflow Result
// ============================================================================
struct BlockedEllGetNpuResult {
    aclsparseStatus_t createRet{ACL_SPARSE_STATUS_SUCCESS};
    // Get 未执行（Create 失败/参数非法提前返回）时保持 NOT_INITIALIZED，不误报 SUCCESS
    aclsparseStatus_t getRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};

    // BlockedEllGet 读回的标量字段（Host 元数据，非张量计算）
    int64_t rows{0};
    int64_t cols{0};
    int64_t ellBlockSize{0};
    int64_t ellCols{0};
    aclsparseIndexType_t idxType{};
    aclsparseIndexBase_t idxBase{};
    aclDataType valueType{};

    // Get 读回指针是否与 Create 传入的设备指针逐字一致（指针值在 wrapper 内比较，
    // 不向外部暴露已释放的设备指针）
    bool devPtrMatch{false};
};

// ============================================================================
// NPU wrapper: CreateBlockedEll -> BlockedEllGet accessor 验证
// 全程在 NPU 侧建设备缓冲存指针；Host 仅做元数据生成与 accessor 断言。
// 仅覆盖 I32 索引路径（64I round-trip 由测试用例直接绑定 int64 缓冲覆盖）。
// ============================================================================
inline BlockedEllGetNpuResult BlockedEllGetNpu(aclrtStream stream,
                                               const std::vector<int32_t> &hostColInd,
                                               int rows, int cols, int ellBlockSize,
                                               int ellCols, int idx_base, bool isFp16)
{
    using namespace sparse_test;
    (void)stream;
    BlockedEllGetNpuResult r{};

    // ellBlockSize 是后续容量换算的唯一除数，单独前置判零
    //（G.EXP.22：CI 仅识别 == 0 / != 0 形式的判零，<= 0 不被认可）
    if (ellBlockSize == 0) {
        r.createRet = ACL_SPARSE_STATUS_INVALID_VALUE;
        return r;
    }
    if (rows < 0 || cols < 0 || ellBlockSize < 0 || ellCols < 0) {
        r.createRet = ACL_SPARSE_STATUS_INVALID_VALUE;
        return r;
    }

    // 容量换算与 BlockedEllCapacity 一致：补齐后的块网格 * bs^2
    const int64_t blockRows = (static_cast<int64_t>(rows) + ellBlockSize - 1) / ellBlockSize;
    const int64_t slots = (static_cast<int64_t>(ellCols) + ellBlockSize - 1) / ellBlockSize;
    const int64_t indexCount = blockRows * slots;
    const int64_t valueCount = indexCount * static_cast<int64_t>(ellBlockSize) * ellBlockSize;
    if (indexCount > 0 && static_cast<int64_t>(hostColInd.size()) != indexCount) {
        r.createRet = ACL_SPARSE_STATUS_INVALID_VALUE;
        return r;
    }

    // 输入张量迁移到 NPU 设备（accessor 不读数据，设备缓冲存入描述符供回读断言）
    auto dCol = DeviceBuffer::copyFrom(hostColInd.data(),
        static_cast<size_t>(indexCount) * sizeof(int32_t));

    aclDataType valueType = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;
    aclsparseIndexBase_t idxBase =
        (idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    size_t valBytes = isFp16 ? (static_cast<size_t>(valueCount) * sizeof(uint16_t))
                             : (static_cast<size_t>(valueCount) * sizeof(float));
    auto dVal = DeviceBuffer::alloc(valBytes);

    // aclsparseCreateBlockedEll
    aclsparseSpMatDescr_t descr = nullptr;
    r.createRet = aclsparseCreateBlockedEll(&descr, rows, cols, ellBlockSize, ellCols,
        dCol.get(), dVal.get(), ACL_SPARSE_INDEX_32I, idxBase, valueType);
    if (r.createRet != ACL_SPARSE_STATUS_SUCCESS || descr == nullptr) {
        std::cerr << "[NPU] aclsparseCreateBlockedEll failed: " << r.createRet << std::endl;
        return r;
    }

    // aclsparseBlockedEllGet（设备指针读回到局部变量，仅作指针值比较）
    void *devColInd = nullptr;
    void *devValue = nullptr;
    r.getRet = aclsparseBlockedEllGet(descr, &r.rows, &r.cols, &r.ellBlockSize, &r.ellCols,
        &devColInd, &devValue, &r.idxType, &r.idxBase, &r.valueType);

    if (r.getRet != ACL_SPARSE_STATUS_SUCCESS) {
        aclsparseDestroySpMat(descr);
        std::cerr << "[NPU] aclsparseBlockedEllGet failed: " << r.getRet << std::endl;
        return r;
    }

    // accessor 语义断言：Get 读回的设备指针应与 Create 传入的逐字一致
    //（指针值比较后即丢弃，不随结果结构体返回已释放的设备指针）
    r.devPtrMatch = (devColInd == dCol.get() && devValue == dVal.get());

    aclsparseDestroySpMat(descr);
    return r;
}

// ============================================================================
// 生成 Blocked-ELL 测试数据（host 侧仅生成输入，不在 CPU 做算子计算）
// 列块索引取 [idxBase, ceil(cols/bs)-1+idxBase]，随机撒 -1 表示空块。
// ============================================================================
inline void GenerateBlockedEllData(int rows, int cols, int ellBlockSize, int ellCols,
                                   int idx_base, uint32_t seed,
                                   std::vector<int32_t> &colInd)
{
    colInd.clear();
    // ellBlockSize 是后续块网格换算的唯一除数，单独前置判零
    //（G.EXP.22：CI 仅识别 == 0 / != 0 形式的判零，<= 0 不被认可）
    if (ellBlockSize == 0) {
        return;
    }
    if (rows <= 0 || cols <= 0 || ellBlockSize < 0 || ellCols < 0) {
        return;
    }
    const int64_t blockRows = (static_cast<int64_t>(rows) + ellBlockSize - 1) / ellBlockSize;
    const int64_t slots = (static_cast<int64_t>(ellCols) + ellBlockSize - 1) / ellBlockSize;
    const int64_t indexCount = blockRows * slots;
    const int64_t maxBlockCol = (static_cast<int64_t>(cols) + ellBlockSize - 1) / ellBlockSize;
    colInd.reserve(static_cast<size_t>(indexCount));
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> colDist(0, static_cast<int32_t>(maxBlockCol) - 1);
    std::uniform_int_distribution<int32_t> holeDist(0, 9);
    const int32_t base = (idx_base == 1) ? 1 : 0;
    for (int64_t i = 0; i < indexCount; i++) {
        // 约 1/10 的 slot 填 -1 表示空块（Blocked-ELL 合法取值）
        colInd.push_back((holeDist(rng) == 0) ? -1 : (colDist(rng) + base));
    }
}
