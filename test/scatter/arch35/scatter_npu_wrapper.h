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

#ifndef TEST_SCATTER_ARCH35_SCATTER_NPU_WRAPPER_H_
#define TEST_SCATTER_ARCH35_SCATTER_NPU_WRAPPER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "fill.h"
#include "sparse_test.h"

// aclsparseScatter 声明已由 cann_ops_sparse.h 提供，无需重复前向声明。

namespace sparse_test {

// ============================================================================
// Helper: byte size per element for a given ACL data type
//
// TODO: 此处 ScatterAclDataTypeSize 与 csr2csc_ex2 等算子的同类实现重复，
// 属框架级缺失正向 size 工具的问题。待测试框架补齐公共 aclDataType → size
// 工具后，各算子应统一改用框架公共实现，删除本算子内的重复定义。
// ============================================================================
inline size_t ScatterAclDataTypeSize(aclDataType dt) {
    switch (dt) {
        case ACL_INT8:    return 1;
        case ACL_FLOAT16: return 2;
        case ACL_BF16:    return 2;
        case ACL_FLOAT:   return 4;
        case ACL_COMPLEX64: return 8;
        default:          return 4;
    }
}

// ============================================================================
// ScatterNpuResult: output of the NPU scatter wrapper
// ============================================================================
struct ScatterNpuResult {
    std::vector<uint8_t> yBytes;   // dnSize * valSize bytes (raw output Y)
    aclsparseStatus_t execRet;     // return code from aclsparseScatter
};

// ============================================================================
// NPU wrapper for aclsparseScatter.
//
// Full workflow:
//   1. Allocate device buffers for indices, values, Y (with 16-byte alignment)
//   2. Copy host data H2D
//   3. Create SpVec (const) and DnVec descriptors via RAII managers
//   4. Call aclsparseScatter
//   5. Synchronize stream
//   6. Copy Y D2H
//
// Device memory managed via DeviceBuffer RAII.
// Descriptors managed via SpVecManager / DnVecManager RAII.
// ============================================================================
struct ScatterCallFields {
    aclrtStream stream = nullptr;
    int64_t vecSize = 0;
    int64_t nnz = 0;
    int64_t dnSize = 0;
    const void* indicesHost = nullptr;
    const void* valuesHost = nullptr;
    const void* yInitHost = nullptr;
    aclsparseIndexType_t idxType = ACL_SPARSE_INDEX_32I;
    aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO;
    aclDataType valType = ACL_FLOAT;
};

struct ScatterNpuArgs : ScatterCallFields {
    HandleManager* handle = nullptr;
};

inline ScatterNpuResult ScatterNpu(const ScatterNpuArgs& a)
{
    ScatterNpuResult result{};
    size_t valSize = ScatterAclDataTypeSize(a.valType);
    size_t idxSize = (a.idxType == ACL_SPARSE_INDEX_64I) ? sizeof(int64_t) : sizeof(int32_t);

    a.handle->setStream(a.stream);

    // 1. Device buffers (RAII)
    //    Ensure minimum 1-byte allocation for nnz=0 / dnSize=0 to avoid null ptr
    size_t idxBytes = std::max(static_cast<size_t>(a.nnz) * idxSize, size_t(1));
    size_t valBytes = std::max(static_cast<size_t>(a.nnz) * valSize, size_t(1));
    size_t yBytes   = std::max(static_cast<size_t>(a.dnSize) * valSize, size_t(1));

    auto dIndices = (a.nnz > 0 && a.indicesHost != nullptr)
        ? DeviceBuffer::copyFrom(a.indicesHost, static_cast<size_t>(a.nnz) * idxSize)
        : DeviceBuffer::alloc(idxBytes);

    auto dValues = (a.nnz > 0 && a.valuesHost != nullptr)
        ? DeviceBuffer::copyFrom(a.valuesHost, static_cast<size_t>(a.nnz) * valSize)
        : DeviceBuffer::alloc(valBytes);

    auto dY = (a.dnSize > 0 && a.yInitHost != nullptr)
        ? DeviceBuffer::copyFrom(a.yInitHost, static_cast<size_t>(a.dnSize) * valSize)
        : DeviceBuffer::alloc(yBytes);

    // 2. Create descriptors (RAII, auto-destroy on scope exit)
    SpVecManager vecX = SpVecManager::createConst(
        a.vecSize, a.nnz, dIndices.get(), dValues.get(),
        a.idxType, a.idxBase, a.valType);
    DnVecManager vecY = DnVecManager::create(
        a.dnSize, dY.get(), a.valType);

    // 3. Call aclsparseScatter
    result.execRet = aclsparseScatter(a.handle->get(), vecX.cget(), vecY.get());
    if (result.execRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseScatter failed: " << result.execRet << std::endl;
        return result;
    }

    // 4. Synchronize
    if (aclrtSynchronizeStream(a.stream) != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed" << std::endl;
        result.execRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return result;
    }

    // 5. D2H copy Y
    result.yBytes.resize(static_cast<size_t>(a.dnSize) * valSize);
    if (a.dnSize > 0) {
        dY.copyToHost(result.yBytes.data(), static_cast<size_t>(a.dnSize) * valSize);
    }
    return result;
}  // exit scope → RAII auto-destroy descriptors + free device memory

}  // namespace sparse_test

#endif  // TEST_SCATTER_ARCH35_SCATTER_NPU_WRAPPER_H_
