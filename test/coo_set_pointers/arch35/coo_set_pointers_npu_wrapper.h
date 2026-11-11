/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_COO_SET_POINTERS_COO_SET_POINTERS_NPU_WRAPPER_H_
#define TEST_COO_SET_POINTERS_COO_SET_POINTERS_NPU_WRAPPER_H_

#include <cstdint>
#include <iostream>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "aclsparse_descr_internal.h"  // 白盒：ptrs/idxs 复用同步 + structureGeneration 锁定

// ============================================================================
// NPU Workflow Result：每步返回码 + CooGet 九字段回读 + 白盒复用/代际字段。
// 未执行步骤（Create 失败提前返回）保持 NOT_INITIALIZED 哨兵，不误报 SUCCESS。
// ============================================================================
struct CooSetPointersNpuResult {
    aclsparseStatus_t createRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t setRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t getRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};

    // CooGet roundtrip 九字段（指针在 wrapper 内与 P1 比较，不向外部暴露
    // 已释放的设备指针）
    int64_t rows{-1};
    int64_t cols{-1};
    int64_t nnz{-1};
    bool outRowIndsMatchP1{false};
    bool outColIndsMatchP1{false};
    bool outValuesMatchP1{false};
    aclsparseIndexType_t idxType{static_cast<aclsparseIndexType_t>(-1)};
    aclsparseIndexBase_t idxBase{static_cast<aclsparseIndexBase_t>(-1)};
    aclDataType valueType{static_cast<aclDataType>(-1)};

    // 白盒字段：COO 的 SetPointers 同时写入复用字段 ptrs/idxs（descr.cpp:1083-1087），
    // CooGet 读回的正是 ptrs/idxs——复用同步断言锁定该写入语义；
    // COO 路径不调用 AclsparseMarkSpMatStructureChanged，generation 恒不变（行为锁定）。
    bool whiteBoxPtrsMatchesRowInds{false};
    bool whiteBoxIdxsMatchesColInds{false};
    uint64_t whiteBoxGenerationBefore{0};
    uint64_t whiteBoxGenerationAfter{0};
};

// ============================================================================
// 缓冲准备阶段（阶段函数）：P0/P1 指针组占位设备缓冲
//（Set/Create/Get 均不解引用，仅按占位规模分配；nnz=0 时按 1 元素最小
// 占位以保证指针非空可区分；nullPtrs=1 时 P1 留空——三指针全传 null，
// 仅 nnz=0 行合法）。
// ============================================================================
struct CooPointerBuffers {
    sparse_test::DeviceBuffer p0RowInds;
    sparse_test::DeviceBuffer p0ColInds;
    sparse_test::DeviceBuffer p0Values;
    sparse_test::DeviceBuffer p1RowInds;
    sparse_test::DeviceBuffer p1ColInds;
    sparse_test::DeviceBuffer p1Values;
};

inline CooPointerBuffers AllocCooPointerBuffers(int64_t nnz, int idxType, bool isFp16,
                                                bool nullPtrs)
{
    using namespace sparse_test;
    // 索引大小随 idxType，值大小随 dtype
    const size_t idxBytes = (idxType == 1) ? sizeof(int64_t) : sizeof(int32_t);
    const size_t valBytes = isFp16 ? sizeof(uint16_t) : sizeof(float);
    const size_t elemCount = static_cast<size_t>(nnz > 0 ? nnz : 1);
    CooPointerBuffers b;
    b.p0RowInds = DeviceBuffer::alloc(elemCount * idxBytes);
    b.p0ColInds = DeviceBuffer::alloc(elemCount * idxBytes);
    b.p0Values = DeviceBuffer::alloc(elemCount * valBytes);
    if (!nullPtrs) {
        b.p1RowInds = DeviceBuffer::alloc(elemCount * idxBytes);
        b.p1ColInds = DeviceBuffer::alloc(elemCount * idxBytes);
        b.p1Values = DeviceBuffer::alloc(elemCount * valBytes);
    }
    return b;
}

// ============================================================================
// Get 回读阶段（阶段函数）：CooGet 九字段 roundtrip，指针==P1 在 wrapper 内
// 比较后落位结果结构（不向外部暴露已释放的设备指针）；Get 失败时几何/指针
// 回读字段保持哨兵不动。
// ============================================================================
inline aclsparseStatus_t CooGetRoundTrip(aclsparseSpMatDescr_t descr, const void *p1RowInds,
                                         const void *p1ColInds, const void *p1Values,
                                         CooSetPointersNpuResult &r)
{
    int64_t outRows = -1;
    int64_t outCols = -1;
    int64_t outNnz = -1;
    void *outRowInds = nullptr;
    void *outColInds = nullptr;
    void *outValues = nullptr;
    const aclsparseStatus_t getRet = aclsparseCooGet(descr, &outRows, &outCols, &outNnz,
        &outRowInds, &outColInds, &outValues, &r.idxType, &r.idxBase, &r.valueType);
    if (getRet == ACL_SPARSE_STATUS_SUCCESS) {
        r.rows = outRows;
        r.cols = outCols;
        r.nnz = outNnz;
        r.outRowIndsMatchP1 = (outRowInds == p1RowInds);
        r.outColIndsMatchP1 = (outColInds == p1ColInds);
        r.outValuesMatchP1 = (outValues == p1Values);
    }
    return getRet;
}

// ============================================================================
// NPU wrapper：CreateCoo(几何，初始指针组 P0) -> SetPointers(P1) -> CooGet 回读。
// 纯 Host 侧描述符操作，不启动 NPU kernel；Create/Set/Get 均不解引用数组内容，
// 设备缓冲仅按占位规模分配（P0/P1 各自独立 DeviceBuffer；nnz=0 时按 1 元素最小
// 占位以保证指针非空可区分）。
// COO 使用 SpMatManager::createCoo 工厂（RAII，失败抛 std::runtime_error，
// 由本 wrapper 捕获并置失败返回码）。
// ============================================================================
inline CooSetPointersNpuResult CooSetPointersNpu(aclrtStream stream,
    int64_t rows, int64_t cols, int64_t nnz, int idxType, int idxBase,
    bool isFp16, bool nullPtrs)
{
    using namespace sparse_test;
    (void)stream;
    CooSetPointersNpuResult r{};

    const aclsparseIndexType_t cooIdxType =
        (idxType == 1) ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    const aclsparseIndexBase_t idxBaseEnum =
        (idxBase == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
    const aclDataType valueType = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;

    // 1. 缓冲准备阶段
    auto buf = AllocCooPointerBuffers(nnz, idxType, isFp16, nullPtrs);

    // 2. aclsparseCreateCoo（P0 初始指针组）
    SpMatManager mat;
    try {
        mat = SpMatManager::createCoo(rows, cols, nnz, buf.p0RowInds.get(),
                                      buf.p0ColInds.get(), buf.p0Values.get(), cooIdxType,
                                      idxBaseEnum, valueType);
    } catch (const std::exception &e) {
        std::cerr << "[NPU] aclsparseCreateCoo failed: " << e.what() << std::endl;
        r.createRet = ACL_SPARSE_STATUS_INVALID_VALUE;  // 工厂抛异常 = Create 未成功
        return r;
    }
    r.createRet = ACL_SPARSE_STATUS_SUCCESS;

    // 白盒记录 Set 前代际（COO 路径不失效 structure，after 应等于 before）
    r.whiteBoxGenerationBefore = mat.get()->structureGeneration;

    // 3. aclsparseCooSetPointers（P1 新指针组重绑）
    r.setRet = aclsparseCooSetPointers(mat.get(), buf.p1RowInds.get(), buf.p1ColInds.get(),
                                       buf.p1Values.get());
    if (r.setRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseCooSetPointers failed: " << r.setRet << std::endl;
        return r;
    }

    // 白盒：复用字段同步 + 代际不变
    r.whiteBoxPtrsMatchesRowInds = (mat.get()->ptrs == mat.get()->rowInds);
    r.whiteBoxIdxsMatchesColInds = (mat.get()->idxs == mat.get()->colInds);
    r.whiteBoxGenerationAfter = mat.get()->structureGeneration;

    // 4. Get 回读阶段（九字段；指针==P1 在 wrapper 内比较）
    r.getRet = CooGetRoundTrip(mat.get(), buf.p1RowInds.get(), buf.p1ColInds.get(),
                               buf.p1Values.get(), r);
    if (r.getRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseCooGet failed: " << r.getRet << std::endl;
        return r;
    }
    return r;
}

#endif  // TEST_COO_SET_POINTERS_COO_SET_POINTERS_NPU_WRAPPER_H_
