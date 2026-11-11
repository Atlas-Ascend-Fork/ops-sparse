/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSR_SET_POINTERS_CSR_SET_POINTERS_NPU_WRAPPER_H_
#define TEST_CSR_SET_POINTERS_CSR_SET_POINTERS_NPU_WRAPPER_H_

#include <cstdint>
#include <iostream>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "aclsparse_descr_internal.h"  // 白盒：structureGeneration +1 断言

// ============================================================================
// NPU Workflow Result：每步返回码 + CsrGet 十字段回读 + 白盒代际字段。
// 未执行步骤（Create 失败提前返回）保持 NOT_INITIALIZED 哨兵，不误报 SUCCESS。
// ============================================================================
struct CsrSetPointersNpuResult {
    aclsparseStatus_t createRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t setRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t getRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};

    // CsrGet roundtrip 十字段（指针在 wrapper 内与 P1 比较，不向外部暴露
    // 已释放的设备指针）
    int64_t rows{-1};
    int64_t cols{-1};
    int64_t nnz{-1};
    bool outOffsetsMatchP1{false};
    bool outColIndMatchP1{false};
    bool outValuesMatchP1{false};
    aclsparseIndexType_t ptrType{static_cast<aclsparseIndexType_t>(-1)};
    aclsparseIndexType_t idxType{static_cast<aclsparseIndexType_t>(-1)};
    aclsparseIndexBase_t idxBase{static_cast<aclsparseIndexBase_t>(-1)};
    aclDataType valueType{static_cast<aclDataType>(-1)};

    // 白盒代际：CSV 成功行的 P1 相对 P0 均发生 ptrs/idxs 变化，
    // SetCompressedPointers 的 structureChanged 分支应使 generation+1（:1006-1008）
    uint64_t whiteBoxGenerationBefore{0};
    uint64_t whiteBoxGenerationAfter{0};
};

// ============================================================================
// 缓冲准备阶段（阶段函数）：P0/P1 指针组占位设备缓冲
//（Create/Set/Get 均不解引用；P1 按 null_ptrs 行构造：0=三指针全新有效缓冲；
// 1=offsets 有效 + idx/val null（仅 nnz=0 合法）；2=offsets null + idx/val 有效
//（仅零维行合法）。零规模数组按 1 元素最小占位分配，保证指针非空可区分）。
// ============================================================================
struct CsrPointerBuffers {
    sparse_test::DeviceBuffer p0Offsets;
    sparse_test::DeviceBuffer p0ColInd;
    sparse_test::DeviceBuffer p0Values;
    sparse_test::DeviceBuffer p1Offsets;
    sparse_test::DeviceBuffer p1ColInd;
    sparse_test::DeviceBuffer p1Values;
};

inline CsrPointerBuffers AllocCsrPointerBuffers(int64_t rows, int64_t nnz, int idxType,
                                                bool isFp16, int nullPtrs)
{
    using namespace sparse_test;
    // offsets 为 rows+1 端点（恒 >=1 元素）；idx/val 按 nnz
    const size_t idxBytes = (idxType == 1) ? sizeof(int64_t) : sizeof(int32_t);
    const size_t valBytes = isFp16 ? sizeof(uint16_t) : sizeof(float);
    const size_t offsetsCount = static_cast<size_t>(rows) + 1u;
    const size_t elemCount = static_cast<size_t>(nnz > 0 ? nnz : 1);
    CsrPointerBuffers b;
    b.p0Offsets = DeviceBuffer::alloc(offsetsCount * idxBytes);
    b.p0ColInd = DeviceBuffer::alloc(elemCount * idxBytes);
    b.p0Values = DeviceBuffer::alloc(elemCount * valBytes);
    if (nullPtrs == 0) {
        b.p1Offsets = DeviceBuffer::alloc(offsetsCount * idxBytes);
        b.p1ColInd = DeviceBuffer::alloc(elemCount * idxBytes);
        b.p1Values = DeviceBuffer::alloc(elemCount * valBytes);
    } else if (nullPtrs == 1) {
        // offsets 有效新缓冲；indices/values 传 null（仅 nnz=0 行合法）
        b.p1Offsets = DeviceBuffer::alloc(offsetsCount * idxBytes);
    } else {
        // offsets 传 null（仅零维行 rows=0 或 cols=0 合法）；indices/values 有效
        b.p1ColInd = DeviceBuffer::alloc(elemCount * idxBytes);
        b.p1Values = DeviceBuffer::alloc(elemCount * valBytes);
    }
    return b;
}

// ============================================================================
// Get 回读阶段（阶段函数）：CsrGet 十字段 roundtrip，指针==P1 在 wrapper 内
// 比较后落位结果结构（不向外部暴露已释放的设备指针）；Get 失败时几何/指针
// 回读字段保持哨兵不动。
// ============================================================================
inline aclsparseStatus_t CsrGetRoundTrip(aclsparseSpMatDescr_t descr, const void *p1Offsets,
                                         const void *p1ColInd, const void *p1Values,
                                         CsrSetPointersNpuResult &r)
{
    int64_t outRows = -1;
    int64_t outCols = -1;
    int64_t outNnz = -1;
    void *outOffsets = nullptr;
    void *outColInd = nullptr;
    void *outValues = nullptr;
    const aclsparseStatus_t getRet = aclsparseCsrGet(descr, &outRows, &outCols, &outNnz,
        &outOffsets, &outColInd, &outValues, &r.ptrType, &r.idxType, &r.idxBase,
        &r.valueType);
    if (getRet == ACL_SPARSE_STATUS_SUCCESS) {
        r.rows = outRows;
        r.cols = outCols;
        r.nnz = outNnz;
        r.outOffsetsMatchP1 = (outOffsets == p1Offsets);
        r.outColIndMatchP1 = (outColInd == p1ColInd);
        r.outValuesMatchP1 = (outValues == p1Values);
    }
    return getRet;
}

// ============================================================================
// NPU wrapper：CreateCsr(几何，初始指针组 P0) -> SetPointers(P1) -> CsrGet 回读。
// 纯 Host 侧描述符操作，不启动 NPU kernel；Create/Set/Get 均不解引用数组内容。
// CSR 使用 SpMatManager::createCsr 工厂（RAII，失败抛 std::runtime_error，
// 由本 wrapper 捕获并置失败返回码）。
// ============================================================================
inline CsrSetPointersNpuResult CsrSetPointersNpu(aclrtStream stream,
    int64_t rows, int64_t cols, int64_t nnz, int idxType, int idxBase,
    bool isFp16, int nullPtrs)
{
    using namespace sparse_test;
    (void)stream;
    CsrSetPointersNpuResult r{};

    const aclsparseIndexType_t csrIdxType =
        (idxType == 1) ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    const aclsparseIndexBase_t idxBaseEnum =
        (idxBase == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
    const aclDataType valueType = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;

    // 1. 缓冲准备阶段（P1 按 null_ptrs 行构造）
    auto buf = AllocCsrPointerBuffers(rows, nnz, idxType, isFp16, nullPtrs);

    // 2. aclsparseCreateCsr（P0 初始指针组）
    SpMatManager mat;
    try {
        mat = SpMatManager::createCsr(rows, cols, nnz, buf.p0Offsets.get(),
                                      buf.p0ColInd.get(), buf.p0Values.get(), csrIdxType,
                                      csrIdxType, idxBaseEnum, valueType);
    } catch (const std::exception &e) {
        std::cerr << "[NPU] aclsparseCreateCsr failed: " << e.what() << std::endl;
        r.createRet = ACL_SPARSE_STATUS_INVALID_VALUE;  // 工厂抛异常 = Create 未成功
        return r;
    }
    r.createRet = ACL_SPARSE_STATUS_SUCCESS;

    // 白盒记录 Set 前代际（CSV 行 P1 相对 P0 均发生 ptrs/idxs 变化，after 应为 before+1）
    r.whiteBoxGenerationBefore = mat.get()->structureGeneration;

    // 3. aclsparseCsrSetPointers（P1 指针组重绑）
    r.setRet = aclsparseCsrSetPointers(mat.get(), buf.p1Offsets.get(), buf.p1ColInd.get(),
                                       buf.p1Values.get());
    if (r.setRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseCsrSetPointers failed: " << r.setRet << std::endl;
        return r;
    }
    r.whiteBoxGenerationAfter = mat.get()->structureGeneration;

    // 4. Get 回读阶段（十字段；指针==P1 在 wrapper 内比较）
    r.getRet = CsrGetRoundTrip(mat.get(), buf.p1Offsets.get(), buf.p1ColInd.get(),
                               buf.p1Values.get(), r);
    if (r.getRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseCsrGet failed: " << r.getRet << std::endl;
        return r;
    }
    return r;
}

#endif  // TEST_CSR_SET_POINTERS_CSR_SET_POINTERS_NPU_WRAPPER_H_
