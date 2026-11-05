/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

// ============================================================================
// NPU Workflow Result
// ============================================================================
struct CscGetNpuResult {
    aclsparseStatus_t createRet;
    aclsparseStatus_t getRet;

    // CscGet 读回的标量/枚举字段（Host 元数据，非张量计算）
    int64_t rows{0};
    int64_t cols{0};
    int64_t nnz{0};
    aclsparseIndexType_t colOffsetsType{};
    aclsparseIndexType_t rowIndType{};
    aclsparseIndexBase_t idxBase{};
    aclDataType valueType{};

    // CscGet 读回的设备指针（应与 Create 传入的设备指针一致）
    void *devColOffsets{nullptr};
    void *devRowInd{nullptr};
    void *devValues{nullptr};
    bool devPtrMatch{false};  // 读回指针是否与 Create 传入的设备指针逐字一致
};

// ============================================================================
// 生成 CSC 测试数据（host 侧仅生成输入，不在 CPU 做算子计算）。
// colOffsets 长 cols+1（CSC 规格，区别于 CSR 的 rows+1）、单调非降、
// 首元素=idx_base、末元素=nnz+idx_base；rowInd 长 nnz、取值 ∈ [idx_base, rows-1+idx_base]。
// std::mt19937(seed) 固定种子可复现。test/frame/fill.h 的 makeSparseCsc 为稀疏度
// 驱动且仅支持 int32+idxBase0，无法提供精确 nnz/idxBase/I64 宽度控制，故专用此生成器。
// ============================================================================
template <typename PtrT, typename IdxT>
inline void GenerateCscData(int rows, int cols, int nnz, int idx_base, uint32_t seed,
                            std::vector<PtrT> &colOffsets, std::vector<IdxT> &rowInd)
{
    colOffsets.clear();
    rowInd.clear();
    if (rows <= 0 || cols <= 0 || nnz < 0) {
        return;
    }
    const PtrT base = static_cast<PtrT>(idx_base == 1 ? 1 : 0);
    std::mt19937 rng(seed);
    // 按列随机撒 nnz 个元素，再前缀累加得到列偏移（保证单调非降且末元素=base+nnz）
    std::vector<int64_t> counts(static_cast<size_t>(cols), 0);
    if (nnz > 0) {
        std::uniform_int_distribution<int> colDist(0, cols - 1);
        for (int i = 0; i < nnz; i++) {
            counts[static_cast<size_t>(colDist(rng))]++;
        }
    }
    colOffsets.reserve(static_cast<size_t>(cols) + 1);
    colOffsets.push_back(base);
    int64_t acc = 0;
    for (int c = 0; c < cols; c++) {
        acc += counts[static_cast<size_t>(c)];
        colOffsets.push_back(static_cast<PtrT>(base + acc));
    }
    // 行索引随机生成（accessor 不读数据，仅需取值域合法）
    rowInd.reserve(static_cast<size_t>(nnz));
    std::uniform_int_distribution<int> rowDist(0, rows - 1);
    const IdxT rowBase = static_cast<IdxT>(idx_base == 1 ? 1 : 0);
    for (int i = 0; i < nnz; i++) {
        rowInd.push_back(static_cast<IdxT>(rowDist(rng) + rowBase));
    }
}

// ============================================================================
// NPU wrapper 模板实现: CreateCsc -> CscGet accessor 验证。
// 全程在 NPU 侧建设备缓冲存指针；Host 仅做元数据生成与 accessor 断言。
// nnz=0 时 rowInd/values 为空 DeviceBuffer（nullptr），CreateCsc 允许空指针
// （延迟绑定语义），Get 读回 nullptr，指针恒等断言 nullptr==nullptr 仍成立。
// 描述符生命周期用 RAII（SpMatManager::createCsc 工厂）。
// ============================================================================
template <typename PtrT, typename IdxT>
inline CscGetNpuResult CscGetNpuImpl(int rows, int cols, int nnz, int idx_base, bool isFp16,
                                     aclsparseIndexType_t colOffsetsType, aclsparseIndexType_t rowIndType,
                                     uint32_t seed)
{
    using namespace sparse_test;
    CscGetNpuResult r{};

    // 非法维度在进入分配路径前直接拒绝：负 rows/cols/nnz 若继续下走，
    // static_cast<size_t>(负 nnz) 会得到接近 SIZE_MAX 的分配尺寸，
    // 且分配发生在下方 try 之外，异常会冲出 wrapper 导致 gtest 进程终止。
    if (rows < 0 || cols < 0 || nnz < 0) {
        std::cerr << "[NPU] invalid dims: rows=" << rows << " cols=" << cols << " nnz=" << nnz
                  << std::endl;
        r.createRet = ACL_SPARSE_STATUS_INVALID_VALUE;
        r.getRet = ACL_SPARSE_STATUS_INVALID_VALUE;
        return r;
    }

    std::vector<PtrT> hostColOffsets;
    std::vector<IdxT> hostRowInd;
    GenerateCscData(rows, cols, nnz, idx_base, seed, hostColOffsets, hostRowInd);

    // 输入迁移到 NPU 设备（accessor 不读数据，values 缓冲仅占位）
    auto dOff = DeviceBuffer::copyFrom(hostColOffsets.data(), hostColOffsets.size() * sizeof(PtrT));
    auto dIdx = DeviceBuffer::copyFrom(hostRowInd.data(), hostRowInd.size() * sizeof(IdxT));
    size_t valBytes = isFp16 ? (static_cast<size_t>(nnz) * sizeof(uint16_t))
                             : (static_cast<size_t>(nnz) * sizeof(float));
    auto dVal = DeviceBuffer::alloc(valBytes);

    const aclDataType valueType = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;
    const aclsparseIndexBase_t idxBase =
        (idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    try {
        SpMatManager mat = SpMatManager::createCsc(rows, cols, nnz, dOff.get(), dIdx.get(), dVal.get(),
                                                   colOffsetsType, rowIndType, idxBase, valueType);
        r.createRet = ACL_SPARSE_STATUS_SUCCESS;
        r.getRet = aclsparseCscGet(mat.get(), &r.rows, &r.cols, &r.nnz, &r.devColOffsets, &r.devRowInd,
                                   &r.devValues, &r.colOffsetsType, &r.rowIndType, &r.idxBase,
                                   &r.valueType);
        if (r.getRet != ACL_SPARSE_STATUS_SUCCESS) {
            std::cerr << "[NPU] aclsparseCscGet failed: " << r.getRet << std::endl;
            return r;
        }
        // accessor 语义断言：Get 读回的设备指针应与 Create 传入的逐字一致
        r.devPtrMatch = (r.devColOffsets == dOff.get() && r.devRowInd == dIdx.get() &&
                         r.devValues == dVal.get());
    } catch (const std::exception &e) {
        // SpMatManager::createCsc 工厂失败时抛异常；转为非 SUCCESS 状态供测试断言
        std::cerr << "[NPU] aclsparseCreateCsc failed: " << e.what() << std::endl;
        r.createRet = ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    return r;
}

// ============================================================================
// NPU wrapper 入口：按 cscColOffsetsType / cscRowIndType 分发到对应宽度组合
// （I32/I32、I32/I64、I64/I32、I64/I64 四种组合均支持）
// ============================================================================
inline CscGetNpuResult CscGetNpu(int rows, int cols, int nnz, int idx_base, bool isFp16,
                                 aclsparseIndexType_t colOffsetsType, aclsparseIndexType_t rowIndType,
                                 uint32_t seed)
{
    if (colOffsetsType == ACL_SPARSE_INDEX_32I && rowIndType == ACL_SPARSE_INDEX_32I) {
        return CscGetNpuImpl<int32_t, int32_t>(rows, cols, nnz, idx_base, isFp16, colOffsetsType,
                                               rowIndType, seed);
    }
    if (colOffsetsType == ACL_SPARSE_INDEX_32I && rowIndType == ACL_SPARSE_INDEX_64I) {
        return CscGetNpuImpl<int32_t, int64_t>(rows, cols, nnz, idx_base, isFp16, colOffsetsType,
                                               rowIndType, seed);
    }
    if (colOffsetsType == ACL_SPARSE_INDEX_64I && rowIndType == ACL_SPARSE_INDEX_32I) {
        return CscGetNpuImpl<int64_t, int32_t>(rows, cols, nnz, idx_base, isFp16, colOffsetsType,
                                               rowIndType, seed);
    }
    return CscGetNpuImpl<int64_t, int64_t>(rows, cols, nnz, idx_base, isFp16, colOffsetsType,
                                           rowIndType, seed);
}
