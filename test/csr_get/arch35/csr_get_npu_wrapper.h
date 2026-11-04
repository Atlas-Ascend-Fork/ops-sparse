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
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

// ============================================================================
// NPU Workflow Result
// ============================================================================
struct CsrGetNpuResult {
    aclsparseStatus_t createRet;
    aclsparseStatus_t getRet;

    // CsrGet 读回的标量字段（Host 元数据，非张量计算）
    int64_t rows{0};
    int64_t cols{0};
    int64_t nnz{0};
    aclsparseIndexType_t rowOffsetsType{};
    aclsparseIndexType_t colIndType{};
    aclsparseIndexBase_t idxBase{};
    aclDataType valueType{};

    // CsrGet 读回的设备指针（应与 Create 传入的设备指针一致）
    void *devRowOffsets{nullptr};
    void *devColInd{nullptr};
    void *devValues{nullptr};
    bool devPtrMatch{false};  // 读回指针是否与 Create 传入的设备指针逐字一致
};

// ============================================================================
// 生成 CSR 测试数据（host 侧仅生成输入，不在 CPU 做算子计算）。
// 与 COO 生成器的关键差异：rowOffsets 长度恒为 rows+1（只由 rows 决定，与 nnz
// 无关），nnz == 0 时 colInd 为空但 rowOffsets 仍生成 rows+1 项（全等于 base）。
// mt19937(seed) 在 rows*cols 内无重复采样，按 (row, col) 字典序有序。
// ============================================================================
template <typename RowOffT, typename ColIndT>
inline void GenerateCsrData(int rows, int cols, int nnz, int idx_base, uint32_t seed,
                            std::vector<RowOffT> &rowOffsets, std::vector<ColIndT> &colInd)
{
    rowOffsets.clear();
    colInd.clear();
    if (rows < 0 || cols < 0 || nnz < 0) {
        return;
    }
    const RowOffT base = (idx_base == 1) ? static_cast<RowOffT>(1) : static_cast<RowOffT>(0);
    const int64_t totalCells = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);

    // 1) 在 rows*cols 单元格内无重复采样 nnz 个 (row, col)（set 保证字典序有序）
    std::set<std::pair<int, int>> coords;
    if (totalCells > 0) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> rowDist(0, rows - 1);
        std::uniform_int_distribution<int> colDist(0, cols - 1);
        while (static_cast<int64_t>(coords.size()) < nnz &&
               static_cast<int64_t>(coords.size()) < totalCells) {
            coords.insert(std::make_pair(rowDist(rng), colDist(rng)));
        }
    }

    // 2) rowOffsets：逐行计数前缀和 + base（长度 rows+1，恒生成）
    std::vector<int64_t> perRowCount(static_cast<size_t>(rows) + 1, 0);
    for (const auto &c : coords) {
        perRowCount[static_cast<size_t>(c.first) + 1] += 1;
    }
    rowOffsets.assign(static_cast<size_t>(rows) + 1, base);
    for (int i = 0; i < rows; i++) {
        const size_t cur = static_cast<size_t>(i);
        rowOffsets[cur + 1] = static_cast<RowOffT>(static_cast<int64_t>(rowOffsets[cur]) +
                                                   perRowCount[cur + 1]);
    }

    // 3) colInd：对应列号 + base（长度等于实际采样数，合法输入下等于 nnz）
    colInd.reserve(coords.size());
    for (const auto &c : coords) {
        colInd.push_back(static_cast<ColIndT>(static_cast<int64_t>(c.second) +
                                              static_cast<int64_t>(base)));
    }
}

// ============================================================================
// 索引类型 -> aclsparseIndexType_t 枚举换算（int32_t -> 32I，int64_t -> 64I）
// ============================================================================
template <typename T>
inline aclsparseIndexType_t CsrIndexTypeEnumOf()
{
    return std::is_same<T, int64_t>::value ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
}

// ============================================================================
// NPU wrapper（模板内层）：GenerateCsrData -> DeviceBuffer 迁移 ->
// SpMatManager::createCsr（RAII）-> aclsparseCsrGet 全字段读回 -> devPtrMatch。
// 全程在 NPU 侧建设备缓冲存指针；Host 仅做元数据生成与 accessor 断言。
// ============================================================================
template <typename RowOffT, typename ColIndT>
inline CsrGetNpuResult CsrGetNpuTyped(aclrtStream stream, int rows, int cols, int nnz,
                                      int idx_base, bool isFp16, uint32_t seed)
{
    using namespace sparse_test;
    (void)stream;
    CsrGetNpuResult r{};

    // 1. 生成 CSR 输入数据（rowOffsets 恒 rows+1 项；colInd 为 nnz 项）
    std::vector<RowOffT> hostRowOffsets;
    std::vector<ColIndT> hostColInd;
    GenerateCsrData(rows, cols, nnz, idx_base, seed, hostRowOffsets, hostColInd);

    // 2. 入参合法性预检（nnz<0 或数组长度不符 -> INVALID_VALUE 提前返回）
    const bool lenOk =
        (static_cast<int64_t>(hostRowOffsets.size()) == static_cast<int64_t>(rows) + 1) &&
        (static_cast<int64_t>(hostColInd.size()) == nnz);
    if (nnz < 0 || !lenOk) {
        r.createRet = ACL_SPARSE_STATUS_INVALID_VALUE;
        return r;
    }

    // 3-6. 设备分配（DeviceBuffer 抛异常）、CreateCsr（RAII Manager 抛异常）与
    // CsrGet 读回全部纳入同一 try：任一设备侧失败都转为受控返回（createRet 置
    // INVALID_VALUE），不向 GTest 逃逸未捕获异常；缓冲与描述符由 RAII 保证释放。
    try {
        // 3. 输入迁移到 NPU 设备（size==0 时框架返回空缓冲，即 nullptr）
        auto dRow = DeviceBuffer::copyFrom(hostRowOffsets.data(),
                                           hostRowOffsets.size() * sizeof(RowOffT));
        auto dCol = DeviceBuffer::copyFrom(hostColInd.data(),
                                           hostColInd.size() * sizeof(ColIndT));

        // 4. values 仅占位不初始化（accessor 不读设备数据，与 coo_get 先例一致）
        const size_t valBytes = isFp16 ? (static_cast<size_t>(nnz) * sizeof(uint16_t))
                                       : (static_cast<size_t>(nnz) * sizeof(float));
        auto dVal = DeviceBuffer::alloc(valBytes);

        const aclDataType valueType = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;
        const aclsparseIndexBase_t idxBase =
            (idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
        const aclsparseIndexType_t rowOffEnum = CsrIndexTypeEnumOf<RowOffT>();
        const aclsparseIndexType_t colIndEnum = CsrIndexTypeEnumOf<ColIndT>();

        // 5. CreateCsr（RAII Manager）+ CsrGet 全字段读回
        auto mgr = SpMatManager::createCsr(rows, cols, nnz, dRow.get(), dCol.get(), dVal.get(),
                                           rowOffEnum, colIndEnum, idxBase, valueType);
        r.createRet = ACL_SPARSE_STATUS_SUCCESS;
        r.getRet = aclsparseCsrGet(mgr.get(), &r.rows, &r.cols, &r.nnz, &r.devRowOffsets,
                                   &r.devColInd, &r.devValues, &r.rowOffsetsType, &r.colIndType,
                                   &r.idxBase, &r.valueType);
        if (r.getRet != ACL_SPARSE_STATUS_SUCCESS) {
            std::cerr << "[NPU] aclsparseCsrGet failed: " << r.getRet << std::endl;
            return r;
        }

        // 6. accessor 语义断言素材：读回设备指针应与 Create 传入的逐字一致
        r.devPtrMatch = (r.devRowOffsets == dRow.get() && r.devColInd == dCol.get() &&
                         r.devValues == dVal.get());
    } catch (const std::exception &e) {
        // e.what() 区分来源：aclrtMalloc/aclrtMemcpy 失败（设备分配/拷贝）或
        // aclsparseCreateCsr 失败（描述符构造）
        std::cerr << "[NPU] CsrGetNpu device setup failed: " << e.what() << std::endl;
        r.createRet = ACL_SPARSE_STATUS_INVALID_VALUE;
        return r;
    }
    return r;
}

// ============================================================================
// NPU wrapper 入口：按两个索引类型字符串（"32I"/"64I"，独立取值）分发到
// <int32_t,int32_t> / <int32_t,int64_t> / <int64_t,int32_t> / <int64_t,int64_t>
// 四个模板实例。
// ============================================================================
inline CsrGetNpuResult CsrGetNpu(aclrtStream stream, int rows, int cols, int nnz, int idx_base,
                                 bool isFp16, uint32_t seed, const std::string &rowOffType,
                                 const std::string &colIndType)
{
    if (rowOffType == "32I" && colIndType == "32I") {
        return CsrGetNpuTyped<int32_t, int32_t>(stream, rows, cols, nnz, idx_base, isFp16, seed);
    }
    if (rowOffType == "32I" && colIndType == "64I") {
        return CsrGetNpuTyped<int32_t, int64_t>(stream, rows, cols, nnz, idx_base, isFp16, seed);
    }
    if (rowOffType == "64I" && colIndType == "32I") {
        return CsrGetNpuTyped<int64_t, int32_t>(stream, rows, cols, nnz, idx_base, isFp16, seed);
    }
    if (rowOffType == "64I" && colIndType == "64I") {
        return CsrGetNpuTyped<int64_t, int64_t>(stream, rows, cols, nnz, idx_base, isFp16, seed);
    }
    CsrGetNpuResult r{};
    r.createRet = ACL_SPARSE_STATUS_INVALID_VALUE;  // 未知索引类型字符串
    return r;
}
