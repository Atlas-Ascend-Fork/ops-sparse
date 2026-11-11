/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CREATE_BSR_CREATE_BSR_NPU_WRAPPER_H_
#define TEST_CREATE_BSR_CREATE_BSR_NPU_WRAPPER_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "sparse_dtype_utils.h"    // ParseValueType：CSV dtype 列名 -> aclDataType（BSR 族共享）
#include "spmat_test_helpers.h"    // SpMatCreateGetResult/ReadBackGenericGets：Create 系公共回读段
#include "aclsparse_descr_internal.h"  // 白盒：BSR 无公共 Get 访问器，核对块几何字段

// frame 共享助手统一置于 sparse_test 命名空间，本文件的结构体定义经
// using 声明引用（函数体内已有 using namespace sparse_test）
using sparse_test::SpMatCreateGetResult;

// ============================================================================
// CSV order 列名 -> 枚举映射（ROW/COL，仅 create_bsr 使用）
// ============================================================================
inline aclsparseOrder_t ParseBsrOrder(const std::string &order)
{
    return (order == "COL") ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW;
}

// ============================================================================
// NPU Workflow Result：Create + 通用 Get 回读公共段 + 白盒字段（纯 Host 元数据）。
// 未执行步骤（Create 失败提前返回）保持 NOT_INITIALIZED 哨兵，不误报 SUCCESS。
// ============================================================================
struct CreateBsrNpuResult : public SpMatCreateGetResult {
    // 白盒字段（先例 csr_set_strided_batch：BSR 无公共 Get 访问器，
    // 块几何字段只能经内部结构核对，防 int64 截断）
    int64_t whiteBoxBlockRows{-1};
    int64_t whiteBoxBlockCols{-1};
    int64_t whiteBoxBlockNnz{-1};
    int64_t whiteBoxRowBlockSize{-1};
    int64_t whiteBoxColBlockSize{-1};
    int64_t whiteBoxBsrValuesCount{-1};
    aclsparseOrder_t whiteBoxBsrOrder{static_cast<aclsparseOrder_t>(-1)};
};

// ============================================================================
// 创建阶段（阶段函数）：mutable / const 入口分发，const 变体句柄转存
// mutable 视图（底层为同一 aclsparseSpMatDescr 对象）。
// ============================================================================
inline aclsparseStatus_t CreateBsrDispatch(const std::string &entry, int64_t blockRows,
    int64_t blockCols, int64_t blockNnz, int64_t blockSize, aclsparseIndexBase_t idxBaseEnum,
    aclDataType valueType, aclsparseOrder_t order, void *offsets, void *colInd, void *vals,
    aclsparseSpMatDescr_t *out)
{
    if (entry == "const_bsr") {
        aclsparseConstSpMatDescr_t constDescr = nullptr;
        const aclsparseStatus_t status = aclsparseCreateConstBsr(&constDescr, blockRows, blockCols,
            blockNnz, blockSize, blockSize, offsets, colInd, vals, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_32I, idxBaseEnum, valueType, order);
        *out = const_cast<aclsparseSpMatDescr_t>(constDescr);
        return status;
    }
    return aclsparseCreateBsr(out, blockRows, blockCols, blockNnz, blockSize, blockSize, offsets,
        colInd, vals, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, idxBaseEnum, valueType, order);
}

// ============================================================================
// NPU wrapper：CreateBsr / CreateConstBsr -> 通用 Get 访问器验证。
// 纯 Host 侧描述符创建，不启动 NPU kernel。
//
// 注意（wrapper 适用范围声明）：设备缓冲仅按单矩阵规模占位分配，用于承载描述符
// 持有的设备指针；Create/Get 均只做地址算术、不访问数组内容。BSR 无
// SpMatManager 工厂，沿用 spmat_get_strided_batch 的 unique_ptr +
// aclsparseDestroySpMat 删除器 RAII 形态（先例 csr_set_strided_batch wrapper）。
// CSV 行 block_rows 均为小几何（≤8）；INT32 边界内极大值由 TEST_F 直调覆盖
//（BSR-S01，占位指针不被解引用，无需按 INT32_MAX 规模分配）。
// ============================================================================
inline CreateBsrNpuResult CreateBsrNpu(aclrtStream stream,
    const std::string &entry, int64_t blockRows, int64_t blockCols, int64_t blockNnz,
    int64_t blockSize, int idxBase, aclDataType valueType, aclsparseOrder_t order,
    bool nullPtrs)
{
    using namespace sparse_test;
    (void)stream;
    CreateBsrNpuResult r{};

    const aclsparseIndexBase_t idxBaseEnum =
        (idxBase == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    // offsets 恒传有效指针（含 nnzb=0：单端点数组占位，Create 不解引用）；
    // colInd/values 在 nullPtrs=1（仅 blockNnz=0 行）时传 null，否则按块规模占位
    //（缓冲准备阶段：元素宽度按 valueType 经共享 ElementBytes 计算）
    const std::vector<int32_t> hostOffsets(static_cast<size_t>(blockRows) + 1u, 0);
    auto dOffsets = DeviceBuffer::copyFrom(hostOffsets.data(),
                                           hostOffsets.size() * sizeof(int32_t));
    DeviceBuffer dColInd;
    DeviceBuffer dVals;
    if (!nullPtrs) {
        dColInd = DeviceBuffer::alloc(static_cast<size_t>(blockNnz) * sizeof(int32_t));
        const int64_t valuesCount = blockNnz * blockSize * blockSize;
        dVals = DeviceBuffer::alloc(static_cast<size_t>(valuesCount) *
                                    sparse_test::ElementBytes(valueType));
    }

    // 创建阶段（unique_ptr + aclsparseDestroySpMat 删除器 RAII）
    aclsparseSpMatDescr_t descr = nullptr;
    r.createRet = CreateBsrDispatch(entry, blockRows, blockCols, blockNnz, blockSize,
                                    idxBaseEnum, valueType, order, dOffsets.get(),
                                    dColInd.get(), dVals.get(), &descr);
    if (r.createRet != ACL_SPARSE_STATUS_SUCCESS || descr == nullptr) {
        std::cerr << "[NPU] aclsparseCreateBsr(" << entry << ") failed: " << r.createRet
                  << std::endl;
        return r;
    }
    std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)> mat(
        descr, &aclsparseDestroySpMat);

    // Get 回读阶段（Create 语义载体，成功路径；四个通用访问器共享回读）
    if (!sparse_test::ReadBackGenericGets(mat.get(), r)) {
        std::cerr << "[NPU] generic Get accessor failed after create" << std::endl;
        return r;
    }

    // 白盒核对阶段：块几何字段与传入一致（GetSize 语义 = 块展开维度，另由 TEST_P 断言）
    r.whiteBoxBlockRows = mat.get()->blockRows;
    r.whiteBoxBlockCols = mat.get()->blockCols;
    r.whiteBoxBlockNnz = mat.get()->blockNnz;
    r.whiteBoxRowBlockSize = mat.get()->rowBlockSize;
    r.whiteBoxColBlockSize = mat.get()->colBlockSize;
    r.whiteBoxBsrValuesCount = mat.get()->bsrValuesCount;
    r.whiteBoxBsrOrder = mat.get()->bsrOrder;
    return r;
}

#endif  // TEST_CREATE_BSR_CREATE_BSR_NPU_WRAPPER_H_
