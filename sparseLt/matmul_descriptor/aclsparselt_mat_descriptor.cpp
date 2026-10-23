/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file aclsparselt_mat_descriptor.cpp
 * \brief ops-sparseLt 矩阵描述符初始化与销毁函数实现。
 */

#include "cann_ops_sparseLt.h"

#include "aclsparselt_mat_descriptor_internal.h"

#include <cstdint>
#include <new>

namespace
{

/**
 * @brief 判断 valueType 是否在本期支持列表内。
 *
 * 支持列表：FP32 / FP16 / BF16 / INT8 / HIFLOAT8 / FP8_E4M3 / FP8_E5M2 /
 * FP8_E8M0 / FP4_E2M1 / FP4_E1M2。
 */
inline bool IsSupportedValueType(aclDataType t)
{
    switch (t)
    {
        case ACL_FLOAT:
        case ACL_FLOAT16:
        case ACL_BF16:
        case ACL_INT8:
        case ACL_INT32:
        case ACL_HIFLOAT8:
        case ACL_FLOAT8_E4M3FN:
        case ACL_FLOAT8_E5M2:
        case ACL_FLOAT8_E8M0:
        case ACL_FLOAT4_E2M1:
        case ACL_FLOAT4_E1M2:
            return true;
        default:
            return false;
    }
}

/**
 * @brief 获取对齐倍数（rows/cols/ld 须为其整数倍）。
 *
 * 按 valueType 分档，Dense 与 Structured 倍数不同。
 * @return 对齐倍数；valueType 不在支持列表内时返回 0。
 */
inline int64_t GetAlignMultiple(aclDataType valueType, bool isStructured)
{
    switch (valueType)
    {
        case ACL_INT8:
        case ACL_HIFLOAT8:
        case ACL_FLOAT8_E4M3FN:
        case ACL_FLOAT8_E5M2:
        case ACL_FLOAT8_E8M0:
        case ACL_FLOAT4_E2M1:
        case ACL_FLOAT4_E1M2:
            return isStructured ? 32 : 16;
        case ACL_FLOAT16:
        case ACL_BF16:
            return isStructured ? 16 : 8;
        case ACL_FLOAT:
        case ACL_INT32:
            return isStructured ? 8 : 4;
        default:
            return 0;
    }
}

/**
 * @brief 校验 order 是否为合法枚举值。
 */
inline bool IsValidOrder(aclsparseOrder_t order)
{
    return order == ACL_SPARSE_ORDER_ROW || order == ACL_SPARSE_ORDER_COL;
}

/**
 * @brief 校验 ld 约束：COL 序 ld >= rows，ROW 序 ld >= cols。
 *
 * 调用前须确保 order 为合法枚举值（ROW/COL）。
 */
inline bool IsLdValid(aclsparseOrder_t order, int64_t ld, int64_t rows, int64_t cols)
{
    return (order == ACL_SPARSE_ORDER_COL && ld >= rows) || (order == ACL_SPARSE_ORDER_ROW && ld >= cols);
}

/**
 * @brief 填充矩阵描述符字段（dense / structured 共用）。
 */
inline void FillMatDescriptor(aclsparseLtMatDescriptor* desc, int64_t rows, int64_t cols, int64_t ld,
                              uint32_t alignment, aclDataType valueType, aclsparseOrder_t order,
                              aclsparseLtSparsity_t sparsity, bool isStructured)
{
    desc->rows = rows;
    desc->cols = cols;
    desc->ld = ld;
    desc->alignment = alignment;
    desc->valueType = valueType;
    desc->order = order;
    desc->sparsity = sparsity;
    desc->isStructured = isStructured;
    desc->numBatches = 1;
    desc->batchStride = 0;
}

/**
 * @brief
 * 公共参数校验（handle/matDescr/rows/cols/ld/valueType/order/alignment）。
 *
 * Dense 与 Structured 共用：句柄/输出指针/基本维度/数据类型/存储序/对齐。
 */
inline aclsparseStatus_t ValidateMatDescCommonParams(const aclsparseLtHandle_t* handle,
                                                     const aclsparseLtMatDescriptor_t* matDescr, int64_t rows,
                                                     int64_t cols, int64_t ld, uint32_t alignment,
                                                     aclDataType valueType, aclsparseOrder_t order)
{
    if (handle == nullptr)
    {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matDescr == nullptr || *matDescr != nullptr)
    {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rows <= 0 || cols <= 0 || ld <= 0)
    {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsSupportedValueType(valueType))
    {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!IsValidOrder(order))
    {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (alignment == 0U || (alignment % 16U) != 0U)
    {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/**
 * @brief 公共对齐倍数 + ld 约束校验并分配描述符内存。
 *
 * @param isStructured true=Structured（GetAlignMultiple 用 structured
 * 档），false=Dense。
 * @param outDesc 输出参数，成功时指向新分配的描述符。
 */
inline aclsparseStatus_t AllocateAndValidateMatDesc(int64_t rows, int64_t cols, int64_t ld, aclDataType valueType,
                                                     aclsparseOrder_t order, bool isStructured,
                                                     aclsparseLtMatDescriptor*& outDesc)
{
    int64_t alignMultiple = GetAlignMultiple(valueType, isStructured);
    if (alignMultiple == 0 || (rows % alignMultiple) != 0 || (cols % alignMultiple) != 0 || (ld % alignMultiple) != 0)
    {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // ld 约束：COL 序 ld >= rows，ROW 序 ld >= cols（order 已校验为 ROW/COL）
    if (!IsLdValid(order, ld, rows, cols))
    {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // ld 上界：ld 为 int64 参数，但下游消费点（tiling 的 ldA/ldB/ldC/ldD、
    // prune 的 td.ld）均为 int32 字段——超 INT32_MAX 的 ld 经静默截断变为
    // 垃圾行距（kernel 按错误行距访问，静默数据损坏）。描述符层显式拒绝
    // （与 prune_check 侧既有 INT32_MAX 复检同口径），使全部收窄点安全。
    if (ld > INT32_MAX)
    {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    outDesc = new (std::nothrow) aclsparseLtMatDescriptor();
    if (outDesc == nullptr)
    {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

extern "C"
{
    aclsparseStatus_t aclsparseLtDenseDescriptorInit(const aclsparseLtHandle_t* handle,
                                                     aclsparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols,
                                                     int64_t ld, uint32_t alignment, aclDataType valueType,
                                                     aclsparseOrder_t order)
    {
        aclsparseStatus_t ret = ValidateMatDescCommonParams(handle, matDescr, rows, cols, ld, alignment, valueType,
                                                            order);
        if (ret != ACL_SPARSE_STATUS_SUCCESS)
        {
            return ret;
        }

        aclsparseLtMatDescriptor* desc = nullptr;
        ret = AllocateAndValidateMatDesc(rows, cols, ld, valueType, order, false, desc);
        if (ret != ACL_SPARSE_STATUS_SUCCESS)
        {
            return ret;
        }

        FillMatDescriptor(desc, rows, cols, ld, alignment, valueType, order, ACL_SPARSE_LT_SPARSITY_50_PERCENT, false);
        *matDescr = desc;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    aclsparseStatus_t aclsparseLtStructuredDescriptorInit(const aclsparseLtHandle_t* handle,
                                                          aclsparseLtMatDescriptor_t* matDescr, int64_t rows,
                                                          int64_t cols, int64_t ld, uint32_t alignment,
                                                          aclDataType valueType, aclsparseOrder_t order,
                                                          aclsparseLtSparsity_t sparsity)
    {
        aclsparseStatus_t ret = ValidateMatDescCommonParams(handle, matDescr, rows, cols, ld, alignment, valueType,
                                                            order);
        if (ret != ACL_SPARSE_STATUS_SUCCESS)
        {
            return ret;
        }

        if (sparsity != ACL_SPARSE_LT_SPARSITY_50_PERCENT)
        {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }

        aclsparseLtMatDescriptor* desc = nullptr;
        ret = AllocateAndValidateMatDesc(rows, cols, ld, valueType, order, true, desc);
        if (ret != ACL_SPARSE_STATUS_SUCCESS)
        {
            return ret;
        }

        FillMatDescriptor(desc, rows, cols, ld, alignment, valueType, order, sparsity, true);
        *matDescr = desc;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    aclsparseStatus_t aclsparseLtMatDescriptorDestroy(aclsparseLtMatDescriptor_t* matDescr)
    {
        if (matDescr == nullptr)
        {
            return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
        }
        if (*matDescr == nullptr)
        {
            return ACL_SPARSE_STATUS_SUCCESS;
        }

        auto* desc = *matDescr;
        delete desc;
        *matDescr = nullptr;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

} // extern "C"
