/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file aclsparselt_matmul_descriptor_internal.h
 * \brief ops-sparseLt matmul 描述符内部结构定义（不对外暴露）。
 */

#ifndef ACLSPARSELT_MATMUL_DESCRIPTOR_INTERNAL_H
#define ACLSPARSELT_MATMUL_DESCRIPTOR_INTERNAL_H

#include <cfloat>
#include "cann_ops_sparseLt.h"
#include "aclsparselt_mat_descriptor_internal.h"

/**
 * @brief matmul 描述符内部结构体。
 *
 * 持有 matmul 运算的完整元信息：操作类型、四个矩阵描述符引用、计算精度。
 * matA/matB/matC/matD 为非所有权引用（Destroy 时不会销毁它们）。
 *
 * bias + activation 属性字段（与 7 个 MatmulDesc 枚举属性一一对应）。
 * bias dtype = C dtype（非 INT8 路径），INT8 路径 bias dtype = FP32（cuSPARSELt 规定）。
 * activationType 统一编码：0=无, 1=ReLU, 2=GeLU（在 SetAttribute 时映射，不存在于结构体）。
 */
struct aclsparseLtMatmulDescriptor {
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE; ///< 作用于 A 的操作
    aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE; ///< 作用于 B 的操作
    aclsparseLtMatDescriptor_t matA = nullptr;  ///< 矩阵 A 描述符引用（非所有权）
    aclsparseLtMatDescriptor_t matB = nullptr;  ///< 矩阵 B 描述符引用（非所有权）
    aclsparseLtMatDescriptor_t matC = nullptr;  ///< 矩阵 C 描述符引用（非所有权）
    aclsparseLtMatDescriptor_t matD = nullptr;  ///< 矩阵 D 描述符引用（非所有权）
    aclsparseComputeType_t computeType = ACL_SPARSE_COMPUTE_16F; ///< 计算精度
    // —— bias + activation 属性（与 MatmulDesc 枚举 2~8 一一对应）——
    void* biasPointer = nullptr;               ///< BIAS_POINTER: bias device 指针，nullptr=无 bias
    int64_t biasStride = 0;                    ///< BIAS_STRIDE: batch 间 bias 步长，0=广播
    int32_t activationRelu = 0;                ///< ACTIVATION_RELU: 0=禁用, nonzero=启用 ReLU
    float reluUpperBound = FLT_MAX;            ///< ACTIVATION_RELU_UPPERBOUND: 默认 FLT_MAX
    float reluThreshold = 0.0f;                 ///< ACTIVATION_RELU_THRESHOLD: 默认 0.0f
    int32_t activationGelu = 0;                ///< ACTIVATION_GELU: 0=禁用, nonzero=启用 GeLU
    float geluScaling = 1.0f;                   ///< ACTIVATION_GELU_SCALING: 默认 1.0f
    void* sparseMatPointer = nullptr;           ///< SPARSE_MAT_POINTER: 存储不参与执行（Matmul/Search 以
                                                ///< matA/matB 函数参数为准，参数优先；仅兼容 cuSPARSELt
                                                ///< 官方样例的调用序列）
};

/** @brief 对外句柄安全转换为内部结构体指针。 */
inline aclsparseLtMatmulDescriptor* ToInternal(aclsparseLtMatmulDescriptor_t desc)
{
    return reinterpret_cast<aclsparseLtMatmulDescriptor*>(desc);
}

#endif // ACLSPARSELT_MATMUL_DESCRIPTOR_INTERNAL_H
