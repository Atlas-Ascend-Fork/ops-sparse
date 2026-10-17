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
 * \file sparse2dense.h
 * \brief SparseToDense Host 侧辅助函数：描述符转换、值类型映射。
 */

#ifndef SPARSE2DENSE_H_
#define SPARSE2DENSE_H_

#include <cstdint>

#include "cann_ops_sparse.h"
#include "sparse2dense_tiling_data.h"

/// 将 opaque handle 转为内部结构体指针
inline struct aclsparseContext *Sparse2DenseToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

/// 将 SpMat 描述符转为内部结构体指针
inline struct aclsparseSpMatDescr *Sparse2DenseToMatInner(aclsparseSpMatDescr_t desc)
{
    return reinterpret_cast<struct aclsparseSpMatDescr *>(desc);
}

/// 将 const SpMat 描述符转为内部结构体指针（只读路径）
inline const struct aclsparseSpMatDescr *Sparse2DenseToConstMatInner(aclsparseConstSpMatDescr_t desc)
{
    return reinterpret_cast<const struct aclsparseSpMatDescr *>(desc);
}

/// 将 DnMat 描述符转为内部结构体指针
inline struct aclsparseDnMatDescr *Sparse2DenseToDnMatInner(aclsparseDnMatDescr_t desc)
{
    return reinterpret_cast<struct aclsparseDnMatDescr *>(desc);
}

/// 将 const DnMat 描述符转为内部结构体指针（只读路径）
inline const struct aclsparseDnMatDescr *Sparse2DenseToConstDnMatInner(aclsparseConstDnMatDescr_t desc)
{
    return reinterpret_cast<const struct aclsparseDnMatDescr *>(desc);
}

/// 返回 ACL 数据类型的字节大小（调用方需先通过 Validate 校验，此函数不含非法值）
static inline uint32_t AclDataTypeSize(aclDataType valueType)
{
    if (valueType == ACL_INT8) {
        return 1;
    }
    if (valueType == ACL_FLOAT16 || valueType == ACL_BF16) {
        return 2;
    }
    if (valueType == ACL_FLOAT || valueType == ACL_INT32) {
        return 4;
    }
    if (valueType == ACL_COMPLEX64) {
        return 8; // Reg 路径按 uint64_t bit-copy
    }
    return 0;
}

/// 将 ACL 数据类型映射为 kernel 内部值类型编码（调用方需先通过 Validate 校验，此函数不含非法值）
static inline int32_t Sparse2DenseValTypeFromAcl(aclDataType valueType)
{
    if (valueType == ACL_FLOAT16) {
        return SPARSE2DENSE_VAL_F16;
    }
    if (valueType == ACL_BF16) {
        return SPARSE2DENSE_VAL_BF16;
    }
    if (valueType == ACL_INT32) {
        return SPARSE2DENSE_VAL_I32;
    }
    if (valueType == ACL_INT8) {
        return SPARSE2DENSE_VAL_I8;
    }
    if (valueType == ACL_COMPLEX64) {
        return SPARSE2DENSE_VAL_COMPLEX64;
    }
    return SPARSE2DENSE_VAL_F32;
}

/// 将 ACL 稀疏格式映射为 kernel 内部格式编码（调用方需先通过 Validate 校验，此函数不含非法值）
static inline int32_t Sparse2DenseFormatFromAcl(aclsparseFormat_t format)
{
    if (format == ACL_SPARSE_FORMAT_CSC) { return SPARSE2DENSE_FMT_CSC; }
    if (format == ACL_SPARSE_FORMAT_COO) { return SPARSE2DENSE_FMT_COO; }
    return SPARSE2DENSE_FMT_CSR;
}

#endif  // SPARSE2DENSE_H_
