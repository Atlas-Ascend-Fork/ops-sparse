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

/*!
 * \file aclsparse_descr_guard.h
 * \brief 描述符 Get 类访问器（SparseAccessor）的公共守卫与 OUT 回写辅助（header-only）。
 *
 * 供 CsrGet / CscGet / CooGet / BlockedEllGet / GetIndexBase 等描述符访问类入口复用：
 *   - CheckSpMatHandle：null 与签名守卫，拒绝误传的其他类型描述符句柄，
 *     返回 ACL_SPARSE_STATUS_INVALID_VALUE；适用于不做格式限制的入口
 *     （如按任意格式读描述符属性的 GetIndexBase 类接口）。
 *   - CheckSpMatFormat：格式守卫，描述符格式与入口要求不符时返回
 *     ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED；需要格式分派的入口
 *     在 CheckSpMatHandle 通过后调用。
 *   - WriteOutIfRequested：OUT 字段按需回写（OUT 指针允许为 nullptr，
 *     构成部分读取语义）。
 *   - ReadCompressedSpMat：压缩格式（CSR / CSC 等共用 aclsparseSpMatDescr
 *     字段布局）描述符的标准 10 字段回读，组合上述守卫与回写，压缩格式
 *     访问器直接复用。
 */

#ifndef ACLSPARSE_DESCR_GUARD_H
#define ACLSPARSE_DESCR_GUARD_H

#include "log/log.h"
#include "aclsparse_descr_internal.h"

// null 与签名守卫：拒绝 null 句柄与签名不匹配句柄（误传其他类型描述符对象），
// 两者均返回 ACL_SPARSE_STATUS_INVALID_VALUE。
static inline aclsparseStatus_t CheckSpMatHandle(const char *fn, const aclsparseSpMatDescr *d)
{
    if (d == nullptr) {
        OP_LOGE(fn, "spMatDescr is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (d->signature != kSpMatSignature) {
        OP_LOGE(fn, "spMatDescr signature mismatch, not a sparse matrix descriptor");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 格式守卫：描述符格式与入口要求的格式不符时返回
// ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED。调用前须先通过 CheckSpMatHandle。
static inline aclsparseStatus_t CheckSpMatFormat(const char *fn, const aclsparseSpMatDescr *d,
                                                 aclsparseFormat_t expectedFormat)
{
    if (d->format != expectedFormat) {
        OP_LOGE(fn, "spMatDescr format %d does not match the expected format %d",
                static_cast<int>(d->format), static_cast<int>(expectedFormat));
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 仅在调用方请求该字段时回写（OUT 指针允许为 nullptr，构成部分读取语义）。
// OutPtr 形如 int64_t* / void** / const void** 等；Val 与 *OutPtr 赋值兼容，
// const 变体传入 void* 值时经限定转换写入 const void* 目标。
template <typename OutPtr, typename Val>
static inline void WriteOutIfRequested(OutPtr out, Val value)
{
    if (out != nullptr) {
        *out = value;
    }
}

// 压缩格式（CSR / CSC 等共用 aclsparseSpMatDescr 字段布局）描述符的标准字段
// 回读：句柄与格式守卫通过后，把 rows/cols/nnz、压缩数组与值三指针
// （ptrs/idxs/values，CSR 映射 csrRowOffsets/csrColInd/csrValues，CSC 映射
// cscColOffsets/cscRowInd/cscValues）、两个独立索引类型（ptrType/IdxType）、
// 索引基与值类型按请求回写。三个数据指针 OUT 的类型由模板参数 PtrOut 决定
// ——非 const 入口为 void**，const 入口为 const void**。
template <typename PtrOut>
static inline aclsparseStatus_t ReadCompressedSpMat(const char *fn, const aclsparseSpMatDescr *d,
    aclsparseFormat_t expectedFormat,
    int64_t *rows, int64_t *cols, int64_t *nnz,
    PtrOut offsetsOut, PtrOut indicesOut, PtrOut valuesOut,
    aclsparseIndexType_t *offsetsTypeOut, aclsparseIndexType_t *indicesTypeOut,
    aclsparseIndexBase_t *idxBaseOut, aclDataType *valueTypeOut)
{
    const aclsparseStatus_t handleSt = CheckSpMatHandle(fn, d);
    if (handleSt != ACL_SPARSE_STATUS_SUCCESS) {
        return handleSt;
    }
    const aclsparseStatus_t formatSt = CheckSpMatFormat(fn, d, expectedFormat);
    if (formatSt != ACL_SPARSE_STATUS_SUCCESS) {
        return formatSt;
    }
    WriteOutIfRequested(rows, static_cast<int64_t>(d->rows));
    WriteOutIfRequested(cols, static_cast<int64_t>(d->cols));
    WriteOutIfRequested(nnz, static_cast<int64_t>(d->nnz));
    WriteOutIfRequested(offsetsOut, d->ptrs);
    WriteOutIfRequested(indicesOut, d->idxs);
    WriteOutIfRequested(valuesOut, d->values);
    WriteOutIfRequested(offsetsTypeOut, d->ptrType);
    WriteOutIfRequested(indicesTypeOut, d->IdxType);
    WriteOutIfRequested(idxBaseOut, d->baseType);
    WriteOutIfRequested(valueTypeOut, d->valueType);
    return ACL_SPARSE_STATUS_SUCCESS;
}

#endif  // ACLSPARSE_DESCR_GUARD_H
