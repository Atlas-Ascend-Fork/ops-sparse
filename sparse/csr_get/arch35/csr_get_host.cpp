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
 * \file csr_get_host.cpp
 * \brief aclsparseCsrGet / aclsparseConstCsrGet Host 侧实现（arch35）。
 *
 * 对标 cuSPARSE cusparseCreateCsr / cusparseCsrGet。CsrGet 为 SparseAccessor：
 * 从 CSR 稀疏矩阵描述符中读出 rows / cols / nnz / csrRowOffsets / csrColInd /
 * csrValues / csrRowOffsetsType / csrColIndType / idxBase / valueType 全部字段，
 * 语义与 cuSPARSE 完全一致，确保 cuSPARSE 用户可无缝迁移。
 *
 * 与 COO 的关键差异：CSR 的行偏移索引类型（csrRowOffsetsType）与列索引类型
 * （csrColIndType）相互独立，各自可取 32I / 64I（四种组合均合法），因此 Get
 * 有两个独立的索引类型 OUT 参数。
 *
 * 守卫与字段回读复用 sparse/common/aclsparse_descr_guard.h 的
 * ReadCompressedSpMat（句柄/格式守卫 + 压缩格式标准字段按需回写），本文件
 * 仅声明 CSR 的 OUT 形参映射：
 *   - ptrs  <- csrRowOffsets（行偏移数组，长 rows+1）
 *   - idxs  <- csrColInd（长 nnz）
 *   - values<- csrValues（长 nnz）
 *   - ptrType <- csrRowOffsetsType，IdxType <- csrColIndType（相互独立）
 *   - baseType <- idxBase, valueType <- valueType
 */

#include "cann_ops_sparse.h"
#include "aclsparse_descr_guard.h"

extern "C" {

aclsparseStatus_t aclsparseCsrGet(aclsparseSpMatDescr_t matDescr, int64_t *rowNum, int64_t *colNum,
    int64_t *nonZeroCount, void **rowOffsets, void **colIndices, void **nonZeroValues,
    aclsparseIndexType_t *rowOffsetsType, aclsparseIndexType_t *colIndicesType,
    aclsparseIndexBase_t *baseIndex, aclDataType *dataType)
{
    return ReadCompressedSpMat("aclsparseCsrGet", matDescr, ACL_SPARSE_FORMAT_CSR, rowNum, colNum,
        nonZeroCount, rowOffsets, colIndices, nonZeroValues, rowOffsetsType, colIndicesType,
        baseIndex, dataType);
}

aclsparseStatus_t aclsparseConstCsrGet(aclsparseConstSpMatDescr_t matDescr, int64_t *rowNum,
    int64_t *colNum, int64_t *nonZeroCount, const void **rowOffsets, const void **colIndices,
    const void **nonZeroValues, aclsparseIndexType_t *rowOffsetsType,
    aclsparseIndexType_t *colIndicesType, aclsparseIndexBase_t *baseIndex, aclDataType *dataType)
{
    return ReadCompressedSpMat("aclsparseConstCsrGet", matDescr, ACL_SPARSE_FORMAT_CSR, rowNum,
        colNum, nonZeroCount, rowOffsets, colIndices, nonZeroValues, rowOffsetsType,
        colIndicesType, baseIndex, dataType);
}

} // extern "C"
