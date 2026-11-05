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
 * \file csc_get_host.cpp
 * \brief aclsparseCscGet / aclsparseConstCscGet Host 侧实现（arch35）。
 *
 * 对标 cuSPARSE cusparseCreateCsc / cusparseCscGet。CscGet 为 SparseAccessor：
 * 从 CSC 稀疏矩阵描述符中读出 rows / cols / nnz / cscColOffsets / cscRowInd /
 * cscValues / cscColOffsetsType / cscRowIndType / idxBase / valueType 全部字段，
 * 语义与 cuSPARSE 完全一致，确保 cuSPARSE 用户可无缝迁移。
 *
 * 与 CSR/COO 的关键差异：CSC 的列偏移数组 cscColOffsets 长度为 cols+1
 * （区别于 CSR 的 rows+1）；列偏移索引类型（cscColOffsetsType）与行索引
 * 类型（cscRowIndType）相互独立，各自可取 32I / 64I（四种组合均合法），
 * 因此 Get 有两个独立的索引类型 OUT 参数。
 *
 * 守卫与字段回读复用 sparse/common/aclsparse_descr_guard.h 的
 * ReadCompressedSpMat（句柄/签名/格式守卫 + 压缩格式标准字段按需回写），
 * 本文件仅声明 CSC 的 OUT 形参映射：
 *   - ptrs  <- cscColOffsets（列偏移数组，长 cols+1，区别于 CSR 的 rows+1）
 *   - idxs  <- cscRowInd（长 nnz）
 *   - values<- cscValues（长 nnz）
 *   - ptrType <- cscColOffsetsType，IdxType <- cscRowIndType（相互独立）
 *   - baseType <- idxBase, valueType <- valueType
 */

#include "cann_ops_sparse.h"
#include "aclsparse_descr_guard.h"

extern "C" {

aclsparseStatus_t aclsparseCscGet(aclsparseSpMatDescr_t matDescr, int64_t *rowNum, int64_t *colNum,
    int64_t *nonZeroCount, void **colOffsets, void **rowIndices, void **nonZeroValues,
    aclsparseIndexType_t *colOffsetsType, aclsparseIndexType_t *rowIndicesType,
    aclsparseIndexBase_t *baseIndex, aclDataType *dataType)
{
    return ReadCompressedSpMat("aclsparseCscGet", matDescr, ACL_SPARSE_FORMAT_CSC, rowNum, colNum,
        nonZeroCount, colOffsets, rowIndices, nonZeroValues, colOffsetsType, rowIndicesType, baseIndex,
        dataType);
}

aclsparseStatus_t aclsparseConstCscGet(aclsparseConstSpMatDescr_t matDescr, int64_t *rowNum,
    int64_t *colNum, int64_t *nonZeroCount, const void **colOffsets, const void **rowIndices,
    const void **nonZeroValues, aclsparseIndexType_t *colOffsetsType, aclsparseIndexType_t *rowIndicesType,
    aclsparseIndexBase_t *baseIndex, aclDataType *dataType)
{
    return ReadCompressedSpMat("aclsparseConstCscGet", matDescr, ACL_SPARSE_FORMAT_CSC, rowNum,
        colNum, nonZeroCount, colOffsets, rowIndices, nonZeroValues, colOffsetsType, rowIndicesType,
        baseIndex, dataType);
}

} // extern "C"
