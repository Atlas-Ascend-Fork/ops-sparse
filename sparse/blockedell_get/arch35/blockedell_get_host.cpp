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
 * \file blockedell_get_host.cpp
 * \brief aclsparseBlockedEllGet / aclsparseConstBlockedEllGet Host 侧实现（arch35）。
 *
 * 对标 cuSPARSE cusparseCreateBlockedEll / cusparseBlockedEllGet。BlockedEllGet 为
 * SparseAccessor：从 Blocked-ELL 稀疏矩阵描述符中读回 rows / cols / ellBlockSize /
 * ellCols / ellColInd / ellValue / ellIdxType / idxBase / valueType 全部字段，
 * 语义与 cuSPARSE 完全一致，确保 cuSPARSE 用户可无缝迁移。
 *
 * 守卫与 OUT 回写复用 sparse/common/aclsparse_descr_guard.h（与 CsrGet 等压缩格式
 * 访问器同源）：CheckSpMatHandle（null + 签名校验）、CheckSpMatFormat（格式校验）、
 * WriteOutIfRequested（OUT 可空的部分读取语义）。Blocked-ELL 字段与压缩格式不同
 * （ellBlockSize/ellCols/单索引数组），不走 ReadCompressedSpMat，按同款结构组合。
 *
 * 描述符内部结构复用 aclsparseSpMatDescr（见 aclsparse_descr_internal.h），
 * 字段映射与 aclsparseCreateBlockedEll 一一对应：
 *   - ellColInd <- ellColInd（Create 同时写入 idxs，读取专用字段 ellColInd）
 *   - ellValue <- values
 *   - ellIdxType <- IdxType（Create 时 ptrType == IdxType == indexType）
 *   - baseType <- indexBase, valueType <- valueType,
 *     format <- ACL_SPARSE_FORMAT_BLOCKED_ELL
 */

#include "cann_ops_sparse.h"
#include "aclsparse_descr_guard.h"

template <typename PtrOut>
static inline aclsparseStatus_t BlockedEllGetImpl(const char *fn, const aclsparseSpMatDescr *d,
    int64_t *rows, int64_t *cols, int64_t *ellBlockSize, int64_t *ellCols,
    PtrOut ellColInd, PtrOut ellValue, aclsparseIndexType_t *ellIdxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType)
{
    const aclsparseStatus_t handleSt = CheckSpMatHandle(fn, d);
    if (handleSt != ACL_SPARSE_STATUS_SUCCESS) {
        return handleSt;
    }
    const aclsparseStatus_t formatSt = CheckSpMatFormat(fn, d, ACL_SPARSE_FORMAT_BLOCKED_ELL);
    if (formatSt != ACL_SPARSE_STATUS_SUCCESS) {
        return formatSt;
    }
    WriteOutIfRequested(rows, static_cast<int64_t>(d->rows));
    WriteOutIfRequested(cols, static_cast<int64_t>(d->cols));
    WriteOutIfRequested(ellBlockSize, static_cast<int64_t>(d->ellBlockSize));
    WriteOutIfRequested(ellCols, static_cast<int64_t>(d->ellCols));
    WriteOutIfRequested(ellColInd, d->ellColInd);
    WriteOutIfRequested(ellValue, d->values);
    WriteOutIfRequested(ellIdxType, d->IdxType);
    WriteOutIfRequested(idxBase, d->baseType);
    WriteOutIfRequested(valueType, d->valueType);
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" {

aclsparseStatus_t aclsparseBlockedEllGet(aclsparseSpMatDescr_t spMatDescr, int64_t *rows,
    int64_t *cols, int64_t *ellBlockSize, int64_t *ellCols, void **ellColInd, void **ellValue,
    aclsparseIndexType_t *ellIdxType, aclsparseIndexBase_t *idxBase, aclDataType *valueType)
{
    return BlockedEllGetImpl("aclsparseBlockedEllGet", spMatDescr, rows, cols, ellBlockSize,
        ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType);
}

aclsparseStatus_t aclsparseConstBlockedEllGet(aclsparseConstSpMatDescr_t spMatDescr,
    int64_t *rows, int64_t *cols, int64_t *ellBlockSize, int64_t *ellCols,
    const void **ellColInd, const void **ellValue, aclsparseIndexType_t *ellIdxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType)
{
    return BlockedEllGetImpl("aclsparseConstBlockedEllGet", spMatDescr, rows, cols, ellBlockSize,
        ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType);
}

} // extern "C"
