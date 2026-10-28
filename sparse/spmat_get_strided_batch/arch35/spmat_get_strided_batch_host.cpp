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
 * \file spmat_get_strided_batch_host.cpp
 * \brief aclsparseSpMatGetStridedBatch Host 侧实现（arch35）。
 *
 * 对标 cuSPARSE cusparseSpMatGetStridedBatch。纯 Host 侧描述符访问器，
 * 不启动 NPU kernel、不涉及 stream 同步：从任意格式（COO/CSR/CSC/SELL/
 * BLOCKED_ELL）稀疏矩阵描述符读回 batch 数。未 batch 化的新建描述符返回 1。
 *
 * 仅返回 batchCount 而不返回 batchStride，与 cuSPARSE 完全一致：stride 签名
 * 是格式相关的（COO 1 个 / CSR 2 个 / BSR 3 个），通用 getter 唯一能跨格式
 * 统一暴露的公共量是 batchCount。batch 元数据的写入由各格式的 SetStridedBatch
 * 接口完成（当前为 aclsparseCooSetStridedBatch）。
 */

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"

extern "C" {

aclsparseStatus_t aclsparseSpMatGetStridedBatch(aclsparseConstSpMatDescr_t spMatDescr, int* batchCount)
{
    // 签名为各描述符结构体首字段（偏移 0），对任意错型句柄读取签名本身安全；
    // 拒绝 nullptr 与签名非法（错型/损坏）句柄，与 aclsparseSpMatGetFormat /
    // aclsparseSpMatGetSize / aclsparseDnMatGetStridedBatch 的防御约定一致。
    if (spMatDescr == nullptr || spMatDescr->signature != kSpMatSignature) {
        OP_LOGE("aclsparseSpMatGetStridedBatch", "spMatDescr is nullptr or invalid");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (batchCount == nullptr) {
        OP_LOGE("aclsparseSpMatGetStridedBatch", "batchCount is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *batchCount = spMatDescr->batchCount;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
