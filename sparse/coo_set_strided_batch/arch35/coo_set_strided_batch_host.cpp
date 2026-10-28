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
 * \file coo_set_strided_batch_host.cpp
 * \brief aclsparseCooSetStridedBatch Host 侧实现（arch35）。
 *
 * 对标 cuSPARSE cusparseCooSetStridedBatch。纯 Host 侧描述符操作，
 * 不启动 NPU kernel、不涉及 stream 同步：将 batchCount / batchStride 写入
 * COO 稀疏矩阵描述符，使其可表达 batch 化稀疏矩阵；batchStride 为相邻 batch
 * 的 cooRowInd / cooColInd / cooValues 起始地址间的元素偏移，0 表示广播（所有
 * batch 共享同一组数组，对齐 cuSPARSE 官方样例与本仓 BSR 约定）。batchCount
 * 的读回由配套算子 aclsparseSpMatGetStridedBatch（sparse/spmat_get_strided_batch/）提供。
 */

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"

#include <limits>

// 本文件私有副本（sparse/common/aclsparse_descr.cpp 存在同名文件内定义但未导出；
// 为控制本 PR 对 sparse/common 的侵入面，不将共享常量与受检运算提升到公共头，
// 数值语义与 descr.cpp 版本保持一致，若后续 common 统一导出可届时删除副本）。
namespace {

constexpr int kMaxStridedBatchCount = 65535;

bool CheckedNonnegativeMul(int64_t lhs, int64_t rhs, int64_t* result)
{
    if (result == nullptr || lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool CheckedNonnegativeAdd(int64_t lhs, int64_t rhs, int64_t* result)
{
    if (result == nullptr || lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool CheckedStridedExtent(int batchCount, int64_t stride, int64_t span, int64_t* extent)
{
    if (batchCount < 1 || stride < 0 || span < 0 || extent == nullptr) {
        return false;
    }
    int64_t lastBatchOffset = 0;
    if (!CheckedNonnegativeMul(static_cast<int64_t>(batchCount - 1), stride, &lastBatchOffset)) {
        return false;
    }
    return CheckedNonnegativeAdd(lastBatchOffset, span, extent);
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseCooSetStridedBatch(aclsparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride)
{
    if (spMatDescr == nullptr || spMatDescr->signature != kSpMatSignature) {
        OP_LOGE("aclsparseCooSetStridedBatch", "spMatDescr is nullptr or invalid");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (spMatDescr->format != ACL_SPARSE_FORMAT_COO) {
        OP_LOGE("aclsparseCooSetStridedBatch", "descriptor format %d is not COO", static_cast<int>(spMatDescr->format));
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (batchCount < 1 || batchCount > kMaxStridedBatchCount) {
        OP_LOGE("aclsparseCooSetStridedBatch", "batchCount %d is out of range [1, %d]", batchCount,
                kMaxStridedBatchCount);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // batchStride 语义（对齐 cuSPARSE 官方样例与本仓 BSR 的 0=广播约定）：
    // 0 表示广播——所有 batch 共享同一组 row/col/values 数组（样例
    // CUDALibrarySamples/cuSPARSE/spmm_coo_batched: cusparseCooSetStridedBatch(matA, num_batches, 0)），
    // 此时单一物理副本按 nnz 布局，无跨 batch 地址递进，不存在溢出；
    // 正值必须 >= nnz（相邻 batch 数据区间不重叠），且 (batchCount-1)*batchStride+nnz
    // 不溢出 int64；负值非法。nnz 受物理内存约束不会超过 INT64_MAX，转型安全。
    const int64_t nnz = static_cast<int64_t>(spMatDescr->nnz);
    if (batchStride < 0) {
        OP_LOGE("aclsparseCooSetStridedBatch", "batchStride %lld is negative",
                static_cast<long long>(batchStride));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (batchStride > 0) {
        if (batchStride < nnz) {
            OP_LOGE("aclsparseCooSetStridedBatch", "batchStride %lld is invalid: must be 0 (broadcast) or >= nnz %lld",
                    static_cast<long long>(batchStride), static_cast<long long>(nnz));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        // 地址范围受检乘加：验证 (batchCount-1)*batchStride + nnz 不超出 int64_t，
        // 与 aclsparseBsrSetStridedBatch / aclsparseDnMatSetStridedBatch 的溢出
        // 防护保持一致（CheckedStridedExtent 为本文件私有副本，语义同 descr.cpp）。
        int64_t extent = 0;
        if (!CheckedStridedExtent(batchCount, batchStride, nnz, &extent)) {
            OP_LOGE("aclsparseCooSetStridedBatch",
                    "strided batch extent overflows int64: batchCount %d, batchStride %lld", batchCount,
                    static_cast<long long>(batchStride));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    const bool changed = spMatDescr->batchCount != batchCount || spMatDescr->batchStride != batchStride;
    spMatDescr->batchCount = batchCount;
    spMatDescr->batchStride = batchStride;
    if (changed) {
        // 与 aclsparseBsrSetStridedBatch 语义一致：batch 元数据实际变化时失效既有
        // 预处理缓存（sddmm 等），避免消费算子按旧 batch 布局复用缓存数据。
        AclsparseMarkSpMatStructureChanged(spMatDescr);
        spMatDescr->sddmmState.reset();
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
