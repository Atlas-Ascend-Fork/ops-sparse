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
 * \file csr_set_strided_batch_host.cpp
 * \brief aclsparseCsrSetStridedBatch Host 侧实现（arch35）。
 *
 * 对标 cuSPARSE cusparseCsrSetStridedBatch。纯 Host 侧描述符操作，
 * 不启动 NPU kernel、不涉及 stream 同步：将 batchCount 与两个 batch 步长写入
 * CSR 稀疏矩阵描述符，使其可表达 batch 化稀疏矩阵。步长按元素个数计：
 * offsetsBatchStride 为相邻 batch 的 csrRowOffsets 起始地址间元素偏移（span =
 * rows + 1）；columnsValuesBatchStride 为相邻 batch 的 csrColInd / csrValues
 * 起始地址间元素偏移（两数组共用，span = nnz，写入描述符共享字段
 * columnsBatchStride 与 valuesBatchStride 同值）。两个步长取 0 均表示广播——
 * 所有 batch 共享对应数组（对齐 cuSPARSE 官方样例
 * CUDALibrarySamples/cuSPARSE/spmm_csr_batched 的 offsets 单份 + stride=0 用法，
 * 与本仓 BSR / COO 的 0=广播约定一致）。batchCount 的读回由配套算子
 * aclsparseSpMatGetStridedBatch（sparse/spmat_get_strided_batch/）提供。
 */

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"

#include <limits>

// 本文件私有副本（sparse/common/aclsparse_descr.cpp 存在同名文件内定义但未导出；
// 为控制本 PR 对 sparse/common 的侵入面，不将共享常量与受检运算提升到公共头，
// 数值语义与 descr.cpp 版本保持一致，与 coo_set_strided_batch_host.cpp 相同，
// 若后续 common 统一导出可届时删除副本）。
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

// 校验单个 batch 步长（对齐 cuSPARSE 官方样例与本仓 BSR / COO / DnMat 的 0=广播约定）：
// 0 表示广播——所有 batch 共享对应数组，单一物理副本按数组长度布局，无跨 batch
// 地址递进，不存在溢出；正值必须不小于对应数组跨度 span（相邻 batch 数据区间
// 不重叠），且 (batchCount-1)*stride+span 不溢出 int64（受检乘加，CheckedStridedExtent
// 为本文件私有副本，语义同 descr.cpp，与 aclsparseBsrSetStridedBatch /
// aclsparseDnMatSetStridedBatch / aclsparseCooSetStridedBatch 的溢出防护一致）；
// 负值非法。spanName / strideName 用于日志定位。
static aclsparseStatus_t ValidateBatchStride(const char *fn, int batchCount, int64_t stride,
                                              int64_t span, const char *strideName, const char *spanName)
{
    if (stride < 0) {
        OP_LOGE(fn, "%s %lld is negative", strideName, static_cast<long long>(stride));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (stride == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (stride < span) {
        OP_LOGE(fn, "%s %lld is invalid: must be 0 (broadcast) or >= %s %lld", strideName,
                static_cast<long long>(stride), spanName, static_cast<long long>(span));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int64_t extent = 0;
    if (!CheckedStridedExtent(batchCount, stride, span, &extent)) {
        OP_LOGE(fn, "%s strided batch extent overflows int64: batchCount %d, %s %lld",
                strideName, batchCount, strideName, static_cast<long long>(stride));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" {

aclsparseStatus_t aclsparseCsrSetStridedBatch(aclsparseSpMatDescr_t spMatDescr, int batchCount,
                                              int64_t offsetsBatchStride, int64_t columnsValuesBatchStride)
{
    if (spMatDescr == nullptr || spMatDescr->signature != kSpMatSignature) {
        OP_LOGE("aclsparseCsrSetStridedBatch", "spMatDescr is nullptr or invalid");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (spMatDescr->format != ACL_SPARSE_FORMAT_CSR) {
        OP_LOGE("aclsparseCsrSetStridedBatch", "descriptor format %d is not CSR",
                static_cast<int>(spMatDescr->format));
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (batchCount < 1 || batchCount > kMaxStridedBatchCount) {
        OP_LOGE("aclsparseCsrSetStridedBatch", "batchCount %d is out of range [1, %d]", batchCount,
                kMaxStridedBatchCount);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 两个步长分别作用于行偏移数组（跨度 rows + 1）与列索引/值数组（跨度 nnz），
    // 各自按 0=广播 / 正值 >= 跨度 / 负值非法 / 受检乘加溢出防护校验（见
    // ValidateBatchStride）。rows / nnz 由 aclsparseCreateCsr 校验非负后存入，但
    // rows 允许到达 INT64_MAX（延迟绑定下无需真实分配），rows + 1 须受检加法构造，
    // 溢出返回 INVALID_VALUE 且不触碰描述符既有状态；nnz 无加法、转型安全。
    int64_t offsetsSpan = 0;
    if (!CheckedNonnegativeAdd(static_cast<int64_t>(spMatDescr->rows), 1, &offsetsSpan)) {
        OP_LOGE("aclsparseCsrSetStridedBatch", "rows %lld overflows offsets span",
                static_cast<long long>(spMatDescr->rows));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const int64_t columnsValuesSpan = static_cast<int64_t>(spMatDescr->nnz);
    const aclsparseStatus_t offsetsSt = ValidateBatchStride("aclsparseCsrSetStridedBatch", batchCount,
        offsetsBatchStride, offsetsSpan, "offsetsBatchStride", "rows + 1");
    if (offsetsSt != ACL_SPARSE_STATUS_SUCCESS) {
        return offsetsSt;
    }
    const aclsparseStatus_t columnsValuesSt = ValidateBatchStride("aclsparseCsrSetStridedBatch",
        batchCount, columnsValuesBatchStride, columnsValuesSpan, "columnsValuesBatchStride", "nnz");
    if (columnsValuesSt != ACL_SPARSE_STATUS_SUCCESS) {
        return columnsValuesSt;
    }
    // columnsValuesBatchStride 同时作用于列索引与值两数组（cuSPARSE CSR 语义），
    // 写入描述符共享拆分字段同值，后续批量消费算子可与 BSR 拆分字段同源读取。
    const bool changed = spMatDescr->batchCount != batchCount ||
                         spMatDescr->offsetsBatchStride != offsetsBatchStride ||
                         spMatDescr->columnsBatchStride != columnsValuesBatchStride ||
                         spMatDescr->valuesBatchStride != columnsValuesBatchStride;
    spMatDescr->batchCount = batchCount;
    spMatDescr->offsetsBatchStride = offsetsBatchStride;
    spMatDescr->columnsBatchStride = columnsValuesBatchStride;
    spMatDescr->valuesBatchStride = columnsValuesBatchStride;
    if (changed) {
        // 与 aclsparseBsrSetStridedBatch / aclsparseCooSetStridedBatch 语义一致：
        // batch 元数据实际变化时失效既有预处理缓存（sddmm 等），避免消费算子
        // 按旧 batch 布局复用缓存数据。
        AclsparseMarkSpMatStructureChanged(spMatDescr);
        spMatDescr->sddmmState.reset();
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
