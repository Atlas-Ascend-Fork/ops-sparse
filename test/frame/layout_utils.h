/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_FRAME_LAYOUT_UTILS_H_
#define TEST_FRAME_LAYOUT_UTILS_H_

// =============================================================================
// ld + order 布局基建（共享层）。
//
// 物理布局语义（与算子侧 SpltCdOffset 同一语义，
// 对齐 cusparseLt：cusparseOrder_t 为内存布局，ld ≥ 物理维度允许 padding）：
//   ROW: 物理元素 (i, j) 位于 i*ld + j（行步长 ld，行尾 pad 跳过）
//   COL: 物理元素 (i, j) 位于 i + j*ld（列主序：物理连续段为列，列步长 ld）
// 紧凑（ROW ld == 物理列数）时与行主序紧凑摆放逐位一致；COL 的紧凑摆放为
// 转置形态（恒经 gather/scatter 重排）。
//
// 消费方：ltmatmul_test_utils.h（chain 布局覆盖）、prune_npu_wrapper.h
// （L9 padded 输入）、matmulSearch / ltmatmul test.cpp（C/D 输出解包）。
// =============================================================================

#include <cstdint>
#include <vector>

#include "cann_ops_sparseLt.h"

namespace sparse_test {

// PackPhysicalPadded: 紧凑物理行主序 → (order, ld) padded 物理布局。
// 快路径仅 ROW 且 ld==physCols（行主序紧凑与目标逐位一致）；COL 的紧凑
// 摆放为转置形态（物理 (i,j) at i + j*rows），恒经 gather 重排。
template <typename T>
inline std::vector<T> PackPhysicalPadded(const std::vector<T>& compactPhys,
    int64_t physRows, int64_t physCols, aclsparseOrder_t order, int64_t ld)
{
    if (order == ACL_SPARSE_ORDER_ROW &&
        (ld <= 0 || ld == physCols)) {
        return compactPhys;
    }
    const int64_t effLd = (order == ACL_SPARSE_ORDER_COL)
        ? ((ld > 0) ? ld : physRows) : ld;
    const size_t total = static_cast<size_t>(
        (order == ACL_SPARSE_ORDER_COL) ? physCols * effLd : physRows * effLd);
    std::vector<T> out(total, T{0});
    for (int64_t i = 0; i < physRows; ++i) {
        for (int64_t j = 0; j < physCols; ++j) {
            const T v = compactPhys[static_cast<size_t>(i) * physCols + j];
            if (order == ACL_SPARSE_ORDER_COL) {
                out[static_cast<size_t>(j) * effLd + i] = v;
            } else {
                out[static_cast<size_t>(i) * effLd + j] = v;
            }
        }
    }
    return out;
}

// UnpackPhysicalToLogical: (order, ld) padded 物理 → 紧凑行主序（C/D 回读解包）。
// 快路径仅 ROW 且 ld==cols；COL 恒经 scatter 还原。
template <typename T>
inline std::vector<T> UnpackPhysicalToLogical(const std::vector<T>& phys,
    int64_t rows, int64_t cols, aclsparseOrder_t order, int64_t ld)
{
    if (order == ACL_SPARSE_ORDER_ROW && (ld <= 0 || ld == cols)) {
        return phys;
    }
    const int64_t effLd = (order == ACL_SPARSE_ORDER_COL)
        ? ((ld > 0) ? ld : rows) : ld;
    std::vector<T> out(static_cast<size_t>(rows) * cols);
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            if (order == ACL_SPARSE_ORDER_COL) {
                out[static_cast<size_t>(i) * cols + j] =
                    phys[static_cast<size_t>(j) * effLd + i];
            } else {
                out[static_cast<size_t>(i) * cols + j] =
                    phys[static_cast<size_t>(i) * effLd + j];
            }
        }
    }
    return out;
}

// MatmulLayoutOpts: chain 布局扩展参数（默认值 = 既有紧凑行为，零回归锚点）。
//   ldSparse  — structured 侧 ld（A-sparse 的 A / B-sparse 的 B）；0 = 紧凑
//   ldDense   — dense 侧 ld（A-sparse 的 B / B-sparse 的 A）；0 = 紧凑
//   denseOrder— dense 侧 order（默认 ROW）
//   cdOrder   — C/D order（默认 ROW；C/D 须同 order，ValidateCDConsistency）
//   ldCd      — C/D ld；0 = 紧凑（ROW → n，COL → m）
struct MatmulLayoutOpts {
    int64_t ldSparse = 0;
    int64_t ldDense = 0;
    aclsparseOrder_t denseOrder = ACL_SPARSE_ORDER_ROW;
    aclsparseOrder_t cdOrder = ACL_SPARSE_ORDER_ROW;
    int64_t ldCd = 0;
    // (b) 预剪直喂开关：稀疏侧不经 SpMMAPrune，直接以 CPU golden 逻辑剪枝
    // 结果按 (order, ldSparse) 物理散布喂 matmul——独立验证 matmul 稀疏侧
    // (order, ld) 贯通（剥离 prune 依赖）。经 layout 参数链透传（链路行为
    // 开关与布局覆盖同渠道）。
    bool directFeed = false;

    bool hasSparseLd() const { return ldSparse > 0; }
    bool hasDenseLayout() const { return ldDense > 0 || denseOrder == ACL_SPARSE_ORDER_COL; }
    bool hasCdLayout() const
    {
        return ldCd > 0 || cdOrder == ACL_SPARSE_ORDER_COL;
    }
};

// CdLdOf / CdSpanElems: C/D 的实际 ld 与单 batch 物理 span（元素数）。
inline int64_t CdLdOf(int32_t m, int32_t n, const MatmulLayoutOpts& o)
{
    if (o.ldCd > 0) { return o.ldCd; }
    return (o.cdOrder == ACL_SPARSE_ORDER_COL) ? m : n;
}
inline size_t CdSpanElems(int32_t m, int32_t n, const MatmulLayoutOpts& o)
{
    const int64_t ld = CdLdOf(m, n, o);
    return static_cast<size_t>(
        (o.cdOrder == ACL_SPARSE_ORDER_COL) ? static_cast<int64_t>(n) * ld
                                            : static_cast<int64_t>(m) * ld);
}

// PackBatchStridedPhysical: 逻辑 batch-strided 数据（每 batch 前 rows*cols 个
// 有效元素 + stride padding）→ 每 batch (order, ld) 物理布局（batch 间同 stride
// 推进，物理 span 内 pad 置零）。nb==1 退化为 PackPhysicalPadded。
template <typename T>
inline std::vector<T> PackBatchStridedPhysical(const std::vector<T>& logicalBatch,
    int32_t rows, int32_t cols, aclsparseOrder_t order, int64_t ld,
    int32_t numBatches, int64_t batchStride)
{
    if (numBatches <= 1) {
        return PackPhysicalPadded<T>(logicalBatch, rows, cols, order, ld);
    }
    if (order == ACL_SPARSE_ORDER_ROW && (ld <= 0 || ld == cols)) {
        return logicalBatch;  // ROW 紧凑 batch-strided 原样
    }
    const int64_t effLd = (order == ACL_SPARSE_ORDER_COL)
        ? ((ld > 0) ? ld : rows) : ld;
    const size_t total = static_cast<size_t>(numBatches) * batchStride;
    std::vector<T> out(total, T{0});
    for (int32_t b = 0; b < numBatches; ++b) {
        const size_t srcOff = static_cast<size_t>(b) * batchStride;
        const size_t dstOff = srcOff;
        for (int32_t i = 0; i < rows; ++i) {
            for (int32_t j = 0; j < cols; ++j) {
                const T v = logicalBatch[srcOff + static_cast<size_t>(i) * cols + j];
                if (order == ACL_SPARSE_ORDER_COL) {
                    out[dstOff + static_cast<size_t>(j) * effLd + i] = v;
                } else {
                    out[dstOff + static_cast<size_t>(i) * effLd + j] = v;
                }
            }
        }
    }
    return out;
}

// UnpackCdBatchSegment: 单 batch 段解包（(order, ld) 物理 → packed 逻辑
// 行，写入 packed + dstOff 基址）。自 ExtractCdLogicalBatch 的内层双循环
// 拆出（控制函数嵌套深度 ≤5——codecheck 超大深度函数整改）。
template <typename T>
inline void UnpackCdBatchSegment(std::vector<T>& packed, size_t dstOff,
    const std::vector<T>& physBatch, size_t srcOff,
    int32_t m, int32_t n, aclsparseOrder_t cdOrder, int64_t ld)
{
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            if (cdOrder == ACL_SPARSE_ORDER_COL) {
                packed[dstOff + static_cast<size_t>(i) * n + j] =
                    physBatch[srcOff + static_cast<size_t>(j) * ld + i];
            } else {
                packed[dstOff + static_cast<size_t>(i) * n + j] =
                    physBatch[srcOff + static_cast<size_t>(i) * ld + j];
            }
        }
    }
}

// ExtractCdLogicalBatch: C/D 物理 batch-strided 输出 → packed 逻辑输出
// （每 batch 解包 rows*cols 后拼接；ld == min 且 ROW 时仅做 batch 抽取）。
template <typename T>
inline std::vector<T> ExtractCdLogicalBatch(const std::vector<T>& physBatch,
    int32_t m, int32_t n, const MatmulLayoutOpts& o,
    int32_t numBatches, int64_t batchStride)
{
    const int64_t ld = CdLdOf(m, n, o);
    const size_t mn = static_cast<size_t>(m) * n;
    if (numBatches <= 1) {
        return UnpackPhysicalToLogical<T>(physBatch, m, n, o.cdOrder, ld);
    }
    std::vector<T> packed(static_cast<size_t>(numBatches) * mn);
    for (int32_t b = 0; b < numBatches; ++b) {
        const size_t srcOff = static_cast<size_t>(b) * batchStride;
        const size_t dstOff = static_cast<size_t>(b) * mn;
        if (o.cdOrder == ACL_SPARSE_ORDER_COL || ld != n) {
            UnpackCdBatchSegment<T>(packed, dstOff, physBatch, srcOff, m, n, o.cdOrder, ld);
        } else {
            std::copy(physBatch.begin() + srcOff, physBatch.begin() + srcOff + mn,
                      packed.begin() + dstOff);
        }
    }
    return packed;
}

}  // namespace sparse_test

#endif  // TEST_FRAME_LAYOUT_UTILS_H_
