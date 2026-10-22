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

#ifndef SPMM_ARCH22_CSR_MAT_H
#define SPMM_ARCH22_CSR_MAT_H

#include <vector>

#include "spmm.h"
#include "aclsparse_descr_internal.h"

// 行块端点的 int64 截断：start + width 在 int64 域相加后与 limit 取小再窄化为 int32。
// 抽成内联纯函数供装箱逻辑与单元测试共用，测试可直接以 m 接近 INT32_MAX 的边界值
// 调用，无需分配 m 大小的数组；旧实现用 int32 做 lo+blk / (b+1)*rowsPerBin 会回绕。
static inline int32_t SpmmArch22ClampBinEnd(int64_t start, int64_t width, int64_t limit)
{
    const int64_t end = start + width;
    return static_cast<int32_t>(end > limit ? limit : end);
}

class SpmmArch22CsrMat {
public:
    // D1 fix: callers (spmm_host.cpp) already reject nullptr matA before
    // constructing this object, but this constructor must not crash the
    // whole process on unexpected input (G.STD.16 forbids abort()/exit()).
    // A nullptr inner degrades to an empty (rows=cols=nnz=0) matrix instead;
    // DoPreProcess() below re-checks matDesc and returns an error status.
    SpmmArch22CsrMat(aclsparseSpMatDescr *inner, int32_t blockDim)
        : matDesc(inner),
          blockDim(blockDim)
    {
        if (inner == nullptr) {
            rows = 0;
            cols = 0;
            nnz = 0;
            return;
        }
        rows = static_cast<int64_t>(inner->rows);
        cols = static_cast<int64_t>(inner->cols);
        nnz = static_cast<int64_t>(inner->nnz);
    }

    // rowBlock: 装箱粒度。0 或 1 表示逐行装箱（原行为）；>=2 表示按连续 rowBlock 行
    // 一块装箱，让 kernel 侧一组内的行号连续。取值必须与 kernel 侧的 rowGroup 相等，
    // 且必须与写进 tiling 的 rowBlockHost 一致——kernel 靠后者判断能否信任连续性。
    // 调用方统一走 SpmmArch22RowGroup + SpmmArch22EffectiveRowBlock 取值。
    // deterministic=true 时忽略 rowBlock，固定使用原始 CSR 行序和均匀行切分。
    aclsparseStatus_t DoPreProcess(uint8_t *dWorkspace, int64_t reorderOff, int64_t binEdgeOff,
        int64_t transposeRowOff, int64_t transposeSourceRowOff,
        int64_t transposePermutationOff, uint32_t rowBlock = 0, bool transpose = false,
        bool deterministic = false);

private:
    aclsparseStatus_t ValidateRowOffsets(const std::vector<uint32_t> &rowOff) const;
    aclsparseStatus_t ValidateColumnIndices() const;
    void GreedyRowBinPack(const uint32_t *rowNnz, int32_t *reorder, int32_t *binEdges);
    void GreedyBlockBinPack(const uint32_t *rowNnz, uint32_t rowBlock, int32_t *reorder,
        int32_t *binEdges);
    void BuildIdentityReorder(int32_t count, int32_t *reorder, int32_t *binEdges);
    void BuildPreprocessReorder(const std::vector<uint32_t> &rowOff, int32_t count,
        uint32_t rowBlock, bool transpose, bool deterministic, int32_t *reorder,
        int32_t *binEdges);
    aclsparseStatus_t BuildTransposeWorkspace(const std::vector<uint32_t> &rowOff,
        uint8_t *workspace, int64_t rowOffset, int64_t sourceRowOffset,
        int64_t permutationOffset);

    aclsparseSpMatDescr *matDesc;
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    int32_t blockDim;
};

#endif
