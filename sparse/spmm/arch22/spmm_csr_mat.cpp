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

#include "spmm_csr_mat.h"
#include "spmm.h"
#include "acl/acl.h"
#include <algorithm>
#include <string.h>
#include <vector>
#include <numeric>

void SpmmArch22CsrMat::GreedyRowBinPack(const uint32_t *rowNnz, int32_t *reorder, int32_t *binEdges)
{
    /* blockDim 来自 GetSpmmBlockDim 恒 >=1，但这是跨函数不变量，静态分析看不到；
     * 显式兜底，避免 blockDim==0 时 bins/binLoad 为空 vector 被下标访问。 */
    if (blockDim <= 0) {
        binEdges[0] = 0;
        return;
    }
    int32_t m = static_cast<int32_t>(rows);
    std::vector<int32_t> sortedIdx(m);
    std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
    std::sort(sortedIdx.begin(), sortedIdx.end(), [&](int32_t a, int32_t b) {
        if (rowNnz[a] != rowNnz[b]) {
            return rowNnz[a] > rowNnz[b];
        }
        return a < b;
    });

    std::vector<uint64_t> binLoad(blockDim, 0);
    std::vector<std::vector<int32_t>> bins(blockDim);

    for (int32_t i = 0; i < m; i++) {
        int32_t minBin = 0;
        for (int32_t b = 1; b < blockDim; b++) {
            if (binLoad[b] < binLoad[minBin]) { minBin = b; }
        }
        bins[minBin].push_back(sortedIdx[i]);
        binLoad[minBin] += rowNnz[sortedIdx[i]];
    }

    int32_t pos = 0;
    binEdges[0] = 0;
    for (int32_t b = 0; b < blockDim; b++) {
        for (int32_t r : bins[b]) {
            reorder[pos++] = r;
        }
        binEdges[b + 1] = pos;
    }
}

/* 与 GreedyRowBinPack 同一套 LPT 贪心，但装箱单位是连续的 rowBlock 行而不是单行。
 *
 * 逐行装箱把行号最大程度打散，于是 kernel 侧一组行的 rowOffsets、C 的读入与写出
 * 都只能逐行发散落的访问。实测(P-02 尺寸/nnz 对齐的规则合成 CSR，fp32)每行残留
 * 754ns 就主要
 * 是这些散读：把分组从 16 行加到 64 行只再快 4.5%，说明瓶颈不在每组同步而在每行
 * 访问。按块装箱后一组内行号连续，kernel 可以一次连续 DMA 取回整组 rowOffsets，
 * C 也能合成一条多块 DMA。
 *
 * 均衡度的代价可控：块内行度数被抹平，但块数远大于核数(P-02 是 169343/64=2646 块
 * 对 40 核，P-03 是 38266 块对 40 核)，LPT 仍有足够多的块可调度。M 小的时候块数
 * 不够，调用方会退回逐行装箱。
 *
 * 尾块不满 rowBlock 时会被排到所在 bin 的最后，这样每个 bin 内除末尾外的组都正好
 * 对齐一个整块，kernel 的连续性判定才不会因为错位而大面积落空。 */
void SpmmArch22CsrMat::GreedyBlockBinPack(const uint32_t *rowNnz, uint32_t rowBlock,
    int32_t *reorder, int32_t *binEdges)
{
    int32_t m = static_cast<int32_t>(rows);
    int32_t blk = static_cast<int32_t>(rowBlock);
    /* Callers only reach here with rowBlock >= 2 (see DoPreProcess), but that
     * invariant lives at the call site, not here -- guard directly so the
     * division below is provably safe on its own. */
    if (blk <= 0) {
        blk = 1;
    }
    // 用 int64_t 做向上取整除法，避免 m 接近 INT32_MAX 时 (m + blk - 1) 溢出。
    int32_t nBlocks = static_cast<int32_t>((static_cast<int64_t>(m) + blk - 1) / blk);

    std::vector<uint64_t> blockNnz(nBlocks, 0);
    for (int32_t i = 0; i < m; i++) {
        blockNnz[i / blk] += rowNnz[i];
    }

    std::vector<int32_t> order(nBlocks);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        if (blockNnz[a] != blockNnz[b]) {
            return blockNnz[a] > blockNnz[b];
        }
        return a < b;
    });

    std::vector<uint64_t> binLoad(blockDim, 0);
    std::vector<std::vector<int32_t>> bins(blockDim);
    for (int32_t i = 0; i < nBlocks; i++) {
        int32_t minBin = 0;
        for (int32_t b = 1; b < blockDim; b++) {
            if (binLoad[b] < binLoad[minBin]) { minBin = b; }
        }
        bins[minBin].push_back(order[i]);
        binLoad[minBin] += blockNnz[order[i]];
    }

    int32_t pos = 0;
    binEdges[0] = 0;
    for (int32_t b = 0; b < blockDim; b++) {
        // 尾块(唯一可能不满的块)排到本 bin 最后，保持前面各组与整块对齐。
        auto tail = std::find(bins[b].begin(), bins[b].end(), nBlocks - 1);
        if (tail != bins[b].end() && m % blk != 0) {
            bins[b].erase(tail);
            bins[b].push_back(nBlocks - 1);
        }
        for (int32_t blkId : bins[b]) {
            // 端点用 int64 计算并先截断，避免 m 接近 INT32_MAX 时 lo+blk 越过上限
            // 回绕成负值（尾块 lo=2147483584、blk=64 时 lo+blk 已超 INT32_MAX），
            // 使末尾几行不进 reorder/bin 范围。
            int64_t lo = static_cast<int64_t>(blkId) * blk;
            int32_t hi = SpmmArch22ClampBinEnd(lo, blk, m);
            for (int64_t r = lo; r < hi; r++) {
                reorder[pos++] = static_cast<int32_t>(r);
            }
        }
        binEdges[b + 1] = pos;
    }
}

void SpmmArch22CsrMat::BuildIdentityReorder(int32_t count, int32_t *reorder, int32_t *binEdges)
{
    int32_t m = count;
    for (int32_t i = 0; i < m; i++) { reorder[i] = i; }

    /* blockDim comes from the host core count via the constructor and is
     * never 0 in practice, but that's invisible to a per-function static
     * check, so guard the divisor explicitly. binEdges is sized blockDim+1
     * by the caller, so the loop bound below stays on the real blockDim. */
    int32_t divisor = (blockDim > 0) ? blockDim : 1;
    // 用 int64_t 做向上取整除法，避免 m 接近 INT32_MAX 时 (m + divisor - 1) 溢出。
    int32_t rowsPerBin = static_cast<int32_t>((static_cast<int64_t>(m) + divisor - 1) / divisor);
    binEdges[0] = 0;
    for (int32_t b = 0; b < blockDim; b++) {
        // 端点同样用 int64 计算并截断，避免 (b+1)*rowsPerBin 越过 INT32_MAX 回绕。
        binEdges[b + 1] = SpmmArch22ClampBinEnd(static_cast<int64_t>(b) * rowsPerBin, rowsPerBin, m);
    }
}

aclsparseStatus_t SpmmArch22CsrMat::ValidateRowOffsets(const std::vector<uint32_t> &rowOff) const
{
    const uint64_t base = matDesc->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    const uint64_t end = static_cast<uint64_t>(nnz) + base;
    if (end > static_cast<uint64_t>(INT32_MAX) || rowOff.empty() || rowOff.front() != base ||
        rowOff.back() != end) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    for (size_t i = 1; i < rowOff.size(); i++) {
        if (rowOff[i] < rowOff[i - 1] || rowOff[i] > end) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t SpmmArch22CsrMat::ValidateColumnIndices() const
{
    if (nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const int32_t base = matDesc->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    const int64_t upper = cols + base;
    const size_t chunkElems = 1U << 20;
    std::vector<int32_t> hostIdx(std::min(static_cast<size_t>(nnz), chunkElems));
    for (size_t begin = 0; begin < static_cast<size_t>(nnz); begin += hostIdx.size()) {
        const size_t count = std::min(hostIdx.size(), static_cast<size_t>(nnz) - begin);
        const uint8_t *src = static_cast<const uint8_t *>(matDesc->idxs) + begin * sizeof(int32_t);
        const aclError ret = aclrtMemcpy(hostIdx.data(), count * sizeof(int32_t), src,
                                        count * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        for (size_t i = 0; i < count; i++) {
            if (hostIdx[i] < base || static_cast<int64_t>(hostIdx[i]) >= upper) {
                return ACL_SPARSE_STATUS_INVALID_VALUE;
            }
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

void SpmmArch22CsrMat::BuildPreprocessReorder(const std::vector<uint32_t> &rowOff,
    int32_t count, uint32_t rowBlock, bool transpose, bool deterministic,
    int32_t *reorder, int32_t *binEdges)
{
    if (count <= 0 || transpose || deterministic) {
        BuildIdentityReorder(count, reorder, binEdges);
        return;
    }
    std::vector<uint32_t> rowNnz(count);
    for (int32_t i = 0; i < count; i++) {
        rowNnz[i] = rowOff[i + 1] - rowOff[i];
    }
    if (rowBlock >= 2) {
        GreedyBlockBinPack(rowNnz.data(), rowBlock, reorder, binEdges);
    } else {
        GreedyRowBinPack(rowNnz.data(), reorder, binEdges);
    }
}

aclsparseStatus_t SpmmArch22CsrMat::BuildTransposeWorkspace(
    const std::vector<uint32_t> &rowOff, uint8_t *workspace, int64_t rowOffset,
    int64_t sourceRowOffset, int64_t permutationOffset)
{
    const uint32_t base = matDesc->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    std::vector<int32_t> columnIndices(static_cast<size_t>(nnz));
    const size_t indexBytes = static_cast<size_t>(nnz) * sizeof(int32_t);
    if (indexBytes > 0 && aclrtMemcpy(columnIndices.data(), indexBytes, matDesc->idxs,
        indexBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    std::vector<uint32_t> offsets(static_cast<size_t>(cols) + 1, 0);
    for (int32_t column : columnIndices) {
        offsets[static_cast<size_t>(column - static_cast<int32_t>(base)) + 1]++;
    }
    for (size_t column = 1; column < offsets.size(); column++) {
        offsets[column] += offsets[column - 1];
    }
    std::vector<uint32_t> cursor(offsets.begin(), offsets.end() - 1);
    std::vector<uint32_t> sourceRows(static_cast<size_t>(nnz));
    std::vector<uint32_t> permutation(static_cast<size_t>(nnz));
    for (uint32_t row = 0; row < static_cast<uint32_t>(rows); row++) {
        const uint32_t begin = rowOff[row] - base;
        const uint32_t end = rowOff[row + 1] - base;
        for (uint32_t index = begin; index < end; index++) {
            const uint32_t column = static_cast<uint32_t>(columnIndices[index] - static_cast<int32_t>(base));
            const uint32_t destination = cursor[column]++;
            sourceRows[destination] = row;
            permutation[destination] = index;
        }
    }
    const size_t offsetBytes = offsets.size() * sizeof(uint32_t);
    if (aclrtMemcpy(workspace + rowOffset, offsetBytes, offsets.data(), offsetBytes,
        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    if (indexBytes > 0 &&
        (aclrtMemcpy(workspace + sourceRowOffset, indexBytes, sourceRows.data(), indexBytes,
             ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
         aclrtMemcpy(workspace + permutationOffset, indexBytes, permutation.data(), indexBytes,
             ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE)) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t SpmmArch22CsrMat::DoPreProcess(uint8_t *dWorkspace, int64_t reorderOff,
    int64_t binEdgeOff, int64_t transposeRowOff, int64_t transposeSourceRowOff,
    int64_t transposePermutationOff, uint32_t rowBlock, bool transpose, bool deterministic)
{
    // D1 fix: matDesc may be nullptr if the constructor was given a nullptr
    // inner descriptor (see spmm_csr_mat.h); fail gracefully instead of
    // dereferencing it below.
    if (matDesc == nullptr || dWorkspace == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    int32_t m = static_cast<int32_t>(transpose ? cols : rows);
    std::vector<int32_t> reorder(m);
    std::vector<int32_t> binEdges(blockDim + 1);

    const size_t rowCount = static_cast<size_t>(rows) + 1;
    std::vector<uint32_t> rowOff(rowCount);
    const size_t rowBytes = rowCount * sizeof(uint32_t);
    aclError aclRet = aclrtMemcpy(rowOff.data(), rowBytes, matDesc->ptrs, rowBytes,
                                  ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    aclsparseStatus_t st = ValidateRowOffsets(rowOff);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateColumnIndices();
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (transpose) {
        st = BuildTransposeWorkspace(rowOff, dWorkspace, transposeRowOff,
            transposeSourceRowOff, transposePermutationOff);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }
    BuildPreprocessReorder(
        rowOff, m, rowBlock, transpose, deterministic, reorder.data(), binEdges.data());

    if (m > 0) {
        aclRet = aclrtMemcpy(dWorkspace + reorderOff, m * sizeof(int32_t),
                             reorder.data(), m * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    aclRet = aclrtMemcpy(dWorkspace + binEdgeOff, (blockDim + 1) * sizeof(int32_t),
                         binEdges.data(), (blockDim + 1) * sizeof(int32_t),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
