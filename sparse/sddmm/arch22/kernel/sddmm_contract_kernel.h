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
#ifndef SDDMM_CONTRACT_KERNEL_H_
#define SDDMM_CONTRACT_KERNEL_H_

#include "sddmm_transform_kernel.h"

namespace {
__aicore__ inline bool HasNonzeroDifference(AscendC::LocalTensor<int32_t> diff, AscendC::LocalTensor<float> errors,
                                            AscendC::LocalTensor<float> reduced, uint32_t count)
{
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(errors, diff, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    Abs(errors, errors, count);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceMax(reduced, errors, diff.ReinterpretCast<float>(), count, false);
    Fence<AscendC::HardEvent::V_S>();
    return reduced.GetValue(0) != 0.0f;
}

__aicore__ inline bool ValidatePatternTile(AscendC::LocalTensor<int32_t> input, AscendC::LocalTensor<int32_t> saved,
                                           AscendC::LocalTensor<int32_t> diff, AscendC::LocalTensor<int32_t> offsets,
                                           AscendC::LocalTensor<float> errors, AscendC::LocalTensor<float> reduced,
                                           uint32_t count, uint32_t extra, bool row, bool compare, uint32_t nnz,
                                           uint32_t columns, int32_t base)
{
    bool failed = false;
    if (compare) {
        AscendC::Sub(diff, input, saved, count);
    } else {
        Maxs(diff, input, base, count);
        AscendC::PipeBarrier<PIPE_V>();
        Mins(diff, diff, int32_t(row ? nnz + base : columns - 1 + base), count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(diff, input, diff, count);
    }
    failed |= HasNonzeroDifference(diff, errors, reduced, count);
    if (!compare && row) {
        const uint32_t pairs = count + extra - 1;
        if (pairs) {
            AscendC::Gather(saved, input, offsets.ReinterpretCast<uint32_t>(), 0, pairs);
            AscendC::PipeBarrier<PIPE_V>();
            Max(diff, input, saved, pairs);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(diff, diff, saved, pairs);
            failed |= HasNonzeroDifference(diff, errors, reduced, pairs);
        }
    }
    return failed;
}

__aicore__ inline void StorePatternValidation(bool failed, AscendC::GlobalTensor<int32_t> rowGm,
                                              AscendC::GlobalTensor<int32_t> result, AscendC::LocalTensor<int32_t> diff,
                                              uint32_t rows, uint32_t nnz, int32_t base)
{
    if (AscendC::GetBlockIdx() == 0) {
        failed |= rowGm.GetValue(0) != base || rowGm.GetValue(rows) != int32_t(nnz + base);
    }
    Fence<AscendC::HardEvent::MTE3_V>();
    AscendC::Duplicate(diff, int32_t(failed), 8);
    Fence<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopy(result[AscendC::GetBlockIdx() * 8], diff, 8);
}

__aicore__ inline void InitPatternOffsets(AscendC::LocalTensor<int32_t> offsets)
{
    AscendC::CreateVecIndex(offsets, 1, kTile);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(offsets, offsets, int32_t(4), kTile);
    AscendC::PipeBarrier<PIPE_V>();
}

__global__ __aicore__ void CheckPattern(GM_ADDR rp, GM_ADDR ci, GM_ADDR sr, GM_ADDR sc, GM_ADDR flags, uint32_t rows,
                                        uint32_t columns, uint32_t nnz, int32_t base, bool compare)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, (kTile + 8) * 6 * 4);
    auto input = storage.Get<int32_t>();
    auto saved = input[kTile + 8];
    auto diff = input[2 * (kTile + 8)];
    auto offsets = input[3 * (kTile + 8)];
    auto errors = input[4 * (kTile + 8)].ReinterpretCast<float>();
    auto reduced = input[5 * (kTile + 8)].ReinterpretCast<float>();
    AscendC::GlobalTensor<int32_t> rowGm, colGm, rowSaved, colSaved, result;
    rowGm.SetGlobalBuffer((__gm__ int32_t*)rp);
    colGm.SetGlobalBuffer((__gm__ int32_t*)ci);
    rowSaved.SetGlobalBuffer((__gm__ int32_t*)sr);
    colSaved.SetGlobalBuffer((__gm__ int32_t*)sc);
    result.SetGlobalBuffer((__gm__ int32_t*)flags);
    const uint32_t rowTasks = (rows + 1 + kTile - 1) / kTile;
    const uint32_t colTasks = (nnz + kTile - 1) / kTile;
    bool failed = false;
    InitPatternOffsets(offsets);
    for (uint32_t task = AscendC::GetBlockIdx(); task < rowTasks + colTasks; task += AscendC::GetBlockNum()) {
        const bool row = task < rowTasks;
        const uint32_t begin = (row ? task : task - rowTasks) * kTile;
        const uint32_t total = row ? rows + 1 : nnz;
        const uint32_t count = total - begin < kTile ? total - begin : kTile;
        const uint32_t extra = row && begin + count < total ? 1 : 0;
        auto source = row ? rowGm : colGm;
        auto snapshot = row ? rowSaved : colSaved;
        Fence<AscendC::HardEvent::V_MTE2>();
        Fence<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::DataCopyPad(input, source[begin], {1, (count + extra) * 4, 0, 0, 0}, {false, 0, 0, 0});
        if (compare) {
            AscendC::DataCopyPad(saved, snapshot[begin], {1, count * 4, 0, 0, 0}, {false, 0, 0, 0});
        }
        Fence<AscendC::HardEvent::MTE2_V>();
        failed |= ValidatePatternTile(input, saved, diff, offsets, errors, reduced, count, extra, row, compare, nnz,
                                      columns, base);
        if (!compare) {
            Fence<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(snapshot[begin], input, {1, count * 4, 0, 0, 0});
        }
    }
    StorePatternValidation(failed, rowGm, result, diff, rows, nnz, base);
}

// Compare every int32 word with the validated snapshot. Bitwise OR cannot
// cancel nonzero differences, unlike a checksum, and only the final reduction
// crosses from the vector pipe to the scalar pipe.
__aicore__ inline void StorePatternComparison(AscendC::LocalTensor<int32_t> input,
                                              AscendC::LocalTensor<int32_t> accumulated,
                                              AscendC::LocalTensor<int32_t> diff, AscendC::LocalTensor<float> errors,
                                              AscendC::LocalTensor<float> reduced,
                                              AscendC::GlobalTensor<int32_t> result)
{
    constexpr uint32_t tile = 2048;
    AscendC::Cast(errors, accumulated, AscendC::RoundMode::CAST_NONE, tile);
    AscendC::PipeBarrier<PIPE_V>();
    Abs(errors, errors, tile);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceMax(reduced, errors, diff.ReinterpretCast<float>(), tile, false);
    Fence<AscendC::HardEvent::V_S>();
    const int32_t failed = reduced.GetValue(0) != 0.0f;
    AscendC::Duplicate(input, failed, 8);
    Fence<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopy(result[AscendC::GetBlockIdx() * 8], input, 8);
}

__global__ __aicore__ void ComparePattern(GM_ADDR rp, GM_ADDR ci, GM_ADDR sr, GM_ADDR sc, GM_ADDR flags, uint32_t rows,
                                          uint32_t nnz)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t tile = 2048;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, (5 * tile + 8) * 4);
    auto input = storage.Get<int32_t>();
    auto saved = input[tile];
    auto diff = input[2 * tile];
    auto accumulated = input[3 * tile];
    auto errors = input[4 * tile].ReinterpretCast<float>();
    auto reduced = input[5 * tile].ReinterpretCast<float>();
    AscendC::GlobalTensor<int32_t> rowGm, colGm, rowSaved, colSaved, result;
    rowGm.SetGlobalBuffer((__gm__ int32_t*)rp);
    colGm.SetGlobalBuffer((__gm__ int32_t*)ci);
    rowSaved.SetGlobalBuffer((__gm__ int32_t*)sr);
    colSaved.SetGlobalBuffer((__gm__ int32_t*)sc);
    result.SetGlobalBuffer((__gm__ int32_t*)flags);
    const uint32_t rowTasks = (rows + 1 + tile - 1) / tile;
    const uint32_t colTasks = (nnz + tile - 1) / tile;
    AscendC::Duplicate(accumulated, int32_t(0), tile);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t task = AscendC::GetBlockIdx(); task < rowTasks + colTasks; task += AscendC::GetBlockNum()) {
        const bool row = task < rowTasks;
        const uint32_t begin = (row ? task : task - rowTasks) * tile;
        const uint32_t total = row ? rows + 1 : nnz;
        const uint32_t count = total - begin < tile ? total - begin : tile;
        auto source = row ? rowGm : colGm;
        auto snapshot = row ? rowSaved : colSaved;
        Fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyPad(input, source[begin], {1, count * 4, 0, 0, 0}, {false, 0, 0, 0});
        AscendC::DataCopyPad(saved, snapshot[begin], {1, count * 4, 0, 0, 0}, {false, 0, 0, 0});
        Fence<AscendC::HardEvent::MTE2_V>();
        AscendC::Sub(diff, input, saved, count);
        AscendC::PipeBarrier<PIPE_V>();
        Or(accumulated.ReinterpretCast<uint16_t>(), accumulated.ReinterpretCast<uint16_t>(),
           diff.ReinterpretCast<uint16_t>(), count * 2);
        AscendC::PipeBarrier<PIPE_V>();
    }
    StorePatternComparison(input, accumulated, diff, errors, reduced, result);
}

__aicore__ inline uint32_t FindNnzRow(AscendC::GlobalTensor<int32_t>& rows, uint32_t rowCount, uint32_t offset,
                                      int32_t base)
{
    uint32_t lo = 0, hi = rowCount;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (static_cast<uint32_t>(rows.GetValue(mid + 1) - base) <= offset) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__aicore__ inline uint32_t PreferRowBoundary(AscendC::GlobalTensor<int32_t>& rows, uint32_t rowCount, uint32_t offset,
                                             uint32_t nnz, int32_t base, uint32_t outAlign)
{
    if (offset == 0 || offset >= nnz) {
        return offset;
    }
    const uint32_t row = FindNnzRow(rows, rowCount, offset, base);
    const uint32_t start = static_cast<uint32_t>(rows.GetValue(row) - base);
    // Keep whole short rows together, without sacrificing parallelism for long
    // or skewed rows. This monotone mapping preserves contiguous core ownership.
    const uint32_t safeOutAlign = outAlign == 0 ? 1u : outAlign;
    return offset - start <= 128 && start % safeOutAlign == 0 ? start : offset;
}

__aicore__ inline void StorePartitionBounds(AscendC::GlobalTensor<int32_t> rowGm, AscendC::GlobalTensor<int32_t> hintGm,
                                            AscendC::LocalTensor<int32_t> hints, uint32_t rows, int32_t base,
                                            uint32_t outAlign)
{
    // Numeric execution partitions whole output DMA blocks. These searches
    // depend only on the validated CSR pattern, not on X, Y, alpha or beta.
    const uint32_t nnz = static_cast<uint32_t>(rowGm.GetValue(rows) - base);
    const uint32_t safeOutAlign = outAlign == 0 ? 1u : outAlign;
    const uint32_t units = (nnz + safeOutAlign - 1) / safeOutAlign;
    const uint32_t blockCount = AscendC::GetBlockNum();
    const uint32_t cores = blockCount == 0 ? 1u : blockCount;
    const uint32_t per = units / cores, rem = units % cores;
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t begin = (core * per + (core < rem ? core : rem)) * safeOutAlign;
    const uint32_t end = begin + (per + (core < rem ? 1u : 0u)) * safeOutAlign;
    const uint32_t first = PreferRowBoundary(rowGm, rows, begin, nnz, base, safeOutAlign);
    const uint32_t last = PreferRowBoundary(rowGm, rows, end < nnz ? end : nnz, nnz, base, safeOutAlign);
    Fence<AscendC::HardEvent::MTE3_S>();
    for (uint32_t i = 0; i < 8; ++i) {
        hints.SetValue(i, 0);
    }
    if (first < last) {
        hints.SetValue(0, FindNnzRow(rowGm, rows, first, base));
        hints.SetValue(1, FindNnzRow(rowGm, rows, last - 1, base));
    }
    hints.SetValue(2, first);
    hints.SetValue(3, last);
    Fence<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopy(hintGm[(rows + 7) / 8 * 8 + core * 8], hints, 8);
}

__aicore__ inline int32_t ClassifyCsrRow(AscendC::GlobalTensor<int32_t> colGm, AscendC::LocalTensor<int32_t> raw,
                                         AscendC::LocalTensor<int32_t> sequence, AscendC::LocalTensor<int32_t> diff,
                                         AscendC::LocalTensor<float> errors, AscendC::LocalTensor<float> reduced,
                                         uint32_t begin, uint32_t length, int32_t base)
{
    Fence<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopyPad(raw, colGm[begin], {1, length * 4, 0, 0, 0}, {false, 0, 0, 0});
    Fence<AscendC::HardEvent::MTE2_S>();
    const int32_t column = raw.GetValue(0);
    AscendC::CreateVecIndex(sequence, column, length);
    Fence<AscendC::HardEvent::MTE2_V>();
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(diff, raw, sequence, length);
    if (!HasNonzeroDifference(diff, errors, reduced, length)) {
        return column - base;
    }
    return -1;
}

__global__ __aicore__ void ClassifyCsrRows(GM_ADDR rp, GM_ADDR ci, GM_ADDR output, uint32_t rows, int32_t base,
                                           uint32_t outAlign)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 5 * 128 * 4 + 32);
    auto raw = storage.Get<int32_t>();
    auto sequence = raw[128];
    auto diff = raw[256];
    auto errors = raw[384].ReinterpretCast<float>();
    auto reduced = raw[512].ReinterpretCast<float>();
    auto hints = raw[640];
    AscendC::GlobalTensor<int32_t> rowGm, colGm, hintGm;
    rowGm.SetGlobalBuffer((__gm__ int32_t*)rp);
    colGm.SetGlobalBuffer((__gm__ int32_t*)ci);
    hintGm.SetGlobalBuffer((__gm__ int32_t*)output);
    for (uint32_t first = AscendC::GetBlockIdx() * 8; first < rows; first += AscendC::GetBlockNum() * 8) {
        const uint32_t count = rows - first < 8 ? rows - first : 8;
        Fence<AscendC::HardEvent::MTE3_S>();
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t begin = rowGm.GetValue(first + i) - base;
            const uint32_t length = rowGm.GetValue(first + i + 1) - base - begin;
            int32_t hint = -1;
            if (length && length <= 128 && length % 8 == 0) {
                hint = ClassifyCsrRow(colGm, raw, sequence, diff, errors, reduced, begin, length, base);
            }
            hints.SetValue(i, hint);
        }
        Fence<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(hintGm[first], hints, {1, count * 4, 0, 0, 0});
    }
    StorePartitionBounds(rowGm, hintGm, hints, rows, base, outAlign);
}

} // namespace

#endif
