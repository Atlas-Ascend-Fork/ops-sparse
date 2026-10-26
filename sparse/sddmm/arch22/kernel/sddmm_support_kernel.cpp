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
#include "kernel_operator.h"
#include "sddmm_contract_kernel.h"
#include "../host/sddmm_launch.h"

namespace {
template <typename Word, uint32_t Lanes>
__aicore__ inline void InitDensePackMap(AscendC::LocalTensor<int32_t> index, AscendC::LocalTensor<int32_t> map,
                                        AscendC::LocalTensor<int32_t> tmp)
{
    constexpr uint32_t tile = 256;
    AscendC::CreateVecIndex(index, 0, tile * Lanes);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (Lanes == 2) {
        AscendC::ShiftRight(map, index, int32_t(1), tile * Lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tmp, map, int32_t(2), tile * Lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tmp, index, tmp, tile * Lanes);
        AscendC::Muls(map, map, int32_t(32), tile * Lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tmp, tmp, int32_t(sizeof(Word)), tile * Lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(map, map, tmp, tile * Lanes);
    } else {
        AscendC::Muls(map, index, int32_t(32), tile);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename Word, uint32_t Lanes>
__aicore__ inline AscendC::DataCopyExtParams DensePackCopy(uint32_t count, uint32_t colStep)
{
    AscendC::DataCopyExtParams copy{1, count * Lanes * uint32_t(sizeof(Word)), 0, 0, 0};
    if (colStep != 1) {
        copy.blockCount = count;
        copy.blockLen = Lanes * sizeof(Word);
        copy.srcStride = (colStep - 1) * Lanes * sizeof(Word);
    }
    return copy;
}

template <typename Word, uint32_t Lanes>
__global__ __aicore__ void PackDense(GM_ADDR source, GM_ADDR dest, uint32_t rows, uint32_t columns, uint32_t rowStep,
                                     uint32_t colStep)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> input;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> output;
    AscendC::TBuf<AscendC::TPosition::VECCALC> table;
    constexpr uint32_t tile = 256;
    pipe.InitBuffer(input, 1, tile * 32);
    pipe.InitBuffer(output, 1, tile * Lanes * sizeof(Word));
    pipe.InitBuffer(table, tile * Lanes * 3 * 4);
    AscendC::GlobalTensor<Word> src, dst;
    src.SetGlobalBuffer((__gm__ Word*)source);
    dst.SetGlobalBuffer((__gm__ Word*)dest);
    auto index = table.Get<int32_t>();
    auto map = index[tile * Lanes];
    auto tmp = map[tile * Lanes];
    InitDensePackMap<Word, Lanes>(index, map, tmp);
    const uint64_t total = uint64_t(rows) * columns;
    for (uint64_t begin = uint64_t(AscendC::GetBlockIdx()) * tile; begin < total;
         begin += uint64_t(AscendC::GetBlockNum()) * tile) {
        const uint64_t end = begin + tile < total ? begin + tile : total;
        for (uint64_t pos = begin; pos < end;) {
            const uint32_t row = pos / columns, col = pos % columns;
            const uint32_t count = end - pos < columns - col ? end - pos : columns - col;
            auto in = input.AllocTensor<Word>();
            const uint64_t offset = (uint64_t(row) * rowStep + uint64_t(col) * colStep) * Lanes;
            auto copy = DensePackCopy<Word, Lanes>(count, colStep);
            AscendC::DataCopyPad(in, src[offset], copy, {false, 0, 0, Word(0)});
            input.EnQue(in);
            in = input.DeQue<Word>();
            auto out = output.AllocTensor<Word>();
            if (colStep == 1) {
                AscendC::DataCopy(out, in,
                                  (count * Lanes + 31 / sizeof(Word)) / (32 / sizeof(Word)) * (32 / sizeof(Word)));
            } else {
                AscendC::Gather(out, in, map.ReinterpretCast<uint32_t>(), 0, count * Lanes);
            }
            output.EnQue(out);
            out = output.DeQue<Word>();
            AscendC::DataCopyPad(dst[pos * Lanes], out, {1, count * Lanes * uint32_t(sizeof(Word)), 0, 0, 0});
            output.FreeTensor(out);
            input.FreeTensor(in);
            pos += count;
        }
    }
}

// Invalid int64 indices stay negative after narrowing, so pattern validation
// rejects them without an additional host reduction or synchronization.
__global__ __aicore__ void NarrowIndices(GM_ADDR source, GM_ADDR dest, uint32_t count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 6 * kTile * 4);
    auto raw = storage.Get<int32_t>();
    auto low = raw[2 * kTile];
    auto high = low[kTile];
    auto map = high[kTile];
    auto scratch = map[kTile].ReinterpretCast<float>();
    AscendC::GlobalTensor<int32_t> src, dst;
    src.SetGlobalBuffer((__gm__ int32_t*)source);
    dst.SetGlobalBuffer((__gm__ int32_t*)dest);
    AscendC::CreateVecIndex(map, 0, kTile);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(map, map, int32_t(8), kTile);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint64_t begin = uint64_t(AscendC::GetBlockIdx()) * kTile; begin < count;
         begin += uint64_t(AscendC::GetBlockNum()) * kTile) {
        const uint32_t size = count - begin < kTile ? count - begin : kTile;
        Fence<AscendC::HardEvent::V_MTE2>();
        Fence<AscendC::HardEvent::MTE3_V>();
        AscendC::DataCopyPad(raw, src[begin * 2], {1, size * 8, 0, 0, 0}, {false, 0, 0, 0});
        Fence<AscendC::HardEvent::MTE2_V>();
        AscendC::Gather(low, raw, map.ReinterpretCast<uint32_t>(), 0, size);
        AscendC::Gather(high, raw, map.ReinterpretCast<uint32_t>(), 4, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(scratch, high, AscendC::RoundMode::CAST_NONE, size);
        AscendC::PipeBarrier<PIPE_V>();
        Abs(scratch, scratch, size);
        AscendC::PipeBarrier<PIPE_V>();
        Mins(scratch, scratch, 1.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scratch, scratch, -2147483648.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(high, scratch, AscendC::RoundMode::CAST_TRUNC, size);
        AscendC::PipeBarrier<PIPE_V>();
        Or(low.ReinterpretCast<uint16_t>(), low.ReinterpretCast<uint16_t>(), high.ReinterpretCast<uint16_t>(),
           size * 2);
        Fence<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyPad(dst[begin], low, {1, size * 4, 0, 0, 0});
    }
}

template <bool Complex>
__aicore__ inline void InitScaleMap(AscendC::LocalTensor<int32_t> seq, AscendC::LocalTensor<int32_t> map,
                                    AscendC::LocalTensor<float> factor, float bi)
{
    if constexpr (Complex) {
        AscendC::CreateVecIndex(seq, 0, kTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(map, seq, int32_t(1), kTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(map, map, int32_t(2), kTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(seq, seq, map, kTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(factor, seq, AscendC::RoundMode::CAST_NONE, kTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(factor, factor, 2 * bi, kTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(factor, factor, -bi, kTile);
        AscendC::Sub(map, map, seq, kTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(map, map, int32_t(1), kTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(map, map, int32_t(4), kTile);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <bool Complex>
__aicore__ inline void ScaleTile(AscendC::LocalTensor<float> data, AscendC::LocalTensor<float> swapped,
                                 AscendC::LocalTensor<int32_t> map, AscendC::LocalTensor<float> factor, uint32_t size,
                                 float br, float bi)
{
    if constexpr (Complex) {
        if (bi == 0.0f) {
            AscendC::Duplicate(swapped, 0.0f, size);
        } else {
            AscendC::Gather(swapped, data, map.ReinterpretCast<uint32_t>(), 0, size);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(swapped, swapped, factor, size);
        }
        if (br == 0.0f) {
            AscendC::Duplicate(data, 0.0f, size);
        } else {
            AscendC::Muls(data, data, br, size);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(data, data, swapped, size);
    } else {
        AscendC::Muls(data, data, br, size);
    }
}

template <typename T>
__aicore__ inline void CastScaledOutput(AscendC::LocalTensor<T> out, AscendC::LocalTensor<float> data, uint32_t size)
{
    if constexpr (sizeof(T) == 4) {
        AscendC::DataCopy(out, data, (size + 7) / 8 * 8);
    } else {
        AscendC::Cast(out, data, AscendC::RoundMode::CAST_RINT, size);
    }
}

template <typename T, bool Complex = false>
__global__ __aicore__ void ScaleValues(GM_ADDR values, uint32_t count, float br, float bi)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> input;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> output;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(input, 1, kTile * sizeof(T));
    pipe.InitBuffer(output, 1, kTile * sizeof(T));
    pipe.InitBuffer(storage, kTile * 5 * 4);
    auto data = storage.Get<float>();
    auto swapped = data[kTile];
    auto factor = swapped[kTile];
    auto map = factor[kTile].ReinterpretCast<int32_t>();
    auto seq = map[kTile];
    InitScaleMap<Complex>(seq, map, factor, bi);
    AscendC::GlobalTensor<T> gm;
    gm.SetGlobalBuffer((__gm__ T*)values);
    const uint64_t total = uint64_t(count) * (Complex ? 2 : 1);
    for (uint64_t begin = uint64_t(AscendC::GetBlockIdx()) * kTile; begin < total;
         begin += uint64_t(AscendC::GetBlockNum()) * kTile) {
        const uint32_t size = total - begin < kTile ? total - begin : kTile;
        auto out = output.AllocTensor<T>();
        if (br == 0 && bi == 0) {
            AscendC::Duplicate(data, 0.0f, size);
        } else {
            auto in = input.AllocTensor<T>();
            AscendC::DataCopyPad(in, gm[begin], {1, size * uint32_t(sizeof(T)), 0, 0, 0}, {false, 0, 0, T(0)});
            input.EnQue(in);
            in = input.DeQue<T>();
            if constexpr (sizeof(T) == 4) {
                AscendC::DataCopy(data, in, (size + 7) / 8 * 8);
            } else {
                AscendC::Cast(data, in, AscendC::RoundMode::CAST_NONE, size);
            }
            AscendC::PipeBarrier<PIPE_V>();
            ScaleTile<Complex>(data, swapped, map, factor, size, br, bi);
            input.FreeTensor(in);
        }
        AscendC::PipeBarrier<PIPE_V>();
        CastScaledOutput(out, data, size);
        output.EnQue(out);
        out = output.DeQue<T>();
        AscendC::DataCopyPad(gm[begin], out, {1, size * uint32_t(sizeof(T)), 0, 0, 0});
        output.FreeTensor(out);
        Fence<AscendC::HardEvent::MTE3_V>();
    }
}

} // namespace

namespace {
__aicore__ inline void InitBsrExpansionMap(AscendC::LocalTensor<int32_t> index, AscendC::LocalTensor<int32_t> offset,
                                           AscendC::LocalTensor<int32_t> lane, uint32_t bd)
{
    uint32_t shift = 0;
    for (uint32_t d = bd; d > 1; d >>= 1) {
        ++shift;
    }
    AscendC::CreateVecIndex(index, 0, kTile);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(offset, index, int32_t(shift), kTile);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(lane, offset, int32_t(bd), kTile);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(lane, index, lane, kTile);
    AscendC::Muls(offset, offset, int32_t(4), kTile);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ExpandRowTile(AscendC::LocalTensor<int32_t> raw, AscendC::LocalTensor<int32_t> offset,
                                     AscendC::LocalTensor<int32_t> starts, AscendC::LocalTensor<int32_t> ends,
                                     AscendC::LocalTensor<int32_t> lane, AscendC::LocalTensor<int32_t> out,
                                     uint32_t count, uint32_t bd, int32_t base)
{
    AscendC::Gather(starts, raw, offset.ReinterpretCast<uint32_t>(), 0, count);
    AscendC::Gather(ends, raw, offset.ReinterpretCast<uint32_t>(), 4, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(ends, ends, starts, count);
    AscendC::Adds(starts, starts, -base, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(ends, ends, lane, count);
    AscendC::Muls(starts, starts, int32_t(bd * bd), count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ends, ends, int32_t(bd), count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(out, starts, ends, count);
}

__global__ __aicore__ void ExpandBsrRows(GM_ADDR rows, GM_ADDR scalarRows, uint32_t m, uint32_t bd, int32_t base)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 7 * (kTile + 8) * 4);
    auto raw = storage.Get<int32_t>();
    auto index = raw[kTile + 8];
    auto offset = index[kTile + 8];
    auto starts = offset[kTile + 8];
    auto ends = starts[kTile + 8];
    auto lane = ends[kTile + 8];
    auto out = lane[kTile + 8];
    AscendC::GlobalTensor<int32_t> rp, sr;
    rp.SetGlobalBuffer((__gm__ int32_t*)rows);
    sr.SetGlobalBuffer((__gm__ int32_t*)scalarRows);
    const uint32_t safeBd = bd == 0 ? 1u : bd;
    InitBsrExpansionMap(index, offset, lane, safeBd);
    for (uint32_t begin = AscendC::GetBlockIdx() * kTile; begin <= m; begin += AscendC::GetBlockNum() * kTile) {
        const uint32_t count = m - begin < kTile ? m - begin : kTile;
        Fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyPad(raw, rp[begin / safeBd], {1, (count / safeBd + 1) * 4, 0, 0, 0}, {false, 0, 0, 0});
        Fence<AscendC::HardEvent::MTE2_V>();
        if (!count) {
            Fence<AscendC::HardEvent::MTE2_S>();
            out.SetValue(0, (raw.GetValue(0) - base) * bd * bd);
            Fence<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(sr[begin], out, {1, 4, 0, 0, 0});
            break;
        }
        ExpandRowTile(raw, offset, starts, ends, lane, out, count, bd, base);
        const bool tail = count < kTile;
        if (tail) {
            Fence<AscendC::HardEvent::V_S>();
            out.SetValue(count, (raw.GetValue(count / safeBd) - base) * bd * bd);
            Fence<AscendC::HardEvent::S_MTE3>();
        }
        Fence<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyPad(sr[begin], out, {1, (count + uint32_t(tail)) * 4, 0, 0, 0});
        Fence<AscendC::HardEvent::MTE3_V>();
    }
}

__aicore__ inline uint32_t AlignBsrRow(AscendC::GlobalTensor<int32_t> rp, uint32_t row, uint32_t blockRows, uint32_t bd,
                                       int32_t base)
{
    while (row < blockRows && ((rp.GetValue(row) - base) * bd * bd) % 8) {
        ++row;
    }
    return row;
}

__global__ __aicore__ void ExpandBsrColumns(GM_ADDR rows, GM_ADDR columns, GM_ADDR scalarColumns, uint32_t m,
                                            uint32_t bd, int32_t base)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 5 * kTile * 4);
    auto raw = storage.Get<int32_t>();
    auto map = raw[kTile];
    auto lane = map[kTile];
    auto out = lane[kTile];
    auto seq = out[kTile];
    AscendC::GlobalTensor<int32_t> rp, ci, dest;
    rp.SetGlobalBuffer((__gm__ int32_t*)rows);
    ci.SetGlobalBuffer((__gm__ int32_t*)columns);
    dest.SetGlobalBuffer((__gm__ int32_t*)scalarColumns);
    const uint32_t safeBd = bd == 0 ? 1u : bd;
    InitBsrExpansionMap(seq, map, lane, safeBd);
    const uint32_t blockRows = m / safeBd;
    const uint32_t blockCount = AscendC::GetBlockNum();
    const uint32_t cores = blockCount == 0 ? 1u : blockCount;
    const uint32_t per = (blockRows + cores - 1) / cores;
    uint32_t first = AscendC::GetBlockIdx() * per < blockRows ? AscendC::GetBlockIdx() * per : blockRows;
    uint32_t last = first + per < blockRows ? first + per : blockRows;
    first = AlignBsrRow(rp, first, blockRows, bd, base);
    last = AlignBsrRow(rp, last, blockRows, bd, base);
    for (uint32_t row = first; row < last; ++row) {
        const uint32_t start = rp.GetValue(row) - base;
        const uint32_t blocks = rp.GetValue(row + 1) - base - start;
        for (uint32_t off = 0; off < blocks * bd; off += kTile) {
            const uint32_t count = blocks * bd - off < kTile ? blocks * bd - off : kTile;
            Fence<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyPad(raw, ci[start + off / safeBd], {1, count / safeBd * 4, 0, 0, 0}, {false, 0, 0, 0});
            Fence<AscendC::HardEvent::MTE2_V>();
            AscendC::Gather(out, raw, map.ReinterpretCast<uint32_t>(), 0, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(out, out, -base, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(out, out, int32_t(bd), count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(out, out, lane, count);
            Fence<AscendC::HardEvent::V_MTE3>();
            for (uint32_t r = 0; r < bd; ++r) {
                const uint64_t pos = uint64_t(start) * bd * bd + uint64_t(r) * blocks * bd + off;
                AscendC::DataCopyPad(dest[pos], out, {1, count * 4, 0, 0, 0});
            }
            Fence<AscendC::HardEvent::MTE3_V>();
        }
    }
}
} // namespace

extern "C" aclsparseStatus_t sddmm_csr_row_hints_launch(GM_ADDR rows, GM_ADDR columns, GM_ADDR hints, uint32_t rowCount,
                                                        int32_t base, uint32_t outAlign, uint32_t cores, void* stream)
{
    ClassifyCsrRows<<<cores, nullptr, stream>>>(rows, columns, hints, rowCount, base, outAlign);
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t sddmm_pattern_check_launch(GM_ADDR rp, GM_ADDR ci, GM_ADDR sr, GM_ADDR sc, GM_ADDR flags,
                                                        uint32_t rows, uint32_t columns, uint32_t nnz, int32_t base,
                                                        bool compare, uint32_t cores, void* stream)
{
    if (compare) {
        ComparePattern<<<cores, nullptr, stream>>>(rp, ci, sr, sc, flags, rows, nnz);
    } else {
        CheckPattern<<<cores, nullptr, stream>>>(rp, ci, sr, sc, flags, rows, columns, nnz, base, false);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t sddmm_index32_launch(GM_ADDR source, GM_ADDR dest, uint32_t count, uint32_t cores,
                                                  void* stream)
{
    if (count) {
        NarrowIndices<<<cores, nullptr, stream>>>(source, dest, count);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t sddmm_scale_launch(GM_ADDR values, uint32_t count, aclDataType type, float br, float bi,
                                                uint32_t cores, void* stream)
{
    if (type == ACL_COMPLEX64) {
        ScaleValues<float, true><<<cores, nullptr, stream>>>(values, count, br, bi);
    } else if (type == ACL_FLOAT) {
        ScaleValues<float><<<cores, nullptr, stream>>>(values, count, br, bi);
    } else if (type == ACL_FLOAT16) {
        ScaleValues<half><<<cores, nullptr, stream>>>(values, count, br, bi);
    } else {
        ScaleValues<bfloat16_t><<<cores, nullptr, stream>>>(values, count, br, bi);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t sddmm_pack_dense_launch(GM_ADDR src, GM_ADDR dst, uint32_t rows, uint32_t cols,
                                                     uint32_t rowStep, uint32_t colStep, aclDataType type,
                                                     uint32_t cores, void* stream)
{
    if (type == ACL_COMPLEX64) {
        PackDense<uint32_t, 2><<<cores, nullptr, stream>>>(src, dst, rows, cols, rowStep, colStep);
    } else if (type == ACL_FLOAT) {
        PackDense<uint32_t, 1><<<cores, nullptr, stream>>>(src, dst, rows, cols, rowStep, colStep);
    } else {
        PackDense<uint16_t, 1><<<cores, nullptr, stream>>>(src, dst, rows, cols, rowStep, colStep);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t sddmm_bsr_expand_launch(GM_ADDR rp, GM_ADDR ci, GM_ADDR sr, GM_ADDR sc, uint32_t m,
                                                     uint32_t blocks, uint32_t bd, int32_t base, uint32_t cores,
                                                     void* stream)
{
    ExpandBsrRows<<<cores, nullptr, stream>>>(rp, sr, m, bd, base);
    if (blocks) {
        ExpandBsrColumns<<<cores, nullptr, stream>>>(rp, ci, sc, m, bd, base);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
