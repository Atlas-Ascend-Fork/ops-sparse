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
#ifndef SDDMM_TRANSFORM_KERNEL_H_
#define SDDMM_TRANSFORM_KERNEL_H_

#include "kernel_operator.h"
#include "../sddmm_tiling_fields.h"

namespace {
constexpr uint32_t kTile = SddmmBudget::kPatternChunk;

template <AscendC::HardEvent Event>
__aicore__ inline void Fence()
{
    const auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    AscendC::SetFlag<Event>(id);
    AscendC::WaitFlag<Event>(id);
}

} // namespace

template <typename Word, uint32_t WordsPerValue>
class SddmmBsrRepack {
public:
    __aicore__ inline void Init(GM_ADDR blockValues, GM_ADDR scalarValues, GM_ADDR rows, uint32_t blockRows,
                                uint32_t blockNnz, uint32_t blockDim, int32_t base, bool columnMajor, bool unpack,
                                AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;
        rowCount = blockRows;
        nnzb = blockNnz;
        bd = blockDim;
        indexBase = base;
        column = columnMajor;
        reverse = unpack;
        side = bd < 32u ? bd : 32u;
        group = bd > 32u ? 1u : 4096u / (side * side * WordsPerValue);
        if (group > 32u) {
            group = 32u;
        }
        words = group * side * side * WordsPerValue;
        rowWords = group * side * WordsPerValue;
        blockWords = side * side * WordsPerValue;
        blockGm.SetGlobalBuffer((__gm__ Word*)blockValues, static_cast<uint64_t>(nnzb) * bd * bd * WordsPerValue);
        scalarGm.SetGlobalBuffer((__gm__ Word*)scalarValues, static_cast<uint64_t>(nnzb) * bd * bd * WordsPerValue);
        rowGm.SetGlobalBuffer((__gm__ int32_t*)rows, blockRows + 1u);
        pipe->InitBuffer(input, 1, words * sizeof(Word));
        pipe->InitBuffer(output, 1, words * sizeof(Word));
        pipe->InitBuffer(table, 5u * words * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (nnzb == 0) {
            return;
        }
        BuildMap();
        const uint32_t blockCount = AscendC::GetBlockNum();
        const uint32_t cores = blockCount == 0 ? 1u : blockCount;
        const uint32_t per = (nnzb + cores - 1u) / cores;
        uint32_t first = AscendC::GetBlockIdx() * per;
        uint32_t last = first + per < nnzb ? first + per : nnzb;
        // Narrow scalar rows may share a 32B store unit. Assign complete block
        // rows, joining adjacent rows until both layouts have an aligned edge.
        if (bd * WordsPerValue * sizeof(Word) < 32u) {
            const uint32_t rowPer = (rowCount + cores - 1u) / cores;
            uint32_t rowFirst = AscendC::GetBlockIdx() * rowPer;
            uint32_t rowLast = rowFirst + rowPer;
            rowFirst = AlignedRow(rowFirst < rowCount ? rowFirst : rowCount);
            rowLast = AlignedRow(rowLast < rowCount ? rowLast : rowCount);
            first = rowGm.GetValue(rowFirst) - indexBase;
            last = rowGm.GetValue(rowLast) - indexBase;
        }
        if (first >= last) {
            return;
        }
        uint32_t lo = 0, hi = rowCount;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2u;
            if (rowGm.GetValue(mid + 1u) - indexBase <= static_cast<int32_t>(first)) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        for (uint32_t row = lo; row < rowCount && first < last; ++row) {
            const uint32_t begin = rowGm.GetValue(row) - indexBase;
            const uint32_t end = rowGm.GetValue(row + 1u) - indexBase;
            const uint32_t limit = end < last ? end : last;
            for (; first < limit;) {
                const uint32_t count = limit - first < group ? limit - first : group;
                for (uint32_t r = 0; r < bd; r += side) {
                    for (uint32_t c = 0; c < bd; c += side) {
                        Repack(first, begin, end - begin, count, r, c);
                    }
                }
                first += count;
            }
        }
    }

private:
    __aicore__ inline uint32_t AlignedRow(uint32_t row)
    {
        const uint32_t bytes = bd * bd * WordsPerValue * sizeof(Word);
        for (; row < rowCount; ++row) {
            if ((static_cast<uint64_t>(rowGm.GetValue(row) - indexBase) * bytes) % 32u == 0u) {
                break;
            }
        }
        return row;
    }

    __aicore__ inline uint32_t Log2(uint32_t n)
    {
        uint32_t result = 0;
        for (; n > 1u; n >>= 1u) {
            ++result;
        }
        return result;
    }

    __aicore__ inline void Split(AscendC::LocalTensor<int32_t> value, AscendC::LocalTensor<int32_t> quotient,
                                 AscendC::LocalTensor<int32_t> remainder, AscendC::LocalTensor<int32_t> scratch,
                                 uint32_t divisor)
    {
        AscendC::ShiftRight(quotient, value, static_cast<int32_t>(Log2(divisor)), words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scratch, quotient, static_cast<int32_t>(divisor), words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(remainder, value, scratch, words);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddScaled(AscendC::LocalTensor<int32_t> dst, AscendC::LocalTensor<int32_t> value,
                                     AscendC::LocalTensor<int32_t> scratch, uint32_t factor)
    {
        AscendC::Muls(scratch, value, static_cast<int32_t>(factor), words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(dst, dst, scratch, words);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildMap()
    {
        auto map = table.Get<int32_t>();
        auto position = map[words];
        auto rem = map[2u * words];
        auto quotient = map[3u * words];
        auto scratch = map[4u * words];
        AscendC::CreateVecIndex(position, 0, words);
        AscendC::PipeBarrier<PIPE_V>();
        if (!reverse) {
            Split(position, quotient, rem, scratch, rowWords);
            AscendC::Muls(map, quotient, static_cast<int32_t>(column ? WordsPerValue : side * WordsPerValue), words);
            AscendC::PipeBarrier<PIPE_V>();
            Split(rem, quotient, rem, scratch, side * WordsPerValue);
            AddScaled(map, quotient, scratch, blockWords);
            if (column) {
                Split(rem, quotient, rem, scratch, WordsPerValue);
                AddScaled(map, quotient, scratch, side * WordsPerValue);
            }
        } else {
            Split(position, quotient, rem, scratch, blockWords);
            AscendC::Muls(map, quotient, static_cast<int32_t>(side * WordsPerValue), words);
            AscendC::PipeBarrier<PIPE_V>();
            Split(rem, quotient, rem, scratch, side * WordsPerValue);
            AddScaled(map, quotient, scratch, column ? WordsPerValue : rowWords);
            if (column) {
                Split(rem, quotient, rem, scratch, WordsPerValue);
                AddScaled(map, quotient, scratch, rowWords);
            }
        }
        AscendC::Add(map, map, rem, words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(map, map, static_cast<int32_t>(sizeof(Word)), words);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Repack(uint32_t block, uint32_t rowBegin, uint32_t rowBlocks, uint32_t count, uint32_t r,
                                  uint32_t c)
    {
        constexpr uint32_t valueBytes = WordsPerValue * sizeof(Word);
        const uint32_t validWidth = count * side * valueBytes;
        const uint32_t paddedWidth = (validWidth + 31u) / 32u * 32u;
        const uint32_t localStride = (rowWords * sizeof(Word) - paddedWidth) / 32u;
        const uint64_t blockOffset = (static_cast<uint64_t>(block) * bd * bd + (column ? c * bd + r : r * bd + c)) *
                                     WordsPerValue;
        const uint64_t scalarOffset = (static_cast<uint64_t>(rowBegin) * bd * bd +
                                       static_cast<uint64_t>(r) * rowBlocks * bd + (block - rowBegin) * bd + c) *
                                      WordsPerValue;
        auto src = input.AllocTensor<Word>();
        AscendC::DataCopyPadExtParams<Word> pad{false, 0, 0, static_cast<Word>(0)};
        AscendC::DataCopyExtParams blockCopy{1, count * blockWords * static_cast<uint32_t>(sizeof(Word)), 0, 0, 0};
        if (bd > side) {
            blockCopy.blockCount = side;
            blockCopy.blockLen = side * valueBytes;
            blockCopy.srcStride = (bd - side) * valueBytes;
        }
        if (!reverse) {
            AscendC::DataCopyPad(src, blockGm[blockOffset], blockCopy, pad);
        } else {
            AscendC::DataCopyExtParams cp{static_cast<uint16_t>(side), validWidth,
                                          (rowBlocks * bd - count * side) * valueBytes, localStride, 0};
            AscendC::DataCopyPad(src, scalarGm[scalarOffset], cp, pad);
        }
        input.EnQue(src);
        src = input.DeQue<Word>();
        auto dst = output.AllocTensor<Word>();
        auto offsets = table.Get<int32_t>().ReinterpretCast<uint32_t>();
        AscendC::Gather(dst, src, offsets, 0, reverse ? count * blockWords : words);
        output.EnQue(dst);
        dst = output.DeQue<Word>();
        if (!reverse) {
            AscendC::DataCopyExtParams cp{static_cast<uint16_t>(side), validWidth, localStride,
                                          (rowBlocks * bd - count * side) * valueBytes, 0};
            AscendC::DataCopyPad(scalarGm[scalarOffset], dst, cp);
        } else {
            blockCopy.dstStride = blockCopy.srcStride;
            blockCopy.srcStride = 0;
            AscendC::DataCopyPad(blockGm[blockOffset], dst, blockCopy);
        }
        input.FreeTensor(src);
        output.FreeTensor(dst);
    }

    AscendC::TPipe* pipe = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> input;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> output;
    AscendC::TBuf<AscendC::TPosition::VECCALC> table;
    AscendC::GlobalTensor<Word> blockGm, scalarGm;
    AscendC::GlobalTensor<int32_t> rowGm;
    uint32_t rowCount = 0, nnzb = 0, bd = 0, side = 0, group = 0, words = 0, rowWords = 0, blockWords = 0;
    int32_t indexBase = 0;
    bool column = false, reverse = false;
};

namespace {
// 转置按 kYtTile x kYtTile 的方 tile 走：tile 内用一次 Gather 完成转置，源和目的
// 都是 kYtTile 个 32B 块，两边的 DMA 分块都对齐。
constexpr uint32_t kYtTile = 64;
constexpr uint32_t kYtElems = kYtTile * kYtTile;

__aicore__ inline uint32_t YtMinU32(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

// 把 [begin, end) 这段 tile 下标均分给 blockDim 个核。
__aicore__ inline void YtSlice(uint32_t count, uint32_t& first, uint32_t& last)
{
    const uint32_t blockCount = AscendC::GetBlockNum();
    const uint32_t blocks = blockCount == 0 ? 1u : blockCount;
    const uint32_t per = (count + blocks - 1) / blocks;
    first = AscendC::GetBlockIdx() * per;
    last = YtMinU32(first + per, count);
}
} // namespace

// Yt[j * pitch + k] = Y[k * ldy + j]
//
// 为什么需要它：SDDMM 的每一条点积都要把 B 的一整列（K 个元素）搬到 UB。Y 是行主
// 序时，一列的元素彼此相隔 ldy 个元素，逐元素搬一个 32B 块只装得下 2~4 字节有用
// 数据 —— 带宽浪费 8~16 倍，这是标量路径慢的根因。转置之后一列变成连续的 K 个元
// 素，一次多块 DMA 就整行搬进来了。
//
// 转置本身按 64x64 的方 tile 做：
//   读：blockCount = tile 内的行数，blockLen = 64 个元素，srcStride = 行间空隙 ；
//   转：每条 Gather 收一列（64 项偏移表跨 64 行，srcBaseAddr 前移 c 个元素）；
//   写：blockCount = 64，blockLen = 64 个元素，dstStride = (pitch - 64) * sizeof(T)。
// tile 内的行主序下标是 r * 64 + c，第 c 列的元素间隔 64，所以一张「步长 64」的
// 64 项偏移表就能喂给所有列，cols 那一维只体现在 srcBaseAddr 上。
//
// pitch 由 host 定成 AlignUp(K, 64)，因此 Yt 的每一行都补到 64 的整数倍：行首
// 32B 对齐、tile 在 k 方向正好整除、多出来的尾巴是垃圾但内核永远不会读到。
template <typename XType>
class SddmmYTranspose {
public:
    __aicore__ inline SddmmYTranspose() {}

    __aicore__ inline void Init(GM_ADDR y, GM_ADDR yt, uint32_t rows, uint32_t cols, uint32_t yLd, uint32_t tPitch,
                                AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;
        kRows = rows;
        colCount = cols;
        ldy = yLd;
        pitch = tPitch;
        const uint64_t ySize = static_cast<uint64_t>(rows) * ((yLd > cols) ? yLd : cols);
        yGm.SetGlobalBuffer((__gm__ XType*)y, ySize);
        ytGm.SetGlobalBuffer((__gm__ XType*)yt, static_cast<uint64_t>(cols) * tPitch);
        pipe->InitBuffer(inQueue, 1, kYtElems * sizeof(XType));
        pipe->InitBuffer(outQueue, 1, kYtElems * sizeof(XType));
        pipe->InitBuffer(offBuf, kYtTile * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (kRows == 0 || colCount == 0) {
            return;
        }
        // host 只在 colCount >= kYtTile 时走这条路；tile 的最后一列往回重叠，
        // 保证读不会越过行尾。
        if (colCount < kYtTile || ldy < kYtTile) {
            return;
        }
        BuildOffsets();
        const uint32_t kTiles = (kRows + kYtTile - 1u) / kYtTile;
        const uint32_t cTiles = (colCount + kYtTile - 1u) / kYtTile;
        uint32_t first = 0;
        uint32_t last = 0;
        YtSlice(kTiles * cTiles, first, last);
        for (uint32_t t = first; t < last; ++t) {
            // 相邻 tile 沿列方向连续：读顺着 Y 的行走，写落在 Yt 里一段连续的列上。
            DoTile(t / cTiles, t % cTiles);
        }
    }

private:
    // 偏移表只有 kYtTile 项：第 i 项是 tile 内「行首」的字节偏移 i * kYtTile * sizeof(T)，
    // 也就是元素步长 64（跨行）。取第 c 列只要把 srcBaseAddr 往前挪 c 个元素。
    //
    // 为什么是「64 项 + count=64」而不是「4096 项 + count=512」：Gather 的 count 形式
    // 内部按每次迭代 256 字节（8 个 DataBlock）切分，只有**单次迭代**（count == 单次
    // 迭代元素数，fp32 为 64、fp16 为 128）这条路径在本平台是可靠的 —— 官方样例、
    // docs 里的高维样例，以及本仓库里唯一被验证过的调用（GatherChunk 的 cnt<=64、
    // 偏移表恒从索引 0 起）都只用单次迭代。传 512 时只有第一次迭代写对了目的位置。
    // 所以这里刻意让每次调用都落在单次迭代上：count 恒为 kYtTile，偏移表恒从 0 起。
    __aicore__ inline void BuildOffsets()
    {
        AscendC::LocalTensor<int32_t> offs = offBuf.Get<int32_t>();
        AscendC::CreateVecIndex<int32_t>(offs, 0, static_cast<int32_t>(kYtTile));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(offs, offs, static_cast<int32_t>(kYtTile * sizeof(XType)), static_cast<int32_t>(kYtTile));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DoTile(uint32_t kTile, uint32_t cTile)
    {
        const uint32_t k0 = kTile * kYtTile;
        const uint32_t ktr = YtMinU32(kRows - k0, kYtTile);
        uint32_t c0 = cTile * kYtTile;
        if (c0 + kYtTile > colCount) {
            // 最后一列 tile 往回重叠。重叠的那几列被两个 tile 写同样的值，幂等。
            c0 = colCount - kYtTile;
        }

        AscendC::LocalTensor<XType> raw = inQueue.AllocTensor<XType>();
        AscendC::DataCopyExtParams rp{static_cast<uint16_t>(ktr), kYtTile * static_cast<uint32_t>(sizeof(XType)),
                                      (ldy - kYtTile) * static_cast<uint32_t>(sizeof(XType)), 0, 0};
        AscendC::DataCopyPadExtParams<XType> pd{false, 0, 0, static_cast<XType>(0)};
        AscendC::DataCopyPad(raw, yGm[static_cast<uint64_t>(k0) * ldy + c0], rp, pd);
        inQueue.EnQue(raw);
        raw = inQueue.DeQue<XType>();

        AscendC::LocalTensor<XType> out = outQueue.AllocTensor<XType>();
        AscendC::LocalTensor<uint32_t> offs = offBuf.Get<int32_t>().ReinterpretCast<uint32_t>();
        // 一列一条 Gather：源基址前进 c 个元素，偏移表跨 64 行，结果正好是 tile 内
        // 第 c 列在 out 里的一段连续 kYtTile 个元素（也就是 Yt 的一行）。
        for (uint32_t c = 0; c < kYtTile; ++c) {
            AscendC::Gather<XType>(out[c * kYtTile], raw, offs, c * static_cast<uint32_t>(sizeof(XType)), kYtTile);
            AscendC::PipeBarrier<PIPE_V>();
        }
        inQueue.FreeTensor(raw);

        outQueue.EnQue(out);
        out = outQueue.DeQue<XType>();
        // dstStride 是「前一个数据块的结束地址与后一个数据块的起始地址之差」，
        // 也就是**间隙**而不是行距：每块长 kYtTile 个元素，所以要减掉块长。
        AscendC::DataCopyExtParams wp{static_cast<uint16_t>(kYtTile), kYtTile * static_cast<uint32_t>(sizeof(XType)), 0,
                                      (pitch - kYtTile) * static_cast<uint32_t>(sizeof(XType)), 0};
        AscendC::DataCopyPad(ytGm[static_cast<uint64_t>(c0) * pitch + k0], out, wp);
        outQueue.FreeTensor(out);
    }

    AscendC::TPipe* pipe = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offBuf;
    AscendC::GlobalTensor<XType> yGm;
    AscendC::GlobalTensor<XType> ytGm;
    uint32_t kRows = 0;
    uint32_t colCount = 0;
    uint32_t ldy = 0;
    uint32_t pitch = 0;
};

#endif
