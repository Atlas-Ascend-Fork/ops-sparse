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
#ifndef SDDMM_ARCH22_COMPLEX_KERNEL_H_
#define SDDMM_ARCH22_COMPLEX_KERNEL_H_

#include "kernel_operator.h"
#include "../sddmm_tiling_fields.h"

namespace {
// 每次处理的复数元素个数（拆 X 用）。X 是 m x k，比 Y 小得多，这个值只要让
// gather 的下标表短一些就够了。
constexpr uint32_t kCplxSplitChunk = SddmmBudget::kComplexSplitChunk;
// 每次处理的 CSR 条目数（展开索引用）。交错 gather 要跨 2 * kCplxIdxChunk 个
// 元素，所以这里留得比较保守。
constexpr uint32_t kCplxIdxChunk = SddmmBudget::kComplexIndexChunk;
// 成对索引表的长度。CplxBuildPairIndex 为**每个**元素建一张长度 2*n 的表
// （元素 i 的表项在 i 与 i^1 之间跳），所以表的步长是 2 * kCplxIdxChunk 而不是
// kCplxIdxChunk —— 缓冲区必须按这个步长乘 3 来开。
constexpr uint32_t kCplxPairTable = 2u * kCplxIdxChunk;
// 每次处理的 fp32 槽位数（合成用）。必须是偶数：奇偶表假定每个分块都从
// (实部, 虚部) 这对的边界开始。
constexpr uint32_t kCplxDotChunk = SddmmBudget::kComplexOutputChunk;

__aicore__ inline uint32_t CplxMinU32(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

// 把 [begin, end) 这段下标均分给 blockDim 个核，返回本核的 [first, last)。
__aicore__ inline void CplxSlice(uint32_t count, uint32_t& first, uint32_t& last)
{
    const uint32_t blockCount = AscendC::GetBlockNum();
    const uint32_t blocks = blockCount == 0 ? 1u : blockCount;
    const uint32_t per = (count + blocks - 1) / blocks;
    first = AscendC::GetBlockIdx() * per;
    last = CplxMinU32(first + per, count);
}

// 建两张表（长度都是 2*srcElems）：
//   dupByteOffset[i] = 4 * floor(i/2)   —— 源元素下标的字节偏移
//   pairParity[i]    = i & 1            —— 位置奇偶
//
// arch22 上没有 Ands（整个接口不支持），And 又只吃 16 位整数，所以这里不用按位
// 与：非负数的算术右移 1 位就是 floor(i/2)，`i - 2*floor(i/2)` 就是 i&1。
// `scratch` 是长度 2*srcElems 的临时区，得和另外两张表不重叠。
__aicore__ inline void CplxBuildPairIndex(AscendC::LocalTensor<int32_t>& dupByteOffset,
                                          AscendC::LocalTensor<int32_t>& pairParity,
                                          AscendC::LocalTensor<int32_t>& scratch, uint32_t srcElems)
{
    const int32_t twins = static_cast<int32_t>(2u * srcElems);
    AscendC::CreateVecIndex<int32_t>(pairParity, 0, twins);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight<int32_t>(scratch, pairParity, 1, twins);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(dupByteOffset, scratch, 4, twins);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(scratch, scratch, 2, twins);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(pairParity, pairParity, scratch, twins);
    AscendC::PipeBarrier<PIPE_V>();
}
} // namespace

// 把 X（complex64、行主序、ld == k）拆成实部平面和虚部平面。
//
// 一次连续 DMA 取回一段交错的 float 对，然后用两个 Gather 按字节偏移
// 0,8,16,... 和 4,12,20,... 把两个平面挑出来。X 很小（m x k）且每次调用只跑
// 一遍，所以这里用最直白的 gather 就够，不值得再为 DMA 对齐绕弯子。
class SddmmComplexSplitX {
public:
    __aicore__ inline SddmmComplexSplitX() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR xr, GM_ADDR xi, uint32_t complexCount, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;
        total = complexCount;
        const uint64_t elems = static_cast<uint64_t>(complexCount);
        xGm.SetGlobalBuffer((__gm__ float*)x, 2u * elems);
        xrGm.SetGlobalBuffer((__gm__ float*)xr, elems);
        xiGm.SetGlobalBuffer((__gm__ float*)xi, elems);
        pipe->InitBuffer(inQueue, 1, kCplxSplitChunk * 2u * sizeof(float));
        pipe->InitBuffer(outQueueR, 1, kCplxSplitChunk * sizeof(float));
        pipe->InitBuffer(outQueueI, 1, kCplxSplitChunk * sizeof(float));
        pipe->InitBuffer(offsetBuf, 2u * kCplxSplitChunk * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (total == 0) {
            return;
        }
        BuildOffsets();
        const uint32_t chunks = (total + kCplxSplitChunk - 1u) / kCplxSplitChunk;
        uint32_t first = 0;
        uint32_t last = 0;
        CplxSlice(chunks, first, last);
        for (uint32_t c = first; c < last; ++c) {
            DoChunk(c);
        }
    }

private:
    // 实部在 8*i 字节、虚部在 8*i+4 字节；两张表一次建好，逐块复用。
    __aicore__ inline void BuildOffsets()
    {
        AscendC::LocalTensor<int32_t> realOff = offsetBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> imagOff = realOff[kCplxSplitChunk];
        AscendC::CreateVecIndex<int32_t>(realOff, 0, static_cast<int32_t>(kCplxSplitChunk));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(realOff, realOff, 8, static_cast<int32_t>(kCplxSplitChunk));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(imagOff, realOff, 4, static_cast<int32_t>(kCplxSplitChunk));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DoChunk(uint32_t chunk)
    {
        const uint32_t base = chunk * kCplxSplitChunk;
        const uint32_t cnt = CplxMinU32(total - base, kCplxSplitChunk);

        AscendC::LocalTensor<int32_t> offs = offsetBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> realOff = offs.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<uint32_t> imagOff = offs[kCplxSplitChunk].ReinterpretCast<uint32_t>();

        AscendC::LocalTensor<float> raw = inQueue.AllocTensor<float>();
        AscendC::DataCopyExtParams rp{1, cnt * 2u * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(raw, xGm[2u * static_cast<uint64_t>(base)], rp, pad);
        inQueue.EnQue(raw);
        raw = inQueue.DeQue<float>();

        AscendC::LocalTensor<float> xr = outQueueR.AllocTensor<float>();
        AscendC::LocalTensor<float> xi = outQueueI.AllocTensor<float>();
        AscendC::Gather<float>(xr, raw, realOff, 0, cnt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather<float>(xi, raw, imagOff, 0, cnt);
        AscendC::PipeBarrier<PIPE_V>();
        inQueue.FreeTensor(raw);

        outQueueR.EnQue(xr);
        outQueueI.EnQue(xi);
        xr = outQueueR.DeQue<float>();
        xi = outQueueI.DeQue<float>();
        AscendC::DataCopyExtParams wp{1, cnt * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(xrGm[base], xr, wp);
        AscendC::DataCopyPad(xiGm[base], xi, wp);
        outQueueR.FreeTensor(xr);
        outQueueI.FreeTensor(xi);
    }

    AscendC::TPipe* pipe = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueR;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueI;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> xrGm;
    AscendC::GlobalTensor<float> xiGm;
    uint32_t total = 0;
};

// 把 CSR 结构展开成"每个逻辑元素各占一个实数列"的形式。
//
//   rowPtr2[i]  = 2 * rowPtr[i] - base
//   colIdx2[2e] = 2 * (colIdx[e] - base) + base
//   colIdx2[2e+1] = colIdx2[2e] + 1
//
// 两次 SDDMM 用同一份展开后的结构：第 2e 个槽位取 Y 第 2j 个实数（实部），
// 第 2e+1 个槽位取第 2j+1 个（虚部）。基值原样透传，减掉 base 之后两次都得到
// 0 基的列号，所以 0 基和 1 基共用一条路径。
class SddmmComplexPrepCsr {
public:
    __aicore__ inline SddmmComplexPrepCsr() {}

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR colIdx, GM_ADDR rowPtr2, GM_ADDR colIdx2, uint32_t rows,
                                uint32_t nnz, int32_t idxBase, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;
        rowCount = rows;
        nnzCount = nnz;
        base = idxBase;
        const uint64_t rowElems = static_cast<uint64_t>(rows) + 1u;
        const uint64_t idxElems = static_cast<uint64_t>(nnz) * 2u;
        rowPtrGm.SetGlobalBuffer((__gm__ int32_t*)rowPtr, rowElems);
        colIdxGm.SetGlobalBuffer((__gm__ int32_t*)colIdx, nnz);
        rowPtr2Gm.SetGlobalBuffer((__gm__ int32_t*)rowPtr2, rowElems);
        colIdx2Gm.SetGlobalBuffer((__gm__ int32_t*)colIdx2, idxElems);
        pipe->InitBuffer(inQueue, 1, kCplxIdxChunk * sizeof(int32_t));
        pipe->InitBuffer(outQueue, 1, 2u * kCplxIdxChunk * sizeof(int32_t));
        // dupOff / parity / scratch 三张等长表，步长是 kCplxPairTable（2 * chunk）
        pipe->InitBuffer(tableBuf, 3u * kCplxPairTable * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        BuildTables();
        ExpandRowOffsets();
        ExpandColIndices();
    }

private:
    __aicore__ inline void BuildTables()
    {
        AscendC::LocalTensor<int32_t> dupOff = tableBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> parity = dupOff[kCplxPairTable];
        AscendC::LocalTensor<int32_t> scratch = dupOff[2u * kCplxPairTable];
        CplxBuildPairIndex(dupOff, parity, scratch, kCplxIdxChunk);
    }

    __aicore__ inline void ExpandRowOffsets()
    {
        const uint32_t chunks = (rowCount + 1u + kCplxIdxChunk - 1u) / kCplxIdxChunk;
        uint32_t first = 0;
        uint32_t last = 0;
        CplxSlice(chunks, first, last);
        for (uint32_t c = first; c < last; ++c) {
            const uint32_t off = c * kCplxIdxChunk;
            const uint32_t cnt = CplxMinU32(rowCount + 1u - off, kCplxIdxChunk);
            AscendC::LocalTensor<int32_t> src = Pull(rowPtrGm, off, cnt);
            AscendC::LocalTensor<int32_t> dst = outQueue.AllocTensor<int32_t>();
            AscendC::Muls(dst, src, 2, static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(dst, dst, -base, static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_V>();
            inQueue.FreeTensor(src);
            Push(rowPtr2Gm, off, dst, cnt);
        }
    }

    __aicore__ inline void ExpandColIndices()
    {
        if (nnzCount == 0) {
            return;
        }
        const uint32_t chunks = (nnzCount + kCplxIdxChunk - 1u) / kCplxIdxChunk;
        uint32_t first = 0;
        uint32_t last = 0;
        CplxSlice(chunks, first, last);
        AscendC::LocalTensor<int32_t> dupOff = tableBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> dupOffU = dupOff.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<int32_t> parity = dupOff[kCplxPairTable];
        for (uint32_t c = first; c < last; ++c) {
            const uint32_t off = c * kCplxIdxChunk;
            const uint32_t cnt = CplxMinU32(nnzCount - off, kCplxIdxChunk);
            AscendC::LocalTensor<int32_t> src = Pull(colIdxGm, off, cnt);
            // 先算 0 基列号对应的"实数列"起点，再复制成相邻的一对。
            AscendC::Muls(src, src, 2, static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(src, src, -base, static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<int32_t> dst = outQueue.AllocTensor<int32_t>();
            AscendC::Gather<int32_t>(dst, src, dupOffU, 0, 2u * cnt);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(dst, dst, parity, static_cast<int32_t>(2u * cnt));
            AscendC::PipeBarrier<PIPE_V>();
            inQueue.FreeTensor(src);
            Push(colIdx2Gm, 2u * off, dst, 2u * cnt);
        }
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> Pull(AscendC::GlobalTensor<int32_t>& gm, uint32_t off, uint32_t cnt)
    {
        AscendC::LocalTensor<int32_t> buf = inQueue.AllocTensor<int32_t>();
        AscendC::DataCopyExtParams p{1, cnt * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(buf, gm[off], p, pad);
        inQueue.EnQue(buf);
        return inQueue.DeQue<int32_t>();
    }

    __aicore__ inline void Push(AscendC::GlobalTensor<int32_t>& gm, uint32_t off, AscendC::LocalTensor<int32_t>& buf,
                                uint32_t cnt)
    {
        outQueue.EnQue(buf);
        AscendC::LocalTensor<int32_t> out = outQueue.DeQue<int32_t>();
        AscendC::DataCopyExtParams p{1, cnt * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(gm[off], out, p);
        outQueue.FreeTensor(out);
    }

    AscendC::TPipe* pipe = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tableBuf;
    AscendC::GlobalTensor<int32_t> rowPtrGm;
    AscendC::GlobalTensor<int32_t> colIdxGm;
    AscendC::GlobalTensor<int32_t> rowPtr2Gm;
    AscendC::GlobalTensor<int32_t> colIdx2Gm;
    uint32_t rowCount = 0;
    uint32_t nnzCount = 0;
    int32_t base = 0;
};

// Two FP32 lane accumulators form Re/Im in UB. A rolling Y window is shared
// across consecutive CSR rows; no P/Q intermediates or combine launch remain.
class SddmmComplexFusedProducts {
public:
    __aicore__ inline void Init(GM_ADDR xr, GM_ADDR xi, GM_ADDR yt, GM_ADDR rp, GM_ADDR ci, GM_ADDR values,
                                uint32_t rows, uint32_t cols, uint32_t k, uint32_t pitch, uint32_t nnz, int32_t base,
                                float ar, float ai, float br, float bi, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;
        rowCount = rows;
        columnCount = cols;
        K = k;
        yPitch = pitch;
        total = nnz;
        indexBase = base;
        alphaR = ar;
        alphaI = ai;
        betaR = br;
        betaI = bi;
        tile = CplxMinU32(64u, (SddmmBudget::kComplexTileElements / (2u * pitch)) / 8u * 8u);
        spanCapacity = SddmmBudget::kComplexWindowElements / (2u * pitch);
        xrGm.SetGlobalBuffer((__gm__ float*)xr, static_cast<uint64_t>(rows) * k);
        xiGm.SetGlobalBuffer((__gm__ float*)xi, static_cast<uint64_t>(rows) * k);
        ytGm.SetGlobalBuffer((__gm__ float*)yt, static_cast<uint64_t>(2u * cols) * pitch);
        rpGm.SetGlobalBuffer((__gm__ int32_t*)rp, rows + 1u);
        ciGm.SetGlobalBuffer((__gm__ int32_t*)ci, nnz);
        valuesGm.SetGlobalBuffer((__gm__ float*)values, 2ull * nnz);
        pipe->InitBuffer(aBuf, 3u * pitch * sizeof(float));
        pipe->InitBuffer(bBuf, SddmmBudget::kComplexWindowElements * sizeof(float));
        pipe->InitBuffer(indexBuf, tile * sizeof(int32_t));
        pipe->InitBuffer(productBuf, 2u * tile * 64u * sizeof(float));
        pipe->InitBuffer(sumBuf, 2u * tile * sizeof(float));
        pipe->InitBuffer(scaledBuf, 2u * tile * sizeof(float));
        pipe->InitBuffer(tableBuf, 6u * tile * sizeof(int32_t));
        pipe->InitBuffer(outP, 1, 2u * tile * sizeof(float));
        pipe->InitBuffer(inC, 1, 2u * tile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t units = (total + 3u) / 4u;
        uint32_t firstUnit = 0, lastUnit = 0;
        CplxSlice(units, firstUnit, lastUnit);
        uint32_t first = CplxMinU32(firstUnit * 4u, total);
        const uint32_t last = CplxMinU32(lastUnit * 4u, total);
        if (first >= last) {
            return;
        }
        auto offsets = tableBuf.Get<int32_t>();
        auto parity = offsets[2u * tile];
        auto scratch = offsets[4u * tile];
        CplxBuildPairIndex(offsets, parity, scratch, tile);
        AscendC::Muls(parity, parity, static_cast<int32_t>(tile * sizeof(float)), 2u * tile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(offsets, offsets, parity, 2u * tile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CreateVecIndex(scratch, 0, tile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scratch, scratch, 8, tile);
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t lo = 0, hi = rowCount;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2u;
            if (rpGm.GetValue(mid + 1u) - indexBase <= static_cast<int32_t>(first)) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        for (uint32_t row = lo; row < rowCount && first < last; ++row) {
            const uint32_t end = CplxMinU32(rpGm.GetValue(row + 1u) - indexBase, last);
            if (first == end) {
                continue;
            }
            Fence<AscendC::HardEvent::V_MTE2>();
            auto a = aBuf.Get<float>();
            AscendC::DataCopyExtParams cp{1, K * 4u, 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(a, xrGm[static_cast<uint64_t>(row) * K], cp, pad);
            AscendC::DataCopyPad(a[yPitch], xiGm[static_cast<uint64_t>(row) * K], cp, pad);
            negativeReady = false;
            for (; first < end;) {
                const uint32_t count = CplxMinU32(tile, end - first);
                Compute(first, count);
                first += count;
            }
        }
    }

private:
    template <AscendC::HardEvent Event>
    __aicore__ inline void Fence()
    {
        const event_t id = static_cast<event_t>(pipe->FetchEventID(Event));
        AscendC::SetFlag<Event>(id);
        AscendC::WaitFlag<Event>(id);
    }

    __aicore__ inline AscendC::LocalTensor<float> LoadProducts(uint32_t first, uint32_t count)
    {
        auto indices = indexBuf.Get<int32_t>();
        Fence<AscendC::HardEvent::S_MTE2>();
        AscendC::DataCopyExtParams cp{1, count * 4u, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> ip{false, 0, 0, 0};
        AscendC::DataCopyPad(indices, ciGm[first], cp, ip);
        Fence<AscendC::HardEvent::MTE2_S>();
        auto b = bBuf.Get<float>();
        const int32_t c0 = indices.GetValue(0) - indexBase;
        bool consecutive = c0 >= 0 && static_cast<uint32_t>(c0) + count <= columnCount;
        for (uint32_t j = 1; j < count && consecutive; ++j) {
            consecutive = indices.GetValue(j) - indexBase == c0 + static_cast<int32_t>(j);
        }
        if (consecutive) {
            const uint32_t col = static_cast<uint32_t>(c0);
            if (!spanValid || col < spanFirst || col + count > spanFirst + spanCount) {
                Fence<AscendC::HardEvent::V_MTE2>();
                spanFirst = col;
                spanCount = CplxMinU32(spanCapacity, columnCount - col);
                AscendC::DataCopy(b, ytGm[2ull * col * yPitch], 2u * spanCount * yPitch);
                spanValid = true;
            }
            b = b[2u * (col - spanFirst) * yPitch];
        } else {
            spanValid = false;
            Fence<AscendC::HardEvent::V_MTE2>();
            for (uint32_t j = 0; j < count;) {
                const int32_t col = indices.GetValue(j) - indexBase;
                uint32_t run = 1;
                while (j + run < count && indices.GetValue(j + run) - indexBase == col + static_cast<int32_t>(run)) {
                    ++run;
                }
                if (col >= 0 && static_cast<uint32_t>(col) + run <= columnCount) {
                    AscendC::DataCopy(b[2u * j * yPitch], ytGm[2ull * col * yPitch], 2u * run * yPitch);
                }
                j += run;
            }
        }
        return b;
    }

    __aicore__ inline void ReduceProducts(AscendC::LocalTensor<float> b, uint32_t count)
    {
        Fence<AscendC::HardEvent::MTE2_V>();
        auto a = aBuf.Get<float>();
        if (!negativeReady) {
            AscendC::Muls(a[2u * yPitch], a[yPitch], -1.0f, K);
            AscendC::PipeBarrier<PIPE_V>();
            negativeReady = true;
        }
        auto prod = productBuf.Get<float>();
        const uint32_t slab = tile * 64u;
        const AscendC::BinaryRepeatParams mp{1, 1, 1, 8, 0, static_cast<uint8_t>(2u * yPitch / 8u)};
        const uint8_t repeat = static_cast<uint8_t>(count);
        for (uint32_t k0 = 0; k0 < K; k0 += 64u) {
            const uint64_t mask = CplxMinU32(64u, K - k0);
            if (k0 == 0) {
                AscendC::Mul(prod, a[k0], b[k0], mask, repeat, mp);
                AscendC::Mul(prod[slab], a[k0], b[yPitch + k0], mask, repeat, mp);
            } else {
                AscendC::MulAddDst(prod, a[k0], b[k0], mask, repeat, mp);
                AscendC::MulAddDst(prod[slab], a[k0], b[yPitch + k0], mask, repeat, mp);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(prod, a[2u * yPitch + k0], b[yPitch + k0], mask, repeat, mp);
            AscendC::MulAddDst(prod[slab], a[yPitch + k0], b[k0], mask, repeat, mp);
            AscendC::PipeBarrier<PIPE_V>();
        }
        auto sums = sumBuf.Get<float>();
        for (uint32_t term = 0; term < 2u; ++term) {
            AscendC::WholeReduceSum<float>(sums[term * tile], prod[term * slab], CplxMinU32(K, 64u), count, 1, 1, 8);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ScaleProducts(uint32_t first, uint32_t count)
    {
        auto sums = sumBuf.Get<float>();
        auto scaled = scaledBuf.Get<float>();
        if (alphaR == 0.0f) {
            AscendC::Duplicate(scaled, 0.0f, count);
            AscendC::Duplicate(scaled[tile], 0.0f, count);
        } else {
            AscendC::Muls(scaled, sums, alphaR, count);
            AscendC::Muls(scaled[tile], sums[tile], alphaR, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (alphaI != 0.0f) {
            AscendC::Axpy(scaled, sums[tile], -alphaI, count);
            AscendC::Axpy(scaled[tile], sums, alphaI, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (betaR != 0.0f || betaI != 0.0f) {
            AddPreviousProducts(first, count);
        }
    }

    __aicore__ inline void AddPreviousProducts(uint32_t first, uint32_t count)
    {
        auto sums = sumBuf.Get<float>();
        auto scaled = scaledBuf.Get<float>();
        auto old = inC.AllocTensor<float>();
        AscendC::DataCopyExtParams cp{1, count * 8u, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(old, valuesGm[2ull * first], cp, pad);
        inC.EnQue(old);
        old = inC.DeQue<float>();
        auto even = tableBuf.Get<int32_t>()[4u * tile].ReinterpretCast<uint32_t>();
        AscendC::Gather(sums, old, even, 0, count);
        AscendC::Gather(sums[tile], old, even, 4, count);
        AscendC::PipeBarrier<PIPE_V>();
        inC.FreeTensor(old);
        if (betaR != 0.0f) {
            AscendC::Axpy(scaled, sums, betaR, count);
            AscendC::Axpy(scaled[tile], sums[tile], betaR, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (betaI != 0.0f) {
            AscendC::Axpy(scaled, sums[tile], -betaI, count);
            AscendC::Axpy(scaled[tile], sums, betaI, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreProducts(uint32_t first, uint32_t count)
    {
        auto scaled = scaledBuf.Get<float>();
        auto offsets = tableBuf.Get<int32_t>().ReinterpretCast<uint32_t>();
        auto p = outP.AllocTensor<float>();
        AscendC::Gather(p, scaled, offsets, 0, count * 2u);
        outP.EnQue(p);
        p = outP.DeQue<float>();
        AscendC::DataCopyExtParams out{1, count * 8u, 0, 0, 0};
        AscendC::DataCopyPad(valuesGm[2ull * first], p, out);
        outP.FreeTensor(p);
    }

    __aicore__ inline void Compute(uint32_t first, uint32_t count)
    {
        auto b = LoadProducts(first, count);
        ReduceProducts(b, count);
        ScaleProducts(first, count);
        StoreProducts(first, count);
    }

    AscendC::TPipe* pipe = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aBuf, bBuf, indexBuf, productBuf, sumBuf, tableBuf, scaledBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outP;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inC;
    AscendC::GlobalTensor<float> xrGm, xiGm, ytGm, valuesGm;
    AscendC::GlobalTensor<int32_t> rpGm, ciGm;
    uint32_t rowCount = 0, columnCount = 0, K = 0, yPitch = 0, total = 0, tile = 0;
    int32_t indexBase = 0;
    uint32_t spanCapacity = 0, spanFirst = 0, spanCount = 0;
    bool spanValid = false, negativeReady = false;
    float alphaR = 0, alphaI = 0, betaR = 0, betaI = 0;
};

#endif
