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
#ifndef SDDMM_BSR_SMALL_KERNEL_H_
#define SDDMM_BSR_SMALL_KERNEL_H_

namespace {
constexpr uint32_t kColumns = 32;
constexpr uint32_t kLanes = 64;

template <AscendC::HardEvent Event>
__aicore__ inline void BsrFence()
{
    const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    AscendC::SetFlag<Event>(id);
    AscendC::WaitFlag<Event>(id);
}

__aicore__ inline uint32_t RoundUp(uint32_t value, uint32_t alignment)
{
    const uint32_t safeAlignment = alignment == 0 ? 1u : alignment;
    return (value + safeAlignment - 1) / safeAlignment * safeAlignment;
}

template <typename T, typename CT>
struct SmallTileViews {
    AscendC::LocalTensor<float> a, stage, b, prod, output, old;
    AscendC::LocalTensor<int32_t> offsets;
    AscendC::LocalTensor<T> aRaw, bRaw;
    AscendC::LocalTensor<CT> cRaw, outRaw;
    uint32_t bd, k, n, pitch;
    bool columnMajor;
    float alpha, beta;
};

template <typename T, typename CT>
__aicore__ inline void LoadSmallTile(SmallTileViews<T, CT>& v, AscendC::GlobalTensor<T> xG, AscendC::GlobalTensor<T> yG,
                                     AscendC::GlobalTensor<CT> cG, uint32_t row, uint32_t col, uint32_t block,
                                     uint32_t width, uint32_t count, bool newRow)
{
    const uint32_t alignT = 32 / sizeof(T);
    const uint32_t paddedK = RoundUp(v.k, alignT);
    const uint32_t paddedWidth = RoundUp(width, alignT);
    // Retire all readers before recycling DMA destinations or the v.output.
    BsrFence<AscendC::HardEvent::V_MTE2>();
    BsrFence<AscendC::HardEvent::MTE3_V>();
    if (newRow) {
        AscendC::DataCopyPad(v.aRaw, xG[uint64_t(row) * v.bd * v.k],
                             {uint16_t(v.bd), uint32_t(v.k * sizeof(T)), 0, (v.pitch - paddedK) / alignT, 0},
                             AscendC::DataCopyPadExtParams<T>{true, 0, uint8_t(paddedK - v.k), T(0)});
    }
    // GM strides are bytes; the UB destination stride is in 32B blocks.
    AscendC::DataCopyPad(v.bRaw, yG[uint64_t(col) * v.bd],
                         {uint16_t(v.k), uint32_t(width * sizeof(T)), uint32_t((v.n - width) * sizeof(T)),
                          (kColumns - paddedWidth) / alignT, 0},
                         AscendC::DataCopyPadExtParams<T>{true, 0, uint8_t(paddedWidth - width), T(0)});
    if (v.beta != 0.0f) {
        AscendC::DataCopyPad(v.cRaw, cG[uint64_t(block) * v.bd * v.bd], {1, uint32_t(count * sizeof(CT)), 0, 0, 0},
                             AscendC::DataCopyPadExtParams<CT>{false, 0, 0, CT(0)});
    }
    BsrFence<AscendC::HardEvent::MTE2_V>();
    if constexpr (!AscendC::IsSameType<T, float>::value) {
        if (newRow) {
            AscendC::Cast(v.a, v.aRaw, AscendC::RoundMode::CAST_NONE, v.bd * v.pitch);
        }
        AscendC::Cast(v.stage, v.bRaw, AscendC::RoundMode::CAST_NONE, v.k * kColumns);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if constexpr (!AscendC::IsSameType<CT, float>::value) {
        if (v.beta != 0.0f) {
            AscendC::Cast(v.old, v.cRaw, AscendC::RoundMode::CAST_NONE, count);
        }
    }
}

template <typename T, typename CT>
__aicore__ inline void ComputeSmallTile(SmallTileViews<T, CT>& v, uint32_t width, uint32_t run)
{
    AscendC::Gather(v.b, v.stage, v.offsets.template ReinterpretCast<uint32_t>(), 0, width * v.pitch);
    AscendC::PipeBarrier<PIPE_V>();
    const AscendC::BinaryRepeatParams multiply{1, 1, 1, 8, 0, uint8_t(v.pitch / 8)};
    for (uint32_t r = 0; r < v.bd; ++r) {
        AscendC::Mul(v.prod, v.a[r * v.pitch], v.b, uint64_t(v.k < kLanes ? v.k : kLanes), uint8_t(width), multiply);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t off = kLanes; off < v.k; off += kLanes) {
            const uint32_t len = v.k - off < kLanes ? v.k - off : kLanes;
            AscendC::MulAddDst(v.prod, v.a[r * v.pitch + off], v.b[off], uint64_t(len), uint8_t(width), multiply);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (v.columnMajor) {
            // Column j lands at j*v.bd+r, including across grouped blocks.
            // The reduction destination stride is measured in elements.
            AscendC::WholeReduceSum(v.output[r], v.prod, int32_t(v.k < kLanes ? v.k : kLanes), int32_t(width),
                                    int32_t(v.bd), 1, 8);
        } else {
            for (uint32_t g = 0; g < run; ++g) {
                AscendC::WholeReduceSum(v.output[g * v.bd * v.bd + r * v.bd], v.prod[g * v.bd * kLanes],
                                        int32_t(v.k < kLanes ? v.k : kLanes), int32_t(v.bd), 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T, typename CT>
__aicore__ inline void StoreSmallTile(SmallTileViews<T, CT>& v, AscendC::GlobalTensor<CT> cG, uint32_t block,
                                      uint32_t count)
{
    AscendC::Muls(v.output, v.output, v.alpha, count);
    AscendC::PipeBarrier<PIPE_V>();
    if (v.beta != 0.0f) {
        AscendC::Axpy(v.output, v.old, v.beta, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if constexpr (!AscendC::IsSameType<CT, float>::value) {
        AscendC::Cast(v.outRaw, v.output, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    BsrFence<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopyPad(cG[uint64_t(block) * v.bd * v.bd], v.outRaw, {1, uint32_t(count * sizeof(CT)), 0, 0, 0});
}

template <typename T, typename CT>
__aicore__ inline void InitSmallBuffers(
    AscendC::TPipe& pipe, uint32_t bd, uint32_t pitch, AscendC::TBuf<AscendC::TPosition::VECCALC>& aBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& stageBuf, AscendC::TBuf<AscendC::TPosition::VECCALC>& bBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& offsetsBuf, AscendC::TBuf<AscendC::TPosition::VECCALC>& prodBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& outputBuf, AscendC::TBuf<AscendC::TPosition::VECCALC>& oldBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& aRawBuf, AscendC::TBuf<AscendC::TPosition::VECCALC>& bRawBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& cRawBuf, AscendC::TBuf<AscendC::TPosition::VECCALC>& outRawBuf)
{
    const uint32_t outputCapacity = bd * kColumns;
    // At bd=8, K=256 the widest type pair uses <144 KiB. No live
    // widening conversion aliases its input, and every row starts on 32B.
    pipe.InitBuffer(aBuf, bd * pitch * sizeof(float));
    pipe.InitBuffer(stageBuf, kColumns * pitch * sizeof(float));
    pipe.InitBuffer(bBuf, kColumns * pitch * sizeof(float));
    pipe.InitBuffer(offsetsBuf, kColumns * pitch * sizeof(int32_t));
    pipe.InitBuffer(prodBuf, kColumns * kLanes * sizeof(float));
    pipe.InitBuffer(outputBuf, outputCapacity * sizeof(float));
    pipe.InitBuffer(oldBuf, outputCapacity * sizeof(float));
    if constexpr (!AscendC::IsSameType<T, float>::value) {
        pipe.InitBuffer(aRawBuf, bd * pitch * sizeof(T));
        pipe.InitBuffer(bRawBuf, kColumns * pitch * sizeof(T));
    }
    if constexpr (!AscendC::IsSameType<CT, float>::value) {
        pipe.InitBuffer(cRawBuf, outputCapacity * sizeof(CT));
        pipe.InitBuffer(outRawBuf, outputCapacity * sizeof(CT));
    }
}

template <typename T, typename CT>
struct SmallStorage {
    AscendC::TBuf<AscendC::TPosition::VECCALC> aBuf, stageBuf, bBuf, offsetsBuf, prodBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf, oldBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aRawBuf, bRawBuf, cRawBuf, outRawBuf;
    __aicore__ inline SmallTileViews<T, CT> Init(AscendC::TPipe& pipe, uint32_t bd, uint32_t k, uint32_t n,
                                                 uint32_t pitch, bool columnMajor, float alpha, float beta)
    {
        InitSmallBuffers<T, CT>(pipe, bd, pitch, aBuf, stageBuf, bBuf, offsetsBuf, prodBuf, outputBuf, oldBuf, aRawBuf,
                                bRawBuf, cRawBuf, outRawBuf);
        auto a = aBuf.Get<float>();
        auto stage = stageBuf.Get<float>();
        auto b = bBuf.Get<float>();
        auto offsets = offsetsBuf.Get<int32_t>();
        auto prod = prodBuf.Get<float>();
        auto output = outputBuf.Get<float>();
        auto old = oldBuf.Get<float>();
        AscendC::LocalTensor<T> aRaw, bRaw;
        if constexpr (AscendC::IsSameType<T, float>::value) {
            aRaw = a;
            bRaw = stage;
        } else {
            aRaw = aRawBuf.Get<T>();
            bRaw = bRawBuf.Get<T>();
        }
        AscendC::LocalTensor<CT> cRaw, outRaw;
        if constexpr (AscendC::IsSameType<CT, float>::value) {
            cRaw = old;
            outRaw = output;
        } else {
            cRaw = cRawBuf.Get<CT>();
            outRaw = outRawBuf.Get<CT>();
        }

        return {a,    stage,  b,  prod, output, old,   offsets,     aRaw,  bRaw,
                cRaw, outRaw, bd, k,    n,      pitch, columnMajor, alpha, beta};
    }
};

__aicore__ inline void InitSmallOffsets(AscendC::LocalTensor<int32_t> offsets, uint32_t pitch)
{
    // Transpose [K,32] -> [32,Kpitch] with one resident byte-offset table.
    AscendC::CreateVecIndex(offsets, int32_t(0), pitch);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(offsets, offsets, int32_t(kColumns * sizeof(float)), pitch);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t c = 1; c < kColumns; ++c) {
        AscendC::Adds(offsets[c * pitch], offsets, int32_t(c * sizeof(float)), pitch);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline uint32_t FindSmallBlockRow(AscendC::GlobalTensor<int32_t> rowG, uint32_t rows, uint32_t first,
                                             int32_t base)
{
    uint32_t lo = 0, hi = rows;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (uint32_t(rowG.GetValue(mid + 1) - base) <= first) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template <typename T, typename CT>
__global__ __aicore__ void BsrSmall(GM_ADDR x, GM_ADDR y, GM_ADDR rows, GM_ADDR cols, GM_ADDR values, uint32_t m,
                                    uint32_t n, uint32_t k, uint32_t bd, uint32_t blocks, int32_t base,
                                    bool columnMajor, float alpha, float beta)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t blockCount = AscendC::GetBlockNum();
    const uint32_t cores = blockCount == 0 ? 1u : blockCount;
    const uint32_t safeBd = bd == 0 ? 1u : bd;
    const uint32_t perCore = (blocks + cores - 1) / cores;
    const uint32_t first = AscendC::GetBlockIdx() * perCore;
    const uint32_t last = first + perCore < blocks ? first + perCore : blocks;
    if (first >= last) {
        return;
    }
    const uint32_t pitch = RoundUp(k, 16);
    AscendC::TPipe pipe;
    SmallStorage<T, CT> storage;
    auto tileView = storage.Init(pipe, bd, k, n, pitch, columnMajor, alpha, beta);
    auto offsets = tileView.offsets;
    InitSmallOffsets(offsets, pitch);

    AscendC::GlobalTensor<T> xG, yG;
    AscendC::GlobalTensor<CT> cG;
    AscendC::GlobalTensor<int32_t> rowG, colG;
    xG.SetGlobalBuffer((__gm__ T*)x);
    yG.SetGlobalBuffer((__gm__ T*)y);
    cG.SetGlobalBuffer((__gm__ CT*)values);
    rowG.SetGlobalBuffer((__gm__ int32_t*)rows);
    colG.SetGlobalBuffer((__gm__ int32_t*)cols);
    const uint32_t lo = FindSmallBlockRow(rowG, m / safeBd, first, base);
    uint32_t row = lo, cachedRow = m / safeBd;
    uint32_t rowEnd = rowG.GetValue(row + 1) - base;
    for (uint32_t block = first; block < last;) {
        while (block >= rowEnd) {
            rowEnd = rowG.GetValue(++row + 1) - base;
        }
        const uint32_t col = colG.GetValue(block) - base;
        uint32_t run = 1;
        while (run < kColumns / safeBd && block + run < last && block + run < rowEnd &&
               uint32_t(colG.GetValue(block + run) - base) == col + run) {
            ++run;
        }
        const uint32_t width = run * bd, count = width * bd;
        LoadSmallTile(tileView, xG, yG, cG, row, col, block, width, count, row != cachedRow);
        cachedRow = row;
        ComputeSmallTile(tileView, width, run);
        StoreSmallTile(tileView, cG, block, count);
        block += run;
    }
    BsrFence<AscendC::HardEvent::MTE3_V>();
}

struct ComplexVectorViews {
    AscendC::LocalTensor<float> rawA, rawB, aR, aI, bR, bI, negative, pR, pI, accR, accI, scaledR, scaledI, output, old;
    AscendC::LocalTensor<int32_t> mapA, mapB, map, index, temp;
    uint32_t rowsPerTile, maxColumns, width, pitch, count, bd, k, n;
    bool columnMajor;
    float ar, ai, br, bi;
};

__aicore__ inline void InitComplexVectorMaps(ComplexVectorViews& v)
{
    constexpr uint32_t capacity = 64;
    AscendC::CreateVecIndex(v.index, 0, v.pitch);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t r = 0; r < v.rowsPerTile; ++r) {
        AscendC::Muls(v.mapA[r * v.pitch], v.index, int32_t(v.columnMajor ? 32 : 8), v.pitch);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(v.mapA[r * v.pitch], v.mapA[r * v.pitch], int32_t(v.columnMajor ? r * 8 : r * 2 * v.pitch * 4),
                      v.pitch);
    }
    for (uint32_t c = 0; c < v.width; ++c) {
        AscendC::Muls(v.mapB[c * v.pitch], v.index, int32_t(v.columnMajor ? 8 : 2 * v.maxColumns * 4), v.pitch);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(v.mapB[c * v.pitch], v.mapB[c * v.pitch], int32_t(v.columnMajor ? c * 2 * v.pitch * 4 : c * 8),
                      v.pitch);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(v.mapA[v.rowsPerTile * v.pitch], v.mapA, int32_t(4), v.rowsPerTile * v.pitch);
    AscendC::Adds(v.mapB[v.width * v.pitch], v.mapB, int32_t(4), v.width * v.pitch);
    AscendC::CreateVecIndex(v.index, 0, 2 * capacity);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(v.map, v.index, int32_t(1), 2 * capacity);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(v.temp, v.map, int32_t(2), 2 * capacity);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(v.temp, v.index, v.temp, 2 * capacity);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(v.temp, v.temp, int32_t(capacity), 2 * capacity);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(v.map, v.map, v.temp, 2 * capacity);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(v.map, v.map, int32_t(4), 2 * capacity);
    AscendC::Muls(v.index, v.index, int32_t(8), capacity);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(v.index[capacity], v.index, int32_t(4), capacity);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void LoadComplexVectorTile(ComplexVectorViews& v, AscendC::GlobalTensor<float> xG,
                                             AscendC::GlobalTensor<float> yG, AscendC::GlobalTensor<float> cG,
                                             uint32_t col, uint32_t lo, uint32_t rowOffset, uint32_t colOffset,
                                             uint64_t dest)
{
    BsrFence<AscendC::HardEvent::V_MTE2>();
    BsrFence<AscendC::HardEvent::MTE3_V>();
    if (v.columnMajor) {
        AscendC::DataCopyPad(v.rawA, yG[uint64_t(col * v.bd + rowOffset) * 2],
                             {uint16_t(v.k), v.rowsPerTile * 8, (v.n - v.rowsPerTile) * 8, 0, 0},
                             AscendC::DataCopyPadExtParams<float>{true, 0, uint8_t(8 - 2 * v.rowsPerTile), 0});
        AscendC::DataCopyPad(v.rawB, xG[uint64_t(lo * v.bd + colOffset) * v.k * 2],
                             {uint16_t(v.width), v.k * 8, 0, (2 * v.pitch - RoundUp(2 * v.k, 8)) / 8, 0},
                             AscendC::DataCopyPadExtParams<float>{true, 0, uint8_t(RoundUp(2 * v.k, 8) - 2 * v.k), 0});
    } else {
        AscendC::DataCopyPad(v.rawA, xG[uint64_t(lo * v.bd + rowOffset) * v.k * 2],
                             {uint16_t(v.rowsPerTile), v.k * 8, 0, (2 * v.pitch - RoundUp(2 * v.k, 8)) / 8, 0},
                             AscendC::DataCopyPadExtParams<float>{true, 0, uint8_t(RoundUp(2 * v.k, 8) - 2 * v.k), 0});
        AscendC::DataCopyPad(v.rawB, yG[uint64_t(col * v.bd + colOffset) * 2],
                             {uint16_t(v.k), v.width * 8, (v.n - v.width) * 8, (v.maxColumns - v.width) / 4, 0},
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
    }
    if (v.br != 0 || v.bi != 0) {
        AscendC::DataCopyPad(v.old, cG[dest], {uint16_t(v.rowsPerTile), v.width * 8, (v.bd - v.width) * 8, 0, 0},
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
    }
    BsrFence<AscendC::HardEvent::MTE2_V>();
    AscendC::Gather(v.aR, v.rawA, v.mapA.template ReinterpretCast<uint32_t>(), 0, 2 * v.rowsPerTile * v.pitch);
    AscendC::Gather(v.bR, v.rawB, v.mapB.template ReinterpretCast<uint32_t>(), 0, 2 * v.width * v.pitch);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(v.negative, v.bI, -1.0f, v.width * v.pitch);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ComputeComplexVectorTile(ComplexVectorViews& v)
{
    const AscendC::BinaryRepeatParams repeat{1, 1, 1, 8, 0, uint8_t(v.pitch / 8)};
    for (uint32_t r = 0; r < v.rowsPerTile; ++r) {
        for (uint32_t off = 0; off < v.k; off += kLanes) {
            const uint64_t len = v.k - off < kLanes ? v.k - off : kLanes;
            if (off == 0) {
                AscendC::Mul(v.pR, v.aR[r * v.pitch], v.bR, len, uint8_t(v.width), repeat);
                AscendC::Mul(v.pI, v.aR[r * v.pitch], v.bI, len, uint8_t(v.width), repeat);
            } else {
                AscendC::MulAddDst(v.pR, v.aR[r * v.pitch + off], v.bR[off], len, uint8_t(v.width), repeat);
                AscendC::MulAddDst(v.pI, v.aR[r * v.pitch + off], v.bI[off], len, uint8_t(v.width), repeat);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(v.pR, v.aI[r * v.pitch + off], v.negative[off], len, uint8_t(v.width), repeat);
            AscendC::MulAddDst(v.pI, v.aI[r * v.pitch + off], v.bR[off], len, uint8_t(v.width), repeat);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::WholeReduceSum(v.accR[r * v.width], v.pR, int32_t(v.k < kLanes ? v.k : kLanes), int32_t(v.width), 1, 1,
                                8);
        AscendC::WholeReduceSum(v.accI[r * v.width], v.pI, int32_t(v.k < kLanes ? v.k : kLanes), int32_t(v.width), 1, 1,
                                8);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void StoreComplexVectorTile(ComplexVectorViews& v, AscendC::GlobalTensor<float> cG, uint64_t dest)
{
    constexpr uint32_t capacity = 64;
    if (v.ar == 0.0f) {
        AscendC::Duplicate(v.scaledR, 0.0f, v.count);
        AscendC::Duplicate(v.scaledI, 0.0f, v.count);
    } else {
        AscendC::Muls(v.scaledR, v.accR, v.ar, v.count);
        AscendC::Muls(v.scaledI, v.accI, v.ar, v.count);
    }
    AscendC::PipeBarrier<PIPE_V>();
    if (v.ai != 0) {
        AscendC::Axpy(v.scaledR, v.accI, -v.ai, v.count);
        AscendC::Axpy(v.scaledI, v.accR, v.ai, v.count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (v.br != 0 || v.bi != 0) {
        AscendC::Gather(v.pR, v.old, v.index.template ReinterpretCast<uint32_t>(), 0, v.count);
        AscendC::Gather(v.pI, v.old, v.index[capacity].template ReinterpretCast<uint32_t>(), 0, v.count);
        AscendC::PipeBarrier<PIPE_V>();
        if (v.br != 0.0f) {
            AscendC::Axpy(v.scaledR, v.pR, v.br, v.count);
            AscendC::Axpy(v.scaledI, v.pI, v.br, v.count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (v.bi != 0) {
            AscendC::Axpy(v.scaledR, v.pI, -v.bi, v.count);
            AscendC::Axpy(v.scaledI, v.pR, v.bi, v.count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    AscendC::Gather(v.output, v.scaledR, v.map.template ReinterpretCast<uint32_t>(), 0, v.count * 2);
    BsrFence<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopyPad(cG[dest], v.output, {uint16_t(v.rowsPerTile), v.width * 8, 0, (v.bd - v.width) * 8, 0});
}

struct ComplexVectorStorage {
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawABuf, rawBBuf, aBuf, bBuf, negativeBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mapABuf, mapBBuf, prodBuf, accBuf, scaledBuf, outBuf, oldBuf, tableBuf;
    __aicore__ inline ComplexVectorViews Init(AscendC::TPipe& pipe, uint32_t rowsPerTile, uint32_t maxColumns,
                                              uint32_t width, uint32_t pitch, uint32_t count, uint32_t bd, uint32_t k,
                                              uint32_t n, bool columnMajor, float ar, float ai, float br, float bi)
    {
        constexpr uint32_t capacity = 64;
        pipe.InitBuffer(rawABuf, 8 * pitch * sizeof(float));
        pipe.InitBuffer(rawBBuf, 2 * maxColumns * pitch * sizeof(float));
        pipe.InitBuffer(aBuf, 8 * pitch * sizeof(float));
        pipe.InitBuffer(bBuf, 2 * width * pitch * sizeof(float));
        pipe.InitBuffer(negativeBuf, width * pitch * sizeof(float));
        pipe.InitBuffer(mapABuf, 8 * pitch * sizeof(uint32_t));
        pipe.InitBuffer(mapBBuf, 2 * width * pitch * sizeof(uint32_t));
        pipe.InitBuffer(prodBuf, 2 * width * kLanes * sizeof(float));
        pipe.InitBuffer(accBuf, 2 * capacity * sizeof(float));
        pipe.InitBuffer(scaledBuf, 2 * capacity * sizeof(float));
        pipe.InitBuffer(outBuf, 2 * capacity * sizeof(float));
        pipe.InitBuffer(oldBuf, 2 * capacity * sizeof(float));
        pipe.InitBuffer(tableBuf, 6 * capacity * sizeof(int32_t));
        auto rawA = rawABuf.Get<float>(), rawB = rawBBuf.Get<float>();
        auto aR = aBuf.Get<float>(), aI = aR[rowsPerTile * pitch];
        auto bR = bBuf.Get<float>(), bI = bR[width * pitch];
        auto negative = negativeBuf.Get<float>();
        auto mapA = mapABuf.Get<int32_t>(), mapB = mapBBuf.Get<int32_t>();
        auto pR = prodBuf.Get<float>(), pI = pR[width * kLanes];
        auto accR = accBuf.Get<float>(), accI = accR[capacity];
        auto scaledR = scaledBuf.Get<float>(), scaledI = scaledR[capacity];
        auto output = outBuf.Get<float>(), old = oldBuf.Get<float>();
        auto map = tableBuf.Get<int32_t>(), index = map[2 * capacity];
        auto temp = index[2 * capacity];
        return {rawA,    rawB,    aR,     aI,  bR,   bI,   negative,    pR,    pI,   accR,        accI,
                scaledR, scaledI, output, old, mapA, mapB, map,         index, temp, rowsPerTile, maxColumns,
                width,   pitch,   count,  bd,  k,    n,    columnMajor, ar,    ai,   br,          bi};
    }
};

__global__ __aicore__ void BsrComplexVector(GM_ADDR x, GM_ADDR y, GM_ADDR rows, GM_ADDR cols, GM_ADDR values,
                                            uint32_t m, uint32_t n, uint32_t k, uint32_t bd, uint32_t blocks,
                                            int32_t base, bool columnMajor, float ar, float ai, float br, float bi)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t rowsPerTile = bd >= 16 ? 2 : 4;
    const uint32_t maxColumns = bd >= 16 ? 32 : 16;
    const uint32_t width = bd < maxColumns ? bd : maxColumns;
    const uint32_t pitch = RoundUp(k, 8);
    const uint32_t rowTiles = bd / rowsPerTile, colTiles = bd / width;
    const uint32_t tilesPerBlock = rowTiles * colTiles;
    const uint32_t count = rowsPerTile * width;
    AscendC::TPipe pipe;
    ComplexVectorStorage storage;
    auto tileView = storage.Init(pipe, rowsPerTile, maxColumns, width, pitch, count, bd, k, n, columnMajor, ar, ai, br,
                                 bi);
    InitComplexVectorMaps(tileView);
    AscendC::GlobalTensor<float> xG, yG, cG;
    AscendC::GlobalTensor<int32_t> rowG, colG;
    xG.SetGlobalBuffer((__gm__ float*)x);
    yG.SetGlobalBuffer((__gm__ float*)y);
    cG.SetGlobalBuffer((__gm__ float*)values);
    rowG.SetGlobalBuffer((__gm__ int32_t*)rows);
    colG.SetGlobalBuffer((__gm__ int32_t*)cols);
    for (uint32_t task = AscendC::GetBlockIdx(); task < blocks * tilesPerBlock; task += AscendC::GetBlockNum()) {
        const uint32_t block = task / tilesPerBlock, tile = task % tilesPerBlock;
        const uint32_t rowOffset = tile / colTiles * rowsPerTile, colOffset = tile % colTiles * width;
        const uint32_t safeBd = bd == 0 ? 1u : bd;
        uint32_t lo = 0, hi = m / safeBd;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (uint32_t(rowG.GetValue(mid + 1) - base) <= block) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        const uint32_t col = colG.GetValue(block) - base;
        const uint64_t dest = (uint64_t(block) * bd * bd + rowOffset * bd + colOffset) * 2;
        LoadComplexVectorTile(tileView, xG, yG, cG, col, lo, rowOffset, colOffset, dest);
        ComputeComplexVectorTile(tileView);
        StoreComplexVectorTile(tileView, cG, dest);
    }
    BsrFence<AscendC::HardEvent::MTE3_V>();
}

} // namespace

#endif
