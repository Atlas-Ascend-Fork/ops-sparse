/**
 * Copyright (c) 2026 liutaw.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "../spsm_kernel.h"
#include "spsm_dispatch.h"
#define SPSM_PACKED_INLINE __aicore__ inline
#include "spsm_packed_math.h"
using namespace AscendC;

// Packed stage.
using namespace spsm_packed;
namespace {
__aicore__ inline bool Prepared(GlobalTensor<int32_t> &status, int32_t blocks)
{
    for (int32_t b = 0; b < blocks; ++b) {
        if (status.GetValue(b * kSpsmA2StatusStride) != 0) { return false; }
    }
    return true;
}
__aicore__ inline bool PackedReady(GlobalTensor<int32_t> &status, int32_t blocks, int32_t condition)
{
    if (!Prepared(status, blocks) || !spsm_a2::DispatchReady(status, blocks, condition)) { return false; }
    for (int32_t b = 0; b < blocks; ++b) {
        if (status.GetValue(b * kSpsmA2StatusStride + 1) != 0 ||
            status.GetValue(b * kSpsmA2StatusStride + 2) != 0) { return false; }
    }
    return true;
}
// Examine exponent bits in contiguous DMA tiles. Clearing sign/mantissa makes
// every INF/NaN an identical positive infinity before reduction; NaN ordering
// of ReduceMax is therefore irrelevant. Do not infer finiteness from abs/compare.
class FiniteScanner {
// Scan at most 512 FP32 components per DMA and only valid tail elements.
// Masking with 0x7f800000 turns every INF/NaN into the same positive pattern,
// avoiding reliance on ReduceMax's NaN ordering. V/MTE2/S events protect reuse,
// input visibility and reading the scalar reduction result.
public:
    __aicore__ inline void Init(TPipe &pipe)
    {
        pipe.InitBuffer(input_, kTile * 4); pipe.InitBuffer(mask_, kTile * 4);
        pipe.InitBuffer(scratch_, kTile * 4); pipe.InitBuffer(maximum_, 32);
        Duplicate(mask_.Get<uint32_t>(), uint32_t(0x7f800000), kTile);
        reuse_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
        loaded_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        reduced_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
    }
    __aicore__ inline bool Nonfinite(GlobalTensor<float> &source, int64_t size)
    {
        auto data = input_.Get<float>();
        for (int64_t start = int64_t(GetBlockIdx()) * kTile; start < size;
             start += int64_t(GetBlockNum()) * kTile) {
            int32_t count = size - start < kTile ? size - start : kTile;
            SetFlag<HardEvent::V_MTE2>(reuse_); WaitFlag<HardEvent::V_MTE2>(reuse_);
            DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(data, source[start], cp, pad);
            SetFlag<HardEvent::MTE2_V>(loaded_); WaitFlag<HardEvent::MTE2_V>(loaded_);
            And(data.ReinterpretCast<uint16_t>(), data.ReinterpretCast<uint16_t>(),
                mask_.Get<uint16_t>(), count * 2);
            PipeBarrier<PIPE_V>();
            ReduceMax(maximum_.Get<float>(), data, scratch_.Get<float>(), count, false);
            SetFlag<HardEvent::V_S>(reduced_); WaitFlag<HardEvent::V_S>(reduced_);
            if (maximum_.Get<uint32_t>().GetValue(0) == 0x7f800000u) { return true; }
        }
        return false;
    }
private:
    static constexpr int32_t kTile = 512;
    TBuf<TPosition::VECCALC> input_, mask_, scratch_, maximum_;
    event_t reuse_{}, loaded_{}, reduced_{};
};
class WordPlanes {
// Two 32-bit word planes hold packed components without increasing BufferSize.
// A tagged cache batches at most 32 rows; writes never cross the split plane's
// allocation boundary. Misses read flushed rows; completion protects both UB
// reuse and later consumers. Live permutation/column/snapshot storage is excluded.
public:
    __aicore__ inline void Init(GM_ADDR dense, GM_ADDR first, GM_ADDR tail,
        int64_t lowRows, int32_t rhs, TPipe &pipe)
    {
        lowRows_ = lowRows; rhs_ = rhs;
        dense_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(dense));
        first_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(first));
        tail_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tail));
        pipe.InitBuffer(buffer_, 64);
        pipe.InitBuffer(pending_, 2 * 32 * 32);
        pipe.InitBuffer(inputRows_, 32 * 32);
        readReuse_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE2));
        loaded_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
        writeReady_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
        writeReuse_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
        visible_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    }
    __aicore__ inline int64_t HighOffset(int32_t row, int32_t col)
    {
        return ((row < lowRows_ ? row : row - lowRows_) * rhs_ + col) * 2;
    }
    __aicore__ inline GlobalTensor<uint32_t> High(int32_t row)
    {
        return row < lowRows_ ? first_ : tail_;
    }
    __aicore__ inline void Load(int32_t row, int32_t col, uint64_t &real, uint64_t &imag)
    {
        auto low = buffer_.Get<uint32_t>(); auto high = low[8];
        DataCopyExtParams cp{1, 8, 0, 0, 0}; DataCopyPadExtParams<uint32_t> pad{true, 0, 6, 0};
        SetFlag<HardEvent::S_MTE2>(readReuse_); WaitFlag<HardEvent::S_MTE2>(readReuse_);
        DataCopyPad(low, dense_[(int64_t(row) * rhs_ + col) * 2], cp, pad);
        auto source = High(row); DataCopyPad(high, source[HighOffset(row, col)], cp, pad);
        SetFlag<HardEvent::MTE2_S>(loaded_); WaitFlag<HardEvent::MTE2_S>(loaded_);
        real = uint64_t(low.GetValue(0)) | (uint64_t(high.GetValue(0)) << 32);
        imag = uint64_t(low.GetValue(1)) | (uint64_t(high.GetValue(1)) << 32);
    }
    __aicore__ inline void Input(int32_t row, int32_t col, int32_t rows, float &real, float &imag)
    {
        const uint64_t stride = (static_cast<uint64_t>(rhs_) - 1) * 8;
        if (stride <= UINT32_MAX) {
            const int32_t first = row / 32 * 32;
            if (inputFirst_ != first || inputCol_ != col) {
                const int32_t count = rows - first < 32 ? rows - first : 32;
                SetFlag<HardEvent::S_MTE2>(readReuse_); WaitFlag<HardEvent::S_MTE2>(readReuse_);
                DataCopyExtParams cp{static_cast<uint16_t>(count), 8,
                    static_cast<uint32_t>(stride), 0, 0};
                DataCopyPad(inputRows_.Get<uint32_t>(), dense_[(int64_t(first) * rhs_ + col) * 2],
                    cp, DataCopyPadExtParams<uint32_t>{true, 0, 6, 0});
                SetFlag<HardEvent::MTE2_S>(loaded_); WaitFlag<HardEvent::MTE2_S>(loaded_);
                inputFirst_ = first; inputCol_ = col;
            }
            // A core owns each RHS column throughout this chunk. Read its original
            // B rows before any pending output overwrites them, in either direction.
            auto input = inputRows_.Get<float>()[(row - first) * 8];
            real = input.GetValue(0); imag = input.GetValue(1);
            return;
        }
        // Keep the existing single-row path when the DMA stride cannot represent rhs.
        auto low = buffer_.Get<uint32_t>();
        DataCopyExtParams cp{1, 8, 0, 0, 0}; DataCopyPadExtParams<uint32_t> pad{true, 0, 6, 0};
        SetFlag<HardEvent::S_MTE2>(readReuse_); WaitFlag<HardEvent::S_MTE2>(readReuse_);
        DataCopyPad(low, dense_[(int64_t(row) * rhs_ + col) * 2], cp, pad);
        SetFlag<HardEvent::MTE2_S>(loaded_); WaitFlag<HardEvent::MTE2_S>(loaded_);
        Bits a, b; a.u = low.GetValue(0); b.u = low.GetValue(1); real = a.f; imag = b.f;
    }
    __aicore__ inline void Store(int32_t row, int32_t col, uint64_t real, uint64_t imag)
    {
        // Pending rows fit in the 32-entry tagged cache. Any dependency not in
        // that cache is older than this batch and has already reached GM.
        // Never batch across the two high-word workspace regions.
        if (pendingFirst_ >= 0 && (row / 32 != pendingFirst_ / 32 || col != pendingCol_ ||
            (row < lowRows_) != (pendingFirst_ < lowRows_))) { Flush(); }
        if (pendingFirst_ < 0) { pendingFirst_ = row; pendingLast_ = row; pendingCol_ = col; }
        if (row < pendingFirst_) { pendingFirst_ = row; }
        if (row > pendingLast_) { pendingLast_ = row; }
        auto low = pending_.Get<uint32_t>()[(row % 32) * 8]; auto high = low[32 * 8];
        low.SetValue(0, uint32_t(real)); low.SetValue(1, uint32_t(imag));
        high.SetValue(0, uint32_t(real >> 32)); high.SetValue(1, uint32_t(imag >> 32));
    }
    __aicore__ inline void Flush()
    {
        if (pendingFirst_ < 0) { return; }
        auto low = pending_.Get<uint32_t>()[(pendingFirst_ % 32) * 8]; auto high = low[32 * 8];
        const uint64_t stride = (static_cast<uint64_t>(rhs_) - 1) * 8;
        SetFlag<HardEvent::S_MTE3>(writeReady_); WaitFlag<HardEvent::S_MTE3>(writeReady_);
        auto destination = High(pendingFirst_);
        if (stride <= UINT32_MAX) {
            DataCopyExtParams cp{static_cast<uint16_t>(pendingLast_ - pendingFirst_ + 1),
                8, 0, static_cast<uint32_t>(stride), 0};
            DataCopyPad(dense_[(int64_t(pendingFirst_) * rhs_ + pendingCol_) * 2], low, cp);
            DataCopyPad(destination[HighOffset(pendingFirst_, pendingCol_)], high, cp);
        } else {
            // The public API accepts 64-bit dense strides. A single block copy
            // avoids truncating the DMA's 32-bit inter-block stride field.
            DataCopyExtParams cp{1, 8, 0, 0, 0};
            for (int32_t row = pendingFirst_; row <= pendingLast_; ++row) {
                int32_t offset = (row - pendingFirst_) * 8;
                DataCopyPad(dense_[(int64_t(row) * rhs_ + pendingCol_) * 2], low[offset], cp);
                DataCopyPad(destination[HighOffset(row, pendingCol_)], high[offset], cp);
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(visible_); WaitFlag<HardEvent::MTE3_MTE2>(visible_);
        SetFlag<HardEvent::MTE3_S>(writeReuse_); WaitFlag<HardEvent::MTE3_S>(writeReuse_);
        pendingFirst_ = -1;
    }
    __aicore__ inline void FloatOutput(int32_t row, int32_t col, uint64_t real, uint64_t imag)
    {
        auto low = buffer_.Get<uint32_t>();
        low.SetValue(0, PackedFloatBits(real)); low.SetValue(1, PackedFloatBits(imag));
        DataCopyExtParams cp{1, 8, 0, 0, 0};
        SetFlag<HardEvent::S_MTE3>(writeReady_); WaitFlag<HardEvent::S_MTE3>(writeReady_);
        DataCopyPad(dense_[(int64_t(row) * rhs_ + col) * 2], low, cp);
        SetFlag<HardEvent::MTE3_S>(writeReuse_); WaitFlag<HardEvent::MTE3_S>(writeReuse_);
    }
    __aicore__ inline void Failure(GlobalTensor<int32_t> &status, uint32_t error)
    {
        auto low = buffer_.Get<uint32_t>(); low.SetValue(0, error);
        DataCopyExtParams cp{1, 4, 0, 0, 0};
        SetFlag<HardEvent::S_MTE3>(writeReady_); WaitFlag<HardEvent::S_MTE3>(writeReady_);
        DataCopyPad(status[GetBlockIdx() * kSpsmA2StatusStride + 2], low.ReinterpretCast<int32_t>(), cp);
        SetFlag<HardEvent::MTE3_S>(writeReuse_); WaitFlag<HardEvent::MTE3_S>(writeReuse_);
    }
private:
    int64_t lowRows_{0}; int32_t rhs_{0};
    GlobalTensor<uint32_t> dense_, first_, tail_;
    TBuf<TPosition::VECCALC> buffer_, pending_, inputRows_;
    int32_t inputFirst_{-1}, inputCol_{-1};
    int32_t pendingFirst_{-1}, pendingLast_{-1}, pendingCol_{0};
    event_t readReuse_{}, loaded_{}, writeReady_{}, writeReuse_{}, visible_{};
};
__aicore__ inline void AccumulatePackedRow(GlobalTensor<int32_t> &rp, GlobalTensor<int32_t> &ci,
    GlobalTensor<float> &av, LocalTensor<uint64_t> &recent, WordPlanes &memory,
    const SpsmA2TilingData &t, int32_t row, int32_t k, ScaledComplex &current,
    float &dr, float &di, uint32_t &error)
{
    int32_t start = rp.GetValue(row), finish = rp.GetValue(row + 1);
    for (int32_t p = start; p < finish; ++p) {
        int32_t col = ci.GetValue(p); float vr = av.GetValue(int64_t(p) * 2), vi = av.GetValue(int64_t(p) * 2 + 1);
        if (col == row) {
            if (!t.unit) { dr = AddScalar(dr, FlushSubnormal(vr)); di = AddScalar(di, FlushSubnormal(vi)); }
            continue;
        }
        if (col < 0 || col >= t.rows || (t.upper ? col < row : col > row)) { error = 5; break; }
        uint64_t re, im; int32_t slot = (col % 32) * 4;
        if (recent.GetValue(slot) == uint64_t(col)) {
            re = recent.GetValue(slot + 1);
            im = recent.GetValue(slot + 2);
        } else { memory.Load(col, k, re, im); }
        ScaledComplex term = TimesNormalized({Decode(re), Decode(im)}, -vr, -vi);
        current = {AddScaled(current.real, term.real), AddScaled(current.imag, term.imag)};
    }
}
}
extern "C" __global__ __aicore__ void spsm_packed_guard(GM_ADDR offsets, GM_ADDR values,
    GM_ADDR dense, GM_ADDR alpha, GM_ADDR status, SpsmA2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<int32_t> rp, flags; GlobalTensor<float> a, b, scalar;
    rp.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(offsets));
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(values));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dense));
    scalar.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(alpha));
    if (!Prepared(flags, t.statusBlocks)) { return; }
    float ar = t.deviceAlpha ? scalar.GetValue(0) : t.alpha;
    float ai = t.deviceAlpha ? scalar.GetValue(1) : t.alphaImag;
    bool reject = !IsFinite(ar) || !IsFinite(ai);
    TPipe pipe; FiniteScanner scan; scan.Init(pipe);
    reject = reject || scan.Nonfinite(a, int64_t(rp.GetValue(t.rows)) * 2) ||
        scan.Nonfinite(b, int64_t(t.rows) * t.rhs * 2);
    TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf, 32);
    auto words = buf.Get<int32_t>(); words.SetValue(0, reject ? 1 : 0); words.SetValue(1, 0); words.SetValue(2, 0);
    auto event = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(event); WaitFlag<HardEvent::S_MTE3>(event);
    DataCopyExtParams cp{1, 12, 0, 0, 0};
    DataCopyPad(flags[GetBlockIdx() * kSpsmA2StatusStride + 1], words, cp);
    auto done = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(done); WaitFlag<HardEvent::MTE3_S>(done);
}
// A finite input with any nonfinite high or low row component retries from B.
// Inspect all stored rows: an earlier overflow must not disappear just because
// the final row happens not to depend on it. Nonfinite inputs retain old behavior.
extern "C" __global__ __aicore__ void spsm_packed_retry(GM_ADDR dense, GM_ADDR first,
    GM_ADDR tail, GM_ADDR status, int64_t lowRows, SpsmA2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<int32_t> flags;
    GlobalTensor<float> high, lowFirst, lowTail;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
    high.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dense));
    lowFirst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(first));
    lowTail.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tail));
    if (!Prepared(flags, t.statusBlocks)) { return; }
    bool eligible = true, retry = false;
    for (int32_t block = 0; block < t.statusBlocks; ++block) {
        eligible = eligible && flags.GetValue(block * kSpsmA2StatusStride + 1) == 0;
    }
    int64_t split = lowRows * t.rhs * 2;
    int64_t total = int64_t(t.rows) * t.rhs * 2;
    TPipe pipe; FiniteScanner scan; scan.Init(pipe);
    retry = eligible && (scan.Nonfinite(high, total) || scan.Nonfinite(lowFirst, split) ||
        scan.Nonfinite(lowTail, total - split));
    TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf, 32);
    auto words = buf.Get<int32_t>(); words.SetValue(0, retry ? 1 : 0);
    auto ready = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(ready); WaitFlag<HardEvent::S_MTE3>(ready);
    DataCopyExtParams cp{1, 4, 0, 0, 0};
    DataCopyPad(flags[GetBlockIdx() * kSpsmA2StatusStride + 3], words, cp);
    auto done = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(done); WaitFlag<HardEvent::MTE3_S>(done);
}
extern "C" __global__ __aicore__ void spsm_packed_solve(GM_ADDR offsets, GM_ADDR indices,
    GM_ADDR values, GM_ADDR dense, GM_ADDR alpha, GM_ADDR status, GM_ADDR first, GM_ADDR tail,
    int64_t lowRows, int32_t begin, int32_t end, SpsmA2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<int32_t> rp, ci, flags; GlobalTensor<float> av, scalar;
    rp.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(offsets));
    ci.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices));
    av.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(values));
    scalar.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(alpha));
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
    if (!PackedReady(flags, t.statusBlocks, t.packedFallback)) { return; }
    float ar = t.deviceAlpha ? scalar.GetValue(0) : t.alpha;
    float ai = t.deviceAlpha ? scalar.GetValue(1) : t.alphaImag;
    TPipe pipe; WordPlanes memory; memory.Init(dense, first, tail, lowRows, t.rhs, pipe);
    TBuf<TPosition::VECCALC> cache; pipe.InitBuffer(cache, 32 * 8 * 4);
    auto recent = cache.Get<uint64_t>(); uint32_t error = 0;
    for (int64_t column = GetBlockIdx(); !error && column < t.rhs; column += GetBlockNum()) {
        int32_t k = static_cast<int32_t>(column);
        for (int32_t i = 0; i < 32; ++i) { recent.SetValue(i * 4, UINT64_MAX); }
        for (int32_t step = begin; step < end; ++step) {
            int32_t row = t.upper ? t.rows - 1 - step : step;
            float br, bi; memory.Input(row, k, t.rows, br, bi);
            ScaledComplex current = TimesNormalized({FromFloat(br), FromFloat(bi)}, ar, ai);
            float dr = t.unit ? 1 : 0, di = 0;
            AccumulatePackedRow(rp, ci, av, recent, memory, t, row, k, current, dr, di, error);
            if (error) { break; }
            if (FlushSubnormal(dr) == 0 && FlushSubnormal(di) == 0) { error = 4; break; }
            if (!t.unit) { current = Divide(current, dr, di); }
            uint64_t re = Encode(current.real, error), im = Encode(current.imag, error);
            if (error) { break; }
            int32_t slot = (row % 32) * 4;
            recent.SetValue(slot + 1, re); recent.SetValue(slot + 2, im);
            recent.SetValue(slot, uint64_t(row)); memory.Store(row, k, re, im);
        }
        memory.Flush();
    }
    memory.Failure(flags, error);
}
extern "C" __global__ __aicore__ void spsm_packed_convert(GM_ADDR dense, GM_ADDR first,
    GM_ADDR tail, GM_ADDR status, int64_t lowRows, SpsmA2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<int32_t> flags; flags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
    if (!PackedReady(flags, t.statusBlocks, t.packedFallback)) { return; }
    TPipe pipe; WordPlanes memory; memory.Init(dense, first, tail, lowRows, t.rhs, pipe);
    for (int64_t i = GetBlockIdx(); i < int64_t(t.rows) * t.rhs; i += GetBlockNum()) {
        int32_t row = i / t.rhs, col = i % t.rhs; uint64_t re, im;
        memory.Load(row, col, re, im); memory.FloatOutput(row, col, re, im);
    }
}
extern "C" void SpsmA2PackedGuardLaunch(GM_ADDR offsets, GM_ADDR values, GM_ADDR dense,
    GM_ADDR alpha, GM_ADDR status, const SpsmA2TilingData &t, uint32_t blocks, void *stream)
{ spsm_packed_guard<<<blocks, nullptr, stream>>>(offsets, values, dense, alpha, status, t); }
extern "C" void SpsmA2PackedSolveLaunch(GM_ADDR offsets, GM_ADDR indices, GM_ADDR values,
    GM_ADDR dense, GM_ADDR alpha, GM_ADDR status, GM_ADDR first, GM_ADDR tail, int64_t lowRows,
    int32_t begin, int32_t end, const SpsmA2TilingData &t, uint32_t blocks, void *stream)
{ spsm_packed_solve<<<blocks, nullptr, stream>>>(offsets, indices, values, dense, alpha, status, first, tail, lowRows, begin, end, t); }
extern "C" void SpsmA2PackedConvertLaunch(GM_ADDR dense, GM_ADDR first, GM_ADDR tail,
    GM_ADDR status, int64_t lowRows, const SpsmA2TilingData &t, uint32_t blocks, void *stream)
{ spsm_packed_convert<<<blocks, nullptr, stream>>>(dense, first, tail, status, lowRows, t); }

extern "C" void SpsmA2PackedRetryLaunch(GM_ADDR dense, GM_ADDR first, GM_ADDR tail,
    GM_ADDR status, int64_t lowRows, const SpsmA2TilingData &t, uint32_t blocks, void *stream)
{ spsm_packed_retry<<<blocks, nullptr, stream>>>(dense, first, tail, status, lowRows, t); }
