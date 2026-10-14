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
#include "spsm_scan_sum.h"
using namespace AscendC;

// Format stage.
namespace {
constexpr int32_t kTile = 512;
class FormatExpand {
public:
    __aicore__ inline void Init(GM_ADDR first, GM_ADDR second, GM_ADDR values,
        GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outValues, GM_ADDR status,
        SpsmA2FormatTilingData t, TPipe *pipe)
    {
        t_ = t; parts_ = t.complex ? 2 : 1;
        first_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(first));
        second_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(second));
        values_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(values));
        rows_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(outRows));
        cols_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(outCols));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outValues));
        status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
        pipe->InitBuffer(major_, (kTile + 8) * 4);
        pipe->InitBuffer(minor_, kTile * 4);
        pipe->InitBuffer(data_, kTile * 8);
        pipe->InitBuffer(mask_, kTile * 8);
        pipe->InitBuffer(scratch_, kTile * 8);
        loadEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_S));
        scalarEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
        storeEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE3));
        reuseEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_MTE2));
    }
    __aicore__ inline void Validate()
    {
        int32_t code = 0;
        if (t_.format != 2) { code = ValidateOffsets(); }
        for (int64_t begin = static_cast<int64_t>(GetBlockIdx()) * kTile;
            begin < t_.nnz && code == 0; begin += static_cast<int64_t>(GetBlockNum()) * kTile) {
            int32_t count = t_.nnz - begin < kTile ? t_.nnz - begin : kTile;
            LoadIndices(begin, count);
            auto minor = minor_.Get<int32_t>(); auto major = major_.Get<int32_t>();
            for (int32_t p = 0; p < count; ++p) {
                int64_t col = static_cast<int64_t>(minor.GetValue(p)) - t_.base;
                int64_t row = t_.format == 2 ? static_cast<int64_t>(major.GetValue(p)) - t_.base : 0;
                if (col < 0 || col >= t_.rows || row < 0 || row >= t_.rows) { code = 1; }
            }
        }
        status_.SetValue(GetBlockIdx() * kSpsmA2StatusStride, code);
    }
    __aicore__ inline void Expand()
    {
        for (int32_t block = 0; block < t_.blocks; ++block) {
            if (status_.GetValue(block * kSpsmA2StatusStride) != 0) { return; }
        }
        InitMask();
        int32_t code = 0;
        for (int64_t begin = static_cast<int64_t>(GetBlockIdx()) * kTile;
            begin < t_.nnz && code == 0; begin += static_cast<int64_t>(GetBlockNum()) * kTile) {
            int32_t count = t_.nnz - begin < kTile ? t_.nnz - begin : kTile;
            Load(begin, count);
            code = Coordinates(begin, count);
            if (code == 0) { TransformAndStore(begin, count); }
        }
        status_.SetValue(GetBlockIdx() * kSpsmA2StatusStride, code);
    }
private:
    __aicore__ inline int32_t ValidateOffsets()
    {
        if (first_.GetValue(0) != t_.base ||
            static_cast<int64_t>(first_.GetValue(t_.rows)) != static_cast<int64_t>(t_.nnz) + t_.base) {
            return 1;
        }
        for (int64_t row = static_cast<int64_t>(GetBlockIdx()) * kTile; row < t_.rows;
            row += static_cast<int64_t>(GetBlockNum()) * kTile) {
            int32_t count = t_.rows - row < kTile ? t_.rows - row : kTile;
            auto offsets = major_.Get<int32_t>();
            DataCopyExtParams cp{1, static_cast<uint32_t>((count + 1) * 4), 0, 0, 0};
            DataCopyPad(offsets, first_[row], cp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(loadEvent_); WaitFlag<HardEvent::MTE2_S>(loadEvent_);
            for (int32_t p = 0; p < count; ++p) {
                int64_t begin = static_cast<int64_t>(offsets.GetValue(p)) - t_.base;
                int64_t end = static_cast<int64_t>(offsets.GetValue(p + 1)) - t_.base;
                if (begin < 0 || end < begin || end > t_.nnz) { return 1; }
            }
        }
        return 0;
    }
    __aicore__ inline void InitMask()
    {
        auto mask = mask_.Get<uint16_t>();
        for (int32_t lane = 0; lane < 16; ++lane) {
            mask.SetValue(lane, lane % 4 == 3 ? static_cast<uint16_t>(0x8000) : 0);
        }
        SetFlag<HardEvent::S_V>(scalarEvent_); WaitFlag<HardEvent::S_V>(scalarEvent_);
        for (int32_t block = 1; block < kTile / 4; ++block) {
            DataCopy(mask[block * 16], mask, 16);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadIndices(int64_t begin, int32_t count)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4), 0, 0, 0};
        DataCopyPad(minor_.Get<int32_t>(), second_[begin], cp,
            DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        if (t_.format == 2) {
            DataCopyPad(major_.Get<int32_t>(), first_[begin], cp,
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_S>(loadEvent_); WaitFlag<HardEvent::MTE2_S>(loadEvent_);
    }
    __aicore__ inline void Load(int64_t begin, int32_t count)
    {
        LoadIndices(begin, count);
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * parts_ * 4), 0, 0, 0};
        DataCopyPad(data_.Get<float>(), values_[begin * parts_], cp,
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(loadEvent_); WaitFlag<HardEvent::MTE2_S>(loadEvent_);
    }
    __aicore__ inline int32_t FindMajor(int64_t position)
    {
        // Compressed offsets are validated by the preceding kernel. Binary
        // search only the first entry of a tile; walk row boundaries thereafter.
        int32_t lo = 0, hi = t_.rows;
        while (lo < hi) {
            int32_t mid = lo + (hi - lo) / 2;
            if (static_cast<int64_t>(first_.GetValue(mid + 1)) - t_.base <= position) { lo = mid + 1; }
            else { hi = mid; }
        }
        return lo;
    }
    __aicore__ inline int32_t Coordinates(int64_t begin, int32_t count)
    {
        auto major = major_.Get<int32_t>(); auto minor = minor_.Get<int32_t>();
        int32_t row = t_.format == 2 ? 0 : FindMajor(begin);
        int64_t boundary = t_.format == 2 ? 0 : static_cast<int64_t>(first_.GetValue(row + 1)) - t_.base;
        for (int32_t p = 0; p < count; ++p) {
            if (t_.format == 2) { row = major.GetValue(p) - t_.base; }
            else {
                while (row < t_.rows && boundary <= begin + p) {
                    ++row;
                    if (row < t_.rows) { boundary = static_cast<int64_t>(first_.GetValue(row + 1)) - t_.base; }
                }
                // Irregular sparse address metadata only; values are batch DMA
                // copied and conjugated by SIMD below, never scalar-copied.
                major.SetValue(p, row + t_.base);
            }
            int32_t col = minor.GetValue(p) - t_.base;
            bool upper = t_.format == 1 ? !t_.upper : t_.upper;
            if (upper ? col < row : col > row) { return 1; }
        }
        return 0;
    }
    __aicore__ inline void Conjugate(int32_t count)
    {
        auto bits = data_.Get<uint16_t>(); auto mask = mask_.Get<uint16_t>();
        auto tmp = scratch_.Get<uint16_t>();
        Or(tmp, bits, mask, count * 4); PipeBarrier<PIPE_V>();
        And(bits, bits, mask, count * 4); PipeBarrier<PIPE_V>();
        Not(bits, bits, count * 4); PipeBarrier<PIPE_V>();
        And(bits, tmp, bits, count * 4); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void TransformAndStore(int64_t begin, int32_t count)
    {
        SetFlag<HardEvent::S_V>(scalarEvent_); WaitFlag<HardEvent::S_V>(scalarEvent_);
        Adds(major_.Get<int32_t>(), major_.Get<int32_t>(), -t_.base, count);
        Adds(minor_.Get<int32_t>(), minor_.Get<int32_t>(), -t_.base, count);
        if (t_.complex && t_.operation == 2) { Conjugate(count); }
        SetFlag<HardEvent::V_MTE3>(storeEvent_); WaitFlag<HardEvent::V_MTE3>(storeEvent_);
        bool swap = (t_.format == 1) != (t_.operation != 0);
        auto row = swap ? minor_.Get<int32_t>() : major_.Get<int32_t>();
        auto col = swap ? major_.Get<int32_t>() : minor_.Get<int32_t>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4), 0, 0, 0};
        DataCopyPad(rows_[begin], row, cp); DataCopyPad(cols_[begin], col, cp);
        cp.blockLen *= parts_;
        DataCopyPad(output_[begin * parts_], data_.Get<float>(), cp);
        SetFlag<HardEvent::MTE3_MTE2>(reuseEvent_); WaitFlag<HardEvent::MTE3_MTE2>(reuseEvent_);
    }
    SpsmA2FormatTilingData t_{};
    int32_t parts_{1};
    event_t loadEvent_{}, scalarEvent_{}, storeEvent_{}, reuseEvent_{};
    GlobalTensor<int32_t> first_, second_, rows_, cols_, status_;
    GlobalTensor<float> values_, output_;
    TBuf<TPosition::VECCALC> major_, minor_, data_, mask_, scratch_;
};
}
extern "C" __global__ __aicore__ void spsm_a2_format_validate(GM_ADDR first, GM_ADDR second,
    GM_ADDR values, GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outValues, GM_ADDR status,
    SpsmA2FormatTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; FormatExpand op;
    op.Init(first, second, values, outRows, outCols, outValues, status, tiling, &pipe); op.Validate();
}
extern "C" __global__ __aicore__ void spsm_a2_format_expand(GM_ADDR first, GM_ADDR second,
    GM_ADDR values, GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outValues, GM_ADDR status,
    SpsmA2FormatTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; FormatExpand op;
    op.Init(first, second, values, outRows, outCols, outValues, status, tiling, &pipe); op.Expand();
}
extern "C" void SpsmA2ExpandLaunch(GM_ADDR first, GM_ADDR second, GM_ADDR values,
    GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outValues, GM_ADDR status,
    const SpsmA2FormatTilingData &tiling, void *stream)
{
    spsm_a2_format_validate<<<tiling.blocks, nullptr, stream>>>(first, second, values, outRows, outCols, outValues, status, tiling);
    spsm_a2_format_expand<<<tiling.blocks, nullptr, stream>>>(first, second, values, outRows, outCols, outValues, status, tiling);
}

// Sort stage.
namespace {
class StableSort {
public:
    __aicore__ inline void Init(GM_ADDR rows, GM_ADDR cols, GM_ADDR values, GM_ADDR order,
        GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outValues, GM_ADDR outOrder, GM_ADDR status,
        SpsmA2SortTilingData t, TPipe *pipe)
    {
        t_ = t; parts_ = t.complex ? 2 : 1;
        rows_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rows));
        cols_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cols));
        order_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(order));
        values_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(values));
        outRows_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(outRows));
        outCols_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(outCols));
        outOrder_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(outOrder));
        outValues_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outValues));
        status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
        pipe->InitBuffer(rowIn_, kTile * 8); pipe->InitBuffer(colIn_, kTile * 8);
        pipe->InitBuffer(orderIn_, kTile * 8); pipe->InitBuffer(valueIn_, kTile * 16);
        pipe->InitBuffer(rowOut_, kTile * 4); pipe->InitBuffer(colOut_, kTile * 4);
        pipe->InitBuffer(orderOut_, kTile * 4); pipe->InitBuffer(valueOut_, kTile * 8);
        pipe->InitBuffer(indices_, kTile * 4); pipe->InitBuffer(valueIndices_, kTile * 8);
        loadEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_S));
        scalarEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
        vectorEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
        storeEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE3));
        reuseEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_MTE2));
    }
    __aicore__ inline void Process()
    {
        for (int32_t core = 0; core < t_.blocks; ++core) {
            if (status_.GetValue(core * kSpsmA2StatusStride) != 0) { return; }
        }
        for (int64_t begin = static_cast<int64_t>(GetBlockIdx()) * kTile; begin < t_.nnz;
            begin += static_cast<int64_t>(GetBlockNum()) * kTile) {
            int32_t count = Min(t_.nnz - begin, kTile);
            if (t_.run == 0) { SortTile(begin, count); }
            else { MergeTile(begin, count); }
            Store(begin, count);
        }
    }
private:
    __aicore__ inline int64_t Min(int64_t a, int64_t b) { return a < b ? a : b; }
    __aicore__ inline bool Less(int32_t ar, int32_t ac, int32_t br, int32_t bc)
    {
        return ar < br || (ar == br && ac < bc);
    }
    __aicore__ inline bool GlobalLess(int64_t a, int64_t b)
    {
        return Less(rows_.GetValue(a), cols_.GetValue(a), rows_.GetValue(b), cols_.GetValue(b));
    }
    __aicore__ inline bool LocalLess(int32_t a, int32_t b)
    {
        auto rows = rowIn_.Get<int32_t>(); auto cols = colIn_.Get<int32_t>();
        return Less(rows.GetValue(a), cols.GetValue(a), rows.GetValue(b), cols.GetValue(b));
    }
    __aicore__ inline void Load(int64_t begin, int32_t count, int32_t offset)
    {
        if (count == 0) { return; }
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4), 0, 0, 0};
        auto row = rowIn_.Get<int32_t>(); auto col = colIn_.Get<int32_t>(); auto order = orderIn_.Get<int32_t>();
        DataCopyPad(row[offset], rows_[begin], cp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        DataCopyPad(col[offset], cols_[begin], cp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        if (t_.run != 0) {
            DataCopyPad(order[offset], order_[begin], cp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
        cp.blockLen *= parts_;
        auto value = valueIn_.Get<float>();
        DataCopyPad(value[offset * parts_], values_[begin * parts_], cp,
            DataCopyPadExtParams<float>{false, 0, 0, 0});
    }
    __aicore__ inline void SetIndex(int32_t destination, int32_t source)
    {
        // Merge decisions produce address metadata. SIMD Gather performs all
        // coordinate, permutation and floating-point payload movement.
        indices_.Get<uint32_t>().SetValue(destination, source * 4);
        auto value = valueIndices_.Get<uint32_t>();
        value.SetValue(destination * parts_, source * parts_ * 4);
        if (parts_ == 2) { value.SetValue(destination * parts_ + 1, source * parts_ * 4 + 4); }
    }
    __aicore__ inline void MapMerge(int32_t a, int32_t aEnd, int32_t b, int32_t bEnd,
        int32_t destination, int32_t count)
    {
        for (int32_t p = 0; p < count; ++p) {
            bool useA = b == bEnd || (a < aEnd && !LocalLess(b, a));
            SetIndex(destination + p, useA ? a : b);
            if (useA) { ++a; } else { ++b; }
        }
    }
    __aicore__ inline void GatherOutput(int32_t count)
    {
        SetFlag<HardEvent::S_V>(scalarEvent_); WaitFlag<HardEvent::S_V>(scalarEvent_);
        auto indices = indices_.Get<uint32_t>();
        Gather(rowOut_.Get<int32_t>(), rowIn_.Get<int32_t>(), indices, 0, count);
        Gather(colOut_.Get<int32_t>(), colIn_.Get<int32_t>(), indices, 0, count);
        Gather(orderOut_.Get<int32_t>(), orderIn_.Get<int32_t>(), indices, 0, count);
        Gather(valueOut_.Get<float>(), valueIn_.Get<float>(), valueIndices_.Get<uint32_t>(), 0, count * parts_);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Feedback(int32_t count)
    {
        int32_t aligned = (count + 7) / 8 * 8;
        DataCopy(rowIn_.Get<int32_t>(), rowOut_.Get<int32_t>(), aligned);
        DataCopy(colIn_.Get<int32_t>(), colOut_.Get<int32_t>(), aligned);
        DataCopy(orderIn_.Get<int32_t>(), orderOut_.Get<int32_t>(), aligned);
        DataCopy(valueIn_.Get<float>(), valueOut_.Get<float>(), (count * parts_ + 7) / 8 * 8);
        SetFlag<HardEvent::V_S>(vectorEvent_); WaitFlag<HardEvent::V_S>(vectorEvent_);
    }
    __aicore__ inline void SortTile(int64_t begin, int32_t count)
    {
        Load(begin, count, 0);
        SetFlag<HardEvent::MTE2_S>(loadEvent_); WaitFlag<HardEvent::MTE2_S>(loadEvent_);
        auto order = orderIn_.Get<int32_t>();
        for (int32_t lane = 0; lane < 8; ++lane) { order.SetValue(lane, begin + lane); }
        SetFlag<HardEvent::S_V>(scalarEvent_); WaitFlag<HardEvent::S_V>(scalarEvent_);
        for (int32_t offset = 8; offset < count; offset += 8) { Adds(order[offset], order, offset, 8); }
        SetFlag<HardEvent::V_S>(vectorEvent_); WaitFlag<HardEvent::V_S>(vectorEvent_);
        if (count == 1) { SetIndex(0, 0); GatherOutput(count); return; }
        for (int32_t run = 1; run < count; run *= 2) {
            for (int32_t start = 0; start < count; start += run * 2) {
                int32_t mid = Min(start + run, count), end = Min(start + run * 2, count);
                MapMerge(start, mid, mid, end, start, end - start);
            }
            GatherOutput(count); Feedback(count);
        }
    }
    __aicore__ inline int64_t Partition(int64_t left, int64_t nLeft, int64_t right, int64_t nRight, int64_t rank)
    {
        int64_t lo = rank > nRight ? rank - nRight : 0, hi = Min(rank, nLeft);
        while (lo <= hi) {
            int64_t a = lo + (hi - lo) / 2, b = rank - a;
            if (a > 0 && b < nRight && GlobalLess(right + b, left + a - 1)) { hi = a - 1; }
            else if (b > 0 && a < nLeft && !GlobalLess(right + b - 1, left + a)) { lo = a + 1; }
            else { return a; }
        }
        return lo;
    }
    __aicore__ inline void MergeTile(int64_t begin, int32_t count)
    {
        int64_t group = begin / (2 * t_.run) * (2 * t_.run);
        int64_t nLeft = Min(t_.run, t_.nnz - group);
        int64_t right = group + nLeft, nRight = Min(t_.run, t_.nnz - right);
        int64_t a = Partition(group, nLeft, right, nRight, begin - group);
        int64_t b = begin - group - a;
        int32_t takeA = Min(nLeft - a, count), takeB = Min(nRight - b, count);
        Load(group + a, takeA, 0); Load(right + b, takeB, kTile);
        SetFlag<HardEvent::MTE2_S>(loadEvent_); WaitFlag<HardEvent::MTE2_S>(loadEvent_);
        MapMerge(0, takeA, kTile, kTile + takeB, 0, count);
        GatherOutput(count);
    }
    __aicore__ inline void Store(int64_t begin, int32_t count)
    {
        SetFlag<HardEvent::V_MTE3>(storeEvent_); WaitFlag<HardEvent::V_MTE3>(storeEvent_);
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4), 0, 0, 0};
        DataCopyPad(outRows_[begin], rowOut_.Get<int32_t>(), cp);
        DataCopyPad(outCols_[begin], colOut_.Get<int32_t>(), cp);
        DataCopyPad(outOrder_[begin], orderOut_.Get<int32_t>(), cp);
        cp.blockLen *= parts_; DataCopyPad(outValues_[begin * parts_], valueOut_.Get<float>(), cp);
        SetFlag<HardEvent::MTE3_MTE2>(reuseEvent_); WaitFlag<HardEvent::MTE3_MTE2>(reuseEvent_);
    }
    SpsmA2SortTilingData t_{}; int32_t parts_{1};
    event_t loadEvent_{}, scalarEvent_{}, vectorEvent_{}, storeEvent_{}, reuseEvent_{};
    GlobalTensor<int32_t> rows_, cols_, order_, outRows_, outCols_, outOrder_, status_;
    GlobalTensor<float> values_, outValues_;
    TBuf<TPosition::VECCALC> rowIn_, colIn_, orderIn_, valueIn_, rowOut_, colOut_, orderOut_, valueOut_;
    TBuf<TPosition::VECCALC> indices_, valueIndices_;
};
}
extern "C" __global__ __aicore__ void spsm_a2_stable_sort(GM_ADDR rows, GM_ADDR cols, GM_ADDR values,
    GM_ADDR order, GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outValues, GM_ADDR outOrder,
    GM_ADDR status, SpsmA2SortTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; StableSort op;
    op.Init(rows, cols, values, order, outRows, outCols, outValues, outOrder, status, tiling, &pipe);
    op.Process();
}
extern "C" void SpsmA2SortPassLaunch(GM_ADDR rows, GM_ADDR cols, GM_ADDR values, GM_ADDR order,
    GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outValues, GM_ADDR outOrder, GM_ADDR status,
    const SpsmA2SortTilingData &tiling, void *stream)
{
    spsm_a2_stable_sort<<<tiling.blocks, nullptr, stream>>>(rows, cols, values, order,
        outRows, outCols, outValues, outOrder, status, tiling);
}

// Canonical stage.
namespace {
constexpr int32_t kRowTile = 64;
class Canonical {
public:
    __aicore__ inline void Init(GM_ADDR rows, GM_ADDR cols, GM_ADDR values,
        GM_ADDR rowStarts, GM_ADDR rowCounts, GM_ADDR rowPtr, GM_ADDR outCols, GM_ADDR outValues,
        GM_ADDR groupStarts, GM_ADDR groupEnds, GM_ADDR status, SpsmA2CanonicalTilingData t, TPipe *pipe)
    {
        t_ = t; parts_ = t.complex ? 2 : 1;
        rows_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rows));
        cols_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cols));
        values_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(values));
        rowStarts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rowStarts));
        rowCounts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rowCounts));
        rowPtr_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rowPtr));
        outCols_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(outCols));
        outValues_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outValues));
        groupStarts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(groupStarts));
        groupEnds_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(groupEnds));
        status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
        pipe->InitBuffer(colsBuf_, kTile * 4);
        pipe->InitBuffer(rowMeta_, (kTile + 8) * 4); pipe->InitBuffer(countMeta_, kTile * 4);
        pipe->InitBuffer(acc_, kTile * 8); pipe->InitBuffer(prev_, kTile * 8);
        pipe->InitBuffer(low_, kTile * 8); pipe->InitBuffer(prevLow_, kTile * 8);
        pipe->InitBuffer(tmp_, kTile * 8);
        pipe->InitBuffer(mask_, kTile * 8); pipe->InitBuffer(indices_, kTile * 8);
        pipe->InitBuffer(out_, kTile * 8); pipe->InitBuffer(outColsBuf_, kTile * 4);
        pipe->InitBuffer(startBuf_, kTile * 4); pipe->InitBuffer(endBuf_, kTile * 4);
        pipe->InitBuffer(carry_, 32); pipe->InitBuffer(carryLow_, 32);
        scanSum_.Init(pipe);
        loadEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_S));
        scalarEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
        vectorEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
        scalarStore_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_MTE3));
        vectorStore_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE3));
        reuseEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_MTE2));
        reuseScalar_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_S));
    }
    __aicore__ inline void Count()
    {
        if (!Ready()) { return; }
        auto starts = rowMeta_.Get<int32_t>(); auto counts = countMeta_.Get<int32_t>();
        for (int64_t first = static_cast<int64_t>(GetBlockIdx()) * kRowTile; first < t_.rows;
            first += static_cast<int64_t>(GetBlockNum()) * kRowTile) {
            int32_t count = Min(t_.rows - first, kRowTile);
            for (int32_t p = 0; p < count; ++p) {
                int32_t begin = LowerBound(first + p), end = LowerBound(first + p + 1);
                starts.SetValue(p, begin); counts.SetValue(p, CountRow(begin, end));
            }
            SetFlag<HardEvent::S_MTE3>(scalarStore_); WaitFlag<HardEvent::S_MTE3>(scalarStore_);
            CopyMeta(rowStarts_, starts, first, count); CopyMeta(rowCounts_, counts, first, count);
            SetFlag<HardEvent::MTE3_S>(reuseScalar_); WaitFlag<HardEvent::MTE3_S>(reuseScalar_);
        }
    }
    __aicore__ inline void Prefix()
    {
        if (!Ready()) { return; }
        auto counts = countMeta_.Get<int32_t>(); auto offsets = rowMeta_.Get<int32_t>();
        int32_t total = 0;
        for (int64_t begin = 0; begin < t_.rows; begin += kTile) {
            int32_t count = Min(t_.rows - begin, kTile);
            LoadMeta(counts, rowCounts_, begin, count);
            offsets.SetValue(0, total);
            // Counts/row pointers are address metadata. The Device scan is
            // batched; no sparse values or solve arithmetic runs on the Host.
            for (int32_t p = 0; p < count; ++p) { total += counts.GetValue(p); offsets.SetValue(p + 1, total); }
            SetFlag<HardEvent::S_MTE3>(scalarStore_); WaitFlag<HardEvent::S_MTE3>(scalarStore_);
            CopyMeta(rowPtr_, offsets, begin, count + 1);
            SetFlag<HardEvent::MTE3_S>(reuseScalar_); WaitFlag<HardEvent::MTE3_S>(reuseScalar_);
        }
        if (t_.rows == 0) {
            offsets.SetValue(0, 0);
            SetFlag<HardEvent::S_MTE3>(scalarStore_); WaitFlag<HardEvent::S_MTE3>(scalarStore_);
            CopyMeta(rowPtr_, offsets, 0, 1);
            SetFlag<HardEvent::MTE3_S>(reuseScalar_); WaitFlag<HardEvent::MTE3_S>(reuseScalar_);
        }
        offsets.SetValue(0, t_.nnz);
        SetFlag<HardEvent::S_MTE3>(scalarStore_); WaitFlag<HardEvent::S_MTE3>(scalarStore_);
        CopyMeta(rowStarts_, offsets, t_.rows, 1);
    }
    __aicore__ inline void Reduce()
    {
        if (!Ready()) { return; }
        int32_t code = 0;
        for (int64_t row = GetBlockIdx(); row < t_.rows && code == 0; row += GetBlockNum()) { code = ReduceRow(row); }
        status_.SetValue(GetBlockIdx() * kSpsmA2StatusStride, code);
    }
private:
    __aicore__ inline int32_t Min(int64_t a, int32_t b) { return a < b ? a : b; }
    __aicore__ inline bool Ready()
    {
        for (int32_t core = 0; core < t_.blocks; ++core) {
            if (status_.GetValue(core * kSpsmA2StatusStride) != 0) { return false; }
        }
        return true;
    }
    __aicore__ inline int32_t LowerBound(int64_t row)
    {
        int64_t lo = 0, hi = t_.nnz;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (rows_.GetValue(mid) < row) { lo = mid + 1; } else { hi = mid; }
        }
        return lo;
    }
    __aicore__ inline void LoadMeta(LocalTensor<int32_t> dst, GlobalTensor<int32_t> src, int64_t begin, int32_t count)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4), 0, 0, 0};
        DataCopyPad(dst, src[begin], cp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(loadEvent_); WaitFlag<HardEvent::MTE2_S>(loadEvent_);
    }
    __aicore__ inline void CopyMeta(GlobalTensor<int32_t> dst, LocalTensor<int32_t> src, int64_t begin, int32_t count)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4), 0, 0, 0}; DataCopyPad(dst[begin], src, cp);
    }
    __aicore__ inline int32_t CountRow(int32_t begin, int32_t end)
    {
        int32_t count = 0, previous = -1; auto cols = colsBuf_.Get<int32_t>();
        for (int64_t offset = begin; offset < end; offset += kTile) {
            int32_t length = Min(end - offset, kTile); LoadMeta(cols, cols_, offset, length);
            for (int32_t p = 0; p < length; ++p) {
                int32_t col = cols.GetValue(p); count += col != previous; previous = col;
            }
        }
        return count;
    }
    __aicore__ inline void LoadValues(int64_t begin, int32_t count)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * parts_ * 4), 0, 0, 0};
        DataCopyPad(acc_.Get<float>(), values_[begin * parts_], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(loadEvent_); WaitFlag<HardEvent::MTE2_S>(loadEvent_);
    }
    __aicore__ inline bool BuildStep(int32_t count, int32_t step)
    {
        auto cols = colsBuf_.Get<int32_t>(); auto index = indices_.Get<uint32_t>(); auto mask = mask_.Get<int32_t>();
        bool combines = false;
        for (int32_t p = 0; p < count; ++p) {
            bool same = p >= step && cols.GetValue(p) == cols.GetValue(p - step);
            combines = combines || same;
            for (int32_t part = 0; part < parts_; ++part) {
                index.SetValue(p * parts_ + part, ((same ? p - step : p) * parts_ + part) * 4);
                mask.SetValue(p * parts_ + part, same ? -1 : 0);
            }
        }
        return combines;
    }
    __aicore__ inline void SelectSums(LocalTensor<float> target, LocalTensor<float> source, int32_t count)
    {
        auto mask = mask_.Get<uint16_t>(); auto tmp = tmp_.Get<uint16_t>();
        auto sum = source.ReinterpretCast<uint16_t>(); auto acc = target.ReinterpretCast<uint16_t>();
        And(sum, sum, mask, count * parts_ * 2); PipeBarrier<PIPE_V>();
        Not(tmp, mask, count * parts_ * 2); PipeBarrier<PIPE_V>();
        And(acc, acc, tmp, count * parts_ * 2); PipeBarrier<PIPE_V>();
        Or(acc, sum, acc, count * parts_ * 2); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ScanValues(int32_t count, bool continuation)
    {
        auto acc = acc_.Get<float>(); auto low = low_.Get<float>();
        Duplicate(low, 0.0f, count * parts_); PipeBarrier<PIPE_V>();
        if (continuation) {
            scanSum_.AddPair(carry_.Get<float>(), carryLow_.Get<float>(), acc, low, parts_);
            Duplicate(mask_.Get<int32_t>(), -1, parts_); PipeBarrier<PIPE_V>();
            SelectSums(acc, scanSum_.High(), 1); SelectSums(low, scanSum_.Low(), 1);
        }
        SetFlag<HardEvent::V_S>(vectorEvent_); WaitFlag<HardEvent::V_S>(vectorEvent_);
        for (int32_t step = 1; step < count; step *= 2) {
            if (!BuildStep(count, step)) { break; }
            SetFlag<HardEvent::S_V>(scalarEvent_); WaitFlag<HardEvent::S_V>(scalarEvent_);
            Gather(prev_.Get<float>(), acc, indices_.Get<uint32_t>(), 0, count * parts_); PipeBarrier<PIPE_V>();
            Gather(prevLow_.Get<float>(), low, indices_.Get<uint32_t>(), 0, count * parts_); PipeBarrier<PIPE_V>();
            scanSum_.AddPair(prev_.Get<float>(), prevLow_.Get<float>(), acc, low, count * parts_);
            SelectSums(acc, scanSum_.High(), count); SelectSums(low, scanSum_.Low(), count);
            SetFlag<HardEvent::V_S>(vectorEvent_); WaitFlag<HardEvent::V_S>(vectorEvent_);
        }
        auto index = indices_.Get<uint32_t>();
        for (int32_t part = 0; part < parts_; ++part) { index.SetValue(part, ((count - 1) * parts_ + part) * 4); }
        SetFlag<HardEvent::S_V>(scalarEvent_); WaitFlag<HardEvent::S_V>(scalarEvent_);
        Gather(carry_.Get<float>(), acc, index, 0, parts_);
        Gather(carryLow_.Get<float>(), low, index, 0, parts_);
        SetFlag<HardEvent::V_S>(vectorEvent_); WaitFlag<HardEvent::V_S>(vectorEvent_);
    }
    __aicore__ inline int32_t SelectGroups(int64_t begin, int32_t count, int32_t rowEnd, int32_t &groupStart)
    {
        auto cols = colsBuf_.Get<int32_t>(); auto index = indices_.Get<uint32_t>();
        auto outCols = outColsBuf_.Get<int32_t>(); auto starts = startBuf_.Get<int32_t>(); auto ends = endBuf_.Get<int32_t>();
        int32_t output = 0;
        for (int32_t p = 0; p < count; ++p) {
            int32_t next = p + 1 < count ? cols.GetValue(p + 1) : begin + count < rowEnd ? cols_.GetValue(begin + count) : -1;
            if (cols.GetValue(p) != next) {
                outCols.SetValue(output, cols.GetValue(p)); starts.SetValue(output, groupStart);
                ends.SetValue(output, begin + p + 1); groupStart = begin + p + 1;
                for (int32_t part = 0; part < parts_; ++part) { index.SetValue(output * parts_ + part, (p * parts_ + part) * 4); }
                ++output;
            }
        }
        return output;
    }
    __aicore__ inline bool StoreGroups(int32_t row, int32_t destination, int32_t count)
    {
        if (count == 0) { return false; }
        SetFlag<HardEvent::S_V>(scalarEvent_); WaitFlag<HardEvent::S_V>(scalarEvent_);
        Gather(out_.Get<float>(), acc_.Get<float>(), indices_.Get<uint32_t>(), 0, count * parts_);
        SetFlag<HardEvent::V_S>(vectorEvent_); WaitFlag<HardEvent::V_S>(vectorEvent_);
        bool nonzeroDiagonal = false; auto vals = out_.Get<float>(); auto cols = outColsBuf_.Get<int32_t>();
        for (int32_t p = 0; p < count; ++p) {
            if (cols.GetValue(p) == row) {
                nonzeroDiagonal = vals.GetValue(p * parts_) != 0 || (parts_ == 2 && vals.GetValue(p * parts_ + 1) != 0);
            }
        }
        SetFlag<HardEvent::V_MTE3>(vectorStore_); WaitFlag<HardEvent::V_MTE3>(vectorStore_);
        SetFlag<HardEvent::S_MTE3>(scalarStore_); WaitFlag<HardEvent::S_MTE3>(scalarStore_);
        CopyMeta(outCols_, cols, destination, count); CopyMeta(groupStarts_, startBuf_.Get<int32_t>(), destination, count);
        CopyMeta(groupEnds_, endBuf_.Get<int32_t>(), destination, count);
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * parts_ * 4), 0, 0, 0};
        DataCopyPad(outValues_[static_cast<int64_t>(destination) * parts_], vals, cp);
        SetFlag<HardEvent::MTE3_MTE2>(reuseEvent_); WaitFlag<HardEvent::MTE3_MTE2>(reuseEvent_);
        return nonzeroDiagonal;
    }
    __aicore__ inline int32_t ReduceRow(int32_t row)
    {
        int32_t begin = rowStarts_.GetValue(row), end = row + 1 < t_.rows ? rowStarts_.GetValue(row + 1) : t_.nnz;
        int32_t destination = rowPtr_.GetValue(row), groupStart = begin, previous = -1;
        bool diagonal = t_.unit != 0;
        for (int64_t offset = begin; offset < end; offset += kTile) {
            int32_t count = Min(end - offset, kTile);
            LoadMeta(colsBuf_.Get<int32_t>(), cols_, offset, count); LoadValues(offset, count);
            bool continuation = previous == colsBuf_.Get<int32_t>().GetValue(0);
            ScanValues(count, continuation); previous = colsBuf_.Get<int32_t>().GetValue(count - 1);
            int32_t groups = SelectGroups(offset, count, end, groupStart);
            diagonal = StoreGroups(row, destination, groups) || diagonal; destination += groups;
        }
        return diagonal ? 0 : 2;
    }
    SpsmA2CanonicalTilingData t_{}; int32_t parts_{1};
    event_t loadEvent_{}, scalarEvent_{}, vectorEvent_{}, scalarStore_{}, vectorStore_{}, reuseEvent_{}, reuseScalar_{};
    GlobalTensor<int32_t> rows_, cols_, rowStarts_, rowCounts_, rowPtr_, outCols_, groupStarts_, groupEnds_, status_;
    GlobalTensor<float> values_, outValues_;
    spsm_a2::ScanSum<kTile * 2> scanSum_;
    TBuf<TPosition::VECCALC> colsBuf_, rowMeta_, countMeta_, acc_, prev_, low_, prevLow_, tmp_, mask_, indices_, out_, outColsBuf_, startBuf_, endBuf_, carry_, carryLow_;
};
}
#define SPSM_CANONICAL_ARGS GM_ADDR rows, GM_ADDR cols, GM_ADDR values, GM_ADDR rowStarts, GM_ADDR rowCounts, GM_ADDR rowPtr, GM_ADDR outCols, GM_ADDR outValues, GM_ADDR groupStarts, GM_ADDR groupEnds, GM_ADDR status, SpsmA2CanonicalTilingData tiling
#define SPSM_CANONICAL_INIT op.Init(rows, cols, values, rowStarts, rowCounts, rowPtr, outCols, outValues, groupStarts, groupEnds, status, tiling, &pipe)
extern "C" __global__ __aicore__ void spsm_a2_canonical_count(SPSM_CANONICAL_ARGS)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); TPipe pipe; Canonical op; SPSM_CANONICAL_INIT; op.Count();
}
extern "C" __global__ __aicore__ void spsm_a2_canonical_prefix(SPSM_CANONICAL_ARGS)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); TPipe pipe; Canonical op; SPSM_CANONICAL_INIT; op.Prefix();
}
extern "C" __global__ __aicore__ void spsm_a2_canonical_reduce(SPSM_CANONICAL_ARGS)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); TPipe pipe; Canonical op; SPSM_CANONICAL_INIT; op.Reduce();
}
#undef SPSM_CANONICAL_ARGS
#undef SPSM_CANONICAL_INIT
extern "C" void SpsmA2CanonicalLaunch(GM_ADDR rows, GM_ADDR cols, GM_ADDR values,
    GM_ADDR rowStarts, GM_ADDR rowCounts, GM_ADDR rowPtr, GM_ADDR outCols, GM_ADDR outValues,
    GM_ADDR groupStarts, GM_ADDR groupEnds, GM_ADDR status,
    const SpsmA2CanonicalTilingData &tiling, void *stream)
{
    spsm_a2_canonical_count<<<tiling.blocks, nullptr, stream>>>(rows, cols, values, rowStarts, rowCounts, rowPtr, outCols, outValues, groupStarts, groupEnds, status, tiling);
    spsm_a2_canonical_prefix<<<1, nullptr, stream>>>(rows, cols, values, rowStarts, rowCounts, rowPtr, outCols, outValues, groupStarts, groupEnds, status, tiling);
    spsm_a2_canonical_reduce<<<tiling.blocks, nullptr, stream>>>(rows, cols, values, rowStarts, rowCounts, rowPtr, outCols, outValues, groupStarts, groupEnds, status, tiling);
}

// Levels stage.
namespace {
class DependencyLevels {
public:
    __aicore__ inline void Init(GM_ADDR offsets, GM_ADDR columns, GM_ADDR order,
        GM_ADDR levelOffsets, int32_t rows, int32_t upper, TPipe *pipe)
    {
        rows_ = rows; upper_ = upper;
        rp_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(offsets));
        cols_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(columns));
        orderOut_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(order));
        offsetsOut_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(levelOffsets));
        pipe->InitBuffer(levels_, (kSpsmA2LevelRows + 8) * 4);
        pipe->InitBuffer(counts_, (kSpsmA2LevelRows + 8) * 4);
        pipe->InitBuffer(offsets_, (kSpsmA2LevelRows + 8) * 4);
        pipe->InitBuffer(order_, kSpsmA2LevelRows * 4);
        vectorToScalar_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
        scalarToStore_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_MTE3));
        storeToScalar_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_S));
    }
    __aicore__ inline void Process()
    {
        auto counts = counts_.Get<int32_t>();
        auto offsets = offsets_.Get<int32_t>();
        auto levels = levels_.Get<int32_t>();
        auto order = order_.Get<int32_t>();
        Duplicate(counts, 0, kSpsmA2LevelRows + 8);
        Duplicate(offsets, 0, kSpsmA2LevelRows + 8);
        SetFlag<HardEvent::V_S>(vectorToScalar_); WaitFlag<HardEvent::V_S>(vectorToScalar_);
        int32_t depth = 0;
        for (int32_t step = 0; step < rows_; ++step) {
            int32_t row = upper_ ? rows_ - 1 - step : step;
            int32_t level = 0;
            for (int32_t p = rp_.GetValue(row); p < rp_.GetValue(row + 1); ++p) {
                int32_t col = cols_.GetValue(p);
                if (col != row) {
                    int32_t candidate = levels.GetValue(col) + 1;
                    if (candidate > level) { level = candidate; }
                }
            }
            levels.SetValue(row, level);
            counts.SetValue(level, counts.GetValue(level) + 1);
            if (level + 1 > depth) { depth = level + 1; }
        }
        int32_t total = 0;
        for (int32_t level = 0; level < depth; ++level) {
            int32_t count = counts.GetValue(level);
            offsets.SetValue(level, total);
            counts.SetValue(level, total);
            total += count;
        }
        offsets.SetValue(depth, total);
        for (int32_t row = 0; row < rows_; ++row) {
            int32_t level = levels.GetValue(row);
            int32_t position = counts.GetValue(level);
            order.SetValue(position, row);
            counts.SetValue(level, position + 1);
        }
        FinalizePhases(depth);
    }
private:
    __aicore__ inline void FinalizePhases(int32_t depth)
    {
        auto counts = counts_.Get<int32_t>();
        auto offsets = offsets_.Get<int32_t>();
        auto order = order_.Get<int32_t>();
        // Keep wide levels parallel, but combine consecutive narrow levels into
        // one serial phase. A long dependency chain needs one launch, not one per row.
        // counts now holds the end offset of each level after the stable scatter.
        int32_t phases = 0, level = 0;
        uint32_t parallelLow = 0, parallelHigh = 0;
        offsets.SetValue(0, 0);
        while (level < depth) {
            if (phases == kSpsmA2MaxLevels) { phases = 0; break; }
            int32_t width = counts.GetValue(level) - (level ? counts.GetValue(level - 1) : 0);
            bool parallel = width >= 8;
            int32_t end = level + 1;
            if (!parallel) {
                while (end < depth && counts.GetValue(end) - counts.GetValue(end - 1) < 8) { ++end; }
            } else if (phases < 32) {
                parallelLow |= uint32_t(1) << phases;
            } else {
                parallelHigh |= uint32_t(1) << (phases - 32);
            }
            offsets.SetValue(phases + 1, counts.GetValue(end - 1));
            ++phases; level = end;
        }
        if (phases == 1 && parallelLow == 0) { phases = 0; }
        offsets.SetValue(rows_ + 1, phases);
        offsets.SetValue(rows_ + 2, static_cast<int32_t>(parallelLow));
        offsets.SetValue(rows_ + 3, static_cast<int32_t>(parallelHigh));
        SetFlag<HardEvent::S_MTE3>(scalarToStore_); WaitFlag<HardEvent::S_MTE3>(scalarToStore_);
        DataCopyExtParams cp{1, static_cast<uint32_t>(rows_ * 4), 0, 0, 0};
        DataCopyPad(orderOut_, order, cp);
        cp.blockLen = (rows_ + 4) * 4;
        DataCopyPad(offsetsOut_, offsets, cp);
        SetFlag<HardEvent::MTE3_S>(storeToScalar_); WaitFlag<HardEvent::MTE3_S>(storeToScalar_);
    }
    int32_t rows_{}, upper_{};
    event_t vectorToScalar_{}, scalarToStore_{}, storeToScalar_{};
    GlobalTensor<int32_t> rp_, cols_, orderOut_, offsetsOut_;
    TBuf<TPosition::VECCALC> levels_, counts_, offsets_, order_;
};
}
extern "C" __global__ __aicore__ void spsm_a2_dependency_levels(GM_ADDR offsets, GM_ADDR columns,
    GM_ADDR order, GM_ADDR levelOffsets, int32_t rows, int32_t upper)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; DependencyLevels op;
    op.Init(offsets, columns, order, levelOffsets, rows, upper, &pipe);
    op.Process();
}
extern "C" void SpsmA2LevelsLaunch(GM_ADDR offsets, GM_ADDR columns, GM_ADDR order,
    GM_ADDR levelOffsets, int32_t rows, int32_t upper, void *stream)
{
    spsm_a2_dependency_levels<<<1, nullptr, stream>>>(offsets, columns, order, levelOffsets, rows, upper);
}
