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
#include "spsm_math.h"
using namespace AscendC;

// Spsm stage.
namespace {
// CSR FP32/complex64 core. Canonicalization and layout dispatch are added separately.
// Each core owns complete RHS columns, eliminating inter-core dependency waits.
// A row's output reaches GM before subsequent rows read it through MTE2.
template<bool Complex> class SpsmA2Kernel {
    static constexpr int32_t kParts = Complex ? 2 : 1;
    static constexpr int32_t kBatchLanes = 512;
    static constexpr int32_t kBatchTerms = 128;
    using Scalar = spsm_a2::Scalar;
public:
    __aicore__ inline void Init(GM_ADDR offsets, GM_ADDR indices, GM_ADDR values,
        GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR status, SpsmA2TilingData t, TPipe *pipe,
        GM_ADDR lowFirst, GM_ADDR lowTail, int64_t lowRows, GM_ADDR levelOrder, GM_ADDR levelOffsets)
    {
        t_ = t; pipe_ = pipe;
        lowFirst_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lowFirst));
        lowTail_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lowTail));
        lowRows_ = lowRows;
        levelOrder_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(levelOrder));
        levelOffsets_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(levelOffsets));
        rp_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(offsets));
        ci_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices));
        av_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(values));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        c_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c));
        alpha_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(alpha));
        status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
        pipe_->InitBuffer(indicesBuf_, kSpsmA2NnzChunk * sizeof(int32_t));
        pipe_->InitBuffer(valuesBuf_, kSpsmA2NnzChunk * sizeof(float) * kParts);
        math_.Init(pipe_);
        if (t_.independentMode != 1) {
            batchMath_.Init(pipe_);
            pipe_->InitBuffer(batchHigh_, kBatchTerms * 16 * sizeof(float));
            pipe_->InitBuffer(batchLow_, kBatchTerms * 16 * sizeof(float));
            pipe_->InitBuffer(compactHigh_, kBatchLanes * kParts * sizeof(float));
            pipe_->InitBuffer(compactLow_, kBatchLanes * kParts * sizeof(float));
            pipe_->InitBuffer(coeffReal_, kBatchLanes * sizeof(float));
            pipe_->InitBuffer(coeffImag_, kBatchLanes * sizeof(float));
            pipe_->InitBuffer(batchMaps_, 5 * kBatchLanes * sizeof(uint32_t));
        }
        batchReady_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        batchReuse_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        mapReady_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_V));
        mapReuse_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        pipe_->InitBuffer(in_, 1, kSpsmA2RhsTile * sizeof(float) * kParts);
        pipe_->InitBuffer(out_, 1, kSpsmA2RhsTile * sizeof(float) * kParts);
        metadataEvent_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
        outputEvent_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        outputReuseEvent_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void Solve()
    {
        if (!spsm_a2::PrepareSolve<Complex>(t_, status_, alpha_)) { return; }
        const int32_t tiles = (t_.rhs - 1) / kSpsmA2SolveColumns + 1;
        if (t_.independentMode == 1) {
            // Each eligible row contains only its diagonal (or is empty/unit).
            // Its RHS lanes are independent, so use the existing full-width
            // arithmetic buffers without changing any sparse reduction tree.
            const int32_t independentTiles = (t_.rhs - 1) / kSpsmA2RhsTile + 1;
            int64_t units = static_cast<int64_t>(t_.rows) * independentTiles;
            for (int64_t unit = GetBlockIdx(); unit < units; unit += GetBlockNum()) {
                int32_t row = unit / independentTiles;
                int32_t start = (unit % independentTiles) * kSpsmA2RhsTile;
                if (Independent(row)) { SolveRow(row, start, Min(t_.rhs - start, kSpsmA2RhsTile)); }
            }
            return;
        }
        if (t_.independentMode == 3) {
            const int32_t first = levelOffsets_.GetValue(t_.level);
            const int32_t end = levelOffsets_.GetValue(t_.level + 1);
            const int64_t units = static_cast<int64_t>(end - first) * tiles;
            int32_t previousCount = -1;
            for (int64_t unit = GetBlockIdx(); unit < units; unit += GetBlockNum()) {
                const int32_t row = levelOrder_.GetValue(first + unit / tiles);
                const int32_t start = (unit % tiles) * kSpsmA2SolveColumns;
                const int32_t count = Min(t_.rhs - start, kSpsmA2SolveColumns);
                if (count != previousCount) { InitBatchMaps(count); previousCount = count; }
                SolveRow(row, start, count);
            }
            return;
        }
        if (t_.independentMode == 4) {
            const int32_t first = levelOffsets_.GetValue(t_.level);
            const int32_t end = levelOffsets_.GetValue(t_.level + 1);
            for (int32_t tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum()) {
                const int32_t start = tile * kSpsmA2SolveColumns;
                const int32_t count = Min(t_.rhs - start, kSpsmA2SolveColumns);
                InitBatchMaps(count);
                for (int32_t position = first; position < end; ++position) {
                    SolveRow(levelOrder_.GetValue(position), start, count);
                }
            }
            return;
        }
        for (int32_t tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum()) {
            const int32_t start = tile * kSpsmA2SolveColumns;
            const int32_t count = Min(t_.rhs - start, kSpsmA2SolveColumns);
            InitBatchMaps(count);
            for (int32_t step = 0; step < t_.rows; ++step) {
                int32_t row = t_.upper ? t_.rows - 1 - step : step;
                if (t_.independentMode != 2 || !Independent(row)) { SolveRow(row, start, count); }
            }
        }
    }
private:
    __aicore__ inline int32_t Min(int32_t a, int32_t b) { return a < b ? a : b; }
    __aicore__ inline bool Independent(int32_t row)
    {
        int32_t first = rp_.GetValue(row) - t_.base;
        int32_t end = rp_.GetValue(row + 1) - t_.base;
        return (end == first && t_.unit) ||
            (end == first + 1 && ci_.GetValue(first) - t_.base == row);
    }

    __aicore__ inline void InitBatchMaps(int32_t rhs)
    {
        if (rhs <= 0) { return; }
        // Parallel row scheduling can alternate full RHS tiles and a short tail
        // on one core. Finish the previous gathers before scalar writes remap UB.
        SetFlag<HardEvent::V_S>(mapReuse_);
        WaitFlag<HardEvent::V_S>(mapReuse_);
        batchCapacity_ = 1;
        while (batchCapacity_ * 2 <= kBatchTerms && batchCapacity_ * 2 * rhs <= kBatchLanes) {
            batchCapacity_ *= 2;
        }
        batchPitch_ = (rhs * kParts + 7) / 8 * 8;
        auto maps = batchMaps_.Get<uint32_t>();
        for (int32_t term = 0; term < batchCapacity_; ++term) {
            for (int32_t lane = 0; lane < rhs * kParts; ++lane) {
                maps.SetValue(term * rhs * kParts + lane, (term * batchPitch_ + lane) * 4);
            }
            for (int32_t lane = 0; lane < rhs; ++lane) {
                int32_t index = term * rhs + lane;
                maps.SetValue(2 * kBatchLanes + index, term * kParts * 4);
                maps.SetValue(3 * kBatchLanes + index, (2 * term * rhs + lane) * 4);
                maps.SetValue(4 * kBatchLanes + index, ((2 * term + 1) * rhs + lane) * 4);
            }
        }
        SetFlag<HardEvent::S_V>(mapReady_);
        WaitFlag<HardEvent::S_V>(mapReady_);
    }

    __aicore__ inline void SubtractTerms(int32_t first, int32_t terms, int32_t start, int32_t rhs)
    {
        // TBuf does not provide queue ownership: finish every vector read before
        // reusing its storage for the next batch of DMA loads.
        SetFlag<HardEvent::V_MTE2>(batchReuse_);
        WaitFlag<HardEvent::V_MTE2>(batchReuse_);
        auto high = batchHigh_.Get<float>();
        auto low = batchLow_.Get<float>();
        auto cols = indicesBuf_.Get<int32_t>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(rhs * kParts * 4), 0, 0, 0};
        for (int32_t i = 0; i < terms; ++i) {
            int32_t col = cols.GetValue(first + i) - t_.base;
            int64_t offset = static_cast<int64_t>(col) * t_.ldc + start;
            DataCopyPad(high[i * batchPitch_], c_[offset * kParts], cp,
                DataCopyPadExtParams<float>{false, 0, 0, 0});
            if (col < lowRows_) {
                DataCopyPad(low[i * batchPitch_], lowFirst_[offset * kParts], cp,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
            } else {
                offset = (col - lowRows_) * t_.ldc + start;
                DataCopyPad(low[i * batchPitch_], lowTail_[offset * kParts], cp,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
        }
        SetFlag<HardEvent::MTE2_V>(batchReady_);
        WaitFlag<HardEvent::MTE2_V>(batchReady_);
        auto maps = batchMaps_.Get<uint32_t>();
        int32_t lanes = terms * rhs;
        Gather(compactHigh_.Get<float>(), high, maps, 0, lanes * kParts);
        Gather(compactLow_.Get<float>(), low, maps, 0, lanes * kParts);
        Gather(coeffReal_.Get<float>(), valuesBuf_.Get<float>(), maps[2 * kBatchLanes],
            first * kParts * 4, lanes);
        if constexpr (Complex) {
            Gather(coeffImag_.Get<float>(), valuesBuf_.Get<float>(), maps[2 * kBatchLanes],
                first * kParts * 4 + 4, lanes);
        }
        PipeBarrier<PIPE_V>();
        batchMath_.NegativeProducts(compactHigh_.Get<float>(), compactLow_.Get<float>(),
            coeffReal_.Get<float>(), coeffImag_.Get<float>(), lanes);
        batchMath_.FoldTerms(maps[3 * kBatchLanes], maps[4 * kBatchLanes], terms, rhs);
        for (int32_t part = 0; part < kParts; ++part) {
            math_.AddReduced(part, batchMath_.High(part), batchMath_.Low(part), rhs);
        }
    }

    __aicore__ inline void LoadEntries(int32_t first, int32_t count)
    {
        SetFlag<HardEvent::V_MTE2>(batchReuse_);
        WaitFlag<HardEvent::V_MTE2>(batchReuse_);
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4), 0, 0, 0};
        DataCopyPad(indicesBuf_.Get<int32_t>(), ci_[first], cp,
            DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        cp.blockLen *= kParts;
        DataCopyPad(valuesBuf_.Get<float>(), av_[static_cast<int64_t>(first) * kParts], cp,
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(metadataEvent_);
        WaitFlag<HardEvent::MTE2_S>(metadataEvent_);
    }

    __aicore__ inline LocalTensor<float> LoadDense(GlobalTensor<float> &matrix, int64_t offset, int32_t count)
    {
        auto input = in_.AllocTensor<float>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4 * kParts), 0, 0, 0};
        DataCopyPad(input, matrix[offset * kParts], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        in_.EnQue(input);
        return in_.DeQue<float>();
    }

    __aicore__ inline void SubtractChunk(int32_t row, int32_t start, int32_t rhs,
        int32_t count, Scalar &diag)
    {
        auto cols = indicesBuf_.Get<int32_t>();
        auto vals = valuesBuf_.Get<float>();
        for (int32_t p = 0; p < count; ++p) {
            int32_t col = cols.GetValue(p) - t_.base;
            Scalar value = ReadValue(vals, p);
            if (col == row) {
                diag.real += value.real; diag.imag += value.imag;
            } else {
                int32_t available = Min(count - p, batchCapacity_);
                for (int32_t i = 1; i < available; ++i) {
                    if (cols.GetValue(p + i) - t_.base == row) { available = i; break; }
                }
                int32_t terms = 1;
                while (terms * 2 <= available) { terms *= 2; }
                if (terms >= 4) {
                    SubtractTerms(p, terms, start, rhs);
                    p += terms - 1;
                    continue;
                }
                auto prev = LoadDense(c_, static_cast<int64_t>(col) * t_.ldc + start, rhs);
                math_.ReadHigh(prev, rhs);
                in_.FreeTensor(prev);
                auto low = LoadLow(col, start, rhs);
                math_.Subtract(low, value, rhs);
                in_.FreeTensor(low);
            }
        }
    }

    __aicore__ inline void SolveRow(int32_t row, int32_t start, int32_t count)
    {
        auto input = LoadDense(b_, static_cast<int64_t>(row) * t_.ldb + start, count);
        math_.Begin(input, {t_.alpha, t_.alphaImag}, count);
        in_.FreeTensor(input);
        int32_t first = rp_.GetValue(row) - t_.base;
        int32_t end = rp_.GetValue(row + 1) - t_.base;
        Scalar diag{};
        while (first < end) {
            int32_t nz = Min(end - first, kSpsmA2NnzChunk);
            LoadEntries(first, nz);
            SubtractChunk(row, start, count, nz, diag);
            first += nz;
        }
        if (!t_.unit) {
            // Device Analysis validated the same immutable structure/values.
            math_.Divide(diag, count);
        } else if constexpr (Complex) {
            math_.DivideUnit(count);
        }
        StoreRow(row, start, count);
    }

    __aicore__ inline LocalTensor<float> LoadLow(int32_t row, int32_t start, int32_t count)
    {
        if (row < lowRows_) {
            return LoadDense(lowFirst_, static_cast<int64_t>(row) * t_.ldc + start, count);
        }
        return LoadDense(lowTail_, (row - lowRows_) * t_.ldc + start, count);
    }

    __aicore__ inline void StoreLow(int32_t row, int32_t start, int32_t count)
    {
        auto output = out_.AllocTensor<float>();
        math_.PackLow(output, count);
        out_.EnQue(output);
        output = out_.DeQue<float>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4 * kParts), 0, 0, 0};
        if (row < lowRows_) {
            DataCopyPad(lowFirst_[(static_cast<int64_t>(row) * t_.ldc + start) * kParts], output, cp);
        } else {
            DataCopyPad(lowTail_[((row - lowRows_) * t_.ldc + start) * kParts], output, cp);
        }
        out_.FreeTensor(output);
    }

    __aicore__ inline void StoreRow(int32_t row, int32_t start, int32_t count)
    {
        auto output = out_.AllocTensor<float>();
        math_.Pack(output, count);
        out_.EnQue(output);
        output = out_.DeQue<float>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4 * kParts), 0, 0, 0};
        DataCopyPad(c_[(static_cast<int64_t>(row) * t_.ldc + start) * kParts], output, cp);
        out_.FreeTensor(output);
        SetFlag<HardEvent::MTE3_V>(outputReuseEvent_);
        WaitFlag<HardEvent::MTE3_V>(outputReuseEvent_);
        StoreLow(row, start, count);
        SetFlag<HardEvent::MTE3_MTE2>(outputEvent_);
        WaitFlag<HardEvent::MTE3_MTE2>(outputEvent_);
    }

    __aicore__ inline Scalar ReadValue(LocalTensor<float> vals, int32_t position)
    {
        Scalar value{vals.GetValue(position * kParts), 0};
        if constexpr (Complex) { value.imag = vals.GetValue(position * kParts + 1); }
        return value;
    }

    spsm_a2::CompensatedMath<Complex> math_;
    spsm_a2::CompensatedMath<Complex, kBatchLanes> batchMath_;
    SpsmA2TilingData t_{};
    TPipe *pipe_{nullptr};
    event_t metadataEvent_{}, outputEvent_{}, outputReuseEvent_{};
    event_t batchReady_{}, batchReuse_{}, mapReady_{}, mapReuse_{};
    int32_t batchCapacity_{}, batchPitch_{};
    TBuf<TPosition::VECCALC> batchHigh_, batchLow_, compactHigh_, compactLow_, coeffReal_, coeffImag_, batchMaps_;
    GlobalTensor<int32_t> rp_, ci_, status_, levelOrder_, levelOffsets_;
    GlobalTensor<float> av_, b_, c_, alpha_, lowFirst_, lowTail_;
    int64_t lowRows_{0};
    TBuf<TPosition::VECCALC> indicesBuf_, valuesBuf_;
    TQue<TPosition::VECIN, 1> in_;
    TQue<TPosition::VECOUT, 1> out_;
};
}

extern "C" __global__ __aicore__ void spsm_a2_compensated_fp32(GM_ADDR offsets, GM_ADDR indices,
    GM_ADDR values, GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR status, GM_ADDR lowFirst, GM_ADDR lowTail,
    int64_t lowRows, GM_ADDR levelOrder, GM_ADDR levelOffsets, SpsmA2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; SpsmA2Kernel<false> op;
    op.Init(offsets, indices, values, b, c, alpha, status, tiling, &pipe, lowFirst, lowTail, lowRows, levelOrder, levelOffsets);
    op.Solve();
}

extern "C" __global__ __aicore__ void spsm_a2_compensated_complex64(GM_ADDR offsets, GM_ADDR indices,
    GM_ADDR values, GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR status, GM_ADDR lowFirst, GM_ADDR lowTail,
    int64_t lowRows, GM_ADDR levelOrder, GM_ADDR levelOffsets, SpsmA2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; SpsmA2Kernel<true> op;
    op.Init(offsets, indices, values, b, c, alpha, status, tiling, &pipe, lowFirst, lowTail, lowRows, levelOrder, levelOffsets);
    op.Solve();
}
extern "C" void SpsmA2CompensatedLaunch(GM_ADDR offsets, GM_ADDR indices, GM_ADDR values,
    GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR status, GM_ADDR lowFirst, GM_ADDR lowTail,
    int64_t lowRows, GM_ADDR levelOrder, GM_ADDR levelOffsets,
    const SpsmA2TilingData &tiling, uint32_t blocks, void *stream)
{
    if (tiling.complex) {
        spsm_a2_compensated_complex64<<<blocks, nullptr, stream>>>(offsets, indices, values, b, c, alpha,
            status, lowFirst, lowTail, lowRows, levelOrder, levelOffsets, tiling);
    } else {
        spsm_a2_compensated_fp32<<<blocks, nullptr, stream>>>(offsets, indices, values, b, c, alpha,
            status, lowFirst, lowTail, lowRows, levelOrder, levelOffsets, tiling);
    }
}

// Diagonal stage.
namespace {
// Shortcut for a canonical matrix whose every row holds exactly its diagonal entry.
// Such a solve carries no row-to-row dependency, so rows parallelise across cores
// instead of the general kernel's single RHS tile. The per-row arithmetic is the
// compensated product and reciprocal refinement used by the general kernel.
// FP32 packs several rows into adjacent lanes to amortize instruction overhead.
// No low-word store: nothing reads another row's compensation residual here.
template<bool Complex> class SpsmA2DiagonalKernel {
    static constexpr int32_t kParts = Complex ? 2 : 1;
    static constexpr int32_t kLanes = Complex ? kSpsmA2RhsTile : 256;
    using Scalar = spsm_a2::Scalar;
public:
    __aicore__ inline void Init(GM_ADDR offsets, GM_ADDR values,
        GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR status, SpsmA2TilingData t, TPipe *pipe)
    {
        t_ = t; pipe_ = pipe;
        rp_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(offsets));
        av_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(values));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        c_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c));
        alpha_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(alpha));
        status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
        math_.Init(pipe_);
        if constexpr (!Complex) {
            pipe_->InitBuffer(diagonalRaw_, kLanes * sizeof(float));
            pipe_->InitBuffer(diagonalLanes_, kLanes * sizeof(float));
            pipe_->InitBuffer(inverseLanes_, kLanes * sizeof(float));
            diagonalReady_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
            diagonalReuse_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE2));
            scalarReady_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_V));
            scalarReuse_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        }
        pipe_->InitBuffer(in_, 1, kLanes * sizeof(float) * kParts);
        pipe_->InitBuffer(out_, 1, kLanes * sizeof(float) * kParts);
        outputEvent_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Solve()
    {
        if (!spsm_a2::PrepareSolve<Complex>(t_, status_, alpha_)) { return; }
        if constexpr (!Complex) {
            SolveFlat();
            return;
        }
        const int32_t tiles = (t_.rhs - 1) / kLanes + 1;
        // Rows are independent, so the grid covers rows times RHS tiles. Fill order
        // keeps whole tiles on one core, matching the general kernel's access shape.
        const int64_t units = static_cast<int64_t>(t_.rows) * tiles;
        for (int64_t unit = GetBlockIdx(); unit < units; unit += GetBlockNum()) {
            const int32_t row = static_cast<int32_t>(unit / tiles);
            const int32_t tile = static_cast<int32_t>(unit % tiles);
            const int32_t start = tile * kLanes;
            const int32_t count = Min(t_.rhs - start, kLanes);
            SolveRow(row, start, count);
        }
    }
private:
    __aicore__ inline int32_t Min(int32_t a, int32_t b) { return a < b ? a : b; }

    __aicore__ inline void SolveFlat()
    {
        int64_t elements = static_cast<int64_t>(t_.rows) * t_.rhs;
        int64_t chunks = (elements + kLanes - 1) / kLanes;
        for (int64_t chunk = GetBlockIdx(); chunk < chunks; chunk += GetBlockNum()) {
            int64_t first = chunk * kLanes;
            int32_t count = static_cast<int32_t>(elements - first < kLanes ?
                elements - first : kLanes);
            auto input = in_.AllocTensor<float>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
            DataCopyPad(input, b_[first], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
            in_.EnQue(input);
            input = in_.DeQue<float>();
            math_.Begin(input, {t_.alpha, 0.0f}, count);
            in_.FreeTensor(input);
            if (!t_.unit && !DivideFlat(first, count)) { return; }
            auto output = out_.AllocTensor<float>();
            math_.Pack(output, count);
            out_.EnQue(output);
            output = out_.DeQue<float>();
            DataCopyPad(c_[first], output, cp);
            out_.FreeTensor(output);
            SetFlag<HardEvent::MTE3_MTE2>(outputEvent_);
            WaitFlag<HardEvent::MTE3_MTE2>(outputEvent_);
        }
    }

    __aicore__ inline bool DivideFlat(int64_t first, int32_t count)
    {
        if (t_.rhs <= 0 || count <= 0) { return false; }
        int32_t firstRow = first / t_.rhs;
        int32_t rows = (first + count - 1) / t_.rhs - firstRow + 1;
        SetFlag<HardEvent::S_MTE2>(diagonalReuse_);
        WaitFlag<HardEvent::S_MTE2>(diagonalReuse_);
        DataCopyExtParams diagonalCopy{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
        DataCopyPad(diagonalRaw_.Get<float>(), av_[firstRow], diagonalCopy,
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(diagonalReady_);
        WaitFlag<HardEvent::MTE2_S>(diagonalReady_);
        SetFlag<HardEvent::V_S>(scalarReuse_);
        WaitFlag<HardEvent::V_S>(scalarReuse_);
        int32_t previousRow = -1;
        float divisor = 1.0f, inverse = 1.0f;
        for (int32_t lane = 0; lane < count; ++lane) {
            const int32_t row = (first + lane) / t_.rhs - firstRow;
            if (row != previousRow) {
                divisor = diagonalRaw_.Get<float>().GetValue(row);
                // NON_UNIT zeros are rejected during Analysis/Update before Solve.
                if (divisor == 0.0f) { return false; }
                inverse = 1.0f / divisor;
                previousRow = row;
            }
            diagonalLanes_.Get<float>().SetValue(lane, divisor);
            inverseLanes_.Get<float>().SetValue(lane, inverse);
        }
        SetFlag<HardEvent::S_V>(scalarReady_);
        WaitFlag<HardEvent::S_V>(scalarReady_);
        math_.DivideRealVector(diagonalLanes_.Get<float>(), inverseLanes_.Get<float>(), count);
        return true;
    }

    __aicore__ inline Scalar DiagonalValue(int32_t row)
    {
        // Host confirmed one entry per row at the diagonal before selecting this path.
        const int32_t first = rp_.GetValue(row) - t_.base;
        Scalar value{};
        value.real = av_.GetValue(static_cast<int64_t>(first) * kParts);
        if constexpr (Complex) { value.imag = av_.GetValue(static_cast<int64_t>(first) * kParts + 1); }
        return value;
    }

    __aicore__ inline void SolveRow(int32_t row, int32_t start, int32_t count)
    {
        auto input = in_.AllocTensor<float>();
        DataCopyExtParams load{1, static_cast<uint32_t>(count * 4 * kParts), 0, 0, 0};
        DataCopyPad(input, b_[(static_cast<int64_t>(row) * t_.ldb + start) * kParts], load,
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        in_.EnQue(input);
        input = in_.DeQue<float>();
        math_.Begin(input, {t_.alpha, t_.alphaImag}, count);
        in_.FreeTensor(input);
        if (!t_.unit) {
            math_.Divide(DiagonalValue(row), count);
        } else if constexpr (Complex) {
            math_.DivideUnit(count);
        }
        auto output = out_.AllocTensor<float>();
        math_.Pack(output, count);
        out_.EnQue(output);
        output = out_.DeQue<float>();
        DataCopyExtParams store{1, static_cast<uint32_t>(count * 4 * kParts), 0, 0, 0};
        DataCopyPad(c_[(static_cast<int64_t>(row) * t_.ldc + start) * kParts], output, store);
        out_.FreeTensor(output);
        SetFlag<HardEvent::MTE3_MTE2>(outputEvent_);
        WaitFlag<HardEvent::MTE3_MTE2>(outputEvent_);
    }

    GlobalTensor<int32_t> rp_, status_;
    GlobalTensor<float> av_, b_, c_, alpha_;
    TQue<QuePosition::VECIN, 1> in_;
    TQue<QuePosition::VECOUT, 1> out_;
    spsm_a2::CompensatedMath<Complex, kLanes> math_;
    SpsmA2TilingData t_;
    TPipe *pipe_{nullptr};
    event_t outputEvent_{};
    event_t diagonalReady_{}, diagonalReuse_{}, scalarReady_{}, scalarReuse_{};
    TBuf<TPosition::VECCALC> diagonalRaw_, diagonalLanes_, inverseLanes_;
};
}
extern "C" __global__ __aicore__ void spsm_a2_diagonal_fp32(GM_ADDR offsets,
    GM_ADDR values, GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR status, SpsmA2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; SpsmA2DiagonalKernel<false> op;
    op.Init(offsets, values, b, c, alpha, status, tiling, &pipe);
    op.Solve();
}
extern "C" __global__ __aicore__ void spsm_a2_diagonal_complex64(GM_ADDR offsets,
    GM_ADDR values, GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR status, SpsmA2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; SpsmA2DiagonalKernel<true> op;
    op.Init(offsets, values, b, c, alpha, status, tiling, &pipe);
    op.Solve();
}
extern "C" void SpsmA2DiagonalLaunch(GM_ADDR offsets, GM_ADDR values,
    GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR status, const SpsmA2TilingData &tiling,
    uint32_t blocks, void *stream)
{
    if (tiling.complex) {
        spsm_a2_diagonal_complex64<<<blocks, nullptr, stream>>>(offsets, values, b, c, alpha,
            status, tiling);
    } else {
        spsm_a2_diagonal_fp32<<<blocks, nullptr, stream>>>(offsets, values, b, c, alpha,
            status, tiling);
    }
}
