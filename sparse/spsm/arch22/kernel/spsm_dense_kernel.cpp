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
using namespace AscendC;

// Dense stage.
namespace {
constexpr int32_t kTile = 64;
constexpr int32_t kBlockFloats = 8;
class DenseCopy {
public:
    __aicore__ inline void Init(GM_ADDR source, GM_ADDR destination, GM_ADDR status,
        SpsmA2DenseTilingData t, TPipe *pipe)
    {
        t_ = t; parts_ = t.complex ? 2 : 1;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(source));
        dst_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(destination));
        status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
        pipe->InitBuffer(input_, 1, kTile * 8);
        pipe->InitBuffer(output_, 1, kTile * 32);
        pipe->InitBuffer(padded_, kTile * 32);
        pipe->InitBuffer(loadOffsets_, kTile * 8);
        pipe->InitBuffer(storeOffsets_, kTile * 32);
        pipe->InitBuffer(sign_, kTile * 8);
        loadEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_V));
        InitOffsets(pipe);
    }
    __aicore__ inline void Process()
    {
        for (int32_t core = 0; core < t_.statusBlocks; ++core) {
            if (status_.GetValue(core * kSpsmA2StatusStride) != 0) { return; }
        }
        if (!spsm_a2::DispatchReady(status_, t_.statusBlocks, t_.packedFallback)) { return; }
        int64_t colTiles = (static_cast<int64_t>(t_.cols) + kTile - 1) / kTile;
        int64_t tiles = colTiles * t_.rows;
        for (int64_t tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum()) {
            int32_t row = tile / colTiles, col = (tile % colTiles) * kTile;
            int32_t count = t_.cols - col < kTile ? t_.cols - col : kTile;
            Copy(row, col, count);
        }
    }
private:
    __aicore__ inline void InitOffsets(TPipe *pipe)
    {
        auto load = loadOffsets_.Get<int32_t>();
        auto store = storeOffsets_.Get<int32_t>(); auto sign = sign_.Get<uint16_t>();
        // Seed one hardware block of address/sign metadata, then vector-expand.
        for (int32_t lane = 0; lane < kBlockFloats; ++lane) {
            load.SetValue(lane, (lane / parts_) * 32 + (lane % parts_) * 4);
            store.SetValue(lane, (lane % parts_) * 4);

        }
        for (int32_t lane = 0; lane < kBlockFloats * 2; ++lane) {
            sign.SetValue(lane, lane % 4 == 3 ? static_cast<uint16_t>(0x8000) : 0);
        }
        auto event = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event); WaitFlag<HardEvent::S_V>(event);
        for (int32_t group = 1; group < kTile; ++group) {
            Adds(store[group * kBlockFloats], store, group * parts_ * 4, kBlockFloats);
        }
        for (int32_t group = 1; group < kTile * parts_ / kBlockFloats; ++group) {
            Adds(load[group * kBlockFloats], load, group * kBlockFloats / parts_ * 32, kBlockFloats);
            DataCopy(sign[group * kBlockFloats * 2], sign, kBlockFloats * 2);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline LocalTensor<float> Load(int64_t offset, int32_t count)
    {
        auto input = input_.AllocTensor<float>();
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        if (t_.srcColStride == 1) {
            DataCopyExtParams cp{1, static_cast<uint32_t>(count * parts_ * 4), 0, 0, 0};
            DataCopyPad(input, src_[offset * parts_], cp, pad);
        } else {
            DataCopyExtParams cp{static_cast<uint16_t>(count), static_cast<uint32_t>(parts_ * 4),
                static_cast<uint32_t>((t_.srcColStride - 1) * parts_ * 4), 0, 0};
            auto padded = padded_.Get<float>();
            DataCopyPad(padded, src_[offset * parts_], cp, pad);
            SetFlag<HardEvent::MTE2_V>(loadEvent_); WaitFlag<HardEvent::MTE2_V>(loadEvent_);
            Gather(input, padded, loadOffsets_.Get<uint32_t>(), 0, count * parts_);
            PipeBarrier<PIPE_V>();
        }
        input_.EnQue(input);
        return input_.DeQue<float>();
    }
    __aicore__ inline void Conjugate(LocalTensor<float> input, int32_t count)
    {
        // Flip only the imaginary sign bit, preserving signed zeros and NaN payloads.
        auto bits = input.ReinterpretCast<uint16_t>();
        auto mask = sign_.Get<uint16_t>(); auto tmp = padded_.Get<uint16_t>();
        Or(tmp, bits, mask, count * 4);
        PipeBarrier<PIPE_V>();
        And(bits, bits, mask, count * 4);
        PipeBarrier<PIPE_V>();
        Not(bits, bits, count * 4);
        PipeBarrier<PIPE_V>();
        And(bits, tmp, bits, count * 4);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Copy(int32_t row, int32_t col, int32_t count)
    {
        auto input = Load(row * t_.srcRowStride + col * t_.srcColStride, count);
        if (t_.complex && t_.conjugate) {
            Conjugate(input, count);
        }
        auto output = output_.AllocTensor<float>();
        if (t_.dstColStride == 1) {
            // A numeric add would normalize negative zero. Copy bytes so the
            // layout operation preserves floating-point representations.
            DataCopy(output, input, (count * parts_ + kBlockFloats - 1) / kBlockFloats * kBlockFloats);
        } else {
            Gather(output, input, storeOffsets_.Get<uint32_t>(), 0, count * kBlockFloats);
        }
        output_.EnQue(output); output = output_.DeQue<float>();
        Store(output, row * t_.dstRowStride + col * t_.dstColStride, count);
        output_.FreeTensor(output); input_.FreeTensor(input);
    }
    __aicore__ inline void Store(LocalTensor<float> output, int64_t offset, int32_t count)
    {
        if (t_.dstColStride == 1) {
            DataCopyExtParams cp{1, static_cast<uint32_t>(count * parts_ * 4), 0, 0, 0};
            DataCopyPad(dst_[offset * parts_], output, cp);
        } else {
            DataCopyExtParams cp{static_cast<uint16_t>(count), static_cast<uint32_t>(parts_ * 4),
                0, static_cast<uint32_t>((t_.dstColStride - 1) * parts_ * 4), 0};
            DataCopyPad(dst_[offset * parts_], output, cp);
        }
    }
    SpsmA2DenseTilingData t_{};
    int32_t parts_{1}; event_t loadEvent_{};
    GlobalTensor<float> src_, dst_;
    GlobalTensor<int32_t> status_;
    TQue<TPosition::VECIN, 1> input_;
    TQue<TPosition::VECOUT, 1> output_;
    TBuf<TPosition::VECCALC> padded_, loadOffsets_, storeOffsets_, sign_;
};
}
extern "C" __global__ __aicore__ void spsm_a2_dense_copy(GM_ADDR source,
    GM_ADDR destination, GM_ADDR status, SpsmA2DenseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; DenseCopy copy;
    copy.Init(source, destination, status, tiling, &pipe); copy.Process();
}
extern "C" void SpsmA2DenseLaunch(GM_ADDR source, GM_ADDR destination, GM_ADDR status,
    const SpsmA2DenseTilingData &tiling, uint32_t blocks, void *stream)
{
    spsm_a2_dense_copy<<<blocks, nullptr, stream>>>(source, destination, status, tiling);
}
