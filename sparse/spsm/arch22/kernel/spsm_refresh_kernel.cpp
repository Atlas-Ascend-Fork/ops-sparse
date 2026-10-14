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
using namespace AscendC;

// Refresh stage.
namespace {
constexpr int32_t kTile = 512;
class Refresh {
public:
    __aicore__ inline void Init(GM_ADDR rows, GM_ADDR cols, GM_ADDR values,
        GM_ADDR snapshot, GM_ADDR status, SpsmA2RefreshTilingData t, TPipe *pipe)
    {
        t_ = t; parts_ = t.complex ? 2 : 1;
        rows_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rows));
        cols_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cols));
        values_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(values));
        snapshot_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(snapshot));
        status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
        pipe->InitBuffer(rowsBuf_, kTile * 4); pipe->InitBuffer(colsBuf_, kTile * 4);
        pipe->InitBuffer(new_, kTile * 8); pipe->InitBuffer(old_, kTile * 8);
        pipe->InitBuffer(mask_, kTile * 8); pipe->InitBuffer(tmp_, kTile * 8);
        pipe->InitBuffer(out_, kTile * 8);
        load_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_S));
        scalar_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
        store_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE3));
        reuse_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_MTE2));
    }
    __aicore__ inline void Process()
    {
        for (int32_t core = 0; core < t_.blocks; ++core) {
            if (status_.GetValue(core * kSpsmA2StatusStride) != 0) { return; }
        }
        for (int64_t begin = static_cast<int64_t>(GetBlockIdx()) * kTile; begin < t_.nnz;
            begin += static_cast<int64_t>(GetBlockNum()) * kTile) {
            int32_t count = t_.nnz - begin < kTile ? t_.nnz - begin : kTile;
            Load(begin, count); Select(count); Store(begin, count);
        }
    }
private:
    __aicore__ inline void Load(int64_t begin, int32_t count)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * parts_ * 4), 0, 0, 0};
        DataCopyPad(new_.Get<float>(), values_[begin * parts_], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        if (t_.diagonal) {
            DataCopyPad(old_.Get<float>(), snapshot_[begin * parts_], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
            cp.blockLen = count * 4;
            DataCopyPad(rowsBuf_.Get<int32_t>(), rows_[begin], cp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            DataCopyPad(colsBuf_.Get<int32_t>(), cols_[begin], cp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_S>(load_); WaitFlag<HardEvent::MTE2_S>(load_);
    }
    __aicore__ inline void Select(int32_t count)
    {
        if (!t_.diagonal) {
            SetFlag<HardEvent::S_V>(scalar_); WaitFlag<HardEvent::S_V>(scalar_);
            DataCopy(out_.Get<float>(), new_.Get<float>(), (count * parts_ + 7) / 8 * 8);
            return;
        }
        auto rows = rowsBuf_.Get<int32_t>(); auto cols = colsBuf_.Get<int32_t>(); auto mask = mask_.Get<int32_t>();
        for (int32_t p = 0; p < count; ++p) {
            int32_t keep = rows.GetValue(p) == cols.GetValue(p) ? -1 : 0;
            mask.SetValue(p * parts_, keep);
            if (parts_ == 2) { mask.SetValue(p * parts_ + 1, keep); }
        }
        SetFlag<HardEvent::S_V>(scalar_); WaitFlag<HardEvent::S_V>(scalar_);
        auto bits = mask_.Get<uint16_t>(); auto out = out_.Get<uint16_t>(); auto tmp = tmp_.Get<uint16_t>();
        And(out, new_.Get<uint16_t>(), bits, count * parts_ * 2); PipeBarrier<PIPE_V>();
        Not(tmp, bits, count * parts_ * 2); PipeBarrier<PIPE_V>();
        And(tmp, old_.Get<uint16_t>(), tmp, count * parts_ * 2); PipeBarrier<PIPE_V>();
        Or(out, out, tmp, count * parts_ * 2); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Store(int64_t begin, int32_t count)
    {
        SetFlag<HardEvent::V_MTE3>(store_); WaitFlag<HardEvent::V_MTE3>(store_);
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * parts_ * 4), 0, 0, 0};
        DataCopyPad(snapshot_[begin * parts_], out_.Get<float>(), cp);
        if (t_.diagonal) { DataCopyPad(values_[begin * parts_], out_.Get<float>(), cp); }
        SetFlag<HardEvent::MTE3_MTE2>(reuse_); WaitFlag<HardEvent::MTE3_MTE2>(reuse_);
    }
    SpsmA2RefreshTilingData t_{}; int32_t parts_{1};
    event_t load_{}, scalar_{}, store_{}, reuse_{};
    GlobalTensor<int32_t> rows_, cols_, status_;
    GlobalTensor<float> values_, snapshot_;
    TBuf<TPosition::VECCALC> rowsBuf_, colsBuf_, new_, old_, mask_, tmp_, out_;
};
}
extern "C" __global__ __aicore__ void spsm_a2_refresh_values(GM_ADDR rows, GM_ADDR cols,
    GM_ADDR values, GM_ADDR snapshot, GM_ADDR status, SpsmA2RefreshTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; Refresh op; op.Init(rows, cols, values, snapshot, status, tiling, &pipe); op.Process();
}
extern "C" void SpsmA2RefreshLaunch(GM_ADDR rows, GM_ADDR cols, GM_ADDR values,
    GM_ADDR snapshot, GM_ADDR status, const SpsmA2RefreshTilingData &tiling, void *stream)
{
    spsm_a2_refresh_values<<<tiling.blocks, nullptr, stream>>>(rows, cols, values, snapshot, status, tiling);
}
