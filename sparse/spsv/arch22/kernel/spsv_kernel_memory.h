/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_KERNEL_MEMORY_H_
#define SPSV_KERNEL_MEMORY_H_
#include "kernel_operator.h"
#include "spsv_tiling_data.h"

namespace spsv_a2 {
using namespace AscendC;

// Scalar arithmetic only touches UB. GM transfers use bounded DMA blocks.
template <typename T, uint32_t Capacity>
class UbCache {
public:
    static constexpr uint32_t kBytes = spsv_a2_ub::BufferBytes(Capacity, sizeof(T));

    __aicore__ inline void Init(TPipe &pipe, GM_ADDR address, int64_t size)
    {
        pipe.InitBuffer(buffer_, Capacity * sizeof(T));
        local_ = buffer_.template Get<T>();
        global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address), size);
        size_ = size;
    }
    __aicore__ inline T Read(int64_t position)
    {
        Load(position);
        return local_.GetValue(static_cast<uint32_t>(position - start_));
    }
    __aicore__ inline void Rebind(GM_ADDR address, int64_t size)
    {
        Flush();
        global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address), size);
        size_ = size;
        start_ = -1;
        length_ = 0;
    }
    __aicore__ inline void Write(int64_t position, T value)
    {
        Load(position);
        local_.SetValue(static_cast<uint32_t>(position - start_), value);
        dirty_ = true;
    }
    // A span stays valid until the next cache-window change. Callers bound
    // their loop to available elements, never crossing a DMA window.
    __aicore__ inline LocalTensor<T> ReadSpan(int64_t position, uint32_t &available)
    {
        Load(position);
        uint32_t offset = static_cast<uint32_t>(position - start_);
        available = static_cast<uint32_t>(length_) - offset;
        return local_[offset];
    }
    __aicore__ inline LocalTensor<T> WritableSpan(int64_t position, uint32_t &available)
    {
        auto span = ReadSpan(position, available);
        dirty_ = true;
        return span;
    }
    // Only used when the complete tensor fits this cache. No subsequent
    // window change is allowed before Flush writes the scalar stores to GM.
    __aicore__ inline LocalTensor<T> WritableWhole()
    {
        Load(0);
        dirty_ = true;
        return local_;
    }
    __aicore__ inline void Flush()
    {
        if (dirty_) {
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(global_[start_], local_,
                        DataCopyExtParams{1, static_cast<uint32_t>(length_ * sizeof(T)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            dirty_ = false;
        }
    }
private:
    __aicore__ inline void Load(int64_t position)
    {
        if (position < start_ || position >= start_ + length_) {
            Flush();
            start_ = position / Capacity * Capacity;
            length_ = size_ - start_;
            if (length_ > Capacity)
                length_ = Capacity;
            DataCopyPad(local_, global_[start_],
                        DataCopyExtParams{1, static_cast<uint32_t>(length_ * sizeof(T)), 0, 0, 0},
                        DataCopyPadExtParams<T>{false, 0, 0, T{}});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
    }
    TBuf<TPosition::VECCALC> buffer_;
    LocalTensor<T> local_;
    GlobalTensor<T> global_;
    int64_t size_ = 0;
    int64_t start_ = -1;
    int64_t length_ = 0;
    bool dirty_ = false;
};

// Preserve the original right-hand side before an aliased solve can publish
// partial results. Every fallback then reads the same immutable input.
__aicore__ inline GM_ADDR PreserveAliasedInput(UbCache<float, spsv_a2_ub::kInputCapacity>& x, GM_ADDR input,
                                               GM_ADDR output, GM_ADDR workspace, const SpsvA2Tiling& tiling)
{
    GM_ADDR sourceInput = input;
    if (input == output) {
        sourceInput = workspace + tiling.inputOffset;
        int64_t total = tiling.m * tiling.components;
        GlobalTensor<float> saved;
        saved.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sourceInput), total);
        for (int64_t offset = 0; offset < total;) {
            uint32_t count = 0;
            auto source = x.ReadSpan(offset, count);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(saved[offset], source,
                        DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            offset += count;
        }
        x.Rebind(sourceInput, total);
    }
    return sourceInput;
}

} // namespace spsv_a2
#endif
