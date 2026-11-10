/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_UPDATE_H_
#define SPSV_UPDATE_H_
#include "spsv_kernel_memory.h"

namespace spsv_a2 {
using namespace AscendC;

// With no filtered entries, CSR N (or a purely diagonal transpose) retains
// source order. No permutation lookup, arithmetic or conjugation is needed.
__aicore__ inline void CopyGeneralValues(TPipe &pipe, GM_ADDR source, GM_ADDR workspace,
    const SpsvA2Tiling &tiling)
{
    constexpr int64_t capacity = 4096;
    static_assert(spsv_a2_ub::BufferBytes(capacity, sizeof(float)) <= spsv_a2_ub::kRequiredBytes,
        "General update copy exceeds the UB budget");
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, capacity * sizeof(float));
    auto local = buffer.Get<float>();
    GlobalTensor<float> input;
    GlobalTensor<float> output;
    int64_t total = tiling.nnz * tiling.components;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(source), total);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + tiling.valueOffset), total);
    for (int64_t offset = 0; offset < total; offset += capacity) {
        int64_t count = total - offset;
        if (count > capacity) count = capacity;
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(local, input[offset], copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(output[offset], local, copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
}

class Update {
public:
    __aicore__ inline Update(TPipe &pipe, GM_ADDR source, GM_ADDR workspace,
        SpsvA2Tiling tiling) : t_(tiling)
    {
        row_.Init(pipe, workspace + t_.rowOffset, t_.m + 1);
        col_.Init(pipe, workspace + t_.colOffset, t_.nnz);
        perm_.Init(pipe, workspace + t_.permOffset, t_.nnz);
        value_.Init(pipe, workspace + t_.valueOffset, t_.nnz * t_.components);
        // Entry sets sourceNnz to m for DIAGONAL and to stored nnz for GENERAL.
        source_.Init(pipe, source, t_.sourceNnz * t_.components);
    }
    __aicore__ inline void Run(bool diagonal)
    {
        for (int64_t row = 0; row < t_.m; ++row) {
            bool found = false;
            for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
                if (diagonal && col_.Read(p) != row) continue;
                int64_t original = diagonal ? row : perm_.Read(p);
                float real = diagonal && found ? 0 : source_.Read(original * t_.components);
                value_.Write(p * t_.components, real);
                if (t_.components == 2) {
                    float imag = diagonal && found ? 0 : source_.Read(original * 2 + 1);
                    value_.Write(p * 2 + 1, t_.conjugate ? -imag : imag);
                }
                found = true;
            }
        }
        value_.Flush();
    }
private:
    SpsvA2Tiling t_;
    UbCache<int32_t, 2048> row_, col_, perm_;
    UbCache<float, 4096> value_, source_;
    static_assert(decltype(row_)::kBytes + decltype(col_)::kBytes + decltype(perm_)::kBytes +
        decltype(value_)::kBytes + decltype(source_)::kBytes <= spsv_a2_ub::kRequiredBytes,
        "Update exceeds the UB budget");
};

} // namespace spsv_a2
#endif
