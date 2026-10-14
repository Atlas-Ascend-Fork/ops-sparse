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

// Gather stage.
namespace {
// Without duplicate coordinates canonicalization is a pure permutation: canonical
// entry k holds the sorted entry k, and the sort recorded where that entry came from.
// Refreshing values is then a gather, with no addition at all, so the merge scan's
// accumulation order cannot be perturbed. Matrices that do hold duplicates keep the
// original sort-and-merge path.
template<bool Complex> class SpsmA2GatherKernel {
    static constexpr int32_t kParts = Complex ? 2 : 1;
public:
    __aicore__ inline void Init(GM_ADDR perm, GM_ADDR source, GM_ADDR cols, GM_ADDR rowPtr,
        GM_ADDR output, GM_ADDR status, SpsmA2GatherTilingData t, TPipe *pipe)
    {
        t_ = t;
        pipe->InitBuffer(out_buf_, kTile * sizeof(float) * kParts);
        storeEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_MTE3));
        reuseEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_S));
        perm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(perm));
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(source));
        cols_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cols));
        rowPtr_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rowPtr));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
        status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
    }

    __aicore__ inline void Run()
    {
        // An earlier stage may already have rejected the matrix.
        for (int32_t block = 0; block < t_.statusBlocks; ++block) {
            if (status_.GetValue(block * kSpsmA2StatusStride) != 0) { return; }
        }
        int32_t failure = 0;
        // Whole rows per block keeps the diagonal check local: the values it reads are
        // the ones this block has just written.
        // The terminating increment can exceed INT32_MAX even though each valid row fits.
        for (int64_t row = GetBlockIdx(); row < t_.rows; row += GetBlockNum()) {
            const int32_t first = rowPtr_.GetValue(row);
            const int32_t end = rowPtr_.GetValue(row + 1);
            bool diagonal = t_.unit != 0;
            // Stage through UB and store with DataCopyPad, the same discipline the rest
            // of the pipeline uses; a scalar write straight to GM has no flush guarantee
            // before the next kernel reads it.
            for (int32_t base = first; base < end;) {
                const int32_t count = Min(end - base, kTile);
                auto buffer = out_buf_.Get<float>();
                for (int32_t i = 0; i < count; ++i) {
                    const int32_t k = base + i;
                    const int64_t from = static_cast<int64_t>(perm_.GetValue(k)) * kParts;
                    const float real = src_.GetValue(from);
                    buffer.SetValue(i * kParts, real);
                    float imag = 0.0f;
                    if constexpr (Complex) {
                        imag = src_.GetValue(from + 1);
                        buffer.SetValue(i * kParts + 1, imag);
                    }
                    if (!t_.unit && cols_.GetValue(k) == row) {
                        diagonal = real != 0.0f || (Complex && imag != 0.0f);
                    }
                }
                SetFlag<HardEvent::S_MTE3>(storeEvent_); WaitFlag<HardEvent::S_MTE3>(storeEvent_);
                DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4 * kParts), 0, 0, 0};
                DataCopyPad(out_[static_cast<int64_t>(base) * kParts], buffer, cp);
                SetFlag<HardEvent::MTE3_S>(reuseEvent_); WaitFlag<HardEvent::MTE3_S>(reuseEvent_);
                // Advance only by valid entries so the final value is end, without overflow.
                base += count;
            }
            if (!diagonal) { failure = 2; }
        }
        if (failure != 0) {
            status_.SetValue(GetBlockIdx() * kSpsmA2StatusStride, failure);
        }
    }
private:
    __aicore__ inline int32_t Min(int32_t a, int32_t b) { return a < b ? a : b; }
    static constexpr int32_t kTile = 128;
    TBuf<QuePosition::VECCALC> out_buf_;
    event_t storeEvent_{}, reuseEvent_{};
    GlobalTensor<int32_t> perm_, cols_, rowPtr_, status_;
    GlobalTensor<float> src_, out_;
    SpsmA2GatherTilingData t_{};
};
}
extern "C" __global__ __aicore__ void spsm_a2_gather_fp32(GM_ADDR perm, GM_ADDR source, GM_ADDR cols,
    GM_ADDR rowPtr, GM_ADDR output, GM_ADDR status, SpsmA2GatherTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; SpsmA2GatherKernel<false> op;
    op.Init(perm, source, cols, rowPtr, output, status, tiling, &pipe);
    op.Run();
}
extern "C" __global__ __aicore__ void spsm_a2_gather_complex64(GM_ADDR perm, GM_ADDR source, GM_ADDR cols,
    GM_ADDR rowPtr, GM_ADDR output, GM_ADDR status, SpsmA2GatherTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; SpsmA2GatherKernel<true> op;
    op.Init(perm, source, cols, rowPtr, output, status, tiling, &pipe);
    op.Run();
}
extern "C" void SpsmA2GatherLaunch(GM_ADDR perm, GM_ADDR source, GM_ADDR cols, GM_ADDR rowPtr,
    GM_ADDR output, GM_ADDR status, const SpsmA2GatherTilingData &tiling, uint32_t blocks, void *stream)
{
    if (tiling.complex) {
        spsm_a2_gather_complex64<<<blocks, nullptr, stream>>>(perm, source, cols, rowPtr, output, status, tiling);
    } else {
        spsm_a2_gather_fp32<<<blocks, nullptr, stream>>>(perm, source, cols, rowPtr, output, status, tiling);
    }
}
