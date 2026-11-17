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
#include "kernel_operator.h"
#include "../spmv_stride_kernel.h"

using namespace AscendC;
namespace {
constexpr uint32_t kBatch = 256;
constexpr uint32_t kSlot = 32;

class SpmvByteCopy {
public:
    __aicore__ inline void Init(TPipe *pipe) {
        pipe_ = pipe;
        pipe_->InitBuffer(slots_, kBatch * kSlot);
        pipe_->InitBuffer(dense_, kBatch * 8); // widest supported element is complex64
    }
    __aicore__ inline void Run(GM_ADDR source, GM_ADDR destination, const SpmvStrideCopy &copy,
                               bool scatter) {
        const uint64_t destStep = scatter ? copy.strideBytes : copy.width;
        uint32_t blocks = GetBlockNum();
        const uint64_t base = reinterpret_cast<uint64_t>(destination);
        // Misaligned native elements may cross cache lines. One block owns the whole output.
        if (base % copy.width != 0) {
            if (GetBlockIdx() != 0) return;
            blocks = 1;
        }
        uint64_t begin = Boundary(copy.length, GetBlockIdx(), blocks, base, destStep);
        uint64_t end = Boundary(copy.length, GetBlockIdx() + 1, blocks, base, destStep);
        GlobalTensor<uint8_t> src, dst;
        src.SetGlobalBuffer(source);
        dst.SetGlobalBuffer(destination);
        auto slots = slots_.Get<uint8_t>();
        auto dense = dense_.Get<uint8_t>();
        const uint64_t gap = copy.strideBytes - copy.width;
        for (uint64_t start = begin; start < end;) {
            uint32_t count = static_cast<uint32_t>(end - start > kBatch ? kBatch : end - start);
            // DMA stride fields are uint32 on DAV_2201. Preserve 64-bit offsets for large strides.
            if (gap > 0xffffffffULL) count = 1;
            const uint32_t dmaGap = count == 1 ? 0 : static_cast<uint32_t>(gap);
            CopyBatch(src, dst, slots, dense, copy, scatter, start, count, dmaGap);
            // Both local buffers are reused in the next batch/field, including scalar reads/writes.
            Fence<HardEvent::MTE3_S>();
            Fence<HardEvent::MTE3_MTE2>();
            start += count;
        }
    }
private:
    __aicore__ inline void CopyBatch(GlobalTensor<uint8_t> &src, GlobalTensor<uint8_t> &dst,
                                     LocalTensor<uint8_t> &slots, LocalTensor<uint8_t> &dense,
                                     const SpmvStrideCopy &copy, bool scatter, uint64_t start,
                                     uint32_t count, uint32_t dmaGap) {
        // Preserve native bits while reshaping each fixed batch between 32B UB slots and dense UB bytes.
        if (!scatter) {
            DataCopyExtParams in{static_cast<uint16_t>(count), copy.width, dmaGap, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            DataCopyPad(slots, src[start * copy.strideBytes], in, pad);
            Fence<HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < count; ++i)
                for (uint32_t b = 0; b < copy.width; ++b)
                    dense.SetValue(i * copy.width + b, slots.GetValue(i * kSlot + b));
            Fence<HardEvent::S_MTE3>();
            DataCopyExtParams out{1, count * copy.width, 0, 0, 0};
            DataCopyPad(dst[start * copy.width], dense, out);
        } else {
            DataCopyExtParams in{1, count * copy.width, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            DataCopyPad(dense, src[start * copy.width], in, pad);
            Fence<HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < count; ++i)
                for (uint32_t b = 0; b < copy.width; ++b)
                    slots.SetValue(i * kSlot + b, dense.GetValue(i * copy.width + b));
            Fence<HardEvent::S_MTE3>();
            DataCopyExtParams out{static_cast<uint16_t>(count), copy.width, 0, dmaGap, 0};
            DataCopyPad(dst[start * copy.strideBytes], slots, out);
        }
    }

    template<HardEvent event> __aicore__ inline void Fence() {
        auto id = static_cast<event_t>(pipe_->FetchEventID(event));
        SetFlag<event>(id); WaitFlag<event>(id);
    }
    // Advance a logical partition to the first element on a new physical 64B cache line.
    // This is O(1), even when the stride spans billions of bytes; no physical holes are visited.
    __aicore__ inline uint64_t Boundary(uint64_t length, uint32_t part, uint32_t parts,
                                       uint64_t base, uint64_t step) {
        // Invalid partitions yield identical begin/end boundaries, so no copy is issued.
        if (parts == 0) return length;
        if (step == 0) return length;
        if (part == 0) return 0;
        if (part >= parts) return length;
        const uint64_t q = length / parts, r = length % parts;
        uint64_t index = q * part + (r * part + parts - 1) / parts;
        if (index >= length) return length;
        const uint64_t address = base + index * step;
        const uint64_t remainder = (64 - (address & 63)) & 63;
        index += remainder / step + (remainder % step != 0);
        return index > length ? length : index;
    }
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> slots_, dense_;
};
}

extern "C" __global__ __aicore__ void spmv_stride_pack(
    GM_ADDR rows, GM_ADDR columns, GM_ADDR values, GM_ADDR x, GM_ADDR y,
    GM_ADDR workspace, SpmvStridePlan plan, uint32_t readY) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SpmvByteCopy copier; copier.Init(&pipe);
    for (uint32_t i = 0; i < 5; ++i) {
        const auto &copy = plan.copies[i];
        if (!copy.enabled || (i == 4 && !readY)) continue;
        GM_ADDR source = i == 0 ? rows : i == 1 ? columns : i == 2 ? values : i == 3 ? x : y;
        copier.Run(source, workspace + copy.offset, copy, false);
    }
}
extern "C" __global__ __aicore__ void spmv_stride_scatter(
    GM_ADDR workspace, GM_ADDR y, SpmvStrideCopy copy) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SpmvByteCopy copier; copier.Init(&pipe);
    copier.Run(workspace + copy.offset, y, copy, true);
}
// Host launchers deliberately use a separate TU from the numerical dispatcher.
extern "C" void spmv_stride_pack_do(uint8_t *rows, uint8_t *columns, uint8_t *values,
    uint8_t *x, uint8_t *y, uint8_t *workspace, SpmvStridePlan plan, bool readY,
    uint32_t blocks, void *stream) {
    bool needed = false;
    for (unsigned i = 0; i < 5; ++i) needed |= plan.copies[i].enabled && (i != 4 || readY);
    if (needed) spmv_stride_pack<<<blocks, nullptr, stream>>>(rows, columns, values, x, y, workspace, plan, readY);
}
extern "C" void spmv_stride_scatter_do(uint8_t *workspace, uint8_t *y,
    SpmvStrideCopy copy, uint32_t blocks, void *stream) {
    spmv_stride_scatter<<<blocks, nullptr, stream>>>(workspace, y, copy);
}
