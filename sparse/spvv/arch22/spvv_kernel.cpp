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
#include "spvv_tiling_data.h"
#include "spvv_kernel.h"
#include <acl/acl.h>

using namespace AscendC;

// ---------------------------------------------------------------------------
// KernelSpvv — vectorized sparse-dense dot product.
// InputT = float / InputSz = 4 → FP32: Gather + Mul + ReduceSum<float>
// InputT = half  / InputSz = 2 → Half:  Gather + Mul + Cast→float + ReduceSum<float>
// Both paths accumulate in FP32 and produce a single float result via AtomicAdd.
//
// Each core processes a contiguous chunk of nnz (nnzPerCore). Per tile:
//   1. Load full-tile indices (for segment scan) and x values (final Mul).
//   2. y range = [idx[0], idx[tn-1]] (indices sorted ascending).
//   3. Walk the y range in segments of YSLICE_MAX. For each segment:
//        - count the in-range indices; round the count down to ALIGN_ELEM
//          (32B) so the Gather dst offset (gatherOff) stays 32B-aligned.
//          The rounded-off indices are deferred to the next segment.
//        - load y from a 32B-aligned address (yLoadStart); use Gather's
//          srcBaseAddr to bridge yLoadStart → segYStart.
//        - Gather into yGathered[gatherOff], concatenating segments.
//   4. One Mul(yGathered, xval) + ReduceSum over all tn, accumulate.
// Cross-core reduction via AtomicAdd to output.
// ---------------------------------------------------------------------------
template<typename InputT, uint32_t InputSz>
class KernelSpvv {
    static constexpr uint32_t ALIGN_ELEM = 32 / InputSz;          // 8 (FP32) / 16 (FP16)
    static constexpr uint32_t YSLICE_MAX = (InputSz == 4) ? 32000 : 64000;

public:
    __aicore__ inline KernelSpvv() {}

    __aicore__ inline void Init(
        __gm__ uint8_t* xIndices, __gm__ uint8_t* xValues,
        __gm__ uint8_t* y, __gm__ uint8_t* output, SpvvTilingData tiling)
    {
        blockIdx_ = GetBlockIdx();

        nnz_ = tiling.nnz;
        yLen_ = tiling.yLen;
        nnzPerCore_ = tiling.nnzPerCore;

        // Compute this core's nnz range. Host guarantees nnzStart_ < nnz for
        // every launched core, so nnzCount_ > 0 here.
        nnzStart_ = blockIdx_ * nnzPerCore_;
        uint32_t remain = nnz_ - nnzStart_;
        nnzCount_ = (nnzPerCore_ < remain) ? nnzPerCore_ : remain;

        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(xIndices), nnz_);
        xvalGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(xValues), nnz_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(y), yLen_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), 1);

        // Output is zeroed on the host side (aclrtMemsetAsync on the same stream)
        // before this kernel launches, so every core can AtomicAdd into a
        // known-zero result without any in-kernel zeroing or cross-core sync.

        pipe_.InitBuffer(qIdx_, 1, SPVV_TILE_LENGTH * sizeof(int32_t));
        pipe_.InitBuffer(qXval_, 1, SPVV_TILE_LENGTH * InputSz);
        // ySlice spans up to YSLICE_MAX elements (one segment's y range).
        pipe_.InitBuffer(qYSlice_, 1, YSLICE_MAX * InputSz);
        pipe_.InitBuffer(qYGathered_, 1, SPVV_TILE_LENGTH * InputSz);
        if constexpr (InputSz == 2) {
            // FP16: ReduceSum runs in FP32 → needs a float dst+tmp buffer
            pipe_.InitBuffer(qReduce_, 1, SPVV_TILE_LENGTH * sizeof(float));
            pipe_.InitBuffer(qYGatheredFp32_, 1, SPVV_TILE_LENGTH * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        ComputeTiles();
        CrossCoreReduce();
    }

private:
    // Load tile indices + x values and capture the y range; FP16 values are cast to FP32.
    // idxUb/xvalUb are pre-allocated; xvalFp32 is allocated here for FP16 only.
    __aicore__ inline void LoadTile(uint32_t ts, uint32_t tn,
        LocalTensor<int32_t>& idxUb, LocalTensor<InputT>& xvalUb,
        LocalTensor<float>& xvalFp32, int32_t& yStart, int32_t& yEnd)
    {
        DataCopyPadParams pad = { false, 0, 0, 0 };
        DataCopyParams dpI = { 1, static_cast<uint16_t>(tn * sizeof(int32_t)), 0, 0 };
        DataCopyPad(idxUb, idxGm_[nnzStart_ + ts], dpI, pad);
        // Ensure idxUb is ready before reading its bounds; xvalUb's DMA overlaps this wait.
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        DataCopyParams dpV = { 1, static_cast<uint16_t>(tn * InputSz), 0, 0 };
        DataCopyPad(xvalUb, xvalGm_[nnzStart_ + ts], dpV, pad);
        if constexpr (InputSz == 2) {
            // Mark xvalUb ready so its FP32 cast runs before the segment loop.
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        yStart = idxUb.GetValue(0);
        yEnd = idxUb.GetValue(tn - 1);
        // Keep indices in element units; absolute byte offsets can overflow signed 32-bit.

        // FP16: cast xval to FP32 now so it overlaps the y-load MTE2 work.
        if constexpr (InputSz == 2) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            xvalFp32 = qReduce_.AllocTensor<float>();
            Cast<float, half>(xvalFp32, xvalUb, RoundMode::CAST_NONE, tn);
        }
    }

    // Fast path: tile's y range fits one YSLICE_MAX load — single Gather.
    __aicore__ inline void FastGather(uint32_t tn, int32_t yStart, int32_t yEnd,
        LocalTensor<int32_t>& idxUb, LocalTensor<InputT>& ySliceUb,
        LocalTensor<InputT>& yGathered)
    {
        DataCopyPadExtParams<InputT> padExt = { false, 0, 0, 0 };
        uint32_t yLoadLen = static_cast<uint32_t>(static_cast<int64_t>(yEnd) - yStart + 1);
        DataCopyExtParams dpY = { 1, yLoadLen * InputSz, 0, 0, 0 };
        DataCopyPad(ySliceUb, yGm_[yStart], dpY, padExt);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

        // Convert bounded element deltas to byte offsets for Gather.
        Adds(idxUb, idxUb, -yStart, tn);
        Muls(idxUb, idxUb, static_cast<int32_t>(InputSz), tn);
        LocalTensor<uint32_t> idxByteOff = idxUb.ReinterpretCast<uint32_t>();

        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Gather(yGathered, ySliceUb, idxByteOff, 0, tn);
    }

    // Binary search (upper bound): count indices in [nnzPos, tn) <= segYEnd.
    __aicore__ inline uint32_t FindSegmentCount(LocalTensor<int32_t>& idxUb,
        uint32_t nnzPos, uint32_t tn, int32_t segYEnd)
    {
        uint32_t lo = nnzPos;
        uint32_t hi = tn;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (idxUb.GetValue(mid) <= segYEnd) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo - nnzPos;
    }

    // Scalar gather for sparse segments (< ALIGN_ELEM) to keep nnzPos aligned.
    // Returns true if the segment loop should break (nnzPos reached tn).
    __aicore__ inline bool ScalarGatherSegment(LocalTensor<int32_t>& idxUb,
        LocalTensor<InputT>& yGathered, uint32_t& nnzPos, uint32_t tn)
    {
        uint32_t n = (nnzPos + ALIGN_ELEM > tn) ? (tn - nnzPos) : ALIGN_ELEM;
        for (uint32_t i = 0; i < n; i++) {
            // idxUb retains element indices; vector offsets are scaled only after segmentation.
            int32_t idx = idxUb.GetValue(nnzPos + i);
            // Index addressing remains ACL_SPARSE_INDEX_32I; widen only for the yLen_ comparison.
            if (idx < 0 || static_cast<uint64_t>(idx) >= yLen_) {
                yGathered.SetValue(nnzPos + i, static_cast<InputT>(0));
            } else {
                yGathered.SetValue(nnzPos + i, yGm_.GetValue(idx));
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        nnzPos += n;
        return nnzPos >= tn;
    }

    // Slow path: y range exceeds YSLICE_MAX — split into 32B-aligned segments,
    // Gather-concatenate into yGathered.
    __aicore__ inline void SlowGather(uint32_t tn, int32_t yEnd,
        LocalTensor<int32_t>& idxUb, LocalTensor<InputT>& ySliceUb,
        LocalTensor<InputT>& yGathered)
    {
        DataCopyPadExtParams<InputT> padExt = { false, 0, 0, 0 };
        uint32_t nnzPos = 0;   // index position == gather dst offset (aligned)
        while (nnzPos < tn) {
            int32_t segYStart = idxUb.GetValue(nnzPos);
            int64_t segYEndWide = static_cast<int64_t>(segYStart) + YSLICE_MAX - 1;
            if (segYEndWide > yEnd) { segYEndWide = yEnd; }
            int32_t segYEnd = static_cast<int32_t>(segYEndWide);

            uint32_t count = FindSegmentCount(idxUb, nnzPos, tn, segYEnd);
            bool isLast = (nnzPos + count >= tn);

            // Sparse segment: scalar-gather ALIGN_ELEM to keep nnzPos aligned.
            if (!isLast && count < ALIGN_ELEM) {
                if (ScalarGatherSegment(idxUb, yGathered, nnzPos, tn)) {
                    break;
                }
                continue;
            }

            // Round count down to ALIGN_ELEM for dst alignment (last exempt).
            if (!isLast) {
                count = (count / ALIGN_ELEM) * ALIGN_ELEM;
            }
            // Trim segYEnd to the last actually-gathered index (smaller DMA).
            segYEnd = idxUb.GetValue(nnzPos + count - 1);

            uint32_t yLoadLen = static_cast<uint32_t>(segYEnd - segYStart + 1);
            DataCopyExtParams dpY = { 1, yLoadLen * InputSz, 0, 0, 0 };
            DataCopyPad(ySliceUb, yGm_[segYStart], dpY, padExt);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

            // Convert bounded element deltas to byte offsets for Gather.
            LocalTensor<int32_t> idxSeg = idxUb[nnzPos];
            Adds(idxSeg, idxSeg, -segYStart, count);
            Muls(idxSeg, idxSeg, static_cast<int32_t>(InputSz), count);
            LocalTensor<uint32_t> idxByteOff = idxSeg.ReinterpretCast<uint32_t>();

            // The next segment boundary is read by the scalar unit from idxUb.
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Gather(yGathered[nnzPos], ySliceUb, idxByteOff, 0, count);

            nnzPos += count;
        }
    }

    // Final element-wise multiply + reduce over all tn, accumulate into partialSum_.
    __aicore__ inline void MulReduce(uint32_t tn,
        LocalTensor<InputT>& xvalUb, LocalTensor<float>& xvalFp32,
        LocalTensor<InputT>& yGathered)
    {
        if constexpr (InputSz == 4) {
            // FP32: reuse xvalUb (consumed by Mul) as ReduceSum dst+tmp.
            Mul(yGathered, xvalUb, yGathered, tn);
            ReduceSum<float>(xvalUb, yGathered, xvalUb, tn);
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            partialSum_ += xvalUb.GetValue(0);
        } else {
            // FP16: cast yGathered to FP32, Mul with pre-cast xvalFp32.
            LocalTensor<float> yGatheredFp32 = qYGatheredFp32_.AllocTensor<float>();
            Cast<float, half>(yGatheredFp32, yGathered, RoundMode::CAST_NONE, tn);
            Mul<float>(yGatheredFp32, xvalFp32, yGatheredFp32, tn);
            ReduceSum<float>(xvalFp32, yGatheredFp32, xvalFp32, tn);
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            partialSum_ += xvalFp32.GetValue(0);
            qReduce_.FreeTensor(xvalFp32);
            qYGatheredFp32_.FreeTensor(yGatheredFp32);
        }
    }

    __aicore__ inline void ComputeTiles()
    {
        partialSum_ = 0.0f;
        for (uint32_t ts = 0; ts < nnzCount_; ts += SPVV_TILE_LENGTH) {
            uint32_t tn = (ts + SPVV_TILE_LENGTH > nnzCount_) ? (nnzCount_ - ts) : SPVV_TILE_LENGTH;

            LocalTensor<int32_t> idxUb = qIdx_.AllocTensor<int32_t>();
            LocalTensor<InputT> xvalUb = qXval_.AllocTensor<InputT>();
            LocalTensor<InputT> ySliceUb = qYSlice_.AllocTensor<InputT>();
            LocalTensor<InputT> yGathered = qYGathered_.AllocTensor<InputT>();

            int32_t yStart = 0, yEnd = 0;
            LocalTensor<float> xvalFp32;
            LoadTile(ts, tn, idxUb, xvalUb, xvalFp32, yStart, yEnd);

            if (static_cast<int64_t>(yEnd) - yStart + 1 <= YSLICE_MAX) {
                FastGather(tn, yStart, yEnd, idxUb, ySliceUb, yGathered);
            } else {
                SlowGather(tn, yEnd, idxUb, ySliceUb, yGathered);
            }

            MulReduce(tn, xvalUb, xvalFp32, yGathered);

            qYSlice_.FreeTensor(ySliceUb);
            qYGathered_.FreeTensor(yGathered);
            qIdx_.FreeTensor(idxUb);
            qXval_.FreeTensor(xvalUb);
        }
    }

    // Cross-core reduction via SetAtomicAdd directly to output. The host-side
    // aclrtMemsetAsync guarantees outGm_ is 0 before any core reaches here, so
    // AtomicAdd needs no cross-core ordering. Do NOT add SyncAll() before this
    // (or anywhere in this kernel): hardware SyncAll deadlocks the FP16 path
    // regardless of blockNum or kernel type (see memory: ascendc-syncall-deadlock).
    __aicore__ inline void CrossCoreReduce()
    {
        LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
        LocalTensor<float> ub = ubAllocator.Alloc<float, 1>();
        DataCopyParams cp1 = { 1, static_cast<uint16_t>(sizeof(float)), 0, 0 };

        ub.SetValue(0, partialSum_);
        SetAtomicAdd<float>();
        DataCopyPad(outGm_, ub, cp1);
        DisableDmaAtomic();
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe_;

    uint32_t blockIdx_;
    uint32_t nnz_;
    uint64_t yLen_;
    uint32_t nnzPerCore_;
    uint32_t nnzStart_, nnzCount_;
    float partialSum_;

    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<InputT> xvalGm_;
    GlobalTensor<InputT> yGm_;
    GlobalTensor<float> outGm_;

    TQue<TPosition::VECIN, 1> qIdx_;        // full-tile indices (scan + offset workspace)
    TQue<TPosition::VECIN, 1> qXval_;       // full-tile x values (final Mul)
    TQue<TPosition::VECIN, 1> qYSlice_;     // per-segment y (aligned load)
    TQue<TPosition::VECIN, 1> qYGathered_;  // gather concat + mul dst
    TQue<TPosition::VECIN, 1> qReduce_;     // ReduceSum dst/tmp
    TQue<TPosition::VECIN, 1> qYGatheredFp32_;  // Only used when InputSz == 2
};

// ---------------------------------------------------------------------------
// Kernel entry points
// ---------------------------------------------------------------------------

__global__ __aicore__ void spvv_kernel_fp32(
    GM_ADDR xIndices, GM_ADDR xValues, GM_ADDR y,
    GM_ADDR output, SpvvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelSpvv<float, 4> op;
    op.Init(xIndices, xValues, y, output, tiling);
    op.Process();
}

__global__ __aicore__ void spvv_kernel_half(
    GM_ADDR xIndices, GM_ADDR xValues, GM_ADDR y,
    GM_ADDR output, SpvvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelSpvv<half, 2> op;
    op.Init(xIndices, xValues, y, output, tiling);
    op.Process();
}

// ---------------------------------------------------------------------------
// Host-side launcher (called from spvv_host.cpp)
// ---------------------------------------------------------------------------

void spvv_kernel_do(GM_ADDR xIndices, GM_ADDR xValues, GM_ADDR y,
    GM_ADDR output, const SpvvTilingData &tiling, uint32_t blockNum,
    aclDataType valueType, void* stream)
{
    if (valueType == ACL_FLOAT) {
        spvv_kernel_fp32<<<blockNum, nullptr, stream>>>(
            xIndices, xValues, y, output, tiling);
    } else {
        spvv_kernel_half<<<blockNum, nullptr, stream>>>(
            xIndices, xValues, y, output, tiling);
    }
}
