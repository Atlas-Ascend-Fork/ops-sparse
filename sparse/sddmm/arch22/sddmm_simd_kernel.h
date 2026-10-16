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

/*!
 * \file sddmm_simd_kernel.h
 * \brief sddmm arch22 kernel 实现（SIMD：TPipe/TBuf/LocalTensor + 低阶 AscendC API）。
 */

#ifndef SDDMM_ARCH22_SIMD_KERNEL_H_
#define SDDMM_ARCH22_SIMD_KERNEL_H_

#include "kernel_operator.h"
#include "sddmm.h"

namespace {
// UB block granularity for fp32: 32 bytes / 4 bytes = 8 elements.
constexpr uint32_t kAlign = 8;
// Elements a single vector repeat can cover for fp32 (256B / 4B).
constexpr uint32_t kElemPerRep = 64;
// Repeat-stride fields are 8-bit, so a staged row pitch may span at most 255 blocks.
constexpr uint32_t kMaxPitchBlocks = 255;
// Staging capacity for the CSR value / column-index / output segments.
constexpr uint32_t kOutStageCap = 4096;
// Nonzeros processed per batched segment. Each needs one kElemPerRep-wide product
// row in mulBuf and one staged B column in bColBuf, so this bounds both buffers.
constexpr uint32_t kJTileCap = 128;
// Scanning for consecutive column runs only pays off above ~1/kRunDensityNum
// density; sparser patterns almost never yield runs longer than one.
constexpr uint32_t kRunDensityNum = 4;
constexpr uint32_t kHugeKCap = 8192;
constexpr uint32_t kStreamKCap = 16448;
// Include one extra vector repeat beyond 16K so a short tail can be reduced
// with the main chunk. ReduceSum needs at most 257 first-stage partials;
// 512 scratch elements cover their aligned layout and subsequent reduction.
constexpr uint32_t kStreamReduceScratch = 512;
constexpr int kACacheNone = 0;
constexpr int kACacheSingle = 1;
constexpr int kACacheFull = 2;
constexpr uint32_t kSmallDmaMinK = 32;
constexpr uint32_t kSmallDmaMinRows = 256;
// DMA alignment requirement: DataCopy requires lenBytes to be 32-byte aligned.
constexpr uint32_t kDmaAlignBytes = 32;
constexpr uint32_t kDmaAlignMask = kDmaAlignBytes - 1;
// Strided GM->UB gather: DataCopyPad lands each element on its own 32B block, so
// the staging buffer holds kStrideChunk elements at kStrideElemBytes pitch.
constexpr uint32_t kStrideElemBytes = 32;
// One 256-element padded gather amortizes DMA/event overhead for strided K
// tails while keeping the worst-case FP32 UB footprint below 192 KiB.
constexpr uint32_t kStrideChunk = 256;

__aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t a) {
    if (a == 0) {
        return x;
    }
    return (x + a - 1) / a * a;
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

struct SddmmWorkRange {
    uint32_t begin;
    uint32_t count;
};

// Keep the uneven work split identical for the FP32 and FP16 kernels.  The
// helper also makes the partition boundary explicit at each call site.
__aicore__ inline SddmmWorkRange GetSddmmWorkRange(uint32_t perCore, uint32_t remainder) {
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t extra = (block < remainder) ? 1u : 0u;
    return {block * perCore + ((block < remainder) ? block : remainder), perCore + extra};
}

// Share the row traversal as well as its bounds across both precision paths.
// Friendship grants access to ProcessRow without widening the kernel interface.
template <typename Kernel>
struct SddmmRowPartition {
    __aicore__ static inline void Run(Kernel &kernel) {
        const SddmmWorkRange range = GetSddmmWorkRange(kernel.rowsPerCore, kernel.remainderRows);
        for (uint32_t g = range.begin;
             g < range.begin + range.count && g < kernel.rowCount; ++g) {
            kernel.ProcessRow(g);
        }
    }
};

// Shape checks and X contiguity are shared; Y contiguity stays path-specific.
template <typename Kernel>
struct SddmmCommonLayout {
    __aicore__ static inline bool Setup(Kernel &kernel) {
        if (kernel.nnz == 0 || kernel.rowCount == 0) {
            return false;
        }
        const uint32_t innerA = (kernel.opAValue == 0) ? kernel.matACols : kernel.matARows;
        const uint32_t innerB = (kernel.opBValue == 0) ? kernel.matBRows : kernel.matBCols;
        if (innerA != innerB) {
            return false;
        }
        kernel.K = innerA;
        kernel.colCount = (kernel.opBValue == 0) ? kernel.matBCols : kernel.matBRows;
        kernel.aContig = (kernel.orderXCol == 0 && kernel.opAValue == 0) ||
                         (kernel.orderXCol == 1 && kernel.opAValue == 1);
        return true;
    }
};

template <AscendC::HardEvent Event>
__aicore__ inline void SyncSddmmEvent() {
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    AscendC::SetFlag<Event>(event);
    AscendC::WaitFlag<Event>(event);
}

template <typename RowOffsets>
__aicore__ inline uint32_t FindSddmmRowForNnz(const RowOffsets &rowOffsets,
                                               uint32_t rowCount, uint32_t nnzIndex) {
    uint32_t lo = 0;
    uint32_t hi = rowCount;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (rowOffsets.GetValue(mid + 1) <= static_cast<int32_t>(nnzIndex)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}
} // namespace

template <bool StridedB>
class KernelSparseSddmm {
public:
    __aicore__ inline KernelSparseSddmm() {}

    __aicore__ inline void Init(GM_ADDR matA, GM_ADDR matB, GM_ADDR csrRowOffsets,
                                GM_ADDR csrColIndices, GM_ADDR csrValues,
                                const SddmmTilingData &td, AscendC::TPipe *pipeIn) {
        pipe = pipeIn;
        CopyTiling(td);
        BindGlobalBuffers(matA, matB, csrRowOffsets, csrColIndices, csrValues);
        InitBuffers();
    }

    __aicore__ inline void BindGlobalBuffers(GM_ADDR matA, GM_ADDR matB, GM_ADDR csrRowOffsets,
                                              GM_ADDR csrColIndices, GM_ADDR csrValues) {
        matAGm.SetGlobalBuffer((__gm__ float *)matA,
                               static_cast<uint64_t>(matARows) * ldx);
        const uint64_t bGmSize = static_cast<uint64_t>(matBRows) *
                                 ((ldy > matBCols) ? ldy : matBCols);
        matBGm.SetGlobalBuffer((__gm__ float *)matB, bGmSize);
        rowOffsetsGm.SetGlobalBuffer((__gm__ int32_t *)csrRowOffsets,
                                     static_cast<uint64_t>(rowCount + 1));
        colIndicesGm.SetGlobalBuffer((__gm__ int32_t *)csrColIndices, nnz);
        csrValuesGm.SetGlobalBuffer((__gm__ float *)csrValues, nnz);
    }

    __aicore__ inline void InitBuffers() {
        const uint32_t effectiveK = (opAValue == 0) ? matACols : matARows;
        // Above the full-row cache capacity, stream larger vectors and avoid
        // allocating the unused full-A cache.
        if (effectiveK > kHugeKCap) {
            kBlockLen = kStreamKCap;
        }
        const uint32_t kbuf = AlignUp(kBlockLen, kAlign);
        kbufAligned = kbuf;
        hugeKPath = effectiveK > kBlockLen && effectiveK <= kHugeKCap;
        // The huge-K full-cache path never reads aRowBuf.  Reclaim that UB for
        // a single full-width B vector and reduction workspace.
        const bool fullKStorage = hugeKPath;
        const uint32_t aRowElems = fullKStorage ? kAlign : kbuf;
        uint32_t bColElems = fullKStorage ? kHugeKCap :
                             (effectiveK > kHugeKCap ? kbuf : 2 * kbuf);
        if constexpr (!StridedB) {
            // A complete 128-column tile avoids a third short segment at
            // K=65..128. The maximum extra 32KiB still fits the arch22 UB.
            if (effectiveK <= 128u) {
                const uint32_t fullTile = kJTileCap * AlignUp(effectiveK, kAlign);
                bColElems = bColElems > fullTile ? bColElems : fullTile;
            }
        }
        bTileCapacity = bColElems;
        InitPrimaryBuffers(aRowElems, bColElems);
        InitRowBuffers(effectiveK, kbuf, fullKStorage);
        InitGatherBuffers(effectiveK);
    }

    __aicore__ inline void InitPrimaryBuffers(uint32_t aRowElems, uint32_t bColElems) {
        pipe->InitBuffer(aRowBuf, aRowElems * sizeof(float));
        // Doubles as the double-buffered scalar B column pair and as the batched B tile.
        pipe->InitBuffer(bColBuf, bColElems * sizeof(float));
        if (hugeKPath) {
            pipe->InitBuffer(aFullBuf, AlignUp(kHugeKCap, kAlign) * sizeof(float));
        }
    }

    __aicore__ inline void InitRowBuffers(uint32_t effectiveK, uint32_t kbuf,
                                          bool fullKStorage) {
        const uint32_t reduceTmpElems = effectiveK > kHugeKCap ? kStreamReduceScratch :
                                        (fullKStorage ? kHugeKCap : kbuf);
        if (nnzPartition == 0) {
            pipe->InitBuffer(inQueueCsr, 1, kOutStageCap * sizeof(float));
            pipe->InitBuffer(colIndBuf, kOutStageCap * sizeof(int32_t));
        }
        if (nnzPartition != 0) {
            pipe->InitBuffer(scalarOutputBuf, kAlign * sizeof(float));
        } else {
            pipe->InitBuffer(outQueueY, 1, kOutStageCap * sizeof(float));
        }
        // mulBuf holds either one K-length product vector (scalar path) or
        // kJTileCap product rows of kElemPerRep each (batched path).
        const uint32_t mulElems = effectiveK > kHugeKCap ? kAlign :
                                 ((kJTileCap * kElemPerRep > kbuf) ? (kJTileCap * kElemPerRep) : kbuf);
        pipe->InitBuffer(mulBuf, mulElems * sizeof(float));
        pipe->InitBuffer(reduceTmpBuf, reduceTmpElems * sizeof(float));
        pipe->InitBuffer(reduceDstBuf, kAlign * sizeof(float));
        // Dot-product accumulator, kept separate from reduceTmpBuf because
        // ReduceSum overwrites its sharedTmpBuffer argument.
        if (nnzPartition == 0) {
            pipe->InitBuffer(accBuf, AlignUp(kJTileCap, kAlign) * sizeof(float));
            pipe->InitBuffer(partialBuf, AlignUp(kJTileCap, kAlign) * sizeof(float));
        }
    }

    __aicore__ inline void InitGatherBuffers(uint32_t effectiveK) {
        // The strided gather is only reachable when a row of X or a column of Y is
        // non-contiguous in GM; skip its staging buffers otherwise so contiguous
        // shapes (notably the huge-K path) keep the UB headroom.
        const bool aCtg = (orderXCol == 0 && opAValue == 0) || (orderXCol == 1 && opAValue == 1);
        const bool bCtg = (orderYCol == 0 && opBValue == 1) || (orderYCol == 1 && opBValue == 0);
        needStrideGather = (!aCtg || !bCtg);
        strideCapacity = (StridedB && effectiveK <= 256) ? 1024 : kStrideChunk;
        if (needStrideGather || StridedB) {
            pipe->InitBuffer(stridePadBuf, strideCapacity * kStrideElemBytes);
            pipe->InitBuffer(strideOffsetBuf, strideCapacity * sizeof(int32_t));
        }
    }


    __aicore__ inline void Process() {
        if (!SetupLayout()) {
            return;
        }
        if (nnzPartition != 0) {
            RunNnzPartition();
        } else {
            SetupBatchPlan();
            RunRowPartition();
        }
    }

private:
    friend struct SddmmRowPartition<KernelSparseSddmm>;
    friend struct SddmmCommonLayout<KernelSparseSddmm>;

    // Split the reduction dimension K and the contiguity flags out of the tiling
    // data. Returns false when there is nothing to do or the shapes disagree.
    __aicore__ inline bool SetupLayout() {
        if (!SddmmCommonLayout<KernelSparseSddmm>::Setup(*this)) {
            return false;
        }
        bContig = !StridedB;
        // The strided gather (and hence its offset table) is only reachable when a
        // row of X or a column of Y is non-contiguous in GM.
        if (needStrideGather) {
            InitStrideOffsets();
        }
        return SetupDmaFlags();
    }

    // Each core takes a contiguous slice of nonzeros; used for huge-K shapes where
    // a single row would not fill the cores.
    __aicore__ inline void RunNnzPartition() {
        const SddmmWorkRange range = GetSddmmWorkRange(nnzPerCore, remainderNnz);
        for (uint32_t j = 0; j < range.count; ++j) {
            if (range.begin + j < nnz) {
                ProcessNnz(range.begin + j, j + 1u < range.count);
            }
        }
    }

    // Each core takes a contiguous slice of rows (the common case).
    __aicore__ inline void RunRowPartition() {
        SddmmRowPartition<KernelSparseSddmm>::Run(*this);
    }

    // Contiguous small-K loads are only worth a DMA above a size threshold.
    __aicore__ inline bool SetupDmaFlags() {
        const uint32_t totalRows = rowCount;
        const bool kInRange = (K <= kBlockLen) && (K >= kSmallDmaMinK);
        smallKContigADma = aContig && kInRange && (totalRows >= kSmallDmaMinRows);
        smallKContigBDma = bContig && kInRange;
        return true;
    }
    // Copy the scalar tiling fields into members.
    __aicore__ inline void CopyTiling(const SddmmTilingData &td) {
        nnz = static_cast<uint64_t>(td.nnz);
        rowCount = td.rowCount;
        matARows = td.matARows;
        matACols = td.matACols;
        matBRows = td.matBRows;
        matBCols = td.matBCols;
        rowsPerCore = td.rowsPerCore;
        remainderRows = td.remainderRows;
        kBlockLen = td.kBlockLen;
        nnzPartition = td.nnzPartition;
        nnzPerCore = td.nnzPerCore;
        remainderNnz = td.remainderNnz;
        alphaValue = td.alphaHost;
        betaValue = td.betaHost;
        opAValue = static_cast<int64_t>(td.opX);
        opBValue = static_cast<int64_t>(td.opY);
        ldx = td.ldx;
        ldy = td.ldy;
        orderXCol = td.orderX;
        orderYCol = td.orderY;
    }
    // Decide whether the batched (vector-reduction) path is usable, and size its tile.
    //
    // The batched path stages jTile B columns contiguously in UB as a jTile x kPitch
    // matrix, multiplies them against the cached A row using a repeat-stride-0
    // broadcast, then collapses each staged row to a scalar with a single
    // WholeReduceSum. That removes the per-nonzero ReduceSum + GetValue vector-to-scalar
    // stall that dominates the scalar path.
    __aicore__ inline void SetupBatchPlan() {
        batchEnabled = false;
        if (K == 0 || K > kBlockLen || (StridedB && K > strideCapacity)) {
            return;
        }
        kPitch = AlignUp(K, kAlign);
        if (kPitch == 0) {
            return;
        }
        // src1RepStride / dstRepStride are 8-bit block counts.
        if ((kPitch / kAlign) > kMaxPitchBlocks) {
            return;
        }
        // B tile lives in bColBuf (2 * kbufAligned floats).
        const uint32_t byTile = bTileCapacity / kPitch;
        const uint32_t stridedCap = StridedB ? (strideCapacity / kPitch) : kJTileCap;
        const uint32_t cap = MinU32(byTile, stridedCap);
        if (cap == 0) {
            return;
        }
        jTile = cap;
        batchEnabled = true;
        // Multi-block staging needs contiguous destination slots (dstStride == 0),
        // which only holds when K is already block-aligned, and needs the source
        // gap to be non-negative. Only worth the index scan when C is dense enough
        // that consecutive column runs actually occur.
        runBatchable = (kPitch == K) && (ldy >= K) &&
                       (static_cast<uint64_t>(nnz) * kRunDensityNum >
                        static_cast<uint64_t>(rowCount) * colCount);
    }

    // Build the byte-offset table 0, 32, 64, ... used by the strided gather below.
    // Computed once per kernel invocation.
    __aicore__ inline void InitStrideOffsets() {
        AscendC::LocalTensor<int32_t> offs = strideOffsetBuf.Get<int32_t>();
        const uint32_t count = StridedB ? strideCapacity : MinU32(K, kStrideChunk);
        AscendC::CreateVecIndex<int32_t>(offs, 0, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(offs, offs, static_cast<int32_t>(kStrideElemBytes),
                      static_cast<int32_t>(count));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Strided GM->UB load: DataCopyPad pulls `len` single-element blocks (each
    // padded to 32B by the blockCount form), then one Gather compacts them into
    // ub[0..len). Vectorised throughout — no per-element scalar access.
    __aicore__ inline void GatherChunk(AscendC::LocalTensor<float> &ub,
                                       AscendC::GlobalTensor<float> &gm,
                                       uint64_t off, uint32_t stride, uint32_t len,
                                       bool protectFinalReuse = true) {
        if (stride == 1u) {
            CopyContig(ub, gm, off, len);
            return;
        }
        AscendC::LocalTensor<float> pad32 = stridePadBuf.Get<float>();
        AscendC::LocalTensor<int32_t> offs = strideOffsetBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> offsU = offs.ReinterpretCast<uint32_t>();
        const uint32_t srcGapBytes = (stride - 1u) * (uint32_t)sizeof(float);
        for (uint32_t base = 0; base < len; base += kStrideChunk) {
            const uint32_t cnt = MinU32(len - base, kStrideChunk);
            AscendC::DataCopyExtParams cp{static_cast<uint16_t>(cnt), (uint32_t)sizeof(float),
                                          srcGapBytes, 0, 0};
            AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(pad32, gm[off + static_cast<uint64_t>(base) * stride], cp, pd);
            SyncMte2ToV();
            AscendC::Gather<float>(ub[base], pad32, offsU, 0, cnt);
            AscendC::PipeBarrier<PIPE_V>();
            if (base + cnt < len || protectFinalReuse) {
                SyncVToMte2();
            }
        }
    }

    __aicore__ inline void CopyContig(AscendC::LocalTensor<float> &ub,
                                      AscendC::GlobalTensor<float> &gm,
                                      uint64_t off, uint32_t len) {
        const uint32_t lenBytes = len * (uint32_t)sizeof(float);
        if ((lenBytes & kDmaAlignMask) == 0 && lenBytes > 0) {
            AscendC::DataCopy(ub, gm[off], len);
        } else {
            AscendC::DataCopyExtParams p{1, lenBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(ub, gm[off], p, pad);
        }
    }

    // Stage `run` consecutive B columns (GM row pitch ldy) into `run` adjacent UB
    // slots of kPitch elements each, as a single multi-block DMA.
    //
    // Only used when kPitch == K, so the destination blocks are back-to-back and
    // dstStride is 0. srcStride is the inter-block gap in bytes, matching the
    // convention used by the merged arch35 path.
    __aicore__ inline void CopyBRun(AscendC::LocalTensor<float> dst, uint64_t gmOff, uint32_t run) {
        if (run == 1) {
            CopyContig(dst, matBGm, gmOff, K);
            return;
        }
        AscendC::DataCopyExtParams p{static_cast<uint16_t>(run), K * (uint32_t)sizeof(float),
                                     (ldy - K) * (uint32_t)sizeof(float), 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(dst, matBGm[gmOff], p, pad);
    }

    __aicore__ inline void SyncMte2ToV() {
        SyncSddmmEvent<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ inline void SyncMte2ToS() {
        SyncSddmmEvent<AscendC::HardEvent::MTE2_S>();
    }

    __aicore__ inline void SyncVToS() {
        SyncSddmmEvent<AscendC::HardEvent::V_S>();
    }

    __aicore__ inline void SyncVToMte2() {
        SyncSddmmEvent<AscendC::HardEvent::V_MTE2>();
    }

    __aicore__ inline void SyncSToMte3() {
        SyncSddmmEvent<AscendC::HardEvent::S_MTE3>();
    }

    __aicore__ inline void SyncMte3ToS() {
        SyncSddmmEvent<AscendC::HardEvent::MTE3_S>();
    }

    __aicore__ inline float DotProduct(uint64_t aOff, uint32_t aStride, uint64_t bOff,
                                       uint32_t bStride, int aCache) {
        if (aCache == kACacheFull && bStride == 1u && K <= kHugeKCap) {
            return DotProductFullCache(bOff);
        }
        return DotProductByChunks(aOff, aStride, bOff, bStride, aCache);
    }

    __aicore__ inline float DotProductFullCache(uint64_t bOff) {
        AscendC::LocalTensor<float> bC = bColBuf.Get<float>();
        AscendC::LocalTensor<float> ml = K > kHugeKCap ? bC : mulBuf.Get<float>();
        AscendC::LocalTensor<float> rt = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> rd = reduceDstBuf.Get<float>();
        AscendC::LocalTensor<float> aFull = aFullBuf.Get<float>();
        CopyContig(bC, matBGm, bOff, K);
        SyncMte2ToV();
        AscendC::Mul(ml, aFull, bC, K);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float, true>(rd, ml, rt, static_cast<int32_t>(K));
        SyncVToS();
        return rd.GetValue(0);
    }

    __aicore__ inline float DotProductByChunks(uint64_t aOff, uint32_t aStride, uint64_t bOff,
                                                uint32_t bStride, int aCache) {
        AscendC::LocalTensor<float> aC = aRowBuf.Get<float>();
        AscendC::LocalTensor<float> bC = bColBuf.Get<float>();
        AscendC::LocalTensor<float> ml = K > kHugeKCap ? bC : mulBuf.Get<float>();
        AscendC::LocalTensor<float> rt = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> rd = reduceDstBuf.Get<float>();
        float acc = 0.0f;
        const bool bUseDma = (bStride == 1u && (aCache == kACacheFull || smallKContigBDma));
        for (uint32_t base = 0; base < K; base += kBlockLen) {
            const uint32_t len = (K - base < kBlockLen) ? (K - base) : kBlockLen;
            if (K > kHugeKCap && len < kAlign) {
                for (uint32_t i = 0; i < len; ++i) {
                    const float av = matAGm.GetValue(aOff + static_cast<uint64_t>(base + i) * aStride);
                    const float bv = matBGm.GetValue(bOff + static_cast<uint64_t>(base + i) * bStride);
                    acc += av * bv;
                }
                break;
            }
            acc += DotProductChunk(aC, bC, ml, rt, rd, aOff, aStride, bOff, bStride,
                                   base, len, aCache, bUseDma);
        }
        return acc;
    }

    __aicore__ inline float DotProductChunk(AscendC::LocalTensor<float> aC,
                                            AscendC::LocalTensor<float> bC,
                                            AscendC::LocalTensor<float> product,
                                            AscendC::LocalTensor<float> temporary,
                                            AscendC::LocalTensor<float> result,
                                            uint64_t aOff, uint32_t aStride, uint64_t bOff,
                                            uint32_t bStride, uint32_t base, uint32_t len,
                                            int aCache, bool bUseDma) {
        AscendC::LocalTensor<float> aOp = aC;
        if (aCache == kACacheNone) {
            GatherChunk(aC, matAGm, aOff + static_cast<uint64_t>(base) * aStride, aStride, len);
        } else if (aCache == kACacheFull) {
            aOp = aFullBuf.Get<float>()[base];
        }
        if (bUseDma) {
            CopyContig(bC, matBGm, bOff + static_cast<uint64_t>(base) * bStride, len);
        } else {
            GatherChunk(bC, matBGm, bOff + static_cast<uint64_t>(base) * bStride, bStride, len, false);
        }
        if (bStride == 1u) {
            SyncMte2ToV();
        }
        AscendC::Mul(product, aOp, bC, len);
        AscendC::PipeBarrier<PIPE_V>();
        if (len <= kElemPerRep) {
            AscendC::WholeReduceSum<float>(result, product, static_cast<int32_t>(len), 1, 1, 1, 8);
        } else {
            AscendC::ReduceSum<float, true>(result, product, temporary, static_cast<int32_t>(len));
        }
        SyncVToS();
        return result.GetValue(0);
    }

    __aicore__ inline void ProcessNnz(uint32_t gnnz, bool reuseOutput) {
        const uint32_t row = FindSddmmRowForNnz(rowOffsetsGm, rowCount, gnnz);
        const uint64_t aOff = aContig ? static_cast<uint64_t>(row) * ldx : static_cast<uint64_t>(row);
        const uint32_t aStride = aContig ? 1u : ldx;
        int aCache = kACacheNone;
        if (alphaValue != 0.0f) {
            if (K <= kBlockLen) {
                aCache = kACacheSingle;
                AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
                GatherChunk(a, matAGm, aOff, aStride, K);
            } else if (hugeKPath && K <= kHugeKCap) {
                aCache = kACacheFull;
                AscendC::LocalTensor<float> a = aFullBuf.Get<float>();
                if (aStride == 1u) {
                    CopyContig(a, matAGm, aOff, K);
                } else {
                    GatherChunk(a, matAGm, aOff, aStride, K);
                }
            }
        }
        const int32_t col = colIndicesGm.GetValue(gnnz);
        float dot = 0.0f;
        if (alphaValue != 0.0f && col >= 0 && static_cast<uint32_t>(col) < colCount) {
            const uint64_t bOff = bContig ? static_cast<uint64_t>(col) * ldy
                                          : static_cast<uint64_t>(col);
            const uint32_t bStr = bContig ? 1u : ldy;
            dot = DotProduct(aOff, aStride, bOff, bStr, aCache);
        }
        float value = alphaValue * dot;
        if (betaValue != 0.0f) {
            value += betaValue * csrValuesGm.GetValue(gnnz);
        }
        AscendC::LocalTensor<float> y = scalarOutputBuf.Get<float>();
        y.SetValue(0, value);
        SyncSToMte3();
        AscendC::DataCopyExtParams op{1, (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(csrValuesGm[gnnz], y, op);
        if (reuseOutput) {
            SyncMte3ToS();
        }
    }

    // Load the A row (or A column when opA/order make it strided) into UB once per row.
    __aicore__ inline int LoadARow(uint64_t aOff, uint32_t aStride) {
        if (alphaValue == 0.0f) {
            return kACacheNone;
        }
        if (K <= kBlockLen) {
            AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
            if (smallKContigADma || (batchEnabled && aStride == 1u)) {
                CopyContig(a, matAGm, aOff, K);
            } else {
                // Contiguous batched B does not overwrite the gather pad.
                // The next segment acquires the input queue before pad reuse.
                GatherChunk(a, matAGm, aOff, aStride, K, !(batchEnabled && !StridedB));
            }
            return kACacheSingle;
        }
        if (hugeKPath && K <= kHugeKCap) {
            AscendC::LocalTensor<float> a = aFullBuf.Get<float>();
            if (aStride == 1u) {
                CopyContig(a, matAGm, aOff, K);
            } else {
                // Contiguous batched B does not overwrite the gather pad.
                // The next segment acquires the input queue before pad reuse.
                GatherChunk(a, matAGm, aOff, aStride, K, !(batchEnabled && !StridedB));
            }
            return kACacheFull;
        }
        return kACacheNone;
    }

    // Batched segment: compute `segNnz` dot products with vector instructions only.
    //
    // Layout in UB:
    //   bTile[j * kPitch .. + K)   staged B column for nonzero j
    //   prod[j * kElemPerRep ..)   per-chunk products, one repeat row per nonzero
    //   acc[j]                     running dot product
    __aicore__ inline void ComputeSegmentBatched(AscendC::LocalTensor<int32_t> &colInd,
                                                 uint32_t segBase, uint32_t segNnz,
                                                 uint64_t bBB,
                                                 AscendC::LocalTensor<float> &aOp,
                                                 AscendC::LocalTensor<float> &acc) {
        AscendC::LocalTensor<float> bTile = bColBuf.Get<float>();
        StageBTile(colInd, segBase, segNnz, bBB, bTile);
        ReduceBTile(bTile, aOp, acc, segNnz);
    }

    // Stage this segment's B columns into `bTile` as segNnz rows of kPitch each.
    // CSR column indices ascend, so runs of consecutive columns are common; each
    // run is issued as one multi-block DMA instead of one DMA per nonzero.
    // Out-of-range columns contribute zero.
    __aicore__ inline void StageBTile(AscendC::LocalTensor<int32_t> &colInd,
                                      uint32_t segBase, uint32_t segNnz, uint64_t bBB,
                                      AscendC::LocalTensor<float> &bTile) {
        if constexpr (StridedB) {
            AscendC::LocalTensor<float> pad32 = stridePadBuf.Get<float>();
            const uint32_t srcGapBytes = (ldy - 1u) * (uint32_t)sizeof(float);
            for (uint32_t j = 0; j < segNnz; ++j) {
                const int32_t col = colInd.GetValue(segBase + j);
                if (col < 0 || static_cast<uint32_t>(col) >= colCount) {
                    continue;
                }
                AscendC::DataCopyExtParams cp{static_cast<uint16_t>(K), (uint32_t)sizeof(float),
                                              srcGapBytes, 0, 0};
                AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(pad32[j * kPitch * kAlign],
                                     matBGm[bBB + static_cast<uint64_t>(col)], cp, pd);
            }
            SyncMte2ToV();
            AscendC::LocalTensor<int32_t> offs = strideOffsetBuf.Get<int32_t>();
            AscendC::LocalTensor<uint32_t> offsU = offs.ReinterpretCast<uint32_t>();
            AscendC::Gather<float>(bTile, pad32, offsU, 0, segNnz * kPitch);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < segNnz; ++j) {
                const int32_t col = colInd.GetValue(segBase + j);
                if (col < 0 || static_cast<uint32_t>(col) >= colCount) {
                    AscendC::Duplicate(bTile[j * kPitch], 0.0f, static_cast<int32_t>(kPitch));
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        uint32_t j = 0;
        while (j < segNnz) {
            const int32_t col = colInd.GetValue(segBase + j);
            if (col < 0 || static_cast<uint32_t>(col) >= colCount) {
                AscendC::Duplicate(bTile[j * kPitch], 0.0f, static_cast<int32_t>(kPitch));
                ++j;
                continue;
            }
            uint32_t run = 1;
            if (runBatchable) {
                while (j + run < segNnz &&
                       colInd.GetValue(segBase + j + run) == col + static_cast<int32_t>(run)) {
                    ++run;
                }
            }
            CopyBRun(bTile[j * kPitch], bBB + static_cast<uint64_t>(col) * ldy, run);
            j += run;
        }
        SyncMte2ToV();
    }

    // Multiply the staged tile by the cached A row and collapse each staged row to
    // one dot product. acc[0..segNnz) receives the results.
    __aicore__ inline void ReduceBTile(AscendC::LocalTensor<float> &bTile,
                                       AscendC::LocalTensor<float> &aOp,
                                       AscendC::LocalTensor<float> &acc, uint32_t segNnz) {
        AscendC::LocalTensor<float> prod = mulBuf.Get<float>();
        const uint8_t rep = static_cast<uint8_t>(segNnz);
        const int32_t repStride = static_cast<int32_t>(kElemPerRep / kAlign);
        AscendC::BinaryRepeatParams mp;
        mp.dstBlkStride = 1;
        mp.src0BlkStride = 1;
        mp.src1BlkStride = 1;
        mp.dstRepStride = static_cast<uint8_t>(repStride);
        // src0RepStride == 0 broadcasts the single cached A row across all repeats.
        mp.src0RepStride = 0;
        mp.src1RepStride = static_cast<uint8_t>(kPitch / kAlign);

        if (K <= kElemPerRep) {
            // Single chunk: reduce straight into acc, no accumulator pass needed.
            AscendC::Mul(prod, aOp, bTile, static_cast<uint64_t>(K), rep, mp);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(acc, prod, static_cast<int32_t>(K),
                                           static_cast<int32_t>(segNnz), 1, 1, repStride);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        AscendC::LocalTensor<float> partial = partialBuf.Get<float>();
        AscendC::Duplicate(acc, 0.0f, static_cast<int32_t>(AlignUp(segNnz, kAlign)));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t base = 0; base < K; base += kElemPerRep) {
            const uint32_t len = MinU32(K - base, kElemPerRep);
            // prod[j][0..len) = aOp[base..base+len) * bTile[j][base..base+len)
            AscendC::Mul(prod, aOp[base], bTile[base], static_cast<uint64_t>(len), rep, mp);
            AscendC::PipeBarrier<PIPE_V>();
            // One vcadd collapses each staged repeat row to a single dot-product term.
            AscendC::WholeReduceSum<float>(partial, prod, static_cast<int32_t>(len),
                                           static_cast<int32_t>(segNnz), 1, 1, repStride);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(acc, acc, partial, static_cast<int32_t>(segNnz));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Scalar fallback for layouts the batched path cannot serve (strided B, huge K).
    __aicore__ inline void ComputeSegmentScalar(AscendC::LocalTensor<int32_t> &colInd,
                                                 uint32_t segBase, uint32_t segNnz,
                                                 uint64_t bBB, uint64_t aOff, uint32_t aStride,
                                                 int aCache, AscendC::LocalTensor<float> &csrStage,
                                                 AscendC::LocalTensor<float> &yStage) {
        for (uint32_t j = 0; j < segNnz; ++j) {
            const int32_t col = colInd.GetValue(segBase + j);
            float dot = 0.0f;
            if (alphaValue != 0.0f && col >= 0 && static_cast<uint32_t>(col) < colCount) {
                const uint64_t bOff = bContig ? (bBB + static_cast<uint64_t>(col) * ldy)
                                              : (bBB + static_cast<uint64_t>(col));
                const uint32_t bStr = bContig ? 1u : ldy;
                dot = DotProduct(aOff, aStride, bOff, bStr, aCache);
            }
            const float cv = (betaValue != 0.0f) ? csrStage.GetValue(j) : 0.0f;
            yStage.SetValue(j, alphaValue * dot + betaValue * cv);
        }
    }

    // alpha == 0 case: y = beta * C over the row, in segments of kOutStageCap.
    __aicore__ inline void ScaleRowOnly(uint64_t rowOff, uint32_t rowNnz) {
        for (uint32_t segBase = 0; segBase < rowNnz; segBase += kOutStageCap) {
            const uint32_t segNnz = MinU32(rowNnz - segBase, kOutStageCap);
            const uint64_t segOff = rowOff + segBase;
            const uint32_t segBytes = segNnz * (uint32_t)sizeof(float);
            AscendC::LocalTensor<float> yStage = outQueueY.AllocTensor<float>();
            if (betaValue == 0.0f) {
                AscendC::Duplicate(yStage, 0.0f, static_cast<int32_t>(AlignUp(segNnz, kAlign)));
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::LocalTensor<float> csrIn = inQueueCsr.AllocTensor<float>();
                AscendC::DataCopyExtParams p{1, segBytes, 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(csrIn, csrValuesGm[segOff], p, pad);
                inQueueCsr.EnQue(csrIn);
                AscendC::LocalTensor<float> csrStage = inQueueCsr.DeQue<float>();
                AscendC::Muls(yStage, csrStage, betaValue, static_cast<int32_t>(segNnz));
                AscendC::PipeBarrier<PIPE_V>();
                inQueueCsr.FreeTensor(csrStage);
            }
            outQueueY.EnQue(yStage);
            AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
            AscendC::DataCopyExtParams outP{1, segBytes, 0, 0, 0};
            AscendC::DataCopyPad(csrValuesGm[segOff], yOut, outP);
            outQueueY.FreeTensor(yOut);
        }
    }

    __aicore__ inline void ProcessRow(uint32_t row) {
        const int32_t start = rowOffsetsGm.GetValue(row);
        const int32_t end = rowOffsetsGm.GetValue(row + 1);
        const uint32_t rowNnz = static_cast<uint32_t>(end - start);
        if (rowNnz == 0) {
            return;
        }

        const uint64_t aOff = aContig ? (static_cast<uint64_t>(row) * ldx) : static_cast<uint64_t>(row);
        const uint32_t aStride = aContig ? 1u : ldx;

        if (alphaValue == 0.0f) {
            ScaleRowOnly(static_cast<uint64_t>(start), rowNnz);
            return;
        }

        const uint32_t segCap = batchEnabled ? jTile : kOutStageCap;
        int aCache = kACacheNone;
        for (uint32_t segBase = 0; segBase < rowNnz; segBase += segCap) {
            const uint32_t segNnz = MinU32(rowNnz - segBase, segCap);
            ProcessSegment(static_cast<uint64_t>(start) + segBase, segNnz,
                           0ULL, aOff, aStride, aCache);
        }
    }

    // Load one segment's column indices (and C values when beta != 0), compute the
    // dot products, then write back y = alpha * dot + beta * C.
    __aicore__ inline void ProcessSegment(uint64_t segOff, uint32_t segNnz, uint64_t bBB,
                                          uint64_t aOff, uint32_t aStride, int &aCache) {
        AscendC::LocalTensor<int32_t> colInd = colIndBuf.Get<int32_t>();
        AscendC::DataCopyExtParams cp{1, segNnz * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> cpd{false, 0, 0, 0};
        AscendC::DataCopyPad(colInd, colIndicesGm[segOff], cp, cpd);

        AscendC::LocalTensor<float> csrIn = inQueueCsr.AllocTensor<float>();
        // Alloc consumes the preceding FreeTensor V->MTE2 event before
        // reusing any A/B/gather staging in this segment.
        if (aCache == kACacheNone) {
            aCache = LoadARow(aOff, aStride);
        }
        if (betaValue != 0.0f) {
            AscendC::DataCopyExtParams p{1, segNnz * (uint32_t)sizeof(float), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(csrIn, csrValuesGm[segOff], p, pad);
        }
        // Both the batch staging and scalar fallback read column indices on S.
        SyncMte2ToS();
        inQueueCsr.EnQue(csrIn);
        AscendC::LocalTensor<float> csrStage = inQueueCsr.DeQue<float>();

        AscendC::LocalTensor<float> yStage = outQueueY.AllocTensor<float>();
        if (batchEnabled) {
            AscendC::LocalTensor<float> acc = accBuf.Get<float>();
            AscendC::LocalTensor<float> aOp = (aCache == kACacheFull) ? aFullBuf.Get<float>()
                                                                     : aRowBuf.Get<float>();
            ComputeSegmentBatched(colInd, 0, segNnz, bBB, aOp, acc);
            CombineBatched(yStage, acc, csrStage, segNnz);
        } else {
            ComputeSegmentScalar(colInd, 0, segNnz, bBB, aOff, aStride, aCache, csrStage, yStage);
            SyncSToMte3();
        }
        inQueueCsr.FreeTensor(csrStage);

        outQueueY.EnQue(yStage);
        AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams outP{1, segNnz * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(csrValuesGm[segOff], yOut, outP);
        outQueueY.FreeTensor(yOut);
    }

    // y = alpha * acc + beta * csr, entirely in the vector pipe.
    __aicore__ inline void CombineBatched(AscendC::LocalTensor<float> &yStage,
                                          AscendC::LocalTensor<float> &acc,
                                          AscendC::LocalTensor<float> &csrStage,
                                          uint32_t segNnz) {
        AscendC::Muls(yStage, acc, alphaValue, static_cast<int32_t>(segNnz));
        AscendC::PipeBarrier<PIPE_V>();
        if (betaValue != 0.0f) {
            AscendC::Axpy<float, float>(yStage, csrStage, betaValue,
                                        static_cast<int32_t>(segNnz));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::TPipe *pipe = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aRowBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bColBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aFullBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mulBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceDstBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> colIndBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partialBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarOutputBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stridePadBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> strideOffsetBuf;
    bool needStrideGather = false;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueCsr;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::GlobalTensor<float> matAGm;
    AscendC::GlobalTensor<float> matBGm;
    AscendC::GlobalTensor<int32_t> rowOffsetsGm;
    AscendC::GlobalTensor<int32_t> colIndicesGm;
    AscendC::GlobalTensor<float> csrValuesGm;
    uint64_t nnz = 0;
    uint32_t rowCount = 0;
    uint32_t matARows = 0;
    uint32_t matACols = 0;
    uint32_t matBRows = 0;
    uint32_t matBCols = 0;
    uint32_t rowsPerCore = 0;
    uint32_t remainderRows = 0;
    uint32_t kBlockLen = 0;
    uint32_t K = 0;
    uint32_t colCount = 0;
    int64_t opAValue = 0;
    int64_t opBValue = 0;
    float alphaValue = 0.0f;
    float betaValue = 0.0f;
    bool hugeKPath = false;
    bool smallKContigADma = false;
    bool smallKContigBDma = false;
    bool aContig = true;
    bool bContig = true;
    bool batchEnabled = false;
    uint32_t strideCapacity = kStrideChunk;
    bool runBatchable = false;
    uint32_t nnzPartition = 0;
    uint32_t nnzPerCore = 0;
    uint32_t remainderNnz = 0;
    uint32_t ldx = 0;
    uint32_t ldy = 0;
    uint32_t orderXCol = 0;
    uint32_t orderYCol = 0;
    uint32_t bTileCapacity = 0;
    uint32_t kbufAligned = 0;
    uint32_t kPitch = 0;
    uint32_t jTile = 0;
};

#include "sddmm_fp16_kernel.h"

#endif // SDDMM_ARCH22_SIMD_KERNEL_H_
